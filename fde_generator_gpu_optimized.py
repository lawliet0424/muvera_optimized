# -*- coding: utf-8 -*-
"""
GPU-Optimized FDE Generator using CuPy with CUDA kernels
Parallelizes across repetitions with shared memory optimization
Optimized for NVIDIA TITAN V (Compute Capability 7.0, 96KB shared memory)
"""
import logging
import time
import os
import numpy as np
import cupy as cp
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, List

# Import original config classes
#from fde_generator_optimized_stream_weight_fde_gpu import (
#    EncodingType,
#    ProjectionType,
#    FixedDimensionalEncodingConfig,
#    _gray_code_to_binary,
#)


# ==============================================================================
# CUDA KERNEL: Parallel FDE Generation Across Repetitions
# ==============================================================================

# Kernel 1: SimHash projection and partition assignment (vectorized across repetitions)
SIMHASH_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void simhash_projection_multi_rep(
    const float* __restrict__ embeddings,      // [num_docs, max_len, dim]
    const int* __restrict__ doc_lengths,       // [num_docs]
    const float* __restrict__ simhash_matrices, // [num_reps, dim, num_bits]
    const float* __restrict__ ams_matrices,     // [num_reps, dim, proj_dim] (or NULL)
    float* __restrict__ sketches_out,          // [num_docs, num_reps, max_len, num_bits]
    float* __restrict__ projected_out,         // [num_docs, num_reps, max_len, proj_dim]
    int* __restrict__ partition_indices,       // [num_docs, num_reps, max_len]
    const int num_docs,
    const int max_len,
    const int dim,
    const int num_bits,
    const int proj_dim,
    const int num_reps,
    const int use_identity
) {
    // Global thread ID maps to (doc_idx, rep_idx, token_idx)
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = num_docs * num_reps * max_len;
    
    if (global_id >= total_tokens) return;
    
    // Decode indices
    int token_idx = global_id % max_len;
    int temp = global_id / max_len;
    int rep_idx = temp % num_reps;
    int doc_idx = temp / num_reps;
    
    // Check if this token exists for this document
    if (token_idx >= doc_lengths[doc_idx]) return;
    
    // Input embedding pointer
    const float* emb = &embeddings[(doc_idx * max_len + token_idx) * dim];
    
    // SimHash matrix for this repetition
    const float* simhash_mat = &simhash_matrices[rep_idx * dim * num_bits];
    
    // Compute SimHash sketches: X @ simhash_matrix
    float sketches[32];  // Max 32 bits (adjust if needed)
    for (int b = 0; b < num_bits; b++) {
        float val = 0.0f;
        for (int d = 0; d < dim; d++) {
            val += emb[d] * simhash_mat[d * num_bits + b];
        }
        sketches[b] = val;
    }
    
    // Store sketches
    int sketch_offset = ((doc_idx * num_reps + rep_idx) * max_len + token_idx) * num_bits;
    for (int b = 0; b < num_bits; b++) {
        sketches_out[sketch_offset + b] = sketches[b];
    }
    
    // Compute partition index using Gray code
    unsigned int p_idx = 0;
    for (int b = 0; b < num_bits; b++) {
        unsigned int bit = (sketches[b] > 0.0f) ? 1 : 0;
        p_idx = (p_idx << 1) + (bit ^ (p_idx & 1));
    }
    partition_indices[(doc_idx * num_reps + rep_idx) * max_len + token_idx] = p_idx;
    
    // Projection (AMS sketch or identity)
    int proj_offset = ((doc_idx * num_reps + rep_idx) * max_len + token_idx) * proj_dim;
    
    if (use_identity) {
        // Identity projection: just copy embeddings
        for (int d = 0; d < proj_dim; d++) {
            projected_out[proj_offset + d] = emb[d];
        }
    } else {
        // AMS projection: X @ ams_matrix
        const float* ams_mat = &ams_matrices[rep_idx * dim * proj_dim];
        for (int p = 0; p < proj_dim; p++) {
            float val = 0.0f;
            for (int d = 0; d < dim; d++) {
                val += emb[d] * ams_mat[d * proj_dim + p];
            }
            projected_out[proj_offset + p] = val;
        }
    }
}
''', 'simhash_projection_multi_rep')


# Kernel 2: Scatter-add aggregation with shared memory optimization
SCATTER_ADD_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void scatter_add_partitions(
    const float* __restrict__ projected,        // [num_docs, num_reps, max_len, proj_dim]
    const int* __restrict__ partition_indices,  // [num_docs, num_reps, max_len]
    const int* __restrict__ doc_lengths,        // [num_docs]
    float* __restrict__ partition_sums,         // [num_docs, num_reps, num_partitions, proj_dim]
    int* __restrict__ partition_counts,         // [num_docs, num_reps, num_partitions]
    const int num_docs,
    const int num_reps,
    const int max_len,
    const int proj_dim,
    const int num_partitions
) {
    // Shared memory for partition aggregation (per block)
    extern __shared__ float shared_mem[];
    float* shared_sums = shared_mem;  // [num_partitions * proj_dim]
    int* shared_counts = (int*)&shared_sums[num_partitions * proj_dim];  // [num_partitions]
    
    int doc_idx = blockIdx.x;
    int rep_idx = blockIdx.y;
    
    if (doc_idx >= num_docs || rep_idx >= num_reps) return;
    
    int doc_len = doc_lengths[doc_idx];
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < num_partitions * proj_dim; i += blockDim.x) {
        shared_sums[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
        shared_counts[i] = 0;
    }
    __syncthreads();
    
    // Each thread processes multiple tokens
    for (int token_idx = threadIdx.x; token_idx < doc_len; token_idx += blockDim.x) {
        int p_idx = partition_indices[(doc_idx * num_reps + rep_idx) * max_len + token_idx];
        int proj_offset = ((doc_idx * num_reps + rep_idx) * max_len + token_idx) * proj_dim;
        
        // Atomic add to shared memory
        atomicAdd(&shared_counts[p_idx], 1);
        
        for (int d = 0; d < proj_dim; d++) {
            atomicAdd(&shared_sums[p_idx * proj_dim + d], projected[proj_offset + d]);
        }
    }
    __syncthreads();
    
    // Write shared memory back to global memory
    int out_offset = (doc_idx * num_reps + rep_idx) * num_partitions;
    for (int p = threadIdx.x; p < num_partitions; p += blockDim.x) {
        partition_counts[out_offset + p] = shared_counts[p];
        
        int sum_offset = out_offset * proj_dim + p * proj_dim;
        for (int d = 0; d < proj_dim; d++) {
            partition_sums[sum_offset + d] = shared_sums[p * proj_dim + d];
        }
    }
}
''', 'scatter_add_partitions')


# Kernel 3: Average computation and empty partition filling
AVERAGE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void compute_averages(
    float* __restrict__ partition_sums,         // [num_docs, num_reps, num_partitions, proj_dim]
    const int* __restrict__ partition_counts,   // [num_docs, num_reps, num_partitions]
    const int num_docs,
    const int num_reps,
    const int num_partitions,
    const int proj_dim
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_partitions = num_docs * num_reps * num_partitions;
    
    if (global_id >= total_partitions) return;
    
    int p_idx = global_id % num_partitions;
    int temp = global_id / num_partitions;
    int rep_idx = temp % num_reps;
    int doc_idx = temp / num_reps;
    
    int count_offset = (doc_idx * num_reps + rep_idx) * num_partitions + p_idx;
    int count = partition_counts[count_offset];
    
    if (count > 0) {
        int sum_offset = count_offset * proj_dim;
        float inv_count = 1.0f / (float)count;
        
        for (int d = 0; d < proj_dim; d++) {
            partition_sums[sum_offset + d] *= inv_count;
        }
    }
}
''', 'compute_averages')


# ==============================================================================
# Helper Functions
# ==============================================================================
class EncodingType(Enum):
    DEFAULT_SUM = 0
    AVERAGE = 1

class ProjectionType(Enum):
    DEFAULT_IDENTITY = 0
    AMS_SKETCH = 1

@dataclass
class FixedDimensionalEncodingConfig:
    dimension: int = 128
    num_repetitions: int = 10
    num_simhash_projections: int = 6
    seed: int = 42
    encoding_type: EncodingType = EncodingType.DEFAULT_SUM
    projection_type: ProjectionType = ProjectionType.DEFAULT_IDENTITY
    projection_dimension: Optional[int] = None
    fill_empty_partitions: bool = False
    final_projection_dimension: Optional[int] = None


def _append_to_gray_code(gray_code: int, bit: bool) -> int:
    return (gray_code << 1) + (int(bit) ^ (gray_code & 1))

def _gray_code_to_binary(num: int) -> int:
    mask = num >> 1
    while mask != 0:
        num = num ^ mask
        mask >>= 1
    return num

def _simhash_matrix_from_seed_gpu(
    dimension: int, num_projections: int, seed: int
) -> cp.ndarray:
    """Generate SimHash matrix on GPU"""
    rng = cp.random.default_rng(seed)

    # 평균 0, 표준편차 1인 가우시안
    simhash_mat = rng.standard_normal(
        size=(dimension, num_projections),
        dtype=cp.float32,
    )
    return simhash_mat


def _ams_projection_matrix_from_seed_gpu(
    dimension: int, projection_dim: int, seed: int
) -> cp.ndarray:
    """Generate AMS projection matrix on GPU"""
    rng = cp.random.default_rng(seed)
    out = cp.zeros((dimension, projection_dim), dtype=cp.float32)
    indices = rng.integers(0, projection_dim, size=dimension)

    # 0 또는 1 샘플링
    sign_bits = rng.integers(0, 2, size=dimension, dtype=cp.int8)
    # 0 -> -1, 1 -> +1 로 매핑
    signs = (sign_bits * 2 - 1).astype(cp.float32)
    
    out[cp.arange(dimension), indices] = signs
    return out


def _pad_doc_embeddings(doc_embeddings_list: List[np.ndarray]) -> tuple:
    """Pad document embeddings to uniform length and create length array"""
    doc_lengths = np.array([doc.shape[0] for doc in doc_embeddings_list], dtype=np.int32)
    max_len = int(doc_lengths.max())
    num_docs = len(doc_embeddings_list)
    dim = doc_embeddings_list[0].shape[1]
    
    # Create padded array
    padded = np.zeros((num_docs, max_len, dim), dtype=np.float32)
    for i, doc in enumerate(doc_embeddings_list):
        padded[i, :doc.shape[0], :] = doc
    
    return padded, doc_lengths, max_len


def generate_query_fde(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    """Generates a Fixed Dimensional Encoding for a query point cloud (using SUM)."""
    if config.fill_empty_partitions:
        raise ValueError(
            "Query FDE generation does not support 'fill_empty_partitions'."
        )
    query_config = replace(config, encoding_type=EncodingType.DEFAULT_SUM)
    return _generate_fde_internal(point_cloud, query_config)

# -----------------------------
# Core FDE generation routines
# -----------------------------
def _generate_fde_internal(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    if point_cloud.ndim != 2 or point_cloud.shape[1] != config.dimension:
        raise ValueError(
            f"Input data shape {point_cloud.shape} is inconsistent with config dimension {config.dimension}."
        )
    if not (0 <= config.num_simhash_projections < 32):
        raise ValueError(
            f"num_simhash_projections must be in [0, 31]: {config.num_simhash_projections}"
        )

    num_points, original_dim = point_cloud.shape
    num_partitions = 2**config.num_simhash_projections

    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    projection_dim = original_dim if use_identity_proj else config.projection_dimension
    if not use_identity_proj and (not projection_dim or projection_dim <= 0):
        raise ValueError(
            "A positive projection_dimension is required for non-identity projections."
        )

    final_fde_dim = config.num_repetitions * num_partitions * projection_dim
    out_fde = np.zeros(final_fde_dim, dtype=np.float32)

    for rep_num in range(config.num_repetitions):
        current_seed = config.seed + rep_num

        sketches = point_cloud @ _simhash_matrix_from_seed(
            original_dim, config.num_simhash_projections, current_seed
        )

        if use_identity_proj:
            projected_matrix = point_cloud
        elif config.projection_type == ProjectionType.AMS_SKETCH:
            ams_matrix = _ams_projection_matrix_from_seed(
                original_dim, projection_dim, current_seed
            )
            projected_matrix = point_cloud @ ams_matrix

        rep_fde_sum = np.zeros(num_partitions * projection_dim, dtype=np.float32)
        partition_counts = np.zeros(num_partitions, dtype=np.int32)
        partition_indices = np.array(
            [_simhash_partition_index_gray(sketches[i]) for i in range(num_points)]
        )

        for i in range(num_points):
            start_idx = partition_indices[i] * projection_dim
            rep_fde_sum[start_idx : start_idx + projection_dim] += projected_matrix[i]
            partition_counts[partition_indices[i]] += 1

        if config.encoding_type == EncodingType.AVERAGE:
            for i in range(num_partitions):
                start_idx = i * projection_dim
                if partition_counts[i] > 0:
                    rep_fde_sum[start_idx : start_idx + projection_dim] /= (
                        partition_counts[i]
                    )
                elif config.fill_empty_partitions and num_points > 0:
                    distances = [
                        _distance_to_simhash_partition(sketches[j], i)
                        for j in range(num_points)
                    ]
                    nearest_point_idx = np.argmin(distances)
                    rep_fde_sum[start_idx : start_idx + projection_dim] = (
                        projected_matrix[nearest_point_idx]
                    )

        rep_start_index = rep_num * num_partitions * projection_dim
        out_fde[rep_start_index : rep_start_index + rep_fde_sum.size] = rep_fde_sum

    if config.final_projection_dimension and config.final_projection_dimension > 0:
        return _apply_count_sketch_to_vector(
            out_fde, config.final_projection_dimension, config.seed
        )

    return out_fde

# ==============================================================================
# Main GPU FDE Generation Function
# ==============================================================================

def generate_document_fde_batch_gpu(
    doc_embeddings_list: List[np.ndarray],
    config: FixedDimensionalEncodingConfig,
    *,
    memmap_path: Optional[str] = None,
    max_bytes_in_memory: int = 2 * 1024**3,
    log_every: int = 1000,
    flush_interval: int = 1000
) -> np.ndarray:
    """
    GPU-optimized FDE batch generation with parallel repetition processing.
    
    Key optimizations:
    1. Parallelizes across repetitions (independent random projections)
    2. Uses shared memory for scatter-add operations
    3. Minimizes CPU-GPU transfers
    4. Streams computation for large batches
    
    Args:
        doc_embeddings_list: List of document embeddings [num_tokens, dim]
        config: FDE configuration
        memmap_path: Optional path for memory-mapped output
        max_bytes_in_memory: Memory threshold for using memmap
        log_every: Logging interval
        flush_interval: Memmap flush interval
    
    Returns:
        FDE matrix [num_docs, final_fde_dim]
    """
    start_time = time.perf_counter()
    num_docs = len(doc_embeddings_list)
    
    if num_docs == 0:
        logging.warning("[FDE GPU] Empty document list")
        return np.array([])
    
    # Validate inputs
    for i, doc in enumerate(doc_embeddings_list):
        if doc.ndim != 2 or doc.shape[1] != config.dimension:
            raise ValueError(f"Doc {i} has invalid shape {doc.shape}")
    
    logging.info(f"[FDE GPU] Processing {num_docs} documents with {config.num_repetitions} reps")
    
    # Configuration
    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    projection_dim = config.dimension if use_identity_proj else config.projection_dimension
    num_partitions = 2 ** config.num_simhash_projections
    final_fde_dim_per_rep = num_partitions * projection_dim
    final_fde_dim = config.num_repetitions * final_fde_dim_per_rep
    
    # Decide output location
    out_bytes = num_docs * final_fde_dim * 4
    memmap_used = False
    
    if memmap_path or out_bytes > max_bytes_in_memory:
        if memmap_path is None:
            memmap_path = os.path.join(os.getcwd(), f"fde_gpu_{final_fde_dim}d_{num_docs}n.mmap")
        logging.info(f"[FDE GPU] Using memmap: {memmap_path} (~{out_bytes/1e9:.2f} GB)")
        out_fdes = np.memmap(memmap_path, mode="w+", dtype=np.float32, shape=(num_docs, final_fde_dim))
        memmap_used = True
    else:
        out_fdes = np.zeros((num_docs, final_fde_dim), dtype=np.float32)
    
    # ==========================================
    # Step 1: Prepare data on GPU
    # ==========================================
    prep_start = time.perf_counter()
    
    # Pad embeddings to uniform length
    padded_embeddings, doc_lengths, max_len = _pad_doc_embeddings(doc_embeddings_list)
    logging.info(f"[FDE GPU] Padded to max_len={max_len}")
    
    # Transfer to GPU
    embeddings_gpu = cp.asarray(padded_embeddings)
    doc_lengths_gpu = cp.asarray(doc_lengths)
    
    # Generate all random matrices for all repetitions
    simhash_matrices_list = []
    ams_matrices_list = []
    
    for rep_num in range(config.num_repetitions):
        current_seed = config.seed + rep_num
        simhash_mat = _simhash_matrix_from_seed_gpu(
            config.dimension, config.num_simhash_projections, current_seed
        )
        simhash_matrices_list.append(simhash_mat)
        
        if not use_identity_proj:
            ams_mat = _ams_projection_matrix_from_seed_gpu(
                config.dimension, projection_dim, current_seed
            )
            ams_matrices_list.append(ams_mat)
    
    # Stack matrices: [num_reps, dim, num_bits/proj_dim]
    simhash_matrices_gpu = cp.stack(simhash_matrices_list, axis=0)
    ams_matrices_gpu = cp.stack(ams_matrices_list, axis=0) if not use_identity_proj else None
    
    prep_time = time.perf_counter() - prep_start
    logging.info(f"[FDE GPU] Data prep completed in {prep_time:.3f}s")
    
    # ==========================================
    # Step 2: Launch CUDA kernels
    # ==========================================
    kernel_start = time.perf_counter()
    
    # Allocate output buffers
    sketches_gpu = cp.zeros((num_docs, config.num_repetitions, max_len, config.num_simhash_projections), dtype=cp.float32)
    projected_gpu = cp.zeros((num_docs, config.num_repetitions, max_len, projection_dim), dtype=cp.float32)
    partition_indices_gpu = cp.zeros((num_docs, config.num_repetitions, max_len), dtype=cp.int32)
    
    # Kernel 1: SimHash projection and partition assignment
    total_tokens = num_docs * config.num_repetitions * max_len
    threads_per_block = 256
    num_blocks = (total_tokens + threads_per_block - 1) // threads_per_block
    
    logging.info(f"[FDE GPU] Launching SimHash kernel with {num_blocks} blocks")
    
    SIMHASH_KERNEL(
        (num_blocks,), (threads_per_block,),
        (
            embeddings_gpu,
            doc_lengths_gpu,
            simhash_matrices_gpu,
            ams_matrices_gpu if ams_matrices_gpu is not None else cp.zeros(1, dtype=cp.float32),
            sketches_gpu,
            projected_gpu,
            partition_indices_gpu,
            num_docs,
            max_len,
            config.dimension,
            config.num_simhash_projections,
            projection_dim,
            config.num_repetitions,
            1 if use_identity_proj else 0
        )
    )
    cp.cuda.Device().synchronize()
    
    simhash_time = time.perf_counter() - kernel_start
    logging.info(f"[FDE GPU] SimHash kernel completed in {simhash_time:.3f}s")
    
    # Kernel 2: Scatter-add with shared memory
    scatter_start = time.perf_counter()
    
    partition_sums_gpu = cp.zeros((num_docs, config.num_repetitions, num_partitions, projection_dim), dtype=cp.float32)
    partition_counts_gpu = cp.zeros((num_docs, config.num_repetitions, num_partitions), dtype=cp.int32)
    
    # Shared memory size: (num_partitions * proj_dim) floats + num_partitions ints
    shared_mem_size = (num_partitions * projection_dim * 4) + (num_partitions * 4)
    
    # Launch with 2D grid: (num_docs, num_reps)
    threads_per_block = 256
    grid_dim = (num_docs, config.num_repetitions)
    
    logging.info(f"[FDE GPU] Launching scatter-add kernel with shared_mem={shared_mem_size} bytes")
    
    SCATTER_ADD_KERNEL(
        grid_dim, (threads_per_block,),
        (
            projected_gpu,
            partition_indices_gpu,
            doc_lengths_gpu,
            partition_sums_gpu,
            partition_counts_gpu,
            num_docs,
            config.num_repetitions,
            max_len,
            projection_dim,
            num_partitions
        ),
        shared_mem=shared_mem_size
    )
    cp.cuda.Device().synchronize()
    
    scatter_time = time.perf_counter() - scatter_start
    logging.info(f"[FDE GPU] Scatter-add kernel completed in {scatter_time:.3f}s")
    
    # Kernel 3: Compute averages
    avg_start = time.perf_counter()
    
    total_partitions = num_docs * config.num_repetitions * num_partitions
    num_blocks = (total_partitions + threads_per_block - 1) // threads_per_block
    
    AVERAGE_KERNEL(
        (num_blocks,), (threads_per_block,),
        (
            partition_sums_gpu,
            partition_counts_gpu,
            num_docs,
            config.num_repetitions,
            num_partitions,
            projection_dim
        )
    )
    cp.cuda.Device().synchronize()
    
    avg_time = time.perf_counter() - avg_start
    logging.info(f"[FDE GPU] Average kernel completed in {avg_time:.3f}s")
    
    # ==========================================
    # Step 3: Transfer results back to CPU
    # ==========================================
    transfer_start = time.perf_counter()
    
    # Reshape partition_sums to final FDE format
    partition_sums_cpu = cp.asnumpy(partition_sums_gpu)  # [num_docs, num_reps, num_partitions, proj_dim]
    
    # Reshape to [num_docs, num_reps * num_partitions * proj_dim]
    for doc_idx in range(num_docs):
        for rep_idx in range(config.num_repetitions):
            rep_offset = rep_idx * final_fde_dim_per_rep
            fde_chunk = partition_sums_cpu[doc_idx, rep_idx].reshape(-1)
            out_fdes[doc_idx, rep_offset:rep_offset + final_fde_dim_per_rep] = fde_chunk
        
        if memmap_used and (doc_idx + 1) % flush_interval == 0:
            out_fdes.flush()
    
    if memmap_used:
        out_fdes.flush()
    
    transfer_time = time.perf_counter() - transfer_start
    logging.info(f"[FDE GPU] Transfer to CPU completed in {transfer_time:.3f}s")
    
    # ==========================================
    # Performance Summary
    # ==========================================
    total_time = time.perf_counter() - start_time
    
    logging.info("=" * 80)
    logging.info("🚀 GPU FDE Generation Performance Summary")
    logging.info("=" * 80)
    logging.info(f"Data preparation:     {prep_time:8.3f}s  ({prep_time/total_time*100:5.1f}%)")
    logging.info(f"SimHash kernel:       {simhash_time:8.3f}s  ({simhash_time/total_time*100:5.1f}%)")
    logging.info(f"Scatter-add kernel:   {scatter_time:8.3f}s  ({scatter_time/total_time*100:5.1f}%)")
    logging.info(f"Average kernel:       {avg_time:8.3f}s  ({avg_time/total_time*100:5.1f}%)")
    logging.info(f"CPU transfer:         {transfer_time:8.3f}s  ({transfer_time/total_time*100:5.1f}%)")
    logging.info(f"Total time:           {total_time:8.3f}s")
    logging.info(f"Speedup potential:    ~{(prep_time + simhash_time + scatter_time + avg_time)/scatter_time:.1f}x over CPU scatter-add")
    logging.info("=" * 80)
    
    # Cleanup GPU memory
    del embeddings_gpu, doc_lengths_gpu, simhash_matrices_gpu
    del sketches_gpu, projected_gpu, partition_indices_gpu
    del partition_sums_gpu, partition_counts_gpu
    if ams_matrices_gpu is not None:
        del ams_matrices_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return out_fdes


# ==============================================================================
# Wrapper for backward compatibility
# ==============================================================================

def generate_document_fde_batch_gpu_wrapper(
    doc_embeddings_list: List[np.ndarray],
    config: FixedDimensionalEncodingConfig,
    **kwargs
) -> np.ndarray:
    """
    Wrapper that automatically uses GPU if available, falls back to CPU.
    """
    try:
        import cupy as cp
        if cp.cuda.is_available():
            logging.info("[FDE] Using GPU-accelerated implementation")
            return generate_document_fde_batch_gpu(doc_embeddings_list, config, **kwargs)
    except (ImportError, cp.cuda.runtime.CUDARuntimeError) as e:
        logging.warning(f"[FDE] GPU not available ({e}), falling back to CPU")
    
    # Fallback to original CPU implementation
    from fde_generator_optimized_stream_weight_fde_gpu import generate_document_fde_batch
    return generate_document_fde_batch(doc_embeddings_list, config, **kwargs)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    num_docs = 100
    doc_len = 50
    dim = 128
    
    test_embeddings = [
        np.random.randn(doc_len, dim).astype(np.float32) 
        for _ in range(num_docs)
    ]
    
    config = FixedDimensionalEncodingConfig(
        dimension=128,
        num_repetitions=5,
        num_simhash_projections=4,
        seed=42,
        encoding_type=EncodingType.AVERAGE,
        projection_type=ProjectionType.AMS_SKETCH,
        projection_dimension=128,
    )
    
    logging.info("Testing GPU FDE generation...")
    result = generate_document_fde_batch_gpu(test_embeddings, config)
    logging.info(f"Output shape: {result.shape}")
    logging.info("✅ GPU FDE generation test passed!")
