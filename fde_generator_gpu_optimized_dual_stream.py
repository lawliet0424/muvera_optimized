# -*- coding: utf-8 -*-
"""
GPU-Optimized FDE Generator with DUAL CUDA STREAMS
Overlaps CPU↔GPU transfers with kernel computation
Optimized for NVIDIA TITAN V to eliminate transfer bottleneck
"""
import logging
import time
import os
import numpy as np
import cupy as cp
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, List


# ==============================================================================
# CUDA KERNELS (Same as before)
# ==============================================================================

# Kernel 1: SimHash projection and partition assignment
SIMHASH_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void simhash_projection_multi_rep(
    const float* __restrict__ embeddings,
    const int* __restrict__ doc_lengths,
    const float* __restrict__ simhash_matrices,
    const float* __restrict__ ams_matrices,
    float* __restrict__ sketches_out,
    float* __restrict__ projected_out,
    int* __restrict__ partition_indices,
    const int num_docs,
    const int max_len,
    const int dim,
    const int num_bits,
    const int proj_dim,
    const int num_reps,
    const int use_identity
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = num_docs * num_reps * max_len;
    
    if (global_id >= total_tokens) return;
    
    int token_idx = global_id % max_len;
    int temp = global_id / max_len;
    int rep_idx = temp % num_reps;
    int doc_idx = temp / num_reps;
    
    if (token_idx >= doc_lengths[doc_idx]) return;
    
    const float* emb = &embeddings[(doc_idx * max_len + token_idx) * dim];
    const float* simhash_mat = &simhash_matrices[rep_idx * dim * num_bits];
    
    float sketches[32];
    for (int b = 0; b < num_bits; b++) {
        float val = 0.0f;
        for (int d = 0; d < dim; d++) {
            val += emb[d] * simhash_mat[d * num_bits + b];
        }
        sketches[b] = val;
    }
    
    int sketch_offset = ((doc_idx * num_reps + rep_idx) * max_len + token_idx) * num_bits;
    for (int b = 0; b < num_bits; b++) {
        sketches_out[sketch_offset + b] = sketches[b];
    }
    
    unsigned int p_idx = 0;
    for (int b = 0; b < num_bits; b++) {
        unsigned int bit = (sketches[b] > 0.0f) ? 1 : 0;
        p_idx = (p_idx << 1) + (bit ^ (p_idx & 1));
    }
    partition_indices[(doc_idx * num_reps + rep_idx) * max_len + token_idx] = p_idx;
    
    int proj_offset = ((doc_idx * num_reps + rep_idx) * max_len + token_idx) * proj_dim;
    
    if (use_identity) {
        for (int d = 0; d < proj_dim; d++) {
            projected_out[proj_offset + d] = emb[d];
        }
    } else {
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


# Kernel 2: Scatter-add with shared memory
SCATTER_ADD_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void scatter_add_partitions(
    const float* __restrict__ projected,
    const int* __restrict__ partition_indices,
    const int* __restrict__ doc_lengths,
    float* __restrict__ partition_sums,
    int* __restrict__ partition_counts,
    const int num_docs,
    const int num_reps,
    const int max_len,
    const int proj_dim,
    const int num_partitions
) {
    extern __shared__ float shared_mem[];
    float* shared_sums = shared_mem;
    int* shared_counts = (int*)&shared_sums[num_partitions * proj_dim];
    
    int doc_idx = blockIdx.x;
    int rep_idx = blockIdx.y;
    
    if (doc_idx >= num_docs || rep_idx >= num_reps) return;
    
    int doc_len = doc_lengths[doc_idx];
    
    for (int i = threadIdx.x; i < num_partitions * proj_dim; i += blockDim.x) {
        shared_sums[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
        shared_counts[i] = 0;
    }
    __syncthreads();
    
    for (int token_idx = threadIdx.x; token_idx < doc_len; token_idx += blockDim.x) {
        int p_idx = partition_indices[(doc_idx * num_reps + rep_idx) * max_len + token_idx];
        int proj_offset = ((doc_idx * num_reps + rep_idx) * max_len + token_idx) * proj_dim;
        
        atomicAdd(&shared_counts[p_idx], 1);
        
        for (int d = 0; d < proj_dim; d++) {
            atomicAdd(&shared_sums[p_idx * proj_dim + d], projected[proj_offset + d]);
        }
    }
    __syncthreads();
    
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


# Kernel 3: Average computation
AVERAGE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void compute_averages(
    float* __restrict__ partition_sums,
    const int* __restrict__ partition_counts,
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
    """Pad document embeddings to uniform length"""
    doc_lengths = np.array([doc.shape[0] for doc in doc_embeddings_list], dtype=np.int32)
    max_len = int(doc_lengths.max())
    num_docs = len(doc_embeddings_list)
    dim = doc_embeddings_list[0].shape[1]
    
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
# DUAL-STREAM GPU FDE GENERATION (NEW!)
# ==============================================================================

# Global variables for cumulative timing across all batches
_GPU_CUMULATIVE_TIMING = {
    'prep_time': 0.0,
    'simhash_time': 0.0,
    'scatter_time': 0.0,
    'average_time': 0.0,
    'transfer_time': 0.0,
}

def reset_gpu_cumulative_timing():
    """Reset cumulative timing for GPU operations"""
    global _GPU_CUMULATIVE_TIMING
    _GPU_CUMULATIVE_TIMING = {
        'prep_time': 0.0,
        'simhash_time': 0.0,
        'scatter_time': 0.0,
        'average_time': 0.0,
        'transfer_time': 0.0,
    }

def generate_document_fde_batch_gpu_streaming(
    doc_embeddings_list: List[np.ndarray],
    config: FixedDimensionalEncodingConfig,
    *,
    memmap_path: Optional[str] = None,
    max_bytes_in_memory: int = 2 * 1024**3,
    log_every: int = 1000,
    flush_interval: int = 1000,
    mini_batch_size: int = 500  # Ignored - kept for backward compatibility
) -> np.ndarray:
    """
    🚀 GPU FDE generation processing all documents in a single batch!
    
    Strategy:
    1. Process all documents in one batch (no mini-batching)
    2. Use single CUDA stream for computation
    3. Transfer results to CPU after all computation is done
    
    Args:
        mini_batch_size: Ignored (kept for backward compatibility)
    """
    start_time = time.perf_counter()
    num_docs = len(doc_embeddings_list)
    
    if num_docs == 0:
        logging.warning("[FDE GPU Streaming] Empty document list")
        return np.array([])
    
    # Validate
    for i, doc in enumerate(doc_embeddings_list):
        if doc.ndim != 2 or doc.shape[1] != config.dimension:
            raise ValueError(f"Doc {i} has invalid shape {doc.shape}")
    
    logging.info(f"[FDE GPU Streaming] Processing {num_docs} documents with {config.num_repetitions} reps")
    logging.info(f"[FDE GPU Streaming] Processing all documents in a single batch (no mini-batching)")
    
    # Configuration
    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    projection_dim = config.dimension if use_identity_proj else config.projection_dimension
    num_partitions = 2 ** config.num_simhash_projections
    final_fde_dim_per_rep = num_partitions * projection_dim
    final_fde_dim = config.num_repetitions * final_fde_dim_per_rep

    # Output allocation
    out_bytes = num_docs * final_fde_dim * 4
    memmap_used = False
    
    if memmap_path or out_bytes > max_bytes_in_memory:
        if memmap_path is None:
            memmap_path = os.path.join(os.getcwd(), f"fde_gpu_stream_{final_fde_dim}d_{num_docs}n.mmap")
        logging.info(f"[FDE GPU Streaming] Using memmap: {memmap_path} (~{out_bytes/1e9:.2f} GB)")
        out_fdes = np.memmap(memmap_path, mode="w+", dtype=np.float32, shape=(num_docs, final_fde_dim))
        memmap_used = True
    else:
        out_fdes = np.zeros((num_docs, final_fde_dim), dtype=np.float32)
    
    # ==========================================
    # Create CUDA stream for computation
    # ==========================================
    stream_compute = cp.cuda.Stream()
    
    # Generate random matrices
    prep_start = time.perf_counter()
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
    
    simhash_matrices_gpu = cp.stack(simhash_matrices_list, axis=0)
    ams_matrices_gpu = cp.stack(ams_matrices_list, axis=0) if not use_identity_proj else None
    
    prep_time = time.perf_counter() - prep_start
    logging.info(f"[FDE GPU Streaming] Random matrices prepared in {prep_time:.3f}s")
    
    # ==========================================
    # Process all documents in a single batch
    # ==========================================
    logging.info(f"[FDE GPU Streaming] Processing all {num_docs} documents in one batch")
    
    # Detailed kernel timing
    total_simhash_time = 0.0
    total_scatter_time = 0.0
    total_average_time = 0.0
    total_transfer_time = 0.0
    
    # ========================================
    # STEP 1: Prepare all document data
    # ========================================
    padded_embeddings, doc_lengths, max_len = _pad_doc_embeddings(doc_embeddings_list)
    
    # ========================================
    # STEP 2: Upload to GPU
    # ========================================
    with stream_compute:
        embeddings_gpu = cp.asarray(padded_embeddings)
        doc_lengths_gpu = cp.asarray(doc_lengths)
        
        # Allocate output buffers
        sketches_gpu = cp.zeros((num_docs, config.num_repetitions, max_len, config.num_simhash_projections), dtype=cp.float32)
        projected_gpu = cp.zeros((num_docs, config.num_repetitions, max_len, projection_dim), dtype=cp.float32)
        partition_indices_gpu = cp.zeros((num_docs, config.num_repetitions, max_len), dtype=cp.int32)
    
    # ========================================
    # STEP 3: Launch kernels
    # ========================================
    compute_start = time.perf_counter()
    
    with stream_compute:
        # Kernel 1: SimHash + Projection
        simhash_start = time.perf_counter()
        
        total_tokens = num_docs * config.num_repetitions * max_len
        threads_per_block = 256
        num_blocks = (total_tokens + threads_per_block - 1) // threads_per_block
        
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
        stream_compute.synchronize()
        simhash_time = time.perf_counter() - simhash_start
        total_simhash_time = simhash_time
        
        # Kernel 2: Scatter-add
        scatter_start = time.perf_counter()
        
        partition_sums_gpu = cp.zeros((num_docs, config.num_repetitions, num_partitions, projection_dim), dtype=cp.float32)
        partition_counts_gpu = cp.zeros((num_docs, config.num_repetitions, num_partitions), dtype=cp.int32)
        
        shared_mem_size = (num_partitions * projection_dim * 4) + (num_partitions * 4)
        grid_dim = (num_docs, config.num_repetitions)
        
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
        stream_compute.synchronize()
        scatter_time = time.perf_counter() - scatter_start
        total_scatter_time = scatter_time
        
        # Kernel 3: Average
        average_start = time.perf_counter()
        
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
        stream_compute.synchronize()
        average_time = time.perf_counter() - average_start
        total_average_time = average_time
    
    compute_time = simhash_time + scatter_time + average_time
    
    # ========================================
    # STEP 4: Transfer results to CPU
    # ========================================
    transfer_start = time.perf_counter()
    
    partition_sums_cpu = cp.asnumpy(partition_sums_gpu)
    
    # Write to output
    for doc_idx in range(num_docs):
        for rep_idx in range(config.num_repetitions):
            rep_offset = rep_idx * final_fde_dim_per_rep
            fde_chunk = partition_sums_cpu[doc_idx, rep_idx].reshape(-1)
            out_fdes[doc_idx, rep_offset:rep_offset + final_fde_dim_per_rep] = fde_chunk
    
    if memmap_used:
        out_fdes.flush()
    
    transfer_time = time.perf_counter() - transfer_start
    total_transfer_time = transfer_time
    
    # Cleanup intermediate buffers
    del embeddings_gpu, doc_lengths_gpu, sketches_gpu, projected_gpu, partition_indices_gpu
    del partition_counts_gpu, partition_sums_gpu
    
    # ==========================================
    # Performance Summary (Detailed Breakdown)
    # ==========================================
    total_time = time.perf_counter() - start_time
    
    # Calculate total measured time for this batch
    total_measured_time = prep_time + total_simhash_time + total_scatter_time + total_average_time + total_transfer_time
    
    # Accumulate this batch's times to global cumulative timing
    global _GPU_CUMULATIVE_TIMING
    _GPU_CUMULATIVE_TIMING['prep_time'] += prep_time
    _GPU_CUMULATIVE_TIMING['simhash_time'] += total_simhash_time
    _GPU_CUMULATIVE_TIMING['scatter_time'] += total_scatter_time
    _GPU_CUMULATIVE_TIMING['average_time'] += total_average_time
    _GPU_CUMULATIVE_TIMING['transfer_time'] += total_transfer_time
    
    # Get cumulative times (across all batches)
    cumul_prep = _GPU_CUMULATIVE_TIMING['prep_time']
    cumul_simhash = _GPU_CUMULATIVE_TIMING['simhash_time']
    cumul_scatter = _GPU_CUMULATIVE_TIMING['scatter_time']
    cumul_average = _GPU_CUMULATIVE_TIMING['average_time']
    cumul_transfer = _GPU_CUMULATIVE_TIMING['transfer_time']
    
    # Try to get flush time from global CUMULATIVE_TIMING if available (from index method)
    # Use sys.modules to avoid circular import
    flush_time = 0.0
    try:
        import sys
        # Try multiple possible module names
        module_names = [
            'main_weight_fde_gpu_dual_stream',
            '__main__',
        ]
        
        main_module = None
        for mod_name in module_names:
            if mod_name in sys.modules:
                main_module = sys.modules[mod_name]
                # Check if this module has CUMULATIVE_TIMING (it's the right module)
                if hasattr(main_module, 'CUMULATIVE_TIMING'):
                    break
                main_module = None
        
        if main_module is None:
            # Fallback: search all loaded modules
            for mod_name, mod in sys.modules.items():
                if hasattr(mod, 'CUMULATIVE_TIMING') and hasattr(mod, 'TIMING'):
                    main_module = mod
                    logging.info(f"[DEBUG] Found module with CUMULATIVE_TIMING: {mod_name}")
                    break
        
        if main_module is not None:
            # Check CUMULATIVE_TIMING first (accumulated across all batches)
            if hasattr(main_module, 'CUMULATIVE_TIMING'):
                flush_time = main_module.CUMULATIVE_TIMING.get('flush', 0.0)
                logging.info(f"[DEBUG] Retrieved flush time from CUMULATIVE_TIMING: {flush_time:.3f}s")
            # Fallback to TIMING if CUMULATIVE_TIMING doesn't have it
            if flush_time == 0.0 and hasattr(main_module, 'TIMING'):
                flush_time = main_module.TIMING.get('flush', 0.0)
                if flush_time > 0.0:
                    logging.info(f"[DEBUG] Retrieved flush time from TIMING: {flush_time:.3f}s")
        else:
            logging.warning("[DEBUG] Could not find module with CUMULATIVE_TIMING in sys.modules")
            # List available modules for debugging
            available_modules = [name for name in sys.modules.keys() if 'main_weight' in name or 'fde' in name.lower()]
            logging.warning(f"[DEBUG] Available modules with 'main_weight' or 'fde': {available_modules}")
    except (AttributeError, KeyError, Exception) as e:
        logging.warning(f"[DEBUG] Could not get flush time: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        pass
    
    # Calculate total cumulative time (across all batches)
    total_measured_with_flush = cumul_prep + cumul_simhash + cumul_scatter + cumul_average + cumul_transfer + flush_time
    
    logging.info("=" * 80)
    logging.info("🚀 GPU FDE Performance Summary (Single Batch)")
    logging.info("=" * 80)
    logging.info(f"Data preparation:    {prep_time:8.3f}s  ({prep_time/total_time*100:5.1f}%)")
    logging.info(f"SimHash kernel:      {total_simhash_time:8.3f}s  ({total_simhash_time/total_time*100:5.1f}%)")
    logging.info(f"Scatter-add kernel:  {total_scatter_time:8.3f}s  ({total_scatter_time/total_time*100:5.1f}%)")
    logging.info(f"Average kernel:      {total_average_time:8.3f}s  ({total_average_time/total_time*100:5.1f}%)")
    logging.info(f"CPU transfer:        {total_transfer_time:8.3f}s  ({total_transfer_time/total_time*100:5.1f}%)")
    logging.info(f"Flush:               {flush_time:8.3f}s  ({flush_time/total_time*100:5.1f}%)" if flush_time > 0.0 else f"Flush:               {flush_time:8.3f}s  (  0.0%)")
    logging.info(f"Total time:          {total_time:8.3f}s")
    logging.info("=" * 80)
    logging.info("📊 Cumulative Time Breakdown (All Operations - Across All Batches):")
    logging.info("-" * 80)
    logging.info(f"   Data preparation (cumulative):    {cumul_prep:8.3f}s")
    logging.info(f"   SimHash kernel (cumulative):      {cumul_simhash:8.3f}s")
    logging.info(f"   Scatter-add kernel (cumulative): {cumul_scatter:8.3f}s")
    logging.info(f"   Average kernel (cumulative):     {cumul_average:8.3f}s")
    logging.info(f"   CPU transfer (cumulative):       {cumul_transfer:8.3f}s")
    logging.info(f"   Flush (cumulative):              {flush_time:8.3f}s")
    logging.info("-" * 80)
    logging.info(f"   Total cumulative time:            {total_measured_with_flush:8.3f}s")
    logging.info("=" * 80)
    
    # Cleanup
    del simhash_matrices_gpu
    if ams_matrices_gpu is not None:
        del ams_matrices_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    return out_fdes


# ==============================================================================
# Wrapper with automatic mini-batch size tuning
# ==============================================================================

def generate_document_fde_batch_gpu_wrapper(
    doc_embeddings_list: List[np.ndarray],
    config: FixedDimensionalEncodingConfig,
    **kwargs
) -> np.ndarray:
    """
    Smart wrapper that uses dual-stream GPU if available.
    
    Automatically tunes mini-batch size based on GPU memory.
    """
    try:
        import cupy as cp
        if cp.cuda.is_available():
            # Auto-tune mini-batch size based on GPU memory
            device = cp.cuda.Device(0)
            free_mem, total_mem = device.mem_info
            
            # Use ~30% of free memory per mini-batch
            # Estimate: each doc uses ~(max_len * dim * num_reps * 8) bytes
            avg_len = np.mean([doc.shape[0] for doc in doc_embeddings_list])
            bytes_per_doc = avg_len * config.dimension * config.num_repetitions * 8
            
            optimal_batch_size = int((free_mem * 0.3) / bytes_per_doc)
            optimal_batch_size = max(100, min(optimal_batch_size, 1000))  # Clamp to 100-1000
            
            logging.info(f"[GPU Wrapper] Auto-tuned mini-batch size: {optimal_batch_size}")
            
            return generate_document_fde_batch_gpu_streaming(
                doc_embeddings_list,
                config,
                mini_batch_size=optimal_batch_size,
                **kwargs
            )
    except (ImportError, cp.cuda.runtime.CUDARuntimeError) as e:
        logging.warning(f"[GPU Wrapper] GPU not available ({e}), falling back to CPU")
    
    # Fallback to CPU
    from fde_generator_optimized_stream_weight_fde_gpu import generate_document_fde_batch
    return generate_document_fde_batch(doc_embeddings_list, config, **kwargs)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    num_docs = 1000
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
    
    logging.info("Testing DUAL-STREAM GPU FDE generation...")
    result = generate_document_fde_batch_gpu_streaming(
        test_embeddings, 
        config,
        mini_batch_size=500
    )
    logging.info(f"Output shape: {result.shape}")
    logging.info("✅ Dual-stream GPU FDE test passed!")