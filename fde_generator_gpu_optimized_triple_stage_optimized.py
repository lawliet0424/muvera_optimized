# -*- coding: utf-8 -*-
import logging
import time
import os
import numpy as np
import cupy as cp
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum

# ==============================================================================
# CUDA KERNELS 
# ==============================================================================

# Kernel 1: SimHash partition assignment
SIMHASH_PARTITION_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void simhash_partition_multi_rep(
    const float* __restrict__ sketches_out,
    const int* __restrict__ doc_lengths,
    int* __restrict__ partition_indices,
    const int num_docs,
    const int max_len,
    const int num_bits,
    const int num_reps
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = num_docs * num_reps * max_len;
    if (global_id >= total_tokens) return;

    int token_idx = global_id % max_len;
    int temp = global_id / max_len;
    int rep_idx = temp % num_reps;
    int doc_idx = temp / num_reps;

    if (token_idx >= doc_lengths[doc_idx]) return;

    int sketch_offset = ((doc_idx * num_reps + rep_idx) * max_len + token_idx) * num_bits;
    const float* sketches = &sketches_out[sketch_offset];

    unsigned int p_idx = 0;
    for (int b = 0; b < num_bits; b++) {
        unsigned int bit = (sketches[b] > 0.0f) ? 1 : 0;
        p_idx = (p_idx << 1) + (bit ^ (p_idx & 1));
    }

    partition_indices[(doc_idx * num_reps + rep_idx) * max_len + token_idx] = p_idx;
}
''', 'simhash_partition_multi_rep')


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

# Kernel 4: Fill empty partitions (sketch Hamming vs gray->binary target bits; matches CPU)
FILL_EMPTY_PARTITIONS_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void fill_empty_partitions(
    float* __restrict__ partition_sums,
    const int* __restrict__ partition_counts,
    const float* __restrict__ sketches,
    const float* __restrict__ projected,
    const int* __restrict__ doc_lengths,
    const unsigned char* __restrict__ partition_bits_table,
    const int num_docs,
    const int num_reps,
    const int num_partitions,
    const int max_len,
    const int num_bits,
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
    if (partition_counts[count_offset] > 0) return;

    int doc_len = doc_lengths[doc_idx];
    if (doc_len == 0) return;

    const unsigned char* target_bits = &partition_bits_table[p_idx * num_bits];
    int min_dist = num_bits + 1;
    int nearest_token_idx = -1;

    for (int token_idx = 0; token_idx < doc_len; token_idx++) {
        int sketch_offset = ((doc_idx * num_reps + rep_idx) * max_len + token_idx) * num_bits;
        const float* sketch = &sketches[sketch_offset];

        int dist = 0;
        for (int b = 0; b < num_bits; b++) {
            unsigned char sketch_bit = (sketch[b] > 0.0f) ? 1 : 0;
            if (sketch_bit != target_bits[b]) {
                dist++;
            }
        }

        if (dist < min_dist) {
            min_dist = dist;
            nearest_token_idx = token_idx;
        }
    }

    if (nearest_token_idx >= 0) {
        int proj_offset = ((doc_idx * num_reps + rep_idx) * max_len + nearest_token_idx) * proj_dim;
        int sum_offset = count_offset * proj_dim;
        for (int d = 0; d < proj_dim; d++) {
            partition_sums[sum_offset + d] = projected[proj_offset + d];
        }
    }
}
''', 'fill_empty_partitions')


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


def _pad_doc_embeddings(doc_embeddings_list: List[np.ndarray]) -> tuple:
    #Pad document embeddings to uniform length
    doc_lengths = np.array([doc.shape[0] for doc in doc_embeddings_list], dtype=np.int32)
    max_len = int(doc_lengths.max())
    num_docs = len(doc_embeddings_list)
    dim = doc_embeddings_list[0].shape[1]
    
    padded = np.zeros((num_docs, max_len, dim), dtype=np.float32)
    for i, doc in enumerate(doc_embeddings_list):
        padded[i, :doc.shape[0], :] = doc
    
    return padded, doc_lengths, max_len

# ==============================================================================
# Helper functions
# ==============================================================================

def _simhash_matrix_from_seed(
    dimension: int, num_projections: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(dimension, num_projections)).astype(
        np.float32
    )

def _ams_projection_matrix_from_seed(
    dimension: int, projection_dim: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((dimension, projection_dim), dtype=np.float32)
    indices = rng.integers(0, projection_dim, size=dimension)
    signs = rng.choice([-1.0, 1.0], size=dimension)
    out[np.arange(dimension), indices] = signs
    return out

def _apply_count_sketch_to_vector(
    input_vector: np.ndarray, final_dimension: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros(final_dimension, dtype=np.float32)
    indices = rng.integers(0, final_dimension, size=input_vector.shape[0])
    signs = rng.choice([-1.0, 1.0], size=input_vector.shape[0])
    np.add.at(out, indices, signs * input_vector)
    return out

def _partition_bits_table(num_bits: int) -> np.ndarray:
    """
    Returns an array of shape [num_partitions, num_bits] with binary bits (0/1)
    corresponding to *binary* code for each partition index where the index was
    originally generated as Gray code (Gray->Binary performed here).
    """
    P = 1 << num_bits
    gray = np.arange(P, dtype=np.uint32)
    binary = gray.copy()
    g = gray.copy()
    # vectorized gray->binary via iterative XOR with right shift
    while True:
        g >>= 1
        if not g.any():
            break
        binary ^= g
    shifts = np.arange(num_bits - 1, -1, -1, dtype=np.uint32)
    bits = ((binary[:, None] >> shifts[None, :]) & 1).astype(np.uint8)  # [P, b]
    return bits

# ==============================================================================
# Global variables for cumulative timing across all batches
# ==============================================================================
_GPU_CUMULATIVE_TIMING = {
    'prep_time': 0.0,
    'upload_time': 0.0,
    'simhash_time': 0.0,
    'partition_time': 0.0,
    'scatter_time': 0.0,
    'average_time': 0.0,
    'fill_time': 0.0,
    'compute_time': 0.0,
    'download_time': 0.0,
    'reshape_time': 0.0,
    'flush_time': 0.0,
}

def reset_gpu_cumulative_timing():
    """Reset cumulative timing for GPU operations"""
    global _GPU_CUMULATIVE_TIMING
    _GPU_CUMULATIVE_TIMING = {
        'prep_time': 0.0,
        'upload_time': 0.0,
        'simhash_time': 0.0,
        'partition_time': 0.0,
        'scatter_time': 0.0,
        'average_time': 0.0,
        'fill_time': 0.0,
        'compute_time': 0.0,
        'download_time': 0.0,
        'reshape_time': 0.0,
        'flush_time': 0.0,
    }

def generate_query_fde_gpu(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    """
    Query FDE on GPU: GEMM (SimHash + projection) + kernel 1 (partition) + kernel 2 (SUM).
    No average (kernel 3) or fill-empty (kernel 4).
    """
    if config.fill_empty_partitions:
        raise ValueError(
            "Query FDE generation does not support 'fill_empty_partitions'."
        )
    if point_cloud.ndim != 2 or point_cloud.shape[1] != config.dimension:
        raise ValueError(
            f"Input shape {point_cloud.shape} inconsistent with dimension {config.dimension}."
        )
    if not (0 <= config.num_simhash_projections < 32):
        raise ValueError(
            f"num_simhash_projections must be in [0, 31]: {config.num_simhash_projections}"
        )

    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    projection_dim = (
        config.dimension if use_identity_proj else config.projection_dimension
    )
    if not use_identity_proj and (not projection_dim or projection_dim <= 0):
        raise ValueError(
            "A positive projection_dimension is required for non-identity projections."
        )

    num_tokens = point_cloud.shape[0]
    num_docs = 1
    max_len = num_tokens
    num_bits = config.num_simhash_projections
    reps = config.num_repetitions
    num_partitions = 2 ** num_bits
    final_fde_dim_per_rep = num_partitions * projection_dim
    final_fde_dim = reps * final_fde_dim_per_rep

    simhash_matrices_list = []
    ams_matrices_list = []
    for rep_num in range(reps):
        current_seed = config.seed + rep_num
        simhash_matrices_list.append(
            cp.asarray(
                _simhash_matrix_from_seed(
                    config.dimension, num_bits, current_seed
                ),
                dtype=cp.float32,
            )
        )
        if not use_identity_proj:
            ams_matrices_list.append(
                cp.asarray(
                    _ams_projection_matrix_from_seed(
                        config.dimension, projection_dim, current_seed
                    ),
                    dtype=cp.float32,
                )
            )
    simhash_matrices_gpu = cp.stack(simhash_matrices_list, axis=0)
    ams_matrices_gpu = (
        cp.stack(ams_matrices_list, axis=0) if not use_identity_proj else None
    )

    embeddings_gpu = cp.asarray(
        point_cloud.astype(np.float32, copy=False)[np.newaxis, :, :]
    )
    doc_lengths_gpu = cp.asarray(np.array([num_tokens], dtype=np.int32))

    sketches_gpu = cp.zeros((num_docs, reps, max_len, num_bits), dtype=cp.float32)
    projected_gpu = cp.zeros(
        (num_docs, reps, max_len, projection_dim), dtype=cp.float32
    )
    partition_indices_gpu = cp.zeros((num_docs, reps, max_len), dtype=cp.int32)
    partition_sums_gpu = cp.zeros(
        (num_docs, reps, num_partitions, projection_dim), dtype=cp.float32
    )
    partition_counts_gpu = cp.zeros(
        (num_docs, reps, num_partitions), dtype=cp.int32
    )

    dim = config.dimension
    embeddings_2d = embeddings_gpu.reshape(num_docs * max_len, dim)

    for rep_idx in range(reps):
        sketches_rep = embeddings_2d @ simhash_matrices_gpu[rep_idx]
        sketches_gpu[:, rep_idx, :, :] = sketches_rep.reshape(
            num_docs, max_len, num_bits
        )
        if use_identity_proj:
            projected_gpu[:, rep_idx, :, :] = embeddings_gpu
        else:
            proj_rep = embeddings_2d @ ams_matrices_gpu[rep_idx]
            projected_gpu[:, rep_idx, :, :] = proj_rep.reshape(
                num_docs, max_len, projection_dim
            )

    threads_per_block = 256
    total_tokens_all_reps = num_docs * reps * max_len
    num_blocks = (total_tokens_all_reps + threads_per_block - 1) // threads_per_block

    SIMHASH_PARTITION_KERNEL(
        (num_blocks,),
        (threads_per_block,),
        (
            sketches_gpu,
            doc_lengths_gpu,
            partition_indices_gpu,
            num_docs,
            max_len,
            num_bits,
            reps,
        ),
    )

    shared_mem_size = (num_partitions * projection_dim * 4) + (num_partitions * 4)
    SCATTER_ADD_KERNEL(
        (num_docs, reps),
        (threads_per_block,),
        (
            projected_gpu,
            partition_indices_gpu,
            doc_lengths_gpu,
            partition_sums_gpu,
            partition_counts_gpu,
            num_docs,
            reps,
            max_len,
            projection_dim,
            num_partitions,
        ),
        shared_mem=shared_mem_size,
    )

    cp.cuda.Stream.null.synchronize()

    partition_sums_cpu = cp.asnumpy(partition_sums_gpu)
    out_fde = np.zeros(final_fde_dim, dtype=np.float32)
    for rep_idx in range(reps):
        rep_offset = rep_idx * final_fde_dim_per_rep
        out_fde[rep_offset : rep_offset + final_fde_dim_per_rep] = (
            partition_sums_cpu[0, rep_idx].reshape(-1)
        )

    if config.final_projection_dimension and config.final_projection_dimension > 0:
        out_fde = _apply_count_sketch_to_vector(
            out_fde, config.final_projection_dimension, config.seed
        )

    return out_fde


def generate_query_fde(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    """Alias for GPU query FDE (SUM via kernel 1 + 2)."""
    return generate_query_fde_gpu(point_cloud, config)


def generate_document_fde_batch_gpu_3stage(
    doc_embeddings_list: List[np.ndarray],
    config: FixedDimensionalEncodingConfig,
    fde_memmap,  # Pre-created memmap from main code
    batch_start_idx: int,  # Where to write in memmap
    *,
    mini_batch_size: int = 500,  # Ignored - kept for backward compatibility
    log_every: int = 1000
) -> dict:

    start_time = time.perf_counter()
    num_docs = len(doc_embeddings_list)
    
    if num_docs == 0:
        logging.warning("[FDE 3-Stream] Empty document list")
        return {}
    
    logging.info(f"[FDE 3-Stream] Processing {num_docs} documents with 3-stream pipeline")
    logging.info(f"[FDE 3-Stream] Processing all documents in a single batch (no mini-batching)")
    
    # Configuration
    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    projection_dim = config.dimension if use_identity_proj else config.projection_dimension
    num_partitions = 2 ** config.num_simhash_projections
    final_fde_dim_per_rep = num_partitions * projection_dim
    final_fde_dim = config.num_repetitions * final_fde_dim_per_rep
    
    # ==========================================
    # Random matrices preparation (shared across all batches)
    # ==========================================
    prep_start = time.perf_counter()
    
    simhash_matrices_list = []
    ams_matrices_list = []

    for rep_num in range(config.num_repetitions):
        current_seed = config.seed + rep_num
        simhash_mat = cp.asarray(
            _simhash_matrix_from_seed(
                config.dimension, config.num_simhash_projections, current_seed
            ),
            dtype=cp.float32,
        )
        simhash_matrices_list.append(simhash_mat)

        if not use_identity_proj:
            ams_mat = cp.asarray(
                _ams_projection_matrix_from_seed(
                    config.dimension, projection_dim, current_seed
                ),
                dtype=cp.float32,
            )
            ams_matrices_list.append(ams_mat)

    simhash_matrices_gpu = cp.stack(simhash_matrices_list, axis=0)
    ams_matrices_gpu = cp.stack(ams_matrices_list, axis=0) if not use_identity_proj else None

    prep_time = time.perf_counter() - prep_start
    logging.info(f"[FDE 3-Stream] Random matrices prepared in {prep_time:.3f}s")
    
    # ==========================================
    # Create 3 CUDA streams
    # ==========================================
    stream_upload = cp.cuda.Stream(non_blocking=True)    # Stream 1: Upload
    stream_compute = cp.cuda.Stream(non_blocking=True)   # Stream 2: Compute
    stream_download = cp.cuda.Stream(non_blocking=True)  # Stream 3: Download
    
    # ==========================================
    # Process all documents in a single batch
    # ==========================================
    logging.info(f"[FDE 3-Stream] Processing all {num_docs} documents in one batch")
    
    # Timing accumulators
    upload_time = 0.0
    simhash_time = 0.0
    partition_time = 0.0
    scatter_time = 0.0
    average_time = 0.0
    fill_time = 0.0
    compute_time = 0.0
    download_time = 0.0
    reshape_time = 0.0
    flush_time = 0.0
    
    # ========================================
    # STEP 1: Prepare all document data
    # ========================================
    padded_embeddings, doc_lengths, max_len = _pad_doc_embeddings(doc_embeddings_list)
    
    # ========================================
    # STEP 2: Upload to GPU (Stream 1)
    # ========================================
    upload_start = time.perf_counter()
    
    with stream_upload:
        embeddings_gpu = cp.asarray(padded_embeddings)
        doc_lengths_gpu = cp.asarray(doc_lengths)

        sketches_gpu = cp.zeros(
            (num_docs, config.num_repetitions, max_len, config.num_simhash_projections),
            dtype=cp.float32,
        )
        projected_gpu = cp.zeros(
            (num_docs, config.num_repetitions, max_len, projection_dim), dtype=cp.float32
        )
        partition_indices_gpu = cp.zeros(
            (num_docs, config.num_repetitions, max_len), dtype=cp.int32
        )
        partition_sums_gpu = cp.zeros(
            (num_docs, config.num_repetitions, num_partitions, projection_dim),
            dtype=cp.float32,
        )
        partition_counts_gpu = cp.zeros(
            (num_docs, config.num_repetitions, num_partitions), dtype=cp.int32
        )

    stream_upload.synchronize()
    upload_time = time.perf_counter() - upload_start
    
    # ========================================
    # STEP 3: Compute on GPU (Stream 2)
    # ========================================
    with stream_compute:
        simhash_start = time.perf_counter()

        total_tokens = num_docs * max_len
        dim = config.dimension
        num_bits = config.num_simhash_projections
        reps = config.num_repetitions

        embeddings_2d = embeddings_gpu.reshape(total_tokens, dim)

        for rep_idx in range(reps):
            simhash_mat_rep = simhash_matrices_gpu[rep_idx]
            sketches_rep = embeddings_2d @ simhash_mat_rep
            sketches_gpu[:, rep_idx, :, :] = sketches_rep.reshape(num_docs, max_len, num_bits)

            if use_identity_proj:
                projected_gpu[:, rep_idx, :, :] = embeddings_gpu
            else:
                ams_mat_rep = ams_matrices_gpu[rep_idx]
                proj_rep = embeddings_2d @ ams_mat_rep
                projected_gpu[:, rep_idx, :, :] = proj_rep.reshape(
                    num_docs, max_len, projection_dim
                )

        stream_compute.synchronize()
        simhash_time = time.perf_counter() - simhash_start

        partition_start = time.perf_counter()

        total_tokens_all_reps = num_docs * reps * max_len
        threads_per_block = 256
        num_blocks = (total_tokens_all_reps + threads_per_block - 1) // threads_per_block

        SIMHASH_PARTITION_KERNEL(
            (num_blocks,),
            (threads_per_block,),
            (
                sketches_gpu,
                doc_lengths_gpu,
                partition_indices_gpu,
                num_docs,
                max_len,
                num_bits,
                reps,
            ),
        )
        stream_compute.synchronize()
        partition_time = time.perf_counter() - partition_start

        scatter_start = time.perf_counter()

        shared_mem_size = (num_partitions * projection_dim * 4) + (num_partitions * 4)
        grid_dim = (num_docs, config.num_repetitions)

        SCATTER_ADD_KERNEL(
            grid_dim,
            (threads_per_block,),
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
                num_partitions,
            ),
            shared_mem=shared_mem_size,
        )
        stream_compute.synchronize()
        scatter_time = time.perf_counter() - scatter_start

        average_start = time.perf_counter()

        total_partitions = num_docs * config.num_repetitions * num_partitions
        num_blocks = (total_partitions + threads_per_block - 1) // threads_per_block

        AVERAGE_KERNEL(
            (num_blocks,),
            (threads_per_block,),
            (
                partition_sums_gpu,
                partition_counts_gpu,
                num_docs,
                config.num_repetitions,
                num_partitions,
                projection_dim,
            ),
        )
        stream_compute.synchronize()
        average_time = time.perf_counter() - average_start

        if config.fill_empty_partitions:
            fill_start = time.perf_counter()

            part_bits_tbl_gpu = cp.asarray(
                _partition_bits_table(config.num_simhash_projections),
                dtype=cp.uint8,
            )
            FILL_EMPTY_PARTITIONS_KERNEL(
                (num_blocks,),
                (threads_per_block,),
                (
                    partition_sums_gpu,
                    partition_counts_gpu,
                    sketches_gpu,
                    projected_gpu,
                    doc_lengths_gpu,
                    part_bits_tbl_gpu,
                    num_docs,
                    config.num_repetitions,
                    num_partitions,
                    max_len,
                    num_bits,
                    projection_dim,
                ),
            )
            stream_compute.synchronize()
            fill_time = time.perf_counter() - fill_start

    compute_time = simhash_time + partition_time + scatter_time + average_time + fill_time
    
    # ========================================
    # STEP 4: Download to CPU (Stream 3)
    # ========================================
    download_start = time.perf_counter()
    
    with stream_download:
        partition_sums_cpu = cp.asnumpy(partition_sums_gpu)
    
    stream_download.synchronize()
    download_time = time.perf_counter() - download_start
    
    # ========================================
    # STEP 5: Reshape on CPU
    # ========================================
    reshape_start = time.perf_counter()
    
    fde_cpu = np.zeros((num_docs, final_fde_dim), dtype=np.float32)
    for doc_idx in range(num_docs):
        for rep_idx in range(config.num_repetitions):
            rep_offset = rep_idx * final_fde_dim_per_rep
            fde_chunk = partition_sums_cpu[doc_idx, rep_idx].reshape(-1)
            fde_cpu[doc_idx, rep_offset : rep_offset + final_fde_dim_per_rep] = fde_chunk
    
    reshape_time = time.perf_counter() - reshape_start
    
    # ========================================
    # STEP 6: Optional final count-sketch projection (CPU, matches stream batch)
    # ========================================
    if config.final_projection_dimension and config.final_projection_dimension > 0:
        target_dim = config.final_projection_dimension
        if fde_memmap.shape[1] != target_dim:
            raise ValueError(
                f"fde_memmap width {fde_memmap.shape[1]} must equal "
                f"final_projection_dimension {target_dim}"
            )
        for doc_idx in range(num_docs):
            fde_cpu[doc_idx] = _apply_count_sketch_to_vector(
                fde_cpu[doc_idx], target_dim, config.seed
            )

    # ========================================
    # STEP 7: Write to memmap and flush
    # ========================================
    flush_start = time.perf_counter()

    write_width = (
        config.final_projection_dimension
        if config.final_projection_dimension and config.final_projection_dimension > 0
        else final_fde_dim
    )
    fde_memmap[batch_start_idx:batch_start_idx + num_docs, :write_width] = (
        fde_cpu[:, :write_width]
    )
    fde_memmap.flush()  # Flush to disk
    
    flush_time = time.perf_counter() - flush_start
    
    logging.info(f"[FDE 3-Stream] Completed: upload={upload_time:.3f}s, compute={compute_time:.3f}s, download={download_time:.3f}s, reshape={reshape_time:.3f}s, flush={flush_time:.3f}s")
    
    # ==========================================
    # Performance Summary
    # ==========================================
    total_time = time.perf_counter() - start_time
    
    # Accumulate this batch's times to global cumulative timing
    global _GPU_CUMULATIVE_TIMING
    _GPU_CUMULATIVE_TIMING['prep_time'] += prep_time
    _GPU_CUMULATIVE_TIMING['upload_time'] += upload_time
    _GPU_CUMULATIVE_TIMING['simhash_time'] += simhash_time
    _GPU_CUMULATIVE_TIMING['partition_time'] += partition_time
    _GPU_CUMULATIVE_TIMING['scatter_time'] += scatter_time
    _GPU_CUMULATIVE_TIMING['average_time'] += average_time
    _GPU_CUMULATIVE_TIMING['fill_time'] += fill_time
    _GPU_CUMULATIVE_TIMING['compute_time'] += compute_time
    _GPU_CUMULATIVE_TIMING['download_time'] += download_time
    _GPU_CUMULATIVE_TIMING['reshape_time'] += reshape_time
    _GPU_CUMULATIVE_TIMING['flush_time'] += flush_time
    
    # Get cumulative times (across all batches) - use dictionary directly to avoid duplication
    cumul = _GPU_CUMULATIVE_TIMING
    
    # Final memmap flush time from main_weight module
    final_memmap_flush_time = 0.0
    try:
        import sys
        module_names = [
            'main_weight_fde_gpu_triple_stream',
            '__main__',
        ]
        
        main_module = None
        for mod_name in module_names:
            if mod_name in sys.modules:
                main_module = sys.modules[mod_name]
                if hasattr(main_module, 'CUMULATIVE_TIMING'):
                    break
                main_module = None
        
        if main_module is None:
            for mod_name, mod in sys.modules.items():
                if hasattr(mod, 'CUMULATIVE_TIMING') and hasattr(mod, 'TIMING'):
                    main_module = mod
                    break
        
        if main_module is not None:
            if hasattr(main_module, 'CUMULATIVE_TIMING'):
                final_memmap_flush_time = main_module.CUMULATIVE_TIMING.get('flush', 0.0)
            elif hasattr(main_module, 'TIMING'):
                final_memmap_flush_time = main_module.TIMING.get('flush', 0.0)
    except (AttributeError, KeyError, Exception):
        pass
    
    # Calculate total measured time
    total_measured_time = prep_time + upload_time + compute_time + download_time + reshape_time + flush_time
    total_measured_with_final_flush = cumul['prep_time'] + cumul['upload_time'] + cumul['compute_time'] + cumul['download_time'] + cumul['reshape_time'] + cumul['flush_time'] + final_memmap_flush_time
    
    logging.info("=" * 80)
    logging.info("🚀 GPU FDE Performance Summary (Single Batch)")
    logging.info("=" * 80)
    logging.info(f"Data preparation:    {prep_time:8.3f}s  ({prep_time/total_time*100:5.1f}%)")
    logging.info(f"Upload time:         {upload_time:8.3f}s  ({upload_time/total_time*100:5.1f}%)")
    logging.info(f"SimHash(Projection) kernel:      {simhash_time:8.3f}s  ({simhash_time/total_time*100:5.1f}%)")
    logging.info(f"Partition kernel:    {partition_time:8.3f}s  ({partition_time/total_time*100:5.1f}%)")
    logging.info(f"Scatter-add kernel:  {scatter_time:8.3f}s  ({scatter_time/total_time*100:5.1f}%)")
    logging.info(f"Average kernel:      {average_time:8.3f}s  ({average_time/total_time*100:5.1f}%)")
    if fill_time > 0.0:
        logging.info(f"Fill-empty kernel:   {fill_time:8.3f}s  ({fill_time/total_time*100:5.1f}%)")
    logging.info(f"Compute time:        {compute_time:8.3f}s  ({compute_time/total_time*100:5.1f}%)")
    logging.info(f"Download time:       {download_time:8.3f}s  ({download_time/total_time*100:5.1f}%)")
    logging.info(f"Reshape time:        {reshape_time:8.3f}s  ({reshape_time/total_time*100:5.1f}%)")
    logging.info(f"Flush time:          {flush_time:8.3f}s  ({flush_time/total_time*100:5.1f}%)")
    logging.info(f"Final memmap flush: {final_memmap_flush_time:8.3f}s  ({final_memmap_flush_time/total_time*100:5.1f}%)" if final_memmap_flush_time > 0.0 else f"Final memmap flush: {final_memmap_flush_time:8.3f}s  (  0.0%)")
    logging.info(f"Total time:          {total_time:8.3f}s")
    logging.info("=" * 80)
    logging.info("📊 Cumulative Time Breakdown (All Operations - Across All Batches):")
    logging.info("-" * 80)
    logging.info(f"   Data preparation (cumulative):    {cumul['prep_time']:8.3f}s")
    logging.info(f"   Upload (cumulative):             {cumul['upload_time']:8.3f}s")
    logging.info(f"   SimHash(Projection) kernel (cumulative):      {cumul['simhash_time']:8.3f}s")
    logging.info(f"   Partition kernel (cumulative):    {cumul['partition_time']:8.3f}s")
    logging.info(f"   Scatter-add kernel (cumulative): {cumul['scatter_time']:8.3f}s")
    logging.info(f"   Average kernel (cumulative):     {cumul['average_time']:8.3f}s")
    logging.info(f"   Fill-empty kernel (cumulative):  {cumul['fill_time']:8.3f}s")
    logging.info(f"   Compute (cumulative):           {cumul['compute_time']:8.3f}s")
    logging.info(f"   Download (cumulative):          {cumul['download_time']:8.3f}s")
    logging.info(f"   Reshape (cumulative):          {cumul['reshape_time']:8.3f}s")
    logging.info(f"   Flush (cumulative):             {cumul['flush_time']:8.3f}s")
    logging.info(f"   Final memmap flush (cumulative): {final_memmap_flush_time:8.3f}s")
    logging.info("-" * 80)
    logging.info(f"   Total cumulative time:            {total_measured_with_final_flush:8.3f}s")
    logging.info("=" * 80)
    
    # Cleanup
    del simhash_matrices_gpu
    if ams_matrices_gpu is not None:
        del ams_matrices_gpu
    cp.get_default_memory_pool().free_all_blocks()
    
    # Cleanup intermediate buffers
    del embeddings_gpu, doc_lengths_gpu, sketches_gpu, projected_gpu, partition_indices_gpu
    del partition_counts_gpu, partition_sums_gpu
    
    
    return {
        'prep_time': prep_time,
        'upload_time': upload_time,
        'simhash_time': simhash_time,
        'partition_time': partition_time,
        'scatter_time': scatter_time,
        'average_time': average_time,
        'fill_time': fill_time,
        'compute_time': compute_time,
        'download_time': download_time,
        'reshape_time': reshape_time,
        'flush_time': flush_time,
        'total_time': total_time,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with small dataset
    num_docs = 1000
    test_embeddings = [
        np.random.randn(50, 128).astype(np.float32) 
        for _ in range(num_docs)
    ]
    
    config = FixedDimensionalEncodingConfig(
        dimension=128,
        num_repetitions=5,
        num_simhash_projections=4,
        seed=42,
        encoding_type=EncodingType.AVERAGE,
        fill_empty_partitions=True,
        projection_type=ProjectionType.AMS_SKETCH,
        projection_dimension=128,
    )
    
    # Create memmap
    num_partitions = 2 ** config.num_simhash_projections
    final_fde_dim = config.num_repetitions * num_partitions * 128
    fde_memmap = np.memmap("test_fde.mmap", mode="w+", dtype=np.float32, shape=(num_docs, final_fde_dim))
    
    logging.info("Testing 3-STREAM PIPELINE...")
    stats = generate_document_fde_batch_gpu_3stage(
        test_embeddings,
        config,
        fde_memmap,
        batch_start_idx=0,
        mini_batch_size=200
    )
    
    logging.info("✅ Test passed!")
    logging.info(f"Final FDE shape: {fde_memmap.shape}")