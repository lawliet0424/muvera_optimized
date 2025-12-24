# -*- coding: utf-8 -*-
import logging
import time
import os
import numpy as np
import cupy as cp
from typing import Optional, List
import threading
from queue import Queue
from dataclasses import dataclass, replace
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
    const int num_reps,
    const int ignore_bit,
    const int force_bit_value
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
        unsigned int bit;
        if (ignore_bit >= 0 && b == ignore_bit) {
            // Bit ablation: force the bit value (force_bit_value가 지정되면 그 값, 아니면 기본값 0)
            bit = (force_bit_value >= 0) ? (unsigned int)force_bit_value : 0;
        } else {
            bit = (sketches[b] > 0.0f) ? 1 : 0;
        }
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

def _simhash_partition_index_gray(sketch_vector) -> int:
    """Compute Gray code partition index from sketch vector (supports both numpy and cupy arrays)."""
    partition_index = 0
    # Convert to numpy if cupy array
    if hasattr(sketch_vector, 'get'):  # cupy array
        sketch_vector = cp.asnumpy(sketch_vector)
    for val in sketch_vector:
        partition_index = _append_to_gray_code(partition_index, val > 0)
    return partition_index

def _distance_to_simhash_partition(sketch_vector, partition_index: int) -> int:
    """Compute Hamming distance to partition (supports both numpy and cupy arrays)."""
    num_projections = sketch_vector.size
    binary_representation = _gray_code_to_binary(partition_index)
    # Convert to numpy if cupy array
    if hasattr(sketch_vector, 'get'):  # cupy array
        sketch_vector = cp.asnumpy(sketch_vector)
    sketch_bits = (sketch_vector > 0).astype(int)
    binary_array = (binary_representation >> np.arange(num_projections - 1, -1, -1)) & 1
    return int(np.sum(sketch_bits != binary_array))

def _apply_count_sketch_to_vector(
    input_vector: np.ndarray, final_dimension: int, seed: int
) -> np.ndarray:
    """Apply count sketch projection (supports both numpy and cupy arrays)."""
    # Convert to numpy if cupy array
    if hasattr(input_vector, 'get'):  # cupy array
        input_vector = cp.asnumpy(input_vector)
    rng = np.random.default_rng(seed)
    out = np.zeros(final_dimension, dtype=np.float32)
    indices = rng.integers(0, final_dimension, size=input_vector.shape[0])
    signs = rng.choice([-1.0, 1.0], size=input_vector.shape[0])
    np.add.at(out, indices, signs * input_vector)
    return out

def _simhash_matrix_from_seed_gpu(
    dimension: int, num_projections: int, seed: int
) -> cp.ndarray:
    #Generate SimHash matrix on GPU
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
    #Generate AMS projection matrix on GPU
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
    #Pad document embeddings to uniform length
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
    #Generates a Fixed Dimensional Encoding for a query point cloud (using SUM).
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
    out_fde_gpu = cp.zeros(final_fde_dim, dtype=cp.float32)

    # Convert input to GPU
    point_cloud_gpu = cp.asarray(point_cloud.astype(np.float32))
    
    # Query는 단일 쿼리이므로 num_docs=1, max_len=num_points
    num_docs = 1
    max_len = num_points
    doc_lengths_gpu = cp.array([num_points], dtype=cp.int32)

    # Prepare all random matrices for all repetitions at once
    simhash_matrices_list = []
    ams_matrices_list = []
    
    for rep_num in range(config.num_repetitions):
        current_seed = config.seed + rep_num
        simhash_mat_gpu = _simhash_matrix_from_seed_gpu(
            original_dim, config.num_simhash_projections, current_seed
        )
        simhash_matrices_list.append(simhash_mat_gpu)
        
        if not use_identity_proj:
            ams_matrix_gpu = _ams_projection_matrix_from_seed_gpu(
                original_dim, projection_dim, current_seed
            )
            ams_matrices_list.append(ams_matrix_gpu)
    
    # Stack matrices: [num_reps, dim, num_bits] or [num_reps, dim, proj_dim]
    simhash_matrices_gpu = cp.stack(simhash_matrices_list, axis=0)  # [num_reps, dim, num_bits]
    if not use_identity_proj:
        ams_matrices_gpu = cp.stack(ams_matrices_list, axis=0)  # [num_reps, dim, proj_dim]
    else:
        ams_matrices_gpu = None

    # Prepare data structures for kernel calls (batch format: [num_docs=1, num_reps, max_len, ...])
    sketches_batch_gpu = cp.zeros((num_docs, config.num_repetitions, max_len, config.num_simhash_projections), dtype=cp.float32)
    projected_batch_gpu = cp.zeros((num_docs, config.num_repetitions, max_len, projection_dim), dtype=cp.float32)
    partition_indices_batch_gpu = cp.zeros((num_docs, config.num_repetitions, max_len), dtype=cp.int32)
    partition_sums_batch_gpu = cp.zeros((num_docs, config.num_repetitions, num_partitions, projection_dim), dtype=cp.float32)
    partition_counts_batch_gpu = cp.zeros((num_docs, config.num_repetitions, num_partitions), dtype=cp.int32)

    # Compute SimHash and Projection for all repetitions (same pattern as document batch processing)
    # 1-Dimensional Embedding으로 펼치기: T = num_points
    total_tokens = num_points
    dim = original_dim
    num_bits = config.num_simhash_projections
    reps = config.num_repetitions
    
    point_cloud_2d = point_cloud_gpu.reshape(total_tokens, dim)  # (T, D)
    
    for rep_idx in range(reps):
        # SimHash: (T, D) @ (D, num_bits) -> (T, num_bits)
        simhash_mat_rep = simhash_matrices_gpu[rep_idx]  # (D, num_bits)
        sketches_rep = point_cloud_2d @ simhash_mat_rep  # (T, num_bits)
        
        # (1, num_points, num_bits)로 reshape 후, rep 축에 넣기
        sketches_rep_4d = sketches_rep.reshape(num_docs, max_len, num_bits)
        sketches_batch_gpu[:, rep_idx, :, :] = sketches_rep_4d
        
        # Projection: identity or AMS
        if use_identity_proj:
            # projection이 필요없는 경우 그대로 복사
            projected_batch_gpu[:, rep_idx, :, :] = point_cloud_gpu.reshape(num_docs, max_len, dim)
        else:
            ams_mat_rep = ams_matrices_gpu[rep_idx]  # (D, proj_dim)
            proj_rep = point_cloud_2d @ ams_mat_rep  # (T, proj_dim)
            proj_rep_4d = proj_rep.reshape(num_docs, max_len, projection_dim)
            projected_batch_gpu[:, rep_idx, :, :] = proj_rep_4d

    # Call SIMHASH_PARTITION_KERNEL for all repetitions at once
    total_tokens_all_reps = num_docs * config.num_repetitions * max_len
    threads_per_block = 256
    num_blocks = (total_tokens_all_reps + threads_per_block - 1) // threads_per_block
    
    SIMHASH_PARTITION_KERNEL(
        (num_blocks,), (threads_per_block,),
        (sketches_batch_gpu, doc_lengths_gpu, partition_indices_batch_gpu,
         num_docs, max_len, config.num_simhash_projections, config.num_repetitions,
         -1, -1)  # ignore_bit=-1, force_bit_value=-1 (no bit ablation for queries)
    )
    cp.cuda.Device().synchronize()

    # Call SCATTER_ADD_KERNEL for all repetitions
    shared_mem_size = (num_partitions * projection_dim * 4) + (num_partitions * 4)  # floats + ints
    grid_dim = (num_docs, config.num_repetitions)
    
    SCATTER_ADD_KERNEL(
        grid_dim, (threads_per_block,),
        (projected_batch_gpu, partition_indices_batch_gpu, doc_lengths_gpu,
         partition_sums_batch_gpu, partition_counts_batch_gpu,
         num_docs, config.num_repetitions, max_len, projection_dim, num_partitions),
        shared_mem=shared_mem_size
    )
    cp.cuda.Device().synchronize()

    # Call AVERAGE_KERNEL if needed
    '''
    if config.encoding_type == EncodingType.AVERAGE:
        total_partitions = num_docs * config.num_repetitions * num_partitions
        num_blocks = (total_partitions + threads_per_block - 1) // threads_per_block
        
        AVERAGE_KERNEL(
            (num_blocks,), (threads_per_block,),
            (partition_sums_batch_gpu, partition_counts_batch_gpu, num_docs, config.num_repetitions,
             num_partitions, projection_dim)
        )
        cp.cuda.Device().synchronize()

    # Handle fill_empty_partitions if needed (CPU fallback for now, can be optimized later)
    
    if config.fill_empty_partitions and config.encoding_type == EncodingType.AVERAGE:
        # Check for empty partitions and fill them
        for rep_num in range(config.num_repetitions):
            for p_idx in range(num_partitions):
                count = int(partition_counts_batch_gpu[0, rep_num, p_idx])
                if count == 0 and num_points > 0:
                    # Find nearest point (fallback to CPU for now)
                    sketches_cpu = cp.asnumpy(sketches_batch_gpu[0, rep_num, :, :])
                    nearest_point_idx = None
                    min_dist = float('inf')
                    for j in range(num_points):
                        dist = _distance_to_simhash_partition(sketches_cpu[j], p_idx)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_point_idx = j
                    if nearest_point_idx is not None:
                        # partition_sums_batch_gpu shape: [num_docs, num_reps, num_partitions, proj_dim]
                        partition_sums_batch_gpu[0, rep_num, p_idx, :] = (
                            projected_batch_gpu[0, rep_num, nearest_point_idx, :]
                        )
    '''
    
    # Reshape results to final FDE format
    for rep_num in range(config.num_repetitions):
        rep_start_index = rep_num * num_partitions * projection_dim
        # partition_sums_batch_gpu shape: [num_docs, num_reps, num_partitions, proj_dim]
        # Flatten: [num_partitions, projection_dim] -> [num_partitions * projection_dim]
        rep_fde_flat = partition_sums_batch_gpu[0, rep_num].reshape(-1)
        out_fde_gpu[rep_start_index : rep_start_index + rep_fde_flat.size] = rep_fde_flat

    # Convert final result to CPU
    out_fde = cp.asnumpy(out_fde_gpu)

    if config.final_projection_dimension and config.final_projection_dimension > 0:
        return _apply_count_sketch_to_vector(
            out_fde, config.final_projection_dimension, config.seed
        )

    return out_fde

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
        'compute_time': 0.0,
        'download_time': 0.0,
        'reshape_time': 0.0,
        'flush_time': 0.0,
    }

def generate_document_fde_batch_gpu_3stage(
    doc_embeddings_list: List[np.ndarray],
    config: FixedDimensionalEncodingConfig,
    fde_memmap,  # Pre-created memmap from main code
    batch_start_idx: int,  # Where to write in memmap
    *,
    ignore_bit=None,
    force_bit_value=None,  # 0 or 1 to force the bit value when ignore_bit is set
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
    
    # Bit ablation 설정 로깅
    if ignore_bit is not None:
        forced_val = force_bit_value if force_bit_value is not None else 0
        logging.info(f"[FDE 3-Stream] Bit ablation enabled: ignore_bit={ignore_bit}, force_bit_value={forced_val}")
    else:
        logging.info(f"[FDE 3-Stream] Bit ablation disabled")
    
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
        
        # Allocate output buffers
        sketches_gpu = cp.zeros((num_docs, config.num_repetitions, max_len, config.num_simhash_projections), dtype=cp.float32)
        projected_gpu = cp.zeros((num_docs, config.num_repetitions, max_len, projection_dim), dtype=cp.float32)
        partition_indices_gpu = cp.zeros((num_docs, config.num_repetitions, max_len), dtype=cp.int32)
        partition_sums_gpu = cp.zeros((num_docs, config.num_repetitions, num_partitions, projection_dim), dtype=cp.float32)
        partition_counts_gpu = cp.zeros((num_docs, config.num_repetitions, num_partitions), dtype=cp.int32)
    
    stream_upload.synchronize()
    upload_time = time.perf_counter() - upload_start
    
    # ========================================
    # STEP 3: Compute on GPU (Stream 2)
    # ========================================
    compute_start = time.perf_counter()
    
    with stream_compute:
        # ===========================
        # 3-1. CuPy GEMM -> projection 수행
        # ===========================
        simhash_start = time.perf_counter()
        
        # 1-Dimensional Embedding으로 펼치기: T = num_docs * max_len
        total_tokens = num_docs * max_len
        dim = config.dimension
        num_bits = config.num_simhash_projections
        reps = config.num_repetitions
        
        embeddings_2d = embeddings_gpu.reshape(total_tokens, dim)  # (T, D)
        
        for rep_idx in range(reps):
            # SimHash: (T, D) @ (D, num_bits) -> (T, num_bits)
            simhash_mat_rep = simhash_matrices_gpu[rep_idx]             # (D, num_bits)
            sketches_rep = embeddings_2d @ simhash_mat_rep              # (T, num_bits)
            
            # (num_docs, max_len, num_bits)로 reshape 후, rep 축에 넣기
            sketches_rep_4d = sketches_rep.reshape(num_docs, max_len, num_bits)
            sketches_gpu[:, rep_idx, :, :] = sketches_rep_4d
            
            # Projection: identity or AMS
            if use_identity_proj:
                # projection이 필요없는 경우 그대로 복사
                projected_gpu[:, rep_idx, :, :] = embeddings_gpu
            else:
                ams_mat_rep = ams_matrices_gpu[rep_idx]                 # (D, proj_dim)
                proj_rep = embeddings_2d @ ams_mat_rep                  # (T, proj_dim)
                proj_rep_4d = proj_rep.reshape(num_docs, max_len, projection_dim)
                projected_gpu[:, rep_idx, :, :] = proj_rep_4d
        
        stream_compute.synchronize()
        simhash_time = time.perf_counter() - simhash_start

        # ===========================
        # 3-2. partition 계산 커널 호출
        # ===========================
        partition_start = time.perf_counter()
        
        total_tokens_all_reps = num_docs * reps * max_len
        threads_per_block = 256
        num_blocks = (total_tokens_all_reps + threads_per_block - 1) // threads_per_block
        
        # Bit ablation 파라미터 설정
        ignore_bit_val = ignore_bit if ignore_bit is not None else -1
        force_bit_val = force_bit_value if force_bit_value is not None else -1
        
        SIMHASH_PARTITION_KERNEL(
            (num_blocks,), (threads_per_block,),
            (sketches_gpu, doc_lengths_gpu, partition_indices_gpu,
             num_docs, max_len, num_bits, reps, ignore_bit_val, force_bit_val)
        )
        stream_compute.synchronize()
        partition_time = time.perf_counter() - partition_start
        
        # Kernel 2: Scatter-add
        scatter_start = time.perf_counter()
        
        shared_mem_size = (num_partitions * projection_dim * 4) + (num_partitions * 4)
        grid_dim = (num_docs, config.num_repetitions)
        
        SCATTER_ADD_KERNEL(
            grid_dim, (threads_per_block,),
            (projected_gpu, partition_indices_gpu, doc_lengths_gpu, partition_sums_gpu, partition_counts_gpu,
             num_docs, config.num_repetitions, max_len, projection_dim, num_partitions),
            shared_mem=shared_mem_size
        )
        stream_compute.synchronize()
        scatter_time = time.perf_counter() - scatter_start
        
        # Kernel 3: Average
        average_start = time.perf_counter()
        
        total_partitions = num_docs * config.num_repetitions * num_partitions
        num_blocks = (total_partitions + threads_per_block - 1) // threads_per_block
        
        AVERAGE_KERNEL(
            (num_blocks,), (threads_per_block,),
            (partition_sums_gpu, partition_counts_gpu, num_docs, config.num_repetitions,
             num_partitions, projection_dim)
        )
        stream_compute.synchronize()
        average_time = time.perf_counter() - average_start
    
    compute_time = simhash_time + scatter_time + average_time
    
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
            fde_cpu[doc_idx, rep_offset:rep_offset + final_fde_dim_per_rep] = fde_chunk
    
    reshape_time = time.perf_counter() - reshape_start
    
    # ========================================
    # STEP 6: Write to memmap and flush
    # ========================================
    flush_start = time.perf_counter()
    
    fde_memmap[batch_start_idx:batch_start_idx + num_docs] = fde_cpu
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
        projection_type=ProjectionType.AMS_SKETCH,
        projection_dimension=128,
    )
    
    # Create memmap
    num_partitions = 2 ** config.num_simhash_projections
    final_fde_dim = config.num_repetitions * num_partitions * 128
    fde_memmap = np.memmap("test_fde.mmap", mode="w+", dtype=np.float32, shape=(num_docs, final_fde_dim))
    
    logging.info("Testing 3-STREAM PIPELINE...")
    stats = generate_document_fde_batch_gpu_3stream_pipeline(
        test_embeddings,
        config,
        fde_memmap,
        batch_start_idx=0,
        mini_batch_size=200
    )
    
    logging.info("✅ Test passed!")
    logging.info(f"Final FDE shape: {fde_memmap.shape}")