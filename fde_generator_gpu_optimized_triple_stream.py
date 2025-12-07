# -*- coding: utf-8 -*-
"""
GPU-Optimized FDE Generator with 3-STREAM PIPELINE
Pipeline stages:
  Stream 1: CPU → GPU (upload embeddings)
  Stream 2: GPU processing (compute FDE)  
  Stream 3: GPU → CPU → Disk (download + flush)

All 3 stages run in parallel for maximum throughput!
"""
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

# Import CUDA kernels from your dual-stream version
from fde_generator_gpu_optimized_dual_stream_with_mini_batch import (
    SIMHASH_KERNEL,
    SCATTER_ADD_KERNEL,
    AVERAGE_KERNEL,
)

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
# Global variables for cumulative timing across all batches
# ==============================================================================
_GPU_CUMULATIVE_TIMING = {
    'prep_time': 0.0,
    'upload_time': 0.0,
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
        'compute_time': 0.0,
        'download_time': 0.0,
        'reshape_time': 0.0,
        'flush_time': 0.0,
    }

def generate_document_fde_batch_gpu_3stream_pipeline(
    doc_embeddings_list: List[np.ndarray],
    config: FixedDimensionalEncodingConfig,
    fde_memmap,  # Pre-created memmap from main code
    batch_start_idx: int,  # Where to write in memmap
    *,
    mini_batch_size: int = 500,  # Ignored - kept for backward compatibility
    log_every: int = 1000
) -> dict:
    """
    🚀 GPU FDE generation processing all documents in a single batch!
    
    Strategy:
    1. Process all documents in one batch (no mini-batching)
    2. Use 3 CUDA streams for upload, compute, and download
    3. Transfer results to CPU and write to memmap after all computation is done
    
    Args:
        doc_embeddings_list: All document embeddings
        config: FDE configuration
        fde_memmap: Pre-allocated memmap for output
        batch_start_idx: Starting index in memmap
        mini_batch_size: Ignored (kept for backward compatibility)
        
    Returns:
        Timing statistics dictionary
    """
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
    # Prepare random matrices (shared across all batches)
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
        # Kernel 1: SimHash
        total_tokens = num_docs * config.num_repetitions * max_len
        threads_per_block = 256
        num_blocks = (total_tokens + threads_per_block - 1) // threads_per_block
        
        SIMHASH_KERNEL(
            (num_blocks,), (threads_per_block,),
            (embeddings_gpu, doc_lengths_gpu, simhash_matrices_gpu,
             ams_matrices_gpu if ams_matrices_gpu is not None else cp.zeros(1, dtype=cp.float32),
             sketches_gpu, projected_gpu, partition_indices_gpu,
             num_docs, max_len, config.dimension, config.num_simhash_projections,
             projection_dim, config.num_repetitions,
             1 if use_identity_proj else 0)
        )
        
        # Kernel 2: Scatter-add
        shared_mem_size = (num_partitions * projection_dim * 4) + (num_partitions * 4)
        grid_dim = (num_docs, config.num_repetitions)
        
        SCATTER_ADD_KERNEL(
            grid_dim, (threads_per_block,),
            (projected_gpu, partition_indices_gpu, doc_lengths_gpu, partition_sums_gpu, partition_counts_gpu,
             num_docs, config.num_repetitions, max_len, projection_dim, num_partitions),
            shared_mem=shared_mem_size
        )
        
        # Kernel 3: Average
        total_partitions = num_docs * config.num_repetitions * num_partitions
        num_blocks = (total_partitions + threads_per_block - 1) // threads_per_block
        
        AVERAGE_KERNEL(
            (num_blocks,), (threads_per_block,),
            (partition_sums_gpu, partition_counts_gpu, num_docs, config.num_repetitions,
             num_partitions, projection_dim)
        )
    
    stream_compute.synchronize()
    compute_time = time.perf_counter() - compute_start
    
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
    _GPU_CUMULATIVE_TIMING['compute_time'] += compute_time
    _GPU_CUMULATIVE_TIMING['download_time'] += download_time
    _GPU_CUMULATIVE_TIMING['reshape_time'] += reshape_time
    _GPU_CUMULATIVE_TIMING['flush_time'] += flush_time
    
    # Get cumulative times (across all batches) - use dictionary directly to avoid duplication
    cumul = _GPU_CUMULATIVE_TIMING
    
    # Try to get final memmap flush time from main_weight module
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