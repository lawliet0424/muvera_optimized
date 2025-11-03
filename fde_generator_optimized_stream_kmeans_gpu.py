# -*- coding: utf-8 -*-
import logging
import time, pathlib, os
import numpy as np
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, List, Tuple
from collections import defaultdict
from joblib import Parallel, delayed  # pip install joblib

#[1103] GPU 사용 시작
from cuml.cluster import KMeans

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
    num_simhash_projections: int = 6  # 이제 partition 개수로 사용
    seed: int = 42
    encoding_type: EncodingType = EncodingType.DEFAULT_SUM
    projection_type: ProjectionType = ProjectionType.DEFAULT_IDENTITY
    projection_dimension: Optional[int] = None
    fill_empty_partitions: bool = False
    final_projection_dimension: Optional[int] = None
    # K-means 관련 설정
    kmeans_sample_ratio: float = 0.2  # 기본 샘플링 비율 (동적으로 조정됨)
    use_kmeans_partition: bool = True  # K-means 사용 여부
    use_memory_based_sampling: bool = True  # 메모리 기반 샘플링 사용 여부
    target_memory_gb: float = 2.0  # 목표 메모리 사용량 (GB)

# ------------------------------
# Gray code / hashing utilities
# ------------------------------
def _append_to_gray_code(gray_code: int, bit: bool) -> int:
    return (gray_code << 1) + (int(bit) ^ (gray_code & 1))

def _gray_code_to_binary(num: int) -> int:
    mask = num >> 1
    while mask != 0:
        num = num ^ mask
        mask >>= 1
    return num

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

def _simhash_partition_index_gray(sketch_vector: np.ndarray) -> int:
    partition_index = 0
    for val in sketch_vector:
        partition_index = _append_to_gray_code(partition_index, val > 0)
    return partition_index

def _distance_to_simhash_partition(
    sketch_vector: np.ndarray, partition_index: int
) -> int:
    num_projections = sketch_vector.size
    binary_representation = _gray_code_to_binary(partition_index)
    sketch_bits = (sketch_vector > 0).astype(int)
    binary_array = (binary_representation >> np.arange(num_projections - 1, -1, -1)) & 1
    return int(np.sum(sketch_bits != binary_array))

# ------------------------------
# K-means partition utilities
# ------------------------------
def _kmeans_partition_index(projected_points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """K-means centers와의 거리로 partition index 계산"""
    distances = np.linalg.norm(projected_points[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    return np.argmin(distances, axis=1)

def _calculate_memory_based_sample_ratio(
    num_docs: int,
    embedding_dim: int,
    num_partitions: int,
    target_memory_gb: float = 2.0,
    min_ratio: float = 0.05,
    max_ratio: float = 0.3,
    actual_tokens_per_doc: Optional[int] = None
) -> float:
    """문서 임베딩 크기와 메모리 사용량을 고려한 동적 샘플링 비율 계산"""
    
    # 메모리 사용량 계산 (GB 단위)
    # float32 = 4 bytes
    # 샘플링된 문서들의 projected points + K-means centers + 기타 오버헤드
    
    # 1. 샘플링된 문서들의 projected points 메모리
    def calculate_memory_for_sample_ratio(sample_ratio: float, actual_tokens_per_doc: Optional[int] = None) -> float:
        n_sample_docs = max(int(num_docs * sample_ratio), 1)
        
        # 실제 토큰 수가 제공되면 사용, 아니면 보수적 추정
        if actual_tokens_per_doc is not None:
            avg_tokens_per_doc = actual_tokens_per_doc
        else:
            avg_tokens_per_doc = 100  # 보수적 추정
        
        # 샘플링된 문서들의 projected points 메모리
        sample_points_memory = n_sample_docs * avg_tokens_per_doc * embedding_dim * 4  # bytes
        
        # K-means centers 메모리
        centers_memory = num_partitions * embedding_dim * 4  # bytes
        
        # psutil을 사용한 전체 시스템 메모리 기반 오버헤드 계산
        try:
            import psutil
            # 전체 시스템 메모리 정보
            total_memory_gb = psutil.virtual_memory().total / (1024**3)  # GB
            available_memory_gb = psutil.virtual_memory().available / (1024**3)  # GB
            
            # 시스템 메모리 대비 오버헤드 팩터 계산 (일관된 비율)
            if total_memory_gb > 0:
                # 시스템 메모리 크기에 따라 오버헤드 조정
                if total_memory_gb >= 32:  # 32GB 이상
                    overhead_factor = 0.6
                elif total_memory_gb >= 16:  # 16GB 이상
                    overhead_factor = 0.7
                elif total_memory_gb >= 8:   # 8GB 이상
                    overhead_factor = 0.8
                else:  # 8GB 미만
                    overhead_factor = 1.0
            else:
                overhead_factor = 0.8  # 기본값
        except ImportError:
            overhead_factor = 0.8  # psutil 없을 때 기본값
        
        overhead_memory = sample_points_memory * overhead_factor
        total_memory_bytes = sample_points_memory + centers_memory + overhead_memory
        total_memory_gb = total_memory_bytes / (1024**3)
        
        return total_memory_gb
    
    # 2. 이진 탐색으로 적절한 샘플링 비율 찾기
    low_ratio, high_ratio = min_ratio, max_ratio
    best_ratio = min_ratio
    
    for _ in range(10):  # 최대 20회 반복
        mid_ratio = (low_ratio + high_ratio) / 2
        memory_usage = calculate_memory_for_sample_ratio(mid_ratio, actual_tokens_per_doc)
        
        if memory_usage <= target_memory_gb:
            best_ratio = mid_ratio
            low_ratio = mid_ratio
        else:
            high_ratio = mid_ratio
        
        if high_ratio - low_ratio < 0.001:  # 수렴 조건
            break
    
    # 3. 최종 비율 제한
    final_ratio = np.clip(best_ratio, min_ratio, max_ratio)
    
    return float(final_ratio)

def _calculate_dynamic_sample_ratio(
    num_simhash_projections: int, 
    base_sample_ratio: float = 0.2,
    min_ratio: float = 0.1,
    max_ratio: float = 0.3
) -> float:
    """num_simhash_projections에 따라 동적으로 샘플링 비율 계산 (기존 방식)"""
    # partition 개수가 많을수록 더 많은 샘플이 필요
    # log2(num_partitions)에 비례하여 증가
    num_partitions = 2 ** num_simhash_projections
    log_partitions = np.log2(num_partitions)
    
    # 기본 비율에 log 스케일 팩터 적용
    # 2^6=64일 때 1.0, 2^10=1024일 때 1.67 정도
    log_factor = log_partitions / 6.0  # 2^6=64를 기준으로 정규화
    
    # 동적 샘플링 비율 계산
    dynamic_ratio = base_sample_ratio * log_factor
    
    # 최소/최대 비율로 제한
    dynamic_ratio = np.clip(dynamic_ratio, min_ratio, max_ratio)
    
    return float(dynamic_ratio)

def _sample_and_train_kmeans(
    all_projected_points: np.ndarray, 
    num_partitions: int, 
    seed: int,
    sample_ratio: float = 0.1,
    actual_tokens_per_doc: Optional[int] = None
) -> np.ndarray:
    """전체 데이터에서 샘플링하여 K-means centers 학습"""
    # 전체 데이터에서 임의 샘플링
    rng = np.random.default_rng(seed)
    n_samples = max(int(len(all_projected_points) * sample_ratio), num_partitions * 2)
    sample_indices = rng.choice(len(all_projected_points), size=n_samples, replace=False)
    sampled_points = all_projected_points[sample_indices]
    
    # GPU에서의 랜덤 30% 재할당 최적화된 K-means 학습
    kmeans = KMeans(n_clusters=num_partitions, init="k-means++", random_state=seed, n_init=10, max_iter=300, tol=1e-4, oversampling_factor=2.0)
    logging.info(f"[K-means Pre-training] Fitting K-means with {num_partitions} clusters and {n_samples} samples")
    start_time = time.time()
    kmeans.fit(sampled_points)
    end_time = time.time()
    logging.info(f"[K-means Pre-training] K-means fitting time: {end_time - start_time:.2f} seconds")

    return kmeans.cluster_centers_

# -----------------------------------------
# Vectorized/parallel helpers (NEW/UPDATED)
# -----------------------------------------
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

def _fill_empty_partitions_for_doc(
    doc_idx: int,
    num_bits: int,
    num_partitions: int,
    doc_start: int,
    doc_end: int,
    all_sketches: np.ndarray,       # [total_vectors, num_bits], float32
    projected_points: np.ndarray,   # [total_vectors, proj_dim], float32
    empty_parts_bool_row: np.ndarray,  # [num_partitions], bool
    part_bits_tbl: np.ndarray       # [num_partitions, num_bits], uint8
):
    """
    Returns (doc_idx, empties, fill_block)
      - empties: np.array of empty partition indices for this doc
      - fill_block: np.ndarray [len(empties), proj_dim] to assign into rep_fde_sum[doc_idx, empties, :]
    """
    empties = np.flatnonzero(empty_parts_bool_row)
    if empties.size == 0 or (doc_end - doc_start) == 0:
        return doc_idx, empties, None

    doc_sketches = all_sketches[doc_start:doc_end]            # [Ld, b], float
    doc_bits = (doc_sketches > 0).astype(np.uint8)            # [Ld, b]

    # target bits for the empty partitions: [P_e, b]
    target_bits = part_bits_tbl[empties]                      # [P_e, b]
    # Broadcast XOR to compute Hamming distances: [P_e, Ld, b] -> sum over last axis -> [P_e, Ld]
    distances = np.sum(target_bits[:, None, :] ^ doc_bits[None, :, :], axis=2)  # [P_e, Ld]
    nearest_local = np.argmin(distances, axis=1)              # [P_e]
    nearest_global = doc_start + nearest_local                # [P_e]
    fill_block = projected_points[nearest_global]             # [P_e, proj_dim]
    return doc_idx, empties, fill_block

# -----------------------------
# Core FDE generation routines
# -----------------------------
def _generate_fde_internal(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig, query_or_doc: bool,
    kmeans_centers: Optional[np.ndarray] = None  # Query용 centers 추가
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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

    #[1017] simhash별 indice별 원소 개수 저장 필요------------------------------------
    partition_counts_all = np.zeros((config.num_repetitions, num_partitions), dtype=np.int32)
    #------------------------------------------------------------------------
    
    # K-means centers 저장용
    learned_centers = np.zeros((config.num_repetitions, num_partitions, projection_dim), dtype=np.float32)

    for rep_num in range(config.num_repetitions):
        current_seed = config.seed + rep_num

        if use_identity_proj:
            projected_matrix = point_cloud
        elif config.projection_type == ProjectionType.AMS_SKETCH:
            ams_matrix = _ams_projection_matrix_from_seed(
                original_dim, projection_dim, current_seed
            )
            projected_matrix = point_cloud @ ams_matrix

        rep_fde_sum = np.zeros(num_partitions * projection_dim, dtype=np.float32)
        partition_counts = np.zeros(num_partitions, dtype=np.int32)

        # [수정] K-means 기반 partition index 계산
        if query_or_doc and kmeans_centers is not None:  # Query인 경우: 미리 학습된 centers 사용
            partition_indices = _kmeans_partition_index(projected_matrix, kmeans_centers[rep_num])
        else:  # Document인 경우: 현재 데이터로 K-means 학습
            # 메모리 기반 또는 동적 샘플링 비율 계산
            if config.use_memory_based_sampling:
                # 단일 문서의 경우 전체 데이터 사용 (메모리 기반 계산 불가)
                dynamic_sample_ratio = config.kmeans_sample_ratio
            else:
                dynamic_sample_ratio = _calculate_dynamic_sample_ratio(
                    config.num_simhash_projections, config.kmeans_sample_ratio
                )
            learned_centers[rep_num] = _sample_and_train_kmeans(
                projected_matrix, num_partitions, current_seed, dynamic_sample_ratio
            )
            partition_indices = _kmeans_partition_index(projected_matrix, learned_centers[rep_num])

        for i in range(num_points):
            start_idx = partition_indices[i] * projection_dim
            rep_fde_sum[start_idx : start_idx + projection_dim] += projected_matrix[i]
            partition_counts[partition_indices[i]] += 1
            #[1017] simhash별 indice별 원소 개수 저장 필요------------------------------------
            partition_counts_all[rep_num][partition_indices[i]] += 1
            #------------------------------------------------------------------------

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
        final_fde = _apply_count_sketch_to_vector(
            out_fde, config.final_projection_dimension, config.seed
        )
        return final_fde, partition_counts_all, learned_centers

    return out_fde, partition_counts_all, learned_centers

def generate_query_fde(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig, query_or_doc: bool,
    kmeans_centers: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a Fixed Dimensional Encoding for a query point cloud (using SUM)."""
    if config.fill_empty_partitions:
        raise ValueError(
            "Query FDE generation does not support 'fill_empty_partitions'."
        )
    query_config = replace(config, encoding_type=EncodingType.DEFAULT_SUM)
    return _generate_fde_internal(point_cloud, query_config, query_or_doc, kmeans_centers)

def generate_document_fde(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig, query_or_doc: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a Fixed Dimensional Encoding for a document point cloud (using AVERAGE)."""
    doc_config = replace(config, encoding_type=EncodingType.AVERAGE)
    return _generate_fde_internal(point_cloud, doc_config, query_or_doc)

def generate_fde(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig, query_or_doc: bool,
    kmeans_centers: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if config.encoding_type == EncodingType.DEFAULT_SUM:
        return generate_query_fde(point_cloud, config, query_or_doc, kmeans_centers)
    elif config.encoding_type == EncodingType.AVERAGE:
        return generate_document_fde(point_cloud, config, query_or_doc)
    else:
        raise ValueError(f"Unsupported encoding type in config: {config.encoding_type}")

# ---------------------------------------------
# Batch document FDE generation (vectorized)
# + parallel empty-partition filling (UPDATED)
# ---------------------------------------------
def generate_document_fde_batch(
    doc_embeddings_list: List[np.ndarray],
    config: FixedDimensionalEncodingConfig,
    *,
    memmap_path: Optional[str] = None,           # e.g., "/path/to/fde_index.mmap"
    max_bytes_in_memory: int = 2 * 1024**3,      # 2GB safety threshold
    log_every: int = 10000,                       # progress logging
    flush_interval: int = 1000,                   # 배치별 flush 간격
    kmeans_centers: Optional[np.ndarray] = None,  # 사전 학습된 K-means centers
    #[1017] simhash별 indice별 원소 개수 저장 필요------------------------------------
    simhash_count_path: Optional[str] = None,
    simhash_counter_array: Optional[np.ndarray] = None,
    #------------------------------------------------------------------------
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Streaming implementation: no np.vstack; processes docs one by one.
    Optionally writes output to a numpy.memmap on disk if the full matrix would be large.
    """
    batch_start_time = time.perf_counter()
    num_docs = len(doc_embeddings_list)

    if num_docs == 0:
        logging.warning("[FDE Batch] Empty document list provided")
        return np.array([]), np.array([]), np.array([])

    # Validate + summarize
    for i, doc in enumerate(doc_embeddings_list):
        if doc.ndim != 2:
            raise ValueError(f"Document {i} has invalid shape (ndim={doc.ndim})")
        if doc.shape[1] != config.dimension:
            raise ValueError(
                f"Document {i} has incorrect dim: expected {config.dimension}, got {doc.shape[1]}"
            )

    use_identity_proj = config.projection_type == ProjectionType.DEFAULT_IDENTITY
    if use_identity_proj:
        projection_dim = config.dimension
        logging.info(f"[FDE Batch] Using identity projection (dim={projection_dim})")
    else:
        if not config.projection_dimension or config.projection_dimension <= 0:
            raise ValueError(
                "A positive projection_dimension must be specified for non-identity projections"
            )
        projection_dim = config.projection_dimension
        logging.info(
            f"[FDE Batch] Using {config.projection_type.name} projection: "
            f"{config.dimension} -> {projection_dim}"
        )

    num_partitions = 2 ** config.num_simhash_projections
    final_fde_dim_per_rep = num_partitions * projection_dim
    final_fde_dim = config.num_repetitions * final_fde_dim_per_rep

    # Decide where to place output (RAM or memmap)
    out_bytes = num_docs * final_fde_dim * 4  # float32
    
    if memmap_path or out_bytes > max_bytes_in_memory:
        if memmap_path is None:
            memmap_path = os.path.join(
                pathlib.Path(".").absolute(),
                f"fde_index_{final_fde_dim}d_{num_docs}n.mmap",
            )
        logging.info(f"[FDE Batch] Using memmap for output: {memmap_path} (~{out_bytes/1e9:.2f} GB)")
        out_fdes = np.memmap(memmap_path, mode="w+", dtype=np.float32, shape=(num_docs, final_fde_dim))
        memmap_used = True
    else:
        out_fdes = np.zeros((num_docs, final_fde_dim), dtype=np.float32)
        memmap_used = False

    # Precompute partition target bits table for empty-part filling (small; P x b)
    def _partition_bits_table(num_bits: int) -> np.ndarray:
        P = 1 << num_bits
        gray = np.arange(P, dtype=np.uint32)
        binary = gray.copy()
        g = gray.copy()
        while True:
            g >>= 1
            if not g.any():
                break
            binary ^= g
        shifts = np.arange(num_bits - 1, -1, -1, dtype=np.uint32)
        bits = ((binary[:, None] >> shifts[None, :]) & 1).astype(np.uint8)  # [P, b]
        return bits

    part_bits_tbl = _partition_bits_table(config.num_simhash_projections) if config.fill_empty_partitions else None

    # partition_counter[doc_idx][rep_num][partition_idx] = count
    # 3D numpy array: (num_docs, num_repetitions, num_partitions)
    partition_counter = np.zeros((num_docs, config.num_repetitions, num_partitions), dtype=np.int32)
    
    # K-means centers 저장용
    kmeans_centers_all = np.zeros((config.num_repetitions, num_partitions, projection_dim), dtype=np.float32)

    # For each repetition, stream over docs
    for rep_num in range(config.num_repetitions):
        current_seed = config.seed + rep_num
        if rep_num % 5 == 0:
            logging.info(
                f"[FDE Batch] Processing repetition {rep_num + 1}/{config.num_repetitions}"
            )

        if use_identity_proj:
            ams_matrix = None
        elif config.projection_type == ProjectionType.AMS_SKETCH:
            ams_matrix = _ams_projection_matrix_from_seed(
                config.dimension, projection_dim, current_seed
            )  # [D, Pdim]
        else:
            raise ValueError(f"Unsupported projection type: {config.projection_type}")

        rep_offset = rep_num * final_fde_dim_per_rep
        
        # [수정] 사전 학습된 K-means centers 사용 또는 새로 학습
        if kmeans_centers is not None:
            # 사전 학습된 centers 사용
            kmeans_centers_all[rep_num] = kmeans_centers[rep_num]
            num_centers = kmeans_centers[rep_num].shape[0]
            logging.info(f"[FDE Batch] Rep {rep_num}: Using pre-trained K-means centers with {num_centers} partitions (shape: {kmeans_centers[rep_num].shape})")
        else:
            # 기존 방식: 각 repetition마다 랜덤 샘플링으로 K-means centers 학습
            if config.use_memory_based_sampling:
                dynamic_sample_ratio = _calculate_memory_based_sample_ratio(
                    num_docs, projection_dim, num_partitions, config.target_memory_gb,
                    min_ratio=0.05, max_ratio=0.3
                )
            else:
                dynamic_sample_ratio = _calculate_dynamic_sample_ratio(
                    config.num_simhash_projections, config.kmeans_sample_ratio
                )
            
            # 랜덤으로 문서들을 선택하여 projected points 수집
            rng = np.random.default_rng(current_seed)
            n_sample_docs = max(int(num_docs * dynamic_sample_ratio), 1)
            sample_doc_indices = rng.choice(num_docs, size=n_sample_docs, replace=False)
            
            logging.info(f"[FDE Batch] Rep {rep_num}: Dynamic sampling ratio={dynamic_sample_ratio:.3f}, "
                        f"Sampling {n_sample_docs} docs from {num_docs} total docs for K-means")
            
            all_projected_points = []
            for doc_idx in sample_doc_indices:
                X_temp = doc_embeddings_list[doc_idx].astype(np.float32, copy=False)
                if use_identity_proj:
                    Pts_temp = X_temp
                else:
                    Pts_temp = X_temp @ ams_matrix
                all_projected_points.append(Pts_temp)
            
            all_projected_points = np.vstack(all_projected_points)
            kmeans_centers_all[rep_num] = _sample_and_train_kmeans(
                all_projected_points, num_partitions, current_seed, dynamic_sample_ratio
            )
            
            logging.info(f"[FDE Batch] Rep {rep_num}: K-means centers learned with {num_partitions} partitions from {len(all_projected_points)} points")

        # Stream over documents (no vstack)
        for d in range(num_docs):
            X = doc_embeddings_list[d].astype(np.float32, copy=False)  # [Ld, D]
            Ld = X.shape[0]
            
            # if Ld == 0:
            #     # leave zeros for this doc/rep
            #     if (d + 1) % log_every == 0:
            #         logging.info(f"[FDE Batch] rep {rep_num} doc {d+1}/{num_docs}: empty doc")
            #     continue

            # [수정] K-means 기반 partition index 계산
            if use_identity_proj:
                Pts = X
            else:
                Pts = X @ ams_matrix
            
            p_idx = _kmeans_partition_index(Pts, kmeans_centers_all[rep_num])

            # Aggregate per partition for this doc
            rep_sum = np.zeros((num_partitions, projection_dim), dtype=np.float32)  # [P, Pdim]
            counts = np.zeros(num_partitions, dtype=np.int32)

            # counts
            np.add.at(counts, p_idx, 1)

            # sums (scatter-add per feature)
            # rep_sum[p_idx[k], :] += Pts[k, :]
            for feat in range(projection_dim):
                np.add.at(rep_sum[:, feat], p_idx, Pts[:, feat])

            # Average where counts > 0
            nz = counts > 0
            if nz.any():
                rep_sum[nz, :] /= counts[nz, None]

            # Optional: fill empty partitions with nearest point (by Euclidean distance)
            if config.fill_empty_partitions and (~nz).any():
                empties = np.flatnonzero(~nz)
                # Find nearest points using Euclidean distance in projected space
                if len(Pts) > 0:
                    # Calculate distances from empty partition centers to all points
                    empty_centers = kmeans_centers_all[rep_num][empties]  # [E, projection_dim]
                    # distances: [E, Ld] - distance from each empty center to each point
                    distances = np.sqrt(np.sum((empty_centers[:, None, :] - Pts[None, :, :]) ** 2, axis=2))
                    nearest_local = np.argmin(distances, axis=1)   # [E]
                    rep_sum[empties, :] = Pts[nearest_local, :]

            # [Changed] Store partition counts for this document and repetition
            partition_counter[d, rep_num, :] = counts

            # Write this doc's rep chunk
            out_fdes[d, rep_offset:rep_offset + final_fde_dim_per_rep] = rep_sum.reshape(-1)

            # 배치별 flush (메모리 효율성)
            if (d + 1) % flush_interval == 0 and memmap_used and hasattr(out_fdes, "flush"):
                out_fdes.flush()

            if (d + 1) % log_every == 0:
                logging.info(f"[FDE Batch] rep {rep_num} doc {d+1}/{num_docs} processed")

        # If using memmap, ensure dirty pages are flushed each repetition
        if memmap_used and hasattr(out_fdes, "flush"):
            out_fdes.flush()

    # Final projection (count-sketch) if requested — done per-doc to stay streaming
    if config.final_projection_dimension and config.final_projection_dimension > 0:
        target_dim = config.final_projection_dimension
        logging.info(
            f"[FDE Batch] Applying final projection: {final_fde_dim} -> {target_dim}"
        )

        # If we already used memmap for intermediate, create a new memmap for final
        if memmap_used:
            final_path = os.path.splitext(memmap_path)[0] + f".final_{target_dim}.mmap"
            logging.info(f"[FDE Batch] Using memmap for final projection: {final_path}")
            final_out = np.memmap(final_path, mode="w+", dtype=np.float32, shape=(num_docs, target_dim))
        else:
            # Decide if final fits in RAM
            final_bytes = num_docs * target_dim * 4
            if final_bytes > max_bytes_in_memory:
                final_path = os.path.join(
                    pathlib.Path(".").absolute(),
                    f"fde_index_final_{target_dim}d_{num_docs}n.mmap",
                )
                logging.info(f"[FDE Batch] Using memmap for final projection: {final_path}")
                final_out = np.memmap(final_path, mode="w+", dtype=np.float32, shape=(num_docs, target_dim))
            else:
                final_out = np.zeros((num_docs, target_dim), dtype=np.float32)

        for d in range(num_docs):
            vec = out_fdes[d]  # 1D view
            final_out[d] = _apply_count_sketch_to_vector(vec, target_dim, config.seed)
            
            # 배치별 flush (메모리 효율성)
            if (d + 1) % flush_interval == 0 and hasattr(final_out, "flush"):
                final_out.flush()
            
            if (d + 1) % log_every == 0:
                logging.info(f"[FDE Batch] final-proj doc {d+1}/{num_docs}")

        if hasattr(final_out, "flush"):
            final_out.flush()
        out_fdes = final_out  # replace with final

    total_time = time.perf_counter() - batch_start_time
    logging.info(f"[FDE Batch] Batch generation completed in {total_time:.3f}s")
    logging.info(f"[FDE Batch] Output shape: {out_fdes.shape}")
    logging.info(f"[FDE Batch] Partition counter keys: {len(partition_counter)} documents")

    return out_fdes, partition_counter, kmeans_centers_all

# -------------------------
# Simple sanity test runner
# -------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print(f"\n{'=' * 20} SCENARIO 1: Basic FDE Generation {'=' * 20}")

    base_config = FixedDimensionalEncodingConfig(
        dimension=128, num_repetitions=2, num_simhash_projections=4, seed=42
    )

    # 임베딩 데이터 종류별 token 설정 : 쿼리 = 32 tokens, 문서 = 80 tokens
    query_data = np.random.randn(32, base_config.dimension).astype(np.float32)
    doc_data = np.random.randn(80, base_config.dimension).astype(np.float32)

    # FDE 생성
    #[1017] simhash별 indice별 원소 개수 저장 필요------------------------------------
    query_or_doc = True  # True: query, False: doc
    #------------------------------------------------------------------------

    query_fde, partition_counts, _ = generate_query_fde(query_data, base_config, query_or_doc)

    #[1017] simhash bool 변경 필요------------------------------------
    query_or_doc = False
    #------------------------------------------------------------------------
    doc_fde, partition_counts, learned_centers = generate_document_fde(
        doc_data, replace(base_config, fill_empty_partitions=True), query_or_doc
    )
    
    # Query FDE를 document에서 학습한 centers로 재생성
    query_fde_with_centers, _, _ = generate_query_fde(query_data, base_config, True, learned_centers)

    expected_dim = (
        base_config.num_repetitions
        * (2**base_config.num_simhash_projections)
        * base_config.dimension
    )
    # 동적 샘플링 비율 예시 출력
    dynamic_ratio = _calculate_dynamic_sample_ratio(base_config.num_simhash_projections, base_config.kmeans_sample_ratio)
    print(f"Dynamic sampling ratio for {base_config.num_simhash_projections} projections: {dynamic_ratio:.3f}")
    
    # 메모리 기반 샘플링 비율 예시 출력
    memory_ratio = _calculate_memory_based_sample_ratio(
        num_docs=1000,  # 가상의 문서 수
        embedding_dim=base_config.dimension,
        num_partitions=2**base_config.num_simhash_projections,
        target_memory_gb=2.0
    )
    print(f"Memory-based sampling ratio for {base_config.dimension}D embeddings: {memory_ratio:.3f}")
    
    print(f"Query FDE Shape: {query_fde.shape} (Expected: {expected_dim})")
    print(f"Document FDE Shape: {doc_fde.shape} (Expected: {expected_dim})")
    print(f"Query FDE with Centers Shape: {query_fde_with_centers.shape} (Expected: {expected_dim})")
    print(f"Similarity Score (original): {np.dot(query_fde, doc_fde):.4f}")
    print(f"Similarity Score (with centers): {np.dot(query_fde_with_centers, doc_fde):.4f}")
    assert query_fde.shape[0] == expected_dim
    assert query_fde_with_centers.shape[0] == expected_dim

    print(f"\n{'=' * 20} SCENARIO 2: Inner Projection (AMS Sketch) {'=' * 20}")

    ams_config = replace(
        base_config, projection_type=ProjectionType.AMS_SKETCH, projection_dimension=16
    )
    query_fde_ams, _, _ = generate_query_fde(query_data, ams_config, query_or_doc)
    expected_dim_ams = (
        ams_config.num_repetitions
        * (2**ams_config.num_simhash_projections)
        * ams_config.projection_dimension
    )
    print(f"AMS Sketch FDE Shape: {query_fde_ams.shape} (Expected: {expected_dim_ams})")
    assert query_fde_ams.shape[0] == expected_dim_ams

    print(f"\n{'=' * 20} SCENARIO 3: Final Projection (Count Sketch) {'=' * 20}")

    final_proj_config = replace(base_config, final_projection_dimension=1024)
    query_fde_final, _, _ = generate_query_fde(query_data, final_proj_config, query_or_doc)
    print(
        f"Final Projection FDE Shape: {query_fde_final.shape} (Expected: {final_proj_config.final_projection_dimension})"
    )
    assert query_fde_final.shape[0] == final_proj_config.final_projection_dimension

    print(f"\n{'=' * 20} SCENARIO 4: Top-level `generate_fde` wrapper {'=' * 20}")

    query_fde_2, _, _ = generate_fde(
        query_data, replace(base_config, encoding_type=EncodingType.DEFAULT_SUM), query_or_doc
    )
    doc_fde_2, _, _ = generate_fde(
        doc_data, replace(base_config, encoding_type=EncodingType.AVERAGE), query_or_doc
    )

    print(
        f"Wrapper-generated Query FDE is identical: {np.allclose(query_fde, query_fde_2)}"
    )
    print(
        f"Wrapper-generated Document FDE is identical: {np.allclose(doc_fde, doc_fde_2)}"
    )

    print("\nAll test scenarios completed successfully.")