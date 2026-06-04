# -*- coding: utf-8 -*-
import logging
import time, pathlib, os
import numpy as np
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, List, Tuple
from collections import defaultdict
from joblib import Parallel, delayed

import faiss

# PCA 사용을 위한 import
from sklearn.decomposition import PCA

class EncodingType(Enum):
    DEFAULT_SUM = 0
    AVERAGE = 1

class ProjectionType(Enum):
    DEFAULT_IDENTITY = 0
    AMS_SKETCH = 1
    PCA_PROJECTION = 2  # PCA projection 타입 추가


# =========================
# PCA projection matrices
# =========================
_PCA_PROJ_MATRICES = None  # type: Optional[object]
_PCA_TOP10_INDICES = None  # type: Optional[List[np.ndarray]]  # Top-20 dimension indices per repetition

def set_pca_projection_matrices(pca_mats, top10_indices=None):
    """Register PCA projection matrices.
    pca_mats: list[np.ndarray] or np.ndarray with shape (D, P).
    top10_indices: list of top-20 dimension indices per repetition (optional).
    """
    global _PCA_PROJ_MATRICES, _PCA_TOP10_INDICES
    _PCA_PROJ_MATRICES = pca_mats
    _PCA_TOP10_INDICES = top10_indices

def clear_pca_projection_matrices():
    global _PCA_PROJ_MATRICES, _PCA_TOP10_INDICES
    _PCA_PROJ_MATRICES = None
    _PCA_TOP10_INDICES = None

def _get_proj_matrix_for_rep(
    rep_idx: int,
    config: "FixedDimensionalEncodingConfig",
    original_dim: int,
    projection_dim: int,
    current_seed: int,
):
    """Return projection matrix for this repetition.
    Priority: identity > registered PCA > random AMS (if AMS_SKETCH).
    """
    if config.projection_type == ProjectionType.DEFAULT_IDENTITY:
        return None

    if config.projection_type == ProjectionType.PCA_PROJECTION:
        global _PCA_PROJ_MATRICES, _PCA_TOP10_INDICES
        if _PCA_PROJ_MATRICES is not None:
            mat = _PCA_PROJ_MATRICES[rep_idx] if isinstance(_PCA_PROJ_MATRICES, (list, tuple)) else _PCA_PROJ_MATRICES
            if mat is None:
                raise ValueError("PCA projection matrices registered but mat is None.")
            if mat.shape[0] != original_dim:
                raise ValueError(f"PCA proj mat shape mismatch: got {mat.shape}, expected ({original_dim}, ...)")
            
            # Top-20 dimension만 선택
            if _PCA_TOP10_INDICES is not None and rep_idx < len(_PCA_TOP10_INDICES):
                top10_idx = _PCA_TOP10_INDICES[rep_idx]
                mat_top10 = mat[:, top10_idx]  # (original_dim, 20)
                return mat_top10.astype(np.float32, copy=False)
            else:
                # Top-10 indices가 없으면 전체 차원 사용 (fallback)
                if mat.shape[1] != projection_dim:
                    raise ValueError(f"PCA proj mat shape mismatch: got {mat.shape}, expected ({original_dim}, {projection_dim})")
                return mat.astype(np.float32, copy=False)
        else:
            # PCA가 등록되지 않았으면 에러
            raise ValueError("PCA_PROJECTION type specified but no PCA matrices registered. Call set_pca_projection_matrices() first.")
    
    if config.projection_type == ProjectionType.AMS_SKETCH:
        # fallback to original AMS random sketch
        return _ams_projection_matrix_from_seed(original_dim, projection_dim, current_seed)

    raise ValueError(f"Unsupported projection_type: {config.projection_type}")


@dataclass
class FixedDimensionalEncodingConfig:
    dimension: int = 128
    num_repetitions: int = 10
    num_simhash_projections: int = 6  # partition 개수로 사용
    seed: int = 42
    encoding_type: EncodingType = EncodingType.DEFAULT_SUM
    projection_type: ProjectionType = ProjectionType.PCA_PROJECTION  # PCA를 기본값으로 변경
    projection_dimension: Optional[int] = None
    fill_empty_partitions: bool = False
    final_projection_dimension: Optional[int] = None
    # PCA 관련 설정
    pca_sample_ratio: float = 0.2  # PCA 학습을 위한 샘플링 비율
    use_pca_partition: bool = True  # PCA 사용 여부
    use_memory_based_sampling: bool = True
    target_memory_gb: float = 2.0

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
# PCA partition utilities
# ------------------------------
def _pca_partition_index(projected_points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """PCA projection 결과와 partition centers 간의 거리로 partition index 계산"""
    distances = np.linalg.norm(projected_points[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    return np.argmin(distances, axis=1)

def _pca_partition_index_gpu(projected_points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """GPU를 이용한 빠른 partition assignment"""
    X = projected_points.astype('float32', copy=False)
    C = centers.astype('float32', copy=False)
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, C.shape[1])
    index.add(C)
    _, labels = index.search(X, 1)
    return labels.ravel()

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
    
    def calculate_memory_for_sample_ratio(sample_ratio: float, actual_tokens_per_doc: Optional[int] = None) -> float:
        n_sample_docs = max(int(num_docs * sample_ratio), 1)
        
        if actual_tokens_per_doc is not None:
            avg_tokens_per_doc = actual_tokens_per_doc
        else:
            avg_tokens_per_doc = 100  # 보수적 추정
        
        sample_points_memory = n_sample_docs * avg_tokens_per_doc * embedding_dim * 4  # bytes
        centers_memory = num_partitions * embedding_dim * 4  # bytes
        overhead_memory = sample_points_memory * 0.2  # 20% overhead
        total_memory_bytes = sample_points_memory + centers_memory + overhead_memory
        total_memory_gb = total_memory_bytes / (1024 ** 3)
        
        return total_memory_gb
    
    # Binary search for optimal ratio
    low, high = min_ratio, max_ratio
    optimal_ratio = min_ratio
    
    for _ in range(10):
        mid = (low + high) / 2
        mem = calculate_memory_for_sample_ratio(mid, actual_tokens_per_doc)
        
        if mem <= target_memory_gb:
            optimal_ratio = mid
            low = mid
        else:
            high = mid
    
    return optimal_ratio

def _calculate_dynamic_sample_ratio(
    num_partitions: int,
    base_ratio: float = 0.2,
    min_ratio: float = 0.05,
    max_ratio: float = 0.3
) -> float:
    """Partition 수에 따라 동적으로 샘플링 비율 조정"""
    if num_partitions <= 16:
        return max_ratio
    elif num_partitions <= 64:
        return base_ratio
    else:
        return max(min_ratio, base_ratio * (64.0 / num_partitions))

def _train_pca_on_sample(
    embeddings_list: List[np.ndarray],
    config: FixedDimensionalEncodingConfig,
    num_partitions: int
) -> Tuple[List[PCA], List[np.ndarray]]:
    """
    샘플링된 데이터로 PCA를 학습하고 partition centers 계산
    
    Returns:
        pca_models: repetition별 PCA 모델 리스트
        pca_centers: repetition별 partition centers 리스트
    """
    original_dim = config.dimension
    projection_dim = config.projection_dimension or original_dim
    
    num_docs = len(embeddings_list)
    
    # 메모리 기반 샘플링 비율 계산
    if config.use_memory_based_sampling:
        avg_tokens = int(np.mean([len(emb) for emb in embeddings_list]))
        sample_ratio = _calculate_memory_based_sample_ratio(
            num_docs=num_docs,
            embedding_dim=original_dim,
            num_partitions=num_partitions,
            target_memory_gb=config.target_memory_gb,
            actual_tokens_per_doc=avg_tokens
        )
        logging.info(f"[PCA Training] Memory-based sampling ratio: {sample_ratio:.4f}")
    else:
        sample_ratio = _calculate_dynamic_sample_ratio(num_partitions, config.pca_sample_ratio)
        logging.info(f"[PCA Training] Dynamic sampling ratio: {sample_ratio:.4f}")
    
    # 샘플 문서 선택
    n_sample_docs = max(int(num_docs * sample_ratio), min(100, num_docs))
    sample_indices = np.random.choice(num_docs, size=n_sample_docs, replace=False)
    logging.info(f"[PCA Training] Selected {n_sample_docs} documents for PCA training")
    
    # 샘플 데이터 수집
    sample_points = []
    for idx in sample_indices:
        sample_points.append(embeddings_list[idx])
    sample_points = np.vstack(sample_points).astype(np.float32)
    logging.info(f"[PCA Training] Sample points shape: {sample_points.shape}")
    
    pca_models = []
    pca_centers = []
    top10_indices_list = []
    
    # repetition별로 PCA 학습
    for rep_num in range(config.num_repetitions):
        logging.info(f"[PCA Training] Training PCA for repetition {rep_num}")
        
        # PCA 학습 (전체 차원으로 학습)
        pca = PCA(n_components=projection_dim, random_state=config.seed + rep_num)
        pca.fit(sample_points)
        pca_models.append(pca)
        
        # PCA로 샘플 데이터 변환
        transformed_sample = pca.transform(sample_points).astype(np.float32)
        
        # Top-20 dimension만 선택 (explained_variance_ratio가 높은 상위 20개)
        explained_var = pca.explained_variance_ratio_
        top_10_indices = np.argsort(explained_var)[-20:][::-1]  # 상위 20개 인덱스 (내림차순)
        top_10_indices = np.sort(top_10_indices)  # 인덱스를 오름차순으로 정렬
        top10_indices_list.append(top_10_indices)
        
        logging.info(f"[PCA Training] Rep {rep_num}: Selected top-20 dimensions: {top_10_indices.tolist()}")
        logging.info(f"[PCA Training] Rep {rep_num}: Explained variance ratio (top-20): {explained_var[top_10_indices]}")
        
        # Top-20 dimension만 사용하여 변환
        transformed_sample_top10 = transformed_sample[:, top_10_indices]
        
        # 변환된 공간에서 partition centers 계산 (균등 분할)
        # num_partitions개의 centers를 계산하기 위해 transformed space를 균등 분할
        # Top-20 dimension만 사용하므로 centers도 20차원
        centers = np.zeros((num_partitions, 20), dtype=np.float32)
        
        # 각 차원을 num_partitions의 비트 수만큼 분할
        n_bits = config.num_simhash_projections
        for p_idx in range(num_partitions):
            # Gray code를 binary로 변환
            binary_idx = _gray_code_to_binary(p_idx)
            
            # 각 비트에 대응하는 축 값 설정 (top-20 dimension만 사용)
            for bit_pos in range(min(n_bits, 20)):
                bit_val = (binary_idx >> bit_pos) & 1
                # 해당 차원의 중앙값을 기준으로 +/- 설정
                dim_median = np.median(transformed_sample_top10[:, bit_pos])
                dim_std = np.std(transformed_sample_top10[:, bit_pos])
                centers[p_idx, bit_pos] = dim_median + (1 if bit_val else -1) * dim_std * 0.5
        
        pca_centers.append(centers)
        logging.info(f"[PCA Training] Repetition {rep_num} - PCA centers shape: {centers.shape} (top-20 dimensions)")
    
    return pca_models, pca_centers, top10_indices_list


def generate_query_fde(
    query_embeddings: np.ndarray,
    config: FixedDimensionalEncodingConfig,
    query_or_doc: bool,
    learned_pca_centers: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, defaultdict, Optional[List[np.ndarray]]]:
    """
    Query용 FDE 생성 (PCA 기반)
    
    Args:
        query_embeddings: (Lq, D) shape의 query token embeddings
        config: FDE 설정
        query_or_doc: True면 query, False면 document (partition counter 기록용)
        learned_pca_centers: document에서 학습된 PCA centers (있으면 사용)
    
    Returns:
        fde: 1D FDE vector
        partition_counter: partition별 token 개수
        pca_centers: PCA centers (learned_pca_centers가 None이면 새로 생성)
    """
    original_dim = config.dimension
    projection_dim = config.projection_dimension or original_dim
    num_partitions = 2 ** config.num_simhash_projections
    
    Lq = query_embeddings.shape[0]
    
    # PCA centers가 제공되지 않았으면 에러
    if learned_pca_centers is None and config.projection_type == ProjectionType.PCA_PROJECTION:
        raise ValueError("PCA centers must be provided for query FDE generation")
    
    pca_centers = learned_pca_centers
    
    # PCA projection matrices 등록 확인
    global _PCA_PROJ_MATRICES
    if _PCA_PROJ_MATRICES is None and config.projection_type == ProjectionType.PCA_PROJECTION:
        raise ValueError("PCA projection matrices not registered. Call set_pca_projection_matrices() first.")
    
    # Partition counter
    partition_counter = defaultdict(lambda: np.zeros((config.num_repetitions, num_partitions), dtype=np.int32))
    
    # FDE 배열 초기화 (top-20 dimension만 사용)
    final_fde_dim = config.num_repetitions * num_partitions * 20  # top-20 dimension만
    fde = np.zeros(final_fde_dim, dtype=np.float32)
    
    # Top-20 indices 가져오기
    global _PCA_TOP10_INDICES
    top10_idx = None
    if _PCA_TOP10_INDICES is not None and len(_PCA_TOP10_INDICES) > 0:
        # 첫 번째 repetition의 indices를 사용 (모든 repetition이 동일한 top-20을 사용한다고 가정)
        # 실제로는 각 repetition마다 다른 indices를 가질 수 있으므로 rep_num에 맞는 것을 사용
        if len(_PCA_TOP10_INDICES) > 0:
            top10_idx = _PCA_TOP10_INDICES[0] if len(_PCA_TOP10_INDICES) == 1 else None
    
    # Repetition별 처리
    for rep_num in range(config.num_repetitions):
        rep_offset = rep_num * num_partitions * 20  # top-20 dimension만
        
        # Top-20 indices 가져오기 (repetition별)
        if _PCA_TOP10_INDICES is not None and rep_num < len(_PCA_TOP10_INDICES):
            top10_idx = _PCA_TOP10_INDICES[rep_num]
        
        # Projection matrix 가져오기 (top-20 dimension만 반환)
        proj_matrix = _get_proj_matrix_for_rep(
            rep_num, config, original_dim, 20, config.seed + rep_num  # projection_dim=20
        )
        
        # Project embeddings (top-20 dimension만)
        if proj_matrix is not None:
            Pts_full = query_embeddings @ proj_matrix  # (Lq, Pdim)
            # proj_matrix가 top-20만 반환하지 않은 경우를 대비해 확인
            if Pts_full.shape[1] == 20:
                Pts = Pts_full  # (Lq, 20)
            elif top10_idx is not None:
                # 전체 차원으로 projection된 경우 top-20만 선택
                Pts = Pts_full[:, top10_idx].astype(np.float32, copy=False)  # (Lq, 20)
            else:
                # Fallback: 처음 20개 차원 사용
                Pts = Pts_full[:, :20].astype(np.float32, copy=False)  # (Lq, 20)
        else:
            # Identity projection인 경우에도 top-20 dimension만 선택
            if top10_idx is not None:
                Pts = query_embeddings[:, top10_idx].astype(np.float32, copy=False)  # (Lq, 20)
            else:
                # Fallback: 처음 20개 차원 사용
                Pts = query_embeddings[:, :20].astype(np.float32, copy=False)  # (Lq, 20)
        
        # Shape 검증
        if Pts.shape[1] != 20:
            raise ValueError(f"Pts shape mismatch: expected (Lq, 20), got {Pts.shape}. "
                           f"proj_matrix shape: {proj_matrix.shape if proj_matrix is not None else None}, "
                           f"top10_idx: {top10_idx}")
        
        # Assign to partitions using PCA centers (top-20 dimension)
        p_idx = _pca_partition_index(Pts, pca_centers[rep_num])  # (Lq,)
        
        # Aggregate per partition - 차원별 합산 후 평균 (top-20 dimension만)
        rep_sum = np.zeros((num_partitions, 20), dtype=np.float32)  # top-20 dimension만
        counts = np.zeros(num_partitions, dtype=np.int32)
        
        # Count tokens per partition
        np.add.at(counts, p_idx, 1)
        
        # Sum features per partition (top-20 dimension만)
        for feat in range(20):
            np.add.at(rep_sum[:, feat], p_idx, Pts[:, feat])
        
        # Average where counts > 0 (차원별 평균)
        nz = counts > 0
        if nz.any():
            rep_sum[nz, :] /= counts[nz, None]
        
        # Optional: fill empty partitions
        if config.fill_empty_partitions and (~nz).any():
            empties = np.flatnonzero(~nz)
            if len(Pts) > 0:
                empty_centers = pca_centers[rep_num][empties]
                distances = np.sqrt(np.sum((empty_centers[:, None, :] - Pts[None, :, :]) ** 2, axis=2))
                nearest_local = np.argmin(distances, axis=1)
                rep_sum[empties, :] = Pts[nearest_local, :]
        
        # Store partition counts
        partition_counter[0][rep_num, :] = counts
        
        # Write to FDE (top-20 dimension만)
        fde[rep_offset:rep_offset + num_partitions * 20] = rep_sum.reshape(-1)
    
    # Final projection if requested
    if config.final_projection_dimension and config.final_projection_dimension > 0:
        fde = _apply_count_sketch_to_vector(fde, config.final_projection_dimension, config.seed)
    
    return fde, partition_counter, pca_centers


def generate_document_fde(
    doc_embeddings: np.ndarray,
    config: FixedDimensionalEncodingConfig,
    query_or_doc: bool,
) -> Tuple[np.ndarray, defaultdict, List[np.ndarray]]:
    """
    Single document용 FDE 생성 (PCA 기반)
    
    Args:
        doc_embeddings: (Ld, D) shape의 document token embeddings
        config: FDE 설정
        query_or_doc: partition counter 기록용
    
    Returns:
        fde: 1D FDE vector
        partition_counter: partition별 token 개수
        pca_centers: 학습된 PCA centers
    """
    original_dim = config.dimension
    projection_dim = config.projection_dimension or original_dim
    num_partitions = 2 ** config.num_simhash_projections
    
    # PCA 학습 (단일 문서이므로 자기 자신으로 학습)
    pca_models, pca_centers, top10_indices_list = _train_pca_on_sample([doc_embeddings], config, num_partitions)
    
    # PCA projection matrices 등록 (top-10 indices 포함)
    pca_proj_matrices = [pca.components_.T for pca in pca_models]  # (D, P)
    set_pca_projection_matrices(pca_proj_matrices, top10_indices_list)
    
    # Query FDE 생성 함수 재사용
    fde, partition_counter, _ = generate_query_fde(
        doc_embeddings, config, query_or_doc, learned_pca_centers=pca_centers
    )
    
    return fde, partition_counter, pca_centers


def generate_document_fde_batch(
    embeddings_list: List[np.ndarray],
    config: FixedDimensionalEncodingConfig,
    query_or_doc: bool,
    memmap_path: Optional[str] = None,
    max_bytes_in_memory: float = 2e9,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, defaultdict, List[np.ndarray], List[np.ndarray]]:
    """
    배치 문서용 FDE 생성 (PCA 기반, 차원별 평균)
    
    Args:
        embeddings_list: 문서별 token embeddings 리스트
        config: FDE 설정
        query_or_doc: partition counter 기록용
        memmap_path: 메모리맵 파일 경로
        max_bytes_in_memory: 메모리 제한
        n_jobs: 병렬 처리 job 수
    
    Returns:
        out_fdes: (num_docs, final_fde_dim) FDE 배열
        partition_counter: partition별 token 개수
        pca_centers: 학습된 PCA centers
        top10_indices_list: Top-20 dimension indices per repetition
    """
    batch_start_time = time.perf_counter()
    
    num_docs = len(embeddings_list)
    original_dim = config.dimension
    projection_dim = config.projection_dimension or original_dim
    num_partitions = 2 ** config.num_simhash_projections
    
    logging.info(f"[FDE Batch PCA] Starting batch generation for {num_docs} documents")
    logging.info(f"[FDE Batch PCA] Config: dim={original_dim}, proj_dim={projection_dim}, "
                 f"reps={config.num_repetitions}, partitions={num_partitions}")
    
    # Step 1: PCA 학습
    logging.info("[FDE Batch PCA] Step 1: Training PCA models...")
    pca_start = time.perf_counter()
    pca_models, pca_centers, top10_indices_list = _train_pca_on_sample(embeddings_list, config, num_partitions)
    pca_time = time.perf_counter() - pca_start
    logging.info(f"[FDE Batch PCA] PCA training completed in {pca_time:.3f}s")
    
    # PCA projection matrices 등록 (top-10 indices 포함)
    pca_proj_matrices = [pca.components_.T for pca in pca_models]  # (D, P)
    set_pca_projection_matrices(pca_proj_matrices, top10_indices_list)
    
    # Step 2: FDE 생성 준비 (top-20 dimension만 사용)
    final_fde_dim_per_rep = num_partitions * 20  # top-20 dimension만 사용
    final_fde_dim = config.num_repetitions * final_fde_dim_per_rep
    
    if config.final_projection_dimension and config.final_projection_dimension > 0:
        out_dim = config.final_projection_dimension
    else:
        out_dim = final_fde_dim
    
    # 메모리 결정
    total_bytes = num_docs * out_dim * 4
    memmap_used = False
    
    if total_bytes > max_bytes_in_memory:
        if memmap_path is None:
            memmap_path = os.path.join(
                pathlib.Path(".").absolute(),
                f"fde_index_pca_{out_dim}d_{num_docs}n.mmap",
            )
        logging.info(f"[FDE Batch PCA] Using memmap: {memmap_path} ({total_bytes / 1e9:.2f} GB)")
        out_fdes = np.memmap(memmap_path, mode="w+", dtype=np.float32, shape=(num_docs, final_fde_dim))
        memmap_used = True
    else:
        logging.info(f"[FDE Batch PCA] Using RAM ({total_bytes / 1e9:.2f} GB)")
        out_fdes = np.zeros((num_docs, final_fde_dim), dtype=np.float32)
    
    partition_counter = {}
    
    # Step 3: 각 문서별 FDE 생성
    logging.info("[FDE Batch PCA] Step 3: Generating FDEs...")
    
    flush_interval = max(1, num_docs // 20)
    log_every = max(1, num_docs // 10)
    
    for rep_num in range(config.num_repetitions):
        rep_offset = rep_num * final_fde_dim_per_rep
        
        # Projection matrix (top-20 dimension만 사용)
        proj_matrix = _get_proj_matrix_for_rep(
            rep_num, config, original_dim, 20, config.seed + rep_num  # projection_dim=20
        )
        
        # Top-20 indices 가져오기
        global _PCA_TOP10_INDICES
        top10_idx = None
        if _PCA_TOP10_INDICES is not None and rep_num < len(_PCA_TOP10_INDICES):
            top10_idx = _PCA_TOP10_INDICES[rep_num]
        
        for d in range(num_docs):
            Ld = embeddings_list[d].shape[0]
            
            # 첫 번째 repetition에서 partition counter 초기화
            if rep_num == 0:
                partition_counter[d] = np.zeros((config.num_repetitions, num_partitions), dtype=np.int32)
            
            # Project (top-20 dimension만 사용)
            if proj_matrix is not None:
                Pts_full = embeddings_list[d] @ proj_matrix  # (Ld, Pdim)
                # proj_matrix가 top-20만 반환하지 않은 경우를 대비해 확인
                if Pts_full.shape[1] == 20:
                    Pts = Pts_full  # (Ld, 20)
                elif top10_idx is not None:
                    # 전체 차원으로 projection된 경우 top-20만 선택
                    Pts = Pts_full[:, top10_idx].astype(np.float32, copy=False)  # (Ld, 20)
                else:
                    # Fallback: 처음 20개 차원 사용
                    Pts = Pts_full[:, :20].astype(np.float32, copy=False)  # (Ld, 20)
            else:
                # Identity projection인 경우에도 top-20 dimension만 선택
                if top10_idx is not None:
                    Pts = embeddings_list[d][:, top10_idx].astype(np.float32, copy=False)  # (Ld, 20)
                else:
                    # Fallback: 처음 20개 차원 사용
                    Pts = embeddings_list[d][:, :20].astype(np.float32, copy=False)  # (Ld, 20)
            
            # Shape 검증
            if Pts.shape[1] != 20:
                raise ValueError(f"Pts shape mismatch: expected (Ld, 20), got {Pts.shape}. "
                               f"proj_matrix shape: {proj_matrix.shape if proj_matrix is not None else None}, "
                               f"top10_idx: {top10_idx}")
            
            # Partition assignment (top-20 dimension centers 사용)
            p_idx = _pca_partition_index(Pts, pca_centers[rep_num])  # (Ld,)
            
            # Aggregate per partition - 차원별 합산 후 평균 (top-20 dimension만)
            rep_sum = np.zeros((num_partitions, 20), dtype=np.float32)  # top-20 dimension만
            counts = np.zeros(num_partitions, dtype=np.int32)
            
            # Count
            np.add.at(counts, p_idx, 1)
            
            # Sum per feature dimension (top-20 dimension만)
            for feat in range(20):
                np.add.at(rep_sum[:, feat], p_idx, Pts[:, feat])
            
            # Average (차원별 평균)
            nz = counts > 0
            if nz.any():
                rep_sum[nz, :] /= counts[nz, None]
            
            # Fill empty partitions
            if config.fill_empty_partitions and (~nz).any():
                empties = np.flatnonzero(~nz)
                if len(Pts) > 0:
                    empty_centers = pca_centers[rep_num][empties]
                    distances = np.sqrt(np.sum((empty_centers[:, None, :] - Pts[None, :, :]) ** 2, axis=2))
                    nearest_local = np.argmin(distances, axis=1)
                    rep_sum[empties, :] = Pts[nearest_local, :]
            
            # Store counts
            partition_counter[d][rep_num, :] = counts
            
            # Write to output
            out_fdes[d, rep_offset:rep_offset + final_fde_dim_per_rep] = rep_sum.reshape(-1)
            
            # Flush periodically
            if (d + 1) % flush_interval == 0 and memmap_used and hasattr(out_fdes, "flush"):
                out_fdes.flush()
            
            if (d + 1) % log_every == 0:
                logging.info(f"[FDE Batch PCA] rep {rep_num} doc {d+1}/{num_docs} processed")
        
        # Flush after each repetition
        if memmap_used and hasattr(out_fdes, "flush"):
            out_fdes.flush()
    
    # Step 4: Final projection (optional)
    if config.final_projection_dimension and config.final_projection_dimension > 0:
        target_dim = config.final_projection_dimension
        logging.info(f"[FDE Batch PCA] Applying final projection: {final_fde_dim} -> {target_dim}")
        
        if memmap_used:
            final_path = os.path.splitext(memmap_path)[0] + f".final_{target_dim}.mmap"
            logging.info(f"[FDE Batch PCA] Using memmap for final projection: {final_path}")
            final_out = np.memmap(final_path, mode="w+", dtype=np.float32, shape=(num_docs, target_dim))
        else:
            final_bytes = num_docs * target_dim * 4
            if final_bytes > max_bytes_in_memory:
                final_path = os.path.join(
                    pathlib.Path(".").absolute(),
                    f"fde_index_pca_final_{target_dim}d_{num_docs}n.mmap",
                )
                logging.info(f"[FDE Batch PCA] Using memmap for final projection: {final_path}")
                final_out = np.memmap(final_path, mode="w+", dtype=np.float32, shape=(num_docs, target_dim))
            else:
                final_out = np.zeros((num_docs, target_dim), dtype=np.float32)
        
        for d in range(num_docs):
            vec = out_fdes[d]
            final_out[d] = _apply_count_sketch_to_vector(vec, target_dim, config.seed)
            
            if (d + 1) % flush_interval == 0 and hasattr(final_out, "flush"):
                final_out.flush()
            
            if (d + 1) % log_every == 0:
                logging.info(f"[FDE Batch PCA] final-proj doc {d+1}/{num_docs}")
        
        if hasattr(final_out, "flush"):
            final_out.flush()
        out_fdes = final_out
    
    total_time = time.perf_counter() - batch_start_time
    logging.info(f"[FDE Batch PCA] Batch generation completed in {total_time:.3f}s")
    logging.info(f"[FDE Batch PCA] Output shape: {out_fdes.shape}")
    
    return out_fdes, partition_counter, pca_centers, top10_indices_list


def generate_fde(
    embeddings: np.ndarray,
    config: FixedDimensionalEncodingConfig,
    query_or_doc: bool,
    learned_pca_centers: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, defaultdict, Optional[List[np.ndarray]]]:
    """
    통합 FDE 생성 wrapper (PCA 기반)
    """
    if learned_pca_centers is not None:
        # Query mode
        return generate_query_fde(embeddings, config, query_or_doc, learned_pca_centers)
    else:
        # Document mode
        return generate_document_fde(embeddings, config, query_or_doc)


# -------------------------
# Simple sanity test runner
# -------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print(f"\n{'=' * 20} SCENARIO 1: Basic FDE Generation with PCA {'=' * 20}")

    base_config = FixedDimensionalEncodingConfig(
        dimension=128,
        num_repetitions=2,
        num_simhash_projections=4,
        seed=42,
        projection_type=ProjectionType.PCA_PROJECTION,
        projection_dimension=64,  # PCA로 64차원으로 축소
    )

    # 테스트 데이터
    query_data = np.random.randn(32, base_config.dimension).astype(np.float32)
    doc_data = np.random.randn(80, base_config.dimension).astype(np.float32)

    # Document FDE 생성 (PCA 학습 포함)
    query_or_doc = False
    doc_fde, doc_partition_counts, learned_centers = generate_document_fde(
        doc_data, replace(base_config, fill_empty_partitions=True), query_or_doc
    )
    
    # Query FDE 생성 (학습된 PCA centers 사용)
    query_or_doc = True
    query_fde, query_partition_counts, _ = generate_query_fde(
        query_data, base_config, query_or_doc, learned_centers
    )

    num_partitions = 2 ** base_config.num_simhash_projections
    expected_dim = base_config.num_repetitions * num_partitions * base_config.projection_dimension
    
    print(f"Query FDE Shape: {query_fde.shape} (Expected: {expected_dim})")
    print(f"Document FDE Shape: {doc_fde.shape} (Expected: {expected_dim})")
    print(f"Similarity Score: {np.dot(query_fde, doc_fde):.4f}")
    assert query_fde.shape[0] == expected_dim
    assert doc_fde.shape[0] == expected_dim

    print(f"\n{'=' * 20} SCENARIO 2: Batch FDE Generation {'=' * 20}")

    # 배치 문서 데이터
    num_docs = 100
    doc_batch = [np.random.randn(50, base_config.dimension).astype(np.float32) for _ in range(num_docs)]
    
    batch_fdes, batch_partition_counts, batch_centers = generate_document_fde_batch(
        doc_batch,
        replace(base_config, fill_empty_partitions=True),
        query_or_doc=False,
        n_jobs=1,
    )
    
    print(f"Batch FDE Shape: {batch_fdes.shape} (Expected: ({num_docs}, {expected_dim}))")
    assert batch_fdes.shape == (num_docs, expected_dim)

    print(f"\n{'=' * 20} SCENARIO 3: Final Projection {'=' * 20}")

    final_proj_config = replace(base_config, final_projection_dimension=512)
    query_fde_final, _, _ = generate_query_fde(
        query_data, final_proj_config, True, learned_centers
    )
    print(f"Final Projection FDE Shape: {query_fde_final.shape} (Expected: {final_proj_config.final_projection_dimension})")
    assert query_fde_final.shape[0] == final_proj_config.final_projection_dimension

    print("\n✅ All test scenarios completed successfully with PCA!")