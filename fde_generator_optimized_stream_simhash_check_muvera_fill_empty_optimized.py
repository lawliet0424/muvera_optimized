# -*- coding: utf-8 -*-
import logging
import time, pathlib, os
import numpy as np
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, List
from collections import defaultdict
from joblib import Parallel, delayed  # pip install joblib

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
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig, query_or_doc: bool
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

    #[1017] simhash별 indice별 원소 개수 저장 필요------------------------------------
    partition_counts_all = np.zeros((config.num_repetitions, num_partitions), dtype=np.int32)
    #------------------------------------------------------------------------

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
        return _apply_count_sketch_to_vector(
            out_fde, config.final_projection_dimension, config.seed
        )

    return out_fde, partition_counts_all

def generate_query_fde(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig, query_or_doc: bool
) -> np.ndarray:
    """Generates a Fixed Dimensional Encoding for a query point cloud (using SUM)."""
    if config.fill_empty_partitions:
        raise ValueError(
            "Query FDE generation does not support 'fill_empty_partitions'."
        )
    query_config = replace(config, encoding_type=EncodingType.DEFAULT_SUM)
    return _generate_fde_internal(point_cloud, query_config, query_or_doc)

def generate_document_fde(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig, query_or_doc: bool
) -> np.ndarray:
    """Generates a Fixed Dimensional Encoding for a document point cloud (using AVERAGE)."""
    doc_config = replace(config, encoding_type=EncodingType.AVERAGE)
    return _generate_fde_internal(point_cloud, doc_config, query_or_doc)

def generate_fde(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig, query_or_doc: bool
) -> np.ndarray:
    if config.encoding_type == EncodingType.DEFAULT_SUM:
        return generate_query_fde(point_cloud, config, query_or_doc)
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
    #[1017] simhash별 indice별 원소 개수 저장 필요------------------------------------
    simhash_count_path: Optional[str] = None,
    simhash_counter_array: Optional[np.ndarray] = None,
    #------------------------------------------------------------------------
) -> np.ndarray:
    """
    Streaming implementation: no np.vstack; processes docs one by one.
    Optionally writes output to a numpy.memmap on disk if the full matrix would be large.
    """
    batch_start_time = time.perf_counter()
    num_docs = len(doc_embeddings_list)

    if num_docs == 0:
        logging.warning("[FDE Batch] Empty document list provided")
        return np.array([])

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

    # For each repetition, stream over docs
    for rep_num in range(config.num_repetitions):
        current_seed = config.seed + rep_num
        if rep_num % 5 == 0:
            logging.info(
                f"[FDE Batch] Processing repetition {rep_num + 1}/{config.num_repetitions}"
            )

        # Projection matrices for this repetition
        simhash_matrix = _simhash_matrix_from_seed(
            config.dimension, config.num_simhash_projections, current_seed
        )  # [D, b]

        if use_identity_proj:
            ams_matrix = None
        elif config.projection_type == ProjectionType.AMS_SKETCH:
            ams_matrix = _ams_projection_matrix_from_seed(
                config.dimension, projection_dim, current_seed
            )  # [D, Pdim]
        else:
            raise ValueError(f"Unsupported projection type: {config.projection_type}")

        rep_offset = rep_num * final_fde_dim_per_rep

        # Stream over documents (no vstack)
        for d in range(num_docs):
            X = doc_embeddings_list[d].astype(np.float32, copy=False)  # [Ld, D]
            Ld = X.shape[0]
            
            # if Ld == 0:
            #     # leave zeros for this doc/rep
            #     if (d + 1) % log_every == 0:
            #         logging.info(f"[FDE Batch] rep {rep_num} doc {d+1}/{num_docs}: empty doc")
            #     continue

            # SimHash sketches
            sketches = X @ simhash_matrix                    # [Ld, b]
            bits = (sketches > 0).astype(np.uint32)          # [Ld, b]
            # Gray-code partition index (vectorized)
            p_idx = np.zeros(Ld, dtype=np.uint32)
            for b in range(config.num_simhash_projections):
                p_idx = (p_idx << 1) + (bits[:, b] ^ (p_idx & 1))  # Gray append

            # Projection
            if use_identity_proj:
                Pts = X                                       # [Ld, Pdim]
            else:
                Pts = X @ ams_matrix                           # [Ld, Pdim]

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

            # Optional: fill empty partitions with nearest point (by Hamming dist in sketch space)
            if config.fill_empty_partitions and (~nz).any():
                empties = np.flatnonzero(~nz)
                # Build doc bit table once: [Ld, b]
                doc_bits = (sketches > 0).astype(np.uint8)     # [Ld, b]
                tgt_bits = part_bits_tbl[empties]              # [E, b]
                # distances: [E, Ld]
                distances = np.sum(tgt_bits[:, None, :] ^ doc_bits[None, :, :], axis=2)
                
                # --------------------------------------------------------------
                # 각 empty partition에 대해 최소 해밍 거리 찾기
                min_distances = np.min(distances, axis=1, keepdims=True)  # [E, 1]
                # 최소 거리와 같은 거리를 가진 모든 점들을 마스킹
                mask = (distances == min_distances)  # [E, Ld]
                
                # 각 empty partition에 대해 마스킹된 점들의 평균 계산
                # Pts는 [Ld, proj_dim], mask는 [E, Ld]
                # 브로드캐스트: [E, Ld, proj_dim]
                masked_pts = Pts[None, :, :] * mask[:, :, None]  # [E, Ld, proj_dim]
                sum_pts = np.sum(masked_pts, axis=1)  # [E, proj_dim]
                count_pts = np.sum(mask, axis=1, keepdims=True)  # [E, 1]
                
                # 평균 계산 (0으로 나누기 방지)
                fill_values = np.where(count_pts > 0, sum_pts / count_pts, 0.0)  # [E, proj_dim]
                rep_sum[empties, :] = fill_values
                # --------------------------------------------------------------

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

    return out_fdes, partition_counter

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

    query_fde, partition_counts = generate_query_fde(query_data, base_config, query_or_doc)

    #[1017] simhash bool 변경 필요------------------------------------
    query_or_doc = False
    #------------------------------------------------------------------------
    doc_fde, partition_counts = generate_document_fde(
        doc_data, replace(base_config, fill_empty_partitions=True), query_or_doc
    )

    expected_dim = (
        base_config.num_repetitions
        * (2**base_config.num_simhash_projections)
        * base_config.dimension
    )
    print(f"Query FDE Shape: {query_fde.shape} (Expected: {expected_dim})")
    print(f"Document FDE Shape: {doc_fde.shape} (Expected: {expected_dim})")
    print(f"Similarity Score: {np.dot(query_fde, doc_fde):.4f}")
    assert query_fde.shape[0] == expected_dim

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
    query_fde_final, _ = generate_query_fde(query_data, final_proj_config, query_or_doc)
    print(
        f"Final Projection FDE Shape: {query_fde_final.shape} (Expected: {final_proj_config.final_projection_dimension})"
    )
    assert query_fde_final.shape[0] == final_proj_config.final_projection_dimension

    print(f"\n{'=' * 20} SCENARIO 4: Top-level `generate_fde` wrapper {'=' * 20}")

    query_fde_2, _ = generate_fde(
        query_data, replace(base_config, encoding_type=EncodingType.DEFAULT_SUM), query_or_doc
    )
    doc_fde_2, _ = generate_fde(
        doc_data, replace(base_config, encoding_type=EncodingType.AVERAGE), query_or_doc
    )

    print(
        f"Wrapper-generated Query FDE is identical: {np.allclose(query_fde, query_fde_2)}"
    )
    print(
        f"Wrapper-generated Document FDE is identical: {np.allclose(doc_fde, doc_fde_2)}"
    )

    print("\nAll test scenarios completed successfully.")