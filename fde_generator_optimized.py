# -*- coding: utf-8 -*-
import logging
import time
import numpy as np
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, List
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

def generate_document_fde(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    """Generates a Fixed Dimensional Encoding for a document point cloud (using AVERAGE)."""
    doc_config = replace(config, encoding_type=EncodingType.AVERAGE)
    return _generate_fde_internal(point_cloud, doc_config)

def generate_fde(
    point_cloud: np.ndarray, config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    if config.encoding_type == EncodingType.DEFAULT_SUM:
        return generate_query_fde(point_cloud, config)
    elif config.encoding_type == EncodingType.AVERAGE:
        return generate_document_fde(point_cloud, config)
    else:
        raise ValueError(f"Unsupported encoding type in config: {config.encoding_type}")

# ---------------------------------------------
# Batch document FDE generation (vectorized)
# + parallel empty-partition filling (UPDATED)
# ---------------------------------------------
def generate_document_fde_batch(
    doc_embeddings_list: List[np.ndarray], config: FixedDimensionalEncodingConfig
) -> np.ndarray:
    """
    Generates FDEs for a batch of documents using highly optimized NumPy vectorization.
    Fully compliant with C++ implementation including all projection types.
    Now parallelizes the empty-partition filling step across documents.
    """
    batch_start_time = time.perf_counter()
    num_docs = len(doc_embeddings_list)

    if num_docs == 0:
        logging.warning("[FDE Batch] Empty document list provided")
        return np.array([])

    logging.info(f"[FDE Batch] Starting batch FDE generation for {num_docs} documents")

    # Input validation
    valid_docs = []
    for i, doc in enumerate(doc_embeddings_list):
        if doc.ndim != 2:
            logging.warning(
                f"[FDE Batch] Document {i} has invalid shape (ndim={doc.ndim}), skipping"
            )
            continue
        if doc.shape[1] != config.dimension:
            raise ValueError(
                f"Document {i} has incorrect dimension: expected {config.dimension}, got {doc.shape[1]}"
            )
        if doc.shape[0] == 0:
            logging.warning(f"[FDE Batch] Document {i} has no vectors, skipping")
            continue
        valid_docs.append(doc)

    if len(valid_docs) == 0:
        logging.warning("[FDE Batch] No valid documents after filtering")
        return np.array([])

    num_docs = len(valid_docs)
    doc_embeddings_list = valid_docs

    # Determine projection dimension (matching C++ logic)
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

    # Configuration summary
    num_partitions = 2**config.num_simhash_projections
    logging.info(
        f"[FDE Batch] Configuration: {config.num_repetitions} repetitions, "
        f"{num_partitions} partitions, projection_dim={projection_dim}"
    )

    # Document tracking
    doc_lengths = np.array([len(doc) for doc in doc_embeddings_list], dtype=np.int32)
    total_vectors = np.sum(doc_lengths)
    doc_boundaries = np.insert(np.cumsum(doc_lengths), 0, 0)
    doc_indices = np.repeat(np.arange(num_docs), doc_lengths)

    logging.info(
        f"[FDE Batch] Total vectors: {total_vectors}, avg per doc: {total_vectors / num_docs:.1f}"
    )

    # Concatenate all embeddings
    concat_start = time.perf_counter()
    all_points = np.vstack(doc_embeddings_list).astype(np.float32)
    concat_time = time.perf_counter() - concat_start
    logging.info(f"[FDE Batch] Concatenation completed in {concat_time:.3f}s")

    # Pre-allocate output
    final_fde_dim = config.num_repetitions * num_partitions * projection_dim
    out_fdes = np.zeros((num_docs, final_fde_dim), dtype=np.float32)
    logging.info(f"[FDE Batch] Output FDE dimension: {final_fde_dim}")

    # Process each repetition
    for rep_num in range(config.num_repetitions):
        current_seed = config.seed + rep_num

        if rep_num % 5 == 0:  # Log every 5 repetitions
            logging.info(
                f"[FDE Batch] Processing repetition {rep_num + 1}/{config.num_repetitions}"
            )

        # Step 1: SimHash projection
        simhash_start = time.perf_counter()
        simhash_matrix = _simhash_matrix_from_seed(
            config.dimension, config.num_simhash_projections, current_seed
        )
        all_sketches = all_points @ simhash_matrix
        simhash_time = time.perf_counter() - simhash_start

        # Step 2: Apply dimensionality reduction if configured
        proj_start = time.perf_counter()
        if use_identity_proj:
            projected_points = all_points
        elif config.projection_type == ProjectionType.AMS_SKETCH:
            ams_matrix = _ams_projection_matrix_from_seed(
                config.dimension, projection_dim, current_seed
            )
            projected_points = all_points @ ams_matrix
        else:
            raise ValueError(f"Unsupported projection type: {config.projection_type}")
        proj_time = time.perf_counter() - proj_start

        # Step 3: Vectorized partition index calculation
        partition_start = time.perf_counter()
        bits = (all_sketches > 0).astype(np.uint32)
        partition_indices = np.zeros(total_vectors, dtype=np.uint32)
        for bit_idx in range(config.num_simhash_projections):
            partition_indices = (partition_indices << 1) + (
                bits[:, bit_idx] ^ (partition_indices & 1)
            )
        partition_time = time.perf_counter() - partition_start

        # Step 4: Vectorized aggregation
        agg_start = time.perf_counter()

        rep_fde_sum = np.zeros(
            (num_docs * num_partitions * projection_dim,), dtype=np.float32
        )
        partition_counts = np.zeros((num_docs, num_partitions), dtype=np.int32)

        # Count vectors per partition per document
        np.add.at(partition_counts, (doc_indices, partition_indices), 1)

        # Aggregate vectors using flattened indexing for efficiency
        doc_part_indices = doc_indices * num_partitions + partition_indices
        base_indices = doc_part_indices * projection_dim
        for d in range(projection_dim):
            flat_indices = base_indices + d
            np.add.at(rep_fde_sum, flat_indices, projected_points[:, d])

        rep_fde_sum = rep_fde_sum.reshape(num_docs, num_partitions, projection_dim)
        agg_time = time.perf_counter() - agg_start

        # Step 5: Convert sums to averages + parallel empty fill (UPDATED)
        avg_start = time.perf_counter()

        # Vectorized division where counts > 0
        non_zero_mask = partition_counts > 0
        counts_3d = partition_counts[:, :, np.newaxis]  # [D, P, 1]
        np.divide(rep_fde_sum, counts_3d, out=rep_fde_sum, where=counts_3d > 0)

        # Fill empty partitions if configured — PARALLEL over documents
        empty_filled = 0
        if config.fill_empty_partitions:
            # Precompute partition target bits (Gray->Binary -> bits) once per repetition
            part_bits_tbl = _partition_bits_table(config.num_simhash_projections)  # [P, b]
            empty_mask = ~non_zero_mask  # [D, P]

            # Document ranges
            doc_ranges = [(doc_boundaries[d], doc_boundaries[d + 1]) for d in range(num_docs)]

            # Parallel execution per document
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(_fill_empty_partitions_for_doc)(
                    d,
                    config.num_simhash_projections,
                    num_partitions,
                    doc_ranges[d][0],
                    doc_ranges[d][1],
                    all_sketches,
                    projected_points,
                    empty_mask[d],
                    part_bits_tbl
                )
                for d in range(num_docs)
            )

            # Apply fills
            for d, empties, fill_block in results:
                if fill_block is None or empties.size == 0:
                    continue
                rep_fde_sum[d, empties, :] = fill_block
                empty_filled += empties.size

        avg_time = time.perf_counter() - avg_start

        # Step 6: Copy results to output array
        rep_output_start = rep_num * num_partitions * projection_dim
        out_fdes[
            :, rep_output_start : rep_output_start + num_partitions * projection_dim
        ] = rep_fde_sum.reshape(num_docs, -1)

        # Log timing for first repetition
        if rep_num == 0:
            logging.info("[FDE Batch] Repetition timing breakdown:")
            logging.info(f"  - SimHash: {simhash_time:.3f}s")
            logging.info(f"  - Projection: {proj_time:.3f}s")
            logging.info(f"  - Partition indices: {partition_time:.3f}s")
            logging.info(f"  - Aggregation: {agg_time:.3f}s")
            logging.info(f"  - Averaging+Fill(parallel): {avg_time:.3f}s")
            if config.fill_empty_partitions:
                logging.info(f"  - Filled {empty_filled} empty partitions")

    # Step 7: Apply final projection if configured
    if config.final_projection_dimension and config.final_projection_dimension > 0:
        logging.info(
            f"[FDE Batch] Applying final projection: {final_fde_dim} -> "
            f"{config.final_projection_dimension}"
        )
        final_proj_start = time.perf_counter()

        # Process in chunks to avoid memory issues
        chunk_size = min(100, num_docs)
        final_fdes = []

        for i in range(0, num_docs, chunk_size):
            chunk_end = min(i + chunk_size, num_docs)
            chunk_fdes = np.array(
                [
                    _apply_count_sketch_to_vector(
                        out_fdes[j], config.final_projection_dimension, config.seed
                    )
                    for j in range(i, chunk_end)
                ]
            )
            final_fdes.append(chunk_fdes)

        out_fdes = np.vstack(final_fdes)
        final_proj_time = time.perf_counter() - final_proj_start
        logging.info(
            f"[FDE Batch] Final projection completed in {final_proj_time:.3f}s"
        )

    # Final statistics and validation
    total_time = time.perf_counter() - batch_start_time
    logging.info(f"[FDE Batch] Batch generation completed in {total_time:.3f}s")
    logging.info(
        f"[FDE Batch] Average time per document: {total_time / num_docs * 1000:.2f}ms"
    )
    logging.info(f"[FDE Batch] Throughput: {num_docs / total_time:.1f} docs/sec")
    logging.info(f"[FDE Batch] Output shape: {out_fdes.shape}")

    # Validate output dimensions
    expected_dim = (
        final_fde_dim
        if not config.final_projection_dimension
        else config.final_projection_dimension
    )
    assert out_fdes.shape == (num_docs, expected_dim), (
        f"Output shape mismatch: {out_fdes.shape} != ({num_docs}, {expected_dim})"
    )

    return out_fdes

# -------------------------
# Simple sanity test runner
# -------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print(f"\n{'=' * 20} SCENARIO 1: Basic FDE Generation {'=' * 20}")

    base_config = FixedDimensionalEncodingConfig(
        dimension=128, num_repetitions=2, num_simhash_projections=4, seed=42
    )
    query_data = np.random.randn(32, base_config.dimension).astype(np.float32)
    doc_data = np.random.randn(80, base_config.dimension).astype(np.float32)

    query_fde = generate_query_fde(query_data, base_config)
    doc_fde = generate_document_fde(
        doc_data, replace(base_config, fill_empty_partitions=True)
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
    query_fde_ams = generate_query_fde(query_data, ams_config)
    expected_dim_ams = (
        ams_config.num_repetitions
        * (2**ams_config.num_simhash_projections)
        * ams_config.projection_dimension
    )
    print(f"AMS Sketch FDE Shape: {query_fde_ams.shape} (Expected: {expected_dim_ams})")
    assert query_fde_ams.shape[0] == expected_dim_ams

    print(f"\n{'=' * 20} SCENARIO 3: Final Projection (Count Sketch) {'=' * 20}")

    final_proj_config = replace(base_config, final_projection_dimension=1024)
    query_fde_final = generate_query_fde(query_data, final_proj_config)
    print(
        f"Final Projection FDE Shape: {query_fde_final.shape} (Expected: {final_proj_config.final_projection_dimension})"
    )
    assert query_fde_final.shape[0] == final_proj_config.final_projection_dimension

    print(f"\n{'=' * 20} SCENARIO 4: Top-level `generate_fde` wrapper {'=' * 20}")

    query_fde_2 = generate_fde(
        query_data, replace(base_config, encoding_type=EncodingType.DEFAULT_SUM)
    )
    doc_fde_2 = generate_fde(
        doc_data, replace(base_config, encoding_type=EncodingType.AVERAGE)
    )
    print(
        f"Wrapper-generated Query FDE is identical: {np.allclose(query_fde, query_fde_2)}"
    )
    print(
        f"Wrapper-generated Document FDE is identical: {np.allclose(doc_fde, doc_fde_2)}"
    )

    print("\nAll test scenarios completed successfully.")