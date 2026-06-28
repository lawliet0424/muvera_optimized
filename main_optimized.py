# -*- coding: utf-8 -*-
import argparse
import socket
import time
from dataclasses import replace
from typing import Optional
import hashlib
import json
import logging
import os
import pathlib

import joblib
import nltk
import numpy as np
import torch

import neural_cherche.models as neural_cherche_models
import neural_cherche.rank as neural_cherche_rank

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from fde_generator_optimized_stream import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_document_fde_batch,
)
from perf_logger import PerfLogger, ShardPerfRecord

# ======================
# --- Configuration ----
# ======================
DATASET_REPO_ID = "treccovid"  # 사용되는 데이터셋 식별자(캐시 키에 포함)
DATASETS_ROOT = "/media/dcceris/datasets"
COLBERT_MODEL_NAME = "raphaelsty/neural-cherche-colbert"
TOP_K = 10
FILENAME = "main_optimized"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# 캐시 루트 (임베딩 전용 — cache_muvera)
CACHE_ROOT = os.path.join("/media/dcceris", "muvera_optimized", "cache_muvera", DATASET_REPO_ID, FILENAME)
os.makedirs(CACHE_ROOT, exist_ok=True)

# 결과물 저장 루트 (mmap, final_manifest.json, meta.json, metrics.csv, fde_index.pkl, doc_ids.json)
STORAGE_ROOT = os.path.join(os.path.expanduser("~"), "Desktop", "muvera_optimized", "storage", DATASET_REPO_ID, FILENAME)
os.makedirs(STORAGE_ROOT, exist_ok=True)

# 공통 문서 임베딩 디렉터리
COMMON_EMBEDS_DIR = os.path.join("/media/dcceris", "muvera_optimized", "cache_muvera", DATASET_REPO_ID)
COMMON_DOC_EMBEDS_DIR = os.path.join(COMMON_EMBEDS_DIR, "doc_embeds")
os.makedirs(COMMON_DOC_EMBEDS_DIR, exist_ok=True)

ATOMIC_BATCH_SIZE = 12000  # 배치 크기 기본값 (--batch-size 미지정 시)

# ======================
# --- Dataset Setup ----
# ======================
# dataset = "treccovid"
# url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# data_path = util.download_and_unzip(url, out_dir)

# ======================
# --- Logging Setup ----
# ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Using device: {DEVICE}")
logging.info(f"Embeddings cache (cache_muvera): {COMMON_DOC_EMBEDS_DIR}")
logging.info(f"Results storage: {STORAGE_ROOT}")


# ===========================
# --- Helper Functions  -----
# ===========================
def load_nanobeir_dataset(repo_id: str):
    """BEIR 데이터셋을 DATASETS_ROOT/{repo_id} 에서 로드."""
    data_path = os.path.join(DATASETS_ROOT, repo_id)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"  기대 경로: {DATASETS_ROOT}/<dataset_name>/ (BEIR 디렉터리)"
        )
    logging.info(f"Loading dataset from local path (BEIR): {data_path}")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    logging.info(f"Dataset loaded: {len(corpus)} documents, {len(queries)} queries.")
    return corpus, queries, qrels


def evaluate_recall(results: dict, qrels: dict, k: int) -> float:
    hits, total_queries = 0, 0
    for query_id, ranked_docs in results.items():
        relevant_docs = set(qrels.get(str(query_id), {}).keys())
        if not relevant_docs:
            continue
        total_queries += 1
        top_k_docs = set(list(ranked_docs.keys())[:k])
        if not relevant_docs.isdisjoint(top_k_docs):
            hits += 1
    return hits / total_queries if total_queries > 0 else 0.0


def to_numpy(tensor_or_array) -> np.ndarray:
    """Safely convert a PyTorch Tensor or a NumPy array to a float32 NumPy array."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().detach().numpy().astype(np.float32)
    elif isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    else:
        raise TypeError(f"Unsupported type for conversion: {type(tensor_or_array)}")


# =====================================
# --- ColBERT + FDE with Caching  -----
# =====================================
class ColbertFdeRetriever:
    """
    Uses a real ColBERT model to generate embeddings, then FDE for search.

    Caching:
      - After building the doc index once, saves it to disk.
      - On next runs, loads from disk if available.
      - Also caches query embeddings and FDE vectors per (dataset, model, config).
    """

    def __init__(
        self,
        model_name=COLBERT_MODEL_NAME,
        atomic_batch_size: int = ATOMIC_BATCH_SIZE,
        external_doc_embeds_dir: Optional[str] = COMMON_DOC_EMBEDS_DIR,
        num_repetitions: int = 1,
        num_simhash_projections: int = 7,
    ):
        model = neural_cherche_models.ColBERT(model_name_or_path=model_name, device=DEVICE)
        self.ranker = neural_cherche_rank.ColBERT(key="id", on=["title", "text"], model=model)

        if atomic_batch_size <= 0:
            raise ValueError(f"atomic_batch_size must be positive, got {atomic_batch_size}")
        if num_repetitions <= 0:
            raise ValueError(f"num_repetitions must be positive, got {num_repetitions}")
        if num_simhash_projections <= 0:
            raise ValueError(f"num_simhash_projections must be positive, got {num_simhash_projections}")
        self.atomic_batch_size = atomic_batch_size
        self.external_doc_embeds_dir = external_doc_embeds_dir
        self._doc_pos = {}

        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=128,
            num_repetitions=num_repetitions,
            num_simhash_projections=num_simhash_projections,
            seed=42,
            fill_empty_partitions=True,
        )

        self.fde_index: Optional[np.ndarray] = None
        self.doc_ids = []

        # 캐시 경로 구성
        self._model_name = model_name
        self._cache_dir = self._compute_cache_dir(
            dataset=DATASET_REPO_ID,
            model_name=model_name,
            cfg=self.doc_config,
        )
        self._fde_path = os.path.join(self._cache_dir, "fde_index.pkl")
        self._ids_path = os.path.join(self._cache_dir, "doc_ids.json")
        self._meta_path = os.path.join(self._cache_dir, "meta.json")
        self._queries_dir = os.path.join(self._cache_dir, "queries")

        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._queries_dir, exist_ok=True)

    # --------- 캐시 경로/키 유틸 ---------
    def _compute_cache_dir(self, dataset: str, model_name: str, cfg) -> str:
        model_key = model_name.replace("/", "_")
        cfg_str = f"d{cfg.dimension}_r{cfg.num_repetitions}_p{cfg.num_simhash_projections}_seed{cfg.seed}_fill{int(cfg.fill_empty_partitions)}"
        raw = f"{dataset}|{model_key}|{cfg_str}"
        key = hashlib.md5(raw.encode()).hexdigest()[:10]
        dir_name = f"{dataset.replace('/', '_')}__{model_key}__{cfg_str}__{key}"
        # return os.path.join(CACHE_ROOT, dir_name)  # (기존) cache_fde
        return os.path.join(STORAGE_ROOT, dir_name)  # (신규) storage 에 결과물 저장

    def _cache_exists(self) -> bool:
        return os.path.exists(self._fde_path) and os.path.exists(self._ids_path)

    def _save_cache(self):
        joblib.dump(self.fde_index, self._fde_path)
        with open(self._ids_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f, ensure_ascii=False)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": DATASET_REPO_ID,
                    "model": self._model_name,
                    "doc_count": len(self.doc_ids),
                    "config": {
                        "dimension": self.doc_config.dimension,
                        "num_repetitions": self.doc_config.num_repetitions,
                        "num_simhash_projections": self.doc_config.num_simhash_projections,
                        "seed": self.doc_config.seed,
                        "fill_empty_partitions": self.doc_config.fill_empty_partitions,
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logging.info(f"[{self.__class__.__name__}] Saved index cache -> {self._cache_dir}")

    def _load_cache(self) -> bool:
        if not self._cache_exists():
            return False
        try:
            self.fde_index = joblib.load(self._fde_path)
            with open(self._ids_path, "r", encoding="utf-8") as f:
                self.doc_ids = json.load(f)
            logging.info(f"[{self.__class__.__name__}] Loaded index cache from {self._cache_dir}")
            return True
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Cache load failed ({e}). Will rebuild.")
            return False

    def _fde_memmap_path(self) -> str:
        return os.path.join(
            self._cache_dir,
            f"fde_index_memmap_{self.doc_config.num_repetitions}_"
            f"{self.doc_config.num_simhash_projections}.mmap",
        )

    def _cleanup_fde_cache(self, mmap_path: Optional[str] = None):
        """FDE 인덱스 캐시 삭제 — 다음 실행에서 FDE를 다시 생성하도록 함."""
        self.fde_index = None
        mmap_path = mmap_path or self._fde_memmap_path()
        for path in (self._fde_path, self._ids_path, self._meta_path, mmap_path):
            if not path or not os.path.exists(path):
                continue
            try:
                os.remove(path)
                logging.info(f"[{self.__class__.__name__}] Removed FDE cache: {path}")
            except OSError as e:
                logging.warning(f"[{self.__class__.__name__}] Failed to remove {path}: {e}")

    # --------- 쿼리 캐시 유틸 ---------
    def _query_key(self, query: str, query_id: Optional[str]) -> str:
        if query_id:
            return str(query_id)
        return hashlib.md5(query.encode("utf-8")).hexdigest()[:16]

    def _query_paths(self, key: str):
        emb_path = os.path.join(self._queries_dir, f"{key}.emb.npy")  # ColBERT 쿼리 임베딩
        fde_path = os.path.join(self._queries_dir, f"{key}.fde.npy")  # FDE 쿼리 벡터
        return emb_path, fde_path

    def _save_query_cache(self, key: str, query_embeddings: np.ndarray, query_fde: np.ndarray):
        emb_path, fde_path = self._query_paths(key)
        np.save(emb_path, query_embeddings)
        np.save(fde_path, query_fde)

    def _load_query_cache(self, key: str):
        emb_path, fde_path = self._query_paths(key)
        if os.path.exists(fde_path):
            try:
                query_fde = np.load(fde_path)
                query_embeddings = np.load(emb_path) if os.path.exists(emb_path) else None
                return query_embeddings, query_fde
            except Exception:
                return None, None
        return None, None

    # --------- Public API ---------
    def index(self, corpus: dict, force_rebuild: bool = False):
        # 캐시 로드 시도
        if not force_rebuild and self._load_cache():
            if len(self.doc_ids) == len(corpus):
                return
            else:
                logging.warning(
                    f"[{self.__class__.__name__}] Corpus size changed ({len(self.doc_ids)} -> {len(corpus)}). Rebuilding index."
                )

        self.doc_ids = list(corpus.keys())
        self._doc_pos = {d: i for i, d in enumerate(self.doc_ids)}

        num_partitions = 2 ** self.doc_config.num_simhash_projections
        final_fde_dim = self.doc_config.num_repetitions * num_partitions * self.doc_config.dimension

        fde_memmap_path = self._fde_memmap_path()
        fde_index = np.memmap(
            fde_memmap_path, mode="w+", dtype=np.float32, shape=(len(self.doc_ids), final_fde_dim)
        )

        batch_size = self.atomic_batch_size
        logging.info(
            f"[{self.__class__.__name__}] Processing {len(self.doc_ids)} documents "
            f"in atomic batches of {batch_size}..."
        )

        perf = PerfLogger("standalone")
        worker_id = f"standalone-{socket.gethostname()}"
        pipeline_start_ts = time.time()
        cumulative_flush_time = 0.0

        for batch_start in range(0, len(self.doc_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(self.doc_ids))
            batch_doc_ids = self.doc_ids[batch_start:batch_end]
            t_batch_start_perf = time.perf_counter()
            t_batch_req_ts = time.time()
            embed_time = 0.0

            logging.info(
                f"[Atomic Batch] Processing batch {batch_start // batch_size + 1}/"
                f"{(len(self.doc_ids) + batch_size - 1) // batch_size}: "
                f"docs {batch_start}-{batch_end - 1}"
            )

            batch_missing_ids = []
            for doc_id in batch_doc_ids:
                if self.external_doc_embeds_dir:
                    pos = self._doc_pos.get(doc_id)
                    if pos is not None:
                        ext = os.path.join(self.external_doc_embeds_dir, f"{pos:08d}.npy")
                        if os.path.exists(ext):
                            continue
                batch_missing_ids.append(doc_id)

            encoded_map = {}
            if batch_missing_ids:
                logging.info(f"[Atomic Batch] Encoding {len(batch_missing_ids)} missing documents...")
                t_embed_start = time.perf_counter()
                to_encode_docs = [{"id": did, **corpus[did]} for did in batch_missing_ids]
                encoded_map = self.ranker.encode_documents(documents=to_encode_docs)
                for did in batch_missing_ids:
                    arr = to_numpy(encoded_map[did])
                    if self.external_doc_embeds_dir:
                        pos = self._doc_pos.get(did)
                        if pos is not None:
                            common_path = os.path.join(self.external_doc_embeds_dir, f"{pos:08d}.npy")
                            if not os.path.exists(common_path):
                                os.makedirs(os.path.dirname(common_path), exist_ok=True)
                                np.save(common_path, arr)
                embed_time = time.perf_counter() - t_embed_start

            batch_embeddings = []
            for doc_id in batch_doc_ids:
                if doc_id in batch_missing_ids:
                    batch_embeddings.append(to_numpy(encoded_map[doc_id]))
                elif self.external_doc_embeds_dir:
                    pos = self._doc_pos[doc_id]
                    ext = os.path.join(self.external_doc_embeds_dir, f"{pos:08d}.npy")
                    batch_embeddings.append(np.load(ext).astype(np.float32))
                else:
                    raise RuntimeError(f"Missing embedding for doc_id={doc_id}")

            t_get_shard_done_ts = time.time()
            t_process_start_ts = time.time()
            t_fde_start_perf = time.perf_counter()
            batch_fde, fde_stats = generate_document_fde_batch(
                batch_embeddings, self.doc_config, return_stats=True
            )
            t_write_start_perf = time.perf_counter()
            fde_index[batch_start:batch_end] = batch_fde
            write_time = time.perf_counter() - t_write_start_perf
            fde_total = time.perf_counter() - t_fde_start_perf
            t_process_done_ts = time.time()

            _stats = fde_stats or {}
            pipeline_flush_time = float(_stats.get("flush_time", 0.0))
            if pipeline_flush_time > 0.0:
                logging.info(
                    f"[Atomic Batch] Pipeline flush time: {pipeline_flush_time:.3f} seconds"
                )

            flush_start = time.perf_counter()
            fde_index.flush()
            additional_flush_time = time.perf_counter() - flush_start
            if additional_flush_time > 0.001:
                logging.info(
                    f"[Atomic Batch] Additional flush time: {additional_flush_time:.3f} seconds"
                )

            batch_flush_time = pipeline_flush_time + additional_flush_time
            cumulative_flush_time += batch_flush_time

            t_batch_done_ts = time.time()
            end_to_end = time.perf_counter() - t_batch_start_perf

            perf.record_shard(ShardPerfRecord(
                shard_index=batch_start // batch_size,
                worker_id=worker_id,
                status="success",
                num_docs=len(batch_doc_ids),
                doc_start=batch_start,
                doc_end=batch_end,
                embed_time=embed_time,
                prep_time=float(_stats.get("prep_time", 0.0)),
                upload_time=0.0,
                simhash_time=float(_stats.get("simhash_time", 0.0)),
                partition_time=float(_stats.get("partition_time", 0.0)),
                scatter_time=float(_stats.get("scatter_time", 0.0)),
                average_time=float(_stats.get("average_time", 0.0)),
                fill_time=float(_stats.get("fill_time", 0.0)),
                compute_time=float(_stats.get("compute_time", 0.0)),
                download_time=0.0,
                reshape_time=write_time,
                flush_time=batch_flush_time,
                fde_total=fde_total,
                grpc_recv_time=0.0,
                grpc_send_time=0.0,
                remote_flush_time=0.0,
                end_to_end=end_to_end,
                worker_get_shard_req_ts=t_batch_req_ts,
                worker_get_shard_done_ts=t_get_shard_done_ts,
                worker_process_start_ts=t_process_start_ts,
                worker_process_done_ts=t_process_done_ts,
                worker_upload_req_ts=0.0,
                worker_upload_done_ts=0.0,
                worker_status_report_ts=t_batch_done_ts,
                storage_get_shard_recv_ts=0.0,
                storage_get_shard_send_done_ts=0.0,
                storage_upload_recv_start_ts=0.0,
                storage_upload_done_ts=0.0,
                storage_status_recv_ts=0.0,
                worker_wall_total_s=t_batch_done_ts - t_batch_req_ts,
                storage_shard_wall_s=0.0,
            ))
            logging.info(
                "[PERF] batch=%d worker=%s docs=%d embed=%.4fs fde=%.4fs "
                "end_to_end=%.4fs wall_total=%.4fs ts_req=%.3f ts_done=%.3f",
                batch_start // batch_size, worker_id, len(batch_doc_ids),
                embed_time, fde_total, end_to_end, t_batch_done_ts - t_batch_req_ts,
                t_batch_req_ts, t_batch_done_ts,
            )

        final_flush_start = time.perf_counter()
        fde_index.flush()
        final_flush_time = time.perf_counter() - final_flush_start
        cumulative_flush_time += final_flush_time
        logging.info(f"[FDE Integration] Final integrated memmap completed: {fde_memmap_path}")
        logging.info(f"[FDE Integration] Final flush time: {final_flush_time:.3f} seconds")
        logging.info(f"[FDE Integration] Final shape: {fde_index.shape}")
        self.fde_index = fde_index

        pipeline_end_ts = time.time()
        pipeline_wall = {
            "first_get_shard_recv_ts": pipeline_start_ts,
            "last_status_recv_ts": pipeline_end_ts,
            "pipeline_wall_total_s": round(pipeline_end_ts - pipeline_start_ts, 3),
        }
        metrics_csv = os.path.join(self._cache_dir, f"metrics_{batch_size}.csv")
        manifest_path = os.path.join(self._cache_dir, f"final_manifest_{batch_size}.json")
        perf.save_csv(metrics_csv)
        perf.save_final_manifest(
            manifest_path,
            mode="standalone",
            pipeline_wall_clock=pipeline_wall,
            mmap_path=fde_memmap_path,
        )
        perf.print_summary()
        logging.info(
            "[PERF] metrics saved: %s  manifest: %s  pipeline_wall_total=%.3fs  "
            "flush_total=%.3fs",
            metrics_csv, manifest_path, pipeline_wall["pipeline_wall_total_s"],
            cumulative_flush_time,
        )

        # # (기존) 전체 코퍼스 일괄 인코딩 + FDE 생성
        # documents_for_ranker = [{"id": doc_id, **corpus[doc_id]} for doc_id in self.doc_ids]
        # logging.info(f"[{self.__class__.__name__}] Generating native multi-vector embeddings...")
        # doc_embeddings_map = self.ranker.encode_documents(documents=documents_for_ranker)
        # doc_embeddings_list = [to_numpy(doc_embeddings_map[doc_id]) for doc_id in self.doc_ids]
        # logging.info(f"[{self.__class__.__name__}] Generating FDEs from ColBERT embeddings in BATCH mode...")
        # self.fde_index = generate_document_fde_batch(doc_embeddings_list, self.doc_config)

        # metrics/manifest는 유지하고, FDE 캐시만 삭제 (batch-size별 재실행 보장)
        self._cleanup_fde_cache(fde_memmap_path)

    def precompute_queries(self, queries: dict):
        """queries: {query_id: query_text} (BEIR 포맷). 캐시에 없는 쿼리만 생성하여 저장."""
        missing = 0
        for qid, qtext in queries.items():
            key = self._query_key(qtext, str(qid))
            _, fde = self._load_query_cache(key)
            if fde is not None:
                continue
            query_embeddings_map = self.ranker.encode_queries(queries=[qtext])
            query_embeddings = to_numpy(next(iter(query_embeddings_map.values())))
            query_config = replace(self.doc_config, fill_empty_partitions=False)
            query_fde = generate_query_fde(query_embeddings, query_config)
            self._save_query_cache(key, query_embeddings, query_fde)
            missing += 1
        logging.info(f"[{self.__class__.__name__}] Precomputed {missing} uncached queries.")

    def search(self, query: str, query_id: Optional[str] = None) -> dict:
        # 인덱스 준비 확인
        if self.fde_index is None or not self.doc_ids:
            if not self._load_cache():
                raise RuntimeError("FDE index is not built. Call index(corpus) first.")

        # 쿼리 캐시 확인
        key = self._query_key(query, query_id)
        _, cached_fde = self._load_query_cache(key)

        if cached_fde is None:
            query_embeddings_map = self.ranker.encode_queries(queries=[query])
            query_embeddings = to_numpy(next(iter(query_embeddings_map.values())))
            query_config = replace(self.doc_config, fill_empty_partitions=False)
            query_fde = generate_query_fde(query_embeddings, query_config)
            self._save_query_cache(key, query_embeddings, query_fde)
        else:
            query_fde = cached_fde

        scores = self.fde_index @ query_fde
        return dict(sorted(zip(self.doc_ids, scores), key=lambda item: item[1], reverse=True))


# ======================
# --- Main Script ------
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--simhash", type=int, default=7)
    parser.add_argument(
        "--batch-size", type=int, default=ATOMIC_BATCH_SIZE,
        help=f"Documents per batch (default: {ATOMIC_BATCH_SIZE})",
    )
    args = parser.parse_args()
    if args.rep <= 0:
        parser.error("--rep must be a positive integer")
    if args.simhash <= 0:
        parser.error("--simhash must be a positive integer")
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")

    # nltk tokenizer (일부 환경에서 필요한 경우)
    nltk.download('punkt', quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        pass

    corpus, queries, qrels = load_nanobeir_dataset(DATASET_REPO_ID)

    logging.info("Initializing retrieval models...")
    logging.info(
        f"Parameters: rep={args.rep}, simhash={args.simhash}, batch_size={args.batch_size}"
    )
    retriever = ColbertFdeRetriever(
        atomic_batch_size=args.batch_size,
        num_repetitions=args.rep,
        num_simhash_projections=args.simhash,
    )

    logging.info("--- PHASE 1: INDEXING ---")
    start_time = time.perf_counter()
    retriever.index(corpus)
    total_indexing_time = time.perf_counter() - start_time
    logging.info(f"Indexing finished in {total_indexing_time:.2f} seconds.")

    # logging.info("--- PHASE 2: SEARCH & EVALUATION ---")
    # retrievers = {"2. ColBERT + FDE": retriever}
    # timings, final_results = {}, {}
    # for name, r in retrievers.items():
    #     logging.info(f"Running search for '{name}' on {len(queries)} queries...")
    #     if hasattr(r, "precompute_queries"):
    #         r.precompute_queries(queries)
    #     query_times = []
    #     results = {}
    #     for query_id, query_text in queries.items():
    #         start_time = time.perf_counter()
    #         results[str(query_id)] = r.search(query_text, query_id=str(query_id))
    #         query_times.append(time.perf_counter() - start_time)
    #     timings[name] = {"avg_query_time": np.mean(query_times)}
    #     final_results[name] = results
    #     logging.info(f"'{name}' search finished. Avg query time: {timings[name]['avg_query_time'] * 1000:.2f} ms.")
    #
    # print("\n" + "=" * 85)
    # print(f"{'FINAL REPORT':^85}")
    # print(f"(Dataset: {DATASET_REPO_ID})")
    # print("=" * 85)
    # print(f"{'Retriever':<25} | {'Indexing Time (s)':<20} | {'Avg Query Time (ms)':<22} | {'Recall@{k}'.format(k=TOP_K):<10}")
    # print("-" * 85)
    # for name in retrievers.keys():
    #     recall = evaluate_recall(final_results[name], qrels, k=TOP_K)
    #     query_time_ms = timings[name]["avg_query_time"] * 1000
    #     print(f"{name:<25} | {total_indexing_time:<20.2f} | {query_time_ms:<22.2f} | {recall:<10.4f}")
    # print("=" * 85)