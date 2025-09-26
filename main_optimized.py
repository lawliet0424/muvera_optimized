# -*- coding: utf-8 -*-
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

from fde_generator_optimized import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_document_fde_batch,
)

# ======================
# --- Configuration ----
# ======================
DATASET_REPO_ID = "BeIR/scidocs"  # 사용되는 데이터셋 식별자(캐시 키에 포함)
COLBERT_MODEL_NAME = "raphaelsty/neural-cherche-colbert"
TOP_K = 10
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# 캐시 루트 디렉터리
CACHE_ROOT = os.path.join(pathlib.Path(__file__).parent.absolute(), "cache_fde")
os.makedirs(CACHE_ROOT, exist_ok=True)

# ======================
# --- Dataset Setup ----
# ======================
# scidocs 다운로드 (BEIR 포맷)
dataset = "scidocs"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

# ======================
# --- Logging Setup ----
# ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Using device: {DEVICE}")


# ===========================
# --- Helper Functions  -----
# ===========================
def load_nanobeir_dataset(repo_id: str):
    """Loads BEIR dataset from local 'data_path' in test split."""
    logging.info(f"Loading dataset from local path (BEIR): '{repo_id}'...")
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

    def __init__(self, model_name=COLBERT_MODEL_NAME):
        model = neural_cherche_models.ColBERT(model_name_or_path=model_name, device=DEVICE)
        self.ranker = neural_cherche_rank.ColBERT(key="id", on=["title", "text"], model=model)

        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=128,
            num_repetitions=1,
            num_simhash_projections=7,
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
        return os.path.join(CACHE_ROOT, dir_name)

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

        # 새로 빌드
        self.doc_ids = list(corpus.keys())
        documents_for_ranker = [{"id": doc_id, **corpus[doc_id]} for doc_id in self.doc_ids]

        logging.info(f"[{self.__class__.__name__}] Generating native multi-vector embeddings...")
        doc_embeddings_map = self.ranker.encode_documents(documents=documents_for_ranker)

        doc_embeddings_list = [to_numpy(doc_embeddings_map[doc_id]) for doc_id in self.doc_ids]

        logging.info(f"[{self.__class__.__name__}] Generating FDEs from ColBERT embeddings in BATCH mode...")
        self.fde_index = generate_document_fde_batch(doc_embeddings_list, self.doc_config)

        # 저장
        self._save_cache()

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
    # nltk tokenizer (일부 환경에서 필요한 경우)
    nltk.download('punkt', quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        pass

    corpus, queries, qrels = load_nanobeir_dataset(DATASET_REPO_ID)

    logging.info("Initializing retrieval models...")
    retrievers = {
        # "1. ColBERT (Native)": ColbertNativeRetriever(),  # 필요 시 동일 패턴으로 캐싱 적용 가능
        "2. ColBERT + FDE": ColbertFdeRetriever()
    }

    timings, final_results = {}, {}

    logging.info("--- PHASE 1: INDEXING ---")
    for name, retriever in retrievers.items():
        start_time = time.perf_counter()
        # 첫 실행에서 인덱스 생성 후 디스크 저장, 재실행에서는 자동 로드
        retriever.index(corpus)
        timings[name] = {"indexing_time": time.perf_counter() - start_time}
        logging.info(f"'{name}' indexing finished in {timings[name]['indexing_time']:.2f} seconds.")

    logging.info("--- PHASE 2: SEARCH & EVALUATION ---")
    for name, retriever in retrievers.items():
        logging.info(f"Running search for '{name}' on {len(queries)} queries...")

        # 선택: 전체 쿼리를 미리 캐싱해두면 반복 실행 시 매우 빠름
        if hasattr(retriever, "precompute_queries"):
            retriever.precompute_queries(queries)

        query_times = []
        results = {}
        for query_id, query_text in queries.items():
            start_time = time.perf_counter()
            # 캐시 키 안정화를 위해 query_id 전달
            results[str(query_id)] = retriever.search(query_text, query_id=str(query_id))
            query_times.append(time.perf_counter() - start_time)

        timings[name]["avg_query_time"] = np.mean(query_times)
        final_results[name] = results
        logging.info(f"'{name}' search finished. Avg query time: {timings[name]['avg_query_time'] * 1000:.2f} ms.")

    print("\n" + "=" * 85)
    print(f"{'FINAL REPORT':^85}")
    print(f"(Dataset: {DATASET_REPO_ID})")
    print("=" * 85)
    print(f"{'Retriever':<25} | {'Indexing Time (s)':<20} | {'Avg Query Time (ms)':<22} | {'Recall@{k}'.format(k=TOP_K):<10}")
    print("-" * 85)

    for name in retrievers.keys():
        recall = evaluate_recall(final_results[name], qrels, k=TOP_K)
        idx_time = timings[name]["indexing_time"]
        query_time_ms = timings[name]["avg_query_time"] * 1000
        print(f"{name:<25} | {idx_time:<20.2f} | {query_time_ms:<22.2f} | {recall:<10.4f}")

    print("=" * 85)