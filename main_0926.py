# -*- coding: utf-8 -*-
import os, json, time, hashlib, logging, pathlib
from collections import OrderedDict
from dataclasses import replace
from typing import Optional, List, Tuple

import nltk
import numpy as np
import torch
import joblib
import time

import neural_cherche.models as neural_cherche_models
import neural_cherche.rank as neural_cherche_rank
from datasets import load_dataset

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES


# FDE 구현 (업로드된 파일 사용)
from fde_generator_optimized_stream import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_document_fde_batch,
)

# ======================
# --- Configuration ----
# ======================
DATASET_REPO_ID = "scidocs"
COLBERT_MODEL_NAME = "raphaelsty/neural-cherche-colbert"
TOP_K = 10

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# 캐시 루트
CACHE_ROOT = os.path.join(pathlib.Path(__file__).parent.absolute(), "cache_muvera")
os.makedirs(CACHE_ROOT, exist_ok=True)

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
    # 데이터셋 준비 (BEIR trec-covid)
    dataset = "scidocs"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")

    if not os.path.exists(os.path.join(out_dir, dataset)):
        data_path = util.download_and_unzip(url, out_dir)
        logging.info(
            f"[Dataset] Downloaded and unzipped dataset from {url} to {out_dir}"
        )
    else:
        data_path = os.path.join(out_dir, dataset)
        logging.info(
            f"[Dataset] Dataset already exists in {out_dir}"
        )

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
# --- ColBERT + FDE + Reranking  ------
# =====================================
class ColbertFdeRetriever:
    """
    1) ColBERT 토큰 임베딩 → 문서 FDE 인덱스 생성(디스크 캐시)
    2) 쿼리 FDE로 1차 점수
    3) 상위 후보 N개를 Chamfer(MaxSim)로 재랭킹 후 반환
       - 외부 임베딩 디렉터리(external_doc_embeds_dir)에서 .npy가 있으면 인코딩 생략
       - 문서/쿼리 임베딩과 FDE 캐시
       - 쿼리별 지연시간을 파일에 "QID\\tSearch\\tRerank" 형식(ms)으로 기록
    """

    def __init__(
        self,
        model_name: str = COLBERT_MODEL_NAME,
        rerank_candidates: int = 100,
        enable_rerank: bool = True,
        save_doc_embeds: bool = True,
        latency_log_path: Optional[str] = None,
        external_doc_embeds_dir: Optional[str] = None,  # ★ 추가: 외부 임베딩 디렉터리
    ):
        model = neural_cherche_models.ColBERT(model_name_or_path=model_name, device=DEVICE)
        self.ranker = neural_cherche_rank.ColBERT(key="id", on=["title", "text"], model=model)

        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=128,
            num_repetitions=2,
            num_simhash_projections=5,
            seed=42,
            fill_empty_partitions=True,
        )

        self.fde_index: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        self._doc_pos = {}     # doc_id -> position
        self._corpus = None    # for on-the-fly encoding

        self.enable_rerank = enable_rerank
        self.rerank_candidates = rerank_candidates
        self.save_doc_embeds = save_doc_embeds
        self.external_doc_embeds_dir = external_doc_embeds_dir  # ★

        # 캐시 경로
        self._model_name = model_name
        self._cache_dir = self._compute_cache_dir(
            dataset=DATASET_REPO_ID, model_name=model_name, cfg=self.doc_config
        )
        self._fde_path = os.path.join(self._cache_dir, "fde_index.pkl")
        self._ids_path = os.path.join(self._cache_dir, "doc_ids.json")
        self._meta_path = os.path.join(self._cache_dir, "meta.json")
        self._queries_dir = os.path.join(self._cache_dir, "queries")
        self._doc_emb_dir = os.path.join(self._cache_dir, "doc_embeds")

        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._queries_dir, exist_ok=True)
        if self.save_doc_embeds:
            os.makedirs(self._doc_emb_dir, exist_ok=True)

        # 지연시간 로그 파일 (헤더 없이 누적)
        self._latency_log_path = latency_log_path or os.path.join(self._cache_dir, "latency.tsv")

    # --------- 경로/키 유틸 ---------
    def _compute_cache_dir(self, dataset: str, model_name: str, cfg) -> str:
        model_key = model_name.replace("/", "_")
        cfg_str = f"d{cfg.dimension}_r{cfg.num_repetitions}_p{cfg.num_simhash_projections}_seed{cfg.seed}_fill{int(cfg.fill_empty_partitions)}"
        raw = f"{dataset}|{model_key}|{cfg_str}"
        key = hashlib.md5(raw.encode()).hexdigest()[:10]
        dir_name = f"{dataset.replace('/', '_')}__{model_key}__{cfg_str}__{key}"
        return os.path.join(CACHE_ROOT, dataset)

    def _query_key(self, query_text: str, query_id: Optional[str]) -> str:
        base = (query_id or "") + "||" + query_text
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _query_paths(self, key: str) -> Tuple[str, str]:
        return (
            os.path.join(self._queries_dir, f"{key}.emb.npy"),
            os.path.join(self._queries_dir, f"{key}.fde.npy"),
        )

    def _doc_emb_path(self, doc_id: str) -> str:
        pos = self._doc_pos[doc_id]
        return os.path.join(self._doc_emb_dir, f"{pos:08d}.npy")

    def _external_doc_emb_path(self, doc_id: str) -> Optional[str]:
        """외부 디렉터리에서 기대하는 파일 경로(문서 순번 8자리 파일명)."""
        if not self.external_doc_embeds_dir:
            return None
        pos = self._doc_pos.get(doc_id)
        if pos is None:
            return None
        return os.path.join(self.external_doc_embeds_dir, f"{pos:08d}.npy")

    # --------- 저장/로드 ---------
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
        logging.info(f"[{self.__class__.__name__}] Saved FDE index cache to: {self._cache_dir}")

    def _load_cache(self) -> bool:
        # (사용자 코드 유지: 존재 체크 주석 처리)
        self.fde_index = joblib.load(self._fde_path)
        with open(self._ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        self._doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        logging.info(
            f"[{self.__class__.__name__}] Loaded FDE index cache: "
            f"{self.fde_index.shape} for {len(self.doc_ids)} docs"
        )
        return True

    def _save_query_cache(self, key: str, query_embeddings: np.ndarray, query_fde: np.ndarray):
        emb_path, fde_path = self._query_paths(key)
        np.save(emb_path, query_embeddings)
        np.save(fde_path, query_fde)

    def _load_query_cache(self, key: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        emb_path, fde_path = self._query_paths(key)
        emb = np.load(emb_path) if os.path.exists(emb_path) else None
        fde = np.load(fde_path) if os.path.exists(fde_path) else None
        return emb, fde

    # --------- Chamfer(MaxSim) ---------
    @staticmethod
    def _chamfer(query_tok: np.ndarray, doc_tok: np.ndarray) -> float:
        sim = query_tok @ doc_tok.T  # [m, n]
        return float(sim.max(axis=1).sum())

    def _get_doc_embeddings(self, doc_id: str, allow_build: bool = True) -> np.ndarray:
        """재랭킹 시 문서 임베딩 로드: 외부 디렉터리 → 내부 캐시 → 필요시 on-the-fly 인코딩"""
        # 1) 외부 디렉터리 우선
        ext_path = self._external_doc_emb_path(doc_id)
        if ext_path and os.path.exists(ext_path):
            #logging.info(f"[doc-embed] external load: id={doc_id} path={ext_path}")
            return np.load(ext_path)

        # 2) 내부 캐시
        int_path = self._doc_emb_path(doc_id)
        if os.path.exists(int_path):
            #logging.info(f"[doc-embed] internal load: id={doc_id} path={int_path}")
            return np.load(int_path)

        # 3) 필요 시 빌드
        if not allow_build:
            raise FileNotFoundError(ext_path or int_path)

        if self._corpus is None:
            raise RuntimeError("Corpus not set; cannot build document embeddings on the fly.")
        doc = {"id": doc_id, **self._corpus[doc_id]}
        emap = self.ranker.encode_documents(documents=[doc])
        arr = to_numpy(emap[doc_id])

        # 내부 캐시에 저장(선택)
        np.save(int_path, arr)
        #logging.info(f"[doc-embed] built & saved: id={doc_id} path={int_path}")
        return arr

    # --------- Latency log ---------
    def _log_latency(self, qid: str, search_s: float, rerank_s: float):
        try:
            with open(self._latency_log_path, "a", encoding="utf-8") as f:
                f.write(f"{qid}\t{search_s*1000:.3f}\t{rerank_s*1000:.3f}\n")
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write latency log: {e}")

    # --------- Public API ---------
    def index(self, corpus: dict):
        self._corpus = corpus

        # # (사용자 설정대로 캐시 로드 스킵 가능)
        # if self._load_cache():
        #     return

        # 문서 아이디 & 포지션 확정
        self.doc_ids = list(corpus.keys())
        self._doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        documents_for_ranker = [{"id": doc_id, **corpus[doc_id]} for doc_id in self.doc_ids]

        # ---------- 외부/내부 임베딩 로드 & 부족분만 인코딩 ----------
        doc_embeddings_map = {}
        missing_doc_ids: List[str] = []

        # 1) 외부/내부에서 가능한 만큼 채운다
        for doc_id in self.doc_ids:
            ext = self._external_doc_emb_path(doc_id)            
            if ext and os.path.exists(ext):
                doc_embeddings_map[doc_id] = np.load(ext).astype(np.float32)                
                # 필요 시 내부 캐시에도 채움
                if self.save_doc_embeds:
                    dst = self._doc_emb_path(doc_id)
                    if not os.path.exists(dst):
                        np.save(dst, doc_embeddings_map[doc_id])
                continue

            # 내부 캐시 확인
            dst = self._doc_emb_path(doc_id)
            if os.path.exists(dst): # shape(256, 128)
                print(f"[inner shape]: {np.load(dst).shape}")
                doc_embeddings_map[doc_id] = np.load(dst).astype(np.float32)
            else:
                missing_doc_ids.append(doc_id)

        logging.info(
            f"[index] preloaded from external/internal: {len(doc_embeddings_map)} / {len(self.doc_ids)}, "
            f"to-encode: {len(missing_doc_ids)}"
        )

        # 2) 외부/내부에 없는 문서만 배치 인코딩
        if missing_doc_ids:
            to_encode_docs = [{"id": did, **corpus[did]} for did in missing_doc_ids]
            logging.info(f"[index] encoding {len(to_encode_docs)} documents that are missing from precomputed files...")
            encoded_map = self.ranker.encode_documents(documents=to_encode_docs)
            for did in missing_doc_ids:
                arr = to_numpy(encoded_map[did])
                doc_embeddings_map[did] = arr
                if self.save_doc_embeds:
                    np.save(self._doc_emb_path(did), arr)

        # 3) 리스트로 정렬
        doc_embeddings_list = [doc_embeddings_map[doc_id] for doc_id in self.doc_ids]

        # ---------- FDE 인덱스 생성 ----------
        logging.info(f"[{self.__class__.__name__}] Generating FDEs from ColBERT embeddings in BATCH mode...")
        self.fde_index = generate_document_fde_batch(
            doc_embeddings_list,
            self.doc_config,
            memmap_path=os.path.join(self._cache_dir, "fde_index.mmap"),
            max_bytes_in_memory=2 * 1024**3,  # optional guard
            log_every=50000
        )# generate_document_fde_batch(doc_embeddings_list, self.doc_config)

        # 저장
        self._save_cache()

    def precompute_queries(self, queries: dict):
        missing = 0
        for qid, qtext in queries.items():
            key = self._query_key(qtext, str(qid))
            emb, fde = self._load_query_cache(key)
            if fde is not None and emb is not None:
                continue
            query_embeddings_map = self.ranker.encode_queries(queries=[qtext])
            query_embeddings = to_numpy(next(iter(query_embeddings_map.values())))
            query_config = replace(self.doc_config, fill_empty_partitions=False)
            query_fde = generate_query_fde(query_embeddings, query_config)
            self._save_query_cache(key, query_embeddings, query_fde)
            missing += 1
        logging.info(f"[{self.__class__.__name__}] Precomputed {missing} uncached queries.")

    def search(self, query: str, query_id: Optional[str] = None) -> dict:
        if self.fde_index is None or not self.doc_ids:
            if not self._load_cache():
                raise RuntimeError("FDE index is not built. Call index(corpus) first.")

        key = self._query_key(query, query_id)
        cached_emb, cached_fde = self._load_query_cache(key)

        # 1) FDE 검색 시간
        t0 = time.perf_counter()

        if cached_emb is None or cached_fde is None:
            query_embeddings_map = self.ranker.encode_queries(queries=[query])
            query_embeddings = to_numpy(next(iter(query_embeddings_map.values())))
            query_config = replace(self.doc_config, fill_empty_partitions=False)
            query_fde = generate_query_fde(query_embeddings, query_config)
            self._save_query_cache(key, query_embeddings, query_fde)
        else:
            query_embeddings = cached_emb
            query_fde = cached_fde

        start_fde_scores = time.perf_counter()
        fde_scores = self.fde_index @ query_fde
        end_fde_scores = time.perf_counter() - start_fde_scores
        start_argsort = time.perf_counter()
        order_fde = np.argsort(-fde_scores)
        end_argsort = time.perf_counter() - start_argsort
        search_time = time.perf_counter() - t0

        # 2) 재랭킹 시간
        rerank_time = 0.0
        if not self.enable_rerank or self.rerank_candidates <= 0:
            self._log_latency(str(query_id) if query_id is not None else "", search_time, rerank_time)
            logging.info(f"[search] QID={query_id} reranked=0 search_ms={search_time*1000:.3f} rerank_ms=0.000")
            return OrderedDict((self.doc_ids[i], float(fde_scores[i])) for i in order_fde)

        t1 = time.perf_counter()
        N = min(self.rerank_candidates, len(self.doc_ids))
        cand_idx = order_fde[:N]
        cand_ids = [self.doc_ids[i] for i in cand_idx]

        reranked = []
        for did in cand_ids:
            d_tok = self._get_doc_embeddings(did, allow_build=True)
            score = self._chamfer(query_embeddings, d_tok)
            reranked.append((did, score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        reranked_ids = {did for did, _ in reranked}
        tail = [(self.doc_ids[i], float(fde_scores[i])) for i in order_fde if self.doc_ids[i] not in reranked_ids]

        out = OrderedDict()
        for did, sc in reranked:
            out[did] = float(sc)
        for did, sc in tail:
            out[did] = sc

        rerank_time = time.perf_counter() - t1
        self._log_latency(str(query_id) if query_id is not None else "", search_time, rerank_time)
        logging.info(
            f"[search] QID={query_id} reranked={len(cand_ids)} "
            f"search_ms={search_time*1000:.3f} rerank_ms={rerank_time*1000:.3f}, scoring_ms={end_fde_scores*1000:.3f}, argsort_ms={end_argsort*1000:.3f}"
        )

        return out

# ======================
# --- Main Script ------
# ======================
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        pass

    # 데이터셋 로드
    corpus, queries, qrels = load_nanobeir_dataset(DATASET_REPO_ID)


    logging.info("Initializing retrieval models...")
    retrievers = {
        "2. ColBERT + FDE (+Chamfer rerank)": ColbertFdeRetriever(
            model_name=COLBERT_MODEL_NAME,
            rerank_candidates=100,
            enable_rerank=True,
            save_doc_embeds=True,
            latency_log_path=os.path.join(CACHE_ROOT, "latency.tsv"),  # QID\tSearch\tRerank
            external_doc_embeds_dir="/home/hyunji/muvera_optimized/cache_muvera/scidocs/doc_embeds",  # ★ 외부 임베딩 디렉터리 지정
        )
    }

    timings, final_results = {}, {}

    logging.info("--- PHASE 1: INDEXING ---")
    for name, retriever in retrievers.items():
        start_time = time.perf_counter()
        retriever.index(corpus)
        timings[name] = {"indexing_time": time.perf_counter() - start_time}
        logging.info(f"'{name}' indexing finished in {timings[name]['indexing_time']:.2f} seconds.")

    logging.info("--- PHASE 2: SEARCH & EVALUATION ---")
    for name, retriever in retrievers.items():
        logging.info(f"Running search for '{name}' on {len(queries)} queries...")

        if hasattr(retriever, "precompute_queries"):
            retriever.precompute_queries(queries)

        query_times = []
        results = {}
        for query_id, query_text in queries.items():
            start_time = time.perf_counter()
            results[str(query_id)] = retriever.search(query_text, query_id=str(query_id))
            query_times.append(time.perf_counter() - start_time)

        timings[name]["avg_query_time"] = np.mean(query_times)
        final_results[name] = results
        logging.info(f"'{name}' search finished. Avg query time: {timings[name]['avg_query_time'] * 1000:.2f} ms.")

    print("\n" + "=" * 85)
    print(f"{'FINAL REPORT':^85}")
    print(f"(Dataset: {DATASET_REPO_ID})")
    print("=" * 85)
    print(f"{'Retriever':<30} | {'Indexing Time (s)':<20} | {'Avg Query Time (ms)':<22} | {'Recall@{k}'.format(k=TOP_K):<10}")
    print("-" * 85)

    for name in retrievers.keys():
        #recall = evaluate_recall(final_results[name], qrels, k=TOP_K)
        idx_time = timings[name]["indexing_time"]
        query_time_ms = timings[name]["avg_query_time"] * 1000
        #print(f"{name:<30} | {idx_time:<20.2f} | {query_time_ms:<22.2f} | {recall:<10.4f}")
        print(f"{name:<30} | {idx_time:<20.2f} | {query_time_ms:<22.2f}")

    print("=" * 85)