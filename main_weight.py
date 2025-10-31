# -*- coding: utf-8 -*-
import os, json, time, hashlib, logging, pathlib, math
from collections import OrderedDict
from dataclasses import replace
from typing import Optional, List, Tuple, Dict
from itertools import islice

import nltk
import numpy as np
import torch
import joblib
import time
import psutil
import gc

import neural_cherche.models as neural_cherche_models
import neural_cherche.rank as neural_cherche_rank
from datasets import load_dataset

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import argparse

# FDE 구현 (업로드된 파일 사용)
from fde_generator_optimized_stream_simhash_check import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_document_fde_batch,
)

# ======================
# --- Configuration ----
# ======================
DATASET_REPO_ID = "trec-covid"
COLBERT_MODEL_NAME = "raphaelsty/neural-cherche-colbert"
TOP_K = 10
FILENAME = "main_weight"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# 캐시 루트
CACHE_ROOT = os.path.join("/media/hyunji", "muvera_optimized", "cache_muvera", DATASET_REPO_ID, FILENAME)
os.makedirs(CACHE_ROOT, exist_ok=True)

# 쿼리 검색 디렉터리
dataset = "trec-covid"
QUERY_SEARCH_DIR = os.path.join(CACHE_ROOT, "query_search")
os.makedirs(QUERY_SEARCH_DIR, exist_ok=True)

# 공통 문서 임베딩 디렉터리 설정
COMMON_EMBEDS_DIR = os.path.join("/media/hyunji", "muvera_optimized", "cache_muvera", DATASET_REPO_ID)
COMMON_DOC_EMBEDS_DIR = os.path.join(COMMON_EMBEDS_DIR, "doc_embeds")
COMMON_QUERY_EMBEDS_DIR = os.path.join(COMMON_EMBEDS_DIR, "query_embeds")
os.makedirs(COMMON_DOC_EMBEDS_DIR, exist_ok=True)
os.makedirs(COMMON_QUERY_EMBEDS_DIR, exist_ok=True)

# ======================
# --- Logging Setup ----
# ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Using device: {DEVICE}")

# ===========================
# --- Helper Functions  -----
# ===========================

# 메모리 사용량 확인 함수
def log_memory_usage(stage: str):
    """현재 메모리 사용량을 로깅"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024  # MB 단위
    memory_gb = memory_mb / 1024  # GB 단위
    logging.info(f"[MEMORY] {stage}: {memory_mb:.1f} MB ({memory_gb:.2f} GB)")
    return memory_mb
def load_nanobeir_dataset(repo_id: str):
    """Loads BEIR dataset from local 'data_path' in test split."""
    # 데이터셋 준비 (BEIR trec-covid)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join("/media/hyunji/7672b947-0099-4e49-8e90-525a208d54b8", "muvera_optimized", "datasets")

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

# === (NEW) Per-query Recall@K ===
def per_query_recall_at_k(results: dict, qrels: dict, k: int) -> Dict[str, float]:
    recalls: Dict[str, float] = {}
    for qid, ranked_docs in results.items():
        qid_str = str(qid)
        rel = set(qrels.get(qid_str, {}).keys())
        if not rel:
            continue

        if isinstance(ranked_docs, OrderedDict):
            topk_ids = set(islice(ranked_docs.keys(), k))
        else:
            topk_ids = set(
                doc for doc, _ in sorted(
                    ranked_docs.items(), key=lambda x: x[1], reverse=True
                )[:k]
            )

        hit_rel = rel.intersection(topk_ids)
        recall = len(hit_rel) / len(rel)
        recalls[qid_str] = recall
    return recalls

def evaluate_hit_k(results: dict, qrels: dict, k: int) -> float:
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

# === (nDCG with qrels) ===
def _dcg_at_k(ranked_docids: List[str], qrels_q: Dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked_docids[:k], start=1):
        rel = int(qrels_q.get(str(did), 0))
        if rel <= 0:
            continue
        gain = (2 ** rel) - 1
        dcg += gain / math.log2(i + 1)
    return dcg

def per_query_ndcg_at_k(results: Dict[str, OrderedDict], qrels: Dict[str, Dict[str, int]], k: int) -> Dict[str, float]:
    ndcgs: Dict[str, float] = {}
    for qid, ranked_docs in results.items():
        qrels_q = qrels.get(str(qid), {})
        if not any(int(v) > 0 for v in qrels_q.values()):
            continue

        if isinstance(ranked_docs, OrderedDict):
            topk_ids = list(islice(ranked_docs.keys(), k))
        else:
            topk_ids = [doc for doc, _ in sorted(ranked_docs.items(), key=lambda x: x[1], reverse=True)[:k]]

        dcg = _dcg_at_k(topk_ids, qrels_q, k)
        ideal_gains = sorted([int(v) for v in qrels_q.values() if int(v) > 0], reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_gains[:k], start=1):
            gain = (2 ** rel) - 1
            idcg += gain / math.log2(i + 1)
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcgs[qid] = ndcg
    return ndcgs

def evaluate_ndcg_at_k(results: dict, qrels: dict, k: int) -> float:
    """Calculate average nDCG@K across all queries."""
    per_query_ndcgs = per_query_ndcg_at_k(results, qrels, k)
    if not per_query_ndcgs:
        return 0.0
    return sum(per_query_ndcgs.values()) / len(per_query_ndcgs)

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
        num_repetitions: int = 2,
        num_simhash_projections: int = 5,
    ):
        model = neural_cherche_models.ColBERT(model_name_or_path=model_name, device=DEVICE)
        self.ranker = neural_cherche_rank.ColBERT(key="id", on=["title", "text"], model=model)

        # 추가된 인자
        self.num_repetitions = num_repetitions
        self.num_simhash_projections = num_simhash_projections

        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=128,
            num_repetitions=self.num_repetitions,
            num_simhash_projections=self.num_simhash_projections,
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
        
        # 공통 문서 임베딩 디렉터리 설정
        self.common_doc_embeds_dir = COMMON_DOC_EMBEDS_DIR
        
        # 공통 쿼리 임베딩 디렉터리 설정
        self.common_query_embeds_dir = COMMON_QUERY_EMBEDS_DIR

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
        # 개별 하위 디렉터리에 doc_embeds 저장하지 않음 (공통 디렉터리 사용)

        # 지연시간 로그 파일 (헤더 없이 누적)
        self._latency_log_path = latency_log_path or os.path.join(self._cache_dir, "latency.tsv")

    # --------- 경로/키 유틸 ---------
    def _compute_cache_dir(self, dataset: str, model_name: str, cfg) -> str:
        model_key = model_name.replace("/", "_")
        cfg_str = f"d{cfg.dimension}_r{cfg.num_repetitions}_p{cfg.num_simhash_projections}_seed{cfg.seed}_fill{int(cfg.fill_empty_partitions)}"
        raw = f"{dataset}|{model_key}|{cfg_str}"
        key = hashlib.md5(raw.encode()).hexdigest()[:10]
        dir_name = f"{dataset.replace('/', '_')}__{model_key}__{cfg_str}__{key}"
        return os.path.join(CACHE_ROOT, dir_name)

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

    def _common_doc_emb_path(self, doc_id: str) -> Optional[str]:
        """공통 디렉터리에서 기대하는 파일 경로(문서 순번 8자리 파일명)."""
        if not self.common_doc_embeds_dir:
            return None
        pos = self._doc_pos.get(doc_id)
        if pos is None:
            return None
        return os.path.join(self.common_doc_embeds_dir, f"{pos:08d}.npy")

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
        # 공통 디렉터리에 쿼리 임베딩 저장
        if hasattr(self, 'common_query_embeds_dir') and self.common_query_embeds_dir:
            # query_id 추출 (key에서)
            query_id = key.split('||')[0] if '||' in key else None
            if query_id and query_id.strip():  # 빈 문자열 체크 추가
                common_emb_path = os.path.join(self.common_query_embeds_dir, f"query_{query_id}.npy")
                if not os.path.exists(common_emb_path):
                    os.makedirs(os.path.dirname(common_emb_path), exist_ok=True)
                    np.save(common_emb_path, query_embeddings)
                    logging.info(f"[query-embed] saved to common directory: {common_emb_path}")
        
        # FDE만 개별 하위 디렉터리에 저장 (백업 제거)
        _, fde_path = self._query_paths(key)
        np.save(fde_path, query_fde)

    def _load_query_cache(self, key: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        # 공통 디렉터리에서 쿼리 임베딩 로드 시도
        if hasattr(self, 'common_query_embeds_dir') and self.common_query_embeds_dir:
            query_id = key.split('||')[0] if '||' in key else None
            if query_id and query_id.strip():  # 빈 문자열 체크 추가
                common_emb_path = os.path.join(self.common_query_embeds_dir, f"query_{query_id}.npy")
                if os.path.exists(common_emb_path):
                    emb = np.load(common_emb_path)
                    # FDE는 개별 하위 디렉터리에서 로드
                    _, fde_path = self._query_paths(key)
                    fde = np.load(fde_path) if os.path.exists(fde_path) else None
                    return emb, fde
        
        # 공통 디렉터리에 없으면 개별 하위 디렉터리에서 로드 (fallback)
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
        """재랭킹 시 문서 임베딩 로드: 공통 디렉터리 → 필요시 on-the-fly 인코딩"""
        # 1) 공통 디렉터리에서 로드
        ext_path = self._external_doc_emb_path(doc_id)
        if ext_path and os.path.exists(ext_path):
            #logging.info(f"[doc-embed] common load: id={doc_id} path={ext_path}")
            return np.load(ext_path)

        # 2) 필요 시 빌드 (개별 저장 없이)
        if not allow_build:
            raise FileNotFoundError(ext_path)

        if self._corpus is None:
            raise RuntimeError("Corpus not set; cannot build document embeddings on the fly.")
        doc = {"id": doc_id, **self._corpus[doc_id]}
        emap = self.ranker.encode_documents(documents=[doc])
        arr = to_numpy(emap[doc_id])

        # 공통 디렉터리에 저장
        if ext_path:
            os.makedirs(os.path.dirname(ext_path), exist_ok=True)
            np.save(ext_path, arr)
            #logging.info(f"[doc-embed] built & saved to common: id={doc_id} path={ext_path}")
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
                # 공통 디렉터리에서 로드했으므로 개별 저장 불필요
                continue

            # 내부 캐시 확인
            dst = self._doc_emb_path(doc_id)
            if os.path.exists(dst):
                try:
                    loaded_emb = np.load(dst)
                    # shape 검증: (256, 128) 또는 (128,) 형태여야 함
                    if loaded_emb.ndim == 2 and loaded_emb.shape[1] == 128:
                        doc_embeddings_map[doc_id] = loaded_emb.astype(np.float32)
                        print(f"[inner shape]: {loaded_emb.shape}")
                    else:
                        print(f"[inner shape invalid]: {loaded_emb.shape}, expected (256, 128) or (128,), will regenerate")
                        missing_doc_ids.append(doc_id)
                except Exception as e:
                    print(f"[inner load error]: {e}, will regenerate")
                    missing_doc_ids.append(doc_id)
            else:
                missing_doc_ids.append(doc_id)

        logging.info(
            f"[index] preloaded from external/internal: {len(doc_embeddings_map)} / {len(self.doc_ids)}, "
            f"to-encode: {len(missing_doc_ids)}"
        )

        # ---------- 배치 단위 처리: 인코딩 → FDE 생성 → 저장 ----------
        ATOMIC_BATCH_SIZE = 1000  # 배치 크기 (메모리 매핑으로 안전하게 처리)
        
        #[1017] simhash별 indice별 원소 개수 csv 파일 저장 필요------------------------------------
        simhash_count_dir = os.path.join(QUERY_SEARCH_DIR, f"rep{args.rep}_simhash{args.simhash}_rerank{args.rerank}")
        os.makedirs(simhash_count_dir, exist_ok=True)
        simhash_count_path = os.path.join(simhash_count_dir, "simhash_count.csv")
        with open(simhash_count_path, "w", encoding="utf-8") as f:
            f.write("doc_idx,rep_num,partition_idx,count\n")
        #------------------------------------------------------------------------
        
        # FDE 인덱스 초기화 (메모리 매핑으로)
        num_partitions = 2 ** self.doc_config.num_simhash_projections
        final_fde_dim_per_rep = num_partitions * (self.doc_config.projection_dimension or self.doc_config.dimension)
        final_fde_dim = self.doc_config.num_repetitions * final_fde_dim_per_rep
        
        # FDE 인덱스 memmap 생성
        fde_memmap_path = os.path.join(self._cache_dir, f"fde_index_memmap_{args.rep}_{args.simhash}.mmap")
        fde_index = np.memmap(fde_memmap_path, mode="w+", dtype=np.float32, 
                             shape=(len(self.doc_ids), final_fde_dim))
        
        log_memory_usage("Before atomic batch processing")
        
        logging.info(f"[{self.__class__.__name__}] Processing {len(self.doc_ids)} documents in atomic batches of {ATOMIC_BATCH_SIZE}...")
        
        for batch_start in range(0, len(self.doc_ids), ATOMIC_BATCH_SIZE):
            batch_end = min(batch_start + ATOMIC_BATCH_SIZE, len(self.doc_ids))
            batch_doc_ids = self.doc_ids[batch_start:batch_end]
            
            logging.info(f"[Atomic Batch] Processing batch {batch_start//ATOMIC_BATCH_SIZE + 1}/{(len(self.doc_ids) + ATOMIC_BATCH_SIZE - 1)//ATOMIC_BATCH_SIZE}: docs {batch_start}-{batch_end-1}")
            
            # Step 1: 배치용 임베딩 수집 (파일에서 직접 로드)
            batch_embeddings = []
            batch_missing_ids = []
            
            for doc_id in batch_doc_ids:
                # 외부 디렉터리에서 로드
                ext = self._external_doc_emb_path(doc_id)
                if ext and os.path.exists(ext):
                    batch_embeddings.append(np.load(ext).astype(np.float32))
                    continue
                
                # 내부 캐시에서 로드
                dst = self._doc_emb_path(doc_id)
                if os.path.exists(dst):
                    try:
                        loaded_emb = np.load(dst)
                        if loaded_emb.ndim == 2 and loaded_emb.shape[1] == 128:
                            batch_embeddings.append(loaded_emb.astype(np.float32))
                            continue
                        else:
                            print(f"[inner shape invalid]: {loaded_emb.shape}, will regenerate")
                    except Exception as e:
                        print(f"[inner load error]: {e}, will regenerate")
                
                batch_missing_ids.append(doc_id)
            
            # Step 2: 누락된 문서들 배치 인코딩
            if batch_missing_ids:
                logging.info(f"[Atomic Batch] Encoding {len(batch_missing_ids)} missing documents...")
                to_encode_docs = [{"id": did, **corpus[did]} for did in batch_missing_ids]
                encoded_map = self.ranker.encode_documents(documents=to_encode_docs)
                
                for did in batch_missing_ids:
                    arr = to_numpy(encoded_map[did])
                    batch_embeddings.append(arr)
                    
                    # 공통 디렉터리에 저장 (없을 때만)
                    common_path = self._common_doc_emb_path(did)
                    if common_path and not os.path.exists(common_path):
                        os.makedirs(os.path.dirname(common_path), exist_ok=True)
                        np.save(common_path, arr)
                        logging.info(f"[doc-embed] saved to common directory: {common_path}")
                    
                    # 공통 디렉터리에 저장했으므로 개별 저장 불필요
                    del encoded_map[did]
                    del arr
                
                del to_encode_docs
                del encoded_map
            
            # Step 3: 배치 FDE 생성
            logging.info(f"[Atomic Batch] Generating FDE for {len(batch_embeddings)} documents...")
            # 배치별 임시 memmap 파일 생성
            batch_memmap_path = os.path.join(self._cache_dir, f"batch_{batch_start//ATOMIC_BATCH_SIZE}.mmap")
            batch_fde_result = generate_document_fde_batch(
                batch_embeddings,
                self.doc_config,
                memmap_path=batch_memmap_path,  # 배치별 memmap 사용
                max_bytes_in_memory=512 * 1024**2,  # 512MB로 제한
                log_every=ATOMIC_BATCH_SIZE,
                flush_interval=ATOMIC_BATCH_SIZE,
            )
            
            if isinstance(batch_fde_result, tuple):
                batch_fde, partition_counter = batch_fde_result
            else:
                batch_fde = batch_fde_result
                partition_counter = None
            
            # Step 4: FDE 인덱스에 통합 저장 (메모리 매핑에 직접 저장)
            fde_index[batch_start:batch_end] = batch_fde
            logging.info(f"[FDE Integration] Integrated batch {batch_start//ATOMIC_BATCH_SIZE + 1} into final memmap")
            
            # Step 5: 배치별 flush (즉시 디스크 저장)
            fde_index.flush()
            
            # Step 6: Simhash 통계 저장
            if partition_counter is not None:
                for doc_idx in range(partition_counter.shape[0]):
                    global_doc_idx = batch_start + doc_idx
                    for rep_num in range(partition_counter.shape[1]):
                        for partition_idx in range(partition_counter.shape[2]):
                            count = partition_counter[doc_idx, rep_num, partition_idx]
                            with open(simhash_count_path, "a", encoding="utf-8") as f:
                                f.write(f"{global_doc_idx},{rep_num},{partition_idx},{count}\n")
            
            # Step 7: 배치 완료 후 메모리 해제
            del batch_embeddings
            if not (batch_memmap_path and os.path.exists(batch_memmap_path)):
                del batch_fde  # memmap이 아닌 경우만 삭제
            if partition_counter is not None:
                del partition_counter
            gc.collect()
            
            # Step 8: 임시 배치 memmap 파일 정리
            if batch_memmap_path and os.path.exists(batch_memmap_path):
                try:
                    os.remove(batch_memmap_path)
                    logging.info(f"[Atomic Batch] Cleaned up batch memmap: {batch_memmap_path}")
                except Exception as e:
                    logging.warning(f"[Atomic Batch] Failed to clean up {batch_memmap_path}: {e}")
            
            log_memory_usage(f"After atomic batch {batch_start//ATOMIC_BATCH_SIZE + 1}")
        
        # Step 8: 최종 통합 memmap 완성 및 저장
        fde_index.flush()
        logging.info(f"[FDE Integration] Final integrated memmap completed: {fde_memmap_path}")
        logging.info(f"[FDE Integration] Final shape: {fde_index.shape}")
        
        # 최종 통합 memmap을 인스턴스에 할당
        self.fde_index = fde_index
        
        # FDE 인덱스 참조 해제 (메모리 절약)
        del fde_index
        gc.collect()
        
        logging.info(f"[Atomic Batch] Completed processing {len(self.doc_ids)} documents")
        logging.info(f"[Atomic Batch] Integrated FDE index saved to: {fde_memmap_path}")
        log_memory_usage("After atomic batch processing")
        
        # 메모리 해제
        logging.info(f"[{self.__class__.__name__}] Memory cleanup completed")
        log_memory_usage("After memory cleanup")
        
        # 저장
        self._save_cache()
        log_memory_usage("Index completed")

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
            query_fde_result = generate_query_fde(query_embeddings, query_config, True)
            
            # query_fde_result가 튜플인 경우 첫 번째 요소만 사용
            if isinstance(query_fde_result, tuple):
                query_fde = query_fde_result[0]
            else:
                query_fde = query_fde_result

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
            query_fde_result = generate_query_fde(query_embeddings, query_config, True)
            
            # query_fde_result가 튜플인 경우 첫 번째 요소만 사용
            if isinstance(query_fde_result, tuple):
                query_fde = query_fde_result[0]
            else:
                query_fde = query_fde_result
            
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

    # args 받기
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep", type=int, default=2)
    parser.add_argument("--simhash", type=int, default=5)
    parser.add_argument("--rerank", type=int, default=100)
    args = parser.parse_args()

    nltk.download('punkt', quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        pass

    # 데이터셋 로드
    corpus, queries, qrels = load_nanobeir_dataset(DATASET_REPO_ID)
    
    # 쿼리를 첫 100개로 제한 (1:100)
    queries = dict(list(queries.items())[:100])
    logging.info(f"Limited queries to first 100: {len(queries)} queries.")


    logging.info("Initializing retrieval models...")

    retrievers = {
        "2. ColBERT + FDE (+Chamfer rerank)": ColbertFdeRetriever(
            model_name=COLBERT_MODEL_NAME,
            rerank_candidates=args.rerank,
            enable_rerank=True,
            save_doc_embeds=False,  # 공통 디렉터리에만 저장, 하위 디렉터리 중복 저장 방지
            latency_log_path=os.path.join(QUERY_SEARCH_DIR, f"rep{args.rep}_simhash{args.simhash}_rerank{args.rerank}", "latency.tsv"),  # QID\tSearch\tRerank
            external_doc_embeds_dir=COMMON_DOC_EMBEDS_DIR,  # ★ 공통 문서 임베딩 디렉터리
            num_repetitions=args.rep,
            num_simhash_projections=args.simhash,
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

        # 지연시간 로그 파일 초기화
        latency_dir = os.path.join(QUERY_SEARCH_DIR, f"rep{args.rep}_simhash{args.simhash}_rerank{args.rerank}")
        os.makedirs(latency_dir, exist_ok=True)
        with open(os.path.join(latency_dir, "latency.tsv"), "w", encoding="utf-8") as f:
            f.write("QID\tSearch\tRerank\n")
        
        # 결과 저장 파일 경로 설정
        results_file = os.path.join(latency_dir, "results.txt")

        for query_id, query_text in queries.items():
            start_time = time.perf_counter()
            results[str(query_id)] = retriever.search(query_text, query_id=str(query_id))
            query_times.append(time.perf_counter() - start_time)

        timings[name]["avg_query_time"] = np.mean(query_times)
        final_results[name] = results
        logging.info(f"'{name}' search finished. Avg query time: {timings[name]['avg_query_time'] * 1000:.2f} ms.")

    # 결과 출력 및 파일 저장
    report_lines = []
    report_lines.append("\n" + "=" * 85)
    report_lines.append(f"{'FINAL REPORT':^85}")
    report_lines.append(f"(Dataset: {DATASET_REPO_ID})")
    report_lines.append(f"Parameters: rep={args.rep}, simhash={args.simhash}, rerank={args.rerank}")
    report_lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 85)
    report_lines.append(f"{'Retriever':<30} | {'Indexing Time (s)':<20} | {'Avg Query Time (ms)':<22} | {'Recall@{k}'.format(k=TOP_K):<10} | {'Hit@{k}'.format(k=TOP_K):<10} | {'nDCG@{k}'.format(k=TOP_K):<10}")
    report_lines.append("-" * 120)

    for name in retrievers.keys():
        recall = evaluate_recall(final_results[name], qrels, k=TOP_K)
        hit_rate = evaluate_hit_k(final_results[name], qrels, k=TOP_K)
        ndcg = evaluate_ndcg_at_k(final_results[name], qrels, k=TOP_K)
        idx_time = timings[name]["indexing_time"]
        query_time_ms = timings[name]["avg_query_time"] * 1000
        line = f"{name:<30} | {idx_time:<20.2f} | {query_time_ms:<22.2f} | {recall:<10.4f} | {hit_rate:<10.4f} | {ndcg:<10.4f}"
        report_lines.append(line)

    report_lines.append("=" * 120)
    
    # 추가 통계 정보
    report_lines.append(f"\nAdditional Statistics:")
    report_lines.append(f"- Total queries processed: {len(queries)}")
    report_lines.append(f"- Total documents: {len(corpus)}")
    report_lines.append(f"- Device used: {DEVICE}")
    
    # Per-query metrics 상세 정보
    report_lines.append(f"\nPer-Query Metrics (for first retriever):")
    first_retriever_name = list(retrievers.keys())[0]
    per_query_recalls = per_query_recall_at_k(final_results[first_retriever_name], qrels, k=TOP_K)
    per_query_ndcgs = per_query_ndcg_at_k(final_results[first_retriever_name], qrels, k=TOP_K)
    
    if per_query_recalls:
        avg_recall = sum(per_query_recalls.values()) / len(per_query_recalls)
        min_recall = min(per_query_recalls.values())
        max_recall = max(per_query_recalls.values())
        report_lines.append(f"- Average Per Query Recall@{TOP_K}: {avg_recall:.4f}")
        report_lines.append(f"- Min Per Query Recall@{TOP_K}: {min_recall:.4f}")
        report_lines.append(f"- Max Per Query Recall@{TOP_K}: {max_recall:.4f}")
    
    if per_query_ndcgs:
        avg_ndcg = sum(per_query_ndcgs.values()) / len(per_query_ndcgs)
        min_ndcg = min(per_query_ndcgs.values())
        max_ndcg = max(per_query_ndcgs.values())
        report_lines.append(f"- Average nDCG@{TOP_K}: {avg_ndcg:.4f}")
        report_lines.append(f"- Min nDCG@{TOP_K}: {min_ndcg:.4f}")
        report_lines.append(f"- Max nDCG@{TOP_K}: {max_ndcg:.4f}")
    
    # Per-query metrics를 별도 파일에 저장
    per_query_file = os.path.join(latency_dir, f"per_query_metrics_{TOP_K}.tsv")
    try:
        with open(per_query_file, "w", encoding="utf-8") as f:
            f.write("qid\trecall_at_k\tndcg_at_k\n")
            for qid in per_query_recalls.keys():
                recall_val = per_query_recalls.get(qid, 0.0)
                ndcg_val = per_query_ndcgs.get(qid, 0.0)
                f.write(f"{qid}\t{recall_val:.6f}\t{ndcg_val:.6f}\n")
        logging.info(f"Per-query metrics saved to: {per_query_file}")
    except Exception as e:
        logging.warning(f"Failed to save per-query metrics: {e}")
    
    # 콘솔에 출력
    for line in report_lines:
        print(line)
    
    # 파일에 저장
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    logging.info(f"Results saved to: {results_file}")
