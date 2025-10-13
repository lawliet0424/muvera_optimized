# -*- coding: utf-8 -*-
"""
Naive-baseline pipeline (deadline 제거, 고정 배치 크기 100):
  - ANN: 100개 모이면 XQ를 묶어 faiss_index.search(XQ, k)
  - Rerank: 100개 모이면 배치 시작(내부는 문서별 순차 Chamfer; 대형 GEMM/세그먼트 리듀스 없음)

실험 파라미터:
  TARGET_NUM_QUERIES, RANDOM_SEED
  ANN_BATCH_SIZE = 100
  RERANK_BATCH_QUERIES = 100
  RERANK_TOPN (= rerank_candidates)
  OMP/BLAS thread caps (기본 1)
"""
import os, json, time, hashlib, logging, pathlib, random, threading, argparse, sys, resource
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Optional, List, Tuple, Dict
from queue import Queue
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

import nltk
import numpy as np
import torch
import joblib

import neural_cherche.models as neural_cherche_models
import neural_cherche.rank as neural_cherche_rank

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from threading import Lock
try:
    # 선택: BLAS 스레드 과다 생성 방지 (NumPy/OpenBLAS/MKL)
    from threadpoolctl import threadpool_limits
    # BLAS thread 비활성화
    _TPCTL = threadpool_limits(limits=1)  # 프로그램 시작 시
    _TPCTL_OK = True
except Exception:
    _TPCTL_OK = False

# ---- 환경 제한(권장: 내부 라이브러리 멀티스레딩 억제) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ========== Faiss (CPU) ==========
try:
    import faiss  # pip install faiss-cpu
    _FAISS_OK = True
except Exception as _e:
    _FAISS_OK = False
    faiss = None

# ======================
# --- Configuration ----
# ======================
DATASET_REPO_ID = "scidocs"
COLBERT_MODEL_NAME = "raphaelsty/neural-cherche-colbert"
TOP_K = 40

# ----- Rerank 워커 개수 -----
RERANK_WORKERS = 16  # 코어/메모리/I/O 상황에 맞춰 조정

# ----- 실험 스케일 -----
TARGET_NUM_QUERIES = 100
RANDOM_SEED = 42

# ----- ANN 배치 (고정 크기) -----
ANN_BATCH_SIZE = 4          # ← 100개 모이면 배치 검색 (지금은 16으로 운영)
FAISS_NLIST = 1000
FAISS_NPROBE = 50
FAISS_CANDIDATES = 50        # over-fetch; rerank보다 크거나 같게 권장
FAISS_NUM_THREADS = 1         # OpenMP 스레드 수(권장: 1 또는 소수)

# ----- Rerank 배치(나이브, 고정 크기) -----
RERANK_BATCH_QUERIES = 4      # ← 100개 모이면 배치 시작(현 코드에서는 즉시 처리였음)
RERANK_TOPN = 50              # top-N만 재랭크 (나이브)

# ====== Rerank Batch Mode Switches ======
# 'immediate' : 쿼리 도착 즉시 Rerank (워커 병렬, mega GEMM)
# 'batch'     : Rerank 작업을 BATCH_RERANK_SIZE 개 모아 한 번에 멀티스레드 나이브 rerank
BATCH_RERANK_MODE = "batch"   # "immediate" or "batch"
BATCH_RERANK_SIZE = 4        # 배치 모을 크기

# ----- 캐시(기본 끔: 정말 나이브) -----
ENABLE_DOC_EMB_LRU_CACHE = False
DOC_EMB_LRU_SIZE = 0          # 0이면 캐시 안씀

# ----- FDE 설정(예: 1024 차원) -----
FDE_DIM = 128
FDE_NUM_REPETITIONS = 2
FDE_NUM_SIMHASH = 3

# ----- 디바이스 -----
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# 데이터셋 경로
dataset = "scidocs"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

# 캐시 루트
CACHE_ROOT = os.path.join(pathlib.Path(__file__).parent.absolute(), "cache_muvera")
os.makedirs(CACHE_ROOT, exist_ok=True)

# ======================
# --- Logging Setup ----
# ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Using device: {DEVICE}  |  FAISS={'on' if _FAISS_OK else 'off'}")
logging.info(f"[RERANK MODE] {BATCH_RERANK_MODE} (batch_size={BATCH_RERANK_SIZE})")

# ======================
# --- Metric Setup ----
# ======================
avg_search_time_list = []
avg_ann_time_list = []
avg_rerank_time_list = []
avg_rerank_cp_list = []
avg_rerank_io_list = []
avg_rerank_wait_list = []
avg_dup_ratio_list = []

# ===========================
# --- Helper Functions  -----
# ===========================
def load_nanobeir_dataset(repo_id: str):
    logging.info(f"Loading dataset from local path (BEIR): '{repo_id}'...")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    target_queries = dict(islice(queries.items(), TARGET_NUM_QUERIES))
    logging.info(f"Dataset loaded: {len(corpus)} documents, {len(target_queries)} queries.")
    return corpus, target_queries, qrels

# === (NEW) Per-query Recall@K ===
def per_query_recall_at_k(results: dict, qrels: dict, k: int) -> float:    
    # recalls = {}
    recalls: Dict[str, float] = {}
    for qid, ranked_docs in results.items():
        rel = set(qrels.get(str(qid), {}).keys())
        if not rel:
            continue
        
        if isinstance(ranked_docs, OrderedDict):            
            topk_ids = list(islice(ranked_docs.keys(), k))
        else:
            topk_ids = [doc for doc, _ in sorted(
                ranked_docs.items(), key=lambda x: x[1], reverse=True)[:k]]
        
        # 3) recall 계산
        hit_rel = rel.intersection(topk_ids)
        recall = len(hit_rel) / len(rel)
        recalls[qid] = recall        
        
        try:
            with open(f"/home/dccbeta/muvera_optimized/cache_muvera/per_query_{TOP_K}.tsv", "a", encoding="utf-8") as f:                
                f.write(f"{qid}\t{recalls[qid]}\n")
        except Exception as e:
            logging.warning(f"Failed to write per-query row: {e}")
    return recalls

# === (NEW) Per-query Recall@K ===
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

def to_numpy(tensor_or_array) -> np.ndarray:
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().detach().numpy().astype(np.float32)
    elif isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    else:
        raise TypeError(f"Unsupported type for conversion: {type(tensor_or_array)}")

# 랜덤 I/O 계측 헬퍼
def _read_proc_io_bytes() -> Optional[int]:
    """
    /proc/self/io 에서 read_bytes를 읽어온다 (Linux 한정).
    실패/비지원 시 None.
    """
    try:
        with open("/proc/self/io", "r") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except Exception:
        return None
    return None

def _get_rusage_faults():
    """(minflt, majflt) 튜플 반환"""
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(ru.ru_minflt), int(ru.ru_majflt)
    except Exception:
        return None, None

# =====================================
# --- FDE Query/Doc Generator Stubs  ---
# =====================================
from fde_generator_optimized_stream import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_document_fde_batch,
)

# =====================================
# --- Retriever (ANN 배치 + Rerank) ---
# =====================================
class ColbertFdeRetrieverNaive:
    def __init__(
        self,
        model_name: str = COLBERT_MODEL_NAME,
        rerank_candidates: int = RERANK_TOPN,
        enable_rerank: bool = True,
        save_doc_embeds: bool = True,
        latency_log_path: Optional[str] = None,
        external_doc_embeds_dir: Optional[str] = None,
        # FAISS params
        use_faiss_ann: bool = True,
        faiss_nlist: int = FAISS_NLIST,
        faiss_nprobe: int = FAISS_NPROBE,
        faiss_candidates: int = FAISS_CANDIDATES,
        faiss_num_threads: int = FAISS_NUM_THREADS,
        # FDE config
        fde_dim: int = FDE_DIM,
        fde_reps: int = FDE_NUM_REPETITIONS,
        fde_simhash: int = FDE_NUM_SIMHASH,
    ):
        self.faiss_num_threads = max(1, int(faiss_num_threads))
        model = neural_cherche_models.ColBERT(model_name_or_path=model_name, device=DEVICE)
        self.ranker = neural_cherche_rank.ColBERT(key="id", on=["title", "text"], model=model)

        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=fde_dim,
            num_repetitions=fde_reps,
            num_simhash_projections=fde_simhash,
            seed=42,
            fill_empty_partitions=True,
        )

        self.fde_index: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        self._doc_pos = {}
        self._corpus = None

        self.enable_rerank = enable_rerank
        self.rerank_candidates = rerank_candidates
        self.save_doc_embeds = save_doc_embeds
        self.external_doc_embeds_dir = external_doc_embeds_dir

        # FAISS
        self.use_faiss_ann = use_faiss_ann and _FAISS_OK
        self.faiss_nlist = faiss_nlist
        self.faiss_nprobe = faiss_nprobe
        self.faiss_candidates = faiss_candidates
        self.faiss_index = None

        # cache paths
        self._model_name = model_name
        self._cache_dir = self._compute_cache_dir(dataset=DATASET_REPO_ID)

        # 경로는 main 이후에 결정되는 전역 이름에 의존 (기존 코드 유지)
        self._fde_path = os.path.join(self._cache_dir, in_default)
        self._ids_path = os.path.join(self._cache_dir, "doc_ids.json")
        self._meta_path = os.path.join(self._cache_dir, meta_default)
        self._queries_dir = os.path.join(self._cache_dir, "queries")
        self._doc_emb_dir = os.path.join(self._cache_dir, "doc_embeds")
        self._faiss_path = os.path.join(self._cache_dir, faiss_default)

        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._queries_dir, exist_ok=True)
        if self.save_doc_embeds:
            os.makedirs(self._doc_emb_dir, exist_ok=True)

        self._latency_log_path = latency_log_path or os.path.join(self._cache_dir, "latency.tsv")
        self._log_lock = threading.Lock()

        # (선택) 작은 LRU 캐시 – 기본 비활성
        self._lru_enabled = ENABLE_DOC_EMB_LRU_CACHE and DOC_EMB_LRU_SIZE > 0
        if self._lru_enabled:
            from collections import OrderedDict as _OD
            self._lru = _OD()
            self._lru_lock = threading.Lock()
            self._lru_cap = DOC_EMB_LRU_SIZE
        
        # 헤더 기록 (이미 존재하면 이어쓰기)
        try:
            with self._log_lock:
                if not os.path.exists(self._latency_log_path):
                    with open(self._latency_log_path, "a", encoding="utf-8") as f:
                        f.write(
                            "qid\tann_ms\trerank_ms\trerank_compute_ms\trerank_io_ms\t"
                            "wait_ms\tdup_ratio\tread_bytes\tminflt\tmajflt\t"
                            "docloads\tuniqdocs\tmean_pos_delta\tp95_pos_delta\n"
                        )
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write latency header: {e}")
        
        # 헤더 기록 (이미 존재하면 이어쓰기), query 별 로깅 파일 생성
        self._per_query_log_path = os.path.join(CACHE_ROOT, f"per_query_{TOP_K}.tsv") # os.path.join(self._cache_dir, "latency.tsv")
        try:
            with self._log_lock:                
                if not os.path.exists(self._per_query_log_path):                    
                    with open(self._per_query_log_path, "a", encoding="utf-8") as f:                        
                        f.write("qid\trecall_at_k\n")
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write per-query header: {e}")        

    def _compute_cache_dir(self, dataset: str) -> str:
        return os.path.join(CACHE_ROOT, dataset)

    def _set_faiss_threads(self):
        if not self.use_faiss_ann:
            return
        try:
            faiss.omp_set_num_threads(self.faiss_num_threads)
        except Exception as e:
            logging.warning(f"[FAISS] omp_set_num_threads failed: {e}")

    def _query_key(self, query_text: str, query_id: Optional[str]) -> str:
        base = (query_id or "") + "||" + query_text
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _query_paths(self, key: str) -> Tuple[str, str]:
        return (
            os.path.join(self._queries_dir, f"{key}.emb.npy"),
            os.path.join(self._queries_dir, f"{key}.fde.npy"),
        )

    def _doc_emb_path(self) -> str:
        raise NotImplementedError  # 내부 pos 기반 파일 이름 사용

    def _external_doc_emb_path(self, doc_id: str) -> Optional[str]:
        if not self.external_doc_embeds_dir:
            return None
        pos = self._doc_pos.get(doc_id)
        if pos is None:
            return None
        return os.path.join(self.external_doc_embeds_dir, f"{pos:08d}.npy")

    def _internal_doc_emb_path(self, doc_id: str) -> str:
        pos = self._doc_pos[doc_id]
        return os.path.join(self._doc_emb_dir, f"{pos:08d}.npy")

    def _load_cache(self) -> bool:
        self.fde_index = joblib.load(self._fde_path)
        with open(self._ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        self._doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        logging.info(f"[{self.__class__.__name__}] Loaded FDE index cache: "
                     f"{self.fde_index.shape} for {len(self.doc_ids)} docs")

        # meta로 실제 FDE 설정 동기화
        try:
            with open(self._meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cfg = meta.get("config", {})
            self.doc_config = FixedDimensionalEncodingConfig(
                dimension=cfg.get("dimension", self.doc_config.dimension),
                num_repetitions=cfg.get("num_repetitions", self.doc_config.num_repetitions),
                num_simhash_projections=cfg.get("num_simhash_projections", self.doc_config.num_simhash_projections),
                seed=cfg.get("seed", self.doc_config.seed),
                fill_empty_partitions=cfg.get("fill_empty_partitions", self.doc_config.fill_empty_partitions),
            )
            logging.info(f"[FDE] Synchronized doc_config to cache meta: {self.doc_config}")
        except Exception as e:
            logging.warning(f"[FDE] Could not read meta.json; keep current doc_config. ({e})")

        # FAISS 로드 & 차원 검증
        if self.use_faiss_ann and os.path.exists(self._faiss_path):
            try:
                self.faiss_index = faiss.read_index(self._faiss_path)
                self.faiss_index.nprobe = self.faiss_nprobe
                if hasattr(self.faiss_index, "d") and self.faiss_index.d != int(self.fde_index.shape[1]):
                    logging.warning(f"[FAISS] dim mismatch: index.d={self.faiss_index.d} vs FDE={self.fde_index.shape[1]} ⇒ rebuild")
                    self.faiss_index = None
            except Exception as e:
                logging.warning(f"[FAISS] load failed: {e}")
                self.faiss_index = None
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

    @staticmethod
    def _chamfer(query_tok: np.ndarray, doc_tok: np.ndarray) -> float:
        # [m, d] @ [d, n] = [m, n] → row-wise max → sum
        sim = query_tok @ doc_tok.T
        return float(sim.max(axis=1).sum())

    def _get_doc_embeddings(self, doc_id: str, allow_build: bool = True) -> np.ndarray:
        # (나이브) 캐시 미사용 시 그냥 파일 로드
        if self._lru_enabled:
            with self._lru_lock:
                if doc_id in self._lru:
                    arr = self._lru.pop(doc_id)
                    self._lru[doc_id] = arr
                    return arr

        ext_path = self._external_doc_emb_path(doc_id)
        if ext_path and os.path.exists(ext_path):
            arr = np.load(ext_path)
        else:
            int_path = self._internal_doc_emb_path(doc_id)
            if os.path.exists(int_path):
                arr = np.load(int_path)
            else:
                if not allow_build:
                    raise FileNotFoundError(ext_path or int_path)
                if self._corpus is None:
                    raise RuntimeError("Corpus not set.")
                doc = {"id": doc_id, **self._corpus[doc_id]}
                emap = self.ranker.encode_documents(documents=[doc])
                arr = to_numpy(emap[doc_id])
                np.save(int_path, arr)

        if self._lru_enabled:
            with self._lru_lock:
                self._lru[doc_id] = arr
                if len(self._lru) > self._lru_cap:
                    self._lru.popitem(last=False)
        return arr

    # per_query logging
    def _log_per_query(self, qid: str, recall_at_k: int):
        try:
            with self._log_lock:
                with open(self._per_query_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{qid}\t{recall_at_k}\n")
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write per-query row: {e}")
    
    def _log_latency(
        self,
        qid: str,
        search_s: float,
        ann_s: float,
        rerank_s: float,
        rerank_compute_s: float,
        rerank_io_s: float,
        wait_s: float,
        dup_ratio: Optional[float] = None,
        # ── 랜덤 I/O 측정 추가 ──
        read_bytes: Optional[int] = None,
        minflt_delta: Optional[int] = None,
        majflt_delta: Optional[int] = None,
        docloads: Optional[int] = None,
        uniqdocs: Optional[int] = None,
        mean_pos_delta: Optional[float] = None,
        p95_pos_delta: Optional[float] = None,
    ):
        try:
            divided_ann_s = ann_s / ANN_BATCH_SIZE
            dr = -1.0 if (dup_ratio is None) else float(dup_ratio)
            rb = -1 if (read_bytes is None) else int(read_bytes)
            mf = -1 if (minflt_delta is None) else int(minflt_delta)
            M  = -1 if (majflt_delta is None) else int(majflt_delta)
            dl = -1 if (docloads is None) else int(docloads)
            ud = -1 if (uniqdocs is None) else int(uniqdocs)
            mp = -1.0 if (mean_pos_delta is None) else float(mean_pos_delta)
            p95= -1.0 if (p95_pos_delta is None) else float(p95_pos_delta)
            with self._log_lock:
                with open(self._latency_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{qid}\t{divided_ann_s*1000:.3f}\t{rerank_s*1000:.3f}\t"
                        f"{rerank_compute_s*1000:.3f}\t{rerank_io_s*1000:.3f}\t{wait_s*1000:.3f}\t"
                        f"{dr:.6f}\t{rb}\t{mf}\t{M}\t{dl}\t{ud}\t{mp:.6f}\t{p95:.6f}\n"
                    )
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write latency log: {e}")

    def _build_or_load_faiss_index(self):
        if not self.use_faiss_ann:
            return
        if self.faiss_index is not None and os.path.exists(self._fde_path):
            return
        self._set_faiss_threads()
        dim   = int(self.fde_index.shape[1])
        nvecs = int(self.fde_index.shape[0])

        logging.info(f"[FAISS] Building IVFFlat(IP) nlist={self.faiss_nlist} for {nvecs} vectors (dim={dim})")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, self.faiss_nlist, faiss.METRIC_INNER_PRODUCT)

        # Train
        try:
            x_train = self.fde_index
            if x_train.dtype != np.float32 or not x_train.flags["C_CONTIGUOUS"]:
                raise RuntimeError("Training would copy; fallback.")
            index.train(x_train)
        except Exception as e:
            logging.warning(f"[FAISS] train all failed: {e}; sampling.")
            min_per_list = 100
            target_train = max(self.faiss_nlist * min_per_list, 50_000)
            train_size = min(nvecs, target_train)
            rng = np.random.default_rng(42)
            train_x = np.empty((train_size, dim), dtype=np.float32, order="C")
            filled = 0
            block = 65536
            for start in range(0, nvecs, block):
                end = min(start + block, nvecs)
                Xblk = self.fde_index[start:end]
                need = train_size - filled
                if need <= 0:
                    break
                p = need / (nvecs - start)
                sel = np.nonzero(rng.random(end - start) < p)[0]
                k = min(need, sel.size)
                if k > 0:
                    train_x[filled:filled + k] = Xblk[sel[:k]]
                    filled += k
            if filled < train_size:
                need = train_size - filled
                step = max(1, nvecs // need)
                pos = 0
                for i in range(need):
                    train_x[filled + i] = self.fde_index[pos]
                    pos = (pos + step) % nvecs
            index.train(train_x)

        # Add
        add_bs = 100_000
        for start in range(0, nvecs, add_bs):
            end = min(start + add_bs, nvecs)
            xb = np.ascontiguousarray(self.fde_index[start:end], dtype=np.float32)
            index.add(xb)

        faiss.write_index(index, self._faiss_path)
        index.nprobe = self.faiss_nprobe
        self.faiss_index = index

    # --------- Public API ---------
    def index(self, corpus: dict):
        self._corpus = corpus
        if self._load_cache():
            if self.use_faiss_ann and self.faiss_index is None:
                try:
                    self._build_or_load_faiss_index()
                except Exception as e:
                    logging.warning(f"[FAISS] Build/load skipped: {e}")
            return
        raise RuntimeError("fde_index cache missing; create fde_index.pkl first.")

    def precompute_queries(self, queries: dict):
        missing = 0
        exp_dim = int(self.fde_index.shape[1])
        for qid, qtext in queries.items():
            key = self._query_key(qtext, str(qid))
            emb, fde = self._load_query_cache(key)
            need = (emb is None or fde is None or (fde is not None and fde.shape[0] != exp_dim))
            if need:
                if fde is not None and fde.shape[0] != exp_dim:
                    logging.warning(f"[QueryCache] dim mismatch qid={qid}: {fde.shape[0]} vs {exp_dim}. Rebuild.")
                qmap = self.ranker.encode_queries(queries=[qtext])
                qemb = to_numpy(next(iter(qmap.values())))
                qcfg = replace(self.doc_config, fill_empty_partitions=False)
                qfde = generate_query_fde(qemb, qcfg)
                self._save_query_cache(key, qemb, qfde)
                missing += 1
        logging.info(f"[{self.__class__.__name__}] Precomputed {missing} queries.")

    # ----- 배치 ANN (XQ 한 번에) -----
    def ann_search_batch(self, XQ_batch: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, float]:
        assert XQ_batch.ndim == 2
        if self.faiss_index is None:
            self._build_or_load_faiss_index()
        # 검색 직전에도 스레드 수 보장
        self._set_faiss_threads()
        t0 = time.perf_counter()
        D, I = self.faiss_index.search(XQ_batch, k)
        ann_time = time.perf_counter() - t0
        return D, I, ann_time

# ============== 공통: 하나의 task를 대형 GEMM + 세그먼트-리듀스로 재랭크 ==============
def _rerank_task_with_mega_gemm(
    retriever: ColbertFdeRetrieverNaive,
    task: "RerankTask",
    top_k: int,
) -> Tuple[OrderedDict, float, float, float, float, dict]:
    """
    반환: (out_pairs, rerank_total_s, compute_s, io_s, sort_s, meta)
    meta: 랜덤 I/O 계측 { 'docloads','uniqdocs','read_bytes','minflt_delta','majflt_delta','mean_pos_delta','p95_pos_delta' }
    """
    q_emb = task.query_embeddings  # [m, d] float32
    N_compute = min(top_k, retriever.rerank_candidates, len(task.initial_candidates))
    compute_ids = [did for (did, _) in task.initial_candidates[:N_compute]]

    # ── 계측 시작 ──
    io_before = _read_proc_io_bytes()
    mf0, M0   = _get_rusage_faults()
    pos_seq: List[int] = []
    # ────────────────

    # ----- I/O: 문서 토큰 모두 로드 & 이어붙이기 -----
    t_io0 = time.perf_counter()
    doc_spans: List[Tuple[str, int, int]] = []
    blocks = []
    col_start = 0
    for did in compute_ids:
        # pos 기록(순차성 근사)
        try:
            pos_seq.append(int(retriever._doc_pos[did]))
        except Exception:
            pass
        d_tok = retriever._get_doc_embeddings(did, allow_build=True)  # np.ndarray [n_i, d]
        n_i = int(d_tok.shape[0])
        blocks.append(d_tok)
        doc_spans.append((did, col_start, col_start + n_i))  # [start, end) in D_all
        col_start += n_i
    D_all = None
    if blocks:
        # [sum(n_i), d], C-contiguous
        D_all = np.ascontiguousarray(np.vstack(blocks).astype(np.float32))
    io_s = time.perf_counter() - t_io0

    # ----- Compute: 한 번의 큰 GEMM -----
    t_c0 = time.perf_counter()
    reranked_pairs: List[Tuple[str, float]] = []

    if D_all is not None and D_all.size > 0:
        # S: [m, sum(n_i)]
        S = q_emb @ D_all.T
        # 문서별 세그먼트 리듀스 (row-wise max → sum)
        for did, s, e in doc_spans:
            if e - s == 0:
                score = -1e9  # empty defensive
            else:
                score = float(S[:, s:e].max(axis=1).sum())
            reranked_pairs.append((did, score))
    compute_s = time.perf_counter() - t_c0

    # ----- Sort + Tail(미재채점 후보) 유지 -----
    t_sort0 = time.perf_counter()
    reranked_pairs.sort(key=lambda x: x[1], reverse=True)
    computed_set = {did for (did, _) in reranked_pairs}
    tail_pairs = [(did, sc) for (did, sc) in task.initial_candidates if did not in computed_set]
    out = OrderedDict()
    for did, sc in reranked_pairs:
        out[did] = float(sc)
    for did, sc in tail_pairs:
        out[did] = float(sc)
    sort_s = time.perf_counter() - t_sort0

    total_s = io_s + compute_s + sort_s

    # ── 계측 종료/집계 ──
    io_after = _read_proc_io_bytes()
    mf1, M1  = _get_rusage_faults()
    read_bytes = (io_after - io_before) if (io_before is not None and io_after is not None) else None
    minflt_delta = (mf1 - mf0) if (mf0 is not None and mf1 is not None) else None
    majflt_delta = (M1 - M0) if (M0 is not None and M1 is not None) else None

    # pos-delta 통계
    mean_pos_delta = None
    p95_pos_delta = None
    if len(pos_seq) >= 2:
        deltas = [abs(pos_seq[i] - pos_seq[i-1]) for i in range(1, len(pos_seq))]
        deltas_sorted = sorted(deltas)
        mean_pos_delta = float(sum(deltas) / len(deltas))
        p95_pos_delta = float(deltas_sorted[int(0.95*(len(deltas_sorted)-1))])

    meta = dict(
        docloads=len(compute_ids),
        uniqdocs=len(set(compute_ids)),
        read_bytes=read_bytes,
        minflt_delta=minflt_delta,
        majflt_delta=majflt_delta,
        mean_pos_delta=mean_pos_delta,
        p95_pos_delta=p95_pos_delta,
    )
    return out, total_s, compute_s, io_s, sort_s, meta

# ============== 나이브 per-task rerank (멀티스레드 배치용) ==============
def _rerank_task_naive(
    retriever: ColbertFdeRetrieverNaive,
    task: "RerankTask",
    top_k: int,
) -> Tuple[OrderedDict, float, float, float, float, dict]:
    """
    mega_gemm을 쓰지 않는 나이브 per-task rerank:
      - 상위 N 문서 각각에 대해 d_tok 로드 → Q @ D_doc.T → row-wise max→sum (Chamfer)
      - 정렬 후 tail 이어붙임
    반환: (out_pairs, total_s, compute_s, io_s, sort_s, meta)
    meta: 랜덤 I/O 계측 딕셔너리
    """
    q_emb = task.query_embeddings  # [m, d]
    N_compute = min(top_k, retriever.rerank_candidates, len(task.initial_candidates))
    compute_ids = [did for (did, _) in task.initial_candidates[:N_compute]]

    # ── 계측 시작 ──
    io_before = _read_proc_io_bytes()
    mf0, M0   = _get_rusage_faults()
    pos_seq: List[int] = []
    # ────────────────

    io_s = 0.0
    compute_s = 0.0
    reranked_pairs: List[Tuple[str, float]] = []

    for did in compute_ids:
        # pos 추출(순차성 프록시)
        try:
            pos_seq.append(int(retriever._doc_pos[did]))
        except Exception:
            pass

        t_io = time.perf_counter()
        d_tok = retriever._get_doc_embeddings(did, allow_build=True)  # [n_i, d]
        io_s += time.perf_counter() - t_io

        t_cp = time.perf_counter()
        score = retriever._chamfer(q_emb, d_tok)  # Q @ D_doc.T → row-wise max → sum
        compute_s += time.perf_counter() - t_cp
        reranked_pairs.append((did, float(score)))

    t_sort0 = time.perf_counter()
    reranked_pairs.sort(key=lambda x: x[1], reverse=True)
    computed_set = {did for (did, _) in reranked_pairs}
    tail_pairs = [(did, sc) for (did, sc) in task.initial_candidates if did not in computed_set]

    out = OrderedDict()
    for did, sc in reranked_pairs:
        out[did] = sc
    for did, sc in tail_pairs:
        out[did] = float(sc)

    sort_s = time.perf_counter() - t_sort0
    total_s = io_s + compute_s + sort_s

    # ── 계측 종료/집계 ──
    io_after = _read_proc_io_bytes()
    mf1, M1  = _get_rusage_faults()
    read_bytes = (io_after - io_before) if (io_before is not None and io_after is not None) else None
    minflt_delta = (mf1 - mf0) if (mf0 is not None and mf1 is not None) else None
    majflt_delta = (M1 - M0) if (M0 is not None and M1 is not None) else None

    # pos-delta 통계
    mean_pos_delta = None
    p95_pos_delta = None
    if len(pos_seq) >= 2:
        deltas = [abs(pos_seq[i] - pos_seq[i-1]) for i in range(1, len(pos_seq))]
        deltas_sorted = sorted(deltas)
        mean_pos_delta = float(sum(deltas) / len(deltas))
        p95_pos_delta = float(deltas_sorted[int(0.95*(len(deltas_sorted)-1))])

    meta = dict(
        docloads=len(compute_ids),
        uniqdocs=len(set(compute_ids)),
        read_bytes=read_bytes,
        minflt_delta=minflt_delta,
        majflt_delta=majflt_delta,
        mean_pos_delta=mean_pos_delta,
        p95_pos_delta=p95_pos_delta,
    )
    return out, total_s, compute_s, io_s, sort_s, meta

# ============== 배치 오케스트레이션(나이브, 고정 크기) ==============
@dataclass
class AnnItem:
    qid: str
    qtext: str
    t_enqueue: float

@dataclass
class RerankTask:
    qid: str
    qtext: str
    query_embeddings: np.ndarray
    initial_candidates: List[Tuple[str, float]]
    search_time_s: float
    ann_time_s: float
    enqueued_time_s: float

def make_random_query_sample(orig_queries: Dict[str, str], n_target: int, seed: int = 42) -> Dict[str, str]:
    rnd = random.Random(seed)
    items = list(orig_queries.items())
    m = len(items)
    out: Dict[str, str] = {}
    for i in range(n_target):
        oid, otext = items[rnd.randrange(m)]
        out[f"{oid}__rep{i}"] = otext
    return out

def build_qrels_for_sample(orig_qrels: Dict[str, Dict[str, int]], sample_queries: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    new_qrels = {}
    for new_qid in sample_queries.keys():
        base = new_qid.split("__rep")[0] if "__rep" in new_qid else new_qid
        if base in orig_qrels:
            new_qrels[new_qid] = orig_qrels[base]
    return new_qrels

# ---- ANN Aggregator (배치: batch_size 모이면 flush) ----
def ann_aggregator_loop(retriever: ColbertFdeRetrieverNaive,
                        in_q: Queue, out_q: Queue,
                        k: int,
                        batch_size: int = ANN_BATCH_SIZE):
    exp_dim = int(retriever.fde_index.shape[1])
    XQ_list: List[np.ndarray] = []
    metas: List[Tuple[str, str, np.ndarray, float]] = [] # flush 전 qid, qtext 등

    def flush():
        if not XQ_list:
            return
        XQb = np.vstack(XQ_list)  # [B, d]
        D, I, ann_time = retriever.ann_search_batch(XQb, k)
        t_now = time.perf_counter()
        for i, (qid, qtext, qemb, t_enq) in enumerate(metas):
            mask = I[i] >= 0
            cand_ids = [retriever.doc_ids[idx] for idx in I[i][mask]]
            cand_scores = D[i][mask].tolist()
            initial_candidates = list(zip(cand_ids, cand_scores))
            task = RerankTask(
                qid=qid, qtext=qtext, query_embeddings=qemb,
                initial_candidates=initial_candidates,
                search_time_s=(t_now - t_enq),  # 큐에 들어온 시점부터 ANN 결과를 내기까지
                ann_time_s=ann_time, enqueued_time_s=t_now
            )
            out_q.put(task)
        XQ_list.clear(); metas.clear()

    while True:
        item = in_q.get()  # 블로킹
        if item == "__STOP__":
            # 마지막 남은 배치 flush 후 STOP 전파
            flush()
            out_q.put("__STOP__")
            break

        # AnnItem 처리
        qid, qtext, t_enq = item.qid, item.qtext, item.t_enqueue
        key = retriever._query_key(qtext, qid)
        qemb, qfde = retriever._load_query_cache(key)
        if (qemb is None) or (qfde is None) or (fde := qfde) is None or (fde.shape[0] != exp_dim):
            qmap = retriever.ranker.encode_queries(queries=[qtext])
            qemb = to_numpy(next(iter(qmap.values())))
            qcfg = replace(retriever.doc_config, fill_empty_partitions=False)
            qfde = generate_query_fde(qemb, qcfg)
            retriever._save_query_cache(key, qemb, qfde)

        XQ_list.append(np.ascontiguousarray(qfde.reshape(1, -1).astype(np.float32)))
        metas.append((qid, qtext, qemb, t_enq))

        if len(XQ_list) >= batch_size:
            flush()

# ---- Rerank Aggregator (즉시/배치 모드 모두 지원) ----
def rerank_aggregator_loop(retriever: ColbertFdeRetrieverNaive,
                           in_q: Queue,
                           out_dict: Dict[str, OrderedDict],
                           batch_queries: int = RERANK_BATCH_QUERIES,  # 호환용
                           top_k: int = TOP_K,
                           num_workers: int = RERANK_WORKERS):
    """
    - immediate 모드: in_q에서 task를 받는 즉시 워커가 _rerank_task_with_mega_gemm 수행
    - batch 모드    : BATCH_RERANK_SIZE 만큼 모아 멀티스레드 나이브 per-task rerank
    """
    results_lock = Lock()
    stop_token = "__STOP__"

    def _commit_result(
        task: RerankTask,
        out_pairs: OrderedDict,
        rerank_time: float,
        compute_rerank_time: float,
        io_rerank_time: float,
        wait_s: float,
        dup_ratio: Optional[float] = None,
        meta: Optional[dict] = None,
    ):
        with results_lock:
            out_dict[task.qid] = out_pairs
        total_search_time = task.ann_time_s + rerank_time
        avg_search_time_list.append(total_search_time)
        avg_ann_time_list.append(task.ann_time_s/ANN_BATCH_SIZE)
        avg_rerank_time_list.append(rerank_time)
        avg_rerank_cp_list.append(compute_rerank_time)
        avg_rerank_io_list.append(io_rerank_time)
        avg_rerank_wait_list.append(wait_s)
        if dup_ratio is not None:
            avg_dup_ratio_list.append(dup_ratio)

        # meta 풀기
        rb  = meta.get("read_bytes")      if meta else None
        mf  = meta.get("minflt_delta")    if meta else None
        M   = meta.get("majflt_delta")    if meta else None
        dl  = meta.get("docloads")        if meta else None
        ud  = meta.get("uniqdocs")        if meta else None
        mp  = meta.get("mean_pos_delta")  if meta else None
        p95 = meta.get("p95_pos_delta")   if meta else None

        retriever._log_latency(
            task.qid, total_search_time, task.ann_time_s,
            rerank_time, compute_rerank_time, io_rerank_time, wait_s,
            dup_ratio=dup_ratio,
            read_bytes=rb, minflt_delta=mf, majflt_delta=M,
            docloads=dl, uniqdocs=ud, mean_pos_delta=mp, p95_pos_delta=p95,
        )

    # -------- Immediate Mode: 도착 즉시 워커 처리 (mega GEMM) --------
    if BATCH_RERANK_MODE == "immediate":
        def process_one(task: RerankTask):
            t_start = time.perf_counter()
            wait_s = t_start - task.enqueued_time_s
            t0 = time.perf_counter()
            out_pairs, total_rerank_s, compute_s, io_s, sort_s, meta = _rerank_task_with_mega_gemm(
                retriever, task, top_k)
            rerank_time = time.perf_counter() - t0  # 전체 함수 감싼 시간(검증용)
            _commit_result(task, out_pairs, rerank_time, compute_s, io_s, wait_s, dup_ratio=None, meta=meta)

        workers = []
        def worker_loop():
            while True:
                item = in_q.get()
                if item == stop_token:
                    in_q.put(stop_token)  # 다른 워커 종료 유도
                    break
                process_one(item)

        for _ in range(max(1, int(num_workers))):
            t = threading.Thread(target=worker_loop, daemon=True)
            t.start()
            workers.append(t)
        for t in workers:
            t.join()
        return

    # -------- Batch Mode: 배치로 모아 멀티스레드 처리 (나이브 per-task) --------
    elif BATCH_RERANK_MODE == "batch":
        buffer: List[RerankTask] = []

        def _process_task(task: RerankTask, dup_ratio_for_batch: Optional[float]):
            # 각 task를 스레드에서 독립적으로 나이브 rerank
            t_start = time.perf_counter()
            wait_s = t_start - task.enqueued_time_s
            t0 = time.perf_counter()
            out_pairs, total_rerank_s, compute_s, io_s, sort_s, meta = _rerank_task_naive(
                retriever, task, top_k
            )
            rerank_time = time.perf_counter() - t0
            _commit_result(
                task, out_pairs, rerank_time, compute_s, io_s, wait_s,
                dup_ratio=dup_ratio_for_batch, meta=meta
            )

        def _flush_batch(buf: List[RerankTask]):
            if not buf:
                return
            # --- 배치 내 문서 중복율 계산 ---
            all_doc_ids: List[str] = []
            for task in buf:
                N_compute = min(top_k, retriever.rerank_candidates, len(task.initial_candidates))
                all_doc_ids.extend([did for (did, _) in task.initial_candidates[:N_compute]])
            total_docs = len(all_doc_ids)
            if total_docs > 0:
                unique_docs = len(set(all_doc_ids))
                dup_ratio_for_batch = 1.0 - (unique_docs / total_docs)
            else:
                dup_ratio_for_batch = 0.0

            # --- 멀티스레드 실행 ---
            parallelism = min(max(1, int(num_workers)), len(buf))
            with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="rerank-batch") as ex:
                futures = [ex.submit(_process_task, t, dup_ratio_for_batch) for t in buf]
                for f in as_completed(futures):
                    _ = f.result()  # 예외 전파

        while True:
            item = in_q.get()
            if item == stop_token:
                _flush_batch(buffer)
                break
            buffer.append(item)
            if len(buffer) >= max(1, int(BATCH_RERANK_SIZE)):
                _flush_batch(buffer)
                buffer.clear()
        return

    else:
        raise ValueError(f"Unknown BATCH_RERANK_MODE={BATCH_RERANK_MODE}")

# ======================
# --- Main Script ------
# ======================
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        pass

    # === [ADDED] CLI for parameterized FAISS IVF-IP build ===
    def _positive_int(x: str) -> int:
        v = int(x)
        if v <= 0:
            raise argparse.ArgumentTypeError("must be > 0")
        return v

    parser = argparse.ArgumentParser(description="Build FAISS IVF-Flat (IP) from fde_index_{P}_{R}.pkl")
    parser.add_argument("--num_simhash_projections", "--p", type=_positive_int, required=True,
                        help="Number of simhash projections (P)")
    parser.add_argument("--num_repetitions", "--r", type=_positive_int, required=True,
                        help="Number of repetitions (R)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Optional explicit input pickle (defaults to fde_index_{P}_{R}.pkl)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional explicit FAISS output path (defaults to ivf1000_ip_{P}_{R}.faiss)")
    parser.add_argument("--nlist", type=_positive_int, default=1000,
                        help="IVF list count (default: 1000)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite output if it exists")
    args, _ = parser.parse_known_args()

    P = args.num_simhash_projections
    R = args.num_repetitions

    in_default = f"fde_index_{P}_{R}.pkl"
    faiss_default = f"ivf{args.nlist}_ip_{P}_{R}.faiss"
    meta_default = f"meta_{P}_{R}.json"

    # 실수 방지 가드
    assert FAISS_CANDIDATES >= RERANK_TOPN, \
        f"FAISS_CANDIDATES({FAISS_CANDIDATES}) must be >= RERANK_TOPN({RERANK_TOPN})"
    
    corpus, queries, qrels = load_nanobeir_dataset(DATASET_REPO_ID)

    retriever = ColbertFdeRetrieverNaive(
        model_name=COLBERT_MODEL_NAME,
        rerank_candidates=RERANK_TOPN,
        enable_rerank=True,
        save_doc_embeds=True,
        latency_log_path=os.path.join(CACHE_ROOT, "latency.tsv"),
        external_doc_embeds_dir=None,  # 있으면 경로 지정
        use_faiss_ann=True,
        faiss_nlist=FAISS_NLIST,
        faiss_nprobe=FAISS_NPROBE,
        faiss_candidates=FAISS_CANDIDATES,
        faiss_num_threads=FAISS_NUM_THREADS,
        fde_dim=FDE_DIM,
        fde_reps=FDE_NUM_REPETITIONS,
        fde_simhash=FDE_NUM_SIMHASH,
    )

    # 인덱스/캐시 로드
    t_ready0 = time.perf_counter()
    retriever.index(corpus)
    t_ready = time.perf_counter() - t_ready0
    logging.info(f"Retriever ready in {t_ready:.2f}s")    

    retriever.precompute_queries(queries)

    # 파이프 큐
    ann_in_q: Queue = Queue(maxsize=4096)
    rerank_in_q: Queue = Queue(maxsize=4096)

    # 출력
    results: Dict[str, OrderedDict] = {}

    # 스레드: ANN Aggregator, Rerank Aggregator
    start_time = time.perf_counter()
    ann_thr = threading.Thread(target=ann_aggregator_loop, args=(retriever, ann_in_q, rerank_in_q,
                                                                 max(FAISS_CANDIDATES, RERANK_TOPN),
                                                                 ANN_BATCH_SIZE),
                               daemon=True)
    rr_thr = threading.Thread(target=rerank_aggregator_loop, args=(retriever, rerank_in_q, results,
                                                                   RERANK_BATCH_QUERIES),
                              daemon=True)
    ann_thr.start()
    rr_thr.start()

    # 프론트: 쿼리 주입
    q_start_times: Dict[str, float] = {}
    for qid, qtext in queries.items():
        q_start_times[qid] = time.perf_counter()
        ann_in_q.put(AnnItem(qid=qid, qtext=qtext, t_enqueue=q_start_times[qid]))

    # 종료 시그널
    ann_in_q.put("__STOP__")
    ann_thr.join()
    rerank_in_q.put("__STOP__")
    rr_thr.join()

    end_time = time.perf_counter() - start_time

    # 성능 리포트
    print("\n" + "=" * 105)
    print(f"{'FINAL REPORT':^105}")
    print(f"(Dataset: {DATASET_REPO_ID}, Queries: {len(queries)} | "
          f"ANN_BATCH={ANN_BATCH_SIZE}, RERANK_BATCH_Q={RERANK_BATCH_QUERIES}, "
          f"TOPN={RERANK_TOPN}, TOTAL_TIME= {end_time:.2f}, MODE={BATCH_RERANK_MODE})")
    print("=" * 105)
    total_search_s = mean(avg_ann_time_list) + mean(avg_rerank_time_list) if avg_ann_time_list and avg_rerank_time_list else 0.0
    print(f"[Average] Search: {total_search_s*1000:.2f} ms, ANN: {mean(avg_ann_time_list)*1000:.2f} ms, "
          f"Rerank(TT): {mean(avg_rerank_time_list)*1000:.2f} ms, Rerank(CP): {mean(avg_rerank_cp_list)*1000:.2f} ms, "
          f"Rerank(IO): {mean(avg_rerank_io_list)*1000:.2f} ms, Rerank(WT): {mean(avg_rerank_wait_list)*1000:.2f} ms")
    if avg_dup_ratio_list:
        print(f"[Batch] Avg dup_ratio: {mean(avg_dup_ratio_list):.6f}")
    print("=" * 105)

    per_q_recall = per_query_recall_at_k(results, qrels, TOP_K)
    macro_recall = sum(per_q_recall.values()) / len(per_q_recall)

    # Recall@K
    recall = evaluate_hit_k(results, qrels, k=TOP_K)
    print(f"Ready Time (s): {t_ready:.2f}")
    print(f"Hit@{TOP_K}: {recall:.4f}")
    print(f"Recall@{TOP_K}: {macro_recall:.4f}")