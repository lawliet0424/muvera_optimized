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
import os, json, time, hashlib, logging, pathlib, random, threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Optional, List, Tuple, Dict
from queue import Queue
from statistics import mean

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
DATASET_REPO_ID = "quora"
COLBERT_MODEL_NAME = "raphaelsty/neural-cherche-colbert"
TOP_K = 10

# ----- Rerank 워커 개수 -----
RERANK_WORKERS = 16  # 코어/메모리/I/O 상황에 맞춰 조정

# ----- 실험 스케일 -----
TARGET_NUM_QUERIES = 1000
RANDOM_SEED = 42

# ----- ANN 배치 (고정 크기) -----
ANN_BATCH_SIZE = 8          # ← 100개 모이면 배치 검색
FAISS_NLIST = 1000
FAISS_NPROBE = 50
FAISS_CANDIDATES = 100        # over-fetch; rerank보다 크거나 같게 권장
FAISS_NUM_THREADS = 1         # OpenMP 스레드 수(권장: 1 또는 소수)

# ----- Rerank 배치(나이브, 고정 크기) -----
RERANK_BATCH_QUERIES = 2    # ← 100개 모이면 배치 시작(내부는 문서별 순차 Chamfer)
RERANK_TOPN = 100             # top-N만 재랭크 (나이브)

# ----- 캐시(기본 끔: 정말 나이브) -----
ENABLE_DOC_EMB_LRU_CACHE = False
DOC_EMB_LRU_SIZE = 0          # 0이면 캐시 안씀

# ----- FDE 설정(예: 1024 차원) -----
FDE_DIM = 1024
FDE_NUM_REPETITIONS = 2
FDE_NUM_SIMHASH = 7

# ----- 디바이스 -----
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# 데이터셋 경로
dataset = "quora"
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

# ======================
# --- Metric Setup ----
# ======================
avg_search_time_list = []
avg_ann_time_list = []
avg_rerank_time_list = []
avg_rerank_cp_list = []
avg_rerank_io_list = []

# ===========================
# --- Helper Functions  -----
# ===========================
def load_nanobeir_dataset(repo_id: str):
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
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().detach().numpy().astype(np.float32)
    elif isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    else:
        raise TypeError(f"Unsupported type for conversion: {type(tensor_or_array)}")

# =====================================
# --- FDE Query/Doc Generator Stubs  ---
# =====================================
from fde_generator_optimized_stream import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_document_fde_batch,
)

# =====================================
# --- Retriever (ANN 배치 + Rerank 나이브) ---
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
        self._fde_path = os.path.join(self._cache_dir, "fde_index.pkl")
        self._ids_path = os.path.join(self._cache_dir, "doc_ids.json")
        self._meta_path = os.path.join(self._cache_dir, "meta.json")
        self._queries_dir = os.path.join(self._cache_dir, "queries")
        self._doc_emb_dir = os.path.join(self._cache_dir, "doc_embeds")
        self._faiss_path = os.path.join(self._cache_dir, f"ivf{self.faiss_nlist}_ip.faiss")

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

    def _compute_cache_dir(self, dataset: str) -> str:
        return os.path.join(CACHE_ROOT, dataset)

    def _set_faiss_threads(self):
        if not self.use_faiss_ann:
            return
        try:
            faiss.omp_set_num_threads(self.faiss_num_threads)
            logging.info(f"[FAISS] omp_set_num_threads({self.faiss_num_threads})")
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
        # 내부 저장 시 pos 기반 파일 이름을 사용
        raise NotImplementedError  # 이 함수는 사용하지 않음 (외부/내부 경로에서 직접 처리)

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
    # retriever._log_latency(task.qid, total_search_time, task.ann_time_s, rerank_time, compute_rerank_time, io_rerank_time)
    def _log_latency(self, qid: str, search_s: float, ann_s: float, rerank_s: float, rerank_compute_s: float, rerank_io_s: float, rerank_sort_s: float):
        try:
            with self._log_lock:
                with open(self._latency_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{qid}\t{search_s*1000:.3f}\t{ann_s*1000:.3f}\t{rerank_s*1000:.3f}\t{rerank_compute_s*1000:.3f}\t{rerank_io_s*1000:.3f}\t{rerank_sort_s*1000:.3f}\n")
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write latency log: {e}")

    def _build_or_load_faiss_index(self):
        if not self.use_faiss_ann:
            return
        if self.faiss_index is not None and os.path.exists(self._faiss_path):
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
        logging.info(f"[FAISS] Saved to {self._faiss_path} (nprobe={self.faiss_nprobe})")

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
        t0 = time.perf_counter()
        D, I = self.faiss_index.search(XQ_batch, k)
        ann_time = time.perf_counter() - t0
        return D, I, ann_time

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

# ---- ANN Aggregator (배치: 100개 모이면 flush) ----
def ann_aggregator_loop(retriever: ColbertFdeRetrieverNaive,
                        in_q: Queue, out_q: Queue,
                        k: int,
                        batch_size: int = ANN_BATCH_SIZE):
    exp_dim = int(retriever.fde_index.shape[1])
    XQ_list: List[np.ndarray] = []
    metas: List[Tuple[str, str, np.ndarray, float]] = []

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
        if (qemb is None) or (qfde is None) or (qfde.shape[0] != exp_dim):
            qmap = retriever.ranker.encode_queries(queries=[qtext])
            qemb = to_numpy(next(iter(qmap.values())))
            qcfg = replace(retriever.doc_config, fill_empty_partitions=False)
            qfde = generate_query_fde(qemb, qcfg)
            retriever._save_query_cache(key, qemb, qfde)

        XQ_list.append(np.ascontiguousarray(qfde.reshape(1, -1).astype(np.float32)))
        metas.append((qid, qtext, qemb, t_enq))

        if len(XQ_list) >= batch_size:
            flush()

def rerank_aggregator_loop(retriever: ColbertFdeRetrieverNaive,
                           in_q: Queue,
                           out_dict: Dict[str, OrderedDict],
                           batch_queries: int = RERANK_BATCH_QUERIES,  # 더 이상 사용되지 않음
                           top_k: int = TOP_K,
                           num_workers: int = RERANK_WORKERS):
    """
    - in_q에서 task를 받는 즉시 워커가 처리 (즉시 flush 의미 유지)
    - 워커 수(num_workers)만큼 병렬 처리
    - 상위 top_k만 토큰 로드/Chamfer 계산, 나머지는 ANN 순서 유지
    """
    results_lock = Lock()
    stop_token = "__STOP__"

    def process_one(task: RerankTask):
        # 배치 시작 대신, '해당 작업이 실제로 시작되는 시점'을 기준으로 wait 측정
        t_start = time.perf_counter()
        wait_s = t_start - task.enqueued_time_s

        # 상위 TOP_K만 재채점 (상한: retriever.rerank_candidates)
        N_compute = min(top_k, retriever.rerank_candidates, len(task.initial_candidates))
        compute_ids = [did for (did, _) in task.initial_candidates[:N_compute]]

        # 재채점
        t0 = time.perf_counter()
        compute_rerank_time = 0
        io_rerank_time = 0
        sort_rerank_time = 0

        reranked_pairs = []

        for did in compute_ids:
            t_io = time.perf_counter()
            d_tok = retriever._get_doc_embeddings(did, allow_build=True) # I/O
            io_rerank_time += time.perf_counter() - t_io
            t_compute = time.perf_counter()
            score = retriever._chamfer(task.query_embeddings, d_tok) # compute
            compute_rerank_time += time.perf_counter() - t_compute                    
            reranked_pairs.append((did, score))

        # 선택: BLAS 스레드 과다 생성 방지 (있으면 1로 한정)
        # if _TPCTL_OK:
        #     with threadpool_limits(limits=1):  # OpenBLAS/MKL을 워커별 1스레드로
        #         reranked_pairs = []
        #         for did in compute_ids:
        #             t_io = time.perf_counter()
        #             d_tok = retriever._get_doc_embeddings(did, allow_build=True) # I/O
        #             io_rerank_time += time.perf_counter() - t_io
        #             t_compute = time.perf_counter()
        #             score = retriever._chamfer(task.query_embeddings, d_tok) # compute
        #             compute_rerank_time += time.perf_counter() - t_compute                    
        #             reranked_pairs.append((did, score))
        # else:
        #     reranked_pairs = []
        #     for did in compute_ids:
        #         d_tok = retriever._get_doc_embeddings(did, allow_build=True)
        #         score = retriever._chamfer(task.query_embeddings, d_tok)
        #         reranked_pairs.append((did, score))

        t_sort = time.perf_counter()
        reranked_pairs.sort(key=lambda x: x[1], reverse=True)
        sort_rerank_time = time.perf_counter() - t_sort
        
        rerank_time = time.perf_counter() - t0
        # 나머지 후보는 ANN 순서 유지
        computed_set = {did for (did, _) in reranked_pairs}
        tail_pairs = [(did, sc) for (did, sc) in task.initial_candidates if did not in computed_set]

        # 출력(재채점 결과 + ANN 꼬리)
        out = OrderedDict()
        for did, sc in reranked_pairs:
            out[did] = float(sc)
        for did, sc in tail_pairs:
            out[did] = float(sc)

        # 기록 compute_rerank_time | io_rerank_time
        with results_lock:
            out_dict[task.qid] = out
        total_search_time = task.ann_time_s + rerank_time
        
        avg_search_time_list.append(total_search_time)
        avg_ann_time_list.append(task.ann_time_s)
        avg_rerank_time_list.append(rerank_time)
        avg_rerank_cp_list.append(compute_rerank_time)
        avg_rerank_io_list.append(io_rerank_time)
        
        retriever._log_latency(task.qid, total_search_time, task.ann_time_s, rerank_time, compute_rerank_time, io_rerank_time, sort_rerank_time)

    # ---- 워커 풀 구성 ----
    workers = []
    def worker_loop():
        while True:
            item = in_q.get()
            if item == stop_token:
                # 다른 워커들도 멈출 수 있도록 토큰을 다시 넣는다.
                in_q.put(stop_token)
                break
            process_one(item)

    # 워커 시작
    for _ in range(max(1, int(num_workers))):
        t = threading.Thread(target=worker_loop, daemon=True)
        t.start()
        workers.append(t)

    # 메인 스레드: 종료 대기
    for t in workers:
        t.join()

# ======================
# --- Main Script ------
# ======================
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        pass

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

    # 1,000개 샘플
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

    sample_queries = make_random_query_sample(queries, TARGET_NUM_QUERIES, seed=RANDOM_SEED)
    sample_qrels   = build_qrels_for_sample(qrels, sample_queries)
    retriever.precompute_queries(sample_queries)

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
    for qid, qtext in sample_queries.items():
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
    print(f"(Dataset: {DATASET_REPO_ID}, Queries: {len(sample_queries)} | "
          f"ANN_BATCH={ANN_BATCH_SIZE}, RERANK_BATCH_Q={RERANK_BATCH_QUERIES}, "
          f"TOPN={RERANK_TOPN}, TOTAL_TIME= {end_time:.2f})")
    print("=" * 105)
    # avg_rerank_cp_list.append(compute_rerank_time) avg_rerank_io_list.append(io_rerank_time)
    print(f"[Average] Search: {mean(avg_search_time_list)*1000:.2f}, ANN: {mean(avg_ann_time_list)*1000:.2f}, "
          f"Rerank: {mean(avg_rerank_time_list)*1000:.2f}, Rerank(CP): {mean(avg_rerank_cp_list)*1000:.2f}, Rerank(IO): {mean(avg_rerank_io_list)*1000:.2f}")
    print("=" * 105)

    # Recall@K
    recall = evaluate_recall(results, sample_qrels, k=TOP_K)
    print(f"Ready Time (s): {t_ready:.2f}")
    print(f"Recall@{TOP_K}: {recall:.4f}")