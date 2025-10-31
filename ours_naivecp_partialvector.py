# -*- coding: utf-8 -*-
"""
Naive-baseline pipeline (+ Pack build & Half-Doc from pack):
  - ANN: 배치 검색 (FAISS)
  - Rerank: 배치 모아 문서별 Chamfer (대형 GEMM/세그먼트 리듀스 없음)
  - PackBuilder/PackReader 통합
  - HALF_POLICY( front | back | stride2 )로 pack에서 문서 토큰의 절반만 읽기
  - (Two-Stage) ANN 상위 num_rank_candidates만 재랭크 + 스케치(idx 사이드카)로 해당 행만 읽기

실험 파라미터:
  TARGET_NUM_QUERIES, RANDOM_SEED
  ANN_BATCH_SIZE, RERANK_BATCH_QUERIES, RERANK_TOPN(=rerank_candidates)
"""

import os, json, time, hashlib, logging, pathlib, random, threading, argparse, sys, resource, csv, math, heapq
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
    from threadpoolctl import threadpool_limits
    _TPCTL = threadpool_limits(limits=1)
    _TPCTL_OK = True
except Exception:
    _TPCTL_OK = False

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ========== Faiss (CPU) ==========
try:
    import faiss
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False
    faiss = None

# ======================
# --- Configuration ----
# ======================
DATASET_REPO_ID = "treccovid"
COLBERT_MODEL_NAME = "raphaelsty/neural-cherche-colbert"
TOP_K = 10

dataset = "trec-covid"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

RERANK_WORKERS = 16
TARGET_NUM_QUERIES = 100
RANDOM_SEED = 42

ANN_BATCH_SIZE = 10
FAISS_NLIST = 1000
FAISS_NPROBE = 50
FAISS_CANDIDATES = 100
FAISS_NUM_THREADS = 1

RERANK_BATCH_QUERIES = 10
RERANK_TOPN = 0

BATCH_RERANK_MODE = "batch"   # "immediate" or "batch"
BATCH_RERANK_SIZE = 10

ENABLE_DOC_EMB_LRU_CACHE = False
DOC_EMB_LRU_SIZE = 0

FDE_DIM = 128
FDE_NUM_REPETITIONS = 2
FDE_NUM_SIMHASH = 3

# --------- Two-Stage 설정 ----------
ENABLE_TWO_STAGE = False         # --enable_two_stage 로 켬
STAGE1_SKETCH_TOKENS = 256       # 스케치 크기 (문서 내 선택 행 수)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

CACHE_ROOT = os.path.join(pathlib.Path(__file__).parent.absolute(), "cache_muvera")
os.makedirs(CACHE_ROOT, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Using device: {DEVICE}  |  FAISS={'on' if _FAISS_OK else 'off'}")
logging.info(f"[RERANK MODE] {BATCH_RERANK_MODE} (batch_size={BATCH_RERANK_SIZE})")

avg_search_time_list = []
avg_ann_time_list = []
avg_rerank_time_list = []
avg_rerank_cp_list = []
avg_rerank_io_list = []
avg_rerank_wait_list = []
avg_dup_ratio_list = []

def load_nanobeir_dataset(repo_id: str):
    logging.info(f"Loading dataset from local path (BEIR): '{repo_id}'...")
    corpus, queries, _ = GenericDataLoader(data_folder=data_path).load(split="test")
    target_queries = dict(islice(queries.items(), TARGET_NUM_QUERIES))

    candidates = [
        os.path.join(data_path, "qrels", "test.tsv"),
        os.path.join(data_path, "test.tsv"),
    ]
    qrels_pos = {}
    tsv_path = next((p for p in candidates if os.path.exists(p)), None)

    if tsv_path is None:
        logging.warning("[qrels] test.tsv not found; fallback to BEIR loader qrels.")
        _, _, qrels_beir = GenericDataLoader(data_folder=data_path).load(split="test")
        qrels_pos = qrels_beir
    else:
        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            def _get(row, *keys):
                for k in keys:
                    if k in row:
                        return row[k]
                raise KeyError(f"Missing any of keys {keys} in row: {row}")
            kept, skipped = 0, 0
            for row in reader:
                try:
                    qid   = _get(row, "query-id", "qid", "query_id")
                    docid = _get(row, "corpus-id", "docid", "doc_id")
                    score = int(_get(row, "score", "label"))
                except Exception as e:
                    logging.warning(f"[qrels] skip malformed row: {e}")
                    continue
                if score > 0:
                    qrels_pos.setdefault(str(qid), {})[str(docid)] = 1
                    kept += 1
                else:
                    skipped += 1
        logging.info(f"[qrels] loaded from {tsv_path}: kept positives={kept}, skipped non-positives={skipped}")

    logging.info(f"Dataset loaded: {len(corpus)} documents, {len(target_queries)} queries, "
                 f"{sum(len(v) for v in qrels_pos.values())} positive qrels.")
    return corpus, target_queries, qrels_pos

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
            topk_ids = set(doc for doc, _ in sorted(ranked_docs.items(), key=lambda x: x[1], reverse=True)[:k])
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

def _dcg_at_k(ranked_docids: List[str], qrels_q: Dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked_docids[:k], start=1):
        rel = int(qrels_q.get(str(did), 0))
        if rel <= 0: continue
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

def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy().astype(np.float32)
    elif isinstance(x, np.ndarray):
        return x.astype(np.float32)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def _read_proc_io_bytes() -> Optional[int]:
    try:
        with open("/proc/self/io", "r") as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except Exception:
        return None
    return None

def _get_rusage_faults():
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(ru.ru_minflt), int(ru.ru_majflt)
    except Exception:
        return None, None

# =====================================
# FDE stubs
# =====================================
from fde_generator_optimized_stream import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_document_fde_batch,
)

# =====================================
# Pack Builder & Reader
# =====================================
@dataclass
class PackPaths:
    tokens_bin: str
    doc_ptrs: str
    block_ptrs: str
    block_max_norms: str
    pack_meta: str

class PackBuilder:
    """pack 파일(연속 row-major float32) 생성기"""
    def __init__(self, pack_dir: str, block_size: int, dim: int):
        os.makedirs(pack_dir, exist_ok=True)
        self.paths = PackPaths(
            tokens_bin=os.path.join(pack_dir, "tokens.bin"),
            doc_ptrs=os.path.join(pack_dir, "doc_ptrs.npy"),
            block_ptrs=os.path.join(pack_dir, "block_ptrs.npy"),
            block_max_norms=os.path.join(pack_dir, "block_max_norms.npy"),
            pack_meta=os.path.join(pack_dir, "pack_meta.json"),
        )
        self.block_size = max(1, int(block_size))
        self.dim = int(dim)
        self.itemsize = 4

    @staticmethod
    def _l2_norm_rows(x: np.ndarray) -> np.ndarray:
        return np.sqrt((x * x).sum(axis=1))

    def build(self, retriever: "ColbertFdeRetrieverNaive", corpus: dict, doc_ids: List[str]):
        logging.info(f"[PackBuilder] building pack to {os.path.dirname(self.paths.tokens_bin)} "
                     f"(block_size={self.block_size})")
        f_tokens = open(self.paths.tokens_bin, "wb", buffering=0)
        doc_ptrs: List[int] = [0]
        block_ptrs: List[int] = [0]
        block_max_norms: List[float] = []

        total_rows = 0
        cur_block_rows: List[np.ndarray] = []
        cur_cnt = 0

        def _flush_block():
            nonlocal cur_block_rows, cur_cnt, total_rows
            if cur_cnt == 0:
                return
            blk = np.vstack(cur_block_rows).astype(np.float32, copy=False)
            maxn = float(self._l2_norm_rows(blk).max()) if blk.shape[0] > 0 else 0.0
            block_max_norms.append(maxn)
            f_tokens.write(blk.tobytes(order="C"))
            total_rows += blk.shape[0]
            block_ptrs.append(total_rows)
            cur_block_rows = []
            cur_cnt = 0

        for did in doc_ids:
            d_tok = retriever._get_doc_embeddings(did, allow_build=True)
            if isinstance(d_tok, torch.Tensor):
                d_tok = d_tok.detach().cpu().numpy()
            d_tok = d_tok.astype(np.float32, copy=False)
            n_i, d = d_tok.shape
            if d != self.dim:
                raise ValueError(f"[PackBuilder] dim mismatch: doc '{did}' has {d} vs pack dim {self.dim}")

            off = 0
            while off < n_i:
                remain = self.block_size - cur_cnt
                take = min(remain, n_i - off)
                cur_block_rows.append(d_tok[off:off+take])
                cur_cnt += take
                off += take
                if cur_cnt == self.block_size:
                    _flush_block()

            if cur_cnt > 0:
                _flush_block()

            doc_ptrs.append(total_rows)

        f_tokens.close()
        np.save(self.paths.doc_ptrs, np.asarray(doc_ptrs, dtype=np.int64))
        np.save(self.paths.block_ptrs, np.asarray(block_ptrs, dtype=np.int64))
        np.save(self.paths.block_max_norms, np.asarray(block_max_norms, dtype=np.float32))
        pack_meta = {
            "dtype": "float32",
            "dim": int(self.dim),
            "total_rows": int(total_rows),
            "n_docs": int(len(doc_ptrs) - 1),
            "n_blocks": int(len(block_ptrs) - 1),
            "block_size": int(self.block_size),
            "tokens_bin": os.path.basename(self.paths.tokens_bin),
            "doc_ptrs": os.path.basename(self.paths.doc_ptrs),
            "block_ptrs": os.path.basename(self.paths.block_ptrs),
            "block_max_norms": os.path.basename(self.paths.block_max_norms),
        }
        with open(self.paths.pack_meta, "w", encoding="utf-8") as f:
            json.dump(pack_meta, f, ensure_ascii=False, indent=2)
        logging.info(f"[PackBuilder] done: rows={total_rows}, blocks={len(block_ptrs)-1}")

class PackReader:
    """pack 구조를 읽는 경량 리더: os.pread 기반 연속 블록/행 읽기 + 임의행(span) 읽기"""
    def __init__(self, pack_dir: str):
        meta_path = os.path.join(pack_dir, "pack_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["dtype"] == "float32"
        self.dim = int(meta["dim"])
        self.itemsize = 4
        self.tokens_bin_path = os.path.join(pack_dir, meta["tokens_bin"])
        self.doc_ptrs = np.load(os.path.join(pack_dir, meta["doc_ptrs"])).astype(np.int64)
        self.block_ptrs = np.load(os.path.join(pack_dir, meta["block_ptrs"])).astype(np.int64)
        self.block_max_norms = np.load(os.path.join(pack_dir, meta["block_max_norms"])).astype(np.float32)
        self.block_size = int(meta["block_size"])
        self.n_blocks = int(meta["n_blocks"])
        self.total_rows = int(meta["total_rows"])
        self.fd = os.open(self.tokens_bin_path, os.O_RDONLY)

    @property
    def n_docs(self) -> int:
        return self.doc_ptrs.shape[0] - 1

    def close(self):
        try:
            os.close(self.fd)
        except Exception:
            pass

    def doc_row_span(self, doc_idx: int) -> Tuple[int, int]:
        s = int(self.doc_ptrs[doc_idx]); e = int(self.doc_ptrs[doc_idx+1])
        return s, e

    def pread_rows(self, start_row: int, n_rows: int) -> np.ndarray:
        if n_rows <= 0:
            return np.empty((0, self.dim), dtype=np.float32)
        byte_off = start_row * self.dim * self.itemsize
        byte_len = n_rows * self.dim * self.itemsize
        buf = os.pread(self.fd, byte_len, byte_off)
        arr = np.frombuffer(buf, dtype=np.float32, count=n_rows * self.dim)
        return np.ascontiguousarray(arr.reshape(n_rows, self.dim))

    def pread_row_spans(self, spans: List[Tuple[int, int]]) -> np.ndarray:
        """여러 (start_row, length) span을 순서대로 읽어 vstack"""
        if not spans:
            return np.empty((0, self.dim), dtype=np.float32)
        chunks = [self.pread_rows(s, n) for (s, n) in spans if n > 0]
        return np.ascontiguousarray(np.vstack(chunks)) if chunks else np.empty((0, self.dim), np.float32)

    @staticmethod
    def rows_to_spans(row_ids: np.ndarray) -> List[Tuple[int, int]]:
        """정렬되지 않은 row id 집합을 정렬/중복제거하고 연속 구간으로 합쳐 span으로 반환"""
        if row_ids.size == 0: return []
        r = np.unique(row_ids.astype(np.int64))
        spans = []
        s = int(r[0]); prev = int(r[0])
        for x in r[1:]:
            x = int(x)
            if x == prev + 1:
                prev = x; continue
            spans.append((s, prev - s + 1))
            s, prev = x, x
        spans.append((s, prev - s + 1))
        return spans

# =====================================
# Retriever
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
        use_faiss_ann: bool = True,
        faiss_nlist: int = FAISS_NLIST,
        faiss_nprobe: int = FAISS_NPROBE,
        faiss_candidates: int = FAISS_CANDIDATES,
        faiss_num_threads: int = FAISS_NUM_THREADS,
        fde_dim: int = FDE_DIM,
        fde_reps: int = FDE_NUM_REPETITIONS,
        fde_simhash: int = FDE_NUM_SIMHASH,
        # pack/half 옵션
        use_pack_half: bool = False,
        half_policy: str = "front",             # "front" | "back" | "stride2"
        pack_block_size: int = 512,
        build_pack_if_missing: bool = False,
    ):
        self.faiss_num_threads = max(1, int(faiss_num_threads))
        model = neural_cherche_models.ColBERT(model_name_or_path=model_name, device=DEVICE)
        self.ranker = neural_cherche_rank.ColBERT(key="id", on=["title", "text"], model=model)

        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=fde_dim, num_repetitions=fde_reps, num_simhash_projections=fde_simhash,
            seed=42, fill_empty_partitions=True,
        )

        self.fde_index: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        self._doc_pos = {}
        self._corpus = None

        self.enable_rerank = enable_rerank
        self.rerank_candidates = rerank_candidates
        self.save_doc_embeds = save_doc_embeds
        self.external_doc_embeds_dir = external_doc_embeds_dir

        self.use_faiss_ann = use_faiss_ann and _FAISS_OK
        self.faiss_nlist = faiss_nlist
        self.faiss_nprobe = faiss_nprobe
        self.faiss_candidates = faiss_candidates
        self.faiss_index = None

        self._model_name = model_name
        self._cache_dir = self._compute_cache_dir(dataset=DATASET_REPO_ID)

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

        # (NEW) sketch sidecar 저장 위치
        self._sketch_dir = os.path.join(self._cache_dir, "doc_stage")
        os.makedirs(self._sketch_dir, exist_ok=True)

        self._latency_log_path = latency_log_path or os.path.join(self._cache_dir, "latency.tsv")
        self._log_lock = threading.Lock()

        self._lru_enabled = ENABLE_DOC_EMB_LRU_CACHE and DOC_EMB_LRU_SIZE > 0
        if self._lru_enabled:
            from collections import OrderedDict as _OD
            self._lru = _OD()
            self._lru_lock = threading.Lock()
            self._lru_cap = DOC_EMB_LRU_SIZE
        
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
        
        self._per_query_log_path = os.path.join(CACHE_ROOT, f"per_query_{TOP_K}.tsv")
        try:
            with self._log_lock:                
                if not os.path.exists(self._per_query_log_path):                    
                    with open(self._per_query_log_path, "a", encoding="utf-8") as f:                        
                        f.write("qid\trecall_at_k\n")
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write per-query header: {e}")

        # pack 관련 상태
        self.use_pack_half = bool(use_pack_half)
        self.half_policy = str(half_policy)
        self.pack_block_size = int(pack_block_size)
        self.build_pack_if_missing = bool(build_pack_if_missing)
        self.pack_dir = os.path.join(self._cache_dir, "pack")
        self.pack: Optional[PackReader] = None  # ensure_pack()에서 채움

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

    def _external_doc_emb_path(self, doc_id: str) -> Optional[str]:
        if not self.external_doc_embeds_dir:
            return None
        pos = self._doc_pos.get(doc_id)
        if pos is None:
            return None
        return os.path.join(self._doc_emb_dir, f"{pos:08d}.npy")

    def _internal_doc_emb_path(self, doc_id: str) -> str:
        pos = self._doc_pos[doc_id]
        return os.path.join(self._doc_emb_dir, f"{pos:08d}.npy")

    # ---- sketch sidecar paths
    def _doc_sketch_path(self, doc_id: str) -> str:
        pos = self._doc_pos[doc_id]
        return os.path.join(self._sketch_dir, f"{pos:08d}.npy")
    def _doc_sketch_idx_path(self, doc_id: str) -> str:
        pos = self._doc_pos[doc_id]
        return os.path.join(self._sketch_dir, f"{pos:08d}.idx.npy")

    def _load_cache(self) -> bool:
        self.fde_index = joblib.load(self._fde_path)
        with open(self._ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        self._doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        logging.info(f"[{self.__class__.__name__}] Loaded FDE index cache: "
                     f"{self.fde_index.shape} for {len(self.doc_ids)} docs")

    def ensure_pack(self, corpus: dict):
        if not (self.use_pack_half or ENABLE_TWO_STAGE):
            return
        os.makedirs(self.pack_dir, exist_ok=True)
        need = not all(os.path.exists(os.path.join(self.pack_dir, p)) for p in
                       ["tokens.bin", "doc_ptrs.npy", "block_ptrs.npy", "block_max_norms.npy", "pack_meta.json"])
        if need:
            if not self.build_pack_if_missing:
                raise FileNotFoundError(
                    f"[Pack] pack files not found in {self.pack_dir}. "
                    f"Run with --build_pack_if_missing or prebuild them."
                )
            logging.info("[Pack] building pack (missing files detected)...")
            builder = PackBuilder(self.pack_dir, block_size=self.pack_block_size, dim=int(self.fde_index.shape[1]))
            builder.build(self, corpus, self.doc_ids)
        self.pack = PackReader(self.pack_dir)
        logging.info("[Pack] ready.")

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
        sim = query_tok @ doc_tok.T
        return float(sim.max(axis=1).sum()) if doc_tok.size else -1e9

    def _get_doc_rows_half_from_pack(self, doc_id: str, policy: str) -> np.ndarray:
        assert self.pack is not None, "pack reader not ready"
        # logging.info(f"[DDD] _get_doc_rows_half_from_pack")
        di = self._doc_pos[doc_id]
        s, e = self.pack.doc_row_span(di)
        n = max(0, e - s)
        if n <= 0:
            return np.empty((0, self.fde_index.shape[1]), dtype=np.float32)
        half = max(1, n // 2)
        if policy == "front":
            return self.pack.pread_rows(s, half)
        elif policy == "back":
            return self.pack.pread_rows(s + (n - half), half)
        elif policy == "stride2":
            q = max(1, half // 2)
            front = self.pack.pread_rows(s, q)
            back  = self.pack.pread_rows(s + (n - q), q)
            return np.vstack([front, back])
        else:
            return self.pack.pread_rows(s, half)

    def _get_doc_embeddings(self, doc_id: str, allow_build: bool = True) -> np.ndarray:
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

    # ==== Sketch build/load (원본 row-id 사이드카 포함) ====
    def _build_doc_sketch(self, doc_id: str, L: int = STAGE1_SKETCH_TOKENS) -> np.ndarray:
        d_tok = self._get_doc_embeddings(doc_id, allow_build=True)   # [n_i, d]
        n_i = d_tok.shape[0]
        if n_i <= L:
            idx = np.arange(n_i, dtype=np.int64)
            sketch = np.ascontiguousarray(d_tok.astype(np.float32))
        else:
            norms = np.sqrt((d_tok * d_tok).sum(axis=1))
            idx = np.argpartition(norms, -L)[-L:].astype(np.int64)   # original row ids
            sketch = np.ascontiguousarray(d_tok[idx].astype(np.float32))
        np.save(self._doc_sketch_path(doc_id), sketch)
        np.save(self._doc_sketch_idx_path(doc_id), idx)
        return sketch

    def _get_doc_sketch_and_idx(self, doc_id: str, L: int = STAGE1_SKETCH_TOKENS) -> Tuple[np.ndarray, np.ndarray]:
        sp = self._doc_sketch_path(doc_id)
        ip = self._doc_sketch_idx_path(doc_id)
        if os.path.exists(sp) and os.path.exists(ip):
            return np.load(sp), np.load(ip)
        sk = self._build_doc_sketch(doc_id, L=L)
        idx = np.load(self._doc_sketch_idx_path(doc_id))
        return sk, idx

    def index(self, corpus: dict):
        self._corpus = corpus
        # load FDE + doc_ids + faiss if any
        self.fde_index = joblib.load(self._fde_path)
        with open(self._ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        self._doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        logging.info(f"[{self.__class__.__name__}] Loaded FDE index cache: "
                     f"{self.fde_index.shape} for {len(self.doc_ids)} docs")

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
        except Exception:
            pass

        if self.use_faiss_ann and os.path.exists(self._faiss_path):
            try:
                self.faiss_index = faiss.read_index(self._faiss_path)
                self.faiss_index.nprobe = self.faiss_nprobe
                if hasattr(self.faiss_index, "d") and self.faiss_index.d != int(self.fde_index.shape[1]):
                    logging.warning(f"[FAISS] dim mismatch: index.d={self.faiss_index.d} vs FDE={self.fde_index.shape[1]} ⇒ rebuild")
                    self.faiss_index = None
            except Exception:
                self.faiss_index = None

        # (중요) two-stage 또는 pack-half가 켜져 있으면 pack 준비
        if self.use_pack_half or ENABLE_TWO_STAGE:
            self.ensure_pack(corpus)

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

    def _build_or_load_faiss_index(self):
        if not self.use_faiss_ann:
            return
        if self.faiss_index is not None and os.path.exists(self._fde_path):
            return
        self._set_faiss_threads()
        dim   = int(self.fde_index.shape[1])
        nvecs = int(self.fde_index.shape[0])
        logging.info(f"[FAISS] Building IVFFlat(IP) nlist={self.faiss_nlist} for {nvecs} (dim={dim})")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, self.faiss_nlist, faiss.METRIC_INNER_PRODUCT)
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
                if need <= 0: break
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

        add_bs = 100_000
        for start in range(0, nvecs, add_bs):
            end = min(start + add_bs, nvecs)
            xb = np.ascontiguousarray(self.fde_index[start:end], dtype=np.float32)
            index.add(xb)

        faiss.write_index(index, self._faiss_path)
        index.nprobe = self.faiss_nprobe
        self.faiss_index = index

    def ann_search_batch(self, XQ_batch: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, float]:
        assert XQ_batch.ndim == 2
        if self.faiss_index is None:
            self._build_or_load_faiss_index()
        self._set_faiss_threads()
        t0 = time.perf_counter()
        D, I = self.faiss_index.search(XQ_batch, k)
        ann_time = time.perf_counter() - t0
        return D, I, ann_time

    def _log_per_query(self, qid: str, recall_at_k: int):
        try:
            with self._log_lock:
                with open(self._per_query_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{qid}\t{recall_at_k}\n")
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write per-query row: {e}")

    def _log_latency(self, qid: str, search_s: float, ann_s: float, rerank_s: float,
                     rerank_compute_s: float, rerank_io_s: float, wait_s: float,
                     dup_ratio: Optional[float] = None):
        try:
            divided_ann_s = ann_s / ANN_BATCH_SIZE
            dr = -1.0 if (dup_ratio is None) else float(dup_ratio)            
            with self._log_lock:
                with open(self._latency_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{qid}\t{divided_ann_s*1000:.3f}\t{rerank_s*1000:.3f}\t"
                        f"{rerank_compute_s*1000:.3f}\t{rerank_io_s*1000:.3f}\t{wait_s*1000:.3f}\n"
                    )
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write latency log: {e}")

# ============== Rerank (mega GEMM) ==============
def _rerank_task_with_mega_gemm(retriever: ColbertFdeRetrieverNaive, task: "RerankTask", top_k: int):
    q_emb = task.query_embeddings
    N_compute = min(top_k if top_k > 0 else len(task.initial_candidates),
                    retriever.rerank_candidates if retriever.rerank_candidates > 0 else len(task.initial_candidates),
                    len(task.initial_candidates))
    compute_ids = [did for (did, _) in task.initial_candidates[:N_compute]]    

    pos_seq: List[int] = []

    t_io0 = time.perf_counter()
    doc_spans: List[Tuple[str, int, int]] = []
    blocks = []
    col_start = 0
    for did in compute_ids:
        try:
            pos_seq.append(int(retriever._doc_pos[did]))
        except Exception:
            pass

        # ------------------------------
        # Two-Stage ON: 스케치 row만 읽기 (pack 필요)
        # ------------------------------
        if ENABLE_TWO_STAGE and (retriever.pack is not None):
            # 1) 스케치와 원본 row-id(sidecar) 로드
            sketch, idx = retriever._get_doc_sketch_and_idx(did, L=STAGE1_SKETCH_TOKENS)
            # 2) 문서의 절대 row-id로 변환 후 연속 span으로 압축
            di = retriever._doc_pos[did]
            ds, de = retriever.pack.doc_row_span(di)
            abs_rows = ds + idx.astype(np.int64)
            spans = PackReader.rows_to_spans(abs_rows)
            # 3) 해당 span만 pread
            d_tok = retriever.pack.pread_row_spans(spans)

        else:
            # 기존 절반 읽기(팩 있으면 Half-Doc), 아니면 전체 임베딩
            if (retriever.pack is not None) and retriever.use_pack_half:
                d_tok = retriever._get_doc_rows_half_from_pack(did, retriever.half_policy)
            else:
                d_tok = retriever._get_doc_embeddings(did, allow_build=True)

        n_i = int(d_tok.shape[0])
        blocks.append(d_tok.astype(np.float32, copy=False))
        doc_spans.append((did, col_start, col_start + n_i))
        col_start += n_i

    D_all = None
    if blocks:
        D_all = np.ascontiguousarray(np.vstack(blocks).astype(np.float32))
    io_s = time.perf_counter() - t_io0

    t_c0 = time.perf_counter()
    reranked_pairs: List[Tuple[str, float]] = []

    if D_all is not None and D_all.size > 0:
        S = q_emb @ D_all.T
        for did, s, e in doc_spans:
            score = -1e9 if e - s == 0 else float(S[:, s:e].max(axis=1).sum())
            reranked_pairs.append((did, score))
    compute_s = time.perf_counter() - t_c0

    reranked_pairs.sort(key=lambda x: x[1], reverse=True)
    computed_set = {did for (did, _) in reranked_pairs}
    tail_pairs = [(did, sc) for (did, sc) in task.initial_candidates if did not in computed_set]
    out = OrderedDict()
    for did, sc in reranked_pairs: out[did] = float(sc)
    for did, sc in tail_pairs:     out[did] = float(sc)
    total_s = io_s + compute_s

    meta = dict()
    return out, total_s, compute_s, io_s, meta

# ============== 배치 오케스트레이션 ==============
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

def ann_aggregator_loop(retriever: ColbertFdeRetrieverNaive,
                        in_q: Queue, out_q: Queue,
                        k: int,
                        batch_size: int = ANN_BATCH_SIZE):
    exp_dim = int(retriever.fde_index.shape[1])
    XQ_list: List[np.ndarray] = []
    metas: List[Tuple[str, str, np.ndarray, float]] = []

    def flush():
        if not XQ_list: return
        XQb = np.vstack(XQ_list)
        D, I, ann_time = retriever.ann_search_batch(XQb, k)
        t_now = time.perf_counter()
        for i, (qid, qtext, qemb, t_enq) in enumerate(metas):
            mask = I[i] >= 0
            cand_ids = [retriever.doc_ids[idx] for idx in I[i][mask]]
            cand_scores = D[i][mask].tolist()
            initial_candidates = list(zip(cand_ids, cand_scores))
            task = RerankTask(qid=qid, qtext=qtext, query_embeddings=qemb,
                              initial_candidates=initial_candidates,
                              search_time_s=(t_now - t_enq), ann_time_s=ann_time,
                              enqueued_time_s=t_now)
            out_q.put(task)
        XQ_list.clear(); metas.clear()

    while True:
        item = in_q.get()
        if item == "__STOP__":
            flush()
            out_q.put("__STOP__")
            break

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
                           batch_queries: int = RERANK_BATCH_QUERIES,
                           top_k: int = TOP_K,
                           num_workers: int = RERANK_WORKERS):
    results_lock = Lock()
    stop_token = "__STOP__"

    def _commit_result(task: RerankTask, out_pairs: OrderedDict,
                       rerank_time: float, compute_rerank_time: float, io_rerank_time: float,
                       wait_s: float, dup_ratio: Optional[float] = None, meta: Optional[dict] = None):
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
        
        retriever._log_latency(task.qid, total_search_time, task.ann_time_s,
                               rerank_time, compute_rerank_time, io_rerank_time, wait_s,
                               dup_ratio=dup_ratio)

    def _process_task(task: RerankTask, dup_ratio_for_batch: Optional[float]):
        # (Two-Stage) ANN 상위 num_rank_candidates로 절단 (문서 수 제한)
        if ENABLE_TWO_STAGE and retriever.rerank_candidates > 0:
            topN = min(retriever.rerank_candidates, len(task.initial_candidates))
            task = replace(task, initial_candidates=task.initial_candidates[:topN])

        t_start = time.perf_counter()
        wait_s = t_start - task.enqueued_time_s
        t0 = time.perf_counter()
        out_pairs, total_rerank_s, compute_s, io_s, meta = _rerank_task_with_mega_gemm(retriever, task, top_k)
        rerank_time = time.perf_counter() - t0
        _commit_result(task, out_pairs, rerank_time, compute_s, io_s, wait_s,
                       dup_ratio=dup_ratio_for_batch, meta=meta)

    if BATCH_RERANK_MODE == "immediate":
        workers = []
        def worker_loop():
            while True:
                item = in_q.get()
                if item == "__STOP__":
                    in_q.put("__STOP__")
                    break
                _process_task(item, dup_ratio_for_batch=None)
        for _ in range(max(1, int(num_workers))):
            t = threading.Thread(target=worker_loop, daemon=True)
            t.start(); workers.append(t)
        for t in workers: t.join()
        return

    elif BATCH_RERANK_MODE == "batch":
        buffer: List[RerankTask] = []

        def _flush_batch(buf: List[RerankTask]):
            if not buf: return
            all_doc_ids: List[str] = []
            for task in buf:
                N_compute = min(top_k if top_k>0 else len(task.initial_candidates),
                                retriever.rerank_candidates if retriever.rerank_candidates>0 else len(task.initial_candidates),
                                len(task.initial_candidates))
                all_doc_ids.extend([did for (did, _) in task.initial_candidates[:N_compute]])
            total_docs = len(all_doc_ids)
            dup_ratio_for_batch = 0.0 if total_docs == 0 else 1.0 - (len(set(all_doc_ids)) / total_docs)

            parallelism = min(max(1, int(num_workers)), len(buf))
            with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="rerank-batch") as ex:
                futures = [ex.submit(_process_task, t, dup_ratio_for_batch) for t in buf]
                for f in as_completed(futures):
                    _ = f.result()

        while True:
            item = in_q.get()
            if item == "__STOP__":
                _flush_batch(buffer)
                break
            buffer.append(item)
            if len(buffer) >= max(1, int(BATCH_RERANK_SIZE)):
                _flush_batch(buffer); buffer.clear()
        return
    else:
        raise ValueError(f"Unknown BATCH_RERANK_MODE={BATCH_RERANK_MODE}")

# ============================
# --- GLOBAL (BF build) ------
# ============================
BF_WORKERS = max(1, (os.cpu_count() or 4) // 2)
BF_CHUNK_SIZE = 256
_DOC_BUILD_LOCK = threading.Lock()

def _safe_get_doc_embeddings(retriever: ColbertFdeRetrieverNaive, did: str) -> np.ndarray:
    # two-stage + pack이 준비되어 있으면 스케치 행만 읽기
    if ENABLE_TWO_STAGE and (retriever.pack is not None):
        _, idx = retriever._get_doc_sketch_and_idx(did, L=STAGE1_SKETCH_TOKENS)
        di = retriever._doc_pos[did]
        ds, de = retriever.pack.doc_row_span(di)
        abs_rows = ds + idx.astype(np.int64)
        spans = PackReader.rows_to_spans(abs_rows)
        return retriever.pack.pread_row_spans(spans)
    # 아니면 기존 half-read 또는 full
    if (retriever.pack is not None) and retriever.use_pack_half:
        return retriever._get_doc_rows_half_from_pack(did, retriever.half_policy)
    int_path = retriever._internal_doc_emb_path(did)
    if os.path.exists(int_path):
        return np.load(int_path)
    with _DOC_BUILD_LOCK:
        if os.path.exists(int_path):
            return np.load(int_path)
        return retriever._get_doc_embeddings(did, allow_build=True)

def _bf_chunk_worker(retriever: ColbertFdeRetrieverNaive, q_emb: np.ndarray, doc_ids: List[str], k: int):
    local_heap: List[Tuple[float, str]] = []
    push = heapq.heappush; replace = heapq.heapreplace
    for did in doc_ids:
        d_tok = _safe_get_doc_embeddings(retriever, did)
        score = retriever._chamfer(q_emb, d_tok)
        if len(local_heap) < k: push(local_heap, (score, did))
        else:
            if score > local_heap[0][0]: replace(local_heap, (score, did))
    return local_heap

def _compute_bf_topk_for_query(retriever: ColbertFdeRetrieverNaive, qid: str, qtext: str, k: int,
                               workers: int = None, chunk_size: int = 256):
    if workers is None: workers = max(1, (os.cpu_count() or 4) // 2)
    key = retriever._query_key(qtext, qid)
    qemb, qfde = retriever._load_query_cache(key)
    if qemb is None:
        qmap = retriever.ranker.encode_queries(queries=[qtext])
        qemb = to_numpy(next(iter(qmap.values())))
        qcfg = replace(retriever.doc_config, fill_empty_partitions=False)
        qfde = generate_query_fde(qemb, qcfg)
        retriever._save_query_cache(key, qemb, qfde)

    doc_ids = retriever.doc_ids
    chunks: List[List[str]] = [doc_ids[i:i+chunk_size] for i in range(0, len(doc_ids), chunk_size)]

    global_heap: List[Tuple[float, str]] = []
    push = heapq.heappush; replace = heapq.heapreplace

    with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="bf-doc") as ex:
        futures = [ex.submit(_bf_chunk_worker, retriever, qemb, ch, k) for ch in chunks]
        for fut in as_completed(futures):
            local_heap = fut.result()
            for sc, did in local_heap:
                if len(global_heap) < k: push(global_heap, (sc, did))
                else:
                    if sc > global_heap[0][0]: replace(global_heap, (sc, did))

    top_sorted = sorted(((did, sc) for sc, did in global_heap), key=lambda x: x[1], reverse=True)
    return top_sorted

def compute_and_persist_bf_topk(retriever: ColbertFdeRetrieverNaive, queries: Dict[str, str], k: int, outfile: str):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    seen_qids = _load_existing_bf_qids(outfile)
    will_process = [ (qid, qtext) for qid, qtext in queries.items() if str(qid) not in seen_qids ]
    if not will_process:
        logging.info(f"[BF] Nothing to compute. All {len(queries)} queries already exist in {outfile}.")
        return
    logging.info(f"[BF] Computing brute-force Top-{k} for {len(will_process)} queries (append to {outfile}) "
                 f"| workers={BF_WORKERS}, chunk={BF_CHUNK_SIZE}")
    for qid, qtext in will_process:
        t0 = time.perf_counter()
        topk = _compute_bf_topk_for_query(retriever, str(qid), qtext, k,
                                          workers=BF_WORKERS, chunk_size=BF_CHUNK_SIZE)
        _append_bf_topk(outfile, str(qid), topk)
        logging.info(f"[BF] qid={qid} done in {time.perf_counter()-t0:.2f}s")

def _load_existing_bf_qids(path: str) -> set:
    if not os.path.exists(path): return set()
    seen = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                qid = line.split("\t", 1)[0]
                seen.add(qid)
    except Exception as e:
        logging.warning(f"[BF] failed to read existing file: {e}")
    return seen

def _append_bf_topk(path: str, qid: str, topk: List[Tuple[str, float]]):
    with open(path, "a", encoding="utf-8") as f:
        for rank, (docid, score) in enumerate(topk, start=1):
            f.write(f"{qid}\t{docid}\t{score:.8f}\t{rank}\n")

def load_bf_truth(outfile: str) -> Dict[str, List[Tuple[str, float]]]:
    truth: Dict[str, List[Tuple[str, float]]] = {}
    if not os.path.exists(outfile): return truth
    with open(outfile, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            qid, docid, score, rank = line.rstrip("\n").split("\t")
            score = float(score)
            truth.setdefault(qid, []).append((docid, score))
    for qid in truth.keys():
        truth[qid] = sorted(truth[qid], key=lambda x: x[1], reverse=True)[:TOP_K]
    return truth

def system_topk_from_results(results: Dict[str, OrderedDict], k: int) -> Dict[str, List[str]]:
    out = {}
    for qid, ranked in results.items():
        out[str(qid)] = list(islice(ranked.keys(), k))
    return out

def recall_at_k_wrt_bf(results_topk: Dict[str, List[str]], bf_truth: Dict[str, List[Tuple[str, float]]], k: int) -> float:
    hits = 0; total = 0
    for qid, bf_list in bf_truth.items():
        bf_set = {doc for doc, _ in bf_list[:k]}
        sys_set = set(results_topk.get(qid, [])[:k])
        if not bf_set: continue
        total += 1
        hits += len(bf_set.intersection(sys_set)) / len(bf_set)
    return hits / total if total > 0 else 0.0

def hit_at_k_wrt_bf(results_topk: Dict[str, List[str]], bf_truth: Dict[str, List[Tuple[str, float]]], k: int) -> float:
    hits = 0; total = 0
    for qid, bf_list in bf_truth.items():
        bf_set = {doc for doc, _ in bf_list[:k]}
        if not bf_set: continue
        total += 1
        sys_set = set(results_topk.get(qid, [])[:k])
        hits += 1 if len(bf_set.intersection(sys_set)) > 0 else 0
    return hits / total if total > 0 else 0.0

def ndcg_at_k_wrt_bf(results_topk: Dict[str, List[str]], bf_truth: Dict[str, List[Tuple[str, float]]], k: int) -> float:
    def _discount(i: int) -> float: return 1.0 / math.log2(i + 1)
    per_q = []
    for qid, ideal in bf_truth.items():
        ideal_k = ideal[:k]
        if not ideal_k: continue
        ideal_gains = [max(s, 0.0) for _, s in ideal_k]
        idcg = sum(g * _discount(i+1) for i, g in enumerate(ideal_gains))
        if idcg <= 0: 
            per_q.append(0.0); continue
        sys_docs = results_topk.get(qid, [])[:k]
        ideal_map = {doc: max(sc, 0.0) for doc, sc in ideal_k}
        dcg = 0.0
        for i, d in enumerate(sys_docs, start=1):
            g = ideal_map.get(d, 0.0)
            dcg += g * _discount(i)
        per_q.append(dcg / idcg)
    return sum(per_q)/len(per_q) if per_q else 0.0

# ======================
# --- Main Script ------
# ======================
if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        pass

    def _positive_int(x: str) -> int:
        v = int(x)
        if v <= 0:
            raise argparse.ArgumentTypeError("must be > 0")
        return v

    parser = argparse.ArgumentParser(description="ANN + (optional) pack-half rerank + (optional) two-stage truncation (sketch row I/O)")
    parser.add_argument("--num_simhash_projections", "--p", type=_positive_int, required=True,
                        help="Number of simhash projections (P)")
    parser.add_argument("--num_repetitions", "--r", type=_positive_int, required=True,
                        help="Number of repetitions (R)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Optional explicit input pickle (defaults to fde_index_{P}_{R}.pkl)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional FAISS output path (defaults to ivf{nlist}_ip_{P}_{R}.faiss)")
    parser.add_argument("--nlist", "-nl", type=_positive_int, default=1000,
                        help="IVF list count (default: 1000)")
    parser.add_argument("--num_rerank_cand", "-rc", type=int, required=True,
                        help="number of rerank candidates")
    # pack/half 옵션
    parser.add_argument("--use_pack_half", action="store_true",
                        help="use pack layout to read only half of doc rows during rerank")
    parser.add_argument("--half_policy", type=str, default="front",
                        choices=["front","back","stride2"], help="Half-Doc policy for pack reading")
    parser.add_argument("--build_pack_if_missing", action="store_true",
                        help="build pack files if they don't exist")
    parser.add_argument("--pack_block_size", type=int, default=512,
                        help="pack block size for building tokens.bin")
    # two-stage 옵션
    parser.add_argument("--enable_two_stage", action="store_true",
                        help="take top-N from ANN (num_rerank_cand) then rerank using sketch-row subset I/O")

    args, _ = parser.parse_known_args()

    P = args.num_simhash_projections
    R = args.num_repetitions
    argslist = args.nlist
    num_rank_candidates = args.num_rerank_cand

    in_default = f"fde_index_{P}_{R}.pkl"
    faiss_default = f"ivf{argslist}_ip_{P}_{R}.faiss"
    meta_default = f"meta_{P}_{R}.json"

    ENABLE_TWO_STAGE = False # 이렇게 해야 half policy를 먹일 수 있음
    
    corpus, queries, qrels = load_nanobeir_dataset(DATASET_REPO_ID)

    retriever = ColbertFdeRetrieverNaive(
        model_name=COLBERT_MODEL_NAME,
        rerank_candidates=num_rank_candidates,
        enable_rerank=True,
        save_doc_embeds=True,
        latency_log_path=os.path.join(CACHE_ROOT, "latency.tsv"),
        external_doc_embeds_dir=None,
        use_faiss_ann=True,
        faiss_nlist=FAISS_NLIST,
        faiss_nprobe=FAISS_NPROBE,
        faiss_candidates=FAISS_CANDIDATES,
        faiss_num_threads=FAISS_NUM_THREADS,
        fde_dim=FDE_DIM,
        fde_reps=FDE_NUM_REPETITIONS,
        fde_simhash=FDE_NUM_SIMHASH,
        # pack/half
        use_pack_half=args.use_pack_half,               # two-stage와 별개, 두-stage가 켜지면 pack도 시도
        half_policy=args.half_policy,
        pack_block_size=args.pack_block_size,
        build_pack_if_missing=args.build_pack_if_missing,
    )

    t_ready0 = time.perf_counter()
    retriever.index(corpus)  # two-stage 또는 pack-half면 내부에서 pack 준비
    t_ready = time.perf_counter() - t_ready0
    logging.info(f"Retriever ready in {t_ready:.2f}s (two_stage={ENABLE_TWO_STAGE}, "
                 f"pack_used={retriever.pack is not None}, policy={retriever.half_policy}, L={STAGE1_SKETCH_TOKENS})")

    retriever.precompute_queries(queries)

    # (선택) 브루트포스 상한선 계산
    BF_OUTFILE = os.path.join(CACHE_ROOT, f"{DATASET_REPO_ID}_bruteforce_top{TOP_K}.tsv")
    compute_and_persist_bf_topk(retriever, queries, TOP_K, BF_OUTFILE)
    bf_truth = load_bf_truth(BF_OUTFILE)

    ann_in_q: Queue = Queue(maxsize=4096)
    rerank_in_q: Queue = Queue(maxsize=4096)
    results: Dict[str, OrderedDict] = {}

    start_time = time.perf_counter()
    ann_thr = threading.Thread(target=ann_aggregator_loop, args=(retriever, ann_in_q, rerank_in_q,
                                                                 max(FAISS_CANDIDATES, RERANK_TOPN),
                                                                 ANN_BATCH_SIZE),
                               daemon=True)
    rr_thr = threading.Thread(target=rerank_aggregator_loop, args=(retriever, rerank_in_q, results,
                                                                   RERANK_BATCH_QUERIES),
                              daemon=True)
    ann_thr.start(); rr_thr.start()

    q_start_times: Dict[str, float] = {}
    for qid, qtext in queries.items():
        q_start_times[qid] = time.perf_counter()
        ann_in_q.put(AnnItem(qid=qid, qtext=qtext, t_enqueue=q_start_times[qid]))

    ann_in_q.put("__STOP__")
    ann_thr.join()
    rerank_in_q.put("__STOP__")
    rr_thr.join()

    end_time = time.perf_counter() - start_time

    total_search_s = mean(avg_ann_time_list) + mean(avg_rerank_time_list) if avg_ann_time_list and avg_rerank_time_list else 0.0

    sys_topk = system_topk_from_results(results, TOP_K)
    bf_recall = recall_at_k_wrt_bf(sys_topk, bf_truth, TOP_K)
    bf_hit = hit_at_k_wrt_bf(sys_topk, bf_truth, TOP_K)
    bf_ndcg = ndcg_at_k_wrt_bf(sys_topk, bf_truth, TOP_K)

    _per_experiment_log_path = os.path.join(CACHE_ROOT, f"per_experiment_{DATASET_REPO_ID}")
    try:
        with open(_per_experiment_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Dataset: {DATASET_REPO_ID}, Queries: {len(queries)}, FirstCand: {FAISS_CANDIDATES} | "
                f"ANN_BATCH:{ANN_BATCH_SIZE}, RERANK_BATCH_Q:{RERANK_BATCH_QUERIES}, "
                f"RERANK_TOTAL: {mean(avg_rerank_time_list)*1000:.2f} | "
                f"RERANK_CAND:{num_rank_candidates}, Search: {total_search_s*1000:.2f} | "
                f"ANN: {mean(avg_ann_time_list)*1000:.2f} Rerank(CP): {mean(avg_rerank_cp_list)*1000:.2f} | "
                f"Rerank(IO): {mean(avg_rerank_io_list)*1000:.2f} | "
                f"Recall@{TOP_K}(BF): {bf_recall:.4f}, nDCG@{TOP_K}(BF): {bf_ndcg:.4f}\n"                
            )
    except Exception as e:
        logging.warning(f"Failed to write per-experiment: {e}")