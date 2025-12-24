# -*- coding: utf-8 -*-
import os, json, time, hashlib, logging, pathlib, math, heapq, threading, csv
from collections import OrderedDict
from dataclasses import replace
from typing import Optional, List, Tuple, Dict
from itertools import islice
from statistics import mean

import nltk
import numpy as np
import torch
import joblib
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

from concurrent.futures import ThreadPoolExecutor, as_completed
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

# FDE 구현 (업로드된 파일 사용)
from fde_generator_optimized_stream_weight_partition_gpu import (
    FixedDimensionalEncodingConfig,
    ProjectionType,
    generate_query_fde,
    generate_document_fde_batch_gpu_3stage,
)

# ======================
# --- Configuration ----
# ======================
DATASET_REPO_ID = "scidocs"
COLBERT_MODEL_NAME = "raphaelsty/neural-cherche-colbert"
TOP_K = 10
FILENAME = "main_weight_partition_gpu_naive_ann"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# 캐시 루트
CACHE_ROOT = os.path.join("/media/dcceris", "muvera_optimized", "cache_muvera", DATASET_REPO_ID, FILENAME)
os.makedirs(CACHE_ROOT, exist_ok=True)

# 쿼리 검색 디렉터리
dataset = DATASET_REPO_ID
QUERY_SEARCH_DIR = os.path.join(CACHE_ROOT, "query_search")
os.makedirs(QUERY_SEARCH_DIR, exist_ok=True)

# 공통 문서 임베딩 디렉터리 설정
COMMON_EMBEDS_DIR = os.path.join("/media/dcceris", "muvera_optimized", "cache_muvera", DATASET_REPO_ID)
COMMON_DOC_EMBEDS_DIR = os.path.join(COMMON_EMBEDS_DIR, "doc_embeds")
COMMON_QUERY_EMBEDS_DIR = os.path.join(COMMON_EMBEDS_DIR, "query_embeds")
os.makedirs(COMMON_DOC_EMBEDS_DIR, exist_ok=True)
os.makedirs(COMMON_QUERY_EMBEDS_DIR, exist_ok=True)

# ======================
# --- FAISS Configuration ----
# ======================
FAISS_NLIST = 1000
FAISS_NPROBE = 50
FAISS_CANDIDATES = 100  # over-fetch; rerank보다 크거나 같게 권장
FAISS_NUM_THREADS = 1   # OpenMP 스레드 수(권장: 1 또는 소수)

# ====== Rerank Batch Mode Switches ======
# 'immediate' : 쿼리 도착 즉시 Rerank (워커 병렬, mega GEMM)
# 'batch'     : Rerank 작업을 BATCH_RERANK_SIZE 개 모아 한 번에 멀티스레드 나이브 rerank
BATCH_RERANK_MODE = "batch"   # "immediate" or "batch"
BATCH_RERANK_SIZE = 1         # 배치 모을 크기

# ----- 캐시(기본 끔: 정말 나이브) -----
ENABLE_DOC_EMB_LRU_CACHE = False
DOC_EMB_LRU_SIZE = 0          # 0이면 캐시 안씀

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
avg_cpu_matmul_list = []
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
    out_dir = os.path.join("/media/dcceris", "muvera_optimized", "datasets")

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
    corpus, queries, _ = GenericDataLoader(data_folder=data_path).load(split="test")
    # TARGET_NUM_QUERIES가 정의되어 있으면 사용, 없으면 전체 쿼리 사용
    if 'TARGET_NUM_QUERIES' in globals() and TARGET_NUM_QUERIES > 0:
        target_queries = dict(islice(queries.items(), TARGET_NUM_QUERIES))
    else:
        target_queries = queries

    # ---- qrels: test.tsv에서 score>0만 포함 ----
    candidates = [
        os.path.join(data_path, "qrels", "test.tsv"),
        os.path.join(data_path, "test.tsv"),
    ]
    qrels_pos = {}
    tsv_path = next((p for p in candidates if os.path.exists(p)), None)

    if tsv_path is None:
        logging.warning("[qrels] test.tsv not found; falling back to BEIR loader qrels (may already be filtered).")
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
    from itertools import islice
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

# ============================
# --- NEW: Bruteforce Top-K ---
# ============================
# 병렬 브루트포스 설정
BF_WORKERS = max(1, (os.cpu_count() or 4) // 2)
BF_CHUNK_SIZE = 256

# 전역 빌드 락(문서 임베딩이 없을 때 생성 구간 직렬화)
_DOC_BUILD_LOCK = threading.Lock()

# 안전 로더 (BF와 rerank 병렬 시 파일 write 충돌 방지)
def _safe_get_doc_embeddings(retriever, did: str) -> np.ndarray:
    """문서 임베딩 안전 로드 (병렬 환경에서 파일 충돌 방지)"""
    common_path = retriever._common_doc_emb_path(did)
    if common_path and os.path.exists(common_path):
        return np.load(common_path)
    
    ext_path = retriever._external_doc_emb_path(did)
    if ext_path and os.path.exists(ext_path):
        return np.load(ext_path)
    
    # 생성이 필요할 수 있으니 락
    with _DOC_BUILD_LOCK:
        # 다른 스레드가 방금 생성했을 수 있으니 재확인
        if common_path and os.path.exists(common_path):
            return np.load(common_path)
        if ext_path and os.path.exists(ext_path):
            return np.load(ext_path)
        return retriever._get_doc_embeddings(did, allow_build=True)

def _load_existing_bf_qids(path: str) -> set:
    """이미 저장된 qid 집합을 반환 (파일 없으면 빈 집합)."""
    if not os.path.exists(path):
        return set()
    seen = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                qid = line.split("\t", 1)[0]
                seen.add(qid)
    except Exception as e:
        logging.warning(f"[BF] failed to read existing file: {e}")
    return seen

def _append_bf_topk(path: str, qid: str, topk: List[Tuple[str, float]]):
    """브루트포스 Top-K를 파일에 append (qid당 K줄)."""
    with open(path, "a", encoding="utf-8") as f:
        for rank, (docid, score) in enumerate(topk, start=1):
            f.write(f"{qid}\t{docid}\t{score:.8f}\t{rank}\n")

def _bf_chunk_worker(retriever,
                     q_emb: np.ndarray,
                     doc_ids: List[str],
                     k: int) -> List[Tuple[float, str]]:
    """한 청크의 문서들에 대한 로컬 Top-K 반환: [(score, docid), ...] (min-heap 유지)"""
    local_heap: List[Tuple[float, str]] = []
    push = heapq.heappush
    heapreplace = heapq.heapreplace

    for did in doc_ids:
        d_tok = _safe_get_doc_embeddings(retriever, did)
        score = retriever._chamfer(q_emb, d_tok)
        if len(local_heap) < k:
            push(local_heap, (score, did))
        else:
            if score > local_heap[0][0]:
                heapreplace(local_heap, (score, did))
    return local_heap

def _compute_bf_topk_for_query(retriever,
                               qid: str,
                               qtext: str,
                               k: int,
                               workers: int = None,
                               chunk_size: int = 256) -> List[Tuple[str, float]]:
    """쿼리 하나에 대해 Chamfer 정확 점수로 전 코퍼스를 병렬 브루트포스하고 Top-K 반환."""
    if workers is None:
        workers = max(1, (os.cpu_count() or 4) // 2)

    # 쿼리 임베딩 준비(토큰)
    key = retriever._query_key(qtext, qid)
    qemb, qfde = retriever._load_query_cache(key)
    if qemb is None:
        query_embeddings_map = retriever.ranker.encode_queries(queries=[qtext])
        qemb = to_numpy(next(iter(query_embeddings_map.values())))
        query_config = replace(retriever.doc_config, fill_empty_partitions=False)
        query_fde_result = generate_query_fde(qemb, query_config)
        if isinstance(query_fde_result, tuple):
            qfde = query_fde_result[0]
        else:
            qfde = query_fde_result
        retriever._save_query_cache(key, qemb, qfde)

    # 문서 id를 청크로 분할
    doc_ids = retriever.doc_ids
    chunks: List[List[str]] = [doc_ids[i:i+chunk_size] for i in range(0, len(doc_ids), chunk_size)]

    # 각 청크를 병렬로 처리하여 로컬 top-k 반환 → 전역 병합
    global_heap: List[Tuple[float, str]] = []
    push = heapq.heappush
    heapreplace = heapq.heapreplace

    with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="bf-doc") as ex:
        futures = [ex.submit(_bf_chunk_worker, retriever, qemb, ch, k) for ch in chunks]
        for fut in as_completed(futures):
            local_heap = fut.result()
            for sc, did in local_heap:
                if len(global_heap) < k:
                    push(global_heap, (sc, did))
                else:
                    if sc > global_heap[0][0]:
                        heapreplace(global_heap, (sc, did))

    # 큰 점수 우선 내림차순 정렬
    top_sorted = sorted(((did, sc) for sc, did in global_heap), key=lambda x: x[1], reverse=True)
    return top_sorted

def compute_and_persist_bf_topk(retriever,
                                queries: Dict[str, str],
                                k: int,
                                outfile: str):
    """ANN 전: 각 쿼리에 대해 브루트포스 Top-K를 계산해 outfile에 append 저장(이미 있으면 스킵)."""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    seen_qids = _load_existing_bf_qids(outfile)
    will_process = [ (qid, qtext) for qid, qtext in queries.items() if str(qid) not in seen_qids ]
    if not will_process:
        logging.info(f"[BF] Nothing to compute. All {len(queries)} queries already exist in {outfile}.")
        return

    logging.info(f"[BF] Computing brute-force Top-{k} for {len(will_process)} queries (append to {outfile}) | workers={BF_WORKERS}, chunk={BF_CHUNK_SIZE}")
    for qid, qtext in will_process:
        t0 = time.perf_counter()
        topk = _compute_bf_topk_for_query(retriever, str(qid), qtext, k,
                                          workers=BF_WORKERS, chunk_size=BF_CHUNK_SIZE)
        _append_bf_topk(outfile, str(qid), topk)
        logging.info(f"[BF] qid={qid} done in {time.perf_counter()-t0:.2f}s")

def load_bf_truth(outfile: str) -> Dict[str, List[Tuple[str, float]]]:
    """파일에서 브루트포스 Top-K 진리값을 로드: {qid: [(docid, score), ...] (desc)}"""
    truth: Dict[str, List[Tuple[str, float]]] = {}
    if not os.path.exists(outfile):
        return truth
    with open(outfile, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, docid, score, rank = line.rstrip("\n").split("\t")
            score = float(score)
            truth.setdefault(qid, []).append((docid, score))
    # rank가 보장되긴 하지만, 안전하게 점수 내림차순 정렬
    for qid in truth.keys():
        truth[qid] = sorted(truth[qid], key=lambda x: x[1], reverse=True)
        # 상위 K만 유지(파일이 중복 append되더라도 방어)
        truth[qid] = truth[qid][:TOP_K]
    return truth

# === (NEW) Metrics w.r.t. Brute-force Top-K ===
def system_topk_from_results(results: Dict[str, OrderedDict], k: int) -> Dict[str, List[str]]:
    """시스템 결과를 Top-K 리스트로 변환"""
    out = {}
    for qid, ranked in results.items():
        out[str(qid)] = list(islice(ranked.keys(), k))
    return out

def recall_at_k_wrt_bf(results_topk: Dict[str, List[str]],
                       bf_truth: Dict[str, List[Tuple[str, float]]],
                       k: int) -> float:
    """BF 기준 Recall@K"""
    hits = 0
    total = 0
    for qid, bf_list in bf_truth.items():
        bf_set = {doc for doc, _ in bf_list[:k]}
        sys_set = set(results_topk.get(qid, [])[:k])
        if not bf_set:
            continue
        total += 1
        hits += len(bf_set.intersection(sys_set)) / len(bf_set)
    return hits / total if total > 0 else 0.0

def hit_at_k_wrt_bf(results_topk: Dict[str, List[str]],
                    bf_truth: Dict[str, List[Tuple[str, float]]],
                    k: int) -> float:
    """BF 기준 Hit@K"""
    hits = 0
    total = 0
    for qid, bf_list in bf_truth.items():
        bf_set = {doc for doc, _ in bf_list[:k]}
        if not bf_set:
            continue
        total += 1
        sys_set = set(results_topk.get(qid, [])[:k])
        hits += 1 if len(bf_set.intersection(sys_set)) > 0 else 0
    return hits / total if total > 0 else 0.0

def ndcg_at_k_wrt_bf(sys_topk: Dict[str, List[str]],
                     bf_truth: Dict[str, List[Tuple[str, float]]],
                     k: int) -> Tuple[float, List[float]]:
    """
    sys_topk를 기준으로 순회하며 qid별 nDCG@k 계산.
    - bf_truth[qid]에 정답이 없거나 비어 있으면 해당 qid는 스킵.
    - 이상적 이득(idcg)은 bf_truth의 상위 k개 점수(>=0으로 컷)로 계산.
    - 시스템 이득(dcg)은 sys_topk[qid]의 순서에 따라 bf_truth 점수를 매핑해 계산.
    """
    def _discount(i: int) -> float:
        return 1.0 / math.log2(i + 1)

    per_q: List[float] = []
    for qid, sys_docs in sys_topk.items():
        ideal = bf_truth.get(str(qid), [])
        if not ideal:
            continue

        ideal_k = ideal[:k]
        if not ideal_k:
            continue

        # IDCG: bf_truth의 점수 기반
        ideal_gains = [max(s, 0.0) for _, s in ideal_k]
        idcg = sum(g * _discount(i+1) for i, g in enumerate(ideal_gains))
        if idcg <= 0:
            per_q.append(0.0)
            continue

        # DCG: 시스템 결과 순서에 bf_truth 점수를 매핑
        ideal_map = {doc: max(sc, 0.0) for doc, sc in ideal_k}
        dcg = 0.0
        for i, d in enumerate(sys_docs[:k], start=1):
            g = ideal_map.get(d, 0.0)
            dcg += g * _discount(i)

        per_q.append(dcg / idcg)

    mean_ndcg = sum(per_q) / len(per_q) if per_q else 0.0
    return mean_ndcg, per_q

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
        projection_dimension: int = 128,  # ★ 추가: projection dimension
        # FAISS params
        use_faiss_ann: bool = True,
        faiss_nlist: int = FAISS_NLIST,
        faiss_nprobe: int = FAISS_NPROBE,
        faiss_candidates: int = FAISS_CANDIDATES,
        faiss_num_threads: int = FAISS_NUM_THREADS,
    ):
        model = neural_cherche_models.ColBERT(model_name_or_path=model_name, device=DEVICE)
        self.ranker = neural_cherche_rank.ColBERT(key="id", on=["title", "text"], model=model)
        self.projection_dimension = projection_dimension

        # 추가된 인자
        self.num_repetitions = num_repetitions
        self.num_simhash_projections = num_simhash_projections

        # projection_dimension이 128이 아닌 경우 AMS_SKETCH projection 사용
        use_projection = self.projection_dimension != 128
        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=128,   # ColBERT 임베딩의 원본 차원 (고정)
            projection_dimension=self.projection_dimension if use_projection else None,  # projection 후 차원
            projection_type=ProjectionType.AMS_SKETCH if use_projection else ProjectionType.DEFAULT_IDENTITY,
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
        
        # FAISS
        self.use_faiss_ann = use_faiss_ann and _FAISS_OK
        self.faiss_nlist = faiss_nlist
        self.faiss_nprobe = faiss_nprobe
        self.faiss_candidates = faiss_candidates
        self.faiss_num_threads = max(1, int(faiss_num_threads))
        self.faiss_index = None
        
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
        
        # FAISS 인덱스 경로 (캐시 디렉터리 기반)
        proj_dim = self.projection_dimension
        proj_suffix = f"_{proj_dim}" if proj_dim != 128 else ""
        self._faiss_path = os.path.join(self._cache_dir, f"faiss_ivf{self.faiss_nlist}_ip_{self.num_repetitions}_{self.num_simhash_projections}{proj_suffix}.faiss")

        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._queries_dir, exist_ok=True)
        # 개별 하위 디렉터리에 doc_embeds 저장하지 않음 (공통 디렉터리 사용)

        # 지연시간 로그 파일 (헤더 없이 누적)
        self._latency_log_path = latency_log_path or os.path.join(self._cache_dir, "latency.tsv")

    def run_bit_ablation(self, corpus, queries, qrels, top_k=10):
        """
        전체 FDE index(self.fde_index) 생성이 끝난 뒤 실행.
        simhash bit을 하나씩 제거하여 전체 recall 변화 측정.
        """
        if self.fde_index is None:
            raise RuntimeError("FDE index is not built yet.")

        b = self.doc_config.num_simhash_projections
        rep = self.doc_config.num_repetitions

        logging.info(f"[Ablation] Running bit ablation: {b} simhash bits")

        # (0) full score 먼저 계산
        full_results = {}
        for qid, qtext in queries.items():
            full_results[str(qid)] = self.search(qtext, query_id=str(qid))

        full_recall = evaluate_recall(full_results, qrels, k=top_k)
        full_ndcg   = evaluate_ndcg_at_k(full_results, qrels, k=top_k)

        ablation_output = {
            "full_recall": full_recall,
            "full_ndcg": full_ndcg,
            "details": []
        }

        # 모든 문서 임베딩을 먼저 수집 (한 번만)
        logging.info("[Ablation] Collecting all document embeddings...")
        all_embeddings = []
        for doc_id in self.doc_ids:
            emb = self._get_doc_embeddings(doc_id)
            all_embeddings.append(emb)

        # FDE 차원 계산
        num_partitions = 2 ** self.doc_config.num_simhash_projections
        proj_dim = self.doc_config.projection_dimension if self.doc_config.projection_dimension else self.doc_config.dimension
        final_fde_dim_per_rep = num_partitions * proj_dim
        final_fde_dim = self.doc_config.num_repetitions * final_fde_dim_per_rep

        # (1) Bit 별 실험 (각 bit에 대해 0과 1 두 가지 실험 수행)
        for k in range(b):
            logging.info(f"[Ablation] Testing bit {k}/{b-1} (both 0 and 1)")

            # 각 bit 값(0, 1)에 대해 실험 수행
            for forced_value in [0, 1]:
                logging.info(f"[Ablation] Testing bit {k} forced to {forced_value}")

                # GPU 버전을 사용하여 배치로 한 번에 FDE 생성
                # 임시 memmap 파일 생성
                ablation_memmap_path = os.path.join(self._cache_dir, f"ablation_bit{k}_val{forced_value}.mmap")
                ablation_memmap = np.memmap(ablation_memmap_path, mode="w+", dtype=np.float32,
                                            shape=(len(all_embeddings), final_fde_dim))
                
                # GPU 버전 함수 호출
                gpu_stats = generate_document_fde_batch_gpu_3stage(
                    all_embeddings,
                    self.doc_config,
                    ablation_memmap,
                    batch_start_idx=0,
                    ignore_bit=k,
                    force_bit_value=forced_value,
                    log_every=1000,
                )
                
                # memmap에서 결과 읽기
                ablated_fde = np.array(ablation_memmap)
                
                # 임시 memmap 파일 정리
                del ablation_memmap
                try:
                    os.remove(ablation_memmap_path)
                except Exception as e:
                    logging.warning(f"[Ablation] Failed to clean up {ablation_memmap_path}: {e}")

                # 검색 점수 계산
                results_k = {}
                old_index = self.fde_index
                self.fde_index = ablated_fde  # 임시 교체

                for qid, qtext in queries.items():
                    results_k[str(qid)] = self.search(qtext, query_id=str(qid))

                # 되돌리기
                self.fde_index = old_index

                recall_k = evaluate_recall(results_k, qrels, k=top_k)
                ndcg_k   = evaluate_ndcg_at_k(results_k, qrels, k=top_k)

                ablation_output["details"].append({
                    "bit": k,
                    "forced_value": forced_value,
                    "recall": recall_k,
                    "recall_drop": full_recall - recall_k,
                    "ndcg": ndcg_k,
                    "ndcg_drop": full_ndcg - ndcg_k
                })

                logging.info(
                    f"[Ablation] bit={k}, forced={forced_value}: "
                    f"Recall={recall_k:.4f} (Δ {full_recall - recall_k:.4f}), "
                    f"nDCG={ndcg_k:.4f} (Δ {full_ndcg - ndcg_k:.4f})"
                )

        # (2) 저장
        save_path = os.path.join(self._cache_dir, "bit_ablation_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(ablation_output, f, indent=2, ensure_ascii=False)

        logging.info(f"[Ablation] Saved results → {save_path}")
        
        return ablation_output

    # --------- FAISS 관련 메서드 ---------
    def _set_faiss_threads(self):
        """FAISS OpenMP 스레드 수 설정"""
        if not self.use_faiss_ann:
            return
        try:
            faiss.omp_set_num_threads(self.faiss_num_threads)
        except Exception as e:
            logging.warning(f"[FAISS] omp_set_num_threads failed: {e}")

    def _build_or_load_faiss_index(self):
        """FAISS IVFFlat 인덱스 빌드 또는 로드"""
        if not self.use_faiss_ann:
            return
        if self.faiss_index is not None:
            return
        
        # 이미 존재하는 인덱스 로드 시도
        if os.path.exists(self._faiss_path):
            try:
                self.faiss_index = faiss.read_index(self._faiss_path)
                self.faiss_index.nprobe = self.faiss_nprobe
                if hasattr(self.faiss_index, "d") and self.faiss_index.d != int(self.fde_index.shape[1]):
                    logging.warning(f"[FAISS] dim mismatch: index.d={self.faiss_index.d} vs FDE={self.fde_index.shape[1]} ⇒ rebuild")
                    self.faiss_index = None
                else:
                    logging.info(f"[FAISS] Loaded existing index from {self._faiss_path}")
                    return
            except Exception as e:
                logging.warning(f"[FAISS] load failed: {e}, will rebuild")
                self.faiss_index = None
        
        # 인덱스 빌드
        self._set_faiss_threads()
        dim = int(self.fde_index.shape[1])
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
        logging.info(f"[FAISS] Built and saved index to {self._faiss_path}")

    def ann_search_batch(self, XQ_batch: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """배치 ANN 검색 (FAISS 사용)"""
        assert XQ_batch.ndim == 2
        if self.faiss_index is None:
            self._build_or_load_faiss_index()
        self._set_faiss_threads()
        t0 = time.perf_counter()
        D, I = self.faiss_index.search(XQ_batch, k)
        ann_time = time.perf_counter() - t0
        return D, I, ann_time

    # --------- 경로/키 유틸 ---------
    def _compute_cache_dir(self, dataset: str, model_name: str, cfg) -> str:
        model_key = model_name.replace("/", "_")
        proj_dim = cfg.projection_dimension if cfg.projection_dimension else cfg.dimension
        cfg_str = f"d{proj_dim}_r{cfg.num_repetitions}_p{cfg.num_simhash_projections}_seed{cfg.seed}_fill{int(cfg.fill_empty_partitions)}"
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
        
        # FAISS 로드 & 차원 검증
        if self.use_faiss_ann and os.path.exists(self._faiss_path):
            try:
                self.faiss_index = faiss.read_index(self._faiss_path)
                self.faiss_index.nprobe = self.faiss_nprobe
                if hasattr(self.faiss_index, "d") and self.faiss_index.d != int(self.fde_index.shape[1]):
                    logging.warning(f"[FAISS] dim mismatch: index.d={self.faiss_index.d} vs FDE={self.fde_index.shape[1]} ⇒ rebuild")
                    self.faiss_index = None
                else:
                    logging.info(f"[FAISS] Loaded existing index from {self._faiss_path}")
            except Exception as e:
                logging.warning(f"[FAISS] load failed: {e}")
                self.faiss_index = None
        
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
    def _log_latency(self, qid: str, search_s: float, rerank_s: float, ann_s: Optional[float] = None):
        try:
            if ann_s is not None:
                # ANN 시간이 제공되면 분리하여 로깅
                with open(self._latency_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{qid}\t{ann_s*1000:.3f}\t{rerank_s*1000:.3f}\n")
            else:
                # 기존 형식 (하위 호환성)
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
        ATOMIC_BATCH_SIZE = 3000  # 배치 크기 (메모리 매핑으로 안전하게 처리)
        
        #[1017] simhash별 indice별 원소 개수 csv 파일 저장 필요------------------------------------
        # partition_count 파일 경로 생성 (main script에서 사용할 경로와 동일한 구조)
        proj_suffix = f"_proj{self.projection_dimension}" if self.projection_dimension != 128 else "_proj128"
        simhash_count_dir = os.path.join(QUERY_SEARCH_DIR, f"rep{self.num_repetitions}_simhash{self.num_simhash_projections}_rerank{self.rerank_candidates}{proj_suffix}")
        os.makedirs(simhash_count_dir, exist_ok=True)
        simhash_count_path = os.path.join(simhash_count_dir, "partition_count.csv")
        with open(simhash_count_path, "w", encoding="utf-8") as f:
            f.write("doc_idx,rep_num,partition_idx,count\n")
        #------------------------------------------------------------------------
        
        # FDE 인덱스 초기화 (메모리 매핑으로)
        num_partitions = 2 ** self.doc_config.num_simhash_projections
        final_fde_dim_per_rep = num_partitions * (self.doc_config.projection_dimension or self.doc_config.dimension)
        final_fde_dim = self.doc_config.num_repetitions * final_fde_dim_per_rep
        
        # FDE 인덱스 memmap 생성
        proj_suffix = f"_{self.projection_dimension}" if self.projection_dimension != 128 else ""
        fde_memmap_path = os.path.join(self._cache_dir, f"fde_index_memmap_{self.num_repetitions}_{self.num_simhash_projections}{proj_suffix}.mmap")
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
            
            # Step 3: 배치 FDE 생성 (GPU 버전 사용)
            logging.info(f"[Atomic Batch] Generating FDE for {len(batch_embeddings)} documents using GPU...")
            # GPU 버전은 직접 fde_index memmap에 쓰므로 별도 memmap 불필요
            # GPU 함수는 dict를 반환하므로, 결과를 직접 fde_index에 저장
            gpu_stats = generate_document_fde_batch_gpu_3stage(
                batch_embeddings,
                self.doc_config,
                fde_index,  # 전체 memmap 전달
                batch_start,  # 시작 인덱스
                log_every=ATOMIC_BATCH_SIZE,
            )
            
            # GPU 버전은 memmap에 직접 쓰므로, 결과를 다시 읽어올 필요 없음
            # 대신 batch_fde를 None으로 설정하고, partition_counter는 GPU 버전에서 반환하지 않음
            batch_fde = None  # GPU 버전은 memmap에 직접 쓰므로 불필요
            partition_counter = None  # GPU 버전에서는 partition_counter를 반환하지 않음

            
            # Step 4: GPU 버전은 이미 memmap에 직접 쓰므로 flush만 수행
            logging.info(f"[FDE Integration] GPU batch {batch_start//ATOMIC_BATCH_SIZE + 1} written to memmap")
            
            # Step 5: 배치별 flush (즉시 디스크 저장)
            fde_index.flush()
            
            # Step 6: Simhash 통계 저장 (GPU 버전에서는 partition_counter를 반환하지 않으므로 스킵)
            # GPU 버전에서는 partition_counter를 계산하지 않으므로 이 부분은 주석 처리
            # 필요시 GPU 버전 함수를 수정하여 partition_counter를 반환하도록 할 수 있음
            
            # Step 7: 배치 완료 후 메모리 해제
            del batch_embeddings
            gc.collect()
            
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
        
        # FAISS 인덱스 빌드/로드
        if self.use_faiss_ann and self.faiss_index is None:
            try:
                self._build_or_load_faiss_index()
            except Exception as e:
                logging.warning(f"[FAISS] Build/load skipped: {e}")
        
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
            query_fde_result = generate_query_fde(query_embeddings, query_config)
            
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
            query_fde_result = generate_query_fde(query_embeddings, query_config)
            
            # query_fde_result가 튜플인 경우 첫 번째 요소만 사용
            if isinstance(query_fde_result, tuple):
                query_fde = query_fde_result[0]
            else:
                query_fde = query_fde_result
            
            self._save_query_cache(key, query_embeddings, query_fde)
        else:
            query_embeddings = cached_emb
            query_fde = cached_fde

        # 1-1) ANN 검색 (FAISS 사용) 또는 직접 행렬 곱셈
        ann_time = 0.0
        if self.use_faiss_ann:
            # FAISS 인덱스가 없으면 빌드/로드 시도
            if self.faiss_index is None:
                try:
                    self._build_or_load_faiss_index()
                except Exception as e:
                    logging.warning(f"[FAISS] Build/load failed in search: {e}, falling back to direct search")
            
            if self.faiss_index is not None:
                # FAISS 사용
                start_ann = time.perf_counter()
                k_search = max(self.faiss_candidates, self.rerank_candidates) if self.enable_rerank else TOP_K
                XQ = np.ascontiguousarray(query_fde.reshape(1, -1).astype(np.float32))
                D, I = self.faiss_index.search(XQ, k_search)
                ann_time = time.perf_counter() - start_ann
                
                # FAISS 결과를 기존 형식으로 변환
                mask = I[0] >= 0
                cand_idx = I[0][mask]
                fde_scores = D[0][mask]
                order_fde = cand_idx  # 이미 정렬된 상태
            else:
                # 직접 행렬 곱셈 (fallback)
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
            # FAISS 미사용 시 fde_scores가 이미 계산됨
            if not self.use_faiss_ann or self.faiss_index is None:
                result_dict = OrderedDict((self.doc_ids[i], float(fde_scores[i])) for i in order_fde)
            else:
                # FAISS 사용 시 점수 매핑
                result_dict = OrderedDict()
                for i, idx in enumerate(order_fde):
                    result_dict[self.doc_ids[idx]] = float(fde_scores[i])
            
            # 메트릭 리스트에 추가 (rerank 없음)
            total_search_time = search_time
            avg_search_time_list.append(total_search_time)
            if ann_time > 0:
                avg_ann_time_list.append(ann_time)
            avg_rerank_time_list.append(0.0)
            avg_rerank_cp_list.append(0.0)
            avg_rerank_io_list.append(0.0)
            avg_rerank_wait_list.append(0.0)
            
            self._log_latency(str(query_id) if query_id is not None else "", search_time, rerank_time, ann_s=ann_time if ann_time > 0 else None)
            ann_ms = ann_time * 1000 if ann_time > 0 else 0.0
            logging.info(f"[search] QID={query_id} reranked=0 search_ms={search_time*1000:.3f} rerank_ms=0.000 ann_ms={ann_ms:.3f}")
            return result_dict

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
        
        # Tail 처리: rerank되지 않은 후보들을 FDE 점수로 추가
        if self.use_faiss_ann and self.faiss_index is not None:
            # FAISS 사용 시: 검색된 후보 중 rerank되지 않은 것들만 tail에 추가
            tail = []
            for i, idx in enumerate(order_fde):
                did = self.doc_ids[idx]
                if did not in reranked_ids:
                    tail.append((did, float(fde_scores[i])))
        else:
            # 직접 행렬 곱셈 사용 시: 전체 문서 중 rerank되지 않은 것들 추가
            tail = [(self.doc_ids[i], float(fde_scores[i])) for i in order_fde if self.doc_ids[i] not in reranked_ids]

        out = OrderedDict()
        for did, sc in reranked:
            out[did] = float(sc)
        for did, sc in tail:
            out[did] = sc

        rerank_time = time.perf_counter() - t1
        
        # 메트릭 리스트에 추가
        total_search_time = search_time
        avg_search_time_list.append(total_search_time)
        if ann_time > 0:
            avg_ann_time_list.append(ann_time)
        avg_rerank_time_list.append(rerank_time)
        # rerank compute time과 I/O time 구분이 어려우므로 rerank_time을 compute time으로 사용
        avg_rerank_cp_list.append(rerank_time)
        avg_rerank_io_list.append(0.0)  # 단일 쿼리 처리 구조에서는 I/O time 측정 불가
        avg_rerank_wait_list.append(0.0)  # 배치 구조가 아니므로 wait time 없음
        
        self._log_latency(str(query_id) if query_id is not None else "", search_time, rerank_time, ann_s=ann_time if ann_time > 0 else None)
        
        # 로깅 메시지 구성
        if self.use_faiss_ann and self.faiss_index is not None:
            logging.info(
                f"[search] QID={query_id} reranked={len(cand_ids)} "
                f"search_ms={search_time*1000:.3f} rerank_ms={rerank_time*1000:.3f} ann_ms={ann_time*1000:.3f}"
            )
        else:
            logging.info(
                f"[search] QID={query_id} reranked={len(cand_ids)} "
                f"search_ms={search_time*1000:.3f} rerank_ms={rerank_time*1000:.3f}"
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
    parser.add_argument("--projection", type=int, default=128, help="Projection dimension (default: 128, uses identity projection)")
    args = parser.parse_args()

    nltk.download('punkt', quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        pass

    # 데이터셋 로드
    corpus, queries, qrels = load_nanobeir_dataset(DATASET_REPO_ID)
    
    # 쿼리를 첫 100개로 제한 (1:100)
    queries = dict(list(queries.items())[:200])
    logging.info(f"Limited queries to first 200: {len(queries)} queries.")


    logging.info("Initializing retrieval models...")

    retrievers = {
        "2. ColBERT + FDE (+Chamfer rerank)": ColbertFdeRetriever(
            model_name=COLBERT_MODEL_NAME,
            rerank_candidates=args.rerank,
            enable_rerank=True,
            save_doc_embeds=False,  # 공통 디렉터리에만 저장, 하위 디렉터리 중복 저장 방지
            latency_log_path=os.path.join(QUERY_SEARCH_DIR, f"rep{args.rep}_simhash{args.simhash}_rerank{args.rerank}_proj{args.projection}", "latency.tsv"),  # QID\tSearch\tRerank
            external_doc_embeds_dir=COMMON_DOC_EMBEDS_DIR,  # ★ 공통 문서 임베딩 디렉터리
            num_repetitions=args.rep,
            num_simhash_projections=args.simhash,
            projection_dimension=args.projection,  # ★ projection dimension 설정
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
    
    # 지연시간 로그 파일 초기화 (for 루프 밖에서 정의)
    latency_dir = os.path.join(QUERY_SEARCH_DIR, f"rep{args.rep}_simhash{args.simhash}_rerank{args.rerank}_proj{args.projection}")
    os.makedirs(latency_dir, exist_ok=True)
    with open(os.path.join(latency_dir, "latency.tsv"), "w", encoding="utf-8") as f:
        f.write("QID\tANN\tRerank\n")  # ANN 시간과 Rerank 시간 분리
    
    # 결과 저장 파일 경로 설정
    results_file = os.path.join(latency_dir, "results.txt")
    
    # --- (NEW) ANN 전: 브루트포스 Top-K 진리 생성 & 저장(append, overwrite 금지) [병렬]
    # 첫 번째 retriever를 사용하여 BF 계산
    first_retriever = list(retrievers.values())[0]
    BF_OUTFILE = os.path.join(latency_dir, f"{DATASET_REPO_ID}_bruteforce_top{TOP_K}.tsv")
    compute_and_persist_bf_topk(first_retriever, queries, TOP_K, BF_OUTFILE)
    # 로드
    bf_truth = load_bf_truth(BF_OUTFILE)
    
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

    # 결과 출력 및 파일 저장
    report_lines = []
    report_lines.append("\n" + "=" * 85)
    report_lines.append(f"{'FINAL REPORT':^85}")
    report_lines.append(f"(Dataset: {DATASET_REPO_ID})")
    report_lines.append(f"Parameters: rep={args.rep}, simhash={args.simhash}, rerank={args.rerank}, projection={args.projection}")
    report_lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 85)
    report_lines.append(f"{'Retriever':<30} | {'Indexing Time (s)':<20} | {'Avg Query Time (ms)':<22} | {'Recall@{k}'.format(k=TOP_K):<10} | {'Hit@{k}'.format(k=TOP_K):<10} | {'nDCG@{k}'.format(k=TOP_K):<10}")
    report_lines.append("-" * 120)

    # BF 기준 평가 (첫 번째 retriever 결과 사용)
    first_retriever_name = list(retrievers.keys())[0]
    sys_topk = system_topk_from_results(final_results[first_retriever_name], TOP_K)
    bf_recall = recall_at_k_wrt_bf(sys_topk, bf_truth, TOP_K)
    bf_hit = hit_at_k_wrt_bf(sys_topk, bf_truth, TOP_K)
    bf_ndcg, bf_ndcg_list = ndcg_at_k_wrt_bf(sys_topk, bf_truth, TOP_K)

    for name in retrievers.keys():
        recall = evaluate_recall(final_results[name], qrels, k=TOP_K)
        hit_rate = evaluate_hit_k(final_results[name], qrels, k=TOP_K)
        ndcg = evaluate_ndcg_at_k(final_results[name], qrels, k=TOP_K)
        idx_time = timings[name]["indexing_time"]
        query_time_ms = timings[name]["avg_query_time"] * 1000
        line = f"{name:<30} | {idx_time:<20.2f} | {query_time_ms:<22.2f} | {recall:<10.4f} | {hit_rate:<10.4f} | {ndcg:<10.4f}"
        report_lines.append(line)

    report_lines.append("=" * 120)
    
    # BF 기준 평가 결과 추가
    report_lines.append(f"\nBrute-force (BF) Based Evaluation:")
    report_lines.append(f"- Recall@{TOP_K} (BF): {bf_recall:.6f}")
    report_lines.append(f"- Hit@{TOP_K} (BF): {bf_hit:.6f}")
    report_lines.append(f"- nDCG@{TOP_K} (BF): {bf_ndcg:.6f}")
    
    # 추가 통계 정보
    report_lines.append(f"\nAdditional Statistics:")
    report_lines.append(f"- Total queries processed: {len(queries)}")
    report_lines.append(f"- Total documents: {len(corpus)}")
    report_lines.append(f"- Device used: {DEVICE}")
    
    # 메트릭 리스트 기반 성능 통계
    if avg_search_time_list:
        report_lines.append(f"\nPerformance Metrics (from metric lists):")
        if avg_ann_time_list:
            report_lines.append(f"- Average ANN time: {mean(avg_ann_time_list)*1000:.2f} ms")
        if avg_rerank_time_list:
            report_lines.append(f"- Average Rerank time: {mean(avg_rerank_time_list)*1000:.2f} ms")
        if avg_rerank_cp_list:
            report_lines.append(f"- Average Rerank compute time: {mean(avg_rerank_cp_list)*1000:.2f} ms")
        if avg_rerank_io_list:
            report_lines.append(f"- Average Rerank I/O time: {mean(avg_rerank_io_list)*1000:.2f} ms")
        if avg_rerank_wait_list:
            report_lines.append(f"- Average Rerank wait time: {mean(avg_rerank_wait_list)*1000:.2f} ms")
        if avg_search_time_list:
            total_search_s = mean(avg_ann_time_list) + mean(avg_rerank_time_list) if avg_ann_time_list and avg_rerank_time_list else mean(avg_search_time_list)
            report_lines.append(f"- Total search time (ANN + Rerank): {total_search_s*1000:.2f} ms")
    
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
    
    # BF 기준 per-query nDCG 저장
    # ndcg_at_k_wrt_bf는 sys_topk.items()의 순서대로 per_q 리스트를 생성하므로,
    # sys_topk.items()의 순서와 일치하도록 매핑 생성
    per_bf_ndcg_file = os.path.join(latency_dir, f"per_bf_ndcg_{TOP_K}.tsv")
    try:
        # sys_topk.items()의 순서대로 bf_ndcg_list를 매핑
        bf_ndcg_dict = {}
        qid_list = list(sys_topk.keys())  # sys_topk.items()의 순서 유지
        for i, qid in enumerate(qid_list):
            if i < len(bf_ndcg_list):
                bf_ndcg_dict[qid] = bf_ndcg_list[i]
            else:
                bf_ndcg_dict[qid] = 0.0
        
        with open(per_bf_ndcg_file, "w", encoding="utf-8") as f:
            f.write("qid\tbf_ndcg_at_k\n")
            for qid in sorted(bf_ndcg_dict.keys()):  # 정렬하여 저장
                f.write(f"{qid}\t{bf_ndcg_dict[qid]:.6f}\n")
        logging.info(f"Per-query BF nDCG saved to: {per_bf_ndcg_file}")
    except Exception as e:
        logging.warning(f"Failed to save per-query BF nDCG: {e}")
    
    # 콘솔에 출력
    for line in report_lines:
        print(line)
    
    # 파일에 저장
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    logging.info(f"Results saved to: {results_file}")

    # Run Bit Ablation
    logging.info("--- PHASE 3: BIT ABLATION ---")
    retriever = list(retrievers.values())[0]
    ablation_results = retriever.run_bit_ablation(corpus, queries, qrels, top_k=TOP_K)
    
    # Bit Ablation 결과를 읽기 쉬운 형식으로 저장
    ablation_dir = os.path.join(QUERY_SEARCH_DIR, f"rep{args.rep}_simhash{args.simhash}_rerank{args.rerank}_proj{args.projection}")
    os.makedirs(ablation_dir, exist_ok=True)
    
    # TSV 형식으로 저장
    ablation_tsv_path = os.path.join(ablation_dir, "bit_ablation_results.tsv")
    with open(ablation_tsv_path, "w", encoding="utf-8") as f:
        f.write("Bit\tForced_Value\tRecall@{}\tnDCG@{}\tRecall_Drop\tnDCG_Drop\n".format(TOP_K, TOP_K))
        f.write("Full\t-\t{:.6f}\t{:.6f}\t0.000000\t0.000000\n".format(
            ablation_results["full_recall"], 
            ablation_results["full_ndcg"]
        ))
        for detail in ablation_results["details"]:
            f.write("{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(
                detail["bit"],
                detail["forced_value"],
                detail["recall"],
                detail["ndcg"],
                detail["recall_drop"],
                detail["ndcg_drop"]
            ))
    logging.info(f"Bit ablation TSV saved to: {ablation_tsv_path}")
    
    # 텍스트 리포트 형식으로 저장
    ablation_report_path = os.path.join(ablation_dir, "bit_ablation_report.txt")
    with open(ablation_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"{'BIT ABLATION RESULTS':^100}\n")
        f.write("=" * 100 + "\n")
        f.write(f"Dataset: {DATASET_REPO_ID}\n")
        f.write(f"Parameters: rep={args.rep}, simhash={args.simhash}, rerank={args.rerank}, projection={args.projection}\n")
        f.write(f"Top-K: {TOP_K}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Full (All bits):\n")
        f.write(f"  Recall@{TOP_K}: {ablation_results['full_recall']:.6f}\n")
        f.write(f"  nDCG@{TOP_K}: {ablation_results['full_ndcg']:.6f}\n\n")
        
        f.write(f"{'Bit':<10} {'Forced':<10} {'Recall@{k}':<15} {'nDCG@{k}':<15} {'Recall Drop':<15} {'nDCG Drop':<15}\n".format(k=TOP_K))
        f.write("-" * 100 + "\n")
        for detail in ablation_results["details"]:
            f.write(f"{detail['bit']:<10} {detail['forced_value']:<10} {detail['recall']:<15.6f} {detail['ndcg']:<15.6f} "
                   f"{detail['recall_drop']:<15.6f} {detail['ndcg_drop']:<15.6f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("Summary:\n")
        f.write(f"  Overall Average Recall Drop: {np.mean([d['recall_drop'] for d in ablation_results['details']]):.6f}\n")
        f.write(f"  Overall Average nDCG Drop: {np.mean([d['ndcg_drop'] for d in ablation_results['details']]):.6f}\n")
        
        # Forced value별 통계
        for forced_val in [0, 1]:
            forced_details = [d for d in ablation_results['details'] if d['forced_value'] == forced_val]
            if forced_details:
                f.write(f"\n  Forced Value = {forced_val}:\n")
                f.write(f"    Average Recall Drop: {np.mean([d['recall_drop'] for d in forced_details]):.6f}\n")
                f.write(f"    Average nDCG Drop: {np.mean([d['ndcg_drop'] for d in forced_details]):.6f}\n")
                max_recall_drop = max(forced_details, key=lambda x: x['recall_drop'])
                max_ndcg_drop = max(forced_details, key=lambda x: x['ndcg_drop'])
                f.write(f"    Max Recall Drop: {max_recall_drop['recall_drop']:.6f} (bit {max_recall_drop['bit']})\n")
                f.write(f"    Max nDCG Drop: {max_ndcg_drop['ndcg_drop']:.6f} (bit {max_ndcg_drop['bit']})\n")
        
        f.write("=" * 100 + "\n")
    
    logging.info(f"Bit ablation report saved to: {ablation_report_path}")

    # Run Repetition Ablation
    logging.info("--- PHASE 4: REPETITION ABLATION ---")
    retriever = list(retrievers.values())[0]
    
    # FDE 차원 구조 계산
    num_partitions_per_rep = 2 ** args.simhash
    partition_size = retriever.doc_config.dimension  # 기본 dimension (projection_dimension이 있으면 그것 사용)
    if hasattr(retriever.doc_config, 'projection_dimension') and retriever.doc_config.projection_dimension:
        partition_size = retriever.doc_config.projection_dimension
    repetition_size = partition_size * num_partitions_per_rep
    
    logging.info(f"✅ [INFO] repetition 개수: {args.rep}, 공간 분할 함수 개수: {args.simhash}")
    logging.info(f"✅ [INFO] 각 repetition당 partition 개수: {num_partitions_per_rep}, 각 partition 크기: {partition_size}")
    logging.info(f"✅ [INFO] 각 repetition당 공간 크기: {repetition_size}")
    
    # Repetition별 results 초기화
    results_repetition = {}
    
    # 각 repetition을 개별적으로 분석: 0번만 -> 1번만 -> 2번만 -> ... -> (rep-1)번만
    for rep_idx in range(args.rep):
        # 각 repetition의 차원 범위 계산
        dim_start = rep_idx * repetition_size
        dim_end = (rep_idx + 1) * repetition_size
        
        logging.info(f"✅ [INFO] Repetition {rep_idx} (개별 분석):")
        logging.info(f"   - 포함된 repetition: {rep_idx}번만 (1개)")
        logging.info(f"   - 차원 범위: [{dim_start}, {dim_end})")
        logging.info(f"   - 검색 수행 중...")
        
        # rep_idx번 repetition만 포함하는 차원들만 선택
        dims_to_keep = list(range(dim_start, dim_end))
        
        # 문서 FDE에서 해당 repetition만 선택
        truncated_fde_index = retriever.fde_index[:, dims_to_keep]
        
        # 검색 수행 (임시로 fde_index 교체)
        old_index = retriever.fde_index
        retriever.fde_index = truncated_fde_index
        
        # 모든 쿼리에 대해 검색 수행
        rep_results = {}
        for qid, qtext in queries.items():
            # Query FDE 생성 (전체 차원으로 생성 후 슬라이싱)
            key = retriever._query_key(qtext, str(qid))
            cached_emb, cached_fde = retriever._load_query_cache(key)
            
            if cached_emb is None or cached_fde is None:
                query_embeddings_map = retriever.ranker.encode_queries(queries=[qtext])
                query_embeddings = to_numpy(next(iter(query_embeddings_map.values())))
                query_config = replace(retriever.doc_config, fill_empty_partitions=False)
                query_fde_result = generate_query_fde(query_embeddings, query_config)
                
                if isinstance(query_fde_result, tuple):
                    query_fde = query_fde_result[0]
                else:
                    query_fde = query_fde_result
            else:
                query_fde = cached_fde
            
            # Query FDE도 동일한 차원만 선택
            truncated_query_fde = query_fde[dims_to_keep]
            
            # 검색 점수 계산
            fde_scores = truncated_fde_index @ truncated_query_fde
            order_fde = np.argsort(-fde_scores)
            
            # 결과 저장
            rep_results[str(qid)] = OrderedDict((retriever.doc_ids[i], float(fde_scores[i])) for i in order_fde)
        
        # 되돌리기
        retriever.fde_index = old_index
        
        # 평가
        recall_rep = evaluate_recall(rep_results, qrels, k=TOP_K)
        ndcg_rep = evaluate_ndcg_at_k(rep_results, qrels, k=TOP_K)
        
        # 결과 저장
        key = f"rep_{rep_idx}_only"
        results_repetition[key] = {
            "rep_idx": rep_idx,
            "included_repetitions": 1,
            "dim_range": [dim_start, dim_end],
            "recall": recall_rep,
            "ndcg": ndcg_rep
        }
        
        logging.info(f"✅ [INFO] {DATASET_REPO_ID}, rep {rep_idx}만 완료")
        logging.info(f"   - Recall@{TOP_K}: {recall_rep:.6f}, nDCG@{TOP_K}: {ndcg_rep:.6f}")
    
    # Repetition ablation 결과 저장
    repetition_dir = os.path.join(QUERY_SEARCH_DIR, f"rep{args.rep}_simhash{args.simhash}_rerank{args.rerank}_proj{args.projection}")
    os.makedirs(repetition_dir, exist_ok=True)
    
    # JSON 형식으로 저장
    repetition_json_path = os.path.join(repetition_dir, "repetition_ablation_results.json")
    with open(repetition_json_path, "w", encoding="utf-8") as f:
        json.dump(results_repetition, f, indent=2, ensure_ascii=False)
    logging.info(f"Repetition ablation JSON saved to: {repetition_json_path}")
    
    # TSV 형식으로 저장
    repetition_tsv_path = os.path.join(repetition_dir, "repetition_ablation_results.tsv")
    with open(repetition_tsv_path, "w", encoding="utf-8") as f:
        f.write("Repetition\tIncluded_Reps\tDim_Range_Start\tDim_Range_End\tRecall@{}\tnDCG@{}\n".format(TOP_K, TOP_K))
        for key, result in sorted(results_repetition.items(), key=lambda x: x[1]['rep_idx']):
            f.write(f"{result['rep_idx']}" + "\t" +
                   f"{result['included_repetitions']}" + "\t" +
                   f"{result['dim_range'][0]}" + "\t" +
                   f"{result['dim_range'][1]}" + "\t" +
                   f"{result['recall']:.6f}" + "\t" +
                   f"{result['ndcg']:.6f}\n")
    logging.info(f"Repetition ablation TSV saved to: {repetition_tsv_path}")
    
    # 텍스트 리포트 형식으로 저장
    repetition_report_path = os.path.join(repetition_dir, "repetition_ablation_report.txt")
    with open(repetition_report_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"{'REPETITION ABLATION RESULTS':^100}\n")
        f.write("=" * 100 + "\n")
        f.write(f"Dataset: {DATASET_REPO_ID}\n")
        f.write(f"Parameters: rep={args.rep}, simhash={args.simhash}, rerank={args.rerank}, projection={args.projection}\n")
        f.write(f"Top-K: {TOP_K}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'Repetition':<15} {'Included':<15} {'Dim Range':<20} {'Recall@{k}':<15} {'nDCG@{k}':<15}\n".format(k=TOP_K))
        f.write("-" * 100 + "\n")
        for key, result in sorted(results_repetition.items(), key=lambda x: x[1]['rep_idx']):
            f.write(f"{result['rep_idx']:<15} {result['included_repetitions']:<15} "
                   f"[{result['dim_range'][0]}, {result['dim_range'][1]}){'':<10} "
                   f"{result['recall']:<15.6f} {result['ndcg']:<15.6f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("Summary:\n")
        recalls = [r['recall'] for r in results_repetition.values()]
        ndcgs = [r['ndcg'] for r in results_repetition.values()]
        best_recall_rep = max(results_repetition.items(), key=lambda x: x[1]['recall'])[1]['rep_idx']
        best_ndcg_rep = max(results_repetition.items(), key=lambda x: x[1]['ndcg'])[1]['rep_idx']
        f.write(f"  Best Recall@{TOP_K}: {max(recalls):.6f} (rep {best_recall_rep} only)\n")
        f.write(f"  Best nDCG@{TOP_K}: {max(ndcgs):.6f} (rep {best_ndcg_rep} only)\n")
        f.write(f"  Average Recall@{TOP_K}: {np.mean(recalls):.6f}\n")
        f.write(f"  Average nDCG@{TOP_K}: {np.mean(ndcgs):.6f}\n")
        f.write(f"  Std Recall@{TOP_K}: {np.std(recalls):.6f}\n")
        f.write(f"  Std nDCG@{TOP_K}: {np.std(ndcgs):.6f}\n")
        f.write("=" * 100 + "\n")
    
    logging.info(f"Repetition ablation report saved to: {repetition_report_path}")
    
    # Top N Repetition 조합 실험 (20%, 40%, 60%, 80%만)
    logging.info("--- PHASE 5: TOP N REPETITION COMBINATION (20%, 40%, 60%, 80% of total) ---")
    
    # 전체 repetition 개수
    total_reps = len(results_repetition)
    
    # 테스트할 percentile 목록
    percentiles = [20, 40, 60, 80]
    num_reps_list = [max(1, int(total_reps * pct / 100)) for pct in percentiles]
    
    logging.info(f"✅ [INFO] Total repetitions: {total_reps}")
    logging.info(f"✅ [INFO] Testing top {percentiles}% repetitions: {num_reps_list}")
    
    # Recall 기준으로 정렬
    sorted_by_recall = sorted(results_repetition.items(), key=lambda x: x[1]['recall'], reverse=True)
    
    # nDCG 기준으로 정렬
    sorted_by_ndcg = sorted(results_repetition.items(), key=lambda x: x[1]['ndcg'], reverse=True)
    
    # 개수별 결과 저장
    top_n_results = {}
    
    # 각 percentile에 대해 실험
    for percentile, num_reps in zip(percentiles, num_reps_list):
        
        # Recall 기준 Top N개
        top_n_recall_reps = [item[1]['rep_idx'] for item in sorted_by_recall[:num_reps]]
        
        # nDCG 기준 Top N개
        top_n_ndcg_reps = [item[1]['rep_idx'] for item in sorted_by_ndcg[:num_reps]]
        
        logging.info(f"✅ [INFO] Testing Top {num_reps} repetitions (out of {total_reps} total, {percentile}%)")
        
        # Recall 기준 실험
        top_n_recall_dims = []
        for rep_idx in sorted(top_n_recall_reps):
            dim_start = rep_idx * repetition_size
            dim_end = (rep_idx + 1) * repetition_size
            top_n_recall_dims.extend(range(dim_start, dim_end))
        
        top_n_recall_dims = sorted(top_n_recall_dims)
        truncated_fde_index_recall = retriever.fde_index[:, top_n_recall_dims]
        
        old_index = retriever.fde_index
        retriever.fde_index = truncated_fde_index_recall
        
        top_n_recall_results = {}
        for qid, qtext in queries.items():
            key = retriever._query_key(qtext, str(qid))
            cached_emb, cached_fde = retriever._load_query_cache(key)
            
            if cached_emb is None or cached_fde is None:
                query_embeddings_map = retriever.ranker.encode_queries(queries=[qtext])
                query_embeddings = to_numpy(next(iter(query_embeddings_map.values())))
                query_config = replace(retriever.doc_config, fill_empty_partitions=False)
                query_fde_result = generate_query_fde(query_embeddings, query_config)
                
                if isinstance(query_fde_result, tuple):
                    query_fde = query_fde_result[0]
                else:
                    query_fde = query_fde_result
            else:
                query_fde = cached_fde
            
            truncated_query_fde = query_fde[top_n_recall_dims]
            fde_scores = truncated_fde_index_recall @ truncated_query_fde
            order_fde = np.argsort(-fde_scores)
            top_n_recall_results[str(qid)] = OrderedDict((retriever.doc_ids[i], float(fde_scores[i])) for i in order_fde)
        
        retriever.fde_index = old_index
        
        recall_top_n = evaluate_recall(top_n_recall_results, qrels, k=TOP_K)
        ndcg_top_n_recall = evaluate_ndcg_at_k(top_n_recall_results, qrels, k=TOP_K)
        
        # nDCG 기준 실험
        top_n_ndcg_dims = []
        for rep_idx in sorted(top_n_ndcg_reps):
            dim_start = rep_idx * repetition_size
            dim_end = (rep_idx + 1) * repetition_size
            top_n_ndcg_dims.extend(range(dim_start, dim_end))
        
        top_n_ndcg_dims = sorted(top_n_ndcg_dims)
        truncated_fde_index_ndcg = retriever.fde_index[:, top_n_ndcg_dims]
        
        retriever.fde_index = truncated_fde_index_ndcg
        
        top_n_ndcg_results = {}
        for qid, qtext in queries.items():
            key = retriever._query_key(qtext, str(qid))
            cached_emb, cached_fde = retriever._load_query_cache(key)
            
            if cached_emb is None or cached_fde is None:
                query_embeddings_map = retriever.ranker.encode_queries(queries=[qtext])
                query_embeddings = to_numpy(next(iter(query_embeddings_map.values())))
                query_config = replace(retriever.doc_config, fill_empty_partitions=False)
                query_fde_result = generate_query_fde(query_embeddings, query_config)
                
                if isinstance(query_fde_result, tuple):
                    query_fde = query_fde_result[0]
                else:
                    query_fde = query_fde_result
            else:
                query_fde = cached_fde
            
            truncated_query_fde = query_fde[top_n_ndcg_dims]
            fde_scores = truncated_fde_index_ndcg @ truncated_query_fde
            order_fde = np.argsort(-fde_scores)
            top_n_ndcg_results[str(qid)] = OrderedDict((retriever.doc_ids[i], float(fde_scores[i])) for i in order_fde)
        
        retriever.fde_index = old_index
        
        recall_top_n_ndcg = evaluate_recall(top_n_ndcg_results, qrels, k=TOP_K)
        ndcg_top_n_ndcg = evaluate_ndcg_at_k(top_n_ndcg_results, qrels, k=TOP_K)
        
        # 결과 저장
        top_n_results[f"top{percentile}pct"] = {
            "num_repetitions": num_reps,
            "percentile": percentile,
            "recall_based": {
                "selected_repetitions": sorted(top_n_recall_reps),
                "dim_range": [min(top_n_recall_dims), max(top_n_recall_dims) + 1] if top_n_recall_dims else [0, 0],
                "recall": recall_top_n,
                "ndcg": ndcg_top_n_recall
            },
            "ndcg_based": {
                "selected_repetitions": sorted(top_n_ndcg_reps),
                "dim_range": [min(top_n_ndcg_dims), max(top_n_ndcg_dims) + 1] if top_n_ndcg_dims else [0, 0],
                "recall": recall_top_n_ndcg,
                "ndcg": ndcg_top_n_ndcg
            }
        }
        
        logging.info(f"   - Recall-based: Recall@{TOP_K}={recall_top_n:.6f}, nDCG@{TOP_K}={ndcg_top_n_recall:.6f}")
        logging.info(f"   - nDCG-based: Recall@{TOP_K}={recall_top_n_ndcg:.6f}, nDCG@{TOP_K}={ndcg_top_n_ndcg:.6f}")
    
    # Top N별 결과 저장
    top_n_json_path = os.path.join(repetition_dir, "top_n_combination_results.json")
    with open(top_n_json_path, "w", encoding="utf-8") as f:
        json.dump(top_n_results, f, indent=2, ensure_ascii=False)
    logging.info(f"Top N combination JSON saved to: {top_n_json_path}")
    
    # TSV 형식으로도 저장
    top_n_tsv_path = os.path.join(repetition_dir, "top_n_combination_results.tsv")
    with open(top_n_tsv_path, "w", encoding="utf-8") as f:
        f.write("Num_Reps\tPercentile\tMetric_Based\tRecall@{}\tnDCG@{}\n".format(TOP_K, TOP_K))
        for key, result in sorted(top_n_results.items(), key=lambda x: x[1]['num_repetitions']):
            num_reps = result['num_repetitions']
            pct = result['percentile']
            f.write(f"{num_reps}\t{pct:.2f}\tRecall\t{result['recall_based']['recall']:.6f}\t{result['recall_based']['ndcg']:.6f}\n")
            f.write(f"{num_reps}\t{pct:.2f}\tnDCG\t{result['ndcg_based']['recall']:.6f}\t{result['ndcg_based']['ndcg']:.6f}\n")
    logging.info(f"Top N combination TSV saved to: {top_n_tsv_path}")
    
    # 리포트 업데이트 (정렬된 repetition 목록과 Percentile 조합 결과 추가)
    with open(repetition_report_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"{'REPETITION RANKING':^100}\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("Recall 기준 순위:\n")
        f.write(f"{'Rank':<10} {'Repetition':<15} {'Recall@{k}':<15} {'nDCG@{k}':<15}\n".format(k=TOP_K))
        f.write("-" * 100 + "\n")
        for rank, (key, result) in enumerate(sorted_by_recall, 1):
            f.write(f"{rank:<10} {result['rep_idx']:<15} {result['recall']:<15.6f} {result['ndcg']:<15.6f}\n")
        
        f.write("\nnDCG 기준 순위:\n")
        f.write(f"{'Rank':<10} {'Repetition':<15} {'Recall@{k}':<15} {'nDCG@{k}':<15}\n".format(k=TOP_K))
        f.write("-" * 100 + "\n")
        for rank, (key, result) in enumerate(sorted_by_ndcg, 1):
            f.write(f"{rank:<10} {result['rep_idx']:<15} {result['recall']:<15.6f} {result['ndcg']:<15.6f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"{'TOP N COMBINATION RESULTS (20%, 40%, 60%, 80% of total)':^100}\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'Num_Reps':<12} {'Percentile':<12} {'Metric':<12} {'Recall@{k}':<15} {'nDCG@{k}':<15}\n".format(k=TOP_K))
        f.write("-" * 100 + "\n")
        for key, result in sorted(top_n_results.items(), key=lambda x: x[1]['num_repetitions']):
            num_reps = result['num_repetitions']
            pct = result['percentile']
            f.write(f"{num_reps:<12} {pct:.2f}%{'':<7} {'Recall':<12} {result['recall_based']['recall']:<15.6f} {result['recall_based']['ndcg']:<15.6f}\n")
            f.write(f"{'':<12} {pct:.2f}%{'':<7} {'nDCG':<12} {result['ndcg_based']['recall']:<15.6f} {result['ndcg_based']['ndcg']:<15.6f}\n")
        
        f.write("=" * 100 + "\n")
    
    logging.info(f"✅ [INFO] 모든 결과 저장 완료:")
    logging.info(f"   - Repetition별 결과: {repetition_json_path}")
    logging.info(f"   - Top N 조합 결과: {top_n_json_path}")
    logging.info(f"   - Top N 조합 TSV: {top_n_tsv_path}")

