# -*- coding: utf-8 -*-
import os, json, time, hashlib, logging, pathlib
from collections import OrderedDict
from dataclasses import replace
from typing import Optional, List, Tuple

import nltk
import numpy as np
import torch
import joblib

import neural_cherche.models as neural_cherche_models
import neural_cherche.rank as neural_cherche_rank

from beir import util
from beir.datasets.data_loader import GenericDataLoader

# FDE 구현 (스트리밍 + 배치 병렬)
from fde_generator_optimized_stream import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_document_fde_batch,
)

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
TOP_K = 10

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# 데이터셋 준비 
dataset = "scidocs"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

# 캐시 루트
CACHE_ROOT = os.path.join(pathlib.Path(__file__).parent.absolute(), "cache_muvera")
os.makedirs(CACHE_ROOT, exist_ok=True)

# 쿼리 검색 디렉터리
QUERY_SEARCH_DIR = os.path.join(CACHE_ROOT, "query_search", dataset, "fde_ivfip")
os.makedirs(QUERY_SEARCH_DIR, exist_ok=True)

# ======================
# --- Logging Setup ----
# ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info(f"Using device: {DEVICE}  |  FAISS={'on' if _FAISS_OK else 'off'}")

# ===========================
# --- Helper Functions  -----
# ===========================
def load_nanobeir_dataset(repo_id: str):
    """Loads BEIR dataset from local 'data_path' in test split."""
    # 데이터셋 준비 
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
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().detach().numpy().astype(np.float32)
    elif isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    else:
        raise TypeError(f"Unsupported type for conversion: {type(tensor_or_array)}")

# =====================================
# --- ColBERT + FDE + FAISS + Rerank ---
# =====================================
class ColbertFdeRetriever:
    """
    1) ColBERT 토큰 임베딩 → 문서 FDE 인덱스 생성(디스크 캐시)
    2) (옵션) FAISS IVFFlat(IP)로 ANN 검색 → 상위 M 후보
    3) 상위 후보 N개를 Chamfer(MaxSim)로 재랭킹
    """

    def __init__(
        self,
        model_name: str = COLBERT_MODEL_NAME,
        rerank_candidates: int = 100,
        enable_rerank: bool = True,
        save_doc_embeds: bool = True,
        latency_log_path: Optional[str] = None,
        external_doc_embeds_dir: Optional[str] = None,

        # ----- FAISS params -----
        use_faiss_ann: bool = True,
        faiss_nlist: int = 10000,
        faiss_nprobe: int = 10,
        faiss_candidates: int = 1000,   # ANN에서 몇 개를 받아 재랭크 소스로 쓸지

        # ----- NEW: threading -----
        faiss_num_threads: Optional[int] = None,
    ):
        self.faiss_num_threads = faiss_num_threads or max(1, (os.cpu_count() or 1) - 0)
        logging.info(f"Using {self.faiss_num_threads} threads for FAISS")

        model = neural_cherche_models.ColBERT(model_name_or_path=model_name, device=DEVICE)
        self.ranker = neural_cherche_rank.ColBERT(key="id", on=["title", "text"], model=model)

        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=128,
            num_repetitions=2,
            num_simhash_projections=7,
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
        self.faiss_index = None  # faiss.IndexIVFFlat

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
        self._faiss_path = os.path.join(self._cache_dir, f"ivf{self.faiss_nlist}_ip.faiss")

        os.makedirs(self._cache_dir, exist_ok=True)
        os.makedirs(self._queries_dir, exist_ok=True)
        if self.save_doc_embeds:
            os.makedirs(self._doc_emb_dir, exist_ok=True)

        self._latency_log_path = latency_log_path or os.path.join(self._cache_dir, "latency.tsv")
    
    def _set_faiss_threads(self):
        # OpenMP 스레드 (Faiss KMeans/train, add/search에 모두 영향)
        try:
            # print(f"omp_set_num_threads: {self.faiss_num_threads}")
            faiss.omp_set_num_threads(self.faiss_num_threads)
            logging.info(f"[FAISS] omp_set_num_threads({self.faiss_num_threads})")
        except Exception as e:
            logging.warning(f"[FAISS] omp_set_num_threads failed: {e}")
        
        # OPENBLAS/LAPACK 류 환경변수(런타임 변경이 안 먹을 수도 있으니, 가능하면 실행 초기에 세팅)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(self.faiss_num_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(self.faiss_num_threads))
        os.environ.setdefault("OMP_NUM_THREADS", str(self.faiss_num_threads))

    # --------- 경로/키 유틸 ---------
    def _compute_cache_dir(self, dataset: str, model_name: str, cfg) -> str:
        # 간단히 dataset 이름으로 폴더 고정 (사용자 코드 유지)
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
        self.fde_index = joblib.load(self._fde_path)
        with open(self._ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        self._doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        logging.info(
            f"[{self.__class__.__name__}] Loaded FDE index cache: "
            f"{self.fde_index.shape} for {len(self.doc_ids)} docs"
        )
        # FAISS index가 있으면 로드 시도
        if self.use_faiss_ann and os.path.exists(self._faiss_path):
            try:
                self.faiss_index = faiss.read_index(self._faiss_path)
                self.faiss_index.nprobe = self.faiss_nprobe
                logging.info(f"[FAISS] Loaded index from {self._faiss_path} (nprobe={self.faiss_nprobe})")
            except Exception as e:
                logging.warning(f"[FAISS] Failed to load index: {e}")
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

    # --------- Chamfer(MaxSim) ---------
    @staticmethod
    def _chamfer(query_tok: np.ndarray, doc_tok: np.ndarray) -> float:
        sim = query_tok @ doc_tok.T  # [m, n]
        return float(sim.max(axis=1).sum())

    def _get_doc_embeddings(self, doc_id: str, allow_build: bool = True) -> np.ndarray:
        ext_path = self._external_doc_emb_path(doc_id)
        if ext_path and os.path.exists(ext_path):
            return np.load(ext_path)
        int_path = self._doc_emb_path(doc_id)
        if os.path.exists(int_path):
            return np.load(int_path)
        if not allow_build:
            raise FileNotFoundError(ext_path or int_path)
        if self._corpus is None:
            raise RuntimeError("Corpus not set; cannot build document embeddings on the fly.")
        doc = {"id": doc_id, **self._corpus[doc_id]}
        emap = self.ranker.encode_documents(documents=[doc])
        arr = to_numpy(emap[doc_id])
        np.save(int_path, arr)
        return arr

    # --------- Latency log ---------
    def _log_latency(self, qid: str, search_s: float, rerank_s: float):
        try:
            with open(self._latency_log_path, "a", encoding="utf-8") as f:
                f.write(f"{qid}\t{search_s*1000:.3f}\t{rerank_s*1000:.3f}\n")
        except Exception as e:
            logging.warning(f"[{self.__class__.__name__}] Failed to write latency log: {e}")
    
    def _build_or_load_faiss_index(self):
        if not self.use_faiss_ann:
            return
        if self.faiss_index is not None and os.path.exists(self._faiss_path):
            return
        
        # >>> NEW
        self._set_faiss_threads()
        # <<<

        dim   = int(self.fde_index.shape[1])
        nvecs = int(self.fde_index.shape[0])

        logging.info(f"[FAISS] Building IVFFlat(IP) nlist={self.faiss_nlist} for {nvecs} vectors (dim={dim})")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, self.faiss_nlist, faiss.METRIC_INNER_PRODUCT)

        # -------------------------
        # A) TRAIN: 복사 없이 전체 사용 시도 → 실패하면 스트리밍 균등 샘플
        # -------------------------
        try:
            x_train = self.fde_index
            if x_train.dtype != np.float32 or not x_train.flags["C_CONTIGUOUS"]:
                raise RuntimeError("Training would copy; fallback to streaming sampler.")

            t0 = time.perf_counter()
            index.train(x_train)   # 메모리맵 포인터로 직접 훈련 (복사 없음)
            logging.info(f"[FAISS] Trained on ALL ({nvecs}) vectors in {time.perf_counter()-t0:.2f}s (no copy).")
        except Exception as e:
            logging.warning(f"[FAISS] Train on ALL failed or would copy ({e}); switching to streaming sampling.")

            # 스트리밍 균등 샘플 크기: 최소 100*nlist, 하한 50k, 상한은 메모리 예산 기반
            min_per_list = 100
            target_train = max(self.faiss_nlist * min_per_list, 50_000)
            train_size = min(nvecs, target_train)

            # 메모리 예산 (필요시 조절)
            max_bytes = 1_000_000_000  # ≈ 1GB
            est_bytes = train_size * dim * 4
            if est_bytes > max_bytes:
                train_size = max(min(int(max_bytes // (dim * 4)), nvecs), self.faiss_nlist * min_per_list)
                logging.info(f"[FAISS] Shrink train_size to {train_size} for memory budget.")

            rng = np.random.default_rng(42)
            train_x = np.empty((train_size, dim), dtype=np.float32, order="C")

            filled = 0
            block = 65536  # 블록 단위 순회
            for start in range(0, nvecs, block):
                end = min(start + block, nvecs)
                Xblk = self.fde_index[start:end]  # memmap view (복사 없음)
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
                logging.info(f"[FAISS] Underfilled ({filled}/{train_size}); topping up.")
                need = train_size - filled
                step = max(1, nvecs // need)
                pos = 0
                for i in range(need):
                    train_x[filled + i] = self.fde_index[pos]
                    pos = (pos + step) % nvecs

            t0 = time.perf_counter()
            index.train(train_x)
            logging.info(f"[FAISS] Trained on {train_size} sampled vectors in {time.perf_counter()-t0:.2f}s")

        # ------------------------------------------------------------------
        # B) (옵션) On-disk inverted lists: 벡터 코드 저장을 디스크로 이동 (RAM 절감)
        # ------------------------------------------------------------------
        use_ondisk = hasattr(faiss, "OnDiskInvertedLists")
        if use_ondisk:
            try:
                ivf = faiss.extract_index_ivf(index)
                invlists_path = self._faiss_path + ".lists"  # 실제 코드 데이터 파일
                invlists = faiss.OnDiskInvertedLists(ivf.nlist, ivf.code_size, invlists_path)
                ivf.replace_invlists(invlists)
                logging.info(f"[FAISS] Using On-Disk inverted lists at: {invlists_path}")
            except Exception as e:
                use_ondisk = False
                logging.warning(f"[FAISS] On-Disk inverted lists not available/failed: {e}")

        # ------------------------------------
        # C) ADD: 큰 배열을 나눠서 배치 추가 (피크 메모리 ↓)
        # ------------------------------------
        add_bs = 100_000  # 배치 크기 (메모리 상황에 맞게 조절)
        t_add = time.perf_counter()
        added = 0
        for start in range(0, nvecs, add_bs):
            end = min(start + add_bs, nvecs)
            # 중요: 배치만큼만 연속 버퍼로 보장 (필요시 그 배치만 복사)
            xb = np.ascontiguousarray(self.fde_index[start:end], dtype=np.float32)
            index.add(xb)
            added += (end - start)
            if added % 200_000 == 0:
                logging.info(f"[FAISS] Added {added}/{nvecs} vectors...")

        logging.info(f"[FAISS] Added all {nvecs} vectors in {time.perf_counter()-t_add:.2f}s")

        # On-disk inverted lists를 썼다면, 아래 write_index는 메타/헤더만 저장하고
        # 실제 코드(벡터)는 .lists 파일에 담겨 있습니다. 두 파일을 함께 보관하세요.
        faiss.write_index(index, self._faiss_path)
        index.nprobe = self.faiss_nprobe
        self.faiss_index = index
        logging.info(f"[FAISS] Saved to {self._faiss_path} (nprobe={self.faiss_nprobe})")

    # --------- Public API ---------
    def index(self, corpus: dict):
        self._corpus = corpus

        # 캐시에서 fde_index / doc_ids 로드 (있으면 그대로 사용)
        if self._load_cache():
            # ensure FAISS index is available/loaded
            if self.use_faiss_ann and self.faiss_index is None:
                try:
                    self._build_or_load_faiss_index()
                except Exception as e:
                    logging.warning(f"[FAISS] Build/load skipped due to error: {e}")
            return

        # (생성 경로는 생략 — 사용자 코드에선 이미 외부 임베딩 활용 후 FDE 생성함)
        raise RuntimeError("fde_index cache missing; run the FDE building step first to create fde_index.pkl")

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

        # make sure FAISS index exists if we want ANN
        if self.use_faiss_ann and self.faiss_index is None:
            try:
                self._build_or_load_faiss_index()
            except Exception as e:
                logging.warning(f"[FAISS] Build/load failed; falling back to brute-force: {e}")
                self.use_faiss_ann = False

        key = self._query_key(query, query_id)
        cached_emb, cached_fde = self._load_query_cache(key)

        # --- 1) FDE for query ---
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

        # --- 2) First-stage retrieval: FAISS ANN or brute-force matmul ---
        if self.use_faiss_ann and (self.faiss_index is not None):
            # ANN with IVFFlat IP
            xq = np.ascontiguousarray(query_fde.reshape(1, -1).astype(np.float32))
            t_ann = time.perf_counter()
            # print(f"[First-stage retrieval] {np.ascontiguousarray}")
            D, I = self.faiss_index.search(xq, max(self.faiss_candidates, self.rerank_candidates))
            ann_time = time.perf_counter() - t_ann

            I = I[0]
            D = D[0]
            mask = I >= 0
            idxs = I[mask]
            scores = D[mask]

            # ordered candidates from ANN
            cand_ids = [self.doc_ids[i] for i in idxs]
            cand_scores = scores.tolist()
            search_time = time.perf_counter() - t0  # includes ANN time

            # build candidate dict (limit to ANN results only)
            initial_candidates = list(zip(cand_ids, cand_scores))
        else:
            # Brute-force fallback: exact top over all docs (as before)
            t_bf = time.perf_counter()
            fde_scores = self.fde_index @ query_fde
            order_fde = np.argsort(-fde_scores)
            bf_time = time.perf_counter() - t_bf
            cand_idx = order_fde[:max(self.faiss_candidates, self.rerank_candidates)]
            initial_candidates = [(self.doc_ids[i], float(fde_scores[i])) for i in cand_idx]
            search_time = time.perf_counter() - t0
            ann_time = bf_time

        # --- 3) Rerank with Chamfer(MaxSim) on top-N (if enabled) ---
        rerank_time = 0.0
        if not self.enable_rerank or self.rerank_candidates <= 0 or len(initial_candidates) == 0:
            # no rerank; return ANN/brute candidates as ordered dict
            self._log_latency(str(query_id) if query_id is not None else "", search_time, rerank_time)
            return OrderedDict(initial_candidates)

        t1 = time.perf_counter()
        N = min(self.rerank_candidates, len(initial_candidates))
        topN_ids = [did for (did, _) in initial_candidates[:N]]

        reranked = []
        for did in topN_ids:
            d_tok = self._get_doc_embeddings(did, allow_build=True)
            score = self._chamfer(query_embeddings, d_tok)
            reranked.append((did, score))
        reranked.sort(key=lambda x: x[1], reverse=True)

        # tail: keep ANN/brute order for the rest
        topN_set = {did for did, _ in reranked}
        tail = [(did, sc) for (did, sc) in initial_candidates if did not in topN_set]

        out = OrderedDict()
        for did, sc in reranked:
            out[did] = float(sc)
        for did, sc in tail:
            out[did] = float(sc)

        rerank_time = time.perf_counter() - t1
        self._log_latency(str(query_id) if query_id is not None else "", search_time, rerank_time)

        logging.info(
            f"[search] QID={query_id} "
            f"backend={'FAISS' if self.use_faiss_ann and self.faiss_index is not None else 'matmul'} "
            f"cands={len(initial_candidates)} reranked={N} "
            f"search_ms={search_time*1000:.3f} rerank_ms={rerank_time*1000:.3f}"
            + (f" ann_ms={ann_time*1000:.3f}" if self.use_faiss_ann else "")
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

    corpus, queries, qrels = load_nanobeir_dataset(DATASET_REPO_ID)

    # 쿼리를 첫 100개로 제한 (1:100)
    queries = dict(list(queries.items())[:100])
    logging.info(f"Limited queries to first 100: {len(queries)} queries.")

    logging.info("Initializing retrieval models...")
    retrievers = {
        "2. ColBERT + FDE (+FAISS IVF IP) + Chamfer": ColbertFdeRetriever(
            model_name=COLBERT_MODEL_NAME,
            rerank_candidates=100,
            enable_rerank=True,
            save_doc_embeds=True,
            latency_log_path=os.path.join(QUERY_SEARCH_DIR, "latency.tsv"),  # QID\tSearch\tRerank
            external_doc_embeds_dir=f"/home/dccbeta/muvera_optimized/cache_muvera/{dataset}/doc_embeds",

            # FAISS 설정
            use_faiss_ann=True,   # False로 끄면 기존 matmul 경로 사용
            faiss_nlist=100,
            faiss_nprobe=10,
            faiss_candidates=100,
        )
    }

    timings, final_results = {}, {}

    # 지연시간 로그 파일 초기화
    with open(os.path.join(QUERY_SEARCH_DIR, "latency.tsv"), "w", encoding="utf-8") as f:
        f.write("QID\tSearch\tRerank\n")

    logging.info("--- PHASE 1: INDEXING / LOAD CACHES ---")
    for name, retriever in retrievers.items():
        start_time = time.perf_counter()
        retriever.index(corpus)   # fde_index.pkl / faiss index 로드(없으면 빌드)
        timings[name] = {"indexing_time": time.perf_counter() - start_time}
        logging.info(f"'{name}' ready in {timings[name]['indexing_time']:.2f} seconds.")

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
        logging.info(f"'{name}' Avg query time: {timings[name]['avg_query_time'] * 1000:.2f} ms.")

    print("\n" + "=" * 85)
    print(f"{'FINAL REPORT':^85}")
    print(f"(Dataset: {DATASET_REPO_ID})")
    print("=" * 85)
    print(f"{'Retriever':<40} | {'Ready Time (s)':<16} | {'Avg Query Time (ms)':<22} | {'Recall@{k}'.format(k=TOP_K):<10}")
    print("-" * 85)

    for name in retrievers.keys():
        #recall = evaluate_recall(final_results[name], qrels, k=TOP_K)
        ready_s = timings[name]["indexing_time"]
        query_time_ms = timings[name]["avg_query_time"] * 1000
        #print(f"{name:<40} | {ready_s:<16.2f} | {query_time_ms:<22.2f} | {recall:<10.4f}")
        print(f"{name:<40} | {ready_s:<16.2f} | {query_time_ms:<22.2f}")

    print("=" * 85)
