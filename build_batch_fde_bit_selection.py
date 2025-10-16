# -*- coding: utf-8 -*-
import os, json, time, hashlib, logging, pathlib, argparse, sys
from collections import OrderedDict
from dataclasses import replace
from typing import Optional, List, Tuple

import nltk
import numpy as np
import torch
import joblib
import psutil

import neural_cherche.models as neural_cherche_models
import neural_cherche.rank as neural_cherche_rank

from beir import util
from beir.datasets.data_loader import GenericDataLoader

import gc

# FDE 구현 (업로드된 파일 사용)
from fde_generator_optimized_stream_bit_selection import (
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    generate_document_fde_batch,
    apply_bit_selection_to_query,
    load_structured_fde_index,
)

# 메모리 사용량 확인 함수
def log_memory_usage(stage: str):
    """현재 메모리 사용량을 로깅"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024  # MB 단위
    memory_gb = memory_mb / 1024  # GB 단위
    logging.info(f"[MEMORY] {stage}: {memory_mb:.1f} MB ({memory_gb:.2f} GB)")
    return memory_mb

# ======================
# --- Configuration ----
# ======================
DATASET_REPO_ID = "scidocs"
COLBERT_MODEL_NAME = "raphaelsty/neural-cherche-colbert"
TOP_K = 10
ATOMIC_BATCH_SIZE = 1000

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# 데이터셋 준비 (BEIR trec-covid)
dataset = "scidocs"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

# [original] 캐시 루트
# CACHE_ROOT = os.path.join(pathlib.Path(__file__).parent.absolute(), "cache_muvera")
# os.makedirs(CACHE_ROOT, exist_ok=True)

# 캐시 루트
FILENAME = "build_fde"
CACHE_ROOT = os.path.join(pathlib.Path(__file__).parent.absolute(), "cache_muvera", DATASET_REPO_ID)
os.makedirs(CACHE_ROOT, exist_ok=True)

# 쿼리 검색 디렉터리
QUERY_SEARCH_DIR = os.path.join(CACHE_ROOT, "query_search", FILENAME)
#os.makedirs(QUERY_SEARCH_DIR, exist_ok=True)

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
        enable_bit_selection: bool = False, # bit selection 사용 여부
        bit_selection_ratio: float = 0.5, # bit selection 비율
        structured_output_dir: Optional[str] = None, # bit selection 결과 저장 경로
    ):
        model = neural_cherche_models.ColBERT(model_name_or_path=model_name, device=DEVICE)
        self.ranker = neural_cherche_rank.ColBERT(key="id", on=["title", "text"], model=model)

        self.doc_config = FixedDimensionalEncodingConfig(
            dimension=128,
            num_repetitions=num_repetitions,
            num_simhash_projections=num_simhash_projections,
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
        
        # Bit selection 관련 속성들
        self.enable_bit_selection = enable_bit_selection
        self.bit_selection_ratio = bit_selection_ratio
        self.structured_output_dir = structured_output_dir
        self.selected_bits = None
        self.bit_selection_metadata = None

        # 캐시 경로
        self._model_name = model_name
        self._cache_dir = self._compute_cache_dir(
            dataset=DATASET_REPO_ID, model_name=model_name, cfg=self.doc_config
        )
        self._fde_path = os.path.join(self._cache_dir, default_out)
        self._ids_path = os.path.join(self._cache_dir, "doc_ids.json")
        self._meta_path = os.path.join(self._cache_dir, meta_default)
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
        return os.path.join(CACHE_ROOT)

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
        # Bit selection과 구조화된 저장이 활성화된 경우
        if self.enable_bit_selection and self.structured_output_dir:
            try:
                # Bit selection metadata 로드
                bits_path = os.path.join(self._cache_dir, "selected_bits.npy")
                metadata_path = os.path.join(self._cache_dir, "bit_selection_metadata.json")
                
                if os.path.exists(bits_path):
                    self.selected_bits = np.load(bits_path)
                    logging.info(f"[{self.__class__.__name__}] Loaded selected bits: {len(self.selected_bits)}")
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.bit_selection_metadata = json.load(f)
                    logging.info(f"[{self.__class__.__name__}] Loaded bit selection metadata")
                
                # 구조화된 FDE 인덱스 로드
                self.fde_index = load_structured_fde_index(self._cache_dir, self.bit_selection_metadata)
                
                # 문서 ID 로드
                with open(self._ids_path, "r", encoding="utf-8") as f:
                    self.doc_ids = json.load(f)
                self._doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
                
                logging.info(
                    f"[{self.__class__.__name__}] Loaded structured FDE index: "
                    f"{self.fde_index.shape} for {len(self.doc_ids)} docs"
                )
                return True
                
            except Exception as e:
                logging.warning(f"[{self.__class__.__name__}] Failed to load structured index: {e}")
                # Fallback to regular cache loading
        
        # 일반 캐시 로드 (fallback)
        self.fde_index = joblib.load(self._fde_path)
        with open(self._ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        self._doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        
        # Bit selection 정보 로드
        if self.enable_bit_selection:
            bits_path = os.path.join(self._cache_dir, "selected_bits.npy")
            metadata_path = os.path.join(self._cache_dir, "bit_selection_metadata.json")
            
            if os.path.exists(bits_path):
                self.selected_bits = np.load(bits_path)
                logging.info(f"[{self.__class__.__name__}] Loaded selected bits: {len(self.selected_bits)}")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.bit_selection_metadata = json.load(f)
                logging.info(f"[{self.__class__.__name__}] Loaded bit selection metadata")
        
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
            logging.info(f"[doc-embed] external load: id={doc_id} path={ext_path}")
            return np.load(ext_path)

        # 2) 내부 캐시
        int_path = self._doc_emb_path(doc_id)
        if os.path.exists(int_path):
            logging.info(f"[doc-embed] internal load: id={doc_id} path={int_path}")
            return np.load(int_path)

        # 3) 필요 시 빌드
        if not allow_build:
            raise FileNotFoundError(ext_path or int_path)

        if self._corpus is None:
            raise RuntimeError("Corpus not set; cannot build document embeddings on the fly.")
        doc = {"id": doc_id, **self._corpus[doc_id]}
        emap = self.ranker.encode_documents(documents=[doc])#, batch_size=128)
        arr = to_numpy(emap[doc_id])

        # 내부 캐시에 저장(선택)
        np.save(int_path, arr)
        logging.info(f"[doc-embed] built & saved: id={doc_id} path={int_path}")
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

        # [1015]---------- 누락된 문서 ID만 수집 (메모리 효율적) ----------
        missing_doc_ids: List[str] = []

        # 1) 누락된 문서만 찾기 (임베딩 로드하지 않음)
        for doc_id in self.doc_ids:
            ext = self._external_doc_emb_path(doc_id)            
            if ext and os.path.exists(ext):
                # 외부에 있으면 스킵
                continue

            # 내부 캐시 확인
            dst = self._doc_emb_path(doc_id)
            if os.path.exists(dst):
                # 내부에 있으면 스킵
                continue
            else:
                missing_doc_ids.append(doc_id)

        logging.info(
            f"[index] missing documents to encode: {len(missing_doc_ids)} / {len(self.doc_ids)}"
        )

        # [1015] 2) Atomic 배치 처리: Encoding → FDE → Index 저장
        log_memory_usage("Before atomic batch processing")

        if missing_doc_ids:
            logging.info(f"[index] encoding {len(missing_doc_ids)} documents that are missing from precomputed files...")
        
        # FDE 인덱스 초기화 (memmap으로)
        fde_memmap_path = os.path.join(self._cache_dir, f"fde_index_{P}_{R}.mmap")
        num_partitions = 2 ** self.doc_config.num_simhash_projections
        final_fde_dim_per_rep = num_partitions * (self.doc_config.projection_dimension or self.doc_config.dimension)
        final_fde_dim = self.doc_config.num_repetitions * final_fde_dim_per_rep
        
        # FDE 인덱스 memmap 생성
        fde_index = np.memmap(fde_memmap_path, mode="w+", dtype=np.float32, 
                             shape=(len(self.doc_ids), final_fde_dim))
        
        # Bit selection용 압축된 FDE 인덱스 초기화
        compressed_fde_index = None
        if self.enable_bit_selection:
            compressed_memmap_path = os.path.join(self._cache_dir, f"compressed_fde_index_{P}_{R}.mmap")
            # 첫 번째 배치에서만 압축된 차원을 알 수 있으므로, 임시로 원본 크기로 초기화
            compressed_fde_index = np.memmap(compressed_memmap_path, mode="w+", dtype=np.float32, 
                                           shape=(len(self.doc_ids), final_fde_dim))
        
        # Atomic 배치 처리 (1000개 문서씩)
        atomic_batch_size = ATOMIC_BATCH_SIZE  # 인코딩과 FDE 배치 크기 통일
        logging.info(f"[{self.__class__.__name__}] Processing {len(missing_doc_ids)} documents in atomic batches of {atomic_batch_size}...")
        
        for batch_start in range(0, len(missing_doc_ids), atomic_batch_size):
            batch_end = min(batch_start + atomic_batch_size, len(missing_doc_ids))
            batch_doc_ids = missing_doc_ids[batch_start:batch_end]
            
            logging.info(f"[Atomic Batch] Processing batch {batch_start//atomic_batch_size + 1}/{(len(missing_doc_ids) + atomic_batch_size - 1)//atomic_batch_size}: docs {batch_start}-{batch_end-1}")
            
            # Step 1: 배치용 임베딩 수집 (파일에서 직접 로드)
            batch_embeddings = []
            batch_missing_ids = []
            
            for doc_id in batch_doc_ids:
                # 파일에서 직접 로드 (메모리 효율적)
                ext = self._external_doc_emb_path(doc_id)
                if ext and os.path.exists(ext):
                    batch_embeddings.append(np.load(ext).astype(np.float32))
                else:
                    dst = self._doc_emb_path(doc_id)
                    if os.path.exists(dst):
                        batch_embeddings.append(np.load(dst).astype(np.float32))
                    else:
                        batch_missing_ids.append(doc_id)
            
            # 누락된 문서들 배치 인코딩
            if batch_missing_ids:
                logging.info(f"[Atomic Batch] Encoding {len(batch_missing_ids)} missing documents...")
                to_encode_docs = [{"id": did, **corpus[did]} for did in batch_missing_ids]
                encoded_map = self.ranker.encode_documents(documents=to_encode_docs)
                
                for did in batch_missing_ids:
                    arr = to_numpy(encoded_map[did])
                    batch_embeddings.append(arr)
                    if self.save_doc_embeds:
                        np.save(self._doc_emb_path(did), arr)
                    del encoded_map[did]
                    del arr
                
                del to_encode_docs
                del encoded_map
            
            #--------- bit selection related ----------
            # Step 2: 배치 FDE 생성
            logging.info(f"[Atomic Batch] Generating FDE for {len(batch_embeddings)} documents...")
            batch_fde = generate_document_fde_batch(
                batch_embeddings,
                self.doc_config,
                memmap_path=None,  # 메모리에서 직접 처리
                max_bytes_in_memory=2 * 1024**3,
                log_every=ATOMIC_BATCH_SIZE,
                flush_interval=atomic_batch_size  # atomic_batch_size 활용
                enable_bit_selection=self.enable_bit_selection,
                bit_selection_ratio=self.bit_selection_ratio,
                structured_output_dir=self._cache_dir,
            )

            #--------- bit selection related ----------
            # Bit selection 결과 로드 (첫 번째 배치에서만)
            if self.enable_bit_selection and batch_start == 0:
                bits_path = os.path.join(self._cache_dir, "selected_bits.npy")
                metadata_path = os.path.join(self._cache_dir, "bit_selection_metadata.json")
                
                if os.path.exists(bits_path):
                    self.selected_bits = np.load(bits_path)
                    logging.info(f"[Index] Loaded selected bits: {len(self.selected_bits)}")
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        self.bit_selection_metadata = json.load(f)
                    logging.info(f"[Index] Loaded bit selection metadata: "
                               f"{self.bit_selection_metadata['total_original_dim']} → "
                               f"{self.bit_selection_metadata['total_compressed_dim']} "
                               f"({self.bit_selection_metadata['compression_ratio']*100:.1f}%)")
                    
                    # 압축된 FDE 인덱스 재생성
                    compressed_dim = self.bit_selection_metadata['total_compressed_dim']
                    compressed_memmap_path = os.path.join(self._cache_dir, f"compressed_fde_index_{P}_{R}.mmap")
                    compressed_fde_index = np.memmap(compressed_memmap_path, mode="w+", dtype=np.float32, 
                                                   shape=(len(self.doc_ids), compressed_dim))
                    logging.info(f"[Index] Created compressed FDE index: {compressed_fde_index.shape}")
            #---------------------------------------------------

            # Step 3: FDE 인덱스에 저장
            # Bit selection이 적용된 경우, 압축된 FDE를 저장
            if self.enable_bit_selection and hasattr(self, 'selected_bits') and self.selected_bits is not None:
                # 압축된 FDE를 압축된 인덱스에 저장
                compressed_fde_index[batch_start:batch_end] = batch_fde
            else:
                # 원본 FDE를 원본 인덱스에 저장
                fde_index[batch_start:batch_end] = batch_fde
            
            # Step 4: 배치별 flush (즉시 디스크 저장)
            fde_index.flush()
            if compressed_fde_index is not None:
                compressed_fde_index.flush()
            
            # Step 5: 배치 완료 후 메모리 해제
            del batch_embeddings
            del batch_fde
            gc.collect()
            
            log_memory_usage(f"After atomic batch {batch_start//atomic_batch_size + 1}")
        
        # FDE 인덱스 저장 및 참조 설정
        fde_index.flush()
        if compressed_fde_index is not None:
            compressed_fde_index.flush()
            self.fde_index = compressed_fde_index  # 압축된 인덱스 사용
            logging.info(f"[Index] Using compressed FDE index: {compressed_fde_index.shape}")
        else:
            self.fde_index = fde_index  # 원본 인덱스 사용
            logging.info(f"[Index] Using original FDE index: {fde_index.shape}")
        
        # FDE 인덱스 참조 해제 (메모리 절약)
        del fde_index
        if compressed_fde_index is not None:
            del compressed_fde_index
        gc.collect()
        
        logging.info(f"[Atomic Batch] Completed processing {len(self.doc_ids)} documents")
        logging.info(f"[Atomic Batch] FDE index saved to: {fde_memmap_path}")
        log_memory_usage("After atomic batch processing")

        #[1014] ---------- 메모리 해제 ----------
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

        #--------- bit selection related ----------
        if self.enable_bit_selection and self.bit_selection_metadata is not None:
            query_fde = apply_bit_selection_to_query(query_fde, self.bit_selection_metadata)
            logging.debug(f"[Search] Applied bit selection: {len(query_fde)} dimensions")
        #---------------------------------------------------

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

    # === [ADDED] CLI for parameterized FDE index build ===
    def _positive_int(x: str) -> int:
        v = int(x)
        if v <= 0:
            raise argparse.ArgumentTypeError("must be > 0")
        return v
    
    parser = argparse.ArgumentParser(description="Build FDE index and save as fde_index_{P}_{R}.pkl")
    parser.add_argument("--num_simhash_projections", "--p", type=_positive_int, required=True,
                        help="Number of simhash projections (P)")
    parser.add_argument("--num_repetitions", "--r", type=_positive_int, required=True,
                        help="Number of repetitions (R)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional explicit output path for FDE pickle")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite output if it exists")
    parser.add_argument("--enable_bit_selection", action="store_true",
                        help="Enable bit selection for FDE compression")
    parser.add_argument("--bit_selection_ratio", type=float, default=0.5,
                        help="Bit selection compression ratio (default: 0.5)")
    parser.add_argument("--structured_output_dir", type=str, default=None,
                        help="Directory for structured bit selection output")
    args, _ = parser.parse_known_args()

    P = args.num_simhash_projections
    R = args.num_repetitions

    # 기본 출력 파일명: fde_index_{P}_{R}.pkl (작업 디렉토리 기준)
    default_out = f"fde_index_{P}_{R}.pkl"
    #out_path = pathlib.Path(args.output or default_out)

    meta_default = f"meta_{P}_{R}.json"
    #meta_set_path = pathlib.Path(meta_default)
    
    t0 = time.time()    

    # === [INTEGRATE HERE] ===
    # 1) 아래 두 줄처럼, 기존 코드에서 P/R을 설정하던 변수를 이 값으로 바인딩하세요.
    #    (이미 같은 변수명이면 이 줄이 기존 값을 덮어쓰게 됩니다.)
    num_simhash_projections = P
    num_repetitions = R

    corpus, queries, qrels = load_nanobeir_dataset(DATASET_REPO_ID)

    logging.info("Initializing retrieval models...")
    retrievers = {
        "2. ColBERT + FDE (+Chamfer rerank)": ColbertFdeRetriever(
            model_name=COLBERT_MODEL_NAME,
            rerank_candidates=100,
            enable_rerank=True,
            save_doc_embeds=True,
            latency_log_path=os.path.join(QUERY_SEARCH_DIR, "latency.tsv"),  # QID\tSearch\tRerank
            external_doc_embeds_dir=f"/home/hyunji/Desktop/muvera_optimized/cache_muvera/{DATASET_REPO_ID}/doc_embeds",  # ★ 외부 임베딩 디렉터리 지정
            enable_bit_selection=args.enable_bit_selection,
            bit_selection_ratio=args.bit_selection_ratio,
            structured_output_dir=args.structured_output_dir,
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
            # results[str(query_id)] = retriever.search(query_text, query_id=str(query_id))
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
        recall = evaluate_recall(final_results[name], qrels, k=TOP_K)
        idx_time = timings[name]["indexing_time"]
        query_time_ms = timings[name]["avg_query_time"] * 1000
        print(f"{name:<30} | {idx_time:<20.2f} | {query_time_ms:<22.2f} | {recall:<10.4f}")

    print("=" * 85)
