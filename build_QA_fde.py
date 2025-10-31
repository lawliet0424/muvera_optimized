# -*- coding: utf-8 -*-
"""
Offline Index Builder for Query-Aware Two-Stage Rerank (with reuse)

Builds:
  - Per-doc token embeddings (.npy)  ← 기존이 있으면 재사용, 없으면 생성
  - doc_ids.json
  - FDE index (joblib)
  - FAISS IVF(IP) index
  - Pack (tokens.bin & pointers)
  - Sketch sidecars (Top-L token rows per doc)
  - Token Routing LSH index (joblib)

Usage (예):
  python offline_index_builder.py \
    --dataset arguana \
    --target_num_docs 0 \
    --pack_block_size 512 \
    --sketch_topL 256 \
    --fde_dim 128 --fde_reps 2 --fde_simhash 3 \
    --nlist 1000 --nprobe 50 \
    --routing_bits 64 --routing_tables 4

출력 루트: cache_muvera/<dataset>/
"""

import os, json, math, time, logging, pathlib, argparse, csv, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, DefaultDict
from collections import defaultdict

import numpy as np
import joblib
import torch

# External deps
import neural_cherche.models as neural_cherche_models
import neural_cherche.rank as neural_cherche_rank

from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader

try:
    import faiss
    _FAISS_OK = True
except Exception:
    faiss = None
    _FAISS_OK = False

# FDE
from fde_generator_optimized_stream import (
    FixedDimensionalEncodingConfig,
    generate_document_fde_batch,
)

# --------------------------
# Logging
# --------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------
# Small utils
# --------------------------
def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    elif isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    raise TypeError(type(x))

def l2_norm_rows(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=1))

# --------------------------
# Pack Builder / Reader
# --------------------------
@dataclass
class PackPaths:
    tokens_bin: str
    doc_ptrs: str
    block_ptrs: str
    block_max_norms: str
    pack_meta: str

class PackBuilder:
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

    def build_from_docs(self, doc_ids: List[str], doc_emb_dir: str):
        logging.info(f"[Pack] build → {os.path.dirname(self.paths.tokens_bin)} "
                     f"(block_size={self.block_size}, dim={self.dim})")

        f_tokens = open(self.paths.tokens_bin, "wb", buffering=0)
        doc_ptrs: List[int] = [0]
        block_ptrs: List[int] = [0]
        block_max_norms: List[float] = []

        total_rows = 0
        cur_block_rows: List[np.ndarray] = []
        cur_cnt = 0

        def _flush_block():
            nonlocal cur_block_rows, cur_cnt, total_rows
            if cur_cnt == 0: return
            blk = np.vstack(cur_block_rows).astype(np.float32, copy=False)
            maxn = float(l2_norm_rows(blk).max()) if blk.shape[0] > 0 else 0.0
            block_max_norms.append(maxn)
            f_tokens.write(blk.tobytes(order="C"))
            total_rows += blk.shape[0]
            block_ptrs.append(total_rows)
            cur_block_rows.clear()
            cur_cnt = 0

        for i, did in enumerate(doc_ids):
            path = os.path.join(doc_emb_dir, f"{i:08d}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            d_tok = np.load(path).astype(np.float32, copy=False)
            if d_tok.ndim != 2 or d_tok.shape[1] != self.dim:
                raise ValueError(f"dim mismatch @ {did}: got {d_tok.shape}")

            off = 0
            while off < d_tok.shape[0]:
                remain = self.block_size - cur_cnt
                take = min(remain, d_tok.shape[0] - off)
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
        meta = {
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
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logging.info(f"[Pack] done. rows={total_rows}, blocks={len(block_ptrs)-1}")

class PackReader:
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
        self.total_rows = int(meta["total_rows"])
        self.fd = os.open(self.tokens_bin_path, os.O_RDONLY)

    def doc_row_span(self, doc_idx: int) -> Tuple[int, int]:
        s = int(self.doc_ptrs[doc_idx]); e = int(self.doc_ptrs[doc_idx+1])
        return s, e

    def pread_rows(self, start_row: int, n_rows: int) -> np.ndarray:
        if n_rows <= 0: return np.empty((0, self.dim), np.float32)
        byte_off = start_row * self.dim * self.itemsize
        byte_len = n_rows   * self.dim * self.itemsize
        buf = os.pread(self.fd, byte_len, byte_off)
        arr = np.frombuffer(buf, dtype=np.float32, count=n_rows * self.dim)
        return np.ascontiguousarray(arr.reshape(n_rows, self.dim))

    @staticmethod
    def rows_to_spans(row_ids: np.ndarray) -> List[Tuple[int, int]]:
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

# --------------------------
# Token Routing LSH
# --------------------------
class TokenRoutingLSH:
    def __init__(self, dim: int, bits: int = 64, n_tables: int = 4, seed: int = 42):
        self.dim = int(dim); self.bits = int(bits); self.n_tables = int(n_tables)
        rng = np.random.default_rng(seed)
        self.proj = [np.ascontiguousarray(rng.standard_normal((dim, bits)).astype(np.float32))
                     for _ in range(n_tables)]
        self.tables: List[DefaultDict[int, List[Tuple[int, int]]]] = [defaultdict(list) for _ in range(n_tables)]

    @staticmethod
    def _bits_to_int(bits: np.ndarray) -> int:
        out = 0
        for b in bits.astype(np.uint8):
            out = (out << 1) | int(b)
        return int(out)

    def add_token(self, table_id: int, hash_int: int, doc_idx: int, abs_row: int):
        self.tables[table_id][hash_int].append((int(doc_idx), int(abs_row)))

    def build_from_pack_topL(self, pack: PackReader, doc_ids: List[str], sketch_dir: str, L: int):
        for di, _ in enumerate(doc_ids):
            idx_path = os.path.join(sketch_dir, f"{di:08d}.idx.npy")
            if not os.path.exists(idx_path):
                raise FileNotFoundError(idx_path)
            idx = np.load(idx_path).astype(np.int64)
            ds, _ = pack.doc_row_span(di)
            abs_rows = ds + idx
            spans = PackReader.rows_to_spans(abs_rows)
            dL = _pread_row_spans(pack, spans)  # (L, dim)
            for t in range(self.n_tables):
                bits = (dL @ self.proj[t] >= 0.0).astype(np.uint8)   # (L, bits)
                for r, b in zip(abs_rows, bits):
                    h = self._bits_to_int(b)
                    self.add_token(t, h, di, int(r))

    def dump(self, path: str):
        joblib.dump(dict(proj=self.proj, tables=self.tables,
                         bits=self.bits, n_tables=self.n_tables, dim=self.dim), path)

def _pread_row_spans(pack: PackReader, spans: List[Tuple[int, int]]) -> np.ndarray:
    chunks = [pack.pread_rows(s, n) for (s, n) in spans if n > 0]
    return np.ascontiguousarray(np.vstack(chunks)) if chunks else np.empty((0, pack.dim), np.float32)

# --------------------------
# BEIR load
# --------------------------
def load_dataset(dataset: str, target_num_docs: int = 0, data_root: Optional[str] = None):
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets") if data_root is None else data_root
    data_path = beir_util.download_and_unzip(url, out_dir)
    corpus, _, _ = GenericDataLoader(data_folder=data_path).load(split="test")
    if target_num_docs and target_num_docs > 0:
        kept = dict(list(corpus.items())[:target_num_docs])
    else:
        kept = corpus
    logging.info(f"[BEIR] dataset={dataset} | docs={len(kept)}")
    return kept

# --------------------------
# encode_and_save_docs_reuse
# --------------------------
def encode_and_save_docs_reuse(
    model_name: str,
    corpus: Dict[str, dict],
    out_dir: str,
    reuse_existing: bool = True,
    quick_validate_samples: int = 3,
) -> List[str]:
    """
    기존 임베딩이 있으면 재사용, 없는 문서만 인코딩하여 저장.
    - doc_ids.json이 있으면 그 순서를 '진실원'으로 사용.
    - doc_ids.json이 없으면 corpus 순서로 새로 만들고, 파일명 인덱스에 매핑.
    - 빠른 유효성 검사(샘플 파일 존재/차원 일관성) 수행.
    """
    os.makedirs(out_dir, exist_ok=True)
    emb_dir = os.path.join(out_dir, "doc_embeds")
    os.makedirs(emb_dir, exist_ok=True)

    doc_ids_path = os.path.join(out_dir, "doc_ids.json")

    # 1) doc_ids 결정
    if os.path.exists(doc_ids_path):
        with open(doc_ids_path, "r", encoding="utf-8") as f:
            doc_ids = json.load(f)
        if len(doc_ids) != len(corpus):
            logging.warning(
                f"[reuse] doc_ids.json(len={len(doc_ids)}) != corpus(len={len(corpus)}). "
                "기존 순서를 유지하지만, 누락/추가 문서는 인코딩/재사용이 엇갈릴 수 있습니다."
            )
    else:
        doc_ids = list(corpus.keys())
        with open(doc_ids_path, "w", encoding="utf-8") as f:
            json.dump(doc_ids, f, ensure_ascii=False, indent=2)
        logging.info(f"[reuse] created doc_ids.json with {len(doc_ids)} entries.")

    # 2) 빠른 유효성 검사(기존 파일 재사용 여부 판단)
    def _npypath(i: int) -> str:
        return os.path.join(emb_dir, f"{i:08d}.npy")

    have_any = any(os.path.exists(_npypath(i)) for i in range(min(len(doc_ids), 10)))
    first_dim: Optional[int] = None

    if reuse_existing and have_any:
        idxs = list(range(len(doc_ids)))
        random.shuffle(idxs)
        idxs = idxs[:max(1, min(quick_validate_samples, len(doc_ids)))]
        ok = True
        for i in idxs:
            p = _npypath(i)
            if not os.path.exists(p):
                ok = False; break
            try:
                arr = np.load(p, mmap_mode="r")
                if arr.ndim != 2:
                    ok = False; break
                if first_dim is None:
                    first_dim = int(arr.shape[1])
                elif int(arr.shape[1]) != first_dim:
                    ok = False; break
            except Exception:
                ok = False; break
        if ok:
            logging.info("[reuse] existing embeddings detected; will reuse and fill missing only.")
        else:
            logging.warning("[reuse] existing files are inconsistent; missing or dim mismatch. "
                            "Will re-encode missing/invalid ones.")
    else:
        logging.info("[reuse] no existing embeddings found (or reuse disabled). Will encode all (as needed).")

    # 3) 모델 준비 (필요 시에만)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    ranker = None

    def _ensure_model():
        nonlocal model, ranker
        if model is None:
            model = neural_cherche_models.ColBERT(model_name_or_path=model_name, device=device)
            ranker = neural_cherche_rank.ColBERT(key="id", on=["title", "text"], model=model)

    # 4) 루프: 파일이 없으면 인코딩, 있으면 스킵
    encoded, skipped = 0, 0
    for i, did in enumerate(doc_ids):
        out_path = _npypath(i)
        if reuse_existing and os.path.exists(out_path):
            try:
                arr = np.load(out_path, mmap_mode="r")
                if arr.ndim == 2 and (first_dim is None or int(arr.shape[1]) == first_dim):
                    skipped += 1
                    continue
                else:
                    logging.warning(f"[reuse] bad shape at {out_path}; will overwrite.")
            except Exception:
                logging.warning(f"[reuse] cannot read {out_path}; will overwrite.")

        _ensure_model()
        doc = {"id": did, **corpus.get(did, {})}
        emb_map = ranker.encode_documents(documents=[doc])
        arr = to_numpy(emb_map[did])  # (n_tokens, d)
        np.save(out_path, arr)
        if first_dim is None:
            first_dim = int(arr.shape[1])
        encoded += 1
        if (i + 1) % 100 == 0:
            logging.info(f"[reuse] processed {i+1}/{len(doc_ids)} (encoded={encoded}, skipped={skipped})")

    logging.info(f"[reuse] done. total={len(doc_ids)}, encoded={encoded}, reused(skipped)={skipped}")
    return doc_ids

# --------------------------
# Core builders
# --------------------------
def build_fde_index(out_dir: str, doc_ids: List[str], fde_cfg: FixedDimensionalEncodingConfig):
    doc_emb_dir = os.path.join(out_dir, "doc_embeds")
    def gen():
        for i, _ in enumerate(doc_ids):
            yield np.load(os.path.join(doc_emb_dir, f"{i:08d}.npy"))
    X, meta = generate_document_fde_batch(gen(), fde_cfg)  # X: (N_docs, fde_dim)
    in_default = f"fde_index_{fde_cfg.num_simhash_projections}_{fde_cfg.num_repetitions}.pkl"
    meta_default = f"meta_{fde_cfg.num_simhash_projections}_{fde_cfg.num_repetitions}.json"
    joblib.dump(X.astype(np.float32, copy=False), os.path.join(out_dir, in_default))
    with open(os.path.join(out_dir, meta_default), "w", encoding="utf-8") as f:
        json.dump({"config": vars(fde_cfg), "shape": list(X.shape)}, f, ensure_ascii=False, indent=2)
    logging.info(f"[FDE] saved: {X.shape}")
    return in_default, meta_default

def build_faiss_ivf_ip(out_dir: str, fde_path: str, nlist: int, nprobe: int) -> str:
    if not _FAISS_OK:
        raise RuntimeError("FAISS not available")
    X = joblib.load(fde_path)  # (N, d)
    dim = int(X.shape[1])
    quant = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quant, dim, int(nlist), faiss.METRIC_INNER_PRODUCT)
    logging.info(f"[FAISS] train nlist={nlist} on X={X.shape}")
    if not X.flags["C_CONTIGUOUS"] or X.dtype != np.float32:
        X = np.ascontiguousarray(X.astype(np.float32))
    index.train(X)
    index.add(X)
    index.nprobe = int(nprobe)
    # derive P,R from filename "..._P_R.pkl"
    base = os.path.basename(fde_path).split(".")[0]  # fde_index_P_R
    _, P, R = base.split("_")
    faiss_path = os.path.join(out_dir, f"ivf{nlist}_ip_{P}_{R}.faiss")
    faiss.write_index(index, faiss_path)
    logging.info(f"[FAISS] saved → {faiss_path}")
    return faiss_path

def build_pack(out_dir: str, doc_ids: List[str], dim: int, block_size: int):
    pb = PackBuilder(os.path.join(out_dir, "pack"), block_size=block_size, dim=dim)
    pb.build_from_docs(doc_ids, os.path.join(out_dir, "doc_embeds"))

def build_sketch_sidecars(out_dir: str, doc_ids: List[str], topL: int):
    sketch_dir = os.path.join(out_dir, "doc_stage")
    os.makedirs(sketch_dir, exist_ok=True)
    doc_emb_dir = os.path.join(out_dir, "doc_embeds")
    for i, did in enumerate(doc_ids):
        arr = np.load(os.path.join(doc_emb_dir, f"{i:08d}.npy")).astype(np.float32, copy=False)
        n = arr.shape[0]
        if n <= topL:
            idx = np.arange(n, dtype=np.int64)
            sk = arr
        else:
            norms = l2_norm_rows(arr)
            idx = np.argpartition(norms, -topL)[-topL:].astype(np.int64)
            sk = arr[idx]
        np.save(os.path.join(sketch_dir, f"{i:08d}.idx.npy"), idx)
        np.save(os.path.join(sketch_dir, f"{i:08d}.npy"), sk)
    logging.info(f"[Sketch] saved L={topL} for {len(doc_ids)} docs")

def build_token_routing_lsh(out_dir: str, bits: int, tables: int, sketch_topL: int):
    pack = PackReader(os.path.join(out_dir, "pack"))
    with open(os.path.join(out_dir, "doc_ids.json"), "r", encoding="utf-8") as f:
        doc_ids = json.load(f)
    router = TokenRoutingLSH(dim=int(pack.dim), bits=int(bits), n_tables=int(tables))
    router.build_from_pack_topL(pack, doc_ids, os.path.join(out_dir, "doc_stage"), L=sketch_topL)
    os.makedirs(os.path.join(out_dir, "routing"), exist_ok=True)
    router_path = os.path.join(out_dir, "routing", "router.joblib")
    router.dump(router_path)
    logging.info(f"[Routing] saved → {router_path}")

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Offline index builder (pack/sketch/FDE/FAISS/LSH) with reuse")
    parser.add_argument("--dataset", type=str, required=True, help="BEIR dataset id (e.g., arguana, scidocs, ...)")
    parser.add_argument("--model", type=str, default="raphaelsty/neural-cherche-colbert")
    parser.add_argument("--cache_root", type=str, default=os.path.join(pathlib.Path(__file__).parent.absolute(), "cache_muvera"))
    parser.add_argument("--target_num_docs", type=int, default=0, help="0이면 전체 문서")
    # FDE
    parser.add_argument("--fde_dim", type=int, default=128)
    parser.add_argument("--fde_reps", type=int, default=2)
    parser.add_argument("--fde_simhash", type=int, default=3)
    # FAISS
    parser.add_argument("--nlist", type=int, default=1000)
    parser.add_argument("--nprobe", type=int, default=50)
    # Pack/Sketch
    parser.add_argument("--pack_block_size", type=int, default=512)
    parser.add_argument("--sketch_topL", type=int, default=256)
    # Routing LSH
    parser.add_argument("--routing_bits", type=int, default=64)
    parser.add_argument("--routing_tables", type=int, default=4)
    # Reuse control
    parser.add_argument("--no_reuse", action="store_true", help="기존 doc_embeds 재사용하지 않고 전부 새로 인코딩")

    args = parser.parse_args()

    dataset = args.dataset
    out_dir = os.path.join(args.cache_root, dataset)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load corpus
    corpus = load_dataset(dataset, target_num_docs=args.target_num_docs)

    # 2) Encode & save per-doc token embeddings (reuse if present)
    t0 = time.perf_counter()
    doc_ids = encode_and_save_docs_reuse(
        model_name=args.model,
        corpus=corpus,
        out_dir=out_dir,
        reuse_existing=(not args.no_reuse),
        quick_validate_samples=3,
    )
    enc_s = time.perf_counter() - t0

    # 3) Build FDE index
    fde_cfg = FixedDimensionalEncodingConfig(
        dimension=args.fde_dim,
        num_repetitions=args.fde_reps,
        num_simhash_projections=args.fde_simhash,
        seed=42,
        fill_empty_partitions=True,
    )
    in_default, meta_default = build_fde_index(out_dir, doc_ids, fde_cfg)

    # 4) Build FAISS IVF(IP)
    # faiss_path = build_faiss_ivf_ip(out_dir, os.path.join(out_dir, in_default), nlist=args.nlist, nprobe=args.nprobe)

    # 5) Build Pack
    first_arr = np.load(os.path.join(out_dir, "doc_embeds", f"{0:08d}.npy"))
    build_pack(out_dir, doc_ids, dim=int(first_arr.shape[1]), block_size=args.pack_block_size)

    # 6) Build Sketch sidecars (Top-L rows per doc)
    build_sketch_sidecars(out_dir, doc_ids, topL=args.sketch_topL)

    # 7) Build Token Routing LSH (Query-Aware routing)
    build_token_routing_lsh(out_dir, bits=args.routing_bits, tables=args.routing_tables, sketch_topL=args.sketch_topL)

    logging.info(f"[DONE] dataset={dataset} | docs={len(doc_ids)} | enc(reuse-aware)={enc_s:.2f}s")
    logging.info(f"  out dir: {out_dir}")
    # logging.info(f"  FDE: {in_default}, meta: {meta_default}")
    logging.info(f"  Pack: {os.path.join(out_dir,'pack')}")
    logging.info(f"  Sketch: {os.path.join(out_dir,'doc_stage')}")
    logging.info(f"  Routing: {os.path.join(out_dir,'routing','router.joblib')}")

if __name__ == "__main__":
    main()