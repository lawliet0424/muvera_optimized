# -*- coding: utf-8 -*-
"""
역할
----
Storage Node 파이프라인 (shard 당):
  1. GetShard 배정 시 shard_{N}.mmap.tmp 사전 확보
  2. UploadFdeShard: gRPC chunk 수신
  3. row offset 기반 preallocated mmap 기록
  4. chunk completeness 검증
  5. flush + fsync + atomic rename (shard_{N}.mmap)
  6. UploadAck 반환

기타:
- corpus shard 분할 및 Worker 에 문서 원문 streaming
- shard 상태 / timing 집계, final_manifest.json / metrics.csv 생성

배포 (dccblue@163.239.199.208)
------------------------------
  코드/실행     /data/muvera_optimized
  datasets_root        /data/datasets
  dataset_collection   flare
  corpus               /data/datasets/flare/{dataset}/corpus.jsonl
  FDE 출력             /data/muvera_optimized/data/fde_out/{dataset}/
  gRPC                 163.239.199.208:50051

  경로 우선순위: --corpus-path > --dataset > {project_dir}/data/corpus.json

FDE 저장 경로
-------------
  {output_dir}/
    shard_{N:04d}.mmap        # Worker 가 전송한 FDE shard
    doc_ids.json              # shard 순서대로 정렬된 전체 doc_id 목록
    final_manifest.json
    metrics.csv

주의
----
- Storage Node 는 문서 원문(title, text)만 전송한다.
  embedding 은 전송하지 않으며, Worker 가 ColBERT 로 인코딩한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import struct
import threading
import time

from storage_node_config import StorageNodeConfig
from gpu_worker_config import MAX_SIMHASH_PROJECTIONS
from collections import defaultdict
from concurrent import futures
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import grpc

import fde_pipeline_pb2 as pb2
import fde_pipeline_pb2_grpc as pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [STORAGE] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ===========================================================================
# Shard 분할 헬퍼
# ===========================================================================

def compute_shards(total_docs: int, shard_doc_size: int) -> list[tuple[int, int]]:
    """corpus 를 shard_doc_size 단위로 분할 (마지막 shard 는 잔여분)."""
    if total_docs <= 0:
        return []
    size = max(1, shard_doc_size)
    shards = []
    for s in range(0, total_docs, size):
        e = min(s + size, total_docs)
        shards.append((s, e))
    return shards


# ===========================================================================
# Shard 배포 상태 관리
# ===========================================================================

class ShardDispatcher:
    """Thread-safe shard 배포, lease 추적, 장애 시 재할당."""

    STATUS_PENDING   = "pending"
    STATUS_INPROGRESS= "inprogress"
    STATUS_DONE      = "done"
    STATUS_FAILED    = "failed"

    def __init__(
        self,
        shards: list[tuple[int, int]],
        *,
        lease_timeout_sec: float = 600.0,
        max_attempts: int = 3,
    ) -> None:
        self._lock   = threading.Lock()
        self._shards = shards
        self._lease_timeout_sec = lease_timeout_sec
        self._max_attempts = max_attempts
        self._status: Dict[int, str]    = {i: self.STATUS_PENDING for i in range(len(shards))}
        self._worker: Dict[int, str]    = {}
        self._timing: Dict[int, dict]   = {}
        self._leased_at: Dict[int, float] = {}
        self._attempts: Dict[int, int]  = defaultdict(int)
        self._start_time = time.time()

    # ------------------------------------------------------------------ #

    def expire_leases(self) -> List[int]:
        """만료된 inprogress shard 를 pending 으로 되돌리고 index 목록을 반환."""
        now = time.time()
        expired: List[int] = []
        with self._lock:
            for idx, st in self._status.items():
                if st != self.STATUS_INPROGRESS:
                    continue
                leased_at = self._leased_at.get(idx, 0.0)
                if now - leased_at > self._lease_timeout_sec:
                    expired.append(idx)
            for idx in expired:
                self._requeue_unlocked(idx, "lease_timeout")
        return expired

    def _requeue_unlocked(self, idx: int, reason: str) -> None:
        self._attempts[idx] = self._attempts.get(idx, 0) + 1
        attempt = self._attempts[idx]
        if attempt >= self._max_attempts:
            self._status[idx] = self.STATUS_FAILED
            self._timing[idx] = {"error": reason, "attempts": attempt}
            logger.error(
                "[Dispatcher] shard=%d permanently FAILED after %d attempts: %s",
                idx, attempt, reason,
            )
        else:
            self._status[idx] = self.STATUS_PENDING
            logger.warning(
                "[Dispatcher] shard=%d requeued (attempt %d/%d): %s",
                idx, attempt, self._max_attempts, reason,
            )
        self._worker.pop(idx, None)
        self._leased_at.pop(idx, None)

    def next_pending(self, worker_id: str) -> Optional[int]:
        """pending 상태인 첫 번째 shard index 를 반환 (없으면 None)."""
        self.expire_leases()
        with self._lock:
            for idx, st in sorted(self._status.items()):
                if st == self.STATUS_PENDING:
                    self._status[idx] = self.STATUS_INPROGRESS
                    self._worker[idx] = worker_id
                    self._leased_at[idx] = time.time()
                    return idx
        return None

    def touch_lease(self, idx: int) -> None:
        """장시간 업로드 중 lease 만료를 방지."""
        with self._lock:
            if self._status.get(idx) == self.STATUS_INPROGRESS:
                self._leased_at[idx] = time.time()

    def mark_done(self, idx: int, timing: dict) -> None:
        with self._lock:
            self._status[idx] = self.STATUS_DONE
            self._timing[idx] = timing
            self._leased_at.pop(idx, None)
            self._worker.pop(idx, None)

    def report_failure(self, idx: int, error: str) -> None:
        """일시 실패 → pending 재할당 (max_attempts 초과 시 permanent failed)."""
        with self._lock:
            if self._status.get(idx) == self.STATUS_DONE:
                return
            # ReportShardStatus 와 GetShard completed 가 중복 보고할 수 있음
            if self._status.get(idx) != self.STATUS_INPROGRESS:
                return
            self._requeue_unlocked(idx, error)

    def mark_failed(self, idx: int, error: str) -> None:
        """하위 호환 alias — report_failure 와 동일."""
        self.report_failure(idx, error)

    def all_done(self) -> bool:
        with self._lock:
            return all(
                s in (self.STATUS_DONE, self.STATUS_FAILED)
                for s in self._status.values()
            )

    def summary(self) -> dict:
        with self._lock:
            counts = defaultdict(int)
            for s in self._status.values():
                counts[s] += 1
            return {
                "total": len(self._shards),
                "done": counts[self.STATUS_DONE],
                "failed": counts[self.STATUS_FAILED],
                "inprogress": counts[self.STATUS_INPROGRESS],
                "pending": counts[self.STATUS_PENDING],
                "elapsed_sec": time.time() - self._start_time,
                "timings": dict(self._timing),
                "workers": dict(self._worker),
            }

    def shard_range(self, idx: int) -> tuple[int, int]:
        return self._shards[idx]

    def expected_num_docs(self, idx: int) -> int:
        doc_start, doc_end = self._shards[idx]
        return doc_end - doc_start

    def assigned_worker(self, idx: int) -> Optional[str]:
        with self._lock:
            return self._worker.get(idx)

    def shard_status(self, idx: int) -> str:
        with self._lock:
            return self._status.get(idx, self.STATUS_PENDING)

    def num_shards(self) -> int:
        return len(self._shards)


# ===========================================================================
# FDE 저장소 (per-shard mmap)
# ===========================================================================

class FdeStore:
    """수신된 FDE 청크를 임시 mmap 에 기록 → flush+close → atomic rename."""

    def __init__(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        self._lock = threading.Lock()
        self._mmaps: Dict[int, np.memmap] = {}
        self._expected: Dict[int, tuple[int, int]] = {}  # shard → (num_docs, fde_dim)

    @staticmethod
    def _sync_memmap(mm: np.memmap) -> None:
        mm.flush()

    def _final_path(self, shard_index: int) -> str:
        return os.path.join(
            self._output_dir, f"shard_{shard_index:04d}.mmap"
        )

    def _tmp_path(self, shard_index: int) -> str:
        return os.path.join(
            self._output_dir, f"shard_{shard_index:04d}.mmap.tmp"
        )

    def _validate_existing_shape(
        self, final_path: str, num_docs: int, fde_dim: int
    ) -> None:
        """재업로드 시 기존 committed 파일 shape 가 일치하는지 검증."""
        if not os.path.isfile(final_path):
            return
        expected = (num_docs, fde_dim)
        expected_bytes = num_docs * fde_dim * 4
        actual_bytes = os.path.getsize(final_path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"existing mmap size mismatch: file={actual_bytes} bytes, "
                f"expected {expected_bytes} for shape {expected} path={final_path}"
            )
        mm = np.memmap(
            final_path, mode="r", dtype=np.float32, shape=expected
        )
        try:
            if mm.shape != expected:
                raise ValueError(
                    f"existing mmap shape mismatch: got {mm.shape}, "
                    f"expected {expected} path={final_path}"
                )
        finally:
            del mm

    def preallocate(
        self, shard_index: int, num_docs: int, fde_dim: int
    ) -> None:
        """shard 배정 시 mmap 공간을 미리 확보한다 (idempotent)."""
        final_path = self._final_path(shard_index)
        if os.path.isfile(final_path):
            self._validate_existing_shape(final_path, num_docs, fde_dim)
            return

        with self._lock:
            if shard_index in self._mmaps:
                mm = self._mmaps[shard_index]
                if mm.shape != (num_docs, fde_dim):
                    raise ValueError(
                        f"preallocated mmap shape mismatch shard={shard_index}: "
                        f"got {mm.shape}, expected ({num_docs}, {fde_dim})"
                    )
                self._expected[shard_index] = (num_docs, fde_dim)
                return

            self._expected[shard_index] = (num_docs, fde_dim)
            tmp_path = self._tmp_path(shard_index)
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

            mm = np.memmap(
                tmp_path, mode="w+", dtype=np.float32, shape=(num_docs, fde_dim)
            )
            self._mmaps[shard_index] = mm
            logger.info(
                "[FdeStore] Preallocated shard=%d path=%s shape=(%d,%d)",
                shard_index, tmp_path, num_docs, fde_dim,
            )

    def open_for_upload(
        self, shard_index: int, num_docs: int, fde_dim: int
    ) -> np.memmap:
        """사전 확보된 mmap 을 반환. shape/선할당 여부를 검증."""
        expected = self._expected.get(shard_index)
        if expected != (num_docs, fde_dim):
            raise ValueError(
                f"upload header ({num_docs},{fde_dim}) != "
                f"preallocated {expected} for shard={shard_index}"
            )
        with self._lock:
            mm = self._mmaps.get(shard_index)
        if mm is None:
            raise ValueError(
                f"shard={shard_index} not preallocated — "
                f"GetShard assignment must precede upload"
            )
        if mm.shape != (num_docs, fde_dim):
            raise ValueError(
                f"mmap shape mismatch shard={shard_index}: "
                f"got {mm.shape}, expected ({num_docs}, {fde_dim})"
            )
        return mm

    def get_or_create(
        self, shard_index: int, num_docs: int, fde_dim: int
    ) -> np.memmap:
        """하위 호환 — preallocate + open_for_upload."""
        self.preallocate(shard_index, num_docs, fde_dim)
        return self.open_for_upload(shard_index, num_docs, fde_dim)

    def finalize(self, shard_index: int) -> float:
        """flush → mmap close → fsync → atomic rename. 소요시간(s) 반환."""
        with self._lock:
            mm = self._mmaps.pop(shard_index, None)
        self._expected.pop(shard_index, None)

        tmp_path = getattr(mm, "filename", None) if mm is not None else None
        t0 = time.perf_counter()
        if mm is not None:
            self._sync_memmap(mm)
            del mm

        if tmp_path:
            fd = os.open(tmp_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            final_path = self._final_path(shard_index)
            os.replace(tmp_path, final_path)
            logger.info(
                "[FdeStore] Finalized shard=%d → %s", shard_index, final_path
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "[FdeStore] Finalized shard=%d  flush+close+rename=%.4fs",
            shard_index, elapsed,
        )
        return elapsed

    def flush(self, shard_index: int) -> float:
        """하위 호환 alias."""
        return self.finalize(shard_index)

    def discard(self, shard_index: int) -> None:
        """검증 실패 등으로 부분 수신된 tmp mmap 을 제거한다 (committed 파일 유지)."""
        with self._lock:
            mm = self._mmaps.pop(shard_index, None)
        self._expected.pop(shard_index, None)
        if mm is not None:
            path = getattr(mm, "filename", None)
            del mm
            if path and os.path.isfile(path):
                try:
                    os.remove(path)
                    logger.warning(
                        "[FdeStore] Discarded partial shard=%d path=%s",
                        shard_index, path,
                    )
                except OSError as exc:
                    logger.warning(
                        "[FdeStore] discard remove failed shard=%d: %s",
                        shard_index, exc,
                    )
            return

        tmp_path = self._tmp_path(shard_index)
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
                logger.warning(
                    "[FdeStore] Discarded orphan tmp shard=%d path=%s",
                    shard_index, tmp_path,
                )
            except OSError as exc:
                logger.warning(
                    "[FdeStore] discard tmp failed shard=%d: %s", shard_index, exc
                )

    def discard_partial(self, shard_index: int) -> None:
        """lease 만료 시 진행 중 업로드만 정리."""
        self.discard(shard_index)

    def mmap_path(self, shard_index: int) -> str:
        return self._final_path(shard_index)


# ===========================================================================
# Corpus 로더
# ===========================================================================

def _load_corpus_json(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return {item["id"]: item for item in raw}
    return raw


def _load_corpus_jsonl(path: str) -> Dict[str, dict]:
    corpus: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc.pop("_id", doc.pop("id", None))
            if doc_id is None:
                raise ValueError(f"corpus.jsonl 문서에 _id/id 없음: {path}")
            corpus[str(doc_id)] = {
                "title": doc.get("title", ""),
                "text":  doc.get("text", ""),
            }
    return corpus


def load_corpus(source: str) -> Dict[str, dict]:
    """
    corpus 소스를 로드한다.

    지원 형식
    ---------
    - 단일 JSON 파일: {doc_id: {title, text}} 또는 [{id, title, text}, ...]
    - BEIR 디렉터리: corpus.jsonl 또는 corpus.json 포함
    """
    if os.path.isfile(source):
        if source.endswith(".jsonl"):
            return _load_corpus_jsonl(source)
        return _load_corpus_json(source)

    if os.path.isdir(source):
        jsonl_path = os.path.join(source, "corpus.jsonl")
        json_path  = os.path.join(source, "corpus.json")
        if os.path.isfile(jsonl_path):
            return _load_corpus_jsonl(jsonl_path)
        if os.path.isfile(json_path):
            return _load_corpus_json(json_path)
        raise FileNotFoundError(
            f"BEIR corpus 디렉터리에 corpus.jsonl/corpus.json 없음: {source}"
        )

    raise FileNotFoundError(f"corpus 소스 없음: {source}")


class CorpusLoader:
    """corpus JSON 파일 또는 BEIR dataset 디렉터리에서 문서 원문을 제공한다."""

    def __init__(self, corpus_source: str) -> None:
        logger.info("[Corpus] Loading %s ...", corpus_source)
        self._corpus = load_corpus(corpus_source)
        self.corpus_source = corpus_source
        self.doc_ids: List[str] = list(self._corpus.keys())
        logger.info("[Corpus] Loaded %d documents from %s", len(self.doc_ids), corpus_source)

    def total_docs(self) -> int:
        return len(self.doc_ids)

    def get_doc(self, pos: int) -> tuple[str, dict]:
        doc_id = self.doc_ids[pos]
        return doc_id, self._corpus[doc_id]

    def save_doc_ids(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f, ensure_ascii=False, indent=2)
        logger.info("[Corpus] doc_ids saved → %s", path)


# ===========================================================================
# FDE config 검증 (Worker 간 일치)
# ===========================================================================

def _fde_config_signature(cfg: pb2.FdeConfig) -> tuple:
    return (
        cfg.dimension,
        cfg.num_repetitions,
        cfg.num_simhash_projections,
        cfg.seed,
        cfg.encoding_type,
        cfg.projection_type,
        cfg.projection_dimension,
        cfg.fill_empty_partitions,
        cfg.final_projection_dimension,
    )


def validate_fde_config_proto(cfg: pb2.FdeConfig) -> Optional[str]:
    if cfg.dimension <= 0:
        return f"fde_config.dimension must be > 0, got {cfg.dimension}"
    if cfg.num_repetitions <= 0:
        return f"fde_config.num_repetitions must be > 0, got {cfg.num_repetitions}"
    if not (1 <= cfg.num_simhash_projections <= MAX_SIMHASH_PROJECTIONS):
        return (
            f"fde_config.num_simhash_projections must be 1.."
            f"{MAX_SIMHASH_PROJECTIONS}, got {cfg.num_simhash_projections}"
        )
    return None


_PROJECTION_DEFAULT_IDENTITY = 0


def fde_output_dim_from_proto(cfg: pb2.FdeConfig) -> int:
    """Worker fde_output_dim() 과 동일한 출력 차원 계산."""
    proj_dim = (
        cfg.dimension
        if cfg.projection_type == _PROJECTION_DEFAULT_IDENTITY
        else cfg.projection_dimension
    )
    if cfg.final_projection_dimension > 0:
        return cfg.final_projection_dimension
    return cfg.num_repetitions * (2 ** cfg.num_simhash_projections) * proj_dim


# ===========================================================================
# FDE 업로드 검증
# ===========================================================================

def _upload_ack_error(
    shard_index: int,
    message: str,
    *,
    bytes_received: int = 0,
    grpc_recv_time: float = 0.0,
) -> pb2.UploadAck:
    logger.error("[Upload] shard=%d ERROR: %s", shard_index, message)
    return pb2.UploadAck(
        shard_index          = shard_index,
        status               = "error",
        message              = message,
        remote_flush_time_s  = 0.0,
        grpc_transfer_time_s = grpc_recv_time,
        bytes_received       = bytes_received,
    )


# ===========================================================================
# gRPC Servicer
# ===========================================================================

class ShardServicer(pb2_grpc.ShardServiceServicer):

    def __init__(
        self,
        corpus: CorpusLoader,
        dispatcher: ShardDispatcher,
        fde_store: FdeStore,
        output_dir: str,
    ) -> None:
        self._corpus     = corpus
        self._dispatcher = dispatcher
        self._store      = fde_store
        self._output_dir = output_dir
        self._status_log: List[dict] = []
        self._log_lock   = threading.Lock()
        self._canonical_fde_config: Optional[pb2.FdeConfig] = None
        self._fde_config_lock = threading.Lock()
        # shard별 Storage Node wall-clock (time.time())
        self._shard_wall_ts: Dict[int, dict] = {}
        self._wall_ts_lock = threading.Lock()
        self._pipeline_wall_ts: Dict[str, Optional[float]] = {
            "first_get_shard_recv_ts": None,
            "last_status_recv_ts":     None,
        }

    def _record_shard_wall_ts(self, shard_idx: int, **fields: float) -> None:
        with self._wall_ts_lock:
            self._shard_wall_ts.setdefault(shard_idx, {}).update(fields)

    def _get_shard_wall_ts(self, shard_idx: int) -> dict:
        with self._wall_ts_lock:
            return dict(self._shard_wall_ts.get(shard_idx, {}))

    @staticmethod
    def _build_timing_from_status(request: pb2.ShardStatus, wall_ts: dict) -> dict:
        """ShardStatus duration + wall-clock timestamps → timing dict."""
        storage_status_recv_ts = wall_ts.get("storage_status_recv_ts", 0.0)
        storage_get_shard_recv_ts = wall_ts.get("storage_get_shard_recv_ts", 0.0)

        timing = {
            "prep_time":        request.prep_time_s,
            "upload_time":      request.upload_time_s,
            "simhash_time":     request.simhash_time_s,
            "partition_time":   request.partition_time_s,
            "scatter_time":     request.scatter_time_s,
            "average_time":     request.average_time_s,
            "fill_time":        request.fill_time_s,
            "compute_time":     request.compute_time_s,
            "download_time":    request.download_time_s,
            "reshape_time":     request.reshape_time_s,
            "flush_time":       request.flush_time_s,
            "fde_total":        request.fde_total_time_s,
            "embed_time":       request.embed_time_s,
            "grpc_recv":        request.grpc_recv_time_s,
            "grpc_send":        request.grpc_send_time_s,
            "remote_flush":     request.remote_flush_time_s,
            "end_to_end":       request.end_to_end_time_s,
            # Worker wall-clock (time.time())
            "worker_get_shard_req_ts":  request.worker_get_shard_req_ts,
            "worker_get_shard_done_ts": request.worker_get_shard_done_ts,
            "worker_process_start_ts":  request.worker_process_start_ts,
            "worker_process_done_ts":   request.worker_process_done_ts,
            "worker_upload_req_ts":     request.worker_upload_req_ts,
            "worker_upload_done_ts":    request.worker_upload_done_ts,
            "worker_status_report_ts":  request.worker_status_report_ts,
            # Storage Node wall-clock (time.time())
            **wall_ts,
        }

        w_req = request.worker_get_shard_req_ts
        w_end = request.worker_status_report_ts
        if w_req > 0 and w_end > 0:
            timing["worker_wall_total_s"] = w_end - w_req

        if storage_get_shard_recv_ts > 0 and storage_status_recv_ts > 0:
            timing["storage_shard_wall_s"] = (
                storage_status_recv_ts - storage_get_shard_recv_ts
            )

        return timing

    def _canonical_fde_dim(self) -> Optional[int]:
        with self._fde_config_lock:
            if self._canonical_fde_config is None:
                return None
            return fde_output_dim_from_proto(self._canonical_fde_config)

    def _ensure_shard_preallocated(self, shard_idx: int) -> None:
        """GetShard 배정 시 해당 shard mmap 만 선할당."""
        fde_dim = self._canonical_fde_dim()
        if fde_dim is None:
            raise RuntimeError("FDE config not established")
        num_docs = self._dispatcher.expected_num_docs(shard_idx)
        self._store.preallocate(shard_idx, num_docs, fde_dim)

    def _validate_upload_assignment(
        self,
        shard_index: int,
        worker_id: str,
        num_docs: int,
        fde_dim: int,
    ) -> Optional[str]:
        if self._canonical_fde_config is None:
            return "FDE config not established — GetShard must precede upload"

        status = self._dispatcher.shard_status(shard_index)
        if status != ShardDispatcher.STATUS_INPROGRESS:
            return (
                f"shard={shard_index} not inprogress (status={status!r})"
            )

        assigned = self._dispatcher.assigned_worker(shard_index)
        if assigned != worker_id:
            return (
                f"shard={shard_index} assigned to {assigned!r}, "
                f"not {worker_id!r}"
            )

        expected_docs = self._dispatcher.expected_num_docs(shard_index)
        if num_docs != expected_docs:
            return (
                f"num_docs mismatch: got {num_docs}, "
                f"expected {expected_docs} for shard={shard_index}"
            )

        expected_dim = fde_output_dim_from_proto(self._canonical_fde_config)
        if fde_dim != expected_dim:
            return (
                f"fde_dim mismatch: got {fde_dim}, expected {expected_dim}"
            )

        return None

    def _validate_worker_fde_config(
        self, request: pb2.ShardRequest, context: grpc.ServicerContext
    ) -> None:
        cfg = request.fde_config
        err = validate_fde_config_proto(cfg)
        if err:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, err)
            return

        sig = _fde_config_signature(cfg)
        with self._fde_config_lock:
            if self._canonical_fde_config is None:
                self._canonical_fde_config = pb2.FdeConfig()
                self._canonical_fde_config.CopyFrom(cfg)
                logger.info("[FDE config] canonical established: %s", sig)
            elif _fde_config_signature(self._canonical_fde_config) != sig:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "FDE config mismatch between workers",
                )
                return

    # ------------------------------------------------------------------ #
    # RPC: GetShard
    # ------------------------------------------------------------------ #

    def GetShard(
        self, request: pb2.ShardRequest, context: grpc.ServicerContext
    ) -> Iterator[pb2.DocumentChunk]:
        worker_id = request.worker_id
        t_get_shard_recv = time.time()
        if self._pipeline_wall_ts["first_get_shard_recv_ts"] is None:
            self._pipeline_wall_ts["first_get_shard_recv_ts"] = t_get_shard_recv

        self._validate_worker_fde_config(request, context)
        self._dispatcher.expire_leases()

        # 이전 shard 완료/실패 처리
        if request.HasField("completed") and request.completed.shard_index >= 0:
            ci = request.completed
            if ci.status == "success":
                logger.info(
                    "[GetShard] worker=%s reports shard=%d SUCCESS",
                    worker_id, ci.shard_index,
                )
            else:
                logger.warning(
                    "[GetShard] worker=%s reports shard=%d FAILURE (already handled via ReportShardStatus): %s",
                    worker_id, ci.shard_index, ci.error_message,
                )

        # 다음 shard 배정
        shard_idx = self._dispatcher.next_pending(worker_id)
        if shard_idx is None:
            # 더 이상 배정할 shard 없음 → 빈 스트림 종료
            logger.info(
                "[GetShard] worker=%s: no more shards to assign", worker_id
            )
            return

        doc_start, doc_end = self._dispatcher.shard_range(shard_idx)
        total_shards = len(self._dispatcher._shards)
        num_docs_in_shard = doc_end - doc_start

        logger.info(
            "[GetShard] Assigning shard=%d docs=[%d,%d) to worker=%s",
            shard_idx, doc_start, doc_end, worker_id,
        )

        self._record_shard_wall_ts(
            shard_idx, storage_get_shard_recv_ts=t_get_shard_recv
        )

        try:
            self._ensure_shard_preallocated(shard_idx)  # shard_{N}.mmap.tmp
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "[GetShard] preallocate failed shard=%d: %s", shard_idx, exc
            )
            self._dispatcher.report_failure(shard_idx, str(exc))
            context.abort(grpc.StatusCode.INTERNAL, f"preallocate failed: {exc}")
            return

        # 문서 원문(title, text)만 streaming 전송 — embedding 은 Worker 가 생성
        for local_idx, pos in enumerate(range(doc_start, doc_end)):
            doc_id, doc = self._corpus.get_doc(pos)
            yield pb2.DocumentChunk(
                shard_index        = shard_idx,
                total_shards       = total_shards,
                doc_start          = doc_start,
                doc_end            = doc_end,
                total_docs         = num_docs_in_shard,
                chunk_index        = 0,
                total_chunks       = 1,
                is_last            = (local_idx == num_docs_in_shard - 1),
                doc_index_in_shard = local_idx,
                seq_len            = 0,
                embed_dim          = 0,
                embedding_data     = b"",
                doc_id             = doc_id,
                corpus_title       = doc.get("title", ""),
                corpus_text        = doc.get("text", ""),
            )

        t_get_shard_send_done = time.time()
        self._record_shard_wall_ts(
            shard_idx, storage_get_shard_send_done_ts=t_get_shard_send_done
        )

        logger.info(
            "[GetShard] Sent shard=%d (%d docs) to worker=%s  "
            "wall_send=%.3fs",
            shard_idx, num_docs_in_shard, worker_id,
            t_get_shard_send_done - t_get_shard_recv,
        )

    # ------------------------------------------------------------------ #
    # RPC: UploadFdeShard
    # ------------------------------------------------------------------ #

    def UploadFdeShard(
        self,
        request_iterator: Iterator[pb2.FdeChunk],
        context: grpc.ServicerContext,
    ) -> pb2.UploadAck:
        self._dispatcher.expire_leases()
        recv_start     = time.perf_counter()
        shard_index    = -1
        worker_id      = ""
        mm             = None
        bytes_received = 0
        embed_time     = 0.0
        fde_time       = 0.0

        num_docs         = 0
        fde_dim          = 0
        total_chunks     = 0
        chunk_indices: set[int] = set()
        row_bitmap: Optional[bytearray] = None
        saw_is_last      = False
        upload_error     = ""

        # gRPC chunk 수신 → row offset mmap 기록
        for chunk in request_iterator:
            if upload_error:
                break

            if mm is None:
                shard_index  = chunk.shard_index
                worker_id    = chunk.worker_id
                num_docs     = chunk.num_docs
                fde_dim      = chunk.fde_dim
                total_chunks = chunk.total_chunks

                if num_docs <= 0 or fde_dim <= 0:
                    upload_error = f"invalid header num_docs={num_docs} fde_dim={fde_dim}"
                    break
                if total_chunks <= 0:
                    upload_error = f"invalid total_chunks={total_chunks}"
                    break

                header_err = self._validate_upload_assignment(
                    shard_index, worker_id, num_docs, fde_dim
                )
                if header_err:
                    upload_error = header_err
                    break

                try:
                    mm = self._store.open_for_upload(shard_index, num_docs, fde_dim)
                except ValueError as exc:
                    upload_error = str(exc)
                    break

                self._record_shard_wall_ts(
                    shard_index,
                    storage_upload_recv_start_ts=time.time(),
                )
                self._dispatcher.touch_lease(shard_index)
                row_bitmap = bytearray(num_docs)
                logger.info(
                    "[Upload] Receiving shard=%d worker=%s num_docs=%d fde_dim=%d "
                    "total_chunks=%d",
                    shard_index, worker_id, num_docs, fde_dim, total_chunks,
                )
            else:
                if chunk.shard_index != shard_index:
                    upload_error = (
                        f"shard_index mismatch: {chunk.shard_index} != {shard_index}"
                    )
                    break
                if chunk.num_docs != num_docs or chunk.fde_dim != fde_dim:
                    upload_error = "num_docs/fde_dim changed mid-stream"
                    break
                if chunk.total_chunks != total_chunks:
                    upload_error = "total_chunks changed mid-stream"
                    break

            if chunk.chunk_index in chunk_indices:
                upload_error = f"duplicate chunk_index={chunk.chunk_index}"
                break
            if not (0 <= chunk.chunk_index < total_chunks):
                upload_error = (
                    f"chunk_index out of range: {chunk.chunk_index} "
                    f"(total_chunks={total_chunks})"
                )
                break
            chunk_indices.add(chunk.chunk_index)

            row_count = chunk.row_end - chunk.row_start
            if row_count < 0:
                upload_error = (
                    f"invalid row range [{chunk.row_start},{chunk.row_end})"
                )
                break
            if chunk.row_end > num_docs:
                upload_error = (
                    f"row_end {chunk.row_end} exceeds num_docs {num_docs}"
                )
                break

            if len(chunk.fde_data) == 0:
                if row_count > 0:
                    upload_error = (
                        f"empty fde_data for rows [{chunk.row_start},{chunk.row_end})"
                    )
                    break
            else:
                expected_bytes = row_count * fde_dim * 4
                if len(chunk.fde_data) != expected_bytes:
                    upload_error = (
                        f"chunk {chunk.chunk_index} byte mismatch: "
                        f"got {len(chunk.fde_data)}, expected {expected_bytes}"
                    )
                    break
                for r in range(chunk.row_start, chunk.row_end):
                    if row_bitmap[r]:
                        upload_error = f"overlapping row {r} in chunk {chunk.chunk_index}"
                        break
                    row_bitmap[r] = 1
                if upload_error:
                    break
                arr = np.frombuffer(chunk.fde_data, dtype=np.float32).reshape(
                    row_count, fde_dim
                )
                mm[chunk.row_start : chunk.row_end, :] = arr
                bytes_received += len(chunk.fde_data)

            if chunk.is_last:
                saw_is_last = True
                embed_time = chunk.embed_time_s
                fde_time   = chunk.fde_time_s
                self._dispatcher.touch_lease(shard_index)

        grpc_recv_time = time.perf_counter() - recv_start

        if upload_error:
            if shard_index >= 0:
                self._store.discard(shard_index)
            return _upload_ack_error(
                shard_index, upload_error,
                bytes_received=bytes_received, grpc_recv_time=grpc_recv_time,
            )

        if mm is None:
            return _upload_ack_error(
                -1, "Empty stream", grpc_recv_time=grpc_recv_time,
            )

        expected_bytes = num_docs * fde_dim * 4
        # chunk completeness 검증
        if not saw_is_last:
            self._store.discard(shard_index)
            return _upload_ack_error(
                shard_index, "missing is_last chunk",
                bytes_received=bytes_received, grpc_recv_time=grpc_recv_time,
            )
        if len(chunk_indices) != total_chunks:
            self._store.discard(shard_index)
            return _upload_ack_error(
                shard_index,
                f"incomplete chunks: got {len(chunk_indices)}/{total_chunks}",
                bytes_received=bytes_received, grpc_recv_time=grpc_recv_time,
            )
        if chunk_indices != set(range(total_chunks)):
            self._store.discard(shard_index)
            return _upload_ack_error(
                shard_index, "non-contiguous chunk_index set",
                bytes_received=bytes_received, grpc_recv_time=grpc_recv_time,
            )
        if row_bitmap is None or not all(row_bitmap):
            self._store.discard(shard_index)
            missing = 0 if row_bitmap is None else num_docs - sum(row_bitmap)
            return _upload_ack_error(
                shard_index,
                f"incomplete row coverage: {missing} rows missing",
                bytes_received=bytes_received, grpc_recv_time=grpc_recv_time,
            )
        if bytes_received != expected_bytes:
            self._store.discard(shard_index)
            return _upload_ack_error(
                shard_index,
                f"bytes_received mismatch: got {bytes_received}, "
                f"expected {expected_bytes}",
                bytes_received=bytes_received, grpc_recv_time=grpc_recv_time,
            )

        # flush + fsync + atomic rename → ACK
        remote_flush_time = self._store.finalize(shard_index)
        self._record_shard_wall_ts(
            shard_index, storage_upload_done_ts=time.time()
        )

        logger.info(
            "[Upload] shard=%d DONE  grpc_recv=%.4fs  remote_flush=%.4fs  "
            "bytes=%d  embed=%.4fs  fde=%.4fs",
            shard_index,
            grpc_recv_time,
            remote_flush_time,
            bytes_received,
            embed_time,
            fde_time,
        )

        # [PERF LOG] Storage Node 수신 + flush
        storage_total = grpc_recv_time + remote_flush_time
        logger.info(
            "[PERF] shard=%d | storage_recv=%.4fs | remote_flush+fsync=%.4fs | "
            "storage_total=%.4fs",
            shard_index, grpc_recv_time, remote_flush_time, storage_total,
        )

        return pb2.UploadAck(
            shard_index          = shard_index,
            status               = "ok",
            message              = "Stored and flushed",
            remote_flush_time_s  = remote_flush_time,
            grpc_transfer_time_s = grpc_recv_time,
            bytes_received       = bytes_received,
        )

    # ------------------------------------------------------------------ #
    # RPC: ReportShardStatus
    # ------------------------------------------------------------------ #

    def ReportShardStatus(
        self, request: pb2.ShardStatus, context: grpc.ServicerContext
    ) -> pb2.StatusAck:
        self._dispatcher.expire_leases()

        t_status_recv = time.time()
        self._pipeline_wall_ts["last_status_recv_ts"] = t_status_recv
        wall_ts = self._get_shard_wall_ts(request.shard_index)
        wall_ts["storage_status_recv_ts"] = t_status_recv
        timing = self._build_timing_from_status(request, wall_ts)

        if request.status == "success":
            self._dispatcher.mark_done(request.shard_index, timing)
        else:
            self._dispatcher.report_failure(request.shard_index, request.error_message)
            self._store.discard_partial(request.shard_index)

        with self._log_lock:
            self._status_log.append({
                "shard_index": request.shard_index,
                "worker_id":   request.worker_id,
                "status":      request.status,
                "num_docs":    request.num_docs,
                "timing":      timing,
            })

        summary = self._dispatcher.summary()
        logger.info(
            "[Status] shard=%d worker=%s status=%s  "
            "progress=%d/%d  end_to_end=%.3fs  fde=%.3fs  "
            "grpc_send=%.3fs  remote_flush=%.3fs",
            request.shard_index,
            request.worker_id,
            request.status,
            summary["done"] + summary["failed"],
            summary["total"],
            request.end_to_end_time_s,
            request.fde_total_time_s,
            request.grpc_send_time_s,
            request.remote_flush_time_s,
        )

        # 모든 shard 완료 시 보고서 생성
        if self._dispatcher.all_done():
            logger.info("[Status] All shards done — generating reports ...")
            self._write_reports()

        return pb2.StatusAck(status="ok", message="recorded")

    # ------------------------------------------------------------------ #
    # RPC: GetProgress
    # ------------------------------------------------------------------ #

    def GetProgress(
        self, request: pb2.ProgressRequest, context: grpc.ServicerContext
    ) -> pb2.ProgressResponse:
        summary = self._dispatcher.summary()
        summaries = []
        for idx, st in self._dispatcher._status.items():
            t = self._dispatcher._timing.get(idx, {})
            summaries.append(
                pb2.ShardSummary(
                    shard_index = idx,
                    status      = st,
                    worker_id   = self._dispatcher._worker.get(idx, ""),
                    num_docs    = (
                        self._dispatcher.shard_range(idx)[1]
                        - self._dispatcher.shard_range(idx)[0]
                    ),
                    total_time_s= t.get("end_to_end", 0.0),
                )
            )
        return pb2.ProgressResponse(
            total_shards     = summary["total"],
            completed_shards = summary["done"],
            failed_shards    = summary["failed"],
            pending_shards   = summary["pending"],
            elapsed_sec      = summary["elapsed_sec"],
            shard_summaries  = summaries,
        )

    # ------------------------------------------------------------------ #
    # 최종 보고서 생성
    # ------------------------------------------------------------------ #

    def _write_reports(self) -> None:
        summary = self._dispatcher.summary()
        logs    = self._status_log[:]

        pts = self._pipeline_wall_ts
        pipeline_wall_total_s = None
        if pts["first_get_shard_recv_ts"] and pts["last_status_recv_ts"]:
            pipeline_wall_total_s = round(
                pts["last_status_recv_ts"] - pts["first_get_shard_recv_ts"], 3
            )

        # ---- final_manifest.json ----
        manifest = {
            "generated_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_shards":         summary["total"],
            "done":                 summary["done"],
            "failed":               summary["failed"],
            "total_elapsed_sec":    round(summary["elapsed_sec"], 3),
            "pipeline_wall_clock": {
                "first_get_shard_recv_ts": pts["first_get_shard_recv_ts"],
                "last_status_recv_ts":     pts["last_status_recv_ts"],
                "pipeline_wall_total_s":   pipeline_wall_total_s,
            },
            "shards": [
                {
                    "shard_index":  e["shard_index"],
                    "worker_id":    e["worker_id"],
                    "status":       e["status"],
                    "num_docs":     e["num_docs"],
                    "mmap_path":    self._store.mmap_path(e["shard_index"]),
                    "timing":       e["timing"],
                }
                for e in sorted(logs, key=lambda x: x["shard_index"])
            ],
        }
        manifest_path = os.path.join(self._output_dir, "final_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info("[Report] final_manifest.json → %s", manifest_path)

        # ---- metrics.csv ----
        csv_path = os.path.join(self._output_dir, "metrics.csv")
        fields = [
            "shard_index", "worker_id", "status", "num_docs",
            "embed_time", "fde_total", "grpc_send", "remote_flush",
            "end_to_end", "worker_wall_total_s", "storage_shard_wall_s",
            "worker_get_shard_req_ts", "worker_status_report_ts",
            "storage_get_shard_recv_ts", "storage_status_recv_ts",
            "prep_time", "upload_time", "compute_time", "download_time",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for e in sorted(logs, key=lambda x: x["shard_index"]):
                t = e["timing"]
                row = {
                    "shard_index":  e["shard_index"],
                    "worker_id":    e["worker_id"],
                    "status":       e["status"],
                    "num_docs":     e["num_docs"],
                    "embed_time":   f"{t.get('embed_time', 0):.4f}",
                    "fde_total":    f"{t.get('fde_total', 0):.4f}",
                    "grpc_send":    f"{t.get('grpc_send', 0):.4f}",
                    "remote_flush": f"{t.get('remote_flush', 0):.4f}",
                    "end_to_end":   f"{t.get('end_to_end', 0):.4f}",
                    "worker_wall_total_s":  f"{t.get('worker_wall_total_s', 0):.4f}",
                    "storage_shard_wall_s": f"{t.get('storage_shard_wall_s', 0):.4f}",
                    "worker_get_shard_req_ts":  f"{t.get('worker_get_shard_req_ts', 0):.3f}",
                    "worker_status_report_ts":  f"{t.get('worker_status_report_ts', 0):.3f}",
                    "storage_get_shard_recv_ts":f"{t.get('storage_get_shard_recv_ts', 0):.3f}",
                    "storage_status_recv_ts":   f"{t.get('storage_status_recv_ts', 0):.3f}",
                    "prep_time":    f"{t.get('prep_time', 0):.4f}",
                    "upload_time":  f"{t.get('upload_time', 0):.4f}",
                    "compute_time": f"{t.get('compute_time', 0):.4f}",
                    "download_time":f"{t.get('download_time', 0):.4f}",
                }
                w.writerow(row)
        logger.info("[Report] metrics.csv → %s", csv_path)


# ===========================================================================
# Lease reaper (백그라운드)
# ===========================================================================

def _lease_reaper_loop(
    dispatcher: ShardDispatcher,
    fde_store: FdeStore,
    interval_sec: float,
) -> None:
    while True:
        time.sleep(interval_sec)
        expired = dispatcher.expire_leases()
        for idx in expired:
            fde_store.discard_partial(idx)


# ===========================================================================
# 서버 진입점
# ===========================================================================


def serve(args: argparse.Namespace) -> None:
    # ── config 생성: CLI 인자가 없는 항목은 storage_node_config.py 기본값 사용
    cfg = StorageNodeConfig.from_args(args)
    cfg.validate()

    logging.basicConfig(level=getattr(logging, cfg.log_level, logging.INFO))

    corpus     = CorpusLoader(cfg.corpus_path)
    shards     = compute_shards(corpus.total_docs(), cfg.shard_doc_size)
    dispatcher = ShardDispatcher(
        shards,
        lease_timeout_sec=cfg.shard_lease_timeout_sec,
        max_attempts=cfg.shard_max_attempts,
    )
    fde_store  = FdeStore(cfg.output_dir)

    os.makedirs(cfg.output_dir, exist_ok=True)
    corpus.save_doc_ids(os.path.join(cfg.output_dir, "doc_ids.json"))

    reaper = threading.Thread(
        target=_lease_reaper_loop,
        args=(dispatcher, fde_store, cfg.lease_reaper_interval_sec),
        daemon=True,
        name="lease-reaper",
    )
    reaper.start()

    servicer = ShardServicer(corpus, dispatcher, fde_store, cfg.output_dir)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=cfg.max_workers),
        options=cfg.grpc_server_options(),
    )
    pb2_grpc.add_ShardServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{cfg.port}")
    server.start()

    logger.info(
        "Storage Node started: project=%s  dataset=%s  port=%d  shards=%d  "
        "shard_doc_size=%d  docs=%d  corpus=%s  output=%s  max_workers=%d  "
        "grpc_max_msg=%d  lease_timeout=%.0fs  max_attempts=%d",
        cfg.project_dir, cfg.dataset, cfg.port, len(shards),
        cfg.shard_doc_size, corpus.total_docs(),
        cfg.corpus_path, cfg.output_dir, cfg.max_workers,
        cfg.grpc_max_message_bytes,
        cfg.shard_lease_timeout_sec, cfg.shard_max_attempts,
    )

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Storage Node shutting down ...")
        server.stop(grace=cfg.server_stop_grace_sec)


def parse_args() -> argparse.Namespace:
    """
    CLI 파서. 기본값은 StorageNodeConfig 에서 가져오므로
    --help 실행 시 config 파일 값이 표시된다.
    """
    _defaults = StorageNodeConfig()
    p = argparse.ArgumentParser(
        description="FDE Storage Node (gRPC server) — 기본값은 storage_node_config.py 참고",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project-dir", default=_defaults.project_dir,
                   help="프로젝트 루트 (dccblue: /data/muvera_optimized)")
    p.add_argument("--datasets-root", default=_defaults.datasets_root,
                   help="dataset 상위 루트 (기본: /data/datasets)")
    p.add_argument("--dataset-collection", default=_defaults.dataset_collection,
                   help="dataset 컬렉션 디렉터리 (기본: flare)")
    p.add_argument("--dataset", default=_defaults.dataset,
                   help="데이터셋 이름 → corpus: "
                        "{datasets_root}/{collection}/{dataset}/corpus.jsonl, "
                        "output: {project_dir}/data/fde_out/{dataset}/")
    p.add_argument("--corpus-path", default=None,
                   help="corpus 직접 지정 (JSON 파일 또는 BEIR 디렉터리). "
                        "지정 시 --dataset 보다 우선")
    p.add_argument("--output-dir",  default=None,
                   help="FDE 출력 디렉터리. 지정 시 --dataset 보다 우선")
    p.add_argument("--shard-doc-size", type=int, default=_defaults.shard_doc_size,
                   help="GetShard 1회당 전송 문서 수 (기본 10000, 권장 8000~12000)")
    p.add_argument("--shard-lease-timeout-sec", type=float,
                   default=_defaults.shard_lease_timeout_sec,
                   help="Worker 장애 시 inprogress shard lease 만료(초)")
    p.add_argument("--shard-max-attempts", type=int,
                   default=_defaults.shard_max_attempts,
                   help="shard 재시도 상한 (초과 시 permanent failed)")
    p.add_argument("--lease-reaper-interval-sec", type=float,
                   default=_defaults.lease_reaper_interval_sec,
                   help="백그라운드 lease reaper 주기(초)")
    p.add_argument("--port",        type=int, default=_defaults.port,
                   help="gRPC 바인딩 포트")
    p.add_argument("--max-workers", type=int, default=_defaults.max_workers,
                   help="gRPC ThreadPoolExecutor 크기")
    return p.parse_args()


if __name__ == "__main__":
    serve(parse_args())
