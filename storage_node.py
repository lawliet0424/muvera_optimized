# -*- coding: utf-8 -*-
"""
역할
----
1. corpus 를 shard 로 분할하고 Worker 요청에 따라 streaming 전송
2. Worker 로부터 FDE shard 를 streaming 수신 → mmap 에 저장 → flush
3. shard 처리 상태 / timing 을 집계하고 진행 현황 조회 서비스 제공
4. 모든 shard 완료 후 final_manifest.json / metrics.csv 생성

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

def compute_shards(total_docs: int, num_shards: int) -> list[tuple[int, int]]:
    """corpus 를 균등하게 shard 로 분할."""
    actual = min(num_shards, total_docs)
    size = math.ceil(total_docs / actual)
    shards = []
    for i in range(actual):
        s = i * size
        e = min(s + size, total_docs)
        if s >= total_docs:
            break
        shards.append((s, e))
    return shards


# ===========================================================================
# Shard 배포 상태 관리
# ===========================================================================

class ShardDispatcher:
    """Thread-safe shard 배포 및 상태 추적."""

    STATUS_PENDING   = "pending"
    STATUS_INPROGRESS= "inprogress"
    STATUS_DONE      = "done"
    STATUS_FAILED    = "failed"

    def __init__(self, shards: list[tuple[int, int]]) -> None:
        self._lock   = threading.Lock()
        self._shards = shards                       # (doc_start, doc_end)
        self._status: Dict[int, str]    = {i: self.STATUS_PENDING for i in range(len(shards))}
        self._worker: Dict[int, str]    = {}
        self._timing: Dict[int, dict]   = {}
        self._start_time = time.time()

    # ------------------------------------------------------------------ #

    def next_pending(self, worker_id: str) -> Optional[int]:
        """pending 상태인 첫 번째 shard index 를 반환 (없으면 None)."""
        with self._lock:
            for idx, st in self._status.items():
                if st == self.STATUS_PENDING:
                    self._status[idx] = self.STATUS_INPROGRESS
                    self._worker[idx] = worker_id
                    return idx
        return None

    def mark_done(self, idx: int, timing: dict) -> None:
        with self._lock:
            self._status[idx] = self.STATUS_DONE
            self._timing[idx] = timing

    def mark_failed(self, idx: int, error: str) -> None:
        with self._lock:
            self._status[idx] = self.STATUS_FAILED
            self._timing[idx] = {"error": error}

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


# ===========================================================================
# FDE 저장소 (per-shard mmap)
# ===========================================================================

class FdeStore:
    """수신된 FDE 청크를 shard mmap 에 기록하고 flush."""

    def __init__(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        self._lock = threading.Lock()
        self._mmaps: Dict[int, np.memmap] = {}

    def get_or_create(
        self, shard_index: int, num_docs: int, fde_dim: int
    ) -> np.memmap:
        with self._lock:
            if shard_index not in self._mmaps:
                path = os.path.join(
                    self._output_dir, f"shard_{shard_index:04d}.mmap"
                )
                mm = np.memmap(
                    path, mode="w+", dtype=np.float32, shape=(num_docs, fde_dim)
                )
                self._mmaps[shard_index] = mm
                logger.info(
                    "[FdeStore] Created mmap shard=%d path=%s shape=(%d,%d)",
                    shard_index, path, num_docs, fde_dim,
                )
            return self._mmaps[shard_index]

    def flush(self, shard_index: int) -> float:
        """flush 하고 소요시간(s)을 반환."""
        with self._lock:
            mm = self._mmaps.get(shard_index)
        if mm is None:
            return 0.0
        t0 = time.perf_counter()
        mm.flush()
        elapsed = time.perf_counter() - t0
        logger.info(
            "[FdeStore] Flushed shard=%d  flush=%.4fs", shard_index, elapsed
        )
        return elapsed

    def mmap_path(self, shard_index: int) -> str:
        return os.path.join(
            self._output_dir, f"shard_{shard_index:04d}.mmap"
        )


# ===========================================================================
# Corpus 로더
# ===========================================================================

class CorpusLoader:
    """corpus.json 에서 문서 원문(title, text)을 제공한다."""

    def __init__(self, corpus_path: str) -> None:
        logger.info("[Corpus] Loading %s ...", corpus_path)
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # corpus 는 dict{doc_id: {title, text}} 또는 list 형식 모두 허용
        if isinstance(raw, list):
            self._corpus: Dict[str, dict] = {
                item["id"]: item for item in raw
            }
        else:
            self._corpus = raw

        self.doc_ids: List[str] = list(self._corpus.keys())
        logger.info("[Corpus] Loaded %d documents", len(self.doc_ids))

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

    # ------------------------------------------------------------------ #
    # RPC: GetShard
    # ------------------------------------------------------------------ #

    def GetShard(
        self, request: pb2.ShardRequest, context: grpc.ServicerContext
    ) -> Iterator[pb2.DocumentChunk]:
        worker_id = request.worker_id

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
                    "[GetShard] worker=%s reports shard=%d FAILURE: %s",
                    worker_id, ci.shard_index, ci.error_message,
                )
                self._dispatcher.mark_failed(ci.shard_index, ci.error_message)

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

        logger.info(
            "[GetShard] Sent shard=%d (%d docs) to worker=%s",
            shard_idx, num_docs_in_shard, worker_id,
        )

    # ------------------------------------------------------------------ #
    # RPC: UploadFdeShard
    # ------------------------------------------------------------------ #

    def UploadFdeShard(
        self,
        request_iterator: Iterator[pb2.FdeChunk],
        context: grpc.ServicerContext,
    ) -> pb2.UploadAck:
        recv_start    = time.perf_counter()
        shard_index   = -1
        worker_id     = ""
        mm            = None
        bytes_received = 0
        embed_time    = 0.0
        fde_time      = 0.0

        for chunk in request_iterator:
            # 첫 청크에서 mmap 초기화
            if mm is None:
                shard_index = chunk.shard_index
                worker_id   = chunk.worker_id
                num_docs    = chunk.num_docs
                fde_dim     = chunk.fde_dim
                mm = self._store.get_or_create(shard_index, num_docs, fde_dim)
                logger.info(
                    "[Upload] Receiving shard=%d worker=%s num_docs=%d fde_dim=%d",
                    shard_index, worker_id, num_docs, fde_dim,
                )

            if len(chunk.fde_data) == 0:
                continue

            # float32 역직렬화 후 mmap 기록
            arr = np.frombuffer(chunk.fde_data, dtype=np.float32).reshape(
                chunk.row_end - chunk.row_start, -1
            )
            mm[chunk.row_start : chunk.row_end, :] = arr
            bytes_received += len(chunk.fde_data)

            if chunk.is_last:
                embed_time = chunk.embed_time_s
                fde_time   = chunk.fde_time_s

        if mm is None:
            return pb2.UploadAck(
                shard_index=-1, status="error", message="Empty stream"
            )

        grpc_recv_time = time.perf_counter() - recv_start

        # ★ Storage Node 측 flush (baseline 비교 포인트)
        remote_flush_time = self._store.flush(shard_index)

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

        # ──────────────────────────────────────────────────────────────
        # [PERF LOG] local SSD flush baseline vs. gRPC + remote flush
        # local SSD baseline: Worker 가 자체 mmap 을 flush 했다면 얼마나 걸릴까?
        # 여기서는 Storage Node 수신 속도로 추정치를 계산한다.
        estimated_local_flush = bytes_received / (500 * 1024 * 1024)  # 500 MB/s SSD 가정
        logger.info(
            "[PERF COMPARE] shard=%d | "
            "grpc_transfer+remote_flush=%.4fs | "
            "estimated_local_flush=%.4fs | "
            "grpc_overhead=%.4fs",
            shard_index,
            grpc_recv_time + remote_flush_time,
            estimated_local_flush,
            (grpc_recv_time + remote_flush_time) - estimated_local_flush,
        )
        # ──────────────────────────────────────────────────────────────

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
            "local_flush":      request.local_flush_time_s,
            "end_to_end":       request.end_to_end_time_s,
        }

        if request.status == "success":
            self._dispatcher.mark_done(request.shard_index, timing)
        else:
            self._dispatcher.mark_failed(request.shard_index, request.error_message)

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
            "grpc_send=%.3fs  remote_flush=%.3fs  local_flush=%.3fs",
            request.shard_index,
            request.worker_id,
            request.status,
            summary["done"] + summary["failed"],
            summary["total"],
            request.end_to_end_time_s,
            request.fde_total_time_s,
            request.grpc_send_time_s,
            request.remote_flush_time_s,
            request.local_flush_time_s,
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

        # ---- final_manifest.json ----
        manifest = {
            "generated_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_shards":         summary["total"],
            "done":                 summary["done"],
            "failed":               summary["failed"],
            "total_elapsed_sec":    round(summary["elapsed_sec"], 3),
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
            "local_flush", "end_to_end",
            "grpc_vs_local_diff",
            "prep_time", "upload_time", "compute_time", "download_time",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for e in sorted(logs, key=lambda x: x["shard_index"]):
                t = e["timing"]
                grpc_total  = t.get("grpc_send", 0) + t.get("remote_flush", 0)
                local_flush = t.get("local_flush", 0)
                row = {
                    "shard_index":        e["shard_index"],
                    "worker_id":          e["worker_id"],
                    "status":             e["status"],
                    "num_docs":           e["num_docs"],
                    "embed_time":         f"{t.get('embed_time', 0):.4f}",
                    "fde_total":          f"{t.get('fde_total', 0):.4f}",
                    "grpc_send":          f"{t.get('grpc_send', 0):.4f}",
                    "remote_flush":       f"{t.get('remote_flush', 0):.4f}",
                    "local_flush":        f"{local_flush:.4f}",
                    "end_to_end":         f"{t.get('end_to_end', 0):.4f}",
                    "grpc_vs_local_diff": f"{grpc_total - local_flush:.4f}",
                    "prep_time":          f"{t.get('prep_time', 0):.4f}",
                    "upload_time":        f"{t.get('upload_time', 0):.4f}",
                    "compute_time":       f"{t.get('compute_time', 0):.4f}",
                    "download_time":      f"{t.get('download_time', 0):.4f}",
                }
                w.writerow(row)
        logger.info("[Report] metrics.csv → %s", csv_path)


# ===========================================================================
# 서버 진입점
# ===========================================================================


def serve(args: argparse.Namespace) -> None:
    # ── config 생성: CLI 인자가 없는 항목은 storage_node_config.py 기본값 사용
    cfg = StorageNodeConfig.from_args(args)
    cfg.validate()

    logging.basicConfig(level=getattr(logging, cfg.log_level, logging.INFO))

    corpus     = CorpusLoader(cfg.corpus_path)
    shards     = compute_shards(corpus.total_docs(), cfg.num_shards)
    dispatcher = ShardDispatcher(shards)
    fde_store  = FdeStore(cfg.output_dir)

    os.makedirs(cfg.output_dir, exist_ok=True)
    corpus.save_doc_ids(os.path.join(cfg.output_dir, "doc_ids.json"))

    servicer = ShardServicer(corpus, dispatcher, fde_store, cfg.output_dir)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=cfg.max_workers),
        options=cfg.grpc_server_options(),
    )
    pb2_grpc.add_ShardServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{cfg.port}")
    server.start()

    logger.info(
        "Storage Node started: port=%d  shards=%d  docs=%d  max_workers=%d  "
        "grpc_max_msg=%d",
        cfg.port, len(shards), corpus.total_docs(), cfg.max_workers,
        cfg.grpc_max_message_bytes,
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
    p.add_argument("--corpus-path", default=_defaults.corpus_path,
                   help="corpus JSON 파일 경로 {doc_id: {title, text}}")
    p.add_argument("--output-dir",  default=_defaults.output_dir,
                   help="FDE shard mmap 및 보고서 저장 디렉터리")
    p.add_argument("--num-shards",  type=int, default=_defaults.num_shards,
                   help="corpus 를 나눌 shard 수")
    p.add_argument("--port",        type=int, default=_defaults.port,
                   help="gRPC 바인딩 포트")
    p.add_argument("--max-workers", type=int, default=_defaults.max_workers,
                   help="gRPC ThreadPoolExecutor 크기")
    return p.parse_args()


if __name__ == "__main__":
    serve(parse_args())
