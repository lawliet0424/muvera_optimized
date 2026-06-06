# -*- coding: utf-8 -*-
"""
출력 지표
---------
- embed_time        : ColBERT 인코딩 시간
- fde_time          : GPU FDE 생성 시간 (CUDA kernel 합계)
  ├ prep_time       : random matrix 준비
  ├ upload_time     : CPU→GPU 전송
  ├ compute_time    : simhash + partition + scatter + average
  ├ download_time   : GPU→CPU 전송
  └ flush_time      : (내부 flush — memmap 을 쓰지 않으면 0)
- grpc_send_time    : Worker → Storage Node UploadFdeShard RPC (remote flush 포함)
- remote_flush_time : Storage Node mmap flush+fsync (grpc_send 의 부분집합)
- end_to_end_time   : 전체 shard 처리 (recv → embed → FDE → send → report)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ===========================================================================
# ShardPerfRecord
# ===========================================================================

@dataclass
class ShardPerfRecord:
    shard_index:       int
    worker_id:         str
    status:            str
    num_docs:          int
    doc_start:         int
    doc_end:           int

    # ColBERT
    embed_time:        float = 0.0

    # GPU FDE (generate_document_fde_batch_gpu_3stage 반환값)
    prep_time:         float = 0.0
    upload_time:       float = 0.0   # CPU→GPU
    simhash_time:      float = 0.0
    partition_time:    float = 0.0
    scatter_time:      float = 0.0
    average_time:      float = 0.0
    fill_time:         float = 0.0
    compute_time:      float = 0.0
    download_time:     float = 0.0   # GPU→CPU
    reshape_time:      float = 0.0
    flush_time:        float = 0.0   # 내부 flush
    fde_total:         float = 0.0

    # I/O 지표
    grpc_recv_time:    float = 0.0   # Storage Node → Worker 수신
    grpc_send_time:    float = 0.0   # Worker → Storage Node 전송
    remote_flush_time: float = 0.0   # Storage Node flush+fsync (grpc_send 에 포함)

    end_to_end:        float = 0.0
    error_message:     str   = ""

    @property
    def docs_per_sec(self) -> float:
        return self.num_docs / max(self.end_to_end, 1e-9)

    def perf_summary_line(self) -> str:
        return (
            f"shard={self.shard_index:04d} worker={self.worker_id} "
            f"docs={self.num_docs} "
            f"embed={self.embed_time:.4f}s "
            f"fde={self.fde_total:.4f}s "
            f"grpc_send={self.grpc_send_time:.6f}s "
            f"remote_flush={self.remote_flush_time:.6f}s "
            f"end_to_end={self.end_to_end:.4f}s "
            f"docs/s={self.docs_per_sec:.1f}"
        )


# ===========================================================================
# PerfLogger
# ===========================================================================

class PerfLogger:
    """Thread-safe 성능 기록 및 집계."""

    def __init__(self, name: str = "perf") -> None:
        self._name    = name
        self._records: List[ShardPerfRecord] = []
        self._start   = time.time()

    def record_shard(self, record: ShardPerfRecord) -> None:
        self._records.append(record)
        logger.info("[PerfLogger:%s] %s", self._name, record.perf_summary_line())

    def from_status_proto(self, msg, remote_flush_time: float = 0.0) -> ShardPerfRecord:
        """ShardStatus proto → ShardPerfRecord."""
        return ShardPerfRecord(
            shard_index       = msg.shard_index,
            worker_id         = msg.worker_id,
            status            = msg.status,
            num_docs          = msg.num_docs,
            doc_start         = msg.doc_start,
            doc_end           = msg.doc_end,
            embed_time        = msg.embed_time_s,
            prep_time         = msg.prep_time_s,
            upload_time       = msg.upload_time_s,
            simhash_time      = msg.simhash_time_s,
            partition_time    = msg.partition_time_s,
            scatter_time      = msg.scatter_time_s,
            average_time      = msg.average_time_s,
            fill_time         = msg.fill_time_s,
            compute_time      = msg.compute_time_s,
            download_time     = msg.download_time_s,
            reshape_time      = msg.reshape_time_s,
            flush_time        = msg.flush_time_s,
            fde_total         = msg.fde_total_time_s,
            grpc_recv_time    = msg.grpc_recv_time_s,
            grpc_send_time    = msg.grpc_send_time_s,
            remote_flush_time = remote_flush_time or msg.remote_flush_time_s,
            end_to_end        = msg.end_to_end_time_s,
            error_message     = msg.error_message,
        )

    # ------------------------------------------------------------------ #
    # 집계
    # ------------------------------------------------------------------ #

    def _success_records(self) -> List[ShardPerfRecord]:
        return [r for r in self._records if r.status == "success"]

    def aggregate(self) -> dict:
        recs = self._success_records()
        if not recs:
            return {}

        def _sum(attr): return sum(getattr(r, attr) for r in recs)
        def _avg(attr): return _sum(attr) / len(recs)
        def _max(attr): return max(getattr(r, attr) for r in recs)

        total_docs     = sum(r.num_docs for r in recs)
        total_elapsed  = time.time() - self._start

        return {
            "total_shards":          len(self._records),
            "success_shards":        len(recs),
            "total_docs":            total_docs,
            "total_elapsed_sec":     round(total_elapsed, 3),
            "throughput_docs_per_s": round(total_docs / max(total_elapsed, 1e-9), 2),

            "embed_time_sum":        round(_sum("embed_time"), 4),
            "embed_time_avg":        round(_avg("embed_time"), 4),

            "fde_total_sum":         round(_sum("fde_total"), 4),
            "fde_total_avg":         round(_avg("fde_total"), 4),
            "compute_time_sum":      round(_sum("compute_time"), 4),
            "download_time_sum":     round(_sum("download_time"), 4),

            "grpc_send_sum":         round(_sum("grpc_send_time"), 6),
            "grpc_send_avg":         round(_avg("grpc_send_time"), 6),
            "grpc_send_max":         round(_max("grpc_send_time"), 6),

            "remote_flush_sum":      round(_sum("remote_flush_time"), 6),
            "remote_flush_avg":      round(_avg("remote_flush_time"), 6),
            "remote_flush_max":      round(_max("remote_flush_time"), 6),

            "e2e_avg":               round(_avg("end_to_end"), 4),
            "e2e_max":               round(_max("end_to_end"), 4),
        }

    # ------------------------------------------------------------------ #
    # 출력
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        agg = self.aggregate()
        if not agg:
            print("[PerfLogger] No successful shards recorded.")
            return

        header = f"\n{'='*72}\n  Performance Summary: {self._name}\n{'='*72}"
        print(header)
        print(f"  Total shards     : {agg['total_shards']}  "
              f"(success={agg['success_shards']})")
        print(f"  Total docs       : {agg['total_docs']}")
        print(f"  Elapsed          : {agg['total_elapsed_sec']:.2f}s  "
              f"({agg['throughput_docs_per_s']:.1f} docs/s)")
        print()
        print(f"  ColBERT embed    : avg={agg['embed_time_avg']:.4f}s  "
              f"sum={agg['embed_time_sum']:.4f}s")
        print(f"  GPU FDE total    : avg={agg['fde_total_avg']:.4f}s  "
              f"sum={agg['fde_total_sum']:.4f}s")
        print(f"    compute        : sum={agg['compute_time_sum']:.4f}s")
        print(f"    download(GPU→CPU): sum={agg['download_time_sum']:.4f}s")
        print()
        print(f"  {'─'*60}")
        print(f"  {'I/O (per shard avg)':}")
        print(f"  {'─'*60}")
        print(f"  grpc send (RPC)  : "
              f"avg={agg['grpc_send_avg']:.6f}s  "
              f"max={agg['grpc_send_max']:.6f}s  "
              f"sum={agg['grpc_send_sum']:.6f}s")
        print(f"  remote flush     : "
              f"avg={agg['remote_flush_avg']:.6f}s  "
              f"max={agg['remote_flush_max']:.6f}s  "
              f"sum={agg['remote_flush_sum']:.6f}s")
        print(f"  {'─'*60}")
        print(f"  End-to-end (avg) : {agg['e2e_avg']:.4f}s  "
              f"max={agg['e2e_max']:.4f}s")
        print(f"{'='*72}\n")

    # ------------------------------------------------------------------ #
    # CSV 저장
    # ------------------------------------------------------------------ #

    CSV_FIELDS = [
        "shard_index", "worker_id", "status", "num_docs", "doc_start", "doc_end",
        "embed_time", "prep_time", "upload_time", "compute_time",
        "download_time", "reshape_time", "flush_time", "fde_total",
        "grpc_recv_time", "grpc_send_time", "remote_flush_time", "end_to_end",
        "docs_per_sec", "error_message",
    ]

    def save_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            w.writeheader()
            for r in sorted(self._records, key=lambda x: x.shard_index):
                row = asdict(r)
                row["docs_per_sec"] = f"{r.docs_per_sec:.2f}"
                for k, v in row.items():
                    if isinstance(v, float):
                        row[k] = f"{v:.6f}"
                w.writerow(row)
        logger.info("[PerfLogger] CSV saved → %s", path)

    def save_json(self, path: str) -> None:
        data = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "aggregate": self.aggregate(),
            "shards": [asdict(r) for r in self._records],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("[PerfLogger] JSON saved → %s", path)


# ===========================================================================
# 독립 실행: final_manifest.json 파싱 → 리포트 생성
# ===========================================================================

def load_from_manifest(manifest_path: str) -> PerfLogger:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    perf = PerfLogger("manifest-analysis")

    for shard in manifest.get("shards", []):
        t = shard.get("timing", {})
        rec = ShardPerfRecord(
            shard_index       = shard["shard_index"],
            worker_id         = shard.get("worker_id", ""),
            status            = shard.get("status", ""),
            num_docs          = shard.get("num_docs", 0),
            doc_start         = t.get("doc_start", 0),
            doc_end           = t.get("doc_end",   0),
            embed_time        = t.get("embed_time",     0.0),
            prep_time         = t.get("prep_time",      0.0),
            upload_time       = t.get("upload_time",    0.0),
            simhash_time      = t.get("simhash_time",   0.0),
            partition_time    = t.get("partition_time", 0.0),
            scatter_time      = t.get("scatter_time",   0.0),
            average_time      = t.get("average_time",   0.0),
            fill_time         = t.get("fill_time",      0.0),
            compute_time      = t.get("compute_time",   0.0),
            download_time     = t.get("download_time",  0.0),
            reshape_time      = t.get("reshape_time",   0.0),
            flush_time        = t.get("flush_time",     0.0),
            fde_total         = t.get("fde_total",      0.0),
            grpc_recv_time    = t.get("grpc_recv",      0.0),
            grpc_send_time    = t.get("grpc_send",      0.0),
            remote_flush_time = t.get("remote_flush",   0.0),
            end_to_end        = t.get("end_to_end",     0.0),
            error_message     = shard.get("error_message", ""),
        )
        perf.record_shard(rec)

    return perf


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="FDE Performance Report Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True,
                   help="Path to final_manifest.json")
    p.add_argument("--output",   default=None,
                   help="Output CSV path (default: next to manifest)")
    args = p.parse_args()

    perf = load_from_manifest(args.manifest)
    perf.print_summary()

    out = args.output or os.path.join(
        os.path.dirname(args.manifest), "perf_report.csv"
    )
    perf.save_csv(out)
    perf.save_json(out.replace(".csv", ".json"))
    print(f"Reports saved:\n  {out}\n  {out.replace('.csv', '.json')}")


if __name__ == "__main__":
    main()
