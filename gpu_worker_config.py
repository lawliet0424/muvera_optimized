# -*- coding: utf-8 -*-
"""
gpu_worker_config.py
====================
GPU Worker 전용 설정.

배포
----
  Storage Node  dccblue@163.239.199.208  /data/muvera_optimized
  Worker 1      dcceris@163.239.199.213  /home/dcceris/Desktop/muvera_optimized
  Worker 2      dccbeta@163.239.199.206  /home/dccbeta/muvera_optimized
  gRPC          163.239.199.208:50051

사용법
------
gpu_worker.py 는 이 모듈을 import 해서 기본값으로 사용한다.
CLI 인자가 전달되면 CLI 값이 우선한다.

    from gpu_worker_config import WorkerConfig
    cfg = WorkerConfig()                    # 파일 기본값 사용
    cfg = WorkerConfig.from_args(args)      # CLI 오버라이드
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 배포 경로
# ---------------------------------------------------------------------------
STORAGE_NODE_ADDR = "163.239.199.208:50051"

# GPU FDE kernel 과 동일 — partition 수 = 2^num_simhash_projections
MAX_SIMHASH_PROJECTIONS = 31

WORKER_PROJECT_DIRS: dict[str, str] = {
    "dcceris": "/home/dcceris/Desktop/muvera_optimized",
    "dccbeta": "/home/dccbeta/muvera_optimized",
}


def _default_worker_project_dir() -> str:
    """호스트명으로 Worker 프로젝트 루트를 결정한다."""
    hostname = socket.gethostname().lower()
    for key, path in WORKER_PROJECT_DIRS.items():
        if key in hostname:
            return path
    return os.path.expanduser("~/muvera_optimized")


@dataclass
class WorkerConfig:
    # ------------------------------------------------------------------
    # gRPC 클라이언트
    # ------------------------------------------------------------------

    # Storage Node 주소 (dccblue@163.239.199.208)
    server: str = STORAGE_NODE_ADDR

    # Worker 식별자. None 이면 런타임에 "hostname-PID" 로 자동 생성
    worker_id: Optional[str] = None

    # Worker 프로젝트 루트 (호스트명으로 자동 선택)
    #   dcceris → /home/dcceris/Desktop/muvera_optimized
    #   dccbeta → /home/dccbeta/muvera_optimized
    project_dir: str = field(default_factory=_default_worker_project_dir)

    # gRPC 단일 메시지 최대 크기 (송신 / 수신 공통, bytes)
    # Storage Node 의 grpc_max_message_bytes 와 반드시 같거나 크게 설정
    grpc_max_message_bytes: int = 4 * 1024 * 1024   # 4 MB

    # ------------------------------------------------------------------
    # RPC 타임아웃 (초)
    # ------------------------------------------------------------------

    # GetShard: 대형 shard 전송 시 충분한 값으로 설정
    get_shard_timeout_sec: float = 300.0

    # UploadFdeShard: FDE 크기에 따라 조정
    upload_fde_timeout_sec: float = 300.0

    # ReportShardStatus: 단방향 메시지이므로 짧게 설정 가능
    report_status_timeout_sec: float = 30.0

    # ------------------------------------------------------------------
    # FDE 설정 (FixedDimensionalEncodingConfig 대응)
    # ------------------------------------------------------------------

    # embedding 차원 — ColBERT 모델 출력 차원과 일치해야 함
    dimension: int = 128

    # SimHash repetition 횟수
    num_repetitions: int = 2

    # SimHash projection 비트 수 (partition 수 = 2^num_simhash_projections, 최대 31)
    num_simhash_projections: int = 5

    # 랜덤 시드
    seed: int = 42

    # AMS projection 차원. None 이면 identity projection 사용
    projection_dimension: Optional[int] = None

    # 빈 partition 을 최근접 토큰으로 채울지 여부
    # 원본 ColbertFdeRetriever 기본값: True
    fill_empty_partitions: bool = True

    # ------------------------------------------------------------------
    # ColBERT 인코더
    # ------------------------------------------------------------------

    # HuggingFace 모델 이름 또는 로컬 경로
    # Storage Node 가 전송한 문서 텍스트를 인코딩하는 데 사용
    colbert_model: str = "raphaelsty/neural-cherche-colbert"

    # PyTorch device ("cuda" / "cpu" / "cuda:0" 등)
    device: str = "cuda"

    # ------------------------------------------------------------------
    # 스트리밍 청크 크기
    # ------------------------------------------------------------------

    # Worker → Storage Node: FDE 업로드 시 한 청크의 최대 바이트 수
    # grpc_max_message_bytes 보다 작게 설정해야 함
    grpc_fde_chunk_bytes: int = 3 * 1024 * 1024    # 3 MB

    # ------------------------------------------------------------------
    # 로깅
    # ------------------------------------------------------------------

    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, args) -> "WorkerConfig":
        """
        argparse.Namespace 의 값으로 config 를 생성한다.
        CLI 에 전달되지 않은 항목은 dataclass 기본값을 유지한다.
        """
        cfg = cls()
        if getattr(args, "server",        None) is not None:
            cfg.server        = args.server
        if getattr(args, "worker_id",     None) is not None:
            cfg.worker_id     = args.worker_id
        if getattr(args, "project_dir",   None) is not None:
            cfg.project_dir   = str(args.project_dir)
        if getattr(args, "rep",           None) is not None:
            cfg.num_repetitions = int(args.rep)
        if getattr(args, "simhash",       None) is not None:
            cfg.num_simhash_projections = int(args.simhash)
        if getattr(args, "projection",    None) is not None:
            cfg.projection_dimension = int(args.projection)
        if getattr(args, "fill_empty",    None) is not None:
            cfg.fill_empty_partitions = bool(args.fill_empty)
        if getattr(args, "colbert_model", None) is not None:
            cfg.colbert_model = args.colbert_model
        if getattr(args, "device",        None) is not None:
            cfg.device        = args.device
        return cfg

    def grpc_channel_options(self) -> list:
        """grpc.insecure_channel(options=...) 에 전달할 리스트를 반환한다."""
        return [
            ("grpc.max_send_message_length",    self.grpc_max_message_bytes),
            ("grpc.max_receive_message_length", self.grpc_max_message_bytes),
        ]

    def validate(self) -> None:
        """기본 검증. 실패 시 ValueError."""
        if not self.server:
            raise ValueError("[WorkerConfig] server 주소가 비어 있습니다.")
        if self.num_repetitions < 1:
            raise ValueError(
                f"[WorkerConfig] num_repetitions 는 1 이상이어야 합니다: {self.num_repetitions}"
            )
        if not (1 <= self.num_simhash_projections <= MAX_SIMHASH_PROJECTIONS):
            raise ValueError(
                f"[WorkerConfig] num_simhash_projections 는 "
                f"1~{MAX_SIMHASH_PROJECTIONS} 사이여야 합니다: "
                f"{self.num_simhash_projections}"
            )
        if self.grpc_fde_chunk_bytes >= self.grpc_max_message_bytes:
            raise ValueError(
                "[WorkerConfig] grpc_fde_chunk_bytes 는 grpc_max_message_bytes 보다 "
                "작아야 합니다."
            )
