# -*- coding: utf-8 -*-
"""
storage_node_config.py
======================
Storage Node 전용 설정.

배포 (dccblue@163.239.199.208)
------------------------------
  project_dir          /data/muvera_optimized
  datasets_root        /data/datasets
  dataset_collection   flare
  dataset              (필수)  → corpus: /data/datasets/flare/{dataset}/corpus.jsonl
                                  output: /data/muvera_optimized/data/fde_out/{dataset}/

경로 결정 우선순위
------------------
corpus (최종 corpus_path — BEIR 디렉터리):
  1. --corpus-path
  2. --dataset  →  {datasets_root}/{dataset_collection}/{dataset}/
  3. {project_dir}/data/corpus.json

output (최종 output_dir):
  1. --output-dir
  2. --dataset 지정 시  → {project_dir}/data/fde_out/{dataset}/
  3. {project_dir}/data/fde_out/

사용법
------
    from storage_node_config import StorageNodeConfig
    cfg = StorageNodeConfig()
    cfg = StorageNodeConfig.from_args(parsed_args)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# 배포 경로 — dccblue@163.239.199.208 (Storage Node)
# ---------------------------------------------------------------------------
DEFAULT_PROJECT_DIR          = "/data/muvera_optimized"
DEFAULT_DATASETS_ROOT        = "/data/datasets"
DEFAULT_DATASET_COLLECTION   = "flare"
DEFAULT_DATASET              = None   # --dataset 로 지정 (예: scidocs)


def dataset_dir(datasets_root: str, collection: str, dataset: str) -> str:
    """BEIR corpus 디렉터리: {root}/{collection}/{dataset}/"""
    return os.path.join(datasets_root, collection, dataset)


@dataclass
class StorageNodeConfig:
    # ------------------------------------------------------------------
    # 데이터 경로 (입력)
    # ------------------------------------------------------------------

    project_dir: str = DEFAULT_PROJECT_DIR
    datasets_root: str = DEFAULT_DATASETS_ROOT
    dataset_collection: str = DEFAULT_DATASET_COLLECTION
    dataset: Optional[str] = DEFAULT_DATASET

    # ------------------------------------------------------------------
    # 데이터 경로 (resolve_paths() 로 계산되는 최종 값)
    # ------------------------------------------------------------------

    corpus_path: str = ""
    output_dir: str = ""

    # CLI 직접 지정 시에만 사용 (resolve_paths 입력)
    _corpus_override: Optional[str] = field(default=None, repr=False)
    _output_override: Optional[str] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Shard 분할
    # ------------------------------------------------------------------

    # GetShard 1회당 전송 문서 수 (worker 수와 무관, 기본 10 000 ≈ 8 000~12 000 최적 구간)
    shard_doc_size: int = 10_000

    # Worker 장애 시 inprogress shard 를 pending 으로 되돌리는 lease
    shard_lease_timeout_sec: float = 600.0
    shard_max_attempts: int = 3
    lease_reaper_interval_sec: float = 30.0

    # ------------------------------------------------------------------
    # gRPC 서버
    # ------------------------------------------------------------------

    port: int = 50051
    max_workers: int = 8
    server_stop_grace_sec: float = 5.0
    grpc_max_message_bytes: int = 4 * 1024 * 1024   # 4 MB
    fde_recv_chunk_rows: int = 512

    # ------------------------------------------------------------------
    # 로깅
    # ------------------------------------------------------------------

    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.resolve_paths()

    def resolve_paths(self) -> None:
        """인자 조합에 따라 corpus_path / output_dir 최종 경로를 계산한다."""
        if self._corpus_override:
            self.corpus_path = self._corpus_override
        elif self.dataset:
            self.corpus_path = dataset_dir(
                self.datasets_root, self.dataset_collection, self.dataset
            )
        else:
            self.corpus_path = os.path.join(self.project_dir, "data", "corpus.json")

        if self._output_override:
            self.output_dir = self._output_override
        elif self.dataset:
            self.output_dir = os.path.join(
                self.project_dir, "data", "fde_out", self.dataset
            )
        else:
            self.output_dir = os.path.join(self.project_dir, "data", "fde_out")

    @classmethod
    def from_args(cls, args) -> "StorageNodeConfig":
        cfg = cls()

        if getattr(args, "project_dir", None) is not None:
            cfg.project_dir = str(args.project_dir)
        if getattr(args, "datasets_root", None) is not None:
            cfg.datasets_root = str(args.datasets_root)
        if getattr(args, "dataset_collection", None) is not None:
            cfg.dataset_collection = str(args.dataset_collection)
        if getattr(args, "dataset", None) is not None:
            raw = str(args.dataset).strip()
            cfg.dataset = raw if raw else None
        if getattr(args, "corpus_path", None) is not None:
            cfg._corpus_override = str(args.corpus_path)
        if getattr(args, "output_dir", None) is not None:
            cfg._output_override = str(args.output_dir)
        if getattr(args, "shard_doc_size", None) is not None:
            cfg.shard_doc_size = int(args.shard_doc_size)
        if getattr(args, "shard_lease_timeout_sec", None) is not None:
            cfg.shard_lease_timeout_sec = float(args.shard_lease_timeout_sec)
        if getattr(args, "shard_max_attempts", None) is not None:
            cfg.shard_max_attempts = int(args.shard_max_attempts)
        if getattr(args, "lease_reaper_interval_sec", None) is not None:
            cfg.lease_reaper_interval_sec = float(args.lease_reaper_interval_sec)
        if getattr(args, "port", None) is not None:
            cfg.port = int(args.port)
        if getattr(args, "max_workers", None) is not None:
            cfg.max_workers = int(args.max_workers)

        cfg.resolve_paths()
        return cfg

    def grpc_server_options(self) -> list:
        return [
            ("grpc.max_send_message_length",    self.grpc_max_message_bytes),
            ("grpc.max_receive_message_length", self.grpc_max_message_bytes),
        ]

    def validate(self) -> None:
        if not _corpus_source_exists(self.corpus_path):
            raise ValueError(
                f"[StorageNodeConfig] corpus 소스 없음: {self.corpus_path}\n"
                f"  기대 경로: {dataset_dir(self.datasets_root, self.dataset_collection, self.dataset or '<dataset>')}/corpus.jsonl\n"
                f"  (dataset={self.dataset!r}, collection={self.dataset_collection!r}, "
                f"datasets_root={self.datasets_root})"
            )
        if self.shard_doc_size < 1:
            raise ValueError(
                f"[StorageNodeConfig] shard_doc_size 는 1 이상이어야 합니다: "
                f"{self.shard_doc_size}"
            )
        if not (1 <= self.port <= 65535):
            raise ValueError(
                f"[StorageNodeConfig] 유효하지 않은 port: {self.port}"
            )
        if self.shard_lease_timeout_sec <= 0:
            raise ValueError(
                f"[StorageNodeConfig] shard_lease_timeout_sec > 0 이어야 합니다: "
                f"{self.shard_lease_timeout_sec}"
            )
        if self.shard_max_attempts < 1:
            raise ValueError(
                f"[StorageNodeConfig] shard_max_attempts >= 1 이어야 합니다: "
                f"{self.shard_max_attempts}"
            )


def _corpus_source_exists(corpus_path: str) -> bool:
    """단일 JSON 파일 또는 BEIR corpus 디렉터리 존재 여부."""
    if os.path.isfile(corpus_path):
        return True
    if os.path.isdir(corpus_path):
        for name in ("corpus.jsonl", "corpus.json"):
            if os.path.isfile(os.path.join(corpus_path, name)):
                return True
    return False
