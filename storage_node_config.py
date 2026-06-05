# -*- coding: utf-8 -*-
"""
storage_node_config.py
======================
Storage Node 전용 설정.

사용법
------
storage_node.py 는 이 모듈을 import 해서 기본값으로 사용한다.
CLI 인자가 전달되면 CLI 값이 우선한다.

    from storage_node_config import StorageNodeConfig
    cfg = StorageNodeConfig()           # 파일 기본값 사용
    cfg = StorageNodeConfig.from_args(parsed_args)   # CLI 오버라이드
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StorageNodeConfig:
    # ------------------------------------------------------------------
    # 데이터 경로
    # ------------------------------------------------------------------

    # corpus JSON 파일 경로 {doc_id: {title, text}}
    corpus_path: str = "/data/corpus.json"

    # 사전 계산된 ColBERT embedding 디렉터리 ({pos:08d}.npy)
    # None 이면 원문(title, text)만 전송하고 Worker 가 자체 인코딩
    embed_dir: Optional[str] = None

    # FDE shard mmap 파일 및 보고서 저장 디렉터리
    output_dir: str = "/data/fde_out"

    # ------------------------------------------------------------------
    # Shard 분할
    # ------------------------------------------------------------------

    # corpus 를 몇 개의 shard 로 나눌지
    # Worker 수 이상으로 설정하면 idle Worker 없이 처리 가능
    num_shards: int = 8

    # ------------------------------------------------------------------
    # gRPC 서버
    # ------------------------------------------------------------------

    # 바인딩 포트
    port: int = 50051

    # ThreadPoolExecutor 크기 (동시 RPC 처리 수)
    # Worker 수 × 2 정도로 설정 권장
    max_workers: int = 8

    # 서버 종료 시 진행 중인 RPC 대기 시간(초)
    server_stop_grace_sec: float = 5.0

    # gRPC 단일 메시지 최대 크기 (송신 / 수신 공통, bytes)
    # embedding 청크가 CHUNK_BYTES 이하이므로 그보다 크게 설정
    grpc_max_message_bytes: int = 4 * 1024 * 1024   # 4 MB

    # ------------------------------------------------------------------
    # 스트리밍 청크 크기
    # ------------------------------------------------------------------

    # Storage Node → Worker: 단일 DocumentChunk.embedding_data 최대 크기 (bytes)
    # 대용량 embedding 을 이 크기로 분할 전송
    embedding_chunk_bytes: int = 2 * 1024 * 1024    # 2 MB

    # Worker → Storage Node: FDE 수신 시 행 단위 버퍼 크기
    # (현재 미사용 — 수신 측에서 row_start/row_end 로 직접 기록)
    fde_recv_chunk_rows: int = 512

    # ------------------------------------------------------------------
    # 로깅
    # ------------------------------------------------------------------

    # Python logging 레벨 문자열 ("DEBUG" / "INFO" / "WARNING" / "ERROR")
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------

    @classmethod
    def from_args(cls, args) -> "StorageNodeConfig":
        """
        argparse.Namespace 의 값으로 config 를 생성한다.
        CLI 에 전달되지 않은 항목은 dataclass 기본값을 유지한다.
        """
        cfg = cls()
        if getattr(args, "corpus_path", None) is not None:
            cfg.corpus_path = str(args.corpus_path)
        if getattr(args, "embed_dir", None) is not None:
            cfg.embed_dir = str(args.embed_dir)
        if getattr(args, "output_dir", None) is not None:
            cfg.output_dir = str(args.output_dir)
        if getattr(args, "num_shards", None) is not None:
            cfg.num_shards = int(args.num_shards)
        if getattr(args, "port", None) is not None:
            cfg.port = int(args.port)
        if getattr(args, "max_workers", None) is not None:
            cfg.max_workers = int(args.max_workers)
        return cfg

    def grpc_server_options(self) -> list:
        """grpc.server(options=...) 에 전달할 리스트를 반환한다."""
        return [
            ("grpc.max_send_message_length",    self.grpc_max_message_bytes),
            ("grpc.max_receive_message_length", self.grpc_max_message_bytes),
        ]

    def validate(self) -> None:
        """필수 경로 존재 여부 등 기본 검증. 실패 시 ValueError."""
        if not os.path.isfile(self.corpus_path):
            raise ValueError(
                f"[StorageNodeConfig] corpus_path 파일 없음: {self.corpus_path}"
            )
        if self.embed_dir is not None and not os.path.isdir(self.embed_dir):
            raise ValueError(
                f"[StorageNodeConfig] embed_dir 디렉터리 없음: {self.embed_dir}"
            )
        if self.num_shards < 1:
            raise ValueError(
                f"[StorageNodeConfig] num_shards 는 1 이상이어야 합니다: {self.num_shards}"
            )
        if not (1 <= self.port <= 65535):
            raise ValueError(
                f"[StorageNodeConfig] 유효하지 않은 port: {self.port}"
            )
        if self.grpc_max_message_bytes < self.embedding_chunk_bytes:
            raise ValueError(
                "[StorageNodeConfig] grpc_max_message_bytes 는 "
                "embedding_chunk_bytes 이상이어야 합니다."
            )
