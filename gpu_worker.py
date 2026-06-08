# -*- coding: utf-8 -*-
"""
파이프라인 (shard 당)
---------------------
1. Storage Node 로부터 문서 원문 streaming 수신
2. ColBERT embedding 생성 (RAM)
3. CUDA FDE 생성 → RAM ndarray 유지
4. FDE chunk gRPC streaming 업로드
5. UploadAck 수신·검증 후 RAM 해제
6. ReportShardStatus 로 timing 전송
7. 다음 shard 요청 (GetShard)

배포
----
  Storage Node  dccblue@163.239.199.208  /data/muvera_optimized
  Worker 1      dcceris@163.239.199.213  ~/Desktop/muvera_optimized
  Worker 2      dccbeta@163.239.199.206  ~/muvera_optimized
  gRPC          163.239.199.208:50051

원본 파이썬 파일 작업 흐름 대비 구현 현황
-------------------------------------------
[구현됨]        ColBERT 모델 로드, FDE config 구성, 원문 수신,
                ColBERT 배치 인코딩, generate_document_fde_batch_gpu_3stage 호출,
                GPU timing 전 필드 ShardStatus 전송

[주석 표시됨]   캐시 디렉터리/경로 계산(Storage Node 담당), partition_count.csv(미반환),
                log_memory_usage(psutil), 인코딩 결과 공통 디렉터리 저장

[미구현]        쿼리 인코딩(encode_queries), generate_query_fde_gpu, FDE 검색(dot-product),
                Chamfer 재랭킹, 쿼리/문서 embedding 캐시 관리, latency.tsv 로깅
                → 이 기능들은 별도 QueryWorker(또는 QueryService RPC)로 분리 권장
"""

from __future__ import annotations

import argparse
import gc
import io
import logging
import math
import os
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import grpc
import numpy as np
import psutil

import fde_pipeline_pb2 as pb2

from gpu_worker_config import WorkerConfig, MAX_SIMHASH_PROJECTIONS
import fde_pipeline_pb2_grpc as pb2_grpc

# 기존 FDE 생성 함수 재사용 (CUDA kernel 수정 없음)
from fde_generator_gpu_optimized_triple_stage_sharding import (
    FixedDimensionalEncodingConfig,
    EncodingType,
    ProjectionType,
    generate_document_fde_batch_gpu_3stage,
    # generate_query_fde_gpu,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WORKER/%(process)d] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 청크 크기 — WorkerConfig 에서 관리.
# run_worker() 진입 시 _init_chunk_constants(cfg) 로 config 값으로 덮어쓴다.
# ---------------------------------------------------------------------------
FDE_UPLOAD_CHUNK_ROWS = 256                 # gpu_worker_config.WorkerConfig 기본값
GRPC_MAX_MSG_BYTES    = 3 * 1024 * 1024    # gpu_worker_config.grpc_fde_chunk_bytes 기본값


def _init_chunk_constants(cfg: WorkerConfig) -> None:
    """모듈 레벨 청크 상수를 config 값으로 갱신한다."""
    global GRPC_MAX_MSG_BYTES
    GRPC_MAX_MSG_BYTES = cfg.grpc_fde_chunk_bytes


# ===========================================================================
# 메모리 사용량 로깅 헬퍼
# ===========================================================================

def log_memory_usage(stage: str) -> float:
    """
    현재 프로세스 RSS 메모리(MB)를 로깅하고 반환한다.
    """
    try:
        import psutil
        process = psutil.Process()
        mb = process.memory_info().rss / 1024 / 1024
        logger.info("[MEMORY] %s: %.1f MB (%.2f GB)", stage, mb, mb / 1024)
        return mb
    except ImportError:
        # psutil 미설치: pip install psutil 로 활성화 가능
        logger.debug("[MEMORY] psutil 없음 — 메모리 측정 생략 (%s)", stage)
        return 0.0


# ===========================================================================
# FDE Config 헬퍼
# ===========================================================================

def build_fde_config(cfg: WorkerConfig) -> FixedDimensionalEncodingConfig:
    proj_type = (
        ProjectionType.AMS_SKETCH
        if (cfg.projection_dimension and cfg.projection_dimension > 0)
        else ProjectionType.DEFAULT_IDENTITY
    )
    return FixedDimensionalEncodingConfig(
        dimension              = cfg.dimension,
        num_repetitions        = cfg.num_repetitions,
        num_simhash_projections= cfg.num_simhash_projections,
        seed                   = cfg.seed,
        encoding_type          = EncodingType.AVERAGE,
        fill_empty_partitions  = cfg.fill_empty_partitions,
        projection_type        = proj_type,
        projection_dimension   = cfg.projection_dimension,
    )


def config_to_proto(config: FixedDimensionalEncodingConfig) -> pb2.FdeConfig:
    return pb2.FdeConfig(
        dimension                  = config.dimension,
        num_repetitions            = config.num_repetitions,
        num_simhash_projections    = config.num_simhash_projections,
        seed                       = config.seed,
        encoding_type              = config.encoding_type.value,
        projection_type            = config.projection_type.value,
        projection_dimension       = config.projection_dimension or 0,
        fill_empty_partitions      = config.fill_empty_partitions,
        final_projection_dimension = config.final_projection_dimension or 0,
    )


def fde_output_dim(config: FixedDimensionalEncodingConfig) -> int:
    proj_dim = (
        config.dimension
        if config.projection_type == ProjectionType.DEFAULT_IDENTITY
        else config.projection_dimension
    )
    if config.final_projection_dimension and config.final_projection_dimension > 0:
        return config.final_projection_dimension
    return config.num_repetitions * (2 ** config.num_simhash_projections) * proj_dim


# ===========================================================================
# Shard 수신 (GetShard 스트림 파싱)
# ===========================================================================

def receive_shard(
    stream: Iterator[pb2.DocumentChunk],
) -> Tuple[int, int, int, List[str], List[dict]]:
    """
    GetShard 스트림을 파싱하여 문서 원문 메타를 반환한다.
    Storage Node 는 텍스트(title, text)만 전송한다.
    """
    shard_index = -1
    doc_start   = 0
    doc_end     = 0
    total_docs  = 0

    doc_meta: Dict[int, dict] = {}

    for chunk in stream:
        if shard_index < 0:
            shard_index = chunk.shard_index
            doc_start   = chunk.doc_start
            doc_end     = chunk.doc_end
            total_docs  = chunk.total_docs

        di = chunk.doc_index_in_shard

        if di not in doc_meta:
            doc_meta[di] = {
                "doc_id": chunk.doc_id,
                "title":  chunk.corpus_title,
                "text":   chunk.corpus_text,
            }

    doc_ids: List[str] = []
    metas: List[dict]  = []

    for di in range(total_docs):
        meta = doc_meta.get(di, {})
        doc_ids.append(meta.get("doc_id", ""))
        metas.append(meta)

    return shard_index, doc_start, doc_end, doc_ids, metas


# ===========================================================================
# ColBERT 인코더
# ===========================================================================

class ColBERTEncoder:
    def __init__(self, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device     = device
        self._ranker     = None

    def _ensure_loaded(self) -> None:
        if self._ranker is not None:
            return
        try:
            import neural_cherche.models as models
            import neural_cherche.rank as rank
            model = models.ColBERT(
                model_name_or_path=self._model_name, device=self._device
            )
            self._ranker = rank.ColBERT(key="id", on=["title", "text"], model=model)
            logger.info("[ColBERT] Model loaded: %s on %s", self._model_name, self._device)
        except ImportError:
            logger.warning(
                "[ColBERT] neural_cherche 없음 — "
                "ColBERT 인코딩을 수행할 수 없습니다."
            )
            self._ranker = None

    def encode(self, docs: List[dict]) -> Dict[str, np.ndarray]:
        """
        docs = [{id, title, text}, ...] → {doc_id: ndarray(seq_len, 128)}

        원본: ranker.encode_documents(documents=to_encode_docs) 와 동일.
        반환값을 to_numpy()로 변환하는 로직도 포함.
        """
        self._ensure_loaded()
        if self._ranker is None:
            return {}
        encoded = self._ranker.encode_documents(documents=docs)
        result = {}
        for doc_id, arr in encoded.items():
            if hasattr(arr, "cpu"):
                arr = arr.cpu().detach().numpy()
            result[str(doc_id)] = arr.astype(np.float32)
        return result

# ===========================================================================
# FDE 업로드 스트림 생성기
# ===========================================================================

def fde_upload_stream(
    shard_index: int,
    worker_id: str,
    fde_array: np.ndarray,
    embed_time: float,
    fde_time: float,
) -> Iterator[pb2.FdeChunk]:
    num_docs, fde_dim = fde_array.shape
    rows_per_chunk = max(1, GRPC_MAX_MSG_BYTES // (fde_dim * 4))
    total_chunks   = math.ceil(num_docs / rows_per_chunk)

    for ci in range(total_chunks):
        row_start = ci * rows_per_chunk
        row_end   = min(row_start + rows_per_chunk, num_docs)
        is_last   = row_end == num_docs

        yield pb2.FdeChunk(
            shard_index  = shard_index,
            worker_id    = worker_id,
            num_docs     = num_docs,
            fde_dim      = fde_dim,
            total_chunks = total_chunks,
            chunk_index  = ci,
            is_last      = is_last,
            row_start    = row_start,
            row_end      = row_end,
            fde_data     = fde_array[row_start:row_end].astype(np.float32).tobytes(),
            embed_time_s = embed_time if is_last else 0.0,
            fde_time_s   = fde_time   if is_last else 0.0,
        )


def validate_upload_ack(
    ack: pb2.UploadAck,
    shard_index: int,
    expected_bytes: int,
) -> None:
    """UploadFdeShard 응답 검증. 실패 시 RuntimeError."""
    if ack.status != "ok":
        raise RuntimeError(
            f"UploadFdeShard rejected: status={ack.status!r} message={ack.message!r}"
        )
    if ack.shard_index != shard_index:
        raise RuntimeError(
            f"UploadAck shard_index mismatch: got {ack.shard_index}, "
            f"expected {shard_index}"
        )
    if ack.bytes_received != expected_bytes:
        raise RuntimeError(
            f"UploadAck bytes_received mismatch: got {ack.bytes_received}, "
            f"expected {expected_bytes}"
        )


# ===========================================================================
# 단일 shard 처리
# ===========================================================================

def process_shard(
    shard_index: int,
    doc_start: int,
    doc_end: int,
    doc_ids: List[str],
    metas: List[dict],
    config: FixedDimensionalEncodingConfig,
    encoder: ColBERTEncoder,
) -> Tuple[np.ndarray, dict]:
    """
    한 shard 를 처리하여 (fde_array, timing_dict) 를 반환한다.

    Step1: ColBERT 인코딩 (Storage Node 가 텍스트만 전송)
    Step2: generate_document_fde_batch_gpu_3stage (메모리 내 ndarray)
    """
    t_shard_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Step 1: ColBERT 인코딩
    # ------------------------------------------------------------------ #
    t_embed_start = time.perf_counter()
    logger.info(
        "[Shard %d] ColBERT 인코딩 시작: %d 문서",
        shard_index, len(metas),
    )
    docs_to_encode = [
        {
            "id":    meta["doc_id"],
            "title": meta.get("title", ""),
            "text":  meta.get("text", ""),
        }
        for meta in metas
    ]
    encoded_map = encoder.encode(docs_to_encode)

    valid_embeddings: List[np.ndarray] = []
    missing_ids: List[str] = []
    for meta in metas:
        did = meta["doc_id"]
        if did in encoded_map:
            valid_embeddings.append(encoded_map[did])
        else:
            missing_ids.append(did)

    if missing_ids:
        preview = missing_ids[:10]
        suffix = (
            f" ... (+{len(missing_ids) - 10} more)"
            if len(missing_ids) > 10
            else ""
        )
        raise RuntimeError(
            f"[Shard {shard_index}] ColBERT 인코딩 실패: "
            f"{len(missing_ids)}/{len(metas)} docs missing: "
            f"{preview}{suffix}"
        )

    del encoded_map
    del docs_to_encode
    gc.collect()
    embed_time = time.perf_counter() - t_embed_start

    # ── [주석] 원본 log_memory_usage 위치 (배치 인코딩 직후)
    log_memory_usage(f"[Shard {shard_index}] after ColBERT encode")

    logger.info(
        "[Shard %d] ColBERT 완료: %.4fs  docs=%d",
        shard_index, embed_time, len(valid_embeddings),
    )

    num_docs      = len(valid_embeddings)
    final_fde_dim = fde_output_dim(config)

    # 메모리 내 numpy array (로컬 SSD 기록 없음)
    fde_array = np.zeros((num_docs, final_fde_dim), dtype=np.float32)

    # ── [주석] 원본 log_memory_usage 위치 (FDE 생성 전)
    log_memory_usage(f"[Shard {shard_index}] before GPU FDE")

    t_fde_start = time.perf_counter()
    fde_timing  = generate_document_fde_batch_gpu_3stage(
        doc_embeddings_list = valid_embeddings,
        config              = config,
        fde_memmap          = fde_array,  # numpy array (SSD 저장 없음)
        batch_start_idx     = 0,
        log_every           = max(1, num_docs // 5),
    )
    fde_time = time.perf_counter() - t_fde_start

    # ── [주석] 원본 log_memory_usage 위치 (FDE 생성 후)
    log_memory_usage(f"[Shard {shard_index}] after GPU FDE")

    logger.info(
        "[Shard %d] GPU FDE 완료: %.4fs  "
        "(compute=%.4fs  upload=%.4fs  download=%.4fs  reshape=%.4fs)",
        shard_index,
        fde_time,
        fde_timing.get("compute_time",  0),
        fde_timing.get("upload_time",   0),
        fde_timing.get("download_time", 0),
        fde_timing.get("reshape_time",  0),
    )

    # 배치 완료 후 메모리 해제 (원본 del batch_embeddings + gc.collect() 패턴)
    del valid_embeddings
    gc.collect()

    t_shard_total = time.perf_counter() - t_shard_start

    timing = {
        "embed_time":              embed_time,
        "fde_time":                fde_time,
        "shard_total_before_send": t_shard_total,
        **fde_timing,
    }

    return fde_array, timing


def _report_shard_abandon(
    stub: pb2_grpc.ShardServiceStub,
    cfg: WorkerConfig,
    worker_id: str,
    shard_index: int,
    reason: str,
) -> None:
    """Worker 조기 종료 시 in-flight shard 를 Storage Node 에 반환."""
    if shard_index < 0:
        return
    msg = pb2.ShardStatus(
        shard_index   = shard_index,
        worker_id     = worker_id,
        status        = "failure",
        error_message = reason,
        num_docs      = 0,
    )
    try:
        stub.ReportShardStatus(msg, timeout=cfg.report_status_timeout_sec)
        logger.warning(
            "[Worker] shard=%d abandon reported: %s", shard_index, reason
        )
    except grpc.RpcError as exc:
        logger.warning("[Worker] abandon ReportShardStatus failed: %s", exc)


# ===========================================================================
# Worker 메인 루프
# ===========================================================================

def run_worker(args: argparse.Namespace) -> None:
    # ── config 생성: CLI 인자가 없는 항목은 gpu_worker_config.py 기본값 사용
    cfg = WorkerConfig.from_args(args)
    cfg.validate()

    # 모듈 레벨 청크 상수를 config 값으로 동기화
    _init_chunk_constants(cfg)

    logging.basicConfig(level=getattr(logging, cfg.log_level, logging.INFO))

    worker_id = cfg.worker_id or f"{socket.gethostname()}-{os.getpid()}"
    fde_config = build_fde_config(cfg)
    encoder    = ColBERTEncoder(
        model_name = cfg.colbert_model,
        device     = cfg.device,
    )

    logger.info(
        "Worker 시작: id=%s  project=%s  server=%s  rep=%d  simhash=%d  "
        "device=%s  fill_empty=%s  grpc_chunk_bytes=%d",
        worker_id, cfg.project_dir, cfg.server,
        cfg.num_repetitions, cfg.num_simhash_projections,
        cfg.device, cfg.fill_empty_partitions,
        cfg.grpc_fde_chunk_bytes,
    )
    log_memory_usage("worker start")

    channel = grpc.insecure_channel(
        cfg.server,
        options=cfg.grpc_channel_options(),
    )
    stub = pb2_grpc.ShardServiceStub(channel)

    shard_count     = 0
    prev_shard_idx  = -1
    prev_status     = ""
    in_flight_shard = -1
    proc_error      = ""
    worker_pipeline_start_ts = time.time()

    # ------------------------------------------------------------------ #
    # 메인 shard 처리 루프
    # ------------------------------------------------------------------ #
    try:
        while True:
            t_loop_start = time.perf_counter()
            t_get_shard_req = time.time()

            # ---- GetShard 요청 ----
            request = pb2.ShardRequest(
                worker_id   = worker_id,
                shard_index = prev_shard_idx,
                fde_config  = config_to_proto(fde_config),
            )
            if prev_shard_idx >= 0:
                request.completed.CopyFrom(
                    pb2.CompletedShardInfo(
                        shard_index   = prev_shard_idx,
                        status        = prev_status,
                        error_message = proc_error if prev_status == "failure" else "",
                    )
                )

            # ---- shard streaming 수신 ----
            t_recv_start = time.perf_counter()
            try:
                stream = stub.GetShard(request, timeout=cfg.get_shard_timeout_sec)
                (
                    shard_index,
                    doc_start,
                    doc_end,
                    doc_ids,
                    metas,
                ) = receive_shard(stream)
            except grpc.RpcError as exc:
                logger.warning("[Worker] GetShard RPC 오류: %s", exc)
                break

            grpc_recv_time = time.perf_counter() - t_recv_start
            t_get_shard_done = time.time()

            if shard_index < 0 or not doc_ids:
                logger.info("[Worker] 더 이상 처리할 shard 없음 — 종료")
                break

            in_flight_shard = shard_index

            logger.info(
                "[Worker] shard=%d docs=[%d,%d) 수신  grpc_recv=%.4fs",
                shard_index, doc_start, doc_end, grpc_recv_time,
            )

            proc_error = ""
            t_process_start = time.time()
            t_process_done  = t_process_start

            # ---- GPU 처리 ----
            try:
                fde_array, timing = process_shard(
                    shard_index = shard_index,
                    doc_start   = doc_start,
                    doc_end     = doc_end,
                    doc_ids     = doc_ids,
                    metas       = metas,
                    config      = fde_config,
                    encoder     = encoder,
                )
                proc_status = "success"
                t_process_done = time.time()
            except Exception as exc:
                logger.exception("[Worker] Shard %d 처리 실패: %s", shard_index, exc)
                proc_status = "failure"
                proc_error  = str(exc)
                fde_array   = None
                timing      = {}
                t_process_done = time.time()

            # ---- FDE 업로드 (gRPC streaming) ----
            grpc_send_time    = 0.0
            remote_flush_time = 0.0
            t_upload_req      = 0.0
            t_upload_done     = 0.0

            if proc_status == "success" and fde_array is not None:
                t_send_start = time.perf_counter()
                t_upload_req = time.time()
                try:
                    ack = stub.UploadFdeShard(
                        fde_upload_stream(
                            shard_index = shard_index,
                            worker_id   = worker_id,
                            fde_array   = fde_array,
                            embed_time  = timing.get("embed_time", 0.0),
                            fde_time    = timing.get("fde_time",   0.0),
                        ),
                        timeout=cfg.upload_fde_timeout_sec,
                    )
                    fde_bytes = fde_array.nbytes
                    validate_upload_ack(ack, shard_index, fde_bytes)
                    grpc_send_time    = time.perf_counter() - t_send_start
                    t_upload_done     = time.time()
                    remote_flush_time = ack.remote_flush_time_s
                    del fde_array
                    fde_array = None
                    gc.collect()

                    logger.info(
                        "[PERF] shard=%d worker=%s | "
                        "grpc_send(RPC total)=%.6fs | "
                        "remote_flush(subset)=%.6fs | "
                        "fde_size_mb=%.3f",
                        shard_index, worker_id,
                        grpc_send_time,
                        remote_flush_time,
                        fde_bytes / (1024 * 1024),
                    )

                except grpc.RpcError as exc:
                    t_upload_done = time.time()
                    logger.error("[Worker] UploadFdeShard RPC 오류: %s", exc)
                    proc_status = "failure"
                    proc_error  = str(exc)
                except RuntimeError as exc:
                    t_upload_done = time.time()
                    logger.error("[Worker] UploadFdeShard 검증 실패: %s", exc)
                    proc_status = "failure"
                    proc_error  = str(exc)

            end_to_end = time.perf_counter() - t_loop_start
            t_status_report = time.time()
            worker_wall_total_s = t_status_report - t_get_shard_req

            # ---- ShardStatus 보고 ----
            status_msg = pb2.ShardStatus(
                shard_index         = shard_index,
                worker_id           = worker_id,
                status              = proc_status,
                error_message       = proc_error,
                num_docs            = len(doc_ids),
                doc_start           = doc_start,
                doc_end             = doc_end,
                prep_time_s         = timing.get("prep_time",      0.0),
                upload_time_s       = timing.get("upload_time",    0.0),
                simhash_time_s      = timing.get("simhash_time",   0.0),
                partition_time_s    = timing.get("partition_time", 0.0),
                scatter_time_s      = timing.get("scatter_time",   0.0),
                average_time_s      = timing.get("average_time",   0.0),
                fill_time_s         = timing.get("fill_time",      0.0),
                compute_time_s      = timing.get("compute_time",   0.0),
                download_time_s     = timing.get("download_time",  0.0),
                reshape_time_s      = timing.get("reshape_time",   0.0),
                flush_time_s        = timing.get("flush_time",     0.0),
                fde_total_time_s    = timing.get("fde_time",       0.0),
                embed_time_s        = timing.get("embed_time",     0.0),
                grpc_recv_time_s    = grpc_recv_time,
                grpc_send_time_s    = grpc_send_time,
                remote_flush_time_s = remote_flush_time,
                end_to_end_time_s   = end_to_end,
                worker_get_shard_req_ts  = t_get_shard_req,
                worker_get_shard_done_ts = t_get_shard_done,
                worker_process_start_ts  = t_process_start,
                worker_process_done_ts   = t_process_done,
                worker_upload_req_ts     = t_upload_req,
                worker_upload_done_ts    = t_upload_done,
                worker_status_report_ts  = t_status_report,
            )

            try:
                stub.ReportShardStatus(status_msg, timeout=cfg.report_status_timeout_sec)
            except grpc.RpcError as exc:
                logger.warning("[Worker] ReportShardStatus 오류: %s", exc)

            in_flight_shard = -1
            prev_shard_idx  = shard_index
            prev_status     = proc_status
            shard_count    += 1

            logger.info(
                "[Worker] shard=%d 완료  end_to_end=%.3fs  wall_total=%.3fs  "
                "(recv=%.3fs embed=%.3fs fde=%.3fs send=%.3fs)  "
                "ts: get_req=%.3f get_done=%.3f proc_done=%.3f "
                "upload_done=%.3f status=%.3f",
                shard_index, end_to_end, worker_wall_total_s,
                grpc_recv_time,
                timing.get("embed_time",  0),
                timing.get("fde_time",    0),
                grpc_send_time,
                t_get_shard_req, t_get_shard_done, t_process_done,
                t_upload_done, t_status_report,
            )
            log_memory_usage(f"shard {shard_index} done")

    finally:
        if in_flight_shard >= 0:
            _report_shard_abandon(
                stub, cfg, worker_id, in_flight_shard, "worker early exit"
            )
        channel.close()
        worker_pipeline_end_ts = time.time()
        logger.info(
            "[Worker] 종료. 처리한 shard 수: %d  pipeline_wall_total=%.3fs  "
            "(start=%.3f end=%.3f)",
            shard_count,
            worker_pipeline_end_ts - worker_pipeline_start_ts,
            worker_pipeline_start_ts,
            worker_pipeline_end_ts,
        )
        log_memory_usage("worker end")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """
    CLI 파서. 기본값은 WorkerConfig 에서 가져오므로
    --help 실행 시 config 파일 값이 표시된다.
    """
    _defaults = WorkerConfig()
    p = argparse.ArgumentParser(
        description="FDE GPU Worker (gRPC client) — 기본값은 gpu_worker_config.py 참고",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--server",       default=_defaults.server,
                   help="Storage Node gRPC address (163.239.199.208:50051)")
    p.add_argument("--project-dir",  default=None,
                   help="Worker 프로젝트 루트 (기본: 호스트명으로 자동 선택)")
    p.add_argument("--worker-id",    default=_defaults.worker_id,
                   help="Worker 식별자 (기본: hostname-PID)")
    p.add_argument("--rep",          type=int, default=_defaults.num_repetitions,
                   help="num_repetitions (FDE 반복 횟수)")
    p.add_argument("--simhash",      type=int, default=_defaults.num_simhash_projections,
                   help=f"num_simhash_projections (1~{MAX_SIMHASH_PROJECTIONS})")
    p.add_argument("--projection",   type=int, default=_defaults.projection_dimension,
                   help="AMS projection_dimension (None=identity)")
    p.add_argument("--fill-empty",   action="store_true",
                   default=_defaults.fill_empty_partitions,
                   help="fill_empty_partitions 활성화")
    p.add_argument("--colbert-model",default=_defaults.colbert_model,
                   help="ColBERT 모델 이름 (문서 텍스트 인코딩에 사용)")
    p.add_argument("--device",       default=_defaults.device,
                   help="PyTorch device (cuda/cpu)")
    return p.parse_args()


if __name__ == "__main__":
    run_worker(parse_args())
