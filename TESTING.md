# gRPC FDE 파이프라인 테스트 가이드

## 전제 조건

| 항목 | 요구사항 |
|---|---|
| Python | 3.10 이상 |
| CUDA | Worker 머신에만 필요 (Storage Node 는 CPU 전용) |
| GPU | Worker 1대 이상 (테스트는 1대로도 가능) |
| 디스크 | 테스트 데이터용 1 GB 여유 공간 |

---

## 디렉터리 구조

모든 파일을 하나의 작업 디렉터리에 모은다.

```
/workspace/
├── fde_pipeline.proto
├── storage_node.py
├── storage_node_config.py
├── gpu_worker.py
├── gpu_worker_config.py
├── perf_logger.py
├── fde_generator_gpu_optimized_triple_stage_optimized.py   ← 원본 파일
└── main_optimized_gpu_triple_stage_optimized.py            ← 원본 파일 (참조용)
```

```bash
mkdir -p /workspace && cd /workspace

# 생성한 파일들을 작업 디렉터리로 복사
cp /path/to/outputs/*.py /path/to/outputs/*.proto .
cp /path/to/fde_generator_gpu_optimized_triple_stage_optimized.py .
```

---

## STEP 1 — 패키지 설치

### Storage Node 머신 (CPU 전용)

```bash
pip install grpcio grpcio-tools numpy
```

### Worker 머신 (GPU 필요)

```bash
pip install grpcio grpcio-tools numpy torch neural-cherche # <-dcceris는 기존에 numpy만 있었음

# CUDA 버전에 맞게 cupy 설치
pip install cupy-cuda12x    # CUDA 12.x
# pip install cupy-cuda11x  # CUDA 11.x
```

---

## STEP 2 — proto 컴파일

Storage Node 와 Worker **양쪽에서** 실행한다.

```bash
cd /workspace

python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    fde_pipeline.proto
```

성공하면 두 파일이 생성된다.

```
fde_pipeline_pb2.py
fde_pipeline_pb2_grpc.py
```

컴파일 오류가 나면 grpcio-tools 버전을 확인한다.

```bash
pip show grpcio-tools   # 1.50.0 이상 권장
```

---

## STEP 3 — 테스트 데이터 생성

실제 BEIR corpus 없이도 동작 확인이 가능한 미니 데이터를 생성한다.

```bash
python3 - << 'EOF'
import json, numpy as np, os

# 테스트용 디렉터리
os.makedirs("/data/doc_embeds", exist_ok=True)
os.makedirs("/data/fde_out",   exist_ok=True)
os.makedirs("/tmp/fde_worker", exist_ok=True)

# corpus.json: 100개 문서
corpus = {
    f"doc_{i:04d}": {
        "title": f"Document {i}",
        "text":  f"This is the body text of document number {i}. " * 5,
    }
    for i in range(100)
}
with open("/data/corpus.json", "w") as f:
    json.dump(corpus, f)
print(f"corpus.json 생성: {len(corpus)}개 문서")

# 사전 계산된 embedding (ColBERT 인코딩 생략 목적)
# shape: (seq_len, 128),  seq_len은 문서마다 다르게
rng = np.random.default_rng(42)
for i in range(100):
    seq_len = rng.integers(20, 80)
    emb = rng.standard_normal((seq_len, 128)).astype(np.float32)
    np.save(f"/data/doc_embeds/{i:08d}.npy", emb)

print("embedding 파일 생성 완료: /data/doc_embeds/00000000.npy ~ 00000099.npy")
EOF
```

---

## STEP 4 — config 파일 수정 (선택)

기본값으로도 테스트 가능하다. 경로나 포트를 바꾸고 싶으면 수정한다.

**`storage_node_config.py`** 핵심 항목:

```python
corpus_path: str = "/data/corpus.json"   # ← STEP 3에서 생성한 경로
embed_dir:   str = "/data/doc_embeds"    # ← 사전 embedding 디렉터리
output_dir:  str = "/data/fde_out"       # ← FDE 결과 저장 경로
num_shards:  int = 4                     # ← 테스트용으로 줄임
port:        int = 50051
```

**`gpu_worker_config.py`** 핵심 항목:

```python
server:                  str = "localhost:50051"  # Storage Node 주소
num_repetitions:         int = 2                  # 빠른 테스트용
num_simhash_projections: int = 4                  # 빠른 테스트용
fill_empty_partitions:  bool = True
device:                  str = "cuda"             # GPU 없으면 "cpu"
tmp_dir:                 str = "/tmp/fde_worker"
```

---

## STEP 5 — Storage Node 실행

터미널 A에서 실행한다.

```bash
cd /workspace

python storage_node.py \
    --corpus-path /data/corpus.json \
    --embed-dir   /data/doc_embeds \
    --output-dir  /data/fde_out \
    --num-shards  4 \
    --port        50051
```

정상 시작 로그:

```
Storage Node started: port=50051  shards=4  docs=100  max_workers=8 ...
```

config 파일의 기본값을 사용하려면 인자 없이도 실행 가능하다
(단, `corpus_path` 와 `output_dir` 이 config 파일에 올바르게 설정되어 있어야 한다).

```bash
python storage_node.py   # config 기본값 사용
```

---

## STEP 6 — GPU Worker 실행

Storage Node 가 실행 중인 상태에서 터미널 B(와 C)에서 실행한다.

### Worker 1대로 테스트

```bash
cd /workspace

CUDA_VISIBLE_DEVICES=0 python gpu_worker.py \
    --server    localhost:50051 \
    --worker-id worker-0 \
    --rep       2 \
    --simhash   4 \
    --fill-empty
```

### Worker 2대로 테스트 (터미널 C 추가)

```bash
# 터미널 C
CUDA_VISIBLE_DEVICES=1 python gpu_worker.py \
    --server    localhost:50051 \
    --worker-id worker-1 \
    --rep       2 \
    --simhash   4 \
    --fill-empty
```

GPU 가 없는 경우 (CPU 모드, FDE 생성 불가 — embedding 수신 테스트용):

```bash
python gpu_worker.py \
    --server  localhost:50051 \
    --device  cpu \
    --rep     2 \
    --simhash 4
```

> **주의**: `fde_generator_gpu_optimized_triple_stage_optimized.py` 는 최상단에
> `import cupy as cp` 가 있으므로 GPU 없이는 import 자체가 실패한다.
> CPU 전용 테스트는 아래 STEP 9 의 단위 테스트를 사용한다.

---

## STEP 7 — 진행 상황 확인

별도 터미널에서 실시간으로 확인한다.

```bash
cd /workspace

python3 - << 'EOF'
import grpc, fde_pipeline_pb2 as pb2, fde_pipeline_pb2_grpc as pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub    = pb2_grpc.ShardServiceStub(channel)
resp    = stub.GetProgress(pb2.ProgressRequest(requester_id="monitor"))

print(f"전체: {resp.total_shards}  완료: {resp.completed_shards}  "
      f"실패: {resp.failed_shards}  대기: {resp.pending_shards}  "
      f"경과: {resp.elapsed_sec:.1f}s")
for s in resp.shard_summaries:
    print(f"  shard={s.shard_index:04d}  status={s.status:<12}  "
          f"worker={s.worker_id}  docs={s.num_docs}  time={s.total_time_s:.2f}s")
channel.close()
EOF
```

---

## STEP 8 — 결과 확인

모든 shard 완료 후 Storage Node 가 자동으로 보고서를 생성한다.

```bash
ls -lh /data/fde_out/
# shard_0000.mmap  shard_0001.mmap  ...
# doc_ids.json
# final_manifest.json
# metrics.csv
```

### 생성된 FDE mmap 내용 검증

```bash
python3 - << 'EOF'
import numpy as np, json

# doc_ids 로드
with open("/data/fde_out/doc_ids.json") as f:
    doc_ids = json.load(f)

# 첫 번째 shard mmap 확인
mm = np.memmap("/data/fde_out/shard_0000.mmap", dtype=np.float32, mode="r")
print(f"shard_0000.mmap raw size: {mm.shape}")

# final_manifest 확인
with open("/data/fde_out/final_manifest.json") as f:
    manifest = json.load(f)

print(f"\n총 shard: {manifest['total_shards']}")
print(f"성공:     {manifest['done']}")
print(f"실패:     {manifest['failed']}")
print(f"총 소요:  {manifest['total_elapsed_sec']:.2f}s")
for s in manifest["shards"]:
    t = s["timing"]
    print(f"  shard={s['shard_index']:04d}  docs={s['num_docs']}  "
          f"e2e={t.get('end_to_end',0):.3f}s  "
          f"fde={t.get('fde_total',0):.3f}s  "
          f"grpc_send={t.get('grpc_send',0):.3f}s  "
          f"local_flush={t.get('local_flush',0):.6f}s")
EOF
```

### metrics.csv 확인

```bash
cat /data/fde_out/metrics.csv
```

---

## STEP 9 — 단위 테스트 (GPU 없는 환경 / 빠른 검증)

Storage Node 와 Worker 를 실제로 띄우지 않고 개별 컴포넌트를 검증한다.

### 9-1. config 파일 검증

```bash
python3 - << 'EOF'
from storage_node_config import StorageNodeConfig
from gpu_worker_config    import WorkerConfig
import os

# Storage Node config 기본값 확인
scfg = StorageNodeConfig()
print("=== StorageNodeConfig ===")
for k, v in vars(scfg).items():
    print(f"  {k} = {v!r}")
print(f"\ngrpc_server_options: {scfg.grpc_server_options()}")

# Worker config 기본값 확인
wcfg = WorkerConfig()
print("\n=== WorkerConfig ===")
for k, v in vars(wcfg).items():
    print(f"  {k} = {v!r}")
print(f"\ngrpc_channel_options: {wcfg.grpc_channel_options()}")

# validate() 테스트 — 파일 없음 예외 확인
try:
    scfg.validate()
except ValueError as e:
    print(f"\n[정상] validate() 예외 확인: {e}")

# 파일이 실제로 있으면 통과
os.makedirs("/data", exist_ok=True)
with open("/data/corpus.json", "w") as f:
    import json; json.dump({"doc_0": {"title": "t", "text": "x"}}, f)
scfg.corpus_path = "/data/corpus.json"
scfg.validate()
print("[정상] 파일 존재 시 validate() 통과")
EOF
```

### 9-2. proto 컴파일 결과 검증

```bash
python3 - << 'EOF'
import fde_pipeline_pb2 as pb2

# ShardTask 메시지 생성/직렬화 테스트
req = pb2.ShardRequest(worker_id="test-worker", shard_index=-1)
serialized = req.SerializeToString()
req2 = pb2.ShardRequest()
req2.ParseFromString(serialized)
assert req2.worker_id == "test-worker"
print("[정상] ShardRequest 직렬화/역직렬화 통과")

# optional completed 필드 HasField 테스트
assert not req.HasField("completed"), "빈 메시지에서 HasField 는 False 여야 함"
req.completed.CopyFrom(pb2.CompletedShardInfo(shard_index=0, status="success"))
assert req.HasField("completed"), "completed 설정 후 HasField 는 True 여야 함"
print("[정상] optional completed HasField 테스트 통과")

# FdeConfig 테스트
cfg = pb2.FdeConfig(dimension=128, num_repetitions=2, num_simhash_projections=4)
print(f"[정상] FdeConfig: dim={cfg.dimension} rep={cfg.num_repetitions}")
EOF
```

### 9-3. Storage Node 핵심 로직 단위 테스트 (grpc 없이)

```bash
python3 - << 'EOF'
import json, math, os
os.makedirs("/data", exist_ok=True)

# 테스트 corpus 생성
corpus = {f"doc_{i}": {"title": f"T{i}", "text": f"Body {i}"} for i in range(20)}
with open("/data/corpus.json", "w") as f:
    json.dump(corpus, f)

# compute_shards 테스트
import sys; sys.path.insert(0, "/workspace")
from storage_node import compute_shards, CorpusLoader, ShardDispatcher

shards = compute_shards(total_docs=20, num_shards=4)
print(f"[정상] compute_shards(20, 4) = {shards}")
assert len(shards) == 4
assert shards[-1][1] == 20

# CorpusLoader 테스트
loader = CorpusLoader("/data/corpus.json", embed_dir=None)
assert loader.total_docs() == 20
doc_id, doc = loader.get_doc(0)
print(f"[정상] CorpusLoader: doc[0] = {doc_id!r} → {doc}")

# ShardDispatcher 테스트
disp = ShardDispatcher(shards)
idx0 = disp.next_pending("worker-0")
idx1 = disp.next_pending("worker-1")
print(f"[정상] ShardDispatcher: worker-0={idx0}, worker-1={idx1}")
assert idx0 != idx1

disp.mark_done(idx0, {"end_to_end": 1.23})
summary = disp.summary()
print(f"[정상] summary: {summary}")
EOF
```

### 9-4. gRPC 서버 연결 테스트 (Storage Node 단독)

Storage Node 를 백그라운드로 띄우고 연결만 확인한다.

```bash
# 백그라운드 실행
python storage_node.py \
    --corpus-path /data/corpus.json \
    --output-dir  /tmp/fde_test_out \
    --num-shards  2 \
    --port        50099 &
STORAGE_PID=$!
sleep 2

# 연결 확인
python3 - << 'EOF'
import grpc, fde_pipeline_pb2 as pb2, fde_pipeline_pb2_grpc as pb2_grpc

channel = grpc.insecure_channel("localhost:50099")
stub    = pb2_grpc.ShardServiceStub(channel)
resp    = stub.GetProgress(pb2.ProgressRequest(requester_id="test"))
print(f"[정상] GetProgress: total={resp.total_shards}  pending={resp.pending_shards}")
channel.close()
EOF

kill $STORAGE_PID 2>/dev/null
echo "Storage Node 종료"
```

---

## STEP 10 — 성능 리포트 생성

```bash
cd /workspace

python perf_logger.py \
    --manifest /data/fde_out/final_manifest.json \
    --output   /data/fde_out/perf_report.csv
```

출력 예시:

```
========================================================================
  Performance Summary: manifest-analysis
========================================================================
  Total shards     : 4  (success=4)
  Total docs       : 100
  ...
  ────────────────────────────────────────────────────
  I/O Comparison (per shard avg)
  ────────────────────────────────────────────────────
  [BASELINE] local SSD flush : avg=0.000821s
  [gRPC]     grpc send       : avg=0.003412s
  [gRPC]     remote flush    : avg=0.000634s
  [gRPC]     total           : avg=0.004046s
  ────────────────────────────────────────────────────
  gRPC overhead vs local SSD : +0.003225s/shard → gRPC is SLOWER
========================================================================
```

---

## 자주 발생하는 오류

### `ModuleNotFoundError: No module named 'fde_pipeline_pb2'`

proto 를 컴파일하지 않았거나 작업 디렉터리가 다르다.

```bash
cd /workspace
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. fde_pipeline.proto
```

### `ModuleNotFoundError: No module named 'cupy'`

Worker 머신에 cupy 가 설치되지 않았다.

```bash
pip install cupy-cuda12x   # CUDA 12.x 기준
python -c "import cupy; print(cupy.__version__)"
```

### `ValueError: [StorageNodeConfig] corpus_path 파일 없음`

config 파일 또는 CLI 인자의 경로를 확인한다.

```bash
ls -la /data/corpus.json
# 없으면 STEP 3 의 데이터 생성 스크립트를 먼저 실행
```

### `grpc._channel._InactiveRpcError: UNAVAILABLE`

Storage Node 가 실행 중이지 않거나 포트/주소가 다르다.

```bash
# Storage Node 가 실행 중인지 확인
ss -tlnp | grep 50051

# Worker 의 --server 인자와 Storage Node 의 --port 가 일치하는지 확인
```

### `HasField() called on non-optional field`

proto 파일에 `optional` 키워드가 빠진 경우다. proto 를 재컴파일한다.

```bash
grep "optional CompletedShardInfo" fde_pipeline.proto
# 위 줄이 없으면 proto 파일을 다시 받아서 재컴파일
```

### `RuntimeError: CUDA error: no kernel image is available`

cupy 의 CUDA 버전과 실제 드라이버가 맞지 않는다.

```bash
nvidia-smi                        # 드라이버 CUDA 버전 확인
python -c "import cupy; cupy.show_config()"
pip install cupy-cuda11x          # 버전에 맞게 재설치
```
