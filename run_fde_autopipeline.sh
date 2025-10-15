#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_fde_pipeline.sh -p <P> -r <R> [-n <NLIST>] [--force]

Examples:
  ./run_fde_pipeline.sh -p 128 -r 8
  ./run_fde_pipeline.sh -p 128 -r 8 -n 2000 --force

Tips:
  # 백그라운드 실행 + 실시간 로그 보기
  nohup ./run_fde_pipeline.sh -p 128 -r 8 > pipeline_P128_R8.log 2>&1 &
  tail -f pipeline_P128_R8.log
USAGE
}

P=""
R=""
NLIST="1000"
FORCE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--num_simhash_projections) P="$2"; shift 2;;
    -r|--num_repetitions) R="$2"; shift 2;;
    -n|--nlist) NLIST="$2"; shift 2;;
    --force) FORCE="--force"; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "${P}" || -z "${R}" ]]; then
  echo "[pipeline] Missing -p/-r"; usage; exit 1
fi

FDE_PKL="fde_index_${P}_${R}.pkl"
FAISS_OUT="ivf${NLIST}_ip_${P}_${R}.faiss"

echo "[pipeline] === STEP 1: Build FDE ===" # build_fde.py build_fdeivf_indexing.py
python3.10 build_batch_fde_pca.py --p "${P}" --r "${R}" -o "${FDE_PKL}" ${FORCE}

echo "[pipeline] === STEP 2: Build FAISS IVF-IP ==="
python3.10 indexing_fdeivf_naive_batching_annrerank.py --p "${P}" --r "${R}" --nlist "${NLIST}" \
  -i "${FDE_PKL}" -o "${FAISS_OUT}" ${FORCE}

echo "[pipeline] DONE: ${FDE_PKL} -> ${FAISS_OUT}"
