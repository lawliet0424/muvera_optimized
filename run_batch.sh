#!/usr/bin/env bash
# run for batch size 12000, 8000, 4000, 2000

drop_os_page_cache() {
  echo "[run_batch] $(date -Iseconds) sync + /proc/sys/vm/drop_caches"
  sync
  if echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null; then
    echo "[run_batch] OS page cache dropped."
  else
    echo "[run_batch] WARNING: drop_caches failed (sudo may be required). Continuing."
  fi
}

run_cmd() {
  echo "[run_batch] === $* ==="
  "$@"
}

# 실험 시작 전 커널 페이지 캐시 비우기
#drop_os_page_cache

#python main_optimized.py --rep 2 --simhash 4 --batch-size 12000
#drop_os_page_cache
#python main_optimized_gpu_triple_stage_optimized.py --rep 4 --simhash 4 --batch-size 12000
python main_optimized_gpu_triple_stage_optimized.py --rep 4 --simhash 4 --batch-size 12000
#drop_os_page_cache

#run_cmd python main_optimized.py --rep 2 --simhash 4 --batch-size 10000
#drop_os_page_cache
#run_cmd python main_optimized_gpu_triple_stage_optimized.py --rep 4 --simhash 4 --batch-size 10000
#drop_os_page_cache
#run_cmd python main_optimized.py --rep 2 --simhash 4 --batch-size 8000
#drop_os_page_cache
python main_optimized_gpu_triple_stage_optimized.py --rep 4 --simhash 4 --batch-size 8000
#drop_os_page_cache
#run_cmd python main_optimized.py --rep 2 --simhash 4 --batch-size 6000
#drop_os_page_cache
python main_optimized_gpu_triple_stage_optimized.py --rep 4 --simhash 4 --batch-size 6000
#drop_os_page_cache
#run_cmd python main_optimized.py --rep 2 --simhash 4 --batch-size 4000
#drop_os_page_cache
#run_cmd python main_optimized_gpu_triple_stage_optimized.py --rep 4 --simhash 4 --batch-size 4000
#drop_os_page_cache
#run_cmd python main_optimized.py --rep 2 --simhash 4 --batch-size 2000
#drop_os_page_cache
python main_optimized_gpu_triple_stage_optimized.py --rep 4 --simhash 4 --batch-size 2000

# 실험 종료 후 커널 페이지 캐시 비우기
#drop_os_page_cache

echo "[run_batch] All experiments done."
