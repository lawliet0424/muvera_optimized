#!/bin/bash
#----rerank: 0 (순수한 FDE 검색을 통한 recall rate 구하기)
rerank=0
filename=$1

for rep in 1 2 3 4 5 6 7 8 9 10; do
    SIMHASH=4
    echo "rep=${rep} | rerank=${rerank} | simhash=${SIMHASH}"
    python3.10 ${filename} --simhash "${SIMHASH}" --rep "${rep}" --rerank "${rerank}"
done

for simhash in 1 2 3 4 5 6 7 8 9; do
    REP=1
    echo "rep=${REP} | rerank=${rerank} | simhash=${simhash}"
    python3.10 ${filename} --simhash "${simhash}" --rep "${REP}" --rerank "${rerank}"
done



