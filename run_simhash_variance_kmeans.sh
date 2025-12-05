#!/bin/bash
#----rerank: 0 (순수한 FDE 검색을 통한 recall rate 구하기)

filename=$1
rerank=0

for simhash in 4; do
    REP=10
    echo "rep=${REP} | rerank=${rerank} | simhash=${simhash}"
    python3.10 ${filename} --simhash "${simhash}" --rep "${REP}" --rerank "${rerank}"
done



