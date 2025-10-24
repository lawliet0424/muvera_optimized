#!/bin/bash
#----rerank: 0 (순수한 FDE 검색을 통한 recall rate 구하기)
rerank=0

for simhash in 1 2 3 4 5 6 7 8 9; do
    REP=1
    echo "rep=${REP} | rerank=${rerank} | simhash=${simhash}"
    python3.10 main_weight_kmeans.py --simhash "${simhash}" --rep "${REP}" --rerank "${rerank}"
done



