#!/bin/bash
#----rerank: 0 (순수한 FDE 검색을 통한 recall rate 구하기)
rerank=0

for rep in 1 2 3 4 5 6 7 8 9 10; do
    SIMHASH=4
    echo "rep=${rep} | rerank=${rerank} | simhash=${SIMHASH}"
    python3.10 main_weight_kmeans.py --simhash "${SIMHASH}" --rep "${rep}" --rerank "${rerank}"
done




