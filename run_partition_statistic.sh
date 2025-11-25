#!/bin/bash

#----rerank: 0 (순수한 FDE 검색을 통한 recall rate 구하기)
rerank=0
statistic_file_name="partition_statistic.py"
csv_file_name="partition_count.csv"
output_file_name="partition_statistic_result.csv"
output_mask_file_name="partition_masking.csv"
output_rep_stats_file_name="partition_utilization.csv"

dataset=$1
filename=$2
method=$3

'usage: ./run_partition_statistic.sh <dataset> <filename> <method> (~/Desktop/muvera_optimized 경로에서 실행)'
'example: ./run_partition_statistic.sh scidocs main_weight_kmeans_gpu kmeans'

#for rep in 1 2 3 4 5 6 7 8 9 10; do
#    PARTITION_IDX=4
#    echo "rep=${rep} | rerank=${rerank} | ${method}=${PARTITION_IDX}"
#    python3 ${statistic_file_name} --dataset "${dataset}" --filename "${filename}" --method "${method}" --csv_file "${csv_file_name}" --rep "${rep}" --partition_idx "${PARTITION_IDX}" --rerank "${rerank}" --output ${output_file_name} --output_mask ${output_mask_file_name} --output_rep_stats ${output_rep_stats_file_name}
#done

for partition_idx in 1 2 3 4 5 6 7 8 9; do
    REP=1
    echo "rep=${REP} | rerank=${rerank} | ${method}=${simhash}"
    python3 ${statistic_file_name} --dataset "${dataset}" --filename "${filename}" --method "${method}" --csv_file "${csv_file_name}" --rep "${REP}" --partition_idx "${partition_idx}" --rerank "${rerank}" --output ${output_file_name} --output_mask ${output_mask_file_name} --output_rep_stats ${output_rep_stats_file_name}
done