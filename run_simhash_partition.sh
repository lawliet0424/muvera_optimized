#!/bin/bash
#----rerank: 0 (순수한 FDE 검색을 통한 recall rate 구하기)

# Usage: ./run_simhash_partition.sh <dataset> <rerank>
# Example: ./run_simhash_partition.sh scidocs 0

if [ $# -ne 2 ]; then
    echo "Usage: $0 <dataset> <rerank>"
    echo "Example: $0 scidocs 0"
    exit 1
fi

dataset="$1"
rerank="$2"

global_dir="/media/hyunji/7672b947-0099-4e49-8e90-525a208d54b8/muvera_optimized/cache_muvera/${dataset}/main_weight/query_search"

for rep in 1 2 3 4 5 6 7 8 9 10; do
    SIMHASH=4
    echo "rep=${rep} | rerank=${rerank} | simhash=${SIMHASH}"
    dir_name="rep${rep}_simhash${SIMHASH}_rerank${rerank}"
    simhash_count_path="${global_dir}/${dir_name}/simhash_count.csv"
    output_path="${global_dir}/${dir_name}/simhash_partition_statistic.csv"
    python3 simhash_partition_statistic.py "${simhash_count_path}" --metrics all --output "${output_path}"
done

for simhash in 1 2 3 4 5 6 7 8 9; do
    REP=1
    echo "rep=${REP} | rerank=${rerank} | simhash=${simhash}"
    dir_name="rep${REP}_simhash${simhash}_rerank${rerank}"
    simhash_count_path="${global_dir}/${dir_name}/simhash_count.csv"
    output_path="${global_dir}/${dir_name}/simhash_partition_statistic.csv"
    python3 simhash_partition_statistic.py "${simhash_count_path}" --metrics all --output "${output_path}"
done



