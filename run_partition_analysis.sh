#!/bin/bash

# Usage: ./run_partition_analysis.sh <dataset> <rerank>
# Example: ./run_partition_analysis.sh scidocs 0

if [ $# -ne 2 ]; then
    echo "Usage: $0 <dataset> <rerank>"
    echo "Example: $0 scidocs 0"
    exit 1
fi

dataset="$1"
rerank="$2"

echo "=========================================="
echo "Running partition analysis for:"
echo "  Dataset: $dataset"
echo "  Rerank: $rerank"
echo "=========================================="

# run_simhash_partition.sh 실행
./run_simhash_partition.sh "$dataset" "$rerank"

echo "=========================================="
echo "Partition analysis completed!"
echo "=========================================="