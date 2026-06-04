#!/bin/bash

rm -rf ./cache_muvera/per_experiment_*

for RC in 50 45 40 35 30 25 20 15 10 5; do
  python3 indexing_fdeivf_search_basedbf.py --p 4 --r 2 --nlist 1000 \
  -i "${FDE_PKL}" -o "${FAISS_OUT}" -rc "${RC}" ${FORCE} 
done