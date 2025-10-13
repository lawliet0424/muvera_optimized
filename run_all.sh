#!/bin/bash

set -euo pipefail

for P in 3 4 5; do
  for R in 2; do
    log="pipeline_P${P}_R${R}.log"
    echo "[$(date +'%F %T')] START P=${P} R=${R}" | tee -a "$log"
    if ./run_fde_autopipeline.sh -p "$P" -r "$R" 2>&1 | tee -a "$log"; then
      echo "[$(date +'%F %T')] DONE  P=${P} R=${R}" | tee -a "$log"
    else
      echo "[$(date +'%F %T')] FAILED P=${P} R=${R}" | tee -a "$log"
      exit 1
    fi
  done
done
