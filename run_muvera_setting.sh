#!/bin/bash
#----rerank: 0 (순수한 FDE 검색을 통한 recall rate 구하기)
rerank=0
filename=$1

PROJECTION_1=8
PROJECTION_2=16

REP=20
SIMHASH_1=3
SIMHASH_2=4
SIMHASH_3=5

T_SIMHASH=4
T_PROJECTION=128
T_REP=8

echo "rep=${REP} | rerank=${rerank} | simhash=${SIMHASH_1} | projection=${PROJECTION_1}"
python3.10 ${filename} --simhash "${SIMHASH_1}" --projection "${PROJECTION_1}" --rep "${REP}" --rerank "${rerank}"

#echo "rep=${T_REP} | rerank=${rerank} | simhash=${T_SIMHASH} | projection=${T_PROJECTION}"
#python3.10 ${filename} --simhash "${T_SIMHASH}" --projection "${T_PROJECTION}" --rep "${T_REP}" --rerank "${rerank}"

#echo "rep=${REP} | rerank=${rerank} | simhash=${SIMHASH_2} | projection=${PROJECTION_1}"
#python3.10 ${filename} --simhash "${SIMHASH_2}" --projection "${PROJECTION_1}" --rep "${REP}" --rerank "${rerank}"

#echo "rep=${REP} | rerank=${rerank} | simhash=${SIMHASH_3} | projection=${PROJECTION_1}"
#python3.10 ${filename} --simhash "${SIMHASH_3}" --projection "${PROJECTION_1}" --rep "${REP}" --rerank "${rerank}"

#echo "rep=${REP} | rerank=${rerank} | simhash=${SIMHASH_3} | projection=${PROJECTION_2}"
#python3.10 ${filename} --simhash "${SIMHASH_3}" --projection "${PROJECTION_2}" --rep "${REP}" --rerank "${rerank}"


