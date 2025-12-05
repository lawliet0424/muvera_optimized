#!/bin/bash
#----rerank: 0 (순수한 FDE 검색을 통한 recall rate 구하기)
rerank=0
filename_1=$1

PROJECTION_1=8
PROJECTION_2=16

REP=20
SIMHASH_1=3
SIMHASH_2=4
SIMHASH_3=5

T_REP1=4
T_REP2=8
T_REP3=12
T_REP4=16


T_PROJECTION=128
T_REP=8

#echo "rep=${T_REP1} | rerank=${rerank} | simhash=${SIMHASH_1} | projection=${PROJECTION_1}"
#python3.10 ${filename_1} --simhash "${SIMHASH_1}" --projection "${PROJECTION_1}" --rep "${T_REP1}" --rerank "${rerank}"

#echo "rep=${T_REP2} | rerank=${rerank} | simhash=${SIMHASH_1} | projection=${PROJECTION_1}"
#python3.10 ${filename_1} --simhash "${SIMHASH_1}" --projection "${PROJECTION_1}" --rep "${T_REP2}" --rerank "${rerank}"

echo "rep=${T_REP4} | rerank=${rerank} | simhash=${SIMHASH_1} | projection=${PROJECTION_1}"
python3.10 ${filename_1} --simhash "${SIMHASH_1}" --projection "${PROJECTION_1}" --rep "${T_REP4}" --rerank "${rerank}"


#echo "rep=${T_REP1} | rerank=${rerank} | simhash=${SIMHASH_1} | projection=${PROJECTION_1}"
#python3.10 ${filename_2} --simhash "${SIMHASH_1}" --projection "${PROJECTION_1}" --rep "${T_REP1}" --rerank "${rerank}"

#echo "rep=${T_REP2} | rerank=${rerank} | simhash=${SIMHASH_1} | projection=${PROJECTION_1}"
#python3.10 ${filename_2} --simhash "${SIMHASH_1}" --projection "${PROJECTION_1}" --rep "${T_REP2}" --rerank "${rerank}"

#echo "rep=${T_REP3} | rerank=${rerank} | simhash=${SIMHASH_1} | projection=${PROJECTION_1}"
#python3.10 ${filename_2} --simhash "${SIMHASH_1}" --projection "${PROJECTION_1}" --rep "${T_REP3}" --rerank "${rerank}"


#echo "rep=${REP} | rerank=${rerank} | simhash=${SIMHASH_2} | projection=${PROJECTION_1}"
#python3.10 ${filename} --simhash "${SIMHASH_2}" --projection "${PROJECTION_1}" --rep "${REP}" --rerank "${rerank}"

#echo "rep=${REP} | rerank=${rerank} | simhash=${SIMHASH_3} | projection=${PROJECTION_1}"
#python3.10 ${filename} --simhash "${SIMHASH_3}" --projection "${PROJECTION_1}" --rep "${REP}" --rerank "${rerank}"

#echo "rep=${REP} | rerank=${rerank} | simhash=${SIMHASH_3} | projection=${PROJECTION_2}"
#python3.10 ${filename} --simhash "${SIMHASH_3}" --projection "${PROJECTION_2}" --rep "${REP}" --rerank "${rerank}"


