[Requirements]
* python 3.10+
* nltk 3.9.1
* neural-cherche 1.4.3
* faiss-gpu 1.7.2
* numpy 1.26.4

[Main files]
- main_0926.py -> BEIR dataset 다운로드 및 Document/Query FDE indexing.
    * FixedDimensionalEncodingConfig 에서 num_repetition, num_simhash_projections 파라미터 세팅
    * 가장 오래걸리는 시간은 GPU로 document/query embedding 하는 것이므로, 한 번 embedding된 npy들은 디스크에 저장되도록 함. 저장 경로는 코드에 나와있음
- fde_generator_optimized_stream.py
    * FDE 작업 동안 한 번에 n차원의 벡터를 numpy에 밀어넣으면(np.vstack) OOM이 발생하므로, OOM 방지를 위해 stream 방식으로 구현함.
- fde_ivfip.py
    * 559 라인에서 IVF 파라미터 설정값을 변경할 수 있음, faiss_nlist를 1000이상으로 할 경우 시간이 오래걸리는데, index train할 떄 걸리는 것임. 왜 멀티스레드가 안되는지 모르겠음. 멀티 스레드가 된다면 개선 후 commit 바람

[Statistic]
e.g. 
python3 muvera_statistic.py latency.tsv --columns Search Rerank --metrics mean p95 p99 --dataset scidocs --method main_weight

[Notifications]
(1009) New files
* 1. FDE(query/document) 생성
* 2. FDE를 IVFIP(Inner Product) 인덱싱
위 2개의 프로세스를 자동화함. 자동화 프로세스에 요구되는 코드는 아래와 같음.
* run_all.sh (자동화 상위 프로그램, 여기서 num_simHashPartition 개수와 num_repetition 수를 설정함)
* run_fde_autopipeline.sh (build_fde.py, indexing_fdeivf_search.py에 넘기는 인자 설정)
* build_fde.py: FDE 생성하는 코드
* indexing_fdeivf_search.py: 기 만들어진 FDE를 이용하여 document IVF Indexing
* output: fde_index_3_2.mmap(이 때, 3: num_simHashPartition, 2: num_repetition), fde_index_3_2.pkl, ivf1000_ip_3_2.faiss, meta_3_2.json) 

(10/17) New File [Batch 단위로, Encoding -> FDE 생성 -> FDE 인덱스 flush]
* 1. build_batch_fde.py : 44 line에 있는 ATOMIC_BATCH_SIZE = 1000에 의해 문서 1000개 단위로 처리
위의 수정에 따라 요구되는 코드 수정
* run_fde_autopipeline.sh의 (index build)에 있는 파일 이름 변경 [build_fde.py -> build_batch_fde.py]

(1017) nDCG@K metrics, make brute-force sets for achieving more accurate results.
    <br />
    * Brute-force sets based recall/nDCG@K metric 구하기
    <br />
      - 한 번이라도, top-k에 대한 정답셋 파일을 생성했으면, 다음에 같은 실험을 할 때 이미 만든 정답셋을 활용하여 metric을 계산함. 데이터셋을 바꾸고 싶을 때는 DATASET_REPO_ID과 dataset 변수를 바꿔주면 됨.
    <br />
      - repeated_baseline.sh: rerank candidate 수를 다르게 하면서 반복실험. 이 파일을 내가 타겟하는 문제를 쉽게 보기 위해 만든 거라 참고만 하면됨(편의용).