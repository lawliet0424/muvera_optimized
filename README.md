(1009) FDE/IVF Indexing Automation
    <br />
    * 1. FDE(query/document) 생성
    <br />
    * 2. FDE를 IVFIP(Inner Product) 인덱싱
    <br />
    위 2개의 프로세스를 자동화함. 자동화 프로세스에 요구되는 코드는 아래와 같음.
    <br />
    * run_all.sh (자동화 상위 프로그램, 여기서 num_simHashPartition 개수와 num_repetition 수를 설정함)
    <br />
    * run_fde_autopipeline.sh (build_fde.py, indexing_fdeivf_search.py에 넘기는 인자 설정)
    <br />
    * build_fde.py: FDE 생성하는 코드
    <br />
    * indexing_fdeivf_search.py: 기 만들어진 FDE를 이용하여 document IVF Indexing
    <br />
    * output: fde_index_3_2.mmap(이 때, 3: num_simHashPartition, 2: num_repetition), fde_index_3_2.pkl, ivf1000_ip_3_2.faiss, meta_3_2.json)
    <br />

(1017) nDCG@K metrics, make brute-force sets for achieving more accurate results.
    <br />
    * Brute-force sets based recall/nDCG@K metric 구하기
    <br />
      - 한 번이라도, top-k에 대한 정답셋 파일을 생성했으면, 다음에 같은 실험을 할 때 이미 만든 정답셋을 활용하여 metric을 계산함. 데이터셋을 바꾸고 싶을 때는 DATASET_REPO_ID과 dataset 변수를 바꿔주면 됨.
    <br />
      - repeated_baseline.sh: rerank candidate 수를 다르게 하면서 반복실험. 이 파일을 내가 타겟하는 문제를 쉽게 보기 위해 만든 거라 참고만 하면됨(편의용).