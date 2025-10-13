[Main files]
- main_0926.py -> BEIR dataset 다운로드 및 Document/Query FDE indexing.
    * FixedDimensionalEncodingConfig 에서 num_repetition, num_simhash_projections 파라미터 세팅
    * 가장 오래걸리는 시간은 GPU로 document/query embedding 하는 것이므로, 한 번 embedding된 npy들은 디스크에 저장되도록 함. 저장 경로는 코드에 나와있음
- fde_generator_optimized_stream.py
    * FDE 작업 동안 한 번에 n차원의 벡터를 numpy에 밀어넣으면(np.vstack) OOM이 발생하므로, OOM 방지를 위해 stream 방식으로 구현함.
- fde_ivfip.py
    * 559 라인에서 IVF 파라미터 설정값을 변경할 수 있음, faiss_nlist를 1000이상으로 할 경우 시간이 오래걸리는데, index train할 떄 걸리는 것임. 왜 멀티스레드가 안되는지 모르겠음. 멀티 스레드가 된다면 개선 후 commit 바람

(1009) New files
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


