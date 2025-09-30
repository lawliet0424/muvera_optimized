[Requirements]
* python 3.10+
* nltk 3.9.1
* neural-cherche 1.4.3
* faiss-gpu 1.7.2

[Main files]
- main_0926.py -> BEIR dataset 다운로드 및 Document/Query FDE indexing.
    * FixedDimensionalEncodingConfig 에서 num_repetition, num_simhash_projections 파라미터 세팅
    * 가장 오래걸리는 시간은 GPU로 document/query embedding 하는 것이므로, 한 번 embedding된 npy들은 디스크에 저장되도록 함. 저장 경로는 코드에 나와있음
- fde_generator_optimized_stream.py
    * FDE 작업 동안 한 번에 n차원의 벡터를 numpy에 밀어넣으면(np.vstack) OOM이 발생하므로, OOM 방지를 위해 stream 방식으로 구현함.
- fde_ivfip.py
    * 559 라인에서 IVF 파라미터 설정값을 변경할 수 있음, faiss_nlist를 1000이상으로 할 경우 시간이 오래걸리는데, index train할 떄 걸리는 것임. 왜 멀티스레드가 안되는지 모르겠음. 멀티 스레드가 된다면 개선 후 commit 바람
