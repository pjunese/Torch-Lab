import numpy as np
import faiss
import time


# 1) 데이터 준비 (예시)
np.random.seed(0)
n = 200_000      # DB 벡터 수
d = 512          # 차원
k = 10           # top-k
nq = 5           # 쿼리 수

xb = np.random.randn(n, d).astype(np.float32)
xq = np.random.randn(nq, d).astype(np.float32)

# 코사인 유사도로 쓰고 싶으면 L2 정규화 후 Inner Product(IP) 사용
def l2_normalize(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

xb = l2_normalize(xb)
xq = l2_normalize(xq)


# 2) HNSW 인덱스 생성
M = 32  # 노드당 연결 수(대략). 커질수록 품질↑/메모리↑
index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)

# 빌드 품질(클수록 품질↑/빌드시간↑)
index.hnsw.efConstruction = 200


# 3) DB 추가(빌드)
t0 = time.time()
index.add(xb)
t1 = time.time()
print("HNSW build time (sec):", t1 - t0)


# 4) 검색 (efSearch가 핵심 튜닝)
index.hnsw.efSearch = 64  # 클수록 recall↑ / latency↑

t0 = time.time()
D, I = index.search(xq, k)
t1 = time.time()
print("HNSW search time (sec):", t1 - t0)

print("Query0 top-k ids:", I[0])
print("Query0 top-k scores:", D[0])


import numpy as np
import faiss
import time


# 1) 데이터 준비 (예시)
np.random.seed(0)
n = 500_000
d = 512
k = 10
nq = 5

xb = np.random.randn(n, d).astype(np.float32)
xq = np.random.randn(nq, d).astype(np.float32)

def l2_normalize(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

# 코사인 유사도 용도라면 정규화 + IP
xb = l2_normalize(xb)
xq = l2_normalize(xq)


# 2) IVF-PQ 파라미터
nlist = 4096  # 클러스터 개수(코스 센트로이드)
m = 32        # PQ subquantizer 개수 (d가 m으로 나누어떨어지면 좋음)
nbits = 8     # 각 subvector를 2^nbits code로 양자화 (보통 8)


# 3) 인덱스 생성 (coarse quantizer + IVFPQ)
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)


# 4) 반드시 train 필요
t0 = time.time()
index.train(xb)
t1 = time.time()
print("IVF-PQ train time (sec):", t1 - t0)


# 5) DB 추가
t0 = time.time()
index.add(xb)
t1 = time.time()
print("IVF-PQ add time (sec):", t1 - t0)


# 6) 검색 (nprobe가 핵심 튜닝)
index.nprobe = 16  # 탐색할 클러스터 수 (8/16/32/64로 튜닝)

t0 = time.time()
D, I = index.search(xq, k)
t1 = time.time()
print("IVF-PQ search time (sec):", t1 - t0)

print("Query0 top-k ids:", I[0])
print("Query0 top-k scores:", D[0])
