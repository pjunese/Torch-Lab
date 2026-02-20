## img_guard

이미지 등록 요청에 대해 **유사도 탐색 + pHash 검증 + 3상태 판정(ALLOW/REVIEW/BLOCK)**을 수행하는 모듈입니다.
로컬(HNSW) 기반 테스트와 서비스(pgvector) 연동을 모두 지원합니다.

**핵심 흐름**
- 이미지 다운로드/로드
- 임베딩 생성(CLIP)
- ANN Top‑K 검색
- pHash 정밀 비교
- 정책 판정(ALLOW/REVIEW/BLOCK)

**구성**
- `app/ann_index.py`: 로컬 HNSW 또는 pgvector 검색
- `app/embedder.py`: CLIP 임베딩
- `app/phash.py`: pHash 비교
- `app/policy.py`: 3상태 판정 규칙
- `app/guard.py`: 파이프라인 오케스트레이션
- `app/api.py`: FastAPI Guard API
- `experiments/`: 특징점 매칭/회전 실험 스크립트

---

## 빠른 시작 (로컬)

```bash
cd /Users/pjunese/Desktop/WATSON/img_guard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

데이터 위치:
- `data/db_images/dataset60/`에 DB 이미지가 있어야 합니다.

로컬 CLI 테스트:
```bash
python3 -m app.main --query data/db_images/dataset60/dt_001.png --json
```

---

## Guard API (FastAPI)

서버 실행:
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

엔드포인트:
- `GET /health`
- `POST /v1/guard/image`

---

## 서비스 모드 (pgvector)

기본은 로컬(HNSW)입니다. 서비스(pgvector)로 전환하려면 환경변수를 설정하세요.

```bash
export ANN_BACKEND=pgvector
export VECTOR_DSN="postgresql://user:pass@host:5432/db"
export VECTOR_TABLE="image_embeddings"
export VECTOR_EMBED_COL="embedding"
export VECTOR_ID_COL="id"
export VECTOR_FILE_COL="file_name"
export VECTOR_KEY_COL="s3_key"
export VECTOR_URL_COL="asset_url"
export VECTOR_PHASH_COL="phash"   # DB에 pHash 저장 시 사용
```

---

## 실험 스크립트

특징점 매칭 + RANSAC (시각화 자동 열기):
```bash
cd /Users/pjunese/Desktop/WATSON/img_guard/experiments
python3 feature_match_ransac.py \
  --query ../data/db_images/dataset60/dt_028.png \
  --cand  ../data/db_images/dataset60/dt_028_1.png \
  --out /tmp/fm_out \
  --method orb \
  --open
```

회전 데이터 생성 (10~270도, 10도 간격):
```bash
cd /Users/pjunese/Desktop/WATSON/img_guard/experiments
python3 rotate_dataset.py \
  --input ../data/db_images/dataset60/dt_028.png \
  --out ../data/db_images/dataset60_rot_10_270
```

---

## 실험 결과

![res1](assets/res1.png)
![res2](assets/res2.png)

