# img_guard: 이미지 유사도 기반 진위 판정 시스템

**img_guard**는 CLIP 임베딩과 HNSW(Hierarchical Navigable Small World) 인덱싱을 결합하여 대규모 이미지 데이터베이스 내에서 유사 이미지를 고속으로 검색하고, 정의된 정책에 따라 배포 가능 여부를 판정하는 시스템입니다.

##  주요 기능

* **CLIP 기반 벡터 추출**: OpenAI의 CLIP(ViT-B-32) 모델을 활용하여 이미지의 의미적 특징을 고차원 벡터로 변환하고 L2 정규화를 적용합니다.
* **고속 ANN 검색**: `hnswlib`을 사용하여 수만 장 이상의 이미지 사이에서도 초고속 근사 최근접 이웃(Approximate Nearest Neighbor) 검색을 수행합니다.
* **DB 변경 감지 및 자동 갱신**: 데이터베이스 내 파일의 수정 시간(mtime)과 크기를 기반으로 시그니처를 생성하며, 변경 사항이 감지될 때만 인덱스를 자동으로 재빌드합니다.
* **3단계 정책 판정**: 코사인 유사도(Cosine Similarity)와 pHash 거리를 조합하여 **ALLOW(허용)**, **REVIEW(검토)**, **BLOCK(차단)**의 3단계 결과를 도출합니다.

##  프로젝트 구조 및 역할

프로젝트의 핵심 로직은 다음 4개의 파일로 구성되어 있습니다.

| 파일명 | 역할 설명 |
| :--- | :--- |
| **`config.py`** | 데이터 경로, 모델 설정, 검색 임계값(Threshold) 및 HNSW 파라미터 관리 |
| **`embedder.py`** | CLIP 모델을 로드하여 이미지로부터 정규화된 numpy 벡터 생성 |
| **`ann_index.py`** | 인덱스 구축, 임베딩 저장(`.npy`), 매니페스트 관리 및 검색 인터페이스 제공 |
| **`policy.py`** | 검색 결과(ANN + pHash)를 분석하여 최종 의사결정을 내리는 정책 엔진 |

##  기술 스택

* **Model**: OpenCLIP (ViT-B-32 / openai)
* **Indexing**: HNSW (hnswlib)
* **Storage**: Numpy (`.npy`), JSON (Manifest), HNSW Index
* **Language**: Python 3.8+ (PyTorch 기반)

##  설정 및 판정 기준 (`config.py`)

검색 정밀도와 판정 기준은 `config.py` 파일 내의 임계값을 통해 조절할 수 있습니다.

* **HNSW 파라미터**: `HNSW_M`, `HNSW_EF_CONSTRUCTION` 등을 통해 빌드 속도와 검색 정확도 사이의 균형을 조절합니다.
* **판정 임계값**:
    * `COS_BLOCK`: 차단 기준 코사인 유사도 (기본값: 0.97)
    * `COS_ALLOW_A/B`: 허용 기준 코사인 유사도 (기본값: 0.90 / 0.85)

##  판정 로직 (3-State Policy)

`PolicyEngine`은 검색된 후보군 중 최상위 매칭 결과를 분석하여 다음과 같이 판정합니다.

1.  **BLOCK**: 유사도가 매우 높고 픽셀 수준(pHash)에서도 거의 동일한 경우.
2.  **ALLOW**: 유사도가 낮거나 의미적/픽셀적 차이가 확실하여 안전하다고 판단되는 경우.
3.  **REVIEW**: 위 두 조건에 해당하지 않는 모호한 구간으로, 수동 검토가 필요한 상태.
