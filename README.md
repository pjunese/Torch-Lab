#  [모각소] 소프트웨어융합 학과 자율 스터디 기록



##  소개
**모각소**는 소프트웨어융합 소속 학생들이 모여 각자의 IT 역량을 강화하고 학습 내용을 기록하는 자율 스터디 모임입니다.

단순한 코드 작성을 넘어, 매주 학습한 딥러닝/AI 모델의 이론적 배경을 정리하고 실제 데이터를 통해 성능을 검증하는 과정을 이 저장소에 담고 있습니다.

---

## 활동 기록 (Weekly Logs)

| 주차 | 주제 (Topic) | 주요 기술 (Tech Stack) | 상세 리드미 |
|:---:|:---|:---|:---:|
| **1주차** | **Deep Image Watermarking** | Encoder-Decoder, PSNR, COCO Dataset | [🔗 바로가기](./week1/) |
| **2주차** | **Semantic Similarity Detection** | OpenAI CLIP, ViT-B/32, Cosine Similarity | [🔗 바로가기](./week2/) |
| **3주차** | *(진행 예정)* | - | - |

---

## 📂 저장소 구조 (Directory Structure)

```text
root/
├── README.md                # 전체 프로젝트 개요 (현재 파일)
├── week1/                   # 1주차 활동: 이미지 워터마킹
│   ├── README.md            # 워터마킹 모델 이론 및 실험 결과
│   └── 인코더-디코더.ipynb    # 학습 및 추론 코드
└── week2/                   # 2주차 활동: 이미지 유사도 탐지
    ├── README.md            # CLIP 모델 이론 및 여행 사진 테스트 결과
    └── 유사도탐지모델.ipynb    # CLIP 임베딩 및 유사도 측정 코드
