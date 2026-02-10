# [모각소] 소프트웨어융합학과 자율 스터디 기록

본 저장소는 **소프트웨어융합학과 모각소(모여서 각자 소프트웨어 공부)** 활동의 일환으로,  
딥러닝·AI 관련 주제를 중심으로 한 **자율 스터디 학습 기록 및 실험 결과**를 정리한 공간입니다.

단순한 코드 구현에 그치지 않고,  
- 모델의 이론적 배경  
- 설계 의도와 한계  
- 실제 데이터 기반 실험 및 성능 검증  
을 함께 기록하는 것을 목표로 합니다.

---

## 1. 스터디 소개

**모각소**는 소프트웨어융합학과 소속 학생들이 자율적으로 모여  
각자의 관심 분야를 중심으로 학습하고, 그 과정을 기록·공유하는 스터디 프로그램입니다.

본 스터디에서는 특히 다음과 같은 방향성을 갖고 활동을 진행합니다.

- 딥러닝 / AI 모델의 구조와 원리 이해  
- 공개된 최신 모델(Open-source / Pretrained Model)의 실제 활용  
- 실험 결과에 대한 정량적 분석과 해석  
- 서비스 및 현실 환경에서의 적용 가능성 고려  

---

## 2. 활동 기록 (Weekly Logs)

| 주차 | 주제 (Topic) | 주요 기술 (Tech Stack) | 상세 기록 |
|:---:|:---|:---|:---:|
| **1주차** | **Deep Image Watermarking** | Encoder–Decoder, PSNR, COCO Dataset | [바로가기](./week1/) |
| **2주차** | **Semantic Similarity Detection** | OpenAI CLIP, ViT-B/32, Cosine Similarity | [바로가기](./week2/) |
| **3주차** | **Document / Text Watermarking** | WAM, PDF → Image, Bit Accuracy, Mask Detection | [바로가기](./week3/) |
| **4주차** | **ANN (Approximate Nearest Neighbor)** | FAISS, HNSW, IVF-PQ, Vector Search | [바로가기](./week4/) |
| **5주차** | **AI 이미지 진위 판정 시스템 설계** | OpenCLIP, hnswlib, 3-State Policy Engine | [바로가기](./week5/) |
| **6주차** | **WAM 기반 비가시성 워터마킹 실험 심화** | Meta WAM, Partial Masking, PSNR/SSIM, Bit Accuracy | [바로가기](./week6/) |

---

## 3. 저장소 구조 (Directory Structure)

```text
root/
├── README.md                 # 전체 스터디 개요 (본 파일)
├── week1/                    # 1주차: 이미지 워터마킹
│   ├── README.md
│   └── 인코더-디코더.ipynb
├── week2/                    # 2주차: 이미지 유사도 탐지
│   ├── README.md
│   └── 유사도탐지모델.ipynb
├── week3/                    # 3주차: 문서 / 텍스트 워터마킹
│   ├── README.md             # PDF 기반 워터마킹 실험 정리
│   └── TXT_wm.ipynb          # PDF → 이미지 → WAM 실험 코드
├── week4/                    # 4주차: 근사 최근접 이웃(ANN) 알고리즘 비교
│   ├── README.md             # HNSW vs IVF-PQ 개념 및 특징 정리
│   └── ANN.py                # FAISS 기반 ANN 알고리즘 구현 및 비교
├── week5/                    # 5주차: 서비스급 이미지 진위 판정 시스템
│   ├── README.md             # 시스템 설계 및 판정 정책(3-State) 정리
│   ├── ann_index.py          # HNSW 인덱싱 및 DB 변경 감지 로직
│   ├── embedder.py           # CLIP 기반 이미지 임베딩 생성
│   ├── policy.py             # Cosine + pHash 기반 정책 엔진
│   └── config.py             # 시스템 전역 설정 및 임계값 관리
└── week6/                    # 6주차: WAM 심화 실험 및 파라미터 분석
    ├── README.md             # WAM 전체 실험 요약 및 해석
    └── Meta_WM_test.ipynb    # Partial Masking / scaling_w 스윕 실험
