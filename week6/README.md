# Meta Watermark Anything (WAM) 테스트 README
---
## 1. 실험 목적

본 노트북은 **Meta Watermark Anything (WAM)** 모델을 이용해  
이미지에 **비가시성 워터마크를 삽입하고**,  
부분 삽입 상황에서도 **시각적 품질 유지**와 **메시지 복원 성능**을 동시에 평가하는 것을 목표로 한다.

평가는 다음 두 축으로 진행된다.

- **시각적 품질**
  - PSNR
  - SSIM
- **워터마크 복원 성능**
  - Bit Accuracy (32-bit 기준)
  - Hamming Distance

---

## 2. 실행 환경 및 구성

- 실행 환경: Google Colab (GPU 사용 가능)
- 데이터 저장소: Google Drive
- 사용 모델: Meta `watermark-anything`
- 체크포인트: `wam_mit.pth` (MIT weights)

### 주요 경로
- 원본 데이터셋:  
  `/content/drive/MyDrive/image/dataset60`
- 워터마크 결과 저장:  
  `/content/drive/MyDrive/image/wm_img`

---

## 3. 데이터 준비

- `dataset60` 폴더 내 이미지 중 **5장을 랜덤 샘플링**
- 지원 포맷: `.jpg`, `.png`, `.jpeg`, `.webp`, `.bmp`
- 모든 실험은 이 5장을 기준으로 반복 수행됨

---

## 4. 모델 로딩 및 핵심 유틸

### 모델
- `load_model_from_checkpoint`로 WAM 로드
- `eval()` 모드에서 추론 전용 사용

### 주요 유틸
- `default_transform`: 이미지 입력 전처리
- `unnormalize_img`: 지표 계산 및 시각화용 복원
- `create_random_mask`: 부분 삽입용 랜덤 마스크
- `msg_predict_inference`: 비트 메시지 디코딩

---

## 5. 워터마크 삽입 및 검출 파이프라인

각 이미지에 대해 아래 절차를 수행한다.

1. **32-bit 랜덤 메시지 생성**
2. **WAM embed**으로 워터마크 이미지 생성
3. **부분 삽입**
   - 랜덤 마스크(`proportion_masked = 0.6`) 영역에만 워터마크 적용
4. **WAM detect**
   - 워터마크 영역(pred mask) 및 비트 로짓 추출
5. **메시지 복원**
   - pred mask 기반으로 32-bit 메시지 추정

---

## 6. 평가 지표

### 6.1 시각적 품질
- PSNR
- SSIM  
→ 원본 이미지와 워터마크 이미지 간 차이를 `[0,1]` 스케일에서 계산

### 6.2 메시지 복원
- Bit Accuracy  
  - 32비트 중 맞춘 비율
- Hamming Distance  
  - 틀린 비트 개수 (0~32)

---

## 7. 시각화 전략

비가시성 워터마크 특성상 변화가 거의 보이지 않기 때문에,  
단순 비교 대신 아래 방식으로 시각화한다.

### 기본 비교
- Original
- Watermarked
- Absolute Difference
- Target Mask (실제 삽입 영역)

### 변화 강조(DIFF)
- Raw diff
- Diff × k (증폭 후 clip)
- Diff 정규화(상대 비교)

### 검출 결과
- Predicted Mask (모델이 워터마크가 있다고 판단한 영역)

---

## 8. 결과 저장

- 워터마크 이미지 저장
  - `{원본명}_wm.png`
- 저장 위치
  - `/content/drive/MyDrive/image/wm_img`

---

## 9. scaling_w 파라미터 스윕 실험

워터마크 **강도 파라미터 `scaling_w`** 변화에 따른  
품질·복원 성능의 트레이드오프를 분석한다.

### 설정
- `scaling_w ∈ [0.5, 1, 2, 4, 8, 10]`
- 공정 비교를 위해:
  - 이미지별 **고정 메시지**
  - 이미지별 **고정 마스크** 사용

### 수집 지표
- PSNR / SSIM
- Bit Accuracy / Hamming Distance

### 산출물
- scaling별 평균·표준편차 테이블
- 에러바 그래프 4종
- 대표 이미지 1장에 대한 scaling별 시각 비교 그리드

---

## 10. 실험 산출물 요약

- Drive 저장 이미지:
  - 부분 삽입 워터마크 결과
- 정량 결과:
  - 이미지 × scaling 전체 결과 테이블
  - scaling별 평균 성능 통계
- 시각 결과:
  - 워터마크 위치 및 변화 강조
  - 강도별 품질 변화 비교

---

## 11. 핵심 정리

- WAM은 **부분 삽입 환경**에서도 메시지 복원이 가능함
- `scaling_w`는
  - 낮을수록 시각 품질 우수
  - 높을수록 메시지 복원 안정
- 품질과 강건성 사이의 **명확한 트레이드오프 곡선**을 확인할 수 있음
- 본 노트북은 **서비스 적용 전 파라미터 탐색용 테스트베드**로 적합한 구조를 가짐

---
