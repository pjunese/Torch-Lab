#  WAM 기반 PDF 페이지 워터마킹 실험

본 저장소는 Meta의 **Watermark Anything Model (WAM)** 오픈소스를 활용해, **PDF 문서를 페이지 이미지로 변환한 뒤 비가시성 워터마크를 삽입(embed)하고 검출(detect) 결과를 저장/시각화**하는 스터디 및 실험 기록이다.

본 실험의 목적은 “문서(PDF) → 페이지 이미지 → 워터마크 삽입 → 검출(마스크/메시지) → 결과 저장 및 오버레이 시각화” 파이프라인의 동작을 확인하는 것이다.

---

## 1) 실험 목표

- PDF 파일을 페이지 단위 이미지로 변환하고 일괄 처리 파이프라인 구성
- WAM 사전학습 가중치(`wam_mit.pth`) 기반으로 워터마크 메시지(32-bit) 삽입
- 워터마킹 적용 영역을 **랜덤 마스크(mask_percentage)** 로 제한하여 부분 삽입 시나리오 테스트
- Detect 결과로부터 아래 산출물을 저장 및 시각적으로 검증
  - 예측 마스크(pred mask)
  - 타깃 마스크(target mask)
  - 메시지 복원 및 bit accuracy

---

## 2) 실행 환경 

- 실행 환경: Google Colab
- 저장소/데이터 경로: Google Drive
- 주요 라이브러리
  - `watermark-anything` (facebookresearch)
  - `pymupdf (fitz)` : PDF → 이미지 변환
  - `torch`, `torchvision`, `PIL`, `matplotlib`
- 디바이스 선택
  - `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

---

## 3) 폴더 구조 및 입출력 경로

노트북에서 사용하는 기본 경로는 아래와 같다.

- 프로젝트 루트(Drive)
  - `WATSON_DIR = /content/drive/MyDrive/WATSON`
- 입력 PDF 폴더
  - `INPUT_PDF_DIR = /content/drive/MyDrive/WATSON/test_pdfs`
- 출력 루트
  - `OUTPUT_DIR = /content/drive/MyDrive/WATSON/wam_outputs`
- 페이지 이미지 폴더
  - `IMG_DIR = /content/drive/MyDrive/WATSON/wam_outputs/pages`
- 워터마킹 결과 폴더
  - `WM_DIR = /content/drive/MyDrive/WATSON/wam_outputs/watermarked`

---

## 4) 파이프라인 요약

###  WAM 레포지토리 준비

- 레포 클론/업데이트: `/content/WAM` 에 `watermark-anything` 클론
- requirements 설치 + `pymupdf` 설치
- 체크포인트 다운로드: `wam_mit.pth` 를 `/content/WAM/checkpoints/` 에 저장

###  PDF → 페이지 이미지 변환

- `pymupdf(fitz)` 사용
- 변환 설정: `DPI = 300`
- 결과 파일명: `{pdf_stem}_p{page_index:03d}.png`

노트북 출력 로그(실측):
- PDF 개수: **1**
- 변환된 이미지 수: **9**

###  워터마크 삽입(Embed) + 부분 적용

- 메시지: `wm_msg = wam.get_random_msg(1)` (32-bit)
- 부분 적용 비율: `mask_pct = 0.15` (약 15% 영역만 워터마크 적용)
- 합성 방식  
  `img_w = outputs["imgs_w"] * mask + img_pt * (1 - mask)`

###  워터마크 검출(Detect) 및 성능 지표 계산

- `preds = wam.detect(img_w)["preds"]`
- 구성
  - `preds[:, 0, :, :]` → 마스크 로짓 → `sigmoid` 로 `mask_preds`
  - `preds[:, 1:, :, :]` → 비트 로짓 → `bit_preds`
- 메시지 복원
  - `pred_message = msg_predict_inference(bit_preds, mask_preds)`
- bit accuracy
  - `bit_acc = (pred_message == wm_msg).float().mean().item()`

노트북 출력(실측):
- 9장 처리 완료
- 샘플 결과(앞 5장): 모두 `bit_acc = 1.0`

###  결과 저장 규칙

각 페이지 이미지에 대해 다음 파일들을 저장한다.

- 워터마크 적용 이미지: `{base}_wm.png`
- 예측 마스크: `{base}_predmask.png`
- 타깃(랜덤) 마스크: `{base}_targetmask.png`

저장 위치:
- `/content/drive/MyDrive/WATSON/wam_outputs/watermarked/`

---

## 5) 결과 시각화 (오버레이)

저장된 결과를 불러와 2×3 그리드로 시각화한다.

표시 항목:
- Original
- Watermarked
- Target Mask
- Pred Mask
- Overlay Target (초록 오버레이)
- Overlay Pred (빨강 오버레이)

제어 옵션:
- `MAX_SHOW = None` 이면 전부 출력, 숫자 지정 시 일부만 출력

---
## 6) 참고

- WAM 레포: facebookresearch/watermark-anything
- 체크포인트: `wam_mit.pth` (노트북에서 다운로드/사용)
