# 인코더-디코더 워터마킹 노트북

이 노트북은 이미지 워터마크 인코더와 ViT 기반 검출기를 학습합니다. 32비트 메시지를 이미지에 삽입하고 간단한 공격 전/후의 복원 정확도를 평가합니다.

## 구성
- Colab 설정: Drive 마운트, DRIVE_ROOT/OUTPUT_DIR 지정
- COCO 2017 다운로드 및 압축 해제
- /content/wm_global32에 models.py, train.py, inference_test.py 생성
- 학습 루프: BCE 비트 손실 + lambda_img * MSE, PSNR 기록, alpha로 워터마크 강도 조절
- 추론 테스트: 날짜 메시지 삽입, resize/crop/JPEG 공격, 결과 이미지 및 diff map 저장

## 요구 사항
- Google Colab (GPU)
- Python 패키지: torch, torchvision, pillow, tqdm
- COCO 2017 데이터셋 (train2017, val2017, annotations)

## 빠른 시작 (Colab)
1) Drive 마운트 및 경로 설정.

```python
from google.colab import drive
drive.mount("/content/drive")
DRIVE_ROOT = "/content/drive/MyDrive/wm_global32"
OUTPUT_DIR = f"{DRIVE_ROOT}/output"
```

2) 패키지 설치.

```bash
!pip -q install pillow tqdm torchvision
```

3) COCO 다운로드 및 압축 해제.

노트북의 데이터셋 셀을 실행해 COCO 2017을 내려받고 압축을 풉니다.

4) 1단계 학습.

```bash
!python /content/wm_global32/train.py \
  --train_dir /content/coco/train2017 \
  --val_dir /content/coco/val2017 \
  --output_dir "$OUTPUT_DIR" \
  --img_size 224 \
  --nbits 32 \
  --batch_size 64 \
  --epochs 10 \
  --lambda_img 0.3 \
  --lr 1e-4 \
  --amp
```

5) (선택) 2단계 학습: 워터마크 강도 범위 확대.

```bash
!python train.py \
  --train_dir /content/coco/train2017 \
  --val_dir /content/coco/val2017 \
  --output_dir "$OUTPUT_DIR" \
  --img_size 224 \
  --nbits 32 \
  --batch_size 64 \
  --epochs 30 \
  --lambda_img 0.5 \
  --lr 1e-4 \
  --alpha_min 0.08 \
  --alpha_max 0.18 \
  --amp
```

6) 단일 이미지 추론.

```bash
!python inference_test.py \
  --img_path /content/drive/MyDrive/WMimg/wm_004.png \
  --message 20260101 \
  --alpha 0.5 \
  --out_dir /content/wm_global32/test_wm004
```

## 출력
- OUTPUT_DIR 체크포인트: last.pt, baseline_v1.pt
- out_dir 결과 이미지:
  - step0_original.png
  - step1_watermarked_off.png
  - step2_attacked.png
  - step3_diff.png

## 참고
- 디텍터는 ViT-B/16 백본을 고정하고 32비트 헤드로 출력합니다.
- 메시지는 날짜 문자열(예: 20260101)을 32비트로 변환합니다.
- Attack ON은 resize, center crop, JPEG 압축으로 강건성을 테스트합니다.
