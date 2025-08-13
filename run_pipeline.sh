#!/usr/bin/env bash
set -euo pipefail

# ====== 스크립트 위치 기준 경로 ======
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASETS="$SCRIPT_DIR/datasets"   # datasets 폴더
OUTPUTS="$SCRIPT_DIR/outputs"     # outputs 폴더

IMAGE="ieee-case:gpu"

# ====== 사전 체크 ======
if [ ! -d "$DATASETS" ]; then
  echo "[ERR] datasets 폴더가 없습니다: $DATASETS"; exit 1
fi
mkdir -p "$OUTPUTS"

echo "== Docker 이미지 빌드 =="
docker build -t "$IMAGE" "$SCRIPT_DIR"

echo "== GPU 진단 =="
docker run --rm --gpus all "$IMAGE" nvidia-smi || {
  echo "[WARN] nvidia-smi 실패. 드라이버/Container Toolkit 확인 필요."; }

# ====== 학습 ======
echo "== 학습 시작 =="
docker run --rm --gpus all \
  -v "$DATASETS":/app/datasets \
  -v "$OUTPUTS":/app/outputs \
  "$IMAGE" \
  python3 train.py \
    --data_dir datasets \
    --save_dir outputs \
    --epochs 500 \
    --batch_size 32 \
    --lr 0.01

# ====== 제출 파일 생성 ======
echo "== 제출 파일 생성 (infer_submit.py) =="
docker run --rm --gpus all \
  -v "$DATASETS":/app/datasets \
  -v "$OUTPUTS":/app/outputs \
  "$IMAGE" \
  python3 infer_submit.py \
    --data_dir datasets \
    --save_dir outputs \
    --out /app/outputs/NIST_Task1.pkl

# ====== 제출 파일 검증 ======
echo "== 제출 파일 형식 검증 (check_submission.py) =="
docker run --rm \
  -v "$OUTPUTS":/app/outputs \
  "$IMAGE" \
  python3 check_submission.py \
    --pkl /app/outputs/NIST_Task1.pkl

# ====== 선택: 라벨셋 평가 ======
if [ -f "$DATASETS/labeled_training_set.pkl" ]; then
  echo "== (옵션) 라벨셋 평가 (eval_val.py) =="
  docker run --rm --gpus all \
    -v "$DATASETS":/app/datasets \
    -v "$OUTPUTS":/app/outputs \
    "$IMAGE" \
    python3 eval_val.py \
      --data_dir datasets \
      --save_dir outputs \
      --thr 0.620
fi

echo "== 완료 =="
echo "제출 파일: $OUTPUTS/NIST_Task1.pkl"
