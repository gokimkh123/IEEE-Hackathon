#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASETS="$SCRIPT_DIR/datasets"
OUTPUTS="$SCRIPT_DIR/outputs"

IMAGE="ieee-case:gpu"

if [ ! -d "$DATASETS" ]; then
  echo "[ERR] datasets folder not found: $DATASETS"; exit 1
fi
mkdir -p "$OUTPUTS"

echo "== Building Docker image =="
docker build -t "$IMAGE" "$SCRIPT_DIR"

echo "== Diagnosing GPU =="
docker run --rm --gpus all "$IMAGE" nvidia-smi || {
  echo "[WARN] nvidia-smi failed. Check drivers/Container Toolkit."; }

echo "== Starting training =="
docker run --rm --gpus all \
  -v "$DATASETS":/app/datasets \
  -v "$OUTPUTS":/app/outputs \
  "$IMAGE" \
  python3 train.py \
    --data_dir datasets \
    --save_dir outputs \
    --epochs 150 \
    --batch_size 16 \
    --lr 1e-3

echo "== Generating submission file (infer_submit.py) =="
docker run --rm --gpus all \
  -v "$DATASETS":/app/datasets \
  -v "$OUTPUTS":/app/outputs \
  "$IMAGE" \
  python3 infer_submit.py \
    --data_dir datasets \
    --save_dir outputs \
    --thr 0.50 \
    --out /app/outputs/NIST_Task1.pkl

echo "== Validating submission file format (check_submission.py) =="
docker run --rm \
  -v "$OUTPUTS":/app/outputs \
  "$IMAGE" \
  python3 check_submission.py \
    --pkl /app/outputs/NIST_Task1.pkl

if [ -f "$DATASETS/labeled_training_set.pkl" ]; then
  echo "== (Optional) Evaluating on labeled set (eval_val.py) =="
  docker run --rm --gpus all \
    -v "$DATASETS":/app/datasets \
    -v "$OUTPUTS":/app/outputs \
    "$IMAGE" \
    python3 eval_val.py \
      --data_dir datasets \
      --save_dir outputs \
      --thr 0.50
fi

echo "== Done =="
echo "Submission file: $OUTPUTS/NIST_Task1.pkl"