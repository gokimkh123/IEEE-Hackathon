#!/bin/bash
set -e

# --- 1. 모델 학습 (Training) ---
echo "===== 1. 모델 학습을 시작합니다... ====="
docker build -t ieee-case-task2-dcgan . && \
docker run --rm --gpus all --shm-size="2g" \
  -v $(pwd)/submission:/app/submission \
  -v $(pwd)/checkpoints:/app/checkpoints \
  ieee-case-task2-dcgan

echo "===== 학습 완료. ====="
echo ""


# --- 2. 최적 모델 평가 (Evaluation) ---
echo "===== 2. 최적 모델 탐색 및 최종 평가를 시작합니다... ====="
docker build -t ieee-case-eval -f Dockerfile.evaluate . && \
docker run --rm --gpus all \
  -v $(pwd)/datasets:/app/datasets:ro \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -v $(pwd)/submission:/app/submission \
  ieee-case-eval

echo "===== 모든 과정이 완료되었습니다. ====="