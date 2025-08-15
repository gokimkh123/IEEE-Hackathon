#!/usr/bin/env bash
set -euo pipefail

# =====================
# Config
# =====================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="case-task2:gpu"
APP_MNT="/app"                # inside container
DATASETS_HOST="${SCRIPT_DIR}/datasets"  # expected: contains labeled_training_set.pkl, unlabeled_training_set.pkl
OUTPUTS_HOST="${SCRIPT_DIR}/outputs_task2"

# Auto GPU detection (set RUN_CPU=1 to force CPU)
RUN_CPU="${RUN_CPU:-0}"
if [[ "$RUN_CPU" == "1" ]]; then
  GPUS_ARG=""
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS_ARG="--gpus all"
  else
    GPUS_ARG=""  # fallback to CPU if no GPU runtime
  fi
fi

# Whether to run eval at the end of 'all'
RUN_EVAL="${RUN_EVAL:-0}"

# ensure host dirs
mkdir -p "$OUTPUTS_HOST"

# =====================
# Functions
# =====================
usage() {
  cat << EOF
Usage: $0 [build|train|generate|eval|all|shell]
  (no args)  same as 'all' → build -> train -> generate [-> eval if RUN_EVAL=1]
  build      Build Docker image
  train      Train Task2 generator (CNN+GNN cVAE)
  generate   Produce NIST_Task2_a/b/c.pkl (100 images each by default)
  eval       Quick local FID/LPIPS check (dev-only, needs torchvision)
  all        build -> train -> generate (and eval if RUN_EVAL=1)
  shell      Open container shell
Env:
  RUN_CPU=1     Force CPU container (no --gpus)
  RUN_EVAL=1    Make 'all' run eval step at the end
EOF
}

check_data() {
  if [[ ! -d "$DATASETS_HOST" ]]; then
    echo "[ERR] datasets folder not found: $DATASETS_HOST"; exit 1
  fi
  if [[ ! -f "$DATASETS_HOST/labeled_training_set.pkl" ]]; then
    echo "[WARN] labeled_training_set.pkl not found under datasets/ (continuing if you only have unlabeled)"
  fi
  if [[ ! -f "$DATASETS_HOST/unlabeled_training_set.pkl" ]]; then
    echo "[WARN] unlabeled_training_set.pkl not found under datasets/"
  fi
}

build() {
  docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"
  echo "== GPU check =="
  if ! docker run --rm $GPUS_ARG "$IMAGE" bash -lc 'nvidia-smi || true' >/dev/null 2>&1; then
    echo "[INFO] GPU not available in container. Falling back to CPU runs."
  fi
}

train() {
  check_data
  docker run --rm $GPUS_ARG \
    -v "$SCRIPT_DIR":${APP_MNT} \
    -v "$DATASETS_HOST":${APP_MNT}/datasets \
    -v "$OUTPUTS_HOST":${APP_MNT}/outputs \
    -w ${APP_MNT} \
    "$IMAGE" \
    python3 Task2_All_In_One_CNNGNN_Generative.py \
      --mode train \
      --data_dir datasets \
      --save_dir outputs \
      --epochs 200 \
      --batch_size 32 \
      --feat 128 --gnn_layers 2 --z_dim 64 \
      --beta 1e-3 --kl_warmup 20
}

generate() {
  docker run --rm $GPUS_ARG \
    -v "$SCRIPT_DIR":${APP_MNT} \
    -v "$DATASETS_HOST":${APP_MNT}/datasets \
    -v "$OUTPUTS_HOST":${APP_MNT}/outputs \
    -w ${APP_MNT} \
    "$IMAGE" \
    python3 Task2_All_In_One_CNNGNN_Generative.py \
      --mode generate \
      --save_dir outputs \
      --out_dir outputs \
      --num 100
  echo "== Generated files =="
  ls -lh "$OUTPUTS_HOST"/NIST_Task2_*.pkl || true
}

qe() {
  # quick eval
  docker run --rm $GPUS_ARG \
    -v "$SCRIPT_DIR":${APP_MNT} \
    -v "$DATASETS_HOST":${APP_MNT}/datasets \
    -v "$OUTPUTS_HOST":${APP_MNT}/outputs \
    -w ${APP_MNT} \
    "$IMAGE" \
    python3 Task2_All_In_One_CNNGNN_Generative.py \
      --mode eval \
      --data_dir datasets \
      --save_dir outputs \
      --num_pairs 200 --real_max 300
}

shell() {
  docker run --rm -it $GPUS_ARG \
    -v "$SCRIPT_DIR":${APP_MNT} \
    -v "$DATASETS_HOST":${APP_MNT}/datasets \
    -v "$OUTPUTS_HOST":${APP_MNT}/outputs \
    -w ${APP_MNT} \
    "$IMAGE" \
    bash
}

all() {
  build
  train
  generate
  if [[ "$RUN_EVAL" == "1" ]]; then
    qe
  fi
}

ACTION="${1:-all}"
case "$ACTION" in
  build) build ;;
  train) train ;;
  generate) generate ;;
  eval) qe ;;
  all) all ;;
  shell) shell ;;
  *) usage; exit 1 ;;
 esac