# --------------------------- train
docker build -t ieee-case-task2-dcgan . && docker run --rm --gpus all --shm-size="2g" -v $(pwd)/submission:/app/submission -v $(pwd)/checkpoints:/app/checkpoints ieee-case-task2-dcgan

# ----------------- eval ----------------------------
docker build -t ieee-case-eval -f Dockerfile.evaluate . && docker run --rm --gpus all -v $(pwd)/datasets:/app/datasets:ro -v $(pwd)/checkpoints:/app/checkpoints:ro -v $(pwd)/submission:/app/submission ieee-case-eval
