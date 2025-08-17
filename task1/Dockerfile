FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git build-essential libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY train.py infer_submit.py eval_val.py check_submission.py showData.py ./

SHELL ["/bin/bash","-lc"]