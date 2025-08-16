# Dockerfile

# NVIDIA CUDA 12.1.1과 cuDNN 8을 포함한 Ubuntu 22.04 이미지를 기반으로 설정
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 시스템 패키지 업데이트 및 Python 설치
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉터리 설정
WORKDIR /app

# requirements.txt 복사
COPY requirements.txt .

# PyTorch를 CUDA 12.1 버전에 맞춰 설치 (A6000 GPU 지원)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 나머지 Python 패키지 설치
RUN pip3 install -r requirements.txt

# 프로젝트 소스 코드 및 데이터셋 복사
COPY src/ ./src/
COPY datasets/ ./datasets/

# 스크립트 실행
CMD ["python3", "src/main.py"]