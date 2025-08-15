# Task 2: CNN+GNN cVAE generator runtime
# Base: official PyTorch image with CUDA 12.1 runtime
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# (Optional) copy the training/generation script into image
# You can also mount your working directory at runtime to override this.
COPY Task2_All_In_One_CNNGNN_Generative.py /app/Task2_All_In_One_CNNGNN_Generative.py

# Default command (overridden by run.sh)
CMD ["bash"]