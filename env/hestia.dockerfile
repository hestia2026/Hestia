FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

# Install system dependencies (including build tools for Triton)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    ninja-build \
    libibverbs1 \
    ibverbs-providers \
    rdma-core \
    libnuma1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install "lm_eval[hf]==0.4.9.2"

# Set environment variables
ENV HF_ENDPOINT=https://hf-mirror.com

CMD ["bash"]
