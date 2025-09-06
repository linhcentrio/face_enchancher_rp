FROM spxiong/pytorch:2.7.1-py3.10.15-cuda12.8.1-ubuntu22.04 AS base

WORKDIR /app

# Set CUDA environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_MODULE_LOADING=LAZY

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10-dev \
    python3.10-distutils \
    build-essential \
    libgl1 \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Upgrade pip và install tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Install InsightFace dependencies first
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    scikit-image \
    Pillow \
    matplotlib \
    scipy \
    easydict \
    cython \
    protobuf \
    onnx

# Install Face Enhancement dependencies với phiên bản giống gốc
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
    opencv-python==4.9.0.80 \
    onnxruntime-gpu==1.21.0 \
    runpod>=1.6.0 \
    minio>=7.0.0 \
    requests>=2.31.0 \
    tqdm>=4.65.0

# Upgrade numpy cuối cùng nếu cần version cụ thể
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade numpy==1.24.3

# Copy source code
COPY . /app/

# Create model directories
RUN mkdir -p /app/models/face_detection && \
    mkdir -p /app/models/face_enhancement

# Download GFPGAN model
RUN echo "=== Downloading GFPGAN model ===" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/GFPGANv1.4.onnx" \
    -O /app/models/face_enhancement/GFPGANv1.4.onnx && \
    echo "✅ GFPGAN model downloaded"

# Download face detection model
RUN echo "=== Downloading face detection model ===" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/OwlMaster/AllFilesRope/resolve/main/scrfd_2.5g_bnkps.onnx" \
    -O /app/models/face_detection/scrfd_2.5g_bnkps.onnx && \
    echo "✅ Face detection model downloaded"

# Download face recognition model
RUN echo "=== Downloading face recognition model ===" && \
    wget --no-check-certificate --timeout=60 --tries=3 \
    "https://huggingface.co/manh-linh/faceID_recognition/resolve/main/recognition.onnx" \
    -O /app/models/face_detection/recognition.onnx && \
    echo "✅ Face recognition model downloaded"

# Verify model files exist
RUN echo "=== Verifying model files ===" && \
    test -f /app/models/face_enhancement/GFPGANv1.4.onnx && echo "✅ GFPGAN model verified" && \
    test -f /app/models/face_detection/scrfd_2.5g_bnkps.onnx && echo "✅ Face detection model verified" && \
    test -f /app/models/face_detection/recognition.onnx && echo "✅ Face recognition model verified"

# Set environment variables
ENV PYTHONPATH="/app"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

CMD ["python", "rp_handler.py"]

