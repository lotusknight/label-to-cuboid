FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    PATH="/opt/conda/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge to guarantee Python 3.12 on Ubuntu 22.04.
RUN curl -L -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm -f /tmp/miniforge.sh

RUN conda create -n app python=3.12 -y && conda clean -afy
SHELL ["conda", "run", "--no-capture-output", "-n", "app", "/bin/bash", "-c"]

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install "sam3 @ https://codeload.github.com/facebookresearch/sam3/zip/refs/heads/main" && \
    pip install -r requirements.txt

# Optional cache warmup for faster first request.
RUN python -c "from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast; \
DepthProForDepthEstimation.from_pretrained('apple/DepthPro-hf'); \
DepthProImageProcessorFast.from_pretrained('apple/DepthPro-hf')"

COPY app /app/app
COPY README.md /app/README.md
COPY PLAN_SAM3_DEPTH_CUBOID.md /app/PLAN_SAM3_DEPTH_CUBOID.md

EXPOSE 8000

CMD ["conda", "run", "--no-capture-output", "-n", "app", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
