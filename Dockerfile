FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

RUN apt-get update -y \
    && apt-get install -y python3-pip curl git ca-certificates build-essential python3-dev ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh  | sh

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:/root/.local/bin:$PATH"
ENV UV_TORCH_BACKEND=cu129

RUN ldconfig /usr/local/cuda-12.9/compat/

# Install the PyTorch versions expected by the custom vLLM branch.
RUN uv pip install --system torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --torch-backend=cu129

ARG VLLM_REPO=https://github.com/seokhyunan/vllm.git
ARG VLLM_BRANCH=v0.20.1-harmony-continuation
ARG VLLM_COMMIT=5158c7f

RUN git clone --filter=blob:none --no-checkout --branch "${VLLM_BRANCH}" --single-branch --depth 1 \
    "${VLLM_REPO}" /vllm-workspace \
    && cd /vllm-workspace \
    && git checkout --detach "${VLLM_COMMIT}"

WORKDIR /vllm-workspace
RUN VLLM_USE_PRECOMPILED=1 uv pip install --system --editable . --torch-backend=cu129
RUN VLLM_DOCKER_BUILD_CONTEXT=1 bash tools/install_deepgemm.sh --cuda-version 12.9
WORKDIR /

# Install additional Python dependencies (after vLLM to avoid PyTorch version conflicts)
COPY builder/requirements.txt /requirements.txt
RUN uv pip install --system -r /requirements.txt --torch-backend=cu129

# Suppress Ray metrics agent warnings and keep tokenizers thread usage bounded.
ENV HF_DATASETS_CACHE="/runpod-volume/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub" \
    HF_HOME="/runpod-volume/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4

ENV PYTHONPATH="/:/vllm-workspace"

COPY src /src
RUN chmod +x /src/start.sh

# Start the handler
CMD ["/bin/bash", "/src/start.sh"]
