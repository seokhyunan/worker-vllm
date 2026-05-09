FROM runpod/base:1.0.3-cuda1300-ubuntu2404

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends python3-pip curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh  | sh

ENV PATH="/root/.local/bin:$PATH"
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

RUN ldconfig /usr/local/cuda-13.0/compat/

# Install the PyTorch versions expected by the custom vLLM branch.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --torch-backend=cu130

RUN git clone --branch v0.20.1-harmony-continuation --depth 1 \
    https://github.com/seokhyunan/vllm.git /vllm-workspace

WORKDIR /vllm-workspace
RUN --mount=type=cache,target=/root/.cache/uv \
    VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=cu130
WORKDIR /

# Install additional Python dependencies (after vLLM to avoid PyTorch version conflicts)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /requirements.txt

# Suppress Ray metrics agent warnings and keep tokenizers thread usage bounded.
ENV RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4

ENV PYTHONPATH="/:/vllm-workspace"

COPY src /src
RUN chmod +x /src/start.sh

# Start the handler
CMD ["/bin/bash", "/src/start.sh"]
