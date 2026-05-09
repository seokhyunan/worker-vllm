#!/bin/bash
set -Eeuo pipefail

on_error() {
    local exit_code=$?
    echo "[worker-vllm] startup script failed at line ${BASH_LINENO[0]} with exit code ${exit_code}" >&2
    exit "${exit_code}"
}
trap on_error ERR

echo "[worker-vllm] starting container entrypoint" >&2
python3 --version >&2

if [ -n "${TRANSFORMERS_VERSION}" ]; then
    echo "[worker-vllm] installing transformers==${TRANSFORMERS_VERSION}" >&2
    uv pip install --system "transformers==${TRANSFORMERS_VERSION}"
fi

echo "[worker-vllm] launching handler" >&2
exec python3 -u /src/handler.py
