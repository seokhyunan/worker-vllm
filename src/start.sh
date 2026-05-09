#!/bin/bash
set -e

if [ -n "${TRANSFORMERS_VERSION}" ]; then
    echo "Installing transformers==${TRANSFORMERS_VERSION}"
    uv pip install --system "transformers==${TRANSFORMERS_VERSION}"
fi

exec python3 /src/handler.py
