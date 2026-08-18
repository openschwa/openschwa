#!/usr/bin/env bash
# One-shot setup for the Linux training laptop (RTX 4060).
# Safe to re-run; touches nothing outside this project + the uv toolchain.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo '== installing uv'
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo '== syncing dependencies (torch + CUDA comes from the PyPI wheels)'
uv sync --extra ml

echo '== verifying CUDA'
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available - check the NVIDIA driver'; print('CUDA OK:', torch.cuda.get_device_name(0))"

echo '== quick smoke test (CPU-safe, 2 training steps)'
uv run python -m pytest -q
echo 'setup complete'
