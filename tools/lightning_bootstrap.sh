#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.10
uv sync

echo
echo "uv sync complete."
echo "For a Blackwell Lightning node, install the matching torch stack, native packages, and run smoke tests:"
echo "  uv run ed3d-bootstrap lightning-all"
echo
echo "Download UltraShape weights if needed:"
echo "  uv run --no-sync ed3d-bootstrap ultrashape-weights"
echo
echo "Start remote UI:"
echo "  uv run --no-sync dev"
