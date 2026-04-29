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
echo "Download UltraShape weights if not committed/copied:"
echo "  python - <<'PY'"
echo "from huggingface_hub import hf_hub_download"
echo "hf_hub_download('infinith/UltraShape', 'ultrashape_v1.pt', local_dir='integrations/UltraShape-1.0/checkpoints')"
echo "PY"
echo
echo "Start remote UI:"
echo "  uv run --no-sync ed3d-remote-lab --host 0.0.0.0 --port 7860"
