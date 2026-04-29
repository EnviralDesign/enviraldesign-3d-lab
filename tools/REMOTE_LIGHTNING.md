# Remote Lightning Notes

This repo now has a lightweight remote UI:

```bash
uv run --no-sync ed3d-remote-lab --host 0.0.0.0 --port 7860
```

The UI wraps `local_image_to_3d.py`, streams logs, opens the latest GLB in
Gradio's model viewer, and exposes generated GLB artifacts for download.

## Suggested Transfer Strategy

The cleanest path is to push this work to a dedicated repo or branch, then clone
it from the Lightning node. Avoid manually copying a mutated working tree over
SSH unless this is a one-off throwaway machine.

Recommended:

1. Create a bespoke repo for this pipeline wrapper, or a dedicated branch if the
   current TRELLIS fork is already remote-friendly.
2. Push:
   - `local_image_to_3d.py`
   - `tools/remote_launcher.py`
   - `tools/lightning_bootstrap.sh`
   - `tools/glb_viewer.html`
   - `integrations/README.md`
   - the UltraShape/FaithC integration patches
3. Do not push large weights such as `ultrashape_v1.pt`.
4. On Lightning, clone the repo, run the bootstrap/setup, then download weights
   from Hugging Face directly on the node.

## Lightning Setup Sketch

```bash
git clone <repo-url> TRELLIS.2
cd TRELLIS.2
uv sync
uv run ed3d-bootstrap lightning-all
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    "infinith/UltraShape",
    "ultrashape_v1.pt",
    local_dir="integrations/UltraShape-1.0/checkpoints",
)
PY
uv run --no-sync ed3d-remote-lab --host 0.0.0.0 --port 7860
```

Why `--no-sync` on launch: the lockfile targets the local RTX 5000/RTX 3070
CUDA 12.4 stack. `ed3d-bootstrap lightning-all` deliberately replaces that
torch stack with CUDA 13.0 wheels for Blackwell Lightning nodes. Plain
`uv run` may try to restore the locked CUDA 12.4 packages, so use `--no-sync`
after the Lightning bootstrap has completed.

The bootstrap command performs the steps that were validated on a Lightning
RTX PRO 6000 Blackwell Server Edition node:

```bash
uv run ed3d-bootstrap lightning-all
```

It does the following:

- installs `torch==2.11.0+cu130`, `torchvision==0.26.0+cu130`, and
  `torchaudio==2.11.0+cu130`
- installs `xformers==0.0.35`
- builds `nvdiffrast`, `nvdiffrec_render`, `cumesh`, `o_voxel`, and `flex_gemm`
  with `TORCH_CUDA_ARCH_LIST=12.0`
- runs torch CUDA and import smoke tests

Individual steps are available if you need to resume after an interruptible
instance preemption:

```bash
uv run ed3d-bootstrap blackwell-torch
uv run --no-sync ed3d-bootstrap native
uv run --no-sync ed3d-bootstrap smoke
```

If Lightning exposes a proxy URL for port `7860`, use that. Otherwise SSH
forward it:

```bash
ssh -L 7860:127.0.0.1:7860 <lightning-host>
```

Then open:

```text
http://127.0.0.1:7860
```

## Artifact Flow

Remote outputs are written under:

```text
tmp/remote-runs/
```

Each run has its own timestamped folder. The Gradio UI exposes GLBs through the
Artifacts file component, so results can be downloaded directly through the
browser instead of pulling paths manually over SSH.

## Repo Choice

A new bespoke repo is cleaner if this is becoming a real pipeline:

- It avoids mixing experimental launcher/integration work with upstream TRELLIS.
- It gives Lightning a simple `git clone && bash tools/lightning_bootstrap.sh`
  path.
- It lets us add wrappers, docs, presets, and future texturing passes without
  pretending they are upstream TRELLIS changes.

Keep model caches and checkpoints out of git.
