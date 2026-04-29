# Remote Lightning Notes

This repo now has a lightweight remote UI:

```bash
uv run ed3d-remote-lab --host 0.0.0.0 --port 7860
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
bash tools/lightning_bootstrap.sh
source .venv/bin/activate
./setup.sh --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    "infinith/UltraShape",
    "ultrashape_v1.pt",
    local_dir="integrations/UltraShape-1.0/checkpoints",
)
PY
uv run ed3d-remote-lab --host 0.0.0.0 --port 7860
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
