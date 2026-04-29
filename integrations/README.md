# TRELLIS.2 Local Integrations

This folder contains optional external-module experiments used by the local launcher and CLI.

## FaithC

Source: `integrations/FaithC`

FaithC/Faithful Contouring is wired as a post-contour/export comparison module. It operates on an existing mesh artifact and can run after:

- Stage 2 shape mesh
- Stage 3 decoded mesh
- Final native TRELLIS GLB

It does not replace Stage 1, Stage 2, or Stage 3 generation. It cannot invent geometry that TRELLIS did not generate, but it can help diagnose whether the native export/remesh path is softening or damaging an already-good mesh.

CLI entry point:

```powershell
.\.venv\Scripts\python.exe tools\run_faithc.py --mesh tmp\input.glb --out tmp\faithc.glb --resolution 256
```

TRELLIS runner flags:

```powershell
--faithc-mode off|after-stage2|after-stage3|after-final
--faithc-resolution 256
--faithc-tri-mode auto
```

Notes:

- `faithcontour`, `atom3d`, `torch_scatter`, `einops`, and support deps were installed into `.venv`.
- Atom3d's installed CUDA source was patched in `.venv` for the Windows `long`/`int64_t` ABI mismatch, matching the kind of fix needed for o-voxel.
- Smoke test passed on the RTX 5000 at resolution `64` with `integrations\FaithC\assets\examples\cloth.glb`.
- Start with resolution `128` or `256`; higher values can use a lot of VRAM.

## UltraShape

Source: `integrations/UltraShape-1.0`

UltraShape is wired as a learned refinement module. It needs:

- Reference image
- Coarse mesh artifact from TRELLIS
- UltraShape checkpoint
- UltraShape config

It can run after:

- Stage 2 shape mesh
- Stage 3 decoded mesh
- Final native TRELLIS GLB

It is not installed wholesale into `.venv` because its published requirements include heavy/conflicting packages such as `flash_attn`, `deepspeed`, an older `transformers`, and its own checkpoint stack. The launcher exposes checkpoint/config fields and calls the upstream inference script when enabled.

TRELLIS runner flags:

```powershell
--ultrashape-mode off|after-stage2|after-stage3|after-final
--ultrashape-ckpt integrations\UltraShape-1.0\checkpoints\ultrashape_v1.pt
--ultrashape-config integrations\UltraShape-1.0\configs\infer_dit_refine.yaml
--ultrashape-steps 12
--ultrashape-num-latents 8192
--ultrashape-chunk-size 2048
--ultrashape-octree-res 512
--ultrashape-low-vram
```

Notes:

- Download the UltraShape checkpoint before enabling this module.
- Keep `--ultrashape-low-vram`, `8192` latents, chunk size `2048`, and octree `512` for first tests on the RTX 5000.
- UltraShape output is a separate observable GLB artifact, not fed back into TRELLIS texture generation yet.
- The non-conflicting Python dependencies were installed, and the repo was patched to fall back from `flash_attn` to PyTorch SDPA.
- `cubvh` did not build on this Windows environment yet. The first failure was CUDA 13.0 vs PyTorch CUDA 12.4; forcing CUDA 12.4 moved the build forward but then failed inside the MSVC/CUDA host include path. UltraShape remains wired but not runnable until `cubvh` is installed or its loader path is replaced.
