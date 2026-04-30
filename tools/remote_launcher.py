import argparse
import importlib
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "tmp" / "remote-runs"

RUN_LOG_AUTOSCROLL_JS = r"""
() => {
  const scrollRunLog = () => {
    ["run-log", "roi-log"].forEach((id) => {
      const root = document.getElementById(id);
      const textarea = root?.querySelector("textarea");
      if (textarea) {
        textarea.scrollTop = textarea.scrollHeight;
      }
    });
  };

  const attachRunLogObserver = () => {
    ["run-log", "roi-log"].forEach((id) => {
      const root = document.getElementById(id);
      if (!root || root.dataset.autoscrollAttached === "true") {
        return;
      }
      root.dataset.autoscrollAttached = "true";
      const observer = new MutationObserver(() => requestAnimationFrame(scrollRunLog));
      observer.observe(root, { childList: true, subtree: true, characterData: true });
      root.addEventListener("input", scrollRunLog);
    });
    scrollRunLog();
  };

  const configureModelViewers = () => {
    document.querySelectorAll("model-viewer").forEach((viewer) => {
      if (viewer.dataset.ed3dLighting === "true") {
        return;
      }
      viewer.dataset.ed3dLighting = "true";
      viewer.setAttribute("environment-image", "neutral");
      viewer.setAttribute("shadow-intensity", "0.95");
      viewer.setAttribute("shadow-softness", "0.8");
      viewer.setAttribute("exposure", "0.65");
      viewer.setAttribute("interaction-prompt", "none");
    });
  };

  const bootObserver = new MutationObserver(attachRunLogObserver);
  bootObserver.observe(document.body, { childList: true, subtree: true });
  const viewerObserver = new MutationObserver(configureModelViewers);
  viewerObserver.observe(document.body, { childList: true, subtree: true });
  attachRunLogObserver();
  configureModelViewers();
  setInterval(scrollRunLog, 1000);
  setInterval(configureModelViewers, 1000);
}
"""


def choose_dense_attention_backend() -> str:
    requested = os.environ.get("ATTN_BACKEND")
    if requested:
        return requested

    for backend, module in (
        ("flash_attn_3", "flash_attn_interface"),
        ("flash_attn", "flash_attn"),
        ("xformers", "xformers.ops"),
    ):
        try:
            importlib.import_module(module)
            return backend
        except Exception:
            continue

    return "sdpa"


PRESETS = {
    "RTX 5000 standard 512": {
        "pipeline": "512",
        "seed": 0,
        "max_tokens": 12288,
        "sparse_res": 32,
        "stage1_max_voxels": 12000,
        "decimation": 100000,
        "texture_size": 1024,
        "ss_steps": 12,
        "shape_steps": 12,
        "tex_steps": 12,
        "ss_guidance": 7.5,
        "shape_guidance": 7.5,
        "tex_guidance": 1.0,
        "ss_rescale": 0.7,
        "shape_rescale": 0.5,
        "tex_rescale": 0.0,
        "ss_rescale_t": 5.0,
        "shape_rescale_t": 3.0,
        "tex_rescale_t": 3.0,
        "ss_interval_start": 0.6,
        "ss_interval_end": 1.0,
        "shape_interval_start": 0.6,
        "shape_interval_end": 1.0,
        "tex_interval_start": 0.6,
        "tex_interval_end": 0.9,
        "export_remesh": True,
        "fill_holes": True,
        "max_hole_perimeter": 0.03,
        "remesh_band": 1.0,
        "remesh_project": 0.9,
        "pre_simplify": 16777216,
        "save_stages": True,
        "stop_after": "none",
        "faithc_mode": "off",
        "faithc_resolution": 256,
        "faithc_tri_mode": "auto",
        "faithc_margin": 0.05,
        "faithc_normalize": False,
        "faithc_clamp": True,
        "faithc_flux": True,
        "ultra_mode": "off",
        "ultra_steps": 25,
        "ultra_latents": 8192,
        "ultra_chunk": 2048,
        "ultra_octree": 512,
        "ultra_scale": 0.99,
        "ultra_input_simplify": 1000000,
        "ultra_low_vram": True,
        "ultra_remove_bg": False,
    },
    "UltraShape 25 / 8192": {
        "pipeline": "512",
        "seed": 0,
        "max_tokens": 12288,
        "sparse_res": 32,
        "stage1_max_voxels": 12000,
        "decimation": 100000,
        "texture_size": 1024,
        "ss_steps": 12,
        "shape_steps": 12,
        "tex_steps": 12,
        "ss_guidance": 7.5,
        "shape_guidance": 7.5,
        "tex_guidance": 1.0,
        "ss_rescale": 0.7,
        "shape_rescale": 0.5,
        "tex_rescale": 0.0,
        "ss_rescale_t": 5.0,
        "shape_rescale_t": 3.0,
        "tex_rescale_t": 3.0,
        "ss_interval_start": 0.6,
        "ss_interval_end": 1.0,
        "shape_interval_start": 0.6,
        "shape_interval_end": 1.0,
        "tex_interval_start": 0.6,
        "tex_interval_end": 0.9,
        "export_remesh": True,
        "fill_holes": True,
        "max_hole_perimeter": 0.03,
        "remesh_band": 1.0,
        "remesh_project": 0.9,
        "pre_simplify": 16777216,
        "save_stages": True,
        "stop_after": "none",
        "faithc_mode": "off",
        "faithc_resolution": 256,
        "faithc_tri_mode": "auto",
        "faithc_margin": 0.05,
        "faithc_normalize": False,
        "faithc_clamp": True,
        "faithc_flux": True,
        "ultra_mode": "after-final",
        "ultra_steps": 25,
        "ultra_latents": 8192,
        "ultra_chunk": 2048,
        "ultra_octree": 512,
        "ultra_scale": 0.99,
        "ultra_input_simplify": 1000000,
        "ultra_low_vram": True,
        "ultra_remove_bg": False,
    },
    "UltraShape 25 / 16384": {
        "pipeline": "512",
        "seed": 0,
        "max_tokens": 12288,
        "sparse_res": 32,
        "stage1_max_voxels": 12000,
        "decimation": 100000,
        "texture_size": 1024,
        "ss_steps": 12,
        "shape_steps": 12,
        "tex_steps": 12,
        "ss_guidance": 7.5,
        "shape_guidance": 7.5,
        "tex_guidance": 1.0,
        "ss_rescale": 0.7,
        "shape_rescale": 0.5,
        "tex_rescale": 0.0,
        "ss_rescale_t": 5.0,
        "shape_rescale_t": 3.0,
        "tex_rescale_t": 3.0,
        "ss_interval_start": 0.6,
        "ss_interval_end": 1.0,
        "shape_interval_start": 0.6,
        "shape_interval_end": 1.0,
        "tex_interval_start": 0.6,
        "tex_interval_end": 0.9,
        "export_remesh": True,
        "fill_holes": True,
        "max_hole_perimeter": 0.03,
        "remesh_band": 1.0,
        "remesh_project": 0.9,
        "pre_simplify": 16777216,
        "save_stages": True,
        "stop_after": "none",
        "faithc_mode": "off",
        "faithc_resolution": 256,
        "faithc_tri_mode": "auto",
        "faithc_margin": 0.05,
        "faithc_normalize": False,
        "faithc_clamp": True,
        "faithc_flux": True,
        "ultra_mode": "after-final",
        "ultra_steps": 25,
        "ultra_latents": 16384,
        "ultra_chunk": 2048,
        "ultra_octree": 512,
        "ultra_scale": 0.99,
        "ultra_input_simplify": 1000000,
        "ultra_low_vram": True,
        "ultra_remove_bg": False,
    },
    "Comfy-style UltraShape before texture": {
        "pipeline": "1024_cascade",
        "seed": 0,
        "max_tokens": 12288,
        "sparse_res": 32,
        "stage1_max_voxels": 12000,
        "decimation": 200000,
        "texture_size": 1024,
        "ss_steps": 30,
        "shape_steps": 30,
        "tex_steps": 12,
        "ss_guidance": 6.6,
        "shape_guidance": 8.2,
        "tex_guidance": 1.0,
        "ss_rescale": 0.1,
        "shape_rescale": 0.1,
        "tex_rescale": 0.0,
        "ss_rescale_t": 1.0,
        "shape_rescale_t": 2.0,
        "tex_rescale_t": 3.0,
        "ss_interval_start": 0.1,
        "ss_interval_end": 1.0,
        "shape_interval_start": 0.1,
        "shape_interval_end": 1.0,
        "tex_interval_start": 0.6,
        "tex_interval_end": 0.9,
        "export_remesh": True,
        "fill_holes": True,
        "max_hole_perimeter": 0.03,
        "remesh_band": 1.0,
        "remesh_project": 0.9,
        "pre_simplify": 16777216,
        "save_stages": True,
        "stop_after": "none",
        "faithc_mode": "off",
        "faithc_resolution": 256,
        "faithc_tri_mode": "auto",
        "faithc_margin": 0.05,
        "faithc_normalize": False,
        "faithc_clamp": True,
        "faithc_flux": True,
        "ultra_mode": "before-texture",
        "ultra_steps": 10,
        "ultra_latents": 8192,
        "ultra_chunk": 2048,
        "ultra_octree": 256,
        "ultra_scale": 0.99,
        "ultra_input_simplify": 1000000,
        "ultra_low_vram": True,
        "ultra_remove_bg": False,
    },
    "Inspect Stage 1": {
        "pipeline": "512",
        "seed": 0,
        "max_tokens": 12288,
        "sparse_res": 32,
        "stage1_max_voxels": 24000,
        "decimation": 100000,
        "texture_size": 1024,
        "ss_steps": 12,
        "shape_steps": 12,
        "tex_steps": 12,
        "ss_guidance": 7.5,
        "shape_guidance": 7.5,
        "tex_guidance": 1.0,
        "ss_rescale": 0.7,
        "shape_rescale": 0.5,
        "tex_rescale": 0.0,
        "ss_rescale_t": 5.0,
        "shape_rescale_t": 3.0,
        "tex_rescale_t": 3.0,
        "ss_interval_start": 0.6,
        "ss_interval_end": 1.0,
        "shape_interval_start": 0.6,
        "shape_interval_end": 1.0,
        "tex_interval_start": 0.6,
        "tex_interval_end": 0.9,
        "export_remesh": True,
        "fill_holes": True,
        "max_hole_perimeter": 0.03,
        "remesh_band": 1.0,
        "remesh_project": 0.9,
        "pre_simplify": 16777216,
        "save_stages": True,
        "stop_after": "stage1",
        "faithc_mode": "off",
        "faithc_resolution": 256,
        "faithc_tri_mode": "auto",
        "faithc_margin": 0.05,
        "faithc_normalize": False,
        "faithc_clamp": True,
        "faithc_flux": True,
        "ultra_mode": "off",
        "ultra_steps": 25,
        "ultra_latents": 8192,
        "ultra_chunk": 2048,
        "ultra_octree": 512,
        "ultra_scale": 0.99,
        "ultra_input_simplify": 1000000,
        "ultra_low_vram": True,
        "ultra_remove_bg": False,
    },
    "Inspect Stage 2": {
        "pipeline": "512",
        "seed": 0,
        "max_tokens": 12288,
        "sparse_res": 32,
        "stage1_max_voxels": 12000,
        "decimation": 100000,
        "texture_size": 1024,
        "ss_steps": 12,
        "shape_steps": 25,
        "tex_steps": 12,
        "ss_guidance": 7.5,
        "shape_guidance": 7.5,
        "tex_guidance": 1.0,
        "ss_rescale": 0.7,
        "shape_rescale": 0.5,
        "tex_rescale": 0.0,
        "ss_rescale_t": 5.0,
        "shape_rescale_t": 3.0,
        "tex_rescale_t": 3.0,
        "ss_interval_start": 0.6,
        "ss_interval_end": 1.0,
        "shape_interval_start": 0.6,
        "shape_interval_end": 1.0,
        "tex_interval_start": 0.6,
        "tex_interval_end": 0.9,
        "export_remesh": True,
        "fill_holes": True,
        "max_hole_perimeter": 0.03,
        "remesh_band": 1.0,
        "remesh_project": 0.9,
        "pre_simplify": 16777216,
        "save_stages": True,
        "stop_after": "stage2",
        "faithc_mode": "off",
        "faithc_resolution": 256,
        "faithc_tri_mode": "auto",
        "faithc_margin": 0.05,
        "faithc_normalize": False,
        "faithc_clamp": True,
        "faithc_flux": True,
        "ultra_mode": "off",
        "ultra_steps": 25,
        "ultra_latents": 8192,
        "ultra_chunk": 2048,
        "ultra_octree": 512,
        "ultra_scale": 0.99,
        "ultra_input_simplify": 1000000,
        "ultra_low_vram": True,
        "ultra_remove_bg": False,
    },
}


def q(value: str) -> str:
    return shlex.quote(str(value))


def abs_path(path: str | Path) -> str:
    return str(Path(path).resolve())


PRESET_OUTPUT_KEYS = [
    "pipeline",
    "seed",
    "max_tokens",
    "sparse_res",
    "stage1_max_voxels",
    "ss_steps",
    "ss_guidance",
    "ss_rescale",
    "ss_rescale_t",
    "ss_interval_start",
    "ss_interval_end",
    "shape_steps",
    "shape_guidance",
    "shape_rescale",
    "shape_rescale_t",
    "shape_interval_start",
    "shape_interval_end",
    "tex_steps",
    "tex_guidance",
    "tex_rescale",
    "tex_rescale_t",
    "tex_interval_start",
    "tex_interval_end",
    "texture_size",
    "decimation",
    "export_remesh",
    "fill_holes",
    "max_hole_perimeter",
    "remesh_band",
    "remesh_project",
    "pre_simplify",
    "save_stages",
    "stop_after",
    "faithc_mode",
    "faithc_resolution",
    "faithc_tri_mode",
    "faithc_margin",
    "faithc_normalize",
    "faithc_clamp",
    "faithc_flux",
    "ultra_mode",
    "ultra_steps",
    "ultra_latents",
    "ultra_chunk",
    "ultra_octree",
    "ultra_scale",
    "ultra_input_simplify",
    "ultra_low_vram",
    "ultra_remove_bg",
]


def apply_preset(name: str):
    p = PRESETS[name]
    return tuple(p[key] for key in PRESET_OUTPUT_KEYS)


def artifact_choices(artifacts: list[str], labels: dict[str, str]) -> list[tuple[str, str]]:
    choices = []
    for path in artifacts:
        label = labels.get(path, Path(path).name)
        choices.append((label, path))
    return choices


def add_artifact(path: str, label: str, artifacts: list[str], labels: dict[str, str]) -> str:
    path = abs_path(path)
    if path not in artifacts:
        artifacts.append(path)
    labels[path] = label
    return path


def select_artifact(path: str | None):
    return path or None


ROI_PRESETS = {
    "Head / face": ((0.5, 0.91, 0.45), (0.65, 0.18, 0.9)),
    "Hair / head silhouette": ((0.5, 0.86, 0.5), (0.62, 0.34, 0.62)),
    "Torso": ((0.5, 0.58, 0.5), (0.62, 0.38, 0.52)),
    "Hands / lower arms": ((0.5, 0.42, 0.5), (0.9, 0.34, 0.65)),
    "Legs / feet": ((0.5, 0.2, 0.5), (0.72, 0.36, 0.54)),
    "Custom": ((0.5, 0.82, 0.5), (0.46, 0.34, 0.46)),
}


def apply_roi_preset(name: str):
    center, size = ROI_PRESETS[name]
    return (*center, *size)


def run_roi_enhance(
    source_mesh,
    image_file,
    gpu,
    roi_mode,
    roi_label,
    roi_center_x,
    roi_center_y,
    roi_center_z,
    roi_size_x,
    roi_size_y,
    roi_size_z,
    roi_context,
    roi_fallback_faces,
    seed,
    ultra_steps,
    ultra_latents,
    ultra_chunk,
    ultra_octree,
    ultra_scale,
    ultra_low_vram,
    ultra_remove_bg,
):
    if not source_mesh:
        yield "Choose a GLB in View artifact first.", None, [], gr.update(choices=[], value=None)
        return
    if roi_mode == "ultrashape" and image_file is None:
        yield "UltraShape ROI refinement needs the original input image.", None, [], gr.update(choices=[], value=None)
        return

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_mesh)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in (roi_label or "roi")).strip("_") or "roi"
    out_dir = RUN_ROOT / f"{run_id}_{source_path.stem}_{safe_label}_roi"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_file.name if hasattr(image_file, "name") else str(image_file) if image_file else None
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "tools" / "roi_enhance.py"),
        "--mesh",
        abs_path(source_path),
        "--out-dir",
        abs_path(out_dir),
        "--label",
        safe_label,
        "--mode",
        roi_mode,
        "--roi-center",
        f"{float(roi_center_x)},{float(roi_center_y)},{float(roi_center_z)}",
        "--roi-size",
        f"{float(roi_size_x)},{float(roi_size_y)},{float(roi_size_z)}",
        "--context-multiplier",
        str(float(roi_context)),
        "--fallback-faces",
        str(int(roi_fallback_faces)),
        "--seed",
        str(int(seed)),
        "--ultrashape-steps",
        str(int(ultra_steps)),
        "--ultrashape-scale",
        str(float(ultra_scale)),
        "--ultrashape-num-latents",
        str(int(ultra_latents)),
        "--ultrashape-chunk-size",
        str(int(ultra_chunk)),
        "--ultrashape-octree-res",
        str(int(ultra_octree)),
    ]
    if image_path:
        cmd.extend(["--image", abs_path(image_path)])
    cmd.append("--ultrashape-low-vram" if ultra_low_vram else "--no-ultrashape-low-vram")
    if ultra_remove_bg:
        cmd.append("--ultrashape-remove-bg")

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu).strip()
    env["PYTHONUNBUFFERED"] = "1"

    log_lines = [
        f"ROI run directory: {out_dir}",
        "Command:",
        " ".join(q(x) for x in cmd),
        "",
    ]
    artifacts: list[str] = []
    labels: dict[str, str] = {}
    preview = None
    yield "\n".join(log_lines), preview, artifacts, gr.update(choices=[], value=None)

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        log_lines.append(line)
        if line.startswith("ROI_ARTIFACT\t"):
            parts = line.split("\t", 2)
            if len(parts) == 3 and Path(parts[2]).exists():
                preview = add_artifact(parts[2], parts[1], artifacts, labels)
        yield "\n".join(log_lines[-500:]), preview, artifacts, gr.update(choices=artifact_choices(artifacts, labels), value=preview)

    code = proc.wait()
    for glb in sorted(out_dir.glob("*.glb")):
        label = glb.stem.replace(f"{safe_label}_", "")
        preview = add_artifact(glb, label, artifacts, labels)
    log_lines.append("")
    log_lines.append(f"Exit code: {code}")
    yield "\n".join(log_lines[-500:]), preview, artifacts, gr.update(choices=artifact_choices(artifacts, labels), value=preview)


def build_command(
    image_path,
    out_path,
    model,
    pipeline,
    seed,
    max_tokens,
    sparse_res,
    stage1_max_voxels,
    decimation,
    texture_size,
    background,
    inference_dtype,
    ss_steps,
    ss_guidance,
    ss_rescale,
    ss_rescale_t,
    ss_interval_start,
    ss_interval_end,
    shape_steps,
    shape_guidance,
    shape_rescale,
    shape_rescale_t,
    shape_interval_start,
    shape_interval_end,
    tex_steps,
    tex_guidance,
    tex_rescale,
    tex_rescale_t,
    tex_interval_start,
    tex_interval_end,
    export_remesh,
    fill_holes,
    max_hole_perimeter,
    remesh_band,
    remesh_project,
    pre_simplify,
    save_stages,
    stop_after,
    faithc_mode,
    faithc_resolution,
    faithc_tri_mode,
    faithc_margin,
    faithc_normalize,
    faithc_clamp,
    faithc_flux,
    ultra_mode,
    ultra_steps,
    ultra_latents,
    ultra_chunk,
    ultra_octree,
    ultra_scale,
    ultra_input_simplify,
    ultra_low_vram,
    ultra_remove_bg,
):
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "local_image_to_3d.py"),
        image_path,
        "--out",
        out_path,
        "--model",
        model,
        "--pipeline-type",
        pipeline,
        "--seed",
        str(int(seed)),
        "--max-num-tokens",
        str(int(max_tokens)),
        "--sparse-structure-resolution",
        str(int(sparse_res)),
        "--stage1-max-voxels",
        str(int(stage1_max_voxels)),
        "--decimation-target",
        str(int(decimation)),
        "--texture-size",
        str(int(texture_size)),
        "--background",
        background,
        "--inference-dtype",
        inference_dtype,
        "--ss-steps",
        str(int(ss_steps)),
        "--ss-guidance-strength",
        str(float(ss_guidance)),
        "--ss-guidance-rescale",
        str(float(ss_rescale)),
        "--ss-rescale-t",
        str(float(ss_rescale_t)),
        "--ss-guidance-interval-start",
        str(float(ss_interval_start)),
        "--ss-guidance-interval-end",
        str(float(ss_interval_end)),
        "--shape-steps",
        str(int(shape_steps)),
        "--shape-guidance-strength",
        str(float(shape_guidance)),
        "--shape-guidance-rescale",
        str(float(shape_rescale)),
        "--shape-rescale-t",
        str(float(shape_rescale_t)),
        "--shape-guidance-interval-start",
        str(float(shape_interval_start)),
        "--shape-guidance-interval-end",
        str(float(shape_interval_end)),
        "--tex-steps",
        str(int(tex_steps)),
        "--tex-guidance-strength",
        str(float(tex_guidance)),
        "--tex-guidance-rescale",
        str(float(tex_rescale)),
        "--tex-rescale-t",
        str(float(tex_rescale_t)),
        "--tex-guidance-interval-start",
        str(float(tex_interval_start)),
        "--tex-guidance-interval-end",
        str(float(tex_interval_end)),
        "--stop-after",
        stop_after,
        "--max-hole-perimeter",
        str(float(max_hole_perimeter)),
        "--remesh-band",
        str(float(remesh_band)),
        "--remesh-project",
        str(float(remesh_project)),
        "--pre-simplify-target",
        str(int(pre_simplify)),
        "--faithc-mode",
        faithc_mode,
        "--faithc-resolution",
        str(int(faithc_resolution)),
        "--faithc-tri-mode",
        faithc_tri_mode,
        "--faithc-margin",
        str(float(faithc_margin)),
        "--ultrashape-mode",
        ultra_mode,
        "--ultrashape-steps",
        str(int(ultra_steps)),
        "--ultrashape-scale",
        str(float(ultra_scale)),
        "--ultrashape-num-latents",
        str(int(ultra_latents)),
        "--ultrashape-chunk-size",
        str(int(ultra_chunk)),
        "--ultrashape-octree-res",
        str(int(ultra_octree)),
        "--ultrashape-input-simplify-target",
        str(int(ultra_input_simplify)),
    ]
    if export_remesh:
        cmd.append("--export-remesh")
    else:
        cmd.append("--no-export-remesh")
    if fill_holes:
        cmd.append("--fill-holes")
    else:
        cmd.append("--no-fill-holes")
    cmd.append("--faithc-normalize" if faithc_normalize else "--no-faithc-normalize")
    cmd.append("--faithc-clamp-anchors" if faithc_clamp else "--no-faithc-clamp-anchors")
    cmd.append("--faithc-compute-flux" if faithc_flux else "--no-faithc-compute-flux")
    if save_stages:
        cmd.extend(["--save-stage1", "--save-stage2", "--save-stage3"])
    if ultra_low_vram:
        cmd.append("--ultrashape-low-vram")
    else:
        cmd.append("--no-ultrashape-low-vram")
    if ultra_remove_bg:
        cmd.append("--ultrashape-remove-bg")
    return cmd


def run_generation(
    image_file,
    gpu,
    model,
    preset,
    pipeline,
    seed,
    max_tokens,
    sparse_res,
    stage1_max_voxels,
    decimation,
    texture_size,
    background,
    inference_dtype,
    ss_steps,
    ss_guidance,
    ss_rescale,
    ss_rescale_t,
    ss_interval_start,
    ss_interval_end,
    shape_steps,
    shape_guidance,
    shape_rescale,
    shape_rescale_t,
    shape_interval_start,
    shape_interval_end,
    tex_steps,
    tex_guidance,
    tex_rescale,
    tex_rescale_t,
    tex_interval_start,
    tex_interval_end,
    export_remesh,
    fill_holes,
    max_hole_perimeter,
    remesh_band,
    remesh_project,
    pre_simplify,
    save_stages,
    stop_after,
    faithc_mode,
    faithc_resolution,
    faithc_tri_mode,
    faithc_margin,
    faithc_normalize,
    faithc_clamp,
    faithc_flux,
    ultra_mode,
    ultra_steps,
    ultra_latents,
    ultra_chunk,
    ultra_octree,
    ultra_scale,
    ultra_input_simplify,
    ultra_low_vram,
    ultra_remove_bg,
):
    if image_file is None:
        yield "Upload an image first.", None, [], gr.update(choices=[], value=None)
        return

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    image_path = image_file.name if hasattr(image_file, "name") else str(image_file)
    safe_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(image_path).stem)
    out_dir = RUN_ROOT / f"{run_id}_{safe_stem}_{preset.replace(' ', '_').replace('/', '-')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_stem}_{pipeline}.glb"

    cmd = build_command(
        abs_path(image_path),
        abs_path(out_path),
        model,
        pipeline,
        seed,
        max_tokens,
        sparse_res,
        stage1_max_voxels,
        decimation,
        texture_size,
        background,
        inference_dtype,
        ss_steps,
        ss_guidance,
        ss_rescale,
        ss_rescale_t,
        ss_interval_start,
        ss_interval_end,
        shape_steps,
        shape_guidance,
        shape_rescale,
        shape_rescale_t,
        shape_interval_start,
        shape_interval_end,
        tex_steps,
        tex_guidance,
        tex_rescale,
        tex_rescale_t,
        tex_interval_start,
        tex_interval_end,
        export_remesh,
        fill_holes,
        max_hole_perimeter,
        remesh_band,
        remesh_project,
        pre_simplify,
        save_stages,
        stop_after,
        faithc_mode,
        faithc_resolution,
        faithc_tri_mode,
        faithc_margin,
        faithc_normalize,
        faithc_clamp,
        faithc_flux,
        ultra_mode,
        ultra_steps,
        ultra_latents,
        ultra_chunk,
        ultra_octree,
        ultra_scale,
        ultra_input_simplify,
        ultra_low_vram,
        ultra_remove_bg,
    )
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu).strip()
    env["ATTN_BACKEND"] = choose_dense_attention_backend()
    env["SPARSE_ATTN_BACKEND"] = env.get("SPARSE_ATTN_BACKEND", "xformers")
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONUNBUFFERED"] = "1"

    log_lines = [
        f"Run directory: {out_dir}",
        "Command:",
        " ".join(q(x) for x in cmd),
        "",
    ]
    artifacts: list[str] = []
    labels: dict[str, str] = {}
    preview = None
    yield "\n".join(log_lines), preview, artifacts, gr.update(choices=[], value=None)

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        log_lines.append(line)
        if line.startswith("STAGE_ARTIFACT\t"):
            parts = line.split("\t", 2)
            if len(parts) == 3 and Path(parts[2]).exists():
                preview = add_artifact(parts[2], parts[1], artifacts, labels)
        if out_path.exists():
            preview = add_artifact(out_path, "final_glb", artifacts, labels)
        yield "\n".join(log_lines[-500:]), preview, artifacts, gr.update(choices=artifact_choices(artifacts, labels), value=preview)

    code = proc.wait()
    if out_path.exists():
        preview = add_artifact(out_path, "final_glb", artifacts, labels)
    for glb in sorted(out_dir.glob("*.glb")):
        label = Path(glb).stem.replace(f"{safe_stem}_", "")
        preview = add_artifact(glb, label, artifacts, labels)
    log_lines.append("")
    log_lines.append(f"Exit code: {code}")
    yield "\n".join(log_lines[-500:]), preview, artifacts, gr.update(choices=artifact_choices(artifacts, labels), value=preview)


def build_ui():
    with gr.Blocks(title="TRELLIS.2 Remote Lab") as app:
        gr.Markdown("## TRELLIS.2 Remote Lab")
        with gr.Row():
            with gr.Column(scale=4, min_width=380):
                image = gr.File(label="Input image", file_types=["image"], file_count="single")
                preset = gr.Dropdown(
                    label="Preset",
                    choices=list(PRESETS),
                    value="UltraShape 25 / 8192",
                    info="Applies a full run configuration across global controls, TRELLIS stages, export, and optional refinement.",
                )

                with gr.Accordion("Global run setup", open=True):
                    with gr.Row():
                        gpu = gr.Textbox(
                            label="CUDA visible GPU",
                            value="0",
                            scale=1,
                            info="Value passed to CUDA_VISIBLE_DEVICES for the subprocess.",
                        )
                        pipeline = gr.Dropdown(
                            ["512", "1024", "1024_cascade", "1536_cascade"],
                            label="TRELLIS pipeline",
                            value="512",
                            scale=2,
                            info="Stage-2 shape path. 512/1024 generate directly; cascade starts at 512 then refines to 1024 or 1536 and uses Cascade max tokens.",
                        )
                    model = gr.Textbox(
                        label="TRELLIS model",
                        value="microsoft/TRELLIS.2-4B",
                        info="Hugging Face model id or local model folder.",
                    )
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=0, precision=0, info="Random seed for TRELLIS and UltraShape.")
                        background = gr.Dropdown(
                            ["keep", "local-white", "trellis"],
                            label="Background",
                            value="keep",
                            info="Input preparation before Stage 1. Use keep for RGBA/transparent inputs.",
                        )
                        inference_dtype = gr.Dropdown(
                            ["auto", "fp16", "bf16", "float32"],
                            label="Inference dtype",
                            value="auto",
                            info="Precision for TRELLIS flow models. Auto uses fp16 on CUDA for broad RTX compatibility.",
                        )

                with gr.Accordion("Stage 1: sparse structure", open=True):
                    with gr.Row():
                        sparse_res = gr.Number(
                            label="Sparse grid resolution",
                            value=32,
                            precision=0,
                            info="Stage-1 coarse occupancy grid resolution before shape generation. This is separate from the 512/1024/1536 Stage-2 shape grid.",
                        )
                        stage1_max_voxels = gr.Number(
                            label="Preview voxel cap",
                            value=12000,
                            precision=0,
                            info="Maximum exported boxes for the Stage 1 diagnostic GLB.",
                        )
                    with gr.Row():
                        ss_steps = gr.Number(label="Steps", value=12, precision=0, info="Sparse structure diffusion steps.")
                        ss_guidance = gr.Number(label="Guidance", value=7.5, info="Image adherence for Stage 1.")
                        ss_rescale = gr.Number(label="Guidance rescale", value=0.7, info="Dampens over-guided artifacts.")
                    with gr.Row():
                        ss_rescale_t = gr.Number(label="Rescale T", value=5.0, info="Time scaling used by the flow sampler.")
                        ss_interval_start = gr.Number(label="Guidance start", value=0.6, info="Fraction of the sampling schedule where guidance begins.")
                        ss_interval_end = gr.Number(label="Guidance end", value=1.0, info="Fraction of the sampling schedule where guidance ends.")

                with gr.Accordion("Stage 2: shape geometry", open=True):
                    max_tokens = gr.Number(
                        label="Cascade max tokens",
                        value=12288,
                        precision=0,
                        info="Only affects 1024_cascade and 1536_cascade. Caps high-res Stage-2 latent tokens during cascade refinement; higher can retain more thin/detail structure but raises VRAM/time sharply.",
                    )
                    with gr.Row():
                        shape_steps = gr.Number(label="Steps", value=12, precision=0, info="Shape latent diffusion steps.")
                        shape_guidance = gr.Number(label="Guidance", value=7.5, info="Image adherence for Stage 2 geometry.")
                        shape_rescale = gr.Number(label="Guidance rescale", value=0.5, info="Dampens over-guided shape artifacts.")
                    with gr.Row():
                        shape_rescale_t = gr.Number(label="Rescale T", value=3.0, info="Time scaling used by the flow sampler.")
                        shape_interval_start = gr.Number(label="Guidance start", value=0.6, info="Fraction of the sampling schedule where guidance begins.")
                        shape_interval_end = gr.Number(label="Guidance end", value=1.0, info="Fraction of the sampling schedule where guidance ends.")

                with gr.Accordion("Stage 3: texture/material latent", open=True):
                    with gr.Row():
                        tex_steps = gr.Number(label="Steps", value=12, precision=0, info="Texture/material latent diffusion steps.")
                        tex_guidance = gr.Number(label="Guidance", value=1.0, info="Image adherence for Stage 3 texture/material generation.")
                        tex_rescale = gr.Number(label="Guidance rescale", value=0.0, info="Dampens over-guided texture artifacts.")
                    with gr.Row():
                        tex_rescale_t = gr.Number(label="Rescale T", value=3.0, info="Time scaling used by the flow sampler.")
                        tex_interval_start = gr.Number(label="Guidance start", value=0.6, info="Fraction of the sampling schedule where guidance begins.")
                        tex_interval_end = gr.Number(label="Guidance end", value=0.9, info="Fraction of the sampling schedule where guidance ends.")
                    texture_size = gr.Number(
                        label="Texture atlas size",
                        value=1024,
                        precision=0,
                        info="Final texture atlas size sampled from the Stage 3 attribute volume.",
                    )

                with gr.Accordion("Final GLB export", open=False):
                    with gr.Row():
                        decimation = gr.Number(label="Decimation target", value=100000, precision=0, info="Target face count for exported GLB.")
                        pre_simplify = gr.Number(label="Pre-simplify target", value=16777216, precision=0, info="Initial simplification before GLB export. 0 disables it.")
                    with gr.Row():
                        export_remesh = gr.Checkbox(label="Remesh during export", value=True, info="Runs o-voxel remeshing before GLB export.")
                        fill_holes = gr.Checkbox(label="Fill holes before export", value=True, info="Fills holes on the decoded TRELLIS mesh before export.")
                    with gr.Row():
                        max_hole_perimeter = gr.Number(label="Max hole perimeter", value=0.03, info="Largest hole perimeter TRELLIS is allowed to fill.")
                        remesh_band = gr.Number(label="Remesh band", value=1.0, info="Narrow-band width used by o-voxel remeshing.")
                        remesh_project = gr.Number(label="Remesh project", value=0.9, info="Projection strength back to the generated surface after remeshing.")

                with gr.Accordion("Diagnostics and run stopping", open=False):
                    with gr.Row():
                        save_stages = gr.Checkbox(label="Save stage artifacts", value=True, info="Exports Stage 1/2/3 GLBs as they complete.")
                        stop_after = gr.Dropdown(
                            ["none", "stage1", "stage2", "ultrashape", "stage3"],
                            label="Stop after",
                            value="none",
                            info="Stops after a selected stage to avoid spending time on later stages. UltraShape requires UltraShape mode before-texture.",
                        )

                with gr.Accordion("Optional stage: FaithC", open=False):
                    faithc_mode = gr.Dropdown(
                        ["off", "after-stage2", "after-stage3", "after-final"],
                        label="Run FaithC",
                        value="off",
                        info="Runs Faithful Contouring against the selected TRELLIS artifact.",
                    )
                    with gr.Row():
                        faithc_resolution = gr.Number(label="Resolution", value=256, precision=0)
                        faithc_tri_mode = gr.Dropdown(["auto", "simple_02", "simple_13", "length", "angle", "normal", "normal_abs"], label="Triangle mode", value="auto")
                        faithc_margin = gr.Number(label="Margin", value=0.05)
                    with gr.Row():
                        faithc_normalize = gr.Checkbox(label="Normalize", value=False)
                        faithc_clamp = gr.Checkbox(label="Clamp anchors", value=True)
                        faithc_flux = gr.Checkbox(label="Compute edge flux", value=True)

                with gr.Accordion("Optional stage: UltraShape", open=True):
                    ultra_mode = gr.Dropdown(
                        ["off", "before-texture", "after-stage2", "after-stage3", "after-final"],
                        label="Run UltraShape",
                        value="after-final",
                        info="before-texture matches the Comfy workflow: Stage 2 mesh -> UltraShape -> TRELLIS texturing.",
                    )
                    with gr.Row():
                        ultra_steps = gr.Number(label="Steps", value=25, precision=0, info="UltraShape diffusion/refinement steps.")
                        ultra_latents = gr.Number(label="Latents", value=8192, precision=0, info="UltraShape latent token count.")
                        ultra_scale = gr.Number(label="Mesh scale", value=0.99, info="Normalization scale passed to UltraShape.")
                    with gr.Row():
                        ultra_chunk = gr.Number(label="Chunk size", value=2048, precision=0, info="Inference chunk size. Lower reduces VRAM and slows refinement.")
                        ultra_octree = gr.Number(label="Octree res", value=512, precision=0, info="UltraShape mesh extraction resolution.")
                        ultra_input_simplify = gr.Number(label="Input simplify target", value=1000000, precision=0, info="Simplifies Stage 2 before before-texture UltraShape. 0 disables.")
                    with gr.Row():
                        ultra_low_vram = gr.Checkbox(label="Low VRAM", value=True, info="Uses UltraShape low-VRAM/offload mode when available.")
                        ultra_remove_bg = gr.Checkbox(label="Remove background", value=False, info="Runs UltraShape background removal. Usually leave off if TRELLIS input is already prepared.")

                run = gr.Button("Run", variant="primary")

            with gr.Column(scale=7, min_width=680):
                artifact_picker = gr.Dropdown(label="View artifact", choices=[], value=None, info="Choose any GLB emitted by this run.")
                viewer = gr.Model3D(label="Selected GLB", height=560, clear_color=(0.18, 0.19, 0.2, 1.0))
                artifacts = gr.File(label="Artifacts", file_count="multiple")
                log = gr.Textbox(label="Run log", lines=22, elem_id="run-log", autoscroll=True)
                with gr.Accordion("Experimental: ROI zoom/enhance", open=False):
                    gr.Markdown(
                        "Consumes the currently selected GLB. Inspect mode exports a cropped ROI, context shell, and overlay. "
                        "UltraShape mode refines only that crop, inverse-transforms it back, and emits a crude merged preview."
                    )
                    with gr.Row():
                        roi_mode = gr.Dropdown(
                            ["inspect", "ultrashape"],
                            label="Mode",
                            value="inspect",
                            info="Inspect is geometry-only and fast. UltraShape runs the ROI through UltraShape and attempts a world-space patch preview.",
                        )
                        roi_preset = gr.Dropdown(
                            list(ROI_PRESETS),
                            label="ROI preset",
                            value="Head / face",
                            info="Normalized bounding-box presets assuming a Y-up character/object.",
                        )
                        roi_label = gr.Textbox(label="ROI label", value="face", info="Used in output filenames.")
                    with gr.Row():
                        roi_center_x = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Center X")
                        roi_center_y = gr.Slider(0.0, 1.0, value=0.91, step=0.01, label="Center Y")
                        roi_center_z = gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="Center Z")
                    with gr.Row():
                        roi_size_x = gr.Slider(0.01, 1.0, value=0.65, step=0.01, label="Size X")
                        roi_size_y = gr.Slider(0.01, 1.0, value=0.18, step=0.01, label="Size Y")
                        roi_size_z = gr.Slider(0.01, 1.0, value=0.9, step=0.01, label="Size Z")
                    with gr.Row():
                        roi_context = gr.Number(label="Context multiplier", value=1.35, info="Exports an expanded context crop around the ROI.")
                        roi_fallback_faces = gr.Number(label="Fallback faces", value=2000, precision=0, info="Nearest faces to use if the ROI box misses the mesh.")
                    with gr.Row():
                        roi_ultra_steps = gr.Number(label="UltraShape steps", value=25, precision=0)
                        roi_ultra_latents = gr.Number(label="UltraShape latents", value=8192, precision=0)
                        roi_ultra_octree = gr.Number(label="UltraShape octree", value=512, precision=0)
                    with gr.Row():
                        roi_ultra_chunk = gr.Number(label="UltraShape chunk", value=2048, precision=0)
                        roi_ultra_scale = gr.Number(label="UltraShape scale", value=0.99)
                        roi_ultra_low_vram = gr.Checkbox(label="Low VRAM", value=True)
                        roi_ultra_remove_bg = gr.Checkbox(label="Remove bg", value=False)
                    roi_run = gr.Button("Run ROI Zoom/Enhance", variant="secondary")
                    roi_picker = gr.Dropdown(label="ROI artifact", choices=[], value=None)
                    roi_viewer = gr.Model3D(label="ROI selected GLB", height=420, clear_color=(0.18, 0.19, 0.2, 1.0))
                    roi_files = gr.File(label="ROI artifacts", file_count="multiple")
                    roi_log = gr.Textbox(label="ROI log", lines=12, elem_id="roi-log", autoscroll=True)

        preset_outputs = [
            pipeline,
            seed,
            max_tokens,
            sparse_res,
            stage1_max_voxels,
            ss_steps,
            ss_guidance,
            ss_rescale,
            ss_rescale_t,
            ss_interval_start,
            ss_interval_end,
            shape_steps,
            shape_guidance,
            shape_rescale,
            shape_rescale_t,
            shape_interval_start,
            shape_interval_end,
            tex_steps,
            tex_guidance,
            tex_rescale,
            tex_rescale_t,
            tex_interval_start,
            tex_interval_end,
            texture_size,
            decimation,
            export_remesh,
            fill_holes,
            max_hole_perimeter,
            remesh_band,
            remesh_project,
            pre_simplify,
            save_stages,
            stop_after,
            faithc_mode,
            faithc_resolution,
            faithc_tri_mode,
            faithc_margin,
            faithc_normalize,
            faithc_clamp,
            faithc_flux,
            ultra_mode,
            ultra_steps,
            ultra_latents,
            ultra_chunk,
            ultra_octree,
            ultra_scale,
            ultra_input_simplify,
            ultra_low_vram,
            ultra_remove_bg,
        ]
        preset.change(
            apply_preset,
            inputs=[preset],
            outputs=preset_outputs,
        )
        run.click(
            run_generation,
            inputs=[
                image,
                gpu,
                model,
                preset,
                pipeline,
                seed,
                max_tokens,
                sparse_res,
                stage1_max_voxels,
                decimation,
                texture_size,
                background,
                inference_dtype,
                ss_steps,
                ss_guidance,
                ss_rescale,
                ss_rescale_t,
                ss_interval_start,
                ss_interval_end,
                shape_steps,
                shape_guidance,
                shape_rescale,
                shape_rescale_t,
                shape_interval_start,
                shape_interval_end,
                tex_steps,
                tex_guidance,
                tex_rescale,
                tex_rescale_t,
                tex_interval_start,
                tex_interval_end,
                export_remesh,
                fill_holes,
                max_hole_perimeter,
                remesh_band,
                remesh_project,
                pre_simplify,
                save_stages,
                stop_after,
                faithc_mode,
                faithc_resolution,
                faithc_tri_mode,
                faithc_margin,
                faithc_normalize,
                faithc_clamp,
                faithc_flux,
                ultra_mode,
                ultra_steps,
                ultra_latents,
                ultra_chunk,
                ultra_octree,
                ultra_scale,
                ultra_input_simplify,
                ultra_low_vram,
                ultra_remove_bg,
            ],
            outputs=[log, viewer, artifacts, artifact_picker],
        )
        artifact_picker.change(select_artifact, inputs=[artifact_picker], outputs=[viewer])
        roi_preset.change(
            apply_roi_preset,
            inputs=[roi_preset],
            outputs=[roi_center_x, roi_center_y, roi_center_z, roi_size_x, roi_size_y, roi_size_z],
        )
        roi_run.click(
            run_roi_enhance,
            inputs=[
                artifact_picker,
                image,
                gpu,
                roi_mode,
                roi_label,
                roi_center_x,
                roi_center_y,
                roi_center_z,
                roi_size_x,
                roi_size_y,
                roi_size_z,
                roi_context,
                roi_fallback_faces,
                seed,
                roi_ultra_steps,
                roi_ultra_latents,
                roi_ultra_chunk,
                roi_ultra_octree,
                roi_ultra_scale,
                roi_ultra_low_vram,
                roi_ultra_remove_bg,
            ],
            outputs=[roi_log, roi_viewer, roi_files, roi_picker],
        )
        roi_picker.change(select_artifact, inputs=[roi_picker], outputs=[roi_viewer])
    return app


def main():
    parser = argparse.ArgumentParser(description="Remote Gradio launcher for TRELLIS.2 experiments.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    app = build_ui()
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[str(ROOT)],
        js=RUN_LOG_AUTOSCROLL_JS,
    )


if __name__ == "__main__":
    main()


def dev_main():
    app = build_ui()
    app.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=[str(ROOT)],
        js=RUN_LOG_AUTOSCROLL_JS,
    )
