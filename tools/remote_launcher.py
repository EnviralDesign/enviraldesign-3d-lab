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
        "export_remesh": True,
        "fill_holes": True,
        "save_stages": True,
        "ultra_mode": "off",
        "ultra_steps": 25,
        "ultra_latents": 8192,
        "ultra_chunk": 2048,
        "ultra_octree": 512,
    },
    "UltraShape 25 / 8192": {
        "pipeline": "512",
        "seed": 0,
        "max_tokens": 12288,
        "sparse_res": 32,
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
        "export_remesh": True,
        "fill_holes": True,
        "save_stages": True,
        "ultra_mode": "after-final",
        "ultra_steps": 25,
        "ultra_latents": 8192,
        "ultra_chunk": 2048,
        "ultra_octree": 512,
    },
    "UltraShape 25 / 16384": {
        "pipeline": "512",
        "seed": 0,
        "max_tokens": 12288,
        "sparse_res": 32,
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
        "export_remesh": True,
        "fill_holes": True,
        "save_stages": True,
        "ultra_mode": "after-final",
        "ultra_steps": 25,
        "ultra_latents": 16384,
        "ultra_chunk": 2048,
        "ultra_octree": 512,
    },
}


def q(value: str) -> str:
    return shlex.quote(str(value))


def abs_path(path: str | Path) -> str:
    return str(Path(path).resolve())


def apply_preset(name: str):
    p = PRESETS[name]
    return (
        p["pipeline"],
        p["seed"],
        p["max_tokens"],
        p["sparse_res"],
        p["decimation"],
        p["texture_size"],
        p["ss_steps"],
        p["shape_steps"],
        p["tex_steps"],
        p["export_remesh"],
        p["fill_holes"],
        p["save_stages"],
        p["ultra_mode"],
        p["ultra_steps"],
        p["ultra_latents"],
        p["ultra_chunk"],
        p["ultra_octree"],
    )


def build_command(
    image_path,
    out_path,
    model,
    pipeline,
    seed,
    max_tokens,
    sparse_res,
    decimation,
    texture_size,
    background,
    inference_dtype,
    ss_steps,
    shape_steps,
    tex_steps,
    export_remesh,
    fill_holes,
    save_stages,
    stop_after,
    ultra_mode,
    ultra_steps,
    ultra_latents,
    ultra_chunk,
    ultra_octree,
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
        "--shape-steps",
        str(int(shape_steps)),
        "--tex-steps",
        str(int(tex_steps)),
        "--stop-after",
        stop_after,
        "--ultrashape-mode",
        ultra_mode,
        "--ultrashape-steps",
        str(int(ultra_steps)),
        "--ultrashape-num-latents",
        str(int(ultra_latents)),
        "--ultrashape-chunk-size",
        str(int(ultra_chunk)),
        "--ultrashape-octree-res",
        str(int(ultra_octree)),
    ]
    if export_remesh:
        cmd.append("--export-remesh")
    else:
        cmd.append("--no-export-remesh")
    if fill_holes:
        cmd.append("--fill-holes")
    else:
        cmd.append("--no-fill-holes")
    if save_stages:
        cmd.extend(["--save-stage1", "--save-stage2", "--save-stage3"])
    if ultra_mode != "off":
        cmd.append("--ultrashape-low-vram")
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
    decimation,
    texture_size,
    background,
    inference_dtype,
    ss_steps,
    shape_steps,
    tex_steps,
    export_remesh,
    fill_holes,
    save_stages,
    stop_after,
    ultra_mode,
    ultra_steps,
    ultra_latents,
    ultra_chunk,
    ultra_octree,
):
    if image_file is None:
        yield "Upload an image first.", None, []
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
        decimation,
        texture_size,
        background,
        inference_dtype,
        ss_steps,
        shape_steps,
        tex_steps,
        export_remesh,
        fill_holes,
        save_stages,
        stop_after,
        ultra_mode,
        ultra_steps,
        ultra_latents,
        ultra_chunk,
        ultra_octree,
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
    preview = None
    yield "\n".join(log_lines), preview, artifacts

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
                artifacts.append(parts[2])
                preview = parts[2]
        if out_path.exists():
            final = abs_path(out_path)
            if final not in artifacts:
                artifacts.append(final)
            preview = final
        yield "\n".join(log_lines[-500:]), preview, artifacts

    code = proc.wait()
    if out_path.exists():
        final = abs_path(out_path)
        if final not in artifacts:
            artifacts.append(final)
        preview = final
    for glb in sorted(out_dir.glob("*.glb")):
        glb_path = abs_path(glb)
        if glb_path not in artifacts:
            artifacts.append(glb_path)
    log_lines.append("")
    log_lines.append(f"Exit code: {code}")
    yield "\n".join(log_lines[-500:]), preview, artifacts


def build_ui():
    with gr.Blocks(title="TRELLIS.2 Remote Lab") as app:
        gr.Markdown("## TRELLIS.2 Remote Lab")
        with gr.Row():
            with gr.Column(scale=4, min_width=380):
                image = gr.File(label="Input image", file_types=["image"], file_count="single")
                with gr.Row():
                    gpu = gr.Textbox(label="CUDA visible GPU", value="0", scale=1)
                    preset = gr.Dropdown(label="Preset", choices=list(PRESETS), value="UltraShape 25 / 8192", scale=3)
                model = gr.Textbox(label="TRELLIS model", value="microsoft/TRELLIS.2-4B")

                with gr.Accordion("Global", open=True):
                    pipeline = gr.Dropdown(["512", "1024", "1024_cascade", "1536_cascade"], label="Pipeline", value="512")
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=0, precision=0)
                        max_tokens = gr.Number(label="Max tokens", value=12288, precision=0)
                        sparse_res = gr.Number(label="Sparse res", value=32, precision=0)
                    with gr.Row():
                        background = gr.Dropdown(["keep", "local-white", "trellis"], label="Background", value="keep")
                        inference_dtype = gr.Dropdown(["auto", "fp16", "bf16", "float32"], label="Inference dtype", value="auto")
                        stop_after = gr.Dropdown(["none", "stage1", "stage2", "stage3"], label="Stop after", value="none")

                with gr.Accordion("TRELLIS Stages", open=False):
                    with gr.Row():
                        ss_steps = gr.Number(label="Stage 1 steps", value=12, precision=0)
                        shape_steps = gr.Number(label="Stage 2 steps", value=12, precision=0)
                        tex_steps = gr.Number(label="Stage 3 steps", value=12, precision=0)
                    save_stages = gr.Checkbox(label="Save stage artifacts", value=True)

                with gr.Accordion("Final Export", open=False):
                    with gr.Row():
                        decimation = gr.Number(label="Decimation target", value=100000, precision=0)
                        texture_size = gr.Number(label="Texture size", value=1024, precision=0)
                    with gr.Row():
                        export_remesh = gr.Checkbox(label="Remesh during export", value=True)
                        fill_holes = gr.Checkbox(label="Fill holes", value=True)

                with gr.Accordion("UltraShape", open=True):
                    ultra_mode = gr.Dropdown(["off", "after-stage2", "after-stage3", "after-final"], label="Run UltraShape", value="after-final")
                    with gr.Row():
                        ultra_steps = gr.Number(label="Steps", value=25, precision=0)
                        ultra_latents = gr.Number(label="Latents", value=8192, precision=0)
                    with gr.Row():
                        ultra_chunk = gr.Number(label="Chunk size", value=2048, precision=0)
                        ultra_octree = gr.Number(label="Octree res", value=512, precision=0)

                run = gr.Button("Run", variant="primary")

            with gr.Column(scale=7, min_width=680):
                viewer = gr.Model3D(label="Latest GLB", height=560, clear_color=(0.18, 0.19, 0.2, 1.0))
                artifacts = gr.File(label="Artifacts", file_count="multiple")
                log = gr.Textbox(label="Run log", lines=22, elem_id="run-log", autoscroll=True)

        preset.change(
            apply_preset,
            inputs=[preset],
            outputs=[
                pipeline,
                seed,
                max_tokens,
                sparse_res,
                decimation,
                texture_size,
                ss_steps,
                shape_steps,
                tex_steps,
                export_remesh,
                fill_holes,
                save_stages,
                ultra_mode,
                ultra_steps,
                ultra_latents,
                ultra_chunk,
                ultra_octree,
            ],
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
                decimation,
                texture_size,
                background,
                inference_dtype,
                ss_steps,
                shape_steps,
                tex_steps,
                export_remesh,
                fill_holes,
                save_stages,
                stop_after,
                ultra_mode,
                ultra_steps,
                ultra_latents,
                ultra_chunk,
                ultra_octree,
            ],
            outputs=[log, viewer, artifacts],
        )
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
    )
