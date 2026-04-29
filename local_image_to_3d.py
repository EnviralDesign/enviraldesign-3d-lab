import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

from PIL import Image
import numpy as np
import torch
import trimesh

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines import Trellis2TexturingPipeline
from trellis2.pipelines import rembg
import o_voxel


class _DisabledRembg:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self

    def cpu(self):
        return self

    def cuda(self):
        return self

    def __call__(self, image):
        raise RuntimeError("Background removal is disabled. Use an RGBA input with alpha.")


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal local TRELLIS.2 image-to-3D runner.")
    parser.add_argument("image", help="Input image path.")
    parser.add_argument("--out", default="tmp/sample.glb", help="Output GLB path.")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B", help="Model id or local model folder.")
    parser.add_argument("--pipeline-type", default="512", choices=["512", "1024", "1024_cascade", "1536_cascade"])
    parser.add_argument("--steps", type=int, default=None, help="Sampling steps for all stages.")
    parser.add_argument("--ss-steps", type=int, default=12, help="Sparse structure sampling steps.")
    parser.add_argument("--ss-guidance-strength", type=float, default=7.5)
    parser.add_argument("--ss-guidance-rescale", type=float, default=0.7)
    parser.add_argument("--ss-rescale-t", type=float, default=5.0)
    parser.add_argument("--ss-guidance-interval-start", type=float, default=0.6)
    parser.add_argument("--ss-guidance-interval-end", type=float, default=1.0)
    parser.add_argument("--shape-steps", type=int, default=12, help="Shape latent sampling steps.")
    parser.add_argument("--shape-guidance-strength", type=float, default=7.5)
    parser.add_argument("--shape-guidance-rescale", type=float, default=0.5)
    parser.add_argument("--shape-rescale-t", type=float, default=3.0)
    parser.add_argument("--shape-guidance-interval-start", type=float, default=0.6)
    parser.add_argument("--shape-guidance-interval-end", type=float, default=1.0)
    parser.add_argument("--tex-steps", type=int, default=12, help="Texture/material latent sampling steps.")
    parser.add_argument("--tex-guidance-strength", type=float, default=1.0)
    parser.add_argument("--tex-guidance-rescale", type=float, default=0.0)
    parser.add_argument("--tex-rescale-t", type=float, default=3.0)
    parser.add_argument("--tex-guidance-interval-start", type=float, default=0.6)
    parser.add_argument("--tex-guidance-interval-end", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-num-tokens", type=int, default=12288)
    parser.add_argument("--sparse-structure-resolution", type=int, default=None)
    parser.add_argument("--decimation-target", type=int, default=100000)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-hole-perimeter", type=float, default=0.03)
    parser.add_argument("--export-remesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--remesh-band", type=float, default=1.0)
    parser.add_argument("--remesh-project", type=float, default=0.9)
    parser.add_argument("--pre-simplify-target", type=int, default=16777216)
    parser.add_argument("--save-stage1", action="store_true", help="Save sparse structure occupancy as a diagnostic GLB.")
    parser.add_argument("--save-stage2", action="store_true", help="Save shape-only decoded geometry as a diagnostic GLB.")
    parser.add_argument("--save-stage3", action="store_true", help="Save decoded textured-stage geometry before GLB remesh/export as a diagnostic GLB.")
    parser.add_argument("--stop-after", choices=["none", "stage1", "stage2", "ultrashape", "stage3"], default="none")
    parser.add_argument("--stage1-max-voxels", type=int, default=12000, help="Cap sparse voxels exported as boxes for stage-1 GLB diagnostics.")
    parser.add_argument("--faithc-mode", default="off", choices=["off", "after-stage2", "after-stage3", "after-final"], help="Run FaithC contour reconstruction on a selected TRELLIS mesh artifact.")
    parser.add_argument("--faithc-resolution", type=int, default=256)
    parser.add_argument("--faithc-tri-mode", default="auto", choices=["auto", "simple_02", "simple_13", "length", "angle", "normal", "normal_abs"])
    parser.add_argument("--faithc-margin", type=float, default=0.05)
    parser.add_argument("--faithc-normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--faithc-clamp-anchors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--faithc-compute-flux", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ultrashape-mode", default="off", choices=["off", "before-texture", "after-stage2", "after-stage3", "after-final"], help="Run UltraShape refinement on a selected TRELLIS mesh artifact.")
    parser.add_argument("--ultrashape-config", default="integrations/UltraShape-1.0/configs/infer_dit_refine.yaml")
    parser.add_argument("--ultrashape-ckpt", default="integrations/UltraShape-1.0/checkpoints/ultrashape_v1.pt")
    parser.add_argument("--ultrashape-steps", type=int, default=12)
    parser.add_argument("--ultrashape-scale", type=float, default=0.99)
    parser.add_argument("--ultrashape-num-latents", type=int, default=8192)
    parser.add_argument("--ultrashape-chunk-size", type=int, default=2048)
    parser.add_argument("--ultrashape-octree-res", type=int, default=512)
    parser.add_argument("--ultrashape-low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ultrashape-remove-bg", action="store_true")
    parser.add_argument("--ultrashape-input-simplify-target", type=int, default=1000000, help="Simplify Stage 2 mesh before --ultrashape-mode before-texture. 0 disables.")
    parser.add_argument(
        "--inference-dtype",
        default="auto",
        choices=["auto", "fp16", "bf16", "float32"],
        help=(
            "Precision for flow model inference. auto uses fp16 on CUDA because "
            "xformers sparse attention rejects bf16 on common RTX cards."
        ),
    )
    parser.add_argument(
        "--background",
        default="keep",
        choices=["keep", "local-white", "trellis"],
        help=(
            "Background handling. 'keep' avoids background removal. "
            "'local-white' makes near-white pixels transparent locally. "
            "'trellis' uses the configured RMBG model."
        ),
    )
    parser.add_argument(
        "--rembg-model",
        default="camenduru/RMBG-2.0:onnx/model_quantized.onnx",
        help=(
            "Hugging Face model id for --background trellis. "
            "Default uses an ungated RMBG-2.0 ONNX mirror."
        ),
    )
    parser.add_argument("--white-threshold", type=int, default=245, help="Threshold for --background local-white.")
    parser.add_argument("--no-export", action="store_true", help="Run generation but skip GLB extraction.")
    args = parser.parse_args()
    if args.stop_after == "ultrashape" and args.ultrashape_mode != "before-texture":
        parser.error("--stop-after ultrashape requires --ultrashape-mode before-texture")
    return args


def artifact_path(out_path: str, label: str) -> str:
    path = Path(out_path)
    return str(path.with_name(f"{path.stem}_{label}.glb"))


def print_artifact(label: str, path: str) -> None:
    print(f"STAGE_ARTIFACT\t{label}\t{os.path.abspath(path)}", flush=True)


def needs_stage_artifact(args, stage: str) -> bool:
    mode = f"after-{stage}"
    if args.faithc_mode == mode or args.ultrashape_mode == mode:
        return True
    if stage == "stage1":
        return args.save_stage1 or args.stop_after == stage
    if stage == "stage2":
        return args.save_stage2 or args.stop_after == stage
    if stage == "stage3":
        return args.save_stage3 or args.stop_after == stage
    return False


def run_command(label: str, command: list[str], cwd: str | None = None, env: dict[str, str] | None = None) -> None:
    print(f"MODULE_START\t{label}", flush=True)
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


def run_faithc_module(args, input_mesh: str, label: str) -> str:
    out_path = artifact_path(args.out, label)
    command = [
        sys.executable,
        str(Path("tools") / "run_faithc.py"),
        "--mesh", os.path.abspath(input_mesh),
        "--out", os.path.abspath(out_path),
        "--resolution", str(args.faithc_resolution),
        "--tri-mode", args.faithc_tri_mode,
        "--margin", str(args.faithc_margin),
    ]
    command.append("--normalize" if args.faithc_normalize else "--no-normalize")
    command.append("--clamp-anchors" if args.faithc_clamp_anchors else "--no-clamp-anchors")
    command.append("--compute-flux" if args.faithc_compute_flux else "--no-compute-flux")
    run_command(label, command)
    print_artifact(label, out_path)
    return out_path


def run_ultrashape_module(args, image_path: str, input_mesh: str, label: str) -> str:
    config = os.path.abspath(args.ultrashape_config)
    ckpt = os.path.abspath(args.ultrashape_ckpt)
    if not os.path.exists(config):
        raise FileNotFoundError(f"UltraShape config not found: {config}")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"UltraShape checkpoint not found: {ckpt}. Download the UltraShape weights and set the checkpoint path in the launcher."
        )

    out_path = artifact_path(args.out, label)
    module_dir = os.path.abspath(str(Path(out_path).with_suffix("")) + "_work")
    os.makedirs(module_dir, exist_ok=True)
    ultra_root = os.path.abspath(str(Path("integrations") / "UltraShape-1.0"))
    command = [
        sys.executable,
        str(Path("scripts") / "infer_dit_refine.py"),
        "--config", config,
        "--ckpt", ckpt,
        "--image", os.path.abspath(image_path),
        "--mesh", os.path.abspath(input_mesh),
        "--output_dir", module_dir,
        "--steps", str(args.ultrashape_steps),
        "--scale", str(args.ultrashape_scale),
        "--num_latents", str(args.ultrashape_num_latents),
        "--chunk_size", str(args.ultrashape_chunk_size),
        "--octree_res", str(args.ultrashape_octree_res),
        "--seed", str(args.seed),
    ]
    if args.ultrashape_low_vram:
        command.append("--low_vram")
    if args.ultrashape_remove_bg:
        command.append("--remove_bg")

    env = os.environ.copy()
    env["PYTHONPATH"] = ultra_root + os.pathsep + env.get("PYTHONPATH", "")
    cuda_124 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    if os.path.exists(cuda_124):
        env.setdefault("CUDA_HOME", cuda_124)
        env.setdefault("CUDA_PATH", cuda_124)
        env["PATH"] = os.path.join(cuda_124, "bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    run_command(label, command, cwd=ultra_root, env=env)

    expected = os.path.join(module_dir, f"{Path(image_path).stem}_refined.glb")
    if not os.path.exists(expected):
        candidates = sorted(Path(module_dir).glob("*.glb"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"UltraShape completed but no GLB was found in {module_dir}")
        expected = str(candidates[0])
    shutil.move(expected, out_path)
    print_artifact(label, out_path)
    return out_path


def load_trimesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Expected a trimesh mesh from {path}, got {type(mesh).__name__}")
    return mesh


def texture_trimesh_with_trellis(args, image: Image.Image, mesh_path: str, sampler_params: dict, resolution: int) -> trimesh.Trimesh:
    print("Loading TRELLIS texturing pipeline for UltraShape-refined mesh.", flush=True)
    texturing_pipeline = Trellis2TexturingPipeline.from_pretrained(args.model)
    apply_inference_dtype(texturing_pipeline, resolve_inference_dtype(args.inference_dtype))
    texturing_pipeline.cuda()
    mesh = load_trimesh(mesh_path)
    texture_resolution = 512 if resolution == 512 else 1024
    textured = texturing_pipeline.run(
        mesh,
        image,
        seed=args.seed,
        tex_slat_sampler_params=sampler_params,
        preprocess_image=False,
        resolution=texture_resolution,
        texture_size=args.texture_size,
    )
    del texturing_pipeline
    torch.cuda.empty_cache()
    return textured


def run_optional_modules(args, image_path: str, stage_paths: dict[str, str], final_path: str | None) -> None:
    targets = {
        "after-stage2": stage_paths.get("stage2"),
        "after-stage3": stage_paths.get("stage3"),
        "after-final": final_path,
    }
    if args.faithc_mode != "off":
        input_mesh = targets.get(args.faithc_mode)
        if not input_mesh or not os.path.exists(input_mesh):
            raise FileNotFoundError(f"FaithC requested {args.faithc_mode}, but its input mesh was not produced.")
        run_faithc_module(args, input_mesh, f"faithc_{args.faithc_mode.replace('-', '_')}")
    if args.ultrashape_mode not in ("off", "before-texture"):
        input_mesh = targets.get(args.ultrashape_mode)
        if not input_mesh or not os.path.exists(input_mesh):
            raise FileNotFoundError(f"UltraShape requested {args.ultrashape_mode}, but its input mesh was not produced.")
        run_ultrashape_module(args, image_path, input_mesh, f"ultrashape_{args.ultrashape_mode.replace('-', '_')}")


def trellis_internal_to_glb_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = vertices.copy()
    vertices[:, 1], vertices[:, 2] = vertices[:, 2].copy(), -vertices[:, 1].copy()
    return vertices


def export_basic_mesh(mesh, path: str, color=(210, 210, 210, 255)) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    vertices = mesh.vertices.detach().float().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    vertices = trellis_internal_to_glb_vertices(vertices)
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tri.visual.vertex_colors = np.tile(np.array(color, dtype=np.uint8), (len(vertices), 1))
    tri.export(path)


def export_sparse_structure(coords: torch.Tensor, resolution: int, path: str, max_voxels: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    coords = coords.detach().cpu()
    if coords.shape[0] == 0:
        raise RuntimeError("Sparse structure produced no active voxels.")
    if coords.shape[0] > max_voxels:
        print(f"Stage 1 has {coords.shape[0]} voxels; exporting {max_voxels} evenly sampled voxels for viewer responsiveness.", flush=True)
        indices = torch.linspace(0, coords.shape[0] - 1, max_voxels).long()
        coords = coords[indices]

    xyz = coords[:, 1:].float().numpy()
    size = 1.0 / float(resolution)
    centers = ((xyz + 0.5) / float(resolution)) - 0.5
    half = size * 0.38
    corners = np.array(
        [
            [-half, -half, -half],
            [ half, -half, -half],
            [ half,  half, -half],
            [-half,  half, -half],
            [-half, -half,  half],
            [ half, -half,  half],
            [ half,  half,  half],
            [-half,  half,  half],
        ],
        dtype=np.float32,
    )
    cube_faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int64,
    )
    vertices = (centers[:, None, :] + corners[None, :, :]).reshape(-1, 3)
    offsets = (np.arange(len(centers), dtype=np.int64) * 8)[:, None, None]
    faces = (cube_faces[None, :, :] + offsets).reshape(-1, 3)
    vertices = trellis_internal_to_glb_vertices(vertices)
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tri.visual.vertex_colors = np.tile(np.array([90, 170, 255, 210], dtype=np.uint8), (len(vertices), 1))
    tri.export(path)


def resolve_inference_dtype(dtype_arg: str) -> torch.dtype | None:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "auto":
        if torch.cuda.is_available():
            return torch.float16
        return None
    raise ValueError(f"Unknown dtype: {dtype_arg}")


def apply_inference_dtype(pipeline, dtype: torch.dtype | None) -> None:
    if dtype is None:
        print("Inference dtype: model default")
        return
    print(f"Inference dtype override: {dtype}")
    for name, model in pipeline.models.items():
        if hasattr(model, "convert_to") and hasattr(model, "dtype"):
            current = getattr(model, "dtype")
            if current != dtype:
                print(f"  {name}: {current} -> {dtype}")
                model.convert_to(dtype)


def has_transparency(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False
    alpha = image.getchannel("A")
    return alpha.getextrema()[0] < 255


def remove_near_white_background(image: Image.Image, threshold: int) -> Image.Image:
    image = image.convert("RGBA")
    pixels = image.load()
    width, height = image.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if r >= threshold and g >= threshold and b >= threshold:
                pixels[x, y] = (255, 255, 255, 0)
            else:
                pixels[x, y] = (r, g, b, a)
    return image


def run_pipeline_staged(pipeline, image: Image.Image, args, ss_sampler_params, shape_sampler_params, tex_sampler_params, preprocess_image: bool):
    stage_paths: dict[str, str] = {}
    pipeline_type = args.pipeline_type
    if pipeline_type == '512':
        assert 'shape_slat_flow_model_512' in pipeline.models, "No 512 resolution shape SLat flow model found."
        assert 'tex_slat_flow_model_512' in pipeline.models, "No 512 resolution texture SLat flow model found."
    elif pipeline_type == '1024':
        assert 'shape_slat_flow_model_1024' in pipeline.models, "No 1024 resolution shape SLat flow model found."
        assert 'tex_slat_flow_model_1024' in pipeline.models, "No 1024 resolution texture SLat flow model found."
    elif pipeline_type in ['1024_cascade', '1536_cascade']:
        assert 'shape_slat_flow_model_512' in pipeline.models, "No 512 resolution shape SLat flow model found."
        assert 'shape_slat_flow_model_1024' in pipeline.models, "No 1024 resolution shape SLat flow model found."
        assert 'tex_slat_flow_model_1024' in pipeline.models, "No 1024 resolution texture SLat flow model found."
    else:
        raise ValueError(f"Invalid pipeline type: {pipeline_type}")

    if preprocess_image:
        image = pipeline.preprocess_image(image)

    torch.manual_seed(args.seed)
    cond_512 = pipeline.get_cond([image], 512)
    cond_1024 = pipeline.get_cond([image], 1024) if pipeline_type != '512' else None

    ss_res = args.sparse_structure_resolution or {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
    coords = pipeline.sample_sparse_structure(cond_512, ss_res, 1, ss_sampler_params)
    print(f"Stage 1 sparse structure: voxels={coords.shape[0]}, resolution={ss_res}", flush=True)
    if needs_stage_artifact(args, "stage1"):
        path = artifact_path(args.out, "stage1_sparse")
        export_sparse_structure(coords, ss_res, path, args.stage1_max_voxels)
        stage_paths["stage1"] = path
        print_artifact("stage1_sparse", path)
    if args.stop_after == "stage1":
        return None, stage_paths

    if pipeline_type == '512':
        shape_slat = pipeline.sample_shape_slat(
            cond_512, pipeline.models['shape_slat_flow_model_512'],
            coords, shape_sampler_params
        )
        tex_cond = cond_512
        tex_model = pipeline.models['tex_slat_flow_model_512']
        res = 512
    elif pipeline_type == '1024':
        shape_slat = pipeline.sample_shape_slat(
            cond_1024, pipeline.models['shape_slat_flow_model_1024'],
            coords, shape_sampler_params
        )
        tex_cond = cond_1024
        tex_model = pipeline.models['tex_slat_flow_model_1024']
        res = 1024
    elif pipeline_type == '1024_cascade':
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            pipeline.models['shape_slat_flow_model_512'], pipeline.models['shape_slat_flow_model_1024'],
            512, 1024,
            coords, shape_sampler_params,
            args.max_num_tokens
        )
        tex_cond = cond_1024
        tex_model = pipeline.models['tex_slat_flow_model_1024']
    else:
        shape_slat, res = pipeline.sample_shape_slat_cascade(
            cond_512, cond_1024,
            pipeline.models['shape_slat_flow_model_512'], pipeline.models['shape_slat_flow_model_1024'],
            512, 1536,
            coords, shape_sampler_params,
            args.max_num_tokens
        )
        tex_cond = cond_1024
        tex_model = pipeline.models['tex_slat_flow_model_1024']

    shape_meshes = None
    if needs_stage_artifact(args, "stage2") or args.ultrashape_mode == "before-texture":
        shape_meshes, _ = pipeline.decode_shape_slat(shape_slat, res)
    if needs_stage_artifact(args, "stage2"):
        path = artifact_path(args.out, "stage2_shape")
        export_basic_mesh(shape_meshes[0], path, color=(220, 220, 220, 255))
        stage_paths["stage2"] = path
        print_artifact("stage2_shape", path)
        torch.cuda.empty_cache()
    if args.stop_after == "stage2":
        return None, stage_paths

    if args.ultrashape_mode == "before-texture":
        assert shape_meshes is not None
        ultra_input_mesh = shape_meshes[0]
        if args.ultrashape_input_simplify_target > 0:
            print(f"Simplifying Stage 2 mesh for UltraShape input: target={args.ultrashape_input_simplify_target}", flush=True)
            ultra_input_mesh.simplify(args.ultrashape_input_simplify_target)
        ultra_input_path = artifact_path(args.out, "stage2_ultrashape_input")
        export_basic_mesh(ultra_input_mesh, ultra_input_path, color=(225, 225, 225, 255))
        stage_paths["stage2_ultrashape_input"] = ultra_input_path
        print_artifact("stage2_ultrashape_input", ultra_input_path)

        pipeline.cpu()
        torch.cuda.empty_cache()
        ultra_path = run_ultrashape_module(args, args.image, ultra_input_path, "ultrashape_before_texture")
        stage_paths["ultrashape_before_texture"] = ultra_path
        if args.stop_after == "ultrashape":
            return None, stage_paths
        torch.cuda.empty_cache()
        textured_mesh = texture_trimesh_with_trellis(args, image, ultra_path, tex_sampler_params, res)
        return [textured_mesh], stage_paths

    tex_slat = pipeline.sample_tex_slat(tex_cond, tex_model, shape_slat, tex_sampler_params)
    torch.cuda.empty_cache()
    meshes = pipeline.decode_latent(
        shape_slat,
        tex_slat,
        res,
        fill_holes=args.fill_holes,
        max_hole_perimeter=args.max_hole_perimeter,
    )
    if needs_stage_artifact(args, "stage3"):
        path = artifact_path(args.out, "stage3_decoded")
        export_basic_mesh(meshes[0], path, color=(205, 205, 205, 255))
        stage_paths["stage3"] = path
        print_artifact("stage3_decoded", path)
    if args.stop_after == "stage3":
        return None, stage_paths
    return meshes, stage_paths


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    image_path = args.image
    image = Image.open(image_path).convert("RGBA")
    preprocess_image = True
    if args.background == "local-white":
        image = remove_near_white_background(image, args.white_threshold)
        rembg.BiRefNet = _DisabledRembg
    elif args.background == "keep":
        rembg.BiRefNet = _DisabledRembg
        preprocess_image = has_transparency(image)
    elif args.background == "trellis":
        os.environ["TRELLIS_REMBG_MODEL"] = args.rembg_model

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    apply_inference_dtype(pipeline, resolve_inference_dtype(args.inference_dtype))
    pipeline.cuda()

    if args.steps is not None:
        args.ss_steps = args.steps
        args.shape_steps = args.steps
        args.tex_steps = args.steps

    ss_sampler_params = {
        "steps": args.ss_steps,
        "guidance_strength": args.ss_guidance_strength,
        "guidance_rescale": args.ss_guidance_rescale,
        "rescale_t": args.ss_rescale_t,
        "guidance_interval": (args.ss_guidance_interval_start, args.ss_guidance_interval_end),
    }
    shape_sampler_params = {
        "steps": args.shape_steps,
        "guidance_strength": args.shape_guidance_strength,
        "guidance_rescale": args.shape_guidance_rescale,
        "rescale_t": args.shape_rescale_t,
        "guidance_interval": (args.shape_guidance_interval_start, args.shape_guidance_interval_end),
    }
    tex_sampler_params = {
        "steps": args.tex_steps,
        "guidance_strength": args.tex_guidance_strength,
        "guidance_rescale": args.tex_guidance_rescale,
        "rescale_t": args.tex_rescale_t,
        "guidance_interval": (args.tex_guidance_interval_start, args.tex_guidance_interval_end),
    }
    meshes, stage_paths = run_pipeline_staged(
        pipeline,
        image,
        args,
        ss_sampler_params,
        shape_sampler_params,
        tex_sampler_params,
        preprocess_image,
    )
    if meshes is None:
        print(f"Stopped after {args.stop_after}.", flush=True)
        del pipeline
        torch.cuda.empty_cache()
        run_optional_modules(args, image_path, stage_paths, None)
        return

    mesh = meshes[0]
    if isinstance(mesh, trimesh.Trimesh):
        print(f"Generated textured trimesh: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
        if args.no_export:
            del pipeline
            del meshes
            del mesh
            torch.cuda.empty_cache()
            run_optional_modules(args, image_path, stage_paths, None)
            return
        mesh.export(args.out)
        print(f"Wrote {args.out}")
        del pipeline
        del meshes
        del mesh
        torch.cuda.empty_cache()
        run_optional_modules(args, image_path, stage_paths, args.out)
        return

    print(f"Generated mesh: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
    if args.no_export:
        del pipeline
        del meshes
        del mesh
        torch.cuda.empty_cache()
        run_optional_modules(args, image_path, stage_paths, None)
        return

    if args.pre_simplify_target > 0:
        mesh.simplify(args.pre_simplify_target)
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=args.export_remesh,
        remesh_band=args.remesh_band,
        remesh_project=args.remesh_project,
        verbose=True,
    )
    glb.export(args.out, extension_webp=True)
    print(f"Wrote {args.out}")
    del pipeline
    del meshes
    del mesh
    torch.cuda.empty_cache()
    run_optional_modules(args, image_path, stage_paths, args.out)


if __name__ == "__main__":
    main()
