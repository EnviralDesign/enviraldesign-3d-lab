import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]


def parse_triplet(value: str, name: str) -> np.ndarray:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"{name} must be three comma-separated numbers")
    try:
        return np.array([float(part) for part in parts], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} contains a non-numeric value") from exc


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", merge_primitives=True)
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Expected mesh from {path}, got {type(mesh).__name__}")
    if len(mesh.faces) == 0:
        raise RuntimeError(f"Mesh has no faces: {path}")
    return mesh


def print_artifact(label: str, path: Path) -> None:
    print(f"ROI_ARTIFACT\t{label}\t{path.resolve()}", flush=True)


def export_mesh(label: str, mesh: trimesh.Trimesh, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    print_artifact(label, path)
    return path


def normalized_roi_to_bounds(mesh: trimesh.Trimesh, center: np.ndarray, size: np.ndarray) -> np.ndarray:
    center = np.clip(center, 0.0, 1.0)
    size = np.clip(size, 0.001, 1.0)
    mesh_min, mesh_max = mesh.bounds
    extents = np.maximum(mesh_max - mesh_min, 1e-8)
    roi_center = mesh_min + center * extents
    roi_size = size * extents
    roi_min = np.maximum(mesh_min, roi_center - roi_size * 0.5)
    roi_max = np.minimum(mesh_max, roi_center + roi_size * 0.5)
    return np.array([roi_min, roi_max], dtype=np.float64)


def expand_bounds(bounds: np.ndarray, multiplier: float, clamp: np.ndarray) -> np.ndarray:
    center = bounds.mean(axis=0)
    size = (bounds[1] - bounds[0]) * max(multiplier, 1.0)
    return np.array(
        [
            np.maximum(clamp[0], center - size * 0.5),
            np.minimum(clamp[1], center + size * 0.5),
        ],
        dtype=np.float64,
    )


def face_mask(mesh: trimesh.Trimesh, bounds: np.ndarray) -> np.ndarray:
    triangles = mesh.triangles
    tri_min = triangles.min(axis=1)
    tri_max = triangles.max(axis=1)
    intersects_box = np.all((tri_max >= bounds[0]) & (tri_min <= bounds[1]), axis=1)
    centers = mesh.triangles_center
    centers_inside = np.all((centers >= bounds[0]) & (centers <= bounds[1]), axis=1)
    return intersects_box | centers_inside


def submesh_from_mask(mesh: trimesh.Trimesh, mask: np.ndarray, fallback_center: np.ndarray, fallback_count: int) -> trimesh.Trimesh:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        distances = np.linalg.norm(mesh.triangles_center - fallback_center[None, :], axis=1)
        count = min(max(fallback_count, 1), len(distances))
        indices = np.argsort(distances)[:count]
        print(f"ROI box matched no faces; using {len(indices)} nearest faces instead.", flush=True)
    parts = mesh.submesh([indices], append=True, repair=False)
    if isinstance(parts, list):
        parts = trimesh.util.concatenate(parts)
    return parts


def colored_copy(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> trimesh.Trimesh:
    copy = mesh.copy()
    copy.visual = trimesh.visual.ColorVisuals(copy, vertex_colors=np.tile(np.array(color, dtype=np.uint8), (len(copy.vertices), 1)))
    return copy


def make_overlay_mesh(source: trimesh.Trimesh, roi: trimesh.Trimesh) -> trimesh.Trimesh:
    base = colored_copy(source, (135, 140, 145, 80))
    roi_colored = colored_copy(roi, (255, 150, 40, 255))
    return trimesh.util.concatenate([base, roi_colored])


def transform_ultrashape_output_to_world(mesh: trimesh.Trimesh, source_bounds: np.ndarray, normalize_scale: float) -> trimesh.Trimesh:
    world = mesh.copy()
    center = source_bounds.mean(axis=0)
    longest_side = float(np.max(source_bounds[1] - source_bounds[0]))
    if longest_side <= 0:
        raise RuntimeError("Cannot inverse-transform UltraShape output for a zero-size ROI.")
    world.apply_scale(longest_side / (2.0 * normalize_scale))
    world.apply_translation(center)
    return world


def merge_refined_patch(source: trimesh.Trimesh, refined_patch: trimesh.Trimesh, remove_mask: np.ndarray) -> trimesh.Trimesh:
    keep_indices = np.flatnonzero(~remove_mask)
    base = source.submesh([keep_indices], append=True, repair=False)
    if isinstance(base, list):
        base = trimesh.util.concatenate(base)
    base = colored_copy(base, (185, 185, 185, 255))
    patch = colored_copy(refined_patch, (255, 210, 120, 255))
    return trimesh.util.concatenate([base, patch])


def run_ultrashape(args: argparse.Namespace, roi_path: Path, work_dir: Path) -> Path:
    config = Path(args.ultrashape_config).resolve()
    ckpt = Path(args.ultrashape_ckpt).resolve()
    image = Path(args.image).resolve()
    if not config.exists():
        raise FileNotFoundError(f"UltraShape config not found: {config}")
    if not ckpt.exists():
        raise FileNotFoundError(f"UltraShape checkpoint not found: {ckpt}")
    if not image.exists():
        raise FileNotFoundError(f"UltraShape image not found: {image}")

    ultra_root = ROOT / "integrations" / "UltraShape-1.0"
    command = [
        sys.executable,
        str(Path("scripts") / "infer_dit_refine.py"),
        "--config",
        str(config),
        "--ckpt",
        str(ckpt),
        "--image",
        str(image),
        "--mesh",
        str(roi_path.resolve()),
        "--output_dir",
        str(work_dir.resolve()),
        "--steps",
        str(args.ultrashape_steps),
        "--scale",
        str(args.ultrashape_scale),
        "--num_latents",
        str(args.ultrashape_num_latents),
        "--chunk_size",
        str(args.ultrashape_chunk_size),
        "--octree_res",
        str(args.ultrashape_octree_res),
        "--seed",
        str(args.seed),
    ]
    if args.ultrashape_low_vram:
        command.append("--low_vram")
    if args.ultrashape_remove_bg:
        command.append("--remove_bg")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ultra_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    cuda_124 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    if os.path.exists(cuda_124):
        env.setdefault("CUDA_HOME", cuda_124)
        env.setdefault("CUDA_PATH", cuda_124)
        env["PATH"] = os.path.join(cuda_124, "bin") + os.pathsep + env.get("PATH", "")

    print("MODULE_START\troi_ultrashape", flush=True)
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ultra_root, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"roi_ultrashape failed with exit code {completed.returncode}")

    candidates = sorted(work_dir.glob("*.glb"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"UltraShape completed but no GLB was found in {work_dir}")
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental mesh-space ROI zoom/enhance sidecar.")
    parser.add_argument("--mesh", required=True, help="Source GLB/OBJ mesh artifact.")
    parser.add_argument("--out-dir", required=True, help="Directory for ROI artifacts.")
    parser.add_argument("--label", default="roi", help="Filename label for generated artifacts.")
    parser.add_argument("--mode", choices=["inspect", "ultrashape"], default="inspect")
    parser.add_argument("--roi-center", type=lambda v: parse_triplet(v, "roi-center"), default="0.5,0.82,0.5", help="Normalized center in source mesh bounds, x,y,z.")
    parser.add_argument("--roi-size", type=lambda v: parse_triplet(v, "roi-size"), default="0.45,0.35,0.45", help="Normalized size in source mesh bounds, x,y,z.")
    parser.add_argument("--context-multiplier", type=float, default=1.35, help="Expanded context box multiplier.")
    parser.add_argument("--fallback-faces", type=int, default=2000, help="Nearest faces to use if the ROI box is empty.")
    parser.add_argument("--image", default=None, help="Original conditioning image for UltraShape mode.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ultrashape-config", default=str(ROOT / "integrations" / "UltraShape-1.0" / "configs" / "infer_dit_refine.yaml"))
    parser.add_argument("--ultrashape-ckpt", default=str(ROOT / "integrations" / "UltraShape-1.0" / "checkpoints" / "ultrashape_v1.pt"))
    parser.add_argument("--ultrashape-steps", type=int, default=25)
    parser.add_argument("--ultrashape-scale", type=float, default=0.99)
    parser.add_argument("--ultrashape-num-latents", type=int, default=8192)
    parser.add_argument("--ultrashape-chunk-size", type=int, default=2048)
    parser.add_argument("--ultrashape-octree-res", type=int, default=512)
    parser.add_argument("--ultrashape-low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ultrashape-remove-bg", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    label = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in args.label).strip("_") or "roi"

    source_path = Path(args.mesh).resolve()
    source = load_mesh(str(source_path))
    roi_bounds = normalized_roi_to_bounds(source, args.roi_center, args.roi_size)
    context_bounds = expand_bounds(roi_bounds, args.context_multiplier, source.bounds)
    roi_center_world = roi_bounds.mean(axis=0)
    roi_mask = face_mask(source, roi_bounds)
    context_mask = face_mask(source, context_bounds)
    roi = submesh_from_mask(source, roi_mask, roi_center_world, args.fallback_faces)
    context = submesh_from_mask(source, context_mask, roi_center_world, max(args.fallback_faces * 2, 1))

    print(f"ROI source: {source_path}", flush=True)
    print(f"Source mesh: vertices={len(source.vertices)}, faces={len(source.faces)}", flush=True)
    print(f"ROI faces: {int(roi_mask.sum())}, context faces: {int(context_mask.sum())}", flush=True)
    print(f"ROI bounds: {roi_bounds.tolist()}", flush=True)

    roi_path = export_mesh("roi_input", roi, out_dir / f"{label}_roi_input.glb")
    export_mesh("roi_context", context, out_dir / f"{label}_roi_context.glb")
    export_mesh("roi_overlay", make_overlay_mesh(source, roi), out_dir / f"{label}_roi_overlay.glb")

    metadata = {
        "source": str(source_path),
        "mode": args.mode,
        "roi_center_normalized": args.roi_center.tolist(),
        "roi_size_normalized": args.roi_size.tolist(),
        "roi_bounds": roi_bounds.tolist(),
        "context_bounds": context_bounds.tolist(),
        "source_vertices": int(len(source.vertices)),
        "source_faces": int(len(source.faces)),
        "roi_faces": int(roi_mask.sum()),
        "context_faces": int(context_mask.sum()),
    }

    if args.mode == "ultrashape":
        if not args.image:
            raise ValueError("--image is required for --mode ultrashape")
        work_dir = out_dir / f"{label}_ultrashape_work"
        refined_normalized = run_ultrashape(args, roi_path, work_dir)
        normalized_out = out_dir / f"{label}_ultrashape_normalized.glb"
        shutil.copy2(refined_normalized, normalized_out)
        print_artifact("roi_ultrashape_normalized", normalized_out)

        refined = load_mesh(str(refined_normalized))
        world = transform_ultrashape_output_to_world(refined, roi.bounds, args.ultrashape_scale)
        world_path = export_mesh("roi_ultrashape_world", world, out_dir / f"{label}_ultrashape_world.glb")
        merged = merge_refined_patch(source, world, roi_mask)
        export_mesh("roi_ultrashape_merged_preview", merged, out_dir / f"{label}_ultrashape_merged_preview.glb")
        metadata["ultrashape_world"] = str(world_path)

    metadata_path = out_dir / f"{label}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"ROI_METADATA\t{metadata_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
