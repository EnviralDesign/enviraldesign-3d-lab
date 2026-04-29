import argparse
import math
import os
import sys

venv_scripts = os.path.dirname(sys.executable)
os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
import trimesh


def load_mesh(path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            raise RuntimeError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Could not load a mesh from {path}")
    mesh = mesh.copy()
    valid = mesh.nondegenerate_faces()
    if valid.sum() < len(mesh.faces):
        mesh.update_faces(valid)
        mesh.remove_unreferenced_vertices()
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh, margin: float) -> trimesh.Trimesh:
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    mesh.vertices -= center
    current_half_size = np.abs(mesh.vertices).max()
    if current_half_size > 1e-8:
        mesh.vertices *= (1.0 - margin) / current_half_size
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FaithC/Faithful Contouring on a mesh artifact.")
    parser.add_argument("--mesh", required=True, help="Input GLB/OBJ/PLY mesh.")
    parser.add_argument("--out", required=True, help="Output GLB path.")
    parser.add_argument("--resolution", type=int, default=256, help="FaithC contour grid resolution. Must be a power of two.")
    parser.add_argument("--margin", type=float, default=0.05, help="Normalization margin when --normalize is enabled.")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tri-mode", default="auto", choices=["auto", "simple_02", "simple_13", "length", "angle", "normal", "normal_abs"])
    parser.add_argument("--clamp-anchors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute-flux", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.resolution <= 0 or (args.resolution & (args.resolution - 1)) != 0:
        raise ValueError("--resolution must be a power of two")
    if not torch.cuda.is_available():
        raise RuntimeError("FaithC integration requires CUDA.")

    from atom3d import MeshBVH
    from atom3d.grid import OctreeIndexer
    from faithcontour import FCTEncoder, FCTDecoder

    mesh = load_mesh(args.mesh)
    if args.normalize:
        mesh = normalize_mesh(mesh, args.margin)

    print(f"FaithC input: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}", flush=True)
    print(f"FaithC resolution: {args.resolution}, tri_mode={args.tri_mode}", flush=True)

    device = "cuda"
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)

    max_level = int(math.log2(args.resolution))
    min_level = min(4, max(1, max_level - 1))
    bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device)

    bvh = MeshBVH(vertices, faces)
    octree = OctreeIndexer(max_level=max_level, bounds=bounds, device=device)
    encoder = FCTEncoder(bvh, octree, device=device)
    fct = encoder.encode(
        min_level=min_level,
        solver_weights={"lambda_n": 1.0, "lambda_d": 1e-3, "weight_power": 1},
        compute_flux=args.compute_flux,
        clamp_anchors=args.clamp_anchors,
    )
    print(f"FaithC active voxels: {fct.active_voxel_indices.shape[0]}", flush=True)
    if fct.active_voxel_indices.shape[0] == 0:
        raise RuntimeError("FaithC produced zero active voxels. Try enabling --faithc-normalize or using a higher resolution.")

    decoder = FCTDecoder(resolution=args.resolution, bounds=bounds, device=device)
    decoded = decoder.decode(
        active_voxel_indices=fct.active_voxel_indices,
        anchors=fct.anchor,
        edge_flux_sign=fct.edge_flux_sign,
        normals=fct.normal,
        triangulation_mode=args.tri_mode,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_mesh = trimesh.Trimesh(
        vertices=decoded.vertices.detach().cpu().numpy(),
        faces=decoded.faces.detach().cpu().numpy(),
        process=False,
    )
    out_mesh.export(args.out)
    print(f"FaithC wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
