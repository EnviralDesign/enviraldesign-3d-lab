"""Bootstrap helpers for remote GPU deployments.

These commands intentionally mutate the local virtual environment. They are
meant for disposable GPU nodes where the base lockfile is close, but the node
needs a platform-specific torch/native-extension setup.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def ensure_venv() -> None:
    if not VENV_PYTHON.exists():
        raise SystemExit(
            f"Missing venv Python at {VENV_PYTHON}. Run `uv sync` first."
        )


def install_blackwell_torch(args: argparse.Namespace) -> None:
    """Install a Blackwell-capable torch stack into the existing venv."""
    ensure_venv()
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(VENV_PYTHON),
            "--index-url",
            args.torch_index,
            f"torch=={args.torch}",
            f"torchvision=={args.torchvision}",
            f"torchaudio=={args.torchaudio}",
        ]
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(VENV_PYTHON),
            f"xformers=={args.xformers}",
        ]
    )
    smoke_torch()


def build_native(args: argparse.Namespace) -> None:
    """Build TRELLIS native dependencies through upstream setup.sh."""
    ensure_venv()
    if platform.system() != "Linux":
        raise SystemExit("Native bootstrap currently expects Linux.")
    # uv-created venvs do not necessarily contain a `pip` executable. The
    # upstream setup.sh calls `pip` directly, so install it into this venv
    # before activating; otherwise Linux images with a base Conda install can
    # accidentally build extensions against the wrong Python/CUDA stack.
    run(["uv", "pip", "install", "--python", str(VENV_PYTHON), "pip"])

    components = args.components
    setup_flags = [f"--{component}" for component in components]
    env = os.environ.copy()
    env.update(
        {
            "TORCH_CUDA_ARCH_LIST": args.cuda_arch,
            "FORCE_CUDA": "1",
            "MAX_JOBS": str(args.max_jobs),
        }
    )

    if args.clean_tmp:
        for name in ("nvdiffrast", "nvdiffrec", "CuMesh", "FlexGEMM", "o-voxel"):
            subprocess.run(["rm", "-rf", f"/tmp/extensions/{name}"], check=False)

    shell = (
        "source .venv/bin/activate && "
        f"bash setup.sh {' '.join(setup_flags)}"
    )
    run(["bash", "-lc", shell], env=env)


def smoke_torch() -> None:
    ensure_venv()
    code = """
import torch
print("torch", torch.__version__, torch.version.cuda)
print("arch", torch.cuda.get_arch_list() if torch.cuda.is_available() else None)
if torch.cuda.is_available():
    x = torch.ones((2, 2), device="cuda")
    print("cuda-matmul", (x @ x).cpu().tolist())
"""
    run([str(VENV_PYTHON), "-c", code])


def smoke_imports() -> None:
    ensure_venv()
    modules = repr([
        "xformers.ops",
        "nvdiffrast.torch",
        "nvdiffrec_render.light",
        "cumesh",
        "o_voxel",
        "flex_gemm",
        "trellis2.pipelines.trellis2_image_to_3d",
        "tools.remote_launcher",
    ])
    code = f"""
import importlib
failed = []
for module in {modules}:
    try:
        importlib.import_module(module)
        print(f"{{module}} OK")
    except Exception as exc:
        failed.append(module)
        print(f"{{module}} FAIL {{type(exc).__name__}}: {{exc}}")
if failed:
    raise SystemExit("Import smoke failed: " + ", ".join(failed))
"""
    run([str(VENV_PYTHON), "-c", code])


def smoke(args: argparse.Namespace | None = None) -> None:
    smoke_torch()
    smoke_imports()


def all_lightning(args: argparse.Namespace) -> None:
    install_blackwell_torch(args)
    build_native(args)
    smoke(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deployment bootstrap helpers.")
    sub = parser.add_subparsers(dest="command", required=True)

    blackwell = sub.add_parser("blackwell-torch", help="Install Blackwell torch/xformers.")
    add_blackwell_args(blackwell)
    blackwell.set_defaults(func=install_blackwell_torch)

    native = sub.add_parser("native", help="Build native TRELLIS extension deps.")
    add_native_args(native)
    native.set_defaults(func=build_native)

    smoke_parser = sub.add_parser("smoke", help="Run torch and import smoke tests.")
    smoke_parser.set_defaults(func=smoke)

    all_parser = sub.add_parser("lightning-all", help="Blackwell torch + native build + smoke.")
    add_blackwell_args(all_parser)
    add_native_args(all_parser)
    all_parser.set_defaults(func=all_lightning)

    return parser


def add_blackwell_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--torch-index", default="https://download.pytorch.org/whl/cu130")
    parser.add_argument("--torch", default="2.11.0+cu130")
    parser.add_argument("--torchvision", default="0.26.0+cu130")
    parser.add_argument("--torchaudio", default="2.11.0+cu130")
    parser.add_argument("--xformers", default="0.0.35")


def add_native_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cuda-arch", default="12.0", help="Blackwell RTX PRO 6000 is sm_120.")
    parser.add_argument("--max-jobs", type=int, default=8)
    parser.add_argument(
        "--no-clean-tmp",
        action="store_false",
        dest="clean_tmp",
        default=True,
        help="Keep /tmp/extensions sources between native build attempts.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["nvdiffrast", "nvdiffrec", "cumesh", "o-voxel", "flexgemm"],
        choices=["basic", "nvdiffrast", "nvdiffrec", "cumesh", "o-voxel", "flexgemm"],
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def smoke_main() -> None:
    smoke()


if __name__ == "__main__":
    main()
