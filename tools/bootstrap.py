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
DEFAULT_DINOV3_MODEL = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_REMBG_MODEL = "camenduru/RMBG-2.0"
DEFAULT_REMBG_ALLOW = "onnx/model_quantized.onnx"
DEFAULT_ULTRASHAPE_MODEL = "infinith/UltraShape"
DEFAULT_ULTRASHAPE_FILE = "ultrashape_v1.pt"
DEFAULT_ULTRASHAPE_DIR = ROOT / "integrations" / "UltraShape-1.0" / "checkpoints"
NATIVE_BUILD_ROOT = ROOT / ".native-build"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def ensure_venv() -> None:
    if not VENV_PYTHON.exists():
        raise SystemExit(
            f"Missing venv Python at {VENV_PYTHON}. Run `uv sync` first."
        )


def remove_conflicting_flash_attention() -> None:
    """Remove FlashAttention packages that can shadow modules xFormers imports."""
    ensure_venv()
    print("+ uv pip uninstall --python", VENV_PYTHON, "flash-attn flash-attn-4", flush=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "uninstall",
            "--python",
            str(VENV_PYTHON),
            "flash-attn",
            "flash-attn-4",
        ],
        cwd=ROOT,
        check=False,
    )


def uninstall_native_extensions() -> None:
    """Remove native extensions before rebuilding against the active Torch ABI."""
    ensure_venv()
    packages = [
        "cumesh",
        "o-voxel",
        "flex-gemm",
        "nvdiffrast",
        "nvdiffrec-render",
    ]
    print("+ uv pip uninstall --python", VENV_PYTHON, " ".join(packages), flush=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "uninstall",
            "--python",
            str(VENV_PYTHON),
            *packages,
        ],
        cwd=ROOT,
        check=False,
    )


def clone_or_update(repo: str, dest: Path, *, branch: str | None = None, recursive: bool = False) -> None:
    if dest.exists():
        run(["git", "-C", str(dest), "fetch", "--all", "--tags"])
        run(["git", "-C", str(dest), "pull", "--ff-only"])
        return

    cmd = ["git", "clone"]
    if branch:
        cmd.extend(["-b", branch])
    if recursive:
        cmd.append("--recursive")
    cmd.extend([repo, str(dest)])
    run(cmd)


def windows_native_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    cuda_home = Path(args.cuda_home)
    if cuda_home.exists():
        env["CUDA_HOME"] = str(cuda_home)
        env["CUDA_PATH"] = str(cuda_home)
        env["PATH"] = str(cuda_home / "bin") + os.pathsep + env.get("PATH", "")
    env["TORCH_CUDA_ARCH_LIST"] = args.cuda_arch
    env["FORCE_CUDA"] = "1"
    env["MAX_JOBS"] = str(args.max_jobs)
    return env


def install_path_no_isolation(path: Path, *, env: dict[str, str], no_deps: bool = False) -> None:
    cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        str(VENV_PYTHON),
        "--no-build-isolation",
        "--force-reinstall",
    ]
    if no_deps:
        cmd.append("--no-deps")
    cmd.append(str(path))
    run(
        cmd,
        env=env,
    )


def patch_flexgemm_windows_source(source_root: Path) -> None:
    path = source_root / "flex_gemm" / "kernels" / "cuda" / "spconv" / "sparse_neighbor_map.cu"
    text = path.read_text()
    patched = text.replace("expanded_keys.data_ptr<T>()", "expanded_keys.template data_ptr<T>()")
    patched = patched.replace("valid_keys.data_ptr<T>()", "valid_keys.template data_ptr<T>()")
    if patched != text:
        path.write_text(patched)


def build_windows_native(args: argparse.Namespace) -> None:
    """Build native TRELLIS dependencies from source on Windows."""
    ensure_venv()
    if os.name != "nt":
        raise SystemExit("windows-native expects Windows. Use `uv run bootstrap` on Linux/Lightning.")

    NATIVE_BUILD_ROOT.mkdir(exist_ok=True)
    env = windows_native_env(args)
    uninstall_native_extensions()

    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(VENV_PYTHON),
            "plyfile>=1.1.3",
            "triton-windows==3.2.0.post21",
            "pip",
        ]
    )

    clone_or_update("https://github.com/NVlabs/nvdiffrast.git", NATIVE_BUILD_ROOT / "nvdiffrast", branch="v0.4.0")
    install_path_no_isolation(NATIVE_BUILD_ROOT / "nvdiffrast", env=env)

    clone_or_update("https://github.com/JeffreyXiang/nvdiffrec.git", NATIVE_BUILD_ROOT / "nvdiffrec", branch="renderutils")
    install_path_no_isolation(NATIVE_BUILD_ROOT / "nvdiffrec", env=env)

    clone_or_update("https://github.com/JeffreyXiang/CuMesh.git", NATIVE_BUILD_ROOT / "CuMesh", recursive=True)
    install_path_no_isolation(NATIVE_BUILD_ROOT / "CuMesh", env=env)

    clone_or_update("https://github.com/JeffreyXiang/FlexGEMM.git", NATIVE_BUILD_ROOT / "FlexGEMM", recursive=True)
    patch_flexgemm_windows_source(NATIVE_BUILD_ROOT / "FlexGEMM")
    install_path_no_isolation(NATIVE_BUILD_ROOT / "FlexGEMM", env=env)

    install_path_no_isolation(ROOT / "o-voxel", env=env, no_deps=True)

    prefetch_aux_models(args)
    download_ultrashape(args)
    smoke_imports()


def install_blackwell_torch(args: argparse.Namespace) -> None:
    """Install a Blackwell-capable torch stack into the existing venv."""
    ensure_venv()
    remove_conflicting_flash_attention()
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


def install_flash_attention(args: argparse.Namespace) -> None:
    """Install an experimental dense FlashAttention package for this GPU node."""
    ensure_venv()
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(VENV_PYTHON),
            "--prerelease",
            "allow",
            args.flash_attention,
        ]
    )
    code = """
from flash_attn.cute import flash_attn_func
print("flash-attn-4 OK", flash_attn_func)
"""
    run([str(VENV_PYTHON), "-c", code])


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
    uninstall_native_extensions()

    components = args.components
    setup_flags = [f"--{component}" for component in components]
    env = os.environ.copy()
    env.update(
        {
            "TORCH_CUDA_ARCH_LIST": args.cuda_arch,
            "FORCE_CUDA": "1",
            "MAX_JOBS": str(args.max_jobs),
            "PIP_NO_CACHE_DIR": "1",
        }
    )

    if args.clean_tmp:
        for name in ("nvdiffrast", "nvdiffrec", "CuMesh", "FlexGEMM", "o-voxel"):
            subprocess.run(["rm", "-rf", f"/tmp/extensions/{name}"], check=False)

    shell = (
        "source .venv/bin/activate && "
        f"PIP_NO_CACHE_DIR=1 bash setup.sh {' '.join(setup_flags)}"
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


def prefetch_aux_models(args: argparse.Namespace) -> None:
    """Download ungated auxiliary model files used by image generation."""
    ensure_venv()
    code = """
from huggingface_hub import hf_hub_download, snapshot_download

dinov3_model = {dinov3_model!r}
rembg_model = {rembg_model!r}
rembg_allow = {rembg_allow!r}

print(f"Downloading DINOv3 mirror: {{dinov3_model}}", flush=True)
snapshot_download(
    dinov3_model,
    allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
)

print(f"Downloading RMBG mirror: {{rembg_model}} ({rembg_allow})", flush=True)
hf_hub_download(rembg_model, rembg_allow)
""".format(
        dinov3_model=args.dinov3_model,
        rembg_model=args.rembg_model,
        rembg_allow=args.rembg_allow,
    )
    run([str(VENV_PYTHON), "-c", code])


def download_ultrashape(args: argparse.Namespace) -> None:
    """Download the UltraShape checkpoint into the expected local path."""
    ensure_venv()
    local_dir = Path(args.local_dir)
    if not local_dir.is_absolute():
        local_dir = ROOT / local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    code = """
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = {repo_id!r}
filename = {filename!r}
local_dir = {local_dir!r}
path = hf_hub_download(repo_id, filename, local_dir=local_dir)
print(f"UltraShape checkpoint: {{path}}")
""".format(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=str(local_dir),
    )
    run([str(VENV_PYTHON), "-c", code])


def smoke(args: argparse.Namespace | None = None) -> None:
    smoke_torch()
    smoke_imports()


def all_lightning(args: argparse.Namespace) -> None:
    install_blackwell_torch(args)
    build_native(args)
    prefetch_aux_models(args)
    download_ultrashape(args)
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

    windows_native = sub.add_parser("windows-native", help="Build native TRELLIS dependencies from source on Windows.")
    add_windows_native_args(windows_native)
    add_hf_model_args(windows_native)
    add_ultrashape_args(windows_native)
    windows_native.set_defaults(func=build_windows_native)

    flash = sub.add_parser("flash-attention", help="Install experimental dense FlashAttention package.")
    add_flash_attention_args(flash)
    flash.set_defaults(func=install_flash_attention)

    smoke_parser = sub.add_parser("smoke", help="Run torch and import smoke tests.")
    smoke_parser.set_defaults(func=smoke)

    hf_parser = sub.add_parser("hf-models", help="Download ungated auxiliary model mirrors.")
    add_hf_model_args(hf_parser)
    hf_parser.set_defaults(func=prefetch_aux_models)

    ultra_parser = sub.add_parser("ultrashape-weights", help="Download UltraShape checkpoint.")
    add_ultrashape_args(ultra_parser)
    ultra_parser.set_defaults(func=download_ultrashape)

    all_parser = sub.add_parser("lightning-all", help="Blackwell torch + native build + smoke.")
    add_blackwell_args(all_parser)
    add_native_args(all_parser)
    add_hf_model_args(all_parser)
    add_ultrashape_args(all_parser)
    all_parser.set_defaults(func=all_lightning)

    return parser


def add_blackwell_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--torch-index", default="https://download.pytorch.org/whl/cu130")
    parser.add_argument("--torch", default="2.11.0+cu130")
    parser.add_argument("--torchvision", default="0.26.0+cu130")
    parser.add_argument("--torchaudio", default="2.11.0+cu130")
    parser.add_argument("--xformers", default="0.0.35")


def add_flash_attention_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--flash-attention",
        default="flash-attn-4[cu13]",
        help="Experimental dense attention package. Not used by default because FlashAttention-4 currently breaks xFormers imports in this stack.",
    )


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


def add_windows_native_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cuda-home",
        default=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        help="CUDA toolkit path matching the local torch CUDA stack.",
    )
    parser.add_argument(
        "--cuda-arch",
        default="7.5;8.6",
        help="Local RTX 5000 is sm_75; RTX 3070 is sm_86.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=4,
        help="Native extension build parallelism.",
    )


def add_hf_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dinov3-model",
        default=os.environ.get("TRELLIS_DINOV3_MODEL", DEFAULT_DINOV3_MODEL),
        help="Ungated DINOv3 Hugging Face repo used instead of facebook/dinov3-vitl16-pretrain-lvd1689m.",
    )
    parser.add_argument(
        "--rembg-model",
        default=DEFAULT_REMBG_MODEL,
        help="Ungated RMBG Hugging Face repo to prefetch for background removal.",
    )
    parser.add_argument(
        "--rembg-allow",
        default=DEFAULT_REMBG_ALLOW,
        help="RMBG file to prefetch from --rembg-model.",
    )


def add_ultrashape_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_ULTRASHAPE_MODEL,
        help="Hugging Face repo containing the UltraShape checkpoint.",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_ULTRASHAPE_FILE,
        help="UltraShape checkpoint filename in --repo-id.",
    )
    parser.add_argument(
        "--local-dir",
        default=str(DEFAULT_ULTRASHAPE_DIR),
        help="Directory where the UltraShape checkpoint should be stored.",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def default_main() -> None:
    parser = build_parser()
    command = "windows-native" if os.name == "nt" else "lightning-all"
    args = parser.parse_args([command, *sys.argv[1:]])
    args.func(args)


def smoke_main() -> None:
    smoke()


if __name__ == "__main__":
    main()
