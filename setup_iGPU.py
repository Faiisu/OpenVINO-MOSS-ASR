"""Install a local Ubuntu 24.04 Intel compute runtime and project Jupyter kernel."""
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from speech_to_text.runtime import igpu_environment

PROJECT_ROOT = Path(__file__).resolve().parent
# These versions target Ubuntu 24.04; newer Ubuntu 26.04 builds need newer glibc.
RELEASES = (
    ("intel/compute-runtime", "25.48.36300.8", ("libze-intel-gpu1_", "libigdgmm12_")),
    ("intel/intel-graphics-compiler", "v2.24.8", ("intel-igc-core-2_", "intel-igc-opencl-2_")),
    ("oneapi-src/level-zero", "v1.34.0", ("libze1_",)),
)


def main():
    """Extract verified driver packages locally without changing system drivers."""
    if Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve():
        raise RuntimeError("Run using uv run --no-project --python .venv setup_iGPU.py")
    root = PROJECT_ROOT / ".tools" / "intel-iGPU"
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for repository, tag, prefixes in RELEASES:
        with urllib.request.urlopen(
            f"https://api.github.com/repos/{repository}/releases/tags/{tag}"
        ) as response:
            release = json.load(response)
        for asset in release["assets"]:
            name = asset["name"]
            if not name.startswith(prefixes) or not name.endswith(".deb"):
                continue
            if name.startswith("libze1_") and "u24.04" not in name:
                continue
            destination = root / name
            if not destination.exists():
                print("Downloading", name, flush=True)
                urllib.request.urlretrieve(asset["browser_download_url"], destination)
            digest = "sha256:" + hashlib.sha256(destination.read_bytes()).hexdigest()
            if not asset.get("digest") or digest != asset["digest"]:
                raise RuntimeError(f"Missing or mismatched release checksum: {name}")
            subprocess.run(["dpkg-deb", "-x", str(destination), str(root)], check=True)
            manifest.append({"name": name, "url": asset["browser_download_url"], "digest": digest})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    subprocess.run([
        sys.executable, "-m", "ipykernel", "install", "--sys-prefix",
        "--name", "moss-igpu", "--display-name", "MOSS Intel iGPU (.venv)",
    ], check=True)
    kernel_path = Path(sys.prefix) / "share/jupyter/kernels/moss-igpu/kernel.json"
    kernel = json.loads(kernel_path.read_text())
    environment = igpu_environment()
    kernel["env"] = {"LD_LIBRARY_PATH": environment["LD_LIBRARY_PATH"]}
    kernel_path.write_text(json.dumps(kernel, indent=2), encoding="utf-8")
    print("Select MOSS Intel iGPU (.venv) and restart the notebook kernel.")


if __name__ == "__main__":
    main()
