"""Check the existing Conda environment without installing or upgrading packages."""
from __future__ import annotations

import importlib
import platform
import sys

PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "torch": "torch",
}


def main() -> int:
    print("=" * 68)
    print("RESEARCH PIPELINE V2 - ENVIRONMENT CHECK")
    print("=" * 68)
    print("Operating system:", platform.platform())
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.split()[0])
    print()

    failed = []
    for package, module_name in PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "version unavailable")
            print(f"[OK] {package}: {version}")
        except Exception as error:
            failed.append(package)
            print(f"[MISSING/ERROR] {package}: {error}")

    try:
        import torch
        print("\nPyTorch CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device:", torch.cuda.get_device_name(0))
        else:
            print("Training will use CPU unless a working CUDA device is available.")
    except Exception:
        pass

    print()
    if failed:
        print("Missing or broken packages:", ", ".join(failed))
        print("Do not install everything blindly. Share this output first.")
        return 1

    print("[OK] All required imports are available.")
    print("No packages were installed, removed or upgraded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
