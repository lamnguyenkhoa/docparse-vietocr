"""Let PaddlePaddle and PyTorch share the GPU in one process on Windows (docs/clash-fix.md). No-op elsewhere.

torch cu128 bundles CUDA 12.8 DLLs (cublas64_12, cudart64_12, cufft, curand, cusolver, cusparse) in torch/lib;
paddlepaddle-gpu cu129 uses the 12.9 ones from the nvidia-* wheels. Same file names, so whichever loads first wins
and the other fails with WinError 127 ("procedure not found"). Both also ship libiomp5md.dll.

Fix: load the newer 12.9 DLLs first (12.8-built torch runs fine on 12.x >= 12.8), then import torch before paddle
so torch's libiomp5md.dll is the one in the process. Call preload() before importing anything that pulls in
torch or paddle.
"""

import ctypes
import os
import sys
from pathlib import Path

ORDER = ["cuda_runtime", "nvjitlink", "cublas", "cufft", "curand", "cusparse", "cusolver", "cudnn"]


def preload() -> None:
    if sys.platform != "win32":
        return
    root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for pkg in ORDER:
        b = root / pkg / "bin"
        if not b.is_dir():
            continue
        os.add_dll_directory(str(b))
        for dll in sorted(b.glob("*.dll")):
            if not dll.name.startswith("nvblas"):
                ctypes.WinDLL(str(dll))
    import torch  # noqa: F401  - must come before paddle (libiomp5md.dll)
