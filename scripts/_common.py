"""Shared bits for scripts/ and tests/: project root on sys.path, the optional dev-box GPU lock."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional dev-box GPU lock for scripts/tests only (never used by app/). Off unless DOCPARSE_GPU_LOCK names the
# lock file, e.g. DOCPARSE_GPU_LOCK=D:\python-projects\.gpu.lock when several agents share the GPU.
GPU_LOCK = os.environ.get("DOCPARSE_GPU_LOCK", "").strip()


class GpuBusy(RuntimeError):
    pass


@contextmanager
def gpu_lock(owner: str = "docparse-vietocr"):
    """Create the lock file only if it does not exist; remove it afterwards (also on errors)."""
    if not GPU_LOCK:
        yield
        return
    path = Path(GPU_LOCK)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = path.read_text(encoding="utf-8", errors="replace").strip() if path.exists() else "?"
        raise GpuBusy(f"GPU lock {path} is held by {holder!r}; retry later") from None
    try:
        os.write(fd, owner.encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)
