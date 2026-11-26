r"""Fetch the Paddle models and VietOCR weights into DOCPARSE_MODEL_DIR (run once; needs internet).

    uv run python scripts/download_models.py [--all-rec] [--from-cache DIR]

--from-cache DIR copies what is already in a local cache instead of downloading:
  DIR/paddlex/official_models/<model>/ and DIR/vietocr/<model>.pth (the bench-era layout under D:\python-projects\.cache).
After this, the service starts with no network access.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

import _common  # noqa: F401  (sys.path)

from app.config import REC_MODELS, Settings, apply_runtime_env


def download(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"downloading {url} -> {dst}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-rec", action="store_true", help="fetch both VietOCR models, not only DOCPARSE_REC_MODEL")
    ap.add_argument("--from-cache", type=Path, help="copy from an existing cache folder instead of downloading")
    a = ap.parse_args()

    s = Settings.from_env()
    apply_runtime_env(s)
    from app.pipeline import DET_MODEL, LAYOUT_MODEL, vietocr_config, vietocr_weights_path

    official = s.paddlex_home / "official_models"
    official.mkdir(parents=True, exist_ok=True)
    s.vietocr_dir.mkdir(parents=True, exist_ok=True)

    # VietOCR weights
    for name in REC_MODELS if a.all_rec else (s.rec_model,):
        dst = vietocr_weights_path(s, name)
        if dst.is_file():
            print(f"ok   {dst}")
            continue
        src = a.from_cache / "vietocr" / dst.name if a.from_cache else None
        if src and src.is_file():
            print(f"copy {src} -> {dst}")
            shutil.copy2(src, dst)
        else:
            download(vietocr_config(name)["weights"], dst)

    # Paddle models: copy from cache, else let paddlex download them by building the predictors on CPU
    missing = []
    for m in (LAYOUT_MODEL, DET_MODEL):
        dst = official / m
        src = a.from_cache / "paddlex" / "official_models" / m if a.from_cache else None
        if (dst / "inference.yml").is_file():
            print(f"ok   {dst}")
        elif src and (src / "inference.yml").is_file():
            print(f"copy {src} -> {dst}")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            missing.append(m)
    if missing:
        import os

        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "False"  # allow the hoster lookup for the download
        from app import cuda_preload

        cuda_preload.preload()
        from paddleocr import LayoutDetection, TextDetection

        for m in missing:
            print(f"fetching {m} via paddlex ...", flush=True)
            (LayoutDetection if m == LAYOUT_MODEL else TextDetection)(model_name=m, device="cpu")
            print(f"ok   {official / m}")
    print(f"models ready in {s.model_dir}")


if __name__ == "__main__":
    sys.exit(main())
