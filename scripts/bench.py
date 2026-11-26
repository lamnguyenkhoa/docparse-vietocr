"""Regression run: the pipeline in-process on the docparse-bench pages -> out/bench/, then the bench scorer.

    uv run python scripts/bench.py [--bench DIR] [--out out/bench] [--no-check]

Uses the same decode path as POST /parse. Takes the GPU lock while models are loaded if DOCPARSE_GPU_LOCK is set
(optional, dev box only). Exits non-zero if
the result misses the regression bar (SERVICE-PLAN.md section 6): ALL WER 2.5% +-0.2, fills form 9/12, p3 17/19,
p4 2/2, stamp 1/1 on p2, tables 2/2 on form, no extra pictures/tables anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from _common import ROOT, gpu_lock

from app.config import Settings, apply_runtime_env

BAR = {
    "wer": (2.3, 2.7),
    "fills": {"form_p1": "9/12", "vanbangoc_p3": "17/19", "vanbangoc_p4": "2/2"},
    "pictures": {"vanbangoc_p2": "1/1"},
    "tables": {"form_p1": "2/2"},
}


def check(score: dict) -> list[str]:
    fails = []
    t = score["total"]
    wer = round(100 * t["e"] / t["n"], 1)
    if not BAR["wer"][0] <= wer <= BAR["wer"][1]:
        fails.append(f"ALL WER {wer}% outside {BAR['wer']}")
    for p in score["pages"]:
        name = p["page"]
        if p.get("missing"):
            fails.append(f"{name}: missing")
            continue
        if name in BAR["fills"] and p["fills"] != BAR["fills"][name]:
            fails.append(f"{name}: fills {p['fills']} != {BAR['fills'][name]}")
        for key in ("pictures", "tables"):
            found, extra = p[key].split(" ", 1)
            if name in BAR[key] and found != BAR[key][name]:
                fails.append(f"{name}: {key} {found} != {BAR[key][name]}")
            if extra != "(+0 extra)":
                fails.append(f"{name}: {key} {extra}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=Path(os.environ.get("DOCPARSE_BENCH_DIR", r"D:\python-projects\docparse-bench")))
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "bench")
    ap.add_argument("--no-check", action="store_true", help="only report, do not enforce the regression bar")
    a = ap.parse_args()
    pages = sorted((a.bench / "data" / "pages").glob("*.png"))
    if not pages:
        sys.exit(f"no pages in {a.bench / 'data' / 'pages'}")
    a.out.mkdir(parents=True, exist_ok=True)

    s = Settings.from_env()
    apply_runtime_env(s)
    from app import cuda_preload

    cuda_preload.preload()
    from app.pipeline import Pipeline, decode_image, warmup_image

    with gpu_lock("docparse-vietocr bench"):
        t = time.perf_counter()
        pipe = Pipeline(s)
        print(f"models loaded in {time.perf_counter() - t:.1f}s ({pipe.model_string}, device {s.device_label})", flush=True)
        pipe.run(warmup_image())
        for png in pages:
            res, stats = pipe.run(decode_image(png.read_bytes(), s.max_mpix), page=png.stem, include_dropped=True)
            (a.out / f"{png.stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{png.stem}: {res['seconds']:.2f}s {json.dumps(res['timings'])} {json.dumps(stats)}", flush=True)
        del pipe

    r = subprocess.run([sys.executable, str(a.bench / "score.py"), str(a.out), "--overlay"], text=True, encoding="utf-8")
    if r.returncode:
        return r.returncode
    fails = check(json.loads((a.out / "score.json").read_text(encoding="utf-8")))
    if fails:
        print("REGRESSION BAR MISSED:\n  " + "\n  ".join(fails))
        return 0 if a.no_check else 1
    print("regression bar: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
