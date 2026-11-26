"""Page image -> bench-contract regions.

layout (PP-DocLayout_plus-L) -> text-line detection on the full page (PP-OCRv5_server_det) -> each line quad
perspective-warped to a strip -> VietOCR -> dotted leaders found on the page -> regions in reading order.

Import this only after `config.apply_runtime_env()` (and on Windows `cuda_preload.preload()`).
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import yaml

from app import __version__
from app.config import ROOT, Settings
from app.geometry import area, centre, contains_point, order_quad, warp_line
from app.leaders import find_dot_leaders
from app.order import reading_order, resolve_overlaps, sort_lines

log = logging.getLogger("docparse.pipeline")

LAYOUT_MODEL = "PP-DocLayout_plus-L"
DET_MODEL = "PP-OCRv5_server_det"
VIETOCR_CONFIG_DIR = ROOT / "app" / "vietocr_config"
VIETOCR_CONFIG_FILES = {"vgg_transformer": "vgg-transformer.yml", "vgg_seq2seq": "vgg-seq2seq.yml"}

KIND = {
    "doc_title": "title",
    "paragraph_title": "title",
    "text": "text",
    "abstract": "text",
    "content": "text",
    "aside_text": "text",
    "reference": "text",
    "reference_content": "text",
    "figure_title": "text",
    "table_title": "text",
    "chart_title": "text",
    "vision_footnote": "footer",
    "header": "header",
    "number": "header",
    "footer": "footer",
    "footnote": "footer",
    "table": "table",
    "seal": "seal",
    "image": "picture",
    "figure": "picture",
    "chart": "picture",
    "list": "list",
}
PICTURE_KINDS = {"picture", "seal"}


# ---------------------------------------------------------------- input


class ImageError(ValueError):
    """Unreadable or empty image (HTTP 400)."""


class ImageTooLarge(ValueError):
    """Over the pixel limit (HTTP 413)."""


def decode_image(data: bytes, max_mpix: float) -> np.ndarray:
    """Bytes (PNG/JPEG/...) -> BGR uint8. The size is checked from the header before decoding."""
    if not data:
        raise ImageError("empty upload")
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Exception:
        raise ImageError("not a readable image") from None
    if w * h > max_mpix * 1e6:
        raise ImageTooLarge(f"image is {w}x{h} = {w * h / 1e6:.1f} MP, limit {max_mpix:g} MP")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0 or min(bgr.shape[:2]) < 8:
        raise ImageError("not a readable image")
    return bgr


# ---------------------------------------------------------------- recognisers


class Recogniser(Protocol):
    """Line crops (RGB uint8) -> (text, confidence). Kept behind this interface so it can move to a child process
    if paddle and torch ever cannot share one (see README, "Linux risk")."""

    name: str

    def __call__(self, crops: list[np.ndarray]) -> list[tuple[str, float | None]]: ...


def vietocr_config(name: str) -> dict:
    """Vendored base.yml + model yml, merged like vietocr's Cfg.load_config_from_name (no vocr.vn fetch)."""
    cfg = yaml.safe_load((VIETOCR_CONFIG_DIR / "base.yml").read_text(encoding="utf-8"))
    cfg.update(yaml.safe_load((VIETOCR_CONFIG_DIR / VIETOCR_CONFIG_FILES[name]).read_text(encoding="utf-8")))
    return cfg


def vietocr_weights_path(s: Settings, name: str) -> Path:
    url = vietocr_config(name)["weights"]
    return s.vietocr_dir / url.split("/")[-1]


class VietOcrRecogniser:
    def __init__(self, s: Settings):
        import torch
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        cfg = Cfg(vietocr_config(s.rec_model))
        weights = vietocr_weights_path(s, s.rec_model)
        if not weights.is_file():
            raise FileNotFoundError(f"VietOCR weights missing: {weights} - run scripts/download_models.py")
        cfg["weights"] = str(weights)
        device = s.rec_device_or_default
        if device.startswith("gpu") and torch.cuda.is_available():
            cfg["device"] = "cuda:" + (device.split(":", 1)[1] if ":" in device else "0")
        else:
            cfg["device"] = "cpu"
        cfg["cnn"]["pretrained"] = False
        cfg["predictor"]["beamsearch"] = False
        self.name = f"vietocr/{s.rec_model}"
        self.device = cfg["device"]
        self.p = Predictor(cfg)

    def __call__(self, crops: list[np.ndarray]) -> list[tuple[str, float | None]]:
        from PIL import Image

        if not crops:
            return []
        texts, probs = self.p.predict_batch([Image.fromarray(c) for c in crops], return_prob=True)
        return [(t, float(p)) for t, p in zip(texts, probs)]


# ---------------------------------------------------------------- pipeline


class Pipeline:
    def __init__(self, s: Settings, recogniser: Recogniser | None = None):
        from paddleocr import LayoutDetection, TextDetection

        self.s = s
        paddle_device = "gpu:0" if s.device == "gpu" else s.device
        self.layout = LayoutDetection(model_name=LAYOUT_MODEL, device=paddle_device)
        self.det = TextDetection(model_name=DET_MODEL, device=paddle_device)
        self.rec = recogniser if recogniser is not None else VietOcrRecogniser(s)
        self.models = {"layout": LAYOUT_MODEL, "det": DET_MODEL, "rec": self.rec.name}

    @property
    def model_string(self) -> str:
        return f"{LAYOUT_MODEL} + {DET_MODEL} + {self.rec.name}"

    def run(self, bgr: np.ndarray, page: str | None = None, include_dropped: bool = False) -> tuple[dict, dict]:
        """BGR page -> (contract JSON, stats for logging). Stats hold counts only, never text."""
        t_start = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = bgr.shape[:2]
        timings: dict[str, float] = {}
        stats: dict[str, int] = {}

        # 1. layout
        t = time.perf_counter()
        lres = next(iter(self.layout.predict(bgr, batch_size=1, layout_nms=True)))
        timings["layout"] = round(time.perf_counter() - t, 3)
        regions = []
        for b in lres["boxes"]:
            if b["score"] < self.s.layout_min_score:
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in b["coordinate"])
            regions.append({"category": b["label"], "kind": KIND.get(b["label"], "other"), "bbox": [x0, y0, x1, y1], "lines": []})
        regions = resolve_overlaps(regions)

        # 2. line detection on the full page
        t = time.perf_counter()
        dres = next(iter(self.det.predict(bgr, batch_size=1)))
        timings["det"] = round(time.perf_counter() - t, 3)
        polys = [np.asarray(p, dtype=np.float32) for p in dres["dt_polys"]]

        # 3. assign lines to regions (smallest region containing the line centre wins)
        orphans = []
        jobs = []  # (line dict, quad) - every detected line is read, also those inside pictures
        in_pictures = 0
        for q in polys:
            x0, y0 = q[:, 0].min(), q[:, 1].min()
            x1, y1 = q[:, 0].max(), q[:, 1].max()
            box = [int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))]
            cx, cy = centre(box)
            inside = [r for r in regions if contains_point(r["bbox"], cx, cy)]
            # quad (tl, tr, br, bl): the tilted line itself - the client derives baseline and page skew from it
            line = {"bbox": box, "quad": order_quad(q).round().astype(int).tolist(), "text": "", "conf": None}
            jobs.append((line, q))
            if inside:
                r = min(inside, key=lambda r: area(r["bbox"]))
                if r["kind"] in PICTURE_KINDS:  # stamp text: not part of the page text
                    in_pictures += 1
                    r.setdefault("_dropped", []).append(line)
                    continue
                r["lines"].append(line)
            else:
                orphans.append({"category": "orphan_line", "kind": "text", "bbox": list(box), "lines": [line]})
        regions += orphans
        stats.update(lines=len(polys), orphan_lines=len(orphans), lines_in_pictures=in_pictures)

        # 4. dotted fill-in leaders, outside detected lines, tables and pictures
        t = time.perf_counter()
        mask_boxes = [l["bbox"] for l, _ in jobs] + [r["bbox"] for r in regions if r["kind"] in PICTURE_KINDS | {"table"}]
        leaders = find_dot_leaders(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), mask_boxes)
        for box in leaders:
            line = {"bbox": box, "text": "……", "conf": None}
            cx, cy = centre(box)
            inside = [r for r in regions if r["kind"] not in PICTURE_KINDS and contains_point(r["bbox"], cx, cy)]
            if inside:
                min(inside, key=lambda r: area(r["bbox"]))["lines"].append(line)
            else:
                regions.append({"category": "dot_leader", "kind": "text", "bbox": list(box), "lines": [line]})
        timings["leaders"] = round(time.perf_counter() - t, 3)
        stats["dot_leaders"] = len(leaders)

        # 5. recognition
        t = time.perf_counter()
        crops = [warp_line(rgb, q) for _, q in jobs]
        for (line, _), (txt, conf) in zip(jobs, self.rec(crops)):
            line["text"], line["conf"] = txt.strip(), (round(conf, 4) if conf is not None else None)
        timings["rec"] = round(time.perf_counter() - t, 3)

        # 6. order + output
        out_regions = []
        for i, r in enumerate(reading_order(regions)):
            dropped = r.pop("_dropped", [])
            lines = sort_lines(r["lines"]) if r["kind"] not in PICTURE_KINDS else []
            region = {
                "order": i,
                "category": r["category"],
                "kind": r["kind"],
                "bbox": r["bbox"],
                "text": "\n".join(l["text"] for l in lines),
                "lines": lines,
                "html": None,
            }
            if include_dropped and dropped:
                region["dropped_lines"] = sort_lines(dropped)
            out_regions.append(region)
        stats["regions"] = len(out_regions)
        return (
            {
                "page": page,
                "image_size": [W, H],
                "method": "vietocr",
                "model": self.model_string,
                "version": __version__,
                "seconds": round(time.perf_counter() - t_start, 3),
                "timings": timings,
                "regions": out_regions,
            },
            stats,
        )


def warmup_image() -> np.ndarray:
    """Small synthetic page (a title, two text lines, a dotted leader) for the start-up warm-up run."""
    img = np.full((400, 1200, 3), 255, np.uint8)
    cv2.putText(img, "WARM UP PAGE", (360, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    cv2.putText(img, "The quick brown fox jumps over the lazy dog", (80, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.putText(img, "Name:", (80, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    for x in range(220, 900, 14):
        cv2.circle(img, (x, 285), 2, (0, 0, 0), -1)
    return img
