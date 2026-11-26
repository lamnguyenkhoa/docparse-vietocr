"""API tests. The fast ones run without models; `gpu` ones load the real models (skipped when not available)."""

from __future__ import annotations

import dataclasses
import io
import os
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import SETTINGS, create_app

BENCH = Path(os.environ.get("DOCPARSE_BENCH_DIR", r"D:\python-projects\docparse-bench"))
KINDS = {"text", "title", "header", "footer", "list", "table", "picture", "seal", "other"}


def png_bytes(w=64, h=48) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), 255, np.uint8)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def client_no_models():
    with TestClient(create_app(SETTINGS, load_models=False)) as c:
        yield c


def test_health_loading_is_503(client_no_models):
    r = client_no_models.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "loading" and body["version"] and body["models"]["rec"].startswith("vietocr/")


def test_garbage_bytes_400(client_no_models):
    r = client_no_models.post("/parse", files={"file": ("x.png", b"not an image at all", "image/png")})
    assert r.status_code == 400


def test_empty_upload_400(client_no_models):
    r = client_no_models.post("/parse", files={"file": ("x.png", b"", "image/png")})
    assert r.status_code == 400


def test_oversize_upload_413():
    s = dataclasses.replace(SETTINGS, max_upload_mb=0.001)  # ~1 KB
    with TestClient(create_app(s, load_models=False)) as c:
        r = c.post("/parse", files={"file": ("x.png", os.urandom(4096), "image/png")})
    assert r.status_code == 413


def test_too_many_pixels_413():
    s = dataclasses.replace(SETTINGS, max_mpix=0.001)  # 1000 px
    with TestClient(create_app(s, load_models=False)) as c:
        r = c.post("/parse", files={"file": ("x.png", png_bytes(64, 48), "image/png")})
    assert r.status_code == 413


def test_valid_image_while_loading_503(client_no_models):
    r = client_no_models.post("/parse", files={"file": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 503 and "retry-after" in r.headers


def test_token_required_when_set():
    s = dataclasses.replace(SETTINGS, token="secret")
    with TestClient(create_app(s, load_models=False)) as c:
        assert c.post("/parse", files={"file": ("x.png", png_bytes(), "image/png")}).status_code == 401
        r = c.post("/parse", files={"file": ("x.png", b"junk", "image/png")}, headers={"Authorization": "Bearer secret"})
        assert r.status_code == 400  # past auth


# ---------------------------------------------------------------- real models


@pytest.fixture(scope="module")
def client_models():
    from app.pipeline import vietocr_weights_path

    if os.environ.get("DOCPARSE_SKIP_GPU_TESTS"):
        pytest.skip("DOCPARSE_SKIP_GPU_TESTS set")
    if not vietocr_weights_path(SETTINGS, SETTINGS.rec_model).is_file():
        pytest.skip("models not downloaded (scripts/download_models.py)")
    from _common import GpuBusy, gpu_lock

    try:
        lock = gpu_lock("docparse-vietocr pytest")
        lock.__enter__()
    except GpuBusy as e:
        pytest.skip(str(e))
    try:
        with TestClient(create_app(SETTINGS, load_models=True)) as c:
            deadline = time.time() + 300
            while time.time() < deadline:
                r = c.get("/health")
                if r.json()["status"] != "loading":
                    break
                time.sleep(1)
            assert r.status_code == 200, r.json()
            yield c
    finally:
        lock.__exit__(None, None, None)


@pytest.mark.gpu
def test_health_ok(client_models):
    body = client_models.get("/health").json()
    assert body["status"] == "ok" and body["models"]["layout"] == "PP-DocLayout_plus-L"


@pytest.mark.gpu
def test_parse_bench_page_matches_contract(client_models):
    page = BENCH / "data" / "pages" / "vanbangoc_p2.png"
    if not page.is_file():
        pytest.skip(f"bench page missing: {page}")
    r = client_models.post("/parse", files={"file": (page.name, page.read_bytes(), "image/png")}, params={"include_dropped": "true"})
    assert r.status_code == 200, r.text[:200]
    d = r.json()
    for k in ("page", "image_size", "method", "model", "seconds", "version", "timings", "regions"):
        assert k in d
    assert d["page"] == "vanbangoc_p2"
    W, H = d["image_size"]
    assert (W, H) == Image.open(page).size
    assert set(d["timings"]) == {"layout", "det", "rec", "leaders"}

    def in_image(b):
        return len(b) == 4 and all(isinstance(v, int) for v in b) and 0 <= b[0] <= b[2] <= W and 0 <= b[1] <= b[3] <= H

    assert [r["order"] for r in d["regions"]] == list(range(len(d["regions"])))
    for reg in d["regions"]:
        assert reg["kind"] in KINDS and isinstance(reg["category"], str)
        assert in_image(reg["bbox"]) and isinstance(reg["text"], str) and reg["html"] is None
        for line in reg["lines"]:
            assert in_image(line["bbox"]) and isinstance(line["text"], str)
            assert line["conf"] is None or 0.0 <= line["conf"] <= 1.0
        if reg["kind"] in ("picture", "seal"):
            assert reg["lines"] == [] and reg["text"] == ""
    pics = [reg for reg in d["regions"] if reg["kind"] in ("picture", "seal")]
    assert len(pics) == 1 and pics[0].get("dropped_lines")  # the stamp, with its printed text kept aside
    assert "Trần Xuân Hà" in "\n".join(reg["text"] for reg in d["regions"])


# ---------------------------------------------------------------- queue / timeout with a fake pipeline


class SlowPipeline:
    models = {"layout": "fake", "det": "fake", "rec": "fake"}

    def __init__(self, seconds: float):
        self.seconds = seconds

    def run(self, bgr, page=None, include_dropped=False):
        time.sleep(self.seconds)
        h, w = bgr.shape[:2]
        timings = {"layout": 0.0, "det": 0.0, "leaders": 0.0, "rec": self.seconds}
        return (
            {"page": page, "image_size": [w, h], "method": "vietocr", "model": "fake", "version": "x", "seconds": self.seconds, "timings": timings, "regions": []},
            {"regions": 0, "lines": 0, "dot_leaders": 0, "lines_in_pictures": 0, "orphan_lines": 0},
        )


def ready_app(s, pipeline):
    app = create_app(s, load_models=False)
    app.state.engine.pipeline = pipeline
    app.state.engine.status = "ok"
    return app


def test_queue_full_503():
    import threading

    s = dataclasses.replace(SETTINGS, max_queue=0)  # one running, none waiting
    with TestClient(ready_app(s, SlowPipeline(1.5))) as c:
        first = {}
        t = threading.Thread(target=lambda: first.setdefault("r", c.post("/parse", files={"file": ("a.png", png_bytes(), "image/png")})))
        t.start()
        time.sleep(0.5)
        r = c.post("/parse", files={"file": ("b.png", png_bytes(), "image/png")})
        t.join()
    assert r.status_code == 503 and r.headers.get("retry-after")
    assert first["r"].status_code == 200 and first["r"].json()["page"] == "a"


def test_timeout_504():
    s = dataclasses.replace(SETTINGS, timeout_s=0.3)
    with TestClient(ready_app(s, SlowPipeline(1.0))) as c:
        r = c.post("/parse", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 504
