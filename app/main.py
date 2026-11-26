"""docparse-vietocr HTTP service.

    GET  /health  -> {"status": "ok"|"loading"|"error", "device", "models", "version"}; 200 only when ready
    POST /parse   -> multipart field `file` (one page image, PNG/JPEG; atoma-chat-db sends 2480 px wide pages)
                     ?include_dropped=true adds `dropped_lines` (text read inside stamp/picture regions)
                     -> the docparse-bench contract for one page + "version", "timings", "image_size"
       errors: 400 unreadable image, 401 bad token, 413 too large, 503 loading/busy (Retry-After), 504 timeout,
               500 internal error (short message; document text is never returned in errors or logged)

Auth: `Authorization: Bearer $DOCPARSE_TOKEN` when DOCPARSE_TOKEN is set (off by default).
Run:  uv run python -m app.main
"""

from __future__ import annotations

# --- order matters: settings + paddle env, then the Windows CUDA DLL preload, then everything else ---
from app.config import Settings, apply_runtime_env

SETTINGS = Settings.from_env()
apply_runtime_env(SETTINGS)

from app import cuda_preload  # noqa: E402

cuda_preload.preload()

import asyncio  # noqa: E402
import hmac  # noqa: E402
import logging  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from app import __version__  # noqa: E402
from app.pipeline import ImageError, ImageTooLarge, decode_image  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("docparse")


class Engine:
    """Owns the pipeline. Models load in a background thread; one GPU job runs at a time."""

    def __init__(self, s: Settings):
        self.s = s
        self.pipeline = None
        self.status = "loading"
        self.error: str | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docparse-gpu")
        self.in_flight = 0  # running + waiting jobs
        self.lock = threading.Lock()

    @property
    def models(self) -> dict:
        if self.pipeline is not None:
            return self.pipeline.models
        from app.pipeline import DET_MODEL, LAYOUT_MODEL

        return {"layout": LAYOUT_MODEL, "det": DET_MODEL, "rec": f"vietocr/{self.s.rec_model}"}

    def load(self) -> None:
        try:
            from app.pipeline import Pipeline, warmup_image

            t = time.perf_counter()
            p = Pipeline(self.s)
            t_load = time.perf_counter() - t
            t = time.perf_counter()
            p.run(warmup_image())
            log.info("models loaded in %.1fs, warm-up %.1fs, device=%s rec=%s", t_load, time.perf_counter() - t, self.s.device_label, p.rec.name)
            log_vram()
            self.pipeline = p
            self.status = "ok"
        except Exception as e:  # keep serving /health with the reason
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"
            log.error("model load failed: %s\n%s", self.error, "".join(traceback.format_tb(e.__traceback__)))

    def try_reserve(self) -> bool:
        with self.lock:
            if self.in_flight >= self.s.max_queue + 1:  # one running + max_queue waiting
                return False
            self.in_flight += 1
            return True

    def release(self, _fut=None) -> None:
        with self.lock:
            self.in_flight -= 1


def log_vram() -> None:
    parts = []
    try:
        import torch

        if torch.cuda.is_available():
            parts.append(f"torch peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    except Exception:
        pass
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            parts.append(
                f"paddle peak {paddle.device.cuda.max_memory_allocated() / 2**20:.0f} MiB"
                f" (reserved {paddle.device.cuda.max_memory_reserved() / 2**20:.0f} MiB)"
            )
    except Exception:
        pass
    if parts:
        log.info("VRAM after warm-up: %s", ", ".join(parts))


def create_app(s: Settings = SETTINGS, load_models: bool = True) -> FastAPI:
    engine = Engine(s)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_models:
            threading.Thread(target=engine.load, name="docparse-load", daemon=True).start()
        yield
        engine.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="docparse-vietocr", version=__version__, lifespan=lifespan)
    app.state.engine = engine

    def check_auth(authorization: str | None) -> None:
        if not s.token:
            return
        expected = f"Bearer {s.token}"
        if not authorization or not hmac.compare_digest(authorization.encode(), expected.encode()):
            raise HTTPException(401, "missing or wrong bearer token", headers={"WWW-Authenticate": "Bearer"})

    @app.get("/health")
    def health():
        body = {"status": engine.status, "device": s.device_label, "models": engine.models, "version": __version__}
        if engine.error:
            body["error"] = engine.error
        return JSONResponse(body, status_code=200 if engine.status == "ok" else 503)

    @app.post("/parse")
    async def parse(
        request: Request,
        file: UploadFile = File(...),
        include_dropped: bool = Query(False),
        authorization: str | None = Header(None),
    ):
        check_auth(authorization)
        t0 = time.perf_counter()
        # size limit: trust Content-Length for an early reject, then enforce while reading
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > s.max_upload_bytes + 64 * 1024:
            raise HTTPException(413, f"upload larger than {s.max_upload_mb:g} MB")
        data = await file.read(s.max_upload_bytes + 1)
        await file.close()
        if len(data) > s.max_upload_bytes:
            raise HTTPException(413, f"upload larger than {s.max_upload_mb:g} MB")
        try:
            bgr = decode_image(data, s.max_mpix)
        except ImageTooLarge as e:
            raise HTTPException(413, str(e)) from None
        except ImageError as e:
            raise HTTPException(400, str(e)) from None
        del data
        h, w = bgr.shape[:2]

        if engine.status != "ok":
            raise HTTPException(503, f"service {engine.status}", headers={"Retry-After": "10"})
        if not engine.try_reserve():
            log.warning("parse rejected: queue full (%d in flight) image=%dx%d", engine.in_flight, w, h)
            raise HTTPException(503, "busy, queue full", headers={"Retry-After": "5"})

        page = Path(file.filename).stem if file.filename else None
        t_queued = time.perf_counter()
        fut = engine.executor.submit(engine.pipeline.run, bgr, page, include_dropped)
        fut.add_done_callback(engine.release)
        try:
            result, stats = await asyncio.wait_for(asyncio.wrap_future(fut), timeout=s.timeout_s)
        except asyncio.TimeoutError:
            fut.cancel()  # only helps if it has not started; a running job finishes and frees its slot then
            log.warning("parse timeout after %.0fs image=%dx%d", s.timeout_s, w, h)
            raise HTTPException(504, f"timed out after {s.timeout_s:g}s") from None
        except Exception as e:
            log.error("parse failed image=%dx%d: %s\n%s", w, h, type(e).__name__, "".join(traceback.format_tb(e.__traceback__)))
            raise HTTPException(500, f"internal error ({type(e).__name__})") from None
        tm = result["timings"]
        log.info(
            "parse ok image=%dx%d regions=%d lines=%d leaders=%d in_pictures=%d orphans=%d "
            "wait=%.2fs layout=%.2fs det=%.2fs leaders=%.2fs rec=%.2fs total=%.2fs",
            w, h, stats["regions"], stats["lines"], stats["dot_leaders"], stats["lines_in_pictures"], stats["orphan_lines"],
            max(0.0, time.perf_counter() - t_queued - result["seconds"]),
            tm["layout"], tm["det"], tm["leaders"], tm["rec"], time.perf_counter() - t0,
        )  # fmt: skip
        return result

    return app


app = create_app()


def main() -> None:
    import uvicorn

    log.info("docparse-vietocr %s on %s:%d device=%s rec=%s models=%s", __version__, SETTINGS.host, SETTINGS.port, SETTINGS.device_label, SETTINGS.rec_model, SETTINGS.model_dir)
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, workers=1)


if __name__ == "__main__":
    main()
