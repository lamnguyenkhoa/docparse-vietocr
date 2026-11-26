# Plan: clean up docparse-vietocr and turn it into an HTTP service

The bake-off is over: this stack (PP-DocLayout_plus-L layout + PP-OCRv5_server_det lines + VietOCR vgg_transformer +
dot-leader finder) won (2.1% WER on the vanbangoc pages, ~2.5 s/page, ~1.7 GB VRAM). See `RESULTS.md`, `CLASH-FIX.md`
and `D:\python-projects\docparse-bench\README.md` (output contract, scorer).

Goal: a small, boring, production-ready service that atoma-chat-db calls once per scanned page:
`POST /parse` (page image) -> the bench contract JSON. Deployed on the Linux box 172.16.20.76 later, next to the
reranker. Building the DOCX stays in atoma-chat-db - not part of this work. Do not touch `D:\atoma\atoma-chat-db`.

## 1. Target layout

Follow the house style of `D:\python-projects\vieneu-tts-server` (uv project, `app/` package, `python -m app.main`,
systemd script).

```
docparse-vietocr/
  app/
    __init__.py
    main.py            FastAPI app + uvicorn entry (python -m app.main)
    config.py          settings from env (.env supported)
    pipeline.py        Pipeline: layout -> line det -> warp -> VietOCR -> leaders -> regions/order (from run.py)
    leaders.py         find_dot_leaders
    order.py           resolve_overlaps, reading_order, sort_lines
    geometry.py        area/inter/overlap_1d/order_quad/warp_line
    cuda_preload.py    unchanged logic (Windows only; no-op elsewhere)
    vietocr_config/    vendored base.yml + vgg_transformer.yml (+ vgg_seq2seq.yml) - no vocr.vn fetch at start
  scripts/
    download_models.py fetch Paddle models + VietOCR weights into MODEL_DIR (run once, needs internet)
    bench.py           run the pipeline in-process on the bench pages -> out/bench/, then docparse-bench/score.py
  tests/               pytest (see §6)
  docs/
    bakeoff-results.md (= old RESULTS.md, unchanged)
    clash-fix.md       (= old CLASH-FIX.md)
  pyproject.toml, uv.lock, .env.example, README.md, linux-service.sh, run-docparse-server.bat
```

## 2. Clean-up (remove what the service does not need)

- **Delete:** `run.py` (after its logic moves into `app/`), `env.sh` (-> `.env.example`), `requirements.txt` and
  `requirements.lock.txt` (-> `pyproject.toml` + `uv.lock`), `__pycache__/`, the whole `out/` folder (bench.py
  regenerates results; scores are kept in docs/bakeoff-results.md), `PLAN.md`.
- **Drop code paths:** the `ppocr-rec` variant and `PaddleRec` (cannot write Vietnamese), `--no-warp`/`axis_crop`
  (ablation only), the variant CLI, the warm-up/VRAM printing in `main()`, `PPOCR_REC_MODEL`.
- **Keep as config, not variants:** recogniser model `vgg_transformer` (default) or `vgg_seq2seq` (fast mode, 3.2% WER,
  0.7 s/page); paddle device (`gpu:0` default, `cpu` allowed).
- **No hard-coded Windows paths** (`D:\python-projects\...`) in `app/`: everything comes from config.
- Do not change the recognition behaviour while moving code: `scripts/bench.py` must reproduce the current scores
  (§6) - that is the proof the refactor is safe.

## 3. API

Port **8126** by default (8124 = reranker, 8125 = .doc converter on 76). Bind host from config.

- `GET /health` -> `{"status": "ok"|"loading", "device": "...", "models": {...names...}, "version": "..."}`.
  200 only when models are loaded; 503 while loading.
- `POST /parse` - `multipart/form-data` field `file`: one page image (PNG/JPEG). atoma-chat-db renders PDF pages
  itself at **2480 px wide** (300 dpi A4); accept any size up to the limits below, return boxes in the pixels of the
  image received. Optional query: `include_dropped=true` to include the stamp's `dropped_lines`.
  Response = the bench contract for one page, plus:
  `"version"` (service version), `"timings": {"layout": s, "det": s, "rec": s, "leaders": s}`, `"image_size"`.
  Field names, `kind` values, `order`, `lines[].bbox/conf` exactly as in the bench README - atoma-chat-db codes against it.
- Errors: 400 unreadable/empty image, 413 too large, 503 busy (queue full, with `Retry-After`), 500 with a short
  message (never the document text).
- Optional `Authorization: Bearer <DOCPARSE_TOKEN>` when `DOCPARSE_TOKEN` is set (off by default - internal network).

No PDF endpoint, no DOCX, no batch endpoint - keep it small.

## 4. Runtime behaviour

- **Load models once at startup** (FastAPI lifespan), then run one warm-up inference on a small generated image so
  the first real request is not slow. `/health` says `loading` until done.
- **One GPU job at a time**: a single worker thread (or an asyncio lock around a threadpool call) runs the pipeline;
  requests wait in a bounded queue (`DOCPARSE_MAX_QUEUE`, default 8) -> 503 when full. Per-request timeout
  (`DOCPARSE_TIMEOUT_S`, default 60).
- **Limits**: upload <= 25 MB, image <= 40 megapixels (both configurable).
- **VRAM**: set `FLAGS_allocator_strategy=auto_growth` before paddle import (it must not grab the card - the reranker
  shares it). Log peak torch/paddle memory at startup.
- **Offline after setup**: VietOCR configs vendored in `app/vietocr_config/`; weights and Paddle models loaded from
  `MODEL_DIR` (`PADDLE_PDX_CACHE_HOME` pointed into it, `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`). Startup must
  work with no internet once `scripts/download_models.py` has run.
- **Logging**: one line per request - image size, region/line counts, timings, status. **Never log recognised text
  or images**: the documents are confidential.
- `cuda_preload.preload()` runs first in `app/main.py` on Windows (see docs/clash-fix.md); on Linux it is a no-op
  unless the same clash shows up there (see §7).

## 5. Config (`app/config.py`, all env, `.env` supported; document every key in `.env.example`)

`DOCPARSE_HOST` (127.0.0.1), `DOCPARSE_PORT` (8126), `DOCPARSE_TOKEN` (empty), `DOCPARSE_DEVICE` (gpu:0),
`DOCPARSE_REC_MODEL` (vgg_transformer), `DOCPARSE_MODEL_DIR` (./models), `DOCPARSE_MAX_QUEUE` (8),
`DOCPARSE_TIMEOUT_S` (60), `DOCPARSE_MAX_UPLOAD_MB` (25), `DOCPARSE_MAX_MPIX` (40), `DOCPARSE_LAYOUT_MIN_SCORE` (0.5),
`CUDA_VISIBLE_DEVICES` passthrough (on 76 the service must use GPU 0 - see §7).

## 6. Tests and the regression bar

- `scripts/bench.py`: runs the pipeline in-process on `docparse-bench/data/pages/*.png`, writes `out/bench/`, runs
  `docparse-bench/score.py`. **Must match the current `vietocr` result** within noise: ALL WER 2.5% (+-0.2), fills
  form 9/12, p3 17/19, p4 2/2, pictures 1/1 on p2, tables 2/2 on form, 0 extras.
- pytest (fast, CPU-friendly where possible):
  - `order.py`: two-column header -> left column first; the 30% band threshold case (`Kính gửi` before `Họ và tên`).
  - `leaders.py`: a synthetic image with a row of evenly spaced dots -> one leader; scattered noise -> none.
  - API: `/health`, `/parse` on one bench page with the real models (mark as `gpu`, skippable) -> response matches the
    contract (keys, types, bbox inside image); 400 on garbage bytes; 413 on oversize.
- GPU lock (`D:\python-projects\.gpu.lock`): only for dev scripts/tests on the Windows box, never in `app/` - the
  service on 76 must not depend on it. The bake-off is over, so it is optional; keep it env-controlled
  (`DOCPARSE_GPU_LOCK`, empty = off).

## 7. Packaging and deployment

- `pyproject.toml` with uv (like vieneu-tts-server): Python 3.12; explicit indexes for `paddlepaddle-gpu` (Paddle
  cu129 index) and `torch`/`torchvision` (PyTorch cu128 index, Blackwell needs >= 12.8); pin the versions that work
  now (see requirements.txt being removed). vietocr pins old deps - declare its real deps with
  `[[tool.uv.dependency-metadata]]` (or `override-dependencies`) instead of `--no-deps`, so `uv sync` reproduces the
  env: einops, gdown, pyyaml, pillow, requests (check imports). Keep `nvidia-cudnn-cu12==9.19.0.56`.
  Recreate `.venv` from `uv sync` and run the bench to prove the lockfile is complete.
- `run-docparse-server.bat` (Windows, local) and `linux-service.sh` (systemd unit like vieneu-tts-server:
  `uv run python -m app.main`, `Restart=always`, user `atoma`, WorkingDirectory `/home/atoma/Atoma/docparse-vietocr`,
  `Environment=CUDA_VISIBLE_DEVICES=0`).
- **Target machine 172.16.20.76** (Linux): GPU 0 = RTX 5060 Ti **8 GB** shared with the TEI reranker (Docker, port
  8124); GPU 1 = RTX 5080 16 GB is full with Ollama - do not use it. Both Blackwell (sm_120). The service must stay
  around 2 GB of VRAM.
- **Linux dependency risk - write it down in README, you cannot test it from here**: on Linux, torch cu128 and
  paddlepaddle-gpu cu129 both depend on `nvidia-*-cu12` pip packages, possibly with conflicting pins (the Windows
  DLL clash may become a resolver conflict instead). Expected fixes, in order: (1) let uv resolve the newest 12.x
  nvidia packages for both (overrides) - newer 12.x works for a 12.8-built torch; (2) if a clash remains at runtime,
  run VietOCR recognition in a child process (`multiprocessing` spawn) so paddle and torch never share a process.
  Make the pipeline structured so (2) is a small change (recogniser behind an interface).
- README: what it is, API with a curl example, config table, install (Windows + Linux), `download_models.py`,
  bench/regression, the Linux risk above, resource numbers (VRAM ~1.7 GB, ~2.5 s/page on RTX 5060, model load ~7 s).

## 8. Deliverables / done when

- Folder matches §1, everything in §2 removed, `uv sync` from scratch works on Windows.
- Service starts offline, `/health` ok, `curl -F file=@...vanbangoc_p2.png http://127.0.0.1:8126/parse` returns the
  contract JSON; `scripts/bench.py` meets the regression bar; pytest green.
- A short `CHANGES.md` (or section in README) listing what was removed/moved, and anything you could not do.

Out of scope: the atoma-chat-db client and DOCX builder, Docker image, table HTML, dots.ocr.
