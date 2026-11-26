# docparse-vietocr

HTTP service that turns one scanned Vietnamese page image into layout regions + text lines with boxes, for
atoma-chat-db's DOCX template builder. Three small models:

1. **Layout**: PP-DocLayout_plus-L (PaddleOCR) - text/title/table/stamp regions.
2. **Text lines**: PP-OCRv5_server_det on the full page - one quad per printed line, perspective-warped to a strip.
3. **Recognition**: VietOCR `vgg_transformer` (Vietnamese, accents included).

Plus a dotted-leader finder (`……` fill-in blanks; the line detector does not box them) and a simple reading order.
Chosen in the docparse-bench bake-off: 2.1% WER on the vanbangoc pages (current Tesseract+qwen merge: 1.7%,
Tesseract: 4.0%), stamp + signature found as one picture, both form tables found. Details: `docs/bakeoff-results.md`.

## API

Port **8126** (76: 8124 = reranker, 8125 = .doc converter).

`GET /health` -> 200 `{"status": "ok", "device": "gpu:0", "models": {"layout", "det", "rec"}, "version"}` once the
models are loaded and warmed up; 503 with `"status": "loading"` (or `"error"` + `"error"` message) before that.

`POST /parse` - `multipart/form-data`, field `file`: one page image (PNG/JPEG). Send pages rendered at
**2480 px wide** (300 dpi A4); other sizes work, boxes are always in the pixels of the image you sent.
Query `include_dropped=true` adds `dropped_lines` to picture/seal regions (text printed inside the stamp, e.g.
"K.T BỘ TRƯỞNG"; normally left out of the page text).

```bash
curl -F file=@vanbangoc_p2.png http://127.0.0.1:8126/parse
# with DOCPARSE_TOKEN set:
curl -H "Authorization: Bearer $DOCPARSE_TOKEN" -F file=@page.png "http://172.16.20.76:8126/parse?include_dropped=true"
```

Response: the docparse-bench output contract (`D:\python-projects\docparse-bench\README.md`), plus `version` and
`timings`:

```json
{
  "page": "vanbangoc_p2",
  "image_size": [2480, 3512],
  "method": "vietocr",
  "model": "PP-DocLayout_plus-L + PP-OCRv5_server_det + vietocr/vgg_transformer",
  "version": "0.1.0",
  "seconds": 2.76,
  "timings": {"layout": 0.04, "det": 0.075, "leaders": 0.023, "rec": 2.61},
  "regions": [
    {"order": 0, "category": "text", "kind": "text", "bbox": [361, 287, 2187, 603],
     "text": "Điều 4. Để đảm bảo ...\n...",
     "lines": [{"bbox": [366, 290, 2185, 352], "text": "Điều 4. Để đảm bảo ...", "conf": 0.97}],
     "html": null},
    {"order": 6, "category": "image", "kind": "picture", "bbox": [1587, 1571, 2304, 2031], "text": "", "lines": [], "html": null}
  ]
}
```

- `page` = the uploaded file name without extension (null if none).
- `kind`: `text | title | header | footer | list | table | picture | seal | other`. The stamp + signature comes out
  as `picture` (layout label `image`). `category` is the raw layout label, or `orphan_line` (a detected line outside
  every layout region) / `dot_leader` (a `……` blank outside every region).
- `lines[].quad`: the detected line as 4 points `[[x,y] tl, tr, br, bl]` (tilted on skewed scans; `bbox` is its
  axis-aligned box). Absent on `……` leader lines.
- `lines[].conf`: VietOCR probability 0-1; `null` for `……` leader lines. Leader boxes are the dots' own extent (a few
  px high) - use the neighbouring label's line height for font size.
- `html` is always null (no table structure; table cell text is in `lines`).

Errors (JSON `{"detail": "..."}`, never containing document text):

| Code | When |
|-|-|
| 400 | empty or unreadable image |
| 401 | `DOCPARSE_TOKEN` set and the bearer token is missing/wrong |
| 413 | upload > `DOCPARSE_MAX_UPLOAD_MB` or image > `DOCPARSE_MAX_MPIX` |
| 503 | models still loading, or queue full - has `Retry-After` |
| 504 | not done within `DOCPARSE_TIMEOUT_S` (queue wait included) |
| 500 | internal error (exception type only) |

One page is parsed at a time (GPU); up to `DOCPARSE_MAX_QUEUE` more wait. Each request logs one line: image size,
region/line/leader counts, timings, status. Recognised text and images are never logged.

## Config

Environment variables, or `.env` in this folder (copy `.env.example`; real environment variables win).

| Key | Default | |
|-|-|-|
| `DOCPARSE_HOST` | `0.0.0.0` | `127.0.0.1` for this machine only |
| `DOCPARSE_PORT` | `8126` | |
| `DOCPARSE_TOKEN` | empty | bearer token for `/parse`; empty = no auth |
| `DOCPARSE_DEVICE` | `gpu:0` | `gpu:N` or `cpu` for all three models (unless `DOCPARSE_REC_DEVICE` is set) |
| `DOCPARSE_REC_DEVICE` | empty | VietOCR only; empty = same as `DOCPARSE_DEVICE`. `DOCPARSE_DEVICE=cpu` + `DOCPARSE_REC_DEVICE=gpu:0`: ~0.5 GB VRAM instead of ~1.7 GB, ~7-8 s/page (CPU line detection ~4.6 s) |
| `DOCPARSE_REC_MODEL` | `vgg_transformer` | `vgg_seq2seq` = fast mode: 3.2% WER, ~0.7 s/page |
| `DOCPARSE_MODEL_DIR` | `./models` | Paddle models in `paddlex/`, VietOCR weights in `vietocr/` |
| `DOCPARSE_MAX_QUEUE` | `8` | waiting requests beyond the running one |
| `DOCPARSE_TIMEOUT_S` | `60` | per request, queue wait included |
| `DOCPARSE_MAX_UPLOAD_MB` | `25` | |
| `DOCPARSE_MAX_MPIX` | `40` | an A4 page at 300 dpi is 8.7 MP |
| `DOCPARSE_LAYOUT_MIN_SCORE` | `0.5` | layout regions below are ignored |
| `CUDA_VISIBLE_DEVICES` | - | standard; on 76 it must be `0` (set in the systemd unit) |

The service sets `FLAGS_allocator_strategy=auto_growth` (Paddle must not grab the shared card),
`PADDLE_PDX_CACHE_HOME=<MODEL_DIR>/paddlex` and `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` itself.

## Install

Needs [uv](https://docs.astral.sh/uv/) and an NVIDIA driver with CUDA >= 12.8 (Blackwell). No CUDA Toolkit: the
torch and paddle wheels bring their own CUDA libraries. Python 3.12 (uv fetches it). ~9 GB of packages
(torch ~5 GB, paddle ~3 GB) - keep the uv cache on a drive with room (`UV_CACHE_DIR`).

```bash
uv sync --frozen --no-dev                       # --no-dev skips pytest/httpx
uv run --frozen python scripts/download_models.py   # once, needs internet: ~360 MB into ./models
uv run --frozen python -m app.main
```

After `download_models.py` the service needs no network: VietOCR's configs are vendored in `app/vietocr_config/`
(upstream fetches them from vocr.vn on every start) and the Paddle model-source check is off.
`download_models.py --all-rec` also fetches the seq2seq weights; `--from-cache DIR` copies from an existing cache
(`DIR/paddlex/official_models/<model>`, `DIR/vietocr/<model>.pth`) instead of downloading.

**Windows**: `run-docparse-server.bat` does all three steps. Paddle and torch ship same-named CUDA DLLs of different
versions; `app/cuda_preload.py` loads the right ones first (`docs/clash-fix.md`). It runs at the top of
`app/main.py` and must stay before any paddle/torch/cv2 import.

**Linux (172.16.20.76)**: clone to `/home/atoma/Atoma/docparse-vietocr`, run the first two commands above as
`atoma`, then `bash linux-service.sh` (installs and starts the systemd unit
`docparse-vietocr`, `CUDA_VISIBLE_DEVICES=0`). Logs: `journalctl -u docparse-vietocr -f`.

### Linux risk (not tested - no Linux box from here)

On Windows torch bundles its CUDA DLLs and only cuDNN had to be aligned. On Linux torch 2.11 cu128 and
paddlepaddle-gpu 3.2.1 cu129 both depend on `nvidia-*-cu12` pip packages with **conflicting exact pins** (torch via
`cuda-toolkit==12.8.1`: cuBLAS 12.8, cuDNN 9.19, NCCL 2.28; paddle: cuBLAS 12.9, cuDNN 9.9, NCCL 2.27, cuFile 1.14,
...). The resolver would refuse the pair, so `pyproject.toml` `override-dependencies` picks one version for each:
the newer one (12.9.x CUDA libraries, cuDNN 9.19, NCCL 2.28.9). `uv lock` resolves for Linux with these, but
nobody has run it there. If it fails, in order:

1. `uv sync` fails or a library errors on import: adjust the overrides (newer 12.x builds generally work for a
   12.8-built torch; check `ldd` of the failing `.so`).
2. Imports work but one framework crashes when both are loaded (symbol clash, as on Windows): run recognition in a
   child process. The recogniser is behind the `Recogniser` interface in `app/pipeline.py` (`Pipeline(s,
   recogniser=...)`); a spawn-based wrapper that owns `VietOcrRecogniser` and receives the crops over a pipe is a
   small change, and the parent process then never imports torch.
3. Then run `scripts/bench.py` on the box to confirm the same scores.

## Resources

RTX 5060 8 GB, 2480 px pages: ~2.5 s/page (layout 0.04 s, lines 0.07 s, leaders 0.03 s, recognition 1.6-2.6 s -
about 75 ms per text line). Peak VRAM ~1.7 GB on a full page (Paddle ~0.9 GB allocated / 1.2 GB reserved, torch
~0.5 GB). Model load ~7-10 s plus ~1 s warm-up. `vgg_seq2seq`: ~0.7 s/page, ~0.4 GB torch.

## Regression bench and tests

```bash
uv run python scripts/bench.py      # docparse-bench pages in-process -> out/bench/ + score.py --overlay
uv run pytest                       # unit + API tests; `-m "not gpu"` skips the two that load the real models
```

`bench.py` fails (exit 1) below the bar it was accepted at: ALL WER 2.5% +-0.2, fills form 9/12, p3 17/19, p4 2/2,
stamp 1/1 on p2, tables 2/2 on form, no extra pictures/tables. Bench folder: `--bench` or `DOCPARSE_BENCH_DIR`
(default `D:\python-projects\docparse-bench`). `DOCPARSE_SKIP_GPU_TESTS=1` skips the GPU tests.

Optional GPU lock (dev box only, off by default): with `DOCPARSE_GPU_LOCK=D:\python-projects\.gpu.lock`, `bench.py`
and the GPU tests create that file while they hold models and remove it afterwards; if it already exists they stop
(the tests skip). The service itself never uses it.

## Layout

```
app/main.py           FastAPI app, queue, limits, auth, logging; python -m app.main
app/config.py         settings from env/.env; Paddle env vars
app/pipeline.py       Pipeline (layout -> lines -> warp -> VietOCR -> leaders -> order), image decode, Recogniser
app/leaders.py        find_dot_leaders
app/order.py          resolve_overlaps, reading_order, sort_lines
app/geometry.py       box/quad helpers, warp_line
app/cuda_preload.py   Windows DLL preload (no-op elsewhere)
app/vietocr_config/   vendored VietOCR configs (base, vgg-transformer, vgg-seq2seq)
scripts/              download_models.py, bench.py, _common.py (optional GPU lock)
tests/                pytest
docs/                 bakeoff-results.md, clash-fix.md
```

Known limits: two-column signature blocks are read column by column; `……/……/……` date blanks that the detector boxes
together with text come out as junk; no table HTML; the stamp's printed text is left out of the page text (see
`include_dropped`).
