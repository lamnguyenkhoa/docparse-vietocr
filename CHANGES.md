# Changes: bake-off folder -> service (0.1.0)

## Moved

| From | To |
|-|-|
| `run.py` geometry helpers, `warp_line` | `app/geometry.py` |
| `run.py` `find_dot_leaders` | `app/leaders.py` |
| `run.py` `resolve_overlaps`, `reading_order`, `sort_lines` | `app/order.py` |
| `run.py` `Pipeline`, `VietOcrRec`, label->kind map | `app/pipeline.py` (`Pipeline`, `VietOcrRecogniser` behind a `Recogniser` interface) |
| `cuda_preload.py` | `app/cuda_preload.py` (logic unchanged) |
| `env.sh` | `.env.example` (service settings; uv cache location is now a README note) |
| `requirements.txt`, `requirements.lock.txt` | `pyproject.toml` + `uv.lock` |
| `RESULTS.md`, `CLASH-FIX.md` | `docs/bakeoff-results.md`, `docs/clash-fix.md` (unchanged) |

## Removed

- Variants and their code: `ppocr-rec` / `PaddleRec` / `PPOCR_REC_MODEL` (PaddleOCR's recognisers lack Vietnamese
  letters), `--no-warp` / `axis_crop` (ablation), `--layout`, the `--variant` CLI, warm-up/VRAM printing in `main()`.
  `vgg_seq2seq` stays as config (`DOCPARSE_REC_MODEL`), the paddle device as `DOCPARSE_DEVICE`.
- Hard-coded `D:\python-projects\...` paths in the app (only `scripts/` and `tests/` default to the dev-box bench
  folder, overridable with `--bench` / `DOCPARSE_BENCH_DIR`).
- The GPU lock is opt-in for dev scripts/tests (`DOCPARSE_GPU_LOCK=<file>`; unset = off) and not used by `app/`.
- The runtime fetch of VietOCR configs from vocr.vn (vendored in `app/vietocr_config/`) and the weight download
  into `%TEMP%` (weights come from `DOCPARSE_MODEL_DIR`, filled by `scripts/download_models.py`).
- Region `score` and the `stats` block from the output (stats are logged as counts instead).
  `dropped_lines` only with `?include_dropped=true`.

## Behaviour

Recognition is unchanged: `scripts/bench.py` gives ALL WER 2.5% (accent-blind 2.1%), fills 9/12, 17/19, 2/2,
stamp 1/1, tables 2/2, no extras, and the page text is byte-identical to the last `run.py --variant vietocr` output.

## Packaging notes

- `pyproject.toml` declares vietocr's real inference deps via `[[tool.uv.dependency-metadata]]` (no more
  `--no-deps`; no imgaug/albumentations/second OpenCV/Pillow 10).
- `override-dependencies`: cuDNN 9.19.0.56 (paddle pins 9.9) and, for Linux, one version per `nvidia-*-cu12`
  package where torch and paddle pin different ones. `uv lock` resolves for win32 + linux; `.venv` was recreated
  from `uv sync` on Windows and passes the bench.

## Not done / open

- Linux is untested (no access to 172.16.20.76 from here). The lock resolves for Linux, but whether paddle and torch
  then load together in one process is unknown - see README "Linux risk"; the child-process fallback is not built.
- Startup VRAM log is measured after the small warm-up page, so it under-reports (~0.8 GB); a full page peaks at
  ~1.7 GB.
