# Results: PP-DocLayout + PP-OCRv5 line detection + VietOCR

Pipeline: `PP-DocLayout_plus-L` (layout) → `PP-OCRv5_server_det` on the full page (line quads) → each quad
perspective-warped to a straight strip with ~18% vertical padding → recogniser. Lines go to the smallest layout
region containing their centre. A page-level dot-leader finder adds the `……` blanks (see Q4). Run:
`python run.py --variant vietocr|vietocr-s2s|ppocr-rec [--pages <glob>] [--no-warp] [--cpu-paddle]`.

**Short answer:** VietOCR `vgg_transformer` reads this scan as well as the current Tesseract + qwen3-vl merge:
1.2% vs 1.7% WER on vanbangoc p1–p3, 2.1% vs 1.7% on p1–p4. It also gives a tight box for every line, finds the
stamp + signature as one picture, and finds both tables, at ~2.5 s/page and ~1.7 GB of VRAM. PaddleOCR's own
recognisers cannot write Vietnamese at all.

## 1. Scores

`vietocr` (variant A, recommended):

| page | wer | wer_no_accents | fills | pictures | tables | regions | line_boxes | seconds |
|-|-|-|-|-|-|-|-|-|
| form_p1 | 8.0 | 7.0 | 9/12 | 0/0 (+0 extra) | 2/2 (+0 extra) | 24 | 30/30 | 1.77 |
| vanbangoc_p1 | 0.5 | 0.5 | 0/0 | 0/0 (+0 extra) | 0/0 (+0 extra) | 14 | 32/32 | 2.44 |
| vanbangoc_p2 | 2.8 | 2.4 | 0/0 | 1/1 (+0 extra) | 0/0 (+0 extra) | 8 | 31/31 | 3.04 |
| vanbangoc_p3 | 0.7 | 0.3 | 17/19 | 0/0 (+0 extra) | 0/0 (+0 extra) | 27 | 49/49 | 2.91 |
| vanbangoc_p4 | 5.0 | 4.3 | 2/2 | 0/0 (+0 extra) | 0/0 (+0 extra) | 9 | 28/28 | 1.99 |

ALL: WER 2.5% (accent-blind 2.1%)

`vietocr-s2s` (variant B):

| page | wer | wer_no_accents | fills | pictures | tables | regions | line_boxes | seconds |
|-|-|-|-|-|-|-|-|-|
| form_p1 | 4.0 | 3.0 | 8/12 | 0/0 (+0 extra) | 2/2 (+0 extra) | 24 | 30/30 | 0.62 |
| vanbangoc_p1 | 1.5 | 1.3 | 0/0 | 0/0 (+0 extra) | 0/0 (+0 extra) | 14 | 32/32 | 0.71 |
| vanbangoc_p2 | 3.5 | 2.4 | 0/0 | 1/1 (+0 extra) | 0/0 (+0 extra) | 8 | 31/31 | 0.94 |
| vanbangoc_p3 | 3.4 | 2.3 | 14/19 | 0/0 (+0 extra) | 0/0 (+0 extra) | 27 | 49/49 | 0.87 |
| vanbangoc_p4 | 5.0 | 4.3 | 1/2 | 0/0 (+0 extra) | 0/0 (+0 extra) | 9 | 28/28 | 0.65 |

ALL: WER 3.2% (accent-blind 2.5%)

`ppocr-rec` (variant C, `PP-OCRv6_medium_rec`, which is what `PaddleOCR(lang="vi")` resolves to in paddleocr 3.7):

| page | wer | wer_no_accents | fills | pictures | tables | regions | line_boxes | seconds |
|-|-|-|-|-|-|-|-|-|
| form_p1 | 49.0 | 24.0 | 11/12 | 0/0 (+0 extra) | 2/2 (+0 extra) | 24 | 30/30 | 0.41 |
| vanbangoc_p1 | 48.2 | 42.5 | 0/0 | 0/0 (+0 extra) | 0/0 (+0 extra) | 14 | 32/32 | 0.55 |
| vanbangoc_p2 | 55.6 | 46.2 | 0/0 | 1/1 (+0 extra) | 0/0 (+0 extra) | 8 | 31/31 | 0.48 |
| vanbangoc_p3 | 52.3 | 46.3 | 15/19 | 0/0 (+0 extra) | 0/0 (+0 extra) | 27 | 49/49 | 0.51 |
| vanbangoc_p4 | 59.9 | 52.5 | 1/2 | 0/0 (+0 extra) | 0/0 (+0 extra) | 9 | 28/28 | 0.42 |

ALL: WER 53.2% (accent-blind 44.8%). The older `latin_PP-OCRv5_mobile_rec` scores 52.7% (`out/ppocr-rec-latin-v5`).
**Both dictionaries lack ộ ủ ữ ạ ị ế ề ở**, so every word that has one loses that letter ("Hà Ni", "QUYÉT ĐNH"). This
is a missing-character problem, not a tuning one. The 11/12 fills on form_p1 are its only win.

Vanbangoc pages only, to compare with the baselines (WER / accent-blind):

| method | p1–p3 | p1–p4 |
|-|-|-|
| baselines/tesseract | 4.4% / 3.8% | 4.0% / 3.1% |
| baselines/current-merge | 1.7% / 1.2% | 1.7% / 1.0% |
| **vietocr** | **1.2% / 1.0%** | 2.1% / 1.8% |
| vietocr-s2s | 2.7% / 2.0% | 3.2% / 2.5% |
| vietocr-nowarp (ablation) | 2.5% / 1.4% | 4.2% / 2.5% |
| ppocr-rec (PP-OCRv6) | 51.6% / 44.8% | 53.5% / 46.5% |

What drives p4's 5.0% is mostly not recognition. The two-column signature block (`TL.BỘ TRƯỞNG…` |
`TỔNG GIÁM ĐỐC…`) comes out column by column while the truth interleaves the rows, and the page number `4` is
emitted as a `header` region (`number` label) that the truth does not have. Real misreads that remain are few:
`hoàn trà`, `để`→`đề`, `bổ`→`bố`, and `Nơi nhận` + stamp bleed → `Nơi nhận Đấ`.

Reading-order rule (band = regions overlapping in y by >30% of the shorter one; inside a band, x-disjoint groups are
columns, left first). It fails on two-column signature blocks where the truth reads row by row (p4). It also needs
the 30% threshold: an earlier "any overlap" version put `Họ và tên` before `Kính gửi` on form_p1 because the two
boxes touched, which pushed form_p1 to 18% WER.

## 2. Stamp + signature (vanbangoc_p2)

- Layout label: **`image`** (score 0.93), not `seal`. Box `[1587,1571,2304,2031]` vs expected `[1580,1580,2310,2045]`,
  so it scores 1/1 with no extras. The region covers the round stamp, the signature and most of its long tail; the
  tail runs a few px past the right edge.
- The printed `K.T BỘ TRƯỞNG / THỨ TRƯỞNG` inside the stamp is detected as lines and read correctly
  (`KT BỘ TRƯỞNG`, `THỨ TRƯỜNG`), plus one junk line `1` from the stamp. All three are dropped from text and kept
  in the region's `dropped_lines` for inspection. Dropping them costs the 4 truth words (≈1 point of p2 WER).
  I'd keep it this way: in the DOCX that text is part of the stamp picture, and keeping it would also bring in the
  junk. A builder that wants it can read `dropped_lines` with conf ≥ 0.9.
- `Trần Xuân Hà` below the stamp is a separate `figure_title` region → mapped to `text` and read correctly.

## 3. Line boxes

Tight, one box per printed line in body text. Lines split only where the page really has a gap: label and value
fields on forms (`Họ và tên:` | dots | `Giới tính:`) and the two-column headers. I saw no merged lines.
Line counts are close to truth (p1 32 vs 31, p2 34 vs 33 incl. 3 stamp lines).
Accents survive: accent-blind WER is only 0.4 points below full WER overall. The 18% vertical padding on the warped
crop keeps the stacked marks (ẫ, ệ, ở). On p1 the only error in 31 lines is one dash.

Lines outside every layout region become their own `text` region (`orphan_line`): form_p1 12/22, p3 8/32, p1 1,
p4 1, p2 0. They are mostly short form labels (`Kính gửi:`, `Trụ sở:`) that PP-DocLayout leaves unboxed. Reading
order still works because each orphan is its own band.

## 4. Dotted blanks

The detector **does not box dot leaders at all**: line boxes stop where the dots start (`out/vietocr/overlay/form_p1.png`).
So a per-crop "mostly dots → `……`" rule never triggers. I tried one: it fired 0 times on warped crops, and on
axis-aligned crops it replaced 9–14 real text lines with `……`. I removed it.

What works instead is `find_dot_leaders()`:
1. Binarize the page (Otsu) and blank out the detected line boxes and the table/picture regions.
2. Keep blobs 2–14 px with a squarish aspect.
3. Group them by baseline (±4 px) and split runs at gaps over 40 px.
4. Keep runs of ≥6 dots, ≥50 px long, with regular spacing (std/mean of gaps ≤ 0.6).

Each run becomes a `……` line with its own box, in the containing region or as a `dot_leader` region.
Fills went from 2/12 → 9/12 on form_p1, 5/19 → 17/19 on p3, and stayed 2/2 on p4, with no false positives
(0 on p1/p2).

Still lost: when the detector boxes the dots together with text, VietOCR reads them as `....` (counted as a fill),
`-`, or junk. Examples: `Ngày sinh: ……/……/…………` → `Ngày sinh: - 1.00.00.00.00.00.000` and `năm 20……` →
`năm 20.`. The leader boxes are the dots' own extent (a few px high), not a text-line height. The DOCX builder
should use the neighbouring label's line height for them.

## 5. Tables (form_p1)

Both ruled tables come out as `table` regions (2/2, no extras). The empty cells just have no lines. Lines inside the
table are the header cells (`STT`, `Họ và tên`, `Quan hệ`, `Năm sinh`, `Số điện thoại`, `Email`), sorted
row-then-column. `html` is null; the paddleocr table pipeline was not added.

## 6. Tilt (~1.2°)

Quad warping helps clearly. The `--no-warp` ablation (axis-aligned crop, same padding) on vanbangoc p1–p4 gives
4.2% WER vs 2.1% warped. Most of the loss is on p2 (7.3% vs 2.8%) and p4 (10.3% vs 5.0%), where long tilted lines
give axis boxes with a neighbour line's ascenders/descenders in them, and accents are the first to go (accent-blind
gap 3.8 vs 7.3). p1/p3 hardly change.

## 7. Speed and memory

Warm, per page, RTX 5060 (layout / detection / recognition, total):

| variant | layout | det | rec | total |
|-|-|-|-|-|
| vietocr (vgg_transformer) | 0.04 s | 0.07 s | 1.6–2.6 s | 1.8–3.0 s |
| vietocr-s2s | 0.04 s | 0.07 s | 0.44–0.76 s | 0.6–0.9 s |
| ppocr-rec (PP-OCRv6) | 0.04 s | 0.07 s | ~0.35 s | 0.4–0.55 s |
| vietocr, paddle on CPU | 0.5 s | 4.6 s | 1.6–3.1 s | 6.8–8.3 s |

Recognition dominates. VietOCR decodes greedily per width-bucket (`predict_batch`), so it costs ~70 ms/line.
Model load takes ~7 s; the first page also pays CUDA warm-up, so `run.py` does one warm-up page that is not timed.

Peak VRAM: Paddle layout + det 869 MiB allocated / 1227 MiB reserved, VietOCR 463 MiB (s2s 402 MiB), so ~1.7 GB
for the whole stack. PP-OCRv6 rec raises Paddle to 1213 / 2388 MiB.

CPU vs GPU for the Paddle parts: detection at full 2480 px is 65× slower on CPU (4.6 s vs 0.07 s). Layout is fine
on CPU (0.5 s). The Paddle GPU wheel is worth it.

## 8. Setup on Windows (what worked)

Python 3.12 (uv). `paddlepaddle-gpu==3.2.1` (cu129 index), `paddleocr==3.7.0` / `paddlex==3.7.2`,
`torch==2.11.0+cu128` / `torchvision==0.26.0+cu128`, `vietocr==0.3.13`. No CUDA Toolkit; the wheels bring their
runtimes. Full list in `requirements.lock.txt`; install order in `requirements.txt`.

Pain points:
- **Paddle + torch in one process.** Both ship `cublas64_12.dll`, `cudnn*64_9.dll`, etc. under the same names, and
  the first one loaded wins. Torch first breaks Paddle (WinError 127 in cublas: 12.8 vs 12.9). Paddle first breaks
  torch (cudnn_cnn64_9: 9.9 vs 9.19). Fix:
  1. Upgrade `nvidia-cudnn-cu12` to **9.19.0.56** to match torch's bundled cuDNN.
  2. `cuda_preload.py` loads the newer nvidia-* 12.9 DLLs explicitly, then imports torch before paddle (for
     `libiomp5md.dll`).

  It must run before cv2/paddle/torch are imported. The fallback, if this ever breaks, is VietOCR in a subprocess.
- **vietocr deps.** A plain `pip install vietocr` downgrades Pillow to 10.2, adds a second `opencv-python` next to
  paddle's `opencv-contrib-python`, and pulls training-only packages (imgaug, albumentations). Use `--no-deps` plus
  `einops gdown`. It works with torch 2.11 (state-dict load is fine under `weights_only`).
- **vietocr weights land in `%TEMP%`** (C:). `run.py` downloads them to `D:\python-projects\.cache\vietocr` and
  points `cfg['weights']` there. Configs are still fetched from vocr.vn on each start, so the first run needs
  internet.
- All other caches are on D: (`env.sh`).

## Overlays

- `out/vietocr/overlay/vanbangoc_p2.png`: the stamp + signature `picture` region and `Trần Xuân Hà` as text.
- `out/vietocr/overlay/form_p1.png`: both tables, label lines split from the dot leaders, orphan label regions.
- `out/vietocr/overlay/vanbangoc_p3.png`: dot-leader boxes on the fill-in lines and tight body line boxes.
