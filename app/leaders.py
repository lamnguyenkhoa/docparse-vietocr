"""Dotted fill-in leaders (……, .....): the text-line detector does not box them, so find them on the page."""

from __future__ import annotations

import cv2
import numpy as np


def find_dot_leaders(gray: np.ndarray, mask_boxes, pad: int = 3) -> list[list[int]]:
    """Rows of small, evenly spaced dots outside `mask_boxes` (the detected text lines) -> one box per run.

    Binarize (Otsu), blank the masked boxes, keep squarish 2-14 px blobs, group them by baseline (+-4 px), split
    runs at gaps > 40 px, keep runs of >= 6 dots, >= 50 px long, with regular spacing (gap std <= 0.6 * mean).
    """
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    for x0, y0, x1, y1 in mask_boxes:
        bw[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad] = 0
    n, _, st, cen = cv2.connectedComponentsWithStats(bw, 8)
    w, h, a = st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_AREA]
    ok = (w >= 2) & (h >= 2) & (w <= 14) & (h <= 14) & (a >= 4) & (np.maximum(w, h) <= 2.5 * np.minimum(w, h))
    idx = np.where(ok)[0] + 1
    dots = sorted(((cen[i][0], cen[i][1], st[i]) for i in idx), key=lambda d: d[1])
    rows: list[list] = []
    for d in dots:  # group by baseline
        if rows and abs(d[1] - np.mean([e[1] for e in rows[-1]])) <= 4:
            rows[-1].append(d)
        else:
            rows.append([d])
    out = []
    for row in rows:
        row.sort(key=lambda d: d[0])
        runs, cur = [], [row[0]]
        for d in row[1:]:
            if d[0] - cur[-1][0] <= 40:
                cur.append(d)
            else:
                runs.append(cur)
                cur = [d]
        runs.append(cur)
        for run in runs:
            if len(run) < 6 or run[-1][0] - run[0][0] < 50:
                continue
            gaps = np.diff([d[0] for d in run])
            if gaps.std() > 0.6 * gaps.mean():
                continue
            xs0 = min(d[2][0] for d in run)
            ys0 = min(d[2][1] for d in run)
            xs1 = max(d[2][0] + d[2][2] for d in run)
            ys1 = max(d[2][1] + d[2][3] for d in run)
            out.append([int(xs0), int(ys0), int(xs1), int(ys1)])
    return out
