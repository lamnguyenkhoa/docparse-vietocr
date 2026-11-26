"""Region de-duplication, reading order and line order."""

from __future__ import annotations

from app.geometry import area, inter, overlap_1d


def resolve_overlaps(regions: list[dict]) -> list[dict]:
    """Drop a region fully (>=90%) inside a bigger one of the same kind."""
    keep = []
    for r in regions:
        drop = any(
            o is not r and o["kind"] == r["kind"] and area(o["bbox"]) > area(r["bbox"]) and inter(o["bbox"], r["bbox"]) >= 0.9 * area(r["bbox"])
            for o in regions
        )
        if not drop:
            keep.append(r)
    return keep


def reading_order(regions: list[dict]) -> list[dict]:
    """Bands of y-overlapping regions; inside a band, x-disjoint groups are columns (left before right).

    Known failure: two-column signature blocks where people read row by row come out column by column.
    """
    rs = sorted(regions, key=lambda r: r["bbox"][1])
    bands: list[list[dict]] = []
    for r in rs:
        y0, y1 = r["bbox"][1], r["bbox"][3]
        # same band only if it overlaps some band member by >30% of the shorter height (touching boxes don't count)
        if bands and any(
            overlap_1d(y0, y1, o["bbox"][1], o["bbox"][3]) > 0.3 * min(y1 - y0, o["bbox"][3] - o["bbox"][1]) for o in bands[-1]
        ):
            bands[-1].append(r)
        else:
            bands.append([r])
    out = []
    for band in bands:
        # columns = connected components of x-overlap
        cols: list[list[dict]] = []
        for r in sorted(band, key=lambda r: r["bbox"][0]):
            hit = [c for c in cols if any(overlap_1d(r["bbox"][0], r["bbox"][2], o["bbox"][0], o["bbox"][2]) > 0 for o in c)]
            if hit:
                merged = [x for c in hit for x in c] + [r]
                cols = [c for c in cols if c not in hit] + [merged]
            else:
                cols.append([r])
        for c in sorted(cols, key=lambda c: min(x["bbox"][0] for x in c)):
            out.extend(sorted(c, key=lambda r: r["bbox"][1]))
    return out


def sort_lines(lines: list[dict]) -> list[dict]:
    """Top->bottom; lines whose y-ranges overlap by >50% of the smaller height are one row (left->right)."""
    ls = sorted(lines, key=lambda l: (l["bbox"][1] + l["bbox"][3]) / 2)
    rows: list[list[dict]] = []
    for l in ls:
        if rows:
            last = rows[-1]
            ref = last[-1]["bbox"]
            b = l["bbox"]
            if overlap_1d(ref[1], ref[3], b[1], b[3]) > 0.5 * min(ref[3] - ref[1], b[3] - b[1]):
                last.append(l)
                continue
        rows.append([l])
    return [l for row in rows for l in sorted(row, key=lambda l: l["bbox"][0])]
