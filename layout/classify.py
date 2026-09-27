"""Rule-based assignment of text lines to layout classes, then merging into regions.

Order of decisions
  1. filler    : text on the scanning background, English/Latin script, pencil
                 (low stroke contrast), or non-line-shaped decorative blobs
  2. side_text : lines outside the horizontal extent of the main text block, or
                 vertical (rotated) writing along a margin
  3. vertical grouping of remaining lines by whitespace; the dominant group is
     main_text, small groups above/below it (separated by a clear gap) are
     header / footer
  4. small marginal items (folio numbers, signatures) in top/bottom corners are
     re-assigned from side_text to header / footer
"""
from dataclasses import dataclass, field

import numpy as np

LABELS = ("header", "footer", "main_text", "side_text", "filler")


@dataclass
class Region:
    label: str
    bbox: list
    confidence: float
    lines: list = field(default_factory=list)
    reasons: list = field(default_factory=list)


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def _assign(ln, label, conf, reason):
    ln.feats["label"] = label
    ln.feats["conf"] = round(_clip01(conf), 3)
    ln.feats["reason"] = reason


def classify_lines(lines, page_bbox, char_h, cfg):
    if not lines:
        return lines
    px1, py1, px2, py2 = page_bbox
    page_h = max(1, py2 - py1)

    page_lines = [l for l in lines if l.source == "page"]
    contrasts = [l.feats.get("contrast", 0) for l in page_lines if l.w > 3 * l.h]
    med_contrast = float(np.median(contrasts)) if contrasts else 0.0
    hs = [l.h for l in page_lines if l.w > 3 * l.h] or [l.h for l in lines]
    line_h = float(np.median(hs))

    # ---------- 0. specks smaller than a glyph carry no layout information ----------
    lines[:] = [l for l in lines if max(l.w, l.h) >= 0.7 * char_h or l.source == "outside"]

    # ---------- 0b. faint blobs touching the page edge are scan artefacts ----------
    def _edge_artefact(l):
        if l.source != "page" or med_contrast <= 0:
            return False
        e = max(3, 0.6 * line_h)  # generous: deskewing shifts edge artefacts inward
        touches = l.bbox[0] <= px1 + e or l.bbox[1] <= py1 + e or l.bbox[2] >= px2 - e or l.bbox[3] >= py2 - e
        faint = l.feats.get("contrast", med_contrast) < 0.45 * med_contrast
        thin_strip = l.h < 0.5 * line_h and l.w > 6 * l.h
        return touches and (faint or thin_strip)
    lines[:] = [l for l in lines if not _edge_artefact(l)]

    # ---------- 1. filler ----------
    body = []
    for ln in lines:
        f = ln.feats
        if ln.source == "outside":
            _assign(ln, "filler", 0.9, "text outside manuscript substrate")
        elif f.get("latin", 0) >= cfg.latin_threshold:
            _assign(ln, "filler", 0.5 + 0.5 * f["latin"], "English/Latin script")
        elif med_contrast > 0 and f.get("contrast", med_contrast) < cfg.pencil_contrast_ratio * med_contrast \
                and ln.h > 0.35 * line_h:  # thin strips are descender fragments, not pencil
            r = f["contrast"] / med_contrast
            _assign(ln, "filler", 0.55 + 0.4 * (1 - r / cfg.pencil_contrast_ratio), "low-contrast (pencil/faint) strokes")
        elif ln.h > 3 * line_h and ln.w > 3 * line_h and ln.ink_px / max(1, ln.w * ln.h) < 0.12 \
                and ln.w < 3 * ln.h:
            _assign(ln, "filler", 0.5, "large non-text-like blob (decoration)")
        else:
            body.append(ln)
    if not body:
        return lines

    # ---------- 2. side_text by horizontal extent ----------
    widths = np.array([l.w for l in body], float)
    p90 = np.percentile(widths, 90)
    core = [l for l in body if l.w >= cfg.core_width_frac * p90 and l.w > 3 * l.h] or body
    core_l, core_r = _core_extent(core, px1, px2)
    tol = cfg.side_margin_tol * char_h + 0.02 * (core_r - core_l)

    rest = []
    for ln in body:
        x1, _, x2, _ = ln.bbox
        outside = max(0.0, core_l - tol - x1) if ln.cx < core_l else 0.0
        outside = max(outside, max(0.0, x2 - (core_r + tol))) if ln.cx > core_r else outside
        frac_out = outside / max(1, ln.w)
        vertical = ln.h > 2.5 * ln.w and ln.h > 3 * line_h
        if vertical:
            _assign(ln, "side_text", 0.75, "vertical writing in margin")
        elif (ln.cx < core_l - tol or ln.cx > core_r + tol) or frac_out > 0.6:
            dist = max(core_l - ln.cx, ln.cx - core_r)
            _assign(ln, "side_text", 0.55 + 0.4 * _clip01(dist / (4 * char_h)), "outside main text column")
        else:
            rest.append(ln)
    if not rest:
        return lines

    # ---------- 3. vertical grouping -> main / header / footer ----------
    rest.sort(key=lambda l: l.bbox[1])
    cys = sorted(l.cy for l in rest if l.w > 3 * l.h)
    diffs = np.diff(cys) if len(cys) > 1 else np.array([])
    diffs = diffs[diffs > 0.6 * line_h]
    pitch = float(np.median(diffs)) if diffs.size else 1.6 * line_h
    gap_thr = max(0.5 * line_h, cfg.gap_pitch_factor * pitch - line_h)

    groups, cur, cur_bottom = [], [rest[0]], rest[0].bbox[3]
    for ln in rest[1:]:
        if ln.bbox[1] - cur_bottom > gap_thr:
            groups.append(cur)
            cur, cur_bottom = [ln], ln.bbox[3]
        else:
            cur.append(ln)
            cur_bottom = max(cur_bottom, ln.bbox[3])
    groups.append(cur)

    mass = [sum(l.w for l in g) for g in groups]
    mi = int(np.argmax(mass))
    main_top = min(l.bbox[1] for l in groups[mi])
    main_bot = max(l.bbox[3] for l in groups[mi])

    for gi, g in enumerate(groups):
        g_top, g_bot = min(l.bbox[1] for l in g), max(l.bbox[3] for l in g)
        small = _count_rows(g) <= 3 and (g_bot - g_top) < 0.25 * page_h
        if gi < mi and small and (g_bot - py1) < 0.4 * page_h:
            gap = main_top - g_bot
            conf = 0.55 + 0.35 * _clip01((gap / pitch - 0.5) / 2)
            for ln in g:
                _assign(ln, "header", conf + (0.05 if ln.w < 0.5 * (core_r - core_l) else 0),
                        "separated block above main text")
        elif gi > mi and small and (py2 - g_top) < 0.4 * page_h:
            gap = g_top - main_bot
            conf = 0.55 + 0.35 * _clip01((gap / pitch - 0.5) / 2)
            for ln in g:
                _assign(ln, "footer", conf + (0.05 if ln.w < 0.5 * (core_r - core_l) else 0),
                        "separated block below main text")
        else:
            for ln in g:
                full = _clip01(ln.w / max(1, core_r - core_l))
                _assign(ln, "main_text", 0.7 + 0.25 * full, "main text block")

    # ---------- 4. small corner items -> header/footer ----------
    for ln in lines:
        if ln.feats.get("label") == "side_text" and ln.w < 4 * line_h:
            if ln.bbox[3] <= main_top:
                _assign(ln, "header", 0.6, "small mark in top margin (folio/number)")
            elif ln.bbox[1] >= main_bot:
                _assign(ln, "footer", 0.6, "small mark in bottom margin (page no./catchword)")
    # ---------- 5. small fragments inherit the label of their row neighbour ----------
    labelled = [l for l in lines if "label" in l.feats]
    for ln in sorted(labelled, key=lambda l: -l.w):  # larger fragments settle first
        if ln.w >= 2 * line_h or ln.source != "page":
            continue
        best, best_gap = None, 2.0 * char_h
        for b in labelled:
            if b is ln or b.source != ln.source or b.w < 1.5 * ln.w:
                continue
            ov = min(ln.bbox[3], b.bbox[3]) - max(ln.bbox[1], b.bbox[1])
            if ov < 0.5 * ln.h:
                continue
            gap = max(ln.bbox[0], b.bbox[0]) - min(ln.bbox[2], b.bbox[2])
            if gap < best_gap:
                best, best_gap = b, gap
        if best is not None and best.feats["label"] != ln.feats["label"]:
            _assign(ln, best.feats["label"], min(best.feats["conf"], 0.7), "fragment adjacent to " + best.feats["label"])
    return lines


def _core_extent(core, px1, px2):
    """Horizontal extent of the text block(s) from an x-coverage histogram of the
    main lines. Unlike a median of line edges this spans every column of a
    multi-column page, while a few marginal notes cannot stretch it."""
    x0 = int(min(px1, min(l.bbox[0] for l in core)))
    x1 = int(max(px2, max(l.bbox[2] for l in core)))
    cov = np.zeros(x1 - x0 + 1, float)
    for l in core:
        cov[l.bbox[0] - x0: l.bbox[2] - x0] += 1
    rows = max(1.0, cov.max())
    xs = np.nonzero(cov >= 0.25 * rows)[0]
    if xs.size == 0:
        return float(min(l.bbox[0] for l in core)), float(max(l.bbox[2] for l in core))
    return float(x0 + xs[0]), float(x0 + xs[-1])


def _count_rows(group):
    """Number of distinct text rows (fragments overlapping vertically count once)."""
    rows = []
    for ln in sorted(group, key=lambda l: l.cy):
        if rows and ln.bbox[1] < rows[-1][1] - 0.4 * ln.h:
            rows[-1][1] = max(rows[-1][1], ln.bbox[3])
        else:
            rows.append([ln.bbox[1], ln.bbox[3]])
    return len(rows)


def _expand(b, dx, dy):
    return [b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy]


def _intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def merge_into_regions(lines, char_h, pitch, cfg):
    """Union line boxes of the same label that are close to each other."""
    dx = cfg.merge_dx_char * char_h
    dy = cfg.merge_dy_pitch * pitch / 2.0
    regions = []
    for label in LABELS:
        group = [l for l in lines if l.feats.get("label") == label]
        clusters = [[l] for l in group]
        boxes = [list(l.bbox) for l in group]
        changed = True
        while changed:
            changed = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    # filler / header / footer items on different rows stay separate
                    ddy = dy if label in ("main_text", "side_text") else dy * 0.5
                    # columns of main text stay separate; scattered notes/marks group more loosely
                    ddx = dx if label == "main_text" else 2.0 * dx
                    if _intersects(_expand(boxes[i], ddx, ddy), _expand(boxes[j], ddx, ddy)):
                        boxes[i] = [min(boxes[i][0], boxes[j][0]), min(boxes[i][1], boxes[j][1]),
                                    max(boxes[i][2], boxes[j][2]), max(boxes[i][3], boxes[j][3])]
                        clusters[i] += clusters[j]
                        del boxes[j], clusters[j]
                        changed = True
                        break
                if changed:
                    break
        for b, c in zip(boxes, clusters):
            w = np.array([max(1, l.w) for l in c], float)
            conf = float(np.sum(w * np.array([l.feats["conf"] for l in c])) / w.sum())
            reasons = sorted({l.feats["reason"] for l in c})
            regions.append(Region(label, b, round(conf, 3), c, reasons))
    regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
    return regions


def estimate_pitch(lines, line_h):
    ls = [l for l in lines if l.feats.get("label") == "main_text" and l.w > 3 * l.h]
    cys = sorted(l.cy for l in ls)
    d = np.diff(cys) if len(cys) > 1 else np.array([])
    d = d[d > 0.6 * line_h]
    return float(np.median(d)) if d.size else 1.6 * line_h
