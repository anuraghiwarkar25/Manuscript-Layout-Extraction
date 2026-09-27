"""Script-agnostic text-line detection from a binary ink map."""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Line:
    bbox: list                  # [x1, y1, x2, y2] in working coordinates
    source: str = "page"        # 'page' (on substrate) or 'outside' (on scanning background)
    ink_px: int = 0
    rect: tuple = None          # cv2.minAreaRect of the blob, for skew estimation
    feats: dict = field(default_factory=dict)

    @property
    def w(self): return self.bbox[2] - self.bbox[0]
    @property
    def h(self): return self.bbox[3] - self.bbox[1]
    @property
    def cx(self): return (self.bbox[0] + self.bbox[2]) / 2.0
    @property
    def cy(self): return (self.bbox[1] + self.bbox[3]) / 2.0


def clean_components(ink, min_px, max_h=None):
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        if a < min_px:
            continue
        if max_h is not None and hh > max_h and ww < hh * 0.2:
            continue  # tall thin stroke: ruling / crack
        keep[i] = True
    out = keep[lbl].astype(np.uint8) * 255
    return out, stats[1:][keep[1:]]


def estimate_char_height(stats, default=20.0, img_shape=None):
    """Area-weighted median height of plausible glyph/word components.

    Weighting by area makes the estimate insensitive to the many tiny specks that
    palm-leaf fibres, stains and bleed-through produce."""
    if stats is None or len(stats) == 0:
        return default
    hs = stats[:, cv2.CC_STAT_HEIGHT].astype(float)
    ws = stats[:, cv2.CC_STAT_WIDTH].astype(float)
    ar = stats[:, cv2.CC_STAT_AREA].astype(float)
    ok = (hs >= 4) & (ws >= 3) & (ws / hs < 25) & (hs / ws < 6)
    if img_shape is not None:
        ok &= (hs < 0.25 * img_shape[0]) & (ws < 0.5 * img_shape[1])
    if ok.sum() < 5:
        return default
    hs, ar = hs[ok], ar[ok]
    order = np.argsort(hs)
    cw = np.cumsum(ar[order])
    return float(hs[order][np.searchsorted(cw, cw[-1] / 2)])


def drop_specks(ink, char_h):
    """Remove components far smaller than a character (fibre texture, dust) and
    tall thin strokes touching the image border (scan edges, rulings)."""
    H, W = ink.shape
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    w, h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    a = stats[:, cv2.CC_STAT_AREA]
    keep = (np.maximum(h, w) >= 0.3 * char_h) & (a >= (0.12 * char_h) ** 2)
    border = (x <= 1) | (y <= 1) | (x + w >= W - 1) | (y + h >= H - 1)
    keep &= ~(border & ((h > 3 * w) | (w > 3 * h)) & (np.maximum(h, w) > 2 * char_h))
    keep[0] = False
    return keep[lbl].astype(np.uint8) * 255


def _strip_autocorr(col_ink, char_h):
    prof = col_ink.sum(axis=1).astype(float)
    nz = np.nonzero(prof)[0]
    if nz.size < 3 * char_h:
        return None
    prof = prof[nz[0]:nz[-1] + 1]
    k = max(3, int(3 * char_h)) | 1
    trend = np.convolve(prof, np.ones(k) / k, mode="same")
    prof = prof - trend                      # remove slow variation (margins, stains)
    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:]
    return ac / ac[0] if ac[0] > 0 else None


def estimate_line_pitch(ink, char_h):
    """Line spacing from the autocorrelation of row-ink profiles.

    Profiles are computed in narrow vertical strips (so slight skew or curved
    lines do not smear the periodicity), detrended, and their autocorrelations
    averaged. Works even when neighbouring lines touch."""
    b = ink > 0
    H, W = b.shape
    sw = max(20, int(8 * char_h))
    acs = []
    for x0 in range(0, W, sw):
        ac = _strip_autocorr(b[:, x0:x0 + sw], char_h)
        if ac is not None:
            acs.append(ac)
    if not acs:
        return 1.6 * char_h
    n = min(len(a) for a in acs)
    ac = np.mean([a[:n] for a in acs], axis=0)
    lo, hi = int(0.9 * char_h), min(n - 1, int(4.0 * char_h))
    if hi <= lo + 2:
        return 1.6 * char_h
    seg = ac[lo:hi]
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.05]
    if not peaks:
        return 1.6 * char_h
    # first strong peak = fundamental; later peaks are harmonics (2x, 3x pitch)
    top = max(seg[i] for i in peaks)
    best = next(i for i in peaks if seg[i] >= 0.6 * top)
    return float(lo + best)


def split_by_profile(ink, box, pitch):
    """Split a blob spanning several touching lines at horizontal-projection minima."""
    x1, y1, x2, y2 = box
    crop = ink[y1:y2, x1:x2] > 0
    n_lines = int(round((y2 - y1) / pitch))
    if n_lines < 2 or crop.shape[0] < 4:
        return [box]
    prof = crop.sum(axis=1).astype(float)
    k = max(3, int(pitch * 0.15)) | 1
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    step = (y2 - y1) / n_lines
    cuts, prev = [], 0
    for j in range(1, n_lines):
        c = int(j * step)
        lo = max(prev + int(0.5 * step), c - int(0.45 * step))
        hi = min(len(prof) - 1, c + int(0.45 * step))
        if hi > lo:
            cut = lo + int(np.argmin(prof[lo:hi]))
            cuts.append(cut)
            prev = cut
    bounds = [0] + cuts + [y2 - y1]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = crop[a:b]
        if rows.sum() < 10:
            continue
        cols = np.where(rows.any(axis=0))[0]
        rr = np.where(rows.any(axis=1))[0]
        out.append([x1 + int(cols[0]), y1 + a + int(rr[0]), x1 + int(cols[-1]) + 1, y1 + a + int(rr[-1]) + 1])
    return out or [box]


def group_lines(ink, char_h, cfg, source="page"):
    """Horizontal smearing of ink into line blobs; blobs spanning several lines are
    split using the page's line pitch. Returns (lines, pitch)."""
    pitch = estimate_line_pitch(ink, char_h)
    kx = max(3, int(cfg.smear_x * char_h))
    smear = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(smear, connectivity=8)

    lines = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if max(w, h) < 0.5 * char_h:
            continue
        blob = (lbl[y:y + h, x:x + w] == i) & (ink[y:y + h, x:x + w] > 0)
        if blob.sum() < cfg.min_component_px * 2:
            continue
        sub = np.zeros_like(ink)
        sub[y:y + h, x:x + w][blob] = 255
        parts = [[int(x), int(y), int(x + w), int(y + h)]]
        if h > cfg.tall_line_split * pitch and w > 2 * pitch:
            parts = split_by_profile(sub, parts[0], pitch)
        for p in parts:
            if (p[2] - p[0]) < 0.35 * char_h and (p[3] - p[1]) > 0.8 * char_h:
                continue  # sliver: ruling / crack fragment
            crop = sub[p[1]:p[3], p[0]:p[2]]
            pts = cv2.findNonZero(crop)
            rect = None
            if pts is not None and len(pts) >= 5:
                (cx, cy), sz, ang = cv2.minAreaRect(pts)
                rect = ((cx + p[0], cy + p[1]), sz, ang)
            lines.append(Line(bbox=p, source=source, ink_px=int(np.count_nonzero(crop)), rect=rect))
    return lines, pitch


def merge_rows(lines, char_h, max_gap_char=2.0):
    """Join fragments that sit on the same text row (words split by wide spacing)."""
    lines = sorted(lines, key=lambda l: l.bbox[0])
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            a = lines[i]
            for j in range(i + 1, len(lines)):
                b = lines[j]
                if a.source != b.source:
                    continue
                ov = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
                if ov < 0.6 * min(a.h, b.h) or max(a.h, b.h) > 1.8 * min(a.h, b.h):
                    continue
                gap = max(a.bbox[0], b.bbox[0]) - min(a.bbox[2], b.bbox[2])
                if gap > max_gap_char * char_h:
                    continue
                a.bbox = [min(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1]),
                          max(a.bbox[2], b.bbox[2]), max(a.bbox[3], b.bbox[3])]
                a.ink_px += b.ink_px
                a.rect = a.rect if (a.rect is not None and a.w >= b.w) else b.rect
                del lines[j]
                changed = True
                break
            if changed:
                break
    return lines
