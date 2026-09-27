"""Locate the manuscript substrate (palm leaf / paper) against the scanning background."""
import cv2
import numpy as np


def detect_page(img_bgr, cfg):
    """Returns (mask uint8, bbox [x1,y1,x2,y2], full_frame bool).

    The background colour is estimated from the image border; pixels far from it in
    Lab space are substrate. If border and centre look alike (paper fills the frame),
    the whole image is the page.
    """
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(2, int(0.02 * min(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    centre = np.median(lab[h // 3: 2 * h // 3, w // 3: 2 * w // 3].reshape(-1, 3), axis=0)

    full = np.full((h, w), 255, np.uint8)
    if np.linalg.norm(centre - bg) < cfg.page_bg_delta:
        return full, [0, 0, w, h], True

    dist = np.linalg.norm(lab - bg, axis=2)
    dist = cv2.GaussianBlur(dist, (0, 0), 3)
    dist8 = np.clip(dist * (255.0 / max(dist.max(), 1e-6)), 0, 255).astype(np.uint8)
    _, fg = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(5, int(0.01 * min(h, w)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3 * k, 3 * k), np.uint8))

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return full, [0, 0, w, h], True
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < cfg.page_min_area_frac * h * w:
        return full, [0, 0, w, h], True

    comp = (lbl == idx).astype(np.uint8) * 255
    # fill interior holes (binding holes, dark stains) by filling the outer contour
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(comp)
    cv2.drawContours(mask, cnts, -1, 255, thickness=cv2.FILLED)
    x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return mask, [x, y, x + bw, y + bh], False
