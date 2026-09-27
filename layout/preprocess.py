"""Resolution normalisation, illumination correction, binarisation, deskew."""
import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def resize_for_work(img, max_side):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return img, scale


def odd(n):
    n = max(3, int(n))
    return n if n % 2 else n + 1


def normalize_illumination(gray, kernel_frac):
    """Divide by an estimated background: removes uneven lighting, stains, yellowing."""
    k = odd(kernel_frac * max(gray.shape))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 4)
    norm = (gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)) * 255.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(norm)


def binarize_dark_ink(norm_gray, mask, window_frac, k):
    """Sauvola local threshold: robust to faded ink and bleed-through."""
    win = odd(window_frac * max(norm_gray.shape))
    smooth = cv2.GaussianBlur(norm_gray, (3, 3), 0)
    thr = threshold_sauvola(smooth, window_size=win, k=k)
    ink = (smooth < thr) & (mask > 0)
    return ink.astype(np.uint8) * 255


def binarize_any_polarity(gray, mask, char_px=15):
    """For text printed on the background (e.g. white stamps on a blue board)."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (odd(char_px), odd(char_px)))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    resp = cv2.max(tophat, blackhat)
    resp[mask == 0] = 0
    vals = resp[mask > 0]
    if vals.size == 0 or vals.max() < 30:
        return np.zeros_like(gray)
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = ((resp > max(t, 40)) & (mask > 0)).astype(np.uint8) * 255
    return out


def remove_rulings(ink, char_h, len_frac):
    """Remove ruling / border / crack lines so they do not glue text lines together.

    Vertical: any straight run longer than ~3.5 character heights cannot be a glyph.
    Horizontal: Indic headlines (shirorekha) can be long, so the horizontal limit is
    a fraction of the page width instead."""
    h, w = ink.shape
    vlen = max(15, int(3.5 * char_h))
    hlen = max(15, int(max(w * len_frac, 25 * char_h)))
    vert = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen)))
    # tolerate slight slant/breaks: close vertically, then re-open
    vert = cv2.morphologyEx(cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((max(3, int(char_h * 0.3)), 1), np.uint8)),
                            cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen))) | vert
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1)))
    rules = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 5), np.uint8))
    return cv2.bitwise_and(ink, cv2.bitwise_not(rules))


def estimate_skew(line_boxes_rot):
    """Median angle of elongated line blobs (list of cv2.minAreaRect results)."""
    angles, weights = [], []
    for (cx, cy), (rw, rh), a in line_boxes_rot:
        if rw < rh:
            rw, rh, a = rh, rw, a - 90
        if rw < 4 * max(rh, 1):
            continue
        while a > 45:
            a -= 90
        while a < -45:
            a += 90
        angles.append(a)
        weights.append(rw)
    if len(angles) < 3:
        return 0.0
    order = np.argsort(angles)
    a = np.array(angles)[order]
    wts = np.cumsum(np.array(weights)[order])
    return float(a[np.searchsorted(wts, wts[-1] / 2)])


def rotation_matrix(shape, angle):
    h, w = shape[:2]
    return cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)


def rotate(img, M, border_value=0, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
