"""Per-line features used by the classifier: Latin-vs-Indic script, stroke contrast."""
import re

import cv2
import numpy as np

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    _HAS_TESS = True
except Exception:  # tesseract binary or wrapper missing -> geometric cues only
    _HAS_TESS = False

_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def has_ocr():
    return _HAS_TESS


def headline_score(ink_crop):
    """Indic scripts (Devanagari, Bengali, ...) carry a strong horizontal headline
    (shirorekha): one row of the line is far denser than the rest. Latin has none."""
    b = ink_crop > 0
    if b.size == 0 or b.sum() < 20:
        return 0.0
    prof = b.sum(axis=1).astype(float)
    ratio = prof.max() / max(prof.mean(), 1e-6)
    fill = prof.max() / max(b.shape[1], 1)
    # ratio ~1.5-2 for Latin, ~3+ for Devanagari with long headlines
    return float(np.clip((ratio - 1.8) / 2.0, 0, 1) * np.clip(fill / 0.35, 0, 1))


def latin_ocr_score(gray_crop):
    """Confidence that a crop is English/Latin text, from Tesseract 'eng'."""
    if not _HAS_TESS or gray_crop.size == 0:
        return None
    h = gray_crop.shape[0]
    if h < 8:
        return None
    scale = 48.0 / h if h < 48 else 1.0
    crop = cv2.resize(gray_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else gray_crop
    # make text dark-on-light regardless of polarity
    if np.mean(crop) < 127:
        crop = 255 - crop
    crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    try:
        d = pytesseract.image_to_data(crop, lang="eng", config="--psm 7",
                                      output_type=pytesseract.Output.DICT)
    except Exception:
        return None
    total_chars, latin_letters, weighted = 0, 0, 0.0
    for txt, c in zip(d["text"], d["conf"]):
        txt = (txt or "").strip()
        try:
            c = float(c)
        except ValueError:
            continue
        if not txt or c < 0:
            continue
        total_chars += len(txt)
        n = sum(ch.isalpha() and ch.isascii() for ch in txt)
        if n >= 2 and _LATIN_WORD.search(txt):
            latin_letters += n
            weighted += n * c / 100.0
    if latin_letters == 0 or total_chars == 0:
        return 0.0
    mean_conf = weighted / latin_letters
    purity = latin_letters / total_chars
    # short fragments ("te", "ER") are typical OCR hallucinations on Indic strokes,
    # so require several recognised letters before trusting the verdict
    support = min(1.0, latin_letters / 8.0)
    return float(np.clip(mean_conf * purity * support, 0, 1))


def stroke_contrast(norm_gray_crop, ink_crop):
    """Background brightness minus ink brightness; pencil and faint notes score low."""
    ink = ink_crop > 0
    if ink.sum() < 10 or (~ink).sum() < 10:
        return 0.0
    return float(np.median(norm_gray_crop[~ink]) - np.median(norm_gray_crop[ink]))


def compute_features(lines, norm_gray, ink_map, cfg, raw_gray=None):
    raw_gray = norm_gray if raw_gray is None else raw_gray
    for ln in lines:
        x1, y1, x2, y2 = ln.bbox
        pad = max(2, int(0.15 * (y2 - y1)))
        H, W = norm_gray.shape
        px1, py1, px2, py2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
        ink_crop = ink_map[y1:y2, x1:x2]
        ln.feats["headline"] = headline_score(ink_crop)
        ln.feats["contrast"] = stroke_contrast(norm_gray[y1:y2, x1:x2], ink_crop)
        ln.feats["aspect"] = (x2 - x1) / max(1, (y2 - y1))
        ocr = None
        if cfg.use_ocr_script_check and ln.feats["aspect"] > 1.5:
            ocr = latin_ocr_score(raw_gray[py1:py2, px1:px2])
        ln.feats["latin_ocr"] = ocr
        if ocr is None:
            latin = 0.5 * (1 - ln.feats["headline"])  # weak prior without OCR
        else:
            latin = ocr * (1.0 - 0.6 * ln.feats["headline"])
        ln.feats["latin"] = float(latin)
    return lines
