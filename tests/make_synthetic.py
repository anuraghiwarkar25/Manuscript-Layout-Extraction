"""Build a synthetic manuscript page that contains every target class, using real
ink crops from a sample page. Used by the tests and handy for tuning thresholds."""
import cv2
import numpy as np


def _darken_paste(canvas, patch, x, y):
    h, w = patch.shape[:2]
    roi = canvas[y:y + h, x:x + w]
    np.minimum(roi, patch[: roi.shape[0], : roi.shape[1]], out=roi)


def _flatten_bg(patch):
    """Push the crop's paper colour to white so pasting leaves no seams."""
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    bg = np.percentile(g, 80)
    scale = 255.0 / max(bg, 1)
    return np.clip(patch.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def make_page(src_bgr, seed=0):
    rng = np.random.default_rng(seed)
    H, W = 1300, 1700
    page = np.full((H, W, 3), (214, 226, 234), np.uint8)             # paper tone
    page = cv2.add(page, rng.integers(0, 8, (H, W, 3), dtype=np.uint8))

    body = _flatten_bg(src_bgr[80:720, 245:1965])                      # 8 real text lines
    body = cv2.resize(body, (1150, int(640 * 1150 / 1720)))
    _darken_paste(page, body, 300, 300)                                # main_text

    word = _flatten_bg(src_bgr[85:160, 245:640])
    head = cv2.resize(word, (300, 57))
    _darken_paste(page, head, 725, 120)                                # header (title)

    catch = cv2.resize(_flatten_bg(src_bgr[650:720, 1700:1960]), (170, 46))
    _darken_paste(page, catch, 1270, 1120)                             # footer (catchword)

    side = _flatten_bg(src_bgr[240:400, 245:560])
    side = cv2.resize(side, (220, 110))
    side = cv2.rotate(side, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _darken_paste(page, side, 110, 450)                                # side_text (vertical marginalia)

    pencil = _flatten_bg(src_bgr[400:480, 900:1300]).astype(np.float32)
    pencil = 255 - (255 - pencil) * 0.38                               # faint grey strokes
    pencil = cv2.resize(pencil.astype(np.uint8), (260, 52))
    _darken_paste(page, pencil, 1300, 820)                             # filler (pencil)

    cv2.putText(page, "Acc. No. 4471  Oriental Library", (300, 1195),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 60, 60), 2, cv2.LINE_AA)  # filler (English)
    return page


if __name__ == "__main__":
    import sys
    src = cv2.imread(sys.argv[1])
    cv2.imwrite(sys.argv[2], make_page(src))
