"""Draw labelled boxes on a copy of the original image."""
import cv2

COLORS = {  # BGR
    "header": (0, 140, 255),
    "footer": (190, 40, 190),
    "main_text": (40, 170, 40),
    "side_text": (255, 120, 0),
    "filler": (40, 40, 220),
}


def draw_regions(img, regions, draw_lines=False, lines=None):
    out = img.copy()
    h, w = out.shape[:2]
    t = max(2, int(round(max(h, w) / 700)))
    fs = max(0.45, max(h, w) / 2400)
    if draw_lines and lines:
        for ln in lines:
            x1, y1, x2, y2 = ln["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), COLORS.get(ln["label"], (128, 128, 128)), 1)
    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        col = COLORS.get(r["label"], (128, 128, 128))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, t)
        txt = f'{r["label"]} {r["confidence"]:.2f}'
        (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t // 2))
        ty = y1 - 4 if y1 - th - 8 > 0 else min(h - 2, y1 + th + 6)
        tx = min(max(0, x1), max(0, w - tw - 4))
        cv2.rectangle(out, (tx, ty - th - 4), (tx + tw + 4, ty + base), col, -1)
        cv2.putText(out, txt, (tx + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, t // 2), cv2.LINE_AA)
    return out
