"""End-to-end processing of one image: page -> ink -> lines -> features -> classes -> regions."""
import cv2
import numpy as np

from . import preprocess as pp
from .classify import classify_lines, estimate_pitch, merge_into_regions
from .config import Config
from .lines import clean_components, drop_specks, estimate_char_height, group_lines, merge_rows
from .page import detect_page
from .script_id import compute_features, has_ocr


class LayoutPipeline:
    def __init__(self, cfg=None, line_backend="cv", kraken_model=None, device="cpu"):
        self.cfg = cfg or Config()
        self.kraken = None
        if line_backend == "kraken":
            from .backends.kraken_lines import KrakenLineDetector
            self.kraken = KrakenLineDetector(kraken_model, device=device)

    # ------------------------------------------------------------------ helpers
    def _detect_lines(self, work, gray, page_mask, full_frame):
        cfg = self.cfg
        H, W = gray.shape
        norm = pp.normalize_illumination(gray, cfg.illum_kernel_frac)

        inner = page_mask
        if not full_frame:
            e = max(3, int(cfg.page_edge_erode_frac * min(H, W)))
            inner = cv2.erode(page_mask, np.ones((e, e), np.uint8))

        ink = pp.binarize_dark_ink(norm, inner, cfg.sauvola_window_frac, cfg.sauvola_k)
        ink, stats = clean_components(ink, cfg.min_component_px)
        char_h = estimate_char_height(stats, default=H / 40.0, img_shape=(H, W))
        ink = pp.remove_rulings(ink, char_h, cfg.rule_line_len_frac)
        ink = drop_specks(ink, char_h)
        _, stats = clean_components(ink, cfg.min_component_px)
        char_h = estimate_char_height(stats, default=char_h, img_shape=(H, W))

        if self.kraken is not None:
            lines = self.kraken.detect(work, inner)
            for ln in lines:
                x1, y1, x2, y2 = ln.bbox
                ln.ink_px = int(np.count_nonzero(ink[y1:y2, x1:x2]))
        else:
            lines, _ = group_lines(ink, char_h, cfg, source="page")

        ink_all = ink.copy()
        if not full_frame:  # text printed on the scanning background (stamps, rulers, labels)
            e = max(5, int(0.01 * min(H, W)))
            outside = cv2.erode(cv2.bitwise_not(page_mask), np.ones((e, e), np.uint8))
            ink_out = pp.binarize_any_polarity(gray, outside, char_px=max(7, int(0.35 * char_h)))
            ink_out, st_out = clean_components(ink_out, cfg.min_component_px, max_h=0.2 * H)
            if st_out is not None and len(st_out):
                ch_out = estimate_char_height(st_out, default=char_h, img_shape=(H, W))
                ink_out = drop_specks(ink_out, ch_out)
                out_lines, _ = group_lines(ink_out, ch_out, cfg, source="outside")
                # keep only text-like strings (labels, stamps), not board texture
                lines += [l for l in out_lines if l.w > 3 * l.h and l.w > 4 * ch_out]
                ink_all = cv2.bitwise_or(ink_all, ink_out)
        lines = merge_rows(lines, char_h, cfg.row_merge_gap)
        return lines, char_h, ink_all, norm

    @staticmethod
    def _map_back(box, Minv, scale, W0, H0):
        x1, y1, x2, y2 = box
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
        if Minv is not None:
            pts = cv2.transform(pts[None], Minv)[0]
        pts /= scale
        bx1, by1 = np.floor(pts.min(axis=0)).astype(int)
        bx2, by2 = np.ceil(pts.max(axis=0)).astype(int)
        # constraint: boxes must stay inside the page/image boundaries
        return [int(np.clip(bx1, 0, W0 - 1)), int(np.clip(by1, 0, H0 - 1)),
                int(np.clip(bx2, 1, W0)), int(np.clip(by2, 1, H0))]

    # ------------------------------------------------------------------ main
    def process(self, img_bgr):
        cfg = self.cfg
        H0, W0 = img_bgr.shape[:2]
        work, scale = pp.resize_for_work(img_bgr, cfg.work_max_side)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        page_mask, page_bbox, full = detect_page(work, cfg)

        lines, char_h, ink, norm = self._detect_lines(work, gray, page_mask, full)

        angle, M, Minv = 0.0, None, None
        if cfg.deskew:
            rects = [l.rect for l in lines if l.rect is not None and l.source == "page"]
            angle = pp.estimate_skew(rects)
            if cfg.min_skew_deg <= abs(angle) <= cfg.max_skew_deg:
                M = pp.rotation_matrix(work.shape, angle)
                Minv = cv2.invertAffineTransform(M)
                border = tuple(int(v) for v in np.median(work[:5, :5].reshape(-1, 3), axis=0))
                work = pp.rotate(work, M, border_value=border)
                gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
                page_mask = pp.rotate(page_mask, M, 0, cv2.INTER_NEAREST)
                if full:
                    page_mask[:] = 255
                x, y, w, h = cv2.boundingRect(page_mask)
                page_bbox = [x, y, x + w, y + h]
                lines, char_h, ink, norm = self._detect_lines(work, gray, page_mask, full)
            else:
                angle = 0.0

        compute_features(lines, norm, ink, cfg, raw_gray=gray)
        classify_lines(lines, page_bbox, char_h, cfg)
        pitch = estimate_pitch(lines, char_h * 1.3)
        regions = merge_into_regions(lines, char_h, pitch, cfg)

        out_regions = []
        for i, r in enumerate(regions):
            out_regions.append({
                "id": i,
                "label": r.label,
                "bbox": self._map_back(r.bbox, Minv, scale, W0, H0),
                "confidence": r.confidence,
                "num_lines": len(r.lines),
                "evidence": r.reasons,
            })
        out_lines = []
        for ln in lines:
            if "label" not in ln.feats:
                continue
            out_lines.append({
                "label": ln.feats["label"],
                "bbox": self._map_back(ln.bbox, Minv, scale, W0, H0),
                "confidence": ln.feats["conf"],
                "reason": ln.feats["reason"],
                "features": {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in ln.feats.items() if k in ("headline", "contrast", "latin", "latin_ocr")},
            })
        return {
            "image_size": [W0, H0],
            "page_bbox": self._map_back(page_bbox, Minv, scale, W0, H0),
            "page_is_full_frame": bool(full),
            "skew_deg": round(float(angle), 3),
            "char_height_px": round(char_h / scale, 1),
            "ocr_script_check": bool(cfg.use_ocr_script_check and has_ocr()),
            "regions": out_regions,
            "lines": out_lines,
        }
