"""Optional line detector: Kraken BLLA neural baseline segmenter.

Recommended model for Sanskrit/Devanagari eGangotri/MIDF material:
    huggingface.co/tadad/midf-sanskrit-ocr  (segmentation/ folder, BLLA topline model)
Any Kraken BLLA segmentation model works (e.g. the default blla.mlmodel).
Install:  pip install kraken
"""
import numpy as np
from PIL import Image

from ..lines import Line


class KrakenLineDetector:
    def __init__(self, model_path=None, device="cpu"):
        try:
            from kraken import blla  # noqa: F401
        except ImportError as e:
            raise ImportError("Kraken backend requested but kraken is not installed: pip install kraken") from e
        self.model = None
        self.device = device
        if model_path:
            self.model = self._load(model_path)

    @staticmethod
    def _load(path):
        # the loading entry point moved between Kraken releases; try the known ones
        errors = []
        try:
            from kraken.lib import vgsl
            return vgsl.TorchVGSLModel.load_model(path)
        except Exception as e:  # pragma: no cover
            errors.append(e)
        try:
            from kraken.models import load_models  # Kraken >= 6
            return load_models(path)
        except Exception as e:  # pragma: no cover
            errors.append(e)
        raise RuntimeError(f"Could not load Kraken model {path}: {errors}")

    def detect(self, img_bgr, page_mask=None):
        from kraken import blla
        im = Image.fromarray(img_bgr[:, :, ::-1])
        kwargs = {"device": self.device}
        if self.model is not None:
            kwargs["model"] = self.model
        seg = blla.segment(im, **kwargs)
        raw_lines = seg.lines if hasattr(seg, "lines") else seg["lines"]
        out = []
        for ln in raw_lines:
            poly = getattr(ln, "boundary", None) or (ln.get("boundary") if isinstance(ln, dict) else None)
            if not poly:
                continue
            pts = np.asarray(poly, dtype=float)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            b = [int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))]
            if page_mask is not None:
                cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                cy = min(max(cy, 0), page_mask.shape[0] - 1)
                cx = min(max(cx, 0), page_mask.shape[1] - 1)
                if page_mask[cy, cx] == 0:
                    continue
            out.append(Line(bbox=b, source="page"))
        return out
