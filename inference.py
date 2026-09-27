#!/usr/bin/env python
"""Batch CLI for manuscript layout region detection.

Example:
    python inference.py --input ./data/test_images --output ./results
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

from layout import Config, LayoutPipeline
from layout.io_utils import list_images, load_image, save_image, save_json
from layout.visualize import draw_regions

log = logging.getLogger("inference")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Detect header / footer / main_text / side_text / filler regions.")
    p.add_argument("--input", "-i", required=True, help="image file or folder")
    p.add_argument("--output", "-o", default="./results", help="output folder")
    p.add_argument("--recursive", action="store_true", help="search sub-folders")
    p.add_argument("--line-backend", choices=["cv", "kraken"], default="cv",
                   help="cv = classical (no weights needed); kraken = neural BLLA segmenter")
    p.add_argument("--kraken-model", default=None, help="path to a Kraken BLLA segmentation model")
    p.add_argument("--device", default="cpu", help="device for neural backends")
    p.add_argument("--no-ocr", action="store_true", help="disable Tesseract English-text check")
    p.add_argument("--no-deskew", action="store_true")
    p.add_argument("--draw-lines", action="store_true", help="also draw individual line boxes")
    p.add_argument("--config", default=None, help="JSON file overriding Config fields")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def build_config(args):
    cfg = Config()
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            for k, v in json.load(f).items():
                if not hasattr(cfg, k):
                    raise KeyError(f"Unknown config key: {k}")
                setattr(cfg, k, v)
    if args.no_ocr:
        cfg.use_ocr_script_check = False
    if args.no_deskew:
        cfg.deskew = False
    return cfg


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    in_path, out_dir = Path(args.input), Path(args.output)
    images = list_images(in_path, args.recursive)
    if not images:
        log.error("No images found at %s", in_path)
        return 1
    cfg = build_config(args)
    pipe = LayoutPipeline(cfg, line_backend=args.line_backend, kraken_model=args.kraken_model, device=args.device)

    base = in_path if in_path.is_dir() else in_path.parent
    summary, failed = [], 0
    for img_path in images:
        rel = img_path.relative_to(base)
        t0 = time.time()
        try:
            img = load_image(img_path)
            result = pipe.process(img)
        except Exception as e:  # keep going on bad pages
            failed += 1
            log.exception("Failed on %s: %s", rel, e)
            continue
        result = {"image": str(rel).replace("\\", "/"), **result, "runtime_s": round(time.time() - t0, 2)}
        stem = out_dir / rel.parent / rel.stem
        save_json(f"{stem}.json", result)
        save_image(f"{stem}_annotated.jpg", draw_regions(img, result["regions"], args.draw_lines, result["lines"]))
        counts = {}
        for r in result["regions"]:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        summary.append({"image": result["image"], "regions": counts, "runtime_s": result["runtime_s"]})
        log.info("%s -> %s (%.1fs)", rel, counts, result["runtime_s"])

    save_json(out_dir / "summary.json", {"config": cfg.to_dict(), "backend": args.line_backend,
                                         "processed": len(summary), "failed": failed, "images": summary})
    log.info("Done: %d processed, %d failed. Results in %s", len(summary), failed, out_dir)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
