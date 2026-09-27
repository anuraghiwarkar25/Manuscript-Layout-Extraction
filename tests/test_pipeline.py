"""End-to-end tests.

Run with:  pytest -q

The synthetic page (tests/make_synthetic.py) is built from real ink crops of a
sample manuscript and contains one instance of every target class, so the tests
check both detection and correct labelling. The degradations in tests/degrade.py
cover the robustness requirements: skew, blur, stains, uneven lighting, faded ink.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from layout import Config, LayoutPipeline  # noqa: E402
from layout.io_utils import load_image  # noqa: E402
from tests.degrade import blur_noise, fade, skew, stains, uneven_light  # noqa: E402
from tests.make_synthetic import make_page  # noqa: E402

SAMPLES = ROOT / "data" / "test_images"

# Where each class sits on the synthetic page (see make_synthetic.py).
EXPECTED = {
    "header": (725, 125, 1027, 176),
    "main_text": (300, 300, 1450, 728),
    "side_text": (110, 448, 224, 672),
    "footer": (1270, 1118, 1438, 1152),
}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def best_iou(regions, label, box):
    return max((iou(r["bbox"], box) for r in regions if r["label"] == label), default=0.0)


@pytest.fixture(scope="session")
def pipe():
    return LayoutPipeline(Config())


@pytest.fixture(scope="session")
def synthetic():
    return make_page(load_image(SAMPLES / "sample_paper.png"))


# --------------------------------------------------------------- basic contract
@pytest.mark.parametrize("name", ["sample_palmleaf.png", "sample_paper.png"])
def test_real_samples_produce_valid_output(pipe, name):
    img = load_image(SAMPLES / name)
    res = pipe.process(img)
    h, w = img.shape[:2]
    assert res["regions"], "no regions detected"
    for r in res["regions"]:
        x1, y1, x2, y2 = r["bbox"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, "box outside page boundaries"
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["label"] in {"header", "footer", "main_text", "side_text", "filler"}
    assert json.dumps(res)  # result must be JSON-serialisable


@pytest.mark.parametrize("name", ["sample_palmleaf.png", "sample_paper.png"])
def test_real_samples_find_main_text(pipe, name):
    res = pipe.process(load_image(SAMPLES / name))
    main = [r for r in res["regions"] if r["label"] == "main_text"]
    assert len(main) == 1
    assert main[0]["num_lines"] >= 5


def test_watermark_text_is_filler(pipe):
    """The eGangotri / CSU-Sringeri captions are English text -> filler, not header/footer."""
    res = pipe.process(load_image(SAMPLES / "sample_palmleaf.png"))
    fillers = [r for r in res["regions"] if r["label"] == "filler"]
    assert len(fillers) >= 2
    img_h = res["image_size"][1]
    assert any(r["bbox"][1] < 0.15 * img_h for r in fillers)   # top caption
    assert any(r["bbox"][3] > 0.85 * img_h for r in fillers)   # bottom caption


def test_inputs_are_not_modified(tmp_path, pipe):
    src = SAMPLES / "sample_paper.png"
    before = src.read_bytes()
    pipe.process(load_image(src))
    assert src.read_bytes() == before


# ------------------------------------------------------------------ all classes
def test_all_five_classes_detected(pipe, synthetic):
    res = pipe.process(synthetic)
    labels = {r["label"] for r in res["regions"]}
    assert labels == {"header", "footer", "main_text", "side_text", "filler"}


@pytest.mark.parametrize("label,box", EXPECTED.items())
def test_class_localisation(pipe, synthetic, label, box):
    res = pipe.process(synthetic)
    assert best_iou(res["regions"], label, box) > 0.5, f"{label} poorly localised"


def test_english_text_is_filler(pipe, synthetic):
    res = pipe.process(synthetic)
    # "Acc. No. 4471  Oriental Library" sits in the bottom margin
    assert best_iou(res["regions"], "filler", (300, 1164, 852, 1213)) > 0.5


# ------------------------------------------------------------------- robustness
SKEW_DEG = 3.0

DEGRADATIONS = {
    "skew": lambda i: skew(i, SKEW_DEG),
    "uneven_light": uneven_light,
    "stains": stains,
    "blur_noise": blur_noise,
    "faded": lambda i: fade(i, 0.45),
}


def rotate_box(box, shape, deg):
    """Where a ground-truth box lands after the page is rotated by `deg`."""
    import numpy as np
    h, w = shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    x1, y1, x2, y2 = box
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    pts = cv2.transform(pts[None], M)[0]
    return [*pts.min(axis=0), *pts.max(axis=0)]


@pytest.mark.parametrize("name,fn", DEGRADATIONS.items())
def test_robust_to_degradation(pipe, synthetic, name, fn):
    res = pipe.process(fn(synthetic))
    labels = {r["label"] for r in res["regions"]}
    for required in ("header", "main_text", "side_text", "footer"):
        assert required in labels, f"{required} lost under {name}"
    for label in ("header", "main_text", "side_text", "footer"):
        box = EXPECTED[label]
        if name == "skew":  # the ground truth moves with the page
            box = rotate_box(box, synthetic.shape, SKEW_DEG)
        assert best_iou(res["regions"], label, box) > 0.35, \
            f"{label} poorly localised under {name}"


def test_skew_is_estimated(pipe, synthetic):
    res = pipe.process(skew(synthetic, SKEW_DEG))
    assert abs(abs(res["skew_deg"]) - SKEW_DEG) < 1.2


# ------------------------------------------------------------------ multi-column
def test_two_columns_are_separate_regions(pipe, synthetic):
    import numpy as np
    src = load_image(SAMPLES / "sample_paper.png")
    body = src[80:720, 245:1965]
    col = cv2.resize(body, (700, int(640 * 700 / 1720)))
    page = np.full((1200, 1700, 3), (214, 226, 234), np.uint8)
    for x in (120, 900):
        for y in (200, 200 + col.shape[0] + 10):
            roi = page[y:y + col.shape[0], x:x + col.shape[1]]
            np.minimum(roi, col[:roi.shape[0], :roi.shape[1]], out=roi)
    res = pipe.process(page)
    mains = [r for r in res["regions"] if r["label"] == "main_text"]
    assert len(mains) == 2, "columns should not be merged into one region"
    assert all(r["bbox"][2] - r["bbox"][0] < 900 for r in mains)


# --------------------------------------------------------------------------- CLI
def test_cli_batch_run(tmp_path):
    out = tmp_path / "results"
    r = subprocess.run([sys.executable, "inference.py", "--input", str(SAMPLES),
                        "--output", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for stem in ("sample_palmleaf", "sample_paper"):
        assert (out / f"{stem}.json").exists()
        assert (out / f"{stem}_annotated.jpg").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["processed"] == 2 and summary["failed"] == 0


def test_cli_rejects_unknown_config_key(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text('{"not_a_real_option": 1}')
    r = subprocess.run([sys.executable, "inference.py", "--input", str(SAMPLES),
                        "--output", str(tmp_path / "o"), "--config", str(cfg)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
