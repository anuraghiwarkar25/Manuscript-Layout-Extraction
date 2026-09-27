# Manuscript Layout Region Detection

Detects and classifies non-body layout regions in historical manuscript images:
**header**, **footer**, **main_text**, **side_text**, **filler**.

Works on palm-leaf and paper manuscripts, multiple scripts, multi-column pages and
degraded scans. Runs on CPU, requires no model weights and no training data.

```bash
python inference.py --input ./data/test_images --output ./results
```

---

## Contents

- [Setup](#setup)
- [Usage](#usage)
- [Output format](#output-format)
- [Model rationale](#model-rationale)
- [How the pipeline works](#how-the-pipeline-works)
- [Classification rules](#classification-rules)
- [Results](#results)
- [Testing](#testing)
- [Configuration](#configuration)
- [Optional neural line detector](#optional-neural-line-detector)
- [Limitations and next steps](#limitations-and-next-steps)
- [Repository layout](#repository-layout)
- [References](#references)

---

## Setup

Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tesseract OCR is an **optional** dependency, used only to recognise English text so it
can be labelled `filler`. Without it the pipeline still runs and falls back to a
script-shape heuristic; pass `--no-ocr` to disable the check explicitly.

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr
# macOS
brew install tesseract
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
```

Alternatively, with conda: `conda env create -f environment.yml && conda activate manuscript-layout`.

## Usage

```bash
# Batch over a folder (relative paths work from the repo root)
python inference.py --input ./data/test_images --output ./results

# Single image
python inference.py --input ./data/test_images/sample_paper.png --output ./results

# Recurse into sub-folders, also draw individual text-line boxes
python inference.py -i ./scans -o ./results --recursive --draw-lines
```

Useful flags:

| Flag | Effect |
|---|---|
| `--recursive` | search sub-folders of `--input` |
| `--draw-lines` | draw per-line boxes as well as merged regions |
| `--no-ocr` | skip the Tesseract English-text check (faster) |
| `--no-deskew` | skip skew estimation and correction |
| `--config FILE` | JSON file overriding any field of `layout/config.py` |
| `--line-backend kraken` | use a neural line segmenter instead of the classical one |
| `-v` | verbose logging |

Input images are opened read-only and are never modified. Damaged or unreadable
pages are logged and skipped; the rest of the batch continues, and the exit code
is non-zero if anything failed.

## Output format

For every input image `X.png` the pipeline writes `X.json` and `X_annotated.jpg`
into the output folder, mirroring the input folder structure, plus one
`summary.json` per run.

```jsonc
{
  "image": "sample_palmleaf.png",
  "image_size": [2048, 834],
  "page_bbox": [143, 81, 1777, 696],     // detected substrate (leaf / sheet)
  "page_is_full_frame": false,
  "skew_deg": 0.0,
  "char_height_px": 46.6,
  "ocr_script_check": true,
  "regions": [
    {
      "id": 0,
      "label": "filler",
      "bbox": [577, 26, 1469, 56],       // [x1, y1, x2, y2], always inside the image
      "confidence": 0.9,
      "num_lines": 1,
      "evidence": ["text outside manuscript substrate"]
    }
  ],
  "lines": [                              // per-line detail behind each region
    {
      "label": "main_text",
      "bbox": [168, 93, 1678, 155],
      "confidence": 0.95,
      "reason": "main text block",
      "features": {"headline": 0.0, "contrast": 187.0, "latin": 0.0}
    }
  ],
  "runtime_s": 3.6
}
```

`evidence` and `reason` record *why* each label was chosen, which makes
disagreements easy to debug and thresholds easy to tune.

## Model rationale

**Why not an off-the-shelf segmentation model.** The five target classes are defined
by *position and role*, not by appearance. A header line and a body line are the same
handwriting in the same ink; only their position on the page distinguishes them. A
promptable segmenter such as SAM 3 matches short visual concepts ("person", "dog") and
has no notion of "top margin" or "outside the text column", so prompts like
`"header text at top of page"` return nothing usable on these images. This was verified
on the provided samples before the pipeline was designed.

**Why not a trained layout detector.** DocLayout-YOLO, Detectron-style instance
segmenters and similar models are the right long-term answer, but they need labelled
data. The public document-layout checkpoints are pretrained on modern printed documents
(DocLayNet, D<sup>4</sup>LA) and transfer poorly to palm-leaf manuscripts; the relevant
academic datasets for this domain are Indiscapes and Indiscapes2 (see
[References](#references)), whose class taxonomy differs from the one required here.
With a two-day budget and no annotated pages, fine-tuning is not reachable.

**What this pipeline does instead.** It separates the two problems:

1. **Where is the text?** — a script-agnostic, geometric problem, solved robustly with
   classical computer vision (illumination normalisation, Sauvola binarisation,
   morphological line grouping). This step makes no assumption about the script, so
   Devanagari, Grantha, Malayalam and Latin all work.
2. **What role does each text block play?** — a *layout* problem, solved with explicit
   geometric rules relative to the detected page and main text block, plus two
   appearance cues (script identity and stroke contrast) for the `filler` class.

Every rule is stated in one file (`layout/classify.py`), is thresholded in units of
character height or line pitch (so it transfers across resolutions and page sizes), and
reports the reason for its decision. This is the most accurate approach available
without training data, and it doubles as a pre-labelling tool for the annotation effort
that a trained model would require.

## How the pipeline works

```
image
  └─ preprocess    resize to a working resolution, estimate and divide out the
                   background illumination (removes uneven lighting, stains, yellowing)
  └─ page          find the substrate against the scanning background by Lab colour
                   distance; fill binding holes; fall back to the full frame when the
                   sheet fills the image
  └─ ink           Sauvola local thresholding (robust to faded ink and bleed-through),
                   removal of ruling lines and cracks, removal of sub-character specks
                   (palm-leaf fibre texture, dust)
  └─ lines         horizontal morphological smearing into line blobs; line pitch
                   estimated from the autocorrelation of detrended row-ink profiles
                   computed in vertical strips, then used to split blobs that span
                   several touching lines; same-row fragments merged
  └─ deskew        median angle of elongated line blobs; if significant, the page is
                   rotated and lines re-detected; boxes are mapped back to the
                   original coordinate frame at the end
  └─ features      per line: Indic headline (shirorekha) strength, stroke contrast,
                   aspect ratio, and a Tesseract-based English/Latin score
  └─ classify      the rules below
  └─ regions       same-label lines merged into regions; confidences area-weighted
  └─ output        JSON + annotated image, boxes clipped to the image bounds
```

## Classification rules

Applied in order; the first match wins.

| Order | Class | Rule |
|---|---|---|
| 1 | **filler** | text lying outside the manuscript substrate (digitisation captions, labels, rulers); English/Latin script; low stroke contrast (pencil, faint modern annotation); large non-text-shaped blobs (decoration) |
| 2 | **side_text** | vertical (rotated) writing in a margin, or a line whose horizontal extent falls outside the main text column |
| 3 | **header / footer** | after grouping the remaining lines vertically by whitespace, the dominant group is `main_text`; small groups above or below it, separated by a clear gap relative to the line pitch, become `header` / `footer` |
| 4 | **header / footer** | small isolated marks in the top or bottom margin (folio numbers, catchwords, signatures) are re-assigned from `side_text` |
| 5 | — | small fragments adopt the label of the larger block they sit next to on the same row |

The main text column extent is derived from an x-coverage histogram of the wide lines,
not from a median of line edges, so it spans **all columns** of a multi-column page while
a handful of marginal notes cannot stretch it.

A note on the sample images: the captions `CC-0. In Public Domain. Digitization by
eGangotri…` and `CSU-Sringeri Campus Rare Book and Manuscript Collection…` look like a
header and a footer, but the specification defines English text as `filler`. Rule 1 fires
before the header/footer rules precisely so that these are labelled correctly.

## Results

Confidence is the detector's certainty multiplied by how clearly the deciding rule fired
(for example, gap size relative to line pitch), then area-weighted across the lines in a
region.

**Provided samples** (`data/test_images/`):

| Image | Detected |
|---|---|
| `sample_palmleaf.png` | 1 × main_text (20 lines, conf 0.94) + 2 × filler — the two eGangotri captions |
| `sample_paper.png` | 1 × main_text (8 lines, conf 0.95) + 2 × filler |

Neither provided sample actually contains a header, footer or side_text region, so those
classes are validated on a synthetic page (`tests/make_synthetic.py`) built from real ink
crops of `sample_paper.png`, containing one instance of every class:

| Test page | Result |
|---|---|
| all five classes, clean | all 5 correct, IoU > 0.5 against ground truth |
| + 3° skew | all correct, skew estimated to within 1.2° |
| + uneven lighting | all correct |
| + stains and bleed | all correct |
| + blur and sensor noise | all correct |
| + faded ink (45 %) | header/footer/main/side correct; the pencil note fades below the detection floor |
| + everything combined | header/footer/main/side correct |
| two-column page | 2 separate `main_text` regions, not merged |

Runtime is roughly 3–4 s per page on CPU (about 1.5 s with `--no-ocr`).

## Testing

```bash
pip install pytest
pytest -q                 # 21 tests, ~80 s
```

The suite covers the output contract (boxes inside page bounds, valid labels,
JSON-serialisable, inputs unmodified), correct labelling and localisation of all five
classes, the five degradation scenarios, multi-column separation, and the CLI end to end.

## Configuration

Every threshold lives in `layout/config.py` as a dataclass field with a comment, and
values are expressed relative to character height, line pitch or page size. Override any
of them without editing code:

```bash
echo '{"gap_pitch_factor": 1.6, "latin_threshold": 0.7}' > tuning.json
python inference.py -i ./scans -o ./results --config tuning.json
```

The fields most worth tuning on a new collection are `gap_pitch_factor` (how large a gap
must be before a block counts as header/footer), `side_margin_tol` (how far outside the
text column a line must sit to be marginalia), `latin_threshold` and
`pencil_contrast_ratio` (the two `filler` cues).

## Optional neural line detector

Line detection can be swapped for Kraken's BLLA neural baseline segmenter, which is
trained on historical manuscripts and handles curved or crowded lines better than the
classical detector. Classification is unchanged.

```bash
pip install kraken
python inference.py -i ./scans -o ./results --line-backend kraken --kraken-model blla.mlmodel
```

For Sanskrit / Devanagari material from the MIDF–eGangotri collection (which the provided
samples come from), a BLLA model fine-tuned on 1,916 reviewed pages of that exact corpus
is published at `tadad/midf-sanskrit-ocr` on Hugging Face and is the recommended
checkpoint. This backend is implemented but untested here, as the sandbox used for
development had no access to model downloads.

## Limitations and next steps

**Known limitations**

- Header, footer and side_text rules are validated on synthetic pages only, because the
  provided samples contain none of those regions. Expect to re-tune `gap_pitch_factor`
  and `side_margin_tol` on real pages from the target collection.
- Very faint pencil annotation (below roughly 45 % of median ink contrast) is dropped
  along with the noise floor rather than labelled `filler`.
- Tesseract occasionally reads short Indic fragments as English words. The score
  requires several recognised letters with high confidence before a line is called
  English, which suppresses most of this, but a dedicated script classifier would be
  more reliable.
- Regions are axis-aligned boxes. Strongly curved palm-leaf text lines are covered
  correctly but not tightly.
- Decorative illustrations are only caught by a coarse shape heuristic; a page with
  substantial artwork would benefit from a trained detector.

**Next steps, in order of expected value**

1. Annotate 100–200 real pages spanning both substrates and several scripts, using this
   pipeline's output as pre-labels in Label Studio or Roboflow. Pre-labelling is where
   most of the annotation cost disappears.
2. Fine-tune **YOLOv11** or **DocLayout-YOLO** on the five classes from those
   annotations. Published comparisons on historical documents put both in the same range,
   with DocLayout-YOLO slightly ahead thanks to document-specific pretraining.
3. Keep this pipeline as the cold-start path for new collections and as a sanity check
   against the trained model, and keep the `evidence` field to triage disagreements.
4. If exact region shapes matter downstream (for OCR cropping, for instance), move from
   boxes to polygons using Palmira-style deformable instance segmentation.

## Repository layout

```
inference.py               CLI: batch processing, JSON + annotated output
layout/
  config.py                every tunable threshold, with comments
  io_utils.py              image discovery and safe read/write (unicode paths, 16-bit)
  preprocess.py            resize, illumination normalisation, binarisation, deskew
  page.py                  substrate detection against the scanning background
  lines.py                 text-line detection, line-pitch estimation, splitting/merging
  script_id.py             Indic headline score, stroke contrast, Latin OCR score
  classify.py              the layout rules and region merging
  visualize.py             annotated output images
  backends/kraken_lines.py optional Kraken BLLA line detector
tests/
  make_synthetic.py        builds a page containing all five classes from real ink
  degrade.py               skew, blur, stains, uneven lighting, fading
  test_pipeline.py         21 tests
data/test_images/          the two provided sample pages
data/synthetic/            generated test pages
```

## References

- Prusty et al., *Indiscapes: Instance Segmentation Networks for Layout Parsing of
  Historical Indic Manuscripts*, ICDAR 2019 — [arXiv:1912.07025](https://arxiv.org/abs/1912.07025)
- Sharan et al., *Palmira: A Deep Deformable Network for Instance Segmentation of Dense
  and Uneven Layouts in Handwritten Manuscripts*, ICDAR 2021 —
  [arXiv:2108.09436](https://arxiv.org/abs/2108.09436)
- Zhao et al., *DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse
  Synthetic Data and Global-to-Local Adaptive Perception*, 2024 —
  [arXiv:2410.12628](https://arxiv.org/abs/2410.12628)
- Kiessling, *Kraken* — an OCR/HTR system for historical and non-Latin material —
  [github.com/mittagessen/kraken](https://github.com/mittagessen/kraken)
- Sauvola & Pietikäinen, *Adaptive document image binarization*, Pattern Recognition, 2000

Sample images are CC-0, digitised by eGangotri (funding: MIDF), from the CSU-Sringeri
Campus Rare Book and Manuscript Collection.
