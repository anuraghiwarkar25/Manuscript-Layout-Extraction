"""All tunable parameters in one place. Values are relative to the estimated
character height or page size, so they transfer across resolutions."""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # --- preprocessing ---
    work_max_side: int = 1800          # images are processed at this long side, boxes mapped back
    illum_kernel_frac: float = 0.04    # background-estimation kernel, fraction of long side
    deskew: bool = True
    max_skew_deg: float = 10.0
    min_skew_deg: float = 0.3

    # --- page / substrate detection ---
    page_bg_delta: float = 18.0        # Lab distance border-vs-centre below which page == whole image
    page_min_area_frac: float = 0.15
    page_edge_erode_frac: float = 0.012  # ignore dark shadows along the leaf edge

    # --- ink / text-line detection ---
    sauvola_window_frac: float = 0.025
    sauvola_k: float = 0.25
    min_component_px: int = 12
    rule_line_len_frac: float = 0.35   # longer straight strokes are treated as ruling, not text
    smear_x: float = 1.1               # horizontal merge distance, x char height
    smear_y: float = 0.15
    row_merge_gap: float = 2.0         # join same-row fragments closer than this x char height
    tall_line_split: float = 1.5       # blobs taller than this x line pitch are split

    # --- classification ---
    core_width_frac: float = 0.5       # lines >= this fraction of p90 width define the text block
    gap_pitch_factor: float = 1.35     # header/footer need a gap of this x line pitch
    side_margin_tol: float = 0.6       # tolerance outside core x-range, x char height
    latin_threshold: float = 0.55
    pencil_contrast_ratio: float = 0.55
    use_ocr_script_check: bool = True  # Tesseract 'eng' used to spot English/Latin text

    # --- region merging ---
    merge_dy_pitch: float = 0.9
    merge_dx_char: float = 1.5

    def to_dict(self):
        return asdict(self)
