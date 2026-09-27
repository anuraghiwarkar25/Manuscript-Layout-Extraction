"""Manuscript layout region detection (header, footer, main_text, side_text, filler)."""
from .config import Config
from .pipeline import LayoutPipeline

__all__ = ["Config", "LayoutPipeline"]
