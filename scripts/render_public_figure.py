#!/usr/bin/env python3
"""Render the authors' unmodified Figure 1 PDF for README and project website.

This is a format conversion, not a redraw. The PDF remains the source of truth.
Run with: python scripts/render_public_figure.py
"""
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures/data/figure_1/figure1_authors.pdf"
TARGET = ROOT / "assets/repert_figure1.png"

def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Original Figure 1 not found: {SOURCE}")
    with pymupdf.open(SOURCE) as document:
        if document.page_count != 1:
            raise ValueError(f"Expected one-page Figure 1, found {document.page_count} pages")
        pixmap = document[0].get_pixmap(
            matrix=pymupdf.Matrix(2.4, 2.4),
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(TARGET)
    if TARGET.stat().st_size < 5_000:
        raise RuntimeError("Rendered Figure 1 is unexpectedly small")
    print(f"Rendered authors' Figure 1: {TARGET} ({pixmap.width}x{pixmap.height})")

if __name__ == "__main__":
    main()
