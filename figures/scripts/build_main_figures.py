"""Render main-text Figures 2-5 and install them in the submission.

Figure 1 is the authors' schematic; fig1_schematic.py applies the current wording to it
and is run separately because it needs PyMuPDF.

Each figure script runs in its own process, checks its source data, fails on any
clipped data point, off-ladder type size or text collision, and writes PDF, SVG
and PNG previews under figures/. The PDFs are then copied to
outputs/figures/ under the names the manuscript includes.

Usage (from the repository root):
    python figures/scripts/build_main_figures.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "figures" / "scripts"
TARGET = ROOT / "outputs" / "figures"
FIGURES = [("fig2_estimation", "figure2.pdf"), ("fig3_crossmodal", "figure3.pdf"), ("fig4_morphology", "figure4.pdf"),
           ("fig5_predictor", "figure5.pdf")]


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for stem, name in FIGURES:
        subprocess.run([sys.executable, str(SCRIPTS / f"{stem}.py")], check=True, cwd=SCRIPTS)
        shutil.copyfile(ROOT / "figures" / f"{stem}.pdf", TARGET / name)
        print(f"installed {name}")


if __name__ == "__main__":
    main()
