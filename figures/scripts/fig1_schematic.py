"""Figure 1: apply the current wording and the downstream cue to the authors' schematic.

The schematic itself is the authors' drawing, stored unchanged at
figures/data/figure_1/figure1_authors.pdf. This script edits that PDF in place of a
redraw, so every icon, colour and font stays the authors' own:

  * panel b heading  "Recover reproducible signals" -> "Construct reproducible response estimates"
  * panel c heading  "Cross-modal residual update"  -> "Conditional cross-modal update"
  * panel b, lower right: the reproducible estimate is passed down as the training
    target of a perturbation model whose input is a compound. The compound reuses
    the panel a molecule icon; arrows copy the existing grey and dashed-blue styles.

New text uses the fonts already embedded in the figure (Arial subsets); the script
fails if a required glyph is missing or if a replaced heading is not found exactly once.

Usage (from the repository root):
    python figures/scripts/fig1_schematic.py
"""
from __future__ import annotations

from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "figures" / "data" / "figure_1" / "figure1_authors.pdf"
OUT = ROOT / "outputs" / "figures" / "figure1.pdf"
PREVIEW = ROOT / "figures" / "fig1_schematic.png"

# (text operators of the original heading, new heading). The first operator carries
# the new text; any further operators of the same heading are emptied.
HEADINGS = [
    ([b"[(Recover )-12(repr)-3(oducibl)-3(e )8(signal)-3(s)] TJ"],
     "Construct reproducible response estimates", 24.0),
    ([b"[(Cros)-4(s)] TJ", b"[(-)] TJ",
      b"[(mod)4(al)-3( )9(re)-3(si)-3(du)4(al)-3( )4(up)4(date)] TJ"],
     "Conditional cross-modal update", 23.64),
]

INK = (0.0667, 0.0824, 0.129)          # text colour used throughout the figure
GREY = (0.38, 0.443, 0.529)            # solid flow arrows
BLUE = (0.173, 0.514, 0.804)           # dashed calibration arrow
RULE = (0.604, 0.659, 0.729)           # panel dividers
CARD = (0.93, 0.95, 0.975)             # light card fill
BODY_PT = 16.9                         # body label size in the figure


def fmt(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


class Stream:
    """Collects PDF operators in top-down page coordinates (y measured from the top)."""

    def __init__(self, height: float):
        self.h = height
        self.ops: list[str] = []

    def y(self, top: float) -> float:
        return self.h - top

    def text(self, font: str, size: float, x: float, baseline_top: float, s: str) -> None:
        esc = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.ops.append(f"BT /{font} {fmt(size)} Tf 1 0 0 1 {fmt(x)} {fmt(self.y(baseline_top))} Tm "
                        f"{' '.join(fmt(c) for c in INK)} rg ({esc}) Tj ET")

    def arrow(self, x0: float, y0: float, x1: float, y1: float, color, width: float,
              dashed: bool = False, head: float = 10.0, half: float = 4.6) -> None:
        """Straight arrow from (x0, y0) to its tip (x1, y1); horizontal or vertical only."""
        dx, dy = x1 - x0, y1 - y0
        n = (dx * dx + dy * dy) ** 0.5
        ux, uy = dx / n, dy / n
        bx, by = x1 - ux * head, y1 - uy * head          # base of the head
        rgb = " ".join(fmt(c) for c in color)
        self.ops.append(f"q {rgb} RG {rgb} rg {fmt(width)} w 0 J")
        if dashed:                                        # discrete segments, as drawn by the authors
            on, off, t = 6.0, 4.5, 0.0
            while t < n - head:
                t1 = min(t + on, n - head)
                self.ops.append(f"{fmt(x0 + ux * t)} {fmt(self.y(y0 + uy * t))} m "
                                f"{fmt(x0 + ux * t1)} {fmt(self.y(y0 + uy * t1))} l S")
                t += on + off
        else:
            self.ops.append(f"{fmt(x0)} {fmt(self.y(y0))} m {fmt(bx)} {fmt(self.y(by))} l S")
        px, py = -uy, ux
        self.ops.append(f"{fmt(x1)} {fmt(self.y(y1))} m "
                        f"{fmt(bx + px * half)} {fmt(self.y(by + py * half))} l "
                        f"{fmt(bx - px * half)} {fmt(self.y(by - py * half))} l h f Q")

    def card(self, x0: float, y0: float, x1: float, y1: float, r: float = 8.0) -> None:
        k = 0.5523 * r
        Y = self.y
        p = [f"{fmt(x0 + r)} {fmt(Y(y0))} m", f"{fmt(x1 - r)} {fmt(Y(y0))} l",
             f"{fmt(x1 - r + k)} {fmt(Y(y0))} {fmt(x1)} {fmt(Y(y0 + r - k))} {fmt(x1)} {fmt(Y(y0 + r))} c",
             f"{fmt(x1)} {fmt(Y(y1 - r))} l",
             f"{fmt(x1)} {fmt(Y(y1 - r + k))} {fmt(x1 - r + k)} {fmt(Y(y1))} {fmt(x1 - r)} {fmt(Y(y1))} c",
             f"{fmt(x0 + r)} {fmt(Y(y1))} l",
             f"{fmt(x0 + r - k)} {fmt(Y(y1))} {fmt(x0)} {fmt(Y(y1 - r + k))} {fmt(x0)} {fmt(Y(y1 - r))} c",
             f"{fmt(x0)} {fmt(Y(y0 + r))} l",
             f"{fmt(x0)} {fmt(Y(y0 + r - k))} {fmt(x0 + r - k)} {fmt(Y(y0))} {fmt(x0 + r)} {fmt(Y(y0))} c h"]
        self.ops.append(f"q {' '.join(fmt(c) for c in CARD)} rg {' '.join(fmt(c) for c in RULE)} RG "
                        f"1.2 w {' '.join(p)} B Q")

    def image(self, name: str, x0: float, y0: float, w: float, h: float) -> None:
        self.ops.append(f"q {fmt(w)} 0 0 {fmt(h)} {fmt(x0)} {fmt(self.y(y0 + h))} cm /{name} Do Q")


def widths(doc: pymupdf.Document, font_xref: int):
    font = doc.xref_object(font_xref)
    first = int(doc.xref_get_key(font_xref, "FirstChar")[1])
    kind, val = doc.xref_get_key(font_xref, "Widths")
    arr = doc.xref_object(int(val.split()[0])) if kind == "xref" else val
    w = [float(v) for v in arr.strip().strip("[]").split()]

    def measure(s: str, size: float) -> float:
        total = 0.0
        for ch in s:
            code = ch.encode("cp1252")[0]
            gw = w[code - first] if 0 <= code - first < len(w) else 0.0
            if gw == 0.0:
                raise ValueError(f"glyph {ch!r} is not in the embedded subset of {font.split()[0]}")
            total += gw
        return total * size / 1000.0
    return measure


def replace_once(raw: bytes, old: bytes, new: bytes) -> bytes:
    if raw.count(old) != 1:
        raise ValueError(f"expected one occurrence of {old[:40]!r}, found {raw.count(old)}")
    return raw.replace(old, new)


def main() -> None:
    doc = pymupdf.open(BASE)
    page = doc[0]
    fonts = {f[4]: f[0] for f in page.get_fonts()}          # resource name -> xref
    bold, regular = widths(doc, fonts["F1"]), widths(doc, fonts["F2"])
    contents = page.get_contents()
    if len(contents) != 1:
        raise ValueError("expected a single content stream")
    raw = doc.xref_stream(contents[0])

    # headings: same font, size and position, new wording
    for ops, heading, size in HEADINGS:
        bold(heading, size)
        raw = replace_once(raw, ops[0], f"[({heading})] TJ".encode("cp1252"))
        pos = raw.find(f"[({heading})] TJ".encode("cp1252"))
        for op in ops[1:]:                    # the next operators of the same heading, in order
            nxt = raw.find(op, pos)
            if nxt < 0 or nxt - pos > 1000:
                raise ValueError(f"heading continuation {op!r} not found after {heading!r}")
            raw = raw[:nxt] + b"[()] TJ" + raw[nxt + len(op):]
            pos = nxt
    doc.update_stream(contents[0], raw)

    # downstream cue in the empty lower right of panel b
    s = Stream(page.rect.height)
    target_x = 1271.5                       # centre of the reproducible-estimate bars (Image30)
    bars_bottom = 280.0
    box = (1188.0, 362.0, 1314.0, 414.0)
    s.card(*box)
    cx = (box[0] + box[2]) / 2
    for line, base in (("Perturbation", 384.5), ("model", 404.0)):
        s.text("F2", BODY_PT, cx - regular(line, BODY_PT) / 2, base, line)
    s.arrow(target_x, bars_bottom + 3.0, target_x, box[1] - 1.0, BLUE, 1.875, dashed=True)
    label = "Training target"
    s.text("F2", BODY_PT, target_x - 8.0 - regular(label, BODY_PT), 335.0, label)
    mol_w, mol_h = 60.0, 60.0 * 112.44 / 135.0              # panel a molecule, same aspect
    mol_x, mol_y = 1085.0, (box[1] + box[3]) / 2 - mol_h / 2
    s.image("Image9", mol_x, mol_y, mol_w, mol_h)
    mid = (box[1] + box[3]) / 2
    s.arrow(mol_x + mol_w + 6.0, mid, box[0] - 3.0, mid, GREY, 1.65)
    xref = doc.get_new_xref()
    doc.update_object(xref, "<<>>")
    doc.update_stream(xref, ("q\n" + "\n".join(s.ops) + "\nQ\n").encode("latin-1"))
    kind, val = doc.xref_get_key(page.xref, "Contents")
    doc.xref_set_key(page.xref, "Contents", f"[{val} {xref} 0 R]" if kind == "xref"
                     else f"{val[:-1]} {xref} 0 R]")

    doc.save(OUT, garbage=3, deflate=True)
    pymupdf.open(OUT)[0].get_pixmap(dpi=150).save(PREVIEW)
    print(f"wrote {OUT.relative_to(ROOT)} and {PREVIEW.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
