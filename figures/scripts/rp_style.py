"""Shared visual contract for the RePert main-text data figures (Figures 2-5).

Every main figure script imports this module and sets no rcParam of its own, so
one colour or one type size changes across the paper in one place.

Geometry. Figures are built at their true printed size, laid out on a canvas no
wider than the manuscript text block (A4 with 22 mm margins, 166 mm), then
trimmed to their content plus a 1 mm margin. A figure is as wide as its content
needs, not as wide as the page; the manuscript embeds it at natural size, so
every printed point size is the size set here. Axes are placed in millimetres measured from the top-left corner of the
canvas, which is how a layout is read and drawn.

Type. Body text 7 pt, secondary annotation 6 pt, panel letters 8 pt bold. No
panel titles: the claim of each panel is stated in the legend.

Colour grammar. One colour means one thing in all five figures:

    Cell Painting (CP)              blue   (as in the Figure 1 schematic)
    gene expression (GE)            green  (as in the Figure 1 schematic)
    ReCA                            orange (the focal estimate)
    IMR, aggregated-target IMR,
    empirical-Bayes shrinkage       a violet ramp, light to dark
    Raw measurement / support mean  neutral grey
    other comparators               neutral greys, separated by lightness

A modality colour is never used for a method, and ReCA's orange is never used
for anything else, so a reader can follow an entity across figures by colour.

Alignment. Panels that a reader sees as sharing an edge are declared with
Sheet.align(edge, ax, ...). save() refuses to write a figure in which a declared
edge differs by more than ALIGN_TOL_MM, and writes a layout manifest (plot-area
boxes in points plus the declared row and column groups) that the nature-figure
panel-alignment audit can read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
MM = 1.0 / 25.4

# REPERT_DRAFT=1 reports text collisions instead of failing, for layout iteration only.
DRAFT = os.environ.get("REPERT_DRAFT") == "1"

SHEET_W = 166.0          # mm; manuscript text block width
PT_BODY = 7.0
PT_SMALL = 6.0
PT_LETTER = 8.0
PT_MIN = 6.0             # nothing but panel letters may print below or above this ladder
ALIGN_TOL_MM = 0.05

C = {
    # modalities
    "cp": "#1F5A99",
    "cp_mid": "#5E8FC6",
    "cp_soft": "#C9D8EC",
    "ge": "#3B8040",
    "ge_mid": "#6FA86F",
    "ge_soft": "#CFE3CE",
    # focal estimate
    "reca": "#D0661C",
    "reca_soft": "#F4D2B8",
    # ReCA components, ordered light to dark
    "imr": "#BCAEDB",
    "agg": "#8670B8",
    "eb": "#4F3A86",
    # neutrals
    "raw": "#8C8C8C",
    "comp_light": "#BDBDBD",
    "comp_mid": "#8C8C8C",
    "comp_dark": "#4F4F4F",
    "ink": "#262626",
    "note": "#6B6B6B",
    "rule": "#9A9A9A",
    "grid": "#E3E3E3",
    "band": "#F1F1F1",
    "white": "#FFFFFF",
}


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": PT_BODY,
        "axes.labelsize": PT_BODY,
        "axes.titlesize": PT_BODY,
        "xtick.labelsize": PT_BODY,
        "ytick.labelsize": PT_BODY,
        "legend.fontsize": PT_SMALL,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "axes.labelpad": 2.0,
        "axes.edgecolor": C["ink"],
        "axes.labelcolor": C["ink"],
        "text.color": C["ink"],
        "xtick.color": C["ink"],
        "ytick.color": C["ink"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "lines.linewidth": 0.9,
        "lines.markersize": 3.0,
        "patch.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.borderpad": 0.0,
        "legend.labelspacing": 0.3,
        "axes.unicode_minus": True,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.facecolor": "white",
        "svg.hashsalt": "repert",
    })


apply()


class Sheet:
    """A figure canvas at true printed size with millimetre, top-left placement."""

    def __init__(self, height_mm: float, width_mm: float = SHEET_W):
        self.w = width_mm
        self.h = height_mm
        self.fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))
        self.fig.patch.set_linewidth(0)
        self.edges: list[tuple[str, list]] = []
        self.panels: dict[str, object] = {}
        self.rows: list[list[str]] = []
        self.cols: list[list[str]] = []
        self.letters: list[tuple[str, float, float]] = []

    def name(self, panel_id: str, ax) -> None:
        """Register a plot area under an id for the layout manifest."""
        self.panels[panel_id] = ax

    def align(self, edge: str, *axes) -> None:
        """Declare that the given axes share one edge: left, right, top or bottom."""
        if edge not in ("left", "right", "top", "bottom"):
            raise ValueError(edge)
        self.edges.append((edge, list(axes)))

    def row(self, *ids: str) -> None:
        """Panels that share top, bottom and height (checked by the nature-figure audit too)."""
        for i in ids:
            if i not in self.panels:
                raise KeyError(i)
        self.rows.append(list(ids))
        axes = [self.panels[i] for i in ids]
        self.align("top", *axes)
        self.align("bottom", *axes)

    def column(self, *ids: str) -> None:
        """Panels that share left, right and width."""
        for i in ids:
            if i not in self.panels:
                raise KeyError(i)
        self.cols.append(list(ids))
        axes = [self.panels[i] for i in ids]
        self.align("left", *axes)
        self.align("right", *axes)

    def ax(self, left: float, top: float, width: float, height: float, **kw):
        """Axes whose box spans [left, left+width] x [top, top+height] mm from the top-left."""
        ax = self.fig.add_axes([left / self.w, 1 - (top + height) / self.h,
                                width / self.w, height / self.h], **kw)
        # The canvas is already white; an axes background patch would only add a
        # filled rectangle that labels running past the axes edge appear to cross.
        ax.patch.set_visible(False)
        return ax

    def blank(self, left: float, top: float, width: float, height: float):
        """An axes with data coordinates equal to millimetres, for schematic drawing.

        x runs left to right and y runs top to bottom, both in mm from the axes
        top-left corner, so shapes are specified exactly as they are measured.
        """
        ax = self.ax(left, top, width, height)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")
        return ax

    def letter(self, letter: str, x: float, top: float) -> None:
        # 0.6 mm below the requested top: the PDF font box of an 8 pt letter reaches
        # above its glyph, and at top = 0 it would cross the page edge.
        self.letters.append((letter, x, top))
        self.fig.text(x / self.w, 1 - (top + 0.6) / self.h, letter, fontsize=PT_LETTER,
                      weight="bold", ha="left", va="top")

    def text(self, x: float, top: float, s: str, **kw):
        return self.fig.text(x / self.w, 1 - top / self.h, s, **kw)


def hline0(ax, y: float = 0.0, **kw) -> None:
    kw = {"color": C["rule"], "lw": 0.5, "ls": (0, (2.5, 1.5)), "zorder": 0, **kw}
    ax.axhline(y, **kw)


def vline0(ax, x: float = 0.0, **kw) -> None:
    kw = {"color": C["rule"], "lw": 0.5, "ls": (0, (2.5, 1.5)), "zorder": 0, **kw}
    ax.axvline(x, **kw)


def rowband(ax, y: float, half: float = 0.5, **kw) -> None:
    """A light grey band behind one row, used to mark the focal row."""
    ax.axhspan(y - half, y + half, color=C["band"], lw=0, zorder=0, **kw)


def only_bottom(ax) -> None:
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def interval(ax, x, y, lo, hi, color, *, horizontal=True, lw=1.0, ms=3.6,
             marker="o", zorder=3, mfc=None):
    """A point estimate with a 95% CI as a plain line (no caps)."""
    if horizontal:
        ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=zorder)
    else:
        ax.plot([x, x], [lo, hi], color=color, lw=lw, solid_capstyle="butt", zorder=zorder)
    ax.plot([x], [y], marker=marker, ms=ms, color=color, mfc=mfc or color,
            mec=color, mew=0.7, lw=0, zorder=zorder + 1)


def seeds(ax, xs, ys, color, *, ms=2.6, zorder=2):
    """Individual training runs as small open circles."""
    ax.plot(xs, ys, "o", ms=ms, mfc="white", mec=color, mew=0.6, lw=0, zorder=zorder)


def fmt(v: float, nd: int = 3, sign: bool = False) -> str:
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "−")


def check_no_clipped_data(fig, tol: float = 1e-9) -> None:
    """Raise if a data-coordinate line or marker falls outside its axes limits."""
    for ax in fig.axes:
        if not ax.axison:
            continue
        (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        for ln in ax.lines:
            if ln.get_transform() is not ax.transData or not ln.get_visible():
                continue
            x = np.asarray(ln.get_xdata(), float)
            y = np.asarray(ln.get_ydata(), float)
            ok = np.isfinite(x) & np.isfinite(y)
            if not ok.any():
                continue
            out = ((x[ok] < x0 - tol) | (x[ok] > x1 + tol) | (y[ok] < y0 - tol) | (y[ok] > y1 + tol))
            if out.any():
                i = int(np.flatnonzero(out)[0])
                raise ValueError(f"point ({x[ok][i]:.4g}, {y[ok][i]:.4g}) outside axes limits "
                                 f"x [{x0:.4g}, {x1:.4g}] y [{y0:.4g}, {y1:.4g}]")


def audit_text(fig) -> dict:
    """Check type sizes, text inside the canvas and pairwise text overlap.

    Returns a small report and raises on any failure, so a figure that breaks the
    type ladder or prints text over text is never written.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    page = fig.bbox
    items = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        size = t.get_fontsize()
        letter = t.get_fontweight() in ("bold", 700) and size == PT_LETTER and len(t.get_text()) == 1
        if not letter and size not in (PT_BODY, PT_SMALL):
            raise ValueError(f"text {t.get_text()!r} is {size} pt; allowed sizes are {PT_BODY} and {PT_SMALL}")
        bb = t.get_window_extent(r)
        if bb.x0 < page.x0 - 0.5 or bb.x1 > page.x1 + 0.5 or bb.y0 < page.y0 - 0.5 or bb.y1 > page.y1 + 0.5:
            if not DRAFT:
                raise ValueError(f"text {t.get_text()!r} leaves the canvas")
            print(f"DRAFT: text {t.get_text()!r} leaves the canvas")
        items.append((t, bb))
    clashes = []
    for i, (a, ba) in enumerate(items):
        for b, bb in items[i + 1:]:
            if ba.overlaps(bb):
                inter_w = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
                inter_h = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
                if inter_w > 0.8 and inter_h > 0.8:
                    clashes.append((a.get_text(), b.get_text()))
    if clashes:
        if DRAFT:
            print(f"DRAFT: overlapping text: {clashes}")
        else:
            raise ValueError(f"overlapping text: {clashes[:6]}")
    return {"n_text": len(items), "overlaps": len(clashes)}


def box_mm(sheet: Sheet, ax) -> dict[str, float]:
    p = ax.get_position()
    return {"left": p.x0 * sheet.w, "right": p.x1 * sheet.w,
            "top": (1 - p.y1) * sheet.h, "bottom": (1 - p.y0) * sheet.h}


def check_alignment(sheet: Sheet) -> dict:
    """Raise if a declared shared edge, or the letters of one row or column, disagree."""
    for edge, axes in sheet.edges:
        vals = [box_mm(sheet, ax)[edge] for ax in axes]
        if max(vals) - min(vals) > ALIGN_TOL_MM:
            raise ValueError(f"{edge} edges differ: {[round(v, 3) for v in vals]} mm")
    return {"shared_edges": len(sheet.edges), "rows": sheet.rows, "columns": sheet.cols}


def layout_manifest(sheet: Sheet) -> dict:
    """Plot-area boxes in points, in the schema of nature-figure's audit_panel_alignment.py."""
    pt = 72 / 25.4
    panels = []
    for pid, ax in sheet.panels.items():
        b = box_mm(sheet, ax)
        panels.append({"id": pid, "bbox_pt": [b["left"] * pt, (sheet.h - b["bottom"]) * pt,
                                              b["right"] * pt, (sheet.h - b["top"]) * pt]})
    out = {"schema_version": 1, "backend": "python-matplotlib",
           "figure": {"width_pt": sheet.w * pt, "height_pt": sheet.h * pt}, "panels": panels}
    if sheet.rows:
        out["row_groups"] = sheet.rows
    if sheet.cols:
        out["column_groups"] = sheet.cols
    return out


def trim(sheet: Sheet, margin_mm: float = 1.0) -> None:
    """Shrink the canvas to the drawn content plus a uniform margin.

    A figure does not have to fill the text width; outer white space is removed
    so the sheet is only as large as its content. Every axes and figure-level
    text is moved by the same offset in inches, so sizes, fonts and all
    declared alignments are unchanged.
    """
    fig = sheet.fig
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(r)                       # inches
    old_w, old_h = fig.get_size_inches()
    m = margin_mm * MM
    x0, y0 = tight.x0 - m, tight.y0 - m
    new_w, new_h = tight.width + 2 * m, tight.height + 2 * m
    boxes = [(ax, ax.get_position()) for ax in fig.axes]
    texts = [(t, t.get_position()) for t in fig.texts]
    fig.set_size_inches(new_w, new_h)
    for ax, pos in boxes:
        ax.set_position([(pos.x0 * old_w - x0) / new_w, (pos.y0 * old_h - y0) / new_h,
                         pos.width * old_w / new_w, pos.height * old_h / new_h])
    for t, (fx, fy) in texts:
        t.set_position(((fx * old_w - x0) / new_w, (fy * old_h - y0) / new_h))
    sheet.w, sheet.h = new_w / MM, new_h / MM


def save(sheet: Sheet, stem: str, notes: dict | None = None) -> Path:
    """Trim, then export PDF, SVG and PNG at true size after the data, text and alignment checks pass."""
    fig = sheet.fig
    trim(sheet)
    check_no_clipped_data(fig)
    report = audit_text(fig)
    report.update(check_alignment(sheet))
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(OUT / f"{stem}.svg", metadata={"Date": None})
    fig.savefig(OUT / f"{stem}.png", dpi=600)
    note_dir = OUT / "notes"
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / f"{stem}_render_audit.json").write_text(json.dumps(
        {"width_mm": sheet.w, "height_mm": sheet.h, **report, **(notes or {})}, indent=2))
    (note_dir / f"{stem}_layout.json").write_text(json.dumps(layout_manifest(sheet), indent=2))
    plt.close(fig)
    print(f"{stem}: {sheet.w:.1f} x {sheet.h:.1f} mm, {report['n_text']} text items")
    return pdf
