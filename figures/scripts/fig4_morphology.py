"""Figure 4: ReCA estimates recover morphological structure seen in independent references.

cpg0004-LINCS, one-support ReCA estimate versus the single measurement it was
derived from (Raw); the reference is the mean of four other measurements.

Panels
  a  Agreement with the reference by feature group: Raw and ReCA means, the paired
     difference with its 95% CI, and the share of compounds that improved.
  b  Recovery of the reference's 25 largest feature changes (overlap and direction),
     per compound, with arm means and 95% compound-bootstrap intervals.
  c  Dose concordance per compound, Raw against ReCA.

Means, paired differences and CIs are read from data/source_data. The
per-compound file is checked against every summary mean before drawing. Arm-level
intervals in b are computed here by a compound bootstrap with a fixed seed; the
all-group row in a is the per-compound mean over the seven groups, asserted equal
to the frozen macro summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import rp_style as S
from rp_style import C

DATA = S.ROOT / "data" / "source_data"
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20260929
GROUPS = ["DNA", "RNA", "ER", "Mito", "AGP", "Shape", "Cross-channel"]


def close(a: float, b: float, what: str, tol: float = 1e-9) -> None:
    if abs(a - b) > tol:
        raise ValueError(f"{what}: {a} != {b}")


def load():
    summary = pd.read_csv(DATA / "figure4_cfra_summary.csv").set_index("endpoint")
    macro = pd.read_csv(DATA / "figure4_cfra_module_macro.csv").iloc[0]
    per = pd.read_csv(DATA / "figure4_cfra_per_compound.csv")
    for endpoint, g in per.groupby("endpoint"):
        if g.compound.duplicated().any():
            raise ValueError(f"duplicate compounds in {endpoint}")
        s = summary.loc[endpoint]
        for col in ("raw", "cfra", "cfra_minus_raw"):
            close(g[col].mean(), s[col], f"{endpoint} {col}")
        if len(g) != int(s.n_compounds):
            raise ValueError(f"{endpoint}: n mismatch")
    groups = per[per.endpoint.str.startswith("A_module_fisher_z_")]
    per_macro = groups.groupby("compound")[["raw", "cfra"]].mean()
    close(per_macro.raw.mean(), macro.raw, "macro raw")
    close(per_macro.cfra.mean(), macro.cfra, "macro ReCA")
    if len(per_macro) != int(macro.n_compounds):
        raise ValueError("macro: n mismatch")
    return summary, macro, per, per_macro


# ------------------------------------------------------------------ panel a

A1 = (24.0, 50.0)   # (left, width) of the absolute-agreement plot area in a
A2 = (79.0, 34.0)   # (left, width) of the paired-difference plot area in a
DELTA_X, IMPROVED_X = 2.0, 20.0   # mm right of A2 for the difference and percentage columns


def panel_a(sh: S.Sheet, summary, macro, per, per_macro, top: float, height: float) -> tuple:
    rows = [("All groups", macro, per_macro.cfra - per_macro.raw)]
    for g in GROUPS:
        e = f"A_module_fisher_z_{g}"
        sub = per[per.endpoint == e]
        rows.append((g, summary.loc[e], sub.cfra - sub.raw))
    ys = [0.0] + [-1.35 - i for i in range(len(GROUPS))]
    ylim = (ys[-1] - 0.6, 1.3)

    ax1 = sh.ax(A1[0], top, A1[1], height)
    ax2 = sh.ax(A2[0], top, A2[1], height)
    for (name, s, diff), y in zip(rows, ys):
        focal = name == "All groups"
        for ax in (ax1, ax2):
            if focal:
                S.rowband(ax, y, half=0.55)
        ax1.plot([s.raw, s.cfra], [y, y], color=C["reca_soft"], lw=1.6, zorder=1,
                 solid_capstyle="butt")
        ax1.plot([s.raw], [y], "o", ms=3.8, color=C["raw"], zorder=3)
        ax1.plot([s.cfra], [y], "o", ms=3.8, color=C["reca"], zorder=3)
        ax1.text(-0.03, y, name, transform=ax1.get_yaxis_transform(), ha="right", va="center",
                 weight="bold" if focal else "normal")
        S.interval(ax2, s.cfra_minus_raw, y, s.ci_low, s.ci_high, C["reca"], ms=3.8, lw=1.1)
        ax2.text(1.0 + DELTA_X / A2[1], y, S.fmt(s.cfra_minus_raw, 3, sign=True),
                 transform=ax2.get_yaxis_transform(), ha="left", va="center", color=C["reca"])
        ax2.text(1.0 + IMPROVED_X / A2[1], y, f"{100 * (diff > 0).mean():.0f}%",
                 transform=ax2.get_yaxis_transform(), ha="right", va="center")
    for ax in (ax1, ax2):
        ax.set_ylim(*ylim)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    ax2.plot([0, 0], [ylim[0], 0.7], color=C["rule"], lw=0.5, ls=(0, (2.5, 1.5)), zorder=0)
    ax1.set_xlim(0.3, 0.95)
    ax1.set_xticks([0.4, 0.6, 0.8])
    ax1.set_xlabel("Agreement with reference (Fisher z)")
    ax2.set_xlim(-0.01, 0.23)
    ax2.set_xticks([0, 0.1, 0.2], ["0", "0.1", "0.2"])
    ax2.set_xlabel("ReCA − Raw (Fisher z)")
    ax2.text(1.0 + DELTA_X / A2[1], 0.95, "Δ", transform=ax2.get_yaxis_transform(), ha="left",
             va="center", color=C["note"], fontsize=S.PT_SMALL)
    ax2.text(1.0 + IMPROVED_X / A2[1], 0.95, "Improved", transform=ax2.get_yaxis_transform(), ha="right",
             va="center", color=C["note"], fontsize=S.PT_SMALL)
    keys = [Line2D([], [], ls="", marker="o", ms=3.8, color=C["raw"]),
            Line2D([], [], ls="", marker="o", ms=3.8, color=C["reca"])]
    ax1.legend(keys, ["Raw", "ReCA"], ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0),
               handletextpad=0.1, columnspacing=1.0, borderaxespad=0.2)
    return ax1, ax2


# ------------------------------------------------------------------ panel b

def panel_b(sh: S.Sheet, summary, per, rng, top: float, height: float, boxes) -> list:
    """boxes: two (left, width) plot areas in mm."""
    items = [("B_top25_overlap", "Top-25 feature overlap", (0.0, 0.8), [0, 0.2, 0.4, 0.6, 0.8]),
             ("B_top25_direction", "Top-25 direction agreement", (0.45, 1.02), [0.5, 0.75, 1.0])]
    out = []
    for (endpoint, label, ylim, ticks), (left, w) in zip(items, boxes):
        g = per[per.endpoint == endpoint]
        s = summary.loc[endpoint]
        ax = sh.ax(left, top, w, height)
        out.append(ax)
        for r in g.itertuples():
            ax.plot([0, 1], [r.raw, r.cfra], color=C["grid"], lw=0.35, zorder=1)
        for x, col, color in ((0, "raw", C["raw"]), (1, "cfra", C["reca"])):
            vals = g[col].to_numpy()
            idx = rng.integers(0, len(vals), size=(BOOTSTRAP_ROUNDS, len(vals)))
            lo, hi = np.percentile(vals[idx].mean(axis=1), [2.5, 97.5])
            S.interval(ax, x, vals.mean(), lo, hi, color, horizontal=False, ms=4.2, lw=1.3, zorder=4)
        ax.plot([0, 1], [g.raw.mean(), g.cfra.mean()], color=C["ink"], lw=0.8, zorder=3)
        if g[["raw", "cfra"]].min().min() < ylim[0] or g[["raw", "cfra"]].max().max() > ylim[1]:
            raise ValueError(f"b: {endpoint} outside plotted range")
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(*ylim)
        ax.set_yticks(ticks)
        ax.set_xticks([0, 1], ["Raw", "ReCA"])
        ax.set_ylabel(label)
        ax.text(0.5, 1.0, f"Δ {S.fmt(s.cfra_minus_raw, 3, sign=True)}", transform=ax.transAxes,
                ha="center", va="bottom", color=C["reca"])
    return out


# ------------------------------------------------------------------ panel c

def panel_c(sh: S.Sheet, summary, per, left: float, top: float, size: float) -> object:
    endpoint = "C_macro_trajectory_spearman"
    g = per[per.endpoint == endpoint]
    s = summary.loc[endpoint]
    up = g.cfra > g.raw
    ax = sh.ax(left, top, size, size)
    ax.plot([-0.4, 1.0], [-0.4, 1.0], color=C["rule"], lw=0.5, ls=(0, (2.5, 1.5)), zorder=1)
    ax.scatter(g.raw[~up], g.cfra[~up], s=4, color=C["raw"], lw=0, alpha=0.8, zorder=2)
    ax.scatter(g.raw[up], g.cfra[up], s=4, color=C["reca"], lw=0, alpha=0.8, zorder=2)
    lo = min(g.raw.min(), g.cfra.min())
    if lo < -0.4 or max(g.raw.max(), g.cfra.max()) > 1.0:
        raise ValueError("c: dose concordance outside plotted range")
    ax.set_xlim(-0.4, 1.0)
    ax.set_ylim(-0.4, 1.0)
    ax.set_aspect("equal")
    ticks = [0, 0.5, 1.0]
    ax.set_xticks(ticks, ["0", "0.5", "1"])
    ax.set_yticks(ticks, ["0", "0.5", "1"])
    ax.set_xlabel("Raw dose concordance")
    ax.set_ylabel("ReCA dose concordance")
    ax.text(-0.36, 0.98, f"Δ {S.fmt(s.cfra_minus_raw, 3, sign=True)}\n"
            f"[{S.fmt(s.ci_low, 3)}, {S.fmt(s.ci_high, 3)}]", ha="left", va="top",
            color=C["reca"], linespacing=1.2)
    ax.text(0.98, -0.37, f"{int(up.sum())} of {len(g)} improved", ha="right", va="bottom",
            color=C["note"], fontsize=S.PT_SMALL)
    return ax


def main() -> None:
    # Compact grid (mm). Row 2 spans the same width as row 1: b starts on the left
    # edge of a's left plot area and c ends under the right end of a's percentage
    # column; b and c share top and bottom. The canvas is trimmed when saved.
    summary, macro, per, per_macro = load()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sh = S.Sheet(height_mm=98.0)
    ax_a1, ax_a2 = panel_a(sh, summary, macro, per, per_macro, top=6.0, height=36.0)
    t2, h2 = 57.0, 30.0
    b_w, b_gap = 26.0, 14.0
    ax_b1, ax_b2 = panel_b(sh, summary, per, rng, t2, h2, [(A1[0], b_w), (A1[0] + b_w + b_gap, b_w)])
    c_right = A2[0] + A2[1] + IMPROVED_X
    ax_c = panel_c(sh, summary, per, left=c_right - h2, top=t2, size=h2)
    for pid, ax in (("a1", ax_a1), ("a2", ax_a2), ("b1", ax_b1), ("b2", ax_b2), ("c", ax_c)):
        sh.name(pid, ax)
    sh.row("a1", "a2")
    sh.row("b1", "b2")
    sh.row("b2", "c")
    sh.align("left", ax_a1, ax_b1)
    for letter, x, top in (("a", 0.0, 0.0), ("b", 0.0, t2 - 6.0), ("c", c_right - h2 - 12.0, t2 - 6.0)):
        sh.letter(letter, x, top)
    S.save(sh, "fig4_morphology")


if __name__ == "__main__":
    main()
