"""Figure 3: a second modality adds compound-specific, conditional evidence.

Panels
  a  Gain from same-compound cross-modal evidence over three comparators, for GE
     evidence added to a CP estimate and CP evidence added to a GE estimate. A
     glyph table states what distinguishes each comparator.
  b  Joint prediction versus conditional updating of the same target-only GE estimate.
  c  Gain from adding GE or a second CP measurement to one CP measurement, without
     or with the other source already present.

Pooled estimates and 95% CIs are read from figures/data/figure_3/figure3_summary.csv.
Per-run points in b are means over conditions within each training seed of
figure3_paired_ge_scores.csv, and those in c are the seed rows of
conditional_marginal_gain.csv. The pooled values that can be recomputed from these
files are asserted against the summary before drawing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rp_style as S
from rp_style import C

DATA = S.ROOT / "figures" / "data" / "figure_3"
SEQ = DATA / "inputs" / "cross_modal" / "sequential_evidence" / "conditional_marginal_gain.csv"


def row(summary: pd.DataFrame, panel: str, contrast: str) -> pd.Series:
    hit = summary[(summary.panel == panel) & (summary.contrast == contrast)]
    if len(hit) != 1:
        raise ValueError(f"expected one row for {panel}/{contrast}, found {len(hit)}")
    return hit.iloc[0]


def close(a: float, b: float, what: str, tol: float = 1e-6) -> None:
    if abs(a - b) > tol:
        raise ValueError(f"{what}: {a} != {b}")


# ------------------------------------------------------------------ panel a

COMPARATORS = [
    # label, (evidence added, same compound, matched condition), contrast
    ("Target-only estimate", ("n", "-", "-"), "correct_minus_p0"),
    ("Shuffled evidence", ("y", "n", "n"), "correct_minus_shuffled"),
    ("Condition-matched evidence", ("y", "n", "y"), "correct_minus_matched_foreign"),
]
DIRECTIONS = [("A", "GE evidence →\nCP estimate", C["ge"]),
              ("B", "CP evidence →\nGE estimate", C["cp"])]


def glyph(ax, x: float, y: float, kind: str) -> None:
    if kind == "y":
        ax.plot([x], [y], "o", ms=4.0, color=C["ink"])
    elif kind == "n":
        ax.plot([x], [y], "o", ms=4.0, mfc="white", mec=C["ink"], mew=0.7)
    else:
        ax.plot([x - 1.0, x + 1.0], [y, y], color=C["note"], lw=0.7)


TABLE_W = 60.0               # glyph table in a
FOREST_LEFT = (63.0, 101.0)  # plot-area left edges of the two forest columns in a
FOREST_W = 25.0


def panel_a(sh: S.Sheet, summary: pd.DataFrame, top: float, height: float) -> list:
    rows_y = [0.0, -1.0, -2.0]
    ylim = (-2.6, 1.05)
    # glyph table, drawn in millimetre coordinates on a blank axes that shares the row positions
    tab = sh.ax(0.0, top, TABLE_W, height)
    tab.set_xlim(0, TABLE_W)
    tab.set_ylim(*ylim)
    tab.axis("off")
    cols = [("Evidence\nadded", 36.0), ("Same\ncompound", 46.0), ("Matched\ncondition", 56.0)]
    for name, x in cols:
        tab.text(x, 0.62, name, ha="center", va="bottom", color=C["note"], fontsize=S.PT_SMALL,
                 linespacing=1.3)
    for (label, marks, _), y in zip(COMPARATORS, rows_y):
        tab.text(0.5, y, label, ha="left", va="center")
        for (_, x), m in zip(cols, marks):
            glyph(tab, x, y, m)
    tab.text(0.5, 0.62, "Comparator", ha="left", va="bottom", color=C["note"],
             fontsize=S.PT_SMALL)

    width = FOREST_W
    out = []
    for left, (panel, title, color) in zip(FOREST_LEFT, DIRECTIONS):
        ax = sh.ax(left, top, width, height)
        out.append(ax)
        for (label, _, contrast), y in zip(COMPARATORS, rows_y):
            r = row(summary, panel, contrast)
            S.interval(ax, r.estimate, y, r.ci_low, r.ci_high, color, ms=4.0, lw=1.1)
            ax.text(1.0 + 0.04, y, S.fmt(r.estimate, 4), transform=ax.get_yaxis_transform(),
                    ha="left", va="center", color=color)
        ax.text(0.5, 0.62, title, transform=ax.get_yaxis_transform(), ha="center", va="bottom",
                color=color, linespacing=1.3)
        ax.plot([0, 0], [ylim[0], 0.45], color=C["rule"], lw=0.5, ls=(0, (2.5, 1.5)), zorder=0)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.003, 0.043)
        ax.set_xticks([0, 0.02, 0.04], ["0", "0.02", "0.04"])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    # one x-axis title shared by both forest columns, centred under the pair
    centre = (FOREST_LEFT[0] + FOREST_LEFT[1] + FOREST_W) / 2
    sh.text(centre, top + height + 4.2, "Gain of same-compound evidence over comparator (Fisher z)",
            ha="center", va="top")
    return out


# ------------------------------------------------------------------ panel b

def panel_b(sh: S.Sheet, summary: pd.DataFrame, box) -> object:
    paired = pd.read_csv(DATA / "figure3_paired_ge_scores.csv")
    cond = paired.groupby(["canonical_compound", "dose"])[["p0", "fusion", "update"]].mean()
    joint_c = cond["fusion"] - cond["p0"]
    upd_c = cond["update"] - cond["p0"]
    close(joint_c.mean(), row(summary, "C", "fusion_minus_p0").estimate, "b joint")
    close(upd_c.mean(), row(summary, "C", "update_minus_p0").estimate, "b update")
    close((upd_c - joint_c).mean(), row(summary, "C", "update_minus_fusion").estimate, "b difference")
    per_seed = paired.groupby("seed").apply(
        lambda g: pd.Series({"fusion_minus_p0": (g["fusion"] - g["p0"]).mean(),
                             "update_minus_p0": (g["update"] - g["p0"]).mean(),
                             "update_minus_fusion": (g["update"] - g["fusion"]).mean()}),
        include_groups=False)
    if len(per_seed) != 3:
        raise ValueError("b: expected three seeds")
    items = [("Joint input: GE and CP", "fusion_minus_p0", C["cp_mid"], 0.0),
             ("Conditional update: GE, then CP", "update_minus_p0", C["cp"], -1.0),
             ("Conditional − joint", "update_minus_fusion", C["ink"], -2.35)]
    ax = sh.ax(*box)
    for label, key, color, y in items:
        r = row(summary, "C", key)
        if key == "update_minus_fusion":
            S.rowband(ax, y, half=0.55)
        S.seeds(ax, per_seed[key].to_numpy(), np.full(3, y - 0.28), color, ms=2.4)
        S.interval(ax, r.estimate, y, r.ci_low, r.ci_high, color, ms=4.0, lw=1.1)
        ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                color=color)
        ax.text(1.0 + 0.04, y, S.fmt(r.estimate, 4), transform=ax.get_yaxis_transform(),
                ha="left", va="center", color=color)
    S.vline0(ax)
    ax.set_ylim(-3.05, 0.6)
    lo = min(per_seed.min().min(), summary[summary.panel == "C"].ci_low.min())
    hi = max(per_seed.max().max(), summary[summary.panel == "C"].ci_high.max())
    if lo < -0.003 or hi > 0.0148:
        raise ValueError(f"b: values [{lo}, {hi}] outside plotted range")
    ax.set_xlim(-0.003, 0.0148)
    ax.set_xticks([0, 0.005, 0.010], ["0", "0.005", "0.010"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Gain in GE agreement over the\ntarget-only GE estimate (Fisher z)")
    return ax


# ------------------------------------------------------------------ panel c

def panel_c(sh: S.Sheet, summary: pd.DataFrame, box) -> object:
    seq = pd.read_csv(SEQ)
    seed_rows = seq[seq.scope == "seed"]
    pooled = seq[seq.scope == "pooled_three_seed"].set_index("contrast")
    lines = [("+ second CP", C["cp"], [("CP2|CP1", "supplement"), ("CP2|CP1+GE", "supplement")], -0.05),
             ("+ GE", C["ge"], [("GE|CP1", "D"), ("GE|CP1+CP2", "D")], 0.05)]
    ax = sh.ax(*box)
    for label, color, pair, dx in lines:
        xs, ys = [], []
        for x, (contrast, panel) in enumerate(pair):
            r = row(summary, panel, contrast)
            close(pooled.loc[contrast, "excess_z_point"], r.estimate, f"c {contrast}")
            sd = seed_rows[seed_rows.contrast == contrast].excess_z_point.to_numpy()
            if len(sd) != 3:
                raise ValueError(f"c: {contrast} has {len(sd)} seeds")
            S.seeds(ax, np.full(3, x + dx + 0.09), sd, color, ms=2.4)
            S.interval(ax, x + dx, r.estimate, r.ci_low, r.ci_high, color, horizontal=False,
                       ms=4.0, lw=1.1)
            xs.append(x + dx)
            ys.append(r.estimate)
        ax.plot(xs, ys, color=color, lw=0.9, zorder=2)
        ax.text(xs[-1] + 0.2, ys[-1], label, color=color, va="center", ha="left")
    S.hline0(ax)
    ax.set_xlim(-0.35, 1.25)
    ax.set_ylim(0, 0.032)
    ax.set_yticks([0, 0.01, 0.02, 0.03], ["0", "0.01", "0.02", "0.03"])
    ax.set_xticks([0, 1], ["1 CP", "1 CP +\nother source"])
    ax.set_xlabel("Evidence already available")
    ax.set_ylabel("CP agreement gain\n(Fisher z)", linespacing=1.2)
    return ax


def main() -> None:
    # Compact grid (mm). Row 1: glyph table and two forest columns. Row 2: b and c
    # packed from the left and sharing top and bottom edges. The canvas is trimmed
    # to its content when saved.
    summary = pd.read_csv(DATA / "figure3_summary.csv")
    sh = S.Sheet(height_mm=80.0)
    ax_a1, ax_a2 = panel_a(sh, summary, top=7.0, height=15.0)
    t2, h2 = 35.0, 21.0
    ax_b = panel_b(sh, summary, (40.0, t2, FOREST_W, h2))
    ax_c = panel_c(sh, summary, (88.0, t2, 26.0, h2))
    for pid, ax in (("a1", ax_a1), ("a2", ax_a2), ("b", ax_b), ("c", ax_c)):
        sh.name(pid, ax)
    sh.row("a1", "a2")
    sh.row("b", "c")
    for letter, x, top in (("a", 0.0, 0.0), ("b", 0.0, t2 - 5.0), ("c", 76.0, t2 - 5.0)):
        sh.letter(letter, x, top)
    S.save(sh, "fig3_crossmodal")


if __name__ == "__main__":
    main()
