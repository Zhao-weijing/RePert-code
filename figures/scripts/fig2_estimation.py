"""Figure 2: independent measurements improve estimates of reproducible responses.

Panels
  a  Per-compound excess agreement in BBBC047 CP with one or two measurements per side.
  b  IMR agreement gain over the support mean, by dataset and support budget.
  c  Specificity gains of the three ReCA components, split into the change in
     agreement with the same compound and with condition-matched other compounds.
  d  Perturbation specificity of all methods in the BBBC047 benchmark.
  e  Same-compound retrieval (mAP@33) for the support mean, ReCA and cpDistiller-C.

Every printed number is read from figures/data/figure_2. Values recomputed here
(medians in a, the specificity identity in c) are asserted against the frozen
summary rows before anything is drawn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import rp_style as S
from rp_style import C

DATA = S.ROOT / "figures" / "data" / "figure_2"
BUDGETS = (1, 2, 3)


def close(a: float, b: float, what: str, tol: float = 1e-9) -> None:
    if abs(a - b) > tol:
        raise ValueError(f"{what}: {a} != {b}")


# ------------------------------------------------------------------ panel a

def panel_a(sh: S.Sheet, box) -> object:
    summary = pd.read_csv(DATA / "figure2_panelA.csv")
    summary = summary[summary.modality == "CP"].set_index("condition")
    per = pd.read_csv(DATA / "figure2_panelA_compounds.csv")
    per = per[per.modality == "CP"]
    one = per["one_to_one_same_minus_null"].to_numpy()
    two = per["two_to_two_same_minus_null"].to_numpy()
    gain = per["aggregation_gain"].to_numpy()
    close(np.median(one), summary.loc["one_to_one", "point"], "a one-vs-one median")
    close(np.median(two), summary.loc["two_to_two", "point"], "a two-vs-two median")
    close(np.median(gain), summary.loc["paired_aggregation_gain", "point"], "a paired gain median")
    close((gain > 0).mean(), summary.loc["paired_aggregation_gain", "fraction_positive"], "a fraction up")
    n = len(per)
    if n != int(summary.loc["one_to_one", "n_compounds"]):
        raise ValueError("a: compound count differs from summary")

    ax = sh.ax(*box)
    edges = np.arange(-0.6, 1.0001, 0.025)
    if one.min() < edges[0] or two.min() < edges[0] or max(one.max(), two.max()) > edges[-1]:
        raise ValueError("a: values outside histogram range")
    rows = [("One vs one", one, C["cp_soft"], C["cp_mid"], 1.15),
            ("Two vs two", two, C["cp_mid"], C["cp"], 0.0)]
    scale = 0.78 / max(np.histogram(v, edges, density=True)[0].max() for _, v, *_ in rows)
    centers = np.repeat(edges, 2)[1:-1]
    for label, v, fill, line, base in rows:
        dens = np.histogram(v, edges, density=True)[0] * scale
        ys = base + np.repeat(dens, 2)
        ax.fill_between(centers, base, ys, color=fill, lw=0, zorder=1)
        ax.plot(centers, ys, color=line, lw=0.6, zorder=2)
        ax.plot([edges[0], edges[-1]], [base, base], color=C["ink"], lw=0.4, zorder=2)
        med = np.median(v)
        ax.plot([med, med], [base, base + 0.84], color=C["ink"], lw=0.8, zorder=3)
        ax.text(med + 0.03, base + 0.84, S.fmt(med), fontsize=S.PT_SMALL, va="bottom", ha="left")
        ax.text(-0.6, base + 0.84, label, va="bottom", ha="left", color=line)
    S.vline0(ax)
    ax.set_xlim(-0.6, 1.0)
    ax.set_ylim(-0.05, 2.62)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-0.5, 0, 0.5, 1.0], ["−0.5", "0", "0.5", "1"])
    ax.set_xlabel("Excess agreement (PCC)")
    up = (gain > 0).mean()
    ax.text(1.0, 2.62, f"Paired change {S.fmt(np.median(gain), sign=True)}\n"
            f"{100 * up:.0f}% of compounds up", fontsize=S.PT_SMALL, ha="right", va="top",
            color=C["note"], linespacing=1.2)
    return ax


# ------------------------------------------------------------------ panel b

def panel_b(sh: S.Sheet, box) -> object:
    pooled = pd.read_csv(DATA / "figure2_panelB.csv")
    seeds = pd.read_csv(DATA / "figure2_panelB_seed.csv")
    ge = pd.read_csv(DATA / "figure2_panelC.csv")
    ge_seed = pd.read_csv(DATA / "figure2_panelC_seed.csv")
    ge_seed = ge_seed[ge_seed.scope == "seed"]
    # label anchors (x, y, ha): cpg0004 and BBBC047 CP meet at three supports,
    # so cpg0004 is labelled at its first point instead of its last.
    anchors = {"BBBC047 CP": (3.12, 0.014, "left"), "cpg0004 CP": (1.13, 0.146, "left"),
               "BBBC047 GE": (2.34, 0.0876, "left")}
    series = [
        ("BBBC047 CP", C["cp"], "o", -0.12,
         pooled[pooled.dataset == "BBBC047"][["budget", "point", "ci_low", "ci_high"]],
         seeds[seeds.dataset == "BBBC047"][["budget", "point"]]),
        ("cpg0004 CP", C["cp_mid"], "s", 0.0,
         pooled[pooled.dataset == "cpg0004"][["budget", "point", "ci_low", "ci_high"]],
         seeds[seeds.dataset == "cpg0004"][["budget", "point"]]),
        ("BBBC047 GE", C["ge"], "D", 0.12,
         ge[["budget", "point", "ci_low", "ci_high"]],
         ge_seed.rename(columns={"excess_z_point": "point"})[["budget", "point"]]),
    ]
    ax = sh.ax(*box)
    for label, color, marker, dx, pts, sd in series:
        pts = pts.sort_values("budget")
        for b, g in sd.groupby("budget"):
            if len(g) != 3:
                raise ValueError(f"b: {label} budget {b} has {len(g)} seeds")
        ax.plot(pts.budget + dx, pts.point, color=color, lw=0.8, zorder=2)
        for r in pts.itertuples():
            S.interval(ax, r.budget + dx, r.point, r.ci_low, r.ci_high, color, horizontal=False,
                       marker=marker, ms=3.4)
        S.seeds(ax, sd.budget + dx + 0.07, sd.point, color, ms=2.2)
        x, yv, ha = anchors[label]
        ax.text(x, yv, label, color=color, va="center", ha=ha)
    S.hline0(ax)
    ax.set_xlim(0.7, 3.25)
    ax.set_ylim(-0.005, 0.16)
    ax.set_xticks(BUDGETS)
    ax.set_yticks([0, 0.05, 0.10, 0.15], ["0", "0.05", "0.10", "0.15"])
    ax.set_xlabel("Support measurements")
    ax.set_ylabel("Agreement gain of IMR (Fisher z)")
    return ax


# ------------------------------------------------------------------ panel c

COMPONENTS = [("IMR", "IMR", C["imr"]), ("LSO", "Agg. IMR", C["agg"]), ("IMCEB", "EB shrinkage", C["eb"])]


def panel_c(sh: S.Sheet, box) -> object:
    d = pd.read_csv(DATA / "figure2_panelD_all_metrics.csv")
    e_rows = pd.read_csv(DATA / "figure2_panelD.csv")
    ax = sh.ax(*box)
    y = 0.0
    ticks, labels = [], []
    for b in BUDGETS:
        ax.text(-0.004, y + 0.05, f"{b} support" + ("s" if b > 1 else ""), ha="right", va="center",
                transform=ax.get_yaxis_transform(), color=C["note"], fontsize=S.PT_SMALL)
        y -= 1.0
        for key, name, color in COMPONENTS:
            g = d[(d.budget == b) & (d.method == key)].set_index("metric")
            same, foreign, de = g.loc["z_same", "delta"], g.loc["z_foreign", "delta"], g.loc["E", "delta"]
            close(same - foreign, de, f"c {key} {b}: E identity", tol=1e-9)
            ref = e_rows[(e_rows.budget == b) & (e_rows.method == key)].point.item()
            close(de, ref, f"c {key} {b}: E vs panel file")
            ax.plot([foreign, same], [y, y], color=color, lw=1.6, solid_capstyle="butt", zorder=2)
            ax.plot([foreign], [y], "o", ms=3.4, mfc="white", mec=color, mew=0.8, zorder=3)
            ax.plot([same], [y], "o", ms=3.4, color=color, zorder=3)
            ax.text(1.0 + 0.02, y, S.fmt(de), transform=ax.get_yaxis_transform(), ha="left",
                    va="center")
            ticks.append(y)
            labels.append(name)
            y -= 1.0
        y -= 0.35
    ax.text(1.0 + 0.02, 0.05, "ΔE", transform=ax.get_yaxis_transform(), ha="left",
            va="center", color=C["note"], fontsize=S.PT_SMALL)
    ax.set_yticks(ticks, labels)
    S.only_bottom(ax)
    S.vline0(ax)
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_xlim(-0.004, 0.14)
    ax.set_xticks([0, 0.05, 0.10], ["0", "0.05", "0.10"])
    ax.set_xlabel("Change from support mean (Fisher z)")
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], ls="", marker="o", ms=3.2, color=C["comp_dark"]),
            Line2D([], [], ls="", marker="o", ms=3.2, mfc="white", mec=C["comp_dark"], mew=0.8)]
    ax.legend(keys, ["same compound", "other compounds"], ncol=2, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), handletextpad=0.1, columnspacing=0.8,
              borderaxespad=0.3, labelcolor=C["note"])
    return ax


# ------------------------------------------------------------------ panel d

BENCH = [("CFRA", "ReCA", C["reca"]),
         ("Mean", "Support mean", C["raw"]),
         ("PCA", "PCA", C["comp_light"]),
         ("Noise2Self", "Noise2Self", C["comp_light"]),
         ("MLP", "MLP", C["comp_dark"]),
         ("ResNet", "ResNet", C["comp_dark"]),
         ("FT-Transformer", "FT-Transformer", C["comp_dark"]),
         ("TabM", "TabM", C["comp_dark"])]


def panel_d(sh: S.Sheet, top: float, height: float, columns) -> list:
    """One plot area per support budget; columns is a list of (left, width) in mm."""
    e = pd.read_csv(DATA / "figure2_panelE.csv")
    if set(e.method) != {k for k, *_ in BENCH}:
        raise ValueError(f"d: unexpected methods {sorted(set(e.method))}")
    ys = {k: -i for i, (k, *_) in enumerate(BENCH)}
    out = []
    for j, (b, (col_left, col_w)) in enumerate(zip(BUDGETS, columns)):
        ax = sh.ax(col_left, top, col_w, height)
        out.append(ax)
        g = e[e.budget == b].set_index("method")
        if len(g) != len(BENCH):
            raise ValueError(f"d: budget {b} has {len(g)} methods")
        lo, hi = g.point.min(), g.point.max()
        ax.set_xlim(lo - 0.012, hi + 0.012)
        S.rowband(ax, ys["CFRA"], half=0.5)
        ax.axvline(g.loc["Mean", "point"], color=C["rule"], lw=0.5, ls=(0, (2.5, 1.5)), zorder=0)
        ax.axhline(-1.5, color=C["grid"], lw=0.5, zorder=0)
        ax.axhline(-3.5, color=C["grid"], lw=0.5, zorder=0)
        for key, name, color in BENCH:
            v = g.loc[key, "point"]
            ax.plot([v], [ys[key]], "o", ms=3.6 if key == "CFRA" else 3.0, color=color, zorder=3)
        v = g.loc["CFRA", "point"]
        ax.text(v, ys["CFRA"] + 0.5, S.fmt(v), ha="center", va="bottom", color=C["reca"],
                fontsize=S.PT_SMALL)
        ax.set_ylim(-len(BENCH) + 0.4, 1.1)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ticks = np.arange(np.ceil((lo - 0.012) / 0.02) * 0.02, hi + 0.012, 0.02)
        ax.set_xticks(ticks, [f"{t:.2f}" for t in ticks])
        head = f"{b} support" + ("s" if b > 1 else "")
        ax.text(0.5, 1.0, head, transform=ax.transAxes, ha="center", va="bottom", color=C["note"],
                fontsize=S.PT_SMALL)
        if j == 0:
            for key, name, color in BENCH:
                ax.text(-0.04, ys[key], name, transform=ax.get_yaxis_transform(), ha="right",
                        va="center", color=C["reca"] if key == "CFRA" else C["ink"],
                        weight="bold" if key == "CFRA" else "normal")
        if j == 1:
            ax.set_xlabel("Perturbation specificity, E (Fisher z)")
    return out


# ------------------------------------------------------------------ panel e

RETRIEVAL = [("CFRA", "ReCA", C["reca"], "o", 0.0),
             ("RawMean", "Support mean", C["raw"], "o", -0.12),
             ("cpDistiller-C", "cpDistiller-C", C["comp_dark"], "s", 0.12)]


def panel_e(sh: S.Sheet, box) -> object:
    f = pd.read_csv(DATA / "figure2_panelF.csv")
    ax = sh.ax(*box)
    label_y = {}
    for key, name, color, marker, dx in RETRIEVAL:
        g = f[f.arm == key].sort_values("budget")
        if list(g.budget) != list(BUDGETS):
            raise ValueError(f"e: {key} budgets {list(g.budget)}")
        ax.plot(g.budget + dx, g.point, color=color, lw=0.8, zorder=2)
        for r in g.itertuples():
            S.interval(ax, r.budget + dx, r.point, r.ci_low, r.ci_high, color, horizontal=False,
                       marker=marker, ms=3.4)
        label_y[name] = (g.point.iloc[-1], color, dx)
    # direct labels, nudged apart where the last points sit close together
    order = sorted(label_y.items(), key=lambda kv: kv[1][0])
    placed = []
    for name, (v, color, dx) in order:
        yv = v if not placed else max(v, placed[-1] + 0.011)
        placed.append(yv)
        ax.text(3.28, yv, name, color=color, va="center", ha="left")
    ax.set_xlim(0.7, 3.25)
    ax.set_ylim(0.415, 0.545)
    ax.set_xticks(BUDGETS)
    ax.set_yticks([0.42, 0.46, 0.50, 0.54])
    ax.set_xlabel("Support measurements")
    ax.set_ylabel("Same-compound retrieval (mAP@33)")
    return ax


def main() -> None:
    # Grid (mm). Row 1: a | b | c share the top and bottom of their plot areas.
    # Row 2: the three columns of d and panel e share top and bottom; c and e
    # share their left edge, so the right-hand column reads as one column.
    sh = S.Sheet(height_mm=96.0)
    t1, h1 = 5.0, 34.0
    t2, h2 = 55.0, 32.0
    right_col = 121.0
    ax_a = panel_a(sh, (2.0, t1, 40.0, h1))
    ax_b = panel_b(sh, (58.5, t1, 29.5, h1))
    ax_c = panel_c(sh, (right_col, t1, 34.0, h1))
    gap = 4.0
    col_w = (101.0 - 22.0 - 2 * gap) / 3
    ax_d = panel_d(sh, t2, h2, [(22.0 + j * (col_w + gap), col_w) for j in range(3)])
    ax_e = panel_e(sh, (right_col, t2, 25.0, h2))
    for pid, ax in (("a", ax_a), ("b", ax_b), ("c", ax_c), ("d1", ax_d[0]), ("d2", ax_d[1]),
                    ("d3", ax_d[2]), ("e", ax_e)):
        sh.name(pid, ax)
    # a, b and c differ in kind, so they are checked pairwise for shared top and
    # bottom; the three columns of d are small multiples and are also checked for
    # equal width and gutter.
    sh.row("a", "b")
    sh.row("b", "c")
    sh.row("d1", "d2", "d3")
    sh.row("d3", "e")
    sh.align("left", ax_c, ax_e)
    for letter, x, top in (("a", 0.0, 0.0), ("b", 47.0, 0.0), ("c", 104.0, 0.0),
                           ("d", 0.0, t2 - 6.0), ("e", 104.0, t2 - 6.0)):
        sh.letter(letter, x, top)
    S.save(sh, "fig2_estimation")


if __name__ == "__main__":
    main()
