"""Figure 5: reproducibility-aware training targets improve unseen-compound prediction in BBBC047.

Raw is trained on the mean of two, three or four selected measurements per compound
(2R, 3R, 4R); ReCA+Res is trained on the same measurements, split into the mean
one-support ReCA estimate and the residual. Both are scored against the same
measured profiles of unseen compounds.

Panels (one column per metric: PCC, delta PCC, RMSE, specificity E)
  a  Mean score of each predictor with its 95% compound-bootstrap CI.
  b  Paired gain of ReCA+Res over Raw with its 95% CI, signed so that a positive
     value favours ReCA+Res (for RMSE, the decrease).

All values are read from data/source_data/predictor_main_source.csv;
nothing is recomputed here. The script checks that every BBBC047 metric-budget
cell is present exactly once and that each paired difference lies between its CI
bounds before drawing.
"""
from __future__ import annotations

import pandas as pd
from matplotlib.lines import Line2D

import rp_style as S
from rp_style import C

DATA = S.ROOT / "data" / "source_data" / "predictor_main_source.csv"
BUDGETS = ["2R", "3R", "4R"]
DODGE = 0.17

# (source metric, panel-a label, panel-b label, sign that makes a gain positive,
#  panel-a limits and ticks, panel-b limits and ticks)
METRICS = [
    ("full_target_pcc", "PCC", "Gain in PCC", 1.0,
     (0.27, 0.40), [0.28, 0.32, 0.36, 0.40], (-0.0012, 0.0136), [0, 0.005, 0.010]),
    ("delta_pcc", "ΔPCC", "Gain in ΔPCC", 1.0,
     (0.31, 0.44), [0.32, 0.36, 0.40, 0.44], (-0.0015, 0.0160), [0, 0.005, 0.010, 0.015]),
    ("raw_target_rmse", "RMSE", "Decrease in RMSE", -1.0,
     (0.60, 0.80), [0.60, 0.70, 0.80], (-0.0010, 0.0106), [0, 0.005, 0.010]),
    ("E", "Perturbation specificity,\nE (Fisher z)", "Gain in E (Fisher z)", 1.0,
     (0.19, 0.36), [0.20, 0.25, 0.30, 0.35], (-0.0025, 0.0300), [0, 0.01, 0.02, 0.03]),
]

AX_W = 25.0      # mm, plot-area width of every column
GAP = 14.5       # mm between columns, room for the tick labels and y-axis title
LEFT = 13.0      # mm, left edge of the first column
TOP_A, H_A = 6.0, 27.0
TOP_B, H_B = 45.0, 24.0


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df[df.dataset == "BBBC047"].copy()
    want = {(m[0], b) for m in METRICS for b in BUDGETS}
    have = list(zip(df.metric, df.setting))
    if sorted(have) != sorted(want):
        raise ValueError(f"expected one row per metric and budget, got {sorted(have)}")
    if (df.comparator_method != "CFRA+Res").any():        # CFRA is the source name of ReCA
        raise ValueError("unexpected comparator in the predictor source file")
    bad = df[(df.difference < df.difference_ci_low) | (df.difference > df.difference_ci_high)]
    if len(bad):
        raise ValueError(f"paired difference outside its CI: {bad[['metric', 'setting']].values}")
    return df.set_index(["metric", "setting"])


def check_range(values, lim, what: str) -> None:
    if min(values) < lim[0] or max(values) > lim[1]:
        raise ValueError(f"{what}: values [{min(values):.4f}, {max(values):.4f}] outside {lim}")


def tick_labels(ticks) -> list[str]:
    """Same number of decimals on every tick of one axis; zero printed as 0."""
    nd = max(len(f"{t:g}".partition(".")[2]) for t in ticks)
    return ["0" if t == 0 else f"{t:.{nd}f}" for t in ticks]


def panel_a(sh: S.Sheet, df: pd.DataFrame, left: float, metric) -> object:
    key, label, _, _, lim, ticks, _, _ = metric
    ax = sh.ax(left, TOP_A, AX_W, H_A)
    vals = []
    for x, b in enumerate(BUDGETS):
        r = df.loc[(key, b)]
        ax.plot([x - DODGE, x + DODGE], [r.raw_mean, r.comparator_mean], color=C["reca_soft"],
                lw=1.4, zorder=1, solid_capstyle="butt")
        S.interval(ax, x - DODGE, r.raw_mean, r.raw_ci_low, r.raw_ci_high, C["raw"],
                   horizontal=False, ms=3.6, lw=1.0)
        S.interval(ax, x + DODGE, r.comparator_mean, r.comparator_ci_low, r.comparator_ci_high,
                   C["reca"], horizontal=False, ms=3.6, lw=1.0)
        vals += [r.raw_ci_low, r.raw_ci_high, r.comparator_ci_low, r.comparator_ci_high]
    check_range(vals, lim, f"a {key}")
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(*lim)
    ax.set_yticks(ticks, tick_labels(ticks))
    ax.set_xticks(range(3), BUDGETS)
    ax.set_ylabel(label, linespacing=1.2)
    return ax


def panel_b(sh: S.Sheet, df: pd.DataFrame, left: float, metric) -> object:
    key, _, label, sign, _, _, lim, ticks = metric
    ax = sh.ax(left, TOP_B, AX_W, H_B)
    vals = []
    for x, b in enumerate(BUDGETS):
        r = df.loc[(key, b)]
        est = sign * r.difference
        lo, hi = sorted((sign * r.difference_ci_low, sign * r.difference_ci_high))
        S.interval(ax, x, est, lo, hi, C["reca"], horizontal=False, ms=3.6, lw=1.1)
        ax.text(x, 1.0, S.fmt(est, 4), transform=ax.get_xaxis_transform(), ha="center",
                va="top", color=C["reca"], fontsize=S.PT_SMALL)
        vals += [lo, hi]
    check_range(vals, lim, f"b {key}")
    # the value row occupies the top 3 mm of the plot area; no interval may reach it
    if max(vals) > lim[1] - (lim[1] - lim[0]) * 3.0 / H_B:
        raise ValueError(f"b {key}: an interval reaches the value row")
    S.hline0(ax)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(*lim)
    ax.set_yticks(ticks, tick_labels(ticks))
    ax.set_xticks(range(3), BUDGETS)
    ax.set_ylabel(label)
    return ax


def main() -> None:
    df = load()
    sh = S.Sheet(height_mm=80.0)
    ids = []
    for i, metric in enumerate(METRICS):
        left = LEFT + i * (AX_W + GAP)
        ax_a = panel_a(sh, df, left, metric)
        ax_b = panel_b(sh, df, left, metric)
        sh.name(f"a{i + 1}", ax_a)
        sh.name(f"b{i + 1}", ax_b)
        sh.column(f"a{i + 1}", f"b{i + 1}")
        ids.append(i + 1)
        if i == 0:
            keys = [Line2D([], [], ls="", marker="o", ms=3.6, color=C["raw"]),
                    Line2D([], [], ls="", marker="o", ms=3.6, color=C["reca"])]
            ax_a.legend(keys, ["Raw", "ReCA+Res"], ncol=2, loc="lower left",
                        bbox_to_anchor=(0.0, 1.0), handletextpad=0.1, columnspacing=1.0,
                        borderaxespad=0.2)
    sh.row(*[f"a{i}" for i in ids])
    sh.row(*[f"b{i}" for i in ids])
    sh.letter("a", 0.0, 0.0)
    sh.letter("b", 0.0, TOP_B - 5.0)
    S.save(sh, "fig5_predictor")


if __name__ == "__main__":
    main()
