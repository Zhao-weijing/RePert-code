"""Render Figure 4 with complementary, endpoint-specific visual grammars.

The renderer is downstream-only.  It reads the locked CFRA tables and NPZ
bundle used by the earlier Figure 4 renderers, then writes a new candidate
directory.  It does not refit, select, or replace any model and it never
touches the manuscript or the formal submission figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
MPL_LIB = ROOT / "publication_figures_20260910" / ".python_libs"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
if str(MPL_LIB) not in sys.path:
    sys.path.insert(0, str(MPL_LIB))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd

from render_figure4_cfra_nature import (
    DOSES,
    EXAMPLE_COMPOUND,
    MODULE_LABELS,
    MODULE_ORDER,
    load_inputs,
)


VERSION = "cpg0004-figure4-cfra-mixed-v10-2026-09-22"
STATUS = "HISTORICAL_LOCKED_RECOMPUTE"
CANVAS_MM = (180.34, 106.00)
BOOTSTRAP_ROUNDS = 10_000
B_BOOTSTRAP_VERSION = "cpg0004-figure4-cfra-physical-support-v1-2026-09-20"

# Figure 2 typography and axes, retaining the current Figure 4 semantic colors.
TEXT = "#25282C"
AXIS = "#555B61"
RAW = "#A5ABB3"
REFERENCE = "#252A31"
PURPLE = "#6F5DA8"
ORANGE = "#C47A37"
ZERO = "#B9BEC5"
CONNECTOR = "#D8DCE2"
WHITE = "#FFFFFF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def b_bootstrap_seed(endpoint: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{B_BOOTSTRAP_VERSION}|{endpoint}".encode("utf-8")).digest()[:8],
        "little",
    )


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.labelsize": 7.4,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
            "legend.fontsize": 6.7,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": AXIS,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.4,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: plt.Axes, *, hide_left: bool = False, hide_bottom: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    if hide_left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    if hide_bottom:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    ax.tick_params(pad=2.0)


def zero_vline(ax: plt.Axes) -> None:
    ax.axvline(0, color=ZERO, lw=0.6, ls=(0, (3, 3)), zorder=0)


def panel_letter(fig: plt.Figure, letter: str, x: float, y: float) -> None:
    # Figure 2 convention: a bare lower-case letter, with no panel subtitle.
    fig.text(x, y, letter.lower(), fontsize=8.3, fontweight="bold", color=TEXT, va="bottom")


def panel_a(fig: plt.Figure, data: dict[str, Any]) -> pd.DataFrame:
    panel_letter(fig, "a", 0.018, 0.940)
    # Reserve a dedicated right-hand strip for the percent summary.  The
    # plotting width is intentionally narrower than the previous candidate so
    # the high RNA/CFRA endpoint cannot touch the 85% label.
    ax = fig.add_axes([0.105, 0.570, 0.175, 0.310])
    frame = data["a"].set_index("module").loc[list(MODULE_ORDER)].reset_index().copy()
    frame["label"] = frame["module"].map(MODULE_LABELS)
    percentages: list[float] = []
    y = np.arange(len(frame))[::-1]
    for yi, (_, row) in zip(y, frame.iterrows()):
        raw = float(row["raw"])
        cfra = float(row["cfra"])
        ax.plot([raw, cfra], [yi, yi], color=CONNECTOR, lw=1.15, solid_capstyle="round", zorder=1)
        ax.plot(raw, yi, marker="o", ms=3.5, color=RAW, mec=WHITE, mew=0.3, zorder=3)
        # Keep the two arms visually comparable; color carries the method.
        ax.plot(cfra, yi, marker="o", ms=3.5, color=PURPLE, mec=WHITE, mew=0.3, zorder=4)
        endpoint = f"A_module_fisher_z_{row['module']}"
        compound = data["per_compound"][data["per_compound"]["endpoint"].astype(str).eq(endpoint)]
        if len(compound) == 0:
            raise ValueError(f"missing compound rows for {endpoint}")
        pct = 100.0 * float(compound["cfra_minus_raw"].astype(float).gt(0).mean())
        percentages.append(pct)
        ax.text(1.17, yi, f"{pct:.0f}%", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.5, color=TEXT)

    ax.set_xlim(0.30, 0.96)
    ax.set_ylim(-0.65, len(frame) - 0.20)
    ax.set_yticks(y, frame["label"].tolist())
    ax.set_xticks([0.4, 0.6, 0.8])
    ax.set_xlabel("Module agreement (Fisher z)", labelpad=2)
    clean_axis(ax)
    # The legend is deliberately one compact row, with line-and-circle
    # swatches separated from their labels (the sample's visual grammar).
    handles = [
        Line2D([0], [0], color=RAW, marker="o", lw=0.9, ms=3.4, mec=WHITE, mew=0.3, label="Raw"),
        Line2D([0], [0], color=PURPLE, marker="o", lw=0.9, ms=3.4, mec=WHITE, mew=0.3, label="ReCA"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.00, 1.035),
        ncol=2,
        frameon=False,
        handlelength=0.95,
        columnspacing=0.85,
        handletextpad=0.30,
        borderaxespad=0,
        fontsize=6.6,
    )
    ax.text(1.17, 1.07, "% improved", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=TEXT)
    # Separate the cross-channel summary from the six module rows.  Extend
    # the divider over the label, plotting, and percent columns so the group
    # boundary reads at the panel level rather than only inside the axes.
    ax.axhline(0.5, color="#D2D7DE", lw=0.55, ls=(0, (3, 3)), zorder=0)
    y_fig = 0.570 + 0.310 * ((0.5 + 0.65) / ((len(frame) - 0.20) + 0.65))
    fig.add_artist(
        Line2D(
            [0.018, 0.314],
            [y_fig, y_fig],
            transform=fig.transFigure,
            color="#D2D7DE",
            lw=0.55,
            ls=(0, (3, 3)),
            zorder=0,
        )
    )
    frame["percent_improved"] = percentages
    return frame


def _bootstrap_b_arms(frame: pd.DataFrame, summary_row: pd.Series) -> pd.DataFrame:
    """Paired compound bootstrap for the two displayed arm means."""
    frame = frame.sort_values("compound", kind="stable").reset_index(drop=True)
    values = frame[["raw", "cfra"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or len(values) < 2:
        raise ValueError("invalid panel B compound arms")
    expected = np.asarray([float(summary_row["raw"]), float(summary_row["cfra"])])
    observed = values.mean(axis=0)
    if not np.allclose(observed, expected, atol=5e-12, rtol=0):
        raise ValueError(f"panel B arm means do not match locked summary: {observed} vs {expected}")
    endpoint = str(summary_row["endpoint"])
    seed = b_bootstrap_seed(endpoint)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_ROUNDS, len(values)))
    samples = values[indices].mean(axis=1)
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    delta_samples = samples[:, 1] - samples[:, 0]
    delta_interval = np.quantile(delta_samples, [0.025, 0.975])
    locked_delta_interval = np.asarray([float(summary_row["ci_low"]), float(summary_row["ci_high"])])
    if not np.allclose(delta_interval, locked_delta_interval, atol=1e-12, rtol=0):
        raise ValueError(f"panel B delta bootstrap CI does not match locked summary: {delta_interval} vs {locked_delta_interval}")
    return pd.DataFrame(
        [
            {
                "endpoint": endpoint,
                "arm": arm,
                "point": float(point),
                "ci_low": float(intervals[0, index]),
                "ci_high": float(intervals[1, index]),
                "delta": float(summary_row["cfra_minus_raw"]),
                "delta_ci_low": float(summary_row["ci_low"]),
                "delta_ci_high": float(summary_row["ci_high"]),
                "n_compounds": int(len(values)),
                "bootstrap_rounds": BOOTSTRAP_ROUNDS,
                "bootstrap_seed": str(seed),
                "interval_origin": "derived paired compound bootstrap of frozen arm values",
            }
            for index, (arm, point) in enumerate(zip(("Raw", "CFRA"), observed))
        ]
    )


def _mini_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    label: str,
    *,
    ylim: tuple[float, float],
    yticks: list[float],
    show_xlabels: bool,
    label_y: float,
) -> None:
    arms = frame.set_index("arm").loc[["Raw", "CFRA"]]
    points = arms["point"].to_numpy(dtype=float)
    lows = arms["ci_low"].to_numpy(dtype=float)
    highs = arms["ci_high"].to_numpy(dtype=float)
    colors = [RAW, PURPLE]
    ax.bar([0, 1], points, width=0.42, color=colors, edgecolor=WHITE, linewidth=0.35, zorder=2)
    for x, point, low, high, color in zip((0, 1), points, lows, highs, colors):
        ax.errorbar(
            x,
            point,
            yerr=[[point - low], [high - point]],
            fmt="none",
            ecolor=color,
            elinewidth=0.60,
            capsize=1.5,
            capthick=0.60,
            zorder=4,
        )
    span = ylim[1] - ylim[0]
    # Keep a readable vertical sequence above each bar: CI cap -> point value
    # -> bracket -> delta.  The extra clearance is carried by the expanded
    # y-limits supplied by panel_b below, rather than by changing any data.
    for x, point, high, color in zip((0, 1), points, highs, colors):
        ax.text(
            x,
            high + 0.026 * span,
            f"{point:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=TEXT,
            zorder=5,
        )

    bracket_y = max(highs) + 0.230 * span
    tick = 0.018 * span
    ax.plot([0, 0, 1, 1], [bracket_y - tick, bracket_y, bracket_y, bracket_y - tick], color=AXIS, lw=0.50, clip_on=False, zorder=5)
    delta = float(frame["delta"].iloc[0])
    ax.text(0.5, bracket_y + 0.030 * span, f"Δ = +{delta:.3f}", ha="center", va="bottom", fontsize=6.2, color=TEXT)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(*ylim)
    if show_xlabels:
        ax.set_xticks([0, 1], ["Raw", "ReCA"])
        ax.get_xticklabels()[0].set_color(RAW)
        ax.get_xticklabels()[1].set_color(PURPLE)
    else:
        ax.set_xticks([])
    ax.set_yticks(yticks)
    ax.text(0.50, label_y, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.7, color=TEXT)
    clean_axis(ax)


def panel_b(fig: plt.Figure, data: dict[str, Any]) -> pd.DataFrame:
    panel_letter(fig, "b", 0.326, 0.940)
    overlap = data["b"][data["b"]["endpoint"].astype(str).eq("B_top25_overlap")].iloc[0]
    direction = data["b"][data["b"]["endpoint"].astype(str).eq("B_top25_direction")].iloc[0]
    overlap_compound = data["per_compound"][data["per_compound"]["endpoint"].astype(str).eq("B_top25_overlap")]
    direction_compound = data["per_compound"][data["per_compound"]["endpoint"].astype(str).eq("B_top25_direction")]
    overlap_arms = _bootstrap_b_arms(overlap_compound, overlap)
    direction_arms = _bootstrap_b_arms(direction_compound, direction)
    source = pd.concat([overlap_arms, direction_arms], ignore_index=True)
    # Give each mini-chart enough vertical height for its bracket, delta, and
    # bar labels while preserving the sample's generous gap between metrics.
    # Shift the axes right to give the shared y label its own quiet column,
    # and raise the lower chart so its category labels clear the row divider.
    ax_top = fig.add_axes([0.365, 0.760, 0.155, 0.135])
    ax_bottom = fig.add_axes([0.365, 0.550, 0.155, 0.135])
    _mini_bars(ax_top, overlap_arms, "Top-25 overlap", ylim=(0.0, 0.55), yticks=[0.0, 0.2, 0.4], show_xlabels=False, label_y=1.20)
    _mini_bars(ax_bottom, direction_arms, "Direction agreement", ylim=(0.0, 1.40), yticks=[0.0, 0.5, 1.0], show_xlabels=True, label_y=1.20)
    fig.text(0.330, 0.725, "Fraction", rotation=90, ha="center", va="center", fontsize=6.5, color=TEXT)
    return source


def _c_deviation_table(curve: pd.DataFrame, curve_audit: dict[str, Any]) -> pd.DataFrame:
    """Return the six-dose signed deviations used by the left C axis.

    The curve values are already method-specifically normalized by
    ``strict_example_curve``.  This table subtracts the reference value at
    the matching dose only; it does not recompute or pool the normalizations.
    """
    reference = (
        curve[curve["method"].astype(str).eq("reference")]
        .set_index("dose_uM")["relative_morphological_distance"]
        .astype(float)
        .to_dict()
    )
    denominators = {str(key): float(value) for key, value in curve_audit["method_denominators"].items()}
    rows: list[dict[str, Any]] = []
    for method in ("raw", "cfra"):
        subset = curve[curve["method"].astype(str).eq(method)].copy()
        if len(subset) != len(DOSES):
            raise ValueError(f"expected six {method} C trajectory rows")
        for dose in DOSES:
            selected = subset[subset["dose_uM"].astype(str).eq(dose)]
            if len(selected) != 1 or dose not in reference:
                raise ValueError(f"missing matched reference dose {dose} for {method}")
            normalized = float(selected.iloc[0]["relative_morphological_distance"])
            reference_value = float(reference[dose])
            rows.append(
                {
                    "compound": str(selected.iloc[0]["compound"]),
                    "method": method,
                    "dose_uM": dose,
                    "relative_morphological_distance": normalized,
                    "reference_value": reference_value,
                    "deviation_from_reference": normalized - reference_value,
                    "method_denominator": denominators[method],
                    "normalization": "mean 1-PCC distance / method-specific maximum across six dose-wise mean distances",
                }
            )
    return pd.DataFrame(rows)


def panel_c(fig: plt.Figure, data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel_letter(fig, "c", 0.535, 0.940)
    hist_ax = fig.add_axes([0.582, 0.570, 0.158, 0.310])
    scatter_ax = fig.add_axes([0.805, 0.570, 0.175, 0.310])
    curve = data["curve"].copy()
    views = data["curve_views"].copy()
    c_compound = data["c_compound"].copy()
    changes = c_compound["cfra_minus_raw"].to_numpy(dtype=float)
    assert np.allclose(changes, c_compound["cfra"] - c_compound["raw"], atol=1e-12)
    # Fixed 0.025 bins match the standalone candidate, with zero exactly a
    # bin boundary. Counts retain the empirical shape; there is no smoothing.
    bin_width = 0.025
    edges = np.arange(np.floor(changes.min() / bin_width), np.ceil(changes.max() / bin_width) + 1) * bin_width
    counts, edges = np.histogram(changes, bins=edges)
    assert int(counts.sum()) == len(c_compound)
    centers = (edges[:-1] + edges[1:]) / 2
    hist_ax.bar(centers, counts, width=bin_width * 0.94,
                color=["#C4C7CD" if x < 0 else "#8A79B5" for x in centers],
                edgecolor="none", linewidth=0, zorder=2)
    hist_ax.axvline(0, color="#858A91", lw=0.65, ls=(0, (3, 3)), zorder=3)
    hist_ax.set_xlim(edges[0] - 0.0125, edges[-1] + 0.0125)
    hist_ax.set_ylim(0, max(counts) * 1.13)
    hist_ax.set_xticks([-0.2, 0, 0.2], ["-0.2", "0", "+0.2"])
    hist_ax.set_yticks([0, 20, 40])
    hist_ax.set_xlabel("Δ concordance (CFRA - Raw)", fontsize=6.2, labelpad=3)
    hist_ax.set_ylabel("Compounds", fontsize=6.7, labelpad=2)
    clean_axis(hist_ax)
    improved = int((changes > 0).sum())
    hist_ax.text(0.5, 1.115, f"{100 * improved / len(changes):.1f}% improved",
                 transform=hist_ax.transAxes, ha="center", va="bottom", fontsize=7.0, color=PURPLE)
    hist_ax.text(0.5, 1.045, f"{improved} of {len(changes)} compounds",
                 transform=hist_ax.transAxes, ha="center", va="bottom", fontsize=6.25, color=AXIS)
    c_histogram = pd.DataFrame({"bin_left": edges[:-1], "bin_right": edges[1:], "compound_count": counts})
    x = c_compound["raw"].to_numpy(dtype=float)
    y = c_compound["cfra"].to_numpy(dtype=float)
    extent = (-0.30, 1.0)
    # Use smaller, lighter points so the cloud retains its distributional
    # texture rather than reading as a near-deterministic line.
    scatter_ax.scatter(x, y, s=4.0, color=PURPLE, alpha=0.23, linewidths=0, rasterized=True, zorder=2)
    scatter_ax.plot(extent, extent, color=ZERO, lw=0.65, ls=(0, (3, 3)), zorder=1)
    scatter_ax.text(0.86, 0.06, f"n = {len(c_compound)}", transform=scatter_ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=AXIS)
    scatter_ax.set_xlim(extent)
    scatter_ax.set_ylim(extent)
    scatter_ax.set_xticks([0.0, 0.5, 1.0])
    scatter_ax.set_yticks([0.0, 0.5, 1.0])
    scatter_ax.set_xlabel("Raw trajectory concordance", labelpad=2, fontsize=6.7)
    scatter_ax.set_ylabel("CFRA trajectory concordance", labelpad=2, fontsize=6.7)
    clean_axis(scatter_ax)
    c_summary = data["summary"][data["summary"]["endpoint"].astype(str).eq("C_macro_trajectory_spearman")]
    if len(c_summary) != 1:
        raise ValueError("missing locked C_macro_trajectory_spearman summary")
    c_summary_row = c_summary.iloc[0]
    scatter_ax.text(0.50, 1.115, "All compounds", transform=scatter_ax.transAxes, ha="center", va="bottom", fontsize=6.5, color=TEXT)
    scatter_ax.text(
        0.50,
        1.045,
        f"Δ = {float(c_summary_row['cfra_minus_raw']):+.3f} [{float(c_summary_row['ci_low']):.3f}, {float(c_summary_row['ci_high']):.3f}]",
        transform=scatter_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.25,
        color=TEXT,
    )
    return curve, views, c_compound, c_histogram


def panel_d(fig: plt.Figure, data: dict[str, Any]) -> pd.DataFrame:
    panel_letter(fig, "d", 0.018, 0.440)
    # Match the reference layout's lower-row balance: D is a single broad
    # comparison, while E gets enough room for two equal transfer cohorts.
    # Reserve a wider left annotation column for the task names and their
    # sample sizes, as in the reference lower-left panel.
    ax = fig.add_axes([0.230, 0.100, 0.225, 0.330])
    order = (
        "confirmed_activity_ap",
        "moa_map",
        "target_map",
        "reactome_target_neighbour_map",
    )
    labels = {
        "confirmed_activity_ap": "Confirmed activity (AP)",
        "moa_map": "Mechanism (mAP)",
        "target_map": "Target (mAP)",
        "reactome_target_neighbour_map": "Pathways (mAP)",
    }
    frame = data["d_contrasts"].set_index("endpoint").loc[list(order)].reset_index().copy()
    y = np.arange(len(frame))[::-1]
    points = frame["point"].to_numpy(dtype=float)
    ax.barh(y, points, height=0.42, color=ORANGE, alpha=0.72, edgecolor=WHITE, linewidth=0.35, zorder=2)
    xerr = np.vstack(
        [
            points - frame["ci_low"].to_numpy(dtype=float),
            frame["ci_high"].to_numpy(dtype=float) - points,
        ]
    )
    ax.errorbar(points, y, xerr=xerr, fmt="none", ecolor=ORANGE, elinewidth=0.75, capsize=1.7, capthick=0.7, zorder=4)
    zero_vline(ax)
    # Match the sample's symmetric downstream scale so near-zero endpoints
    # remain legible rather than collapsing at the zero line.
    ax.set_xlim(-0.05, 0.15)
    ax.set_ylim(-0.65, len(frame) - 0.35)
    ax.set_yticks(y, [labels[value] for value in frame["endpoint"].astype(str)])
    ax.set_xticks([-0.05, 0.00, 0.05, 0.10, 0.15])
    ax.set_xlabel("Change in performance (CFRA - Raw)", labelpad=2)
    clean_axis(ax, hide_left=True)
    # Sample-size labels sit in a quiet column between the task names and the
    # plotting region, using the frozen compound counts.
    ax.tick_params(axis="y", pad=35)
    for yi, (_, row) in zip(y, frame.iterrows()):
        ax.text(
            -0.012,
            yi,
            f"n = {int(row['compound_count'])}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.3,
            color=AXIS,
        )
    return frame


def _target_point_range(data: dict[str, Any], endpoint: str) -> pd.DataFrame:
    target = data["e_target"]
    rows: list[dict[str, Any]] = []
    for line in ("A549", "K562", "MCF7"):
        selected = target[
            target["endpoint"].astype(str).eq(endpoint)
            & target["target_cell_line"].astype(str).eq(line)
            & target["arm"].astype(str).eq("cfra_minus_raw")
        ]
        if len(selected) != 1:
            raise ValueError(f"missing {endpoint} row for {line}")
        row = selected.iloc[0]
        rows.append(
            {
                "endpoint": endpoint,
                "target_cell_line": line,
                "label": line,
                "point": float(row["estimate"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "n_drugs": int(row["n_drugs"]),
                "bootstrap_draws": int(row["bootstrap_draws"]),
                "bootstrap_unit": str(row["bootstrap_unit"]),
            }
        )
    macro = data["e_macro"][data["e_macro"]["endpoint"].astype(str).eq(endpoint)]
    if len(macro) != 1:
        raise ValueError(f"missing {endpoint} macro row")
    row = macro.iloc[0]
    rows.append(
        {
            "endpoint": endpoint,
            "target_cell_line": "Mean",
            "label": "Mean",
            "point": float(row["estimate"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "n_drugs": int(row["n_targets"]),
            "bootstrap_draws": int(row["bootstrap_draws"]),
            "bootstrap_unit": str(row["bootstrap_unit"]),
        }
    )
    return pd.DataFrame(rows)


def _draw_target_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    label: str,
    *,
    open_marker: bool,
) -> None:
    y = np.arange(len(frame))[::-1]
    zero_vline(ax)
    for yi, (_, item) in zip(y, frame.iterrows()):
        point = float(item["point"])
        low = float(item["ci_low"])
        high = float(item["ci_high"])
        marker = "D" if item["label"] == "Mean" else "o"
        ax.errorbar(
            point,
            yi,
            xerr=[[point - low], [high - point]],
            fmt=marker,
            color=ORANGE,
            mfc=WHITE if open_marker else ORANGE,
            mec=ORANGE,
            mew=0.9 if open_marker else 0.35,
            ms=4.0 if item["label"] == "Mean" else 3.6,
            elinewidth=0.85,
            capsize=1.8,
            capthick=0.75,
            zorder=3,
        )
    ax.set_xlim(-0.0035, 0.013)
    ax.set_ylim(-0.65, len(frame) - 0.35)
    ax.set_yticks(y, frame["label"].tolist())
    ax.set_xticks([0.0, 0.004, 0.008, 0.012])
    n_target = int(frame.loc[frame["label"].ne("Mean"), "n_drugs"].iloc[0])
    ax.text(0.50, 1.065, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color=TEXT)
    ax.text(0.50, 1.015, f"n = {n_target} / target", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.05, color=AXIS)
    clean_axis(ax, hide_left=True)


def panel_e(fig: plt.Figure, data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_letter(fig, "e", 0.505, 0.440)
    seen = _target_point_range(data, "pure_seen")
    unseen = _target_point_range(data, "cold_confirmation")
    # Two equal-width cohort axes reproduce the sample's Seen/Unseen rhythm.
    # Their single shared x label prevents the lower row from becoming dense.
    ax_seen = fig.add_axes([0.565, 0.100, 0.175, 0.330])
    ax_unseen = fig.add_axes([0.795, 0.100, 0.175, 0.330])
    _draw_target_axis(ax_seen, seen, "Seen compounds", open_marker=False)
    _draw_target_axis(ax_unseen, unseen, "Unseen compounds", open_marker=True)
    fig.text(
        0.768,
        0.040,
        "Change in specificity (Fisher z)",
        ha="center",
        va="center",
        fontsize=7.4,
        color=TEXT,
    )
    return seen, unseen


def render_audit(fig: plt.Figure) -> dict[str, Any]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_records: list[dict[str, Any]] = []
    clipped: list[str] = []
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text():
            continue
        box = obj.get_window_extent(renderer)
        text_records.append({"text": obj.get_text(), "font_pt": float(obj.get_fontsize())})
        if box.width and box.height and (
            box.x0 < -1
            or box.y0 < -1
            or box.x1 > fig.bbox.width + 1
            or box.y1 > fig.bbox.height + 1
        ):
            clipped.append(obj.get_text())
    return {
        "width_mm": round(fig.get_figwidth() * 25.4, 3),
        "height_mm": round(fig.get_figheight() * 25.4, 3),
        "min_font_pt": min(record["font_pt"] for record in text_records),
        "text_outside_canvas": clipped,
        "texts": text_records,
    }


def render(args: argparse.Namespace) -> None:
    set_style()
    data = load_inputs(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    source_dir = args.outdir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(CANVAS_MM[0] / 25.4, CANVAS_MM[1] / 25.4), facecolor=WHITE)
    fig.add_artist(
        Line2D(
            [0.018, 0.980],
            [0.490, 0.490],
            transform=fig.transFigure,
            color="#E8EBEF",
            lw=0.40,
            zorder=0,
        )
    )
    # Very light panel dividers reproduce the sample's grid without adding a
    # box around individual axes.  The top row has a|b and b|c separators;
    # the lower row has the d|e separator.
    for x0, y0, y1 in ((0.315, 0.490, 0.940), (0.525, 0.490, 0.940), (0.505, 0.040, 0.490)):
        fig.add_artist(
            Line2D(
                [x0, x0],
                [y0, y1],
                transform=fig.transFigure,
                color="#EEF0F2",
                lw=0.40,
                zorder=0,
            )
        )
    a_source = panel_a(fig, data)
    b_source = panel_b(fig, data)
    c_curve, c_views, c_scatter, c_histogram = panel_c(fig, data)
    d_source = panel_d(fig, data)
    e_source = panel_e(fig, data)
    qa = render_audit(fig)
    if qa["text_outside_canvas"]:
        raise ValueError(f"text outside canvas: {qa['text_outside_canvas']}")

    outputs = {
        "pdf": args.outdir / "figure4_cfra_mixed.pdf",
        "svg": args.outdir / "figure4_cfra_mixed.svg",
        "png": args.outdir / "figure4_cfra_mixed.png",
    }
    fig.savefig(outputs["pdf"])
    fig.savefig(outputs["svg"])
    fig.savefig(outputs["png"], dpi=600)
    plt.close(fig)

    sources = {
        "A_dumbbell": a_source,
        "A_compound": data["per_compound"][data["per_compound"]["endpoint"].astype(str).str.startswith("A_module_fisher_z_")].copy(),
        "B_bars": b_source,
        "C_curve": c_curve,
        "C_curve_views": c_views,
        "C_scatter": c_scatter,
        "C_histogram": c_histogram,
        "D_bars": d_source,
        "E_seen": e_source[0],
        "E_unseen": e_source[1],
    }
    for label, frame in sources.items():
        frame.to_csv(source_dir / f"figure4_cfra_mixed_{label}.csv", index=False, float_format="%.12g")

    inputs = [
        args.summary,
        args.per_compound,
        args.panel_d,
        args.panel_d_metrics,
        args.e_summary,
        args.e_macro,
        args.bundle,
        args.feature_mapping,
    ]
    audit = {
        "version": VERSION,
        "status": STATUS,
        "canvas_mm": {"width": CANVAS_MM[0], "height": CANVAS_MM[1]},
        "models_trained": False,
        "sources_are_frozen_results": True,
        "panel_titles_present": False,
        "layout": "compact 180.34-by-106 mm two-row grid; top columns approximately 30:23:47; panel B stacks two raised mini axes with dedicated Fraction and annotation columns",
        "visual_contract": {
            "a": "Raw-to-CFRA dumbbells plus module-level percent improved",
            "b": "paired compact bars for the two 0-to-1 feature endpoints",
            "c": "empirical distribution of paired compound concordance changes plus matched Raw-versus-CFRA scatter",
            "d": "diverging horizontal CFRA-minus-Raw bars with 95% CI",
            "e": "paired seen/unseen point-range estimates for A549, K562, MCF7, and macro mean; seen filled and unseen open",
        },
        "inferential_unit": {
            "a": "compound",
            "b": "compound",
            "c": "compound",
            "d": "compound",
            "e_target_rows": "drug within target cell line",
            "e_mean": "target_cell_line_then_drug",
        },
        "panel_e_sources": {
            "seen": {
                "endpoint": "pure_seen",
                "arm": "cfra_minus_raw",
                "target_rows": int(len(e_source[0]) - 1),
                "macro_rows": 1,
                "n_drugs_by_target": e_source[0].set_index("target_cell_line")["n_drugs"].drop("Mean").astype(int).to_dict(),
                "source_fields": ["endpoint", "target_cell_line", "arm", "estimate", "ci_low", "ci_high", "n_drugs", "bootstrap_draws", "bootstrap_unit"],
            },
            "unseen": {
                "endpoint": "cold_confirmation",
                "arm": "cfra_minus_raw",
                "target_rows": int(len(e_source[1]) - 1),
                "macro_rows": 1,
                "n_drugs_by_target": e_source[1].set_index("target_cell_line")["n_drugs"].drop("Mean").astype(int).to_dict(),
                "source_fields": ["endpoint", "target_cell_line", "arm", "estimate", "ci_low", "ci_high", "n_drugs", "bootstrap_draws", "bootstrap_unit"],
            },
        },
        "strict_c_curve": data["curve_audit"],
        "panel_c_histogram": {
            "source": "C_histogram CSV; individual values in C_scatter CSV",
            "formula": "CFRA - Raw compound-level macro trajectory Spearman concordance",
            "bin_width": 0.025,
            "zero_is_bin_boundary": True,
            "n_compounds": int(len(c_scatter)),
            "n_improved": int(c_scatter["cfra_minus_raw"].gt(0).sum()),
            "histogram_total": int(c_histogram["compound_count"].sum()),
            "color_encoding": "negative change in light gray; positive change in muted purple",
            "smoothing": None,
            "same_population_as_scatter": True,
            "legacy_single_compound_curve_displayed": False,
        },
        "panel_b_arm_bootstrap": {
            "version": B_BOOTSTRAP_VERSION,
            "rounds": BOOTSTRAP_ROUNDS,
            "sampling_unit": "compound",
            "row_order": "stable sort by compound before resampling",
            "paired_draws": "Raw, CFRA, and CFRA-minus-Raw reuse identical compound draws",
            "seed_rule": "little-endian integer from first 8 bytes of SHA256(version|endpoint)",
            "seeds": {
                endpoint: str(b_bootstrap_seed(endpoint))
                for endpoint in ("B_top25_overlap", "B_top25_direction")
            },
            "interval": "2.5th and 97.5th percentiles of arm means from frozen compound rows",
        },
        "render_qa": qa,
        "inputs": {str(path): sha256(path) for path in inputs},
        "outputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in outputs.items()},
        "source_outputs": {
            "C_histogram": {
                "path": str(source_dir / "figure4_cfra_mixed_C_histogram.csv"),
                "sha256": sha256(source_dir / "figure4_cfra_mixed_C_histogram.csv"),
            }
        },
    }
    (args.outdir / "figure4_cfra_mixed_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default = ROOT / "experiments" / "cpg0004_figure4_cfra_20260920"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=default / "remote_results" / "FIGURE4_CFRA_SUMMARY.csv")
    parser.add_argument("--per-compound", type=Path, default=default / "remote_results" / "FIGURE4_CFRA_PER_COMPOUND.csv")
    parser.add_argument("--panel-d", type=Path, default=default / "remote_results" / "panel_d_v5" / "PANEL_D_CONTRASTS.csv")
    parser.add_argument("--panel-d-metrics", type=Path, default=default / "remote_results" / "panel_d_v5" / "PANEL_D_METRICS.csv")
    parser.add_argument("--e-summary", type=Path, default=ROOT / "publication_figures_20260910" / "source_data" / "figure4_E_target_summary.csv")
    parser.add_argument("--e-macro", type=Path, default=ROOT / "publication_figures_20260910" / "source_data" / "figure4_E_target_macro_contrasts.csv")
    parser.add_argument("--bundle", type=Path, default=default / "remote_results" / "CFRA_PER_PHYSICAL_VIEW.npz")
    parser.add_argument("--feature-mapping", type=Path, default=default / "remote_results" / "FEATURE_MAPPING.csv")
    parser.add_argument("--outdir", type=Path, default=default / "rendered_figure4_cfra_mixed")
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
