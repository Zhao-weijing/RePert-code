"""Render a compact Nature-style Figure 4 candidate for the frozen CFRA bundle.

The renderer is deliberately downstream-only: it reads the locked CSV/NPZ
outputs, does not refit or select a model, and writes all candidates to a new
``rendered_figure4_cfra_nature`` directory.  A--C use the purple CFRA accent;
D--E use the orange target-task accent.  Compound is the inferential unit;
seed and support-rotation traces are retained only as descriptive hairlines
in panel C.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
MPL_LIB = ROOT / "publication_figures_20260910" / ".python_libs"
if str(MPL_LIB) not in sys.path:
    sys.path.insert(0, str(MPL_LIB))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


VERSION = "cpg0004-figure4-cfra-nature-v1-2026-09-21"
STATUS = "HISTORICAL_LOCKED_RECOMPUTE"
SEEDS = (3407, 42, 2025)
DOSES = ("0.04", "0.12", "0.37", "1.11", "3.33", "10")
EXAMPLE_COMPOUND = "BRD-K26619122"
MODULE_ORDER = ("DNA", "RNA", "ER", "Mito", "AGP", "Shape", "Cross-channel")
MODULE_LABELS = {
    "DNA": "DNA",
    "RNA": "RNA",
    "ER": "ER",
    "Mito": "Mito",
    "AGP": "AGP",
    "Shape": "Shape",
    "Cross-channel": "Cross-channel",
}

# Locked publication palette requested for this candidate.
RAW = "#A5ABB3"
REFERENCE = "#252A31"
CFRA_AC = "#6F5DA8"
CFRA_DE = "#C47A37"
GUIDE = "#D8DCE2"
WHITE = "#FFFFFF"
TEXT = "#252A31"
MUTED = "#59616B"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def set_style() -> None:
    # The manuscript scales this 180 mm source to a 145 mm slot.  Keep source
    # text at >=8.2 pt so the smallest retained text is still >=6.5 pt.
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.2,
            "axes.labelsize": 8.6,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.2,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.linewidth": 0.55,
            "lines.linewidth": 0.9,
            "lines.markersize": 3.2,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.minor.size": 1.4,
            "ytick.minor.size": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean(ax: plt.Axes, *, zero: bool = False, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.2, pad=2.0)
    if zero:
        ax.axvline(0, color=GUIDE, lw=0.55, zorder=0)
    if grid:
        ax.grid(axis="x", color=GUIDE, lw=0.45, zorder=0)
        ax.set_axisbelow(True)


def panel_label(fig: plt.Figure, letter: str, title: str, x: float, y: float) -> None:
    fig.text(x, y, letter.lower(), fontsize=11.0, fontweight="bold", va="bottom", color=TEXT)
    fig.text(x + 0.022, y + 0.001, title, fontsize=9.2, va="bottom", color=TEXT)


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "NA"
    if abs(value) >= 0.1:
        return f"{value:+.{digits}f}"
    return f"{value:+.3f}"


def canonical_dose(value: object) -> str:
    return f"{float(str(value)):.12g}"


def pcc(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.size < 2 or b.size != a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return math.nan
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return math.nan if denom <= 0 else float(np.dot(a, b) / denom)


def strict_example_curve(
    bundle_path: Path, mapping_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute the fixed six-dose example exactly from physical views.

    The bold curve is the formal mean over the six seed/rotation views after
    within-view 1-PCC calculation.  Thin traces use the formal mean curve's
    method-specific maximum over the six dose-wise mean distances, so their pointwise mean is exactly
    the bold curve (rather than six independently self-normalized curves).
    """
    mapping = pd.read_csv(mapping_path)
    require_columns(mapping, ("feature_index", "biological_module", "biological_endpoint"), "feature mapping")
    mito = mapping[
        mapping["biological_module"].astype(str).eq("Mito")
        & mapping["biological_endpoint"].astype(str).eq("Eligible")
    ]["feature_index"].to_numpy(dtype=int)
    if len(mito) != 17:
        raise ValueError(f"expected 17 eligible Mito features, found {len(mito)}")

    formal_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    with np.load(bundle_path, allow_pickle=False) as payload:
        for seed in SEEDS:
            prefix = f"seed{seed}_"
            compound = payload[prefix + "compound"].astype(str)
            dose = np.asarray([canonical_dose(v) for v in payload[prefix + "dose"]], dtype=str)
            rotation = payload[prefix + "rotation"].astype(int)
            selected = compound == EXAMPLE_COMPOUND
            if int(selected.sum()) != 12:
                raise ValueError(f"{EXAMPLE_COMPOUND} lacks 12 physical views for seed {seed}")
            for support_rotation in (0, 1):
                indices: list[int] = []
                for value in DOSES:
                    hits = np.flatnonzero(selected & (rotation == support_rotation) & (dose == value))
                    if len(hits) != 1:
                        raise ValueError(f"missing fixed dose {value} for seed {seed}, rotation {support_rotation}")
                    indices.append(int(hits[0]))
                for method in ("reference", "raw", "cfra"):
                    profile = np.asarray(payload[prefix + method][indices][:, mito], dtype=float)
                    base = profile[0]
                    distances = np.asarray([1.0 - pcc(row, base) for row in profile], dtype=float)
                    if not np.isfinite(distances).all():
                        raise ValueError(f"non-finite fixed curve for {method}, seed {seed}, rotation {support_rotation}")
                    for value, distance in zip(DOSES, distances):
                        view_rows.append(
                            {
                                "compound": EXAMPLE_COMPOUND,
                                "seed": seed,
                                "support_rotation": support_rotation,
                                "method": method,
                                "dose_uM": value,
                                "distance_1_minus_pcc_to_lowest": float(distance),
                            }
                        )

    view = pd.DataFrame(view_rows)
    formal = (
        view.groupby(["compound", "method", "dose_uM"], sort=False, as_index=False)["distance_1_minus_pcc_to_lowest"]
        .mean()
        .rename(columns={"distance_1_minus_pcc_to_lowest": "mean_distance"})
    )
    denominators = formal.groupby("method", sort=False)["mean_distance"].max().to_dict()
    if any(float(value) <= 0 for value in denominators.values()):
        raise ValueError("fixed C curve has a constant method trajectory")
    formal["relative_morphological_distance"] = formal.apply(
        lambda row: float(row["mean_distance"]) / float(denominators[str(row["method"])]), axis=1
    )
    view["relative_morphological_distance"] = view.apply(
        lambda row: float(row["distance_1_minus_pcc_to_lowest"]) / float(denominators[str(row["method"])]), axis=1
    )
    audit = {
        "compound": EXAMPLE_COMPOUND,
        "methods": ["reference", "raw", "cfra"],
        "seeds": list(SEEDS),
        "support_rotations": [0, 1],
        "mito_feature_indices": mito.tolist(),
        "aggregation": "compute 1-PCC inside every seed/support physical view -> mean view curve -> method-specific maximum across the six dose-wise mean distances",
        "bundle_sha256": sha256(bundle_path),
        "mapping_sha256": sha256(mapping_path),
        "method_denominators": {str(key): float(value) for key, value in denominators.items()},
    }
    return formal, view, audit


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    summary = pd.read_csv(args.summary)
    per_compound = pd.read_csv(args.per_compound)
    d_contrasts = pd.read_csv(args.panel_d)
    d_metrics = pd.read_csv(args.panel_d_metrics)
    e_target = pd.read_csv(args.e_summary)
    e_macro = pd.read_csv(args.e_macro)

    require_columns(summary, ("endpoint", "raw", "cfra", "cfra_minus_raw", "ci_low", "ci_high", "n_compounds"), "CFRA summary")
    require_columns(per_compound, ("endpoint", "compound", "raw", "cfra", "cfra_minus_raw"), "per-compound CFRA")
    require_columns(d_contrasts, ("endpoint", "point", "ci_low", "ci_high"), "panel D contrasts")
    require_columns(d_metrics, ("endpoint", "method", "point"), "panel D arm metrics")
    require_columns(e_target, ("endpoint", "target_cell_line", "arm", "estimate", "ci_low", "ci_high"), "panel E target summary")
    require_columns(e_macro, ("endpoint", "estimate", "ci_low", "ci_high"), "panel E macro")

    a = summary[summary["endpoint"].astype(str).str.startswith("A_module_fisher_z_")].copy()
    a["module"] = a["endpoint"].astype(str).str.replace("A_module_fisher_z_", "", regex=False)
    a["rank"] = a["module"].map({name: i for i, name in enumerate(MODULE_ORDER)})
    a = a.sort_values("rank", kind="stable").reset_index(drop=True)
    if len(a) != 7 or set(a["module"]) != set(MODULE_ORDER):
        raise ValueError("expected all seven Figure 4A modules")

    b = summary[summary["endpoint"].astype(str).isin(["B_top25_overlap", "B_top25_direction"])].copy()
    if len(b) != 2:
        raise ValueError("expected Top-25 overlap and direction summary rows")

    c_summary = summary[summary["endpoint"].astype(str).eq("C_macro_trajectory_spearman")].copy()
    if len(c_summary) != 1:
        raise ValueError("expected one formal C trajectory summary")
    c_compound = per_compound[per_compound["endpoint"].astype(str).eq("C_macro_trajectory_spearman")].copy()
    if len(c_compound) != 289:
        raise ValueError(f"expected 289 C compound rows, found {len(c_compound)}")

    d_contrasts = d_contrasts.copy()
    d_metrics = d_metrics.copy()
    for frame, cols in (
        (a, ("raw", "cfra", "cfra_minus_raw", "ci_low", "ci_high")),
        (b, ("raw", "cfra", "cfra_minus_raw", "ci_low", "ci_high")),
        (c_summary, ("raw", "cfra", "cfra_minus_raw", "ci_low", "ci_high")),
        (d_contrasts, ("point", "ci_low", "ci_high")),
        (d_metrics, ("point",)),
        (e_target, ("estimate", "ci_low", "ci_high")),
        (e_macro, ("estimate", "ci_low", "ci_high")),
    ):
        for col in cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
                raise ValueError(f"non-finite input in {col}")

    curve, curve_views, curve_audit = strict_example_curve(args.bundle, args.feature_mapping)
    return {
        "summary": summary,
        "per_compound": per_compound,
        "a": a,
        "b": b,
        "c_summary": c_summary,
        "c_compound": c_compound,
        "d_contrasts": d_contrasts,
        "d_metrics": d_metrics,
        "e_target": e_target,
        "e_macro": e_macro,
        "curve": curve,
        "curve_views": curve_views,
        "curve_audit": curve_audit,
    }


def draw_panel_a(fig: plt.Figure, data: dict[str, Any]) -> None:
    """Seven module arm dumbbells plus compound delta distributions."""
    panel_label(fig, "A", "Module agreement", 0.045, 0.947)
    fig.text(0.210, 0.905, "mean arm values", fontsize=8.2, color=MUTED, ha="center")
    fig.text(0.445, 0.905, "compound Δ · 95% CI · % positive", fontsize=8.2, color=MUTED, ha="center")

    # 58% of the usable width is reserved for A; the right-hand strip is its
    # compact compound-level distribution view.
    arm = fig.add_axes([0.150, 0.690, 0.160, 0.190])
    dist = fig.add_axes([0.325, 0.690, 0.246, 0.190])
    rng = np.random.default_rng(4127)
    modules = list(data["a"]["module"].astype(str))
    y = np.arange(len(modules))
    for yi, (_, row) in zip(y, data["a"].iterrows()):
        module = str(row["module"])
        raw = float(row["raw"])
        cfra = float(row["cfra"])
        arm.plot([raw, cfra], [yi, yi], color=GUIDE, lw=1.1, zorder=1)
        arm.plot(raw, yi, marker="o", ms=3.2, color=RAW, mec=RAW, zorder=3)
        arm.plot(cfra, yi, marker="D", ms=3.1, color=CFRA_AC, mec=CFRA_AC, zorder=3)

        subset = data["per_compound"][data["per_compound"]["endpoint"].astype(str).eq(f"A_module_fisher_z_{module}")]
        delta = subset["cfra_minus_raw"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.20, 0.20, size=len(delta))
        dist.scatter(delta, yi + jitter, s=2.0, color=CFRA_AC, alpha=0.16, linewidths=0, rasterized=True, zorder=1)
        low, high, point = float(row["ci_low"]), float(row["ci_high"]), float(row["cfra_minus_raw"])
        dist.plot([low, high], [yi, yi], color=CFRA_AC, lw=1.25, zorder=4)
        dist.plot([low, low], [yi - 0.075, yi + 0.075], color=CFRA_AC, lw=0.75, zorder=4)
        dist.plot([high, high], [yi - 0.075, yi + 0.075], color=CFRA_AC, lw=0.75, zorder=4)
        dist.plot(point, yi, marker="D", ms=3.1, color=CFRA_AC, mec=WHITE, mew=0.35, zorder=5)
        pct = 100.0 * float(np.mean(delta > 0))
        dist.text(
            0.900,
            yi,
            f"{pct:.1f}%",
            transform=dist.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.2,
            color=TEXT,
            bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
        )

    arm.set_xlim(0.30, 0.96)
    arm.set_ylim(-0.55, len(y) - 0.45)
    arm.invert_yaxis()
    arm.set_yticks(y, [MODULE_LABELS[m] for m in modules])
    arm.set_xticks([0.4, 0.6, 0.8])
    arm.set_xlabel("Fisher z agreement", labelpad=2)
    clean(arm, grid=True)
    arm.text(0.02, 1.02, "Raw", transform=arm.transAxes, color=RAW, fontsize=8.2, ha="left", va="bottom")
    arm.text(0.33, 1.02, "CFRA", transform=arm.transAxes, color=CFRA_AC, fontsize=8.2, ha="left", va="bottom")
    arm.plot(0.26, 1.03, marker="o", ms=3.0, color=RAW, transform=arm.transAxes, clip_on=False)
    arm.plot(0.57, 1.03, marker="D", ms=2.9, color=CFRA_AC, transform=arm.transAxes, clip_on=False)

    dist.set_xlim(-0.45, 0.90)
    dist.set_ylim(-0.55, len(y) - 0.45)
    dist.invert_yaxis()
    dist.set_yticks([])
    dist.set_xticks([-0.4, 0.0, 0.4, 0.8])
    dist.set_xlabel("CFRA − Raw", labelpad=2)
    clean(dist, zero=True, grid=False)
    dist.spines["left"].set_visible(False)


def _paired_distribution_axis(
    ax: plt.Axes, frame: pd.DataFrame, summary_row: pd.Series, title: str, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    raw = frame["raw"].to_numpy(dtype=float)
    cfra = frame["cfra"].to_numpy(dtype=float)
    n = len(frame)
    raw_y = rng.uniform(-0.13, 0.13, size=n)
    cfra_y = 1.0 + rng.uniform(-0.13, 0.13, size=n)
    for x0, x1, y0, y1 in zip(raw, cfra, raw_y, cfra_y):
        ax.plot([x0, x1], [y0, y1], color=GUIDE, alpha=0.22, lw=0.35, zorder=1, rasterized=True)
    ax.scatter(raw, raw_y, s=1.8, color=RAW, alpha=0.25, linewidths=0, rasterized=True, zorder=2)
    ax.scatter(cfra, cfra_y, s=1.8, color=CFRA_AC, alpha=0.25, linewidths=0, rasterized=True, zorder=2)
    ax.plot(float(frame["raw"].mean()), 0, marker="o", ms=3.7, color=RAW, mec=TEXT, mew=0.25, zorder=4)
    ax.plot(float(frame["cfra"].mean()), 1, marker="D", ms=3.6, color=CFRA_AC, mec=TEXT, mew=0.25, zorder=4)
    point, low, high = (float(summary_row[k]) for k in ("cfra_minus_raw", "ci_low", "ci_high"))
    positive = 100.0 * float(np.mean((cfra - raw) > 0))
    ax.errorbar(
        point,
        0.50,
        xerr=[[point - low], [high - point]],
        fmt="D",
        color=CFRA_AC,
        mec=CFRA_AC,
        ms=3.3,
        capsize=1.6,
        elinewidth=0.85,
        zorder=4,
    )
    ax.text(
        min(high + 0.045, 0.78),
        0.50,
        f"Δ {point:.3f}; {positive:.0f}% +",
        ha="left",
        va="center",
        fontsize=8.0,
        color=MUTED,
    )
    ax.set_title(title, loc="left", fontsize=8.6, pad=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.30, 1.30)
    ax.set_yticks([0, 1], ["Raw", "CFRA"])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("fraction", labelpad=1)
    clean(ax, grid=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def draw_panel_b(fig: plt.Figure, data: dict[str, Any]) -> None:
    panel_label(fig, "B", "Top-25 feature recovery", 0.603, 0.947)
    overlap = data["b"][data["b"]["endpoint"].astype(str).eq("B_top25_overlap")].iloc[0]
    direction = data["b"][data["b"]["endpoint"].astype(str).eq("B_top25_direction")].iloc[0]
    overlap_values = data["per_compound"][data["per_compound"]["endpoint"].astype(str).eq("B_top25_overlap")]
    direction_values = data["per_compound"][data["per_compound"]["endpoint"].astype(str).eq("B_top25_direction")]
    # Deliberately independent side-by-side axes: overlap and sign agreement
    # remain separate endpoints, each with its own fraction axis.
    ax_overlap = fig.add_axes([0.603, 0.695, 0.163, 0.185])
    ax_direction = fig.add_axes([0.790, 0.695, 0.163, 0.185])
    _paired_distribution_axis(ax_overlap, overlap_values, overlap, "Overlap", 191)
    _paired_distribution_axis(ax_direction, direction_values, direction, "Direction", 293)
    ax_overlap.set_xticks([0.0, 0.5], ["0.0", "0.5"])
    ax_direction.set_xticks([0.5, 1.0], ["0.5", "1.0"])


def draw_panel_c(fig: plt.Figure, data: dict[str, Any]) -> None:
    panel_label(fig, "c", "Dose-response geometry", 0.045, 0.598)
    fig.text(0.045, 0.574, f"{EXAMPLE_COMPOUND} · Mito · six doses", fontsize=8.2, color=MUTED)
    fig.text(0.405, 0.574, "289 compounds · macro trajectory concordance", fontsize=8.2, color=MUTED)
    fig.text(0.755, 0.574, "paired compound effect", fontsize=8.2, color=MUTED)

    curve_ax = fig.add_axes([0.105, 0.355, 0.255, 0.170])
    scatter_ax = fig.add_axes([0.405, 0.355, 0.285, 0.170])
    effect_ax = fig.add_axes([0.758, 0.374, 0.201, 0.150])

    curve = data["curve"]
    curve_views = data["curve_views"]
    curve_style = {
        "reference": ("Reference", REFERENCE, "--"),
        "raw": ("Raw", RAW, "-"),
        "cfra": ("CFRA", CFRA_AC, "-"),
    }
    for method in ("reference", "raw", "cfra"):
        subset = curve[curve["method"].astype(str).eq(method)].copy()
        subset["dose_float"] = subset["dose_uM"].astype(float)
        subset = subset.set_index("dose_uM").loc[list(DOSES)].reset_index()
        label, color, linestyle = curve_style[method]
        # Six descriptive seed/rotation traces; the common denominator is the
        # formal method mean's max-dose denominator computed above.
        for (_, _, _), view in curve_views[
            (curve_views["method"].astype(str).eq(method))
        ].groupby(["seed", "support_rotation", "method"], sort=True):
            view = view.set_index("dose_uM").loc[list(DOSES)].reset_index()
            curve_ax.plot(
                view["dose_uM"].astype(float),
                view["relative_morphological_distance"].astype(float),
                color=color,
                lw=0.45,
                alpha=0.28,
                ls=linestyle,
                zorder=1,
            )
        curve_ax.plot(
            subset["dose_float"],
            subset["relative_morphological_distance"],
            color=color,
            lw=1.25,
            ls=linestyle,
            label=label,
            zorder=3,
        )
    curve_ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.02, 1.015),
        ncol=3,
        frameon=False,
        fontsize=8.2,
        handlelength=1.2,
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    curve_ax.set_xscale("log")
    curve_ax.set_xticks([float(value) for value in DOSES], DOSES)
    curve_ax.set_xlim(0.035, 17.5)
    curve_ax.set_ylim(-0.04, 1.06)
    curve_ax.set_xlabel("Dose (µM)", labelpad=2)
    curve_ax.set_ylabel("Relative distance", labelpad=2)
    curve_ax.tick_params(axis="x", rotation=32)
    clean(curve_ax, grid=True)

    c_compound = data["c_compound"]
    x = c_compound["raw"].to_numpy(dtype=float)
    y = c_compound["cfra"].to_numpy(dtype=float)
    extent = (-0.30, 1.0)
    scatter_ax.hexbin(
        x,
        y,
        gridsize=21,
        extent=extent + extent,
        mincnt=1,
        bins="log",
        cmap=LinearSegmentedColormap.from_list("cfracmap", ["#EEEAF5", CFRA_AC]),
        linewidths=0.2,
        edgecolors=WHITE,
        zorder=1,
    )
    scatter_ax.scatter(x, y, s=4.0, color=CFRA_AC, alpha=0.20, linewidths=0, rasterized=True, zorder=2)
    scatter_ax.plot(extent, extent, color=GUIDE, lw=0.8, ls="--", zorder=3)
    scatter_ax.text(0.03, 0.95, "y = x", transform=scatter_ax.transAxes, fontsize=8.2, color=MUTED, va="top")
    scatter_ax.text(0.98, 0.05, "n = 289", transform=scatter_ax.transAxes, fontsize=8.2, color=MUTED, ha="right", va="bottom")
    scatter_ax.set_xlim(extent)
    scatter_ax.set_ylim(extent)
    scatter_ax.set_xlabel("Raw macro TC", labelpad=2)
    scatter_ax.set_ylabel("CFRA macro TC", labelpad=2)
    clean(scatter_ax, grid=True)

    c_row = data["c_summary"].iloc[0]
    point, low, high = (float(c_row[key]) for key in ("cfra_minus_raw", "ci_low", "ci_high"))
    positive = 100.0 * float(np.mean(c_compound["cfra_minus_raw"].to_numpy(dtype=float) > 0))
    effect_ax.errorbar(point, 0.50, xerr=[[point - low], [high - point]], fmt="D", color=CFRA_AC, mec=CFRA_AC, ms=4.2, capsize=2.0, elinewidth=1.0, zorder=3)
    effect_ax.axvline(0, color=GUIDE, lw=0.65, zorder=0)
    effect_ax.set_xlim(-0.01, 0.08)
    effect_ax.set_ylim(0, 1)
    effect_ax.set_yticks([])
    effect_ax.set_xticks([0.0, 0.04, 0.08])
    effect_ax.set_xlabel("CFRA − Raw macro TC", labelpad=2)
    clean(effect_ax, grid=True)
    effect_ax.spines["left"].set_visible(False)
    effect_ax.text(0.50, 0.82, f"Δ {fmt(point)} [{fmt(low)}, {fmt(high)}]", transform=effect_ax.transAxes, ha="center", va="center", fontsize=8.2, color=TEXT)
    effect_ax.text(0.50, 0.18, f"{positive:.1f}% of compounds improved", transform=effect_ax.transAxes, ha="center", va="center", fontsize=8.2, color=CFRA_AC)


def d_endpoint_label(endpoint: str) -> str:
    return {
        "confirmed_activity_ap": "Activity\nAP",
        "moa_map": "MoA\nmAP",
        "target_map": "Target\nmAP",
        "reactome_target_neighbour_map": "Pathway\nmAP",
    }.get(endpoint, endpoint)


def draw_panel_d(fig: plt.Figure, data: dict[str, Any]) -> None:
    """Four aligned endpoints: one shared arm axis and split effect scales."""
    panel_label(fig, "d", "Selective downstream benefit", 0.045, 0.245)
    endpoints = ("confirmed_activity_ap", "moa_map", "target_map", "reactome_target_neighbour_map")
    labels = ("Activity AP", "MoA mAP", "Target mAP", "Pathway mAP")
    row_y = np.array([3.0, 2.0, 1.0, 0.0])

    # Shared absolute arm axis (0--0.20), with the endpoint names as row labels.
    arm_ax = fig.add_axes([0.110, 0.070, 0.215, 0.145])
    arm_ax.set_xlim(0.0, 0.20)
    arm_ax.set_ylim(-0.55, 3.55)
    arm_ax.set_yticks(row_y, labels)
    arm_ax.set_xticks([0.0, 0.1, 0.2])
    arm_ax.set_xlabel("arm value", labelpad=1)
    clean(arm_ax, grid=True)
    arm_ax.text(0.02, 1.04, "Raw", transform=arm_ax.transAxes, color=RAW, fontsize=8.2, ha="left", va="bottom")
    arm_ax.text(0.34, 1.04, "CFRA", transform=arm_ax.transAxes, color=CFRA_DE, fontsize=8.2, ha="left", va="bottom")
    arm_ax.plot(0.27, 1.05, marker="o", ms=3.0, color=RAW, transform=arm_ax.transAxes, clip_on=False)
    arm_ax.plot(0.59, 1.05, marker="D", ms=2.9, color=CFRA_DE, transform=arm_ax.transAxes, clip_on=False)
    for yi, endpoint in zip(row_y, endpoints):
        contrast = data["d_contrasts"][data["d_contrasts"]["endpoint"].astype(str).eq(endpoint)]
        metrics = data["d_metrics"][data["d_metrics"]["endpoint"].astype(str).eq(endpoint)]
        if len(contrast) != 1 or set(metrics["method"].astype(str)) != {"Raw", "CFRA"}:
            raise ValueError(f"missing D endpoint rows for {endpoint}")
        raw = float(metrics[metrics["method"].astype(str).eq("Raw")]["point"].iloc[0])
        cfra = float(metrics[metrics["method"].astype(str).eq("CFRA")]["point"].iloc[0])
        arm_ax.plot([raw, cfra], [yi, yi], color=GUIDE, lw=1.15, zorder=1)
        arm_ax.plot(raw, yi, marker="o", color=RAW, ms=3.8, zorder=3)
        arm_ax.plot(cfra, yi, marker="D", color=CFRA_DE, ms=3.7, zorder=3)
    arm_ax.tick_params(axis="y", length=0, pad=3)

    # Two real numerical effect axes: Activity is independent above, while
    # the three retrieval endpoints share one compact scale below.
    activity_ax = fig.add_axes([0.385, 0.175, 0.185, 0.045])
    retrieval_ax = fig.add_axes([0.385, 0.065, 0.185, 0.060])
    activity_ax.set_xlim(-0.02, 0.14)
    activity_ax.set_ylim(-0.7, 0.7)
    activity_ax.set_yticks([])
    activity_ax.axvline(0, color=GUIDE, lw=0.55, zorder=0)
    activity_ax.set_xticks([-0.02, 0.06, 0.14], ["−.02", "+.06", "+.14"])
    activity_ax.set_xlabel("", labelpad=1)
    activity_ax.set_title("Activity Δ", loc="left", fontsize=8.2, pad=1)
    clean(activity_ax, grid=True)
    activity_ax.spines["left"].set_visible(False)
    activity_ax.tick_params(axis="y", length=0)
    activity_contrast = data["d_contrasts"][data["d_contrasts"]["endpoint"].astype(str).eq("confirmed_activity_ap")].iloc[0]
    activity_point, activity_low, activity_high = (float(activity_contrast[key]) for key in ("point", "ci_low", "ci_high"))
    activity_ax.errorbar(activity_point, 0, xerr=[[activity_point - activity_low], [activity_high - activity_point]], fmt="D", color=CFRA_DE, mec=CFRA_DE, ms=3.8, capsize=1.7, elinewidth=0.9, zorder=3)

    retrieval_ax.set_xlim(-0.007, 0.003)
    retrieval_ax.set_ylim(-0.6, 2.6)
    retrieval_ax.set_yticks([])
    retrieval_ax.axvline(0, color=GUIDE, lw=0.55, zorder=0)
    retrieval_ax.set_xticks([-0.007, 0.0, 0.003], ["−.007", "0", "+.003"])
    retrieval_ax.set_xlabel("Retrieval Δ", labelpad=1)
    clean(retrieval_ax, grid=True)
    retrieval_ax.spines["left"].set_visible(False)
    retrieval_ax.tick_params(axis="y", length=0)
    for yi_retrieval, endpoint in zip((2.0, 1.0, 0.0), endpoints[1:]):
        contrast = data["d_contrasts"][data["d_contrasts"]["endpoint"].astype(str).eq(endpoint)].iloc[0]
        point, low, high = (float(contrast[key]) for key in ("point", "ci_low", "ci_high"))
        retrieval_ax.errorbar(point, yi_retrieval, xerr=[[point - low], [high - point]], fmt="D", color=CFRA_DE, mec=CFRA_DE, ms=3.6, capsize=1.6, elinewidth=0.85, zorder=3)

    # Rightmost sample-size column.
    n_ax = fig.add_axes([0.570, 0.070, 0.040, 0.145])
    n_ax.set_xlim(0, 1)
    n_ax.set_ylim(-0.55, 3.55)
    n_ax.axis("off")
    n_ax.text(0.5, 3.42, "n", fontsize=8.2, color=MUTED, ha="center", va="bottom")
    for yi, endpoint in zip(row_y, endpoints):
        rows = data["d_contrasts"][data["d_contrasts"]["endpoint"].astype(str).eq(endpoint)]
        n_value = rows.iloc[0].get("compound_count", rows.iloc[0].get("n_compounds", np.nan))
        n_ax.text(0.5, yi, f"{int(float(n_value)) if np.isfinite(float(n_value)) else '—'}", fontsize=8.2, color=TEXT, ha="center", va="center")


def draw_panel_e(fig: plt.Figure, data: dict[str, Any]) -> None:
    panel_label(fig, "e", "Held-out target transfer", 0.673, 0.235)
    e_x0, total_w, gap = 0.673, 0.297, 0.016
    width = (total_w - gap) / 2
    seen_ax = fig.add_axes([e_x0, 0.075, width, 0.135])
    cold_ax = fig.add_axes([e_x0 + width + gap, 0.075, width, 0.135])
    targets = ("A549", "K562", "MCF7", "Macro")
    ys = np.arange(len(targets))[::-1]
    e_target = data["e_target"]
    e_macro = data["e_macro"]

    def rows_for(endpoint: str) -> pd.DataFrame:
        rows = []
        for target in targets[:3]:
            selected = e_target[
                e_target["endpoint"].astype(str).eq(endpoint)
                & e_target["target_cell_line"].astype(str).eq(target)
                & e_target["arm"].astype(str).eq("cfra_minus_raw")
            ]
            if len(selected) != 1:
                raise ValueError(f"missing E target row: {endpoint}/{target}")
            row = selected.iloc[0]
            rows.append({"target": target, "estimate": float(row["estimate"]), "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"])})
        selected_macro = e_macro[e_macro["endpoint"].astype(str).eq(endpoint)]
        if len(selected_macro) != 1:
            raise ValueError(f"missing E macro row: {endpoint}")
        row = selected_macro.iloc[0]
        rows.append({"target": "Macro", "estimate": float(row["estimate"]), "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"])})
        return pd.DataFrame(rows)

    for ax, endpoint, title, cold in (
        (seen_ax, "pure_seen", "Seen compounds", False),
        (cold_ax, "cold_confirmation", "Cold confirmation", True),
    ):
        frame = rows_for(endpoint)
        for yi, (_, row) in zip(ys, frame.iterrows()):
            estimate, low, high = float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"])
            marker = "D" if row["target"] == "Macro" else "o"
            ax.errorbar(
                estimate,
                yi,
                xerr=[[estimate - low], [high - estimate]],
                fmt=marker,
                color=CFRA_DE,
                markerfacecolor=WHITE if cold else CFRA_DE,
                markeredgecolor=CFRA_DE,
                markeredgewidth=0.9,
                ms=3.7,
                elinewidth=0.8,
                capsize=1.7,
                zorder=3,
            )
        ax.set_title(title, fontsize=8.2, pad=2)
        ax.set_xlim(-0.0045, 0.0125)
        ax.set_ylim(-0.75, 3.55)
        ax.axvline(0, color=GUIDE, lw=0.65, zorder=0)
        ax.set_xticks([0.0, 0.005, 0.010])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_xlabel("Δ specificity", fontsize=8.2, labelpad=1)
        ax.tick_params(axis="x", labelrotation=0)
        clean(ax, grid=True)
        ax.spines["left"].set_visible(False)
        if ax is seen_ax:
            ax.set_yticks(ys, targets)
        else:
            ax.set_yticks(ys, [])
        ax.tick_params(axis="y", length=0)
    fig.text(
        e_x0 + total_w / 2,
        0.006,
        "Cold cohort inconclusive: MCF7 CI crosses 0",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color=MUTED,
    )


def render(args: argparse.Namespace) -> None:
    set_style()
    data = load_inputs(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    source_dir = args.outdir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("A_summary", data["a"]),
        (
            "A_compound",
            data["per_compound"][
                data["per_compound"]["endpoint"].astype(str).str.startswith("A_module_fisher_z_")
            ],
        ),
        ("B_summary", data["b"]),
        (
            "B_compound",
            data["per_compound"][
                data["per_compound"]["endpoint"].astype(str).isin(["B_top25_overlap", "B_top25_direction"])
            ],
        ),
        ("C_compound", data["c_compound"]),
        ("C_curve", data["curve"]),
        ("C_curve_views", data["curve_views"]),
        ("D_contrasts", data["d_contrasts"]),
        ("D_arm_metrics", data["d_metrics"]),
        ("E_target_summary", data["e_target"]),
        ("E_macro_contrasts", data["e_macro"]),
    ):
        frame.to_csv(source_dir / f"figure4_cfra_nature_{name}.csv", index=False, float_format="%.12g")

    fig = plt.figure(figsize=(180 / 25.4, 145 / 25.4), facecolor=WHITE)
    draw_panel_a(fig, data)
    draw_panel_b(fig, data)
    draw_panel_c(fig, data)
    draw_panel_d(fig, data)
    draw_panel_e(fig, data)
    fig.savefig(args.outdir / "figure4_cfra_nature.pdf")
    fig.savefig(args.outdir / "figure4_cfra_nature.svg")
    fig.savefig(args.outdir / "figure4_cfra_nature.png", dpi=600)
    plt.close(fig)

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
        "canvas_mm": {"width": 180, "height": 145},
        "palette": {"raw": RAW, "reference": REFERENCE, "cfra_A_to_C": CFRA_AC, "cfra_D_to_E": CFRA_DE, "guides": GUIDE, "background": WHITE},
        "inferential_units": {
            "A_to_D": "compound",
            "E_target": "drug",
            "E_macro": "target_cell_line_then_drug",
            "C_seed_rotation_lines": "descriptive_only",
        },
        "panel_layout": {"A_width_fraction": 0.58, "B_width_fraction": 0.42, "C": "full width", "D_width_fraction": 0.65, "E_width_fraction": 0.35},
        "panel_contract": {
            "A": "seven Raw-to-CFRA absolute means plus compound delta distributions, formal paired CI, and percent positive",
            "B": "independent Top-25 overlap and direction paired distributions with formal delta CI",
            "C": "strict six-dose physical-view example; 289-compound raw-vs-CFRA macro TC hexbin; formal paired CI and percent positive",
            "D": "AP and three retrieval mAP facets; arm dumbbells without arm CIs and paired-effect CIs",
            "E": "seen and cold-confirmation forest; cold markers hollow; All-target criterion not met",
        },
        "c_curve": data["curve_audit"],
        "inputs": {str(path): sha256(path) for path in inputs},
    }
    (args.outdir / "figure4_cfra_nature_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


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
    parser.add_argument("--outdir", type=Path, default=default / "rendered_figure4_cfra_nature")
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
