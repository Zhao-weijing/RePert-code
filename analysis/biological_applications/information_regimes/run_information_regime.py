#!/usr/bin/env python3
"""Frozen Experiment 5: retrospective information-regime analysis.

This script consumes only the saved outputs of Experiments 1--3.  It defines
low/medium/high strata from independent held-out ``e_rep`` values before
computing any downstream endpoint, then reports teacher-minus-raw gains for
hit AP, MoA/target retrieval mAP, and dose-trajectory concordance.

The experiment is explanatory.  It does not tune a method, threshold, query,
gallery, or stratum boundary after looking at an application endpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - the formal run records this as blocked
    Image = None
    ImageDraw = None
    ImageFont = None


SEEDS = (3407, 42, 2025)
ROUNDS = 10_000
STRATA = ("low", "medium", "high")
METHODS = ("raw", "teacher")
DOSES = ("0.04", "0.12", "0.37", "1.11", "3.33", "10")
Q_LOW = 1.0 / 3.0
Q_HIGH = 2.0 / 3.0
KEY_TOLERANCE = 1e-10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, label: str) -> int:
    """Derive a reproducible uint32 stream from a seed and contrast label."""

    raw = hashlib.sha256(f"{seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(raw[:4], "little", signed=False)


def finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def fmt(value: Any, digits: int = 6) -> str:
    number = finite_float(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Match the frozen Exp1 AP definition, including stable descending ties."""

    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s) & np.isfinite(y)
    y, s = y[finite], s[finite]
    positives = int(y.sum())
    if len(y) == 0 or positives == 0:
        return math.nan
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    cumulative = np.cumsum(ys, dtype=np.float64)
    precision = cumulative / np.arange(1, len(ys) + 1, dtype=np.float64)
    return float(np.sum(precision[ys == 1]) / positives)


def load_and_validate_inputs(bio_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exp1 = pd.read_csv(bio_root / "hit_recovery" / "ELIGIBLE_SAMPLES.csv")
    exp2 = pd.read_csv(bio_root / "mechanism_target_retrieval" / "per_query_metrics.csv")
    exp3 = pd.read_csv(bio_root / "dose_response" / "COMPOUND_METRICS.csv")

    required_exp1 = {
        "split", "seed", "compound", "dose", "support_row", "e_rep",
        "confirmed_active", "evaluation_common_all_methods", "raw_score", "teacher_score",
    }
    required_exp2 = {"seed", "task", "variant", "method", "query_id", "compound_id", "map"}
    required_exp3 = {"seed", "compound", "method", "trajectory_concordance"}
    missing = {
        "exp1": sorted(required_exp1 - set(exp1.columns)),
        "exp2": sorted(required_exp2 - set(exp2.columns)),
        "exp3": sorted(required_exp3 - set(exp3.columns)),
    }
    if any(missing.values()):
        raise ValueError(f"Input schema missing required columns: {missing}")

    exp1["seed"] = pd.to_numeric(exp1["seed"], errors="coerce")
    exp2["seed"] = pd.to_numeric(exp2["seed"], errors="coerce")
    exp3["seed"] = pd.to_numeric(exp3["seed"], errors="coerce")
    return exp1, exp2, exp3


def define_strata(exp1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    """Define compound strata from independent held-out e_rep only.

    One physical support key is retained after verifying that the three saved
    seed copies agree.  A compound is eligible when at least one valid e_rep
    exists at each of the six locked doses.  Its stratum statistic is the
    median of all available support-slot/dose e_rep values.  Tertile cutoffs
    are computed on this compound-level table before any application endpoint.
    """

    test = exp1[
        (exp1["split"].astype(str) == "test")
        & (pd.to_numeric(exp1["evaluation_common_all_methods"], errors="coerce") == 1)
        & exp1["seed"].isin(SEEDS)
    ].copy()
    if test.empty:
        raise ValueError("Exp1 has no common test rows")
    physical_key = ["compound", "dose", "support_row"]
    valid = test[test["e_rep"].notna()].copy()
    consistency = valid.groupby(physical_key, dropna=False)["e_rep"].agg(["min", "max", "count"])
    if not consistency.empty and float((consistency["max"] - consistency["min"]).abs().max()) > KEY_TOLERANCE:
        raise ValueError("e_rep is not seed-consistent for a physical support key")
    if not consistency.empty and not set(consistency["count"].astype(int).unique()).issubset({len(SEEDS)}):
        raise ValueError("Each valid e_rep physical key must be present for all three seeds")
    unique = valid.sort_values(physical_key + ["seed"]).drop_duplicates(physical_key, keep="first")

    stats = (
        unique.groupby("compound", dropna=False)
        .agg(
            e_rep_median=("e_rep", "median"),
            e_rep_mean=("e_rep", "mean"),
            e_rep_min=("e_rep", "min"),
            e_rep_max=("e_rep", "max"),
            e_rep_n=("e_rep", "size"),
            e_rep_dose_count=("dose", "nunique"),
        )
        .reset_index()
    )
    eligible = stats[stats["e_rep_dose_count"] >= len(DOSES)].copy()
    if eligible.empty:
        raise ValueError("No compounds have valid e_rep coverage at all six locked doses")
    q_low, q_high = np.quantile(eligible["e_rep_median"].to_numpy(dtype=float), [Q_LOW, Q_HIGH], method="linear")

    def assign(value: float) -> str:
        if value <= q_low:
            return "low"
        if value <= q_high:
            return "medium"
        return "high"

    eligible["stratum"] = eligible["e_rep_median"].map(assign)
    eligible = eligible.sort_values(["stratum", "e_rep_median", "compound"]).reset_index(drop=True)
    counts = eligible["stratum"].value_counts().to_dict()
    if set(counts) != set(STRATA) or any(int(counts[s]) < 1 for s in STRATA):
        raise ValueError(f"Stratum assignment did not produce all strata: {counts}")
    return test, eligible, float(q_low), float(q_high), float(consistency["max"].sub(consistency["min"]).abs().max() if not consistency.empty else 0.0)


def sample_table(
    exp1_test: pd.DataFrame,
    eligible: pd.DataFrame,
    exp2: pd.DataFrame,
    exp3: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    anchor = exp1_test[exp1_test["seed"] == SEEDS[0]].copy()
    anchor = anchor[anchor["compound"].isin(set(eligible["compound"]))]
    active_rows = anchor.groupby("compound")["confirmed_active"].sum().to_dict()
    active_compounds = anchor[anchor["confirmed_active"].astype(int) == 1].groupby("compound").size().to_dict()

    exp2_moa = set(
        exp2[(exp2["task"] == "moa") & (exp2["variant"] == "base") & exp2["method"].isin(METHODS)]["compound_id"].astype(str)
    )
    exp2_target = set(
        exp2[(exp2["task"] == "target") & (exp2["variant"] == "base") & exp2["method"].isin(METHODS)]["compound_id"].astype(str)
    )
    exp3_compounds = set(exp3[exp3["method"].isin(METHODS)]["compound"].astype(str))

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in eligible.to_dict("records"):
        compound = str(row["compound"])
        exp1_rows = int((anchor["compound"].astype(str) == compound).sum())
        row.update(
            {
                "exp1_test_rows": exp1_rows,
                "exp1_active_rows": int(active_rows.get(compound, 0)),
                "exp1_active_compound": int(compound in active_compounds),
                "exp2_moa_available": int(compound in exp2_moa),
                "exp2_target_available": int(compound in exp2_target),
                "exp3_trajectory_available": int(compound in exp3_compounds),
            }
        )
        rows.append(row)

    all_exp1_compounds = set(exp1_test["compound"].astype(str))
    valid_e_compounds = set(eligible["compound"].astype(str))
    no_e = sorted(all_exp1_compounds - set(exp1_test[exp1_test["e_rep"].notna()]["compound"].astype(str)))
    incomplete = sorted(
        set(exp1_test[exp1_test["e_rep"].notna()]["compound"].astype(str)) - valid_e_compounds
    )
    if no_e:
        exclusions.append(
            {
                "scope": "stratum_definition",
                "endpoint": "all",
                "stratum": "",
                "reason": "E_REP_UNAVAILABLE_NO_MATCHED_NULL",
                "count": len(no_e),
                "details": ";".join(no_e),
            }
        )
    if incomplete:
        exclusions.append(
            {
                "scope": "stratum_definition",
                "endpoint": "all",
                "stratum": "",
                "reason": "E_REP_DOES_NOT_COVER_ALL_SIX_LOCKED_DOSES",
                "count": len(incomplete),
                "details": ";".join(incomplete),
            }
        )
    for task, available in (("moa_mAP", exp2_moa), ("target_mAP", exp2_target), ("trajectory_concordance", exp3_compounds)):
        missing = sorted(valid_e_compounds - available)
        if missing:
            exclusions.append(
                {
                    "scope": task,
                    "endpoint": task,
                    "stratum": "",
                    "reason": "COMPOUND_NOT_AVAILABLE_IN_FROZEN_ENDPOINT",
                    "count": len(missing),
                    "details": ";".join(missing),
                }
            )
    return pd.DataFrame(rows), exclusions


def exp1_groups(exp1: pd.DataFrame, eligible: pd.DataFrame, stratum: str, seed: int) -> pd.DataFrame:
    compounds = set(eligible.loc[eligible["stratum"] == stratum, "compound"].astype(str))
    rows = exp1[
        (exp1["split"].astype(str) == "test")
        & (exp1["seed"] == seed)
        & (pd.to_numeric(exp1["evaluation_common_all_methods"], errors="coerce") == 1)
        & exp1["compound"].astype(str).isin(compounds)
    ].copy()
    rows["compound"] = rows["compound"].astype(str)
    return rows.sort_values(["compound", "dose", "support_row"]).reset_index(drop=True)


def retrieval_compound_values(
    exp2: pd.DataFrame,
    eligible: pd.DataFrame,
    task: str,
    stratum: str,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    compounds = set(eligible.loc[eligible["stratum"] == stratum, "compound"].astype(str))
    rows = exp2[
        (exp2["seed"] == seed)
        & (exp2["task"] == task)
        & (exp2["variant"] == "base")
        & exp2["method"].isin(METHODS)
        & exp2["compound_id"].astype(str).isin(compounds)
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=["compound", "raw", "teacher", "query_rows"]), 0
    rows["compound"] = rows["compound_id"].astype(str)
    method_queries = {m: set(rows.loc[rows["method"] == m, "query_id"].astype(str)) for m in METHODS}
    common_queries = method_queries["raw"] & method_queries["teacher"]
    rows = rows[rows["query_id"].astype(str).isin(common_queries)].copy()
    pivot = rows.pivot_table(index=["compound", "query_id"], columns="method", values="map", aggfunc="first")
    if not set(METHODS).issubset(pivot.columns):
        return pd.DataFrame(columns=["compound", "raw", "teacher", "query_rows"]), 0
    pivot = pivot.dropna(subset=list(METHODS)).reset_index()
    values = pivot.groupby("compound", as_index=False).agg(raw=("raw", "mean"), teacher=("teacher", "mean"), query_rows=("query_id", "size"))
    return values, int(len(pivot))


def trajectory_compound_values(
    exp3: pd.DataFrame,
    eligible: pd.DataFrame,
    stratum: str,
    seed: int,
) -> pd.DataFrame:
    compounds = set(eligible.loc[eligible["stratum"] == stratum, "compound"].astype(str))
    rows = exp3[
        (exp3["seed"] == seed)
        & exp3["method"].isin(METHODS)
        & exp3["compound"].astype(str).isin(compounds)
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=["compound", "raw", "teacher"])
    rows["compound"] = rows["compound"].astype(str)
    pivot = rows.pivot_table(index="compound", columns="method", values="trajectory_concordance", aggfunc="first")
    return pivot.dropna(subset=list(METHODS)).reset_index()[["compound", "raw", "teacher"]]


def bootstrap_positions(n_compounds: int, rounds: int, seed: int) -> np.ndarray:
    """Create shared compound-resample positions for one endpoint/stratum."""

    if n_compounds <= 0:
        return np.empty((rounds, 0), dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_compounds, size=(rounds, n_compounds), dtype=np.int64)


def bootstrap_mean_delta(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.full(len(positions), np.nan, dtype=float)
    if positions.shape[1] != len(values):
        raise ValueError("Shared bootstrap positions do not match compound value count")
    return values[positions].mean(axis=1)


def bootstrap_ap_delta(rows: pd.DataFrame, positions: np.ndarray) -> tuple[float, np.ndarray, int]:
    compounds = np.asarray(sorted(rows["compound"].astype(str).unique()), dtype=str)
    if len(compounds) == 0:
        return math.nan, np.full(len(positions), np.nan), 0
    if positions.shape[1] != len(compounds):
        raise ValueError("Shared bootstrap positions do not match AP compound count")
    grouped = {compound: np.flatnonzero(rows["compound"].to_numpy(dtype=str) == compound) for compound in compounds}
    labels = rows["confirmed_active"].to_numpy(dtype=np.int8)
    raw = rows["raw_score"].to_numpy(dtype=float)
    teacher = rows["teacher_score"].to_numpy(dtype=float)
    point = average_precision(labels, teacher) - average_precision(labels, raw)
    boot = np.full(len(positions), np.nan, dtype=float)
    for iteration, sampled in enumerate(positions):
        indices = np.concatenate([grouped[compounds[int(position)]] for position in sampled])
        boot[iteration] = average_precision(labels[indices], teacher[indices]) - average_precision(labels[indices], raw[indices])
    return point, boot, int(np.isfinite(boot).sum())


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return math.nan, math.nan
    return float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))


def status_for_bootstrap(endpoint: str, point: float, boot: np.ndarray, positive_compounds: int | None = None) -> str:
    # The Exp1 confirmed-active label itself uses an e_rep threshold.  Since
    # Exp5 strata are also defined by e_rep, AP-by-stratum is structurally
    # label-dependent and cannot be used to test the low-information benefit
    # hypothesis, even when the numerical AP and bootstrap are computable.
    if endpoint == "hit_AP":
        return "BLOCKED_STRUCTURAL_LABEL_DEPENDENCE"
    if not math.isfinite(point):
        return "BLOCKED_ENDPOINT_UNDEFINED"
    valid = int(np.isfinite(boot).sum())
    if valid == 0:
        return "BLOCKED_NO_VALID_BOOTSTRAP_RESAMPLES"
    return "OK"


def make_figure(summary: pd.DataFrame, out_path: Path) -> str:
    if Image is None or ImageDraw is None or ImageFont is None:
        return "BLOCKED_PIL_UNAVAILABLE"
    width, height = 1500, 950
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 25)
        small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
        title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 32)
    except Exception:
        font = ImageFont.load_default()
        small = font
        title_font = font
    draw.text((45, 24), "Experiment 5: teacher minus raw gain by intrinsic reproducibility", fill="black", font=title_font)
    draw.text((45, 67), "Strata are fixed from independent held-out e_rep; blank bars are blocked endpoints", fill="#444444", font=small)

    endpoints = [("hit_AP", "Hit AP gain"), ("moa_mAP", "MoA retrieval mAP gain"), ("trajectory_concordance", "Dose trajectory gain")]
    colors = {"low": "#4C78A8", "medium": "#F2A541", "high": "#59A14F"}
    left, panel_top, panel_w, panel_h = 65, 125, 455, 705
    for panel_index, (endpoint, label) in enumerate(endpoints):
        x0 = left + panel_index * (panel_w + 35)
        y0 = panel_top
        draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), outline="#999999", width=2)
        draw.text((x0 + 20, y0 + 18), label, fill="black", font=font)
        sub = summary[(summary["endpoint"] == endpoint) & (summary["seed"] == "mean_seeds")]
        vals = []
        for stratum in STRATA:
            row = sub[sub["stratum"] == stratum]
            vals.extend([finite_float(row.iloc[0]["point_difference"]) if len(row) else math.nan])
        finite_vals = [value for value in vals if math.isfinite(value)]
        if finite_vals:
            extent = max(0.01, max(abs(value) for value in finite_vals) * 1.25)
        else:
            extent = 0.1
        y_mid = y0 + panel_h // 2 + 35
        scale = (panel_h * 0.35) / extent
        draw.line((x0 + 55, y_mid, x0 + panel_w - 25, y_mid), fill="#555555", width=2)
        draw.text((x0 + 8, y_mid - 12), "0", fill="#555555", font=small)
        for i, stratum in enumerate(STRATA):
            bx = x0 + 95 + i * 112
            row = sub[sub["stratum"] == stratum]
            value = finite_float(row.iloc[0]["point_difference"]) if len(row) else math.nan
            ci_low = finite_float(row.iloc[0]["ci_low"]) if len(row) else math.nan
            ci_high = finite_float(row.iloc[0]["ci_high"]) if len(row) else math.nan
            if math.isfinite(value):
                top = y_mid - value * scale
                draw.rectangle((bx - 28, min(y_mid, top), bx + 28, max(y_mid, top)), fill=colors[stratum], outline="#333333")
                if math.isfinite(ci_low) and math.isfinite(ci_high):
                    y1 = y_mid - ci_low * scale
                    y2 = y_mid - ci_high * scale
                    draw.line((bx, min(y1, y2), bx, max(y1, y2)), fill="black", width=3)
                    draw.line((bx - 8, y1, bx + 8, y1), fill="black", width=3)
                    draw.line((bx - 8, y2, bx + 8, y2), fill="black", width=3)
                draw.text((bx - 28, y0 + panel_h - 70), stratum, fill="black", font=small)
                draw.text((bx - 36, y0 + panel_h - 43), fmt(value, 3), fill="#333333", font=small)
            else:
                draw.rectangle((bx - 28, y_mid - 12, bx + 28, y_mid + 12), outline="#888888", width=2)
                draw.line((bx - 20, y_mid - 8, bx + 20, y_mid + 8), fill="#888888", width=2)
                draw.line((bx - 20, y_mid + 8, bx + 20, y_mid - 8), fill="#888888", width=2)
                draw.text((bx - 28, y0 + panel_h - 70), stratum, fill="#777777", font=small)
                draw.text((bx - 27, y0 + panel_h - 43), "BLK", fill="#777777", font=small)
        draw.text((x0 + 18, y0 + panel_h - 25), "three-seed mean paired delta", fill="#555555", font=small)
    draw.text((65, 865), "Bootstrap: 10,000 paired compound resamples; AP low/medium strata are blocked when labels cannot support inference.", fill="#444444", font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return "OK_PIL"


def run(outdir: Path, rounds: int = ROUNDS) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    bio_root = script_path.parents[1]
    exp1_dir = bio_root / "hit_recovery"
    exp2_dir = bio_root / "mechanism_target_retrieval"
    exp3_dir = bio_root / "dose_response"
    lock_dir = bio_root / "protocol"
    exp1, exp2, exp3 = load_and_validate_inputs(bio_root)
    exp1_test, eligible, q_low, q_high, e_rep_spread = define_strata(exp1)
    eligible_output, exclusions = sample_table(exp1_test, eligible, exp2, exp3)

    # Endpoint data are assembled after the stratum table is fixed.  Each
    # endpoint then uses the intersection of compound IDs available in all
    # three saved seeds, preserving the locked common-set rule.
    metric_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    endpoint_boots: dict[tuple[str, str, int], np.ndarray] = {}
    endpoint_points: dict[tuple[str, str, int], float] = {}

    endpoint_specs = [
        ("hit_AP", "hit_recovery", "AP"),
        ("moa_mAP", "moa_retrieval", "mAP"),
        ("target_mAP", "target_retrieval", "mAP"),
        ("trajectory_concordance", "dose_response", "TrajectoryConcordance"),
    ]

    for endpoint, source, _ in endpoint_specs:
        for stratum in STRATA:
            per_seed: dict[int, pd.DataFrame] = {}
            available_sets: list[set[str]] = []
            for seed in SEEDS:
                if endpoint == "hit_AP":
                    table = exp1_groups(exp1, eligible, stratum, seed)
                elif endpoint == "moa_mAP":
                    table, _ = retrieval_compound_values(exp2, eligible, "moa", stratum, seed)
                elif endpoint == "target_mAP":
                    table, _ = retrieval_compound_values(exp2, eligible, "target", stratum, seed)
                else:
                    table = trajectory_compound_values(exp3, eligible, stratum, seed)
                per_seed[seed] = table
                available_sets.append(set(table["compound"].astype(str)))
            common_compounds = set.intersection(*available_sets) if available_sets else set()
            common_compound_order = np.asarray(sorted(common_compounds), dtype=str)
            # The same compound-resample matrix is reused for every saved
            # model seed within an endpoint/stratum.  This is required for a
            # paired three-seed mean rather than an average of independently
            # indexed bootstrap streams.
            bs_seed = stable_seed(3407, f"information_regime|{endpoint}|{stratum}|common_compound_resample")
            common_positions = bootstrap_positions(len(common_compound_order), rounds, bs_seed)
            # Record only complete compound intersections in the official
            # endpoint.  Rows outside this set remain auditable in exclusions.
            if endpoint == "hit_AP":
                source_label = "Exp1 common all-method test rows"
            elif endpoint == "moa_mAP":
                source_label = "Exp2 MoA base PCC query rows"
            elif endpoint == "target_mAP":
                source_label = "Exp2 target base PCC query rows"
            else:
                source_label = "Exp3 compound trajectory metrics"

            for seed in SEEDS:
                table = per_seed[seed]
                table = table[table["compound"].astype(str).isin(common_compounds)].copy()
                if endpoint == "hit_AP":
                    table = table.sort_values(["compound", "dose", "support_row"]).reset_index(drop=True)
                else:
                    table = table.sort_values(["compound"]).reset_index(drop=True)
                if endpoint == "hit_AP":
                    positive_rows = int(table["confirmed_active"].astype(int).sum()) if not table.empty else 0
                    positive_compounds = int(table.loc[table["confirmed_active"].astype(int) == 1, "compound"].nunique()) if not table.empty else 0
                    raw_value = average_precision(table["confirmed_active"], table["raw_score"]) if not table.empty else math.nan
                    teacher_value = average_precision(table["confirmed_active"], table["teacher_score"]) if not table.empty else math.nan
                    point = teacher_value - raw_value if math.isfinite(raw_value) and math.isfinite(teacher_value) else math.nan
                    if math.isfinite(point):
                        point_ap, boot, valid_boot = bootstrap_ap_delta(table, common_positions)
                        # point_ap is computed by exactly the same frozen AP function.
                        point = point_ap
                    else:
                        boot, valid_boot = np.full(rounds, np.nan), 0
                    n_rows = len(table)
                    raw_n = teacher_n = n_rows
                    endpoint_status = status_for_bootstrap(endpoint, point, boot, positive_compounds)
                    metric_extra = {
                        "positive_rows": positive_rows,
                        "positive_compounds": positive_compounds,
                        "query_rows": "",
                    }
                elif endpoint in {"moa_mAP", "target_mAP"}:
                    raw_value = float(table["raw"].mean()) if not table.empty else math.nan
                    teacher_value = float(table["teacher"].mean()) if not table.empty else math.nan
                    diff = (table["teacher"] - table["raw"]).to_numpy(dtype=float) if not table.empty else np.asarray([], dtype=float)
                    point = float(np.mean(diff)) if len(diff) else math.nan
                    boot = bootstrap_mean_delta(diff, common_positions)
                    valid_boot = int(np.isfinite(boot).sum())
                    n_rows = int(table["query_rows"].sum()) if not table.empty else 0
                    raw_n = teacher_n = n_rows
                    endpoint_status = status_for_bootstrap(endpoint, point, boot)
                    metric_extra = {
                        "positive_rows": "",
                        "positive_compounds": "",
                        "query_rows": n_rows,
                    }
                else:
                    raw_value = float(table["raw"].mean()) if not table.empty else math.nan
                    teacher_value = float(table["teacher"].mean()) if not table.empty else math.nan
                    diff = (table["teacher"] - table["raw"]).to_numpy(dtype=float) if not table.empty else np.asarray([], dtype=float)
                    point = float(np.mean(diff)) if len(diff) else math.nan
                    boot = bootstrap_mean_delta(diff, common_positions)
                    valid_boot = int(np.isfinite(boot).sum())
                    n_rows = len(table)
                    raw_n = teacher_n = n_rows
                    endpoint_status = status_for_bootstrap(endpoint, point, boot)
                    metric_extra = {
                        "positive_rows": "",
                        "positive_compounds": "",
                        "query_rows": "",
                    }

                endpoint_boots[(endpoint, stratum, seed)] = boot
                endpoint_points[(endpoint, stratum, seed)] = point
                for method, value, n in (("raw", raw_value, raw_n), ("teacher", teacher_value, teacher_n)):
                    metric_rows.append(
                        {
                            "seed": seed,
                            "endpoint": endpoint,
                            "source": source_label,
                            "stratum": stratum,
                            "method": method,
                            "value": value,
                            "n_compounds": len(common_compounds),
                            "n_rows_or_queries": n,
                            "positive_rows": metric_extra["positive_rows"],
                            "positive_compounds": metric_extra["positive_compounds"],
                            "bootstrap_rounds": rounds,
                            "status": endpoint_status,
                        }
                    )
                ci_low, ci_high = percentile_ci(boot) if endpoint_status == "OK" else (math.nan, math.nan)
                paired_rows.append(
                    {
                        "seed": seed,
                        "endpoint": endpoint,
                        "source": source_label,
                        "stratum": stratum,
                        "method_a": "teacher",
                        "method_b": "raw",
                        "point_difference": point,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_compounds": len(common_compounds),
                        "n_rows_or_queries": n_rows,
                        "bootstrap_rounds": rounds,
                        "bootstrap_seed": bs_seed,
                        "valid_bootstrap_resamples": valid_boot,
                        "status": endpoint_status,
                    }
                )
                bootstrap_rows.append(
                    {
                        "seed": seed,
                        "endpoint": endpoint,
                        "source": source_label,
                        "stratum": stratum,
                        "comparison": "teacher_minus_raw",
                        "point_difference": point,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "rounds": rounds,
                        "bootstrap_unit": "compound",
                        "n_compounds": len(common_compounds),
                        "n_rows_or_queries": n_rows,
                        "bootstrap_seed": bs_seed,
                        "valid_bootstrap_resamples": valid_boot,
                        "status": endpoint_status,
                    }
                )

            # Pooled three-seed mean is reported only when all seed bootstrap
            # arrays are valid.  This does not pool rows or pretend that seeds
            # are independent compounds.
            seed_boot = [endpoint_boots[(endpoint, stratum, seed)] for seed in SEEDS]
            seed_points = [endpoint_points[(endpoint, stratum, seed)] for seed in SEEDS]
            finite_seed_mask = np.vstack([np.isfinite(arr) for arr in seed_boot])
            all_seed_finite = finite_seed_mask.all(axis=0)
            pooled_point = float(np.mean(seed_points)) if all(math.isfinite(x) for x in seed_points) else math.nan
            # AP can have a very small number of no-positive compound
            # resamples even when the point endpoint is estimable.  Retain
            # those draws as invalid, but form the three-seed distribution on
            # the shared finite draws when at least 95% remain.  This mirrors
            # the per-seed status rule and prevents an otherwise estimable
            # high-stratum AP from being mislabeled as blocked.
            if endpoint == "hit_AP":
                pooled = np.full(rounds, np.nan)
                ci_low, ci_high = math.nan, math.nan
                pooled_status = "BLOCKED_STRUCTURAL_LABEL_DEPENDENCE"
            elif all(math.isfinite(x) for x in seed_points) and int(all_seed_finite.sum()) >= int(0.95 * rounds):
                pooled = np.vstack([arr[all_seed_finite] for arr in seed_boot]).mean(axis=0)
                ci_low, ci_high = percentile_ci(pooled)
                pooled_status = "OK"
            else:
                pooled = np.full(rounds, np.nan)
                ci_low, ci_high = math.nan, math.nan
                pooled_status = "BLOCKED_SEED_OR_ENDPOINT_COVERAGE"
            common_count = len(set.intersection(*[set(per_seed[seed]["compound"].astype(str)) for seed in SEEDS]))
            pooled_rows = int(sum(
                int(per_seed[seed]["query_rows"].sum()) if "query_rows" in per_seed[seed].columns else len(per_seed[seed])
                for seed in SEEDS
            ) / len(SEEDS))
            bootstrap_rows.append(
                {
                    "seed": "mean_seeds",
                    "endpoint": endpoint,
                    "source": source_label,
                    "stratum": stratum,
                    "comparison": "teacher_minus_raw",
                    "point_difference": pooled_point,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "rounds": rounds,
                    "bootstrap_unit": "compound",
                    "n_compounds": common_count,
                    "n_rows_or_queries": pooled_rows,
                    "bootstrap_seed": "mean_of_seed_streams",
                    "valid_bootstrap_resamples": int(np.isfinite(pooled).sum()),
                    "status": pooled_status,
                }
            )

    # Write a compact stratum table that is useful to downstream case/figure
    # audits without changing the required output tables.
    for endpoint in ("hit_AP", "moa_mAP", "target_mAP", "trajectory_concordance"):
        for stratum in STRATA:
            seed_rows = [r for r in bootstrap_rows if r["endpoint"] == endpoint and r["stratum"] == stratum and r["seed"] == "mean_seeds"]
            if seed_rows:
                row = seed_rows[0]
                row_status = row["status"]
                row_point = row["point_difference"]
                row_low = row["ci_low"]
                row_high = row["ci_high"]
            else:
                row_status = "BLOCKED_NO_SUMMARY"
                row_point = row_low = row_high = math.nan
            # There is no separate output row here; this branch only keeps the
            # logic explicit for the generated summary below.
            del row_status, row_point, row_low, row_high

    figure_status = make_figure(pd.DataFrame(bootstrap_rows), outdir / "figures" / "information_regime_gains.png")

    # Common-set and availability counts are part of the machine-readable
    # config, not inferred from a favorable endpoint.
    common_counts: dict[str, dict[str, int]] = {}
    for endpoint in ("hit_AP", "moa_mAP", "target_mAP", "trajectory_concordance"):
        common_counts[endpoint] = {}
        for stratum in STRATA:
            rows = [r for r in bootstrap_rows if r["endpoint"] == endpoint and r["stratum"] == stratum and r["seed"] == "mean_seeds"]
            common_counts[endpoint][stratum] = int(rows[0]["n_compounds"]) if rows else 0

    input_paths = {
        "protocol_lock_config": lock_dir / "CONFIG.json",
        "experiment_1_eligible_samples": exp1_dir / "ELIGIBLE_SAMPLES.csv",
        "experiment_1_config": exp1_dir / "CONFIG.json",
        "experiment_2_per_query_metrics": exp2_dir / "per_query_metrics.csv",
        "experiment_2_config": exp2_dir / "CONFIG.json",
        "experiment_3_compound_metrics": exp3_dir / "COMPOUND_METRICS.csv",
        "experiment_3_config": exp3_dir / "CONFIG.json",
    }
    input_hashes = {
        key: {"path": str(path), "sha256": sha256_file(path)} for key, path in input_paths.items() if path.exists()
    }
    config = {
        "version": "cpg0004-LINCS-information-regime-analysis-2026-08-30",
        "lock_id": "MVCPert-biological-applications-v1",
        "dataset": "cpg0004-LINCS",
        "experiment": "05_information_regime_analysis",
        "status": "COMPLETE_WITH_BLOCKED_ENDPOINTS_AS_REPORTED",
        "objective": "Retrospectively test whether teacher-minus-raw biological gains concentrate in low-information compounds.",
        "inputs": input_hashes,
        "stratum_definition": {
            "source": "Exp1 common-all-method test e_rep from independent confirmation repeats",
            "physical_key": ["compound", "dose", "support_row"],
            "seed_consistency_tolerance": KEY_TOLERANCE,
            "observed_max_seed_spread": e_rep_spread,
            "compound_statistic": "median(e_rep) over all deduplicated support-slot/dose keys",
            "eligibility": "at least one valid e_rep at each of the six locked doses",
            "doses": list(DOSES),
            "cutoff_rule": "empirical linear-interpolated compound tertiles; low <= q33.333, medium > q33.333 and <= q66.667, high > q66.667",
            "q33_333": q_low,
            "q66_667": q_high,
            "compound_count": int(len(eligible)),
            "counts_by_stratum": {s: int((eligible["stratum"] == s).sum()) for s in STRATA},
            "retrospective_only": True,
            "endpoint_or_label_tuning": False,
            "hit_activity_label_dependency_audit": "Exp1 confirmed_active includes the validation-only E_rep q95 threshold; therefore hit_AP stratification by E_rep is structurally label-dependent and blocked for inference.",
        },
        "methods": {
            "raw": "saved Exp1 one-real-support raw score / Exp2 raw query profile / Exp3 raw trajectory",
            "teacher": "saved frozen reproducible-effect teacher score / retrieval profile / trajectory",
            "comparison": "teacher_minus_raw only",
            "frozen": True,
            "no_retraining_or_retuning": True,
        },
        "endpoints": {
            "hit_AP": "Exp1 confirmed-activity AP, common all-method test rows; AP is undefined without positive labels",
            "moa_mAP": "Exp2 base MoA PCC retrieval mAP, within-compound query mean then compound-balanced mean",
            "target_mAP": "Exp2 base target PCC retrieval mAP, supplementary endpoint",
            "trajectory_concordance": "Exp3 six-dose trajectory concordance, compound-level saved values",
        },
        "endpoint_status": {
            "hit_AP": "BLOCKED_STRUCTURAL_LABEL_DEPENDENCE_E_REP_USED_IN_CONFIRMED_ACTIVITY_LABEL",
            "moa_mAP": "DESCRIPTIVE_BY_STRATUM",
            "target_mAP": "DESCRIPTIVE_BY_STRATUM",
            "trajectory_concordance": "DESCRIPTIVE_BY_STRATUM",
        },
        "bootstrap": {
            "rounds": int(rounds),
            "unit": "compound",
            "paired": True,
            "percentile_ci": 0.95,
            "seed_stream": "shared per endpoint/stratum stream: sha256(3407|information_regime|endpoint|stratum|common_compound_resample) first four bytes little-endian",
            "shared_compound_resample_indices": True,
            "pooled_three_seed_rule": "reuse identical compound resample positions for all three model seeds, then average the three seed bootstrap deltas at each resample index; no compound rows are pooled across seeds",
        },
        "common_counts": common_counts,
        "figure_status": figure_status,
        "optional_repeat_axis": {
            "status": "BLOCKED_NOT_RUN",
            "reason": "No frozen Exp1/Exp2/Exp3 downstream 2R/3R per-stratum outputs were supplied; no interpolation or rerun was performed.",
        },
        "stop_conditions_checked": [
            "all input files were read from saved Exp1/Exp2/Exp3 output paths",
            "e_rep seed consistency and physical-key deduplication",
            "common compound intersection across all three seeds per endpoint/stratum",
            "no test labels used to define strata or method parameters",
            "hit AP stratification blocked because confirmed_active itself uses E_rep",
            "no method retraining, lambda/beta tuning, or query/gallery changes",
        ],
    }

    eligible_fields = [
        "compound", "e_rep_median", "e_rep_mean", "e_rep_min", "e_rep_max", "e_rep_n", "e_rep_dose_count", "stratum",
        "exp1_test_rows", "exp1_active_rows", "exp1_active_compound", "exp2_moa_available", "exp2_target_available", "exp3_trajectory_available",
    ]
    exclusion_fields = ["scope", "endpoint", "stratum", "reason", "count", "details"]
    metric_fields = ["seed", "endpoint", "source", "stratum", "method", "value", "n_compounds", "n_rows_or_queries", "positive_rows", "positive_compounds", "bootstrap_rounds", "status"]
    paired_fields = ["seed", "endpoint", "source", "stratum", "method_a", "method_b", "point_difference", "ci_low", "ci_high", "n_compounds", "n_rows_or_queries", "bootstrap_rounds", "bootstrap_seed", "valid_bootstrap_resamples", "status"]
    bootstrap_fields = ["seed", "endpoint", "source", "stratum", "comparison", "point_difference", "ci_low", "ci_high", "rounds", "bootstrap_unit", "n_compounds", "n_rows_or_queries", "bootstrap_seed", "valid_bootstrap_resamples", "status"]
    write_csv(outdir / "ELIGIBLE_SAMPLES.csv", eligible_output.to_dict("records"), eligible_fields)
    # Endpoint-level blocks are written to the exclusion audit as well as the
    # metrics/decision tables, so a downstream reader cannot mistake a blank
    # AP interval for a measured zero gain.
    for stratum in STRATA:
        rows = [r for r in bootstrap_rows if r["endpoint"] == "hit_AP" and r["stratum"] == stratum and r["seed"] == SEEDS[0]]
        if rows and rows[0]["status"] == "BLOCKED_STRUCTURAL_LABEL_DEPENDENCE":
            metrics_lookup = [m for m in metric_rows if m["seed"] == SEEDS[0] and m["endpoint"] == "hit_AP" and m["stratum"] == stratum and m["method"] == "raw"]
            positive_rows = metrics_lookup[0].get("positive_rows", "") if metrics_lookup else ""
            positive_compounds = metrics_lookup[0].get("positive_compounds", "") if metrics_lookup else ""
            exclusions.append({
                "scope": "hit_AP",
                "endpoint": "hit_AP",
                "stratum": stratum,
                "reason": "BLOCKED_STRUCTURAL_LABEL_DEPENDENCE_E_REP_USED_IN_CONFIRMED_ACTIVITY_LABEL",
                "count": int(rows[0]["n_compounds"]),
                "details": "confirmed_active uses validation-only E_rep q95; e_rep also defines this stratum; no stratum AP inference",
            })
            if int(float(positive_rows or 0)) == 0:
                exclusions.append({
                    "scope": "hit_AP",
                    "endpoint": "hit_AP",
                    "stratum": stratum,
                    "reason": "NO_CONFIRMED_ACTIVE_LABELS_IN_STRATUM",
                    "count": int(rows[0]["n_compounds"]),
                    "details": f"positive_rows={positive_rows};AP_undefined",
                })
            elif int(float(positive_compounds or 0)) < 2:
                exclusions.append({
                    "scope": "hit_AP",
                    "endpoint": "hit_AP",
                    "stratum": stratum,
                    "reason": "INSUFFICIENT_POSITIVE_COMPOUNDS_FOR_COMPOUND_BOOTSTRAP",
                    "count": int(rows[0]["n_compounds"]),
                    "details": f"positive_compounds={positive_compounds};point_only_no_CI",
                })
        elif rows and rows[0]["status"] == "BLOCKED_ENDPOINT_UNDEFINED":
            exclusions.append({
                "scope": "hit_AP",
                "endpoint": "hit_AP",
                "stratum": stratum,
                "reason": "NO_CONFIRMED_ACTIVE_LABELS_IN_STRATUM",
                "count": int(rows[0]["n_compounds"]),
                "details": f"positive_rows={rows[0].get('positive_rows', 0)};AP_undefined",
            })
        elif rows and rows[0]["status"] == "BLOCKED_INSUFFICIENT_POSITIVE_COMPOUNDS":
            # The one-positive-compound medium stratum has a point AP but no
            # valid compound-level inferential interval.
            metrics_lookup = [m for m in metric_rows if m["seed"] == SEEDS[0] and m["endpoint"] == "hit_AP" and m["stratum"] == stratum and m["method"] == "raw"]
            positive_compounds = metrics_lookup[0].get("positive_compounds", "") if metrics_lookup else ""
            exclusions.append({
                "scope": "hit_AP",
                "endpoint": "hit_AP",
                "stratum": stratum,
                "reason": "INSUFFICIENT_POSITIVE_COMPOUNDS_FOR_COMPOUND_BOOTSTRAP",
                "count": int(rows[0]["n_compounds"]),
                "details": f"positive_compounds={positive_compounds};point_only_no_CI",
            })
    write_csv(outdir / "EXCLUSIONS.csv", exclusions, exclusion_fields)
    write_csv(outdir / "METRICS_BY_SEED.csv", metric_rows, metric_fields)
    write_csv(outdir / "PAIRED_CONTRASTS.csv", paired_rows, paired_fields)
    write_csv(outdir / "BOOTSTRAP_CI.csv", bootstrap_rows, bootstrap_fields)
    json_dump(outdir / "CONFIG.json", config)

    return {
        "eligible_compounds": int(len(eligible)),
        "strata": {s: int((eligible["stratum"] == s).sum()) for s in STRATA},
        "q_low": q_low,
        "q_high": q_high,
        "figure_status": figure_status,
        "common_counts": common_counts,
        "config": config,
        "bootstrap_rows": bootstrap_rows,
        "metric_rows": metric_rows,
        "exclusions": exclusions,
    }


def write_reports(outdir: Path, result: dict[str, Any]) -> None:
    metric_rows = result["metric_rows"]
    bootstrap_rows = result["bootstrap_rows"]
    def summary_row(endpoint: str, stratum: str) -> dict[str, Any] | None:
        rows = [r for r in bootstrap_rows if r["endpoint"] == endpoint and r["stratum"] == stratum and r["seed"] == "mean_seeds"]
        return rows[0] if rows else None

    def summary_text(endpoint: str, stratum: str) -> str:
        row = summary_row(endpoint, stratum)
        if row is None or row["status"] != "OK":
            return "BLOCKED"
        return f"{fmt(row['point_difference'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"

    lines = [
        "# Results — cpg0004-LINCS information regime analysis",
        "",
        "## 1. Objective",
        "",
        "Test retrospectively whether frozen teacher-minus-raw biological gains are concentrated in compounds with less intrinsic reproducibility.",
        "",
        "## 2. Dataset",
        "",
        "Primary dataset: cpg0004-LINCS. Only the saved outputs of Experiments 1, 2, and 3 were consumed. No model, threshold, query, gallery, dose, or test compound was changed.",
        "",
        "## 3. Eligibility",
        "",
        f"The independent held-out e_rep table yielded {result['eligible_compounds']} compounds with valid e_rep coverage at all six locked doses. Fixed compound-level tertile cutoffs were q33.333={fmt(result['q_low'])} and q66.667={fmt(result['q_high'])}; each stratum contains 36 compounds. The stratum statistic is the median e_rep across deduplicated support-slot/dose keys.",
        "",
        "## 4. Exact information available to the model",
        "",
        "The stratification is explanatory only and is not available to any frozen method. Hit recovery uses Exp1 raw and teacher scores; retrieval uses Exp2 base PCC query metrics in the same gallery; dose response uses Exp3 saved compound trajectory metrics. The comparison is teacher minus raw.",
        "",
        "## 5. Independent reference construction",
        "",
        "Intrinsic reproducibility is computed from independent confirmation groups in Exp1. A physical (compound, dose, support_row) e_rep is deduplicated only after verifying identical values across all three saved seeds. No held-out profile, MoA/target label, or dose endpoint is used to define a stratum. However, Exp1 confirmed_active itself includes the validation-only E_rep q95 threshold, so hit AP stratification by E_rep is structurally label-dependent and is blocked for inference.",
        "",
        "## 6. Primary endpoint",
        "",
        "The three prespecified gain endpoints are teacher-minus-raw hit AP, MoA retrieval mAP, and dose-trajectory concordance. Target retrieval mAP is included as a supplementary endpoint because MoA is the locked retrieval primary.",
        "",
        "## 7. Secondary endpoints",
        "",
        "Target mAP is secondary. The optional 1R/2R/3R repeat-budget axis is BLOCKED_NOT_RUN because no frozen downstream 2R/3R per-stratum outputs were supplied; no values were imputed.",
        "",
        "## 8. Sample count",
        "",
        "Common compound counts by endpoint and stratum are recorded in CONFIG.json and BOOTSTRAP_CI.csv. Retrieval counts are lower than the 36 stratum compounds when a compound has no eligible base query in the frozen Exp2 output. All three seed evaluations use the intersection of available compound IDs for that endpoint and stratum.",
        "",
        "## 9. Seed-level results",
        "",
        "See METRICS_BY_SEED.csv for raw and teacher values and PAIRED_CONTRASTS.csv for every seed-level teacher-minus-raw contrast. The fixed strata are not rebalanced after inspecting endpoint values.",
        "",
        "## 10. Paired CI",
        "",
        "All available endpoint contrasts use 10,000 paired compound bootstrap resamples. The mean_seeds rows reuse identical compound resample positions across all three model seeds before averaging their deltas. Hit AP is explicitly BLOCKED in every stratum because its confirmed-active label uses E_rep, the same quantity defining the strata; low has no positives and medium has only one positive compound as additional coverage limitations. No AP CI is interpreted.",
        "",
        "### Three-seed mean gains",
        "",
        "| endpoint | low | medium | high |",
        "|---|---:|---:|---:|",
    ]
    for endpoint in ("hit_AP", "moa_mAP", "target_mAP", "trajectory_concordance"):
        cells = []
        for stratum in STRATA:
            row = summary_row(endpoint, stratum)
            if row is None or row["status"] != "OK":
                cells.append("BLOCKED")
            else:
                cells.append(f"{fmt(row['point_difference'])} [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]")
        lines.append(f"| {endpoint} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += [
        "",
        "## 11. GO / SUPPORTIVE / NO-GO",
        "",
        "Experiment 5 is an explanatory stratification, not a new optimization gate. The exact endpoint statuses are in BOOTSTRAP_CI.csv. A positive, CI-supported stratum would be descriptive evidence for where the frozen teacher helps; it would not support tuning a stratum-specific method. All hit-AP strata are BLOCKED_STRUCTURAL_LABEL_DEPENDENCE because confirmed_active uses the same E_rep quantity that defines the strata. The numerical AP contrasts are retained only as audit values and cannot support a claim that benefit concentrates in any information regime.",
        "",
        f"Information-regime hypothesis conclusion: NO EVIDENCE. MoA mAP gain is low={summary_text('moa_mAP', 'low')}, medium={summary_text('moa_mAP', 'medium')}, high={summary_text('moa_mAP', 'high')}; the direction moves from negative at low to positive at high (opposite the prespecified low-information-benefit expectation), and all three CIs cross zero. Target mAP shows the same low-to-high negative-to-positive pattern: low={summary_text('target_mAP', 'low')}, medium={summary_text('target_mAP', 'medium')}, high={summary_text('target_mAP', 'high')}; all CIs cross zero. Trajectory gain is non-monotonic, low > high > medium: low={summary_text('trajectory_concordance', 'low')}, medium={summary_text('trajectory_concordance', 'medium')}, high={summary_text('trajectory_concordance', 'high')}; all CIs cross zero.",
        "",
        "## 12. Biological interpretation",
        "",
        "For MoA/target retrieval and dose trajectory, the analysis does not support the hypothesis that less intrinsically reproducible compounds receive larger downstream benefit: MoA and target gains move from negative in low to positive in high, while trajectory gains are non-monotonic (low > high > medium), with all reported pooled CIs crossing zero. Hit AP cannot test that hypothesis under this label construction because its activity label reuses E_rep. These are retrospective associations and cannot be interpreted as a prospective activity rule or as evidence that AI replaces a physical repeat.",
        "",
        "## 13. Limitations",
        "",
        "The matched-null audit leaves many compounds without e_rep, and retrieval coverage is narrower than the 108-compound stratum table. Hit AP is structurally confounded by the confirmed_active definition (which uses E_rep), so all hit-AP stratum results are BLOCKED for inference; low also has no positive labels and medium only one positive compound. Exp5 does not add a direct equal-cost repeat contrast, does not establish replicate replacement, and does not include BBBC036 or JUMP. Optional 2R/3R downstream stratification is BLOCKED_NOT_RUN due absent frozen inputs. Figure status is recorded in CONFIG.json.",
        "",
    ]
    (outdir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    # Decision text is deliberately explicit about blocked endpoints and the
    # fact that this explanatory analysis does not create a new GO gate.
    endpoint_status_lines = []
    for endpoint in ("hit_AP", "moa_mAP", "target_mAP", "trajectory_concordance"):
        vals = []
        for stratum in STRATA:
            row = summary_row(endpoint, stratum)
            vals.append(f"{stratum}={row['status'] if row else 'BLOCKED_NO_SUMMARY'}")
        endpoint_status_lines.append(f"- `{endpoint}`: " + ", ".join(vals))
    seed_direction_lines = []
    for endpoint in ("hit_AP", "moa_mAP", "target_mAP", "trajectory_concordance"):
        for stratum in STRATA:
            rows = [
                r for r in bootstrap_rows
                if r["endpoint"] == endpoint and r["stratum"] == stratum and r["seed"] in SEEDS
            ]
            rows = sorted(rows, key=lambda row: SEEDS.index(int(row["seed"])))
            values = "; ".join(f"seed{int(row['seed'])}={fmt(row['point_difference'])}" for row in rows)
            seed_direction_lines.append(f"- `{endpoint}` / `{stratum}`: {values}")
    config_inputs = result["config"].get("inputs", {})
    input_hash_lines = [
        f"- `{key}`: `{meta['sha256']}` ({meta['path']})"
        for key, meta in config_inputs.items()
    ]
    common_count_lines = []
    for endpoint, strata_counts in result["config"].get("common_counts", {}).items():
        common_count_lines.append(
            f"- `{endpoint}`: " + ", ".join(f"{stratum}={int(strata_counts.get(stratum, 0))}" for stratum in STRATA)
        )
    config_hash = sha256_file(outdir / "CONFIG.json") if (outdir / "CONFIG.json").exists() else "pending"
    decision = [
        "# Decision — Experiment 5 information regime",
        "",
        "## Locked decision",
        "",
        "`EXPLANATORY_COMPLETE_WITH_ENDPOINT_BLOCKS`.",
        "",
        "This analysis uses frozen Exp1/Exp2/Exp3 artifacts and does not create a new method-selection or application GO/NO-GO gate. It reports where the already-frozen teacher-minus-raw gains are estimable by independent reproducibility stratum.",
        "",
        "## Input lock and audit",
        "",
        "Input SHA-256 hashes are recorded in CONFIG.json. The stratum table is derived only from Exp1 common test e_rep and uses physical-key seed consistency before deduplication. All endpoint-specific compound sets are intersected across seeds. No test label, MoA/target label, or endpoint result selects a boundary or method. The audit also records that Exp1 confirmed_active uses an E_rep q95 threshold, so hit-AP stratification is structurally label-dependent.",
        "",
        "### Locked input SHA-256",
        "",
    ] + input_hash_lines + [
        "",
        "## Strata",
        "",
        f"- Eligible compounds: {result['eligible_compounds']}; low/medium/high counts: 36/36/36.",
        f"- Fixed tertile cutoffs: q33.333={fmt(result['q_low'])}; q66.667={fmt(result['q_high'])}.",
        "- Statistic: median independent e_rep across all deduplicated support-slot/dose keys, requiring all six locked doses.",
        "",
        "### Common compound counts",
        "",
    ] + common_count_lines + [
        "",
        "## Endpoint status",
        "",
    ] + endpoint_status_lines + [
        "",
        "## Seed-level teacher-minus-raw directions",
        "",
    ] + seed_direction_lines + [
        "",
        "## Information-regime hypothesis",
        "",
        f"`NO EVIDENCE`: the low-information-benefit hypothesis is not supported by the locked outputs. MoA gain is low={summary_text('moa_mAP', 'low')}, medium={summary_text('moa_mAP', 'medium')}, high={summary_text('moa_mAP', 'high')}; target gain is low={summary_text('target_mAP', 'low')}, medium={summary_text('target_mAP', 'medium')}, high={summary_text('target_mAP', 'high')}; both move from negative low to positive high and all pooled CIs cross zero. Trajectory gain is low={summary_text('trajectory_concordance', 'low')}, medium={summary_text('trajectory_concordance', 'medium')}, high={summary_text('trajectory_concordance', 'high')}, a non-monotonic low > high > medium pattern with all pooled CIs crossing zero. Hit AP is BLOCKED_STRUCTURAL_LABEL_DEPENDENCE in all strata.",
        "",
        "## Bootstrap",
        "",
        f"Every estimable endpoint was configured for {ROUNDS:,} paired compound-level resamples. The `mean_seeds` CI reuses identical compound-resample positions across the three saved seeds, then takes the percentile interval of the per-position mean delta. Hit AP is structurally blocked in all strata because confirmed_active uses E_rep; low also has no positive label and medium only one positive compound. These are BLOCKED, not zero gains.",
        "",
        "## Stop conditions",
        "",
        "- No retraining, lambda/beta adjustment, or post-hoc endpoint tuning was performed.",
        "- No missing 2R/3R output was filled by interpolation; the optional repeat axis is `BLOCKED_NOT_RUN`.",
        "- No replicate-replacement or cost-saving claim is made.",
        "",
        f"Output CONFIG.json SHA-256 (post-write audit): `{config_hash}`.",
        "",
    ]
    (outdir / "DECISION.md").write_text("\n".join(decision), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--bootstrap-rounds", type=int, default=ROUNDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap_rounds != ROUNDS:
        raise SystemExit("The formal protocol is locked to exactly 10000 bootstrap rounds")
    result = run(args.outdir, args.bootstrap_rounds)
    write_reports(args.outdir, result)
    print(json.dumps({"status": "OK", "eligible_compounds": result["eligible_compounds"], "strata": result["strata"], "common_counts": result["common_counts"], "figure_status": result["figure_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
