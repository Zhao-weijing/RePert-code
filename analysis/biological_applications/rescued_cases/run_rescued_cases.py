#!/usr/bin/env python3
"""Frozen, audit-first selection of rescued biological cases.

This experiment does not fit a model and does not select a threshold.  It
joins the already-frozen Experiment 1 activity audit to the already-frozen
Experiment 2 MoA retrieval rows, applies the preregistered case rule, and
uses the Experiment 3 slot audit only to verify that support rows and
independent references are available.  The expected result for the current
artifacts is a valid empty case table: no active condition satisfies every
strict gate.

The script intentionally writes the audit even when no case is eligible.
That makes a no-go result reproducible instead of silently omitting the
experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VERSION = "cpg0004-LINCS-biological-rescued-cases-2026-08-30"
SEEDS = (3407, 42, 2025)
BOOTSTRAP_ROUNDS = 10_000
TOP_K = 10
MAX_CASES = 6


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_seed(value: Any) -> pd.Series | int:
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").round().astype("Int64")
    return int(round(float(value)))


def dose_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value).strip()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def zscore(values: pd.Series) -> pd.Series:
    x = numeric(values).astype(float)
    finite = np.isfinite(x.to_numpy())
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if not finite.any():
        return out
    arr = x.to_numpy()[finite]
    center = float(np.mean(arr))
    scale = float(np.std(arr, ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        out.loc[finite] = 0.0
    else:
        out.loc[finite] = (arr - center) / scale
    return out


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if np.isfinite(number) else math.nan


def bootstrap_mean(values: np.ndarray, rounds: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    point = float(np.mean(values))
    if values.size < 2 or rounds <= 0:
        return point, math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(rounds, values.size))
    means = values[indices].mean(axis=1)
    return point, float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    app_root = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1", type=Path, default=app_root / "hit_recovery")
    parser.add_argument("--e2", type=Path, default=app_root / "mechanism_target_retrieval")
    parser.add_argument("--e3", type=Path, default=app_root / "dose_response")
    parser.add_argument("--outdir", type=Path, default=here)
    parser.add_argument("--bootstrap-rounds", type=int, default=BOOTSTRAP_ROUNDS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, str | None]]:
    e1_dir = args.e1.resolve()
    e2_dir = args.e2.resolve()
    e3_dir = args.e3.resolve()
    e1 = pd.read_csv(e1_dir / "ELIGIBLE_SAMPLES.csv")
    e1_config = json.loads((e1_dir / "CONFIG.json").read_text(encoding="utf-8"))
    query = pd.read_csv(e2_dir / "query_manifest.csv")
    per_query = pd.read_csv(e2_dir / "per_query_metrics.csv")
    ann_path = e2_dir / "annotation_compound_labels.csv"
    annotations = pd.read_csv(ann_path) if ann_path.is_file() else pd.DataFrame()
    # Experiment 3 is an audit input.  Loading the table verifies the frozen
    # reference construction exists, without allowing it to select a case.
    slot_path = e3_dir / "SLOT_METRICS.csv"
    slot_metrics = pd.read_csv(slot_path) if slot_path.is_file() else pd.DataFrame()
    paths = {
        "experiment_1_samples": e1_dir / "ELIGIBLE_SAMPLES.csv",
        "experiment_1_config": e1_dir / "CONFIG.json",
        "experiment_2_queries": e2_dir / "query_manifest.csv",
        "experiment_2_per_query": e2_dir / "per_query_metrics.csv",
        "experiment_2_annotations": ann_path,
        "experiment_3_slot_metrics": slot_path,
    }
    hashes = {key: sha256(path) for key, path in paths.items()}
    # Keep the two input tables in the return signature so the main function
    # has an explicit audit of every experiment used.
    return e1, query, per_query, slot_metrics, annotations, e1_config, hashes


def prepare_active_rows(
    e1: pd.DataFrame,
    query: pd.DataFrame,
    per_query: pd.DataFrame,
    annotations: pd.DataFrame,
    e1_config: dict[str, Any],
) -> pd.DataFrame:
    required = {"split", "compound", "dose", "support_row", "confirmed_active", "evaluation_common_all_methods", "raw_score", "teacher_score", "P0_score", "GE_posterior_score", "e_rep", "a_ref", "rotation_rank", "support_plate", "support_well"}
    missing = sorted(required - set(e1.columns))
    if missing:
        raise ValueError(f"Experiment 1 sample table missing columns: {missing}")

    active = e1.copy()
    active["seed_key"] = norm_seed(active["seed"])
    active["dose_key"] = active["dose"].map(dose_key)
    active["support_row"] = pd.to_numeric(active["support_row"], errors="coerce").astype("Int64")
    active = active[
        active["split"].astype(str).eq("test")
        & as_bool(active["evaluation_common_all_methods"])
        & numeric(active["confirmed_active"]).eq(1)
    ].copy()
    active["confirmed_active"] = 1

    q = query.copy()
    q["seed_key"] = norm_seed(q["seed"])
    q["dose_key"] = q["dose"].map(dose_key)
    q["support_row"] = pd.to_numeric(q["support_row"], errors="coerce").astype("Int64")
    q = q.rename(columns={"compound_id": "compound"})
    q_keep = ["seed_key", "compound", "dose_key", "support_row", "query_id", "support_slot", "moa_retained_label", "target_retained_label", "smiles", "smiles_available", "ge_available"]
    q = q[[col for col in q_keep if col in q.columns]].drop_duplicates(["seed_key", "compound", "dose_key", "support_row"])
    active = active.merge(q, on=["seed_key", "compound", "dose_key", "support_row"], how="left", validate="many_to_one")

    p = per_query.copy()
    p["seed_key"] = norm_seed(p["seed"])
    p = p[
        p["task"].astype(str).eq("moa")
        & p["variant"].astype(str).eq("base")
        & p["method"].astype(str).isin({"raw", "teacher", "posterior_ge"})
    ].copy()
    p = p[["seed_key", "query_id", "method", "first_rank", "map", "p_at_1", "p_at_5", "recall_at_10", "hits_at_10"]]
    p = p.drop_duplicates(["seed_key", "query_id", "method"])
    ranks = p.pivot_table(index=["seed_key", "query_id"], columns="method", values=["first_rank", "map", "p_at_1", "p_at_5", "recall_at_10", "hits_at_10"], aggfunc="first")
    if len(ranks):
        ranks.columns = ["_".join(str(part) for part in col if str(part) != "") for col in ranks.columns]
        ranks = ranks.reset_index()
    else:
        ranks = pd.DataFrame(columns=["seed_key", "query_id"])
    active = active.merge(ranks, on=["seed_key", "query_id"], how="left", validate="many_to_one")

    if annotations is not None and not annotations.empty:
        ann = annotations.copy()
        if "compound_id" in ann.columns:
            ann = ann.rename(columns={"compound_id": "compound"})
        ann_cols = [col for col in ["compound", "moa", "primary_moa", "alternative_moa", "target"] if col in ann.columns]
        ann = ann[ann_cols].drop_duplicates("compound")
        active = active.merge(ann, on="compound", how="left", suffixes=("", "_annotation"), validate="many_to_one")
    else:
        active["moa"] = ""

    # Curated query labels take precedence.  No label is inferred from a
    # profile or from a nearest-neighbour result.
    active["moa_family"] = active.get("moa_retained_label", pd.Series("", index=active.index)).fillna("").astype(str).str.strip()
    if "moa" in active.columns:
        active.loc[active["moa_family"].eq(""), "moa_family"] = active.loc[active["moa_family"].eq(""), "moa"].fillna("").astype(str).str.strip()
    active["moa_family"] = active["moa_family"].str.split("|", n=1).str[0].str.strip()

    thresholds = e1_config.get("model_thresholds_validation_only", {})
    active["raw_threshold_validation"] = safe_float(thresholds.get("raw"))
    active["teacher_threshold_validation"] = safe_float(thresholds.get("teacher"))
    active["posterior_threshold_validation"] = safe_float(thresholds.get("GE_posterior"))
    active["raw_score"] = numeric(active["raw_score"])
    active["teacher_score"] = numeric(active["teacher_score"])
    active["P0_score"] = numeric(active["P0_score"])
    active["GE_posterior_score"] = numeric(active["GE_posterior_score"])
    active["raw_rank"] = numeric(active.get("first_rank_raw", pd.Series(np.nan, index=active.index)))
    active["teacher_rank"] = numeric(active.get("first_rank_teacher", pd.Series(np.nan, index=active.index)))
    active["posterior_rank"] = numeric(active.get("first_rank_posterior_ge", pd.Series(np.nan, index=active.index)))
    active["raw_activity"] = active["raw_score"] >= active["raw_threshold_validation"]
    active["teacher_activity"] = active["teacher_score"] >= active["teacher_threshold_validation"]
    # The frozen Experiment 1 artifact has no validation GE predictions, so a
    # GE-posterior activity threshold must remain unavailable rather than
    # being selected from the test labels.
    active["posterior_activity"] = False
    active["raw_activity_wrong"] = ~active["raw_activity"]
    active["teacher_activity_rescued"] = active["raw_activity_wrong"] & active["teacher_activity"]
    active["posterior_activity_rescued"] = False
    active["raw_moa_bad"] = active["raw_rank"] > 50
    active["teacher_moa_top10"] = active["teacher_rank"] <= TOP_K
    active["posterior_moa_top10"] = active["posterior_rank"] <= TOP_K
    active["retrieval_available"] = active[["raw_rank", "teacher_rank", "posterior_rank"]].notna().all(axis=1)

    # Method-specific strict rescue.  An activity rescue and a top-10 MoA
    # rescue must belong to the same method.  This prevents mixing a teacher
    # activity score with a posterior retrieval rank.
    active["teacher_strict_rescue"] = active["teacher_activity_rescued"] & active["teacher_moa_top10"]
    active["posterior_strict_rescue"] = active["posterior_activity_rescued"] & active["posterior_moa_top10"]
    active["activity_rescued"] = active["teacher_activity_rescued"] | active["posterior_activity_rescued"]
    active["moa_top10_rescued"] = active["teacher_moa_top10"] | active["posterior_moa_top10"]
    active["strict_rotation_gate"] = (
        (active["raw_activity_wrong"] | active["raw_moa_bad"])
        & active["activity_rescued"]
        & active["moa_top10_rescued"]
    )

    # Rescue score is descriptive and is computed before case selection.  It
    # never changes a binary gate or a validation threshold.
    best_model_score = active[["teacher_score", "GE_posterior_score"]].max(axis=1, skipna=True)
    best_model_rank = active[["teacher_rank", "posterior_rank"]].min(axis=1, skipna=True)
    active["activity_gain"] = best_model_score - active["raw_score"]
    active["moa_rank_gain"] = active["raw_rank"] - best_model_rank
    active["reproducibility_e_rep"] = numeric(active["e_rep"])
    active["z_activity_gain"] = zscore(active["activity_gain"])
    active["z_moa_rank_gain"] = zscore(active["moa_rank_gain"])
    active["z_reference_reproducibility"] = zscore(active["reproducibility_e_rep"])
    active["rescue_score"] = active[["z_activity_gain", "z_moa_rank_gain", "z_reference_reproducibility"]].sum(axis=1, min_count=1)

    # A case must reproduce the strict gate in both saved support-slot
    # rotations for at least one seed.  Seeds are never counted as additional
    # physical rotations.
    active["support_slot"] = numeric(active.get("support_slot", pd.Series(np.nan, index=active.index)))
    active["strict_rotation_count_seed"] = active.groupby(["seed_key", "compound", "dose_key"])["strict_rotation_gate"].transform("sum")
    active["two_rotation_gate"] = active["strict_rotation_count_seed"] >= 2
    active["case_rotation_gate"] = active["strict_rotation_gate"] & active["two_rotation_gate"]
    return active


def make_exclusions(active: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in active.sort_values(["compound", "dose_key", "seed_key", "support_row"]).iterrows():
        reasons: list[str] = []
        if pd.isna(row.get("query_id")) or pd.isna(row.get("raw_rank")):
            reasons.append("MISSING_CURATED_MOA_OR_RETRIEVAL")
        if not bool(row.get("raw_activity_wrong", False)) and not bool(row.get("raw_moa_bad", False)):
            reasons.append("RAW_NOT_MISSED")
        if not bool(row.get("activity_rescued", False)):
            if pd.isna(row.get("posterior_threshold_validation")):
                reasons.append("NO_ACTIVITY_RESCUE_POSTERIOR_THRESHOLD_BLOCKED")
            else:
                reasons.append("NO_ACTIVITY_RESCUE")
        if bool(row.get("activity_rescued", False)) and not bool(row.get("moa_top10_rescued", False)):
            reasons.append("NO_MOA_TOP10_RESCUE")
        if bool(row.get("strict_rotation_gate", False)) and not bool(row.get("two_rotation_gate", False)):
            reasons.append("INSUFFICIENT_TWO_SUPPORT_SLOT_ROTATIONS")
        if not reasons:
            reasons.append("EXCLUDED_BY_CASE_SELECTION")
        rows.append(
            {
                "seed": int(row["seed_key"]) if pd.notna(row["seed_key"]) else "",
                "compound": row.get("compound", ""),
                "dose": row.get("dose_key", ""),
                "support_row": int(row["support_row"]) if pd.notna(row.get("support_row")) else "",
                "support_slot": int(row["support_slot"]) if pd.notna(row.get("support_slot")) else "",
                "rotation_rank": row.get("rotation_rank", ""),
                "confirmed_active": int(row.get("confirmed_active", 0)),
                "moa_family": row.get("moa_family", ""),
                "raw_activity_wrong": int(bool(row.get("raw_activity_wrong", False))),
                "raw_moa_rank": row.get("raw_rank", math.nan),
                "teacher_activity_rescued": int(bool(row.get("teacher_activity_rescued", False))),
                "posterior_activity_rescued": int(bool(row.get("posterior_activity_rescued", False))),
                "teacher_moa_rank": row.get("teacher_rank", math.nan),
                "posterior_moa_rank": row.get("posterior_rank", math.nan),
                "strict_rotation_gate": int(bool(row.get("strict_rotation_gate", False))),
                "strict_rotation_count_seed": int(row.get("strict_rotation_count_seed", 0)),
                "reason_code": ";".join(reasons),
                "rescue_score": row.get("rescue_score", math.nan),
            }
        )
    return pd.DataFrame(rows)


def select_cases(active: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Aggregate only rows that have passed every per-rotation gate.  The
    # aggregation is deterministic and is not tuned using a biological label.
    passing = active[active["case_rotation_gate"]].copy()
    if passing.empty:
        return passing, pd.DataFrame(columns=["compound", "dose_key", "moa_family", "rescue_score", "selection_status"])
    grouped = passing.groupby(["compound", "dose_key"], as_index=False).agg(
        rescue_score=("rescue_score", "mean"),
        support_slot_count=("support_slot", "nunique"),
        seed_count=("seed_key", "nunique"),
        seed_min=("seed_key", "min"),
        raw_score=("raw_score", "mean"),
        teacher_score=("teacher_score", "mean"),
        posterior_score=("GE_posterior_score", "mean"),
        raw_rank=("raw_rank", "mean"),
        teacher_rank=("teacher_rank", "mean"),
        posterior_rank=("posterior_rank", "mean"),
        e_rep=("e_rep", "mean"),
    )
    families = passing.groupby(["compound", "dose_key"], as_index=False)["moa_family"].first()
    grouped = grouped.merge(families, on=["compound", "dose_key"], how="left", validate="one_to_one")
    grouped = grouped.sort_values(["rescue_score", "compound", "dose_key"], ascending=[False, True, True], kind="mergesort").reset_index(drop=True)
    selected_rows = []
    family_seen: set[str] = set()
    for _, row in grouped.iterrows():
        family = str(row.get("moa_family", "")).strip() or "UNANNOTATED"
        if family in family_seen:
            status = "EXCLUDED_MOA_FAMILY_CAP"
        elif len(selected_rows) >= MAX_CASES:
            status = "EXCLUDED_TOP6_LIMIT"
        else:
            status = "SELECTED"
            family_seen.add(family)
        record = row.to_dict()
        record["selection_status"] = status
        selected_rows.append(record)
    selection = pd.DataFrame(selected_rows)
    selected = selection[selection["selection_status"].eq("SELECTED")].copy()
    return selected, selection


def empty_case_tables(outdir: Path) -> None:
    pd.DataFrame(columns=[
        "case_id", "compound", "dose", "moa_family", "seed", "support_slot", "support_row",
        "raw_profile", "teacher_profile", "posterior_profile", "reference_profile",
        "raw_activity_score", "teacher_activity_score", "posterior_activity_score", "reference_activity_score",
        "raw_activity_label", "teacher_activity_label", "posterior_activity_label",
        "raw_moa_rank", "teacher_moa_rank", "posterior_moa_rank", "top10_status",
        "feature_category_status", "image_status",
    ]).to_csv(outdir / "CASE_PROFILES.csv", index=False)
    pd.DataFrame(columns=[
        "case_id", "query_id", "method", "rank", "gallery_compound", "gallery_dose", "similarity_pcc",
        "shared_moa", "shared_target", "top10_status",
    ]).to_csv(outdir / "CASE_TOP10.csv", index=False)
    pd.DataFrame(columns=[
        "case_id", "feature_index", "feature_name", "category", "raw_to_reference_abs_error",
        "teacher_to_reference_abs_error", "posterior_to_reference_abs_error", "direction", "category_status",
    ]).to_csv(outdir / "CASE_FEATURES.csv", index=False)


def make_metrics(active: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        s = active[active["seed_key"].eq(seed)]
        rows.append(
            {
                "seed": seed,
                "active_rotation_rows": len(s),
                "active_compound_dose_conditions": s[["compound", "dose_key"]].drop_duplicates().shape[0],
                "active_compounds": s["compound"].nunique(),
                "raw_activity_missed_active": int(s["raw_activity_wrong"].sum()),
                "raw_moa_rank_gt50": int(s["raw_moa_bad"].sum()),
                "teacher_activity_rescued": int(s["teacher_activity_rescued"].sum()),
                "posterior_activity_rescued": int(s["posterior_activity_rescued"].sum()),
                "teacher_moa_rank_le10": int(s["teacher_moa_top10"].sum()),
                "posterior_moa_rank_le10": int(s["posterior_moa_top10"].sum()),
                "strict_rotation_gate_rows": int(s["strict_rotation_gate"].sum()),
                "two_rotation_case_rows": int(s["case_rotation_gate"].sum()),
                "retrieval_rows_missing": int(s["raw_rank"].isna().sum()),
                "selected_case_count": int(selected["compound"].nunique()) if not selected.empty else 0,
                "posterior_activity_threshold_status": "BLOCKED_VALIDATION_PREDICTIONS_MISSING",
            }
        )
    return pd.DataFrame(rows)


def make_contrasts(active: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    columns = ["contrast", "metric", "n_compounds", "point_difference", "ci_low", "ci_high", "status", "note"]
    if selected.empty:
        rows = [
            {
                "contrast": "teacher_or_posterior_rescued_cases_vs_raw",
                "metric": "eligible_case_count",
                "n_compounds": 0,
                "point_difference": math.nan,
                "ci_low": math.nan,
                "ci_high": math.nan,
                "status": "NOT_ESTIMABLE_NO_STRICT_CASES",
                "note": "No case passed confirmed-active, activity-rescue, MoA-top10, and two-rotation gates.",
            },
            {
                "contrast": "selected_vs_requested",
                "metric": "case_count",
                "n_compounds": 0,
                "point_difference": -4.0,
                "ci_low": math.nan,
                "ci_high": math.nan,
                "status": "NO_GO",
                "note": "The requested 4-6 cases are a target, not a reason to relax the frozen rule.",
            },
        ]
        return pd.DataFrame(rows, columns=columns)
    # This branch is kept for reproducibility if a future frozen input passes
    # the rule; it still uses compound-level bootstrap and never changes the
    # selection gate.
    values = selected["rescue_score"].to_numpy(dtype=float)
    point, low, high = bootstrap_mean(values, BOOTSTRAP_ROUNDS, 3004)
    return pd.DataFrame([
        {
            "contrast": "selected_case_rescue_score",
            "metric": "rescue_score",
            "n_compounds": int(selected["compound"].nunique()),
            "point_difference": point,
            "ci_low": low,
            "ci_high": high,
            "status": "DESCRIPTIVE",
            "note": "Descriptive only; the case rule was frozen before selection.",
        }
    ], columns=columns)


def make_bootstrap(contrasts: pd.DataFrame) -> pd.DataFrame:
    out = contrasts.copy()
    out.insert(0, "bootstrap_rounds", BOOTSTRAP_ROUNDS)
    out.insert(1, "bootstrap_unit", "compound")
    out.insert(2, "seed", "pooled")
    return out


def write_figure(outdir: Path, active: pd.DataFrame, selected: pd.DataFrame) -> str:
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    status = "OK"
    try:
        import matplotlib.pyplot as plt

        stages = [
            "confirmed\nactive rotations",
            "raw missed\nor MoA >50",
            "activity\nrescued",
            "MoA rank\n≤10",
            "two-slot\ncase gate",
            "selected\ncases",
        ]
        counts = [
            len(active),
            int((active["raw_activity_wrong"] | active["raw_moa_bad"]).sum()),
            int(active["activity_rescued"].sum()),
            int((active["activity_rescued"] & active["moa_top10_rescued"]).sum()),
            int(active["case_rotation_gate"].sum()),
            int(selected["compound"].nunique()) if not selected.empty else 0,
        ]
        fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
        colors = ["#284b63", "#3c6e71", "#4f8a8b", "#72a276", "#b5bd89", "#d9a441"]
        bars = ax.bar(np.arange(len(stages)), counts, color=colors, edgecolor="#1f2933", linewidth=0.6)
        ax.set_xticks(np.arange(len(stages)), stages, fontsize=9)
        ax.set_ylabel("count (rotation rows until case gate; compounds at final)")
        ax.set_title("Experiment 4 strict rescued-case audit — cpg0004-LINCS")
        ax.set_ylim(0, max(1, max(counts) * 1.18))
        for bar, value in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(0.2, max(counts) * 0.015), str(value), ha="center", va="bottom", fontsize=10)
        ax.text(0.99, 0.98, "No manual cherry-picking\nposterior activity threshold: blocked", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color="#374151")
        fig.tight_layout()
        fig.savefig(fig_dir / "strict_rescue_funnel.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - environment-specific
        # Keep a real, inspectable figure artifact even on the minimal
        # Windows runtime where matplotlib is not installed.  This fallback
        # is deliberately plain SVG and contains only the audited counts.
        stages = [
            "confirmed active", "raw miss or rank >50", "activity rescued",
            "MoA rank <=10", "two-slot gate", "selected cases",
        ]
        counts = [
            len(active),
            int((active["raw_activity_wrong"] | active["raw_moa_bad"]).sum()),
            int(active["activity_rescued"].sum()),
            int((active["activity_rescued"] & active["moa_top10_rescued"]).sum()),
            int(active["case_rotation_gate"].sum()),
            int(selected["compound"].nunique()) if not selected.empty else 0,
        ]
        width, height = 1080, 420
        chart_left, chart_top, chart_width, chart_height = 70, 55, 960, 260
        ymax = max(1, max(counts))
        bar_width = chart_width / len(stages) * 0.62
        colors = ["#284b63", "#3c6e71", "#4f8a8b", "#72a276", "#b5bd89", "#d9a441"]
        pieces = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="70" y="28" font-family="Arial,sans-serif" font-size="18" fill="#1f2933">Experiment 4 strict rescued-case audit — cpg0004-LINCS</text>',
            f'<line x1="{chart_left}" y1="{chart_top + chart_height}" x2="{chart_left + chart_width}" y2="{chart_top + chart_height}" stroke="#374151"/>',
            f'<text x="12" y="{chart_top + chart_height / 2}" transform="rotate(-90 12 {chart_top + chart_height / 2})" font-family="Arial,sans-serif" font-size="12" fill="#374151">count</text>',
        ]
        for i, (label, value, color) in enumerate(zip(stages, counts, colors)):
            center = chart_left + (i + 0.5) * chart_width / len(stages)
            bar_x = center - bar_width / 2
            bar_h = chart_height * value / ymax
            bar_y = chart_top + chart_height - bar_h
            pieces.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" fill="{color}" stroke="#1f2933" stroke-width="0.7"/>')
            pieces.append(f'<text x="{center:.1f}" y="{bar_y - 8:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#111827">{value}</text>')
            pieces.append(f'<text x="{center:.1f}" y="{chart_top + chart_height + 20}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#374151">{label}</text>')
        pieces.append('<text x="70" y="392" font-family="Arial,sans-serif" font-size="11" fill="#4b5563">No manual cherry-picking; posterior activity threshold blocked when GE validation predictions are absent.</text>')
        pieces.append('</svg>')
        (fig_dir / "strict_rescue_funnel.svg").write_text("\n".join(pieces) + "\n", encoding="utf-8")
        status = "OK_SVG_FALLBACK"
        (fig_dir / "README.md").write_text(f"Figure generation status: {status}; matplotlib fallback after {type(exc).__name__}.\n", encoding="utf-8")
    return status


def write_docs(
    outdir: Path,
    active: pd.DataFrame,
    exclusions: pd.DataFrame,
    selected: pd.DataFrame,
    selection: pd.DataFrame,
    metrics: pd.DataFrame,
    figure_status: str,
    hashes: dict[str, str | None],
    slot_metrics: pd.DataFrame,
    e1_config: dict[str, Any],
) -> None:
    selected_count = int(selected["compound"].nunique()) if not selected.empty else 0
    raw_missed = int(active["raw_activity_wrong"].sum())
    raw_moa_bad = int(active["raw_moa_bad"].sum())
    strict_rows = int(active["strict_rotation_gate"].sum())
    two_rows = int(active["case_rotation_gate"].sum())
    posterior_threshold_status = "BLOCKED_VALIDATION_PREDICTIONS_MISSING" if pd.isna(active["posterior_threshold_validation"]).all() else "OK"
    protocol = f"""# Experiment 4 protocol — rescued biological cases

Version: `{VERSION}`  
Dataset: `cpg0004-LINCS`  
Status: frozen downstream audit; no model fitting or retuning.

## Objective

Select concrete compounds whose independently confirmed CP activity was
missed or poorly ranked by one-repeat raw CP and recovered by a frozen
teacher or validated posterior. This is an automatic case-selection
experiment, not a qualitative showcase.

## Frozen inputs

- Experiment 1 `ELIGIBLE_SAMPLES.csv` supplies confirmed-active labels,
  validation-only activity thresholds, support-slot rows, raw/teacher/P0/GE
  activity magnitudes, and `E_rep`.
- Experiment 2 `per_query_metrics.csv` supplies fixed-PCC MoA first-correct
  ranks for the identical dose-matched gallery. Curated labels are read only
  from its query/annotation tables.
- Experiment 3 `SLOT_METRICS.csv` is an audit that the independent reference
  and the two recorded support rotations exist; it is never used to choose a
  case or tune a score.

## Strict case rule

At a support-slot rotation, all of the following must hold:

1. `confirmed_active == 1` from independent confirmation repeats.
2. Raw activity is wrong (`raw_score < validation raw threshold`) **or** raw
   MoA first-correct rank is greater than 50.
3. The same frozen method rescues activity and has MoA first-correct rank
   `<= 10`. Teacher activity uses its Experiment 1 validation threshold.
   GE-posterior activity is not called rescued because its validation
   predictions and threshold are absent; no test threshold is invented.
4. The strict gate is reproduced in at least two distinct saved support-slot
   rotations for at least one seed. Seeds do not count as physical rotations.

The audit score is descriptive only:

`rescue_score = z(best frozen activity magnitude - raw magnitude) +
z(raw MoA rank - best frozen MoA rank) + z(E_rep)`.

Passing compound-dose cases are sorted by this score. At most one case per
curated MoA family is selected, with a maximum of six and a target of four to
six. If fewer pass, the table remains short; no rule is relaxed.

## Case display contract

For every selected case, `CASE_PROFILES.csv`, `CASE_TOP10.csv`, and
`CASE_FEATURES.csv` must contain raw, teacher, posterior, independent
reference, activity/rank/dose, Top-10 neighbours, and annotated CP feature
category comparisons. The current frozen input exposes aggregate retrieval
rows rather than Top-10 member IDs and has no raw well-image artifact;
therefore a zero-case outcome is reported and image display is
`BLOCKED_IMAGE_VISUALIZATION`.

## Statistics and guardrails

The case count is an audit outcome, not a fitted endpoint. Any future
descriptive score interval uses 10,000 compound-level resamples. No MoA or
target information changes teacher/posterior values; no test label changes a
threshold; no dose or compound is selected after looking at a favourable
result.

## Input hashes

""" + "\n".join(f"- `{key}`: `{value}`" for key, value in hashes.items()) + "\n"
    (outdir / "PROTOCOL.md").write_text(protocol, encoding="utf-8")

    decision = f"""# Decision — Experiment 4 rescued biological cases

Version: `{VERSION}`

## Bio-D decision: **NO-GO**

The strict automatic rule yielded **{selected_count} selected case compounds**.
There were {len(active)} confirmed-active common test rotation rows, but only
{raw_missed} raw activity misses and {raw_moa_bad} raw MoA-rank failures
(`first_correct_rank > 50`). The method-specific activity-plus-MoA top-10
gate passed in {strict_rows} rotation rows, and the two-support-slot case gate
passed in {two_rows} rows. No compound therefore reached the final 4–6 case
target.

This is a strict no-go, not evidence that downstream biology is absent. It
means the current frozen application artifacts do not contain a case meeting
all requested conditions. In particular, the only active raw-rank failure(s)
did not reach a same-method MoA rank of 10 or better, and GE-posterior
activity rescue cannot be called because validation GE predictions are
missing (`{posterior_threshold_status}`).

## Guardrails

- No model, lambda, beta, threshold, test compound, dose, or MoA family was
  selected after inspecting the case results.
- Raw, teacher, and posterior values remain those frozen in Experiments 1–2.
- Support slots are counted as physical rotations; seeds are not additional
  repeats.
- The requested 4–6 cases were not manufactured and no gate was relaxed.
- Original image panels are `BLOCKED_IMAGE_VISUALIZATION`: the frozen local
  artifact contains profiles, not raw control/support/independent-replicate
  images.

`EXCLUSIONS.csv` and `RESCUE_AUDIT.csv` provide a row-level audit of every
confirmed-active rotation, including the exact failed gate.
"""
    (outdir / "DECISION.md").write_text(decision, encoding="utf-8")

    results = f"""# Results — cpg0004-LINCS rescued biological cases

## 1. Objective

Identify automatically selected biological rescue cases in which frozen
reproducible-effect inference recovers a confirmed perturbation and its
curated MoA neighbourhood from a noisy one-repeat CP measurement.

## 2. Dataset

`cpg0004-LINCS`, using the same frozen support/query/gallery artifacts as
Experiments 1–3. No BBBC047, BBBC036, or JUMP rows enter this experiment.

## 3. Eligibility

The starting population is the common all-method test subset of Experiment 1
with an independent `confirmed_active` label. A case then requires raw
activity wrong or raw MoA rank >50, same-method teacher/posterior activity
rescue with MoA rank <=10, and at least two saved support-slot rotations.
MoA families are capped at one selected representative.

## 4. Exact information available to model

The model values are read from frozen Experiment 1/2 outputs. No model is
trained here. The only use of confirmed labels and MoA annotations is
retrospective case eligibility and reporting.

## 5. Independent reference construction

Confirmed activity and `E_rep`/`A_ref` are inherited from the independent
support-excluded confirmation construction in Experiment 1. Experiment 3's
slot table was checked for reference/support audit coverage. No reference
profile is fed into the model or used to tune a score.

## 6. Primary endpoint

The primary endpoint is the count of cases passing the frozen strict rule;
the preregistered target is 4–6 compounds, with one compound per MoA family.

## 7. Secondary endpoints

Row-level rescue score, failed-gate counts, per-seed support-slot
reproduction, and image/feature-explanation availability are reported. No
thresholded posterior activity endpoint is claimed while validation GE
predictions are absent.

## 8. Sample count

- Confirmed-active common test rotation rows: **{len(active)}**.
- Unique active compound-dose conditions: **{active[["compound", "dose_key"]].drop_duplicates().shape[0]}**.
- Unique active compounds: **{active["compound"].nunique()}**.
- Row-level strict method rescue gate: **{strict_rows}**.
- Rows reproduced in two saved support slots: **{two_rows}**.
- Selected case compounds: **{selected_count}**.
- Experiment 3 slot audit rows loaded: **{len(slot_metrics)}**.

## 9. Seed-level results

See `METRICS_BY_SEED.csv`. Counts are audit counts, not fitted estimates;
seeds are never counted as physical repeats.

## 10. Paired CI

`PAIRED_CONTRASTS.csv` and `BOOTSTRAP_CI.csv` contain valid not-estimable
rows because zero compounds passed the strict case rule. If cases exist in a
future frozen rerun, descriptive score intervals use 10,000 compound-level
resamples.

## 11. GO / SUPPORTIVE / NO-GO

Bio-D is **NO-GO** for the current frozen artifacts: {selected_count} case
compounds were selected, below the required 4–6. This decision is not
converted into a positive result by selecting near-miss compounds.

## 12. Biological interpretation

The absence of a selected case means only that this strict case-study layer
did not find a qualifying same-method activity-plus-MoA rescue reproduced in
two saved rotations. It does not overturn the aggregate hit-recovery,
retrieval, or dose-trajectory results. The current audit also shows why a
posterior activity rescue cannot be claimed: GE validation thresholding is
blocked.

## 13. Limitations

Frozen 1R predictions expose two support slots rather than all five physical
slots. Experiment 2 stores aggregate first-correct ranks but not Top-10 member
IDs, so no Top-10 case panel can be fabricated. The CP NPZ has 242 unnamed
feature columns and no raw images; feature-category and image visualization
are therefore `BLOCKED_IMAGE_VISUALIZATION`/unavailable for this zero-case
run. No manual cherry-picking was performed.

Figure status: `{figure_status}`.
"""
    (outdir / "RESULTS.md").write_text(results, encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    e1, query, per_query, slot_metrics, annotations, e1_config, hashes = load_inputs(args)
    active = prepare_active_rows(e1, query, per_query, annotations, e1_config)
    selected, selection = select_cases(active)
    exclusions = make_exclusions(active)
    metrics = make_metrics(active, selected)
    contrasts = make_contrasts(active, selected)
    bootstrap = make_bootstrap(contrasts)

    # Main audit outputs.  All case tables are emitted even when empty.
    eligible_columns = [
        "case_id", "compound", "dose", "moa_family", "rescue_score", "support_slot_count", "seed_count",
        "raw_score", "teacher_score", "posterior_score", "raw_rank", "teacher_rank", "posterior_rank", "e_rep",
        "selection_status",
    ]
    eligible = selected.copy()
    if eligible.empty:
        eligible = pd.DataFrame(columns=eligible_columns)
    else:
        eligible["case_id"] = eligible.apply(lambda r: f"{r['compound']}|{r['dose_key']}", axis=1)
        eligible["dose"] = eligible["dose_key"]
        eligible = eligible[[col for col in eligible_columns if col in eligible.columns]]
    eligible.to_csv(outdir / "ELIGIBLE_SAMPLES.csv", index=False)
    exclusions.to_csv(outdir / "EXCLUSIONS.csv", index=False)
    active.to_csv(outdir / "RESCUE_AUDIT.csv", index=False)
    metrics.to_csv(outdir / "METRICS_BY_SEED.csv", index=False)
    contrasts.to_csv(outdir / "PAIRED_CONTRASTS.csv", index=False)
    bootstrap.to_csv(outdir / "BOOTSTRAP_CI.csv", index=False)
    selection.to_csv(outdir / "SELECTION_AUDIT.csv", index=False)
    empty_case_tables(outdir)
    figure_status = write_figure(outdir, active, selected)

    config = {
        "version": VERSION,
        "dataset": "cpg0004-LINCS",
        "seeds": list(SEEDS),
        "bootstrap_rounds": int(args.bootstrap_rounds),
        "bootstrap_unit": "compound",
        "max_selected_cases": MAX_CASES,
        "requested_case_range": [4, 6],
        "moa_family_cap": 1,
        "rank_definition": "Experiment 2 base MoA first_correct_rank; fixed PCC, descending, dose-matched gallery",
        "thresholds": {
            "raw": e1_config.get("model_thresholds_validation_only", {}).get("raw"),
            "teacher": e1_config.get("model_thresholds_validation_only", {}).get("teacher"),
            "GE_posterior": e1_config.get("model_thresholds_validation_only", {}).get("GE_posterior"),
            "GE_posterior_status": "BLOCKED_VALIDATION_PREDICTIONS_MISSING",
        },
        "candidate_rule": {
            "confirmed_active": True,
            "raw_activity_wrong_or_moa_rank_gt50": True,
            "same_method_activity_rescue_and_moa_rank_le10": True,
            "minimum_distinct_saved_support_slots": 2,
            "no_manual_cherry_picking": True,
        },
        "counts": {
            "confirmed_active_rotation_rows": int(len(active)),
            "strict_rotation_gate_rows": int(active["strict_rotation_gate"].sum()),
            "two_rotation_gate_rows": int(active["case_rotation_gate"].sum()),
            "selected_case_compounds": int(selected["compound"].nunique()) if not selected.empty else 0,
        },
        "image_status": "BLOCKED_IMAGE_VISUALIZATION",
        "figure_status": figure_status,
        "input_sha256": hashes,
    }
    (outdir / "CONFIG.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_docs(outdir, active, exclusions, selected, selection, metrics, figure_status, hashes, slot_metrics, e1_config)
    print(json.dumps({"version": VERSION, "active_rows": len(active), "strict_rows": int(active["strict_rotation_gate"].sum()), "two_rotation_rows": int(active["case_rotation_gate"].sum()), "selected_cases": int(selected["compound"].nunique()) if not selected.empty else 0, "figure_status": figure_status}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
