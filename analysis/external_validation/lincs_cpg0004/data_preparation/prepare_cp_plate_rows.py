#!/usr/bin/env python3
"""Prepare plate-preserving CP effect rows for the cpg0004 external benchmark.

This is deliberately a data-only step.  It does not read GE, fit a model, or
choose a test split from a result.  Treatment profiles are converted to
plate-relative effects using the feature-wise median of the control wells on
the same plate.  The output keeps compound, dose, plate, well, batch and the
control vector so that later repeat-budget experiments can be recomputed from
the immutable artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STANDARD_DOSES = (0.04, 0.12, 0.37, 1.11, 3.33, 10.0)
REFERENCE_COMPOUNDS = {"BRD-K50691590", "BRD-K60230970"}
VERSION = "cpg0004-LINCS-CP-plate-rows-2026-08-30"


def clean_id(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def dose_key(value: object) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(x):
        return ""
    nearest = min(STANDARD_DOSES, key=lambda d: abs(d - x))
    return f"{nearest:.12g}" if abs(nearest - x) <= 0.03 else ""


def digest_strings(values: list[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(value.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cp-source",
        type=Path,
        default=Path(
            "/path/to/home/AIDD/baseline/data/MVC/LINCS-Pilot1/CellPainting/"
            "replicate_level_cp_normalized_variable_selected.csv.gz"
        ),
    )
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--split-seed", type=int, default=3407)
    p.add_argument("--train-fraction", type=float, default=0.60)
    p.add_argument("--valid-fraction", type=float, default=0.20)
    p.add_argument("--missing-threshold", type=float, default=0.20)
    p.add_argument("--smoke", action="store_true", help="Read only the first 1,000 rows.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.train_fraction < 1 or not 0 < args.valid_fraction < 1:
        raise ValueError("train/valid fractions must be in (0,1)")
    if args.train_fraction + args.valid_fraction >= 1:
        raise ValueError("train + valid fraction must leave a test set")
    if args.outdir.exists() and any(args.outdir.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        frame = pd.read_csv(args.cp_source, nrows=1000, low_memory=False)
        (args.outdir / "SMOKE_NOT_EVIDENCE.txt").write_text(
            json.dumps({"version": VERSION, "rows": int(len(frame)), "columns": int(frame.shape[1])}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"version": VERSION, "smoke": True, "rows": len(frame), "columns": frame.shape[1]}, indent=2))
        return

    frame = pd.read_csv(args.cp_source, low_memory=False)
    required = {
        "Metadata_broad_sample_type", "Metadata_Plate", "Metadata_Well",
        "Metadata_pert_id", "Metadata_broad_id", "Metadata_dose_recode",
        "Metadata_mmoles_per_liter", "Metadata_Batch_Number", "Metadata_Batch_Date",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing CP metadata columns: {missing}")

    metadata = [c for c in frame.columns if c.startswith("Metadata_")]
    feature_columns = [
        c for c in frame.columns
        if c not in metadata and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not feature_columns:
        raise ValueError("No numeric CP feature columns found")
    treatment_mask = frame["Metadata_broad_sample_type"].astype(str).str.strip().eq("trt")
    control_mask = frame["Metadata_broad_sample_type"].astype(str).str.strip().eq("control")
    if not treatment_mask.any() or not control_mask.any():
        raise ValueError("CP source has no treatment or control rows")

    # Restrict only by information available before the benchmark: treatment
    # type, standard concentration and the two documented 20-mM references.
    ids = frame["Metadata_broad_id"].where(
        frame["Metadata_broad_id"].notna() & frame["Metadata_broad_id"].astype(str).str.strip().ne(""),
        frame["Metadata_pert_id"],
    ).map(clean_id)
    raw_dose = frame["Metadata_dose_recode"].where(
        frame["Metadata_dose_recode"].notna(), frame["Metadata_mmoles_per_liter"]
    )
    dose_labels = raw_dose.map(dose_key)
    keep = treatment_mask & dose_labels.ne("") & ~ids.isin(REFERENCE_COMPOUNDS)
    treatment = frame.loc[keep, feature_columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    control = frame.loc[control_mask, feature_columns].apply(pd.to_numeric, errors="coerce")
    treatment_meta = pd.DataFrame({
        "compound_id": ids.loc[keep].to_numpy(dtype=str),
        "dose": dose_labels.loc[keep].to_numpy(dtype=str),
        "plate": frame.loc[keep, "Metadata_Plate"].map(clean_id).to_numpy(dtype=str),
        "well": frame.loc[keep, "Metadata_Well"].map(clean_id).to_numpy(dtype=str),
        "batch": frame.loc[keep, "Metadata_Batch_Number"].map(clean_id).to_numpy(dtype=str),
        "batch_date": frame.loc[keep, "Metadata_Batch_Date"].map(clean_id).to_numpy(dtype=str),
        "inchikey14": frame.loc[keep, "Metadata_InChIKey14"].map(clean_id).to_numpy(dtype=str)
        if "Metadata_InChIKey14" in frame.columns else np.full(int(keep.sum()), "", dtype=str),
    })
    treatment_meta["raw_row_index"] = np.flatnonzero(keep.to_numpy()).astype(np.int64)
    if treatment_meta["plate"].eq("").any() or treatment_meta["compound_id"].eq("").any():
        raise ValueError("Treatment rows contain empty compound or plate IDs")

    # A plate can contain two or three wells for the same ordinary
    # compound-dose condition.  Those wells are technical replicates, not
    # independent experimental repeats.  Collapse them to one robust
    # feature-wise plate observation while retaining the well list and raw row
    # indices in the manifest.
    key_columns = ["compound_id", "dose", "plate"]
    duplicate_row_count = int(treatment_meta.duplicated(key_columns, keep=False).sum())
    duplicate_group_count = int(treatment_meta.duplicated(key_columns, keep=False).groupby(
        treatment_meta[key_columns].apply(tuple, axis=1)
    ).any().sum())

    # Retain features whose combined treatment/control missingness is at most
    # the pre-registered threshold.  Median imputation is fitted from controls
    # only, then applied to both arms.  No held-out treatment value is used to
    # fit a model or select a method.
    combined = pd.concat([treatment, control], axis=0, ignore_index=True)
    keep_features = [c for c in feature_columns if float(combined[c].isna().mean()) <= args.missing_threshold]
    if not keep_features:
        raise ValueError("No feature survives missingness threshold")
    treatment = treatment[keep_features]
    control = control[keep_features]
    # A small number of features can be absent from every DMSO well while
    # still passing the pooled missingness threshold.  Fall back to the
    # pooled treatment/control median for those technical columns, and drop a
    # column only if it is entirely missing in the source.
    pooled_medians = combined[keep_features].median(axis=0, skipna=True)
    control_medians_global = control.median(axis=0, skipna=True).fillna(pooled_medians)
    keep_features = [c for c in keep_features if pd.notna(control_medians_global[c])]
    control_medians_global = control_medians_global[keep_features]
    treatment = treatment[keep_features]
    control = control[keep_features]
    control = control.fillna(control_medians_global)
    treatment = treatment.fillna(control_medians_global)
    if treatment.isna().any().any() or control.isna().any().any():
        raise ValueError("NaNs remain after control-median imputation")

    # Compute a robust plate reference and subtract it from each treatment.
    control_frame = frame.loc[control_mask, ["Metadata_Plate"]].copy()
    control_frame["plate"] = control_frame["Metadata_Plate"].map(clean_id).to_numpy(dtype=str)
    control_frame = control_frame.reset_index(drop=True)
    control_values = control.to_numpy(dtype=np.float32)
    control_by_plate: dict[str, np.ndarray] = {}
    for plate, indices in control_frame.groupby("plate", sort=False).groups.items():
        if not plate:
            continue
        control_by_plate[plate] = np.median(control_values[np.asarray(list(indices), dtype=np.int64)], axis=0).astype(np.float32)
    missing_plates = sorted(set(treatment_meta["plate"]) - set(control_by_plate))
    if missing_plates:
        raise ValueError(f"No control reference for plates: {missing_plates[:10]}")
    treatment_values = treatment.to_numpy(dtype=np.float32)
    grouped = treatment_meta.groupby(key_columns, sort=True, dropna=False)
    aggregate_rows: list[dict[str, object]] = []
    aggregate_values: list[np.ndarray] = []
    aggregate_baselines: list[np.ndarray] = []
    for (compound, dose, plate), positions in grouped.indices.items():
        ix = np.asarray(list(positions), dtype=np.int64)
        wells = sorted(set(treatment_meta.iloc[ix]["well"].astype(str)))
        batches = sorted(set(x for x in treatment_meta.iloc[ix]["batch"].astype(str) if x))
        dates = sorted(set(x for x in treatment_meta.iloc[ix]["batch_date"].astype(str) if x))
        ikeys = sorted(set(x for x in treatment_meta.iloc[ix]["inchikey14"].astype(str) if x))
        raw_indices = sorted(int(x) for x in treatment_meta.iloc[ix]["raw_row_index"])
        aggregate_rows.append({
            "compound_id": str(compound), "dose": str(dose), "plate": str(plate),
            "well": ";".join(wells), "batch": ";".join(batches),
            "batch_date": ";".join(dates), "inchikey14": ";".join(ikeys),
            "raw_row_index": ";".join(str(x) for x in raw_indices),
            "raw_well_count": int(len(ix)),
        })
        aggregate_values.append(np.median(treatment_values[ix], axis=0).astype(np.float32))
        aggregate_baselines.append(control_by_plate[str(plate)])
    treatment_meta = pd.DataFrame(aggregate_rows)
    treatment_values = np.vstack(aggregate_values).astype(np.float32)
    baseline = np.vstack(aggregate_baselines).astype(np.float32)
    deltas = treatment_values - baseline
    if not np.isfinite(deltas).all():
        raise ValueError("Non-finite CP deltas")

    # Stable, auditable compound-level split.  The split is locked before any
    # benchmark code sees held-out rows.  All doses and plates of a compound
    # remain in the same split.
    compounds = sorted(treatment_meta["compound_id"].unique().tolist())
    rng = np.random.default_rng(args.split_seed)
    perm = np.asarray(compounds, dtype=object)[rng.permutation(len(compounds))]
    n_train = int(round(len(compounds) * args.train_fraction))
    n_valid = int(round(len(compounds) * args.valid_fraction))
    split_values = {
        "train": sorted(str(x) for x in perm[:n_train]),
        "valid": sorted(str(x) for x in perm[n_train:n_train + n_valid]),
        "test": sorted(str(x) for x in perm[n_train + n_valid:]),
    }
    if set(split_values["train"]) & set(split_values["valid"]) or set(split_values["train"]) & set(split_values["test"]) or set(split_values["valid"]) & set(split_values["test"]):
        raise AssertionError("Split overlap")
    split_map = {compound: split for split, values in split_values.items() for compound in values}
    treatment_meta["split"] = treatment_meta["compound_id"].map(split_map)
    if treatment_meta["split"].isna().any():
        raise AssertionError("Missing split assignment")

    np.savez_compressed(
        args.outdir / "cp_plate_rows.npz",
        compound_id=treatment_meta["compound_id"].to_numpy(dtype=str),
        dose=treatment_meta["dose"].to_numpy(dtype=str),
        plate=treatment_meta["plate"].to_numpy(dtype=str),
        well=treatment_meta["well"].to_numpy(dtype=str),
        batch=treatment_meta["batch"].to_numpy(dtype=str),
        batch_date=treatment_meta["batch_date"].to_numpy(dtype=str),
        inchikey14=treatment_meta["inchikey14"].to_numpy(dtype=str),
        split=treatment_meta["split"].to_numpy(dtype=str),
        delta=deltas,
        baseline=baseline,
    )
    treatment_meta.to_csv(args.outdir / "cp_plate_manifest.csv", index=False)
    pd.DataFrame({"feature": keep_features}).to_csv(args.outdir / "feature_names.csv", index=False)
    split_payload = {
        "version": VERSION,
        "split_seed": args.split_seed,
        "fractions": {"train": args.train_fraction, "valid": args.valid_fraction, "test": 1 - args.train_fraction - args.valid_fraction},
        "compound_counts": {key: len(value) for key, value in split_values.items()},
        "sha256_by_split": {key: digest_strings(value) for key, value in split_values.items()},
        "train_compounds": split_values["train"],
        "valid_compounds": split_values["valid"],
        "test_compounds": split_values["test"],
    }
    (args.outdir / "split_lock.json").write_text(json.dumps(split_payload, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "version": VERSION,
        "source": str(args.cp_source),
        "source_sha256": hashlib.sha256(args.cp_source.read_bytes()).hexdigest() if args.cp_source.is_file() else None,
        "raw_rows": int(len(frame)),
        "raw_columns": int(frame.shape[1]),
        "raw_numeric_feature_count": len(feature_columns),
        "retained_feature_count": len(keep_features),
        "control_rows": int(control_mask.sum()),
        "ordinary_treatment_rows": int(len(treatment_meta)),
        "raw_treatment_rows_before_plate_aggregation": int(keep.sum()),
        "ordinary_compound_count": int(treatment_meta["compound_id"].nunique()),
        "dose_labels": sorted(treatment_meta["dose"].unique().tolist(), key=float),
        "plate_count": int(treatment_meta["plate"].nunique()),
        "batch_count": int(treatment_meta["batch"].nunique()),
        "condition_count": int(treatment_meta.groupby(["compound_id", "dose"]).ngroups),
        "duplicate_condition_plate_rows": duplicate_row_count,
        "duplicate_condition_plate_groups": duplicate_group_count,
        "compound_dose_plate_repeat_distribution": {str(k): int(v) for k, v in treatment_meta.groupby(["compound_id", "dose"]).plate.nunique().value_counts().sort_index().items()},
        "split": split_payload,
        "guardrails": [
            "GE is not read or used in this CP-only preparation",
            "20-mM plate-wide reference compounds are excluded",
            "technical wells within a compound-dose-plate are feature-wise median-collapsed and retained in the manifest",
            "plate-level control median is computed before delta construction",
            "all doses and plates for a compound share one split",
        ],
    }
    (args.outdir / "PREPARATION_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
