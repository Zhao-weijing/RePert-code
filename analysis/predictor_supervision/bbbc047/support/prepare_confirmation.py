#!/usr/bin/env python3
"""Freeze a new 60/20/20 dual-repeat BBBC047 chemical confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = Path(__file__).resolve().parents[4] / "analysis"
SOURCE = (Path(__file__).resolve().parents[4] / "analysis/predictor_supervision/bbbc047/support/run_rank_matched_pca_control.py")
_spec = importlib.util.spec_from_file_location("bbbc047_decomp_source", SOURCE)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot import {SOURCE}")
SRC = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = SRC
_spec.loader.exec_module(SRC)

VERSION = "BBBC047-CFRA-Decomposition-confirmation-v2-2026-09-13"
SPLIT_SALT = "bbbc047-cfra-decomposition-confirmation-v1|chemical-split|"
BUDGETS = (1, 2, 3)
PRIMARY = (1, 2)
RANKS = {1: 8, 2: 8, 3: 9}
MASTER_SEEDS = (3407, 42, 2025, 1337, 7331)
BOOTSTRAP_SEED = 3407
DEFAULT_ROWS = Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def stable(compound: str) -> tuple[str, str]:
    return hashlib.sha256((SPLIT_SALT + compound).encode("utf-8")).hexdigest(), compound


def combine_rows(rows_root: Path) -> tuple[Any, dict[str, Any]]:
    rows = SRC.load_rows(rows_root, ("train", "valid"), smoke=False)
    train, valid = rows["train"], rows["valid"]
    checks: dict[str, Any] = {}
    for name, value in (("train", train), ("valid", valid)):
        checks[name] = {
            "n_rows": int(len(value.compound)),
            "n_features": int(value.delta.shape[1]),
            "delta_all_finite": bool(np.isfinite(value.delta).all()),
            "dose_all_finite": bool(np.isfinite(value.dose.astype(float)).all()),
        }
        if value.delta.shape[1] != 775 or not checks[name]["delta_all_finite"] or not checks[name]["dose_all_finite"]:
            raise RuntimeError(f"STOP: invalid {name} source shape/finiteness: {checks[name]}")
    compound = np.concatenate([train.compound, valid.compound])
    dose = np.concatenate([train.dose, valid.dose])
    plate = np.concatenate([train.plate, valid.plate])
    delta = np.concatenate([train.delta, valid.delta])
    if len(set(train.compound.tolist()) & set(valid.compound.tolist())):
        raise RuntimeError("former train and validation compounds overlap")
    return SRC.LEGACY.Rows("BBBC047", "CP", "development", compound, dose, plate, delta), checks


def verify_official_test_excluded(rows_root: Path, development_compounds: set[str]) -> dict[str, Any]:
    test_path = rows_root / "test_cp_plate_rows.npz"
    if not test_path.exists():
        raise FileNotFoundError(f"STOP: official test artifact missing: {test_path}")
    with np.load(test_path, allow_pickle=False) as z:
        test_compounds = set(np.asarray(z["smiles"], dtype=str).tolist())
    overlap = development_compounds & test_compounds
    if overlap:
        raise RuntimeError(f"STOP: development pool overlaps old official test ({len(overlap)} compounds)")
    return {
        "artifact": str(test_path),
        "sha256": sha256_file(test_path),
        "n_compounds": len(test_compounds),
        "development_overlap": 0,
    }


def partition(compounds: set[str]) -> dict[str, set[str]]:
    ordered = sorted(compounds, key=stable)
    n_fit = int(np.floor(0.60 * len(ordered)))
    n_valid = int(np.floor(0.20 * len(ordered)))
    result = {
        "fit": set(ordered[:n_fit]),
        "validation": set(ordered[n_fit:n_fit + n_valid]),
        "confirmation": set(ordered[n_fit + n_valid:]),
    }
    if any(not x for x in result.values()) or sum(map(len, result.values())) != len(ordered):
        raise RuntimeError("invalid 60/20/20 partition")
    if result["fit"] & result["validation"] or result["fit"] & result["confirmation"] or result["validation"] & result["confirmation"]:
        raise RuntimeError("partition overlap")
    return result


def subset(rows: Any, allowed: set[str], name: str) -> Any:
    mask = np.isin(rows.compound, np.asarray(sorted(allowed), dtype=str))
    return SRC.LEGACY.Rows(rows.dataset, rows.modality, name, rows.compound[mask], rows.dose[mask], rows.plate[mask], rows.delta[mask])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty output {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def write_sealed_rows(path: Path, rows: Any) -> str:
    """Materialize one split so Fit never has to open the monolithic source delta."""
    np.savez_compressed(
        path,
        smiles=np.asarray(rows.compound, dtype=str),
        dose=np.asarray(rows.dose, dtype=str),
        plate=np.asarray(rows.plate, dtype=str),
        delta=np.asarray(rows.delta, dtype=np.float32),
    )
    return sha256_file(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rows-root", type=Path, default=DEFAULT_ROWS)
    p.add_argument("--protocol", type=Path, default=HERE / "PROTOCOL.md")
    args = p.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    combined, source_checks = combine_rows(args.rows_root)
    compounds = set(np.asarray(combined.compound, dtype=str).tolist())
    old_test_audit = verify_official_test_excluded(args.rows_root, compounds)
    parts = partition(compounds)
    views = {name: subset(combined, values, name) for name, values in parts.items()}
    bundles = {b: {name: SRC.build_dual_examples(view, b, include_foreign=(name == "confirmation")) for name, view in views.items()} for b in BUDGETS}
    for b in PRIMARY:
        if any(len(bundles[b][name]) == 0 for name in views):
            raise RuntimeError(f"STOP: no dual-held eligible compounds for primary budget {b}")
    args.output_root.mkdir(parents=True)
    sealed_row_hashes = {
        name: write_sealed_rows(args.output_root / f"{name.upper()}_ROWS.npz", views[name])
        for name in ("fit", "validation")
    }
    split_rows = [{"split": name, "compound_id": c, "order_hash": stable(c)[0]} for name in ("fit", "validation", "confirmation") for c in sorted(parts[name], key=stable)]
    write_csv(args.output_root / "COMPOUND_SPLIT.csv", split_rows)
    manifest: list[dict[str, Any]] = []
    for b in BUDGETS:
        for name, ex in bundles[b].items():
            for i in range(len(ex)):
                support = tuple(ex.support_plates[i]); held_a = tuple(ex.held_a_plates[i]); held_b = tuple(ex.held_b_plates[i])
                if set(support) & set(held_a) or set(support) & set(held_b) or set(held_a) & set(held_b):
                    raise RuntimeError("STOP: physical plate roles overlap")
                manifest.append({
                    "split": name, "budget": b, "compound_id": str(ex.compound[i]), "dose": str(ex.dose[i]),
                    "condition": str(ex.condition[i]), "support_rep_ids": "|".join(support),
                    "held_A_rep_id": "|".join(held_a), "held_B_rep_id": "|".join(held_b),
                    "foreign_compound": str(ex.foreign_compound[i]) if name == "confirmation" else "",
                    "foreign_ok": int(ex.foreign_ok[i]) if name == "confirmation" else "",
                })
    write_csv(args.output_root / "ANALYSIS_MANIFEST.csv", manifest)
    counts = {str(b): {name: len(bundles[b][name]) for name in views} for b in BUDGETS}
    foreign = {str(b): {"eligible": int(np.sum(bundles[b]["confirmation"].foreign_ok)), "total": len(bundles[b]["confirmation"])} for b in BUDGETS}
    source_hashes = {name: sha256_file(args.rows_root / f"{name}_cp_plate_rows.npz") for name in ("train", "valid")}
    audit = {
        "version": VERSION, "phase": "prepare", "status": "PASS", "claim": "chemical_confirmation_not_plate_ood",
        "source_universe": "former official train+valid only; old official test excluded", "n_source_compounds": len(compounds),
        "split_salt": SPLIT_SALT, "split_counts": {k: len(v) for k, v in parts.items()}, "dual_repeat_eligible": counts,
        "foreign_coverage": foreign, "primary_budgets": list(PRIMARY), "supportive_budgets": [3], "pca_ranks": RANKS,
        "master_seeds": list(MASTER_SEEDS), "alpha": 1.0, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_rounds": 10000,
        "foreign_rule": "exact nominal dose and identical complete acquisition-plate signature",
        "checkpoint_rule": "validation only", "source_hashes": source_hashes,
        "sealed_fit_validation_rows": sealed_row_hashes,
        "source_shape_finite_checks": source_checks, "old_official_test_exclusion": old_test_audit,
        "protocol_sha256": sha256_file(args.protocol), "split_sha256": sha256_file(args.output_root / "COMPOUND_SPLIT.csv"),
        "manifest_sha256": sha256_file(args.output_root / "ANALYSIS_MANIFEST.csv"),
        "confirmation_loaded_by_data_custodian_prepare": True,
        "confirmation_values_opened_for_repeat_manifest_only": True,
        "confirmation_model_scores_computed_in_prepare": False,
        "confirmation_used_for_training": False,
        "confirmation_used_for_selection": False,
        "sealing_contract": "fit stage receives fit/validation only; confirmation is opened only after FIT_COMPLETE and SELECTION_FREEZE verification",
    }
    dump(args.output_root / "PREPARE_COMPLETE.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
