#!/usr/bin/env python3
"""Post-hoc maximal technical-position matched-foreign diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import audit_cpdistiller_discrepancy as D
import evaluate_cpdistiller_retrieval as E


VERSION = "BBBC047-cpDistiller-technical-matched-foreign-v1-2026-09-19"
N_FOREIGN = 32
ARMS = ("RawMean775", "CP1783_globalZ", "cpDistiller_EMA_mu", "CFRA775")
METRICS = (
    "mAP_at_33",
    "held_rank",
    "top1",
    "held_cosine",
    "mean_foreign_cosine",
    "separation_margin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profiles", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--roles-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rows-root", type=Path, required=True)
    parser.add_argument("--comparator-root", type=Path, required=True)
    parser.add_argument("--frozen-evaluator", type=Path, required=True)
    parser.add_argument("--existing-test-results", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("matched_foreign_frozen_external", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key_text(key: tuple[str, str, str]) -> str:
    return "::".join(key)


def stable_candidate_order(
    candidates: list[tuple[str, str, str]],
    budget: int,
    condition_id: str,
    metadata: dict[tuple[str, str, str], dict[str, set[str]]],
    held_key: tuple[str, str, str],
) -> list[tuple[str, str, str]]:
    def digest(key: tuple[str, str, str]) -> str:
        payload = f"3407|{budget}|{condition_id}|{key_text(key)}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    held = metadata[held_key]

    def ordering(key: tuple[str, str, str]) -> tuple[int, int, int, int, int, str, tuple[str, str, str]]:
        candidate = metadata[key]
        same_row = int(bool(candidate["row"] & held["row"]))
        same_col = int(bool(candidate["col"] & held["col"]))
        same_batch = int(bool(candidate["batch"] & held["batch"]))
        exact_well = int(frozenset(candidate["well"]) == frozenset(held["well"]))
        return (
            -(same_row + same_col + same_batch),
            -exact_well,
            -same_batch,
            -same_row,
            -same_col,
            digest(key),
            key,
        )

    return sorted(candidates, key=ordering)


def support_mean(
    profiles: dict[tuple[str, str, str], np.ndarray], role: dict[str, str]
) -> np.ndarray:
    compound, dose = role["compound_id"], role["dose"]
    values = [profiles[(compound, dose, plate)] for plate in role["support_plate_ids"].split("|")]
    return np.mean(values, axis=0).astype(np.float32)


def assemble_arm(
    profiles: dict[tuple[str, str, str], np.ndarray],
    roles: list[dict[str, str]],
    candidates: list[list[tuple[str, str, str]]],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    query: list[np.ndarray] = []
    held: list[np.ndarray] = []
    foreign: list[np.ndarray] = []
    for role, selected in zip(roles, candidates):
        compound, dose = role["compound_id"], role["dose"]
        query.append(support_mean(profiles, role))
        held.append(profiles[(compound, dose, role["held_plate_id"])])
        foreign.append(np.stack([profiles[key] for key in selected]).astype(np.float32))
    return np.stack(query), np.stack(held), foreign


def metric_values(
    query: np.ndarray, held: np.ndarray, foreign: list[np.ndarray]
) -> dict[str, np.ndarray]:
    values = E.retrieval_metrics(query.astype(np.float64), held.astype(np.float64), foreign)
    held_cosine = values.pop("query_held_cosine")
    foreign_cosine = values.pop("mean_query_foreign_cosine")
    return {
        **values,
        "held_cosine": held_cosine,
        "mean_foreign_cosine": foreign_cosine,
        "separation_margin": held_cosine - foreign_cosine,
    }


def main() -> None:
    args = parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite {args.outdir}")
    evaluator = load_module(args.frozen_evaluator)
    role_audit = evaluator.check_freeze(args.roles_root, args.protocol)
    mapping, test_plates = E.load_test_mapping(args.mapping)
    by_plate, fallback_and_global, transform_audit = E.frozen_transforms(
        args.prep_dir / "TRAIN_INPUTS.npz", test_plates
    )
    raw1783, source_rows = D.load_test_raw(args.source_profiles, mapping)
    stages, stage_audit = D.stage_transforms(raw1783, source_rows, by_plate, fallback_and_global)
    physical_global_z = E.aggregate_physical(stages["CP1783_globalZ"], source_rows)
    metadata = D.physical_metadata(source_rows)

    frozen = np.load(args.existing_test_results / "TEST_WELL_EMBEDDINGS.npz", allow_pickle=False)
    expected_order = np.asarray([int(row["source_row_index"]) for row in source_rows])
    if not np.array_equal(expected_order, np.asarray(frozen["source_row_index"], dtype=np.int64)):
        raise RuntimeError("frozen embedding source-row order mismatch")
    physical_cpdistiller = E.aggregate_physical(np.asarray(frozen["embedding"], dtype=np.float32), source_rows)
    raw775_nested = evaluator.load_profiles(args.rows_root, "test")
    raw775 = {
        (compound, dose, str(plate)): np.asarray(value, dtype=np.float32)
        for (compound, dose), by_plate_values in raw775_nested.items()
        for plate, value in by_plate_values.items()
    }
    common_keys = set(metadata) & set(physical_global_z) & set(physical_cpdistiller) & set(raw775)

    args.outdir.mkdir(parents=True)
    preflight = {
        "version": VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "post-hoc shortcut diagnostic; no fitting, selection, or confirmation claim",
        "candidate_rule": {
            "different_compound": True,
            "same_dose": True,
            "candidate_batch_disjoint_from_support_batches": True,
            "candidate_source": "all common test physical profiles",
            "selection": (
                "maximize the number of row/column/batch matches to held; then exact well, "
                "held-batch, row, and column matches; break ties by SHA256 with seed 3407, "
                "budget, condition, and physical key"
            ),
        },
        "common_candidate_gallery_across_arms": list(ARMS),
        "physical_profile_counts": {
            "mapped_1783D": len(metadata),
            "RawMean775": len(raw775),
            "four_arm_intersection": len(common_keys),
        },
        "role_manifest_sha256": role_audit["manifest"]["sha256"],
        "mapping_sha256": D.sha256_file(args.mapping),
        "frozen_embedding_sha256": D.sha256_file(args.existing_test_results / "TEST_WELL_EMBEDDINGS.npz"),
        "transform": transform_audit,
        "stage_input_audit": stage_audit,
        "test_profile_values_loaded": True,
        "test_used_for_fit_or_selection": False,
        "status": "PASS",
    }
    (args.outdir / "MATCHED_FOREIGN_PREFLIGHT.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    candidate_manifest: list[dict[str, Any]] = []
    eligibility: dict[str, Any] = {}
    for budget in (1, 2, 3):
        roles = evaluator.read_roles(args.roles_root, "test", budget)
        compounds = np.asarray([role["compound_id"] for role in roles])
        conditions = np.asarray([role["condition_id"] for role in roles])
        selected_by_role: list[list[tuple[str, str, str]]] = []
        pool_sizes: list[int] = []
        exact_pool_sizes: list[int] = []
        layout_failures = 0
        for role in roles:
            compound, dose = role["compound_id"], role["dose"]
            support_keys = [
                (compound, dose, plate) for plate in role["support_plate_ids"].split("|")
            ]
            held_key = (compound, dose, role["held_plate_id"])
            missing_role_keys = [key for key in support_keys + [held_key] if key not in common_keys]
            if missing_role_keys:
                raise RuntimeError(
                    f"{budget}R role support/held is unavailable in a diagnostic arm: {missing_role_keys[:3]}"
                )
            held_wells = frozenset(metadata[held_key]["well"])
            if any(frozenset(metadata[key]["well"]) != held_wells for key in support_keys):
                layout_failures += 1
            support_batches = set().union(*(metadata[key]["batch"] for key in support_keys))
            pool = [
                key
                for key in common_keys
                if key[0] != compound
                and key[1] == dose
                and metadata[key]["batch"].isdisjoint(support_batches)
            ]
            exact_pool_sizes.append(
                sum(frozenset(metadata[key]["well"]) == held_wells for key in pool)
            )
            ordered = stable_candidate_order(
                pool, budget, role["condition_id"], metadata, held_key
            )
            pool_sizes.append(len(ordered))
            selected = ordered[:N_FOREIGN]
            selected_by_role.append(selected)
            held_metadata = metadata[held_key]
            selected_flags = {
                field: [int(bool(metadata[key][field] & held_metadata[field])) for key in selected]
                for field in ("row", "col", "batch", "well")
            }
            candidate_manifest.append(
                {
                    "budget": budget,
                    "condition_id": role["condition_id"],
                    "compound_id": compound,
                    "dose": dose,
                    "support_plate_ids": role["support_plate_ids"],
                    "held_plate_id": role["held_plate_id"],
                    "well_signature": "|".join(sorted(held_wells)),
                    "support_batches": "|".join(sorted(support_batches)),
                    "eligible_pool_size": len(ordered),
                    "exact_well_pool_size": exact_pool_sizes[-1],
                    **{
                        f"selected_foreign_same_{field}_fraction": float(np.mean(values))
                        for field, values in selected_flags.items()
                    },
                    "selected_foreign_keys": "|".join(key_text(key) for key in selected),
                }
            )
        too_small = int(np.sum(np.asarray(pool_sizes) < N_FOREIGN))
        eligible_mask = np.asarray(pool_sizes) >= N_FOREIGN
        eligibility[str(budget)] = {
            "conditions": len(roles),
            "eligible_conditions": int(eligible_mask.sum()),
            "support_held_well_signature_failures": layout_failures,
            "conditions_with_fewer_than_32_candidates": too_small,
            "candidate_pool_min": int(np.min(pool_sizes)),
            "candidate_pool_median": float(np.median(pool_sizes)),
            "candidate_pool_max": int(np.max(pool_sizes)),
            "exact_well_pool_min": int(np.min(exact_pool_sizes)),
            "exact_well_pool_median": float(np.median(exact_pool_sizes)),
            "exact_well_pool_max": int(np.max(exact_pool_sizes)),
            "selected_foreign_match_rates": {
                field: float(
                    np.mean(
                        [
                            row[f"selected_foreign_same_{field}_fraction"]
                            for row, keep in zip(candidate_manifest[-len(roles) :], eligible_mask)
                            if keep
                        ]
                    )
                )
                for field in ("row", "col", "batch", "well")
            },
        }
        if layout_failures or not eligible_mask.any():
            raise RuntimeError(f"{budget}R matched-foreign eligibility failed: {eligibility[str(budget)]}")

        eligible_roles = [role for role, keep in zip(roles, eligible_mask) if keep]
        eligible_candidates = [
            selected for selected, keep in zip(selected_by_role, eligible_mask) if keep
        ]
        compounds = compounds[eligible_mask]
        conditions = conditions[eligible_mask]

        arm_inputs: dict[str, tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = {
            "RawMean775": assemble_arm(raw775, eligible_roles, eligible_candidates),
            "CP1783_globalZ": assemble_arm(physical_global_z, eligible_roles, eligible_candidates),
            "cpDistiller_EMA_mu": assemble_arm(physical_cpdistiller, eligible_roles, eligible_candidates),
        }
        comparator = np.load(
            args.comparator_root / f"budget{budget}_TEST_PREDICTIONS.npz", allow_pickle=False
        )
        full_comparator_conditions = np.asarray(comparator["condition"]).astype(str)
        full_role_conditions = np.asarray([role["condition_id"] for role in roles]).astype(str)
        if not np.array_equal(full_comparator_conditions, full_role_conditions):
            raise RuntimeError("CFRA comparator condition order mismatch")
        _, raw_held, raw_foreign = arm_inputs["RawMean775"]
        arm_inputs["CFRA775"] = (
            np.asarray(comparator["CFRA"], dtype=np.float32)[eligible_mask], raw_held, raw_foreign
        )

        by_arm: dict[str, dict[str, np.ndarray]] = {}
        for arm in ARMS:
            by_arm[arm] = metric_values(*arm_inputs[arm])
            for metric in METRICS:
                point, low, high, n_compounds = E.bootstrap_summary(
                    by_arm[arm][metric], compounds, f"matched|{budget}|{arm}|{metric}"
                )
                summary_rows.append(
                    {
                        "budget": budget,
                        "arm": arm,
                        "metric": metric,
                        "point": point,
                        "ci_low": low,
                        "ci_high": high,
                        "n_compounds": n_compounds,
                        "n_conditions": len(conditions),
                        "bootstrap_rounds": E.BOOTSTRAP_ROUNDS,
                    }
                )
            for index, condition in enumerate(conditions):
                condition_rows.append(
                    {
                        "budget": budget,
                        "condition_id": condition,
                        "compound_id": compounds[index],
                        "arm": arm,
                        **{metric: float(by_arm[arm][metric][index]) for metric in METRICS},
                    }
                )
        for comparator_arm in ("RawMean775", "CP1783_globalZ", "CFRA775"):
            for metric in METRICS:
                point, low, high, n_compounds = E.bootstrap_delta(
                    by_arm["cpDistiller_EMA_mu"][metric],
                    by_arm[comparator_arm][metric],
                    compounds,
                    f"matched|{budget}|cpDistiller_EMA_mu-minus-{comparator_arm}|{metric}",
                )
                contrast_rows.append(
                    {
                        "budget": budget,
                        "contrast": f"cpDistiller_EMA_mu - {comparator_arm}",
                        "metric": metric,
                        "point": point,
                        "ci_low": low,
                        "ci_high": high,
                        "n_compounds": n_compounds,
                        "bootstrap_rounds": E.BOOTSTRAP_ROUNDS,
                    }
                )

    write_csv(args.outdir / "MATCHED_FOREIGN_CANDIDATE_MANIFEST.csv", candidate_manifest)
    write_csv(args.outdir / "MATCHED_FOREIGN_ARM_SUMMARY.csv", summary_rows)
    write_csv(args.outdir / "MATCHED_FOREIGN_CONTRASTS.csv", contrast_rows)
    write_csv(args.outdir / "MATCHED_FOREIGN_PER_CONDITION.csv", condition_rows)
    completion = {
        "version": VERSION,
        "status": "PASS",
        "analysis_type": preflight["analysis_type"],
        "eligibility": eligibility,
        "common_candidate_gallery_across_arms": list(ARMS),
        "foreign_per_condition": N_FOREIGN,
        "test_used_for_fit_or_selection": False,
    }
    (args.outdir / "MATCHED_FOREIGN_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
