#!/usr/bin/env python3
"""Post-hoc discrepancy audit for the frozen BBBC047 cpDistiller-C run.

No model is fitted and no configuration is selected here. The script compares
stage-wise representations, verifies the released inference output, evaluates
three retrieval geometries, and audits technical-position shortcuts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import evaluate_cpdistiller_retrieval as E
from cpDistiller.model import GMVAE


VERSION = "BBBC047-cpDistiller-discrepancy-audit-v2-2026-09-19"
ROUNDS = 10_000
GEOMETRIES = ("cosine", "euclidean", "normalized_euclidean")
STAGE_ARMS = (
    "CP1783_raw",
    "CP1783_MAD",
    "CP1783_plateZ",
    "CP1783_globalZ",
    "cpDistiller_frozen_EMA_mu",
    "cpDistiller_EMA_mu",
    "cpDistiller_EMA_projection_evalz",
    "cpDistiller_EMA_projection_mu",
    "cpDistiller_final_mu",
    "RawMean775",
    "CFRA775",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profiles", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--prep-dir", type=Path, required=True)
    parser.add_argument("--train-run-dir", type=Path, required=True)
    parser.add_argument("--roles-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rows-root", type=Path, required=True)
    parser.add_argument("--comparator-root", type=Path, required=True)
    parser.add_argument("--frozen-evaluator", type=Path, required=True)
    parser.add_argument("--existing-test-results", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("discrepancy_frozen_external", path)
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


def load_test_raw(source: Path, mapping: dict[int, dict[str, str]]) -> tuple[np.ndarray, list[dict[str, str]]]:
    offsets, meta_count, payload_count = E.source_schema(source)
    source_ids = sorted(mapping)
    positions = {source_id: index for index, source_id in enumerate(source_ids)}
    raw = np.empty((len(source_ids), E.EXPECTED_FEATURES), dtype=np.float32)
    found: set[int] = set()
    with gzip.open(source, "rt", newline="", encoding="utf-8") as handle:
        next(handle)
        for source_id, line in enumerate(handle):
            output_index = positions.get(source_id)
            if output_index is None:
                continue
            values = next(csv.reader([line]))
            if len(values) != meta_count + payload_count:
                raise RuntimeError(f"source row {source_id}: width mismatch")
            raw[output_index] = np.asarray(
                [values[meta_count + offset] for offset in offsets], dtype=np.float32
            )
            found.add(source_id)
    if found != set(source_ids):
        raise RuntimeError("mapped test source rows are missing")
    return raw, [mapping[source_id] for source_id in source_ids]


def stage_transforms(
    raw: np.ndarray,
    rows: list[dict[str, str]],
    by_plate: dict[str, tuple[np.ndarray, ...]],
    fallback_and_global: tuple[np.ndarray, ...],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fallback = fallback_and_global[:4]
    global_mean, global_std = fallback_and_global[4:]
    plates = np.asarray([row["plate"] for row in rows])
    raw_finite = raw.astype(np.float64)
    mad = np.empty_like(raw, dtype=np.float32)
    plate_z = np.empty_like(raw, dtype=np.float32)
    fallback_rows = 0
    nonfinite_before = int((~np.isfinite(raw_finite)).sum())
    for plate in sorted(set(plates)):
        mask = plates == plate
        center, scale, post_mean, post_std = by_plate.get(plate, fallback)
        if plate not in by_plate:
            fallback_rows += int(mask.sum())
        block = raw_finite[mask].copy()
        invalid = ~np.isfinite(block)
        if invalid.any():
            block[invalid] = np.broadcast_to(center, block.shape)[invalid]
            raw_finite[mask] = block
        mad_block = (block - center) / (scale + 1e-18)
        mad[mask] = mad_block.astype(np.float32)
        plate_z[mask] = ((mad_block - post_mean) / post_std).astype(np.float32)
    global_z = ((plate_z.astype(np.float64) - global_mean) / global_std).astype(np.float32)
    stages = {
        "CP1783_raw": raw_finite.astype(np.float32),
        "CP1783_MAD": mad,
        "CP1783_plateZ": plate_z,
        "CP1783_globalZ": global_z,
    }
    if any(not np.isfinite(value).all() for value in stages.values()):
        raise RuntimeError("non-finite values remain in stage waterfall")
    scale_summary: dict[str, dict[str, float]] = {}
    for label, values in stages.items():
        values64 = values.astype(np.float64)
        row_l2 = np.linalg.norm(values64, axis=1)
        scale_summary[label] = {
            "abs_max": float(np.abs(values64).max()),
            "row_l2_mean": float(row_l2.mean()),
            "row_l2_median": float(np.median(row_l2)),
            "row_l2_p99": float(np.quantile(row_l2, 0.99)),
            "row_l2_p999": float(np.quantile(row_l2, 0.999)),
            "row_l2_max": float(row_l2.max()),
        }
    fallback_mask = np.asarray([plate not in by_plate for plate in plates])
    global_row_l2 = np.linalg.norm(global_z.astype(np.float64), axis=1)
    return stages, {
        "nonfinite_source_values_before_train_frozen_imputation": nonfinite_before,
        "fallback_wells": fallback_rows,
        "stage_scale": scale_summary,
        "global_z_seen_plate_row_l2_max": float(global_row_l2[~fallback_mask].max()),
        "global_z_fallback_plate_row_l2": [float(value) for value in global_row_l2[fallback_mask]],
    }


def embed_variants(
    transformed: np.ndarray,
    source_rows: list[dict[str, str]],
    train_run: Path,
    existing_results: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for frozen embedding audit")
    checkpoint_dir = train_run / "cpDistillerC_memory_safe_sparse_v1"
    checkpoint_paths = {
        "EMA": checkpoint_dir / "final_model_ema.ckpt",
        "final": checkpoint_dir / "final_model.ckpt",
    }
    models: dict[str, GMVAE] = {}
    for label, path in checkpoint_paths.items():
        model = GMVAE(E.EXPECTED_FEATURES, 512, 50, 10).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        model.eval()
        models[label] = model
    output: dict[str, list[np.ndarray]] = defaultdict(list)
    # The released model samples Gumbel categories in eval mode. Run each
    # checkpoint as an independent full pass after resetting the frozen seed;
    # interleaving models would alter the official random-number sequence.
    E.set_seed(E.SEED)
    for start in range(0, len(transformed), 256):
        tensor = torch.from_numpy(transformed[start : start + 256]).to(device)
        with torch.no_grad():
            ema = models["EMA"](tensor)
            output["cpDistiller_EMA_mu"].append(ema["mu"].cpu().numpy().astype(np.float32))
            output["cpDistiller_EMA_projection_evalz"].append(
                ema["project_head"].cpu().numpy().astype(np.float32)
            )
            output["cpDistiller_EMA_projection_mu"].append(
                models["EMA"].encode.projection_head(ema["mu"]).cpu().numpy().astype(np.float32)
            )
    E.set_seed(E.SEED)
    for start in range(0, len(transformed), 256):
        tensor = torch.from_numpy(transformed[start : start + 256]).to(device)
        with torch.no_grad():
            final = models["final"](tensor)
            output["cpDistiller_final_mu"].append(final["mu"].cpu().numpy().astype(np.float32))
    arrays = {key: np.concatenate(values) for key, values in output.items()}
    for key, values in arrays.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite embedding variant: {key}")
    existing = np.load(existing_results / "TEST_WELL_EMBEDDINGS.npz", allow_pickle=False)
    expected_source_order = np.asarray([int(row["source_row_index"]) for row in source_rows])
    frozen_source_order = np.asarray(existing["source_row_index"], dtype=np.int64)
    if not np.array_equal(expected_source_order, frozen_source_order):
        raise RuntimeError("recomputed and frozen embedding source-row order mismatch")
    frozen_embedding = np.asarray(existing["embedding"], dtype=np.float32)
    recomputed_embedding = arrays["cpDistiller_EMA_mu"]
    delta64 = recomputed_embedding.astype(np.float64) - frozen_embedding.astype(np.float64)
    frozen64 = frozen_embedding.astype(np.float64)
    delta_abs = np.abs(delta64)
    relative_frobenius = float(np.linalg.norm(delta64) / max(np.linalg.norm(frozen64), np.finfo(float).tiny))
    frozen_row_norm = np.linalg.norm(frozen64, axis=1)
    delta_row_norm = np.linalg.norm(delta64, axis=1)
    row_relative_l2 = delta_row_norm / np.maximum(frozen_row_norm, np.finfo(float).tiny)
    recomputed64 = recomputed_embedding.astype(np.float64)
    recomputed_row_norm = np.linalg.norm(recomputed64, axis=1)
    row_cosine = np.sum(recomputed64 * frozen64, axis=1) / np.maximum(
        recomputed_row_norm * frozen_row_norm, np.finfo(float).tiny
    )
    direction_error = np.abs(1.0 - np.clip(row_cosine, -1.0, 1.0))
    numerically_equivalent = bool(
        relative_frobenius < 1e-6 and float(np.quantile(direction_error, 0.999)) < 1e-8
    )
    if not numerically_equivalent:
        raise RuntimeError(
            "recomputed EMA embedding is not numerically equivalent to the frozen embedding: "
            f"relative_frobenius={relative_frobenius:.3e}, "
            f"direction_error_p999={np.quantile(direction_error, 0.999):.3e}"
        )
    arrays["cpDistiller_frozen_EMA_mu"] = frozen_embedding
    audit = {
        "official_cpDistiller_eval_default_use_mean": True,
        "official_downstream_output": "GMVAE mu (50D)",
        "triplet_training_output": "projection_head(z)",
        "checkpoint_used_in_frozen_test": "final_model_ema.ckpt",
        "ema_checkpoint_sha256": sha256_file(checkpoint_paths["EMA"]),
        "non_ema_checkpoint_sha256": sha256_file(checkpoint_paths["final"]),
        "source_row_order_exact": True,
        "recomputed_ema_mu_vs_frozen_embedding_max_abs": float(delta_abs.max()),
        "recomputed_ema_mu_vs_frozen_embedding_mean_abs": float(delta_abs.mean()),
        "recomputed_ema_mu_vs_frozen_embedding_relative_frobenius": relative_frobenius,
        "recomputed_ema_mu_vs_frozen_embedding_row_relative_l2_mean": float(row_relative_l2.mean()),
        "recomputed_ema_mu_vs_frozen_embedding_row_relative_l2_p999": float(np.quantile(row_relative_l2, 0.999)),
        "recomputed_ema_mu_vs_frozen_embedding_direction_error_mean": float(direction_error.mean()),
        "recomputed_ema_mu_vs_frozen_embedding_direction_error_p999": float(np.quantile(direction_error, 0.999)),
        "recomputed_ema_mu_vs_frozen_embedding_allclose_rtol_1e_5_atol_1e_3": bool(
            np.allclose(recomputed_embedding, frozen_embedding, rtol=1e-5, atol=1e-3)
        ),
        "recomputed_ema_mu_exact": bool(np.array_equal(recomputed_embedding, frozen_embedding)),
        "recomputed_ema_mu_numerically_equivalent": numerically_equivalent,
        "dimensions": {key: int(value.shape[1]) for key, value in arrays.items()},
        "l2_norm_summary": {
            key: {
                "mean": float(norms.mean()),
                "median": float(np.median(norms)),
                "p99": float(np.quantile(norms, 0.99)),
                "p999": float(np.quantile(norms, 0.999)),
                "max": float(norms.max()),
            }
            for key, value in arrays.items()
            for norms in [np.linalg.norm(value.astype(np.float64), axis=1)]
        },
    }
    return arrays, audit


def aggregate_all(arrays: dict[str, np.ndarray], rows: list[dict[str, str]]) -> dict[str, dict[tuple[str, str, str], np.ndarray]]:
    return {arm: E.aggregate_physical(values, rows) for arm, values in arrays.items()}


def distance_metrics(query: np.ndarray, held: np.ndarray, foreign: list[np.ndarray], geometry: str) -> dict[str, np.ndarray]:
    query64 = query.astype(np.float64)
    held64 = held.astype(np.float64)
    foreign_distance = np.empty(len(query), dtype=np.float64)
    rank = np.empty(len(query), dtype=np.float64)
    if geometry == "cosine":
        held_measure = E.cosine(query64, held64)
        for index, gallery in enumerate(foreign):
            values = E.cosine(
                np.repeat(query64[index : index + 1], len(gallery), axis=0),
                gallery.astype(np.float64),
            )
            foreign_distance[index] = float(values.mean())
            rank[index] = 1.0 + float(np.sum(values > held_measure[index]))
        margin = held_measure - foreign_distance
    else:
        if geometry == "normalized_euclidean":
            query64 = query64 / np.maximum(np.linalg.norm(query64, axis=1, keepdims=True), 1e-12)
            held64 = held64 / np.maximum(np.linalg.norm(held64, axis=1, keepdims=True), 1e-12)
        held_measure = np.linalg.norm(query64 - held64, axis=1)
        for index, gallery_raw in enumerate(foreign):
            gallery = gallery_raw.astype(np.float64)
            if geometry == "normalized_euclidean":
                gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
            values = np.linalg.norm(gallery - query64[index], axis=1)
            foreign_distance[index] = float(values.mean())
            rank[index] = 1.0 + float(np.sum(values < held_measure[index]))
        margin = foreign_distance - held_measure
    return {
        "mAP_at_33": 1.0 / rank,
        "held_rank": rank,
        "top1": (rank == 1).astype(np.float64),
        "held_measure": held_measure,
        "mean_foreign_measure": foreign_distance,
        "separation_margin": margin,
    }


def compound_matrix(
    by_arm: dict[str, dict[str, np.ndarray]], compounds: np.ndarray, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    compound_ids = np.asarray(sorted(set(compounds.astype(str))))
    matrix = np.empty((len(compound_ids), len(by_arm)), dtype=np.float64)
    for column, arm in enumerate(by_arm):
        ids, values = E.compound_values(by_arm[arm][metric], compounds)
        if not np.array_equal(ids, compound_ids):
            raise RuntimeError(f"compound mismatch for {arm}/{metric}")
        matrix[:, column] = values
    return compound_ids, matrix


def bootstrap_columns(matrix: np.ndarray, seed_label: str) -> np.ndarray:
    rng = np.random.default_rng(E.stable_seed(seed_label))
    output = np.empty((ROUNDS, matrix.shape[1]), dtype=np.float64)
    for start in range(0, ROUNDS, 64):
        stop = min(start + 64, ROUNDS)
        indexes = rng.integers(0, len(matrix), size=(stop - start, len(matrix)))
        output[start:stop] = matrix[indexes].mean(axis=1)
    return output


def physical_metadata(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, set[str]]]:
    output: dict[tuple[str, str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"row": set(), "col": set(), "batch": set(), "well": set()}
    )
    for row in rows:
        item = output[(row["compound"], row["dose"], row["plate"])]
        item["row"].add(row["row"])
        item["col"].add(row["column"])
        item["batch"].add(row["batch"])
        item["well"].add(row["well"])
    return dict(output)


def match_flags(support: list[dict[str, set[str]]], candidate: dict[str, set[str]]) -> dict[str, bool]:
    return {
        field: any(bool(item[field] & candidate[field]) for item in support)
        for field in ("row", "col", "batch", "well")
    }


def technical_rows(
    roles: list[dict[str, str]], metadata: dict[tuple[str, str, str], dict[str, set[str]]]
) -> tuple[list[dict[str, Any]], np.ndarray]:
    output: list[dict[str, Any]] = []
    no_row_col = np.empty(len(roles), dtype=bool)
    for index, role in enumerate(roles):
        compound, dose = role["compound_id"], role["dose"]
        support = [metadata[(compound, dose, plate)] for plate in role["support_plate_ids"].split("|")]
        held = metadata[(compound, dose, role["held_plate_id"])]
        positive = match_flags(support, held)
        no_row_col[index] = not positive["row"] and not positive["col"]
        foreign_flags: list[dict[str, bool]] = []
        for donor, plate in zip(
            role["foreign_condition_ids"].split("|"), role["foreign_held_plate_ids"].split("|")
        ):
            donor_compound, donor_dose = E.parse_condition(donor)
            foreign_flags.append(match_flags(support, metadata[(donor_compound, donor_dose, plate)]))
        output.append(
            {
                "compound_id": compound,
                "condition_id": role["condition_id"],
                **{f"held_same_{field}": int(value) for field, value in positive.items()},
                **{
                    f"foreign_same_{field}_fraction": float(np.mean([item[field] for item in foreign_flags]))
                    for field in ("row", "col", "batch", "well")
                },
            }
        )
    return output, no_row_col


def main() -> None:
    args = parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite {args.outdir}")
    train_config_path = args.train_run_dir / "TRAIN_CONFIG.json"
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    completion = json.loads((args.existing_test_results / "TEST_RETRIEVAL_COMPLETE.json").read_text())
    if train_config.get("status") != "PASS" or completion.get("status") != "PASS":
        raise RuntimeError("frozen training/test prerequisites did not pass")
    evaluator = load_module(args.frozen_evaluator)
    role_audit = evaluator.check_freeze(args.roles_root, args.protocol)
    mapping, test_plates = E.load_test_mapping(args.mapping)
    by_plate, fallback_and_global, transform_audit = E.frozen_transforms(
        args.prep_dir / "TRAIN_INPUTS.npz", test_plates
    )
    args.outdir.mkdir(parents=True)
    preflight = {
        "version": VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "post-hoc discrepancy audit; no fitting or selection",
        "source_sha256": sha256_file(args.source_profiles),
        "mapping_sha256": sha256_file(args.mapping),
        "train_config_sha256": sha256_file(train_config_path),
        "existing_test_completion_sha256": sha256_file(args.existing_test_results / "TEST_RETRIEVAL_COMPLETE.json"),
        "role_manifest_sha256": role_audit["manifest"]["sha256"],
        "transform": transform_audit,
        "test_profile_values_loaded": False,
        "test_used_for_fit_or_selection": False,
        "status": "PASS",
    }
    (args.outdir / "AUDIT_PREFLIGHT.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")

    raw, source_rows = load_test_raw(args.source_profiles, mapping)
    input_stages, stage_audit = stage_transforms(raw, source_rows, by_plate, fallback_and_global)
    embedding_variants, output_audit = embed_variants(
        input_stages["CP1783_globalZ"], source_rows, args.train_run_dir, args.existing_test_results
    )
    all_well_arrays = {**input_stages, **embedding_variants}
    physical_by_arm = aggregate_all(all_well_arrays, source_rows)
    metadata = physical_metadata(source_rows)

    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    per_condition_rows: list[dict[str, Any]] = []
    technical_condition_rows: list[dict[str, Any]] = []
    technical_summary_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    geometry_equivalence: dict[str, Any] = {}
    for budget in (1, 2, 3):
        roles = evaluator.read_roles(args.roles_root, "test", budget)
        raw_query, raw_held, compounds, raw_foreign, conditions, _ = evaluator.assemble(
            roles, evaluator.load_profiles(args.rows_root, "test"), foreign=True
        )
        comparator = np.load(args.comparator_root / f"budget{budget}_TEST_PREDICTIONS.npz", allow_pickle=False)
        if not np.array_equal(comparator["condition"].astype(str), conditions.astype(str)):
            raise RuntimeError("comparator condition order mismatch")
        arm_inputs: dict[str, tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = {}
        for arm, profiles in physical_by_arm.items():
            query, held, foreign, cp_compounds, cp_conditions = E.assemble_embeddings(roles, profiles)
            if not np.array_equal(cp_conditions, conditions) or not np.array_equal(cp_compounds, compounds):
                raise RuntimeError(f"role assembly mismatch for {arm}")
            arm_inputs[arm] = (query, held, foreign)
        arm_inputs["RawMean775"] = (raw_query, raw_held, raw_foreign)
        arm_inputs["CFRA775"] = (np.asarray(comparator["CFRA"], dtype=np.float32), raw_held, raw_foreign)

        technical, no_row_col = technical_rows(roles, metadata)
        for row in technical:
            technical_condition_rows.append({"budget": budget, **row})
        for field in ("row", "col", "batch", "well"):
            held_values = np.asarray([row[f"held_same_{field}"] for row in technical], dtype=float)
            foreign_values = np.asarray([row[f"foreign_same_{field}_fraction"] for row in technical], dtype=float)
            _, held_compound = E.compound_values(held_values, compounds)
            _, foreign_compound = E.compound_values(foreign_values, compounds)
            technical_summary_rows.append(
                {
                    "budget": budget,
                    "field": field,
                    "held_match_rate": float(held_compound.mean()),
                    "foreign_match_rate": float(foreign_compound.mean()),
                    "held_minus_foreign": float((held_compound - foreign_compound).mean()),
                    "n_compounds": len(held_compound),
                    "no_held_row_and_no_held_col_conditions": int(no_row_col.sum()),
                }
            )

        metrics_by_geometry: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for geometry in GEOMETRIES:
            by_arm = {
                arm: distance_metrics(query, held, foreign, geometry)
                for arm, (query, held, foreign) in arm_inputs.items()
            }
            metrics_by_geometry[geometry] = by_arm
            compound_ids, map_matrix = compound_matrix(by_arm, compounds, "mAP_at_33")
            draws = bootstrap_columns(map_matrix, f"b{budget}|{geometry}|common-paired")
            arm_order = list(by_arm)
            for arm_index, arm in enumerate(arm_order):
                for metric, values in by_arm[arm].items():
                    _, reduced = E.compound_values(values, compounds)
                    row = {
                        "budget": budget,
                        "geometry": geometry,
                        "arm": arm,
                        "metric": metric,
                        "point": float(reduced.mean()),
                        "n_compounds": len(reduced),
                        "n_conditions": len(compounds),
                    }
                    if metric == "mAP_at_33":
                        row.update(
                            ci_low=float(np.quantile(draws[:, arm_index], 0.025)),
                            ci_high=float(np.quantile(draws[:, arm_index], 0.975)),
                            bootstrap_rounds=ROUNDS,
                        )
                    summary_rows.append(row)
                for condition_index, condition in enumerate(conditions):
                    per_condition_rows.append(
                        {
                            "budget": budget,
                            "geometry": geometry,
                            "arm": arm,
                            "compound_id": compounds[condition_index],
                            "condition_id": condition,
                            **{name: float(values[condition_index]) for name, values in by_arm[arm].items()},
                        }
                    )
            baseline_index = arm_order.index("CP1783_globalZ")
            for arm in (
                "CP1783_raw", "CP1783_MAD", "CP1783_plateZ",
                "cpDistiller_EMA_mu", "cpDistiller_EMA_projection_evalz",
                "cpDistiller_EMA_projection_mu", "cpDistiller_final_mu",
            ):
                arm_index = arm_order.index(arm)
                delta_draws = draws[:, arm_index] - draws[:, baseline_index]
                delta = map_matrix[:, arm_index] - map_matrix[:, baseline_index]
                contrast_rows.append(
                    {
                        "budget": budget,
                        "geometry": geometry,
                        "comparison": f"{arm} - CP1783_globalZ",
                        "metric": "mAP_at_33",
                        "point": float(delta.mean()),
                        "ci_low": float(np.quantile(delta_draws, 0.025)),
                        "ci_high": float(np.quantile(delta_draws, 0.975)),
                        "n_compounds": len(compound_ids),
                        "bootstrap_rounds": ROUNDS,
                    }
                )
            for arm in ("CP1783_globalZ", "cpDistiller_EMA_mu", "RawMean775", "CFRA775"):
                values = by_arm[arm]["mAP_at_33"][no_row_col]
                subset_compounds = compounds[no_row_col]
                _, reduced = E.compound_values(values, subset_compounds)
                subset_rows.append(
                    {
                        "budget": budget,
                        "geometry": geometry,
                        "subset": "held shares neither row nor column with any support well",
                        "arm": arm,
                        "mAP_at_33": float(reduced.mean()) if len(reduced) else None,
                        "n_conditions": int(no_row_col.sum()),
                        "n_compounds": len(reduced),
                    }
                )
        cosine_rank = metrics_by_geometry["cosine"]["cpDistiller_EMA_mu"]["held_rank"]
        normalized_rank = metrics_by_geometry["normalized_euclidean"]["cpDistiller_EMA_mu"]["held_rank"]
        geometry_equivalence[str(budget)] = {
            "cosine_vs_normalized_euclidean_rank_equal": bool(np.array_equal(cosine_rank, normalized_rank)),
            "max_rank_difference": float(np.max(np.abs(cosine_rank - normalized_rank))),
        }

    write_csv(args.outdir / "STAGE_GEOMETRY_SUMMARY.csv", summary_rows)
    write_csv(args.outdir / "STAGE_GEOMETRY_CONTRASTS.csv", contrast_rows)
    write_csv(args.outdir / "STAGE_GEOMETRY_PER_CONDITION.csv", per_condition_rows)
    write_csv(args.outdir / "TECHNICAL_MATCH_PER_CONDITION.csv", technical_condition_rows)
    write_csv(args.outdir / "TECHNICAL_MATCH_SUMMARY.csv", technical_summary_rows)
    write_csv(args.outdir / "NO_ROW_COL_SUBSET.csv", subset_rows)
    marker = {
        "version": VERSION,
        "stage_input_audit": stage_audit,
        "output_layer_and_checkpoint_audit": output_audit,
        "geometry_equivalence": geometry_equivalence,
        "test_profile_values_loaded": True,
        "test_used_for_fit_or_selection": False,
        "status": "PASS",
    }
    (args.outdir / "DISCREPANCY_AUDIT_COMPLETE.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
