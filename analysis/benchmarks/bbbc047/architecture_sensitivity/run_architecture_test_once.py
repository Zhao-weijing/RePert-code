#!/usr/bin/env python3
"""One historical, non-blind BBBC047 test recomputation after validation freeze."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from architecture_common import (
    BUDGETS,
    EXPECTED_ROLE_SHA256,
    SEEDS,
    VERSION,
    build_model,
    forward_prediction,
    load_module,
    load_official_modules,
    model_specs,
    save_json,
    sha256_file,
    torch_modules,
)


ARMS = ("MLP-control", "ResNet", "FT-Transformer", "TabM")
BOOTSTRAP_REPS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles-root", type=Path, required=True)
    parser.add_argument("--test-rows-root", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--vendor-deps", type=Path, required=True)
    parser.add_argument("--reference-script", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--frozen-cfra-per-condition", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


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


def finite_matrix(name: str, value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 775 or not np.isfinite(result).all():
        raise RuntimeError(f"{name} must be finite n-by-775, got {result.shape}")
    return result


def normalize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = np.asarray((x - mean) / scale, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite checkpoint-normalized profile")
    return result


def predict(name: str, model: Any, x: np.ndarray, torch: Any, device: Any) -> np.ndarray:
    micro_batch = int(model_specs()[name]["micro_batch_size"])
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), micro_batch):
            batch = torch.from_numpy(x[start : start + micro_batch]).to(device)
            chunks.append(forward_prediction(name, model, batch).detach().cpu().numpy().astype(np.float32))
    return finite_matrix(name, np.concatenate(chunks, axis=0))


def load_validation_freeze(args: argparse.Namespace) -> dict[str, Any]:
    complete_path = args.validation_dir / "VALIDATION_COMPLETE.json"
    manifest_path = args.validation_dir / "CHECKPOINT_MANIFEST.json"
    if not complete_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("validation completion marker/checkpoint manifest absent")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if complete.get("status") != "PASS" or complete.get("stage") != "validation-only":
        raise RuntimeError("validation is not complete")
    if complete.get("test_profile_values_loaded") is not False or complete.get("test_evaluated") is not False:
        raise RuntimeError("test was not isolated from validation")
    if complete.get("role_manifest_sha256") != EXPECTED_ROLE_SHA256:
        raise RuntimeError("validation did not use the declared role freeze")
    if complete.get("checkpoint_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("checkpoint manifest changed after validation completion")
    expected = {(budget, arm, seed) for budget in BUDGETS for arm in ARMS for seed in SEEDS}
    found: dict[tuple[int, str, int], dict[str, Any]] = {}
    for item in manifest.get("checkpoints", []):
        key = (int(item["budget"]), str(item["arm"]), int(item["seed"]))
        path = Path(item["path"])
        if key in found or not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"checkpoint hash/path failure for {key}")
        found[key] = item
    if set(found) != expected:
        raise RuntimeError("checkpoint set differs from the frozen validation specification")
    return {"complete": complete, "manifest": manifest, "items": found}


def compound_vectors(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if np.isfinite(value):
            grouped[str(row["compound_id"])].append(value)
    return {compound: float(np.mean(values)) for compound, values in grouped.items()}


def paired_contrast(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], metric: str, seed: int
) -> dict[str, Any]:
    left, right = compound_vectors(left_rows, metric), compound_vectors(right_rows, metric)
    compounds = sorted(set(left) & set(right))
    if len(compounds) < 50 or set(left) != set(right):
        raise RuntimeError(f"non-identical or insufficient compound identities for {metric}")
    differences = np.asarray([left[item] - right[item] for item in compounds], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(BOOTSTRAP_REPS, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    return {
        "metric": metric, "n_compounds": len(compounds), "contrast_mean": float(differences.mean()),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
    }


def load_frozen_cfra(path: Path) -> dict[int, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("method", "")).startswith("CFRA"):
                by_budget[int(row["budget"])].append(row)
    if set(by_budget) != set(BUDGETS):
        raise RuntimeError("frozen test per-condition file lacks CFRA at every requested budget")
    return by_budget


def main() -> None:
    args = parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite {args.outdir}")
    freeze = load_validation_freeze(args)
    manifest_path = args.roles_root / "FROZEN_ROLE_MANIFEST.csv"
    if sha256_file(manifest_path) != EXPECTED_ROLE_SHA256:
        raise RuntimeError("role manifest changed since validation")
    torch, _ = torch_modules()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rtdl, tabm = load_official_modules(args.vendor_root, args.vendor_deps)
    reference = load_module("architecture_test_reference", args.reference_script)
    # The first and only profile-value load in this stage is the historical test split.
    test_profiles = reference.load_profiles(args.test_rows_root, "test")
    frozen_cfra = load_frozen_cfra(args.frozen_cfra_per_condition)
    args.outdir.mkdir(parents=True)
    summary: list[dict[str, Any]] = []
    per_condition: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for budget in BUDGETS:
        roles = reference.read_roles(args.roles_root, "test", budget)
        support, held, compounds, foreign, conditions, _ = reference.assemble(roles, test_profiles, foreign=True)
        support, held = finite_matrix("test support", support), finite_matrix("test held", held)
        for arm in ARMS:
            seed_predictions: list[np.ndarray] = []
            for seed in SEEDS:
                item = freeze["items"][(budget, arm, seed)]
                payload = torch.load(Path(item["path"]), map_location="cpu")
                if payload.get("version") != VERSION or payload.get("arm") != arm or int(payload.get("budget")) != budget:
                    raise RuntimeError(f"checkpoint identity mismatch: {item['path']}")
                model = build_model(arm, rtdl, tabm, device)
                model.load_state_dict(payload["state_dict"], strict=True)
                normalizer = payload["normalizer"]
                prediction = predict(
                    arm, model,
                    normalize(support, normalizer["x_mean"], normalizer["x_scale"]), torch, device,
                )
                seed_predictions.append(finite_matrix(
                    "unscaled test prediction", prediction * normalizer["y_scale"] + normalizer["y_mean"]
                ))
                del model
                torch.cuda.empty_cache()
            prediction = finite_matrix("test seed ensemble", np.mean(seed_predictions, axis=0, dtype=np.float32))
            metrics = reference.method_metrics(prediction, held, foreign, support)
            row = reference.score_row(arm, budget, metrics, compounds)
            row.update({"version": VERSION, "split": "test", "claim_status": "supportive_nonblind_frozen_recomputation"})
            summary.append(row)
            records: list[dict[str, Any]] = []
            for index, condition in enumerate(conditions):
                records.append({
                    "version": VERSION, "split": "test", "budget": budget, "method": arm,
                    "condition_id": str(condition), "compound_id": str(compounds[index]),
                    **{metric: float(values[index]) for metric, values in metrics.items()},
                })
            per_condition[(budget, arm)] = records
            del seed_predictions, prediction
            torch.cuda.empty_cache()
    contrasts: list[dict[str, Any]] = []
    metrics = ("E", "z_same", "z_foreign", "pcc", "top25_overlap", "top25_direction", "norm_ratio")
    for budget in BUDGETS:
        for arm in ARMS:
            if arm == "MLP-control":
                continue
            for metric_index, metric in enumerate(metrics):
                value = paired_contrast(
                    per_condition[(budget, arm)], per_condition[(budget, "MLP-control")], metric,
                    seed=3407 + 100 * budget + metric_index,
                )
                value.update({"budget": budget, "left": arm, "right": "MLP-control", "comparison": "modern_minus_capacity_matched_MLP"})
                contrasts.append(value)
        for arm in ARMS:
            for metric_index, metric in enumerate(metrics):
                value = paired_contrast(
                    frozen_cfra[budget], per_condition[(budget, arm)], metric,
                    seed=4407 + 100 * budget + metric_index,
                )
                value.update({"budget": budget, "left": "CFRA", "right": arm, "comparison": "frozen_CFRA_minus_modern"})
                contrasts.append(value)
    flat_per_condition = [row for rows in per_condition.values() for row in rows]
    write_csv(args.outdir / "TEST_METHOD_SUMMARY.csv", summary)
    write_csv(args.outdir / "TEST_PER_CONDITION_METRICS.csv", flat_per_condition)
    write_csv(args.outdir / "TEST_PAIRED_COMPOUND_BOOTSTRAP.csv", contrasts)
    save_json(args.outdir / "TEST_COMPLETE.json", {
        "status": "PASS", "version": VERSION, "stage": "historical-nonblind-test-once",
        "claim_status": "supportive_nonblind_frozen_recomputation", "test_profile_values_loaded": True,
        "validation_checkpoint_manifest_sha256": sha256_file(args.validation_dir / "CHECKPOINT_MANIFEST.json"),
        "role_manifest_sha256": EXPECTED_ROLE_SHA256, "budgets": list(BUDGETS), "arms": list(ARMS),
        "seeds": list(SEEDS), "bootstrap_reps": BOOTSTRAP_REPS,
        "test_rows_file_sha256": sha256_file(args.test_rows_root / "test_cp_plate_rows.npz"),
        "frozen_cfra_per_condition_sha256": sha256_file(args.frozen_cfra_per_condition),
    })


if __name__ == "__main__":
    main()
