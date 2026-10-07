#!/usr/bin/env python3
"""One historical, non-blind BBBC047 test recomputation after native freeze."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from native_common import (
    ARMS,
    BUDGETS,
    EXPECTED_ROLE_SHA256,
    SEEDS,
    VERSION,
    build_model,
    load_module,
    load_official_modules,
    model_specs,
    prediction,
    save_json,
    sha256_file,
    torch_modules,
)


BOOTSTRAP_REPS = 10_000
COMPACT_BY_NATIVE = {
    "ResNet-wide512": "ResNet",
    "FT-Transformer-5block-default": "FT-Transformer",
    "TabM-native-k32": "TabM",
}
EXPECTED_TEST_SHA256 = "64104430bfb3fde4bf34d3f754a3d8d611e11c6cac1e94cf95bfce4c8d6835f0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles-root", type=Path, required=True)
    parser.add_argument("--test-rows-root", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--vendor-deps", type=Path, required=True)
    parser.add_argument("--reference-script", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--compact-per-condition", type=Path, required=True)
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
            chunks.append(prediction(name, model, batch).detach().cpu().numpy().astype(np.float32))
    return finite_matrix(name, np.concatenate(chunks, axis=0))


def load_validation_freeze(args: argparse.Namespace) -> dict[tuple[int, str, int], dict[str, Any]]:
    complete_path = args.validation_dir / "NATIVE_VALIDATION_COMPLETE.json"
    manifest_path = args.validation_dir / "CHECKPOINT_MANIFEST.json"
    if not complete_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("validation completion marker/checkpoint manifest absent")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if complete.get("status") != "PASS" or complete.get("stage") != "validation-only":
        raise RuntimeError("native validation is not complete")
    if complete.get("test_profile_values_loaded") is not False or complete.get("test_evaluated") is not False:
        raise RuntimeError("test was not isolated from native validation")
    if complete.get("role_manifest_sha256") != EXPECTED_ROLE_SHA256:
        raise RuntimeError("native validation did not use the declared role freeze")
    if complete.get("checkpoint_manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("checkpoint manifest changed after native validation completion")
    expected = {(budget, arm, seed) for budget in BUDGETS for arm in ARMS for seed in SEEDS}
    found: dict[tuple[int, str, int], dict[str, Any]] = {}
    for item in manifest.get("checkpoints", []):
        key = (int(item["budget"]), str(item["arm"]), int(item["seed"]))
        path = Path(item["path"])
        if key in found or not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"checkpoint hash/path failure for {key}")
        found[key] = item
    if set(found) != expected:
        raise RuntimeError("checkpoint set differs from the frozen native specification")
    return found


def input_audit(args: argparse.Namespace) -> dict[str, Any]:
    files = {path.name for path in args.test_rows_root.glob("*_cp_plate_rows.npz")}
    if "test_cp_plate_rows.npz" not in files:
        raise RuntimeError(f"historical test profile is absent from {args.test_rows_root}")
    test_path = args.test_rows_root / "test_cp_plate_rows.npz"
    observed = sha256_file(test_path)
    if observed != EXPECTED_TEST_SHA256:
        raise RuntimeError("historical test profile hash mismatch")
    roles_manifest = args.roles_root / "FROZEN_ROLE_MANIFEST.csv"
    if sha256_file(roles_manifest) != EXPECTED_ROLE_SHA256:
        raise RuntimeError("role manifest changed since validation")
    return {
        "test_rows_file_sha256": observed,
        "role_manifest_sha256": EXPECTED_ROLE_SHA256,
        "test_rows_root_profile_files": sorted(files),
        "test_loader_scope": "reference.load_profiles(rows_root, 'test') opens only test_cp_plate_rows.npz",
        "non_test_profile_files_present_but_not_loaded": sorted(files - {"test_cp_plate_rows.npz"}),
    }


def compound_vectors(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[metric])
        if np.isfinite(value):
            grouped[str(row["compound_id"])].append(value)
    return {compound: float(np.mean(values)) for compound, values in grouped.items()}


def paired_contrast(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], metric: str, seed: int) -> dict[str, Any]:
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


def load_method_rows(path: Path, expected_methods: set[str]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = str(row.get("method", ""))
            if method in expected_methods:
                result[(int(row["budget"]), method)].append(row)
    expected = {(budget, method) for budget in BUDGETS for method in expected_methods}
    if set(result) != expected:
        raise RuntimeError(f"comparison file lacks expected method/budget rows: {path}")
    return result


def main() -> None:
    args = parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite {args.outdir}")
    checkpoints = load_validation_freeze(args)
    test_audit = input_audit(args)
    compact = load_method_rows(args.compact_per_condition, set(COMPACT_BY_NATIVE.values()))
    cfra = load_method_rows(args.frozen_cfra_per_condition, {"CFRA"})
    torch, _ = torch_modules()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rtdl, tabm = load_official_modules(args.vendor_root, args.vendor_deps)
    reference = load_module("native_test_reference", args.reference_script)
    # First and only profile-value load in this stage: the historical test split.
    test_profiles = reference.load_profiles(args.test_rows_root, "test")
    args.outdir.mkdir(parents=True)
    summary: list[dict[str, Any]] = []
    native_per_condition: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for budget in BUDGETS:
        roles = reference.read_roles(args.roles_root, "test", budget)
        support, held, compounds, foreign, conditions, _ = reference.assemble(roles, test_profiles, foreign=True)
        support, held = finite_matrix("test support", support), finite_matrix("test held", held)
        for arm in ARMS:
            seed_predictions: list[np.ndarray] = []
            for seed in SEEDS:
                item = checkpoints[(budget, arm, seed)]
                payload = torch.load(Path(item["path"]), map_location="cpu")
                if payload.get("version") != VERSION or payload.get("arm") != arm or int(payload.get("budget")) != budget or int(payload.get("seed")) != seed:
                    raise RuntimeError(f"checkpoint identity mismatch: {item['path']}")
                model = build_model(arm, rtdl, tabm, device)
                model.load_state_dict(payload["state_dict"], strict=True)
                normalizer = payload["normalizer"]
                values = predict(arm, model, normalize(support, normalizer["x_mean"], normalizer["x_scale"]), torch, device)
                seed_predictions.append(finite_matrix("unscaled test prediction", values * normalizer["y_scale"] + normalizer["y_mean"]))
                del model
                torch.cuda.empty_cache()
            mean_prediction = finite_matrix("test seed ensemble", np.mean(seed_predictions, axis=0, dtype=np.float32))
            metrics = reference.method_metrics(mean_prediction, held, foreign, support)
            row = reference.score_row(arm, budget, metrics, compounds)
            row.update({"version": VERSION, "split": "test", "claim_status": "supportive_nonblind_frozen_recomputation"})
            summary.append(row)
            records: list[dict[str, Any]] = []
            for index, condition_id in enumerate(conditions):
                records.append({"version": VERSION, "split": "test", "budget": budget, "method": arm,
                                "condition_id": str(condition_id), "compound_id": str(compounds[index]),
                                **{metric: float(values[index]) for metric, values in metrics.items()}})
            native_per_condition[(budget, arm)] = records
            del seed_predictions, mean_prediction
            torch.cuda.empty_cache()
    contrasts: list[dict[str, Any]] = []
    metrics = ("E", "z_same", "z_foreign", "pcc", "top25_overlap", "top25_direction", "norm_ratio")
    for budget in BUDGETS:
        for arm in ARMS:
            for metric_index, metric in enumerate(metrics):
                native_minus_compact = paired_contrast(
                    native_per_condition[(budget, arm)], compact[(budget, COMPACT_BY_NATIVE[arm])], metric,
                    seed=3407 + 100 * budget + metric_index,
                )
                native_minus_compact.update({"budget": budget, "left": arm, "right": COMPACT_BY_NATIVE[arm],
                                             "comparison": "native_size_minus_corresponding_compact"})
                contrasts.append(native_minus_compact)
                cfra_minus_native = paired_contrast(
                    cfra[(budget, "CFRA")], native_per_condition[(budget, arm)], metric,
                    seed=4407 + 100 * budget + metric_index,
                )
                cfra_minus_native.update({"budget": budget, "left": "CFRA", "right": arm,
                                          "comparison": "frozen_CFRA_minus_native_size"})
                contrasts.append(cfra_minus_native)
    flat_per_condition = [row for rows in native_per_condition.values() for row in rows]
    write_csv(args.outdir / "NATIVE_TEST_METHOD_SUMMARY.csv", summary)
    write_csv(args.outdir / "NATIVE_TEST_PER_CONDITION_METRICS.csv", flat_per_condition)
    write_csv(args.outdir / "NATIVE_TEST_PAIRED_COMPOUND_BOOTSTRAP.csv", contrasts)
    save_json(args.outdir / "NATIVE_TEST_COMPLETE.json", {
        "status": "PASS", "version": VERSION, "stage": "historical-nonblind-test-once",
        "claim_status": "supportive_nonblind_frozen_recomputation", "test_profile_values_loaded": True,
        "validation_checkpoint_manifest_sha256": sha256_file(args.validation_dir / "CHECKPOINT_MANIFEST.json"),
        **test_audit, "budgets": list(BUDGETS), "arms": list(ARMS), "seeds": list(SEEDS),
        "bootstrap_reps": BOOTSTRAP_REPS, "compact_per_condition_sha256": sha256_file(args.compact_per_condition),
        "frozen_cfra_per_condition_sha256": sha256_file(args.frozen_cfra_per_condition),
    })


if __name__ == "__main__":
    main()
