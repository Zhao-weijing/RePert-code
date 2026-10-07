#!/usr/bin/env python3
"""Dose-aware validation fitting for direct IMR, LSO, IMCEB and CFRA."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
EXPERIMENTS = Path(__file__).resolve().parents[4] / "analysis"


def load_module(name: str, path: Path) -> Any:
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EXT = load_module("benchmark_external", (Path(__file__).resolve().parents[4] / "analysis/benchmarks/bbbc047/repeat_estimator/run_external_validation.py"))
STAGE = load_module("benchmark_stage_c", (Path(__file__).resolve().parents[4] / "analysis/repeat_calibration/run_stage_c_strong_baselines.py"))
RCEB = load_module("benchmark_rceb", (Path(__file__).resolve().parents[4] / "analysis/repeat_calibration/run_rceb_screen.py"))

VERSION = "BBBC047-repeat-estimator-repert-validation-v1-2026-09-16"
BUDGETS = (1, 2, 3)
SEEDS = (3407, 42, 2025)
EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
RCEB_RIDGE_RELATIVE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rows-root", type=Path, default=EXT.DEFAULT_ROWS)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--noise2self-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def all_roles(root: Path, split: str, budget: int, *, require_foreign: bool) -> list[dict[str, str]]:
    with (root / "FROZEN_ROLE_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row["split"] == split and int(row["budget"]) == budget and (not require_foreign or row["foreign_ok"] == "True")]
    if len(rows) < 500:
        raise RuntimeError(f"insufficient {split} roles b{budget}, foreign={require_foreign}")
    return rows


def lso_labels(rows: list[dict[str, str]], profiles: dict[tuple[str, str], dict[str, np.ndarray]]) -> tuple[np.ndarray, dict[str, Any]]:
    labels: list[np.ndarray] = []
    plate_counts: list[int] = []
    for row in rows:
        key = (row["compound_id"], row["dose"])
        by_plate = profiles.get(key)
        if by_plate is None:
            raise RuntimeError(f"missing LSO condition {key}")
        support = set(row["support_plate_ids"].split("|"))
        legal = [plate for plate in sorted(by_plate) if plate not in support]
        if support & set(legal) or not legal:
            raise RuntimeError(f"invalid LSO support/label partition: {row['condition_id']}")
        labels.append(np.mean([by_plate[plate] for plate in legal], axis=0, dtype=np.float64).astype(np.float32))
        plate_counts.append(len(legal))
    return np.vstack(labels), {"definition": "mean of all physical plates excluding support", "label_plate_count_min": min(plate_counts), "label_plate_count_mean": float(np.mean(plate_counts)), "label_plate_count_max": max(plate_counts), "support_label_overlap": 0}


def save_fit(path: Path, fit: Any, meta: dict[str, Any], *, budget: int, method: str, seed: int) -> str:
    torch, _ = STAGE.torch_modules()
    payload = {"version": VERSION, "budget": budget, "method": method, "seed": seed, "state_dict": {key: value.detach().cpu() for key, value in fit.model.state_dict().items()}, "input_mean": fit.input_mean, "input_scale": fit.input_scale, "target_mean": fit.target_mean, "target_scale": fit.target_scale, "meta": meta}
    torch.save(payload, path)
    return EXT.sha256_file(path)


def fit_cfra(candidates: dict[str, np.ndarray], held: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    names = ("IMR", "LSO", "IMCEB")
    stacked = np.stack([np.asarray(candidates[name], dtype=np.float64) for name in names], axis=2)
    target = np.asarray(held, dtype=np.float64)

    def objective(weight: np.ndarray) -> float:
        prediction = np.einsum("ndk,k->nd", stacked, weight)
        return float(np.mean((prediction - target) ** 2))

    result = minimize(objective, x0=np.full(3, 1.0 / 3.0), method="SLSQP", bounds=[(0.0, 1.0)] * 3, constraints={"type": "eq", "fun": lambda value: float(np.sum(value) - 1.0)}, options={"ftol": 1e-12, "maxiter": 1000})
    if not result.success or np.any(result.x < -1e-10) or not np.isclose(result.x.sum(), 1.0, atol=1e-8):
        raise RuntimeError(f"CFRA constrained fit failed: {result.message}")
    weights = np.maximum(result.x, 0.0)
    weights /= weights.sum()
    return weights, {"candidate_order": list(names), "objective": "validation independent-held profile MSE", "constraint": "non-negative simplex", "success": bool(result.success), "iterations": int(result.nit), "objective_value": objective(weights), "weights": {name: float(weight) for name, weight in zip(names, weights)}}


def load_marker(path: Path, expected: str, protocol: Path, role_hash: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    marker = json.loads(path.read_text(encoding="utf-8"))
    if marker.get("status") != "PASS" or marker.get("stage") != expected:
        raise RuntimeError(f"invalid prerequisite marker {path}")
    if marker.get("protocol_sha256") != EXT.sha256_file(protocol) or marker.get("role_manifest_sha256") != role_hash:
        raise RuntimeError(f"prerequisite hash mismatch {path}")
    if marker.get("test_loaded") is not False or marker.get("test_used_for_selection") is not False:
        raise RuntimeError(f"prerequisite violates test barrier {path}")
    return marker


def external_champion(external_root: Path, noise_root: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with (external_root / "VALIDATION_EXTERNAL_SUMMARY.csv").open(newline="", encoding="utf-8") as handle:
        rows.extend(dict(row) for row in csv.DictReader(handle))
    noise_marker = json.loads((noise_root / "NOISE2SELF_VALIDATION_COMPLETE.json").read_text(encoding="utf-8"))
    selected_rates = {int(budget): float(value["mask_rate"]) for budget, value in noise_marker["selected"].items()}
    with (noise_root / "VALIDATION_NOISE2SELF_SUMMARY.csv").open(newline="", encoding="utf-8") as handle:
        rows.extend(dict(row) for row in csv.DictReader(handle) if float(row["mask_rate"]) == selected_rates[int(row["budget"])])
    by_method: dict[str, dict[int, float]] = {}
    for row in rows:
        method, budget = row["method"], int(row["budget"])
        by_method.setdefault(method, {})[budget] = float(row["E_mean"])
    valid = {method: values for method, values in by_method.items() if set(values) == set(BUDGETS)}
    if set(valid) != {"Mean", "Median", "MODZ", "PCA", "Ridge", "Noise2Self"}:
        raise RuntimeError(f"missing external candidate metrics: {sorted(valid)}")
    macro = {method: float(np.mean([values[budget] for budget in BUDGETS])) for method, values in valid.items()}
    champion = max(macro, key=lambda method: (macro[method], method))
    return {"candidates": valid, "macro_E_equal_budget_weight": macro, "champion": champion, "selection": "maximum validation E macro average across 1R/2R/3R; lexical tie break"}


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


def main() -> None:
    args = parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite {args.outdir}")
    role_audit = EXT.check_freeze(args.roles_root, args.protocol)
    role_hash = role_audit["manifest"]["sha256"]
    external_marker = load_marker(args.external_root / "EXTERNAL_VALIDATION_COMPLETE.json", "external_validation", args.protocol, role_hash)
    noise_marker = load_marker(args.noise2self_root / "NOISE2SELF_VALIDATION_COMPLETE.json", "noise2self_validation", args.protocol, role_hash)
    if args.device == "auto":
        torch, nn = STAGE.torch_modules()
        device = STAGE.select_device(torch, "auto")
    else:
        torch, nn = STAGE.torch_modules()
        device = STAGE.select_device(torch, args.device)
    train_profiles = EXT.load_profiles(args.rows_root, "train")
    valid_profiles = EXT.load_profiles(args.rows_root, "valid")
    # RCEB is fit once from the complete dose-aware training plate table.
    train_rows = RCEB.load_cp_rows("BBBC047", "train", args.rows_root, Path("unused"))
    imceb_model, imceb_metadata = RCEB.fit_rceb(train_rows, RCEB_RIDGE_RELATIVE)
    args.outdir.mkdir(parents=True)
    RCEB.save_model(args.outdir / "IMCEB_TRAIN_ONLY_MODEL.npz", imceb_model)
    (args.outdir / "IMCEB_TRAIN_ONLY_METADATA.json").write_text(json.dumps(imceb_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summaries: list[dict[str, Any]] = []
    model_hashes: dict[str, str] = {}
    cfra_by_budget: dict[str, Any] = {}
    for budget in BUDGETS:
        train_role_rows = all_roles(args.roles_root, "train", budget, require_foreign=False)
        valid_role_rows = all_roles(args.roles_root, "valid", budget, require_foreign=True)
        train_x, train_held, train_compounds, _, _, _ = EXT.assemble(train_role_rows, train_profiles, foreign=False)
        valid_x, valid_held, valid_compounds, valid_foreign, _, _ = EXT.assemble(valid_role_rows, valid_profiles, foreign=True)
        train_lso, lso_metadata = lso_labels(train_role_rows, train_profiles)
        imr_predictions: list[np.ndarray] = []
        lso_predictions: list[np.ndarray] = []
        budget_dir = args.outdir / f"budget{budget}"
        budget_dir.mkdir()
        for seed in SEEDS:
            imr_fit, imr_meta = STAGE.fit_baseline(torch, nn, train_x, train_held, train_held, train_compounds, name=f"benchmark-imr-b{budget}-seed{seed}", epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, device=device)
            lso_fit, lso_meta = STAGE.fit_baseline(torch, nn, train_x, train_lso, train_held, train_compounds, name=f"benchmark-lso-b{budget}-seed{seed}", epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, device=device)
            imr_path, lso_path = budget_dir / f"IMR_seed{seed}.pt", budget_dir / f"LSO_seed{seed}.pt"
            model_hashes[f"b{budget}/IMR/seed{seed}"] = save_fit(imr_path, imr_fit, imr_meta, budget=budget, method="IMR", seed=seed)
            model_hashes[f"b{budget}/LSO/seed{seed}"] = save_fit(lso_path, lso_fit, lso_meta | lso_metadata, budget=budget, method="LSO", seed=seed)
            imr_predictions.append(STAGE.predict(torch, imr_fit, valid_x, BATCH_SIZE, device))
            lso_predictions.append(STAGE.predict(torch, lso_fit, valid_x, BATCH_SIZE, device))
        candidates = {"IMR": np.mean(imr_predictions, axis=0).astype(np.float32), "LSO": np.mean(lso_predictions, axis=0).astype(np.float32), "IMCEB": RCEB.rceb_predict(imceb_model, valid_x, budget)}
        weights, cfra_meta = fit_cfra(candidates, valid_held)
        cfra = sum(weight * candidates[name] for weight, name in zip(weights, ("IMR", "LSO", "IMCEB"))).astype(np.float32)
        cfra_by_budget[str(budget)] = cfra_meta | {"weights": cfra_meta["weights"], "imr_lso_seed_ensemble": list(SEEDS), "lso_target": lso_metadata}
        for name, prediction in candidates.items():
            summaries.append(EXT.score_row(name, budget, EXT.method_metrics(prediction, valid_held, valid_foreign, valid_x), valid_compounds) | {"repert_seed_ensemble": "|".join(str(seed) for seed in SEEDS) if name in {"IMR", "LSO"} else "not_applicable"})
        summaries.append(EXT.score_row("CFRA", budget, EXT.method_metrics(cfra, valid_held, valid_foreign, valid_x), valid_compounds) | {"repert_seed_ensemble": "|".join(str(seed) for seed in SEEDS)})
    champion = external_champion(args.external_root, args.noise2self_root)
    (args.outdir / "CFRA_VALIDATION_WEIGHTS.json").write_text(json.dumps(cfra_by_budget, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.outdir / "EXTERNAL_CHAMPION.json").write_text(json.dumps(champion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.outdir / "VALIDATION_REPERT_SUMMARY.csv", summaries)
    marker = {"version": VERSION, "stage": "repert_validation", "protocol_sha256": EXT.sha256_file(args.protocol), "role_manifest_sha256": role_hash, "repert": {"seeds": list(SEEDS), "architecture": "775-256-32-256-775 GELU", "optimizer": "AdamW", "epochs_max": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "imceb_ridge_relative": RCEB_RIDGE_RELATIVE, "cfra": "validation independent-held MSE non-negative simplex"}, "model_hashes": model_hashes, "cfra": cfra_by_budget, "external_champion": champion, "prerequisites": {"external": str(args.external_root), "noise2self": str(args.noise2self_root), "external_protocol_sha256": external_marker["protocol_sha256"], "noise2self_protocol_sha256": noise_marker["protocol_sha256"]}, "test_loaded": False, "test_profile_values_loaded": False, "test_used_for_selection": False, "status": "PASS"}
    (args.outdir / "REPERT_VALIDATION_COMPLETE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.outdir / "SELECTION_FREEZE.json").write_text(json.dumps(marker | {"stage": "selection_freeze", "selection_frozen": True}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": marker["stage"], "external_champion": champion["champion"], "cfra_weights": {budget: value["weights"] for budget, value in cfra_by_budget.items()}, "test_loaded": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
