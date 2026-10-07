#!/usr/bin/env python3
"""One frozen, non-blind BBBC047 test evaluation after selection freeze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


HERE = Path(__file__).resolve().parent


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


EXT = load_module("test_external", (Path(__file__).resolve().parents[4] / "analysis/benchmarks/bbbc047/repeat_estimator/run_external_validation.py"))
N2S = load_module("test_n2s", (Path(__file__).resolve().parents[4] / "analysis/benchmarks/bbbc047/repeat_estimator/run_noise2self_validation.py"))
REPERT = load_module("test_repert", (Path(__file__).resolve().parents[4] / "analysis/benchmarks/bbbc047/repeat_estimator/run_repert_validation.py"))
STAGE, RCEB = REPERT.STAGE, REPERT.RCEB

VERSION = "BBBC047-repeat-estimator-test-once-v1-2026-09-16"
BUDGETS = (1, 2, 3)
BOOTSTRAP_ROUNDS = 10_000
METHODS = ("Mean", "Median", "MODZ", "PCA", "Ridge", "Noise2Self", "IMR", "LSO", "IMCEB", "CFRA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rows-root", type=Path, default=EXT.DEFAULT_ROWS)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--noise2self-root", type=Path, required=True)
    parser.add_argument("--repert-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preflight-only", action="store_true", help="validate frozen prerequisites and train-only refits without opening test values")
    return parser.parse_args()


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{VERSION}|{label}".encode()).digest()[:8], "big") % (2**63 - 1)


def read_marker(path: Path, expected_stage: str, protocol: Path, role_hash: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS" or result.get("stage") != expected_stage:
        raise RuntimeError(f"invalid stage marker: {path}")
    if result.get("protocol_sha256") != EXT.sha256_file(protocol) or result.get("role_manifest_sha256") != role_hash:
        raise RuntimeError(f"hash mismatch in prerequisite: {path}")
    if result.get("test_loaded") is not False or result.get("test_used_for_selection") is not False:
        raise RuntimeError(f"test isolation failure in prerequisite: {path}")
    return result


def read_roles(root: Path, split: str, budget: int, require_foreign: bool) -> list[dict[str, str]]:
    with (root / "FROZEN_ROLE_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        output = [dict(row) for row in csv.DictReader(handle) if row["split"] == split and int(row["budget"]) == budget and (not require_foreign or row["foreign_ok"] == "True")]
    if len(output) < 500:
        raise RuntimeError(f"insufficient frozen {split} roles at {budget}R")
    return output


def load_mlp(path: Path, device: Any) -> Any:
    torch, nn = STAGE.torch_modules()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = STAGE._make_model(nn, 775, 256, 32).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return STAGE.FitResult(model, np.asarray(payload["input_mean"], dtype=np.float32), np.asarray(payload["input_scale"], dtype=np.float32), np.asarray(payload["target_mean"], dtype=np.float32), np.asarray(payload["target_scale"], dtype=np.float32), int(payload["meta"]["selected_epoch_zero_based"]), float(payload["meta"]["selected_inner_loss"]), int(payload["meta"]["final_epochs"]))


def compound_values(values: np.ndarray, compounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for compound, value in zip(compounds, values):
        if np.isfinite(value):
            grouped[str(compound)].append(float(value))
    ids = np.asarray(sorted(grouped), dtype=str)
    return ids, np.asarray([np.mean(grouped[key]) for key in ids], dtype=np.float64)


def bootstrap_summary(values: np.ndarray, compounds: np.ndarray, label: str) -> tuple[float, float, float, int]:
    _, reduced = compound_values(values, compounds)
    if not len(reduced):
        return float("nan"), float("nan"), float("nan"), 0
    point = float(np.mean(reduced))
    rng = np.random.default_rng(stable_seed(f"summary|{label}"))
    draws = rng.integers(0, len(reduced), size=(BOOTSTRAP_ROUNDS, len(reduced)))
    sampled = reduced[draws].mean(axis=1)
    return point, float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975)), len(reduced)


def bootstrap_delta(left: np.ndarray, right: np.ndarray, compounds: np.ndarray, label: str) -> tuple[float, float, float, int]:
    left_ids, left_values = compound_values(left, compounds)
    right_ids, right_values = compound_values(right, compounds)
    if not np.array_equal(left_ids, right_ids):
        raise RuntimeError(f"paired compound mismatch in {label}")
    values = left_values - right_values
    rng = np.random.default_rng(stable_seed(f"delta|{label}"))
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_ROUNDS, len(values)))
    sampled = values[draws].mean(axis=1)
    return float(np.mean(values)), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975)), len(values)


def summary_rows(budget: int, method: str, metrics: dict[str, np.ndarray], compounds: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for metric in ("E", "z_same", "z_foreign", "pcc", "top25_overlap", "top25_direction", "norm_ratio"):
        point, low, high, n = bootstrap_summary(metrics[metric], compounds, f"b{budget}|{method}|{metric}")
        rows.append({"budget": budget, "method": method, "metric": metric, "point": point, "ci_low": low, "ci_high": high, "n_compounds": n, "bootstrap_rounds": BOOTSTRAP_ROUNDS, "bootstrap_unit": "compound after dose-condition aggregation"})
    return rows


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
    external = read_marker(args.external_root / "EXTERNAL_VALIDATION_COMPLETE.json", "external_validation", args.protocol, role_hash)
    noise = read_marker(args.noise2self_root / "NOISE2SELF_VALIDATION_COMPLETE.json", "noise2self_validation", args.protocol, role_hash)
    freeze = read_marker(args.repert_root / "SELECTION_FREEZE.json", "selection_freeze", args.protocol, role_hash)
    if freeze.get("selection_frozen") is not True:
        raise RuntimeError("selection freeze not sealed")
    if args.device == "auto":
        torch, _ = STAGE.torch_modules()
        device = STAGE.select_device(torch, "auto")
    else:
        torch, _ = STAGE.torch_modules()
        device = STAGE.select_device(torch, args.device)
    # Complete every train-only refit and checkpoint integrity check before
    # opening any test profile values.
    with (args.repert_root / "CFRA_VALIDATION_WEIGHTS.json").open(encoding="utf-8") as handle:
        cfra_weights = json.load(handle)
    champion = freeze["external_champion"]["champion"]
    if champion not in {"Mean", "Median", "MODZ", "PCA", "Ridge", "Noise2Self"}:
        raise RuntimeError(f"invalid frozen external champion {champion}")
    train_profiles = EXT.load_profiles(args.rows_root, "train")
    imceb_model = RCEB.load_model(args.repert_root / "IMCEB_TRAIN_ONLY_MODEL.npz")
    prepared: dict[int, dict[str, Any]] = {}
    for budget in BUDGETS:
        train_roles = read_roles(args.roles_root, "train", budget, False)
        train_x, train_held, _, _, _, _ = EXT.assemble(train_roles, train_profiles, foreign=False)
        parameters = external["selected_parameters"][str(budget)]
        pca = PCA(n_components=int(parameters["pca_rank"]), svd_solver="randomized", random_state=3407).fit(train_x)
        x_mean, x_scale = train_x.mean(axis=0), np.maximum(train_x.std(axis=0), 1e-6)
        ridge = Ridge(alpha=float(parameters["ridge_alpha"]), fit_intercept=True, solver="svd").fit((train_x - x_mean) / x_scale, train_held)
        imr_fits, lso_fits = [], []
        for seed in REPERT.SEEDS:
            imr_path, lso_path = args.repert_root / f"budget{budget}" / f"IMR_seed{seed}.pt", args.repert_root / f"budget{budget}" / f"LSO_seed{seed}.pt"
            expected_imr = freeze["model_hashes"].get(f"b{budget}/IMR/seed{seed}")
            expected_lso = freeze["model_hashes"].get(f"b{budget}/LSO/seed{seed}")
            if EXT.sha256_file(imr_path) != expected_imr or EXT.sha256_file(lso_path) != expected_lso:
                raise RuntimeError(f"frozen checkpoint hash mismatch b{budget} seed{seed}")
            imr_fits.append(load_mlp(imr_path, device))
            lso_fits.append(load_mlp(lso_path, device))
        n2s_path = Path(noise["selected"][str(budget)]["checkpoint"])
        if EXT.sha256_file(n2s_path) != noise["selected"][str(budget)]["checkpoint_sha256"]:
            raise RuntimeError(f"Noise2Self checkpoint hash mismatch b{budget}")
        n2s_payload = torch.load(n2s_path, map_location="cpu", weights_only=False)
        n2s_model = N2S.MaskedMLP().to(device)
        n2s_model.load_state_dict(n2s_payload["state_dict"], strict=True)
        prepared[budget] = {"pca": pca, "ridge": ridge, "x_mean": x_mean, "x_scale": x_scale, "imr": imr_fits, "lso": lso_fits, "n2s": n2s_model, "n2s_payload": n2s_payload}
    if args.preflight_only:
        print(json.dumps({"stage": "test_preflight", "protocol_sha256": EXT.sha256_file(args.protocol), "external_champion": champion, "prepared_budgets": list(prepared), "test_loaded": False, "test_profile_values_loaded": False}, indent=2, sort_keys=True))
        return
    # This is the single test-profile opening point.  No test-dependent setting
    # can reach any model, rank, weight, or champion after this line.
    test_profiles = EXT.load_profiles(args.rows_root, "test")
    args.outdir.mkdir(parents=True)
    summaries: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    per_condition: list[dict[str, Any]] = []
    macro_values: dict[str, list[np.ndarray]] = {"E": [], "z_same": []}
    for budget in BUDGETS:
        roles = read_roles(args.roles_root, "test", budget, True)
        x, held, compounds, foreign, conditions, raw_support = EXT.assemble(roles, test_profiles, foreign=True)
        fit = prepared[budget]
        pca = fit["pca"]
        pca_prediction = (pca.mean_ + ((x - pca.mean_) @ pca.components_.T) @ pca.components_).astype(np.float32)
        ridge_prediction = np.asarray(fit["ridge"].predict((x - fit["x_mean"]) / fit["x_scale"]), dtype=np.float32)
        imr = np.mean([STAGE.predict(torch, item, x, REPERT.BATCH_SIZE, device) for item in fit["imr"]], axis=0).astype(np.float32)
        lso = np.mean([STAGE.predict(torch, item, x, REPERT.BATCH_SIZE, device) for item in fit["lso"]], axis=0).astype(np.float32)
        n2s = N2S.jinvariant_predict(fit["n2s"], x, float(fit["n2s_payload"]["mask_rate"]), np.asarray(fit["n2s_payload"]["mean"], dtype=np.float32), np.asarray(fit["n2s_payload"]["scale"], dtype=np.float32), device)
        imceb = RCEB.rceb_predict(imceb_model, x, budget)
        weights = cfra_weights[str(budget)]["weights"]
        cfra = (float(weights["IMR"]) * imr + float(weights["LSO"]) * lso + float(weights["IMCEB"]) * imceb).astype(np.float32)
        predictions = {"Mean": x, "Median": np.median(raw_support, axis=1).astype(np.float32), "MODZ": EXT.modz(raw_support), "PCA": pca_prediction, "Ridge": ridge_prediction, "Noise2Self": n2s, "IMR": imr, "LSO": lso, "IMCEB": imceb, "CFRA": cfra}
        metrics_by_method: dict[str, dict[str, np.ndarray]] = {}
        for method in METHODS:
            metrics = EXT.method_metrics(predictions[method], held, foreign, x)
            metrics_by_method[method] = metrics
            summaries.extend(summary_rows(budget, method, metrics, compounds))
            for index, condition in enumerate(conditions):
                per_condition.append({"budget": budget, "method": method, "compound_id": compounds[index], "condition_id": condition, **{name: float(values[index]) for name, values in metrics.items()}})
        for metric in ("E", "z_same", "z_foreign", "pcc", "top25_overlap", "top25_direction", "norm_ratio"):
            point, low, high, n = bootstrap_delta(metrics_by_method["CFRA"][metric], metrics_by_method[champion][metric], compounds, f"b{budget}|CFRA-{champion}|{metric}")
            contrasts.append({"budget": budget, "comparison": f"CFRA-{champion}", "metric": metric, "point": point, "ci_low": low, "ci_high": high, "n_compounds": n, "bootstrap_rounds": BOOTSTRAP_ROUNDS, "bootstrap_unit": "paired compound"})
        for metric in macro_values:
            _, cfra_compound = compound_values(metrics_by_method["CFRA"][metric], compounds)
            _, champion_compound = compound_values(metrics_by_method[champion][metric], compounds)
            macro_values[metric].append(cfra_compound - champion_compound)
        np.savez_compressed(args.outdir / f"budget{budget}_TEST_PREDICTIONS.npz", compound=compounds, condition=conditions, **predictions)
    macro_rows = []
    for metric, values in macro_values.items():
        rng = np.random.default_rng(stable_seed(f"macro|{metric}"))
        draws = []
        for value in values:
            draw = rng.integers(0, len(value), size=(BOOTSTRAP_ROUNDS, len(value)))
            draws.append(value[draw].mean(axis=1))
        sampled = np.mean(np.vstack(draws), axis=0)
        point = float(np.mean([np.mean(value) for value in values]))
        macro_rows.append({"comparison": f"CFRA-{champion}", "metric": metric, "point": point, "ci_low": float(np.quantile(sampled, 0.025)), "ci_high": float(np.quantile(sampled, 0.975)), "budget_aggregation": "equal 1R/2R/3R", "bootstrap_rounds": BOOTSTRAP_ROUNDS, "bootstrap_unit": "independent paired compounds within each budget"})
    macro = {row["metric"]: row for row in macro_rows}
    decision = "higher_perturbation_specificity_only"
    if macro["E"]["ci_low"] > 0 and macro["z_same"]["point"] >= 0:
        decision = "response_recovery_supportive_nonblind"
    elif macro["E"]["ci_high"] <= 0:
        decision = "CFRA_not_supported_vs_external_champion"
    write_csv(args.outdir / "TEST_METHOD_SUMMARY.csv", summaries)
    write_csv(args.outdir / "TEST_CFRA_CHAMPION_CONTRASTS.csv", contrasts)
    write_csv(args.outdir / "TEST_MACRO_CONTRASTS.csv", macro_rows)
    write_csv(args.outdir / "TEST_PER_CONDITION_METRICS.csv", per_condition)
    marker = {"version": VERSION, "stage": "test_once", "protocol_sha256": EXT.sha256_file(args.protocol), "role_manifest_sha256": role_hash, "historical_test_status": "previously exposed before this benchmark; this is a frozen non-blind recomputation, not new blind confirmation", "external_champion": champion, "cfra_weights": cfra_weights, "test_loaded": True, "test_used_for_selection": False, "test_profile_values_loaded": True, "bootstrap": {"rounds": BOOTSTRAP_ROUNDS, "unit": "compound after condition aggregation", "ci": "95% percentile"}, "primary": {"metric": "E", "macro_contrast": macro["E"], "z_same_macro_contrast": macro["z_same"], "decision": decision}, "status": "PASS"}
    (args.outdir / "TEST_COMPLETE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": marker["stage"], "external_champion": champion, "decision": decision, "E_macro": macro["E"], "z_same_macro": macro["z_same"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
