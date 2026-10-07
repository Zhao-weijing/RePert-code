#!/usr/bin/env python3
"""Strict selected-repeat 2R/3R/4R Raw-k versus per-view-CFRA M2 Student."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = Path(__file__).resolve().parents[4] / "analysis"
PREVIOUS = (Path(__file__).resolve().parents[4] / "analysis/predictor_supervision/bbbc047/support/run_repeat_budget_cfra_viewaggregate_m2.py")
spec = importlib.util.spec_from_file_location("strict_selected_base", PREVIOUS)
if spec is None or spec.loader is None:
    raise RuntimeError(PREVIOUS)
BASE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = BASE
spec.loader.exec_module(BASE)

VERSION = "BBBC047-strict-selected-repeat-1R-CFRA-M2-v1-2026-09-16"
SEED = 3407
BUDGETS = (2, 3, 4)
SEEDS = (3407, 42, 2025, 1337, 7331)
FOLDS = 5
CP_DIM = 775
M0 = "M0_RawKAggregate"
M2 = "M2_CFRA1ViewAggregate_Residual"
METHODS = (M0, M2)
DIMS = (256, 80)
BRANCH_PARAMETERS = 806279
TOTAL_PARAMETERS = 1612558


@dataclass
class Examples:
    compound: np.ndarray
    dose: np.ndarray
    plates: list[tuple[str, ...]]
    views: np.ndarray
    raw: np.ndarray
    foreign: np.ndarray
    foreign_ok: np.ndarray


def sha256(path: Path) -> str:
    return BASE.sha256_file(path)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "fit", "test"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, default=HERE / "PROTOCOL.md")
    parser.add_argument("--rows-root", type=Path, default=BASE.DEFAULT_ROWS)
    parser.add_argument("--model-h5", type=Path, default=BASE.DEFAULT_H5)
    parser.add_argument("--aggregate-npz", type=Path, default=BASE.DEFAULT_AGG)
    parser.add_argument("--split-lock", type=Path, default=BASE.DEFAULT_LOCK)
    parser.add_argument("--teacher-epochs", type=int, default=80)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.protocol_file.is_file():
        parser.error("protocol file is missing")
    if not args.smoke and (args.teacher_epochs, args.student_epochs, args.batch_size, args.bootstrap_rounds) != (80, 40, 256, 10000):
        parser.error("formal protocol locks epochs, batch size, and bootstrap rounds")
    return args


def select_examples(rows: Any, budget: int, split: str, foreign: bool) -> Examples:
    means = rows.plate_means()
    by_compound: dict[str, list[tuple[str, str]]] = {}
    for key, plates in means.items():
        if len(plates) >= budget:
            by_compound.setdefault(key[0], []).append(key)
    chosen: list[tuple[str, str]] = []
    for compound, candidates in sorted(by_compound.items()):
        chosen.append(min(candidates, key=lambda key: (BASE.LEGACY.stable_int(SEED, f"{VERSION}|condition|b{budget}|{split}|{compound}|{key[1]}"), key)))
    compounds: list[str] = []
    doses: list[str] = []
    selected: list[tuple[str, ...]] = []
    views: list[np.ndarray] = []
    for compound, dose in chosen:
        order = sorted(means[(compound, dose)], key=lambda plate: (BASE.LEGACY.stable_int(SEED, f"{VERSION}|plate|b{budget}|{split}|{compound}|{dose}|{plate}"), plate))
        plates = tuple(order[:budget])
        if len(plates) != budget:
            raise RuntimeError("selected budget changed")
        compounds.append(compound); doses.append(dose); selected.append(plates)
        views.append(np.stack([means[(compound, dose)][plate] for plate in plates], axis=0))
    tensor = np.stack(views).astype(np.float32)
    raw = tensor.mean(axis=1).astype(np.float32)
    foreign_values = np.full_like(raw, np.nan)
    foreign_ok = np.zeros(len(raw), dtype=bool)
    if foreign:
        index: dict[str, list[int]] = {}
        for i, dose in enumerate(doses):
            index.setdefault(dose, []).append(i)
        for i, (compound, dose) in enumerate(zip(compounds, doses)):
            donors = [j for j in index[dose] if compounds[j] != compound]
            if donors:
                j = donors[BASE.LEGACY.stable_int(SEED, f"{VERSION}|foreign|b{budget}|{split}|{compound}|{dose}") % len(donors)]
                foreign_values[i] = raw[j]; foreign_ok[i] = True
    return Examples(np.asarray(compounds, dtype=str), np.asarray(doses, dtype=str), selected, tensor, raw, foreign_values, foreign_ok)


def pairs(examples: Examples) -> Any:
    support: list[np.ndarray] = []
    target: list[np.ndarray] = []
    names: list[str] = []
    held: list[int] = []
    for i, (compound, views) in enumerate(zip(examples.compound, examples.views)):
        for j in range(views.shape[0]):
            support.append(views[j]); target.append(views[(j + 1) % views.shape[0]]); names.append(str(compound)); held.append(len(held))
    return BASE.LEGACY.EFFECT.LooPairs(np.stack(support).astype(np.float32), np.stack(target).astype(np.float32), names, np.asarray(held, dtype=np.int64))


def restrict(source: Any, compounds: set[str]) -> Any:
    ix = np.asarray([i for i, name in enumerate(source.smiles) if str(name) in compounds], dtype=np.int64)
    if not len(ix):
        raise RuntimeError("empty teacher pair fold")
    return BASE.LEGACY.EFFECT.LooPairs(source.support[ix], source.target[ix], [source.smiles[i] for i in ix], source.held_rows[ix])


def teacher_args(args: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(seed=seed, hidden_dim=256, latent_dim=32, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, epochs=args.teacher_epochs, smoke=args.smoke)


def apply_teacher(model: Any, views: np.ndarray, mean: np.ndarray, scale: np.ndarray, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    n, k, d = views.shape
    output = BASE.LEGACY.predict_effect(model, views.reshape(n * k, d), mean, scale, args.batch_size, device)
    return output.reshape(n, k, d).mean(axis=1).astype(np.float32)


def build_targets(train: Examples, valid: Examples, budget: int, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    assignment = BASE.LEGACY.fold_assignment(train.compound.tolist())
    all_train = set(map(str, train.compound))
    train_pairs, valid_pairs = pairs(train), pairs(valid)
    cfra_train = np.full_like(train.raw, np.nan)
    index = {str(x): i for i, x in enumerate(train.compound)}
    ledger: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        held = {compound for compound, value in assignment.items() if value == fold}
        fitted = all_train - held
        seed = int(BASE.LEGACY.stable_int(SEED, f"{VERSION}|teacher|b{budget}|fold{fold}") % (2**32 - 1))
        model, mean, scale, epoch, loss = BASE.LEGACY.EFFECT.fit_encoder(restrict(train_pairs, fitted), valid_pairs, teacher_args(args, seed), device)
        ix = np.asarray([index[x] for x in sorted(held)], dtype=np.int64)
        cfra_train[ix] = apply_teacher(model, train.views[ix], mean, scale, args, device)
        ledger.append({"budget": budget, "fold": fold, "teacher_regime": "fit-compound-OOF-strict-selected-1R", "fit_compounds": len(fitted), "held_compounds": len(held), "pair_count_fit": int(len(restrict(train_pairs, fitted).support)), "selected_plates_only": True, "held_compounds_seen_during_teacher_fit": False, "best_epoch": int(epoch), "validation_mse": float(loss)})
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    if not np.isfinite(cfra_train).all():
        raise RuntimeError("missing OOF CFRA target")
    seed = int(BASE.LEGACY.stable_int(SEED, f"{VERSION}|teacher|b{budget}|validation") % (2**32 - 1))
    model, mean, scale, epoch, loss = BASE.LEGACY.EFFECT.fit_encoder(train_pairs, valid_pairs, teacher_args(args, seed), device)
    cfra_valid = apply_teacher(model, valid.views, mean, scale, args, device)
    ledger.append({"budget": budget, "fold": "validation", "teacher_regime": "fit-full-validation-1R-strict-selected", "fit_compounds": len(all_train), "held_compounds": len(valid.compound), "pair_count_fit": int(len(train_pairs.support)), "selected_plates_only": True, "held_compounds_seen_during_teacher_fit": False, "best_epoch": int(epoch), "validation_mse": float(loss)})
    return {"raw_train": train.raw, "raw_valid": valid.raw, "cfra_train": cfra_train, "cfra_valid": cfra_valid}, ledger


def split(virtual: Any, examples: Examples, target: np.ndarray) -> Any:
    return BASE.subset_virtual(virtual, examples, target)


def save_branch(root: Path, budget: int, method: str, master: int, branch: str, payload: Mapping[str, Any], definition: str, epoch: int, loss: float) -> str:
    path = root / f"budget{budget}" / method / f"seed{master}" / f"{branch}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": VERSION, "budget": budget, "method": method, "master_seed": master, "branch": branch, "hidden_dims": list(DIMS), "parameter_count": BRANCH_PARAMETERS, "target_definition": definition, "best_epoch": epoch, "validation_mse": loss, "test_loaded": False, **dict(payload)}, path)
    return sha256(path)


def fit_package(root: Path, budget: int, method: str, master: int, tr: Any, va: Any, target: Mapping[str, np.ndarray], args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if method == M0:
        specs = (("raw_a", target["raw_train"], target["raw_valid"], "Raw-k-Aggregate"), ("raw_b", target["raw_train"], target["raw_valid"], "Raw-k-Aggregate")); join = "mean"
    else:
        specs = (("cfra_base", target["cfra_train"], target["cfra_valid"], "mean of per-view 1R-CFRA"), ("raw_minus_cfra", target["raw_train"] - target["cfra_train"], target["raw_valid"] - target["cfra_valid"], "Raw-k-Aggregate minus CFRA-k-View-Aggregate")); join = "sum"
    rows: list[dict[str, Any]] = []; hashes: dict[str, str] = {}; prediction: list[np.ndarray] = []
    fit_args = SimpleNamespace(student_epochs=1 if args.smoke else args.student_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    for branch, train_target, valid_target, definition in specs:
        seed = int(BASE.LEGACY.stable_int(master, f"{VERSION}|b{budget}|{method}|{branch}") % (2**32 - 1))
        payload, pred, loss, epoch = BASE.CAP.fit_single(tr, va, train_target.astype(np.float32), valid_target.astype(np.float32), DIMS, seed, fit_args, device)
        path = f"budget{budget}/{method}/seed{master}/{branch}.pt"
        hashes[path] = save_branch(root, budget, method, master, branch, payload, definition, epoch, loss)
        prediction.append(pred); rows.append({"budget": budget, "method": method, "seed": master, "branch": branch, "best_epoch": epoch, "validation_mse": loss, "parameter_count": BRANCH_PARAMETERS})
    final = (prediction[0] + prediction[1]) / 2 if join == "mean" else prediction[0] + prediction[1]
    rows.append({"budget": budget, "method": method, "seed": master, "branch": "final", "best_epoch": "", "validation_mse": float(np.mean((final - target["raw_valid"]) ** 2)), "parameter_count": TOTAL_PARAMETERS})
    return rows, hashes


def predict(root: Path, budget: int, method: str, seed: int, data: Any, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    names = ("raw_a", "raw_b") if method == M0 else ("cfra_base", "raw_minus_cfra")
    output = []
    for name in names:
        ck = torch.load(root / f"budget{budget}" / method / f"seed{seed}" / f"{name}.pt", map_location=device, weights_only=False)
        model = BASE.CAP.StudentWidth(*DIMS).to(device); model.load_state_dict(ck["state_dict"])
        output.append(BASE.CAP.predict_single(model, data, ck["stats"], args.batch_size, device)); del model
    return ((output[0] + output[1]) / 2 if method == M0 else output[0] + output[1]).astype(np.float32)


def manifest_rows(examples: Mapping[str, Mapping[int, Examples]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, by_budget in examples.items():
        for budget, ex in by_budget.items():
            for compound, dose, plates in zip(ex.compound, ex.dose, ex.plates):
                rows.append({"split": split_name, "budget": budget, "compound_id": str(compound), "dose": str(dose), "selected_plates": "|".join(plates), "selected_plate_count": len(plates), "nonselected_plates_used": False})
    return rows


def teacher_pair_rows(split_name: str, budget: int, ex: Examples) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for compound, dose, plates in zip(ex.compound, ex.dose, ex.plates):
        for index, support_plate in enumerate(plates):
            rows.append({"split": split_name, "budget": budget, "compound_id": str(compound), "dose": str(dose), "support_plate": support_plate, "held_plate": plates[(index + 1) % len(plates)], "selected_plates_only": True})
    return rows


def prepare(args: argparse.Namespace) -> None:
    root = args.output_root
    if root.exists() and any(root.iterdir()): raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = BASE.load_rows(args.rows_root, ("train", "valid")); BASE.assert_lock(rows, args.split_lock)
    selected = {split: {budget: select_examples(rows[split], budget, split, False) for budget in BUDGETS} for split in ("train", "valid")}
    manifest = manifest_rows(selected); write_csv(root / "STRICT_SELECTED_MANIFEST.csv", manifest)
    audit = {"version": VERSION, "status": "PASS", "phase": "prepare", "protocol_sha256": sha256(args.protocol_file), "manifest_sha256": sha256(root / "STRICT_SELECTED_MANIFEST.csv"), "budgets": list(BUDGETS), "eligible": {str(b): {s: len(selected[s][b].compound) for s in selected} for b in BUDGETS}, "teacher_pair_rule": "cyclic 1R directed pairs over only selected plates; exactly k pairs per selected condition", "test_loaded": False, "test_used_for_selection": False}
    dump(root / "PREPARE_COMPLETE.json", audit); print(json.dumps(audit, indent=2))


def fit(args: argparse.Namespace) -> None:
    root = args.output_root; prep = read(root / "PREPARE_COMPLETE.json")
    if prep["protocol_sha256"] != sha256(args.protocol_file) or prep["test_loaded"]: raise RuntimeError("invalid prepare marker")
    rows = BASE.load_rows(args.rows_root, ("train", "valid")); BASE.assert_lock(rows, args.split_lock)
    selected = {split_name: {b: select_examples(rows[split_name], b, split_name, False) for b in BUDGETS} for split_name in ("train", "valid")}
    if sha256(root / "STRICT_SELECTED_MANIFEST.csv") != prep["manifest_sha256"]: raise RuntimeError("manifest changed")
    virtual = BASE.load_virtual_splits(args.model_h5, args.aggregate_npz, args.split_lock, ("train", "valid")); device = BASE.device_for(args.device)
    ledger: list[dict[str, Any]] = []; validation: list[dict[str, Any]] = []; hashes: dict[str, str] = {}; targets_ref: dict[str, Any] = {}
    for budget in BUDGETS:
        target, teachers = build_targets(selected["train"][budget], selected["valid"][budget], budget, args, device)
        bdir = root / f"budget{budget}"; bdir.mkdir()
        target_path = bdir / "TARGETS.npz"; np.savez_compressed(target_path, train_compound=selected["train"][budget].compound, valid_compound=selected["valid"][budget].compound, **target)
        write_csv(bdir / "TEACHER_LEDGER.csv", teachers); ledger.extend(teachers)
        pair_rows = teacher_pair_rows("train", budget, selected["train"][budget]) + teacher_pair_rows("valid", budget, selected["valid"][budget])
        write_csv(bdir / "TEACHER_PAIR_MANIFEST.csv", pair_rows)
        targets_ref[str(budget)] = {"path": str(target_path.relative_to(root)), "sha256": sha256(target_path), "definition": "selected-k-only: mean over CFRA_1R(each selected raw plate)", "raw_definition": "mean over exactly same selected raw plates", "residual_definition": "Raw-k minus CFRA-k"}
        tr, va = split(virtual["train"], selected["train"][budget], target["raw_train"]), split(virtual["valid"], selected["valid"][budget], target["raw_valid"])
        for method in METHODS:
            for seed in SEEDS:
                records, branch_hashes = fit_package(root, budget, method, seed, tr, va, target, args, device); validation.extend(records); hashes.update(branch_hashes)
                if device.type == "cuda": torch.cuda.empty_cache()
    write_csv(root / "VALIDATION_METRICS.csv", validation); write_csv(root / "TEACHER_LEDGER.csv", ledger); dump(root / "TARGET_REFERENCES.json", targets_ref); dump(root / "CHECKPOINT_HASHES.json", hashes)
    expected = len(BUDGETS) * len(METHODS) * len(SEEDS) * 2
    if len(hashes) != expected or any(row["held_compounds_seen_during_teacher_fit"] for row in ledger): raise RuntimeError("fit ledger failure")
    marker = {"version": VERSION, "status": "PASS", "phase": "fit", "protocol_sha256": sha256(args.protocol_file), "manifest_sha256": sha256(root / "STRICT_SELECTED_MANIFEST.csv"), "checkpoint_count": len(hashes), "teacher_selected_plates_only": True, "test_loaded": False, "test_used_for_selection": False}
    dump(root / "FIT_COMPLETE.json", marker); dump(root / "SELECTION_FREEZE.json", {**marker, "phase": "selection", "selection_frozen": True}); print(json.dumps(marker, indent=2))


def test(args: argparse.Namespace) -> None:
    root = args.output_root; fit_marker = read(root / "FIT_COMPLETE.json"); freeze = read(root / "SELECTION_FREEZE.json")
    if fit_marker["status"] != "PASS" or fit_marker["test_loaded"] or not freeze.get("selection_frozen"): raise RuntimeError("test barrier failed")
    if (root / "TEST_COMPLETE.json").exists(): raise FileExistsError("test already executed")
    dump(root / "TEST_AUTHORIZED.json", {"version": VERSION, "status": "AUTHORIZED", "test_loaded_before_authorization": False, "test_used_for_selection": False})
    rows = BASE.load_rows(args.rows_root, ("test",)); BASE.assert_lock(rows, args.split_lock)
    selected = {b: select_examples(rows["test"], b, "test", True) for b in BUDGETS}; virtual = BASE.load_virtual_splits(args.model_h5, args.aggregate_npz, args.split_lock, ("test",)); device = BASE.device_for(args.device)
    per: list[dict[str, Any]] = []; summary: list[dict[str, Any]] = []; method_summary: list[dict[str, Any]] = []; contrasts: list[dict[str, Any]] = []
    for budget in BUDGETS:
        ex = selected[budget]; data = split(virtual["test"], ex, ex.raw); values: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        for method in METHODS:
            for seed in SEEDS:
                prediction = predict(root, budget, method, seed, data, args, device)
                same = BASE.LEGACY.rowwise_corr(prediction, ex.raw); foreign = BASE.LEGACY.rowwise_corr(prediction, ex.foreign); zsame = np.arctanh(np.clip(same, -0.999999, 0.999999)); zforeign = np.arctanh(np.clip(foreign, -0.999999, 0.999999)); index = {str(x): i for i, x in enumerate(virtual["test"].smiles)}; ix = np.asarray([index[str(x)] for x in ex.compound]); control = virtual["test"].control[ix]; full = BASE.LEGACY.rowwise_corr(prediction + control, ex.raw + control)
                values[(method, seed)] = {"same_z": zsame, "delta_pcc": same, "full_target_pcc": full, "foreign_z": zforeign, "E": zsame - zforeign}
                path = root / f"budget{budget}" / f"seed{seed}_{method}_test_predictions.npz"; np.savez_compressed(path, compound=ex.compound, prediction=prediction, raw_k_aggregate=ex.raw, selected_plates=np.asarray(ex.plates, dtype=str))
                for i, compound in enumerate(ex.compound): per.append({"budget": budget, "method": method, "seed": seed, "compound_id": str(compound), "foreign_ok": int(ex.foreign_ok[i]), **{key: float(value[i]) for key, value in values[(method, seed)].items()}})
        for method in METHODS:
            method_row = {"budget": budget, "method": method, "parameter_count": TOTAL_PARAMETERS, "n_compounds": len(ex.compound), "foreign_compounds": int(ex.foreign_ok.sum())}
            for metric in ("same_z", "delta_pcc", "full_target_pcc", "foreign_z", "E"):
                absolute = BASE.LEGACY.nanmean_across_seeds([values[(method, seed)][metric] for seed in SEEDS])
                point, low, high, n = BASE.bootstrap_pair(absolute, np.zeros(len(absolute), dtype=np.float32), ex.compound, BASE.LEGACY.stable_int(SEED, f"{VERSION}|b{budget}|{method}|absolute|{metric}"), args.bootstrap_rounds)
                method_row[f"{metric}_mean"] = point; method_row[f"{metric}_ci_low"] = low; method_row[f"{metric}_ci_high"] = high; method_row[f"{metric}_n"] = n
            method_summary.append(method_row)
        row = {"budget": budget, "comparison": f"{M2}-{M0}", "n_compounds": len(ex.compound), "foreign_compounds": int(ex.foreign_ok.sum())}
        for metric in ("same_z", "delta_pcc", "full_target_pcc", "foreign_z", "E"):
            left = BASE.LEGACY.nanmean_across_seeds([values[(M2, seed)][metric] for seed in SEEDS]); right = BASE.LEGACY.nanmean_across_seeds([values[(M0, seed)][metric] for seed in SEEDS]); point, low, high, n = BASE.bootstrap_pair(left, right, ex.compound, BASE.LEGACY.stable_int(SEED, f"{VERSION}|b{budget}|{metric}"), args.bootstrap_rounds); row[f"{metric}_delta"] = point; row[f"{metric}_ci_low"] = low; row[f"{metric}_ci_high"] = high; row[f"{metric}_n"] = n; contrasts.append({"budget": budget, "metric": metric, "estimate": point, "ci_low": low, "ci_high": high, "n_compounds": n, "rounds": args.bootstrap_rounds})
        row["decision"] = "GO" if row["delta_pcc_ci_low"] > 0 and row["full_target_pcc_ci_low"] > 0 else "NO-GO" if row["delta_pcc_ci_high"] < 0 and row["full_target_pcc_ci_high"] < 0 else "INCONCLUSIVE"; summary.append(row)
    write_csv(root / "TEST_PER_COMPOUND.csv", per); write_csv(root / "TEST_METHOD_SUMMARY.csv", method_summary); write_csv(root / "TEST_CONTRASTS.csv", contrasts); write_csv(root / "TEST_SUMMARY.csv", summary)
    dump(root / "TEST_COMPLETE.json", {"version": VERSION, "status": "PASS", "phase": "test", "test_loaded_after_authorization": True, "test_used_for_selection": False, "target": "Raw-k-Aggregate formed from exactly selected k physical repeats", "independent_held_repeat": False, "bootstrap_unit": "compound", "bootstrap_rounds": args.bootstrap_rounds, "method_summary_path": "TEST_METHOD_SUMMARY.csv", "contrast_path": "TEST_CONTRASTS.csv"}); print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse()
    if args.phase == "prepare": prepare(args)
    elif args.phase == "fit": fit(args)
    else: test(args)


if __name__ == "__main__": main()
