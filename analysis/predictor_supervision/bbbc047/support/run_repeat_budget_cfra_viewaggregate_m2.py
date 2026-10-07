#!/usr/bin/env python3
"""Test-sealed BBBC047 1R/2R/3R per-view-CFRA aggregate M0/M2 comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = Path(__file__).resolve().parents[4] / "analysis"
LEGACY_PATH = (Path(__file__).resolve().parents[4] / "analysis/predictor_supervision/bbbc047/support/run_cfra_target_replacement.py")
if not LEGACY_PATH.is_file():
    raise FileNotFoundError(f"legacy target-replacement source is required: {LEGACY_PATH}")
spec = importlib.util.spec_from_file_location("bbbc047_repeat_legacy", LEGACY_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"unable to import {LEGACY_PATH}")
LEGACY = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = LEGACY
spec.loader.exec_module(LEGACY)

CAPACITY_PATH = (Path(__file__).resolve().parents[4] / "analysis/predictor_supervision/bbbc047/support/run_capacity_audit.py")
capacity_spec = importlib.util.spec_from_file_location("bbbc047_repeat_budget_capacity", CAPACITY_PATH)
if capacity_spec is None or capacity_spec.loader is None:
    raise RuntimeError(f"unable to import {CAPACITY_PATH}")
CAP = importlib.util.module_from_spec(capacity_spec)
sys.modules[capacity_spec.name] = CAP
capacity_spec.loader.exec_module(CAP)


VERSION = "BBBC047-repeat-budget-per-view-CFRA-aggregate-M2-v1-2026-09-16"
PROTOCOL_SEED = 3407
BUDGETS = (1, 2, 3)
PREDICTOR_SEEDS = (3407, 42, 2025, 1337, 7331)
FOLDS = 5
CP_DIM = 775
FP_DIM = 2048
EXPECTED = {"train": 12175, "valid": 4059, "test": 4060}
M0 = "M0_RawKAggregate"
M2 = "M2_CFRA1ViewAggregate_Residual"
METHODS = (M0, M2)
M2_DIMS = (256, 80)
M2_BRANCH_PARAMETERS = 806279
M2_TOTAL_PARAMETERS = 1612558

DEFAULT_ROWS = Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0")
DEFAULT_H5 = DEFAULT_ROWS.parent / "Paired_CP_GE_PlateMedian_v1_model_compat.h5"
DEFAULT_AGG = DEFAULT_ROWS / "molecule_aggregates.npz"
DEFAULT_LOCK = Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "fit", "test"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rows-root", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--model-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--aggregate-npz", type=Path, default=DEFAULT_AGG)
    parser.add_argument("--split-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--protocol-file", type=Path, default=HERE / "PROTOCOL.md")
    parser.add_argument("--teacher-epochs", type=int, default=80)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and (args.teacher_epochs, args.student_epochs, args.batch_size) != (80, 40, 256):
        parser.error("formal protocol locks teacher_epochs/student_epochs/batch_size to 80/40/256")
    if not args.smoke and args.bootstrap_rounds != 10000:
        parser.error("formal protocol locks bootstrap-rounds to 10000")
    if args.bootstrap_rounds < 1:
        parser.error("bootstrap-rounds must be positive")
    return args


def device_for(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def load_rows(rows_root: Path, splits: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in splits:
        path = rows_root / f"{split}_cp_plate_rows.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as z:
            required = {"smiles", "dose", "plate", "delta"}
            if not required.issubset(z.files):
                raise RuntimeError(f"missing keys in {path}: {sorted(required - set(z.files))}")
            result[split] = LEGACY.Rows("BBBC047", "CP", split, z["smiles"], z["dose"], z["plate"], z["delta"])
        if len(set(result[split].compound.tolist())) != EXPECTED[split]:
            raise RuntimeError(f"{split} compound count mismatch")
    return result


def assert_lock(rows: dict[str, Any], split_lock: Path) -> None:
    lock = read_json(split_lock)
    for split, values in rows.items():
        observed = set(values.compound.tolist())
        expected = {str(x) for x in lock[f"{split}_smiles"]}
        if observed != expected:
            raise RuntimeError(f"{split} rows do not match the official compound lock")
    sets = {split: set(values.compound.tolist()) for split, values in rows.items()}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if left in sets and right in sets and sets[left] & sets[right]:
            raise RuntimeError(f"compound overlap: {left}/{right}")


def build_bundles(rows: dict[str, Any], include_foreign: bool = True) -> dict[int, dict[str, Any]]:
    return {
        budget: {
            split: LEGACY.make_examples(
                values,
                None,
                budget,
                PROTOCOL_SEED,
                all_conditions=False,
                require_source=False,
                include_foreign=include_foreign,
            )
            for split, values in rows.items()
        }
        for budget in BUDGETS
    }


def selected_view_tensor(rows: Any, examples: Any) -> np.ndarray:
    """Return the exact selected support plates, never their pre-CFRA mean."""
    plate_means = rows.plate_means()
    views: list[np.ndarray] = []
    for compound, dose, plates in zip(examples.compound, examples.dose, examples.support_plates):
        by_plate = plate_means[(str(compound), str(dose))]
        if len(plates) == 0 or any(str(plate) not in by_plate for plate in plates):
            raise RuntimeError("support plate is missing from the frozen physical-row source")
        views.append(np.stack([np.asarray(by_plate[str(plate)], dtype=np.float32) for plate in plates], axis=0))
    output = np.stack(views, axis=0).astype(np.float32)
    if output.ndim != 3 or output.shape[2] != CP_DIM or not np.isfinite(output).all():
        raise RuntimeError(f"invalid selected view tensor: {output.shape}")
    return output


def held_view_tensor(rows: Any, examples: Any) -> np.ndarray:
    plate_means = rows.plate_means()
    values: list[np.ndarray] = []
    for compound, dose, plates in zip(examples.compound, examples.dose, examples.held_plates):
        if len(plates) != 1:
            raise RuntimeError("repeat-budget validation/test requires exactly one raw held plate")
        by_plate = plate_means[(str(compound), str(dose))]
        values.append(np.asarray(by_plate[str(plates[0])], dtype=np.float32))
    return np.stack(values, axis=0).astype(np.float32)


def teacher_args(args: argparse.Namespace, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=int(seed), hidden_dim=256, latent_dim=32, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        epochs=args.teacher_epochs, smoke=args.smoke,
    )


def aggregate_per_view_cfra(model: Any, views: np.ndarray, mean: np.ndarray, scale: np.ndarray, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    n, k, dim = views.shape
    if dim != CP_DIM:
        raise RuntimeError("CFRA view dimension changed")
    transformed = LEGACY.predict_effect(model, views.reshape(n * k, dim), mean, scale, args.batch_size, device)
    return np.asarray(transformed.reshape(n, k, dim).mean(axis=1), dtype=np.float32)


def build_view_aggregate_targets(rows: dict[str, Any], examples: Mapping[str, Any], budget: int, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, int]]:
    """Fit 1R teachers, apply each selected view separately, then aggregate."""
    train_examples = examples["train"]
    valid_examples = examples["valid"]
    assignment = LEGACY.fold_assignment(train_examples.compound.tolist())
    all_train = set(map(str, rows["train"].compound.tolist()))
    full_pairs = LEGACY.budget_pairs(rows["train"], 1)
    valid_pairs = LEGACY.budget_pairs(rows["valid"], 1)
    train_views = selected_view_tensor(rows["train"], train_examples)
    valid_views = selected_view_tensor(rows["valid"], valid_examples)
    valid_held = held_view_tensor(rows["valid"], valid_examples)
    train_cfra = np.full((len(train_examples), CP_DIM), np.nan, dtype=np.float32)
    metadata: list[dict[str, Any]] = []
    index = {str(compound): i for i, compound in enumerate(train_examples.compound)}
    for fold in range(FOLDS):
        held = {compound for compound, value in assignment.items() if value == fold}
        fitted = all_train - held
        if held & fitted or not held or not fitted:
            raise RuntimeError("invalid OOF teacher fold")
        seed = int(LEGACY.stable_int(PROTOCOL_SEED, f"{VERSION}|teacher|b{budget}|fold{fold}") % (2**32 - 1))
        model, mean, scale, best_epoch, valid_mse = LEGACY.EFFECT.fit_encoder(
            LEGACY.restrict_pairs(full_pairs, fitted), valid_pairs, teacher_args(args, seed), device
        )
        held_ix = np.asarray([index[compound] for compound in sorted(held)], dtype=np.int64)
        train_cfra[held_ix] = aggregate_per_view_cfra(model, train_views[held_ix], mean, scale, args, device)
        metadata.append({
            "budget": budget, "teacher_regime": "fit-compound-OOF-1R-view",
            "fold": fold, "seed": seed, "fit_compounds": len(fitted), "held_compounds": len(held),
            "fit_compound_sha256": LEGACY.digest_compounds(fitted),
            "held_compound_sha256": LEGACY.digest_compounds(held),
            "held_compounds_seen_during_teacher_fit": False,
            "best_epoch": int(best_epoch), "validation_mse": float(valid_mse),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not np.isfinite(train_cfra).all():
        raise RuntimeError("missing/non-finite OOF CFRA aggregate target")
    audit_seed = int(LEGACY.stable_int(PROTOCOL_SEED, f"{VERSION}|teacher|b{budget}|validation-audit") % (2**32 - 1))
    audit_model, mean, scale, best_epoch, valid_mse = LEGACY.EFFECT.fit_encoder(full_pairs, valid_pairs, teacher_args(args, audit_seed), device)
    valid_cfra_support = aggregate_per_view_cfra(audit_model, valid_views, mean, scale, args, device)
    valid_cfra_held = LEGACY.predict_effect(audit_model, valid_held, mean, scale, args.batch_size, device)
    metadata.append({
        "budget": budget, "teacher_regime": "fit-full-validation-audit-1R-view",
        "fold": "validation_audit", "seed": audit_seed, "fit_compounds": len(all_train), "held_compounds": len(valid_examples),
        "held_compounds_seen_during_teacher_fit": False, "best_epoch": int(best_epoch), "validation_mse": float(valid_mse),
    })
    del audit_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    raw_train = np.asarray(train_examples.support, dtype=np.float32)
    raw_valid_held = np.asarray(valid_examples.target, dtype=np.float32)
    if not np.allclose(raw_train, train_views.mean(axis=1), rtol=0.0, atol=1e-6):
        raise RuntimeError("raw K-aggregate is not the mean of the exact selected physical views")
    if not np.allclose(raw_valid_held, valid_held, rtol=0.0, atol=1e-6):
        raise RuntimeError("validation held target identity changed")
    return {
        "raw_train": raw_train, "cfra_train": train_cfra,
        "raw_valid_held": raw_valid_held, "cfra_valid_support": valid_cfra_support,
        "cfra_valid_held": np.asarray(valid_cfra_held, dtype=np.float32),
    }, metadata, assignment


def load_virtual_splits(model_h5: Path, aggregate_npz: Path, split_lock: Path, splits: Iterable[str]) -> dict[str, Any]:
    """Load only requested split rows; in particular, never index test in fit."""
    requested = tuple(splits)
    lock = read_json(split_lock)
    if not model_h5.is_file() or not aggregate_npz.is_file():
        raise FileNotFoundError("model H5 and aggregate NPZ are required")
    with h5py.File(model_h5, "r") as handle:
        all_smiles = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in handle["canonical_smiles"][:]]
        if len(all_smiles) != len(set(all_smiles)):
            raise RuntimeError("duplicate H5 canonical_smiles")
        index = {value: i for i, value in enumerate(all_smiles)}
        output: dict[str, Any] = {}
        with np.load(aggregate_npz, allow_pickle=False) as aggregate:
            for split in requested:
                names = [str(x) for x in lock[f"{split}_smiles"]]
                if len(names) != EXPECTED[split] or len(set(names)) != len(names):
                    raise RuntimeError(f"invalid {split} split lock")
                try:
                    rows = np.asarray([index[name] for name in names], dtype=np.int64)
                except KeyError as exc:
                    raise RuntimeError(f"{split} compound missing in H5: {exc}") from exc
                # h5py requires fancy indices to be increasing; restore the
                # split-lock order after reading the sorted H5 rows.
                order = np.argsort(rows)
                inverse = np.argsort(order)
                sorted_rows = rows[order]
                control = handle["control_CP"][sorted_rows].astype(np.float32)[inverse]
                target = handle["target_CP"][sorted_rows].astype(np.float32)[inverse]
                delta = target - control
                agg_names = [str(x) for x in aggregate[f"{split}_smiles"]]
                agg_index = {name: i for i, name in enumerate(agg_names)}
                if set(agg_index) != set(names):
                    raise RuntimeError(f"aggregate/H5 {split} compound mismatch")
                agg_rows = np.asarray([agg_index[name] for name in names], dtype=np.int64)
                aggregate_delta = aggregate[f"{split}_cp"][agg_rows].astype(np.float32)
                max_error = float(np.max(np.abs(delta - aggregate_delta)))
                if max_error > 1e-5:
                    raise RuntimeError(f"aggregate/H5 {split} delta mismatch: {max_error}")
                fingerprint = LEGACY.VIRTUAL.fingerprints(names)
                output[split] = LEGACY.VIRTUAL.VirtualSplit(names, control, delta, fingerprint)
    return output


def subset_virtual(base: Any, examples: Any, target: np.ndarray) -> Any:
    index = {str(value): i for i, value in enumerate(base.smiles)}
    rows = np.asarray([index[str(x)] for x in examples.compound], dtype=np.int64)
    if len(rows) != len(set(examples.compound.tolist())):
        raise RuntimeError("one predictor row per compound is required")
    return LEGACY.PredictorSplit(
        compound=examples.compound.copy(),
        dose=examples.dose.copy(),
        control=base.control[rows].astype(np.float32),
        fingerprint=base.fingerprint[rows].astype(np.float32),
        target=np.asarray(target, dtype=np.float32),
    )


def manifest_rows(bundles: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for split, examples in bundles[budget].items():
            for i in range(len(examples)):
                rows.append({
                    "dataset": "BBBC047",
                    "split": split,
                    "budget": budget,
                    "protocol_seed": PROTOCOL_SEED,
                    "compound_id": str(examples.compound[i]),
                    "dose": str(examples.dose[i]),
                    "condition": str(examples.condition[i]),
                    "support_rep_ids": "|".join(examples.support_plates[i]),
                    "held_rep_id": "|".join(examples.held_plates[i]),
                })
    return rows


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    LEGACY.write_csv(path, rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def protocol_hash(protocol_file: Path) -> str:
    if not protocol_file.is_file():
        raise FileNotFoundError(protocol_file)
    return sha256_file(protocol_file)


def preflight_audit(args: argparse.Namespace, root: Path, rows: dict[str, Any], bundles: dict[int, dict[str, Any]]) -> dict[str, Any]:
    split_sets = {split: set(values.compound.tolist()) for split, values in rows.items()}
    eligibility = {
        f"budget{budget}": {
            split: int(len(set(bundles[budget][split].compound.tolist())))
            for split in rows
        }
        for budget in BUDGETS
    }
    source_hashes = {
        "protocol": protocol_hash(args.protocol_file),
        "split_lock": sha256_file(args.split_lock),
        "train_cp_plate_rows": sha256_file(args.rows_root / "train_cp_plate_rows.npz"),
        "valid_cp_plate_rows": sha256_file(args.rows_root / "valid_cp_plate_rows.npz"),
        "ablation_common": sha256_file((Path(__file__).resolve().parents[4] / "analysis/baselines/ablation_common.py")),
        "legacy_target_replacement": sha256_file(LEGACY_PATH),
    }
    return {
        "version": VERSION,
        "phase": "prepare",
        "protocol_file": str(args.protocol_file),
        "protocol_sha256": source_hashes["protocol"],
        "protocol_seed": PROTOCOL_SEED,
        "budgets": list(BUDGETS),
        "student_methods": list(METHODS),
        "predictor_seeds": list(PREDICTOR_SEEDS),
        "outer_compounds": {split: len(values) for split, values in split_sets.items()},
        "split_intersections": {
            "train_valid": len(split_sets.get("train", set()) & split_sets.get("valid", set())),
            "train_test": 0,
            "valid_test": 0,
        },
        "eligible_compounds": eligibility,
        "manifest_rows": len(manifest_rows(bundles)),
        "source_hashes": source_hashes,
        "test_loaded": False,
        "test_values_opened": False,
        "test_used_for_selection": False,
        "bootstrap_unit": "compound",
        "bootstrap_rounds": args.bootstrap_rounds,
        "status": "PASS",
    }


def require_prepare(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    marker = root / "PREPARE_COMPLETE.json"
    if not marker.is_file():
        raise RuntimeError(f"missing prepare marker: {marker}")
    audit = read_json(marker)
    if audit.get("version") != VERSION or audit.get("protocol_sha256") != protocol_hash(args.protocol_file):
        raise RuntimeError("prepare marker does not match current protocol")
    if audit.get("test_loaded") or audit.get("test_values_opened"):
        raise RuntimeError("prepare marker claims test was opened")
    return audit


def fit_phase_legacy_unused(args: argparse.Namespace, root: Path) -> None:
    require_prepare(args, root)
    if (root / "FIT_COMPLETE.json").exists() or (root / "TEST_COMPLETE.json").exists():
        raise FileExistsError("fit/test marker already exists; use a fresh versioned output root")
    rows = load_rows(args.rows_root, ("train", "valid"))
    assert_lock(rows, args.split_lock)
    bundles = build_bundles(rows, include_foreign=False)
    manifest = read_manifest(root / "MANIFEST.csv")
    expected_manifest = manifest_rows(bundles)
    if len(manifest) != len(expected_manifest):
        raise RuntimeError("manifest row count changed between prepare and fit")
    if sha256_file(root / "MANIFEST.csv") != read_json(root / "PREPARE_COMPLETE.json")["manifest_sha256"]:
        raise RuntimeError("manifest hash changed between prepare and fit")
    virtual = load_virtual_splits(args.model_h5, args.aggregate_npz, args.split_lock, ("train", "valid"))
    device = device_for(args.device)
    all_logs: list[dict[str, Any]] = []
    all_validation: list[dict[str, Any]] = []
    target_hashes: dict[str, Any] = {}
    checkpoint_hashes: dict[str, str] = {}
    train_target_metadata: dict[str, Any] = {}
    for budget in BUDGETS:
        examples = bundles[budget]
        cfra_target, teacher_meta, assignment = LEGACY.build_oof_cfra(
            rows,
            examples["train"],
            budget,
            args,
            device,
        )
        raw_target = examples["train"].support.astype(np.float32)
        if raw_target.shape != cfra_target.shape:
            raise RuntimeError(f"target shape mismatch for budget {budget}")
        budget_dir = root / f"budget{budget}"
        budget_dir.mkdir(parents=True, exist_ok=False)
        target_path = budget_dir / "training_targets.npz"
        np.savez_compressed(
            target_path,
            compound=examples["train"].compound,
            dose=examples["train"].dose,
            raw_mean=raw_target,
            cfra=cfra_target,
        )
        target_hashes[str(budget)] = {
            "path": str(target_path),
            "sha256": sha256_file(target_path),
            "n_compounds": len(examples["train"]),
            "raw_mean_shape": list(raw_target.shape),
            "cfra_shape": list(cfra_target.shape),
            "teacher": teacher_meta,
            "cfra_fold_assignment": assignment,
        }
        json_dump(budget_dir / "CFRA_TEACHER_METADATA.json", teacher_meta)
        train_split = subset_virtual(virtual["train"], examples["train"], raw_target)
        valid_split = subset_virtual(virtual["valid"], examples["valid"], examples["valid"].target)
        stats = LEGACY.common_stats(train_split, raw_target)
        np.savez_compressed(budget_dir / "student_normalization.npz", **stats)
        for seed in PREDICTOR_SEEDS:
            for method, label in (("raw_mean", raw_target), ("cfra", cfra_target)):
                model, fitted_stats, logs = LEGACY.fit_student(
                    train_split,
                    valid_split,
                    label,
                    stats,
                    seed,
                    args,
                    device,
                    method,
                )
                all_logs.extend({**row, "budget": budget} for row in logs)
                best_row = min(logs, key=lambda row: float(row["validation_mse"]))
                ckpt_dir = budget_dir / method
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"seed{seed}.pt"
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save({
                    "version": VERSION,
                    "budget": budget,
                    "method": method,
                    "seed": seed,
                    "best_epoch": int(best_row["epoch"]),
                    "validation_mse": float(best_row["validation_mse"]),
                    "state_dict": state,
                    "stats": {key: np.asarray(value, dtype=np.float32) for key, value in fitted_stats.items()},
                }, ckpt_path)
                checkpoint_hashes[f"budget{budget}/{method}/seed{seed}"] = sha256_file(ckpt_path)
                validation_pred = LEGACY.predict_student(model, valid_split, fitted_stats, args.batch_size, device)
                metrics = LEGACY.metric_arrays(validation_pred, examples["valid"])
                overlap, direction = LEGACY.top25_metrics(validation_pred, examples["valid"].target)
                all_validation.append({
                    "budget": budget,
                    "method": method,
                    "seed": seed,
                    "split": "valid",
                    "n_compounds": len(valid_split.compound),
                    "best_epoch": int(best_row["epoch"]),
                    "validation_mse": float(best_row["validation_mse"]),
                    "z_same_mean": float(np.nanmean(metrics["same_z"])),
                    "z_foreign_mean": float(np.nanmean(metrics["foreign_z"])),
                    "E_mean": float(np.nanmean(metrics["excess_z"])),
                    "pcc_mean": float(np.nanmean(metrics["pcc"])),
                    "top25_overlap_mean": float(np.nanmean(overlap)),
                    "top25_direction_mean": float(np.nanmean(direction)),
                })
        train_target_metadata[str(budget)] = {
            "raw_mean_definition": "mean of exact selected support plates",
            "cfra_definition": "5-fold compound-OOF teacher prediction from the same support mean",
            "train_test_values_opened": False,
        }
    LEGACY.write_csv(root / "TRAINING_LOG.csv", all_logs)
    LEGACY.write_csv(root / "VALIDATION_METRICS.csv", all_validation)
    json_dump(root / "TARGET_HASHES.json", target_hashes)
    json_dump(root / "CHECKPOINT_HASHES.json", checkpoint_hashes)
    fit_audit = {
        "version": VERSION,
        "phase": "fit",
        "protocol_sha256": protocol_hash(args.protocol_file),
        "manifest_sha256": sha256_file(root / "MANIFEST.csv"),
        "budgets": list(BUDGETS),
        "methods": list(METHODS),
        "predictor_seeds": list(PREDICTOR_SEEDS),
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "checkpoint_count": len(checkpoint_hashes),
        "target_hashes": target_hashes,
        "checkpoint_hashes": checkpoint_hashes,
        "test_loaded": False,
        "test_values_opened": False,
        "test_used_for_selection": False,
        "bootstrap_unit": "compound",
        "bootstrap_rounds": args.bootstrap_rounds,
        "status": "PASS",
    }
    json_dump(root / "FIT_COMPLETE.json", fit_audit)
    selection = {
        "version": VERSION,
        "selection_frozen": True,
        "protocol_sha256": protocol_hash(args.protocol_file),
        "manifest_sha256": sha256_file(root / "MANIFEST.csv"),
        "checkpoint_hashes_sha256": sha256_file(root / "CHECKPOINT_HASHES.json"),
        "checkpoint_count": len(checkpoint_hashes),
        "all_budgets": list(BUDGETS),
        "all_methods": list(METHODS),
        "all_predictor_seeds": list(PREDICTOR_SEEDS),
        "test_loaded": False,
        "test_used_for_selection": False,
        "status": "PASS",
    }
    json_dump(root / "SELECTION_FREEZE.json", selection)
    print(json.dumps({"phase": "fit", "root": str(root), "checkpoint_count": len(checkpoint_hashes), "test_loaded": False}, indent=2))


def save_branch(root: Path, budget: int, method: str, master: int, branch: str, payload: Mapping[str, Any], target_definition: str, best_epoch: int, validation_mse: float) -> str:
    path = root / f"budget{budget}" / method / f"seed{master}" / f"{branch}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": VERSION, "budget": budget, "method": method, "master_seed": master,
        "branch": branch, "hidden_dims": list(M2_DIMS), "parameter_count": M2_BRANCH_PARAMETERS,
        "target_definition": target_definition, "best_epoch": int(best_epoch),
        "validation_mse_original": float(validation_mse), "test_loaded": False, **dict(payload),
    }, path)
    return sha256_file(path)


def fit_m2_package(args: argparse.Namespace, root: Path, budget: int, method: str, train: Any, valid: Any, targets: Mapping[str, np.ndarray], master: int, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if method == M0:
        specs = (
            ("raw_branch_a", targets["raw_train"], targets["raw_valid_held"], "Raw-k-Aggregate"),
            ("raw_branch_b", targets["raw_train"], targets["raw_valid_held"], "Raw-k-Aggregate"),
        )
        combine = "mean"
    elif method == M2:
        specs = (
            ("cfra1_view_aggregate_base", targets["cfra_train"], targets["cfra_valid_held"], "mean over per-view OOF CFRA-1R outputs"),
            ("raw_minus_cfra_correction", targets["raw_train"] - targets["cfra_train"], targets["raw_valid_held"] - targets["cfra_valid_held"], "Raw-k-Aggregate minus CFRA-1R-view-Aggregate"),
        )
        combine = "sum"
    else:
        raise RuntimeError(method)
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    predictions: list[np.ndarray] = []
    fit_args = SimpleNamespace(student_epochs=1 if args.smoke else args.student_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    for branch, train_target, valid_target, definition in specs:
        seed = int(LEGACY.stable_int(master, f"{VERSION}|b{budget}|{method}|{branch}") % (2**32 - 1))
        payload, valid_prediction, best_loss, best_epoch = CAP.fit_single(train, valid, np.asarray(train_target, dtype=np.float32), np.asarray(valid_target, dtype=np.float32), M2_DIMS, seed, fit_args, device)
        sha = save_branch(root, budget, method, master, branch, payload, definition, best_epoch, best_loss)
        relative = f"budget{budget}/{method}/seed{master}/{branch}.pt"
        hashes[relative] = sha
        predictions.append(np.asarray(valid_prediction, dtype=np.float32))
        rows.append({"budget": budget, "method": method, "seed": master, "branch": branch, "branch_seed": seed, "best_epoch": best_epoch, "validation_mse_original": best_loss, "target_definition": definition, "parameter_count": M2_BRANCH_PARAMETERS})
    final = (predictions[0] + predictions[1]) / np.float32(2.0) if combine == "mean" else predictions[0] + predictions[1]
    rows.append({"budget": budget, "method": method, "seed": master, "branch": "package_final", "best_epoch": "", "validation_mse_original": float(np.mean((final.astype(np.float64) - targets["raw_valid_held"].astype(np.float64)) ** 2)), "target_definition": "common Raw held profile", "parameter_count": M2_TOTAL_PARAMETERS})
    return rows, hashes


def fit_phase(args: argparse.Namespace, root: Path) -> None:
    prepare = require_prepare(args, root)
    if (root / "FIT_COMPLETE.json").exists() or (root / "TEST_COMPLETE.json").exists():
        raise FileExistsError("fit/test marker already exists; use a fresh versioned output root")
    rows = load_rows(args.rows_root, ("train", "valid"))
    assert_lock(rows, args.split_lock)
    bundles = build_bundles(rows, include_foreign=False)
    if sha256_file(root / "MANIFEST.csv") != prepare["manifest_sha256"]:
        raise RuntimeError("frozen manifest hash changed before fit")
    if len(read_manifest(root / "MANIFEST.csv")) != len(manifest_rows(bundles)):
        raise RuntimeError("manifest row count changed before fit")
    virtual = load_virtual_splits(args.model_h5, args.aggregate_npz, args.split_lock, ("train", "valid"))
    device = device_for(args.device)
    validation_rows: list[dict[str, Any]] = []
    teacher_records: dict[str, Any] = {}
    target_records: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for budget in BUDGETS:
        examples = {"train": bundles[budget]["train"], "valid": bundles[budget]["valid"]}
        targets, teacher_meta, assignment = build_view_aggregate_targets(rows, examples, budget, args, device)
        bdir = root / f"budget{budget}"
        bdir.mkdir(parents=True, exist_ok=False)
        target_path = bdir / "TARGETS.npz"
        np.savez_compressed(target_path, train_compound=examples["train"].compound, valid_compound=examples["valid"].compound, **targets)
        audit_rows = []
        for split, raw, cfra in (("train", targets["raw_train"], targets["cfra_train"]), ("valid_support", examples["valid"].support, targets["cfra_valid_support"])):
            pcc = LEGACY.rowwise_corr(raw, cfra)
            ratio = np.linalg.norm(cfra, axis=1) / np.maximum(np.linalg.norm(raw, axis=1), 1e-8)
            names = examples["train"].compound if split == "train" else examples["valid"].compound
            audit_rows.extend({"budget": budget, "split": split, "compound_id": str(name), "raw_vs_cfra_pcc": float(value), "cfra_to_raw_norm_ratio": float(norm)} for name, value, norm in zip(names, pcc, ratio))
        LEGACY.write_csv(bdir / "TARGET_AUDIT.csv", audit_rows)
        json_dump(bdir / "CFRA_TEACHER_METADATA.json", {"teachers": teacher_meta, "fold_assignment": assignment})
        target_records[str(budget)] = {"path": str(target_path.relative_to(root)), "sha256": sha256_file(target_path), "view_aggregation": "apply frozen 1R CFRA independently to each selected support plate, then equal-average", "raw_definition": "equal mean of the same selected support plates", "residual_definition": "Raw-k-Aggregate minus CFRA-1R-view-Aggregate", "target_audit_sha256": sha256_file(bdir / "TARGET_AUDIT.csv")}
        teacher_records[str(budget)] = teacher_meta
        train = subset_virtual(virtual["train"], examples["train"], targets["raw_train"])
        valid = subset_virtual(virtual["valid"], examples["valid"], targets["raw_valid_held"])
        for method in METHODS:
            for master in PREDICTOR_SEEDS:
                records, branch_hashes = fit_m2_package(args, root, budget, method, train, valid, targets, master, device)
                validation_rows.extend(records)
                hashes.update(branch_hashes)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del train, valid
    LEGACY.write_csv(root / "VALIDATION_METRICS.csv", validation_rows)
    json_dump(root / "TARGET_REFERENCES.json", target_records)
    json_dump(root / "CFRA_TEACHERS.json", teacher_records)
    json_dump(root / "CHECKPOINT_HASHES.json", hashes)
    expected = len(BUDGETS) * len(METHODS) * len(PREDICTOR_SEEDS) * 2
    if len(hashes) != expected:
        raise RuntimeError(f"checkpoint count mismatch: {len(hashes)} != {expected}")
    marker = {"version": VERSION, "phase": "fit", "status": "PASS", "protocol_sha256": protocol_hash(args.protocol_file), "manifest_sha256": sha256_file(root / "MANIFEST.csv"), "budgets": list(BUDGETS), "methods": list(METHODS), "master_seeds": list(PREDICTOR_SEEDS), "teacher_regime": "fit-compound-OOF-1R-view; validation audit teacher for validation only", "checkpoint_count": len(hashes), "target_references_sha256": sha256_file(root / "TARGET_REFERENCES.json"), "checkpoint_hashes_sha256": sha256_file(root / "CHECKPOINT_HASHES.json"), "test_loaded": False, "test_values_opened": False, "test_used_for_selection": False, "bootstrap_unit": "compound", "bootstrap_rounds": args.bootstrap_rounds}
    json_dump(root / "FIT_COMPLETE.json", marker)
    json_dump(root / "SELECTION_FREEZE.json", {"version": VERSION, "status": "PASS", "selection_frozen": True, "protocol_sha256": protocol_hash(args.protocol_file), "manifest_sha256": sha256_file(root / "MANIFEST.csv"), "checkpoint_count": len(hashes), "test_loaded": False, "test_used_for_selection": False})
    print(json.dumps({"phase": "fit", "status": "PASS", "checkpoint_count": len(hashes), "test_loaded": False}, indent=2))


def bootstrap_pair(left: np.ndarray, right: np.ndarray, compounds: np.ndarray, seed: int, rounds: int) -> tuple[float, float, float, int]:
    return LEGACY.bootstrap_difference(left, right, compounds, seed, rounds)


def classify_primary(point: float, low: float, high: float, e_point: float, e_low: float, e_high: float) -> str:
    if low > 0:
        return "CFRA-GO"
    if high < 0:
        return "CFRA-NO-GO"
    if e_low > 0 and point <= 0:
        return "DISCRIMINABILITY-ONLY"
    return "INCONCLUSIVE"


def test_phase_legacy_unused(args: argparse.Namespace, root: Path) -> None:
    require_prepare(args, root)
    fit_marker = root / "FIT_COMPLETE.json"
    freeze_marker = root / "SELECTION_FREEZE.json"
    if not fit_marker.is_file() or not freeze_marker.is_file():
        raise RuntimeError("test requires FIT_COMPLETE and SELECTION_FREEZE")
    if (root / "TEST_AUTHORIZED.json").exists() or (root / "TEST_COMPLETE.json").exists():
        raise FileExistsError("test authorization/result already exists; use a fresh output root")
    fit_audit = read_json(fit_marker)
    selection = read_json(freeze_marker)
    if fit_audit.get("test_loaded") or fit_audit.get("test_values_opened") or fit_audit.get("test_used_for_selection"):
        raise RuntimeError("fit audit is not test sealed")
    if selection.get("selection_frozen") is not True or selection.get("test_loaded"):
        raise RuntimeError("selection freeze is invalid")
    expected_checkpoints = len(BUDGETS) * len(METHODS) * len(PREDICTOR_SEEDS)
    checkpoint_hashes = read_json(root / "CHECKPOINT_HASHES.json")
    if len(checkpoint_hashes) != expected_checkpoints:
        raise RuntimeError(f"expected {expected_checkpoints} checkpoints, found {len(checkpoint_hashes)}")
    for key, expected_hash in checkpoint_hashes.items():
        budget, method, seed_name = key.split("/")
        path = root / budget / method / f"{seed_name}.pt"
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
    if sha256_file(root / "MANIFEST.csv") != selection.get("manifest_sha256"):
        raise RuntimeError("manifest hash mismatch at test authorization")
    # This marker is intentionally written before any test row/value is loaded.
    auth = {
        "version": VERSION,
        "decision": "TEST-AUTHORIZED",
        "selection_frozen": True,
        "protocol_sha256": protocol_hash(args.protocol_file),
        "manifest_sha256": sha256_file(root / "MANIFEST.csv"),
        "checkpoint_count": expected_checkpoints,
        "test_loaded_before_authorization": False,
        "test_values_opened_before_authorization": False,
        "test_used_for_selection": False,
        "status": "AUTHORIZED",
    }
    json_dump(root / "TEST_AUTHORIZED.json", auth)
    # First test value access occurs after the authorization marker above.
    rows = load_rows(args.rows_root, ("test",))
    assert_lock(rows, args.split_lock)
    bundles = build_bundles(rows, include_foreign=True)
    virtual = load_virtual_splits(args.model_h5, args.aggregate_npz, args.split_lock, ("test",))
    device = device_for(args.device)
    score_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    predictions_index: list[dict[str, Any]] = []
    cache: dict[tuple[int, str, int], dict[str, np.ndarray]] = {}
    for budget in BUDGETS:
        examples = bundles[budget]["test"]
        test_split = subset_virtual(virtual["test"], examples, examples.target)
        budget_dir = root / f"budget{budget}"
        for seed in PREDICTOR_SEEDS:
            for method in METHODS:
                ckpt_path = budget_dir / method / f"seed{seed}.pt"
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = LEGACY.Student().to(device)
                model.load_state_dict(checkpoint["state_dict"])
                stats = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint["stats"].items()}
                prediction = LEGACY.predict_student(model, test_split, stats, args.batch_size, device)
                metrics = LEGACY.metric_arrays(prediction, examples)
                overlap, direction = LEGACY.top25_metrics(prediction, examples.target)
                cache[(budget, method, seed)] = {
                    "z_same": metrics["same_z"],
                    "z_foreign": metrics["foreign_z"],
                    "E": metrics["excess_z"],
                    "pcc": metrics["pcc"],
                    "foreign_pcc": metrics["foreign_pcc"],
                    "delta_pcc": metrics["delta_pcc"],
                    "top25_overlap": overlap,
                    "top25_direction": direction,
                }
                pred_path = budget_dir / f"seed{seed}_test_{method}_predictions.npz"
                np.savez_compressed(
                    pred_path,
                    compound=test_split.compound,
                    dose=test_split.dose,
                    held_target=test_split.target,
                    prediction=prediction,
                    foreign=examples.foreign,
                    foreign_ok=examples.foreign_ok.astype(np.uint8),
                )
                predictions_index.append({
                    "budget": budget,
                    "method": method,
                    "seed": seed,
                    "path": str(pred_path),
                    "sha256": sha256_file(pred_path),
                    "shape": str(tuple(prediction.shape)),
                })
                for i, compound in enumerate(test_split.compound):
                    score_rows.append({
                        "budget": budget,
                        "method": method,
                        "seed": seed,
                        "compound_id": str(compound),
                        "dose": str(test_split.dose[i]),
                        "z_same": cache[(budget, method, seed)]["z_same"][i],
                        "z_foreign": cache[(budget, method, seed)]["z_foreign"][i],
                        "E": cache[(budget, method, seed)]["E"][i],
                        "pcc": cache[(budget, method, seed)]["pcc"][i],
                        "foreign_pcc": cache[(budget, method, seed)]["foreign_pcc"][i],
                        "delta_pcc": cache[(budget, method, seed)]["delta_pcc"][i],
                        "top25_overlap": cache[(budget, method, seed)]["top25_overlap"][i],
                        "top25_direction": cache[(budget, method, seed)]["top25_direction"][i],
                    })
        compounds = np.asarray(examples.compound, dtype=str)
        for metric in ("z_same", "z_foreign", "E", "pcc", "foreign_pcc", "delta_pcc", "top25_overlap", "top25_direction"):
            per_seed = []
            for seed in PREDICTOR_SEEDS:
                left = cache[(budget, "cfra", seed)][metric]
                right = cache[(budget, "raw_mean", seed)][metric]
                point, low, high, n = bootstrap_pair(left, right, compounds, LEGACY.stable_int(PROTOCOL_SEED, f"{VERSION}|test|b{budget}|{metric}|seed{seed}"), args.bootstrap_rounds)
                per_seed.append(point)
                contrast_rows.append({
                    "budget": budget,
                    "comparison": "cfra_minus_raw_mean",
                    "metric": metric,
                    "seed": seed,
                    "mean_difference": point,
                    "ci_low": low,
                    "ci_high": high,
                    "n_compounds": n,
                })
            left_avg = LEGACY.nanmean_across_seeds([cache[(budget, "cfra", seed)][metric] for seed in PREDICTOR_SEEDS])
            right_avg = LEGACY.nanmean_across_seeds([cache[(budget, "raw_mean", seed)][metric] for seed in PREDICTOR_SEEDS])
            point, low, high, n = bootstrap_pair(left_avg, right_avg, compounds, LEGACY.stable_int(PROTOCOL_SEED, f"{VERSION}|test|b{budget}|{metric}|seed-averaged"), args.bootstrap_rounds)
            contrast_rows.append({
                "budget": budget,
                "comparison": "cfra_minus_raw_mean",
                "metric": metric,
                "seed": "seed_averaged",
                "mean_difference": point,
                "ci_low": low,
                "ci_high": high,
                "n_compounds": n,
                "positive_seed_count": sum(value > 0 for value in per_seed),
            })
    LEGACY.write_csv(root / "TEST_PER_COMPOUND.csv", score_rows)
    LEGACY.write_csv(root / "TEST_CONTRASTS.csv", contrast_rows)
    LEGACY.write_csv(root / "TEST_PREDICTION_INDEX.csv", predictions_index)
    summary_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        row: dict[str, Any] = {"budget": budget, "comparison": "cfra_minus_raw_mean"}
        entries = {str(x["metric"]): x for x in contrast_rows if x["budget"] == budget and x["seed"] == "seed_averaged"}
        z = entries["z_same"]
        foreign = entries["z_foreign"]
        e = entries["E"]
        row.update({
            "n_compounds": z["n_compounds"],
            "z_same_delta": z["mean_difference"],
            "z_same_ci_low": z["ci_low"],
            "z_same_ci_high": z["ci_high"],
            "z_foreign_delta": foreign["mean_difference"],
            "z_foreign_ci_low": foreign["ci_low"],
            "z_foreign_ci_high": foreign["ci_high"],
            "E_delta": e["mean_difference"],
            "E_ci_low": e["ci_low"],
            "E_ci_high": e["ci_high"],
            "positive_z_same_seeds": z["positive_seed_count"],
            "classification": classify_primary(z["mean_difference"], z["ci_low"], z["ci_high"], e["mean_difference"], e["ci_low"], e["ci_high"]),
        })
        summary_rows.append(row)
    LEGACY.write_csv(root / "TEST_SUMMARY.csv", summary_rows)
    test_hashes = {
        "test_cp_plate_rows": sha256_file(args.rows_root / "test_cp_plate_rows.npz"),
        "model_h5": sha256_file(args.model_h5),
        "aggregate_npz": sha256_file(args.aggregate_npz),
    }
    json_dump(root / "TEST_COMPLETE.json", {
        "version": VERSION,
        "phase": "test",
        "protocol_sha256": protocol_hash(args.protocol_file),
        "manifest_sha256": sha256_file(root / "MANIFEST.csv"),
        "test_source_hashes": test_hashes,
        "test_loaded": True,
        "test_values_opened": True,
        "test_used_for_selection": False,
        "test_loaded_after_authorization": True,
        "bootstrap_unit": "compound",
        "bootstrap_rounds": args.bootstrap_rounds,
        "status": "PASS",
    })
    print(json.dumps({"phase": "test", "root": str(root), "test_loaded": True, "summary": summary_rows}, indent=2))


def predict_saved_m2(root: Path, budget: int, method: str, seed: int, split: Any, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    branches = ("raw_branch_a", "raw_branch_b") if method == M0 else ("cfra1_view_aggregate_base", "raw_minus_cfra_correction")
    outputs: list[np.ndarray] = []
    for branch in branches:
        path = root / f"budget{budget}" / method / f"seed{seed}" / f"{branch}.pt"
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if tuple(checkpoint.get("hidden_dims", ())) != M2_DIMS or checkpoint.get("parameter_count") != M2_BRANCH_PARAMETERS:
            raise RuntimeError(f"invalid M2 checkpoint: {path}")
        model = CAP.StudentWidth(*M2_DIMS).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        outputs.append(CAP.predict_single(model, split, checkpoint["stats"], args.batch_size, device))
        del model
    prediction = (outputs[0] + outputs[1]) / np.float32(2.0) if method == M0 else outputs[0] + outputs[1]
    if prediction.shape != (len(split.compound), CP_DIM) or not np.isfinite(prediction).all():
        raise RuntimeError("invalid final M0/M2 prediction")
    return np.asarray(prediction, dtype=np.float32)


def test_phase(args: argparse.Namespace, root: Path) -> None:
    prepare = require_prepare(args, root)
    fit = read_json(root / "FIT_COMPLETE.json")
    freeze = read_json(root / "SELECTION_FREEZE.json")
    if fit.get("status") != "PASS" or fit.get("test_loaded") or fit.get("test_values_opened") or not freeze.get("selection_frozen"):
        raise RuntimeError("fit/selection barriers are not sealed")
    if (root / "TEST_AUTHORIZED.json").exists() or (root / "TEST_COMPLETE.json").exists():
        raise FileExistsError("test is single-use; create a fresh root to rerun")
    hashes = read_json(root / "CHECKPOINT_HASHES.json")
    expected = len(BUDGETS) * len(METHODS) * len(PREDICTOR_SEEDS) * 2
    if len(hashes) != expected:
        raise RuntimeError("checkpoint ledger is incomplete")
    json_dump(root / "TEST_AUTHORIZED.json", {"version": VERSION, "status": "AUTHORIZED", "selection_frozen": True, "fit_marker_sha256": sha256_file(root / "FIT_COMPLETE.json"), "test_loaded_before_authorization": False, "test_used_for_selection": False})
    rows = load_rows(args.rows_root, ("test",))
    assert_lock(rows, args.split_lock)
    bundles = build_bundles(rows, include_foreign=True)
    virtual = load_virtual_splits(args.model_h5, args.aggregate_npz, args.split_lock, ("test",))
    device = device_for(args.device)
    per_compound: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, str, int], dict[str, np.ndarray]] = {}
    metrics = ("same_z", "foreign_z", "excess_z", "pcc", "foreign_pcc", "delta_pcc")
    for budget in BUDGETS:
        examples = bundles[budget]["test"]
        split = subset_virtual(virtual["test"], examples, examples.target)
        for method in METHODS:
            for seed in PREDICTOR_SEEDS:
                prediction = predict_saved_m2(root, budget, method, seed, split, args, device)
                payload = LEGACY.metric_arrays(prediction, examples)
                cache[(budget, method, seed)] = payload
                path = root / f"budget{budget}" / f"seed{seed}_{method}_test_predictions.npz"
                np.savez_compressed(path, compound=examples.compound, dose=examples.dose, prediction=prediction, raw_held_target=examples.target, foreign_ok=examples.foreign_ok.astype(np.uint8))
                index_rows.append({"budget": budget, "method": method, "seed": seed, "path": str(path.relative_to(root)), "sha256": sha256_file(path), "shape": str(tuple(prediction.shape))})
                for i, compound in enumerate(examples.compound):
                    per_compound.append({"budget": budget, "method": method, "seed": seed, "compound_id": str(compound), "dose": str(examples.dose[i]), "foreign_ok": int(examples.foreign_ok[i]), **{key: float(payload[key][i]) for key in metrics}})
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    contrast_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        examples = bundles[budget]["test"]
        compounds = np.asarray(examples.compound, dtype=str)
        summary = {"budget": budget, "comparison": f"{M2}-{M0}", "n_compounds": len(compounds), "foreign_compounds": int(np.sum(examples.foreign_ok))}
        for metric in metrics:
            left = LEGACY.nanmean_across_seeds([cache[(budget, M2, seed)][metric] for seed in PREDICTOR_SEEDS])
            right = LEGACY.nanmean_across_seeds([cache[(budget, M0, seed)][metric] for seed in PREDICTOR_SEEDS])
            point, low, high, n = bootstrap_pair(left, right, compounds, LEGACY.stable_int(PROTOCOL_SEED, f"{VERSION}|test|b{budget}|{metric}"), args.bootstrap_rounds)
            summary[f"{metric}_delta"] = point; summary[f"{metric}_ci_low"] = low; summary[f"{metric}_ci_high"] = high; summary[f"{metric}_n"] = n
            contrast_rows.append({"budget": budget, "comparison": f"{M2}-{M0}", "metric": metric, "seed": "seed_averaged", "estimate": point, "ci_low": low, "ci_high": high, "n_compounds": n, "rounds": args.bootstrap_rounds})
        summary["complete_profile_reconstruction_decision"] = "GO" if summary["same_z_ci_low"] > 0 and summary["pcc_ci_low"] > 0 else ("NO-GO" if summary["same_z_ci_high"] < 0 and summary["pcc_ci_high"] < 0 else "INCONCLUSIVE")
        summary_rows.append(summary)
    LEGACY.write_csv(root / "TEST_PER_COMPOUND.csv", per_compound)
    LEGACY.write_csv(root / "TEST_PREDICTION_INDEX.csv", index_rows)
    LEGACY.write_csv(root / "TEST_CONTRASTS.csv", contrast_rows)
    LEGACY.write_csv(root / "TEST_SUMMARY.csv", summary_rows)
    json_dump(root / "TEST_COMPLETE.json", {"version": VERSION, "phase": "test", "status": "PASS", "protocol_sha256": protocol_hash(args.protocol_file), "prepare_marker_sha256": sha256_file(root / "PREPARE_COMPLETE.json"), "fit_marker_sha256": sha256_file(root / "FIT_COMPLETE.json"), "selection_marker_sha256": sha256_file(root / "SELECTION_FREEZE.json"), "test_loaded_after_authorization": True, "test_used_for_selection": False, "budgets": list(BUDGETS), "methods": list(METHODS), "master_seeds": list(PREDICTOR_SEEDS), "bootstrap_unit": "compound", "bootstrap_rounds": args.bootstrap_rounds, "primary_comparison": f"{M2}-{M0}", "raw_target": "independent physical held Raw CP profile", "summary_path": "TEST_SUMMARY.csv"})
    print(json.dumps({"phase": "test", "status": "PASS", "summary": str(root / "TEST_SUMMARY.csv")}, indent=2))


def prepare_phase(args: argparse.Namespace, root: Path) -> None:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {root}")
    root.mkdir(parents=True)
    rows = load_rows(args.rows_root, ("train", "valid"))
    assert_lock(rows, args.split_lock)
    bundles = build_bundles(rows, include_foreign=False)
    manifest = manifest_rows(bundles)
    write_manifest(root / "MANIFEST.csv", manifest)
    audit = preflight_audit(args, root, rows, bundles)
    audit["manifest_sha256"] = sha256_file(root / "MANIFEST.csv")
    audit["manifest_file"] = str(root / "MANIFEST.csv")
    audit["test_data_paths"] = {
        "test_cp_plate_rows": str(args.rows_root / "test_cp_plate_rows.npz"),
        "test_h5": str(args.model_h5),
    }
    json_dump(root / "PREPARE_COMPLETE.json", audit)
    json_dump(root / "RUN_CONFIG.json", {
        "version": VERSION,
        "protocol_sha256": audit["protocol_sha256"],
        "rows_root": str(args.rows_root),
        "model_h5": str(args.model_h5),
        "aggregate_npz": str(args.aggregate_npz),
        "split_lock": str(args.split_lock),
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "bootstrap_rounds": args.bootstrap_rounds,
        "test_loaded": False,
    })
    print(json.dumps({"phase": "prepare", "root": str(root), "manifest_sha256": audit["manifest_sha256"], "test_loaded": False, "eligible_compounds": audit["eligible_compounds"]}, indent=2))


def main() -> None:
    args = parse_args()
    root = args.output_root
    if args.phase == "prepare":
        prepare_phase(args, root)
    elif args.phase == "fit":
        fit_phase(args, root)
    elif args.phase == "test":
        test_phase(args, root)


if __name__ == "__main__":
    main()
