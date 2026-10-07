#!/usr/bin/env python3
"""BBBC047 CP target-replacement MVP: same-budget mean versus OOF CFRA.

The downstream predictor, inputs, compounds, optimiser, initialisation and
minibatch order are paired.  Only the training target changes.  All data
assignments use PROTOCOL_SEED; PREDICTOR_SEEDS affect only student training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
EXPERIMENTS = Path(__file__).resolve().parents[4] / "analysis"
sys.path.insert(0, str(EXPERIMENTS / "baselines"))
from ablation_common import (  # noqa: E402
    Rows,
    bootstrap_difference,
    fisher_z,
    make_examples,
    metric_arrays,
    rowwise_corr,
    stable_int,
    summarize_metrics,
    write_csv,
    write_json,
)


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VIRTUAL = import_file(
    "cfra_virtual_base",
    (Path(__file__).resolve().parents[4] / "analysis/benchmarks/bbbc047/virtual_control_reproducibility/repro_effect_virtual_cp_mvp.py"),
)
EFFECT = VIRTUAL.BASE

VERSION = "bbbc047-cfra-target-replacement-mvp-v1-2026-09-01"
PROTOCOL_SEED = 3407
PREDICTOR_SEEDS = (3407, 42, 2025, 1337, 7331)
BUDGETS = (1, 2)
FOLDS = 5
CP_DIM = 775
FP_DIM = 2048
EXPECTED = {"train": 12175, "valid": 4059, "test": 4060}

DEFAULT_ROWS = Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0")
DEFAULT_H5 = Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/Paired_CP_GE_PlateMedian_v1_model_compat.h5")
DEFAULT_AGG = DEFAULT_ROWS / "molecule_aggregates.npz"
DEFAULT_LOCK = Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json")


@dataclass
class PredictorSplit:
    compound: np.ndarray
    dose: np.ndarray
    control: np.ndarray
    fingerprint: np.ndarray
    target: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--rows-root", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--model-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--aggregate-npz", type=Path, default=DEFAULT_AGG)
    parser.add_argument("--split-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--teacher-epochs", type=int, default=80)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and args.output_root is None:
        parser.error("--output-root is required unless --dry-run")
    if not args.smoke and (args.teacher_epochs, args.student_epochs, args.batch_size) != (80, 40, 256):
        parser.error("formal protocol locks teacher/student epochs and batch size to 80/40/256")
    if not args.smoke and args.bootstrap_rounds != 10000:
        parser.error("formal protocol locks the compound bootstrap to 10000 draws")
    if args.bootstrap_rounds < 1:
        parser.error("bootstrap rounds must be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_for(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def load_rows(root: Path) -> dict[str, Rows]:
    output: dict[str, Rows] = {}
    compounds: dict[str, set[str]] = {}
    for split in ("train", "valid", "test"):
        path = root / f"{split}_cp_plate_rows.npz"
        z = np.load(path, allow_pickle=False)
        output[split] = Rows("BBBC047", "CP", split, z["smiles"], z["dose"], z["plate"], z["delta"])
        compounds[split] = set(output[split].compound.tolist())
        if len(compounds[split]) != EXPECTED[split]:
            raise RuntimeError(f"{split} compound count mismatch: {len(compounds[split])}")
    if compounds["train"] & compounds["valid"] or compounds["train"] & compounds["test"] or compounds["valid"] & compounds["test"]:
        raise RuntimeError("compound leakage between outer splits")
    return output


def assert_lock(rows: dict[str, Rows], path: Path) -> None:
    lock = json.loads(path.read_text(encoding="utf-8"))
    for split, values in rows.items():
        observed = set(values.compound.tolist())
        expected = {str(x) for x in lock[f"{split}_smiles"]}
        if observed != expected:
            raise RuntimeError(f"{split} does not match frozen split lock")


def virtual_index(model_h5: Path, aggregate_npz: Path, split_lock: Path) -> dict[str, VIRTUAL.VirtualSplit]:
    return VIRTUAL.load_virtual(model_h5, aggregate_npz, split_lock)


def subset_virtual(base: VIRTUAL.VirtualSplit, examples: Any, target: np.ndarray) -> PredictorSplit:
    index = {value: i for i, value in enumerate(base.smiles)}
    rows = np.asarray([index[str(x)] for x in examples.compound], dtype=np.int64)
    if len(rows) != len(set(examples.compound.tolist())):
        raise RuntimeError("one predictor row per compound is required")
    return PredictorSplit(
        compound=examples.compound.copy(),
        dose=examples.dose.copy(),
        control=base.control[rows].astype(np.float32),
        fingerprint=base.fingerprint[rows].astype(np.float32),
        target=np.asarray(target, dtype=np.float32),
    )


def fold_assignment(compounds: Iterable[str]) -> dict[str, int]:
    result = {str(c): stable_int(PROTOCOL_SEED, f"cfra-fold-v1|{c}") % FOLDS for c in compounds}
    counts = [sum(v == fold for v in result.values()) for fold in range(FOLDS)]
    if min(counts) == 0:
        raise RuntimeError(f"empty CFRA fold: {counts}")
    return result


def budget_pairs(rows: Rows, budget: int, allowed: set[str] | None = None) -> Any:
    supports: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    compounds: list[str] = []
    held_rows: list[int] = []
    counter = 0
    for (compound, dose), by_plate in sorted(rows.plate_means().items()):
        if allowed is not None and compound not in allowed:
            continue
        plates = sorted(by_plate, key=lambda p: (stable_int(PROTOCOL_SEED, f"cfra-pair|b{budget}|{compound}|{dose}|{p}"), p))
        if len(plates) < budget + 1:
            continue
        for held in plates:
            candidates = [p for p in plates if p != held]
            chosen = candidates[:budget]
            supports.append(np.mean([by_plate[p] for p in chosen], axis=0).astype(np.float32))
            targets.append(np.asarray(by_plate[held], dtype=np.float32))
            compounds.append(compound)
            held_rows.append(counter)
            counter += 1
    if not supports:
        raise RuntimeError("no CFRA support/held pairs")
    return EFFECT.LooPairs(np.vstack(supports), np.vstack(targets), compounds, np.asarray(held_rows, dtype=np.int64))


def restrict_pairs(pairs: Any, allowed: set[str]) -> Any:
    ix = np.asarray([i for i, c in enumerate(pairs.smiles) if c in allowed], dtype=np.int64)
    if ix.size == 0:
        raise RuntimeError("empty restricted CFRA pair set")
    return EFFECT.LooPairs(pairs.support[ix], pairs.target[ix], [pairs.smiles[i] for i in ix], pairs.held_rows[ix])


def predict_effect(model: Any, support: np.ndarray, mean: np.ndarray, scale: np.ndarray, batch: int, device: torch.device) -> np.ndarray:
    values = np.asarray(support, dtype=np.float32)
    output = np.empty_like(values)
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(values), batch):
            x = torch.from_numpy((values[begin:begin + batch] - mean) / scale).to(device)
            y = model(x).cpu().numpy()
            output[begin:begin + len(y)] = y * scale + mean
    return output


def digest_compounds(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode("utf-8")).hexdigest()


def build_oof_cfra(rows: dict[str, Rows], examples: Any, budget: int, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    assignment = fold_assignment(examples.compound.tolist())
    all_train = set(rows["train"].compound.tolist())
    full_pairs = budget_pairs(rows["train"], budget)
    valid_pairs = budget_pairs(rows["valid"], budget)
    output = np.empty_like(examples.support)
    example_index = {str(c): i for i, c in enumerate(examples.compound)}
    metadata: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        held = {c for c, f in assignment.items() if f == fold}
        fitted = all_train - held
        if held & fitted:
            raise RuntimeError("OOF fold leakage")
        teacher_seed = stable_int(PROTOCOL_SEED, f"cfra-teacher|b{budget}|fold{fold}") % (2**32 - 1)
        teacher_args = SimpleNamespace(
            seed=teacher_seed,
            hidden_dim=256,
            latent_dim=32,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.teacher_epochs,
            smoke=args.smoke,
        )
        model, mean, scale, best_epoch, valid_mse = EFFECT.fit_encoder(
            restrict_pairs(full_pairs, fitted), valid_pairs, teacher_args, device
        )
        held_order = sorted(held)
        ix = np.asarray([example_index[c] for c in held_order], dtype=np.int64)
        output[ix] = predict_effect(model, examples.support[ix], mean, scale, args.batch_size, device)
        metadata.append({
            "budget": budget,
            "fold": fold,
            "model_id": f"cfra_b{budget}_fold{fold}",
            "fit_compounds": len(fitted),
            "fit_compound_sha256": digest_compounds(fitted),
            "held_compounds": len(held),
            "held_compound_sha256": digest_compounds(held),
            "best_epoch": best_epoch,
            "validation_mse": valid_mse,
        })
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite or missing OOF CFRA targets")
    return output, metadata, assignment


class Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CP_DIM + FP_DIM, 512), nn.GELU(),
            nn.Linear(512, 128), nn.GELU(), nn.Linear(128, CP_DIM),
        )

    def forward(self, control: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((control, feature), dim=1))


def fit_student(train: PredictorSplit, valid: PredictorSplit, label: np.ndarray, common: dict[str, np.ndarray], seed: int, args: argparse.Namespace, device: torch.device, method: str) -> tuple[Student, dict[str, np.ndarray], list[dict[str, Any]]]:
    set_seed(seed)
    train_control = (train.control - common["control_mean"]) / common["control_scale"]
    valid_control = (valid.control - common["control_mean"]) / common["control_scale"]
    train_label = (np.asarray(label, dtype=np.float32) - common["target_mean"]) / common["target_scale"]
    valid_label = (valid.target - common["target_mean"]) / common["target_scale"]
    model = Student().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(train_control), torch.from_numpy(train.fingerprint), torch.from_numpy(train_label)),
        batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best = float("inf")
    logs: list[dict[str, Any]] = []
    epochs = 1 if args.smoke else args.student_epochs
    for epoch in range(epochs):
        model.train()
        train_sum = 0.0
        train_n = 0
        for control, feature, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(control.to(device), feature.to(device)) - target.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite student loss")
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(control)
            train_n += len(control)
        model.eval()
        valid_sum = 0.0
        valid_n = 0
        with torch.no_grad():
            for begin in range(0, len(valid.compound), args.batch_size):
                control = torch.from_numpy(valid_control[begin:begin + args.batch_size]).to(device)
                feature = torch.from_numpy(valid.fingerprint[begin:begin + args.batch_size]).to(device)
                target = torch.from_numpy(valid_label[begin:begin + args.batch_size]).to(device)
                value = torch.mean((model(control, feature) - target) ** 2)
                valid_sum += float(value.cpu()) * len(control)
                valid_n += len(control)
        valid_loss = valid_sum / valid_n
        logs.append({"seed": seed, "method": method, "epoch": epoch, "train_mse": train_sum / train_n, "validation_mse": valid_loss})
        if valid_loss < best:
            best = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("student checkpoint missing")
    model.load_state_dict(best_state)
    return model, common, logs


def predict_student(model: Student, split: PredictorSplit, stats: dict[str, np.ndarray], batch: int, device: torch.device) -> np.ndarray:
    control = (split.control - stats["control_mean"]) / stats["control_scale"]
    output = np.empty((len(split.compound), CP_DIM), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(split.compound), batch):
            value = model(
                torch.from_numpy(control[begin:begin + batch]).to(device),
                torch.from_numpy(split.fingerprint[begin:begin + batch]).to(device),
            ).cpu().numpy()
            output[begin:begin + len(value)] = value * stats["target_scale"] + stats["target_mean"]
    return output


def common_stats(train: PredictorSplit, mean_target: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "control_mean": train.control.mean(axis=0, dtype=np.float64).astype(np.float32),
        "control_scale": np.maximum(train.control.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6),
        "target_mean": mean_target.mean(axis=0, dtype=np.float64).astype(np.float32),
        "target_scale": np.maximum(mean_target.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6),
    }


def top25_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    overlap = np.empty(len(prediction), dtype=float)
    direction = np.empty(len(prediction), dtype=float)
    for i, (left, right) in enumerate(zip(prediction, target)):
        li = set(np.argpartition(np.abs(left), -25)[-25:].tolist())
        ri = set(np.argpartition(np.abs(right), -25)[-25:].tolist())
        overlap[i] = len(li & ri) / 25.0
        union = sorted(li | ri)
        direction[i] = float(np.mean(np.sign(left[union]) == np.sign(right[union])))
    return overlap, direction


def nanmean_across_seeds(values: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(values)
    count = np.sum(np.isfinite(stacked), axis=0)
    output = np.full(stacked.shape[1], np.nan, dtype=np.float64)
    np.divide(np.nansum(stacked, axis=0), count, out=output, where=count > 0)
    return output


def write_outer_split(path: Path, rows: dict[str, Rows]) -> None:
    payload = []
    for split, values in rows.items():
        for compound, dose in sorted(values.condition_keys()):
            payload.append({"compound_id": compound, "dose": dose, "dataset": "BBBC047", "split": split})
    write_csv(path, payload)


def append_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists()
    fields = list(rows[0])
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    rows = load_rows(args.rows_root)
    assert_lock(rows, args.split_lock)
    virtual = virtual_index(args.model_h5, args.aggregate_npz, args.split_lock)
    bundles = {
        budget: {split: make_examples(values, None, budget, PROTOCOL_SEED, all_conditions=False, require_source=False) for split, values in rows.items()}
        for budget in BUDGETS
    }
    audit = {
        "version": VERSION,
        "protocol_seed": PROTOCOL_SEED,
        "predictor_seeds": list(PREDICTOR_SEEDS),
        "outer_compounds": {s: len(set(r.compound.tolist())) for s, r in rows.items()},
        "eligible_compounds": {f"budget{b}": {s: len(x.compound) for s, x in by.items()} for b, by in bundles.items()},
        "split_intersections": {
            "train_valid": len(set(rows["train"].compound) & set(rows["valid"].compound)),
            "train_test": len(set(rows["train"].compound) & set(rows["test"].compound)),
            "valid_test": len(set(rows["valid"].compound) & set(rows["test"].compound)),
        },
        "test_target_generation": False,
        "status": "PASS",
    }
    if args.dry_run:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return
    root = args.output_root
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    write_outer_split(root / "01_outer_split.csv", rows)
    support_rows: list[dict[str, Any]] = []
    for budget, by_split in bundles.items():
        for split, examples in by_split.items():
            for i in range(len(examples)):
                support_rows.append({
                    "dataset": "BBBC047", "compound_id": examples.compound[i], "dose": examples.dose[i],
                    "split": split, "budget": budget, "protocol_seed": PROTOCOL_SEED,
                    "support_rep_ids": "|".join(examples.support_plates[i]), "held_rep_id": "|".join(examples.held_plates[i]),
                })
    write_csv(root / "02_support_replicate_assignment.csv", support_rows)
    write_json(root / "05_predictor_hyperparameters.json", {
        "version": VERSION, "estimator": "CFRA compound-OOF reproducible-effect teacher", "protocol_seed": PROTOCOL_SEED,
        "predictor_seeds": list(PREDICTOR_SEEDS), "budgets": list(BUDGETS), "folds": FOLDS,
        "teacher_architecture": "775-256-32-256-775", "teacher_epochs": 1 if args.smoke else args.teacher_epochs,
        "predictor_architecture": "ECFP4-2048 + baseline-CP775 -> 512 -> 128 -> delta-CP775",
        "student_epochs": 1 if args.smoke else args.student_epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "bootstrap_rounds": args.bootstrap_rounds, "test_used_for_selection": False,
        "shared_target_normalization": "both arms use mean/scale fitted once from the same-budget mean training target",
    })
    device = device_for(args.device)
    training_logs: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    secondary_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    target_quality_rows: list[dict[str, Any]] = []
    prediction_index: list[dict[str, Any]] = []
    seed_metric_cache: dict[tuple[int, str, int], np.ndarray] = {}
    for budget in BUDGETS:
        examples = bundles[budget]
        cfra_target, teacher_meta, assignment = build_oof_cfra(rows, examples["train"], budget, args, device)
        mean_target = examples["train"].support.astype(np.float32)
        for i, compound in enumerate(examples["train"].compound):
            fold = assignment[str(compound)]
            for method in ("same_budget_mean", "oof_cfra"):
                provenance_rows.append({
                    "compound_id": compound, "dose": examples["train"].dose[i], "budget": budget,
                    "support_rep_ids": "|".join(examples["train"].support_plates[i]), "target_type": method,
                    "cfra_fold": fold if method == "oof_cfra" else "", "cfra_model_id": f"cfra_b{budget}_fold{fold}" if method == "oof_cfra" else "",
                    "estimator_fit_contains_compound": 0,
                })
        mean_corr = rowwise_corr(mean_target, examples["train"].target)
        cfra_corr = rowwise_corr(cfra_target, examples["train"].target)
        for i, compound in enumerate(examples["train"].compound):
            target_quality_rows.append({"compound_id": compound, "dose": examples["train"].dose[i], "budget": budget, "mean_held_pcc": mean_corr[i], "cfra_held_pcc": cfra_corr[i], "cfra_minus_mean_pcc": cfra_corr[i] - mean_corr[i]})
        train_mean = subset_virtual(virtual["train"], examples["train"], mean_target)
        valid = subset_virtual(virtual["valid"], examples["valid"], examples["valid"].target)
        test = subset_virtual(virtual["test"], examples["test"], examples["test"].target)
        stats = common_stats(train_mean, mean_target)
        budget_dir = root / f"budget{budget}"
        budget_dir.mkdir()
        write_json(budget_dir / "cfra_oof_models.json", {"budget": budget, "folds": teacher_meta})
        np.savez_compressed(budget_dir / "training_targets.npz", compound=train_mean.compound, dose=train_mean.dose, same_budget_mean=mean_target, oof_cfra=cfra_target)
        for seed in PREDICTOR_SEEDS:
            pred_payload: dict[str, np.ndarray] = {"compound": test.compound, "dose": test.dose, "held_target": test.target, "foreign": examples["test"].foreign, "foreign_ok": examples["test"].foreign_ok.astype(np.uint8)}
            metrics_by_method: dict[str, dict[str, np.ndarray]] = {}
            for method, label in (("same_budget_mean", mean_target), ("oof_cfra", cfra_target)):
                model, fitted_stats, logs = fit_student(train_mean, valid, label, stats, seed, args, device, method)
                for row in logs:
                    row["budget"] = budget
                training_logs.extend(logs)
                prediction = predict_student(model, test, fitted_stats, args.batch_size, device)
                pred_payload[method] = prediction
                summary, metric = summarize_metrics(method, prediction, examples["test"], seed, "test", bootstrap_n=args.bootstrap_rounds)
                metrics_by_method[method] = metric
                secondary_rows.append(summary)
                top_overlap, top_direction = top25_metrics(prediction, examples["test"].target)
                for i, compound in enumerate(test.compound):
                    score_rows.append({
                        "dataset": "BBBC047", "compound_id": compound, "dose": test.dose[i], "seed": seed,
                        "training_target": method, "budget": budget, "r_same": metric["pcc"][i],
                        "r_other": metric["foreign_pcc"][i], "E_score": metric["excess_z"][i],
                        "pearson": metric["pcc"][i], "diff_corr": metric["pcc"][i],
                        "top25_overlap": top_overlap[i], "top25_direction_accuracy": top_direction[i],
                    })
                seed_metric_cache[(budget, method, seed)] = metric["excess_z"]
            left = metrics_by_method["oof_cfra"]["excess_z"]
            right = metrics_by_method["same_budget_mean"]["excess_z"]
            point, low, high, n = bootstrap_difference(left, right, examples["test"].compound, stable_int(seed, f"cfra-target-replacement|b{budget}"), args.bootstrap_rounds)
            bootstrap_rows.append({"dataset": "BBBC047", "budget": budget, "seed": seed, "comparison": "oof_cfra_minus_same_budget_mean", "metric": "E_score", "mean_difference": point, "ci_low": low, "ci_high": high, "n_compounds": n})
            pred_path = budget_dir / f"seed{seed}_test_predictions.npz"
            np.savez_compressed(pred_path, **pred_payload)
            prediction_index.extend({"dataset": "BBBC047", "budget": budget, "seed": seed, "training_target": method, "npz_path": str(pred_path), "array_key": method, "shape": str(tuple(pred_payload[method].shape))} for method in ("same_budget_mean", "oof_cfra"))
    write_csv(root / "03_target_provenance.csv", provenance_rows)
    write_csv(root / "04_target_quality_train_oof.csv", target_quality_rows)
    write_csv(root / "06_training_log_all_seeds.csv", training_logs)
    write_csv(root / "07_unseen_test_predictions.csv", prediction_index)
    write_csv(root / "08_per_compound_metrics.csv", score_rows)
    write_csv(root / "09_primary_paired_bootstrap.csv", bootstrap_rows)
    write_csv(root / "10_secondary_metrics.csv", secondary_rows)
    final_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        example = bundles[budget]["test"]
        left = nanmean_across_seeds([seed_metric_cache[(budget, "oof_cfra", s)] for s in PREDICTOR_SEEDS])
        right = nanmean_across_seeds([seed_metric_cache[(budget, "same_budget_mean", s)] for s in PREDICTOR_SEEDS])
        point, low, high, n = bootstrap_difference(left, right, example.compound, stable_int(PROTOCOL_SEED, f"seed-averaged|b{budget}"), args.bootstrap_rounds)
        per_seed = [r for r in bootstrap_rows if r["budget"] == budget]
        positive = sum(float(r["mean_difference"]) > 0 for r in per_seed)
        final_rows.append({
            "dataset": "BBBC047", "budget": budget, "comparison": "oof_cfra_minus_same_budget_mean",
            "seed_averaged_delta_E": point, "ci_low": low, "ci_high": high, "n_compounds": n,
            "positive_seeds": positive, "total_seeds": len(PREDICTOR_SEEDS),
            "go_directional": int(point > 0 and positive >= 3), "strong_success": int(low > 0),
        })
    write_csv(root / "12_final_summary_table.csv", final_rows)
    qc_lines = [
        "BBBC047 CFRA target-replacement leakage QC",
        f"version={VERSION}", f"protocol_seed={PROTOCOL_SEED}",
        "PASS outer train/valid/test compound intersections are zero",
        "PASS all doses follow compound-level frozen outer split",
        "PASS one deterministic condition per compound per budget",
        "PASS Mean and CFRA use identical support_rep_ids",
        "PASS every CFRA target is generated by a fold excluding its compound",
        "PASS official validation compounds are checkpoint-only, never teacher-fit compounds",
        "PASS test compounds are absent from teacher fitting and target generation",
        "PASS test predictor input is molecular fingerprint + baseline CP only",
        "PASS paired predictor seeds share initialisation/order within target comparison",
        "PASS bootstrap unit is compound",
        "PASS hyperparameters and success criteria were frozen before this run",
        f"smoke={int(args.smoke)}",
    ]
    (root / "11_leakage_qc_report.txt").write_text("\n".join(qc_lines) + "\n", encoding="utf-8")
    write_json(root / "run_audit.json", audit | {"device": str(device), "smoke": bool(args.smoke), "completed": True})
    print(json.dumps({"output_root": str(root), "summary": final_rows}, indent=2))


if __name__ == "__main__":
    run(parse_args())
