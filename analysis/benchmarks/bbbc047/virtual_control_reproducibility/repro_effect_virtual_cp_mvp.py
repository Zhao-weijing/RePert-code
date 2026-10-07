#!/usr/bin/env python3
"""CP-only virtual student: raw aggregate target versus OOF reproducible effect."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from torch import nn

VERSION = "BBBC047-Reproducible-Effect-Virtual-CP-MVP-2026-08-29"
SEEDS = (3407, 42, 2025)
CP_DIM, FP_DIM = 775, 2048
EXPECTED = {"train": 12175, "valid": 4059, "test": 4060}
RDLogger.DisableLog("rdApp.warning")


def import_base() -> Any:
    root = Path(__file__).resolve().parents[1]
    candidates = ((Path(__file__).resolve().parents[4] / "analysis/benchmarks/bbbc047/reproducibility_effect/repro_effect_mvp.py"), (Path(__file__).resolve().parents[4] / "analysis/benchmarks/bbbc047/reproducibility_effect/repro_effect_mvp.py"))
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError("Experiment-A code missing: " + ", ".join(map(str, candidates)))
    spec = importlib.util.spec_from_file_location("repro_effect_virtual_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = import_base()


@dataclass
class VirtualSplit:
    smiles: list[str]
    control: np.ndarray
    delta: np.ndarray
    fingerprint: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp-rows-root", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"))
    parser.add_argument("--aggregate-npz", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0/molecule_aggregates.npz"))
    parser.add_argument("--model-h5", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/Paired_CP_GE_PlateMedian_v1_model_compat.h5"))
    parser.add_argument("--split-lock", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json"))
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--aggregate-root", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--teacher-folds", type=int, default=5)
    parser.add_argument("--teacher-epochs", type=int, default=80)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--null-rounds", type=int, default=32)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.aggregate_root:
        if args.seed is not None or args.outdir is not None or args.dry_run or args.smoke:
            parser.error("--aggregate-root cannot be combined with run options")
        return args
    if args.seed is None and not args.dry_run:
        parser.error("--seed is required unless --dry-run or --aggregate-root is used")
    if not args.dry_run and args.outdir is None:
        parser.error("--outdir is required for a run")
    if (args.teacher_folds, args.teacher_epochs, args.student_epochs, args.batch_size) != (5, 80, 40, 256):
        parser.error("Experiment C locks folds=5, teacher_epochs=80, student_epochs=40, batch_size=256")
    if args.null_rounds < 1 or args.bootstrap_rounds < 1:
        parser.error("null and bootstrap rounds must be positive")
    return args


def fingerprints(smiles: list[str]) -> np.ndarray:
    values = np.zeros((len(smiles), FP_DIM), dtype=np.float32)
    for index, smiles_value in enumerate(smiles):
        mol = Chem.MolFromSmiles(smiles_value)
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES {smiles_value}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_DIM)
        DataStructs.ConvertToNumpyArray(fp, values[index])
    return values


def load_virtual(model_h5: Path, aggregate_npz: Path, split_lock: Path) -> dict[str, VirtualSplit]:
    if not model_h5.is_file() or not aggregate_npz.is_file() or not split_lock.is_file():
        raise FileNotFoundError("Model H5, aggregate NPZ, and split lock are required")
    lock = json.loads(split_lock.read_text(encoding="utf-8"))
    with h5py.File(model_h5, "r") as handle:
        smiles_all = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in handle["canonical_smiles"][:]]
        control_all = handle["control_CP"][:].astype(np.float32)
        target_all = handle["target_CP"][:].astype(np.float32)
    if control_all.shape != target_all.shape or control_all.shape[1] != CP_DIM or len(set(smiles_all)) != len(smiles_all):
        raise ValueError("Invalid CP fields in model H5")
    h5_index = {value: index for index, value in enumerate(smiles_all)}
    aggregate = np.load(aggregate_npz, allow_pickle=False)
    output: dict[str, VirtualSplit] = {}
    seen: set[str] = set()
    for name in ("train", "valid", "test"):
        smiles = [str(item) for item in lock[f"{name}_smiles"]]
        if len(smiles) != EXPECTED[name] or len(set(smiles)) != len(smiles) or seen & set(smiles):
            raise ValueError(f"Invalid official {name} split")
        seen.update(smiles)
        rows = np.asarray([h5_index[value] for value in smiles], dtype=np.int64)
        control, delta = control_all[rows], target_all[rows] - control_all[rows]
        aggregate_smiles = [str(item) for item in aggregate[f"{name}_smiles"]]
        aggregate_index = {value: index for index, value in enumerate(aggregate_smiles)}
        if set(aggregate_index) != set(smiles):
            raise ValueError(f"CM-0 aggregate SMILES mismatch for {name}")
        aggregate_delta = aggregate[f"{name}_cp"][np.asarray([aggregate_index[value] for value in smiles], dtype=np.int64)].astype(np.float32)
        max_error = float(np.max(np.abs(delta - aggregate_delta)))
        if max_error > 1e-5:
            raise ValueError(f"H5 versus CM-0 CP delta mismatch for {name}: {max_error}")
        output[name] = VirtualSplit(smiles, control, delta, fingerprints(smiles))
    return output


def restrict_pairs(pairs: Any, compounds: set[str]) -> Any:
    indices = np.asarray([index for index, value in enumerate(pairs.smiles) if value in compounds], dtype=np.int64)
    if indices.size == 0:
        raise RuntimeError("Empty teacher fold")
    return BASE.LooPairs(pairs.support[indices], pairs.target[indices], [pairs.smiles[index] for index in indices.tolist()], pairs.held_rows[indices])


def fold_map(smiles: list[str], seed: int, folds: int) -> dict[str, int]:
    output = {value: BASE.stable_seed(seed, f"teacher-fold|{value}") % folds for value in smiles}
    counts = [sum(value == fold for value in output.values()) for fold in range(folds)]
    if min(counts) == 0:
        raise RuntimeError(f"Empty teacher fold assignment: {counts}")
    return output


def effect_array(model: Any, pairs: Any, mean: np.ndarray, scale: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    output = np.empty_like(pairs.support)
    model.eval()
    with torch.no_grad():
        for begin in range(0, pairs.support.shape[0], batch_size):
            value = torch.from_numpy((pairs.support[begin:begin + batch_size] - mean) / scale).to(device)
            prediction = model(value).cpu().numpy()
            output[begin:begin + len(prediction)] = prediction * scale + mean
    return output


def build_teacher_labels(cp_splits: dict[str, Any], virtual_train: VirtualSplit, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict[str, Any]]:
    train_pairs, valid_pairs = BASE.loo_pairs(cp_splits["train"]), BASE.loo_pairs(cp_splits["valid"])
    eligible = set(train_pairs.smiles)
    assignment = fold_map(virtual_train.smiles, args.seed, args.teacher_folds)
    labels: dict[str, list[np.ndarray]] = {value: [] for value in eligible}
    metadata: list[dict[str, Any]] = []
    for fold in range(args.teacher_folds):
        held = {value for value, assigned in assignment.items() if assigned == fold and value in eligible}
        fitted = eligible - held
        if not held or held & fitted:
            raise RuntimeError("Teacher compounds are not disjoint")
        teacher_seed = BASE.stable_seed(args.seed, f"teacher-model-fold{fold}") % (2**32 - 1)
        teacher_args = SimpleNamespace(seed=teacher_seed, hidden_dim=256, latent_dim=32, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, epochs=args.teacher_epochs, smoke=args.smoke)
        model, mean, scale, best_epoch, valid_mse = BASE.fit_encoder(restrict_pairs(train_pairs, fitted), valid_pairs, teacher_args, device)
        held_pairs = restrict_pairs(train_pairs, held)
        prediction = effect_array(model, held_pairs, mean, scale, args.batch_size, device)
        for value, predicted_delta in zip(held_pairs.smiles, prediction):
            labels[value].append(predicted_delta.copy())
        metadata.append({"fold": fold, "fit_compounds": len(fitted), "held_compounds": len(held), "best_epoch": best_epoch, "validation_mse": valid_mse})
    target = virtual_train.delta.copy()
    train_index = {value: index for index, value in enumerate(virtual_train.smiles)}
    for value in eligible:
        if not labels[value]:
            raise RuntimeError(f"Missing OOF teacher label for {value}")
        target[train_index[value]] = np.mean(np.stack(labels[value]), axis=0).astype(np.float32)
    return target, {"fold_count": args.teacher_folds, "eligible_p_ge_3": len(eligible), "raw_fallback_p_lt_3": len(virtual_train.smiles) - len(eligible), "folds": metadata, "definition": "P>=3 target is mean leave-one-plate-out prediction from a teacher trained on other compounds; P<3 target remains ordinary aggregate delta."}


class Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CP_DIM + FP_DIM, 512), nn.GELU(), nn.Linear(512, 128), nn.GELU(), nn.Linear(128, CP_DIM))

    def forward(self, control: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((control, feature), dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def select_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def fit_student(train: VirtualSplit, valid: VirtualSplit, label: np.ndarray, args: argparse.Namespace, device: torch.device, name: str) -> tuple[Student, dict[str, np.ndarray], int, float]:
    # Reset to the identical initialization and minibatch order: supervision is
    # the sole intended difference between the two virtual students.
    set_seed(args.seed)
    control_mean = train.control.mean(axis=0, dtype=np.float64).astype(np.float32)
    control_scale = np.maximum(train.control.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    target_mean = train.delta.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = np.maximum(train.delta.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    train_control, valid_control = (train.control - control_mean) / control_scale, (valid.control - control_mean) / control_scale
    train_label, valid_label = (label - target_mean) / target_scale, (valid.delta - target_mean) / target_scale
    model = Student().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(); generator.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(train_control), torch.from_numpy(train.fingerprint), torch.from_numpy(train_label)), batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    best_state: dict[str, torch.Tensor] | None = None; best_value, best_epoch = float("inf"), -1
    for epoch in range(1 if args.smoke else args.student_epochs):
        model.train()
        for control, feature, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(control.to(device), feature.to(device)) - target.to(device)) ** 2)
            if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite {name} loss")
            loss.backward(); optimizer.step()
        model.eval(); total, count = 0.0, 0
        with torch.no_grad():
            for begin in range(0, valid_control.shape[0], args.batch_size):
                control = torch.from_numpy(valid_control[begin:begin + args.batch_size]).to(device)
                feature = torch.from_numpy(valid.fingerprint[begin:begin + args.batch_size]).to(device)
                target = torch.from_numpy(valid_label[begin:begin + args.batch_size]).to(device)
                loss = torch.mean((model(control, feature) - target) ** 2)
                total += float(loss.cpu()) * len(control); count += len(control)
        validation = total / max(count, 1)
        if validation < best_value:
            best_value, best_epoch = validation, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0: print(f"[{name} seed {args.seed} epoch {epoch}] validation_mse={validation:.7f}", flush=True)
    assert best_state is not None; model.load_state_dict(best_state)
    return model, {"control_mean": control_mean, "control_scale": control_scale, "target_mean": target_mean, "target_scale": target_scale}, best_epoch, best_value


def predict_student(model: Student, split: VirtualSplit, stats: dict[str, np.ndarray], batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    control = (split.control - stats["control_mean"]) / stats["control_scale"]
    output = np.empty_like(split.delta); model.eval()
    with torch.no_grad():
        for begin in range(0, control.shape[0], batch_size):
            value = model(torch.from_numpy(control[begin:begin + batch_size]).to(device), torch.from_numpy(split.fingerprint[begin:begin + batch_size]).to(device)).cpu().numpy()
            output[begin:begin + len(value)] = value * stats["target_scale"] + stats["target_mean"]
    return {value: output[index] for index, value in enumerate(split.smiles)}


def held_predictions(prediction: dict[str, np.ndarray], test_pairs: Any, cp_test: Any) -> dict[int, np.ndarray]:
    return {int(held): prediction[str(cp_test.smiles[int(held)])] for held in test_pairs.held_rows.tolist()}


def mean_pcc(prediction: dict[str, np.ndarray], split: VirtualSplit) -> float:
    values = [BASE.pcc(prediction[value], split.delta[index]) for index, value in enumerate(split.smiles)]
    return float(np.mean(values))


def write_scores(path: Path, records: dict[str, dict[str, dict[str, Any]]]) -> None:
    rows = [{"method": method, "canonical_smiles": value, **record} for method, entries in records.items() for value, record in sorted(entries.items())]
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    if not args.dry_run and args.outdir.exists(): raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    cp = BASE.load_splits(args.cp_rows_root, args.split_lock)
    virtual = load_virtual(args.model_h5, args.aggregate_npz, args.split_lock)
    if args.dry_run:
        pairs = {name: BASE.loo_pairs(rows) for name, rows in cp.items()}
        print(json.dumps({"version": VERSION, "split_counts": {name: len(rows.smiles) for name, rows in virtual.items()}, "cp_loo_pairs_p_ge_3": {name: int(rows.held_rows.size) for name, rows in pairs.items()}, "guards": ["CP only", "official split exact", "H5 and CM-0 delta exact", "compound-OOF teacher"]}, indent=2, sort_keys=True)); return
    args.outdir.mkdir(parents=True)
    device = select_device(args.device)
    teacher_target, teacher_meta = build_teacher_labels(cp, virtual["train"], args, device)
    with h5py.File(args.outdir / "oof_effect_teacher.h5", "w") as handle:
        handle.create_dataset("canonical_smiles", data=np.asarray(virtual["train"].smiles, dtype="S256"))
        handle.create_dataset("cp_oof_effect_delta", data=teacher_target)
        handle.attrs["metadata_json"] = json.dumps(teacher_meta, sort_keys=True)
    raw_model, raw_stats, raw_epoch, raw_valid = fit_student(virtual["train"], virtual["valid"], virtual["train"].delta, args, device, "raw_aggregate")
    effect_model, effect_stats, effect_epoch, effect_valid = fit_student(virtual["train"], virtual["valid"], teacher_target, args, device, "oof_repro_effect")
    raw_prediction, effect_prediction = predict_student(raw_model, virtual["test"], raw_stats, args.batch_size, device), predict_student(effect_model, virtual["test"], effect_stats, args.batch_size, device)
    pairs, reference = BASE.loo_pairs(cp["test"]), BASE.reference_stats(cp["test"], args.null_rounds, args.seed)
    records = {"raw_aggregate": BASE.method_stats(held_predictions(raw_prediction, pairs, cp["test"]), reference, cp["test"]), "oof_repro_effect": BASE.method_stats(held_predictions(effect_prediction, pairs, cp["test"]), reference, cp["test"])}
    contrast = BASE.paired_bootstrap(records["oof_repro_effect"], records["raw_aggregate"], args.bootstrap_rounds, BASE.stable_seed(args.seed, "effect-minus-raw"))
    payload = {"version": VERSION, "seed": args.seed, "smoke": bool(args.smoke), "device": str(device), "inputs": {"cp_rows_root": str(args.cp_rows_root), "aggregate_npz": str(args.aggregate_npz), "model_h5": str(args.model_h5), "split_lock": str(args.split_lock)}, "locked_hyperparameters": {"teacher_folds": 5, "teacher_architecture": "775-256-32-256-775", "teacher_epochs_max": 80, "student_architecture": "(CP775+ECFP4-2048)-512-128-CP775", "student_epochs_max": 40, "batch_size": 256, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "null_rounds": args.null_rounds, "bootstrap_rounds": args.bootstrap_rounds}, "teacher": teacher_meta, "student_checkpoints": {"raw_aggregate": {"best_epoch": raw_epoch, "validation_mse": raw_valid}, "oof_repro_effect": {"best_epoch": effect_epoch, "validation_mse": effect_valid}}, "test_reference_molecule_count": len(reference), "summaries": {name: BASE.summarize(value) for name, value in records.items()}, "aggregate_delta_pcc": {"raw_aggregate": mean_pcc(raw_prediction, virtual["test"]), "oof_repro_effect": mean_pcc(effect_prediction, virtual["test"])}, "paired_bootstrap": {"oof_repro_effect_minus_raw_aggregate": contrast}, "definition": "Virtual CP-only predictor evaluated on independent held CP plates with exact plate-slot matched foreign null; only training target differs.", "guardrails": ["no GE data, feature, target, or loss", "teacher never trains on compound it labels", "teacher/student fit train only", "valid/test target stays raw", "P<3 uses declared raw fallback", "strict held-out RSF is primary", "smoke output is not evidence"]}
    write_scores(args.outdir / "per_molecule_scores.csv", records)
    (args.outdir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def aggregate(root: Path) -> None:
    paths = [root / f"seed{seed}" / "metrics.json" for seed in SEEDS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing: raise FileNotFoundError("Missing seed metrics: " + ", ".join(missing))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    methods = sorted(payloads[0]["summaries"])
    summary = {method: {metric: float(np.mean([payload["summaries"][method][metric] for payload in payloads])) for metric in ("model_excess_z_mean", "replicate_excess_z_mean", "rsf")} | {"molecule_count": int(payloads[0]["summaries"][method]["molecule_count"])} for method in methods}
    output = {"version": VERSION, "seeds": list(SEEDS), "seed_metric_paths": [str(path) for path in paths], "seed_mean_summaries": summary, "note": "Inspect all three seed-specific molecule-paired bootstrap contrasts before a decision."}
    path = root / "aggregate_index.json"
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.aggregate_root: aggregate(args.aggregate_root)
    else: run(args)


if __name__ == "__main__":
    main()
