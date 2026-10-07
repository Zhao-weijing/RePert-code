#!/usr/bin/env python3
"""Train and evaluate a reproducible-effect teacher with GE as target.

The target is a held-out GE plate.  For each compound and budget, a stable
value-blind rule selects one held plate and exactly ``budget`` support plates
from the same dose.  The teacher is fit only on official training compounds;
the validation split selects its checkpoint and the test split is evaluated
once.  A strict foreign null uses the same held/support plate slots and exact
dose, with different foreign compounds in the held and support slots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


VERSION = "MVCPert-GE-Reproducible-Effect-Teacher-2026-08-30"
SEEDS = (3407, 42, 2025)
BUDGETS = (1, 2, 3)
PAIR_MANIFEST_SEED = 3407
FEATURE_DIM = 977
DOSE_TOLERANCE_MILLIMOLAR = 0.1


@dataclass
class PlateRows:
    compound: np.ndarray
    dose: np.ndarray
    plate: np.ndarray
    delta: np.ndarray
    groups: dict[tuple[str, str], np.ndarray]
    by_slot: dict[tuple[str, str], set[str]]
    by_plate: dict[str, list[int]]
    index: dict[tuple[str, str, str], int]


@dataclass
class BudgetPairs:
    support: np.ndarray
    target: np.ndarray
    compound: list[str]
    dose: list[str]
    held_rows: np.ndarray
    support_rows: list[tuple[int, ...]]
    manifest: list[dict[str, Any]]


def decode(values: np.ndarray) -> np.ndarray:
    result = []
    for value in np.asarray(values).reshape(-1):
        result.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return np.asarray(result, dtype=str)


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return (seed + int.from_bytes(digest, "little")) % (2**63 - 1)


def fisher_z(value: float) -> float:
    return float(np.arctanh(np.clip(value, -0.999, 0.999)))


def pcc(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 0 or not np.isfinite(denominator):
        return None
    value = float(np.dot(x, y) / denominator)
    return value if np.isfinite(value) else None


def load_rows(path: Path) -> PlateRows:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        required = {"smiles", "plate", "dose", "delta"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"{path} missing fields {missing}; got {loaded.files}")
        compound = decode(loaded["smiles"])
        plate = decode(loaded["plate"])
        dose = decode(loaded["dose"])
        delta = loaded["delta"].astype(np.float32)
    if delta.ndim != 2 or delta.shape[1] != FEATURE_DIM or delta.shape[0] != len(compound):
        raise ValueError(f"Unexpected GE shape in {path}: {delta.shape}")
    if not np.isfinite(delta).all():
        raise ValueError(f"Non-finite GE delta in {path}")
    if not (len(compound) == len(plate) == len(dose)):
        raise ValueError(f"Metadata length mismatch in {path}")
    groups0: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_slot: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_plate: dict[str, list[int]] = defaultdict(list)
    index: dict[tuple[str, str, str], int] = {}
    for row, (c, d, pl) in enumerate(zip(compound, dose, plate)):
        if not c or not d or not pl:
            raise ValueError(f"Empty compound/dose/plate at row {row} in {path}")
        key = (str(pl), str(d), str(c))
        if key in index:
            raise ValueError(f"Duplicate compound/dose/plate key {key} in {path}")
        index[key] = row
        groups0[(str(c), str(d))].append(row)
        by_slot[(str(pl), str(d))].add(str(c))
        by_plate[str(pl)].append(row)
    return PlateRows(
        compound=compound,
        dose=dose,
        plate=plate,
        delta=delta,
        groups={key: np.asarray(value, dtype=np.int64) for key, value in groups0.items()},
        by_slot=dict(by_slot),
        by_plate=dict(by_plate),
        index=index,
    )


def load_split_rows(data_root: Path, split: str) -> PlateRows:
    return load_rows(data_root / f"{split}_ge_plate_rows.npz")


def load_split_lock(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, set[str]] = {}
    for split in ("train", "valid", "test"):
        values = payload.get(f"{split}_smiles", payload.get(f"{split}_compounds"))
        if values is not None:
            output[split] = {str(x) for x in values}
    if len(output) != 3:
        raise ValueError(f"Unsupported split lock fields: {path}")
    return output


def choose_condition_groups(rows: PlateRows, budget: int, split: str) -> list[tuple[str, str, np.ndarray]]:
    eligible: dict[str, list[tuple[str, str, np.ndarray]]] = defaultdict(list)
    for (compound, dose), indices in rows.groups.items():
        if len(indices) >= budget + 1:
            eligible[compound].append((compound, dose, indices))
    selected = []
    for compound in sorted(eligible):
        choices = sorted(
            eligible[compound],
            key=lambda item: stable_seed(PAIR_MANIFEST_SEED, f"condition|{split}|{budget}|{item[0]}|{item[1]}"),
        )
        selected.append(choices[0])
    return selected


def build_pairs(rows: PlateRows, split: str, budget: int) -> BudgetPairs:
    support: list[np.ndarray] = []
    target: list[np.ndarray] = []
    compounds: list[str] = []
    doses: list[str] = []
    held_rows: list[int] = []
    support_rows: list[tuple[int, ...]] = []
    manifest: list[dict[str, Any]] = []
    for compound, dose, group in choose_condition_groups(rows, budget, split):
        candidates = sorted((int(i) for i in group.tolist()), key=lambda i: (str(rows.plate[i]), i))
        selected = sorted(
            candidates,
            key=lambda i: stable_seed(PAIR_MANIFEST_SEED, f"pair|{split}|{budget}|{compound}|{dose}|{rows.plate[i]}|{i}"),
        )
        held = selected[0]
        source = tuple(sorted(selected[1 : budget + 1]))
        support.append(rows.delta[np.asarray(source, dtype=np.int64)].mean(axis=0, dtype=np.float64).astype(np.float32))
        target.append(rows.delta[held].astype(np.float32))
        compounds.append(compound)
        doses.append(dose)
        held_rows.append(held)
        support_rows.append(source)
        manifest.append(
            {
                "split": split,
                "budget": budget,
                "canonical_compound": compound,
                "dose": dose,
                "n_available_plates": len(candidates),
                "held_row": held,
                "held_plate": str(rows.plate[held]),
                "support_rows": "|".join(map(str, source)),
                "support_plates": "|".join(str(rows.plate[i]) for i in source),
                "pair_manifest_seed": PAIR_MANIFEST_SEED,
                "selection_rule": "stable hash over compound/dose/plate; no GE value, prediction, PCC, or RSF",
            }
        )
    if not support:
        raise RuntimeError(f"No {split} GE compound is eligible for {budget}R")
    return BudgetPairs(
        support=np.vstack(support),
        target=np.vstack(target),
        compound=compounds,
        dose=doses,
        held_rows=np.asarray(held_rows, dtype=np.int64),
        support_rows=support_rows,
        manifest=manifest,
    )


def approximate_slot_candidates(rows: PlateRows, plate: str, target_dose: str) -> dict[str, int]:
    """Return the closest row per compound within the preregistered dose tolerance."""
    target = float(target_dose)
    result: dict[str, tuple[float, int]] = {}
    for row in rows.by_plate.get(plate, []):
        error = abs(float(rows.dose[row]) - target)
        if error > DOSE_TOLERANCE_MILLIMOLAR:
            continue
        compound = str(rows.compound[row])
        previous = result.get(compound)
        if previous is None or (error, row) < (previous[0], previous[1]):
            result[compound] = (error, row)
    return {compound: row for compound, (_error, row) in result.items()}


def strict_reference(rows: PlateRows, pairs: BudgetPairs, null_rounds: int, seed: int) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for index, compound in enumerate(pairs.compound):
        dose = pairs.dose[index]
        held = int(pairs.held_rows[index])
        support_rows = pairs.support_rows[index]
        held_plate = str(rows.plate[held])
        support_plates = tuple(str(rows.plate[row]) for row in support_rows)
        repeat = pcc(pairs.support[index], rows.delta[held])
        held_candidates = approximate_slot_candidates(rows, held_plate, dose)
        support_candidates = [approximate_slot_candidates(rows, plate, dose) for plate in support_plates]
        left = sorted(set(held_candidates) - {compound})
        right = set(support_candidates[0]) - {compound} if support_candidates else set()
        for position in range(1, len(support_plates)):
            right &= set(support_candidates[position])
        right = sorted(right)
        if repeat is None or not left or not right:
            continue
        null_values: list[float] = []
        rng = np.random.default_rng(stable_seed(seed, f"ge-null|{compound}|{dose}|{held_plate}|{'|'.join(support_plates)}"))
        for _ in range(null_rounds):
            for _attempt in range(128):
                held_foreign = left[int(rng.integers(len(left)))]
                support_foreign = right[int(rng.integers(len(right)))]
                if held_foreign != support_foreign:
                    break
            else:
                continue
            held_value = rows.delta[held_candidates[held_foreign]]
            support_value = np.mean(
                [rows.delta[support_candidates[pos][support_foreign]] for pos in range(len(support_plates))],
                axis=0,
                dtype=np.float64,
            )
            value = pcc(held_value, support_value)
            if value is not None:
                null_values.append(fisher_z(value))
        if null_values:
            repeat_z = fisher_z(repeat)
            output[compound] = {
                "dose": dose,
                "held_row": held,
                "support_rows": list(support_rows),
                "held_plate": held_plate,
                "support_plates": list(support_plates),
                "null_z": float(np.mean(null_values)),
                "replicate_z": repeat_z,
                "replicate_excess_z": float(repeat_z - np.mean(null_values)),
                "null_draw_count": len(null_values),
            }
    if not output:
        raise RuntimeError("No GE pair has a dose-tolerant plate-slot matched foreign reference")
    return output


class CrossFitEffectNet(nn.Module):
    def __init__(self, dimension: int = FEATURE_DIM, hidden: int = 256, latent: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(),
            nn.Linear(hidden, latent), nn.GELU(),
            nn.Linear(latent, hidden), nn.GELU(),
            nn.Linear(hidden, dimension),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def fit_teacher(train: BudgetPairs, valid: BudgetPairs, seed: int, epochs: int, batch_size: int, lr: float, weight_decay: float, device: torch.device, smoke: bool) -> tuple[CrossFitEffectNet, np.ndarray, np.ndarray, int, float]:
    set_seed(seed)
    mean = train.support.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(train.support.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    train_x = (train.support - mean) / scale
    train_y = (train.target - mean) / scale
    valid_x = (valid.support - mean) / scale
    valid_y = (valid.target - mean) / scale
    model = CrossFitEffectNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = -1
    limit = 1 if smoke else epochs
    for epoch in range(limit):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x.to(device)) - y.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite GE teacher loss")
            loss.backward()
            optimizer.step()
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(valid_x), batch_size):
                x = torch.from_numpy(valid_x[start : start + batch_size]).to(device)
                y = torch.from_numpy(valid_y[start : start + batch_size]).to(device)
                value = torch.mean((model(x) - y) ** 2)
                total += float(value.detach().cpu()) * len(x)
                count += len(x)
        validation_loss = total / max(count, 1)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 0 or epoch % 10 == 0:
            print(f"[seed {seed}] epoch={epoch} validation_mse={validation_loss:.7f}", flush=True)
    if best_state is None:
        raise RuntimeError("Teacher did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, mean, scale, best_epoch, best_loss


def predict(model: CrossFitEffectNet, support: np.ndarray, mean: np.ndarray, scale: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    output = np.empty_like(support, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(support), batch_size):
            x = (support[start : start + batch_size] - mean) / scale
            value = model(torch.from_numpy(x).to(device)).detach().cpu().numpy()
            output[start : start + len(value)] = value * scale + mean
    return output


def method_records(prediction: np.ndarray, pairs: BudgetPairs, reference: dict[str, dict[str, Any]], method: str) -> dict[str, dict[str, Any]]:
    by_compound = {compound: index for index, compound in enumerate(pairs.compound)}
    output: dict[str, dict[str, Any]] = {}
    for compound, ref in reference.items():
        index = by_compound[compound]
        score = pcc(prediction[index], pairs.target[index])
        if score is None:
            continue
        output[compound] = {
            "method": method,
            "canonical_compound": compound,
            "dose": pairs.dose[index],
            "held_row": int(pairs.held_rows[index]),
            "raw_pcc": score,
            "model_z": fisher_z(score),
            "null_z": ref["null_z"],
            "model_excess_z": fisher_z(score) - ref["null_z"],
            "replicate_z": ref["replicate_z"],
            "replicate_excess_z": ref["replicate_excess_z"],
        }
    return output


def summarize(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise RuntimeError("Empty method records")
    raw = np.asarray([value["raw_pcc"] for value in records.values()], dtype=np.float64)
    model = np.asarray([value["model_excess_z"] for value in records.values()], dtype=np.float64)
    repeat = np.asarray([value["replicate_excess_z"] for value in records.values()], dtype=np.float64)
    return {
        "molecule_count": int(len(records)),
        "raw_pcc_mean": float(raw.mean()),
        "held_delta_pcc_mean": float(raw.mean()),
        "model_excess_z_mean": float(model.mean()),
        "replicate_excess_z_mean": float(repeat.mean()),
        "rsf": float(model.mean() / repeat.mean()) if repeat.mean() != 0 else None,
    }


def bootstrap_difference(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], rounds: int, seed: int) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    if not common:
        raise RuntimeError("No common molecules for paired bootstrap")
    left_excess = np.asarray([left[x]["model_excess_z"] for x in common], dtype=np.float64)
    right_excess = np.asarray([right[x]["model_excess_z"] for x in common], dtype=np.float64)
    left_raw = np.asarray([left[x]["raw_pcc"] for x in common], dtype=np.float64)
    right_raw = np.asarray([right[x]["raw_pcc"] for x in common], dtype=np.float64)
    diff_excess = left_excess - right_excess
    diff_raw = left_raw - right_raw
    rng = np.random.default_rng(seed)
    boot_excess = np.empty(rounds, dtype=np.float64)
    boot_raw = np.empty(rounds, dtype=np.float64)
    for begin in range(0, rounds, 100):
        end = min(rounds, begin + 100)
        draw = rng.integers(0, len(common), size=(end - begin, len(common)))
        boot_excess[begin:end] = diff_excess[draw].mean(axis=1)
        boot_raw[begin:end] = diff_raw[draw].mean(axis=1)
    return {
        "molecule_count": len(common),
        "excess_z_point": float(diff_excess.mean()),
        "excess_z_ci95": [float(np.quantile(boot_excess, 0.025)), float(np.quantile(boot_excess, 0.975))],
        "raw_pcc_point": float(diff_raw.mean()),
        "raw_pcc_ci95": [float(np.quantile(boot_raw, 0.025)), float(np.quantile(boot_raw, 0.975))],
        "bootstrap_rounds": rounds,
        "bootstrap_seed": seed,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"))
    parser.add_argument("--split-lock", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json"))
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--aggregate-root", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--null-rounds", type=int, default=32)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="one epoch and 100 bootstrap draws; never evidence")
    return parser.parse_args()


def dry_run(args: argparse.Namespace) -> None:
    lock = load_split_lock(args.split_lock)
    counts: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        rows = load_split_rows(args.data_root, split)
        counts[split] = {}
        for budget in BUDGETS:
            pairs = build_pairs(rows, split, budget)
            if len(pairs.compound) < 500:
                counts[split][str(budget)] = {"pairs": len(pairs.compound), "strict_reference": None, "status": "BLOCKED_under_500"}
                continue
            eligible = strict_reference(rows, pairs, min(args.null_rounds, 2), 3407)
            counts[split][str(budget)] = {"pairs": len(pairs.compound), "strict_reference": len(eligible), "status": "runnable"}
    print(json.dumps({"version": VERSION, "feature_dim": FEATURE_DIM, "split_lock_counts": {k: len(v) for k, v in lock.items()}, "pair_counts": counts, "guard": "dry-run only; no output or training"}, indent=2, sort_keys=True))


def run_one(args: argparse.Namespace) -> None:
    if args.seed is None or args.outdir is None:
        raise ValueError("--seed and --outdir are required for a seed run")
    if args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    args.outdir.mkdir(parents=True)
    if args.smoke:
        args.bootstrap_rounds = min(args.bootstrap_rounds, 100)
    split_lock = load_split_lock(args.split_lock)
    loaded = {split: load_split_rows(args.data_root, split) for split in ("train", "valid", "test")}
    for split, rows in loaded.items():
        observed = set(rows.compound)
        if observed != split_lock[split]:
            raise ValueError(f"{split} compounds do not match official split lock: {len(observed)} vs {len(split_lock[split])}")
    pairs = {budget: {split: build_pairs(loaded[split], split, budget) for split in ("train", "valid", "test")} for budget in BUDGETS}
    write_csv(args.outdir / "pair_manifest.csv", [row for budget in BUDGETS for split in ("train", "valid", "test") for row in pairs[budget][split].manifest])
    device = choose_device(args.device)
    output: dict[str, Any] = {
        "version": VERSION,
        "seed": args.seed,
        "device": str(device),
        "feature_dim": FEATURE_DIM,
        "budgets": {},
        "guardrails": [
            "official molecule-disjoint split lock is checked",
            "one value-blind held plate and exact-dose support plates per compound/budget",
            "teacher fits training compounds only; validation selects checkpoint",
            "test held-out GE plate is never used for fitting or selection",
            f"strict null uses identical held/support plate slots and foreign-dose matching within {DOSE_TOLERANCE_MILLIMOLAR} mM",
            "smoke is execution validation only and cannot support a method claim",
        ],
    }
    for budget in BUDGETS:
        train, valid, test = pairs[budget]["train"], pairs[budget]["valid"], pairs[budget]["test"]
        if min(len(train.compound), len(valid.compound), len(test.compound)) < 500:
            output["budgets"][str(budget)] = {
                "status": "BLOCKED_under_500",
                "pair_counts": {split: len(pairs[budget][split].compound) for split in ("train", "valid", "test")},
                "decision": "BLOCKED: fewer than 500 eligible compounds in at least one split",
            }
            continue
        reference_valid = strict_reference(loaded["valid"], valid, args.null_rounds, args.seed)
        reference_test = strict_reference(loaded["test"], test, args.null_rounds, args.seed)
        teacher, mean, scale, best_epoch, best_validation_loss = fit_teacher(
            train, valid, args.seed, args.epochs, args.batch_size, args.learning_rate, args.weight_decay, device, args.smoke
        )
        teacher_valid = predict(teacher, valid.support, mean, scale, args.batch_size, device)
        teacher_test = predict(teacher, test.support, mean, scale, args.batch_size, device)
        mean_valid = valid.support
        mean_test = test.support
        valid_teacher_records = method_records(teacher_valid, valid, reference_valid, "GE_teacher")
        valid_mean_records = method_records(mean_valid, valid, reference_valid, "GE_mean")
        test_teacher_records = method_records(teacher_test, test, reference_test, "GE_teacher")
        test_mean_records = method_records(mean_test, test, reference_test, "GE_mean")
        contrast = bootstrap_difference(test_teacher_records, test_mean_records, args.bootstrap_rounds, stable_seed(args.seed, f"GE_teacher_minus_mean|{budget}"))
        write_csv(args.outdir / f"budget{budget}_per_molecule_scores.csv", list(test_teacher_records.values()) + list(test_mean_records.values()))
        np.savez_compressed(
            args.outdir / f"budget{budget}_valid_predictions.npz",
            compound=np.asarray(valid.compound, dtype="U"),
            dose=np.asarray(valid.dose, dtype="U"),
            held_rows=valid.held_rows,
            support_mean=valid.support,
            teacher=teacher_valid,
            target=valid.target,
        )
        np.savez_compressed(
            args.outdir / f"budget{budget}_predictions.npz",
            compound=np.asarray(test.compound, dtype="U"),
            dose=np.asarray(test.dose, dtype="U"),
            held_rows=test.held_rows,
            support_mean=test.support,
            teacher=teacher_test,
            target=test.target,
        )
        output["budgets"][str(budget)] = {
            "status": "completed",
            "pair_counts": {split: len(pairs[budget][split].compound) for split in ("train", "valid", "test")},
            "strict_reference_counts": {"valid": len(reference_valid), "test": len(reference_test)},
            "teacher_checkpoint": {"best_epoch": best_epoch, "best_validation_mse": best_validation_loss},
            "summaries": {"GE_teacher": summarize(test_teacher_records), "GE_mean": summarize(test_mean_records)},
            "validation_summaries": {"GE_teacher": summarize(valid_teacher_records), "GE_mean": summarize(valid_mean_records)},
            "paired_bootstrap": {"GE_teacher_minus_GE_mean": contrast},
        }
    (args.outdir / "metrics.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"version": VERSION, "seed": args.seed, "outdir": str(args.outdir), "budgets": output["budgets"]}, indent=2, sort_keys=True))


def read_scores(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output[row["method"]][row["canonical_compound"]] = {
                "model_excess_z": float(row["model_excess_z"]),
                "raw_pcc": float(row["raw_pcc"]),
            }
    return output


def aggregate(args: argparse.Namespace) -> None:
    root = args.aggregate_root
    if root is None:
        raise ValueError("--aggregate-root is required")
    payloads = []
    for seed in SEEDS:
        path = root / f"seed_{seed}" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for payload in payloads:
        seed = int(payload["seed"])
        for budget in BUDGETS:
            result = payload["budgets"][str(budget)]
            if result.get("status") != "completed":
                continue
            for method, summary in result["summaries"].items():
                metric_rows.append({"seed": seed, "budget": budget, "method": method, **summary})
            contrast = result["paired_bootstrap"]["GE_teacher_minus_GE_mean"]
            bootstrap_rows.append({"scope": "seed", "seed": seed, "budget": budget, "contrast": "GE_teacher_minus_GE_mean", "molecule_count": contrast["molecule_count"], "excess_z_point": contrast["excess_z_point"], "excess_z_ci_low": contrast["excess_z_ci95"][0], "excess_z_ci_high": contrast["excess_z_ci95"][1], "raw_pcc_point": contrast["raw_pcc_point"], "raw_pcc_ci_low": contrast["raw_pcc_ci95"][0], "raw_pcc_ci_high": contrast["raw_pcc_ci95"][1]})
    for budget in BUDGETS:
        if any(payload["budgets"][str(budget)].get("status") != "completed" for payload in payloads):
            continue
        by_seed = [read_scores(root / f"seed_{seed}" / f"budget{budget}_per_molecule_scores.csv") for seed in SEEDS]
        common = set(by_seed[0]["GE_teacher"]) & set(by_seed[0]["GE_mean"])
        for scores in by_seed[1:]:
            common &= set(scores["GE_teacher"]) & set(scores["GE_mean"])
        common = sorted(common)
        diff_excess = np.asarray([np.mean([scores["GE_teacher"][compound]["model_excess_z"] - scores["GE_mean"][compound]["model_excess_z"] for scores in by_seed]) for compound in common], dtype=np.float64)
        diff_raw = np.asarray([np.mean([scores["GE_teacher"][compound]["raw_pcc"] - scores["GE_mean"][compound]["raw_pcc"] for scores in by_seed]) for compound in common], dtype=np.float64)
        rng = np.random.default_rng(stable_seed(20260830, f"pooled|GE_teacher_minus_mean|{budget}"))
        boot_excess = np.empty(args.bootstrap_rounds)
        boot_raw = np.empty(args.bootstrap_rounds)
        for begin in range(0, args.bootstrap_rounds, 100):
            end = min(args.bootstrap_rounds, begin + 100)
            draw = rng.integers(0, len(common), size=(end - begin, len(common)))
            boot_excess[begin:end] = diff_excess[draw].mean(axis=1)
            boot_raw[begin:end] = diff_raw[draw].mean(axis=1)
        bootstrap_rows.append({"scope": "pooled_three_seed", "seed": "all", "budget": budget, "contrast": "GE_teacher_minus_GE_mean", "molecule_count": len(common), "excess_z_point": float(diff_excess.mean()), "excess_z_ci_low": float(np.quantile(boot_excess, .025)), "excess_z_ci_high": float(np.quantile(boot_excess, .975)), "raw_pcc_point": float(diff_raw.mean()), "raw_pcc_ci_low": float(np.quantile(boot_raw, .025)), "raw_pcc_ci_high": float(np.quantile(boot_raw, .975))})
    write_csv(root / "paired_metrics.csv", metric_rows)
    write_csv(root / "bootstrap_summary.csv", bootstrap_rows)
    decisions = {}
    for budget in BUDGETS:
        seed_rows = [row for row in bootstrap_rows if row["scope"] == "seed" and int(row["budget"]) == budget]
        pooled = next((row for row in bootstrap_rows if row["scope"] == "pooled_three_seed" and int(row["budget"]) == budget), None)
        if pooled is None:
            # A budget can be intentionally blocked by the preregistered
            # minimum-count rule (currently GE 3R).  Keep its status in each
            # seed metrics file, but do not manufacture a pooled decision.
            continue
        decisions[str(budget)] = {"seed_points": {str(row["seed"]): row["excess_z_point"] for row in seed_rows}, "positive_seed_count": sum(float(row["excess_z_point"]) > 0 for row in seed_rows), "pooled_excess_z": pooled["excess_z_point"], "pooled_ci95": [pooled["excess_z_ci_low"], pooled["excess_z_ci_high"]], "go": all(float(row["excess_z_point"]) > 0 for row in seed_rows) and pooled["excess_z_ci_low"] > 0}
    aggregate_payload = {"version": VERSION, "seeds": list(SEEDS), "decisions": decisions, "files": {"paired_metrics": str(root / "paired_metrics.csv"), "bootstrap_summary": str(root / "bootstrap_summary.csv")}, "guard": "pooled bootstrap averages per-molecule seed differences before resampling; test is never used to select a checkpoint"}
    (root / "aggregate_index.json").write_text(json.dumps(aggregate_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(aggregate_payload, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.aggregate_root:
        aggregate(args)
    elif args.dry_run:
        dry_run(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
