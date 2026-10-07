#!/usr/bin/env python3
"""Experiment A: cross-fitted CP reproducible-effect learning MVP.

For a molecule with P>=3 independent CP plate deltas, each leave-one-plate-out
support mean is mapped to its held-out plate.  Training/validation/test are
the locked official BBBC047 compound split.  The test score uses the same
plate-slot matched-foreign Fisher-z null as the earlier strict RSF evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn


VERSION = "BBBC047-Reproducible-Effect-CP-MVP-2026-08-29"
SEEDS = (3407, 42, 2025)
EXPECTED_COMPOUNDS = {"train": 12175, "valid": 4059, "test": 4060}
FEATURE_DIM = 775


@dataclass
class PlateRows:
    smiles: np.ndarray
    plate: np.ndarray
    delta: np.ndarray
    groups: dict[str, np.ndarray]
    by_plate: dict[str, set[str]]
    index: dict[tuple[str, str], int]


@dataclass
class LooPairs:
    support: np.ndarray
    target: np.ndarray
    smiles: list[str]
    held_rows: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"))
    parser.add_argument("--split-lock", type=Path,
                        default=Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json"))
    parser.add_argument("--phase2-runs-root", type=Path,
                        default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/phase2_cp_direction_20260828"))
    parser.add_argument("--phase2-prefix", default="phase2_cp_direction_lambda0p1")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--aggregate-root", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--null-rounds", type=int, default=32)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="One-epoch execution check; never an evidence run.")
    args = parser.parse_args()
    if args.aggregate_root:
        if args.seed is not None or args.outdir is not None or args.dry_run or args.smoke:
            parser.error("--aggregate-root cannot be combined with run options")
        return args
    if args.seed is None and not args.dry_run:
        parser.error("--seed is required unless --dry-run or --aggregate-root is used")
    if not args.dry_run and args.outdir is None:
        parser.error("--outdir is required for a run")
    if args.pca_components != 32 or args.latent_dim != 32:
        parser.error("Experiment A locks PCA and MLP bottlenecks to 32")
    if args.epochs != 80 or args.hidden_dim != 256 or args.batch_size != 256:
        parser.error("Experiment A locks epochs=80, hidden_dim=256, batch_size=256")
    if args.null_rounds < 1 or args.bootstrap_rounds < 1:
        parser.error("null and bootstrap rounds must be positive")
    return args


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return (seed + int.from_bytes(digest, "little")) % (2**63 - 1)


def fisher_z(value: float) -> float:
    return float(np.arctanh(np.clip(value, -0.999, 0.999)))


def pcc(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not np.isfinite(denominator) or denominator <= 0:
        return None
    value = float(np.dot(left, right) / denominator)
    return value if np.isfinite(value) else None


def read_rows(path: Path, split_name: str) -> PlateRows:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        required = {"smiles", "plate", "dose", "delta"}
        if set(loaded.files) != required:
            raise ValueError(f"{path} fields must be {sorted(required)}, got {loaded.files}")
        smiles = loaded["smiles"].astype(str)
        plate = loaded["plate"].astype(str)
        delta = loaded["delta"].astype(np.float32)
    if delta.ndim != 2 or delta.shape[1] != FEATURE_DIM or delta.shape[0] != smiles.size:
        raise ValueError(f"Unexpected CP shape in {path}: smiles={smiles.shape}, delta={delta.shape}")
    if not np.isfinite(delta).all():
        raise ValueError(f"Non-finite CP delta in {path}")
    groups0: dict[str, list[int]] = defaultdict(list)
    by_plate: dict[str, set[str]] = defaultdict(set)
    index: dict[tuple[str, str], int] = {}
    for row, (smiles_value, plate_value) in enumerate(zip(smiles, plate)):
        key = (str(plate_value), str(smiles_value))
        if key in index:
            raise ValueError(f"Duplicate compound/plate pair in {path}: {key}")
        index[key] = row
        groups0[str(smiles_value)].append(row)
        by_plate[str(plate_value)].add(str(smiles_value))
    if len(groups0) != EXPECTED_COMPOUNDS[split_name]:
        raise ValueError(f"{split_name} compound count {len(groups0)} != {EXPECTED_COMPOUNDS[split_name]}")
    return PlateRows(smiles=smiles, plate=plate, delta=delta,
                     groups={key: np.asarray(value, dtype=np.int64) for key, value in groups0.items()},
                     by_plate=dict(by_plate), index=index)


def load_splits(data_root: Path, split_lock: Path) -> dict[str, PlateRows]:
    if not split_lock.is_file():
        raise FileNotFoundError(split_lock)
    locked = json.loads(split_lock.read_text(encoding="utf-8"))
    rows = {name: read_rows(data_root / f"{name}_cp_plate_rows.npz", name)
            for name in ("train", "valid", "test")}
    sets = {name: set(value.groups) for name, value in rows.items()}
    if sets["train"] & sets["valid"] or sets["train"] & sets["test"] or sets["valid"] & sets["test"]:
        raise ValueError("SMILES overlap between official CP plate-row splits")
    lock_sets = {name: {str(value) for value in locked[f"{name}_smiles"]} for name in ("train", "valid", "test")}
    for name in rows:
        if sets[name] != lock_sets[name]:
            raise ValueError(f"{name} plate-row smiles do not exactly match the official split lock")
    return rows


def loo_pairs(rows: PlateRows) -> LooPairs:
    support: list[np.ndarray] = []
    target: list[np.ndarray] = []
    smiles: list[str] = []
    held_rows: list[int] = []
    for compound in sorted(rows.groups):
        group = rows.groups[compound]
        if group.size < 3:
            continue
        values = rows.delta[group]
        total = values.sum(axis=0, dtype=np.float64)
        for local, held in enumerate(group.tolist()):
            support.append(((total - values[local]) / (group.size - 1)).astype(np.float32))
            target.append(values[local])
            smiles.append(compound)
            held_rows.append(held)
    if not support:
        raise RuntimeError("No P>=3 CP leave-one-out pairs")
    return LooPairs(np.vstack(support), np.vstack(target), smiles, np.asarray(held_rows, dtype=np.int64))


def reference_stats(rows: PlateRows, null_rounds: int, seed: int) -> dict[str, dict[str, Any]]:
    """Strict test reference with identical held/support slots for foreign nulls."""
    result: dict[str, dict[str, Any]] = {}
    for compound in sorted(rows.groups):
        group = rows.groups[compound]
        if group.size < 3:
            continue
        replicate_z: list[float] = []
        null_z: list[float] = []
        usable: list[int] = []
        for held in group.tolist():
            refs = [row for row in group.tolist() if row != held]
            replicate = pcc(rows.delta[held], rows.delta[refs].mean(axis=0))
            if replicate is None:
                continue
            held_plate = str(rows.plate[held])
            ref_plates = [str(rows.plate[row]) for row in refs]
            left_choices = sorted(rows.by_plate[held_plate] - {compound})
            right: set[str] | None = None
            for plate in ref_plates:
                right = set(rows.by_plate[plate]) if right is None else right & rows.by_plate[plate]
            assert right is not None
            right.discard(compound)
            right_choices = sorted(right)
            if not left_choices or not right_choices:
                continue
            rng = np.random.default_rng(stable_seed(seed, f"null|{compound}|{held_plate}|{'|'.join(ref_plates)}"))
            current: list[float] = []
            for _ in range(null_rounds):
                for _attempt in range(128):
                    left_compound = left_choices[int(rng.integers(len(left_choices)))]
                    right_compound = right_choices[int(rng.integers(len(right_choices)))]
                    if left_compound != right_compound:
                        break
                else:
                    continue
                foreign_held = rows.index[(held_plate, left_compound)]
                foreign_refs = [rows.index[(plate, right_compound)] for plate in ref_plates]
                score = pcc(rows.delta[foreign_held], rows.delta[foreign_refs].mean(axis=0))
                if score is not None:
                    current.append(fisher_z(score))
            if current:
                replicate_z.append(fisher_z(replicate))
                null_z.extend(current)
                usable.append(held)
        if usable:
            result[compound] = {
                "n_plates": int(group.size), "used_holdout_plates": len(usable),
                "held_rows": usable, "null_z": float(np.mean(null_z)),
                "replicate_z": float(np.mean(replicate_z)),
                "replicate_excess_z": float(np.mean(replicate_z) - np.mean(null_z)),
                "null_draw_count": len(null_z),
            }
    if not result:
        raise RuntimeError("No test molecule had a strict P>=3 matched-null reference")
    return result


def method_stats(predictions: dict[int, np.ndarray], reference: dict[str, dict[str, Any]], rows: PlateRows) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for compound, ref in reference.items():
        scores: list[float] = []
        for held in ref["held_rows"]:
            if held not in predictions:
                raise KeyError(f"Missing prediction for held row {held}")
            score = pcc(predictions[held], rows.delta[held])
            if score is not None:
                scores.append(fisher_z(score))
        if scores:
            record = {key: value for key, value in ref.items() if key != "held_rows"}
            record["model_z"] = float(np.mean(scores))
            record["model_excess_z"] = float(record["model_z"] - record["null_z"])
            output[compound] = record
    return output


def summarize(values: dict[str, dict[str, Any]]) -> dict[str, float | int]:
    model = np.asarray([record["model_excess_z"] for record in values.values()], dtype=np.float64)
    reference = np.asarray([record["replicate_excess_z"] for record in values.values()], dtype=np.float64)
    if not model.size or reference.mean() == 0:
        raise RuntimeError("Cannot summarize RSF")
    return {"molecule_count": int(model.size), "model_excess_z_mean": float(model.mean()),
            "replicate_excess_z_mean": float(reference.mean()), "rsf": float(model.mean() / reference.mean())}


def paired_bootstrap(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], rounds: int, seed: int) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    if not common:
        raise RuntimeError("No shared molecules for paired bootstrap")
    left_values = np.asarray([left[key]["model_excess_z"] for key in common], dtype=np.float64)
    right_values = np.asarray([right[key]["model_excess_z"] for key in common], dtype=np.float64)
    ceiling = np.asarray([left[key]["replicate_excess_z"] for key in common], dtype=np.float64)
    diff = left_values - right_values
    rng = np.random.default_rng(seed)
    boot_excess = np.empty(rounds, dtype=np.float64)
    boot_rsf = np.empty(rounds, dtype=np.float64)
    for begin in range(0, rounds, 100):
        end = min(rounds, begin + 100)
        draws = rng.integers(0, len(common), size=(end - begin, len(common)))
        boot_excess[begin:end] = diff[draws].mean(axis=1)
        boot_rsf[begin:end] = left_values[draws].mean(axis=1) / ceiling[draws].mean(axis=1) - right_values[draws].mean(axis=1) / ceiling[draws].mean(axis=1)
    return {"molecule_count": len(common), "candidate_minus_reference_excess_z": float(diff.mean()),
            "candidate_minus_reference_excess_z_ci95": [float(np.quantile(boot_excess, .025)), float(np.quantile(boot_excess, .975))],
            "candidate_minus_reference_rsf": float(left_values.mean() / ceiling.mean() - right_values.mean() / ceiling.mean()),
            "candidate_minus_reference_rsf_ci95": [float(np.quantile(boot_rsf, .025)), float(np.quantile(boot_rsf, .975))]}


class CrossFitEffectNet(nn.Module):
    def __init__(self, dimension: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, latent),
                                     nn.GELU(), nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested if requested != "auto" else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_encoder(train: LooPairs, valid: LooPairs, args: argparse.Namespace, device: torch.device) -> tuple[CrossFitEffectNet, np.ndarray, np.ndarray, int, float]:
    set_seed(args.seed)
    mean = train.support.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.support.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    # The arrays are now normalized in place; targets and all statistics are training-derived.
    train_x = (train.support - mean) / scale
    train_y = (train.target - mean) / scale
    valid_x = (valid.support - mean) / scale
    valid_y = (valid.target - mean) / scale
    model = CrossFitEffectNet(FEATURE_DIM, args.hidden_dim, args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(); generator.manual_seed(args.seed)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss, best_epoch = float("inf"), -1
    epoch_limit = 1 if args.smoke else args.epochs
    for epoch in range(epoch_limit):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x.to(device)) - y.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite cross-fitted MLP loss")
            loss.backward(); optimizer.step()
        model.eval(); total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, valid_x.shape[0], args.batch_size):
                x = torch.from_numpy(valid_x[start:start + args.batch_size]).to(device)
                y = torch.from_numpy(valid_y[start:start + args.batch_size]).to(device)
                value = torch.mean((model(x) - y) ** 2)
                total += float(value.cpu()) * len(x); count += len(x)
        validation_loss = total / max(count, 1)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"[seed {args.seed} epoch {epoch}] validation_mse={validation_loss:.7f}", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, mean, scale, best_epoch, best_loss


def encode_predictions(model: CrossFitEffectNet, pairs: LooPairs, mean: np.ndarray, scale: np.ndarray, batch_size: int, device: torch.device) -> dict[int, np.ndarray]:
    output = np.empty_like(pairs.support)
    model.eval()
    with torch.no_grad():
        for start in range(0, pairs.support.shape[0], batch_size):
            standardized = (pairs.support[start:start + batch_size] - mean) / scale
            prediction = model(torch.from_numpy(standardized).to(device)).cpu().numpy()
            output[start:start + len(prediction)] = prediction * scale + mean
    return {int(held): output[index] for index, held in enumerate(pairs.held_rows.tolist())}


def pca_predictions(train: LooPairs, test: LooPairs, mean: np.ndarray, scale: np.ndarray, args: argparse.Namespace) -> dict[int, np.ndarray]:
    pca = PCA(n_components=args.pca_components, svd_solver="randomized", random_state=args.seed)
    pca.fit((train.support - mean) / scale)
    standardized = (test.support - mean) / scale
    prediction = pca.inverse_transform(pca.transform(standardized)).astype(np.float32) * scale + mean
    return {int(held): prediction[index] for index, held in enumerate(test.held_rows.tolist())}


def phase2_predictions(root: Path, prefix: str, seed: int, expected: set[str]) -> dict[str, np.ndarray]:
    name = f"MVC_{prefix}_seed{seed}_BBBC047_smiles_split"
    matches = list(root.glob(f"*/{name}/ECFP4_Default/predict/test_prediction_profile.h5"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one frozen Phase-2 profile for seed {seed}, found {matches}")
    with h5py.File(matches[0], "r") as handle:
        smiles = handle["smiles"][:]
        smiles = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in smiles]
        prediction = handle["cp_pred"][:].astype(np.float32) - handle["control_cp"][:].astype(np.float32)
    if len(smiles) != len(set(smiles)) or set(smiles) != expected:
        raise ValueError("Frozen Phase-2 profile does not exactly match the official test split")
    return dict(zip(smiles, prediction))


def rows_for_csv(method: str, values: dict[str, dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for smiles, record in sorted(values.items()):
        yield {"method": method, "canonical_smiles": smiles, **record}


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(materialized)


def run(args: argparse.Namespace) -> None:
    if not args.dry_run and args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.outdir}")
    splits = load_splits(args.data_root, args.split_lock)
    pairs = {name: loo_pairs(rows) for name, rows in splits.items()}
    if args.dry_run:
        payload = {"version": VERSION, "split_counts": {name: len(rows.groups) for name, rows in splits.items()},
                   "p_ge_3_loo_pairs": {name: int(value.held_rows.size) for name, value in pairs.items()},
                   "plate_count_histograms": {name: dict(sorted(Counter(len(group) for group in rows.groups.values()).items())) for name, rows in splits.items()},
                   "guards": ["plate identity retained", "official compound split exact", "test target absent from support"]}
        print(json.dumps(payload, indent=2, sort_keys=True)); return
    reference = reference_stats(splits["test"], args.null_rounds, args.seed)
    device = choose_device(args.device)
    model, mean, scale, best_epoch, best_validation_mse = fit_encoder(pairs["train"], pairs["valid"], args, device)
    predictions: dict[str, dict[int, np.ndarray]] = {
        "support_mean": {int(held): pairs["test"].support[index] for index, held in enumerate(pairs["test"].held_rows.tolist())},
        "pca32": pca_predictions(pairs["train"], pairs["test"], mean, scale, args),
        "crossfit_mlp32": encode_predictions(model, pairs["test"], mean, scale, args.batch_size, device),
    }
    phase2 = phase2_predictions(args.phase2_runs_root, args.phase2_prefix, args.seed, set(splits["test"].groups))
    predictions["phase2_virtual"] = {int(held): phase2[str(splits["test"].smiles[int(held)])] for held in pairs["test"].held_rows.tolist()}
    per_method = {name: method_stats(prediction, reference, splits["test"]) for name, prediction in predictions.items()}
    summaries = {name: summarize(values) for name, values in per_method.items()}
    comparisons = {}
    for left, right in (("pca32", "support_mean"), ("crossfit_mlp32", "support_mean"),
                        ("crossfit_mlp32", "pca32"), ("phase2_virtual", "support_mean")):
        comparisons[f"{left}_minus_{right}"] = paired_bootstrap(per_method[left], per_method[right], args.bootstrap_rounds,
                                                                    stable_seed(args.seed, f"{left}|{right}"))
    args.outdir.mkdir(parents=True)
    write_csv(args.outdir / "per_molecule_scores.csv", (row for name, values in per_method.items() for row in rows_for_csv(name, values)))
    payload = {"version": VERSION, "seed": args.seed, "device": str(device),
               "inputs": {"data_root": str(args.data_root), "split_lock": str(args.split_lock), "phase2_runs_root": str(args.phase2_runs_root)},
               "smoke": bool(args.smoke),
               "locked_hyperparameters": {"pca_components": 32, "mlp_hidden_dim": 256, "mlp_latent_dim": 32,
                                           "epochs_max": 80, "batch_size": 256, "learning_rate": args.learning_rate,
                                           "weight_decay": args.weight_decay, "null_rounds": args.null_rounds, "bootstrap_rounds": args.bootstrap_rounds},
               "encoder_checkpoint": {"best_epoch": best_epoch, "validation_mse": best_validation_mse},
               "test_reference_molecule_count": len(reference), "summaries": summaries, "paired_bootstrap": comparisons,
               "definition": "RSF = mean[z(PCC(prediction for held plate, held plate))-z(matched foreign null)] / mean[z(PCC(mean other plates, held plate))-z(matched foreign null)]",
               "guardrails": ["only P>=3 compounds participate", "all support plates exclude the corresponding held plate", "PCA and MLP fit official training compounds only", "validation is used only for MLP checkpoint selection", "test targets are evaluation-only", "matched null uses two distinct foreign test compounds on the exact held/support plate slots", "frozen Phase-2 profile is read-only and receives no post-treatment input", "smoke output is execution validation only and cannot support a method claim"]}
    (args.outdir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def aggregate(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = [root / f"seed{seed}" / "metrics.json" for seed in SEEDS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing fixed-seed metrics: " + ", ".join(missing))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    methods = sorted(payloads[0]["summaries"])
    summary = {method: {metric: float(np.mean([payload["summaries"][method][metric] for payload in payloads]))
                        for metric in ("model_excess_z_mean", "replicate_excess_z_mean", "rsf")}
               | {"molecule_count": int(payloads[0]["summaries"][method]["molecule_count"])} for method in methods}
    index = {"version": VERSION, "seeds": list(SEEDS), "seed_metric_paths": [str(path) for path in paths],
             "seed_mean_summaries": summary, "note": "Inspect each fixed-seed paired bootstrap result before a go/no-go claim; this index is not a pooled bootstrap."}
    path = root / "aggregate_index.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(index, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.aggregate_root:
        aggregate(args.aggregate_root)
    else:
        run(args)


if __name__ == "__main__":
    main()
