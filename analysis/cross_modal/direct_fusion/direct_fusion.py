#!/usr/bin/env python3
"""Comparison-only direct fusion of GE support and aggregate CP evidence."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent.parent / "expression_teacher"))
sys.path.insert(0, str(HERE.parent.parent / "cell_painting_to_expression"))
import ge_teacher_experiment as T  # noqa: E402
import cp_to_ge_residual as C  # noqa: E402


VERSION = "MVCPert-GE-support-plus-CP-direct-fusion-2026-08-30"
SEEDS = T.SEEDS
INPUT_DIM = 977 + 775
OUTPUT_DIM = 977


class DirectFusionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, 256), nn.GELU(),
            nn.Linear(256, 32), nn.GELU(),
            nn.Linear(32, 256), nn.GELU(),
            nn.Linear(256, OUTPUT_DIM),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cm0-npz", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0/molecule_aggregates.npz"))
    p.add_argument("--ge-data-root", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"))
    p.add_argument("--teacher-root", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/multimodal_expansion_20260830/01_GE_teacher/formal_v2"))
    p.add_argument("--cp-residual-root", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/multimodal_expansion_20260830/02_CP_to_GE/formal"))
    p.add_argument("--outdir", type=Path)
    p.add_argument("--aggregate-root", type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--null-rounds", type=int, default=32)
    p.add_argument("--bootstrap-rounds", type=int, default=10000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DirectFusionNet, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    set_seed(seed)
    x_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_scale = np.maximum(x_train.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    y_mean = y_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_scale = np.maximum(y_train.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    tx = (x_train - x_mean) / x_scale
    ty = (y_train - y_mean) / y_scale
    vx = (x_valid - x_mean) / x_scale
    vy = (y_valid - y_mean) / y_scale
    model = DirectFusionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(); generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss, best_epoch = float("inf"), -1
    limit = 1 if args.smoke else args.epochs
    for epoch in range(limit):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x.to(device)) - y.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite direct-fusion loss")
            loss.backward(); optimizer.step()
        model.eval(); total = 0.0; count = 0
        with torch.no_grad():
            for start in range(0, len(vx), args.batch_size):
                x = torch.from_numpy(vx[start:start + args.batch_size]).to(device)
                y = torch.from_numpy(vy[start:start + args.batch_size]).to(device)
                value = torch.mean((model(x) - y) ** 2)
                total += float(value.detach().cpu()) * len(x); count += len(x)
        validation_loss = total / max(count, 1)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or epoch % 10 == 0:
            print(f"[seed {seed}] direct-fusion epoch={epoch} validation_mse={validation_loss:.7f}", flush=True)
    if best_state is None:
        raise RuntimeError("Direct-fusion model did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, x_mean, x_scale, y_mean, y_scale, best_epoch, best_loss


def predict(model: DirectFusionNet, x: np.ndarray, x_mean: np.ndarray, x_scale: np.ndarray, y_mean: np.ndarray, y_scale: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval(); output = np.empty((len(x), OUTPUT_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            value = (x[start:start + batch_size] - x_mean) / x_scale
            prediction = model(torch.from_numpy(value).to(device)).detach().cpu().numpy()
            output[start:start + len(prediction)] = prediction * y_scale + y_mean
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def run_one(args: argparse.Namespace) -> None:
    if args.seed is None or args.outdir is None:
        raise ValueError("--seed and --outdir are required")
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    args.outdir.mkdir(parents=True)
    if args.smoke:
        args.bootstrap_rounds = min(args.bootstrap_rounds, 100)
    data = C.load_aggregates(args.cm0_npz)
    rows = {split: T.load_split_rows(args.ge_data_root, split) for split in ("train", "valid", "test")}
    pairs = {split: T.build_pairs(rows[split], split, 1) for split in ("train", "valid", "test")}
    teacher_valid = C.load_teacher(args.teacher_root / f"seed_{args.seed}" / "budget1_valid_predictions.npz")
    teacher_test = C.load_teacher(args.teacher_root / f"seed_{args.seed}" / "budget1_predictions.npz")
    for expected, observed, label in ((pairs["valid"].compound, teacher_valid["compound"], "valid"), (pairs["test"].compound, teacher_test["compound"], "test")):
        if list(observed) != expected:
            raise ValueError(f"Teacher prediction order mismatch for {label}")
    cp_maps = {split: C.aggregate_map(data, split) for split in ("train", "valid", "test")}
    x = {}
    for split in ("train", "valid", "test"):
        cp = np.vstack([cp_maps[split][c] for c in pairs[split].compound])
        x[split] = np.concatenate([pairs[split].support, cp], axis=1).astype(np.float32)
    device = T.choose_device(args.device)
    model, x_mean, x_scale, y_mean, y_scale, best_epoch, best_loss = fit_model(
        x["train"], pairs["train"].target, x["valid"], pairs["valid"].target, args.seed, args, device
    )
    direct_test = predict(model, x["test"], x_mean, x_scale, y_mean, y_scale, args.batch_size, device)
    reference_test = T.strict_reference(rows["test"], pairs["test"], args.null_rounds, args.seed)
    records = {
        "GE_mean": T.method_records(pairs["test"].support, pairs["test"], reference_test, "GE_mean"),
        "GE_teacher": T.method_records(teacher_test["teacher"], pairs["test"], reference_test, "GE_teacher"),
        "direct_fusion": T.method_records(direct_test, pairs["test"], reference_test, "direct_fusion"),
    }
    contrasts = {
        "direct_fusion_minus_GE_mean": T.bootstrap_difference(records["direct_fusion"], records["GE_mean"], args.bootstrap_rounds, T.stable_seed(args.seed, "direct-minus-mean")),
        "direct_fusion_minus_GE_teacher": T.bootstrap_difference(records["direct_fusion"], records["GE_teacher"], args.bootstrap_rounds, T.stable_seed(args.seed, "direct-minus-teacher")),
    }
    write_csv(args.outdir / "per_molecule_scores.csv", [row for method in records.values() for row in method.values()])
    write_csv(args.outdir / "paired_contrasts.csv", [{"contrast": name, **value} for name, value in contrasts.items()])
    payload = {
        "version": VERSION,
        "seed": args.seed,
        "device": str(device),
        "pair_counts": {split: len(pairs[split].compound) for split in ("train", "valid", "test")},
        "reference_count": len(reference_test),
        "checkpoint": {"best_epoch": best_epoch, "best_validation_mse": best_loss},
        "summaries": {name: T.summarize(value) for name, value in records.items()},
        "contrasts": contrasts,
        "guard": "comparison-only; direct fusion does not enter the hard GO/NO-GO gate",
    }
    (args.outdir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def read_scores(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.setdefault(row["method"], {})[row["canonical_compound"]] = {
                "model_excess_z": float(row["model_excess_z"]),
                "raw_pcc": float(row["raw_pcc"]),
            }
    return output


def pooled_contrast(root: Path, left: str, right: str, rounds: int) -> dict[str, Any]:
    by_seed = [read_scores(root / f"seed_{seed}" / "per_molecule_scores.csv") for seed in SEEDS]
    common = set(by_seed[0][left]) & set(by_seed[0][right])
    for scores in by_seed[1:]:
        common &= set(scores[left]) & set(scores[right])
    common = sorted(common)
    diff_excess = np.asarray([np.mean([scores[left][c]["model_excess_z"] - scores[right][c]["model_excess_z"] for scores in by_seed]) for c in common])
    diff_raw = np.asarray([np.mean([scores[left][c]["raw_pcc"] - scores[right][c]["raw_pcc"] for scores in by_seed]) for c in common])
    rng = np.random.default_rng(T.stable_seed(20260830, f"direct-pooled|{left}|{right}"))
    bx = np.empty(rounds); br = np.empty(rounds)
    for begin in range(0, rounds, 100):
        end = min(rounds, begin + 100); draw = rng.integers(0, len(common), size=(end - begin, len(common)))
        bx[begin:end] = diff_excess[draw].mean(axis=1); br[begin:end] = diff_raw[draw].mean(axis=1)
    return {
        "scope": "pooled_three_seed",
        "left": left,
        "right": right,
        "molecule_count": len(common),
        "excess_z_point": float(diff_excess.mean()),
        "excess_z_ci95": [float(np.quantile(bx, .025)), float(np.quantile(bx, .975))],
        "raw_pcc_point": float(diff_raw.mean()),
        "raw_pcc_ci95": [float(np.quantile(br, .025)), float(np.quantile(br, .975))],
        "bootstrap_rounds": rounds,
    }


def read_cp_scores(path: Path) -> dict[str, dict[str, float]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[row["canonical_compound"]] = {"model_excess_z": float(row["model_excess_z"]), "raw_pcc": float(row["raw_pcc"])}
    return result


def pooled_external_contrast(root: Path, cp_root: Path, rounds: int) -> dict[str, Any]:
    tables = []
    for seed in SEEDS:
        direct = read_scores(root / f"seed_{seed}" / "per_molecule_scores.csv")["direct_fusion"]
        cp = read_cp_scores(cp_root / f"seed_{seed}" / "correct.csv")
        tables.append({"direct_fusion": direct, "CP_residual": cp})
    common = set(tables[0]["direct_fusion"]) & set(tables[0]["CP_residual"])
    for table in tables[1:]:
        common &= set(table["direct_fusion"]) & set(table["CP_residual"])
    common = sorted(common)
    diff_excess = np.asarray([np.mean([table["direct_fusion"][c]["model_excess_z"] - table["CP_residual"][c]["model_excess_z"] for table in tables]) for c in common])
    diff_raw = np.asarray([np.mean([table["direct_fusion"][c]["raw_pcc"] - table["CP_residual"][c]["raw_pcc"] for table in tables]) for c in common])
    rounds = rounds or 10000
    rng = np.random.default_rng(T.stable_seed(20260830, "direct-pooled|direct_fusion|CP_residual"))
    bx = np.empty(rounds); br = np.empty(rounds)
    for begin in range(0, rounds, 100):
        end = min(rounds, begin + 100); draw = rng.integers(0, len(common), size=(end - begin, len(common)))
        bx[begin:end] = diff_excess[draw].mean(axis=1); br[begin:end] = diff_raw[draw].mean(axis=1)
    return {"scope": "pooled_three_seed", "left": "direct_fusion", "right": "CP_residual", "molecule_count": len(common), "excess_z_point": float(diff_excess.mean()), "excess_z_ci95": [float(np.quantile(bx, .025)), float(np.quantile(bx, .975))], "raw_pcc_point": float(diff_raw.mean()), "raw_pcc_ci95": [float(np.quantile(br, .025)), float(np.quantile(br, .975))], "bootstrap_rounds": rounds}


def aggregate(args: argparse.Namespace) -> None:
    if args.aggregate_root is None:
        raise ValueError("--aggregate-root is required")
    root = args.aggregate_root
    metric_rows = []
    contrast_rows = []
    for seed in SEEDS:
        payload = json.loads((root / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))
        for method, summary in payload["summaries"].items():
            metric_rows.append({"scope": "seed", "seed": seed, "method": method, **summary})
        cp_payload = json.loads((args.cp_residual_root / f"seed_{seed}" / "metrics.json").read_text(encoding="utf-8"))
        metric_rows.append({"scope": "seed", "seed": seed, "method": "CP_residual", **cp_payload["test_summaries"]["correct_CP"]})
        for name, value in payload["contrasts"].items():
            contrast_rows.append({"scope": "seed", "seed": seed, "contrast": name, **value})
    for left, right in (("direct_fusion", "GE_mean"), ("direct_fusion", "GE_teacher")):
        contrast_rows.append(pooled_contrast(root, left, right, args.bootstrap_rounds or 10000))
    contrast_rows.append(pooled_external_contrast(root, args.cp_residual_root, args.bootstrap_rounds or 10000))
    write_csv(root / "comparison.csv", metric_rows + contrast_rows)
    write_csv(root / "bootstrap_summary.csv", contrast_rows)
    payload = {
        "version": VERSION,
        "seeds": list(SEEDS),
        "comparison_file": str(root / "comparison.csv"),
        "bootstrap_file": str(root / "bootstrap_summary.csv"),
        "cp_residual_root": str(args.cp_residual_root),
        "decision": "comparison-only; no hard GO/NO-GO decision is assigned",
    }
    (root / "aggregate_index.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def dry_run(args: argparse.Namespace) -> None:
    data = C.load_aggregates(args.cm0_npz)
    rows = {split: T.load_split_rows(args.ge_data_root, split) for split in ("train", "valid", "test")}
    pairs = {split: T.build_pairs(rows[split], split, 1) for split in ("train", "valid", "test")}
    print(json.dumps({"version": VERSION, "pair_counts": {split: len(pairs[split].compound) for split in pairs}, "input_shapes": {split: [len(pairs[split].compound), INPUT_DIM] for split in pairs}, "guard": "dry-run only"}, indent=2, sort_keys=True))


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
