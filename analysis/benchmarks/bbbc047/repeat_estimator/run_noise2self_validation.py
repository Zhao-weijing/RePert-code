#!/usr/bin/env python3
"""Train and freeze the validation-only same-space Noise2Self comparator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
EXTERNAL_PATH = (Path(__file__).resolve().parents[4] / "analysis/benchmarks/bbbc047/repeat_estimator/run_external_validation.py")
spec = importlib.util.spec_from_file_location("external_validation", EXTERNAL_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {EXTERNAL_PATH}")
EXT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = EXT
spec.loader.exec_module(EXT)

VERSION = "BBBC047-repeat-estimator-noise2self-validation-v1-2026-09-16"
MASK_RATES = (0.10, 0.20, 0.30)
BUDGETS = (1, 2, 3)
EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
MASK_SEED = 3407
INFERENCE_PASSES = 10


class MaskedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(775, 256), nn.GELU(),
            nn.Linear(256, 32), nn.GELU(),
            nn.Linear(32, 256), nn.GELU(),
            nn.Linear(256, 775),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rows-root", type=Path, default=EXT.DEFAULT_ROWS)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{MASK_SEED}|{VERSION}|{label}".encode()).digest()[:8], "big") % (2**31 - 1)


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


def all_train_roles(root: Path, budget: int) -> list[dict[str, str]]:
    path = root / "FROZEN_ROLE_MANIFEST.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row["split"] == "train" and int(row["budget"]) == budget]
    if len(rows) < 500:
        raise RuntimeError(f"insufficient train roles for {budget}R")
    return rows


def balanced_masks(rate: float, device: torch.device) -> torch.Tensor:
    """Ten deterministic masks; every coordinate is masked rate*10 times."""
    repetitions = int(round(rate * INFERENCE_PASSES))
    if repetitions not in (1, 2, 3):
        raise RuntimeError(f"unsupported frozen mask rate {rate}")
    generator = torch.Generator(device="cpu").manual_seed(stable_seed(f"inference-mask-{rate}"))
    permutation = torch.randperm(775, generator=generator)
    bins = torch.empty(775, dtype=torch.long)
    bins[permutation] = torch.arange(775) % INFERENCE_PASSES
    masks = torch.zeros((INFERENCE_PASSES, 775), dtype=torch.bool)
    features = torch.arange(775)
    for offset in range(repetitions):
        masks[(bins + offset) % INFERENCE_PASSES, features] = True
    if not torch.all(masks.sum(dim=0) == repetitions):
        raise RuntimeError("balanced inference masks fail exact coordinate coverage")
    return masks.to(device)


def masked_loss(model: nn.Module, values: torch.Tensor, rate: float, generator: torch.Generator) -> torch.Tensor:
    mask = torch.rand(values.shape, device=values.device, generator=generator) < rate
    # 775 dimensions makes an all-unmasked row astronomically unlikely, but
    # guarantee a valid loss denominator rather than silently changing target.
    empty = ~mask.any(dim=1)
    if torch.any(empty):
        mask[empty, 0] = True
    prediction = model(values.masked_fill(mask, 0.0))
    return ((prediction - values).square() * mask).sum() / mask.sum()


@torch.no_grad()
def jinvariant_predict(model: nn.Module, values: np.ndarray, rate: float, mean: np.ndarray, scale: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    masks = balanced_masks(rate, device)
    x = (np.asarray(values, dtype=np.float32) - mean) / scale
    output = np.zeros_like(x, dtype=np.float64)
    counts = masks.sum(dim=0).detach().cpu().numpy().astype(np.int32)
    for start in range(0, len(x), BATCH_SIZE):
        batch = torch.from_numpy(x[start : start + BATCH_SIZE]).to(device)
        for mask in masks:
            prediction = model(batch.masked_fill(mask[None, :], 0.0)).detach().cpu().numpy()
            selected = mask.detach().cpu().numpy()
            output[start : start + len(batch), selected] += prediction[:, selected]
    if np.any(counts == 0):
        raise RuntimeError("a feature was copied from raw input: inference coverage failure")
    return (output / counts[None, :] * scale + mean).astype(np.float32)


def fit_one(train_x: np.ndarray, valid_x: np.ndarray, rate: float, budget: int, device: torch.device) -> tuple[MaskedMLP, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(train_x.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    train_z = (train_x - mean) / scale
    valid_z = (valid_x - mean) / scale
    seed = stable_seed(f"fit-b{budget}-rate{rate}")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = MaskedMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_z)), batch_size=BATCH_SIZE, shuffle=True, generator=loader_generator, num_workers=0)
    mask_generator = torch.Generator(device=device.type).manual_seed(seed + 1)
    validation_generator = torch.Generator(device=device.type).manual_seed(seed + 2)
    fixed_validation = torch.from_numpy(valid_z).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    logs: list[dict[str, Any]] = []
    for epoch in range(EPOCHS):
        model.train()
        total, count = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model, batch, rate, mask_generator)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite train loss b{budget} rate={rate}")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            count += len(batch)
        model.eval()
        with torch.no_grad():
            validation_loss = float(masked_loss(model, fixed_validation, rate, validation_generator).cpu())
        logs.append({"budget": budget, "mask_rate": rate, "epoch": epoch, "train_masked_mse": total / count, "validation_masked_mse": validation_loss})
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no Noise2Self checkpoint")
    model.load_state_dict(best_state)
    return model, mean, scale, logs


def main() -> None:
    args = parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite {args.outdir}")
    audit = EXT.check_freeze(args.roles_root, args.protocol)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_profiles = EXT.load_profiles(args.rows_root, "train")
    valid_profiles = EXT.load_profiles(args.rows_root, "valid")
    args.outdir.mkdir(parents=True)
    summary: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    for budget in BUDGETS:
        train_roles = all_train_roles(args.roles_root, budget)
        valid_roles = EXT.read_roles(args.roles_root, "valid", budget)
        train_x, _, _, _, _, _ = EXT.assemble(train_roles, train_profiles, foreign=False)
        valid_x, valid_held, compounds, valid_foreign, _, _ = EXT.assemble(valid_roles, valid_profiles, foreign=True)
        candidates: dict[float, tuple[MaskedMLP, np.ndarray, np.ndarray, np.ndarray]] = {}
        for rate in MASK_RATES:
            model, mean, scale, fit_logs = fit_one(train_x, valid_x, rate, budget, device)
            logs.extend(fit_logs)
            prediction = jinvariant_predict(model, valid_x, rate, mean, scale, device)
            metrics = EXT.method_metrics(prediction, valid_held, valid_foreign, valid_x)
            row = EXT.score_row("Noise2Self", budget, metrics, compounds)
            row.update({"mask_rate": rate, "architecture": "775-256-32-256-775-GELU", "inference_passes": INFERENCE_PASSES, "inference_feature_copies_from_input": 0})
            summary.append(row)
            candidates[rate] = (model, mean, scale, prediction)
        selected_rate = max(MASK_RATES, key=lambda rate: (next(row["E_mean"] for row in summary if row["budget"] == budget and row["mask_rate"] == rate), -rate))
        model, mean, scale, _ = candidates[selected_rate]
        checkpoint = args.outdir / f"budget{budget}_selected_noise2self.pt"
        torch.save({"version": VERSION, "budget": budget, "mask_rate": selected_rate, "architecture": "775-256-32-256-775-GELU", "inference_passes": INFERENCE_PASSES, "mask_seed": MASK_SEED, "mean": mean, "scale": scale, "state_dict": model.state_dict()}, checkpoint)
        selected[str(budget)] = {"mask_rate": selected_rate, "checkpoint": str(checkpoint), "checkpoint_sha256": EXT.sha256_file(checkpoint), "selection_endpoint": "validation compound-balanced E"}
    write_csv(args.outdir / "VALIDATION_NOISE2SELF_SUMMARY.csv", summary)
    write_csv(args.outdir / "NOISE2SELF_TRAINING_LOG.csv", logs)
    marker = {"version": VERSION, "stage": "noise2self_validation", "protocol_sha256": EXT.sha256_file(args.protocol), "role_manifest_sha256": audit["manifest"]["sha256"], "architecture": "775-256-32-256-775-GELU", "mask_rates": list(MASK_RATES), "inference": {"passes": INFERENCE_PASSES, "balanced_masks": True, "raw_coordinate_copies": 0}, "optimizer": {"name": "AdamW", "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "epochs": EPOCHS, "batch_size": BATCH_SIZE}, "selected": selected, "test_loaded": False, "test_profile_values_loaded": False, "test_used_for_selection": False, "status": "PASS"}
    (args.outdir / "NOISE2SELF_VALIDATION_COMPLETE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": marker["stage"], "selected": selected, "test_loaded": False, "device": str(device)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
