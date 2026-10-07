#!/usr/bin/env python3
"""Validation-only frozen BBBC047 native-size backbone sensitivity.

This program cannot name or read a test profile.  It creates the sealed
checkpoint manifest that the one historical test recomputation must verify.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from native_common import (
    ARMS,
    BUDGETS,
    EFFECTIVE_BATCH_SIZE,
    EXPECTED_ROLE_SHA256,
    MAX_EPOCHS,
    SEEDS,
    VERSION,
    build_model,
    inner_train_mask,
    load_module,
    load_official_modules,
    make_optimizer,
    model_audit,
    model_specs,
    prediction,
    save_json,
    set_seed,
    sha256_file,
    source_audit,
    supervised_mse,
    torch_modules,
    training_output,
)


EXPECTED_INPUT_SHA256 = {
    "train_cp_plate_rows.npz": "d02732a2a409171fd9457eaee48238218d39b540fd2df45c2e3e684c5e803530",
    "valid_cp_plate_rows.npz": "7931e18190f7b42a9d814190429aa0fc80ce8b5d05257d1d196c4f2e5e325c14",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles-root", type=Path, required=True)
    parser.add_argument("--rows-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--vendor-deps", type=Path, required=True)
    parser.add_argument("--reference-script", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


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


def finite_matrix(name: str, value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 775 or not np.isfinite(result).all():
        raise RuntimeError(f"{name} must be finite n-by-775, got {result.shape}")
    return result


def zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(x, axis=0, dtype=np.float64).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return mean, scale


def normalize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = np.asarray((x - mean) / scale, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite normalized tensor")
    return result


def input_audit(args: argparse.Namespace) -> dict[str, Any]:
    row_files = {path.name for path in args.rows_root.glob("*_cp_plate_rows.npz")}
    if row_files != set(EXPECTED_INPUT_SHA256):
        raise RuntimeError(
            "validation rows root must contain exactly train and valid profile values; "
            f"got {sorted(row_files)}"
        )
    hashes = {name: sha256_file(args.rows_root / name) for name in sorted(row_files)}
    if hashes != EXPECTED_INPUT_SHA256:
        raise RuntimeError(f"validation input hash mismatch: {hashes}")
    manifest = args.roles_root / "FROZEN_ROLE_MANIFEST.csv"
    role_audit_path = args.roles_root / "ROLE_FREEZE_AUDIT.json"
    if not manifest.is_file() or not role_audit_path.is_file():
        raise FileNotFoundError("missing frozen role manifest/audit")
    role_audit = json.loads(role_audit_path.read_text(encoding="utf-8"))
    manifest_hash = sha256_file(manifest)
    if manifest_hash != EXPECTED_ROLE_SHA256:
        raise RuntimeError("role manifest differs from the declared 775D benchmark freeze")
    if role_audit.get("status") != "PASS" or role_audit.get("test_profile_values_loaded") is not False:
        raise RuntimeError("role audit is not a passed metadata-only freeze")
    return {
        "status": "PASS",
        "rows_root": str(args.rows_root),
        "row_files": hashes,
        "test_profile_values_loaded": False,
        "test_profile_file_present": False,
        "role_manifest_sha256": manifest_hash,
        "role_freeze_audit": str(role_audit_path),
    }


def train_epoch(
    *, name: str, model: Any, optimizer: Any, x: np.ndarray, y: np.ndarray,
    torch: Any, device: Any, seed: int,
) -> float:
    micro_batch = int(model_specs()[name]["micro_batch_size"])
    accumulation = EFFECTIVE_BATCH_SIZE // micro_batch
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=micro_batch, shuffle=True, num_workers=0, generator=generator
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    group_size = 0
    for index, (batch_x, batch_y) in enumerate(loader):
        if index % accumulation == 0:
            group_size = min(accumulation, len(loader) - index)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = training_output(name, model, batch_x)
        loss = supervised_mse(name, output, batch_y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite training loss: {name}")
        (loss / group_size).backward()
        losses.append(float(loss.detach().cpu()))
        if (index + 1) % accumulation == 0 or index + 1 == len(loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def predict(name: str, model: Any, x: np.ndarray, torch: Any, device: Any) -> np.ndarray:
    micro_batch = int(model_specs()[name]["micro_batch_size"])
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), micro_batch):
            batch = torch.from_numpy(x[start : start + micro_batch]).to(device)
            chunks.append(prediction(name, model, batch).detach().cpu().numpy().astype(np.float32))
    return finite_matrix(f"{name} prediction", np.concatenate(chunks, axis=0))


def fit_seed(
    *, name: str, budget: int, seed: int, train_x: np.ndarray, train_y: np.ndarray,
    compounds: np.ndarray, rtdl: Any, tabm: Any, torch: Any, device: Any,
) -> tuple[Any, dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    select_train = inner_train_mask(compounds)
    select_valid = ~select_train
    if len(np.unique(compounds[select_train])) < 2 or len(np.unique(compounds[select_valid])) < 2:
        raise RuntimeError("inner split has insufficient compound diversity")
    x_mean, x_scale = zscore_fit(train_x[select_train])
    y_mean, y_scale = zscore_fit(train_y[select_train])
    x_inner_train = normalize(train_x[select_train], x_mean, x_scale)
    y_inner_train = normalize(train_y[select_train], y_mean, y_scale)
    x_inner_valid = normalize(train_x[select_valid], x_mean, x_scale)
    y_inner_valid = normalize(train_y[select_valid], y_mean, y_scale)

    set_seed(torch, seed)
    selection_model = build_model(name, rtdl, tabm, device)
    selection_optimizer = make_optimizer(name, selection_model, tabm, torch)
    best_epoch, best_mse = 0, float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(
            name=name, model=selection_model, optimizer=selection_optimizer,
            x=x_inner_train, y=y_inner_train, torch=torch, device=device, seed=seed + epoch,
        )
        valid_prediction = predict(name, selection_model, x_inner_valid, torch, device)
        valid_mse = float(np.mean((valid_prediction - y_inner_valid) ** 2, dtype=np.float64))
        if not np.isfinite(valid_mse):
            raise RuntimeError(f"non-finite inner validation MSE: {name}")
        history.append({"epoch": epoch, "train_mse": train_loss, "inner_valid_mse": valid_mse})
        if valid_mse < best_mse:
            best_epoch, best_mse = epoch, valid_mse
    if not 1 <= best_epoch <= MAX_EPOCHS:
        raise RuntimeError(f"no epoch selected for {name}")
    del selection_model, selection_optimizer
    torch.cuda.empty_cache()

    full_x_mean, full_x_scale = zscore_fit(train_x)
    full_y_mean, full_y_scale = zscore_fit(train_y)
    x_full = normalize(train_x, full_x_mean, full_x_scale)
    y_full = normalize(train_y, full_y_mean, full_y_scale)
    set_seed(torch, seed)
    final_model = build_model(name, rtdl, tabm, device)
    final_optimizer = make_optimizer(name, final_model, tabm, torch)
    final_losses: list[float] = []
    for epoch in range(1, best_epoch + 1):
        final_losses.append(train_epoch(
            name=name, model=final_model, optimizer=final_optimizer, x=x_full, y=y_full,
            torch=torch, device=device, seed=seed + epoch,
        ))
    return final_model, {
        "seed": seed,
        "budget": budget,
        "arm": name,
        "inner_split": {
            "source": "same deterministic compound split as compact track",
            "train_rows": int(select_train.sum()), "valid_rows": int(select_valid.sum()),
            "train_compounds": int(len(np.unique(compounds[select_train]))),
            "valid_compounds": int(len(np.unique(compounds[select_valid]))),
            "best_epoch": best_epoch, "best_inner_valid_mse": best_mse, "all_epochs": history,
        },
        "all_train_refit": {"epochs": best_epoch, "epoch_losses": final_losses},
    }, (full_x_mean, full_x_scale, full_y_mean, full_y_scale)


def main() -> None:
    args = parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite {args.outdir}")
    if not args.reference_script.is_file():
        raise FileNotFoundError(args.reference_script)
    audit = input_audit(args)
    torch, _ = torch_modules()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this native-size sensitivity")
    device = torch.device("cuda")
    rtdl, tabm = load_official_modules(args.vendor_root, args.vendor_deps)
    reference = load_module("native_validation_reference", args.reference_script)
    args.outdir.mkdir(parents=True)
    save_json(args.outdir / "VALIDATION_INPUT_AUDIT.json", audit)
    source = source_audit(args.vendor_root, args.protocol)
    source.update({"version": VERSION, "device": str(device), "torch": str(torch.__version__)})
    save_json(args.outdir / "SOURCE_AUDIT.json", source)
    capacity = {name: model_audit(name, rtdl, tabm, torch, device) for name in ARMS}
    save_json(args.outdir / "MODEL_CAPACITY_AUDIT.json", capacity)

    train_profiles = reference.load_profiles(args.rows_root, "train")
    valid_profiles = reference.load_profiles(args.rows_root, "valid")
    summary: list[dict[str, Any]] = []
    per_condition: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for budget in BUDGETS:
        train_roles = reference.read_roles(args.roles_root, "train", budget)
        valid_roles = reference.read_roles(args.roles_root, "valid", budget)
        train_x, train_y, train_compounds, _, _, _ = reference.assemble(train_roles, train_profiles, foreign=False)
        valid_x, valid_held, valid_compounds, valid_foreign, valid_conditions, _ = reference.assemble(valid_roles, valid_profiles, foreign=True)
        train_x, train_y = finite_matrix("train support", train_x), finite_matrix("train held", train_y)
        valid_x, valid_held = finite_matrix("valid support", valid_x), finite_matrix("valid held", valid_held)
        for name in ARMS:
            seed_predictions: list[np.ndarray] = []
            for seed in SEEDS:
                model, fit_metadata, normalizer = fit_seed(
                    name=name, budget=budget, seed=seed, train_x=train_x, train_y=train_y,
                    compounds=train_compounds, rtdl=rtdl, tabm=tabm, torch=torch, device=device,
                )
                x_mean, x_scale, y_mean, y_scale = normalizer
                prediction_values = predict(name, model, normalize(valid_x, x_mean, x_scale), torch, device)
                seed_predictions.append(finite_matrix("unscaled validation prediction", prediction_values * y_scale + y_mean))
                checkpoint_dir = args.outdir / "checkpoints" / f"{budget}R"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"{name}-seed{seed}.pt"
                torch.save({
                    "version": VERSION, "arm": name, "budget": budget, "seed": seed,
                    "state_dict": model.state_dict(),
                    "normalizer": {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "y_scale": y_scale},
                    "fit_metadata": fit_metadata, "capacity": capacity[name],
                }, checkpoint_path)
                checkpoints.append({"arm": name, "budget": budget, "seed": seed, "path": str(checkpoint_path),
                                    "sha256": sha256_file(checkpoint_path), "selected_epoch": fit_metadata["inner_split"]["best_epoch"]})
                del model
                torch.cuda.empty_cache()
            mean_prediction = finite_matrix("seed ensemble", np.mean(seed_predictions, axis=0, dtype=np.float32))
            metrics = reference.method_metrics(mean_prediction, valid_held, valid_foreign, valid_x)
            summary.append(reference.score_row(name, budget, metrics, valid_compounds))
            for index, condition_id in enumerate(valid_conditions):
                per_condition.append({"version": VERSION, "split": "valid", "budget": budget, "method": name,
                                      "condition_id": str(condition_id), "compound_id": str(valid_compounds[index]),
                                      **{metric: float(values[index]) for metric, values in metrics.items()}})
            del seed_predictions, mean_prediction
            torch.cuda.empty_cache()
    write_csv(args.outdir / "NATIVE_VALIDATION_METHOD_SUMMARY.csv", summary)
    write_csv(args.outdir / "NATIVE_VALIDATION_PER_CONDITION_METRICS.csv", per_condition)
    save_json(args.outdir / "CHECKPOINT_MANIFEST.json", {"version": VERSION, "checkpoints": checkpoints})
    save_json(args.outdir / "NATIVE_VALIDATION_COMPLETE.json", {
        "status": "PASS", "version": VERSION, "stage": "validation-only",
        "test_profile_values_loaded": False, "test_profile_file_present": False, "test_evaluated": False,
        "budgets": list(BUDGETS), "arms": list(ARMS), "seeds": list(SEEDS), "max_epochs": MAX_EPOCHS,
        "role_manifest_sha256": EXPECTED_ROLE_SHA256,
        "checkpoint_manifest_sha256": sha256_file(args.outdir / "CHECKPOINT_MANIFEST.json"),
    })


if __name__ == "__main__":
    main()
