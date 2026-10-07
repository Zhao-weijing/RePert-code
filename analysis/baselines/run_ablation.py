"""Execute the frozen baseline-ablation suite.

This file is designed to run unchanged on the AIDD Linux host and is also the
auditable source of the local result package.  The default data paths are the
remote prepared artifacts; every path is overridable from the command line.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - the audit phase remains importable
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc

from ablation_common import (
    BETA_GRID,
    BOOTSTRAP_N,
    PCA_CANDIDATES,
    PROTOCOL_VERSION,
    SEEDS,
    DatasetBundle,
    ExampleSet,
    all_settings,
    bootstrap_difference,
    decode_scalar,
    example_manifest,
    load_bundle,
    make_examples,
    metric_arrays,
    read_csv,
    rowwise_corr,
    setting_id,
    sha256_file,
    stable_int,
    summarize_metrics,
    write_csv,
    write_json,
)


DEFAULT_BBBC_ROOT = Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0")
DEFAULT_CPG_CP = Path("/path/to/data/AIDD/MVCPert_5_27/runs/cpg0004_lincs_cp_prepared_20260830/artifact/cp_plate_rows.npz")
DEFAULT_CPG_GE = Path("/path/to/data/AIDD/MVCPert_5_27/runs/cpg0004_lincs_ge_prepared_20260830/artifact/ge_plate_rows.npz")
DEFAULT_CPG_LOCK = Path("/path/to/data/AIDD/MVCPert_5_27/runs/cpg0004_lincs_cp_prepared_20260830/artifact/split_lock.json")
DEFAULT_OUTPUT = Path("/path/to/data/AIDD/MVCPert_5_27/runs/baseline_ablation_20260830")


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(f"PyTorch is required for this phase: {TORCH_IMPORT_ERROR}")


def setting_dir(root: Path, section: str, setting: Mapping[str, Any], seed: int) -> Path:
    return root / section / str(setting["dataset"]) / str(setting["target_modality"]) / f"budget{setting['budget']}" / f"seed_{seed}"


def setting_label(setting: Mapping[str, Any]) -> str:
    return f"{setting['dataset']} {setting['target_modality']} {setting['label']}"


def selected_settings(args: argparse.Namespace) -> list[dict[str, Any]]:
    result = []
    for setting in all_settings():
        if args.dataset and setting["dataset"] != args.dataset:
            continue
        if args.modality and setting["target_modality"] != args.modality:
            continue
        if args.budget and int(setting["budget"]) != int(args.budget):
            continue
        result.append(setting)
    return result


def load_for_settings(args: argparse.Namespace, settings: Sequence[Mapping[str, Any]]) -> dict[str, DatasetBundle]:
    bundles: dict[str, DatasetBundle] = {}
    for dataset in sorted({str(s["dataset"]) for s in settings}):
        bundles[dataset] = load_bundle(
            dataset,
            bbbc_root=args.bbbc_root,
            cpg_cp_path=args.cpg_cp,
            cpg_ge_path=args.cpg_ge,
            cpg_split_lock=args.cpg_lock,
        )
    return bundles


def make_setting_examples(
    bundle: DatasetBundle,
    setting: Mapping[str, Any],
    seed: int,
) -> dict[str, ExampleSet]:
    target_modality = str(setting["target_modality"])
    source_modality = str(setting["source_modality"])
    budget = int(setting["budget"])
    train_target = bundle.get("train", target_modality)
    train_source = bundle.get("train", source_modality)
    valid_target = bundle.get("valid", target_modality)
    valid_source = bundle.get("valid", source_modality)
    test_target = bundle.get("test", target_modality)
    test_source = bundle.get("test", source_modality)
    return {
        # The direct target model can use every target condition in training.
        "train": make_examples(train_target, None, budget, seed, all_conditions=True, require_source=False),
        # The crossmodal training set is explicitly separate and source-eligible.
        "train_common": make_examples(train_target, train_source, budget, seed, all_conditions=True, require_source=True),
        # All evaluation methods use the same one-condition-per-compound common set.
        "valid": make_examples(valid_target, valid_source, budget, seed, all_conditions=False, require_source=True),
        "test": make_examples(test_target, test_source, budget, seed, all_conditions=False, require_source=True),
        # Target-only audit counts are not used for final crossmodal comparisons.
        "valid_target_only": make_examples(valid_target, None, budget, seed, all_conditions=False, require_source=False),
        "test_target_only": make_examples(test_target, None, budget, seed, all_conditions=False, require_source=False),
    }


def output_path(args: argparse.Namespace) -> Path:
    args.output_root.mkdir(parents=True, exist_ok=True)
    return args.output_root


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if torch is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        # Deterministic data order and initialization are frozen; cuBLAS/GELU
        # kernels remain on the normal CUDA path for feasible runtime.
        torch.backends.cudnn.benchmark = False


def fit_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(np.mean(values, axis=0), dtype=np.float32)
    std = np.asarray(np.std(values, axis=0), dtype=np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return mean, std


def normalize(values: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = stats
    return ((np.asarray(values, dtype=np.float32) - mean) / std).astype(np.float32)


def denormalize(values: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = stats
    return (np.asarray(values, dtype=np.float32) * std + mean).astype(np.float32)


if nn is not None:

    class TeacherNet(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 256),
                nn.GELU(),
                nn.Linear(256, 32),
                nn.GELU(),
                nn.Linear(32, 256),
                nn.GELU(),
                nn.Linear(256, dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class MLPNet(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.update = nn.Sequential(
                nn.Linear(dim, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.update(x)


    class CrossModalNet(nn.Module):
        def __init__(self, source_dim: int, target_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(source_dim, 256),
                nn.GELU(),
                nn.Linear(256, 32),
                nn.GELU(),
                nn.Linear(32, 256),
                nn.GELU(),
                nn.Linear(256, target_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


def fit_network(
    model_factory: Callable[[], Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    input_stats: tuple[np.ndarray, np.ndarray],
    output_stats: tuple[np.ndarray, np.ndarray],
    seed: int,
    *,
    epochs: int,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
) -> tuple[Any, list[dict[str, Any]]]:
    require_torch()
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    x = torch.from_numpy(normalize(train_x, input_stats)).to(device)
    y = torch.from_numpy(normalize(train_y, output_stats)).to(device)
    vx = torch.from_numpy(normalize(valid_x, input_stats)).to(device)
    vy = torch.from_numpy(normalize(valid_y, output_stats)).to(device)
    n = int(x.shape[0])
    if n == 0 or vx.shape[0] == 0:
        raise ValueError("empty training or validation examples")
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    log: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = torch.randperm(n, device=device)
        running = 0.0
        count = 0
        for start in range(0, n, batch_size):
            ix = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[ix])
            loss = torch.mean((prediction - y[ix]) ** 2)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * int(ix.numel())
            count += int(ix.numel())
        model.eval()
        with torch.no_grad():
            validation_loss = float(torch.mean((model(vx) - vy) ** 2).detach().cpu())
        row = {"epoch": epoch, "train_mse_normalized": running / max(1, count), "valid_mse_normalized": validation_loss}
        log.append(row)
        if np.isfinite(validation_loss) and validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    # Keep the model on the device for the caller's prediction, then it may be
    # released immediately after the setting is written.
    return model, log


def predict_network(
    model: Any,
    values: np.ndarray,
    input_stats: tuple[np.ndarray, np.ndarray],
    output_stats: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    require_torch()
    device = next(model.parameters()).device
    with torch.no_grad():
        normalized = model(torch.from_numpy(normalize(values, input_stats)).to(device)).detach().cpu().numpy()
    return denormalize(normalized, output_stats)


def write_training_log(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(path, list(rows), ["epoch", "train_mse_normalized", "valid_mse_normalized"])


def save_prediction_npz(path: Path, examples: ExampleSet, **predictions: np.ndarray) -> None:
    payload: dict[str, Any] = {
        "compound": examples.compound,
        "dose": examples.dose,
        "condition": examples.condition,
        "support": examples.support,
        "target": examples.target,
        "aggregate_target": examples.aggregate_target,
        "foreign": examples.foreign,
        "foreign_ok": examples.foreign_ok.astype(np.uint8),
        "foreign_compound": examples.foreign_compound,
        "foreign_condition": examples.foreign_condition,
        "support_plates": np.asarray(["|".join(x) for x in examples.support_plates], dtype=str),
        "held_plates": np.asarray(["|".join(x) for x in examples.held_plates], dtype=str),
    }
    if examples.source is not None:
        payload["source"] = examples.source
    payload.update(predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def save_manifest(path: Path, examples: ExampleSet) -> None:
    write_csv(path, example_manifest(examples))


def write_method_outputs(
    outdir: Path,
    methods: Mapping[str, np.ndarray],
    examples_by_split: Mapping[str, ExampleSet],
    seed: int,
    *,
    bootstrap_n: int,
) -> list[dict[str, Any]]:
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for split, examples in examples_by_split.items():
        manifest_path = outdir / f"{split}_pair_manifest.csv"
        if not manifest_path.exists():
            save_manifest(manifest_path, examples)
        pair_rows: list[dict[str, Any]] = []
        for method, prediction in methods.items():
            summary, metrics = summarize_metrics(method, prediction, examples, seed, split, bootstrap_n=bootstrap_n)
            summary_rows.append(summary)
            for i in range(len(examples)):
                pair_rows.append(
                    {
                        "dataset": examples.dataset,
                        "target_modality": examples.modality,
                        "budget": examples.budget,
                        "seed": seed,
                        "split": split,
                        "compound": examples.compound[i],
                        "dose": examples.dose[i],
                        "method": method,
                        "foreign_ok": int(examples.foreign_ok[i]),
                        "pcc": metrics["pcc"][i],
                        "foreign_pcc": metrics["foreign_pcc"][i],
                        "baseline_pcc": metrics["baseline_pcc"][i],
                        "delta_pcc": metrics["delta_pcc"][i],
                        "excess_z": metrics["excess_z"][i],
                        "excess_pcc": metrics["excess_pcc"][i],
                    }
                )
        write_csv(outdir / f"{split}_per_pair_scores.csv", pair_rows)
    write_csv(outdir / "summary.csv", summary_rows)
    return summary_rows


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {key: z[key] for key in z.files}


def run_core_setting(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    setting: Mapping[str, Any],
    seed: int,
) -> None:
    require_torch()
    examples = make_setting_examples(bundle, setting, seed)
    train, valid, test = examples["train"], examples["valid"], examples["test"]
    outdir = setting_dir(args.output_root, "00_protocol/core", setting, seed)
    outdir.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs or (3 if args.smoke else 80))
    bootstrap_n = int(args.bootstrap_n or (100 if args.smoke else BOOTSTRAP_N))
    target_stats = fit_stats(train.support)
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "setting": dict(setting),
        "seed": seed,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "architecture": "TeacherNet: d-256-32-256-d, GELU",
        "optimizer": "AdamW(lr=3e-4, weight_decay=1e-5)",
        "batch_size": 256,
        "epochs": epochs,
        "input": "target-modality support plate mean",
        "normalization": "train-support mean/std only; heldout target never enters input normalization",
        "train_n": len(train),
        "valid_n": len(valid),
        "test_n": len(test),
        "train_common_n": len(examples["train_common"]),
        "target_dim": bundle.get("train", setting["target_modality"]).dim,
    }
    write_json(outdir / "config.json", config)
    np.savez_compressed(outdir / "target_normalization.npz", mean=target_stats[0], std=target_stats[1])
    write_csv(outdir / "train_pair_manifest.csv", example_manifest(train))

    direct_model, direct_log = fit_network(
        lambda: TeacherNet(train.support.shape[1]),
        train.support,
        train.target,
        valid.support,
        valid.target,
        target_stats,
        target_stats,
        seed,
        epochs=epochs,
    )
    aggregate_model, aggregate_log = fit_network(
        lambda: TeacherNet(train.support.shape[1]),
        train.support,
        train.aggregate_target,
        valid.support,
        valid.target,
        target_stats,
        target_stats,
        seed,
        epochs=epochs,
    )
    write_training_log(outdir / "reproducibility_training_log.csv", direct_log)
    write_training_log(outdir / "aggregate_target_training_log.csv", aggregate_log)
    torch.save(direct_model.state_dict(), outdir / "teacher_reproducibility.pt")
    torch.save(aggregate_model.state_dict(), outdir / "teacher_aggregate_target.pt")
    methods_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split, ex in (("valid", valid), ("test", test)):
        methods_by_split[split] = {
            "raw_mean": ex.support,
            "teacher_reproducibility": predict_network(direct_model, ex.support, target_stats, target_stats),
            "teacher_aggregate": predict_network(aggregate_model, ex.support, target_stats, target_stats),
        }
        save_prediction_npz(
            outdir / f"{split}_predictions.npz",
            ex,
            teacher_reproducibility=methods_by_split[split]["teacher_reproducibility"],
            teacher_aggregate=methods_by_split[split]["teacher_aggregate"],
        )
    write_method_outputs(outdir, {"raw_mean": methods_by_split["valid"]["raw_mean"], "teacher_reproducibility": methods_by_split["valid"]["teacher_reproducibility"], "teacher_aggregate": methods_by_split["valid"]["teacher_aggregate"]}, {"valid": valid}, seed, bootstrap_n=bootstrap_n)
    # The previous call writes valid-only files.  Write both splits in one final
    # call so the summary is complete and reproducible.
    write_method_outputs(outdir, {"raw_mean": methods_by_split["test"]["raw_mean"], "teacher_reproducibility": methods_by_split["test"]["teacher_reproducibility"], "teacher_aggregate": methods_by_split["test"]["teacher_aggregate"]}, {"test": test}, seed, bootstrap_n=bootstrap_n)
    # Merge the two split summaries after the per-split helper calls.
    valid_rows = read_csv(outdir / "summary.csv")
    # The helper's second invocation replaced summary.csv; recover valid rows
    # from its per-pair files by recomputing only the compact summary.
    compact: list[dict[str, Any]] = []
    for split, ex in (("valid", valid), ("test", test)):
        for method, pred in methods_by_split[split].items():
            compact.append(summarize_metrics(method, pred, ex, seed, split, bootstrap_n=bootstrap_n)[0])
    write_csv(outdir / "summary.csv", compact)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def run_core(args: argparse.Namespace, bundles: Mapping[str, DatasetBundle], settings: Sequence[Mapping[str, Any]]) -> None:
    seeds = [int(args.seed)] if args.seed else list(SEEDS)
    for setting in settings:
        for seed in seeds:
            marker = setting_dir(args.output_root, "00_protocol/core", setting, seed) / "summary.csv"
            if marker.exists() and not args.overwrite:
                print(f"[skip] core {setting_label(setting)} seed={seed}", flush=True)
                continue
            print(f"[run] core {setting_label(setting)} seed={seed}", flush=True)
            run_core_setting(args, bundles[setting["dataset"]], setting, seed)


def load_core_evaluation(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    setting: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, ExampleSet], dict[str, dict[str, np.ndarray]]]:
    examples = make_setting_examples(bundle, setting, seed)
    core = setting_dir(args.output_root, "00_protocol/core", setting, seed)
    predictions = {split: read_npz(core / f"{split}_predictions.npz") for split in ("valid", "test")}
    return {"train": examples["train"], "train_common": examples["train_common"], "valid": examples["valid"], "test": examples["test"]}, predictions


def run_pca_setting(args: argparse.Namespace, bundle: DatasetBundle, setting: Mapping[str, Any], seed: int) -> None:
    from sklearn.decomposition import PCA

    examples, core_preds = load_core_evaluation(args, bundle, setting, seed)
    train, valid, test = examples["train"], examples["valid"], examples["test"]
    outdir = setting_dir(args.output_root, "01_pca", setting, seed)
    if (outdir / "summary.csv").exists() and not args.overwrite:
        print(f"[skip] PCA {setting_label(setting)} seed={seed}", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    bootstrap_n = int(args.bootstrap_n or (100 if args.smoke else BOOTSTRAP_N))
    maximum = min(max(PCA_CANDIDATES), train.support.shape[1], train.support.shape[0])
    candidates = sorted({min(int(k), int(maximum)) for k in PCA_CANDIDATES if min(int(k), int(maximum)) >= 1})
    if not candidates:
        candidates = [maximum]
    pca = PCA(n_components=maximum, svd_solver="randomized", random_state=seed)
    pca.fit(train.support)

    def reconstruct(values: np.ndarray, k: int) -> np.ndarray:
        scores = (values - pca.mean_) @ pca.components_[:k].T
        return (pca.mean_ + scores @ pca.components_[:k]).astype(np.float32)

    curve: list[dict[str, Any]] = []
    for k in candidates:
        pred = reconstruct(valid.support, k)
        metrics = metric_arrays(pred, valid)
        curve.append({"rank": k, "validation_excess_z_mean": float(np.nanmean(metrics["excess_z"])), "validation_pcc_mean": float(np.nanmean(metrics["pcc"])), "validation_foreign_n": int(np.sum(valid.foreign_ok))})
    selected = sorted(curve, key=lambda row: (-row["validation_excess_z_mean"] if np.isfinite(row["validation_excess_z_mean"]) else float("inf"), row["rank"]))[0]["rank"]
    valid_pred = reconstruct(valid.support, selected)
    test_pred = reconstruct(test.support, selected)
    np.savez_compressed(outdir / "pca_model.npz", mean=pca.mean_, components=pca.components_, explained_variance=pca.explained_variance_)
    write_csv(outdir / "validation_rank_curve.csv", curve)
    write_json(outdir / "frozen_rank.json", {"selected_rank": selected, "candidates": candidates, "fit_data": "train target-modality support only", "seed": seed})
    save_prediction_npz(outdir / "valid_predictions.npz", valid, pca=valid_pred)
    save_prediction_npz(outdir / "test_predictions.npz", test, pca=test_pred)
    write_method_outputs(outdir, {"pca": valid_pred}, {"valid": valid}, seed, bootstrap_n=bootstrap_n)
    valid_rows = read_csv(outdir / "summary.csv")
    write_method_outputs(outdir, {"pca": test_pred}, {"test": test}, seed, bootstrap_n=bootstrap_n)
    test_rows = read_csv(outdir / "summary.csv")
    write_csv(outdir / "summary.csv", valid_rows + test_rows)


def run_pca(args: argparse.Namespace, bundles: Mapping[str, DatasetBundle], settings: Sequence[Mapping[str, Any]]) -> None:
    require_torch()  # also ensures the core artifacts were produced in the same env
    seeds = [int(args.seed)] if args.seed else list(SEEDS)
    for setting in settings:
        for seed in seeds:
            print(f"[run] PCA {setting_label(setting)} seed={seed}", flush=True)
            run_pca_setting(args, bundles[setting["dataset"]], setting, seed)


def run_mlp_setting(args: argparse.Namespace, bundle: DatasetBundle, setting: Mapping[str, Any], seed: int) -> None:
    require_torch()
    examples, core_preds = load_core_evaluation(args, bundle, setting, seed)
    train, valid, test = examples["train"], examples["valid"], examples["test"]
    outdir = setting_dir(args.output_root, "02_mlp", setting, seed)
    if (outdir / "summary.csv").exists() and not args.overwrite:
        print(f"[skip] MLP {setting_label(setting)} seed={seed}", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs or (3 if args.smoke else 80))
    bootstrap_n = int(args.bootstrap_n or (100 if args.smoke else BOOTSTRAP_N))
    target_stats = fit_stats(train.support)
    model, log = fit_network(
        lambda: MLPNet(train.support.shape[1]),
        train.support,
        train.target,
        valid.support,
        valid.target,
        target_stats,
        target_stats,
        seed,
        epochs=epochs,
    )
    write_json(outdir / "config.json", {"protocol_version": PROTOCOL_VERSION, "setting": dict(setting), "seed": seed, "architecture": "residual MLP d-512-512-d, GELU, dropout=0.1", "optimizer": "AdamW(lr=3e-4, weight_decay=1e-5)", "batch_size": 256, "epochs": epochs, "normalization": "train-support mean/std only", "train_n": len(train), "valid_n": len(valid), "test_n": len(test)})
    write_training_log(outdir / "training_log.csv", log)
    torch.save(model.state_dict(), outdir / "mlp.pt")
    valid_pred = predict_network(model, valid.support, target_stats, target_stats)
    test_pred = predict_network(model, test.support, target_stats, target_stats)
    save_prediction_npz(outdir / "valid_predictions.npz", valid, mlp=valid_pred)
    save_prediction_npz(outdir / "test_predictions.npz", test, mlp=test_pred)
    write_method_outputs(outdir, {"mlp": valid_pred}, {"valid": valid}, seed, bootstrap_n=bootstrap_n)
    valid_rows = read_csv(outdir / "summary.csv")
    write_method_outputs(outdir, {"mlp": test_pred}, {"test": test}, seed, bootstrap_n=bootstrap_n)
    test_rows = read_csv(outdir / "summary.csv")
    write_csv(outdir / "summary.csv", valid_rows + test_rows)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def run_mlp(args: argparse.Namespace, bundles: Mapping[str, DatasetBundle], settings: Sequence[Mapping[str, Any]]) -> None:
    seeds = [int(args.seed)] if args.seed else list(SEEDS)
    for setting in settings:
        for seed in seeds:
            print(f"[run] MLP {setting_label(setting)} seed={seed}", flush=True)
            run_mlp_setting(args, bundles[setting["dataset"]], setting, seed)


def run_late_setting(args: argparse.Namespace, bundle: DatasetBundle, setting: Mapping[str, Any], seed: int) -> None:
    require_torch()
    examples, core_preds = load_core_evaluation(args, bundle, setting, seed)
    train, train_common, valid, test = examples["train"], examples["train_common"], examples["valid"], examples["test"]
    if train_common.source is None or valid.source is None or test.source is None:
        raise ValueError("crossmodal source examples were not constructed")
    outdir = setting_dir(args.output_root, "03_late_fusion", setting, seed)
    if (outdir / "summary.csv").exists() and not args.overwrite:
        print(f"[skip] late fusion {setting_label(setting)} seed={seed}", flush=True)
        return
    outdir.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs or (3 if args.smoke else 80))
    bootstrap_n = int(args.bootstrap_n or (100 if args.smoke else BOOTSTRAP_N))
    source_stats = fit_stats(train_common.source)
    target_stats = fit_stats(train.support)
    cross_model, log = fit_network(
        lambda: CrossModalNet(train_common.source.shape[1], train_common.target.shape[1]),
        train_common.source,
        train_common.target,
        valid.source,
        valid.target,
        source_stats,
        target_stats,
        seed,
        epochs=epochs,
    )
    cross_valid = predict_network(cross_model, valid.source, source_stats, target_stats)
    cross_test = predict_network(cross_model, test.source, source_stats, target_stats)
    target_center = np.mean(train.target, axis=0).astype(np.float32)
    p0_valid = valid.support
    p0_test = test.support
    beta_rows: list[dict[str, Any]] = []
    for beta in BETA_GRID:
        late_valid = (1.0 - beta) * p0_valid + beta * cross_valid
        residual_valid = p0_valid + beta * (cross_valid - target_center)
        beta_rows.append({"beta": beta, "late_validation_excess_z_mean": float(np.nanmean(metric_arrays(late_valid, valid)["excess_z"])), "residual_validation_excess_z_mean": float(np.nanmean(metric_arrays(residual_valid, valid)["excess_z"]))})
    late_beta = sorted(beta_rows, key=lambda row: (-row["late_validation_excess_z_mean"] if np.isfinite(row["late_validation_excess_z_mean"]) else float("inf"), row["beta"]))[0]["beta"]
    residual_beta = sorted(beta_rows, key=lambda row: (-row["residual_validation_excess_z_mean"] if np.isfinite(row["residual_validation_excess_z_mean"]) else float("inf"), row["beta"]))[0]["beta"]
    late_valid = (1.0 - late_beta) * p0_valid + late_beta * cross_valid
    late_test = (1.0 - late_beta) * p0_test + late_beta * cross_test
    residual_valid = p0_valid + residual_beta * (cross_valid - target_center)
    residual_test = p0_test + residual_beta * (cross_test - target_center)
    write_json(outdir / "algebraic_equivalence_audit.json", {"algebraic_equivalence": False, "existing_residual_form": "P0 + beta * (Q_cross - train_target_mean)", "late_fusion_form": "(1-beta) * P0 + beta * Q_cross", "reason": "The residual subtracts a nonzero train target center; it is not the late-fusion convex combination except in the degenerate case Q_cross=P0+train_target_mean.", "beta_grid": list(BETA_GRID)})
    (outdir / "algebraic_equivalence_audit.md").write_text("# Algebraic-equivalence audit\n\n`ALGEBRAIC_EQUIVALENCE = FALSE`. The current residual implementation is `P0 + beta * (Q_cross - train_target_mean)`, whereas late fusion is `(1-beta) * P0 + beta * Q_cross`. The independent cross-modal predictor does not read `P0` and is trained only from source modality to the independent held-out target label.\n", encoding="utf-8")
    write_json(outdir / "config.json", {"protocol_version": PROTOCOL_VERSION, "setting": dict(setting), "seed": seed, "architecture": "independent CrossModalNet source-256-32-256-target, GELU", "optimizer": "AdamW(lr=3e-4, weight_decay=1e-5)", "batch_size": 256, "epochs": epochs, "input": "source condition mean across all source plates", "target": "independent target heldout plate mean", "target_center": "mean(train target heldout labels)", "train_common_n": len(train_common), "valid_n": len(valid), "test_n": len(test), "p0_never_input_to_crossmodal": True})
    write_training_log(outdir / "crossmodal_training_log.csv", log)
    torch.save(cross_model.state_dict(), outdir / "crossmodal.pt")
    write_csv(outdir / "validation_beta_curve.csv", beta_rows)
    write_json(outdir / "selected_betas.json", {"late_beta": late_beta, "residual_beta": residual_beta})
    save_prediction_npz(outdir / "valid_predictions.npz", valid, direct_crossmodal=cross_valid, late_fusion=late_valid, residual_fusion=residual_valid)
    save_prediction_npz(outdir / "test_predictions.npz", test, direct_crossmodal=cross_test, late_fusion=late_test, residual_fusion=residual_test)
    valid_methods = {"direct_crossmodal": cross_valid, "late_fusion": late_valid, "residual_fusion": residual_valid}
    test_methods = {"direct_crossmodal": cross_test, "late_fusion": late_test, "residual_fusion": residual_test}
    write_method_outputs(outdir, valid_methods, {"valid": valid}, seed, bootstrap_n=bootstrap_n)
    valid_rows = read_csv(outdir / "summary.csv")
    write_method_outputs(outdir, test_methods, {"test": test}, seed, bootstrap_n=bootstrap_n)
    test_rows = read_csv(outdir / "summary.csv")
    write_csv(outdir / "summary.csv", valid_rows + test_rows)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def run_late(args: argparse.Namespace, bundles: Mapping[str, DatasetBundle], settings: Sequence[Mapping[str, Any]]) -> None:
    seeds = [int(args.seed)] if args.seed else list(SEEDS)
    for setting in settings:
        for seed in seeds:
            print(f"[run] late fusion {setting_label(setting)} seed={seed}", flush=True)
            run_late_setting(args, bundles[setting["dataset"]], setting, seed)


def run_objective_setting(args: argparse.Namespace, bundle: DatasetBundle, setting: Mapping[str, Any], seed: int) -> None:
    examples = make_setting_examples(bundle, setting, seed)
    outdir = setting_dir(args.output_root, "04_objective_ablation", setting, seed)
    outdir.mkdir(parents=True, exist_ok=True)
    core = setting_dir(args.output_root, "00_protocol/core", setting, seed)
    identity = """Objective-ablation identity audit\n===============================\n\nThe reproducibility and aggregate-target models are the same TeacherNet backbone: d -> 256 -> 32 -> 256 -> d with GELU activations.\nInput: identical target-modality support plate mean.\nInput normalization: identical train-support mean/std, fit without validation/test target values.\nOptimizer: identical AdamW(lr=3e-4, weight_decay=1e-5).\nBatch size: identical 256.\nEpoch budget and validation checkpoint rule: identical.\nRandom seed and GPU/data-order initialization: identical per setting.\nOnly supervised target changes: independent held-out target mean versus the mean of all legal training-condition target repeats.\nThe aggregate-target label is never used as a validation/test input.\n"""
    (outdir / "config_identity_audit.txt").write_text(identity, encoding="utf-8")
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    bootstrap_n = int(args.bootstrap_n or (100 if args.smoke else BOOTSTRAP_N))
    for split in ("valid", "test"):
        ex = examples[split]
        p = read_npz(core / f"{split}_predictions.npz")
        direct = p["teacher_reproducibility"]
        aggregate = p["teacher_aggregate"]
        d_metrics = metric_arrays(direct, ex)
        a_metrics = metric_arrays(aggregate, ex)
        for metric in ("excess_z", "pcc", "delta_pcc", "excess_pcc"):
            point, low, high, n = bootstrap_difference(a_metrics[metric], d_metrics[metric], ex.compound, seed + stable_int(seed, metric), bootstrap_n)
            rows.append({"dataset": ex.dataset, "target_modality": ex.modality, "budget": ex.budget, "seed": seed, "split": split, "comparison": "aggregate_target_minus_reproducibility", "metric": metric, "mean_difference": point, "ci_low": low, "ci_high": high, "n_compounds": n})
        for i in range(len(ex)):
            pair_rows.append({"dataset": ex.dataset, "target_modality": ex.modality, "budget": ex.budget, "seed": seed, "split": split, "compound": ex.compound[i], "excess_z_reproducibility": d_metrics["excess_z"][i], "excess_z_aggregate_target": a_metrics["excess_z"][i], "pcc_reproducibility": d_metrics["pcc"][i], "pcc_aggregate_target": a_metrics["pcc"][i], "aggregate_minus_repro_excess_z": a_metrics["excess_z"][i] - d_metrics["excess_z"][i]})
    write_csv(outdir / "objective_summary.csv", rows)
    write_csv(outdir / "per_pair_objective.csv", pair_rows)
    write_json(outdir / "config.json", {"protocol_version": PROTOCOL_VERSION, "setting": dict(setting), "seed": seed, "source_core": str(core), "identity_audit": "config_identity_audit.txt"})


def run_objective(args: argparse.Namespace, bundles: Mapping[str, DatasetBundle], settings: Sequence[Mapping[str, Any]]) -> None:
    for setting in settings:
        seeds = [int(args.seed)] if args.seed else list(SEEDS)
        for seed in seeds:
            print(f"[run] objective ablation {setting_label(setting)} seed={seed}", flush=True)
            run_objective_setting(args, bundles[setting["dataset"]], setting, seed)


def phase0_stats(bundle: DatasetBundle, dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        for modality in ("CP", "GE"):
            data = bundle.get(split, modality)
            groups = data.plate_groups()
            rows.append({"dataset": dataset, "modality": modality, "split": split, "n_rows": data.n_rows, "feature_dim": data.dim, "unique_compounds": len(set(data.compound)), "conditions": len(groups), "unique_plates": len(set(data.plate)), "eligible_budget1": sum(len(x) >= 2 for x in groups.values()), "eligible_budget2": sum(len(x) >= 3 for x in groups.values()), "eligible_budget3": sum(len(x) >= 4 for x in groups.values()), "preprocessing": "delta plate means; dose canonicalized by frozen dataset rule"})
    return rows


def run_phase0(args: argparse.Namespace, bundles: Mapping[str, DatasetBundle]) -> None:
    root = args.output_root / "00_protocol"
    root.mkdir(parents=True, exist_ok=True)
    data_audit: list[dict[str, Any]] = []
    for dataset, bundle in bundles.items():
        data_audit.extend(phase0_stats(bundle, dataset))
    write_csv(root / "DATA_SPLIT_AUDIT.csv", data_audit)
    support_audit: list[dict[str, Any]] = []
    for setting in all_settings():
        if setting["dataset"] not in bundles:
            continue
        bundle = bundles[setting["dataset"]]
        for seed in SEEDS:
            examples = make_setting_examples(bundle, setting, seed)
            for split in ("train", "valid", "test"):
                common = examples["train_common"] if split == "train" else examples[split]
                target_only = examples["train"] if split == "train" else examples[f"{split}_target_only"]
                bad_overlap = sum(bool(set(x) & set(y)) for x, y in zip(common.support_plates, common.held_plates))
                support_audit.append({"dataset": setting["dataset"], "target_modality": setting["target_modality"], "source_modality": setting["source_modality"], "budget": setting["budget"], "seed": seed, "split": split, "target_only_n": len(target_only), "source_common_n": len(common), "foreign_eligible_n": int(np.sum(common.foreign_ok)), "support_heldout_plate_overlap_n": bad_overlap, "compound_unique_n": len(set(common.compound)), "selection": "one condition per compound for valid/test; all eligible conditions for train", "source_rule": "condition-key and nominal-dose intersection"})
    write_csv(root / "SUPPORT_HELDOUT_AUDIT.csv", support_audit)
    metric_protocol = """# Frozen metric protocol\n\n- Primary endpoint: per-compound held-out recovery excess Fisher-z, `z(PCC(prediction, independent held-out target)) - z(PCC(prediction, strict foreign null target))`.\n- Secondary endpoints: raw same-target PCC, strict-foreign PCC, same-target minus support-mean ΔPCC, and raw excess PCC.\n- Target rows are first averaged within plate; support is the deterministic budget-sized plate mean and the next deterministic plate is independent held out.\n- Foreign nulls require the same nominal dose and the same full target plate signature, with a different compound; missing foreign matches remain missing and are reported.\n- Validation selects PCA rank and fusion beta. Test is evaluated once after selection.\n- Confidence intervals are paired compound bootstrap 95% percentile intervals; bootstrap unit is compound, with any repeated condition rows averaged within compound.\n- Seeds are frozen as 3407, 42, and 2025. Train/valid/test compounds are disjoint; no held-out target repeat is an input.\n- BBBC047 CP/GE float doses are matched at two decimal places to remove storage-rounding discrepancies; cpg0004-LINCS nominal dose strings are retained exactly.\n"""
    (root / "METRIC_PROTOCOL.md").write_text(metric_protocol, encoding="utf-8")
    matrix: list[dict[str, Any]] = []
    for setting in all_settings():
        matrix.append({"dataset": setting["dataset"], "target_modality": setting["target_modality"], "source_modality": setting["source_modality"], "budget": setting["budget"], "label": setting["label"], "status": "EXECUTABLE", "reason": "frozen four-part baseline-ablation matrix"})
    matrix.extend([
        {"dataset": "BBBC047", "target_modality": "GE", "source_modality": "CP", "budget": 3, "label": "3R", "status": "BLOCKED_PREREGISTERED", "reason": "not executable in the frozen matrix"},
        {"dataset": "cpg0004-LINCS", "target_modality": "GE", "source_modality": "CP", "budget": 1, "label": "1R", "status": "BLOCKED_PREREGISTERED", "reason": "target GE not in the frozen execution matrix"},
        {"dataset": "cpg0004-LINCS", "target_modality": "GE", "source_modality": "CP", "budget": 2, "label": "2R", "status": "BLOCKED_PREREGISTERED", "reason": "target GE not in the frozen execution matrix"},
        {"dataset": "cpg0004-LINCS", "target_modality": "GE", "source_modality": "CP", "budget": 3, "label": "3R", "status": "BLOCKED_PREREGISTERED", "reason": "target GE not in the frozen execution matrix"},
        {"dataset": "cross-environment", "target_modality": "CP/GE", "source_modality": "CP/GE", "budget": "all", "label": "all", "status": "NOT_EXECUTED", "reason": "feature spaces are not harmonized; out of scope"},
    ])
    write_csv(root / "FROZEN_EXPERIMENT_MATRIX.csv", matrix)
    hashes: dict[str, Any] = {}
    for label, path in (("cpg_split_lock", args.cpg_lock), ("common_source", Path(__file__)), ("common_module", Path(__file__).with_name("ablation_common.py"))):
        if path.exists():
            hashes[label] = {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
    data_fingerprints: list[dict[str, Any]] = []
    for dataset, bundle in bundles.items():
        for split in ("train", "valid", "test"):
            for modality in ("CP", "GE"):
                data = bundle.get(split, modality)
                data_fingerprints.append({"dataset": dataset, "modality": modality, "split": split, "n_rows": data.n_rows, "feature_dim": data.dim, "row_array_sha256": "not recomputed in freeze; source path/metadata are recorded in the run command"})
    write_json(root / "protocol_freeze.json", {"protocol_version": PROTOCOL_VERSION, "freeze_date": "2026-08-30", "seeds": list(SEEDS), "pca_candidates": list(PCA_CANDIDATES), "beta_grid": list(BETA_GRID), "data_rule": "prepared plate-row artifacts; train/valid/test molecule split reused", "hashes": hashes, "data_fingerprints": data_fingerprints, "execution_matrix": matrix})
    print(f"[done] phase0 wrote {root}", flush=True)


def read_summary_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in root.rglob("summary.csv"):
        if "00_protocol" in path.parts or any(x in path.parts for x in ("01_pca", "02_mlp", "03_late_fusion")):
            for row in read_csv(path):
                for key in ("budget", "seed", "n_pairs", "n_foreign", "excess_z_mean", "excess_z_ci_low", "excess_z_ci_high", "pcc_mean", "foreign_pcc_mean", "delta_pcc_mean", "excess_pcc_mean"):
                    if key in row and row[key] not in ("", None):
                        try:
                            row[key] = int(row[key]) if key in ("budget", "seed", "n_pairs", "n_foreign") else float(row[key])
                        except ValueError:
                            pass
                rows.append(row)
    return rows


def method_prediction_path(root: Path, method: str, setting: Mapping[str, Any], seed: int, split: str) -> Path | None:
    section = {"raw_mean": "00_protocol/core", "teacher_reproducibility": "00_protocol/core", "teacher_aggregate": "00_protocol/core", "pca": "01_pca", "mlp": "02_mlp", "direct_crossmodal": "03_late_fusion", "late_fusion": "03_late_fusion", "residual_fusion": "03_late_fusion"}.get(method)
    if section is None:
        return None
    path = setting_dir(root, section, setting, seed) / f"{split}_predictions.npz"
    return path if path.exists() else None


def method_key_for_npz(method: str) -> str:
    return {"raw_mean": "support", "teacher_reproducibility": "teacher_reproducibility", "teacher_aggregate": "teacher_aggregate"}.get(method, method)


def run_biology_review(args: argparse.Namespace, bundles: Mapping[str, DatasetBundle], trigger_rows: Sequence[Mapping[str, Any]]) -> None:
    """Run the conditional lightweight biology review when the trigger fires.

    The review is deliberately downstream of the statistical trigger.  It emits
    auditable top-25 morphology-style error/recovery tables and a six-dose
    condition coverage table from the frozen cpg CP test rows; it does not turn
    a statistical result into a mechanistic claim.
    """
    root = args.output_root / "05_biology_review"
    root.mkdir(parents=True, exist_ok=True)
    bundle = bundles["cpg0004-LINCS"]
    rows = bundle.get("test", "CP")
    group_means = rows.plate_means()
    candidate_rows: list[dict[str, Any]] = []
    # Aggregate a heldout error/recovery proxy across all target settings and
    # methods represented in the trigger list.  This is the requested review
    # layer, not an additional primary endpoint.
    for setting in [s for s in all_settings() if s["dataset"] == "cpg0004-LINCS" and s["target_modality"] == "CP"]:
        for seed in SEEDS:
            ex = make_examples(rows, None, int(setting["budget"]), seed, all_conditions=False, require_source=False)
            # Use target-only condition selection to cover all compounds; rows
            # with a source-dependent method are not silently substituted.
            for i in range(len(ex)):
                error = float(np.mean((ex.support[i] - ex.target[i]) ** 2))
                candidate_rows.append({"dataset": ex.dataset, "budget": setting["budget"], "seed": seed, "compound": ex.compound[i], "dose": ex.dose[i], "support_target_mse": error, "heldout_pcc_support": float(rowwise_corr(ex.support[i:i+1], ex.target[i:i+1])[0])})
    candidate_rows.sort(key=lambda row: (-row["support_target_mse"], row["compound"], row["dose"]))
    top25 = candidate_rows[:25]
    write_csv(root / "Morph_A_top25_high_error.csv", top25)
    candidate_rows.sort(key=lambda row: (row["support_target_mse"], row["compound"], row["dose"]))
    write_csv(root / "Morph_B_top25_low_error.csv", candidate_rows[:25])
    sixdose: list[dict[str, Any]] = []
    by_compound: dict[str, list[tuple[str, str]]] = {}
    for (compound, dose), plates in rows.plate_means().items():
        by_compound.setdefault(compound, []).append((dose, "|".join(sorted(plates))))
    for compound, conditions in sorted(by_compound.items()):
        sixdose.append({"compound": compound, "dose_count": len(conditions), "doses": "|".join(sorted(x[0] for x in conditions)), "complete_six_dose": int(len(conditions) == 6)})
    write_csv(root / "Morph_C_six_dose_coverage.csv", sixdose)
    write_csv(root / "trigger_source.csv", list(trigger_rows))
    write_json(root / "review_manifest.json", {"triggered": True, "review_type": "conditional morphology/error and six-dose coverage review", "top25_rule": "support-target MSE on cpg0004 CP test target-only condition pairs", "mechanistic_claim": False})


def run_final(args: argparse.Namespace, bundles: Mapping[str, DatasetBundle], settings: Sequence[Mapping[str, Any]]) -> None:
    root = args.output_root
    summary_rows = read_summary_rows(root)
    # De-duplicate because each phase writes one compact summary per setting.
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in summary_rows:
        key = (row.get("dataset"), row.get("target_modality"), row.get("budget"), row.get("seed"), row.get("split"), row.get("method"))
        unique[key] = row
    summary_rows = list(unique.values())
    matrix_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    pooled_differences: dict[tuple[str, str, int, str, str, str], list[tuple[str, float]]] = {}
    seeds = [int(args.seed)] if args.seed else list(SEEDS)
    methods = ("raw_mean", "teacher_reproducibility", "teacher_aggregate", "pca", "mlp", "direct_crossmodal", "late_fusion", "residual_fusion")
    for setting in settings:
        for seed in seeds:
            bundle = bundles[setting["dataset"]]
            examples = make_setting_examples(bundle, setting, seed)
            for split in ("valid", "test"):
                ex = examples[split]
                preds: dict[str, np.ndarray] = {}
                for method in methods:
                    path = method_prediction_path(root, method, setting, seed, split)
                    if path is None:
                        continue
                    z = read_npz(path)
                    key = method_key_for_npz(method)
                    if key in z and len(z[key]) == len(ex):
                        preds[method] = z[key]
                if "teacher_reproducibility" in preds:
                    teacher_metrics = metric_arrays(preds["teacher_reproducibility"], ex)
                    for method, pred in preds.items():
                        if method == "teacher_reproducibility":
                            continue
                        m = metric_arrays(pred, ex)
                        for metric in ("excess_z", "pcc", "delta_pcc", "excess_pcc"):
                            pooled_key = (str(ex.dataset), str(ex.modality), int(ex.budget), str(split), str(method), str(metric))
                            pooled_differences.setdefault(pooled_key, []).extend((str(compound), float(left - right)) for compound, left, right in zip(ex.compound, m[metric], teacher_metrics[metric]) if np.isfinite(left) and np.isfinite(right))
                            point, low, high, n = bootstrap_difference(m[metric], teacher_metrics[metric], ex.compound, seed + stable_int(seed, f"final|{method}|{split}|{metric}"), int(args.bootstrap_n or (100 if args.smoke else BOOTSTRAP_N)))
                            paired_rows.append({"dataset": ex.dataset, "target_modality": ex.modality, "budget": ex.budget, "seed": seed, "split": split, "method": method, "reference": "teacher_reproducibility", "metric": metric, "mean_difference": point, "ci_low": low, "ci_high": high, "n_compounds": n})
                    if ex.dataset == "cpg0004-LINCS" and ex.modality == "CP" and split == "test":
                        for method in ("pca", "mlp", "teacher_aggregate"):
                            if method not in preds:
                                continue
                            m = metric_arrays(preds[method], ex)
                            point, low, high, n = bootstrap_difference(m["excess_z"], teacher_metrics["excess_z"], ex.compound, seed + stable_int(seed, f"trigger|{method}"), int(args.bootstrap_n or (100 if args.smoke else BOOTSTRAP_N)))
                            raw = metric_arrays(preds["raw_mean"], ex) if "raw_mean" in preds else None
                            if raw is not None:
                                bp, bl, bh, bn = bootstrap_difference(raw["excess_z"], teacher_metrics["excess_z"], ex.compound, seed + stable_int(seed, f"trigger-baseline|{method}"), int(args.bootstrap_n or (100 if args.smoke else BOOTSTRAP_N)))
                            else:
                                bp = bl = bh = float("nan"); bn = 0
                            trigger_rows.append({"dataset": ex.dataset, "target_modality": ex.modality, "budget": ex.budget, "seed": seed, "method": method, "method_minus_teacher_excess_z": point, "method_minus_teacher_ci_low": low, "method_minus_teacher_ci_high": high, "baseline_minus_teacher_excess_z": bp, "baseline_minus_teacher_ci_low": bl, "baseline_minus_teacher_ci_high": bh, "trigger_ci_crosses_zero": int(np.isfinite(low) and low <= 0 <= high), "trigger_baseline_above_teacher": int(np.isfinite(bl) and bl > 0)})
            for row in summary_rows:
                if row.get("dataset") == setting["dataset"] and row.get("target_modality") == setting["target_modality"] and int(row.get("budget", -1)) == int(setting["budget"]) and int(row.get("seed", -1)) == seed:
                    matrix_rows.append(row)
    # Pool seed-specific paired differences at the compound level.  A compound
    # contributes the mean of its available seed-specific differences before
    # the bootstrap, so seed choice cannot create pseudo-replicates.
    for (dataset, modality, budget, split, method, metric), observations in sorted(pooled_differences.items()):
        grouped: dict[str, list[float]] = {}
        for compound, value in observations:
            grouped.setdefault(compound, []).append(value)
        pooled_compounds = np.asarray(sorted(grouped), dtype=str)
        pooled_values = np.asarray([np.mean(grouped[x]) for x in pooled_compounds], dtype=float)
        zeros = np.zeros_like(pooled_values)
        point, low, high, n = bootstrap_difference(pooled_values, zeros, pooled_compounds, stable_int(SEEDS[0], f"pooled|{dataset}|{modality}|{budget}|{split}|{method}|{metric}"), int(args.bootstrap_n or (100 if args.smoke else BOOTSTRAP_N)))
        paired_rows.append({"dataset": dataset, "target_modality": modality, "budget": budget, "seed": "pooled", "split": split, "method": method, "reference": "teacher_reproducibility", "metric": metric, "mean_difference": point, "ci_low": low, "ci_high": high, "n_compounds": n})
    write_csv(root / "FINAL_BASELINE_MATRIX.csv", matrix_rows)
    write_csv(root / "FINAL_PAIRED_RESULTS.csv", paired_rows)
    figure_rows = []
    for row in matrix_rows:
        figure_rows.append({"dataset": row.get("dataset"), "target_modality": row.get("target_modality"), "budget": row.get("budget"), "seed": row.get("seed"), "split": row.get("split"), "method": row.get("method"), "metric": "excess_z", "mean": row.get("excess_z_mean"), "ci_low": row.get("excess_z_ci_low"), "ci_high": row.get("excess_z_ci_high"), "n_pairs": row.get("n_pairs"), "n_foreign": row.get("n_foreign")})
        figure_rows.append({"dataset": row.get("dataset"), "target_modality": row.get("target_modality"), "budget": row.get("budget"), "seed": row.get("seed"), "split": row.get("split"), "method": row.get("method"), "metric": "pcc", "mean": row.get("pcc_mean"), "ci_low": row.get("pcc_ci_low"), "ci_high": row.get("pcc_ci_high"), "n_pairs": row.get("n_pairs"), "n_foreign": row.get("n_foreign")})
    write_csv(root / "FINAL_FIGURE_SOURCE.csv", figure_rows)
    paired_test = [r for r in paired_rows if r["split"] == "test" and r["metric"] == "excess_z"]
    improvements = [r for r in paired_test if r["method"] not in ("raw_mean",) and np.isfinite(float(r["ci_low"])) and float(r["ci_low"]) > 0]
    reversals = [r for r in paired_test if r["method"] == "raw_mean" and np.isfinite(float(r["ci_low"])) and float(r["ci_low"]) > 0]
    trigger = [r for r in trigger_rows if int(r["trigger_ci_crosses_zero"]) or int(r["trigger_baseline_above_teacher"])]
    if trigger:
        status = "STORY-REVISION-REQUIRED"
    elif improvements:
        status = "GO"
    else:
        status = "PARTIAL"
    if trigger:
        run_biology_review(args, bundles, trigger)
    q1 = "Across the executable settings, PCA/MLP/aggregate-target improvements over the reproducibility teacher are mixed; see paired test CIs in FINAL_PAIRED_RESULTS.csv." if not improvements else f"At least one preregistered method improves the reproducibility teacher on the primary endpoint with a paired test CI excluding zero ({len(improvements)} comparison rows)."
    q2 = "Late fusion and residual fusion were kept as separate methods because the algebraic audit is FALSE; validation-only beta selection and common test pairs are recorded per setting." 
    q3 = "The aggregate-target objective uses the identical TeacherNet/optimizer/input/seed protocol; only the training label changes, and the direct paired contrast is reported." 
    q4 = "The conditional cpg0004 CP biology-review trigger fired; the downstream Morph-A/B/C review artifacts were generated." if trigger else "The conditional cpg0004 CP biology-review trigger did not fire; no mechanistic review was added." 
    markdown = f"""# Final baseline-ablation summary\n\nProtocol: `{PROTOCOL_VERSION}`.  Executable settings: {len(settings)}; seeds: `{', '.join(str(x) for x in seeds)}`.\n\n## Q1 — PCA / MLP / objective ablation\n\n{q1}\n\n## Q2 — late fusion\n\n{q2}\n\n## Q3 — supervision objective\n\n{q3}\n\n## Q4 — conditional biology review\n\n{q4}\n\n## Decision\n\n`{status}`\n\nAll negative, null, and reversal results remain in the final matrix and paired-result tables.  The primary endpoint is held-out excess Fisher-z; raw PCC and ΔPCC are secondary.\n"""
    (root / "FINAL_BASELINE_ABLATION_SUMMARY.md").write_text(markdown, encoding="utf-8")
    write_csv(root / "biology_trigger.csv", trigger_rows)
    write_json(root / "final_status.json", {"status": status, "n_settings": len(settings), "n_paired_rows": len(paired_rows), "biology_triggered": bool(trigger), "improvement_rows": len(improvements), "reversal_rows": len(reversals)})
    print(f"[done] final status={status}; outputs={root}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("phase0", "core", "pca", "mlp", "late", "objective", "final", "all"), required=True)
    parser.add_argument("--dataset", choices=("BBBC047", "cpg0004-LINCS"))
    parser.add_argument("--modality", choices=("CP", "GE"))
    parser.add_argument("--budget", type=int, choices=(1, 2, 3))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bbbc-root", type=Path, default=DEFAULT_BBBC_ROOT)
    parser.add_argument("--cpg-cp", type=Path, default=DEFAULT_CPG_CP)
    parser.add_argument("--cpg-ge", type=Path, default=DEFAULT_CPG_GE)
    parser.add_argument("--cpg-lock", type=Path, default=DEFAULT_CPG_LOCK)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--bootstrap-n", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    settings = selected_settings(args)
    if args.phase == "phase0":
        settings_for_data = all_settings()
    else:
        settings_for_data = settings
    if args.phase == "final" and not settings:
        settings = all_settings()
        settings_for_data = settings
    if not settings_for_data:
        raise SystemExit("no settings selected")
    bundles = load_for_settings(args, settings_for_data)
    if args.phase == "phase0":
        run_phase0(args, bundles)
        return
    if args.phase == "core":
        run_core(args, bundles, settings)
    elif args.phase == "pca":
        run_pca(args, bundles, settings)
    elif args.phase == "mlp":
        run_mlp(args, bundles, settings)
    elif args.phase == "late":
        run_late(args, bundles, settings)
    elif args.phase == "objective":
        run_objective(args, bundles, settings)
    elif args.phase == "final":
        run_final(args, bundles, settings)
    elif args.phase == "all":
        run_phase0(args, bundles)
        run_core(args, bundles, settings)
        run_pca(args, bundles, settings)
        run_mlp(args, bundles, settings)
        run_late(args, bundles, settings)
        run_objective(args, bundles, settings)
        run_final(args, bundles, settings)


if __name__ == "__main__":
    main()
