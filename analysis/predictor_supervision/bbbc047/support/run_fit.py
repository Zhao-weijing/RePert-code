#!/usr/bin/env python3
"""Fit-only runner for the locked CFRA-decomposition confirmation.

This phase intentionally has no test/confirmation code path.  It reads only
the Fit and Validation compound rows named by ``COMPOUND_SPLIT.csv`` from the
prepare artifact, fits all five frozen comparison models, and freezes the
validation-selected branch checkpoints.  Confirmation rows, confirmation
profiles, and the full analysis manifest are never opened here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = Path(__file__).resolve().parents[4] / "analysis"
SOURCE_PATH = (Path(__file__).resolve().parents[4] / "analysis/predictor_supervision/bbbc047/support/run_rank_matched_pca_control.py")
if not SOURCE_PATH.is_file():
    raise FileNotFoundError(SOURCE_PATH)
_spec = importlib.util.spec_from_file_location("bbbc047_decomp_fit_source", SOURCE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"unable to import {SOURCE_PATH}")
SRC = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = SRC
_spec.loader.exec_module(SRC)
LEGACY = SRC.LEGACY


VERSION = "BBBC047-CFRA-Decomposition-confirmation-v2-2026-09-13"
PROTOCOL_SEED = 3407
BUDGETS = (1, 2, 3)
MASTER_SEEDS = (3407, 42, 2025, 1337, 7331)
SMOKE_SEEDS = (3407,)
CP_DIM = 775
FP_DIM = 2048
RANKS = {1: 8, 2: 8, 3: 9}
BRANCHES = {
    "M0_raw": ("raw",),
    "M1_cfra": ("cfra",),
    "M2_raw_2x": ("raw_a", "raw_b"),
    "M3_pca_decomp": ("pca", "pca_residual"),
    "M4_cfra_decomp": ("cfra", "residual"),
}
METHODS = tuple(BRANCHES)

DEFAULT_ROWS = Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0")
DEFAULT_H5 = DEFAULT_ROWS.parent / "Paired_CP_GE_PlateMedian_v1_model_compat.h5"
DEFAULT_AGG = DEFAULT_ROWS / "molecule_aggregates.npz"


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_hash(path: Path) -> str:
    return sha256_file(path)


def device_for(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but CUDA is unavailable")
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit",), default="fit")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--prepare-root", type=Path)
    parser.add_argument("--rows-root", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--model-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--aggregate-npz", type=Path, default=DEFAULT_AGG)
    parser.add_argument("--protocol-file", type=Path, default=HERE / "PROTOCOL.md")
    parser.add_argument("--teacher-epochs", type=int, default=80)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="one-epoch relaxed fit; non-scientific")
    parser.add_argument("--synthetic-smoke", action="store_true", help="run a contract-only synthetic fit")
    args = parser.parse_args()
    relaxed = args.smoke or args.synthetic_smoke
    if not relaxed and (args.teacher_epochs, args.student_epochs, args.batch_size) != (80, 40, 256):
        parser.error("formal protocol locks teacher/student epochs and batch size to 80/40/256")
    if not relaxed and args.bootstrap_rounds != 10000:
        parser.error("formal protocol locks bootstrap-rounds to 10000")
    if args.bootstrap_rounds < 1:
        parser.error("bootstrap-rounds must be positive")
    if not args.synthetic_smoke and args.output_root is None:
        parser.error("--output-root is required for formal/smoke fit")
    if not args.synthetic_smoke and args.prepare_root is None:
        parser.error("--prepare-root is required for formal/smoke fit")
    return args


def canonical_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray([value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values], dtype=str)


def _rows_from_arrays(dataset: str, compound: np.ndarray, dose: np.ndarray, plate: np.ndarray, delta: np.ndarray) -> Any:
    compound = canonical_strings(compound)
    dose = canonical_strings(dose)
    plate = canonical_strings(plate)
    delta = np.asarray(delta, dtype=np.float32)
    if delta.ndim != 2 or delta.shape[1] != CP_DIM or not np.isfinite(delta).all():
        raise RuntimeError(f"invalid CP row array: {delta.shape}")
    return LEGACY.Rows("BBBC047", "CP", dataset, compound, dose, plate, delta)


def read_fit_valid_ids(prepare_root: Path) -> dict[str, set[str]]:
    """Read only Fit/Validation IDs; never parse the confirmation rows."""
    path = prepare_root / "COMPOUND_SPLIT.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {"fit": set(), "validation": set()}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = str(row.get("split", ""))
            if split in result:
                compound = str(row.get("compound_id", ""))
                if not compound:
                    raise RuntimeError("empty Fit/Validation compound ID in COMPOUND_SPLIT.csv")
                result[split].add(compound)
    if not result["fit"] or not result["validation"] or result["fit"] & result["validation"]:
        raise RuntimeError("invalid or overlapping Fit/Validation ID sets")
    return result


def require_prepare(args: argparse.Namespace) -> dict[str, Any]:
    assert args.prepare_root is not None
    marker_path = args.prepare_root / "PREPARE_COMPLETE.json"
    marker = read_json(marker_path)
    if marker.get("version") != VERSION or marker.get("phase") != "prepare" or marker.get("status") != "PASS":
        raise RuntimeError("prepare marker version/phase/status is invalid")
    for key in ("confirmation_used_for_training", "confirmation_used_for_selection", "confirmation_model_scores_computed_in_prepare"):
        if marker.get(key) is not False:
            raise RuntimeError(f"prepare marker violates the confirmation contract at {key}")
    if marker.get("sealing_contract") != "fit stage receives fit/validation only; confirmation is opened only after FIT_COMPLETE and SELECTION_FREEZE verification":
        raise RuntimeError("prepare marker lacks the frozen information-barrier contract")
    manifest = args.prepare_root / "COMPOUND_SPLIT.csv"
    expected = marker.get("split_sha256")
    if expected and sha256_file(manifest) != expected:
        raise RuntimeError("COMPOUND_SPLIT.csv hash changed after prepare")
    # The large ANALYSIS_MANIFEST.csv contains confirmation role values.  Its
    # presence is checked only by prepare; fit deliberately does not open it.
    ids = read_fit_valid_ids(args.prepare_root)
    return {"marker": marker, "ids": ids, "split_sha256": sha256_file(manifest)}


def load_filtered_rows(prepare_root: Path, ids: Mapping[str, set[str]], marker: Mapping[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    """Load only sealed Fit/Validation artifacts; never open source files here."""
    output: dict[str, Any] = {}
    hashes = marker.get("sealed_fit_validation_rows", {})
    for split in ("fit", "validation"):
        path = prepare_root / f"{split.upper()}_ROWS.npz"
        if not path.is_file() or sha256_file(path) != hashes.get(split):
            raise RuntimeError(f"sealed {split} row artifact missing or changed")
        with np.load(path, allow_pickle=False) as payload:
            output[split] = _rows_from_arrays(split, payload["smiles"], payload["dose"], payload["plate"], payload["delta"])
        observed = set(output[split].compound.tolist())
        if observed != set(ids[split]):
            raise RuntimeError(f"{split} rows do not exactly cover frozen IDs: {len(observed)} != {len(ids[split])}")
    if set(output["fit"].compound.tolist()) & set(output["validation"].compound.tolist()):
        raise RuntimeError("Fit/Validation compound leakage")
    return output


def load_virtual_filtered(model_h5: Path, aggregate_npz: Path, ids: Mapping[str, set[str]]) -> dict[str, Any]:
    """Read predictors for Fit/Validation compounds only.

    The official test split and the locked confirmation IDs are never indexed.
    Aggregate files are used only to verify the selected Fit/Validation deltas.
    """
    if not model_h5.is_file() or not aggregate_npz.is_file():
        raise FileNotFoundError("model H5 and aggregate NPZ are required")
    output: dict[str, Any] = {}
    with h5py.File(model_h5, "r") as handle:
        all_names = canonical_strings(handle["canonical_smiles"][:])
        if len(all_names) != len(set(all_names.tolist())):
            raise RuntimeError("duplicate H5 canonical_smiles")
        index = {name: i for i, name in enumerate(all_names.tolist())}
        with np.load(aggregate_npz, allow_pickle=False) as aggregate:
            aggregate_names: dict[str, np.ndarray] = {}
            aggregate_values: dict[str, np.ndarray] = {}
            for outer in ("train", "valid"):
                if f"{outer}_smiles" in aggregate.files and f"{outer}_cp" in aggregate.files:
                    aggregate_names[outer] = canonical_strings(aggregate[f"{outer}_smiles"])
                    aggregate_values[outer] = np.asarray(aggregate[f"{outer}_cp"], dtype=np.float32)
            for split, allowed in ids.items():
                names = np.asarray(sorted(allowed), dtype=str)
                missing = sorted(set(names.tolist()) - set(index))
                if missing:
                    raise RuntimeError(f"{split} compound missing from H5: {missing[:3]}")
                rows = np.asarray([index[name] for name in names], dtype=np.int64)
                order = np.argsort(rows)
                inverse = np.argsort(order)
                sorted_rows = rows[order]
                control = np.asarray(handle["control_CP"][sorted_rows], dtype=np.float32)[inverse]
                target = np.asarray(handle["target_CP"][sorted_rows], dtype=np.float32)[inverse]
                delta = target - control
                # The aggregate verification is restricted to the selected IDs.
                aggregate_delta = None
                for outer in ("train", "valid"):
                    if outer in aggregate_names and set(names.tolist()).issubset(set(aggregate_names[outer].tolist())):
                        amap = {name: i for i, name in enumerate(aggregate_names[outer].tolist())}
                        aggregate_delta = aggregate_values[outer][np.asarray([amap[name] for name in names], dtype=np.int64)]
                        break
                if aggregate_delta is not None and (delta.shape != aggregate_delta.shape or float(np.max(np.abs(delta - aggregate_delta))) > 1e-5):
                    raise RuntimeError(f"aggregate/H5 delta mismatch for {split}")
                if delta.shape[1] != CP_DIM or not np.isfinite(control).all() or not np.isfinite(delta).all():
                    raise RuntimeError(f"invalid predictor arrays for {split}")
                output[split] = LEGACY.PredictorSplit(
                    compound=names,
                    dose=np.asarray(["" for _ in names], dtype=str),
                    control=control,
                    fingerprint=LEGACY.VIRTUAL.fingerprints(names.tolist()),
                    target=delta,
                )
    return output


def subset_virtual(base: Any, examples: Any, target: np.ndarray) -> Any:
    index = {str(value): i for i, value in enumerate(base.compound)}
    rows = np.asarray([index[str(x)] for x in examples.compound], dtype=np.int64)
    if len(rows) != len(set(examples.compound.tolist())):
        raise RuntimeError("duplicate predictor row for compound")
    return LEGACY.PredictorSplit(
        compound=examples.compound.copy(), dose=examples.dose.copy(),
        control=base.control[rows].astype(np.float32), fingerprint=base.fingerprint[rows].astype(np.float32),
        target=np.asarray(target, dtype=np.float32),
    )


def branch_seed(master: int, budget: int, branch: str) -> int:
    return int(LEGACY.stable_int(master, f"{VERSION}|fit|budget{budget}|{branch}") % (2**32 - 1))


def branch_stats(train: Any, target: np.ndarray) -> dict[str, np.ndarray]:
    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 2 or target.shape[1] != CP_DIM or not np.isfinite(target).all():
        raise RuntimeError(f"invalid branch target shape/value: {target.shape}")
    return {
        "control_mean": np.asarray(train.control.mean(axis=0, dtype=np.float64), dtype=np.float32),
        "control_scale": np.maximum(np.asarray(train.control.std(axis=0, dtype=np.float64), dtype=np.float32), 1e-6),
        "target_mean": np.asarray(target.mean(axis=0, dtype=np.float64), dtype=np.float32),
        "target_scale": np.maximum(np.asarray(target.std(axis=0, dtype=np.float64), dtype=np.float32), 1e-6),
    }


def predict_student(model: Any, split: Any, stats: Mapping[str, np.ndarray], batch: int, device: torch.device) -> np.ndarray:
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
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite Student prediction")
    return output


def fit_branch(
    train: Any,
    valid: Any,
    train_target: np.ndarray,
    valid_objective: np.ndarray,
    stats: Mapping[str, np.ndarray],
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    method: str,
    branch: str,
) -> tuple[Any, dict[str, np.ndarray], list[dict[str, Any]], np.ndarray]:
    """Train one frozen architecture and select checkpoint on Validation.

    Training uses branch-normalized targets. Checkpoint selection is made in
    original CP units against the branch-specific frozen Validation target.
    """
    LEGACY.set_seed(seed)
    train_control = (train.control - stats["control_mean"]) / stats["control_scale"]
    train_z = (np.asarray(train_target, dtype=np.float32) - stats["target_mean"]) / stats["target_scale"]
    valid_objective = np.asarray(valid_objective, dtype=np.float32)
    model = LEGACY.Student().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(train_control), torch.from_numpy(train.fingerprint), torch.from_numpy(train_z)),
        batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
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
                raise FloatingPointError(f"non-finite loss in {method}/{branch}")
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(control)
            train_n += len(control)
        valid_pred = predict_student(model, valid, stats, args.batch_size, device)
        valid_loss = float(np.mean((valid_pred.astype(np.float64) - valid_objective.astype(np.float64)) ** 2))
        logs.append({
            "method": method, "branch": branch, "seed": seed, "epoch": epoch,
            "train_mse_normalized": train_sum / max(train_n, 1),
            "validation_mse_original": valid_loss,
        })
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError(f"no checkpoint state for {method}/{branch}")
    model.load_state_dict(best_state)
    best_pred = predict_student(model, valid, stats, args.batch_size, device)
    return model, dict(stats), logs, best_pred


def build_validation_cfra(rows: dict[str, Any], examples: Any, budget: int, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit validation CFRA teacher on Fit compounds only."""
    full_pairs = LEGACY.budget_pairs(rows["train"], budget)
    valid_pairs = LEGACY.budget_pairs(rows["validation"], budget)
    teacher_seed = LEGACY.stable_int(PROTOCOL_SEED, f"{VERSION}|cfra-valid-objective|b{budget}") % (2**32 - 1)
    teacher_args = SimpleNamespace(
        seed=teacher_seed, hidden_dim=256, latent_dim=32, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        epochs=args.teacher_epochs, smoke=args.smoke,
    )
    model, mean, scale, best_epoch, valid_mse = LEGACY.EFFECT.fit_encoder(full_pairs, valid_pairs, teacher_args, device)
    prediction = LEGACY.predict_effect(model, examples.support, mean, scale, args.batch_size, device).astype(np.float32)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"non-finite validation CFRA teacher target at budget {budget}")
    return prediction, {
        "teacher_seed": int(teacher_seed), "fit_split": "fit_compounds_only",
        "selection_split": "validation_physical_pairs", "best_epoch": int(best_epoch),
        "validation_mse": float(valid_mse),
    }


def pca_reconstruct(raw_target: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    values = np.asarray(raw_target, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != CP_DIM:
        raise RuntimeError(f"invalid PCA target shape: {values.shape}")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = (centered.T @ centered) / float(max(len(values) - 1, 1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    components = eigenvectors[:, :rank].astype(np.float32)
    reconstruction = mean + (centered @ components.astype(np.float64)) @ components.astype(np.float64).T
    total = float(eigenvalues.sum())
    retained = float(eigenvalues[:rank].sum() / total) if total > 0 else 1.0
    return reconstruction.astype(np.float32), mean.astype(np.float32), components, eigenvalues.astype(np.float32), retained


def target_identity(raw: np.ndarray, cfra: np.ndarray) -> tuple[np.ndarray, float]:
    raw = np.asarray(raw, dtype=np.float32)
    cfra = np.asarray(cfra, dtype=np.float32)
    residual = np.asarray(raw - cfra, dtype=np.float32)
    error = float(np.max(np.abs(raw - cfra - residual)))
    return residual, error


def save_npz(path: Path, **values: Any) -> str:
    np.savez_compressed(path, **values)
    return sha256_file(path)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def method_prediction(method: str, branch_predictions: Mapping[str, np.ndarray]) -> np.ndarray:
    branches = BRANCHES[method]
    if method == "M2_raw_2x":
        return ((branch_predictions[branches[0]] + branch_predictions[branches[1]]) / np.float32(2.0)).astype(np.float32)
    return np.sum(np.stack([branch_predictions[name] for name in branches], axis=0), axis=0, dtype=np.float32)


def validate_prediction_identity(raw: np.ndarray, cfra: np.ndarray, residual: np.ndarray) -> float:
    error = float(np.max(np.abs(np.asarray(raw, dtype=np.float32) - np.asarray(cfra, dtype=np.float32) - np.asarray(residual, dtype=np.float32))))
    if error >= 1e-8:
        raise RuntimeError(f"Raw=CFRA+D identity failed: maxerr={error}")
    return error


def fit_core(
    args: argparse.Namespace,
    root: Path,
    prepare: dict[str, Any] | None,
    rows: dict[str, Any],
    virtual: dict[str, Any],
    *,
    synthetic: bool = False,
) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing fit root: {root}")
    root.mkdir(parents=True)
    budgets = BUDGETS
    seeds = SMOKE_SEEDS if args.smoke or synthetic else MASTER_SEEDS
    bundles = {budget: {split: SRC.build_dual_examples(rows[split], budget, include_foreign=False) for split in rows} for budget in budgets}
    device = device_for(args.device)
    logs: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    target_hashes: dict[str, Any] = {}
    pca_hashes: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}
    branch_seed_ledger: dict[str, Any] = {}
    identity_errors: dict[str, float] = {}
    all_branch_valid_predictions: dict[tuple[int, int, str, str], np.ndarray] = {}

    for budget in budgets:
        fit_examples = bundles[budget]["fit"]
        valid_examples = bundles[budget]["validation"]
        fit_raw = np.asarray(fit_examples.support, dtype=np.float32)
        valid_raw = np.asarray((valid_examples.held_a + valid_examples.held_b) / np.float32(2.0), dtype=np.float32)
        cfra_fit, teacher_meta, assignment = LEGACY.build_oof_cfra(
            {"train": rows["fit"], "valid": rows["validation"]}, fit_examples, budget, args, device
        )
        cfra_fit = np.asarray(cfra_fit, dtype=np.float32)
        fit_residual, identity_error = target_identity(fit_raw, cfra_fit)
        if identity_error >= 1e-8:
            raise RuntimeError(f"Raw=CFRA+D identity failed at budget {budget}: {identity_error}")
        identity_errors[str(budget)] = identity_error
        valid_cfra, valid_cfra_meta = build_validation_cfra(
            {"train": rows["fit"], "validation": rows["validation"]}, valid_examples, budget, args, device
        )
        valid_residual = np.asarray(valid_raw - valid_cfra, dtype=np.float32)
        rank = int(RANKS[budget])
        fit_pca, pca_mean, pca_components, pca_eigenvalues, retained = pca_reconstruct(fit_raw, rank)
        valid_centered = valid_raw.astype(np.float64) - pca_mean.astype(np.float64)
        valid_pca = (pca_mean.astype(np.float64) + (valid_centered @ pca_components.astype(np.float64)) @ pca_components.astype(np.float64).T).astype(np.float32)
        fit_pca_residual = np.asarray(fit_raw - fit_pca, dtype=np.float32)
        valid_pca_residual = np.asarray(valid_raw - valid_pca, dtype=np.float32)
        if not all(np.isfinite(x).all() for x in (fit_raw, cfra_fit, fit_residual, valid_raw, valid_cfra, valid_residual, fit_pca, fit_pca_residual, valid_pca, valid_pca_residual)):
            raise RuntimeError(f"non-finite target at budget {budget}")
        budget_dir = root / f"budget{budget}"
        budget_dir.mkdir(parents=True, exist_ok=False)
        target_path = budget_dir / "training_targets.npz"
        target_sha = save_npz(
            target_path, compound=fit_examples.compound, dose=fit_examples.dose,
            raw=fit_raw, cfra=cfra_fit, residual=fit_residual, pca=fit_pca, pca_residual=fit_pca_residual,
        )
        valid_path = budget_dir / "validation_targets.npz"
        valid_sha = save_npz(
            valid_path, compound=valid_examples.compound, dose=valid_examples.dose,
            raw_held_mean=valid_raw, cfra_teacher=valid_cfra, residual=valid_residual,
            pca=valid_pca, pca_residual=valid_pca_residual,
        )
        pca_path = budget_dir / "pca_model.npz"
        pca_sha = save_npz(pca_path, mean=pca_mean, components=pca_components, eigenvalues=pca_eigenvalues, rank=np.asarray(rank), retained_variance=np.asarray(retained))
        pca_hashes[str(budget)] = pca_sha
        target_hashes[str(budget)] = {
            "training_path": str(target_path), "training_sha256": target_sha,
            "validation_path": str(valid_path), "validation_sha256": valid_sha,
            "n_fit": len(fit_examples), "n_validation": len(valid_examples),
            "rank": rank, "pca_sha256": pca_sha, "pca_retained_variance": retained,
            "teacher": teacher_meta, "validation_teacher": valid_cfra_meta,
            "cfra_fold_assignment": assignment, "raw_cfra_residual_max_abs_error": identity_error,
            "confirmation_loaded": False,
        }
        json_dump(budget_dir / "CFRA_TEACHER_METADATA.json", {"teacher": teacher_meta, "validation_teacher": valid_cfra_meta, "fold_assignment": assignment})
        labels = {
            ("M0_raw", "raw"): (fit_raw, valid_raw),
            ("M1_cfra", "cfra"): (cfra_fit, valid_cfra),
            ("M2_raw_2x", "raw_a"): (fit_raw, valid_raw),
            ("M2_raw_2x", "raw_b"): (fit_raw, valid_raw),
            ("M3_pca_decomp", "pca"): (fit_pca, valid_pca),
            ("M3_pca_decomp", "pca_residual"): (fit_pca_residual, valid_pca_residual),
            ("M4_cfra_decomp", "cfra"): (cfra_fit, valid_cfra),
            ("M4_cfra_decomp", "residual"): (fit_residual, valid_residual),
        }
        branch_predictions_by_seed: dict[int, dict[str, dict[str, np.ndarray]]] = {}
        for master in seeds:
            branch_predictions_by_seed[master] = {}
            for method, branches in BRANCHES.items():
                branch_predictions_by_seed[master][method] = {}
                for branch in branches:
                    train_label, valid_label = labels[(method, branch)]
                    subseed = branch_seed(master, budget, branch)
                    branch_seed_ledger[f"budget{budget}/{method}/{branch}/master{master}"] = {
                        "master_seed": master, "subseed": subseed,
                    }
                    train_split = subset_virtual(virtual["fit"], fit_examples, train_label)
                    # All branch checkpoints are selected only on independent
                    # validation held-repeat means in original CP space.
                    valid_split = subset_virtual(virtual["validation"], valid_examples, valid_label)
                    stats = branch_stats(train_split, train_label)
                    model, fitted_stats, branch_logs, valid_prediction = fit_branch(
                        train_split, valid_split, train_label, valid_label, stats, subseed, args, device, method, branch
                    )
                    logs.extend({**row, "budget": budget, "master_seed": master, "subseed": subseed} for row in branch_logs)
                    best_row = min(branch_logs, key=lambda row: float(row["validation_mse_original"]))
                    ckpt_dir = budget_dir / method
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = ckpt_dir / f"{branch}_master{master}.pt"
                    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                    torch.save({
                        "version": VERSION, "budget": budget, "method": method, "branch": branch,
                        "master_seed": master, "subseed": subseed,
                        "architecture": "baseline_CP775+ECFP4-2048->512->128->CP775",
                        "target_definition": {"M0_raw": "Raw support target", "M1_cfra": "Fit-compound-cross-fitted CFRA target", "M2_raw_2x": "Raw support target", "M3_pca_decomp": "Fit-only PCA decomposition", "M4_cfra_decomp": "CFRA + Raw-minus-CFRA"}[method],
                        "normalization_scope": "Fit-only branch-specific",
                        "selection_rule": "Validation branch-target original-space MSE",
                        "best_epoch": int(best_row["epoch"]), "validation_mse_original": float(best_row["validation_mse_original"]),
                        "state_dict": state, "stats": {key: np.asarray(value, dtype=np.float32) for key, value in fitted_stats.items()},
                    }, ckpt_path)
                    key = f"budget{budget}/{method}/{branch}/master{master}"
                    checkpoint_hashes[key] = sha256_file(ckpt_path)
                    branch_predictions_by_seed[master][method][branch] = valid_prediction
                    all_branch_valid_predictions[(budget, master, method, branch)] = valid_prediction
                    validation_rows.append({
                        "budget": budget, "method": method, "branch": branch, "master_seed": master,
                        "subseed": subseed, "n_compounds": len(valid_examples), "best_epoch": int(best_row["epoch"]),
                        "validation_mse_original": float(best_row["validation_mse_original"]),
                        "validation_prediction_pcc_mean": float(np.nanmean(LEGACY.rowwise_corr(valid_prediction, valid_raw))),
                    })
            for method in METHODS:
                combined = method_prediction(method, branch_predictions_by_seed[master][method])
                validation_rows.append({
                    "budget": budget, "method": method, "branch": "combined", "master_seed": master,
                    "subseed": "", "n_compounds": len(valid_examples), "best_epoch": "",
                    "validation_mse_original": float(np.mean((combined.astype(np.float64) - valid_raw.astype(np.float64)) ** 2)),
                    "validation_prediction_pcc_mean": float(np.nanmean(LEGACY.rowwise_corr(combined, valid_raw))),
                })
                pred_path = budget_dir / f"validation_master{master}_{method}.npz"
                save_npz(pred_path, compound=valid_examples.compound, dose=valid_examples.dose, prediction=combined)
    write_csv(root / "TRAINING_LOG.csv", logs)
    write_csv(root / "VALIDATION_METRICS.csv", validation_rows)
    json_dump(root / "TARGET_HASHES.json", target_hashes)
    json_dump(root / "PCA_HASHES.json", pca_hashes)
    json_dump(root / "CHECKPOINT_HASHES.json", checkpoint_hashes)
    json_dump(root / "BRANCH_SEEDS.json", branch_seed_ledger)
    artifact_index = {
        f"budget{b}": {
            method: {
                f"seed{master}": ([
                    f"budget{b}/{method}/{BRANCHES[method][0]}_master{master}.pt"
                ] if len(BRANCHES[method]) == 1 else [
                    f"budget{b}/{method}/{branch}_master{master}.pt" for branch in BRANCHES[method]
                ]) for master in seeds
            } for method in METHODS
        } for b in BUDGETS
    }
    json_dump(root / "MODEL_ARTIFACTS.json", artifact_index)
    fit_audit = {
        "version": VERSION, "phase": "fit", "protocol_sha256": protocol_hash(args.protocol_file),
        "prepare_root": None if prepare is None else str(args.prepare_root),
        "prepare_marker_sha256": None if prepare is None else sha256_file(args.prepare_root / "PREPARE_COMPLETE.json"),
        "split_sha256": None if prepare is None else prepare["split_sha256"],
        "budgets": list(BUDGETS), "methods": list(METHODS), "branches": BRANCHES,
        "master_seeds": list(seeds), "all_master_seeds": list(MASTER_SEEDS), "pca_ranks": RANKS, "alpha": 1.0,
        "teacher_epochs": args.teacher_epochs, "student_epochs": args.student_epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "device": str(device), "checkpoint_count": len(checkpoint_hashes), "target_hashes": target_hashes,
        "artifact_index_sha256": sha256_file(root / "MODEL_ARTIFACTS.json"),
        "pca_hashes": pca_hashes, "checkpoint_hashes": checkpoint_hashes, "branch_seed_hash": sha256_file(root / "BRANCH_SEEDS.json"),
        "raw_cfra_residual_max_abs_error": identity_errors,
        "confirmation_loaded": False, "confirmation_values_opened": False, "confirmation_used_for_selection": False,
        "test_loaded": False, "test_values_opened": False, "test_used_for_selection": False,
        "source_rows_used": "sealed FIT_ROWS.npz and VALIDATION_ROWS.npz only", "status": "PASS",
        "synthetic_smoke": bool(synthetic or args.smoke),
    }
    json_dump(root / "FIT_COMPLETE.json", fit_audit)
    selection = {
        "version": VERSION, "phase": "fit", "selection_frozen": True,
        "protocol_sha256": protocol_hash(args.protocol_file), "split_sha256": fit_audit["split_sha256"],
        "checkpoint_hashes_sha256": sha256_file(root / "CHECKPOINT_HASHES.json"), "pca_hashes_sha256": sha256_file(root / "PCA_HASHES.json"),
        "branch_seed_sha256": sha256_file(root / "BRANCH_SEEDS.json"), "artifact_index_sha256": sha256_file(root / "MODEL_ARTIFACTS.json"), "checkpoint_count": len(checkpoint_hashes),
        "methods": list(METHODS), "budgets": list(BUDGETS), "master_seeds": list(seeds), "pca_ranks": RANKS,
        "alpha": 1.0, "selection_rule": "validation branch-target original-space MSE",
        "confirmation_loaded": False, "confirmation_values_opened": False, "confirmation_used_for_selection": False,
        "test_loaded": False, "test_used_for_selection": False, "status": "PASS",
    }
    json_dump(root / "SELECTION_FREEZE.json", selection)
    return {"root": str(root), "checkpoint_count": len(checkpoint_hashes), "fit_audit": fit_audit}


def synthetic_smoke(args: argparse.Namespace) -> None:
    temp = tempfile.TemporaryDirectory(prefix="bbbc047_cfra_decomp_fit_smoke_")
    root = Path(temp.name) / "fit"
    args.smoke = True
    args.device = "cpu"
    source = SRC.synthetic_rows(("train", "valid"))
    rows = {"fit": source["train"], "validation": source["valid"]}
    virtual_raw = SRC.synthetic_virtual(source)
    virtual = {}
    for source_name, split_name in (("train", "fit"), ("valid", "validation")):
        value = virtual_raw[source_name]
        virtual[split_name] = LEGACY.PredictorSplit(
            compound=np.asarray(value.smiles, dtype=str),
            dose=np.asarray(["" for _ in value.smiles], dtype=str),
            control=np.asarray(value.control, dtype=np.float32),
            fingerprint=np.asarray(value.fingerprint, dtype=np.float32),
            target=np.zeros((len(value.smiles), CP_DIM), dtype=np.float32),
        )
    result = fit_core(args, root, None, rows, virtual, synthetic=True)
    expected = len(BUDGETS) * sum(len(branches) for branches in BRANCHES.values()) * len(SMOKE_SEEDS)
    if result["checkpoint_count"] != expected:
        raise RuntimeError(f"synthetic checkpoint count mismatch: {result['checkpoint_count']} != {expected}")
    audit = read_json(root / "FIT_COMPLETE.json")
    if audit.get("confirmation_loaded") is not False or audit.get("raw_cfra_residual_max_abs_error", {}).get("1", 1.0) >= 1e-8:
        raise RuntimeError("synthetic fit sealing/identity audit failed")
    print(json.dumps({"synthetic_smoke": "PASS", "checkpoint_count": expected, "architecture": "2823->512->128->775", "confirmation_loaded": False}, indent=2))
    temp.cleanup()


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        synthetic_smoke(args)
        return
    assert args.output_root is not None and args.prepare_root is not None
    prepare = require_prepare(args)
    if (args.output_root / "FIT_COMPLETE.json").exists() or (args.output_root / "SELECTION_FREEZE.json").exists():
        raise FileExistsError("fit output marker already exists; use a fresh versioned root")
    rows = load_filtered_rows(args.prepare_root, prepare["ids"], prepare["marker"], smoke=args.smoke)
    virtual = load_virtual_filtered(args.model_h5, args.aggregate_npz, prepare["ids"])
    result = fit_core(args, args.output_root, prepare, rows, virtual)
    print(json.dumps({"phase": "fit", "root": result["root"], "checkpoint_count": result["checkpoint_count"], "confirmation_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
