#!/usr/bin/env python3
"""Formal BBBC047 P0-conditioned post-perturbation-CP residual benchmark.

Stages are deliberately irreversible:
  prepare  builds train-only cross-fitted P0 predictions;
  fit      trains a P0+CP -> (GE-P0) mapper and selects beta on validation;
  gate     authorizes test only when all seeds select beta > 0;
  test     performs the locked correct/shuffled/foreign evaluation once;
  aggregate freezes the all-seed decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


THIS = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "expression_teacher"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "cell_painting_to_expression"))
import ge_teacher_experiment as T  # noqa: E402
import cp_to_ge_residual as C  # noqa: E402


VERSION = "MVCPert-BBBC047-P0-Conditioned-CP-Residual-2026-09-02"
SEEDS = T.SEEDS
BUDGET = 1
OOF_FOLDS = 5
BETA_GRID = C.BETA_GRID
CP_DIM = C.CP_DIM
GE_DIM = T.FEATURE_DIM


class P0CPResidualNet(nn.Module):
    """Same small MLP family as CP-to-GE, with P0 concatenated to CP."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(GE_DIM + CP_DIM, 256), nn.GELU(),
            nn.Linear(256, 32), nn.GELU(),
            nn.Linear(32, 256), nn.GELU(),
            nn.Linear(256, GE_DIM),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "fit", "gate", "test", "aggregate", "dry-run"), required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--cm0-npz", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0/molecule_aggregates.npz"))
    parser.add_argument("--ge-data-root", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"))
    parser.add_argument("--split-lock", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json"))
    parser.add_argument("--teacher-root", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/multimodal_expansion_20260830/01_GE_teacher/formal_v2"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--null-rounds", type=int, default=32)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true", help="one epoch / 100 bootstrap only; cannot support evidence")
    args = parser.parse_args()
    if args.stage in {"prepare", "fit", "test"} and args.seed is None:
        parser.error("--seed is required for prepare, fit, and test")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numpy_seed(seed: int) -> int:
    """Make stable 64-bit hash seeds acceptable to NumPy's legacy RNG API."""
    return int(seed % (2**32))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def seed_dir(args: argparse.Namespace) -> Path:
    assert args.seed is not None
    return args.outroot / f"seed_{args.seed}"


def decode(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in np.asarray(values).reshape(-1)]


def load_cp_split(path: Path, split: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        keys = (f"{split}_smiles", f"{split}_cp", f"{split}_cp_plates", f"{split}_cp_rms", f"{split}_max_dose")
        missing = [key for key in keys if key not in loaded.files]
        if missing:
            raise ValueError(f"{path} lacks {missing}")
        smiles = decode(loaded[f"{split}_smiles"])
        cp = loaded[f"{split}_cp"].astype(np.float32)
        plates = loaded[f"{split}_cp_plates"].astype(np.float64)
        rms = loaded[f"{split}_cp_rms"].astype(np.float64)
        max_dose = loaded[f"{split}_max_dose"].astype(np.float64)
    if cp.shape != (len(smiles), CP_DIM):
        raise ValueError(f"Unexpected {split} CP shape: {cp.shape}")
    if not np.isfinite(cp).all():
        raise ValueError(f"Non-finite {split} CP")
    return {"smiles": smiles, "cp": cp, "cp_plates": plates, "cp_rms": rms, "cp_max_dose": max_dose}


def cp_map(data: dict[str, Any]) -> dict[str, np.ndarray]:
    output = {compound: data["cp"][i] for i, compound in enumerate(data["smiles"])}
    if len(output) != len(data["smiles"]):
        raise ValueError("Duplicate CP aggregate compounds")
    return output


def slice_pairs(pairs: T.BudgetPairs, indices: np.ndarray) -> T.BudgetPairs:
    chosen = np.asarray(indices, dtype=np.int64)
    return T.BudgetPairs(
        support=pairs.support[chosen], target=pairs.target[chosen],
        compound=[pairs.compound[int(i)] for i in chosen], dose=[pairs.dose[int(i)] for i in chosen],
        held_rows=pairs.held_rows[chosen], support_rows=[pairs.support_rows[int(i)] for i in chosen],
        manifest=[pairs.manifest[int(i)] for i in chosen],
    )


def fold_membership(compounds: list[str], seed: int, label: str, folds: int) -> np.ndarray:
    return np.asarray([T.stable_seed(seed, f"{label}|{compound}") % folds for compound in compounds], dtype=np.int64)


def assert_official_split(args: argparse.Namespace, split: str, rows: T.PlateRows) -> None:
    lock = T.load_split_lock(args.split_lock)
    observed = set(rows.compound)
    if observed != lock[split]:
        raise ValueError(f"{split} rows disagree with official split lock: {len(observed)} vs {len(lock[split])}")


def build_train_oof_p0(args: argparse.Namespace, pairs: T.BudgetPairs, device: torch.device) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = fold_membership(pairs.compound, args.seed, "p0-oof-outer", OOF_FOLDS)
    if set(labels.tolist()) != set(range(OOF_FOLDS)):
        raise RuntimeError("A deterministic OOF fold is empty")
    output = np.full_like(pairs.target, np.nan, dtype=np.float32)
    metadata: list[dict[str, Any]] = []
    for fold in range(OOF_FOLDS):
        outer = np.flatnonzero(labels == fold)
        remaining = np.flatnonzero(labels != fold)
        inner_labels = fold_membership([pairs.compound[int(i)] for i in remaining], args.seed, f"p0-oof-inner-{fold}", 5)
        inner_valid = remaining[inner_labels == 0]
        inner_train = remaining[inner_labels != 0]
        if min(len(outer), len(inner_train), len(inner_valid)) < 100:
            raise RuntimeError(f"Invalid P0 OOF fold {fold}: outer={len(outer)}, train={len(inner_train)}, valid={len(inner_valid)}")
        teacher, mean, scale, best_epoch, best_loss = T.fit_teacher(
            slice_pairs(pairs, inner_train), slice_pairs(pairs, inner_valid),
            numpy_seed(T.stable_seed(args.seed, f"p0-oof-teacher-{fold}")), args.epochs, args.batch_size,
            args.learning_rate, args.weight_decay, device, args.smoke,
        )
        output[outer] = T.predict(teacher, pairs.support[outer], mean, scale, args.batch_size, device)
        metadata.append({"fold": fold, "outer": int(len(outer)), "inner_train": int(len(inner_train)), "inner_valid": int(len(inner_valid)), "best_epoch": int(best_epoch), "best_validation_mse": float(best_loss)})
    if not np.isfinite(output).all():
        raise RuntimeError("P0 OOF prediction has unset values")
    return output, metadata


def make_residual_arrays(p0: np.ndarray, cp: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not (p0.shape == target.shape and p0.shape[1] == GE_DIM and cp.shape == (len(p0), CP_DIM)):
        raise ValueError(f"Residual array mismatch: p0={p0.shape}, cp={cp.shape}, target={target.shape}")
    return np.concatenate([p0, cp], axis=1).astype(np.float32), (target - p0).astype(np.float32)


def fit_residual_model(
    train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray,
    seed: int, args: argparse.Namespace, device: torch.device,
) -> tuple[P0CPResidualNet, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    seed = numpy_seed(seed)
    C.set_seed(seed)
    x_mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_scale = np.maximum(train_x.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    y_mean = train_y.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_scale = np.maximum(train_y.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    tx = (train_x - x_mean) / x_scale; ty = (train_y - y_mean) / y_scale
    vx = (valid_x - x_mean) / x_scale; vy = (valid_y - y_mean) / y_scale
    model = P0CPResidualNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(); generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty)), batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss, best_epoch = float("inf"), -1
    for epoch in range(1 if args.smoke else args.epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x.to(device)) - y.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite P0-conditioned residual loss")
            loss.backward(); optimizer.step()
        model.eval(); total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(vx), args.batch_size):
                x = torch.from_numpy(vx[start:start + args.batch_size]).to(device)
                y = torch.from_numpy(vy[start:start + args.batch_size]).to(device)
                value = torch.mean((model(x) - y) ** 2)
                total += float(value.detach().cpu()) * len(x); count += len(x)
        loss_valid = total / max(count, 1)
        if loss_valid < best_loss:
            best_loss, best_epoch = loss_valid, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 0 or epoch % 10 == 0:
            print(f"[seed {seed}] residual-inner epoch={epoch} mse={loss_valid:.7f}", flush=True)
    if best_state is None:
        raise RuntimeError("Residual model did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, x_mean, x_scale, y_mean, y_scale, best_epoch, best_loss


def predict_residual(
    model: P0CPResidualNet, x: np.ndarray, x_mean: np.ndarray, x_scale: np.ndarray,
    y_mean: np.ndarray, y_scale: np.ndarray, batch_size: int, device: torch.device,
) -> np.ndarray:
    model.eval(); output = np.empty((len(x), GE_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            value = (x[start:start + batch_size] - x_mean) / x_scale
            prediction = model(torch.from_numpy(value).to(device)).detach().cpu().numpy()
            output[start:start + len(prediction)] = prediction * y_scale + y_mean
    return output


def selected_beta(pairs: T.BudgetPairs, p0: np.ndarray, residual: np.ndarray, reference: dict[str, dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    curve: dict[str, Any] = {}
    for beta in BETA_GRID:
        records = T.method_records((p0 + beta * residual).astype(np.float32), pairs, reference, f"correct_beta_{beta}")
        curve[str(beta)] = T.summarize(records)
    beta = max(BETA_GRID, key=lambda value: (curve[str(value)]["rsf"], -value))
    return float(beta), curve


def p0_artifact(path: Path, expected: list[str]) -> np.ndarray:
    payload = C.load_teacher(path)
    if list(payload["compound"]) != expected:
        raise ValueError(f"Frozen P0 order mismatch: {path}")
    return payload["teacher"]


def residual_inner_split(compounds: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = fold_membership(compounds, seed, "residual-inner", 5)
    valid = np.flatnonzero(labels == 0); train = np.flatnonzero(labels != 0)
    if min(len(train), len(valid)) < 500:
        raise RuntimeError(f"Residual inner split invalid: train={len(train)}, valid={len(valid)}")
    return train, valid


def prepare(args: argparse.Namespace) -> None:
    out = seed_dir(args)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    rows = T.load_split_rows(args.ge_data_root, "train")
    assert_official_split(args, "train", rows)
    pairs = T.build_pairs(rows, "train", BUDGET)
    if len(pairs.compound) < 500:
        raise RuntimeError("Training GE 1R eligibility below 500")
    cp = load_cp_split(args.cm0_npz, "train")
    cp_by_compound = cp_map(cp)
    if set(pairs.compound) - set(cp_by_compound):
        raise RuntimeError("GE training compounds missing CP aggregate")
    device = T.choose_device(args.device)
    p0_oof, folds = build_train_oof_p0(args, pairs, device)
    np.savez_compressed(
        out / "p0_oof_train.npz", compound=np.asarray(pairs.compound, dtype="U"), dose=np.asarray(pairs.dose, dtype="U"),
        held_rows=pairs.held_rows, target=pairs.target, p0_oof=p0_oof,
    )
    write_json(out / "PREPARE_COMPLETE.json", {
        "version": VERSION, "seed": args.seed, "stage": "prepare", "budget": BUDGET,
        "pair_count": len(pairs.compound), "folds": folds, "test_loaded": False,
        "guards": ["official split lock checked", "only train GE and train CP loaded", "P0 train values are OOF"],
        "source_sha256": {"runner": sha256(THIS), "teacher_code": sha256((Path(__file__).resolve().parents[3] / "analysis/cross_modal/expression_teacher/ge_teacher_experiment.py"))},
        "train_cp_source": str(args.cm0_npz),
    })


def fit(args: argparse.Namespace) -> None:
    out = seed_dir(args)
    prepared = read_json(out / "PREPARE_COMPLETE.json")
    if prepared.get("version") != VERSION or prepared.get("test_loaded") is not False:
        raise RuntimeError("Invalid prepare artifact")
    if (out / "FIT_COMPLETE.json").exists():
        raise FileExistsError("Refusing to overwrite fit artifact")
    with np.load(out / "p0_oof_train.npz", allow_pickle=False) as loaded:
        compounds = decode(loaded["compound"]); target = loaded["target"].astype(np.float32); p0_oof = loaded["p0_oof"].astype(np.float32)
    cp_train = load_cp_split(args.cm0_npz, "train"); train_map = cp_map(cp_train)
    if any(compound not in train_map for compound in compounds):
        raise RuntimeError("OOF train P0 and CP compounds disagree")
    train_cp = np.vstack([train_map[compound] for compound in compounds])
    rows_valid = T.load_split_rows(args.ge_data_root, "valid")
    assert_official_split(args, "valid", rows_valid)
    valid = T.build_pairs(rows_valid, "valid", BUDGET)
    cp_valid = load_cp_split(args.cm0_npz, "valid"); valid_map = cp_map(cp_valid)
    if any(compound not in valid_map for compound in valid.compound):
        raise RuntimeError("Validation GE and CP compounds disagree")
    p0_valid_path = args.teacher_root / f"seed_{args.seed}" / "budget1_valid_predictions.npz"
    p0_valid = p0_artifact(p0_valid_path, valid.compound)
    x, y = make_residual_arrays(p0_oof, train_cp, target)
    valid_x, valid_y = make_residual_arrays(p0_valid, np.vstack([valid_map[c] for c in valid.compound]), valid.target)
    inner_train, inner_valid = residual_inner_split(compounds, args.seed)
    device = T.choose_device(args.device)
    model, x_mean, x_scale, y_mean, y_scale, epoch, loss = fit_residual_model(
        x[inner_train], y[inner_train], x[inner_valid], y[inner_valid], T.stable_seed(args.seed, "residual-fit"), args, device,
    )
    residual_valid = predict_residual(model, valid_x, x_mean, x_scale, y_mean, y_scale, args.batch_size, device)
    reference_valid = T.strict_reference(rows_valid, valid, args.null_rounds, args.seed)
    beta, curve = selected_beta(valid, p0_valid, residual_valid, reference_valid)
    torch.save({"version": VERSION, "state_dict": model.state_dict(), "x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "y_scale": y_scale}, out / "residual_mapper.pt")
    write_json(out / "FIT_COMPLETE.json", {
        "version": VERSION, "seed": args.seed, "stage": "fit", "test_loaded": False,
        "p0_valid_sha256": sha256(p0_valid_path), "residual_inner_train": int(len(inner_train)), "residual_inner_valid": int(len(inner_valid)),
        "best_inner_epoch": int(epoch), "best_inner_validation_mse": float(loss), "selected_beta": beta, "validation_curve": curve,
        "validation_strict_reference_compounds": len(reference_valid),
        "guards": ["only train and valid GE/CP/P0 loaded", "beta grid includes 0", "validation only selects beta", "no test feature or label loaded"],
    })


def gate(args: argparse.Namespace) -> None:
    if (args.outroot / "TEST_AUTHORIZED.json").exists() or (args.outroot / "VALIDATION_STOP.json").exists():
        raise FileExistsError("A validation gate already exists")
    fits = {seed: read_json(args.outroot / f"seed_{seed}" / "FIT_COMPLETE.json") for seed in SEEDS}
    if any(payload.get("test_loaded") for payload in fits.values()):
        raise RuntimeError("Fit artifact reports test access")
    beta = {str(seed): float(payload["selected_beta"]) for seed, payload in fits.items()}
    base = {"version": VERSION, "stage": "validation_gate", "test_loaded": False, "selected_beta": beta, "rule": "all three seeds must select beta > 0 before test access"}
    if all(value > 0.0 for value in beta.values()):
        write_json(args.outroot / "TEST_AUTHORIZED.json", {**base, "decision": "TEST-AUTHORIZED"})
    else:
        write_json(args.outroot / "VALIDATION_STOP.json", {**base, "decision": "P0-CONDITIONED-CP-RESIDUAL-NO-GO"})


def load_mapper(path: Path, device: torch.device) -> tuple[P0CPResidualNet, dict[str, np.ndarray]]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("version") != VERSION:
        raise RuntimeError("Residual mapper version mismatch")
    model = P0CPResidualNet().to(device); model.load_state_dict(payload["state_dict"])
    return model, {key: np.asarray(payload[key], dtype=np.float32) for key in ("x_mean", "x_scale", "y_mean", "y_scale")}


def prediction_map(
    pairs: T.BudgetPairs, p0: np.ndarray, cpm: dict[str, np.ndarray], donors: dict[str, str] | None,
    model: P0CPResidualNet, normalizers: dict[str, np.ndarray], beta: float, args: argparse.Namespace, device: torch.device,
) -> dict[str, np.ndarray]:
    compounds: list[str] = []; base: list[np.ndarray] = []; cp_values: list[np.ndarray] = []
    for index, compound in enumerate(pairs.compound):
        donor = donors.get(compound, compound) if donors is not None else compound
        if donor not in cpm:
            continue
        compounds.append(compound); base.append(p0[index]); cp_values.append(cpm[donor])
    if not compounds:
        raise RuntimeError("No CP donors available for prediction")
    values, _ = make_residual_arrays(np.vstack(base), np.vstack(cp_values), np.vstack(base))
    residual = predict_residual(model, values, normalizers["x_mean"], normalizers["x_scale"], normalizers["y_mean"], normalizers["y_scale"], args.batch_size, device)
    return {compound: (base[index] + beta * residual[index]).astype(np.float32) for index, compound in enumerate(compounds)}


def test(args: argparse.Namespace) -> None:
    out = seed_dir(args)
    authorized = read_json(args.outroot / "TEST_AUTHORIZED.json")
    if authorized.get("decision") != "TEST-AUTHORIZED":
        raise RuntimeError("Test not authorized by validation gate")
    if (out / "TEST_COMPLETE.json").exists():
        raise FileExistsError("Refusing to overwrite test artifact")
    fit_artifact = read_json(out / "FIT_COMPLETE.json")
    beta = float(fit_artifact["selected_beta"])
    if beta <= 0:
        raise RuntimeError("A beta=0 seed may not read test")
    rows_test = T.load_split_rows(args.ge_data_root, "test")
    assert_official_split(args, "test", rows_test)
    pairs = T.build_pairs(rows_test, "test", BUDGET)
    cp_test = load_cp_split(args.cm0_npz, "test")
    cp_train = load_cp_split(args.cm0_npz, "train")
    test_map = cp_map(cp_test)
    p0_test_path = args.teacher_root / f"seed_{args.seed}" / "budget1_predictions.npz"
    p0_test = p0_artifact(p0_test_path, pairs.compound)
    device = T.choose_device(args.device)
    model, normalizers = load_mapper(out / "residual_mapper.pt", device)
    reference = T.strict_reference(rows_test, pairs, args.null_rounds, args.seed)
    shuffled = C.derangement(pairs.compound, args.seed, "test")
    foreign = C.matched_foreign(pairs.manifest, cp_test, cp_train, args.seed, "test")
    p0_map = {compound: p0_test[index] for index, compound in enumerate(pairs.compound)}
    predictions = {
        "frozen_P0": p0_map,
        "correct_CP": prediction_map(pairs, p0_test, test_map, None, model, normalizers, beta, args, device),
        "shuffled_CP": prediction_map(pairs, p0_test, test_map, shuffled, model, normalizers, beta, args, device),
        "matched_foreign_CP": prediction_map(pairs, p0_test, test_map, foreign, model, normalizers, beta, args, device),
    }
    records = {name: C.records_for(prediction, pairs, reference, name) for name, prediction in predictions.items()}
    contrasts = {
        "correct_minus_p0": C.contrast(records["correct_CP"], records["frozen_P0"], args.bootstrap_rounds, T.stable_seed(args.seed, "p0cp-correct-p0")),
        "correct_minus_shuffled": C.contrast(records["correct_CP"], records["shuffled_CP"], args.bootstrap_rounds, T.stable_seed(args.seed, "p0cp-correct-shuffled")),
        "correct_minus_matched_foreign": C.contrast(records["correct_CP"], records["matched_foreign_CP"], args.bootstrap_rounds, T.stable_seed(args.seed, "p0cp-correct-foreign")),
    }
    C.write_csv(out / "per_molecule_scores.csv", [row for method in records.values() for row in method.values()])
    write_json(out / "TEST_COMPLETE.json", {
        "version": VERSION, "seed": args.seed, "stage": "test", "test_loaded": True, "selected_beta": beta,
        "pair_count": len(pairs.compound), "strict_reference_compounds": len(reference), "foreign_donors": len(foreign),
        "p0_test_sha256": sha256(p0_test_path), "contrasts": contrasts,
        "guards": ["test was entered only after all-seed validation authorization", "same P0/mapper/beta used in all arms", "paired compound bootstrap"],
    })


def aggregate(args: argparse.Namespace) -> None:
    if not (args.outroot / "TEST_AUTHORIZED.json").is_file():
        raise RuntimeError("Cannot aggregate without test authorization")
    if (args.outroot / "FORMAL_DECISION.json").exists():
        raise FileExistsError("Formal decision already exists")
    results = {seed: read_json(args.outroot / f"seed_{seed}" / "TEST_COMPLETE.json") for seed in SEEDS}
    names = ("correct_minus_p0", "correct_minus_shuffled", "correct_minus_matched_foreign")
    table: list[dict[str, Any]] = []
    all_pass = True
    for seed, payload in results.items():
        for name in names:
            value = payload["contrasts"][name]
            passed = value["excess_z_point"] > 0 and value["excess_z_ci95"][0] > 0
            all_pass = all_pass and passed
            table.append({"seed": seed, "contrast": name, "molecule_count": value["molecule_count"], "excess_z_point": value["excess_z_point"], "excess_z_ci95": value["excess_z_ci95"], "raw_pcc_point": value["raw_pcc_point"], "raw_pcc_ci95": value["raw_pcc_ci95"], "pass": passed})
    with (args.outroot / "paired_contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    write_json(args.outroot / "FORMAL_DECISION.json", {
        "version": VERSION, "decision": "P0-CONDITIONED-CP-RESIDUAL-GO" if all_pass else "P0-CONDITIONED-CP-RESIDUAL-NO-GO",
        "primary_rule": "every seed and every correct-minus-comparator excess-z contrast has positive point estimate and lower 95% paired-bootstrap bound above zero",
        "contrasts": table, "test_loaded": True,
    })


def dry_run(args: argparse.Namespace) -> None:
    rows_train = T.load_split_rows(args.ge_data_root, "train"); rows_valid = T.load_split_rows(args.ge_data_root, "valid")
    assert_official_split(args, "train", rows_train); assert_official_split(args, "valid", rows_valid)
    train = T.build_pairs(rows_train, "train", BUDGET); valid = T.build_pairs(rows_valid, "valid", BUDGET)
    cp_train = load_cp_split(args.cm0_npz, "train"); cp_valid = load_cp_split(args.cm0_npz, "valid")
    missing = {"train": len(set(train.compound) - set(cp_map(cp_train))), "valid": len(set(valid.compound) - set(cp_map(cp_valid)))}
    print(json.dumps({"version": VERSION, "stage": "dry-run", "test_loaded": False, "budget": BUDGET, "pairs": {"train": len(train.compound), "valid": len(valid.compound)}, "cp_missing": missing, "oof_folds": OOF_FOLDS, "beta_grid": list(BETA_GRID)}, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.bootstrap_rounds = min(args.bootstrap_rounds, 100)
    if args.stage == "dry-run":
        dry_run(args)
    elif args.stage == "prepare":
        prepare(args)
    elif args.stage == "fit":
        fit(args)
    elif args.stage == "gate":
        gate(args)
    elif args.stage == "test":
        test(args)
    elif args.stage == "aggregate":
        aggregate(args)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
