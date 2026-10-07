#!/usr/bin/env python3
"""Formal compound-validation MLP-teacher selection and refit runner.

The runner is deliberately isolated from the P0 target/student code.  It has
three stages: ``preflight``, ``select`` and ``refit``.  There is no test stage.
``select`` trains five compound-OOF teachers for each of the three declared
MLP capacities on Fit drugs only.  Each epoch is scored on Validation by the
compound-equal objective: reciprocal repeat directions are averaged within
each (drug,dose) condition, conditions are averaged within each compound, and
compounds are then weighted equally.  The same objective selects the
non-negative CFRA simplex mixture of MLP (IMR/LSO-equivalent) and IMCEB
candidates.  Capacity, fold epochs and fold weights are frozen before
``refit``.

``refit`` trains five OOF teachers on Fit union Validation minus each refit
held fold for the locked number of epochs.  It performs no validation scoring,
checkpoint search, or confirmation profile read.  Checkpoints contain the
normalizer, IMCEB shrinkage and frozen mixture weights so a later P0 target
runner can load them without reopening the selection data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
BASE_PATH = (Path(__file__).resolve().parents[4] / "analysis/predictor_supervision/sci_plex3/support/run_p0.py")
OLD_PATH = (Path(__file__).resolve().parents[4] / "analysis/predictor_supervision/sci_plex3/support/run_own_cell_unseen.py")
spec = importlib.util.spec_from_file_location("sciplex3_formal_teacher_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
import sys
sys.modules["sciplex3_formal_teacher_base"] = base
assert spec and spec.loader
spec.loader.exec_module(base)


TARGET = "K562"
SURFACE = "own_cell_unseen"
MAX_EPOCHS = 160
TEACHER_BATCH = 256
TEACHER_LR = 3e-4
TEACHER_WEIGHT_DECAY = 1e-5
OOF_FOLDS = 5
EXPECTED_UNITS = 613
CAPACITIES: dict[str, tuple[int, int]] = {
    "H256Z32": (256, 32),
    "H512Z128": (512, 128),
    "H512Z256": (512, 256),
}
CAPACITY_ORDER = tuple(CAPACITIES)
VERSION = "sciplex3-cfra-mlpteacher-compoundval-k562-v1-2026-09-16"
DEFAULT_V3_PLAN = Path(
    "/path/to/data/AIDD/MVCPert_5_27/runs/"
    "sciplex3_rawagg_cfra_p0_20260915_v3_own_cell_unseen/PLAN.json"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "select", "refit"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--v3-plan", type=Path, default=DEFAULT_V3_PLAN)
    p.add_argument("--conditions", type=Path, default=Path(
        "/path/to/data/AIDD/MVCPert_5_27/experiments/external_validation_sciplex3_papalexi_20260903/"
        "sciplex3_validation/eligible_conditions_n50.csv"
    ))
    p.add_argument("--group-counts", type=Path, default=Path(
        "/path/to/data/AIDD/MVCPert_5_27/experiments/external_validation_sciplex3_papalexi_20260903/"
        "sciplex3_validation/group_counts.h5"
    ))
    p.add_argument("--split-lock", type=Path, default=Path(
        "/path/to/data/AIDD/MVCPert_5_27/experiments/external_validation_sciplex3_papalexi_20260903/"
        "sciplex3_validation/cfra_confirmation/split_lock.json"
    ))
    p.add_argument("--structure-map", type=Path, default=Path(
        "/path/to/data/AIDD/MVCPert_5_27/experiments/external_validation_sciplex3_papalexi_20260903/"
        "sciplex3_student_crosscell/input_preflight/DRUG_ECFP4_MAPPING.npz"
    ))
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p.parse_args()


class MLPTeacher(nn.Module):
    def __init__(self, dimension: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent),
            nn.GELU(),
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, dimension),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _v3_and_split(args: argparse.Namespace) -> tuple[dict[str, Any], list[tuple[str, float]], dict[str, list[str]]]:
    if not args.v3_plan.is_file():
        raise FileNotFoundError(f"missing frozen v3 plan: {args.v3_plan}")
    v3 = read_json(args.v3_plan)
    if v3.get("confirmation_treated_loaded") is not False:
        raise RuntimeError("frozen v3 plan has confirmation treated profiles loaded")
    input_paths = {
        "conditions": args.conditions,
        "group_counts": args.group_counts,
        "split_lock": args.split_lock,
        "structure_map": args.structure_map,
    }
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = v3.get("inputs", {}).get(name, {}).get("sha256")
        actual = sha256(path)
        if expected != actual:
            raise RuntimeError(f"frozen v3 {name} SHA mismatch: {actual} != {expected}")
    expected_old = v3.get("inputs", {}).get("old_runner", {}).get("sha256")
    if expected_old and expected_old != sha256(OLD_PATH):
        raise RuntimeError("frozen v3 old runner SHA mismatch")
    units = [(str(item[0]), float(item[1])) for item in v3.get("units", [])]
    if len(units) != EXPECTED_UNITS or len(set(units)) != EXPECTED_UNITS:
        raise RuntimeError(f"expected exactly {EXPECTED_UNITS} frozen units, got {len(units)}")
    fold = v3.get("folds", {}).get(TARGET, {}).get(SURFACE)
    if not isinstance(fold, dict) or fold.get("source_cell_lines") != [TARGET]:
        raise RuntimeError("frozen v3 plan lacks K562 own-cell source fold")
    names = ("train_drugs", "valid_drugs", "refit_drugs", "test_drugs")
    if any(not isinstance(fold.get(name), list) or not fold[name] for name in names):
        raise RuntimeError("frozen v3 split is incomplete")
    split = {name: sorted(str(x) for x in fold[name]) for name in names}
    sets = {name: set(values) for name, values in split.items()}
    if sets["train_drugs"] & sets["valid_drugs"] or sets["train_drugs"] & sets["test_drugs"] or sets["valid_drugs"] & sets["test_drugs"] or sets["refit_drugs"] & sets["test_drugs"]:
        raise RuntimeError("frozen v3 split overlap")
    unit_drugs = {drug for drug, _dose in units}
    for name, values in sets.items():
        if not values <= unit_drugs:
            raise RuntimeError(f"{name} includes drugs absent from frozen units")
    if not sets["train_drugs"] <= sets["refit_drugs"] or not sets["valid_drugs"] <= sets["refit_drugs"]:
        raise RuntimeError("refit universe must include Fit and Validation")
    return v3, units, split


def _assign(drugs: Iterable[str]) -> dict[str, int]:
    return base.old.teacher_fold_assignment(set(drugs))


def preflight(args: argparse.Namespace) -> None:
    if args.root.exists() and any(args.root.iterdir()):
        raise FileExistsError(f"output root is not empty: {args.root}")
    v3, units, split = _v3_and_split(args)
    selection_assignment = _assign(split["train_drugs"])
    refit_assignment = _assign(split["refit_drugs"])
    if set(selection_assignment.values()) != set(range(OOF_FOLDS)) or set(refit_assignment.values()) != set(range(OOF_FOLDS)):
        raise RuntimeError("compound OOF assignment does not contain all five folds")
    args.root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    assignment_relation: list[dict[str, Any]] = []
    for drug in sorted(set(split["refit_drugs"])):
        assignment_relation.append({
            "drug": drug,
            "pool": "Fit" if drug in set(split["train_drugs"]) else "Validation",
            "selection_fold": selection_assignment.get(drug, "validation_pool"),
            "refit_fold": refit_assignment[drug],
            "selection_assignment_scope": "Fit_only" if drug in selection_assignment else "not_assigned_in_selection",
            "refit_assignment_scope": "Fit_union_Validation",
        })
    for role, assignment in (("selection", selection_assignment), ("refit", refit_assignment)):
        for fold in range(OOF_FOLDS):
            held = {drug for drug, value in assignment.items() if value == fold}
            if role == "selection":
                fit = set(split["train_drugs"]) - held
                validation = set(split["valid_drugs"])
            else:
                fit = set(split["refit_drugs"]) - held
                validation = set()
            manifest.append({
                "role": role,
                "fold": fold,
                "fit_drugs": len(fit),
                "validation_drugs": len(validation),
                "held_drugs": len(held),
                "fit_validation_overlap": len(fit & validation),
                "fit_held_overlap": len(fit & held),
                "confirmation_in_fit": len(fit & set(split["test_drugs"])),
                "confirmation_in_validation": len(validation & set(split["test_drugs"])),
            })
    plan = {
        "version": VERSION,
        "stage": "preflight",
        "dataset": "sci-Plex3 Zenodo 7041849",
        "target": TARGET,
        "surface": SURFACE,
        "units": [[drug, dose] for drug, dose in units],
        "n_units": len(units),
        "split": split,
        "selection_assignment": selection_assignment,
        "refit_assignment": refit_assignment,
        "capacities": {name: {"hidden": dims[0], "latent": dims[1], "architecture": f"{base.old.N_HVG}-{dims[0]}-{dims[1]}-{dims[0]}-{base.old.N_HVG}-GELU"} for name, dims in CAPACITIES.items()},
        "selection": {
            "fit_rule": "Fit/base drugs minus held compound fold",
            "validation_rule": "Validation/calibration drugs; disjoint from Fit",
            "max_epochs": MAX_EPOCHS,
            "checkpoint_selection": "minimum validation compound-equal normalized MSE after simplex candidate calibration",
            "objective": "For each (drug,dose), mean the two reciprocal held-repeat direction MSEs; average conditions within each compound; average compounds equally",
            "simplex_candidates": ["IMR_MLP", "LSO_equals_IMR_for_two_repeats", "IMCEB"],
            "simplex_objective": "same validation compound-equal normalized MSE",
            "simplex_constraints": "nonnegative weights summing to one",
            "normalizer": "fit support rows only",
        },
        "refit": {
            "fit_rule": "Fit union Validation minus refit held compound fold",
            "epoch_rule": "locked selection epoch + 1; exactly fixed epochs",
            "validation_read_mode": "validation drugs are included in the refit training union",
            "validation_metrics_read": False,
            "checkpoint_search": False,
            "weights": "locked per selection fold before refit",
        },
        "confirmation_treated_loaded": False,
        "has_test_stage": False,
        "inputs": {
            "conditions": {"path": str(args.conditions), "sha256": sha256(args.conditions)},
            "group_counts": {"path": str(args.group_counts), "sha256": sha256(args.group_counts)},
            "split_lock": {"path": str(args.split_lock), "sha256": sha256(args.split_lock)},
            "structure_map": {"path": str(args.structure_map), "sha256": sha256(args.structure_map)},
            "old_runner": {"path": str(OLD_PATH), "sha256": sha256(OLD_PATH)},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
        "frozen_v3_plan": {"path": str(args.v3_plan), "sha256": sha256(args.v3_plan), "source_version": v3.get("version")},
    }
    write_csv(args.root / "OOF_MANIFEST.csv", manifest)
    write_csv(args.root / "OOF_ASSIGNMENT_RELATION.csv", assignment_relation)
    plan["oof_manifest"] = {"path": "OOF_MANIFEST.csv", "sha256": sha256(args.root / "OOF_MANIFEST.csv")}
    plan["assignment_relation"] = {"path": "OOF_ASSIGNMENT_RELATION.csv", "sha256": sha256(args.root / "OOF_ASSIGNMENT_RELATION.csv")}
    write_json(args.root / "PLAN.json", plan)
    write_json(args.root / "PREPARE_COMPLETE.json", {
        "status": "PASS",
        "version": VERSION,
        "target": TARGET,
        "surface": SURFACE,
        "n_units": len(units),
        "capacities": list(CAPACITY_ORDER),
        "selection_max_epochs": MAX_EPOCHS,
        "confirmation_treated_loaded": False,
        "has_test_stage": False,
    })
    print(json.dumps({"status": "PREFLIGHT_COMPLETE", "n_units": len(units), "capacities": list(CAPACITY_ORDER), "confirmation_treated_loaded": False}))


def load_plan(args: argparse.Namespace) -> tuple[dict[str, Any], list[tuple[str, float]], dict[str, list[str]]]:
    plan_path = args.root / "PLAN.json"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = read_json(plan_path)
    if plan.get("version") != VERSION or plan.get("confirmation_treated_loaded") is not False or plan.get("has_test_stage") is not False:
        raise RuntimeError("formal teacher plan version/leakage check failed")
    if plan.get("inputs", {}).get("runner", {}).get("sha256") != sha256(Path(__file__).resolve()):
        raise RuntimeError("formal teacher runner hash drift")
    _v3, units, split = _v3_and_split(args)
    planned_units = [(str(x[0]), float(x[1])) for x in plan.get("units", [])]
    if planned_units != units:
        raise RuntimeError("formal teacher unit universe drift")
    return plan, units, split


def _pairs(
    profile: dict[tuple[str, str, float, str], tuple[np.ndarray, np.ndarray]],
    drugs: Iterable[str],
    units: list[tuple[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    allowed = set(drugs)
    x: list[np.ndarray] = []
    y: list[np.ndarray] = []
    compounds: list[str] = []
    conditions: list[str] = []
    directions: list[str] = []
    for drug, dose in units:
        if drug not in allowed:
            continue
        condition = f"{drug}|{dose:.9g}"
        for support, held in (("rep1", "rep2"), ("rep2", "rep1")):
            x.append(base.old.response(profile, TARGET, drug, dose, support))
            y.append(base.old.response(profile, TARGET, drug, dose, held))
            compounds.append(str(drug))
            conditions.append(condition)
            directions.append(f"{support}>{held}")
    if len(x) < 8:
        raise RuntimeError(f"too few reciprocal repeat rotations: {len(x)}")
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(compounds, dtype=str),
        np.asarray(conditions, dtype=str),
        np.asarray(directions, dtype=str),
    )


def _shrink(
    profile: dict[tuple[str, str, float, str], tuple[np.ndarray, np.ndarray]],
    drugs: Iterable[str],
    units: list[tuple[str, float]],
) -> np.ndarray:
    allowed = set(drugs)
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    for drug, dose in units:
        if drug not in allowed:
            continue
        r1 = base.old.response(profile, TARGET, drug, dose, "rep1")
        r2 = base.old.response(profile, TARGET, drug, dose, "rep2")
        means.append((r1 + r2) / 2.0)
        variances.append(np.var(np.vstack([r1, r2]), axis=0))
    if len(means) < 4:
        raise RuntimeError("too few source units for IMCEB shrinkage")
    between = np.var(np.vstack(means), axis=0)
    within = np.mean(np.vstack(variances), axis=0)
    return (between / (between + within + 1e-8)).astype(np.float32)


def _normalize(value: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(value, dtype=np.float32) - mean) / scale).astype(np.float32)


def _predict_normalized(model: MLPTeacher, value: np.ndarray, device: torch.device, batch: int = TEACHER_BATCH) -> np.ndarray:
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(value), batch):
            out.append(model(torch.from_numpy(value[begin: begin + batch]).to(device)).cpu().numpy().astype(np.float32))
    return np.vstack(out) if out else np.empty_like(value)


def _mse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((pred.astype(np.float64) - truth.astype(np.float64)) ** 2))


def _compound_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    compounds: np.ndarray,
    conditions: np.ndarray,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Return rotation-weighted and compound-equal normalized MSE.

    ``U_c`` is the ordered set of (drug,dose) conditions for compound c.  A
    condition loss first averages the two reciprocal held-repeat outcomes;
    compound loss averages its condition losses; the final metric averages
    compounds equally.
    """
    squared = np.mean((pred.astype(np.float64) - truth.astype(np.float64)) ** 2, axis=1)
    condition_loss: dict[str, float] = {}
    condition_drug: dict[str, str] = {}
    for condition in sorted(set(conditions.tolist())):
        mask = conditions == condition
        condition_loss[condition] = float(np.mean(squared[mask]))
        condition_drug[condition] = str(condition.split("|", 1)[0])
    compound_loss: dict[str, list[float]] = defaultdict(list)
    for condition, loss in condition_loss.items():
        compound_loss[condition_drug[condition]].append(loss)
    rows: list[dict[str, Any]] = []
    for compound in sorted(compound_loss):
        values = compound_loss[compound]
        rows.append({"compound": compound, "compound_equal_normalized_mse": float(np.mean(values)), "n_conditions": len(values)})
    if not rows:
        raise RuntimeError("empty compound validation metric")
    return float(np.mean(squared)), float(np.mean([row["compound_equal_normalized_mse"] for row in rows])), rows


def _simplex_weights(
    model: MLPTeacher,
    support_raw: np.ndarray,
    held_raw: np.ndarray,
    compounds: np.ndarray,
    conditions: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    shrink: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float, float, list[dict[str, Any]]]:
    support_norm = _normalize(support_raw, mean, scale)
    held_norm = _normalize(held_raw, mean, scale)
    imr_norm = _predict_normalized(model, support_norm, device)
    imceb_norm = _normalize(support_raw * shrink, mean, scale)
    best: tuple[float, int, np.ndarray, float, list[dict[str, Any]]] | None = None
    # With two physical repeats LSO is exactly IMR.  Evaluate the equivalent
    # simplex line once per 0.05 tick and retain the deterministic representative
    # [IMR, LSO=0, IMCEB].
    for imr_ticks in range(21):
        imr_weight = imr_ticks / 20.0
        weights = np.asarray([imr_weight, 0.0, 1.0 - imr_weight], dtype=np.float32)
        prediction = imr_weight * imr_norm + (1.0 - imr_weight) * imceb_norm
        rotation_mse, compound_mse, rows = _compound_metrics(prediction, held_norm, compounds, conditions)
        candidate = (compound_mse, imr_ticks, weights, rotation_mse, rows)
        if best is None or (candidate[0], candidate[1]) < (best[0], best[1]):
            best = candidate
    assert best is not None
    return best[2], float(best[0]), float(best[3]), best[4]


def _seed(token: str) -> int:
    return int(base.old.stable_int(base.old.SEED, "formal-compound-teacher", token) % (2**32 - 1))


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8"))
        h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _fit_selection_teacher(
    capacity: str,
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    v_compounds: np.ndarray,
    v_conditions: np.ndarray,
    fit_drugs: set[str],
    profile: dict[tuple[str, str, float, str], tuple[np.ndarray, np.ndarray]],
    units: list[tuple[str, float]],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    hidden, latent = CAPACITIES[capacity]
    base.old.set_seed(seed)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    xn = _normalize(x, mean, scale)
    yn = _normalize(y, mean, scale)
    shrink = _shrink(profile, fit_drugs, units)
    model = MLPTeacher(base.old.N_HVG, hidden, latent).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TEACHER_LR, weight_decay=TEACHER_WEIGHT_DECAY)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(xn), torch.from_numpy(yn))
    loader = torch.utils.data.DataLoader(dataset, batch_size=TEACHER_BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    best_state: dict[str, torch.Tensor] | None = None
    best_weights: np.ndarray | None = None
    best_epoch = -1
    best_compound = float("inf")
    logs: list[dict[str, Any]] = []
    compound_rows: list[dict[str, Any]] = []
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb.to(device)) - yb.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite MLP teacher training loss")
            loss.backward()
            optimizer.step()
        train_pred = _predict_normalized(model, xn, device)
        train_mse = _mse(train_pred, yn)
        weights, compound_mse, rotation_mse, per_compound = _simplex_weights(model, vx, vy, v_compounds, v_conditions, mean, scale, shrink, device)
        log = {
            "capacity": capacity,
            "epoch": epoch,
            "epoch_one_based": epoch + 1,
            "train_rotation_normalized_mse": train_mse,
            "validation_rotation_normalized_mse_at_best_simplex": rotation_mse,
            "validation_compound_equal_normalized_mse": compound_mse,
            "best_simplex_imr_mlp_weight": float(weights[0]),
            "best_simplex_lso_weight": float(weights[1]),
            "best_simplex_imceb_weight": float(weights[2]),
        }
        logs.append(log)
        for row in per_compound:
            compound_rows.append({**log, **row, "row_kind": "compound"})
        if compound_mse < best_compound - 1e-15:
            best_compound = compound_mse
            best_epoch = epoch
            best_weights = weights.copy()
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f"[selection {capacity} seed={seed} epoch={epoch}] train={train_mse:.8f} validation_compound={compound_mse:.8f} weights={weights.tolist()}", flush=True)
    if best_state is None or best_weights is None:
        raise RuntimeError("selection checkpoint search failed")
    final_log = logs[best_epoch]
    return {
        "state_dict": best_state,
        "support_mean": mean,
        "support_scale": scale,
        "imceb_shrink": shrink,
        "selected_epoch": best_epoch,
        "selected_epoch_one_based": best_epoch + 1,
        "selected_train_rotation_normalized_mse": float(final_log["train_rotation_normalized_mse"]),
        "selected_validation_rotation_normalized_mse": float(final_log["validation_rotation_normalized_mse_at_best_simplex"]),
        "selected_validation_compound_equal_normalized_mse": best_compound,
        "weights": best_weights,
        "state_hash": _state_hash(best_state),
    }, logs, compound_rows


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return sha256(path)


def select(args: argparse.Namespace) -> None:
    plan, units, split = load_plan(args)
    device = base.old.choose_device(args.device)
    rows = base.old.build_row_map(base.old.read_conditions(args.conditions))
    _fps, _canonical, _smeta = base.old.load_structure_map(args.structure_map)
    store = base.old.ProfileStore(args.group_counts)
    try:
        hvg, hvg_audit = base.old.fit_vehicle_hvg(store, rows, [TARGET])
        source_drugs = set(split["refit_drugs"])
        if source_drugs & set(split["test_drugs"]):
            raise RuntimeError("refit source overlaps confirmation drugs")
        profile = base.old.load_source_profiles(store, rows, [TARGET], source_drugs, hvg, set(units))
        loaded_confirmation = [key for key in profile if key[1] in set(split["test_drugs"])]
        if loaded_confirmation:
            raise RuntimeError(f"confirmation treated profiles loaded: {loaded_confirmation[:3]}")
        selection_assignment = _assign(split["train_drugs"])
        selection_dir = args.root / "selection_checkpoints"
        log_dir = args.root / "selection_logs"
        selection_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        all_logs: list[dict[str, Any]] = []
        all_compound_rows: list[dict[str, Any]] = []
        capacity_results: dict[str, Any] = {}
        for capacity in CAPACITY_ORDER:
            fold_results: list[dict[str, Any]] = []
            for fold in range(OOF_FOLDS):
                held = {drug for drug, value in selection_assignment.items() if value == fold}
                fit_drugs = set(split["train_drugs"]) - held
                validation_drugs = set(split["valid_drugs"])
                if fit_drugs & validation_drugs or fit_drugs & held or fit_drugs & set(split["test_drugs"]):
                    raise RuntimeError(f"selection {capacity} fold {fold} split overlap")
                x, y, _fit_compounds, _fit_conditions, _fit_directions = _pairs(profile, fit_drugs, units)
                vx, vy, v_compounds, v_conditions, _v_directions = _pairs(profile, validation_drugs, units)
                token = f"{TARGET}|{SURFACE}|selection|{capacity}|fold{fold}"
                seed = _seed(token)
                result, logs, compound_rows = _fit_selection_teacher(capacity, x, y, vx, vy, v_compounds, v_conditions, fit_drugs, profile, units, seed, device)
                log_path = log_dir / f"{capacity}_fold{fold}_epochs.csv"
                write_csv(log_path, logs)
                all_logs.extend([{**row, "fold": fold, "role": "selection"} for row in logs])
                all_compound_rows.extend([{**row, "fold": fold, "role": "selection"} for row in compound_rows])
                metadata = {
                    "capacity": capacity,
                    "hidden_dim": CAPACITIES[capacity][0],
                    "latent_dim": CAPACITIES[capacity][1],
                    "architecture": f"{base.old.N_HVG}-{CAPACITIES[capacity][0]}-{CAPACITIES[capacity][1]}-{CAPACITIES[capacity][0]}-{base.old.N_HVG}-GELU",
                    "role": "selection_fit_only_compound_oof",
                    "fold": fold,
                    "held_drugs": sorted(held),
                    "fit_drugs": sorted(fit_drugs),
                    "validation_drugs": sorted(validation_drugs),
                    "fit_validation_disjoint": True,
                    "fit_excludes_held": True,
                    "fit_excludes_confirmation": True,
                    "validation_objective": "compound_equal_normalized_mse",
                    "simplex_objective": "same compound_equal_normalized_mse",
                    "n_fit_rotations": len(x),
                    "n_validation_rotations": len(vx),
                    "n_validation_compounds": len(set(v_compounds.tolist())),
                    "teacher_seed": seed,
                    "max_epochs": MAX_EPOCHS,
                    "selected_epoch": result["selected_epoch"],
                    "selected_epoch_one_based": result["selected_epoch_one_based"],
                    "selected_train_rotation_normalized_mse": result["selected_train_rotation_normalized_mse"],
                    "selected_validation_rotation_normalized_mse": result["selected_validation_rotation_normalized_mse"],
                    "selected_validation_compound_equal_normalized_mse": result["selected_validation_compound_equal_normalized_mse"],
                    "weights": {"IMR_MLP": float(result["weights"][0]), "LSO_equals_IMR": float(result["weights"][1]), "IMCEB": float(result["weights"][2])},
                    "state_hash": result["state_hash"],
                    "log_path": str(log_path.relative_to(args.root).as_posix()),
                    "confirmation_treated_loaded": False,
                }
                checkpoint = selection_dir / capacity / f"fold{fold}.pt"
                checkpoint_sha = _save_checkpoint(checkpoint, {
                    "version": VERSION,
                    "capacity": capacity,
                    "architecture": metadata["architecture"],
                    "state_dict": result["state_dict"],
                    "support_mean": result["support_mean"],
                    "support_scale": result["support_scale"],
                    "imceb_shrink": result["imceb_shrink"],
                    "weights": result["weights"],
                    "metadata": metadata,
                })
                metadata["checkpoint"] = str(checkpoint.relative_to(args.root).as_posix())
                metadata["checkpoint_sha256"] = checkpoint_sha
                fold_results.append(metadata)
            capacity_results[capacity] = {
                "folds": fold_results,
                "mean_selected_validation_compound_equal_normalized_mse": float(np.mean([x["selected_validation_compound_equal_normalized_mse"] for x in fold_results])),
                "selected_epochs": [int(x["selected_epoch"]) for x in fold_results],
                "selected_epochs_one_based": [int(x["selected_epoch_one_based"]) for x in fold_results],
            }
        write_csv(args.root / "selection_epoch_summary.csv", all_logs)
        write_csv(args.root / "selection_compound_losses.csv", all_compound_rows)
        selection_artifacts: list[dict[str, Any]] = []
        for relative in ("OOF_MANIFEST.csv", "OOF_ASSIGNMENT_RELATION.csv", "selection_epoch_summary.csv", "selection_compound_losses.csv"):
            artifact_path = args.root / relative
            selection_artifacts.append({"path": relative, "sha256": sha256(artifact_path)})
        for capacity in CAPACITY_ORDER:
            for row in capacity_results[capacity]["folds"]:
                selection_artifacts.append({"path": row["checkpoint"], "sha256": row["checkpoint_sha256"], "capacity": capacity, "fold": row["fold"]})
        selected_capacity = min(CAPACITY_ORDER, key=lambda name: (capacity_results[name]["mean_selected_validation_compound_equal_normalized_mse"], CAPACITY_ORDER.index(name)))
        selected_folds = capacity_results[selected_capacity]["folds"]
        freeze = {
            "status": "FROZEN",
            "version": VERSION,
            "target": TARGET,
            "surface": SURFACE,
            "selected_capacity": selected_capacity,
            "selected_capacity_hidden_dim": CAPACITIES[selected_capacity][0],
            "selected_capacity_latent_dim": CAPACITIES[selected_capacity][1],
            "selection_metric": "mean of five fold-specific validation compound-equal normalized MSE after simplex calibration",
            "capacity_means": {name: capacity_results[name]["mean_selected_validation_compound_equal_normalized_mse"] for name in CAPACITY_ORDER},
            "capacity_selection_uses_only_validation": True,
            "locked_folds": [
                {
                    "fold": int(row["fold"]),
                    "selection_held_drugs": row["held_drugs"],
                    "selected_epoch": int(row["selected_epoch"]),
                    "selected_epoch_one_based": int(row["selected_epoch_one_based"]),
                    "weights": row["weights"],
                    "validation_compound_equal_normalized_mse": row["selected_validation_compound_equal_normalized_mse"],
                }
                for row in selected_folds
            ],
            "selection_assignment": selection_assignment,
            "refit_assignment": _assign(split["refit_drugs"]),
            "assignment_relation": {"path": "OOF_ASSIGNMENT_RELATION.csv", "sha256": sha256(args.root / "OOF_ASSIGNMENT_RELATION.csv"), "selection_scope": "Fit only", "refit_scope": "Fit union Validation"},
            "selection_artifacts": selection_artifacts,
            "no_confirmation_treated_loaded": True,
            "checkpoint_search_after_freeze": False,
            "inputs": plan["inputs"],
        }
        freeze_path = args.root / "SELECTION_FREEZE.json"
        write_json(freeze_path, freeze)
        freeze_sha = sha256(freeze_path)
        (args.root / "SELECTION_FREEZE.sha256").write_text(freeze_sha + "\n", encoding="utf-8")
        write_json(args.root / "SELECTION_COMPLETE.json", {
            "status": "PASS",
            "version": VERSION,
            "selected_capacity": selected_capacity,
            "capacity_results": capacity_results,
            "freeze": "SELECTION_FREEZE.json",
            "freeze_sha256": freeze_sha,
            "selection_artifacts": selection_artifacts,
            "teacher_count": len(CAPACITY_ORDER) * OOF_FOLDS,
            "confirmation_treated_loaded": False,
            "has_test_stage": False,
        })
        print(json.dumps({"status": "SELECTION_COMPLETE", "selected_capacity": selected_capacity, "capacity_means": freeze["capacity_means"], "confirmation_treated_loaded": False}), flush=True)
    finally:
        store.close()


def _fit_fixed_teacher(
    capacity: str,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    hidden, latent = CAPACITIES[capacity]
    base.old.set_seed(seed)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    xn = _normalize(x, mean, scale)
    yn = _normalize(y, mean, scale)
    model = MLPTeacher(base.old.N_HVG, hidden, latent).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TEACHER_LR, weight_decay=TEACHER_WEIGHT_DECAY)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(xn), torch.from_numpy(yn))
    loader = torch.utils.data.DataLoader(dataset, batch_size=TEACHER_BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb.to(device)) - yb.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite fixed refit teacher loss")
            loss.backward()
            optimizer.step()
    final_train_mse = _mse(_predict_normalized(model, xn, device), yn)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return {
        "state_dict": state,
        "support_mean": mean,
        "support_scale": scale,
        "final_train_rotation_normalized_mse": final_train_mse,
        "state_hash": _state_hash(state),
    }


def _build_refit_oof_targets(
    profile: dict[tuple[str, str, float, str], tuple[np.ndarray, np.ndarray]],
    units: list[tuple[str, float]],
    refit_assignment: dict[str, int],
    teachers: dict[int, dict[str, Any]],
    refit_drugs: set[str],
) -> tuple[list[tuple[str, float]], np.ndarray, np.ndarray]:
    """Create CFRA-1R-Aggregate labels for every non-confirmation refit unit."""
    target_units = [(drug, float(dose)) for drug, dose in units if drug in refit_drugs]
    if not target_units:
        raise RuntimeError("no non-confirmation units available for refit OOF targets")
    labels: list[np.ndarray] = []
    folds: list[int] = []
    for drug, dose in target_units:
        if drug not in refit_assignment:
            raise RuntimeError(f"missing refit OOF assignment for {drug}")
        fold = int(refit_assignment[drug])
        teacher = teachers[fold]
        repeats = np.vstack([
            base.old.response(profile, TARGET, drug, dose, "rep1"),
            base.old.response(profile, TARGET, drug, dose, "rep2"),
        ]).astype(np.float32)
        labels.append(np.mean(predict_refit_teacher(teacher, repeats), axis=0).astype(np.float32))
        folds.append(fold)
    result = np.vstack(labels).astype(np.float32)
    if result.shape != (len(target_units), base.old.N_HVG) or not np.isfinite(result).all():
        raise RuntimeError(f"invalid refit OOF target matrix: {result.shape}")
    return target_units, result, np.asarray(folds, dtype=np.int64)


def refit(args: argparse.Namespace) -> None:
    plan, units, split = load_plan(args)
    freeze_path = args.root / "SELECTION_FREEZE.json"
    if not freeze_path.is_file():
        raise FileNotFoundError(freeze_path)
    freeze = read_json(freeze_path)
    if freeze.get("status") != "FROZEN" or freeze.get("version") != VERSION or freeze.get("no_confirmation_treated_loaded") is not True:
        raise RuntimeError("selection freeze is missing or invalid")
    selected_capacity = str(freeze["selected_capacity"])
    if selected_capacity not in CAPACITIES:
        raise RuntimeError(f"unknown frozen capacity: {selected_capacity}")
    sidecar = args.root / "SELECTION_FREEZE.sha256"
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256(freeze_path):
        raise RuntimeError("selection freeze sidecar hash check failed")
    if freeze.get("inputs") != plan.get("inputs"):
        raise RuntimeError("selection freeze input hashes differ from PLAN")
    for artifact in freeze.get("selection_artifacts", []):
        artifact_path = args.root / str(artifact["path"])
        if not artifact_path.is_file() or sha256(artifact_path) != artifact.get("sha256"):
            raise RuntimeError(f"selection artifact hash drift: {artifact_path}")
    device = base.old.choose_device(args.device)
    rows = base.old.build_row_map(base.old.read_conditions(args.conditions))
    _fps, _canonical, _smeta = base.old.load_structure_map(args.structure_map)
    store = base.old.ProfileStore(args.group_counts)
    try:
        hvg, hvg_audit = base.old.fit_vehicle_hvg(store, rows, [TARGET])
        source_drugs = set(split["refit_drugs"])
        if source_drugs & set(split["test_drugs"]):
            raise RuntimeError("refit source overlaps confirmation drugs")
        profile = base.old.load_source_profiles(store, rows, [TARGET], source_drugs, hvg, set(units))
        loaded_confirmation = [key for key in profile if key[1] in set(split["test_drugs"])]
        if loaded_confirmation:
            raise RuntimeError(f"confirmation treated profiles loaded before refit: {loaded_confirmation[:3]}")
        refit_assignment = _assign(split["refit_drugs"])
        locked = {int(row["fold"]): row for row in freeze["locked_folds"]}
        if set(locked) != set(range(OOF_FOLDS)):
            raise RuntimeError("selection freeze lacks five locked folds")
        checkpoint_dir = args.root / "refit_checkpoints" / selected_capacity
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        teachers: dict[int, dict[str, Any]] = {}
        for fold in range(OOF_FOLDS):
            held = {drug for drug, value in refit_assignment.items() if value == fold}
            fit_drugs = set(split["refit_drugs"]) - held
            if fit_drugs & held or fit_drugs & set(split["test_drugs"]):
                raise RuntimeError(f"refit fold {fold} split overlap")
            lock = locked[fold]
            epochs = int(lock["selected_epoch"]) + 1
            if epochs < 1 or epochs > MAX_EPOCHS:
                raise RuntimeError(f"invalid locked epoch for fold {fold}: {epochs}")
            weights = {str(k): float(v) for k, v in lock["weights"].items()}
            if abs(sum(weights.values()) - 1.0) > 1e-6 or min(weights.values()) < -1e-8:
                raise RuntimeError(f"invalid locked simplex weights for fold {fold}: {weights}")
            x, y, _fit_compounds, _fit_conditions, _fit_directions = _pairs(profile, fit_drugs, units)
            token = f"{TARGET}|{SURFACE}|refit|{selected_capacity}|fold{fold}"
            result = _fit_fixed_teacher(selected_capacity, x, y, epochs, _seed(token), device)
            metadata = {
                "version": VERSION,
                "capacity": selected_capacity,
                "hidden_dim": CAPACITIES[selected_capacity][0],
                "latent_dim": CAPACITIES[selected_capacity][1],
                "architecture": f"{base.old.N_HVG}-{CAPACITIES[selected_capacity][0]}-{CAPACITIES[selected_capacity][1]}-{CAPACITIES[selected_capacity][0]}-{base.old.N_HVG}-GELU",
                "role": "refit_oof_teacher",
                "fold": fold,
                "refit_held_drugs": sorted(held),
                "fit_drugs": sorted(fit_drugs),
                "fit_universe": "Fit union Validation minus refit held fold",
                "fit_includes_validation_drugs": True,
                "validation_read_mode": "train_only",
                "validation_metrics_read": False,
                "checkpoint_search_during_refit": False,
                "locked_selection_fold": int(lock["fold"]),
                "locked_selection_held_drugs": lock["selection_held_drugs"],
                "locked_selection_epoch": int(lock["selected_epoch"]),
                "locked_fixed_epochs": epochs,
                "locked_weights": weights,
                "n_fit_rotations": len(x),
                "teacher_seed": _seed(token),
                "final_train_rotation_normalized_mse": result["final_train_rotation_normalized_mse"],
                "state_hash": result["state_hash"],
                "confirmation_treated_loaded": False,
            }
            checkpoint = checkpoint_dir / f"fold{fold}.pt"
            shrink = _shrink(profile, fit_drugs, units)
            checkpoint_sha = _save_checkpoint(checkpoint, {
                "version": VERSION,
                "capacity": selected_capacity,
                "architecture": metadata["architecture"],
                "state_dict": result["state_dict"],
                "support_mean": result["support_mean"],
                "support_scale": result["support_scale"],
                "imceb_shrink": shrink,
                "weights": np.asarray([weights["IMR_MLP"], weights["LSO_equals_IMR"], weights["IMCEB"]], dtype=np.float32),
                "metadata": metadata,
            })
            metadata["checkpoint"] = str(checkpoint.relative_to(args.root).as_posix())
            metadata["checkpoint_sha256"] = checkpoint_sha
            records.append(metadata)
            teacher_model = MLPTeacher(base.old.N_HVG, *CAPACITIES[selected_capacity]).to(device)
            teacher_model.load_state_dict(result["state_dict"])
            teacher_model.eval()
            teachers[fold] = {
                "model": teacher_model,
                "mean": result["support_mean"],
                "scale": result["support_scale"],
                "shrink": shrink,
                "weights": np.asarray([weights["IMR_MLP"], weights["LSO_equals_IMR"], weights["IMCEB"]], dtype=np.float32),
                "device": device,
                "capacity": selected_capacity,
                "metadata": metadata,
            }
            print(json.dumps({"role": "refit", "capacity": selected_capacity, "fold": fold, "fixed_epochs": epochs, "train_mse": result["final_train_rotation_normalized_mse"]}), flush=True)
        target_units, target_values, target_folds = _build_refit_oof_targets(profile, units, refit_assignment, teachers, set(split["refit_drugs"]))
        target_path = args.root / "refit_oof_targets.npz"
        np.savez_compressed(
            target_path,
            drug=np.asarray([drug for drug, _dose in target_units], dtype=str),
            dose_nM=np.asarray([dose for _drug, dose in target_units], dtype=np.float32),
            oof_fold=target_folds,
            cfra_1r_aggregate=target_values,
        )
        target_manifest = {
            "version": VERSION,
            "target": TARGET,
            "surface": SURFACE,
            "selected_capacity": selected_capacity,
            "target_file": str(target_path.relative_to(args.root).as_posix()),
            "target_file_sha256": sha256(target_path),
            "array_name": "cfra_1r_aggregate",
            "unit_order": "PLAN.json units filtered to Fit union Validation (confirmation drugs omitted)",
            "n_target_units": len(target_units),
            "n_hvg": base.old.N_HVG,
            "oof_fold_by_unit": [{"drug": drug, "dose_nM": float(dose), "refit_fold": int(fold)} for (drug, dose), fold in zip(target_units, target_folds)],
            "checkpoint_by_fold": [{"fold": int(row["fold"]), "path": row["checkpoint"], "sha256": row["checkpoint_sha256"], "fixed_epochs": row["locked_fixed_epochs"]} for row in records],
            "aggregation": "mean of frozen teacher predictions for rep1 and rep2 at each (drug,dose)",
            "confirmation_treated_loaded": False,
        }
        write_json(args.root / "P0_TARGET_REFERENCE.json", {
            **target_manifest,
            "consumer": "existing sciplex3_rawagg_cfra_p0_20260915/run_p0.py::fit through a frozen-target cfra_agg adapter",
            "loader": "run_formal_teacher.py::load_refit_teacher",
            "predictor": "run_formal_teacher.py::predict_refit_teacher",
            "student_input_order": "use PLAN.json units and labels to construct old.StudentRows; preserve existing M0/M2 fit_one/student selection logic",
            "no_confirmation_target": True,
        })
        manifest = {
            "version": VERSION,
            "selected_capacity": selected_capacity,
            "architecture": records[0]["architecture"],
            "teachers": records,
            "loader": "run_formal_teacher.py::load_refit_teacher",
            "predictor": "run_formal_teacher.py::predict_refit_teacher",
            "p0_target_reference": "P0_TARGET_REFERENCE.json",
            "confirmation_treated_loaded": False,
            "has_test_stage": False,
        }
        write_json(args.root / "P0_TEACHER_MANIFEST.json", manifest)
        write_json(args.root / "REFIT_COMPLETE.json", {
            "status": "PASS",
            "version": VERSION,
            "selected_capacity": selected_capacity,
            "teacher_count": len(records),
            "checkpoints": [{"path": row["checkpoint"], "sha256": row["checkpoint_sha256"], "fold": row["fold"], "fixed_epochs": row["locked_fixed_epochs"]} for row in records],
            "p0_target_reference": "P0_TARGET_REFERENCE.json",
            "p0_target_reference_sha256": sha256(args.root / "P0_TARGET_REFERENCE.json"),
            "refit_oof_targets": "refit_oof_targets.npz",
            "refit_oof_targets_sha256": sha256(target_path),
            "assignment_relation": plan.get("assignment_relation"),
            "hvg_audit": hvg_audit,
            "source_treated_group_read_count": len(store.read_group_ids),
            "source_control_group_read_count": len(store.read_control_ids),
            "confirmation_treated_group_reads": [],
            "confirmation_treated_loaded": False,
            "validation_read_mode": "train_only",
            "validation_metrics_read": False,
            "checkpoint_search_during_refit": False,
            "has_test_stage": False,
        })
        print(json.dumps({"status": "REFIT_COMPLETE", "selected_capacity": selected_capacity, "teacher_count": len(records), "confirmation_treated_loaded": False}), flush=True)
    finally:
        store.close()


def load_refit_teacher(path: Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    """Load one frozen refit checkpoint for a later P0 target generator."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    capacity = str(payload["capacity"])
    if capacity not in CAPACITIES:
        raise RuntimeError(f"unknown teacher capacity in checkpoint: {capacity}")
    target_device = torch.device(device)
    model = MLPTeacher(base.old.N_HVG, *CAPACITIES[capacity]).to(target_device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {
        "model": model,
        "mean": np.asarray(payload["support_mean"], dtype=np.float32),
        "scale": np.asarray(payload["support_scale"], dtype=np.float32),
        "shrink": np.asarray(payload["imceb_shrink"], dtype=np.float32),
        "weights": np.asarray(payload["weights"], dtype=np.float32),
        "capacity": capacity,
        "metadata": payload.get("metadata", {}),
        "device": target_device,
    }


def predict_refit_teacher(teacher: dict[str, Any], value: np.ndarray) -> np.ndarray:
    """Predict a frozen CFRA teacher output for one or more repeat profiles."""
    raw = np.asarray(value, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[None, :]
    model: MLPTeacher = teacher["model"]
    device = teacher.get("device", next(model.parameters()).device)
    norm = _normalize(raw, teacher["mean"], teacher["scale"])
    imr = _predict_normalized(model, norm, device)
    imceb = _normalize(raw * teacher["shrink"], teacher["mean"], teacher["scale"])
    weights = np.asarray(teacher["weights"], dtype=np.float32)
    if weights.shape != (3,) or np.any(weights < -1e-7) or abs(float(weights.sum()) - 1.0) > 1e-5:
        raise RuntimeError("invalid frozen teacher simplex weights")
    answer_norm = (weights[0] + weights[1]) * imr + weights[2] * imceb
    answer = answer_norm * teacher["scale"] + teacher["mean"]
    if not np.isfinite(answer).all():
        raise RuntimeError("non-finite frozen teacher prediction")
    return answer.astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.stage == "preflight":
        preflight(args)
    elif args.stage == "select":
        select(args)
    elif args.stage == "refit":
        refit(args)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
