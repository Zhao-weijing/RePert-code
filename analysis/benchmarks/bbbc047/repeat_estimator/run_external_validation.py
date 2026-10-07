#!/usr/bin/env python3
"""Validation-only same-space external baselines for frozen BBBC047 roles.

This stage loads only train and validation CP profile values.  It refuses a
role freeze with a changed protocol/manifest hash and writes ``test_loaded``
as false in every completion marker.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.stats import rankdata


VERSION = "BBBC047-repeat-estimator-external-validation-v1-2026-09-16"
CP_DIM = 775
RANKS = (16, 32, 64, 128, 256, 384, 512)
RIDGE_ALPHAS = tuple(float(10**power) for power in range(-4, 5))
BUDGETS = (1, 2, 3)
DEFAULT_ROWS = Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rows-root", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def dose_key(value: Any) -> str:
    text = decode(value).strip()
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return text


def condition_id(compound: str, dose: str) -> str:
    return f"{compound}::{dose}"


def parse_condition_id(value: str) -> tuple[str, str]:
    compound, dose = str(value).rsplit("::", 1)
    return compound, dose


def load_profiles(rows_root: Path, split: str) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Load one permitted split and reduce only technical rows within plate."""
    path = rows_root / f"{split}_cp_plate_rows.npz"
    with np.load(path, allow_pickle=False) as payload:
        required = {"smiles", "dose", "plate", "delta"}
        missing = required - set(payload.files)
        if missing:
            raise RuntimeError(f"missing {sorted(missing)} in {path}")
        smiles, dose, plate = payload["smiles"], payload["dose"], payload["plate"]
        delta = np.asarray(payload["delta"], dtype=np.float32)
    if delta.ndim != 2 or delta.shape[1] != CP_DIM or not np.isfinite(delta).all():
        raise RuntimeError(f"invalid 775D finite CP matrix: {path}: {delta.shape}")
    grouped: dict[tuple[str, str], dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for compound_raw, dose_raw, plate_raw, value in zip(smiles, dose, plate, delta):
        grouped[(decode(compound_raw), dose_key(dose_raw))][decode(plate_raw)].append(value)
    profiles: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, by_plate in grouped.items():
        profiles[key] = {plate_id: np.asarray(np.mean(values, axis=0), dtype=np.float32) for plate_id, values in by_plate.items()}
    return profiles


def read_roles(root: Path, split: str, budget: int) -> list[dict[str, str]]:
    manifest = root / "FROZEN_ROLE_MANIFEST.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle) if row["split"] == split and int(row["budget"]) == budget and row["foreign_ok"] == "True"]
    if len(rows) < 500:
        raise RuntimeError(f"insufficient foreign-eligible {split} role rows at {budget}R: {len(rows)}")
    return rows


def check_freeze(roles_root: Path, protocol: Path) -> dict[str, Any]:
    audit_path = roles_root / "ROLE_FREEZE_AUDIT.json"
    manifest = roles_root / "FROZEN_ROLE_MANIFEST.csv"
    if not protocol.is_file() or not audit_path.is_file() or not manifest.is_file():
        raise FileNotFoundError("protocol, role audit and manifest are required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("test_profile_values_loaded") is not False:
        raise RuntimeError("role freeze lacks a PASS metadata-only audit")
    if audit.get("protocol_sha256") != sha256_file(protocol):
        raise RuntimeError("protocol changed after role freeze; create a fresh freeze")
    if audit.get("manifest", {}).get("sha256") != sha256_file(manifest):
        raise RuntimeError("role manifest hash changed after role freeze")
    return audit


def assemble(rows: list[dict[str, str]], profiles: dict[tuple[str, str], dict[str, np.ndarray]], *, foreign: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    support, held, compound, foreign_profiles, conditions, raw_support = [], [], [], [], [], []
    for row in rows:
        key = (row["compound_id"], row["dose"])
        by_plate = profiles.get(key)
        if by_plate is None:
            raise RuntimeError(f"role condition absent from loaded split: {key}")
        support_ids = tuple(row["support_plate_ids"].split("|"))
        held_id = row["held_plate_id"]
        if held_id in support_ids or len(set(support_ids)) != len(support_ids):
            raise RuntimeError(f"support/held overlap in role manifest: {row['condition_id']}")
        if any(plate not in by_plate for plate in (*support_ids, held_id)):
            raise RuntimeError(f"role plate absent from profile source: {row['condition_id']}")
        mean = np.asarray(np.mean([by_plate[plate] for plate in support_ids], axis=0), dtype=np.float32)
        support.append(mean)
        raw_support.append(np.stack([by_plate[plate] for plate in support_ids]).astype(np.float32))
        held.append(by_plate[held_id])
        compound.append(row["compound_id"])
        conditions.append(row["condition_id"])
        if foreign:
            donors = tuple(row["foreign_condition_ids"].split("|"))
            foreign_held = tuple(row["foreign_held_plate_ids"].split("|"))
            if len(donors) != 32 or len(foreign_held) != 32:
                raise RuntimeError(f"foreign identity count changed: {row['condition_id']}")
            values: list[np.ndarray] = []
            for donor_id, plate_id in zip(donors, foreign_held):
                donor = profiles.get(parse_condition_id(donor_id))
                if donor is None or plate_id not in donor:
                    raise RuntimeError(f"foreign profile absent: {donor_id}/{plate_id}")
                values.append(donor[plate_id])
            foreign_profiles.append(np.stack(values).astype(np.float32))
    return (
        np.stack(support).astype(np.float32),
        np.stack(held).astype(np.float32),
        np.asarray(compound, dtype=str),
        foreign_profiles,
        np.asarray(conditions, dtype=str),
        np.stack(raw_support).astype(np.float32),
    )


def rowwise_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    a = left - left.mean(axis=1, keepdims=True)
    b = right - right.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    result = np.full(len(left), np.nan, dtype=np.float64)
    good = denom > 1e-12
    result[good] = np.sum(a[good] * b[good], axis=1) / denom[good]
    return np.clip(result, -0.999999, 0.999999)


def modz(raw_support: np.ndarray) -> np.ndarray:
    """CMap MODZ: Spearman correlation, [0.01, 1] clipping, zero diagonal."""
    output = np.empty((len(raw_support), CP_DIM), dtype=np.float32)
    for index, values in enumerate(raw_support):
        n = len(values)
        if n == 1:
            output[index] = values[0]
            continue
        ranked = np.vstack([rankdata(value, method="average") for value in values])
        corr = np.corrcoef(ranked)
        corr[~np.isfinite(corr)] = 0.01
        corr = np.clip(corr, 0.01, 1.0)
        np.fill_diagonal(corr, 0.0)
        weights = corr.sum(axis=1)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 1e-12:
            output[index] = values.mean(axis=0)
        else:
            output[index] = np.asarray(weights @ values / total, dtype=np.float32)
    return output


def fisher(value: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(value, -0.999999, 0.999999))


def method_metrics(prediction: np.ndarray, held: np.ndarray, foreign: list[np.ndarray], support: np.ndarray) -> dict[str, np.ndarray]:
    same_pcc = rowwise_corr(prediction, held)
    foreign_z = np.empty(len(prediction), dtype=np.float64)
    for index, references in enumerate(foreign):
        repeated = np.repeat(prediction[index : index + 1], len(references), axis=0)
        foreign_z[index] = float(np.mean(fisher(rowwise_corr(repeated, references))))
    same_z = fisher(same_pcc)
    target_top = np.argpartition(np.abs(held), -25, axis=1)[:, -25:]
    pred_top = np.argpartition(np.abs(prediction), -25, axis=1)[:, -25:]
    overlap = np.asarray([len(set(a.tolist()) & set(b.tolist())) / 25.0 for a, b in zip(target_top, pred_top)], dtype=np.float64)
    direction = np.asarray([np.mean(np.sign(prediction[i, target_top[i]]) == np.sign(held[i, target_top[i]])) for i in range(len(prediction))], dtype=np.float64)
    norm_ratio = np.linalg.norm(prediction, axis=1) / np.maximum(np.linalg.norm(support, axis=1), 1e-12)
    return {"E": same_z - foreign_z, "z_same": same_z, "z_foreign": foreign_z, "pcc": same_pcc, "top25_overlap": overlap, "top25_direction": direction, "norm_ratio": norm_ratio}


def compound_means(values: np.ndarray, compounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for compound, value in zip(compounds, values):
        if np.isfinite(value):
            grouped[str(compound)].append(float(value))
    keys = np.asarray(sorted(grouped), dtype=str)
    return keys, np.asarray([np.mean(grouped[key]) for key in keys], dtype=np.float64)


def score_row(method: str, budget: int, metrics: dict[str, np.ndarray], compounds: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {"version": VERSION, "split": "valid", "budget": budget, "method": method, "n_conditions": len(compounds), "n_foreign_per_condition": 32}
    for name, values in metrics.items():
        ids, averaged = compound_means(values, compounds)
        row[f"{name}_mean"] = float(np.mean(averaged))
        row[f"{name}_n_compounds"] = int(len(ids))
    return row


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


def main() -> None:
    args = parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite {args.outdir}")
    role_audit = check_freeze(args.roles_root, args.protocol)
    # This is the only profile-value I/O in this script; test is never named.
    train_profiles = load_profiles(args.rows_root, "train")
    valid_profiles = load_profiles(args.rows_root, "valid")
    args.outdir.mkdir(parents=True)
    summary: list[dict[str, Any]] = []
    rank_curve: list[dict[str, Any]] = []
    ridge_curve: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    per_condition: list[dict[str, Any]] = []
    for budget in BUDGETS:
        train_roles = read_roles(args.roles_root, "train", budget)
        valid_roles = read_roles(args.roles_root, "valid", budget)
        train_x, _, _, _, _, _ = assemble(train_roles, train_profiles, foreign=False)
        valid_x, valid_held, compounds, valid_foreign, conditions, valid_raw_support = assemble(valid_roles, valid_profiles, foreign=True)
        base_predictions: dict[str, np.ndarray] = {
            "Mean": valid_x,
            "Median": np.median(valid_raw_support, axis=1).astype(np.float32),
            "MODZ": modz(valid_raw_support),
        }
        pca = PCA(n_components=min(max(RANKS), len(train_x), CP_DIM), svd_solver="randomized", random_state=3407)
        pca.fit(train_x)
        pca_candidates: dict[int, np.ndarray] = {}
        for rank in RANKS:
            if rank > len(pca.components_):
                continue
            prediction = (pca.mean_ + ((valid_x - pca.mean_) @ pca.components_[:rank].T) @ pca.components_[:rank]).astype(np.float32)
            pca_candidates[rank] = prediction
            point = score_row("PCA", budget, method_metrics(prediction, valid_held, valid_foreign, valid_x), compounds)
            rank_curve.append({"budget": budget, "rank": rank, "validation_E_mean": point["E_mean"], "n_compounds": point["E_n_compounds"]})
        rank = max(pca_candidates, key=lambda value: (next(row["validation_E_mean"] for row in rank_curve if row["budget"] == budget and row["rank"] == value), -value))
        base_predictions["PCA"] = pca_candidates[rank]
        x_mean, x_scale = train_x.mean(axis=0), np.maximum(train_x.std(axis=0), 1e-6)
        train_z, valid_z = (train_x - x_mean) / x_scale, (valid_x - x_mean) / x_scale
        # The independent held response is the sole supervised training label.
        _, train_y, _, _, _, _ = assemble(train_roles, train_profiles, foreign=False)
        ridge_candidates: dict[float, np.ndarray] = {}
        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
            model.fit(train_z, train_y)
            prediction = np.asarray(model.predict(valid_z), dtype=np.float32)
            ridge_candidates[alpha] = prediction
            point = score_row("Ridge", budget, method_metrics(prediction, valid_held, valid_foreign, valid_x), compounds)
            ridge_curve.append({"budget": budget, "alpha": alpha, "validation_E_mean": point["E_mean"], "n_compounds": point["E_n_compounds"]})
        alpha = max(ridge_candidates, key=lambda value: (next(row["validation_E_mean"] for row in ridge_curve if row["budget"] == budget and row["alpha"] == value), -value))
        base_predictions["Ridge"] = ridge_candidates[alpha]
        selected[str(budget)] = {"pca_rank": int(rank), "ridge_alpha": float(alpha), "fit_data": "train only", "selection_data": "validation only"}
        for method, prediction in base_predictions.items():
            metrics = method_metrics(prediction, valid_held, valid_foreign, valid_x)
            summary.append(score_row(method, budget, metrics, compounds))
            for index, condition in enumerate(conditions):
                per_condition.append({"budget": budget, "method": method, "compound_id": compounds[index], "condition_id": condition, **{name: values[index] for name, values in metrics.items()}})
    write_csv(args.outdir / "VALIDATION_EXTERNAL_SUMMARY.csv", summary)
    write_csv(args.outdir / "VALIDATION_PCA_RANK_CURVE.csv", rank_curve)
    write_csv(args.outdir / "VALIDATION_RIDGE_ALPHA_CURVE.csv", ridge_curve)
    write_csv(args.outdir / "VALIDATION_EXTERNAL_PER_CONDITION.csv", per_condition)
    marker = {
        "version": VERSION,
        "stage": "external_validation",
        "protocol_sha256": sha256_file(args.protocol),
        "role_manifest_sha256": role_audit["manifest"]["sha256"],
        "selected_parameters": selected,
        "test_loaded": False,
        "test_profile_values_loaded": False,
        "test_used_for_selection": False,
        "selection_complete": False,
        "reason_selection_incomplete": "Noise2Self and dose-aware IMR/LSO/IMCEB/CFRA have not yet been fit on this manifest.",
        "status": "PASS",
    }
    (args.outdir / "EXTERNAL_VALIDATION_COMPLETE.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": marker["stage"], "test_loaded": False, "selected_parameters": selected}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
