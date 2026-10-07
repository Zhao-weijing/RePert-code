#!/usr/bin/env python3
"""Build and evaluate the physical-support CFRA bundle for Figure 4 A--C.

This program deliberately deploys the *already validation-calibrated* cpg
candidate set.  It never fits an ensemble weight or a candidate on test rows.
The test-side output is a historical locked recomputation, not confirmation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


VERSION = "cpg0004-figure4-cfra-physical-support-v1-2026-09-20"
STATUS = "HISTORICAL_LOCKED_RECOMPUTE"
SEEDS = (3407, 42, 2025)
DOSES = ("0.04", "0.12", "0.37", "1.11", "3.33", "10")
CANDIDATES = ("imr", "lso", "imceb")
EXPECTED_WEIGHTS = np.asarray((0.112853534587566, 0.211512376665863, 0.675634088746571), dtype=np.float64)
BOOTSTRAP_ROUNDS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("preflight", "fit", "predict", "evaluate"))
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-lock", required=True, type=Path)
    parser.add_argument("--weights-csv", required=True, type=Path)
    parser.add_argument("--stage-a-model", type=Path)
    parser.add_argument("--stage-c-root", type=Path)
    parser.add_argument("--test-pair-root", type=Path)
    parser.add_argument("--feature-mapping", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap-rounds", type=int, default=BOOTSTRAP_ROUNDS)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_dose(value: Any) -> str:
    try:
        return f"{float(str(value)):.12g}"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid dose: {value!r}") from exc


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in records:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def require_empty_root(root: Path) -> None:
    if root.exists():
        if any(root.iterdir()):
            raise FileExistsError(f"refusing non-empty output root: {root}")
        return
    root.mkdir(parents=True, exist_ok=False)


def load_rows(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = ("compound_id", "dose", "plate", "well", "split", "delta")
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise RuntimeError(f"prepared CP artifact missing fields: {missing}")
        rows = {key: np.asarray(payload[key]) for key in required}
    n = len(rows["compound_id"])
    if any(len(value) != n for value in rows.values()):
        raise RuntimeError("prepared CP artifact field lengths differ")
    if rows["delta"].ndim != 2 or rows["delta"].shape[0] != n or rows["delta"].shape[1] != 242:
        raise RuntimeError(f"unexpected CP delta shape: {rows['delta'].shape}")
    if not np.isfinite(rows["delta"]).all():
        raise FloatingPointError("prepared CP delta has non-finite values")
    rows["compound_id"] = rows["compound_id"].astype(str)
    rows["dose"] = np.asarray([canonical_dose(value) for value in rows["dose"]], dtype=str)
    rows["plate"] = rows["plate"].astype(str)
    rows["well"] = rows["well"].astype(str)
    rows["split"] = rows["split"].astype(str)
    rows["delta"] = rows["delta"].astype(np.float32)
    return rows


def physical_key_audit(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    keys = list(zip(rows["compound_id"], rows["dose"], rows["plate"], rows["well"]))
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate physical (compound,dose,plate,well) key")
    compound_sets = {split: set(rows["compound_id"][rows["split"] == split].tolist()) for split in ("train", "valid", "test")}
    overlaps = {f"{left}_{right}": len(compound_sets[left] & compound_sets[right]) for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    if any(overlaps.values()):
        raise RuntimeError(f"compound split overlap: {overlaps}")
    return {
        "n_rows": len(keys),
        "n_unique_physical_keys": len(set(keys)),
        "feature_dimension": int(rows["delta"].shape[1]),
        "split_rows": {split: int(np.sum(rows["split"] == split)) for split in ("train", "valid", "test")},
        "split_compounds": {split: len(compound_sets[split]) for split in compound_sets},
        "compound_overlap": overlaps,
    }


def validate_static_inputs(args: argparse.Namespace, rows: dict[str, np.ndarray]) -> dict[str, Any]:
    for path in (args.manifest, args.split_lock, args.weights_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = pd.read_csv(args.manifest)
    required_manifest = {"compound_id", "dose", "plate", "well", "split"}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest or len(manifest) != len(rows["compound_id"]):
        raise RuntimeError(f"manifest lacks a one-to-one physical identity mapping: {sorted(missing_manifest)}")
    manifest_dose = np.asarray([canonical_dose(value) for value in manifest.dose], dtype=str)
    manifest_identity_matches = (
        np.array_equal(manifest.compound_id.astype(str).to_numpy(), rows["compound_id"])
        and np.array_equal(manifest_dose, rows["dose"])
        and np.array_equal(manifest.plate.astype(str).to_numpy(), rows["plate"])
        and np.array_equal(manifest.well.astype(str).to_numpy(), rows["well"])
        and np.array_equal(manifest.split.astype(str).to_numpy(), rows["split"])
    )
    if not manifest_identity_matches:
        raise RuntimeError("manifest physical identity order does not exactly match the prepared artifact")
    lock = json.loads(args.split_lock.read_text(encoding="utf-8"))
    if not isinstance(lock, dict):
        raise RuntimeError("split lock is not a JSON object")
    return {
        "data": {"path": str(args.data), "sha256": sha256(args.data), "bytes": args.data.stat().st_size},
        "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest), "bytes": args.manifest.stat().st_size},
        "split_lock": {"path": str(args.split_lock), "sha256": sha256(args.split_lock), "bytes": args.split_lock.stat().st_size},
        "weights_csv": {"path": str(args.weights_csv), "sha256": sha256(args.weights_csv), "bytes": args.weights_csv.stat().st_size},
        "physical": physical_key_audit(rows),
    }


def load_weights(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {"dataset", "budget", "weight_historical_teacher_traininternal", "weight_lso_traininternal", "weight_rceb"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"CFRA weights file lacks columns: {sorted(missing)}")
    rows = frame[(frame.dataset.astype(str) == "cpg0004-LINCS") & (frame.budget.astype(int) == 1)]
    if len(rows) != 1:
        raise RuntimeError(f"expected one cpg0004 1R deployment row, found {len(rows)}")
    row = rows.iloc[0]
    weight = np.asarray((row.weight_historical_teacher_traininternal, row.weight_lso_traininternal, row.weight_rceb), dtype=np.float64)
    if not np.all(np.isfinite(weight)) or np.any(weight < -1e-12) or not np.isclose(weight.sum(), 1.0, rtol=0, atol=1e-10):
        raise RuntimeError(f"invalid CFRA simplex weights: {weight.tolist()}")
    if not np.allclose(weight, EXPECTED_WEIGHTS, rtol=0, atol=1e-12):
        raise RuntimeError(f"unexpected cpg 1R deployment weights: {weight.tolist()}")
    return weight, {
        "dataset": str(row.dataset), "budget": int(row.budget), "candidate_order": ["IMR", "LSO", "IMCEB"],
        "weights": weight.tolist(), "sum": float(weight.sum()), "source_status": str(row.get("status", "")),
    }


def preflight(args: argparse.Namespace) -> None:
    require_empty_root(args.outdir)
    rows = load_rows(args.data)
    inputs = validate_static_inputs(args, rows)
    weight, weight_audit = load_weights(args.weights_csv)
    if not np.allclose(weight, EXPECTED_WEIGHTS, rtol=0, atol=1e-12):
        raise RuntimeError("unreachable weight guard")
    json_write(args.outdir / "PREFLIGHT_COMPLETE.json", {
        "version": VERSION, "status": STATUS, "stage": "preflight", "test_profiles_loaded": False,
        "test_used_for_fit_or_selection": False, "inputs": inputs, "weight": weight_audit,
        "stop_gates": ["artifact_provenance", "partition_leakage", "physical_identity", "numerical_integrity"],
    })


def torch_modules(device_name: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to deploy frozen IMR/LSO checkpoints") from exc
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    return torch, nn, device


def make_model(nn: Any) -> Any:
    return nn.Sequential(
        nn.Linear(242, 128), nn.GELU(), nn.Linear(128, 32), nn.GELU(),
        nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 242),
    )


def load_checkpoint(torch: Any, nn: Any, path: Path, device: Any) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    required = ("state_dict", "input_mean", "input_scale", "target_mean", "target_scale", "meta")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"checkpoint lacks fields {missing}: {path}")
    model = make_model(nn).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    stats = {key: np.asarray(payload[key], dtype=np.float32) for key in ("input_mean", "input_scale", "target_mean", "target_scale")}
    if any(value.shape != (242,) or not np.isfinite(value).all() for value in stats.values()):
        raise RuntimeError(f"invalid checkpoint normalization: {path}")
    if np.any(stats["input_scale"] <= 0) or np.any(stats["target_scale"] <= 0):
        raise RuntimeError(f"non-positive checkpoint scale: {path}")
    model.eval()
    return model, stats, dict(payload["meta"])


def predict_network(torch: Any, model: Any, stats: dict[str, np.ndarray], support: np.ndarray, device: Any) -> np.ndarray:
    values = np.asarray(support, dtype=np.float32)
    result = np.empty_like(values)
    with torch.no_grad():
        for start in range(0, len(values), 256):
            x = (values[start:start + 256] - stats["input_mean"]) / stats["input_scale"]
            y = model(torch.from_numpy(x).to(device)).detach().cpu().numpy()
            result[start:start + len(y)] = y * stats["target_scale"] + stats["target_mean"]
    if not np.isfinite(result).all():
        raise FloatingPointError("frozen neural candidate produced non-finite values")
    return result.astype(np.float32)


def rceb_predict(model_path: Path, support: np.ndarray) -> np.ndarray:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    with np.load(model_path, allow_pickle=False) as payload:
        required = ("mean", "cholesky_noise", "generalized_eigenvalues", "generalized_eigenvectors")
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise RuntimeError(f"IMCEB model missing {missing}")
        mean = np.asarray(payload["mean"], dtype=np.float64)
        chol = np.asarray(payload["cholesky_noise"], dtype=np.float64)
        eigenvalues = np.asarray(payload["generalized_eigenvalues"], dtype=np.float64)
        eigenvectors = np.asarray(payload["generalized_eigenvectors"], dtype=np.float64)
    if mean.shape != (242,) or chol.shape != (242, 242) or eigenvectors.shape != (242, 242):
        raise RuntimeError("unexpected IMCEB geometry dimensions")
    whitened = np.linalg.solve(chol, (np.asarray(support, dtype=np.float64) - mean).T)
    rotated = eigenvectors.T @ whitened
    gain = np.clip(eigenvalues, 0.0, None) / (np.clip(eigenvalues, 0.0, None) + 1.0)
    output = mean[:, None] + chol @ (eigenvectors @ (gain[:, None] * rotated))
    output = output.T.astype(np.float32)
    if not np.isfinite(output).all():
        raise FloatingPointError("IMCEB produced non-finite values")
    return output


def fit(args: argparse.Namespace) -> None:
    preflight_path = args.outdir / "PREFLIGHT_COMPLETE.json"
    if not preflight_path.is_file() or (args.outdir / "FIT_COMPLETE.json").exists():
        raise RuntimeError("fit requires a fresh completed preflight")
    if args.stage_a_model is None or args.stage_c_root is None:
        raise ValueError("fit requires --stage-a-model and --stage-c-root")
    preflight_audit = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight_audit.get("test_profiles_loaded") is not False:
        raise RuntimeError("preflight did not preserve test isolation")
    torch, nn, device = torch_modules(args.device)
    teacher_path = args.stage_c_root / "historical_teacher_traininternal.pt"
    lso_path = args.stage_c_root / "lso_traininternal.pt"
    teacher, _teacher_stats, teacher_meta = load_checkpoint(torch, nn, teacher_path, device)
    lso, _lso_stats, lso_meta = load_checkpoint(torch, nn, lso_path, device)
    del teacher, lso
    weight, weight_audit = load_weights(args.weights_csv)
    json_write(args.outdir / "FIT_COMPLETE.json", {
        "version": VERSION, "status": STATUS, "stage": "frozen_candidate_deployment", "test_profiles_loaded": False,
        "test_used_for_fit_or_selection": False, "device": str(device), "candidate_order": ["IMR", "LSO", "IMCEB"],
        "weight": weight_audit, "weight_sum": float(weight.sum()),
        "candidate_checkpoints": {
            "IMR": {"path": str(teacher_path), "sha256": sha256(teacher_path), "meta": teacher_meta},
            "LSO": {"path": str(lso_path), "sha256": sha256(lso_path), "meta": lso_meta},
            "IMCEB": {"path": str(args.stage_a_model), "sha256": sha256(args.stage_a_model)},
        },
        "candidate_fit_provenance": "immutable Stage-C train-internal checkpoints and Stage-A train-only IMCEB geometry; no refit in this package",
    })


def condition_rows(rows: dict[str, np.ndarray]) -> dict[tuple[str, str], list[int]]:
    output: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (compound, dose) in enumerate(zip(rows["compound_id"], rows["dose"])):
        if rows["split"][index] == "test":
            output[(str(compound), str(dose))].append(index)
    return output


def support_rotations(pair_path: Path, rows: dict[str, np.ndarray]) -> dict[tuple[str, str], tuple[int, int]]:
    if not pair_path.is_file():
        raise FileNotFoundError(pair_path)
    frame = pd.read_csv(pair_path)
    required = {"compound", "dose", "held_row", "support_rows"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"test pair manifest missing fields {sorted(missing)}: {pair_path}")
    output: dict[tuple[str, str], tuple[int, int]] = {}
    for (compound, dose), group in frame.groupby(["compound", "dose"], sort=True):
        supports: set[int] = set()
        for value in group.support_rows.astype(str):
            values = [item for item in value.replace("|", ";").split(";") if item]
            if len(values) != 1:
                raise RuntimeError(f"1R pair manifest carries non-single support: {compound}|{dose}|{value}")
            supports.add(int(values[0]))
        key = (str(compound), canonical_dose(dose))
        if len(supports) != 2:
            raise RuntimeError(f"expected exactly two historical support rotations for {key}, found {sorted(supports)}")
        rotations = tuple(sorted(supports))
        for support in rotations:
            if support < 0 or support >= len(rows["compound_id"]):
                raise RuntimeError(f"support row out of range: {support}")
            if rows["split"][support] != "test" or rows["compound_id"][support] != key[0] or rows["dose"][support] != key[1]:
                raise RuntimeError(f"support identity mismatch for {key} at row {support}")
        output[key] = rotations
    if not output:
        raise RuntimeError(f"empty pair manifest: {pair_path}")
    return output


def predict(args: argparse.Namespace) -> None:
    fit_path = args.outdir / "FIT_COMPLETE.json"
    if not fit_path.is_file() or (args.outdir / "PREDICT_COMPLETE.json").exists():
        raise RuntimeError("predict requires frozen fit and may run only once")
    if args.test_pair_root is None or args.stage_a_model is None or args.stage_c_root is None:
        raise ValueError("predict requires --test-pair-root, --stage-a-model and --stage-c-root")
    fit_audit = json.loads(fit_path.read_text(encoding="utf-8"))
    if fit_audit.get("test_profiles_loaded") is not False or fit_audit.get("test_used_for_fit_or_selection") is not False:
        raise RuntimeError("fit audit does not establish test isolation")
    rows = load_rows(args.data)
    groups = condition_rows(rows)
    torch, nn, device = torch_modules(args.device)
    imr, imr_stats, _ = load_checkpoint(torch, nn, args.stage_c_root / "historical_teacher_traininternal.pt", device)
    lso, lso_stats, _ = load_checkpoint(torch, nn, args.stage_c_root / "lso_traininternal.pt", device)
    weight, _ = load_weights(args.weights_csv)
    all_records: list[dict[str, Any]] = []
    per_seed_payload: dict[str, dict[str, np.ndarray]] = {}
    rotation_counts: dict[str, dict[str, int]] = {}
    for seed in SEEDS:
        pair_path = args.test_pair_root / f"full_b1_all_seed{seed}" / "test_pair_manifest.csv"
        rotations = support_rotations(pair_path, rows)
        selected: list[tuple[str, str, int, int, list[int]]] = []
        unavailable_conditions = 0
        for (compound, dose), support_rows in sorted(rotations.items()):
            physical = sorted(groups.get((compound, dose), []))
            if len(physical) != 5:
                # Fixed before any profile prediction: Figure 4 requires one
                # support and four disjoint raw physical reference rows.
                unavailable_conditions += 1
                continue
            for rotation, support in enumerate(support_rows):
                reference_rows = [row for row in physical if row != support]
                if len(reference_rows) != 4 or support in reference_rows:
                    raise RuntimeError(f"support/reference overlap for {(compound, dose, support)}")
                selected.append((compound, dose, rotation, support, reference_rows))
        if not selected:
            raise RuntimeError(f"no strict five-row Figure 4 conditions for seed {seed}")
        support = rows["delta"][np.asarray([item[3] for item in selected], dtype=np.int64)]
        imr_prediction = predict_network(torch, imr, imr_stats, support, device)
        lso_prediction = predict_network(torch, lso, lso_stats, support, device)
        imceb_prediction = rceb_predict(args.stage_a_model, support)
        cfra_prediction = (weight[0] * imr_prediction + weight[1] * lso_prediction + weight[2] * imceb_prediction).astype(np.float32)
        reference = np.vstack([rows["delta"][np.asarray(item[4], dtype=np.int64)].mean(axis=0, dtype=np.float64) for item in selected]).astype(np.float32)
        if not all(np.isfinite(value).all() for value in (support, imr_prediction, lso_prediction, imceb_prediction, cfra_prediction, reference)):
            raise FloatingPointError(f"non-finite profile in seed {seed}")
        per_seed_payload[str(seed)] = {
            "compound": np.asarray([item[0] for item in selected], dtype=str), "dose": np.asarray([item[1] for item in selected], dtype=str),
            "rotation": np.asarray([item[2] for item in selected], dtype=np.int8), "support_row": np.asarray([item[3] for item in selected], dtype=np.int64),
            "reference_rows": np.asarray(["|".join(str(row) for row in item[4]) for item in selected], dtype=str),
            "raw": support, "imr": imr_prediction, "lso": lso_prediction, "imceb": imceb_prediction, "cfra": cfra_prediction, "reference": reference,
        }
        rotation_counts[str(seed)] = {
            "eligible_conditions": int(len(selected) // 2),
            "physical_support_views": int(len(selected)),
            "excluded_conditions_not_exactly_five_physical_profiles": int(unavailable_conditions),
        }
        for pos, item in enumerate(selected):
            all_records.append({"seed": seed, "compound": item[0], "dose": item[1], "rotation": item[2], "support_row": item[3],
                                "support_plate": rows["plate"][item[3]], "support_well": rows["well"][item[3]], "reference_rows": "|".join(str(row) for row in item[4]),
                                "reference_plates": "|".join(rows["plate"][item[4]].tolist()), "reference_excludes_support": True, "finite": True, "profile_index": pos})
    np.savez_compressed(args.outdir / "CFRA_PER_PHYSICAL_VIEW.npz", **{f"seed{seed}_{key}": value for seed, bundle in per_seed_payload.items() for key, value in bundle.items()})
    csv_write(args.outdir / "CFRA_PER_PHYSICAL_VIEW_MANIFEST.csv", all_records)
    aggregates: list[dict[str, Any]] = []
    for seed, bundle in per_seed_payload.items():
        frame = pd.DataFrame({"compound": bundle["compound"], "dose": bundle["dose"], "rotation": bundle["rotation"], "profile_index": np.arange(len(bundle["compound"]))})
        for (compound, dose), group in frame.groupby(["compound", "dose"], sort=True):
            if sorted(group.rotation.astype(int).tolist()) != [0, 1]:
                raise RuntimeError(f"rotation aggregate is incomplete for {seed}|{compound}|{dose}")
            indices = group.profile_index.to_numpy(dtype=int)
            aggregates.append({"seed": int(seed), "compound": compound, "dose": dose, "raw": bundle["raw"][indices].mean(axis=0), "cfra": bundle["cfra"][indices].mean(axis=0), "reference": bundle["reference"][indices].mean(axis=0)})
    np.savez_compressed(args.outdir / "CFRA_POST_VIEW_AGGREGATES.npz",
                        seed=np.asarray([row["seed"] for row in aggregates], dtype=np.int32), compound=np.asarray([row["compound"] for row in aggregates], dtype=str), dose=np.asarray([row["dose"] for row in aggregates], dtype=str),
                        raw=np.vstack([row["raw"] for row in aggregates]).astype(np.float32), cfra=np.vstack([row["cfra"] for row in aggregates]).astype(np.float32), reference=np.vstack([row["reference"] for row in aggregates]).astype(np.float32))
    json_write(args.outdir / "PREDICT_COMPLETE.json", {
        "version": VERSION, "status": STATUS, "stage": "predict", "test_profiles_loaded": True, "test_used_for_fit_or_selection": False,
        "candidate_order": ["IMR", "LSO", "IMCEB"], "weights": weight.tolist(), "per_seed_physical_views": rotation_counts,
        "aggregation": "mean_r CFRA_1R(Y_r) after independent physical-support inference; prohibited CFRA_1R(mean_r Y_r) absent",
        "reference": "mean of four raw physical rows excluding each selected support", "bootstrap_unit": "compound",
    })


def fisher(value: float) -> float:
    if not np.isfinite(value) or value <= -1.0 or value >= 1.0:
        return math.nan
    return float(np.arctanh(value))


def pcc(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.size < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return math.nan
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return math.nan if denominator <= 0 else float(np.dot(a, b) / denominator)


def rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy(dtype=float)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return pcc(rank(left), rank(right))


def top_indices(values: np.ndarray, count: int) -> np.ndarray:
    return np.lexsort((np.arange(len(values)), -np.abs(np.asarray(values, dtype=float))))[:count]


def read_profiles(path: Path) -> dict[int, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        result: dict[int, dict[str, np.ndarray]] = {}
        for seed in SEEDS:
            prefix = f"seed{seed}_"
            names = ("compound", "dose", "rotation", "support_row", "raw", "cfra", "reference")
            if any(prefix + name not in payload.files for name in names):
                raise RuntimeError(f"profile bundle lacks seed {seed}")
            result[seed] = {name: np.asarray(payload[prefix + name]) for name in names}
    return result


def paired_summary(records: pd.DataFrame, endpoint: str, rounds: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frame = records[records.endpoint.eq(endpoint)].copy()
    paired: list[dict[str, Any]] = []
    for compound, group in frame.groupby("compound", sort=True):
        raw, cfra = group.raw.to_numpy(dtype=float), group.cfra.to_numpy(dtype=float)
        if not np.isfinite(raw).all() or not np.isfinite(cfra).all():
            continue
        paired.append({"compound": compound, "raw": float(raw.mean()), "cfra": float(cfra.mean()), "cfra_minus_raw": float((cfra - raw).mean())})
    values = pd.DataFrame(paired)
    if values.empty:
        raise RuntimeError(f"no finite compound pairs for {endpoint}")
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{VERSION}|{endpoint}".encode()).digest()[:8], "little"))
    draws = rng.integers(0, len(values), size=(rounds, len(values)))
    bootstrap = values.cfra_minus_raw.to_numpy(dtype=float)[draws].mean(axis=1)
    return {"endpoint": endpoint, "raw": float(values.raw.mean()), "cfra": float(values.cfra.mean()), "cfra_minus_raw": float(values.cfra_minus_raw.mean()),
            "ci_low": float(np.quantile(bootstrap, .025)), "ci_high": float(np.quantile(bootstrap, .975)), "n_compounds": int(len(values)),
            "bootstrap_unit": "compound", "bootstrap_rounds": rounds, "status": STATUS}, paired


def evaluate(args: argparse.Namespace) -> None:
    predict_path = args.outdir / "PREDICT_COMPLETE.json"
    bundle_path = args.outdir / "CFRA_PER_PHYSICAL_VIEW.npz"
    if not predict_path.is_file() or not bundle_path.is_file() or (args.outdir / "EVALUATE_COMPLETE.json").exists():
        raise RuntimeError("evaluate requires a completed prediction bundle and may run only once")
    if args.feature_mapping is None or not args.feature_mapping.is_file():
        raise ValueError("evaluate requires --feature-mapping")
    if args.bootstrap_rounds < 100:
        raise ValueError("bootstrap rounds must be at least 100")
    profiles = read_profiles(bundle_path)
    mapping = pd.read_csv(args.feature_mapping)
    required = {"feature_index", "biological_endpoint", "biological_module"}
    if required - set(mapping.columns):
        raise RuntimeError("feature mapping lacks Figure 4 fields")
    active = mapping[mapping.biological_endpoint.astype(str).eq("Eligible")].copy()
    biological = active.feature_index.astype(int).to_numpy()
    modules = {name: group.feature_index.astype(int).to_numpy() for name, group in active.groupby("biological_module", sort=True) if name and len(group) >= 8}
    expected_modules = {"DNA", "RNA", "ER", "Mito", "AGP", "Shape", "Cross-channel"}
    if set(modules) != expected_modules or len(biological) != 241:
        raise RuntimeError(f"frozen feature-module gate changed: { {key: len(value) for key, value in modules.items()} }")
    rows = load_rows(args.data)
    train = rows["delta"][rows["split"] == "train"]
    center = np.median(train, axis=0)
    scale = 1.4826 * np.median(np.abs(train - center), axis=0)
    fallback = train.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = fallback[~np.isfinite(scale) | (scale < 1e-6)]
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    records: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for seed, values in profiles.items():
        count = len(values["compound"])
        for index in range(count):
            compound, dose, rotation = str(values["compound"][index]), str(values["dose"][index]), int(values["rotation"][index])
            raw, cfra, reference = values["raw"][index], values["cfra"][index], values["reference"][index]
            for name, indices in modules.items():
                records.append({"seed": seed, "compound": compound, "dose": dose, "rotation": rotation, "endpoint": f"A_module_fisher_z_{name}", "raw": fisher(pcc(raw[indices], reference[indices])), "cfra": fisher(pcc(cfra[indices], reference[indices]))})
            raw_z, cfra_z = (raw[biological] - center[biological]) / scale[biological], (cfra[biological] - center[biological]) / scale[biological]
            ref_z = (reference[biological] - center[biological]) / scale[biological]
            reference_top = top_indices(ref_z, 25)
            records.append({"seed": seed, "compound": compound, "dose": dose, "rotation": rotation, "endpoint": "B_top25_overlap", "raw": float(len(set(top_indices(raw_z, 25)) & set(reference_top)) / 25), "cfra": float(len(set(top_indices(cfra_z, 25)) & set(reference_top)) / 25)})
            records.append({"seed": seed, "compound": compound, "dose": dose, "rotation": rotation, "endpoint": "B_top25_direction", "raw": float(np.mean(np.sign(raw_z[reference_top]) == np.sign(ref_z[reference_top]))), "cfra": float(np.mean(np.sign(cfra_z[reference_top]) == np.sign(ref_z[reference_top])) )})
            records.append({"seed": seed, "compound": compound, "dose": dose, "rotation": rotation, "endpoint": "D_full_profile_fisher_z", "raw": fisher(pcc(raw[biological], reference[biological])), "cfra": fisher(pcc(cfra[biological], reference[biological]))})
            records.append({"seed": seed, "compound": compound, "dose": dose, "rotation": rotation, "endpoint": "D_norm_ratio", "raw": float(np.linalg.norm(raw[biological]) / max(np.linalg.norm(reference[biological]), 1e-12)), "cfra": float(np.linalg.norm(cfra[biological]) / max(np.linalg.norm(reference[biological]), 1e-12))})
        table = pd.DataFrame({"compound": values["compound"].astype(str), "dose": values["dose"].astype(str), "rotation": values["rotation"].astype(int), "profile_index": np.arange(count)})
        for (compound, rotation), group in table.groupby(["compound", "rotation"], sort=True):
            by_dose = {str(row.dose): int(row.profile_index) for row in group.itertuples(index=False)}
            if set(by_dose) != set(DOSES):
                continue
            ordered = [by_dose[dose] for dose in DOSES]
            for module, indices in modules.items():
                def distances(name: str) -> np.ndarray:
                    profile = values[name][ordered][:, indices]
                    return np.asarray([1.0 - pcc(profile[left], profile[right]) for left in range(6) for right in range(left + 1, 6)], dtype=float)
                raw_dist, cfra_dist, reference_dist = distances("raw"), distances("cfra"), distances("reference")
                trajectory_rows.append({"seed": seed, "compound": compound, "rotation": rotation, "module": module, "raw": spearman(raw_dist, reference_dist), "cfra": spearman(cfra_dist, reference_dist)})
    trajectory = pd.DataFrame(trajectory_rows)
    if trajectory.empty:
        raise RuntimeError("no complete six-dose trajectories; do not substitute a smaller dose set")
    for (seed, compound, rotation), group in trajectory.groupby(["seed", "compound", "rotation"], sort=True):
        if set(group.module) != expected_modules:
            continue
        records.append({"seed": seed, "compound": compound, "dose": "all_6_doses", "rotation": rotation, "endpoint": "C_macro_trajectory_spearman", "raw": float(group.raw.mean()), "cfra": float(group.cfra.mean())})
    frame = pd.DataFrame(records)
    frame["cfra_minus_raw"] = frame.cfra - frame.raw
    summaries: list[dict[str, Any]] = []
    compound_rows: list[dict[str, Any]] = []
    for endpoint in sorted(frame.endpoint.unique()):
        summary, paired = paired_summary(frame, endpoint, args.bootstrap_rounds)
        summaries.append(summary)
        compound_rows.extend({"endpoint": endpoint, **row} for row in paired)
    csv_write(args.outdir / "FIGURE4_CFRA_PER_VIEW_METRICS.csv", frame.to_dict("records"))
    csv_write(args.outdir / "FIGURE4_CFRA_PER_COMPOUND.csv", compound_rows)
    csv_write(args.outdir / "FIGURE4_CFRA_SUMMARY.csv", summaries)
    json_write(args.outdir / "EVALUATE_COMPLETE.json", {
        "version": VERSION, "status": STATUS, "stage": "evaluate", "test_used_for_fit_or_selection": False,
        "arms": ["Raw physical support", "CFRA physical-support aggregate"], "candidate_order": ["IMR", "LSO", "IMCEB"],
        "aggregation": "metrics are per physical support view before averaging within compound", "bootstrap_unit": "compound", "bootstrap_rounds": args.bootstrap_rounds,
        "panels": {"A": "seven predefined module Fisher-z agreements", "B": "reference-defined Top-25 overlap and direction", "C": "six-dose, 15-pair trajectory Spearman across seven modules", "D": "full-profile agreement and amplitude diagnostics; not the legacy annotation retrieval panel"},
        "n_endpoints": len(summaries), "feature_mapping_sha256": sha256(args.feature_mapping),
    })


def main() -> None:
    args = parse_args()
    if args.stage == "preflight":
        preflight(args)
    elif args.stage == "fit":
        fit(args)
    elif args.stage == "predict":
        predict(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
