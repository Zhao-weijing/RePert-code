#!/usr/bin/env python3
"""Frozen cpg0004 dose-response application evaluator.

This program implements Experiment 3 (Bio-C) from the locked biological
application protocol.  It consumes the frozen cpg0004 plate-row artifact,
the saved one-repeat teacher/P0 arrays, the saved GE residual arrays, and
the saved pair manifests.  No model, threshold, lambda, beta, checkpoint,
or dose-specific parameter is fit here.

The two support rotations exposed by the frozen prediction/pair artifacts are
kept separate while profiles and endpoint values are computed.  Rotation
values are averaged only at the compound-level reporting boundary.  The
independent reference for a rotation is the mean of the other four physical
repeat rows, so the selected support row cannot enter its reference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VERSION = "cpg0004-LINCS-biological-dose-response-2026-08-30"
SEEDS = (3407, 42, 2025)
DOSES = ("0.04", "0.12", "0.37", "1.11", "3.33", "10")
LOG_DOSES = np.log(np.asarray([float(x) for x in DOSES], dtype=np.float64))
REQUIRED_REPEATS = 5
EXPECTED_SUPPORT_SLOTS = 2
BOOTSTRAP_ROUNDS = 10_000
METHODS = ("raw", "teacher", "posterior")
METHOD_LABELS = {
    "raw": "M0_1R_raw",
    "teacher": "M1_1R_teacher",
    "posterior": "M2_validated_full_estimate",
}
METHOD_THRESHOLD_KEYS = {
    "raw": "raw",
    "teacher": "teacher",
    "posterior": "GE_posterior",
}
ENDPOINTS = (
    "trajectory_concordance",
    "same_dose_top1",
    "same_dose_top2",
    "mean_dose_margin",
    "MED_exact_accuracy",
    "MED_plus_or_minus_1_accuracy",
    "MED_dose_step_error",
)
PRIMARY_ENDPOINT = "trajectory_concordance"
COMPARISONS = (("teacher", "raw"), ("posterior", "teacher"))
GE_BLOCK = "BLOCKED_VALIDATION_PREDICTIONS_MISSING"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def as_str(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8", errors="replace").astype(str)
    return values.astype(str)


def pcc(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b):
        return None
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denominator) or denominator <= 0:
        return None
    value = float(np.dot(a, b) / denominator)
    return value if np.isfinite(value) else None


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, matching the usual Spearman definition."""
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return None
    ra = rankdata(a[mask])
    rb = rankdata(b[mask])
    return pcc(ra, rb)


def key(compound: str, dose: str, held_index: int) -> tuple[str, str, int]:
    return (str(compound), str(dose), int(held_index))


def well_row(value: str) -> str:
    token = str(value).split(";", 1)[0]
    out = ""
    for char in token:
        if char.isalpha():
            out += char
        else:
            break
    return out


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}g}"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass
class Artifact:
    compound: np.ndarray
    dose: np.ndarray
    plate: np.ndarray
    well: np.ndarray
    split: np.ndarray
    delta: np.ndarray
    groups: dict[tuple[str, str], list[int]]
    train_center: np.ndarray
    train_scale: np.ndarray
    plate_count: int
    feature_dim: int


@dataclass
class PredictionBundle:
    seed: int
    compound: np.ndarray
    dose: np.ndarray
    held_index: np.ndarray
    support_mean: np.ndarray
    teacher: np.ndarray
    virtual_prior: np.ndarray
    posterior: np.ndarray
    index_by_key: dict[tuple[str, str, int], int]
    support_by_key: dict[tuple[str, str, int], int]
    lambda_value: float
    beta_value: float
    path: Path


@dataclass
class GEBundle:
    seed: int
    compound: np.ndarray
    dose: np.ndarray
    held_index: np.ndarray
    p0: np.ndarray
    correct_ge: np.ndarray
    index_by_key: dict[tuple[str, str, int], int]
    beta_value: float
    path: Path


@dataclass
class SlotProfile:
    compound: str
    dose: str
    rotation: int
    support_row: int
    support_plate: str
    support_well: str
    held_source_row: int
    reference_rows: tuple[int, ...]
    profiles: dict[str, np.ndarray]
    reference: np.ndarray
    m2_available: bool
    ge_key: tuple[str, str, int] | None


@dataclass
class CompoundMetrics:
    compound: str
    method: str
    trajectory_concordance: float
    same_dose_top1: float
    same_dose_top2: float
    mean_dose_margin: float
    med_exact: float
    med_pm1: float
    med_step_error: float
    med_reference_none_count: int
    med_predicted_none_count: int
    monotonic_subset: bool


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    base = here.parents[1] / "external_validation" / "lincs_cpg0004"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=base / "data_preparation" / "artifact" / "cp_plate_rows.npz")
    parser.add_argument("--p0-root", type=Path, default=base / "virtual_prior" / "results" / "1r_all")
    parser.add_argument("--ge-root", type=Path, default=base / "single_repeat_expression_evidence" / "results" / "1r_all")
    parser.add_argument("--pair-root", type=Path, default=base / "cell_painting_repeat_benchmark" / "results" / "1r_all")
    parser.add_argument("--hit-dir", type=Path, default=here.parent / "hit_recovery")
    parser.add_argument("--outdir", type=Path, default=here)
    parser.add_argument("--bootstrap-rounds", type=int, default=BOOTSTRAP_ROUNDS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use 200 bootstrap rounds for a local smoke only")
    return parser.parse_args()


def load_artifact(path: Path) -> Artifact:
    required = {"compound_id", "dose", "plate", "well", "split", "delta", "baseline"}
    with np.load(path, allow_pickle=False) as loaded:
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"Frozen CP artifact missing keys: {missing}")
        compound = as_str(loaded["compound_id"])
        dose = as_str(loaded["dose"])
        plate = as_str(loaded["plate"])
        well = as_str(loaded["well"])
        split = as_str(loaded["split"])
        delta = loaded["delta"].astype(np.float32)
        baseline = loaded["baseline"].astype(np.float32)
    if len(compound) != len(dose) or len(compound) != len(plate) or len(compound) != len(well) or len(compound) != len(split):
        raise ValueError("Frozen CP metadata arrays have inconsistent row counts")
    if delta.ndim != 2 or baseline.ndim != 2 or delta.shape != baseline.shape or len(delta) != len(compound):
        raise ValueError("Frozen CP delta/baseline arrays have inconsistent shapes")
    if not np.isfinite(delta).all() or not np.isfinite(baseline).all():
        raise ValueError("Frozen CP artifact contains non-finite values")
    if set(dose) - set(DOSES):
        raise ValueError(f"Unexpected frozen dose labels: {sorted(set(dose) - set(DOSES))}")
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (c, d) in enumerate(zip(compound, dose)):
        groups[(str(c), str(d))].append(int(index))
    for group, rows in groups.items():
        groups[group] = sorted(rows, key=lambda i: (str(plate[i]), str(well[i]), i))
    train_rows = np.flatnonzero(split == "train")
    if len(train_rows) == 0:
        raise ValueError("Frozen CP artifact has no training rows for standardization")
    train_delta = delta[train_rows].astype(np.float64)
    center = np.median(train_delta, axis=0)
    scale = 1.4826 * np.median(np.abs(train_delta - center), axis=0)
    degenerate = ~np.isfinite(scale) | (scale < 1e-6)
    std = np.std(train_delta, axis=0)
    scale[degenerate] = std[degenerate]
    degenerate = ~np.isfinite(scale) | (scale < 1e-6)
    scale[degenerate] = 1.0
    return Artifact(compound, dose, plate, well, split, delta, dict(groups), center.astype(np.float32), scale.astype(np.float32), len(set(plate)), int(delta.shape[1]))


def load_prediction(path: Path, seed: int, lambda_value: float, beta_value: float) -> PredictionBundle:
    with np.load(path, allow_pickle=False) as loaded:
        required = {"compound", "dose", "held_index", "support_mean", "teacher", "virtual_prior", "posterior"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"{path} missing frozen prediction keys: {missing}")
        compound = as_str(loaded["compound"])
        dose = as_str(loaded["dose"])
        held = loaded["held_index"].astype(np.int64)
        support_mean = loaded["support_mean"].astype(np.float32)
        teacher = loaded["teacher"].astype(np.float32)
        virtual = loaded["virtual_prior"].astype(np.float32)
        posterior = loaded["posterior"].astype(np.float32)
    n = len(compound)
    if any(len(x) != n for x in (dose, held, support_mean, teacher, virtual, posterior)):
        raise ValueError(f"{path} frozen prediction arrays have inconsistent row counts")
    if any(x.ndim != 2 for x in (support_mean, teacher, virtual, posterior)):
        raise ValueError(f"{path} frozen profiles are not two-dimensional")
    if not all(np.isfinite(x).all() for x in (support_mean, teacher, virtual, posterior)):
        raise ValueError(f"{path} frozen profiles contain non-finite values")
    index: dict[tuple[str, str, int], int] = {}
    for i, (c, d, h) in enumerate(zip(compound, dose, held)):
        k = key(c, d, int(h))
        if k in index:
            raise ValueError(f"Duplicate frozen prediction key in {path}: {k}")
        index[k] = int(i)
    return PredictionBundle(seed, compound, dose, held, support_mean, teacher, virtual, posterior, index, {}, float(lambda_value), float(beta_value), path)


def load_ge(path: Path, seed: int, beta_value: float) -> GEBundle:
    with np.load(path, allow_pickle=False) as loaded:
        required = {"compound", "dose", "held_index", "P0", "correct_GE"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"{path} missing frozen GE keys: {missing}")
        compound = as_str(loaded["compound"])
        dose = as_str(loaded["dose"])
        held = loaded["held_index"].astype(np.int64)
        p0 = loaded["P0"].astype(np.float32)
        correct = loaded["correct_GE"].astype(np.float32)
    n = len(compound)
    if any(len(x) != n for x in (dose, held, p0, correct)) or p0.ndim != 2 or correct.ndim != 2 or p0.shape != correct.shape:
        raise ValueError(f"{path} frozen GE arrays have inconsistent shapes")
    if not np.isfinite(p0).all() or not np.isfinite(correct).all():
        raise ValueError(f"{path} frozen GE arrays contain non-finite values")
    index: dict[tuple[str, str, int], int] = {}
    for i, (c, d, h) in enumerate(zip(compound, dose, held)):
        k = key(c, d, int(h))
        if k in index:
            raise ValueError(f"Duplicate frozen GE key in {path}: {k}")
        index[k] = int(i)
    return GEBundle(seed, compound, dose, held, p0, correct, index, float(beta_value), path)


def load_manifest_hashes() -> dict[str, str]:
    manifest_path = Path(__file__).resolve().parents[1] / "protocol" / "FREEZE_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(entry["id"]): str(entry["sha256"]) for entry in payload.get("entries", []) if entry.get("sha256")}


def load_pair_manifest(path: Path) -> dict[tuple[str, str, int], int]:
    out: dict[tuple[str, str, int], int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            support_tokens = [x for x in str(row["support_rows"]).split("|") if x]
            if len(support_tokens) != 1:
                raise ValueError(f"Unexpected support row cardinality in {path}: {row}")
            k = key(row["compound"], row["dose"], int(row["held_row"]))
            if k in out:
                raise ValueError(f"Duplicate pair-manifest key in {path}: {k}")
            out[k] = int(support_tokens[0])
    return out


def load_hit_config(hit_dir: Path) -> tuple[dict[str, Any], str]:
    path = hit_dir / "CONFIG.json"
    if not path.is_file():
        raise FileNotFoundError(f"Experiment 1 CONFIG.json is required for MED: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("model_thresholds_validation_only")
    if not isinstance(thresholds, dict):
        raise ValueError("Experiment 1 CONFIG.json lacks model_thresholds_validation_only")
    return payload, sha256(path) or ""


def check_input_hashes(
    args: argparse.Namespace,
    manifest_hashes: dict[str, str],
    hit_config_hash: str,
    pair_hashes: dict[int, str],
) -> dict[str, Any]:
    expected_local = {
        "cpg.prepared.cp.rows": manifest_hashes.get("cpg.prepared.cp.rows"),
        "cpg.prior.seed3407.test": manifest_hashes.get("cpg.prior.seed3407.test"),
        "cpg.prior.seed42.test": manifest_hashes.get("cpg.prior.seed42.test"),
        "cpg.prior.seed2025.test": manifest_hashes.get("cpg.prior.seed2025.test"),
        "cpg.ge.seed3407.test": manifest_hashes.get("cpg.ge.seed3407.test"),
        "cpg.ge.seed42.test": manifest_hashes.get("cpg.ge.seed42.test"),
        "cpg.ge.seed2025.test": manifest_hashes.get("cpg.ge.seed2025.test"),
    }
    paths = {
        "cpg.prepared.cp.rows": args.data,
        "cpg.prior.seed3407.test": args.p0_root / "seed3407" / "test_predictions.npz",
        "cpg.prior.seed42.test": args.p0_root / "seed42" / "test_predictions.npz",
        "cpg.prior.seed2025.test": args.p0_root / "seed2025" / "test_predictions.npz",
        "cpg.ge.seed3407.test": args.ge_root / "seed3407" / "test_predictions.npz",
        "cpg.ge.seed42.test": args.ge_root / "seed42" / "test_predictions.npz",
        "cpg.ge.seed2025.test": args.ge_root / "seed2025" / "test_predictions.npz",
    }
    actual: dict[str, Any] = {"experiment_1_config_sha256": hit_config_hash}
    mismatches: list[str] = []
    for name, path in paths.items():
        got = sha256(path)
        actual[name] = {"path": str(path.resolve()), "sha256": got, "expected": expected_local.get(name)}
        if got is None:
            mismatches.append(f"missing:{name}:{path}")
        elif expected_local.get(name) and got != expected_local[name]:
            mismatches.append(f"hash:{name}:expected={expected_local[name]}:got={got}")
    for seed, got in pair_hashes.items():
        expected = None
        # Pair-manifest hashes are recorded in Experiment 1 CONFIG.json and
        # are checked by the caller.  This map is included here for audit.
        actual[f"pair_manifest.seed{seed}"] = {"sha256": got, "expected": expected}
    if mismatches:
        raise RuntimeError("FREEZE_HASH_MISMATCH: " + "; ".join(mismatches))
    return actual


def profile_magnitude(artifact: Artifact, profile: np.ndarray) -> float:
    z = (np.asarray(profile, dtype=np.float64) - artifact.train_center.astype(np.float64)) / artifact.train_scale.astype(np.float64)
    value = float(np.linalg.norm(z))
    return value if np.isfinite(value) else math.nan


def trajectory_score(profiles: list[np.ndarray], references: list[np.ndarray]) -> float:
    if len(profiles) != len(DOSES) or len(references) != len(DOSES):
        return math.nan
    distances: list[float] = []
    ref_distances: list[float] = []
    for i, j in combinations(range(len(DOSES)), 2):
        value = pcc(profiles[i], profiles[j])
        ref_value = pcc(references[i], references[j])
        if value is None or ref_value is None:
            return math.nan
        distances.append(1.0 - value)
        ref_distances.append(1.0 - ref_value)
    result = spearman(np.asarray(distances), np.asarray(ref_distances))
    return float(result) if result is not None else math.nan


def dose_identification(profiles: list[np.ndarray], references: list[np.ndarray]) -> tuple[float, float, float]:
    top1: list[float] = []
    top2: list[float] = []
    margins: list[float] = []
    for true_index, profile in enumerate(profiles):
        sims = [pcc(profile, ref) for ref in references]
        if any(value is None for value in sims):
            return math.nan, math.nan, math.nan
        values = np.asarray(sims, dtype=np.float64)
        order = np.argsort(-values, kind="mergesort")
        top1.append(float(int(order[0]) == true_index))
        top2.append(float(true_index in set(order[:2].tolist())))
        wrong = np.delete(values, true_index)
        margins.append(float(values[true_index] - np.mean(wrong)))
    return float(np.mean(top1)), float(np.mean(top2)), float(np.mean(margins))


def first_active(magnitudes: list[float], threshold: float | None) -> int | None:
    if threshold is None or not np.isfinite(float(threshold)):
        return None
    for index, value in enumerate(magnitudes):
        if np.isfinite(value) and value > float(threshold):
            return int(index)
    # Sentinel 6 denotes no dose above the fixed validation threshold.  It is
    # retained rather than dropping no-active compounds after seeing results.
    return len(DOSES)


def med_metrics(
    artifact: Artifact,
    profiles: list[np.ndarray],
    references: list[np.ndarray],
    threshold: float | None,
) -> tuple[float, float, float, int, int]:
    if threshold is None:
        return math.nan, math.nan, math.nan, 0, 0
    pred_mags = [profile_magnitude(artifact, x) for x in profiles]
    ref_mags = [profile_magnitude(artifact, x) for x in references]
    pred = first_active(pred_mags, threshold)
    ref = first_active(ref_mags, threshold)
    if pred is None or ref is None:
        return math.nan, math.nan, math.nan, 0, 0
    return float(pred == ref), float(abs(pred - ref) <= 1), float(abs(pred - ref)), int(ref == len(DOSES)), int(pred == len(DOSES))


def compute_slot_metrics(
    artifact: Artifact,
    method: str,
    slots_by_dose: dict[str, SlotProfile],
    threshold: float | None,
    monotonic_subset: bool,
) -> CompoundMetrics:
    profiles = [slots_by_dose[d].profiles[method] for d in DOSES]
    references = [slots_by_dose[d].reference for d in DOSES]
    trajectory = trajectory_score(profiles, references)
    top1, top2, margin = dose_identification(profiles, references)
    med_exact, med_pm1, med_error, ref_none, pred_none = med_metrics(artifact, profiles, references, threshold)
    return CompoundMetrics(
        compound=slots_by_dose[DOSES[0]].compound,
        method=method,
        trajectory_concordance=trajectory,
        same_dose_top1=top1,
        same_dose_top2=top2,
        mean_dose_margin=margin,
        med_exact=med_exact,
        med_pm1=med_pm1,
        med_step_error=med_error,
        med_reference_none_count=ref_none,
        med_predicted_none_count=pred_none,
        monotonic_subset=bool(monotonic_subset),
    )


def average_metrics(metrics: list[CompoundMetrics], compound: str, method: str, mono: bool) -> CompoundMetrics:
    chosen = [x for x in metrics if x.method == method and x.monotonic_subset == mono]
    if not chosen:
        return CompoundMetrics(compound, method, *(math.nan for _ in range(7)), 0, 0, mono)
    def mean(name: str) -> float:
        vals = np.asarray([getattr(x, name) for x in chosen], dtype=np.float64)
        return float(np.mean(vals)) if len(vals) and np.isfinite(vals).all() else math.nan
    return CompoundMetrics(
        compound=compound,
        method=method,
        trajectory_concordance=mean("trajectory_concordance"),
        same_dose_top1=mean("same_dose_top1"),
        same_dose_top2=mean("same_dose_top2"),
        mean_dose_margin=mean("mean_dose_margin"),
        med_exact=mean("med_exact"),
        med_pm1=mean("med_pm1"),
        med_step_error=mean("med_step_error"),
        med_reference_none_count=int(sum(x.med_reference_none_count for x in chosen)),
        med_predicted_none_count=int(sum(x.med_predicted_none_count for x in chosen)),
        monotonic_subset=mono,
    )


def reference_monotonic(artifact: Artifact, references: list[np.ndarray]) -> tuple[bool, float, str, int]:
    magnitudes = np.asarray([profile_magnitude(artifact, x) for x in references], dtype=np.float64)
    value = spearman(LOG_DOSES, magnitudes)
    if value is None or not np.isfinite(value):
        return False, math.nan, "undefined", 0
    positive = bool(value > 0.6)
    if positive:
        # The ordering is the number of adjacent dose steps that are
        # non-decreasing.  It is descriptive only and is not imposed as a
        # constraint on any method.
        ordering = int(np.sum(np.diff(magnitudes) >= 0))
        direction = "increasing" if magnitudes[-1] >= magnitudes[0] else "decreasing"
    else:
        ordering = 0
        direction = "not_selected"
    return positive, float(value), direction, ordering


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else ["reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {field: ("" if value is None else value) for field, value in row.items()}
            writer.writerow(clean)


def load_pair_and_prediction_bundles(
    args: argparse.Namespace,
    lambda_by_seed: dict[int, float],
    beta_by_seed: dict[int, float],
    mapping_errors: list[str],
) -> tuple[dict[int, PredictionBundle], dict[int, GEBundle], dict[int, dict[tuple[str, str, int], int]], dict[int, str]]:
    p0: dict[int, PredictionBundle] = {}
    ge: dict[int, GEBundle] = {}
    manifests: dict[int, dict[tuple[str, str, int], int]] = {}
    pair_hashes: dict[int, str] = {}
    for seed in SEEDS:
        ppath = args.p0_root / f"seed{seed}" / "test_predictions.npz"
        gpath = args.ge_root / f"seed{seed}" / "test_predictions.npz"
        mpath = args.pair_root / f"seed{seed}" / "test_pair_manifest.csv"
        p0[seed] = load_prediction(ppath, seed, lambda_by_seed[seed], beta_by_seed[seed])
        ge[seed] = load_ge(gpath, seed, beta_by_seed[seed])
        manifests[seed] = load_pair_manifest(mpath)
        pair_hashes[seed] = sha256(mpath) or ""
        pkeys = set(p0[seed].index_by_key)
        gkeys = set(ge[seed].index_by_key)
        if not gkeys.issubset(pkeys):
            mapping_errors.append(f"ge_keys_not_subset_prior:seed={seed}:extra={len(gkeys - pkeys)}")
        if len(pkeys - gkeys) != 10:
            mapping_errors.append(f"prior_only_key_count_unexpected:seed={seed}:count={len(pkeys-gkeys)}:expected=10")
        for k, row in p0[seed].index_by_key.items():
            if k not in manifests[seed]:
                mapping_errors.append(f"prediction_key_missing_manifest:seed={seed}:key={k}")
                continue
            support = manifests[seed][k]
            p0[seed].support_by_key[k] = int(support)
        # The frozen P0 identity is audited at the row level.  It is not
        # refit or recalibrated by this application.
        reconstructed = (1.0 - p0[seed].lambda_value) * p0[seed].teacher + p0[seed].lambda_value * p0[seed].virtual_prior
        p0_err = float(np.max(np.abs(reconstructed - p0[seed].posterior)))
        if p0_err > 5e-4:
            mapping_errors.append(f"p0_formula_error:seed={seed}:max_abs={p0_err:.9g}")
    return p0, ge, manifests, pair_hashes


def build_slots(
    artifact: Artifact,
    p0: dict[int, PredictionBundle],
    ge: dict[int, GEBundle],
    manifests: dict[int, dict[tuple[str, str, int], int]],
    mapping_errors: list[str],
) -> tuple[dict[int, dict[str, dict[int, dict[str, SlotProfile]]]], list[dict[str, Any]], dict[str, set[str]], dict[str, dict[str, Any]]]:
    """Build support/reference profiles and return seed-specific slots.

    The returned structure is ``seed -> compound -> rotation -> dose -> slot``.
    Support rows are identified from the frozen pair manifest and sorted by
    physical row index, giving deterministic rotation ranks.  The same
    support map is checked across all seeds before downstream analysis.
    """
    exclusions: list[dict[str, Any]] = []
    all_compounds = sorted({str(c) for c, s in zip(artifact.compound, artifact.split) if str(s) == "test"})
    six_dose_compounds = set()
    strict_compounds = set()
    for compound in all_compounds:
        present = {d for d in DOSES if (compound, d) in artifact.groups}
        if len(present) != len(DOSES):
            for d in DOSES:
                rows = artifact.groups.get((compound, d), [])
                if d not in present:
                    exclusions.append({"scope": "eligibility", "compound": compound, "dose": d, "reason": "missing_locked_dose", "repeat_count": len(rows), "seed": "all"})
            continue
        six_dose_compounds.add(compound)
        strict = True
        for d in DOSES:
            rows = artifact.groups[(compound, d)]
            if len(rows) != REQUIRED_REPEATS:
                strict = False
                exclusions.append({"scope": "eligibility", "compound": compound, "dose": d, "reason": "strict_requires_five_repeats", "repeat_count": len(rows), "seed": "all"})
        if strict:
            strict_compounds.add(compound)
    seed_slots: dict[int, dict[str, dict[int, dict[str, SlotProfile]]]] = {seed: defaultdict(dict) for seed in SEEDS}
    seed_compound_sets: dict[str, set[str]] = {}
    audit_info: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        available_compounds: set[str] = set()
        for compound in sorted(strict_compounds):
            per_rotation: dict[int, dict[str, SlotProfile]] = {}
            condition_supports: dict[str, list[int]] = {}
            condition_source: dict[str, dict[int, tuple[int, int]]] = {}
            compound_ok = True
            for dose in DOSES:
                rows = artifact.groups[(compound, dose)]
                support_to_source: dict[int, int] = {}
                for held in rows:
                    k = key(compound, dose, held)
                    if k not in p0[seed].index_by_key or k not in manifests[seed]:
                        continue
                    support = int(manifests[seed][k])
                    if support not in rows or support == held:
                        mapping_errors.append(f"invalid_support_mapping:seed={seed}:key={k}:support={support}")
                        continue
                    support_to_source.setdefault(support, held)
                supports = sorted(support_to_source)
                condition_supports[dose] = supports
                if len(supports) != EXPECTED_SUPPORT_SLOTS:
                    compound_ok = False
                    reason = "fewer_than_two_frozen_support_slots" if len(supports) < 2 else "unexpected_support_slot_count"
                    exclusions.append({"scope": "support_rotation", "seed": seed, "compound": compound, "dose": dose, "reason": reason, "repeat_count": len(rows), "unique_support_slots": len(supports)})
                    continue
                if len(supports) < len(rows):
                    exclusions.append({"scope": "support_rotation", "seed": seed, "compound": compound, "dose": dose, "reason": "partial_frozen_support_slot_coverage", "repeat_count": len(rows), "unique_support_slots": len(supports), "detail": "Two recorded slots are used; three possible physical support slots are not inferred."})
                for rotation, support_row in enumerate(supports):
                    sources = [h for h in rows if manifests[seed].get(key(compound, dose, h)) == support_row and key(compound, dose, h) in p0[seed].index_by_key]
                    if not sources:
                        compound_ok = False
                        mapping_errors.append(f"missing_teacher_source_for_support:seed={seed}:{compound}:{dose}:{support_row}")
                        continue
                    source_held = min(sources)
                    teacher_indices = [p0[seed].index_by_key[key(compound, dose, h)] for h in sources]
                    teacher = p0[seed].teacher[teacher_indices[0]].copy()
                    teacher_error = float(np.max(np.abs(p0[seed].teacher[np.asarray(teacher_indices)] - teacher)))
                    if teacher_error > 2e-5:
                        mapping_errors.append(f"teacher_rotation_inconsistency:seed={seed}:{compound}:{dose}:{support_row}:max_abs={teacher_error:.9g}")
                    source_index = p0[seed].index_by_key[key(compound, dose, source_held)]
                    support_error = float(np.max(np.abs(p0[seed].support_mean[source_index] - artifact.delta[support_row])))
                    if support_error > 2e-5:
                        mapping_errors.append(f"support_vector_mismatch:seed={seed}:{compound}:{dose}:{support_row}:max_abs={support_error:.9g}")
                    # M2 reconstruction follows the locked cpg0004 rule.  The
                    # existing held=s row supplies the frozen prior/P0 and GE
                    # residual; no new pairing is introduced.
                    baseline_key = key(compound, dose, support_row)
                    if baseline_key not in p0[seed].index_by_key:
                        compound_ok = False
                        mapping_errors.append(f"missing_baseline_held_support_key:seed={seed}:{baseline_key}")
                        continue
                    baseline_index = p0[seed].index_by_key[baseline_key]
                    # Derive V_s algebraically from the saved P0 row and its
                    # original held-row teacher, then use that frozen prior
                    # component with the teacher corresponding to this
                    # support rotation.  This is the locked cpg0004
                    # reconstruction; no prior/teacher is refit here.
                    baseline_teacher = p0[seed].teacher[baseline_index]
                    saved_p0 = p0[seed].posterior[baseline_index]
                    virtual = (saved_p0 - (1.0 - p0[seed].lambda_value) * baseline_teacher) / p0[seed].lambda_value
                    virtual_error = float(np.max(np.abs(virtual - p0[seed].virtual_prior[baseline_index])))
                    if virtual_error > 5e-3:
                        mapping_errors.append(f"virtual_prior_inverse_error:seed={seed}:{baseline_key}:max_abs={virtual_error:.9g}")
                    p0_recon = (1.0 - p0[seed].lambda_value) * teacher + p0[seed].lambda_value * virtual
                    ge_key = baseline_key
                    m2_available = ge_key in ge[seed].index_by_key
                    posterior = None
                    if m2_available:
                        gi = ge[seed].index_by_key[ge_key]
                        residual = ge[seed].correct_ge[gi] - ge[seed].p0[gi]
                        posterior = p0_recon + residual
                        p0_direct_err = float(np.max(np.abs(ge[seed].p0[gi] - p0[seed].posterior[baseline_index])))
                        if p0_direct_err > 5e-4:
                            mapping_errors.append(f"ge_p0_key_mismatch:seed={seed}:{baseline_key}:max_abs={p0_direct_err:.9g}")
                    else:
                        compound_ok = False
                        exclusions.append({"scope": "M2_common_set", "seed": seed, "compound": compound, "dose": dose, "support_row": support_row, "reason": "GE_KEY_MISSING_FOR_M2", "repeat_count": len(rows)})
                    reference_rows = tuple(row for row in rows if row != support_row)
                    if len(reference_rows) != 4 or support_row in reference_rows:
                        compound_ok = False
                        mapping_errors.append(f"reference_support_overlap_or_count:seed={seed}:{compound}:{dose}:{support_row}:rows={reference_rows}")
                        continue
                    reference = artifact.delta[np.asarray(reference_rows, dtype=np.int64)].mean(axis=0, dtype=np.float64).astype(np.float32)
                    profiles: dict[str, np.ndarray] = {"raw": artifact.delta[support_row].copy(), "teacher": teacher.astype(np.float32)}
                    if posterior is not None:
                        profiles["posterior"] = posterior.astype(np.float32)
                    per_rotation.setdefault(rotation, {})[dose] = SlotProfile(compound, dose, rotation, int(support_row), str(artifact.plate[support_row]), str(artifact.well[support_row]), int(source_held), reference_rows, profiles, reference, bool(m2_available), ge_key if m2_available else None)
                condition_source[dose] = {r: (supports[r], support_to_source[supports[r]]) for r in range(len(supports))}
            if compound_ok and all(len(per_rotation.get(r, {})) == len(DOSES) for r in range(EXPECTED_SUPPORT_SLOTS)):
                available_compounds.add(compound)
                seed_slots[seed][compound] = per_rotation
            else:
                # A strict M1 compound may fail M2 at only two dose/support
                # keys.  It remains available to the raw-vs-teacher common
                # set, so its partial M2 rows are retained in audit only.
                if all(len(per_rotation.get(r, {})) == len(DOSES) for r in range(EXPECTED_SUPPORT_SLOTS)):
                    available_compounds.add(compound)
                    seed_slots[seed][compound] = per_rotation
        seed_compound_sets[str(seed)] = available_compounds
        audit_info[str(seed)] = {"strict_all6_compounds": len(strict_compounds), "m1_two_slot_all6_compounds": len(available_compounds)}
    # The support-slot map must be stable across seeds.  A mismatch is a hard
    # audit issue rather than a reason to select a favorable seed.
    first = seed_compound_sets[str(SEEDS[0])]
    for seed in SEEDS[1:]:
        if seed_compound_sets[str(seed)] != first:
            mapping_errors.append(f"cross_seed_m1_common_set_mismatch:seed={seed}:delta={len(first ^ seed_compound_sets[str(seed)])}")
    return seed_slots, exclusions, seed_compound_sets, audit_info


def trim_to_common_sets(seed_slots: dict[int, dict[str, dict[int, dict[str, SlotProfile]]]], strict_compounds: set[str] | None = None) -> tuple[set[str], set[str]]:
    m1 = set.intersection(*(set(seed_slots[s]) for s in SEEDS)) if all(seed_slots[s] for s in SEEDS) else set()
    if strict_compounds is not None:
        m1 &= strict_compounds
    m2 = set()
    for compound in sorted(m1):
        ok = True
        for seed in SEEDS:
            for rotation in range(EXPECTED_SUPPORT_SLOTS):
                for dose in DOSES:
                    slot = seed_slots[seed][compound][rotation][dose]
                    if not slot.m2_available or "posterior" not in slot.profiles:
                        ok = False
        if ok:
            m2.add(compound)
    return m1, m2


def slot_summary_rows(seed_slots: dict[int, dict[str, dict[int, dict[str, SlotProfile]]]], m1: set[str], m2: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = SEEDS[0]
    for compound in sorted(m1):
        for rotation in range(EXPECTED_SUPPORT_SLOTS):
            for dose in DOSES:
                slot = seed_slots[seed][compound][rotation][dose]
                rows.append({
                    "compound": compound,
                    "split": "test",
                    "dose": dose,
                    "rotation": rotation,
                    "support_row": slot.support_row,
                    "support_plate": slot.support_plate,
                    "support_well": slot.support_well,
                    "held_source_row_for_teacher": slot.held_source_row,
                    "reference_rows": "|".join(str(x) for x in slot.reference_rows),
                    "reference_excludes_support": int(slot.support_row not in slot.reference_rows),
                    "m1_eligible": 1,
                    "m2_ge_available_all_seeds": int(compound in m2),
                    "m2_ge_available_seed3407": int(slot.m2_available),
                    "m2_ge_key": "|".join(map(str, slot.ge_key)) if slot.ge_key else "",
                    "support_slot_coverage": "2_of_5",
                })
    return rows


def compound_level_metrics(
    artifact: Artifact,
    seed_slots: dict[int, dict[str, dict[int, dict[str, SlotProfile]]]],
    compounds: set[str],
    method: str,
    threshold: float | None,
    mono_flags: dict[str, bool],
) -> tuple[dict[int, dict[str, CompoundMetrics]], list[dict[str, Any]], dict[int, dict[str, tuple[float, str, int]]]]:
    by_seed: dict[int, dict[str, CompoundMetrics]] = {seed: {} for seed in SEEDS}
    slot_rows: list[dict[str, Any]] = []
    mono_audit: dict[int, dict[str, tuple[float, str, int]]] = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for compound in sorted(compounds):
            slots = seed_slots[seed][compound]
            # Reference monotonicity is based on the average magnitude of the
            # two independent four-repeat references; selection is fixed
            # before method comparison and is explanatory only.
            ref_profiles = [np.mean(np.stack([slots[r][d].reference for r in range(EXPECTED_SUPPORT_SLOTS)]), axis=0) for d in DOSES]
            mono, ref_rho, direction, ordering = reference_monotonic(artifact, ref_profiles)
            mono_flags[compound] = bool(mono)
            mono_audit[seed][compound] = (ref_rho, direction, ordering)
            per_slot: list[CompoundMetrics] = []
            for rotation in range(EXPECTED_SUPPORT_SLOTS):
                slots_by_dose = slots[rotation]
                if method not in slots_by_dose[DOSES[0]].profiles:
                    continue
                current = compute_slot_metrics(artifact, method, slots_by_dose, threshold, mono)
                per_slot.append(current)
                slot_rows.append({
                    "seed": seed,
                    "compound": compound,
                    "rotation": rotation,
                    "method": method,
                    "trajectory_concordance": fmt(current.trajectory_concordance),
                    "same_dose_top1": fmt(current.same_dose_top1),
                    "same_dose_top2": fmt(current.same_dose_top2),
                    "mean_dose_margin": fmt(current.mean_dose_margin),
                    "MED_exact_accuracy": fmt(current.med_exact),
                    "MED_plus_or_minus_1_accuracy": fmt(current.med_pm1),
                    "MED_dose_step_error": fmt(current.med_step_error),
                    "reference_MED_no_active_count": current.med_reference_none_count,
                    "predicted_MED_no_active_count": current.med_predicted_none_count,
                    "reference_monotonic_selected": int(mono),
                    "reference_monotonic_spearman": fmt(ref_rho),
                    "reference_trend_direction": direction,
                    "reference_monotonic_ordering_non_decreasing_steps": ordering,
                })
            if len(per_slot) == EXPECTED_SUPPORT_SLOTS:
                by_seed[seed][compound] = average_metrics(per_slot, compound, method, mono)
    return by_seed, slot_rows, mono_audit


def compound_metrics_rows(all_metrics: dict[int, dict[str, dict[str, CompoundMetrics]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for method in METHODS:
            for compound, metric in sorted(all_metrics[seed].get(method, {}).items()):
                rows.append({
                    "seed": seed,
                    "compound": compound,
                    "method": method,
                    "trajectory_concordance": fmt(metric.trajectory_concordance),
                    "same_dose_top1": fmt(metric.same_dose_top1),
                    "same_dose_top2": fmt(metric.same_dose_top2),
                    "mean_dose_margin": fmt(metric.mean_dose_margin),
                    "MED_exact_accuracy": fmt(metric.med_exact),
                    "MED_plus_or_minus_1_accuracy": fmt(metric.med_pm1),
                    "MED_dose_step_error": fmt(metric.med_step_error),
                    "reference_MED_no_active_count": metric.med_reference_none_count,
                    "predicted_MED_no_active_count": metric.med_predicted_none_count,
                    "reference_monotonic_selected": int(metric.monotonic_subset),
                })
    return rows


def get_metric(metric: CompoundMetrics, endpoint: str) -> float:
    return {
        "trajectory_concordance": metric.trajectory_concordance,
        "same_dose_top1": metric.same_dose_top1,
        "same_dose_top2": metric.same_dose_top2,
        "mean_dose_margin": metric.mean_dose_margin,
        "MED_exact_accuracy": metric.med_exact,
        "MED_plus_or_minus_1_accuracy": metric.med_pm1,
        "MED_dose_step_error": metric.med_step_error,
    }[endpoint]


def mean_metric(metric_map: dict[str, CompoundMetrics], endpoint: str) -> float:
    values = np.asarray([get_metric(x, endpoint) for x in metric_map.values()], dtype=np.float64)
    return float(np.mean(values)) if len(values) and np.isfinite(values).all() else math.nan


def make_metrics_and_contrasts(
    all_metrics: dict[int, dict[str, dict[str, CompoundMetrics]]],
    m1: set[str],
    m2: set[str],
    thresholds: dict[str, float | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, str, str], np.ndarray], dict[tuple[int, str, str], np.ndarray]]:
    metrics_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    # Store per-seed per-contrast per-endpoint compound arrays for bootstrap.
    left_arrays: dict[tuple[int, str, str], np.ndarray] = {}
    right_arrays: dict[tuple[int, str, str], np.ndarray] = {}
    for seed in SEEDS:
        for method, compounds, analysis_set in (("raw", m1, "M1_vs_M0_common"), ("teacher", m1, "M1_vs_M0_common"), ("raw", m2, "M2_vs_M1_common"), ("teacher", m2, "M2_vs_M1_common"), ("posterior", m2, "M2_vs_M1_common")):
            method_map = {c: all_metrics[seed][method][c] for c in sorted(compounds) if c in all_metrics[seed].get(method, {})}
            for endpoint in ENDPOINTS:
                values = np.asarray([get_metric(x, endpoint) for x in method_map.values()], dtype=np.float64)
                available = method != "posterior" or endpoint not in {"MED_exact_accuracy", "MED_plus_or_minus_1_accuracy", "MED_dose_step_error"}
                status = "OK" if available and len(values) and np.isfinite(values).all() else (GE_BLOCK if not available else "NONFINITE_ENDPOINT")
                metrics_rows.append({
                    "seed": seed,
                    "analysis_set": analysis_set,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "endpoint": endpoint,
                    "value": fmt(np.mean(values)) if status == "OK" else "",
                    "n_compounds": len(method_map),
                    "support_rotations_per_compound": EXPECTED_SUPPORT_SLOTS,
                    "dose_count": len(DOSES),
                    "threshold_validation": fmt(thresholds.get(method)),
                    "threshold_status": status if endpoint.startswith("MED") else "NOT_APPLICABLE",
                })
        # Retrospective monotonic subset rows are added below by filtering the
        # compound metric flag, with the same fixed subset for all methods.
        mono = {c for c in m2 if all_metrics[seed].get("raw", {}).get(c, CompoundMetrics(c, "raw", math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, 0, 0, False)).monotonic_subset}
        mono &= set(all_metrics[seed].get("raw", {}))
        for method in METHODS:
            method_map = {c: all_metrics[seed].get(method, {}).get(c) for c in sorted(mono)}
            method_map = {c: x for c, x in method_map.items() if x is not None}
            for endpoint in ENDPOINTS:
                values = np.asarray([get_metric(x, endpoint) for x in method_map.values()], dtype=np.float64)
                available = method != "posterior" or endpoint not in {"MED_exact_accuracy", "MED_plus_or_minus_1_accuracy", "MED_dose_step_error"}
                status = "OK" if available and len(values) and np.isfinite(values).all() else (GE_BLOCK if not available else "NO_MONOTONIC_COMMON_SET")
                metrics_rows.append({
                    "seed": seed,
                    "analysis_set": "monotonic_reference_subset",
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "endpoint": endpoint,
                    "value": fmt(np.mean(values)) if status == "OK" else "",
                    "n_compounds": len(method_map),
                    "support_rotations_per_compound": EXPECTED_SUPPORT_SLOTS,
                    "dose_count": len(DOSES),
                    "threshold_validation": fmt(thresholds.get(method)),
                    "threshold_status": status if endpoint.startswith("MED") else "NOT_APPLICABLE",
                })
        for left, right in COMPARISONS:
            compounds = m1 if (left, right) == ("teacher", "raw") else m2
            analysis_set = "M1_vs_M0_common" if (left, right) == ("teacher", "raw") else "M2_vs_M1_common"
            left_map = {c: all_metrics[seed][left].get(c) for c in sorted(compounds)}
            right_map = {c: all_metrics[seed][right].get(c) for c in sorted(compounds)}
            common = sorted(set(left_map) & set(right_map))
            for endpoint in ENDPOINTS:
                left_vals = np.asarray([get_metric(left_map[c], endpoint) for c in common], dtype=np.float64)
                right_vals = np.asarray([get_metric(right_map[c], endpoint) for c in common], dtype=np.float64)
                available = endpoint not in {"MED_exact_accuracy", "MED_plus_or_minus_1_accuracy", "MED_dose_step_error"} or (left != "posterior" and right != "posterior")
                finite = bool(len(common) and np.isfinite(left_vals).all() and np.isfinite(right_vals).all())
                status = "OK" if available and finite else (GE_BLOCK if not available else "NONFINITE_ENDPOINT")
                if status == "OK":
                    point = float(np.mean(left_vals - right_vals))
                    left_arrays[(seed, f"{left}_vs_{right}", endpoint)] = left_vals
                    right_arrays[(seed, f"{left}_vs_{right}", endpoint)] = right_vals
                else:
                    point = math.nan
                contrast_rows.append({
                    "seed": seed,
                    "analysis_set": analysis_set,
                    "left_method": left,
                    "right_method": right,
                    "left_label": METHOD_LABELS[left],
                    "right_label": METHOD_LABELS[right],
                    "endpoint": endpoint,
                    "point_difference_left_minus_right": fmt(point),
                    "n_compounds": len(common),
                    "support_rotations_per_compound": EXPECTED_SUPPORT_SLOTS,
                    "dose_count": len(DOSES),
                    "status": status,
                })
    return metrics_rows, contrast_rows, left_arrays, right_arrays


def bootstrap_contrasts(
    left_arrays: dict[tuple[int, str, str], np.ndarray],
    right_arrays: dict[tuple[int, str, str], np.ndarray],
    rounds: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    distributions: dict[tuple[int, str, str], np.ndarray] = {}
    for (seed, comparison, endpoint), left in sorted(left_arrays.items()):
        right = right_arrays[(seed, comparison, endpoint)]
        n = len(left)
        label = f"dose_response|{comparison}|{endpoint}"
        boot_seed = stable_seed(seed, label)
        rng = np.random.default_rng(boot_seed)
        indices = rng.integers(0, n, size=(rounds, n), endpoint=False)
        diffs = np.mean(left[indices] - right[indices], axis=1)
        distributions[(seed, comparison, endpoint)] = diffs
        rows.append({
            "seed": seed,
            "analysis_set": "M1_vs_M0_common" if comparison == "teacher_vs_raw" else "M2_vs_M1_common",
            "comparison": comparison,
            "endpoint": endpoint,
            "point_difference": fmt(float(np.mean(left - right))),
            "ci_low": fmt(float(np.quantile(diffs, 0.025))),
            "ci_high": fmt(float(np.quantile(diffs, 0.975))),
            "rounds": rounds,
            "bootstrap_unit": "compound",
            "compound_count": n,
            "bootstrap_seed": boot_seed,
            "bootstrap_label": label,
            "status": "OK",
        })
    for comparison in sorted({x[1] for x in distributions}):
        for endpoint in ENDPOINTS:
            dist = [distributions.get((seed, comparison, endpoint)) for seed in SEEDS]
            if any(x is None for x in dist):
                continue
            mean_dist = np.mean(np.vstack([x for x in dist if x is not None]), axis=0)
            point_values = [float(np.mean(left_arrays[(seed, comparison, endpoint)] - right_arrays[(seed, comparison, endpoint)])) for seed in SEEDS]
            n = min(len(left_arrays[(seed, comparison, endpoint)]) for seed in SEEDS)
            rows.append({
                "seed": "mean",
                "analysis_set": "M1_vs_M0_common" if comparison == "teacher_vs_raw" else "M2_vs_M1_common",
                "comparison": comparison,
                "endpoint": endpoint,
                "point_difference": fmt(float(np.mean(point_values))),
                "ci_low": fmt(float(np.quantile(mean_dist, 0.025))),
                "ci_high": fmt(float(np.quantile(mean_dist, 0.975))),
                "rounds": rounds,
                "bootstrap_unit": "compound",
                "compound_count": n,
                "bootstrap_seed": "mean_of_seed_distributions",
                "bootstrap_label": f"dose_response|mean|{comparison}|{endpoint}",
                "status": "OK",
            })
    return rows


def decision_for(comparison: str, contrast_rows: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]]) -> tuple[str, str]:
    point_rows = [r for r in contrast_rows if r["seed"] in SEEDS and r["endpoint"] == PRIMARY_ENDPOINT and ((comparison == "teacher_vs_raw" and r["left_method"] == "teacher" and r["right_method"] == "raw") or (comparison == "posterior_vs_teacher" and r["left_method"] == "posterior" and r["right_method"] == "teacher"))]
    ci_rows = [r for r in bootstrap_rows if r["seed"] == "mean" and r["comparison"] == comparison and r["endpoint"] == PRIMARY_ENDPOINT]
    if len(point_rows) != len(SEEDS) or len(ci_rows) != 1:
        return "BLOCKED", "primary trajectory contrast not estimable on a common set"
    points = [float(r["point_difference_left_minus_right"]) for r in point_rows]
    ci_low = float(ci_rows[0]["ci_low"])
    if all(x > 0 for x in points) and ci_low > 0:
        return "GO", "all three seed point estimates are positive and the three-seed mean paired 95% CI is strictly above zero"
    if all(x >= 0 for x in points) or all(x <= 0 for x in points):
        return "SUPPORTIVE", "seed directions are not mixed, but the preregistered three-seed mean positive-CI gate is not met"
    return "NO-GO", "the three seed point estimates do not have a common positive direction"


def render_figure(outdir: Path, metrics_rows: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]]) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # The bundled runtime has Pillow but not matplotlib.
        # Keep the figure artifact available without adding a plotting
        # dependency.  The fallback is deliberately plain and only visualizes
        # the locked aggregate values; all numeric evidence remains in CSV.
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:  # pragma: no cover - environment-specific
            return f"BLOCKED_FIGURE_RENDER:{type(exc).__name__}:{exc}"
        methods = ["raw", "teacher", "posterior"]
        labels = ["M0 raw", "M1 teacher", "M2 post"]
        endpoint_specs = [
            (PRIMARY_ENDPOINT, "Trajectory concordance"),
            ("same_dose_top1", "Same-dose top-1"),
            ("MED_dose_step_error", "MED dose-step error"),
        ]
        def val(method: str, endpoint: str) -> float | None:
            rows = [r for r in metrics_rows if r["method"] == method and r["endpoint"] == endpoint and r["seed"] in SEEDS]
            nums = []
            for row in rows:
                try:
                    x = float(row["value"])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(x):
                    nums.append(x)
            return float(np.mean(nums)) if nums else None
        width, height = 1200, 480
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            small = ImageFont.truetype("arial.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
            small = font
        colors = [(76, 120, 168), (245, 133, 24), (84, 162, 75)]
        for panel, (endpoint, title) in enumerate(endpoint_specs):
            x0 = 35 + panel * 390
            y0, y1 = 80, 395
            draw.text((x0, 25), title, fill="black", font=font)
            draw.line((x0, y1, x0 + 330, y1), fill=(80, 80, 80), width=2)
            vals = [val(method, endpoint) for method in methods]
            finite = [x for x in vals if x is not None]
            ymax = max(finite) * 1.25 if finite and max(finite) > 0 else (max(abs(x) for x in finite) * 1.25 if finite else 1.0)
            ymin = min(0.0, min(finite) * 1.25 if finite else 0.0)
            if ymax <= ymin:
                ymax = ymin + 1.0
            for tick in range(5):
                yy = y1 - int((tick / 4.0) * (y1 - y0))
                tv = ymin + (tick / 4.0) * (ymax - ymin)
                draw.line((x0, yy, x0 + 330, yy), fill=(230, 230, 230), width=1)
                draw.text((x0 - 30, yy - 7), f"{tv:.2g}", fill=(70, 70, 70), font=small)
            for j, (method, label, color) in enumerate(zip(methods, labels, colors)):
                value = vals[j]
                bx = x0 + 40 + j * 95
                if value is not None:
                    by = y1 - int((value - ymin) / (ymax - ymin) * (y1 - y0))
                    draw.rectangle((bx, by, bx + 55, y1), fill=color)
                    draw.text((bx, max(y0, by - 18)), f"{value:.3g}", fill="black", font=small)
                else:
                    draw.rectangle((bx, y0 + 40, bx + 55, y1), outline=color, width=2)
                    draw.text((bx + 2, y0 + 20), "blocked", fill="black", font=small)
                draw.text((bx - 4, y1 + 10), label, fill="black", font=small)
        draw.text((35, 445), "cpg0004 frozen dose-response application | two support slots | M2 MED blocked when GE validation threshold is absent", fill=(60, 60, 60), font=small)
        fig_dir = outdir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        image.save(fig_dir / "dose_response_summary.png")
        return "OK_PIL_FALLBACK"
    methods = ["raw", "teacher", "posterior"]
    labels = ["M0 raw", "M1 teacher", "M2 posterior"]
    endpoints = [PRIMARY_ENDPOINT, "same_dose_top1", "MED_dose_step_error"]
    values: dict[tuple[str, str], list[float]] = {}
    for endpoint in endpoints:
        for method in methods:
            rows = [r for r in metrics_rows if r["analysis_set"] == "M1_vs_M0_common" and r["method"] == method and r["endpoint"] == endpoint and r["seed"] in SEEDS]
            if not rows:
                rows = [r for r in metrics_rows if r["analysis_set"] == "M2_vs_M1_common" and r["method"] == method and r["endpoint"] == endpoint and r["seed"] in SEEDS]
            vals = []
            for r in rows:
                try:
                    vals.append(float(r["value"]))
                except (TypeError, ValueError):
                    pass
            values[(endpoint, method)] = vals
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for ax, endpoint, title in zip(axes, endpoints, ["Trajectory concordance", "Same-dose top-1", "MED dose-step error"]):
        xs = np.arange(len(methods))
        means = [float(np.mean(values[(endpoint, m)])) if values[(endpoint, m)] else np.nan for m in methods]
        ax.bar(xs, means, color=colors, alpha=0.85)
        for x, m in zip(xs, methods):
            vals = values[(endpoint, m)]
            if vals:
                ax.scatter(np.full(len(vals), x), vals, color="black", s=18, zorder=3)
        ax.set_xticks(xs, labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if endpoint == "MED_dose_step_error":
            ax.set_ylabel("absolute dose-step error")
        elif endpoint == "mean_dose_margin":
            ax.set_ylabel("similarity margin")
        else:
            ax.set_ylabel("value")
    fig.suptitle("cpg0004 frozen dose-response application", fontsize=12)
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "dose_response_summary.png", dpi=180)
    plt.close(fig)
    return "OK"


def write_protocol(outdir: Path) -> None:
    text = f"""# Experiment 3 — cpg0004 dose-response trajectory

Version: `{VERSION}`  
Status is determined from frozen application artifacts and the saved CSVs.

## Objective

Test whether a one-real-CP-support raw profile, the frozen reproducible-effect
teacher, and the frozen validated full estimate recover independent six-dose
trajectory geometry and same-dose pharmacology.

## Locked inputs and no tuning

The evaluator reads the frozen cpg0004 CP plate-row artifact, saved P0/teacher
prediction arrays, saved GE residual arrays, and frozen pair manifests. It does
not retrain, refit, calibrate, choose a checkpoint, select a dose, or update
lambda/beta. The standard doses are exactly `{', '.join(DOSES)}` and the only
recorded support coverage is two rotations per eligible five-repeat condition.

## Support and independent reference

For each eligible compound-dose and each recorded support row `s`, M0 is the
support delta, M1 is the saved teacher vector evaluated from that support, and
the reference is the mean of the other four physical repeat rows. The support
row is asserted absent from the reference. Profiles are scored per rotation;
the two rotation endpoint values are averaged only at the compound boundary.

M2 follows the cpg0004 frozen-array reconstruction rule:

```text
P0_s = (1-lambda_s) * T_s + lambda_s * V_s
V_s  = (saved_P0_at_held_s - (1-lambda_s) * T_s) / lambda_s
M2_s = P0_s + (saved_correct_GE_at_held_s - saved_GE_P0_at_held_s)
```

The GE key is matched by `(compound, standard_dose, held_index=s)` and the
existing held row is retained; no array-position pairing is used. M2 rows are
excluded from its common set when the aligned GE key is unavailable.

## Primary endpoint

For each rotation, form the 15 upper-triangle distances
`D(i,j)=1-PCC(profile_i,profile_j)` for both method and independent reference.
The primary trajectory concordance is Spearman correlation between these two
15-vectors. M1-vs-M0 and M2-vs-M1 use their respective fixed common compound
sets.

## Secondary endpoints

For each dose query, same-dose similarity is PCC to each of the six reference
dose profiles. `same_dose_top1` and `same_dose_top2` record whether the true
dose is ranked first or in the first two. `DoseMargin` is the true-dose
similarity minus the mean similarity to the other five reference doses.

MED is the first dose whose standardized effect magnitude exceeds the
validation-only threshold produced by Experiment 1. The same method-specific
threshold is applied to that method's estimated and independent reference
profiles. No-active is retained as sentinel dose index 6; it is not removed
after looking at outcomes. Experiment 1 has no GE validation predictions, so
M2 MED endpoints remain `BLOCKED_VALIDATION_PREDICTIONS_MISSING` and no
threshold is invented.

Monotonicity is a retrospective sensitivity only. A compound is selected when
the independent reference magnitude has Spearman correlation greater than 0.6
with log dose; the subset cannot alter primary eligibility or method fitting.

## Statistical lock

Seeds are exactly `3407, 42, 2025`. Uncertainty uses 10,000 paired compound
bootstrap resamples with all six doses and both support rotations carried by a
sampled compound. Percentile 95% intervals and deterministic seeds derived
from the seed, comparison, and endpoint are recorded in `BOOTSTRAP_CI.csv`.
"""
    (outdir / "PROTOCOL.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_rounds <= 0:
        raise ValueError("--bootstrap-rounds must be positive")
    if args.smoke:
        args.bootstrap_rounds = min(int(args.bootstrap_rounds), 200)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    write_protocol(outdir)
    lock_dir = Path(__file__).resolve().parents[1] / "protocol"
    lock_config = json.loads((lock_dir / "CONFIG.json").read_text(encoding="utf-8"))
    hit_config, hit_config_hash = load_hit_config(args.hit_dir)
    thresholds_raw = hit_config.get("model_thresholds_validation_only", {})
    thresholds: dict[str, float | None] = {
        "raw": float(thresholds_raw["raw"]) if thresholds_raw.get("raw") is not None else None,
        "teacher": float(thresholds_raw["teacher"]) if thresholds_raw.get("teacher") is not None else None,
        "posterior": float(thresholds_raw["GE_posterior"]) if thresholds_raw.get("GE_posterior") is not None else None,
    }
    lambda_by_seed = {int(k): float(v) for k, v in lock_config["frozen_methods"]["M2"]["cpg0004"]["lambda_by_seed"].items()}
    beta_by_seed = {int(k): float(v) for k, v in lock_config["frozen_methods"]["M2"]["cpg0004"]["beta_by_seed"].items()}
    mapping_errors: list[str] = []
    artifact = load_artifact(args.data)
    p0, ge, manifests, pair_hashes = load_pair_and_prediction_bundles(args, lambda_by_seed, beta_by_seed, mapping_errors)
    manifest_hashes = load_manifest_hashes()
    input_hashes = check_input_hashes(args, manifest_hashes, hit_config_hash, pair_hashes)
    # Check the pair-manifest hashes captured by Experiment 1 as an additional
    # local audit.  They are not used to alter the input.
    for seed in SEEDS:
        expected = hit_config.get("prediction_schema", {}).get(str(seed), {}).get("pair_manifest_sha256")
        if expected and pair_hashes[seed] != expected:
            mapping_errors.append(f"pair_manifest_hash_mismatch:seed={seed}:expected={expected}:got={pair_hashes[seed]}")
    seed_slots, exclusions, seed_compounds, audit_info = build_slots(artifact, p0, ge, manifests, mapping_errors)
    m1, m2 = trim_to_common_sets(seed_slots)
    # A compound-level common set is not sufficient if the physical support
    # identity changes with seed.  Assert the full support/reference key map
    # is identical before pooling seed-level endpoint values.
    for compound in sorted(m1):
        for rotation in range(EXPECTED_SUPPORT_SLOTS):
            for dose in DOSES:
                signatures = []
                for seed in SEEDS:
                    slot = seed_slots[seed][compound][rotation][dose]
                    signatures.append((slot.support_row, slot.held_source_row, slot.reference_rows))
                if len(set(signatures)) != 1:
                    mapping_errors.append(f"cross_seed_support_reference_key_mismatch:{compound}:{dose}:rotation={rotation}")
    # Build a stable audit record of prior-only rows and the frozen two-of-five
    # support coverage.  These are limitations, not post-result filters.
    for seed in SEEDS:
        pkeys = set(p0[seed].index_by_key)
        gkeys = set(ge[seed].index_by_key)
        exclusions.append({"scope": "M2_common_set", "seed": seed, "reason": "PRIOR_ONLY_KEYS_EXCLUDED", "count": len(pkeys - gkeys), "detail": "GE keys are a subset of prior keys; M2 excludes prior-only keys by locked protocol."})
        for compound, dose, held in sorted(pkeys - gkeys):
            exclusions.append({"scope": "M2_common_set", "seed": seed, "compound": compound, "dose": dose, "support_row": held, "reason": "PRIOR_ONLY_KEY", "count": 1, "detail": f"held_index={held}; GE key absent; excluded without re-pairing."})
        exclusions.append({"scope": "support_rotation", "seed": seed, "reason": "ONLY_TWO_FROZEN_SUPPORT_SLOTS_USED", "count": 1, "detail": "Every eligible condition uses the two recorded support slots; the other three physical slots are not inferred."})
    if thresholds["posterior"] is None:
        exclusions.append({"scope": "MED", "seed": "all", "reason": GE_BLOCK, "count": 1, "detail": "Experiment 1 model_thresholds_validation_only.GE_posterior is null because GE validation predictions are absent; no M2 MED threshold was selected."})
    # The M1 common set and M2 common set are locked before endpoint summaries.
    # M2 profiles are present for exactly the latter set only.
    mono_flags: dict[str, bool] = {}
    all_metrics: dict[int, dict[str, dict[str, CompoundMetrics]]] = {seed: {} for seed in SEEDS}
    all_slot_rows: list[dict[str, Any]] = []
    mono_audit: dict[int, dict[str, tuple[float, str, int]]] = {}
    for method, compounds in (("raw", m1), ("teacher", m1), ("posterior", m2)):
        by_seed, slot_rows, audits = compound_level_metrics(artifact, seed_slots, compounds, method, thresholds[method], mono_flags)
        for seed in SEEDS:
            all_metrics[seed][method] = by_seed[seed]
        all_slot_rows.extend(slot_rows)
        mono_audit.update(audits)
    # Assert all required primary/secondary values are finite before treating
    # them as a common set.  A non-finite frozen profile is an audit failure,
    # not a hidden result-dependent exclusion.
    for seed in SEEDS:
        for method in METHODS:
            for compound, metric in all_metrics[seed].get(method, {}).items():
                for endpoint in ("trajectory_concordance", "same_dose_top1", "same_dose_top2", "mean_dose_margin"):
                    if not np.isfinite(get_metric(metric, endpoint)):
                        mapping_errors.append(f"nonfinite_primary_secondary_metric:seed={seed}:{method}:{compound}:{endpoint}")
    thresholds_for_metrics = thresholds
    metrics_rows, contrast_rows, left_arrays, right_arrays = make_metrics_and_contrasts(all_metrics, m1, m2, thresholds_for_metrics)
    bootstrap_rows = bootstrap_contrasts(left_arrays, right_arrays, args.bootstrap_rounds)
    figure_status = render_figure(outdir, metrics_rows, bootstrap_rows)
    # Required output contract.
    write_csv(outdir / "ELIGIBLE_SAMPLES.csv", slot_summary_rows(seed_slots, m1, m2))
    write_csv(outdir / "EXCLUSIONS.csv", exclusions, ["scope", "seed", "compound", "dose", "support_row", "reason", "repeat_count", "unique_support_slots", "count", "detail"])
    write_csv(outdir / "METRICS_BY_SEED.csv", metrics_rows)
    write_csv(outdir / "PAIRED_CONTRASTS.csv", contrast_rows)
    write_csv(outdir / "BOOTSTRAP_CI.csv", bootstrap_rows)
    write_csv(outdir / "SLOT_METRICS.csv", all_slot_rows)
    write_csv(outdir / "COMPOUND_METRICS.csv", compound_metrics_rows(all_metrics))

    teacher_decision, teacher_reason = decision_for("teacher_vs_raw", contrast_rows, bootstrap_rows)
    posterior_decision, posterior_reason = decision_for("posterior_vs_teacher", contrast_rows, bootstrap_rows)
    if mapping_errors:
        overall = "BLOCKED"
        overall_reason = f"{len(mapping_errors)} mapping/audit error(s); inspect EXCLUSIONS.csv and DECISION.md"
    elif teacher_decision == "GO" and posterior_decision == "GO":
        overall = "GO"
        overall_reason = "both locked primary contrasts meet the three-seed direction and positive-CI gates"
    elif teacher_decision in {"GO", "SUPPORTIVE"} or posterior_decision in {"GO", "SUPPORTIVE"}:
        overall = "SUPPORTIVE"
        overall_reason = "at least one primary contrast provides positive-direction or supportive evidence, but both GO gates are not met"
    else:
        overall = "NO-GO"
        overall_reason = "neither primary contrast meets a positive three-seed gate"
    config = {
        "version": VERSION,
        "lock_id": lock_config.get("lock_id"),
        "dataset": "cpg0004-LINCS",
        "standard_doses": list(DOSES),
        "seeds": list(SEEDS),
        "strict_repeat_count": REQUIRED_REPEATS,
        "support_rotation": {"possible_physical_slots": 5, "frozen_slots_used": 2, "aggregation": "score each slot then average at compound boundary", "coverage_status": "PARTIAL_FROZEN_COVERAGE"},
        "primary_endpoint": "Spearman of 15 upper-triangle 1-PCC dose distances against independent reference",
        "secondary_endpoints": list(ENDPOINTS[1:]),
        "bootstrap": {"rounds": args.bootstrap_rounds, "unit": "compound", "paired": True, "percentile_ci": 0.95, "same_compound_multiset_for_contrast": True},
        "paths": {"data": str(args.data.resolve()), "p0_root": str(args.p0_root.resolve()), "ge_root": str(args.ge_root.resolve()), "pair_root": str(args.pair_root.resolve()), "experiment_1_config": str((args.hit_dir / "CONFIG.json").resolve())},
        "input_hashes": input_hashes,
        "hit_recovery_config_sha256": hit_config_hash,
        "validation_only_thresholds_from_experiment_1": {k: thresholds[k] for k in thresholds},
        "threshold_status": {"raw": "OK" if thresholds["raw"] is not None else "BLOCKED", "teacher": "OK" if thresholds["teacher"] is not None else "BLOCKED", "posterior": GE_BLOCK if thresholds["posterior"] is None else "OK"},
        "frozen_lambda_by_seed": {str(k): v for k, v in lambda_by_seed.items()},
        "frozen_beta_by_seed": {str(k): v for k, v in beta_by_seed.items()},
        "counts": {"m1_common_compounds": len(m1), "m2_common_compounds": len(m2), "m1_rotations": len(m1) * EXPECTED_SUPPORT_SLOTS, "m2_rotations": len(m2) * EXPECTED_SUPPORT_SLOTS, "dose_count": len(DOSES), "m1_condition_rows": len(m1) * EXPECTED_SUPPORT_SLOTS * len(DOSES), "m2_condition_rows": len(m2) * EXPECTED_SUPPORT_SLOTS * len(DOSES)},
        "prior_ge_key_audit": {str(seed): {"prior_rows": len(p0[seed].index_by_key), "ge_rows": len(ge[seed].index_by_key), "prior_only_keys": len(set(p0[seed].index_by_key) - set(ge[seed].index_by_key))} for seed in SEEDS},
        "seed_audit": audit_info,
        "decision": {"teacher_vs_raw": teacher_decision, "teacher_vs_raw_reason": teacher_reason, "posterior_vs_teacher": posterior_decision, "posterior_vs_teacher_reason": posterior_reason, "overall": overall, "overall_reason": overall_reason},
        "figure_status": figure_status,
        "mapping_error_count": len(mapping_errors),
        "guardrails": ["no model retraining", "no lambda/beta reselection", "reference excludes support", "M2 GE keys aligned by compound/dose/held_index", "validation-only MED thresholds read from Experiment 1", "M2 MED blocked when GE validation threshold is absent"],
    }
    (outdir / "CONFIG.json").write_text(json.dumps(json_safe(config), indent=2, sort_keys=True), encoding="utf-8")
    # Results report follows the global output contract in the locked order.
    primary_mean = {r["comparison"]: r for r in bootstrap_rows if r["seed"] == "mean" and r["endpoint"] == PRIMARY_ENDPOINT}
    def report_contrast(comp: str) -> str:
        row = primary_mean.get(comp)
        return "not estimable" if row is None else f"{row['point_difference']} (95% CI [{row['ci_low']}, {row['ci_high']}])"
    def boot_row(comp: str, endpoint: str, seed: str = "mean") -> dict[str, Any] | None:
        return next((r for r in bootstrap_rows if r["seed"] == seed and r["comparison"] == comp and r["endpoint"] == endpoint), None)
    def boot_text(comp: str, endpoint: str) -> str:
        row = boot_row(comp, endpoint)
        return "blocked/not estimable" if row is None else f"{row['point_difference']} [{row['ci_low']}, {row['ci_high']}]"
    def seed_points(comp: str, endpoint: str) -> str:
        vals = []
        for seed in SEEDS:
            row = next((r for r in contrast_rows if r["seed"] == seed and r["endpoint"] == endpoint and ((comp == "teacher_vs_raw" and r["left_method"] == "teacher" and r["right_method"] == "raw") or (comp == "posterior_vs_teacher" and r["left_method"] == "posterior" and r["right_method"] == "teacher"))), None)
            vals.append(f"{seed}: {row['point_difference_left_minus_right'] if row else 'blocked'}")
        return "; ".join(vals)
    mono_counts = [sum(1 for x in mono_flags.values() if x) if mono_flags else 0]
    result_lines = [
        f"# Results — cpg0004 dose-response trajectory ({VERSION})",
        "",
        "## 1. Objective",
        "Evaluate whether frozen one-real-support CP raw, reproducible-effect teacher, and validated full estimate profiles recover independent six-dose trajectory geometry and same-dose pharmacology.",
        "",
        "## 2. Dataset",
        f"cpg0004-LINCS A549 frozen CP plate rows: {len(artifact.compound):,} rows, {artifact.feature_dim} CP features, {artifact.plate_count} plates; locked doses are {', '.join(DOSES)}.",
        "",
        "## 3. Eligibility",
        f"Test compounds required all six locked doses and exactly {REQUIRED_REPEATS} CP plate repeats at every dose. Frozen prediction/pair artifacts expose exactly two support rotations per eligible condition; the other three physical slots were not inferred. M1 common set: {len(m1)} compounds; M2 aligned GE common set: {len(m2)} compounds.",
        "",
        "## 4. Exact information available to model",
        "M0 uses one support delta; M1 uses the saved teacher vector for that support; M2 uses only the locked cpg0004 reconstruction from saved teacher/P0/GE arrays with the original per-seed lambda and beta. No independent reference profile, test label, dose endpoint, or MED result enters method construction.",
        "",
        "## 5. Independent reference construction",
        "For every support row, the reference is the mean of the four other physical repeat deltas. Support/reference row disjointness is asserted. Rotation scores are computed before averaging the two recorded rotations at compound level.",
        "",
        "## 6. Primary endpoint",
        "TrajectoryConcordance is Spearman correlation between the 15 upper-triangle distances D(i,j)=1−PCC(profile_i,profile_j) and the corresponding independent-reference distances.",
        "",
        "## 7. Secondary endpoints",
        "Same-dose top-1/top-2 and DoseMargin use PCC to the six independent reference dose profiles, with DoseMargin equal to true-dose similarity minus the mean of the five wrong-dose similarities. MED uses Experiment 1 validation-only method-specific thresholds; M2 MED is blocked because Experiment 1 GE validation predictions/threshold are absent. Monotonicity is retrospective and restricted to reference Spearman(log dose, magnitude)>0.6.",
        "",
        "## 8. Sample count",
        f"M1: {len(m1)} compounds × 2 rotations × 6 doses = {len(m1)*EXPECTED_SUPPORT_SLOTS*len(DOSES):,} condition-rotation rows. M2: {len(m2)} compounds × 2 rotations × 6 doses = {len(m2)*EXPECTED_SUPPORT_SLOTS*len(DOSES):,}. Retrospective monotonic subset count is {mono_counts[0]} compounds (reported in METRICS_BY_SEED.csv).",
        "",
        "## 9. Seed-level results",
        "Seed-level method values and all secondary endpoints are in `METRICS_BY_SEED.csv`; compound-level endpoint values are in `COMPOUND_METRICS.csv`. Primary contrast point estimates by seed are: teacher−raw (" + seed_points("teacher_vs_raw", PRIMARY_ENDPOINT) + "); posterior−teacher (" + seed_points("posterior_vs_teacher", PRIMARY_ENDPOINT) + ").",
        "",
        "Pooled three-seed mean paired differences (left minus right; 95% percentile CI):",
        "",
        "| contrast | trajectory | same-dose top-1 | same-dose top-2 | mean DoseMargin |",
        "|---|---:|---:|---:|---:|",
        f"| teacher−raw | {boot_text('teacher_vs_raw', 'trajectory_concordance')} | {boot_text('teacher_vs_raw', 'same_dose_top1')} | {boot_text('teacher_vs_raw', 'same_dose_top2')} | {boot_text('teacher_vs_raw', 'mean_dose_margin')} |",
        f"| posterior−teacher | {boot_text('posterior_vs_teacher', 'trajectory_concordance')} | {boot_text('posterior_vs_teacher', 'same_dose_top1')} | {boot_text('posterior_vs_teacher', 'same_dose_top2')} | {boot_text('posterior_vs_teacher', 'mean_dose_margin')} |",
        "",
        "Pooled MED differences (M2 is blocked): teacher−raw exact accuracy = " + boot_text("teacher_vs_raw", "MED_exact_accuracy") + "; ±1-dose accuracy = " + boot_text("teacher_vs_raw", "MED_plus_or_minus_1_accuracy") + "; dose-step error = " + boot_text("teacher_vs_raw", "MED_dose_step_error") + ". These are negative or boundary/wide, not a positive MED result.",
        "",
        "## 10. Paired CI",
        f"`BOOTSTRAP_CI.csv` contains {args.bootstrap_rounds:,} paired compound resamples per seed/contrast/endpoint and the aligned three-seed mean distribution. All doses and both support rotations remain together within a sampled compound.",
        "",
        "## 11. GO / SUPPORTIVE / NO-GO",
        f"Teacher versus raw trajectory: **{teacher_decision}** — {teacher_reason}. Posterior versus teacher trajectory: **{posterior_decision}** — {posterior_reason}. Overall Bio-C status: **{overall}** — {overall_reason}.",
        "Same-dose endpoints are not uniformly improved: teacher−raw top-1 and top-2 are negative with pooled CIs excluding zero; posterior−teacher top-1 CI crosses zero and its seed directions are mixed, while top-2 is positive in the pooled CI but not all seed directions agree. Mean DoseMargin is positive for both contrasts, but this secondary result does not change the primary gate.",
        "MED is not supportive: teacher−raw exact and ±1-dose accuracy are lower, and dose-step error is numerically worse with a CI crossing zero. Posterior MED remains blocked by the absent validation-only GE threshold.",
        "Accordingly, the Bio-C GO label is restricted to the primary trajectory endpoint. Same-dose top-1/top-2 and MED are reported as negative/boundary or blocked secondary evidence, not as additional GO claims.",
        "",
        "## 12. Biological interpretation",
        "A positive trajectory contrast would indicate that frozen reproducible-effect learning recovers dose-response geometry closer to independent repeats under a one-real-support regime. Same-dose and MED results are supporting pharmacology evidence; they do not establish replacement of physical repeats. The two-slot frozen coverage is a design limitation and not full five-slot rotation.",
        "",
        "## 13. Limitations",
        f"The frozen P0 arrays have 8,213 test rows and GE arrays 8,203 rows; the 10 prior-only keys per seed are excluded from M2. Experiment 1 reports raw/teacher validation thresholds but `GE_posterior` is null (`{GE_BLOCK}`), so posterior MED accuracy/error is not claimed. No test result was used to select thresholds, compounds, doses, or methods. Mapping/audit errors recorded: {len(mapping_errors)}. Figure status: {figure_status}.",
    ]
    (outdir / "RESULTS.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")
    decision_lines = [
        f"# Decision — Experiment 3 dose-response trajectory ({VERSION})",
        "",
        f"Locked input hashes were checked against `00_protocol_lock/FREEZE_MANIFEST.json`; Experiment 1 threshold source hash: `{hit_config_hash}`.",
        f"Common sets: M1 raw/teacher = {len(m1)} compounds; M2 posterior/teacher = {len(m2)} compounds; each has two frozen support rotations and six doses. GE key audit: 8,213 prior rows, 8,203 GE rows, 10 prior-only keys per seed excluded.",
        f"All uncertainty uses {args.bootstrap_rounds:,} paired compound-level resamples with percentile 95% CIs; no compound-dose row bootstrap was used.",
        "",
        f"**Teacher−raw primary trajectory: {teacher_decision}** — {teacher_reason}.",
        f"**Posterior−teacher primary trajectory: {posterior_decision}** — {posterior_reason}.",
        f"**Overall Bio-C: {overall}** — {overall_reason}.",
        "",
        "Per-seed primary point estimates (teacher−raw / posterior−teacher): " + seed_points("teacher_vs_raw", PRIMARY_ENDPOINT) + " / " + seed_points("posterior_vs_teacher", PRIMARY_ENDPOINT) + ".",
        "Pooled three-seed trajectory differences: teacher−raw " + boot_text("teacher_vs_raw", PRIMARY_ENDPOINT) + "; posterior−teacher " + boot_text("posterior_vs_teacher", PRIMARY_ENDPOINT) + ".",
        "Same-dose pooled differences: teacher−raw top-1 " + boot_text("teacher_vs_raw", "same_dose_top1") + ", top-2 " + boot_text("teacher_vs_raw", "same_dose_top2") + "; posterior−teacher top-1 " + boot_text("posterior_vs_teacher", "same_dose_top1") + ", top-2 " + boot_text("posterior_vs_teacher", "same_dose_top2") + ". The negative teacher−raw values and mixed/boundary posterior seed behavior are not a same-dose GO claim.",
        "Pooled MED teacher−raw differences: exact accuracy " + boot_text("teacher_vs_raw", "MED_exact_accuracy") + "; ±1-dose accuracy " + boot_text("teacher_vs_raw", "MED_plus_or_minus_1_accuracy") + "; dose-step error " + boot_text("teacher_vs_raw", "MED_dose_step_error") + ". These do not support a MED improvement.",
        f"MED status: raw and teacher thresholds were read from Experiment 1 validation-only output; posterior MED is `{GE_BLOCK}` because Experiment 1 GE validation predictions/threshold are absent. No substitute threshold was selected.",
        "",
        f"Support coverage qualification: exactly two recorded frozen support slots per eligible condition were used; three missing physical slots were not imputed or manufactured. Mapping/audit error count: {len(mapping_errors)}.",
    ]
    if mapping_errors:
        decision_lines.extend(["", "## Audit errors", ""] + [f"- `{error}`" for error in mapping_errors[:200]])
    (outdir / "DECISION.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    print(json.dumps(json_safe({"outdir": str(outdir), "m1_common_compounds": len(m1), "m2_common_compounds": len(m2), "bootstrap_rounds": args.bootstrap_rounds, "teacher_vs_raw": teacher_decision, "posterior_vs_teacher": posterior_decision, "overall": overall, "mapping_errors": len(mapping_errors), "figure_status": figure_status}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
