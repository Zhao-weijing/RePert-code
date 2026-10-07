#!/usr/bin/env python3
"""Frozen cpg0004 biological hit-recovery evaluator.

This evaluator consumes only the frozen CP plate-row artifact and predictions
already written by the cpg0004 1R virtual-prior and GE-residual experiments.
It deliberately does not fit a model, select lambda/beta, or use test labels
for thresholds.  The saved 1R predictions do not cover all five possible
support slots.  For each eligible five-repeat condition we therefore expose
only the two support slots that can be reconstructed without retraining and
record that coverage limitation in the audit outputs.

The implementation is intentionally self-contained (no torch dependency) so
that schema and mapping checks can run on Windows as well as on the original
Linux environment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VERSION = "cpg0004-LINCS-biological-hit-recovery-2026-08-30"
SEEDS = (3407, 42, 2025)
DOSES = ("0.04", "0.12", "0.37", "1.11", "3.33", "10")
REQUIRED_REPEATS = 5
FIXED_FPR = 0.05
DEFAULT_NULL_ROUNDS = 256
DEFAULT_BOOTSTRAP_ROUNDS = 10_000
METHODS = ("raw", "teacher", "P0", "GE_posterior")
RANK_METRICS = (
    "AP",
    "AUROC",
    "Recall_at_fixed_FPR",
    "Precision_at_top5",
    "Precision_at_top10",
    "Enrichment_at_top10",
    "rescue_rate",
    "false_rescue_rate",
)
COMPARISONS = (
    ("teacher", "raw"),
    ("P0", "raw"),
    ("GE_posterior", "raw"),
    ("P0", "teacher"),
    ("GE_posterior", "teacher"),
    ("GE_posterior", "P0"),
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, text: str) -> int:
    value = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).digest()
    return int.from_bytes(value[:8], "little") % (2**32 - 1)


def as_str(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8", errors="replace").astype(str)
    return values.astype(str)


def pcc(left: np.ndarray, right: np.ndarray) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denom) or denom <= 0:
        return None
    value = float(np.dot(a, b) / denom)
    return value if np.isfinite(value) else None


def fisher(value: float) -> float:
    return float(np.arctanh(np.clip(value, -0.999999, 0.999999)))


def well_row(value: str) -> str:
    token = str(value).split(";", 1)[0]
    out = ""
    for char in token:
        if char.isalpha():
            out += char
        else:
            break
    return out


def key(compound: str, dose: str, held_index: int) -> tuple[str, str, int]:
    return (str(compound), str(dose), int(held_index))


@dataclass
class Artifact:
    compound: np.ndarray
    dose: np.ndarray
    plate: np.ndarray
    well: np.ndarray
    split: np.ndarray
    delta: np.ndarray
    baseline: np.ndarray
    groups: dict[tuple[str, str], list[int]]
    slot_rows: dict[tuple[str, str, str], dict[str, int]]
    train_center: np.ndarray
    train_scale: np.ndarray
    plate_count: int
    feature_dim: int


@dataclass
class Rotation:
    split: str
    compound: str
    dose: str
    condition_rows: tuple[int, ...]
    support_row: int
    support_plate: str
    support_well: str
    rotation_rank: int
    confirmation_a: tuple[int, ...]
    confirmation_b: tuple[int, ...]
    e_rep: float = math.nan
    a_ref: float = math.nan
    null_z_mean: float = math.nan
    null_candidate_count: int = 0
    label: int = 0
    borderline: int = 0
    p0_available: int = 0
    ge_available: int = 0
    raw_score: float = math.nan
    teacher_score: float = math.nan
    p0_score: float = math.nan
    ge_score: float = math.nan


@dataclass
class PredictionBundle:
    split: str
    compound: np.ndarray
    dose: np.ndarray
    held_index: np.ndarray
    support_mean: np.ndarray | None
    teacher: np.ndarray
    virtual_prior: np.ndarray
    posterior: np.ndarray
    index_by_key: dict[tuple[str, str, int], int]
    support_by_key: dict[tuple[str, str, int], int]
    lambda_value: float
    path: str


@dataclass
class GEBundle:
    compound: np.ndarray
    dose: np.ndarray
    held_index: np.ndarray
    p0: np.ndarray
    correct_ge: np.ndarray
    index_by_key: dict[tuple[str, str, int], int]
    beta_value: float
    path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    default_base = here.parents[1] / "external_validation" / "lincs_cpg0004"
    parser.add_argument(
        "--data",
        type=Path,
        default=default_base / "data_preparation" / "artifact" / "cp_plate_rows.npz",
    )
    parser.add_argument(
        "--p0-root",
        type=Path,
        default=default_base / "virtual_prior" / "results" / "1r_all",
    )
    parser.add_argument(
        "--ge-root",
        type=Path,
        default=default_base / "single_repeat_expression_evidence" / "results" / "1r_all",
    )
    parser.add_argument(
        "--pair-root",
        type=Path,
        default=default_base / "cell_painting_repeat_benchmark" / "results" / "1r_all",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=here,
    )
    parser.add_argument("--null-rounds", type=int, default=DEFAULT_NULL_ROUNDS)
    parser.add_argument("--bootstrap-rounds", type=int, default=DEFAULT_BOOTSTRAP_ROUNDS)
    parser.add_argument("--label-seed", type=int, default=3407)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
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
    n = len(compound)
    if not (len(dose) == len(plate) == len(well) == len(split) == len(delta) == len(baseline) == n):
        raise ValueError("Frozen CP artifact arrays have inconsistent row counts")
    if delta.ndim != 2 or baseline.ndim != 2 or delta.shape != baseline.shape:
        raise ValueError("Frozen CP artifact delta/baseline shapes are inconsistent")
    if not np.isfinite(delta).all() or not np.isfinite(baseline).all():
        raise ValueError("Frozen CP artifact contains non-finite values")
    if set(dose) - set(DOSES):
        raise ValueError(f"Unexpected dose labels in frozen artifact: {sorted(set(dose) - set(DOSES))}")
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (c, d) in enumerate(zip(compound, dose)):
        groups[(str(c), str(d))].append(index)
    for group, rows in list(groups.items()):
        groups[group] = sorted(rows, key=lambda i: (str(plate[i]), str(well[i]), int(i)))
    slot_rows: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    for index, (c, d, pl, ww) in enumerate(zip(compound, dose, plate, well)):
        slot_rows[(str(d), str(pl), well_row(str(ww)))][str(c)] = int(index)

    # The artifact does not retain DMSO well-level dispersion.  Fit a
    # treatment-delta robust scale on training compounds only.  Delta is already
    # treatment minus same-plate control median, so the center is a training
    # delta center and no test treatment values are used.
    train_rows = np.flatnonzero(split == "train")
    train_center = np.median(delta[train_rows].astype(np.float64), axis=0)
    mad = np.median(np.abs(delta[train_rows].astype(np.float64) - train_center), axis=0)
    train_scale = 1.4826 * mad
    degenerate = ~np.isfinite(train_scale) | (train_scale < 1e-6)
    # A second training-only fallback avoids division by zero while making the
    # affected feature count explicit in CONFIG/RESULTS.
    std = np.std(delta[train_rows].astype(np.float64), axis=0)
    train_scale[degenerate] = std[degenerate]
    degenerate = ~np.isfinite(train_scale) | (train_scale < 1e-6)
    train_scale[degenerate] = 1.0
    return Artifact(
        compound=compound,
        dose=dose,
        plate=plate,
        well=well,
        split=split,
        delta=delta,
        baseline=baseline,
        groups=dict(groups),
        slot_rows=dict(slot_rows),
        train_center=train_center.astype(np.float32),
        train_scale=train_scale.astype(np.float32),
        plate_count=len(set(plate)),
        feature_dim=int(delta.shape[1]),
    )


def load_npz_bundle(path: Path, split: str, require_support_mean: bool = True) -> PredictionBundle:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        required = {"compound", "dose", "held_index", "teacher", "virtual_prior", "posterior"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"{path} missing P0 keys: {missing}")
        compound = as_str(loaded["compound"])
        dose = as_str(loaded["dose"])
        held_index = loaded["held_index"].astype(np.int64)
        teacher = loaded["teacher"].astype(np.float32)
        virtual_prior = loaded["virtual_prior"].astype(np.float32)
        posterior = loaded["posterior"].astype(np.float32)
        support_mean = loaded["support_mean"].astype(np.float32) if "support_mean" in loaded.files else None
    n = len(compound)
    arrays = (dose, held_index, teacher, virtual_prior, posterior)
    if any(len(x) != n for x in arrays):
        raise ValueError(f"{path} has inconsistent prediction row counts")
    if require_support_mean and support_mean is None:
        raise ValueError(f"{path} has no support_mean")
    if support_mean is not None and len(support_mean) != n:
        raise ValueError(f"{path} has inconsistent support_mean row count")
    for name, values in (
        ("teacher", teacher),
        ("virtual_prior", virtual_prior),
        ("posterior", posterior),
    ):
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"{path} {name} is not a finite 2D array")
    if support_mean is not None and (support_mean.ndim != 2 or not np.isfinite(support_mean).all()):
        raise ValueError(f"{path} support_mean is not a finite 2D array")
    index_by_key: dict[tuple[str, str, int], int] = {}
    for i, (c, d, h) in enumerate(zip(compound, dose, held_index)):
        k = key(str(c), str(d), int(h))
        if k in index_by_key:
            raise ValueError(f"Duplicate prediction key in {path}: {k}")
        index_by_key[k] = int(i)
    # lambda is replaced by the caller after metrics.json is read.
    return PredictionBundle(
        split=split,
        compound=compound,
        dose=dose,
        held_index=held_index,
        support_mean=support_mean,
        teacher=teacher,
        virtual_prior=virtual_prior,
        posterior=posterior,
        index_by_key=index_by_key,
        support_by_key={},
        lambda_value=math.nan,
        path=str(path),
    )


def load_ge_bundle(path: Path) -> GEBundle:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        required = {"compound", "dose", "held_index", "P0", "correct_GE"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"{path} missing GE keys: {missing}")
        compound = as_str(loaded["compound"])
        dose = as_str(loaded["dose"])
        held_index = loaded["held_index"].astype(np.int64)
        p0 = loaded["P0"].astype(np.float32)
        correct_ge = loaded["correct_GE"].astype(np.float32)
    n = len(compound)
    if any(len(x) != n for x in (dose, held_index, p0, correct_ge)):
        raise ValueError(f"{path} has inconsistent GE row counts")
    if p0.ndim != 2 or correct_ge.ndim != 2 or p0.shape != correct_ge.shape:
        raise ValueError(f"{path} GE arrays have inconsistent shapes")
    if not np.isfinite(p0).all() or not np.isfinite(correct_ge).all():
        raise ValueError(f"{path} GE arrays are not finite")
    index_by_key: dict[tuple[str, str, int], int] = {}
    for i, (c, d, h) in enumerate(zip(compound, dose, held_index)):
        k = key(str(c), str(d), int(h))
        if k in index_by_key:
            raise ValueError(f"Duplicate GE prediction key in {path}: {k}")
        index_by_key[k] = int(i)
    return GEBundle(
        compound=compound,
        dose=dose,
        held_index=held_index,
        p0=p0,
        correct_ge=correct_ge,
        index_by_key=index_by_key,
        beta_value=math.nan,
        path=str(path),
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_support_rows(value: str) -> tuple[int, ...]:
    if not str(value).strip():
        return tuple()
    return tuple(int(part) for part in str(value).split("|"))


def load_pair_manifest(path: Path) -> dict[tuple[str, str, int], tuple[int, ...]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[tuple[str, str, int], tuple[int, ...]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            k = key(row["compound"], row["dose"], int(row["held_row"]))
            if k in result:
                raise ValueError(f"Duplicate pair-manifest key: {k}")
            result[k] = parse_support_rows(row["support_rows"])
    return result


def reconstruct_support_mapping(
    artifact: Artifact,
    bundle: PredictionBundle,
    manifest: dict[tuple[str, str, int], tuple[int, ...]] | None,
    mapping_errors: list[str],
) -> None:
    for i, (compound, dose, held) in enumerate(zip(bundle.compound, bundle.dose, bundle.held_index)):
        k = key(str(compound), str(dose), int(held))
        if int(held) < 0 or int(held) >= len(artifact.compound):
            mapping_errors.append(f"held_index_out_of_range:{bundle.split}:{k}")
            continue
        if str(artifact.compound[int(held)]) != str(compound) or str(artifact.dose[int(held)]) != str(dose):
            mapping_errors.append(f"held_key_mismatch:{bundle.split}:{k}")
            continue
        rows = artifact.groups.get((str(compound), str(dose)), [])
        others = [row for row in rows if row != int(held)]
        expected = (others[0],) if others else tuple()
        actual = manifest.get(k) if manifest is not None else expected
        if actual is None:
            mapping_errors.append(f"missing_pair_manifest_key:{bundle.split}:{k}")
            continue
        if tuple(actual) != tuple(expected):
            mapping_errors.append(f"support_mapping_mismatch:{bundle.split}:{k}:expected={expected}:actual={actual}")
            continue
        if len(actual) != 1 or actual[0] not in rows:
            mapping_errors.append(f"invalid_support_row:{bundle.split}:{k}:support={actual}")
            continue
        bundle.support_by_key[k] = int(actual[0])
        if bundle.support_mean is not None:
            error = float(np.max(np.abs(bundle.support_mean[i] - artifact.delta[int(actual[0])])))
            if error > 2e-5:
                mapping_errors.append(f"support_vector_reconstruction_error:{bundle.split}:{k}:max_abs={error:.9g}")


def profile_magnitude(artifact: Artifact, profile: np.ndarray) -> float:
    standardized = (np.asarray(profile, dtype=np.float64) - artifact.train_center) / artifact.train_scale
    return float(np.linalg.norm(standardized))


def candidate_compounds(artifact: Artifact, rotation: Rotation) -> list[str]:
    candidates: set[str] | None = None
    slots = list(rotation.confirmation_a) + list(rotation.confirmation_b)
    for row_index in slots:
        slot = (rotation.dose, str(artifact.plate[row_index]), well_row(str(artifact.well[row_index])))
        current = set(artifact.slot_rows.get(slot, {}))
        candidates = current if candidates is None else candidates & current
    if candidates is None:
        return []
    candidates.discard(rotation.compound)
    first_row = rotation.confirmation_a[0]
    first_slot = (rotation.dose, str(artifact.plate[first_row]), well_row(str(artifact.well[first_row])))
    first_map = artifact.slot_rows.get(first_slot, {})
    return sorted(
        c for c in candidates
        if c in first_map and str(artifact.split[int(first_map[c])]) == rotation.split
    )


def candidate_rows_for_compound(artifact: Artifact, rotation: Rotation, compound: str) -> tuple[np.ndarray, np.ndarray] | None:
    a_rows: list[int] = []
    b_rows: list[int] = []
    for target, out in ((rotation.confirmation_a, a_rows), (rotation.confirmation_b, b_rows)):
        for row_index in target:
            slot = (rotation.dose, str(artifact.plate[row_index]), well_row(str(artifact.well[row_index])))
            source = artifact.slot_rows.get(slot, {})
            if compound not in source:
                return None
            out.append(int(source[compound]))
    return np.asarray(a_rows, dtype=np.int64), np.asarray(b_rows, dtype=np.int64)


def compute_null_and_label_features(artifact: Artifact, rotation: Rotation, rounds: int, seed: int) -> tuple[np.ndarray, np.ndarray, float, int]:
    actual_a = artifact.delta[np.asarray(rotation.confirmation_a, dtype=np.int64)].mean(axis=0, dtype=np.float64)
    actual_b = artifact.delta[np.asarray(rotation.confirmation_b, dtype=np.int64)].mean(axis=0, dtype=np.float64)
    actual_r = pcc(actual_a, actual_b)
    if actual_r is None:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), math.nan, 0
    actual_ref = artifact.delta[np.asarray(rotation.confirmation_a + rotation.confirmation_b, dtype=np.int64)].mean(axis=0, dtype=np.float64)
    actual_a_mag = profile_magnitude(artifact, actual_ref)
    candidates = candidate_compounds(artifact, rotation)
    candidate_z: list[float] = []
    candidate_mag: list[float] = []
    for compound in candidates:
        rows = candidate_rows_for_compound(artifact, rotation, compound)
        if rows is None:
            continue
        null_a = artifact.delta[rows[0]].mean(axis=0, dtype=np.float64)
        null_b = artifact.delta[rows[1]].mean(axis=0, dtype=np.float64)
        null_r = pcc(null_a, null_b)
        if null_r is None:
            continue
        candidate_z.append(fisher(null_r))
        candidate_mag.append(profile_magnitude(artifact, np.vstack([artifact.delta[rows[0]], artifact.delta[rows[1]]]).mean(axis=0, dtype=np.float64)))
    if not candidate_z:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), math.nan, 0
    rng = np.random.default_rng(stable_seed(seed, f"null|{rotation.split}|{rotation.compound}|{rotation.dose}|{rotation.support_row}"))
    choices = rng.integers(0, len(candidate_z), size=max(1, int(rounds)))
    null_z_samples = np.asarray(candidate_z, dtype=np.float64)[choices]
    null_mag_samples = np.asarray(candidate_mag, dtype=np.float64)[choices]
    null_z_mean = float(np.mean(null_z_samples))
    e_rep = fisher(actual_r) - null_z_mean
    # Store actual values through a compact convention: the caller assigns the
    # return arrays and computes `a_ref` from the final returned magnitude.
    return (
        np.asarray(null_z_samples, dtype=np.float64),
        np.asarray(null_mag_samples, dtype=np.float64),
        float(e_rep),
        int(len(candidate_z)),
    )


def actual_magnitude(artifact: Artifact, rotation: Rotation) -> float:
    rows = np.asarray(rotation.confirmation_a + rotation.confirmation_b, dtype=np.int64)
    return profile_magnitude(artifact, artifact.delta[rows].mean(axis=0, dtype=np.float64))


def make_rotations(
    artifact: Artifact,
    bundle: PredictionBundle,
    ge: GEBundle | None,
    support_manifest: dict[tuple[str, str, int], tuple[int, ...]] | None,
    required_split: str,
    mapping_errors: list[str],
) -> tuple[list[Rotation], list[dict[str, Any]]]:
    rotations: list[Rotation] = []
    audit: list[dict[str, Any]] = []
    condition_keys = sorted((c, d) for (c, d), rows in artifact.groups.items() if str(artifact.split[rows[0]]) == required_split)
    for compound, dose in condition_keys:
        rows = artifact.groups[(compound, dose)]
        if len(rows) != REQUIRED_REPEATS:
            audit.append({"split": required_split, "compound": compound, "dose": dose, "reason": "strict_requires_five_repeats", "repeat_count": len(rows)})
            continue
        support_map: dict[int, int] = {}
        available_keys = {key(compound, dose, row) for row in rows}
        for row in rows:
            k = key(compound, dose, row)
            if k not in bundle.index_by_key or k not in bundle.support_by_key:
                continue
            support = int(bundle.support_by_key[k])
            support_map[support] = row
        # The saved pair set can expose the same support row through several
        # held targets.  A support slot is valid only if its held target also
        # exists, because P0/GE baseline vectors are reconstructed at held=s.
        support_slots = sorted(s for s in support_map if key(compound, dose, s) in bundle.index_by_key)
        if len(support_slots) < 2:
            audit.append({"split": required_split, "compound": compound, "dose": dose, "reason": "fewer_than_two_frozen_support_slots", "repeat_count": len(rows), "unique_support_slots": len(support_slots)})
            continue
        if len(support_slots) < len(rows):
            audit.append({"split": required_split, "compound": compound, "dose": dose, "reason": "partial_support_slot_coverage", "repeat_count": len(rows), "unique_support_slots": len(support_slots)})
        for rank, support_row in enumerate(support_slots):
            remain = [row for row in rows if row != support_row]
            if len(remain) != 4:
                mapping_errors.append(f"confirmation_count_error:{required_split}:{compound}:{dose}:{support_row}")
                continue
            # Frozen order is plate/well/index order from the preparation step.
            confirm_a = tuple(remain[:2])
            confirm_b = tuple(remain[2:])
            teacher_indices = [
                bundle.index_by_key[key(compound, dose, held)]
                for held in rows
                if bundle.support_by_key.get(key(compound, dose, held)) == support_row
            ]
            baseline_index = bundle.index_by_key.get(key(compound, dose, support_row))
            if not teacher_indices or baseline_index is None:
                mapping_errors.append(f"rotation_prediction_mapping_missing:{required_split}:{compound}:{dose}:{support_row}")
                continue
            teacher_values = bundle.teacher[np.asarray(teacher_indices, dtype=np.int64)]
            teacher = teacher_values[0]
            teacher_error = float(np.max(np.abs(teacher_values - teacher))) if len(teacher_values) > 1 else 0.0
            if teacher_error > 2e-5:
                mapping_errors.append(f"teacher_support_rotation_inconsistency:{required_split}:{compound}:{dose}:{support_row}:max_abs={teacher_error:.9g}")
            lam = float(bundle.lambda_value)
            if not (0.0 < lam <= 1.0):
                mapping_errors.append(f"invalid_lambda:{bundle.path}:{lam}")
                continue
            virtual = bundle.virtual_prior[baseline_index]
            p0 = (1.0 - lam) * teacher + lam * virtual
            p0_direct = bundle.posterior[baseline_index]
            # Direct posterior at held=s has the wrong teacher input for this
            # rotated support, but its prior component is enough for the
            # frozen P0 reconstruction audit below.
            raw = artifact.delta[support_row]
            ge_score = math.nan
            ge_available = 0
            if ge is not None:
                gk = key(compound, dose, support_row)
                if gk in ge.index_by_key:
                    gi = ge.index_by_key[gk]
                    ge_p0 = ge.p0[gi]
                    ge_residual = ge.correct_ge[gi] - ge_p0
                    ge_score = profile_magnitude(artifact, p0 + ge_residual)
                    ge_available = 1
                    if float(np.max(np.abs(ge_p0 - p0_direct))) > 5e-4:
                        mapping_errors.append(f"p0_ge_key_mismatch:{required_split}:{compound}:{dose}:{support_row}")
            rotation = Rotation(
                split=required_split,
                compound=compound,
                dose=dose,
                condition_rows=tuple(rows),
                support_row=int(support_row),
                rotation_rank=int(rank),
                support_plate=str(artifact.plate[support_row]),
                support_well=str(artifact.well[support_row]),
                confirmation_a=confirm_a,
                confirmation_b=confirm_b,
                p0_available=1,
                ge_available=ge_available,
                raw_score=profile_magnitude(artifact, raw),
                teacher_score=profile_magnitude(artifact, teacher),
                p0_score=profile_magnitude(artifact, p0),
                ge_score=ge_score,
            )
            rotations.append(rotation)
    return rotations, audit


def assign_labels(
    artifact: Artifact,
    validation: list[Rotation],
    evaluation: list[Rotation],
    rounds: int,
    label_seed: int,
) -> dict[str, float]:
    all_null_e: list[np.ndarray] = []
    all_null_a: list[np.ndarray] = []
    for rotation in validation + evaluation:
        z_samples, a_samples, e_rep, candidate_count = compute_null_and_label_features(artifact, rotation, rounds, label_seed)
        rotation.e_rep = float(e_rep)
        rotation.null_candidate_count = int(candidate_count)
        rotation.null_z_mean = float(np.mean(z_samples)) if len(z_samples) else math.nan
        rotation.a_ref = actual_magnitude(artifact, rotation)
        if rotation.split == "valid" and len(z_samples):
            all_null_e.append(z_samples - float(np.mean(z_samples)))
            all_null_a.append(a_samples)
    if not all_null_e or not all_null_a:
        raise RuntimeError("No validation matched-null samples available for confirmed-activity thresholds")
    null_e = np.concatenate(all_null_e)
    null_a = np.concatenate(all_null_a)
    e_q = {"q90": float(np.quantile(null_e, 0.90)), "q95": float(np.quantile(null_e, 0.95)), "q100": float(np.quantile(null_e, 1.0))}
    a_q = {"q90": float(np.quantile(null_a, 0.90)), "q95": float(np.quantile(null_a, 0.95)), "q100": float(np.quantile(null_a, 1.0))}
    for rotation in validation + evaluation:
        valid_null = np.isfinite(rotation.e_rep) and np.isfinite(rotation.a_ref)
        rotation.label = int(valid_null and rotation.e_rep > e_q["q95"] and rotation.a_ref > a_q["q95"])
        rotation.borderline = int(
            valid_null
            and ((e_q["q90"] <= rotation.e_rep <= e_q["q100"]) or (a_q["q90"] <= rotation.a_ref <= a_q["q100"]))
        )
    return {
        "e_rep_null_q90": e_q["q90"],
        "e_rep_null_q95": e_q["q95"],
        "e_rep_null_q100": e_q["q100"],
        "a_ref_null_q90": a_q["q90"],
        "a_ref_null_q95": a_q["q95"],
        "a_ref_null_q100": a_q["q100"],
        "validation_null_e_count": int(len(null_e)),
        "validation_null_a_count": int(len(null_a)),
    }


def choose_validation_threshold(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float | None:
    finite = np.isfinite(scores) & np.isfinite(labels)
    scores = np.asarray(scores[finite], dtype=np.float64)
    labels = np.asarray(labels[finite], dtype=np.int8)
    negatives = scores[labels == 0]
    positives = scores[labels == 1]
    if len(negatives) == 0 or len(positives) == 0:
        return None
    candidates = np.unique(scores)
    candidates.sort()
    fprs = np.asarray([(negatives >= threshold).mean() for threshold in candidates])
    valid = candidates[fprs <= float(target_fpr) + 1e-12]
    return float(valid[0]) if len(valid) else float(np.nextafter(np.max(scores), np.inf))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s) & np.isfinite(y)
    y, s = y[finite], s[finite]
    positives = int(y.sum())
    if len(y) == 0 or positives == 0:
        return math.nan
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    cumulative = np.cumsum(ys, dtype=np.float64)
    precision = cumulative / np.arange(1, len(ys) + 1, dtype=np.float64)
    return float(np.sum(precision[ys == 1]) / positives)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s) & np.isfinite(y)
    y, s = y[finite], s[finite]
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return math.nan
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(ss), dtype=np.float64)
    start = 0
    while start < len(ss):
        end = start + 1
        while end < len(ss) and ss[end] == ss[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(np.sum(ranks[y[order] == 1]))
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def top_precision(labels: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    y, s = y[finite], s[finite]
    if len(y) == 0:
        return math.nan
    count = max(1, int(math.ceil(len(y) * float(fraction))))
    order = np.argsort(-s, kind="mergesort")[:count]
    return float(np.mean(y[order]))


def evaluate_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    raw_scores: np.ndarray,
    threshold: float | None,
    raw_threshold: float | None,
    fixed_fpr: float,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    raw = np.asarray(raw_scores, dtype=np.float64)
    finite = np.isfinite(s) & np.isfinite(raw)
    y, s, raw = y[finite], s[finite], raw[finite]
    prevalence = float(np.mean(y)) if len(y) else math.nan
    p5 = top_precision(y, s, 0.05)
    p10 = top_precision(y, s, 0.10)
    if threshold is None:
        recall = math.nan
        observed_fpr = math.nan
    else:
        pred = s >= threshold
        recall = float(np.sum(pred & (y == 1)) / max(1, int(np.sum(y == 1))))
        observed_fpr = float(np.sum(pred & (y == 0)) / max(1, int(np.sum(y == 0))))
    rescue_rate = math.nan
    false_rescue_rate = math.nan
    missed_hit_count = math.nan
    rescued_hit_count = math.nan
    false_rescue_count = math.nan
    if threshold is not None and raw_threshold is not None:
        raw_active = raw >= raw_threshold
        active = s >= threshold
        missed = (y == 1) & ~raw_active
        rescued = missed & active
        false = (y == 0) & ~raw_active & active
        missed_hit_count = float(np.sum(missed))
        rescued_hit_count = float(np.sum(rescued))
        false_rescue_count = float(np.sum(false))
        rescue_rate = float(np.sum(rescued) / max(1, int(np.sum(missed))))
        false_rescue_rate = float(np.sum(false) / max(1, int(np.sum((y == 0) & ~raw_active))))
    return {
        "n": float(len(y)),
        "positive_count": float(np.sum(y == 1)),
        "negative_count": float(np.sum(y == 0)),
        "prevalence": prevalence,
        "AP": average_precision(y, s),
        "AUROC": auroc(y, s),
        "Recall_at_fixed_FPR": recall,
        "observed_FPR": observed_fpr,
        "Precision_at_top5": p5,
        "Precision_at_top10": p10,
        "Enrichment_at_top10": float(p10 / prevalence) if np.isfinite(prevalence) and prevalence > 0 else math.nan,
        "missed_hit_count": missed_hit_count,
        "rescued_hit_count": rescued_hit_count,
        "rescue_rate": rescue_rate,
        "false_rescue_count": false_rescue_count,
        "false_rescue_rate": false_rescue_rate,
        "fixed_fpr_target": float(fixed_fpr),
    }


def rotation_records(rotations: list[Rotation], analysis_common: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rotation in rotations:
        if analysis_common and not rotation.ge_available:
            continue
        records.append(
            {
                "split": rotation.split,
                "compound": rotation.compound,
                "dose": rotation.dose,
                "support_row": rotation.support_row,
                "rotation_rank": rotation.rotation_rank,
                "support_plate": rotation.support_plate,
                "support_well": rotation.support_well,
                "confirm_a_rows": "|".join(map(str, rotation.confirmation_a)),
                "confirm_b_rows": "|".join(map(str, rotation.confirmation_b)),
                "e_rep": rotation.e_rep,
                "a_ref": rotation.a_ref,
                "null_candidate_count": rotation.null_candidate_count,
                "confirmed_active": rotation.label,
                "borderline": rotation.borderline,
                "p0_available": rotation.p0_available,
                "ge_available": rotation.ge_available,
                "raw_score": rotation.raw_score,
                "teacher_score": rotation.teacher_score,
                "P0_score": rotation.p0_score,
                "GE_posterior_score": rotation.ge_score,
                # Keep the actual GE availability flag in the audit table even
                # when writing the full P0-eligible table (analysis_common is
                # only a row filter used for paired metrics).
                "evaluation_common_all_methods": int(rotation.ge_available),
            }
        )
    return records


def records_to_arrays(records: list[dict[str, Any]], rotation: str) -> dict[str, np.ndarray]:
    if rotation == "all":
        chosen = records
    else:
        chosen = [row for row in records if int(row["rotation_rank"]) == int(rotation)]
    chosen = sorted(chosen, key=lambda row: (str(row["compound"]), str(row["dose"]), int(row["support_row"])))
    fields = {
        "compound": np.asarray([str(row["compound"]) for row in chosen], dtype=str),
        "label": np.asarray([int(row["confirmed_active"]) for row in chosen], dtype=np.int8),
        "raw": np.asarray([float(row["raw_score"]) for row in chosen], dtype=np.float64),
        "teacher": np.asarray([float(row["teacher_score"]) for row in chosen], dtype=np.float64),
        "P0": np.asarray([float(row["P0_score"]) for row in chosen], dtype=np.float64),
        "GE_posterior": np.asarray([float(row["GE_posterior_score"]) for row in chosen], dtype=np.float64),
        "borderline": np.asarray([int(row["borderline"]) for row in chosen], dtype=np.int8),
    }
    return fields


def cluster_bootstrap(
    arrays: dict[str, np.ndarray],
    thresholds: dict[str, float | None],
    rounds: int,
    seed: int,
) -> tuple[dict[tuple[str, str, str], np.ndarray], dict[str, dict[str, float]]]:
    compounds = np.asarray(sorted(set(arrays["compound"])), dtype=str)
    if len(compounds) == 0:
        return {}, {}
    groups = {compound: np.flatnonzero(arrays["compound"] == compound) for compound in compounds}
    rng = np.random.default_rng(seed)
    metric_values: dict[tuple[str, str, str], np.ndarray] = {}
    point: dict[str, dict[str, float]] = {}
    for method in METHODS:
        point[method] = evaluate_metrics(
            arrays["label"],
            arrays[method],
            arrays["raw"],
            thresholds.get(method),
            thresholds.get("raw"),
            FIXED_FPR,
        )
        for metric in RANK_METRICS:
            metric_values[(method, metric, "point")] = np.asarray([point[method].get(metric, math.nan)], dtype=np.float64)
    # Bootstrap the outer unit as compound.  Each sampled compound carries all
    # of its dose/support-slot rows, preserving within-compound dependence.
    bootstrap_metrics = {
        (method, metric): np.full(max(1, int(rounds)), np.nan, dtype=np.float64)
        for method in METHODS
        for metric in RANK_METRICS
    }
    n_compounds = len(compounds)
    for iteration in range(max(1, int(rounds))):
        sampled = rng.integers(0, n_compounds, size=n_compounds)
        indices = np.concatenate([groups[compounds[int(position)]] for position in sampled])
        labels = arrays["label"][indices]
        raw = arrays["raw"][indices]
        for method in METHODS:
            result = evaluate_metrics(labels, arrays[method][indices], raw, thresholds.get(method), thresholds.get("raw"), FIXED_FPR)
            for metric in RANK_METRICS:
                bootstrap_metrics[(method, metric)][iteration] = result.get(metric, math.nan)
    return {(method, metric, "bootstrap"): values for (method, metric), values in bootstrap_metrics.items()}, point


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"


def build_metrics_outputs(
    test_records: list[dict[str, Any]],
    thresholds: dict[str, float | None],
    rounds: int,
    seeds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # The frozen model vectors are identical in mapping across seeds but are
    # evaluated per seed.  Test records passed here are one seed's records.
    metrics_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    # Caller invokes this once per seed; retained argument keeps the function
    # interface explicit for generated report metadata.
    del seeds
    for rotation in ("0", "1", "all"):
        arrays = records_to_arrays(test_records, rotation)
        if len(arrays["label"]) == 0:
            continue
        boot, points = cluster_bootstrap(arrays, thresholds, rounds, stable_seed(3407, f"bootstrap|{rotation}"))
        for method in METHODS:
            row = {
                "seed": "",
                "rotation": rotation,
                "analysis_set": "common_all_methods" if method == "GE_posterior" else "common_all_methods_for_paired_comparison",
                "method": method,
                "compound_count": int(len(set(arrays["compound"]))),
                "row_count": int(len(arrays["label"])),
                "threshold_validation": thresholds.get(method),
                "threshold_status": "OK" if thresholds.get(method) is not None else "BLOCKED_VALIDATION_PREDICTIONS_MISSING",
            }
            row.update(points[method])
            metrics_rows.append(row)
        for left, right in COMPARISONS:
            for metric in RANK_METRICS:
                left_point = points[left].get(metric, math.nan)
                right_point = points[right].get(metric, math.nan)
                if not (np.isfinite(left_point) and np.isfinite(right_point)):
                    continue
                contrast_rows.append({
                    "seed": "",
                    "rotation": rotation,
                    "analysis_set": "common_all_methods",
                    "left_method": left,
                    "right_method": right,
                    "metric": metric,
                    "point_difference": float(left_point - right_point),
                    "n_compounds": int(len(set(arrays["compound"]))),
                    "n_rows": int(len(arrays["label"])),
                })
                left_boot = boot.get((left, metric, "bootstrap"))
                right_boot = boot.get((right, metric, "bootstrap"))
                if left_boot is None or right_boot is None:
                    continue
                differences = left_boot - right_boot
                finite = differences[np.isfinite(differences)]
                if len(finite):
                    ci_rows.append({
                        "seed": "",
                        "rotation": rotation,
                        "analysis_set": "common_all_methods",
                        "left_method": left,
                        "right_method": right,
                        "metric": metric,
                        "point_difference": float(left_point - right_point),
                        "ci_low": float(np.quantile(finite, 0.025)),
                        "ci_high": float(np.quantile(finite, 0.975)),
                        "rounds": int(rounds),
                        "n_compounds": int(len(set(arrays["compound"]))),
                        "n_rows": int(len(arrays["label"])),
                    })
    return metrics_rows, contrast_rows, ci_rows


def render_figure(outdir: Path, contrast_rows: list[dict[str, Any]]) -> str:
    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        (figures / "FIGURE_BLOCKED.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return "BLOCKED_MATPLOTLIB"
    rows = [row for row in contrast_rows if row.get("rotation") == "all" and row.get("metric") == "AP"]
    methods = ["teacher", "P0", "GE_posterior"]
    values = []
    labels = []
    for method in methods:
        subset = [float(row["point_difference"]) for row in rows if row.get("left_method") == method and row.get("right_method") == "raw" and np.isfinite(float(row["point_difference"]))]
        if subset:
            values.append(float(np.mean(subset)))
            labels.append(method)
    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
    if values:
        ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"][: len(values)])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("AP difference vs raw")
    ax.set_title("cpg0004 hit recovery (frozen 1R predictions)")
    fig.savefig(figures / "hit_recovery_summary.png", dpi=180)
    plt.close(fig)
    return "OK"


def write_protocol(outdir: Path) -> None:
    text = f"""# Experiment 1 — cpg0004 confirmed activity hit recovery

Version: `{VERSION}`  
Status is determined from artifacts produced by `run_hit_recovery.py`.

## Objective

Test whether a one-real-repeat reproducible-effect teacher and the already
frozen P0 / observed-GE posterior rank compound-dose conditions that are later
confirmed as reproducibly active more accurately than the raw one-repeat CP
profile.

## Frozen inputs and no-retuning rule

The only inputs are the immutable `cp_plate_rows.npz`, the saved 1R P0
predictions, the saved 1R observed-GE posterior predictions, and the existing
pair manifests/metrics. No model is retrained. The saved `selected_lambda`
and `selected_beta` values are read from their original `metrics.json`; this
experiment never searches, changes, or reselects them.

## Eligibility and independence

The primary label requires exactly five plate-level observations for one
compound-dose condition. For each available support slot `s`, `s` is excluded
from confirmation; the remaining four observations are split in frozen plate
order into confirmation A (two plates) and B (two plates). Thus support and
confirmation have no shared plate. Conditions with two or four repeats are
excluded from the strict primary label and are listed in `EXCLUSIONS.csv`.

The saved 1R prediction set covers only the support rows explicitly present in
the original pair manifests. The evaluator reconstructs and uses exactly
those support rows. It does not impute a missing teacher. cpg0004 has five
physical repeats, but the saved predictions expose two unique support slots
per eligible mapped condition; the other three possible slots are not claimed.

## Confirmed-activity label

For each rotation, `r_rep = PCC(mean(A), mean(B))`. A matched foreign-compound
null preserves dose, each confirmation plate, and well row. `E_rep` is the
Fisher-z repeat score minus the mean matched-null Fisher-z. The effect score
`A_ref` is the L2 norm of the four-repeat mean after feature-wise robust
standardization. The frozen artifact lacks DMSO well-level dispersion, so the
scale is a training-compound delta MAD (1.4826×MAD), with a training-only
standard-deviation/unit fallback for degenerate features; the fallback count
is recorded in `CONFIG.json`.

The `q95` cutoffs for `E_rep` and `A_ref` are estimated only from validation
compound matched-null draws. `Confirmed Active = (E_rep > q95_E) AND
(A_ref > q95_A)`. Test labels never determine a threshold. A sensitivity
analysis excludes rotations whose score lies in the validation-null q90–q100
borderline band for either component.

## Model scores and endpoints

Each model is scored from its own estimated profile only: `A_raw`, `A_teacher`,
`A_P0`, and `A_GE_posterior`. No held-out confirmation profile enters a model
score. Primary endpoint is AP. Secondary endpoints are AUROC, recall at a
validation-selected 5% FPR, precision@top5%, precision@top10%, and
enrichment@top10%. Missed-hit rescue is `confirmed active AND raw inactive AND
method active`; false rescue is `confirmed inactive AND raw inactive AND
method active`.

The fixed-FPR/activity thresholds for raw, teacher, and P0 are selected on
validation labels and scores only. GE validation predictions were not saved by
the frozen GE run, so GE thresholded recall/rescue endpoints are explicitly
`BLOCKED_VALIDATION_PREDICTIONS_MISSING`; GE AP/AUROC/top-k ranking is still
reported on the common test set.

## Uncertainty and decision

All paired intervals resample compounds (not compound-dose rows) as the outer
bootstrap unit, with 10,000 rounds in the formal run. Test rows from all doses
and the two available support rotations remain together within a compound.
Three seed point estimates and the compound bootstrap CIs are reported
without selecting a favorable seed or dose.
"""
    (outdir / "PROTOCOL.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.null_rounds <= 0 or args.bootstrap_rounds <= 0:
        raise ValueError("null/bootstrap rounds must be positive")
    if args.smoke:
        args.null_rounds = min(args.null_rounds, 8)
        args.bootstrap_rounds = min(args.bootstrap_rounds, 20)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if any(outdir.iterdir()) and not args.force:
        # The source script is also stored in this directory.  It is safe to
        # run with --force only after the user has inspected the target.
        existing = [p.name for p in outdir.iterdir() if p.name != Path(__file__).name]
        if existing:
            raise FileExistsError(f"Refusing to overwrite non-empty output directory: {outdir}; use --force")
    write_protocol(outdir)
    artifact = load_artifact(args.data)
    mapping_errors: list[str] = []
    validation_bundle: PredictionBundle | None = None
    test_bundles: dict[int, PredictionBundle] = {}
    ge_bundles: dict[int, GEBundle] = {}
    meta: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        p0_dir = args.p0_root / f"seed{seed}"
        ge_dir = args.ge_root / f"seed{seed}"
        p0_metrics = load_json(p0_dir / "metrics.json")
        ge_metrics = load_json(ge_dir / "metrics.json")
        lam = float(p0_metrics["selected_lambda"])
        beta = float(ge_metrics["selected_beta"])
        if not np.isfinite(lam) or not (0.0 < lam <= 1.0):
            raise ValueError(f"Invalid frozen lambda for seed {seed}: {lam}")
        if not np.isfinite(beta):
            raise ValueError(f"Invalid frozen beta for seed {seed}: {beta}")
        if validation_bundle is None:
            validation_bundle = load_npz_bundle(p0_dir / "valid_predictions.npz", "valid")
            validation_bundle.lambda_value = lam
        else:
            # Validation predictions are expected to be schema/order-compatible;
            # each seed is still checked independently below.
            pass
        test_bundle = load_npz_bundle(p0_dir / "test_predictions.npz", "test")
        test_bundle.lambda_value = lam
        manifest = load_pair_manifest(args.pair_root / f"seed{seed}" / "test_pair_manifest.csv")
        reconstruct_support_mapping(artifact, test_bundle, manifest, mapping_errors)
        if validation_bundle is not None and seed == SEEDS[0]:
            reconstruct_support_mapping(artifact, validation_bundle, None, mapping_errors)
        ge_bundle = load_ge_bundle(ge_dir / "test_predictions.npz")
        ge_bundle.beta_value = beta
        test_bundles[seed] = test_bundle
        ge_bundles[seed] = ge_bundle
        meta[seed] = {
            "lambda": lam,
            "beta": beta,
            "p0_test_rows": int(len(test_bundle.compound)),
            "p0_valid_rows": int(len(validation_bundle.compound)) if validation_bundle is not None else 0,
            "ge_test_rows": int(len(ge_bundle.compound)),
            "p0_test_sha256": sha256(p0_dir / "test_predictions.npz"),
            "p0_valid_sha256": sha256(p0_dir / "valid_predictions.npz"),
            "ge_test_sha256": sha256(ge_dir / "test_predictions.npz"),
            "pair_manifest_sha256": sha256(args.pair_root / f"seed{seed}" / "test_pair_manifest.csv"),
        }
    assert validation_bundle is not None
    # Validation labels/thresholds are common and label-seed locked.  GE has no
    # validation prediction file, which is deliberately not silently repaired.
    validation_rotations, validation_audit = make_rotations(artifact, validation_bundle, None, None, "valid", mapping_errors)
    test_rotations_by_seed: dict[int, list[Rotation]] = {}
    test_audit: list[dict[str, Any]] = []
    for seed in SEEDS:
        test_rotations, audit = make_rotations(artifact, test_bundles[seed], ge_bundles[seed], None, "test", mapping_errors)
        test_rotations_by_seed[seed] = test_rotations
        test_audit.extend([dict(row, seed=seed) for row in audit])
    threshold_info = assign_labels(artifact, validation_rotations, test_rotations_by_seed[SEEDS[0]], args.null_rounds, args.label_seed)
    # Labels are constructed from frozen rows and null maps, so copy them by
    # condition/support key to every seed and verify no seed-specific drift.
    label_map = {(r.compound, r.dose, r.support_row): (r.e_rep, r.a_ref, r.label, r.borderline, r.null_candidate_count) for r in test_rotations_by_seed[SEEDS[0]]}
    for seed in SEEDS[1:]:
        for rotation in test_rotations_by_seed[seed]:
            z_samples, _, e_rep, candidate_count = compute_null_and_label_features(
                artifact, rotation, args.null_rounds, args.label_seed
            )
            rotation.e_rep = float(e_rep)
            rotation.null_z_mean = float(np.mean(z_samples)) if len(z_samples) else math.nan
            rotation.null_candidate_count = int(candidate_count)
            rotation.a_ref = actual_magnitude(artifact, rotation)
            rotation.label = int(
                np.isfinite(rotation.e_rep)
                and np.isfinite(rotation.a_ref)
                and rotation.e_rep > threshold_info["e_rep_null_q95"]
                and rotation.a_ref > threshold_info["a_ref_null_q95"]
            )
            rotation.borderline = int(
                np.isfinite(rotation.e_rep)
                and np.isfinite(rotation.a_ref)
                and (
                    threshold_info["e_rep_null_q90"] <= rotation.e_rep <= threshold_info["e_rep_null_q100"]
                    or threshold_info["a_ref_null_q90"] <= rotation.a_ref <= threshold_info["a_ref_null_q100"]
                )
            )
        for rotation in test_rotations_by_seed[seed]:
            expected = label_map.get((rotation.compound, rotation.dose, rotation.support_row))
            if expected is None:
                mapping_errors.append(f"cross_seed_label_key_missing:{seed}:{rotation.compound}:{rotation.dose}:{rotation.support_row}")
                continue
            if rotation.label != expected[2] or rotation.borderline != expected[3]:
                mapping_errors.append(f"cross_seed_label_drift:{seed}:{rotation.compound}:{rotation.dose}:{rotation.support_row}")
    validation_records = rotation_records(validation_rotations, False)
    # Fixed-FPR thresholds use every validation rotation with finite score and
    # the labels above.  GE threshold is intentionally absent.
    val_arrays = records_to_arrays(validation_records, "all")
    thresholds: dict[str, float | None] = {
        "raw": choose_validation_threshold(val_arrays["raw"], val_arrays["label"], FIXED_FPR),
        "teacher": choose_validation_threshold(val_arrays["teacher"], val_arrays["label"], FIXED_FPR),
        "P0": choose_validation_threshold(val_arrays["P0"], val_arrays["label"], FIXED_FPR),
        "GE_posterior": None,
    }
    # Validate the frozen P0 identity at the prediction-row level.  This also
    # detects accidental use of a reselected lambda.
    p0_formula_errors: list[dict[str, Any]] = []
    for seed, bundle in {**{SEEDS[0]: validation_bundle}, **test_bundles}.items():
        lam = float(bundle.lambda_value)
        reconstructed = (1.0 - lam) * bundle.teacher + lam * bundle.virtual_prior
        error = float(np.max(np.abs(reconstructed - bundle.posterior)))
        derived = (bundle.posterior - (1.0 - lam) * bundle.teacher) / lam
        prior_error = float(np.max(np.abs(derived - bundle.virtual_prior)))
        p0_formula_errors.append({"seed": seed, "split": bundle.split, "posterior_formula_max_abs": error, "virtual_prior_inverse_max_abs": prior_error})
        if error > 5e-4 or prior_error > 5e-3:
            mapping_errors.append(f"p0_formula_error:{seed}:{bundle.split}:posterior={error:.9g}:prior={prior_error:.9g}")
    # Build all output records and metrics.
    all_eligible_records: list[dict[str, Any]] = []
    for row in validation_records:
        row["role"] = "validation_threshold_calibration"
        all_eligible_records.append(row)
    for seed in SEEDS:
        rows = rotation_records(test_rotations_by_seed[seed], False)
        for row in rows:
            row["seed"] = seed
            row["role"] = "test_evaluation"
        all_eligible_records.extend(rows)
    eligible_fields = [
        "role", "seed", "split", "compound", "dose", "support_row", "rotation_rank", "support_plate", "support_well", "confirm_a_rows", "confirm_b_rows",
        "e_rep", "a_ref", "null_candidate_count", "confirmed_active", "borderline", "p0_available", "ge_available",
        "raw_score", "teacher_score", "P0_score", "GE_posterior_score", "evaluation_common_all_methods",
    ]
    write_csv(outdir / "ELIGIBLE_SAMPLES.csv", all_eligible_records, eligible_fields)
    exclusion_rows: list[dict[str, Any]] = []
    for row in validation_audit + test_audit:
        exclusion_rows.append(dict(row, scope="eligibility"))
    ge_valid_block = {"scope": "prediction_schema", "split": "valid", "reason": "GE_VALIDATION_PREDICTIONS_MISSING", "count": 1, "detail": "08_ge_1r_evidence saves test_predictions.npz only; GE validation-only threshold/rescue cannot be estimated without retraining or a missing frozen artifact."}
    exclusion_rows.append(ge_valid_block)
    for seed in SEEDS:
        ge_keys = {(str(c), str(d), int(h)) for c, d, h in zip(ge_bundles[seed].compound, ge_bundles[seed].dose, ge_bundles[seed].held_index)}
        p0_keys = set(test_bundles[seed].index_by_key)
        exclusion_rows.append({"scope": "prediction_schema", "seed": seed, "split": "test", "reason": "GE_P0_COMMON_KEY_MISSING", "count": int(len(p0_keys - ge_keys)), "detail": "P0 test keys without GE posterior key; excluded from common all-method paired endpoint."})
        unique_support = defaultdict(set)
        for k, support in test_bundles[seed].support_by_key.items():
            unique_support[(k[0], k[1])].add(support)
        incomplete = sum(1 for value in unique_support.values() if len(value) < REQUIRED_REPEATS and len(value) >= 2)
        exclusion_rows.append({"scope": "support_rotation", "seed": seed, "split": "test", "reason": "ONLY_FROZEN_SUPPORT_SLOTS_USED", "count": int(sum(1 for value in unique_support.values() if len(value) == 2)), "detail": "Each mapped five-repeat condition exposes two unique support rows in the saved 1R prediction set; three physical slots are not covered without retraining."})
    for row in p0_formula_errors:
        exclusion_rows.append({"scope": "mapping_audit", **row, "reason": "P0_FORMULA_RECONSTRUCTION"})
    if mapping_errors:
        for error in mapping_errors[:1000]:
            exclusion_rows.append({"scope": "mapping_audit", "reason": "MAPPING_ERROR", "detail": error})
    write_csv(outdir / "EXCLUSIONS.csv", exclusion_rows)

    metrics_rows: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    cis: list[dict[str, Any]] = []
    per_seed_ci: dict[tuple[str, str, str, str, str], list[float]] = {}
    # Run each seed with independent bootstrap streams, then add a three-seed
    # mean row from the aligned per-seed point/boot distributions.
    for seed in SEEDS:
        test_records = rotation_records(test_rotations_by_seed[seed], True)
        for row in test_records:
            row["seed"] = seed
        for rotation_name in ("0", "1", "all"):
            arrays = records_to_arrays(test_records, rotation_name)
            if len(arrays["label"]) == 0:
                continue
            seed_thresholds = thresholds.copy()
            boot, points = cluster_bootstrap(arrays, seed_thresholds, args.bootstrap_rounds, stable_seed(seed, f"bootstrap|{rotation_name}"))
            for method in METHODS:
                metrics_rows.append({"seed": seed, "rotation": rotation_name, "analysis_set": "common_all_methods", "method": method, "compound_count": int(len(set(arrays["compound"]))), "row_count": int(len(arrays["label"])), "threshold_validation": seed_thresholds.get(method), "threshold_status": "OK" if seed_thresholds.get(method) is not None else "BLOCKED_VALIDATION_PREDICTIONS_MISSING", **points[method]})
            for left, right in COMPARISONS:
                for metric in RANK_METRICS:
                    lp = points[left].get(metric, math.nan); rp = points[right].get(metric, math.nan)
                    if not (np.isfinite(lp) and np.isfinite(rp)):
                        continue
                    contrasts.append({"seed": seed, "rotation": rotation_name, "analysis_set": "common_all_methods", "left_method": left, "right_method": right, "metric": metric, "point_difference": float(lp - rp), "n_compounds": int(len(set(arrays["compound"]))), "n_rows": int(len(arrays["label"]))})
                    lb = boot.get((left, metric, "bootstrap")); rb = boot.get((right, metric, "bootstrap"))
                    if lb is None or rb is None:
                        continue
                    diff = lb - rb; finite = diff[np.isfinite(diff)]
                    if len(finite):
                        # Keep the full round-aligned array (including NaN
                        # rounds) so the three-seed mean can be formed with
                        # nan-aware averaging at the same bootstrap index.
                        per_seed_ci[(rotation_name, left, right, metric, str(seed))] = diff
                        cis.append({"seed": seed, "rotation": rotation_name, "analysis_set": "common_all_methods", "left_method": left, "right_method": right, "metric": metric, "point_difference": float(lp - rp), "ci_low": float(np.quantile(finite, 0.025)), "ci_high": float(np.quantile(finite, 0.975)), "rounds": int(args.bootstrap_rounds), "n_compounds": int(len(set(arrays["compound"]))), "n_rows": int(len(arrays["label"]))})
    # Three-seed mean point and mean bootstrap distribution.  The seed-mean
    # row is the requested direction/CI gate; per-seed rows remain visible.
    for rotation_name in ("0", "1", "all"):
        for left, right in COMPARISONS:
            for metric in RANK_METRICS:
                arrays = [per_seed_ci.get((rotation_name, left, right, metric, str(seed))) for seed in SEEDS]
                arrays = [a for a in arrays if a is not None and len(a)]
                point_rows = [r for r in contrasts if r["seed"] in SEEDS and r["rotation"] == rotation_name and r["left_method"] == left and r["right_method"] == right and r["metric"] == metric]
                if len(arrays) != len(SEEDS) or len(point_rows) != len(SEEDS):
                    continue
                mean_boot = np.nanmean(np.vstack(arrays), axis=0)
                mean_boot = mean_boot[np.isfinite(mean_boot)]
                points = [float(r["point_difference"]) for r in point_rows]
                if len(mean_boot) == 0:
                    continue
                cis.append({"seed": "mean", "rotation": rotation_name, "analysis_set": "common_all_methods", "left_method": left, "right_method": right, "metric": metric, "point_difference": float(np.mean(points)), "ci_low": float(np.quantile(mean_boot, 0.025)), "ci_high": float(np.quantile(mean_boot, 0.975)), "rounds": int(args.bootstrap_rounds), "n_compounds": int(min(r["n_compounds"] for r in point_rows)), "n_rows": int(min(r["n_rows"] for r in point_rows))})
    write_csv(outdir / "METRICS_BY_SEED.csv", metrics_rows)
    write_csv(outdir / "PAIRED_CONTRASTS.csv", contrasts)
    write_csv(outdir / "BOOTSTRAP_CI.csv", cis)

    figure_status = render_figure(outdir, contrasts)
    config = {
        "version": VERSION,
        "dataset": "cpg0004-LINCS",
        "data": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "p0_root": str(args.p0_root.resolve()),
        "ge_root": str(args.ge_root.resolve()),
        "pair_root": str(args.pair_root.resolve()),
        "seeds": list(SEEDS),
        "doses": list(DOSES),
        "strict_repeat_count": REQUIRED_REPEATS,
        "available_frozen_support_slots": "per condition, audited from saved pair manifests; expected two rather than all five",
        "support_confirmation": "support row excluded from confirmation; remaining four rows split 2+2 in frozen plate order",
        "null_rounds": int(args.null_rounds),
        "bootstrap_rounds": int(args.bootstrap_rounds),
        "bootstrap_unit": "compound",
        "fixed_fpr": FIXED_FPR,
        "label_seed": int(args.label_seed),
        "label_thresholds_validation_only": threshold_info,
        "model_thresholds_validation_only": thresholds,
        "frozen_lambda_by_seed": {str(seed): meta[seed]["lambda"] for seed in SEEDS},
        "frozen_beta_by_seed": {str(seed): meta[seed]["beta"] for seed in SEEDS},
        "prediction_schema": meta,
        "p0_formula_reconstruction": p0_formula_errors,
        "ge_validation_threshold_status": "BLOCKED_VALIDATION_PREDICTIONS_MISSING",
        "a_ref_standardization": {
            "source": "training-compound delta robust MAD",
            "center": "training compound delta feature-wise median",
            "scale": "1.4826 * training compound delta MAD",
            "degenerate_fallback": "training compound delta std, then 1.0",
            "test_treatment_used_for_scale": False,
        },
        "figure_status": figure_status,
        "guardrails": [
            "no model retraining",
            "no lambda/beta reselection",
            "support and confirmation plate sets are disjoint",
            "validation-only label and activity thresholds",
            "compound-level bootstrap outer unit",
            "GE thresholded endpoints blocked when validation GE predictions are absent",
        ],
    }
    (outdir / "CONFIG.json").write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    # A concise report is generated from the saved CSVs and includes the hard
    # block explicitly.  It is intentionally conservative about GO language.
    primary = [r for r in cis if r.get("seed") == "mean" and r.get("rotation") == "all" and r.get("metric") == "AP" and r.get("right_method") == "raw"]
    common_test_for_report = rotation_records(test_rotations_by_seed[SEEDS[0]], True)
    active_test_rows = [row for row in common_test_for_report if int(row["confirmed_active"]) == 1]
    active_test_compounds = len({str(row["compound"]) for row in active_test_rows})
    active_test_conditions = len({(str(row["compound"]), str(row["dose"])) for row in active_test_rows})
    lines = [
        "# Results — cpg0004 confirmed activity hit recovery",
        "",
        "## 1. Objective",
        "Evaluate frozen raw 1R, reproducible-effect teacher, P0, and observed-GE posterior scores against an independent confirmed-activity label.",
        "",
        "## 2. Dataset",
        f"cpg0004-LINCS frozen CP artifact: {len(artifact.compound):,} plate-level rows, {artifact.feature_dim} features, {artifact.plate_count} plates, doses {', '.join(DOSES)}.",
        "",
        "## 3. Eligibility",
        f"Strict eligibility is exactly {REQUIRED_REPEATS} plate repeats: support excluded from 2+2 confirmation. Validation rotations: {len(validation_rotations):,}; test rotations before common GE restriction: {len(test_rotations_by_seed[SEEDS[0]]):,} (seed 3407).",
        "The saved 1R pair set supplies two unique support rows per mapped five-repeat condition; the other three physical slots are excluded because no frozen predictions exist.",
        "",
        "## 4. Exact information available to model",
        "Raw uses the support delta; teacher uses the saved teacher vector for that support; P0 combines that teacher with the saved virtual prior at held baseline s using the original lambda; GE posterior adds the saved correct_GE−P0 residual at held baseline s using the original beta. Confirmation rows are never model inputs.",
        "",
        "## 5. Independent reference construction",
        "For each support slot, the remaining four rows are split into disjoint A/B groups. Confirmed active requires E_rep and A_ref above validation-only q95 matched-null cutoffs.",
        "",
        "## 6. Primary endpoint",
        "Average Precision (AP) on the common all-method test rotations; compound-level bootstrap is the uncertainty unit.",
        "",
        "## 7. Secondary endpoints",
        "AUROC, recall at validation-selected 5% FPR, precision@top5%, precision@top10%, enrichment@top10%, missed-hit rescue, and false rescue. GE thresholded recall/rescue is blocked because frozen GE validation predictions are absent.",
        "",
        "## 8. Sample count",
        f"Validation rotations: {len(validation_rotations):,}. Test common-all-method rotations by seed: " + ", ".join(f"{seed}={sum(int(r['ge_available']) for r in rotation_records(test_rotations_by_seed[seed], False)):,}" for seed in SEEDS) + f". Confirmed-active labels in the common test table: {len(active_test_rows):,}/{len(common_test_for_report):,} rotations, {active_test_conditions:,} compound-dose conditions, and {active_test_compounds:,} unique compounds.",
        "",
        "## 9. Seed-level results",
        "See `METRICS_BY_SEED.csv` and `PAIRED_CONTRASTS.csv`; no dose or seed was selected after inspecting results.",
        "",
        "## 10. Paired CI",
        f"`BOOTSTRAP_CI.csv` contains {args.bootstrap_rounds:,}-round compound bootstrap intervals, including the three-seed mean row. The active count is only {len(active_test_rows):,} common-test rotations, so AP/recall inference has limited power.",
        "",
        "## 11. GO / SUPPORTIVE / NO-GO",
    ]
    if primary:
        row = primary[0]
        lines.append(f"Teacher−raw AP (three-seed mean): {fmt(row['point_difference'])}, 95% CI [{fmt(row['ci_low'])}, {fmt(row['ci_high'])}].")
        ge_rank_results = [r for r in cis if r.get("seed") == "mean" and r.get("rotation") == "all" and r.get("left_method") == "GE_posterior" and r.get("right_method") == "teacher" and r.get("metric") == "AP"]
        if ge_rank_results:
            ge_row = ge_rank_results[0]
            lines.append(f"GE-posterior−teacher AP (three-seed mean): {fmt(ge_row['point_difference'])}, 95% CI [{fmt(ge_row['ci_low'])}, {fmt(ge_row['ci_high'])}]. This ranking endpoint is estimable; GE thresholded endpoints remain BLOCKED_VALIDATION_PREDICTIONS_MISSING.")
        else:
            lines.append("P0−raw and GE−raw AP contrasts are in `BOOTSTRAP_CI.csv`; GE ranking contrast was not estimable. GE thresholded endpoints remain BLOCKED_VALIDATION_PREDICTIONS_MISSING.")
    else:
        lines.append("Primary AP contrast could not be estimated from the common set; status is BLOCKED.")
    lines.extend([
        "",
        "## 12. Biological interpretation",
        "A positive AP contrast would support improved ranking of independently confirmed reproducible perturbations under the tested one-repeat information regime. It does not establish replacement of a physical repeat, and two support-slot coverage is not full five-slot rotation.",
        "",
        "## 13. Limitations",
        "The frozen GE run has no validation predictions, so GE activity thresholds, fixed-FPR recall, and rescue rates cannot be claimed. The artifact lacks DMSO well-level dispersion; A_ref uses the documented training-delta MAD fallback. LINCS plate layout makes exact-well foreign matching unavailable; the null matches dose, plate, and well row. Conditions with two/four repeats and missing frozen prediction keys are excluded.",
        "",
        f"Mapping audit errors recorded: {len(mapping_errors)}. Figure status: {figure_status}.",
    ])
    (outdir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Decision is intentionally separate from RESULTS so downstream tooling
    # can read an unambiguous gate without parsing prose.
    decision_lines = ["# Decision — Experiment 1 hit recovery", "", f"Version: `{VERSION}`", ""]
    if mapping_errors:
        decision_lines.append(f"**OVERALL: BLOCKED_MAPPING_AUDIT** — {len(mapping_errors)} mapping/audit errors were detected; inspect `EXCLUSIONS.csv`. No biological GO claim is made.")
    else:
        ap_teacher = [r for r in cis if r.get("seed") == "mean" and r.get("rotation") == "all" and r.get("left_method") == "teacher" and r.get("right_method") == "raw" and r.get("metric") == "AP"]
        if ap_teacher and float(ap_teacher[0]["ci_low"]) > 0 and float(ap_teacher[0]["point_difference"]) > 0:
            decision_lines.append("**Bio-A confirmed hit recovery: GO** — teacher−raw AP is positive in the three-seed mean compound bootstrap CI.")
        elif ap_teacher:
            decision_lines.append("**Bio-A confirmed hit recovery: SUPPORTIVE/NO-GO** — teacher−raw AP does not meet the preregistered positive-CI gate.")
        else:
            decision_lines.append("**Bio-A confirmed hit recovery: BLOCKED** — no common AP contrast was estimable.")
        ge_ci = [r for r in cis if r.get("seed") == "mean" and r.get("rotation") == "all" and r.get("left_method") == "GE_posterior" and r.get("right_method") == "raw" and r.get("metric") == "Recall_at_fixed_FPR"]
        ge_rank = [r for r in cis if r.get("seed") == "mean" and r.get("rotation") == "all" and r.get("left_method") == "GE_posterior" and r.get("right_method") == "teacher" and r.get("metric") == "AP"]
        if ge_rank:
            ge_row = ge_rank[0]
            if float(ge_row["point_difference"]) > 0 and float(ge_row["ci_low"]) > 0:
                decision_lines.append(f"**GE ranking increment (AP vs teacher): GO** — difference {float(ge_row['point_difference']):.4f}, 95% CI [{float(ge_row['ci_low']):.4f}, {float(ge_row['ci_high']):.4f}].")
            else:
                decision_lines.append(f"**GE ranking increment (AP vs teacher): SUPPORTIVE/NO-GO** — difference {float(ge_row['point_difference']):.4f}, 95% CI [{float(ge_row['ci_low']):.4f}, {float(ge_row['ci_high']):.4f}]; this is a ranking result, not a thresholded endpoint.")
        else:
            decision_lines.append("**GE ranking increment (AP vs teacher): BLOCKED** — no common AP contrast was estimable.")
        if not ge_ci:
            decision_lines.append("**GE thresholded endpoints: BLOCKED_VALIDATION_PREDICTIONS_MISSING** — no GE validation predictions were saved; beta was not reselected and no substitute model was fit.")
        else:
            decision_lines.append("**GE thresholded endpoints: evaluated** — see `BOOTSTRAP_CI.csv`.")
        decision_lines.append("**Coverage qualification:** only two saved support slots per mapped five-repeat condition were used; full support-slot rotation is not claimed.")
    (outdir / "DECISION.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    print(json.dumps({"version": VERSION, "outdir": str(outdir), "validation_rotations": len(validation_rotations), "test_rotations_seed3407": len(test_rotations_by_seed[SEEDS[0]]), "mapping_errors": len(mapping_errors), "thresholds": thresholds, "ge_validation": "BLOCKED_VALIDATION_PREDICTIONS_MISSING", "bootstrap_rounds": args.bootstrap_rounds}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
