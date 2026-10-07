#!/usr/bin/env python3
"""Auditable cpg0004 Figure 4D Raw versus CFRA downstream adapter.

The adapter consumes the *per physical support view* output of the Figure 4
CFRA runner.  It rebuilds the old Panel-D estimands from the frozen CP rows
and curated annotation sources, while keeping Raw and CFRA paired on exactly
the same support rotation.  Historical IMR, virtual-prior, GE, or posterior
prediction files are deliberately not accepted as inputs.

The program is a runner, not a result file.  It is intentionally fail closed:
missing labels, missing pathway knowledge, an ambiguous profile key, or an
invalid two-rotation physical mapping produce a BLOCKED endpoint and an audit
record rather than silently falling back to an older estimator.

Historical endpoint contracts reproduced here are: hit recovery AP from the
label-only ``ELIGIBLE_SAMPLES.csv`` common test rows; MoA/target mAP from
dose-matched CP gallery conditions with >=3 physical repeats and labels from
the curated Broad primary/alternative fields, restricted by the frozen
Panel-D ``query_manifest.csv`` identity cohort; and Path-A Reactome ``k=25``
pathway mAP from top-neighbour target enrichment with per-query BH FDR.  The
bundle contract is the frozen ``CFRA_PER_PHYSICAL_VIEW.npz`` plus manifest,
with ``seed{3407,42,2025}_raw`` and ``seed{3407,42,2025}_cfra`` arrays only
read by this adapter.  ``PANEL_D_METRICS.csv`` contains arm-level AP/mAP,
``PANEL_D_CONTRASTS.csv`` contains paired CFRA-minus-Raw compound-bootstrap
contrasts, and ``BLOCKED.csv`` records any missing public input or failed
identity check.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scipy.stats import hypergeom
except Exception:  # pragma: no cover - checked at runtime and reported BLOCKED
    hypergeom = None


VERSION = "cpg0004-figure4-panel-d-cfra-adapter-v1-2026-09-20"
STATUS = "HISTORICAL_LOCKED_RECOMPUTE"
DEFAULT_BOOTSTRAP_ROUNDS = 10_000
EXPECTED_SEEDS = (3407, 42, 2025)
EXPECTED_FEATURE_DIM = 242
PANEL_ENDPOINTS = (
    "confirmed_activity_ap",
    "moa_map",
    "target_map",
    "reactome_target_neighbour_map",
)
METHODS = ("Raw", "CFRA")
FORBIDDEN_TOKENS = ("imr", "ge", "posterior", "virtual_prior", "p0", "teacher")
LEGACY_ANNOTATION_EXCLUSIONS = {"BRD-K50691590", "BRD-K60230970"}


class BlockedEndpoint(RuntimeError):
    """An endpoint cannot be computed without changing its estimand."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(seed: int, label: str) -> int:
    h = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int((int(seed) + int.from_bytes(h, "little")) % (2**63 - 1))


def canonical_dose(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return text
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return text
    if not number.is_finite():
        return text
    out = format(number.normalize(), "f")
    return "0" if out in {"-0", "-0.0"} else out


def text_array(values: Iterable[Any]) -> np.ndarray:
    return np.asarray([str(x).strip() for x in values], dtype=str)


def require_columns(frame: pd.DataFrame, names: Iterable[str], label: str) -> None:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise BlockedEndpoint(f"{label}:missing_columns={missing}")


def first_column(frame: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise BlockedEndpoint(f"{label}:missing_any_of={list(candidates)}")


def split_labels(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return set()
    return {part.strip() for part in re.split(r"[|,;/]+", text) if part.strip()}


def pcc_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    if q.ndim != 2 or g.ndim != 2 or q.shape[1] != g.shape[1]:
        raise BlockedEndpoint(f"PCC_shape_mismatch={q.shape}:{g.shape}")
    q = q - q.mean(axis=1, keepdims=True)
    g = g - g.mean(axis=1, keepdims=True)
    qnorm = np.linalg.norm(q, axis=1)
    gnorm = np.linalg.norm(g, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = (q @ g.T) / (qnorm[:, None] * gnorm[None, :])
    score[~np.isfinite(score)] = -np.inf
    return score


def pcc_matrix_with_cached_gallery(query: np.ndarray, gallery_centered: np.ndarray, gallery_norm: np.ndarray) -> np.ndarray:
    """Compute PCC with a gallery block prepared by :func:`pcc_matrix`.

    ``retrieval_rows`` evaluates many queries against the same dose-matched
    gallery.  The old implementation re-centered and re-normed that gallery
    for every query.  This helper retains the exact per-query arithmetic and
    ranking inputs of ``pcc_matrix`` while allowing that deterministic gallery
    work to be reused; it deliberately does not batch query rows, which could
    change BLAS reduction order at machine precision and therefore a tie.
    """
    q = np.asarray(query, dtype=np.float64)
    g = np.asarray(gallery_centered, dtype=np.float64)
    norms = np.asarray(gallery_norm, dtype=np.float64)
    if q.ndim != 2 or g.ndim != 2 or norms.ndim != 1 or q.shape[1] != g.shape[1] or g.shape[0] != norms.shape[0]:
        raise BlockedEndpoint(f"PCC_cached_shape_mismatch={q.shape}:{g.shape}:{norms.shape}")
    q = q - q.mean(axis=1, keepdims=True)
    qnorm = np.linalg.norm(q, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = (q @ g.T) / (qnorm[:, None] * norms[None, :])
    score[~np.isfinite(score)] = -np.inf
    return score


def average_precision(labels: np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool)
    n_positive = int(y.sum())
    if n_positive <= 0:
        return float("nan")
    positions = np.flatnonzero(y) + 1
    return float((np.arange(1, len(positions) + 1, dtype=np.float64) / positions).sum() / n_positive)


def ensure_output_dir(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"Refusing nonempty output directory without --force: {path}")
    path.mkdir(parents=True, exist_ok=True)


def read_artifact(data_path: Path, manifest_path: Path, split_lock_path: Path) -> dict[str, Any]:
    for path in (data_path, manifest_path, split_lock_path):
        if not path.is_file():
            raise BlockedEndpoint(f"missing_frozen_input={path}")
    with np.load(data_path, allow_pickle=False) as loaded:
        # This is the same prepared artifact contract consumed by the frozen
        # biological evaluators.  ``baseline`` is not used as a score here,
        # but requiring it prevents a similarly named, reduced profile table
        # from being mistaken for the cpg0004 physical-row artifact.
        needed = {"compound_id", "dose", "plate", "well", "split", "delta", "baseline"}
        missing = sorted(needed - set(loaded.files))
        if missing:
            raise BlockedEndpoint(f"cp_artifact_missing_keys={missing}")
        compound = text_array(loaded["compound_id"])
        dose = np.asarray([canonical_dose(x) for x in loaded["dose"]], dtype=str)
        plate = text_array(loaded["plate"])
        well = text_array(loaded["well"])
        split = text_array(loaded["split"])
        delta = np.asarray(loaded["delta"], dtype=np.float32)
        baseline = np.asarray(loaded["baseline"], dtype=np.float32)
    n = len(compound)
    if any(len(x) != n for x in (dose, plate, well, split, delta, baseline)):
        raise BlockedEndpoint("cp_artifact_inconsistent_row_counts")
    if delta.ndim != 2 or baseline.ndim != 2 or baseline.shape != delta.shape or not np.isfinite(delta).all() or not np.isfinite(baseline).all():
        raise BlockedEndpoint("cp_artifact_delta_baseline_not_finite_or_shape_mismatch")
    if delta.shape[1] != EXPECTED_FEATURE_DIM:
        raise BlockedEndpoint(f"cp_artifact_feature_dim={delta.shape[1]}:expected={EXPECTED_FEATURE_DIM}")
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    require_columns(manifest, ("compound_id", "dose", "plate", "well", "split"), "cp_manifest")
    if len(manifest) != n:
        raise BlockedEndpoint(f"cp_manifest_row_count={len(manifest)}:npz={n}")
    for name, values in (
        ("compound_id", compound),
        ("dose", dose),
        ("plate", plate),
        ("well", well),
        ("split", split),
    ):
        actual = np.asarray([canonical_dose(x) if name == "dose" else str(x).strip() for x in manifest[name]], dtype=str)
        if not np.array_equal(actual, values):
            raise BlockedEndpoint(f"cp_manifest_npz_mismatch={name}")
    physical = list(zip(compound, dose, plate, well))
    if len(set(physical)) != n:
        raise BlockedEndpoint("cp_physical_key_duplicate")
    lock = json.loads(split_lock_path.read_text(encoding="utf-8"))
    split_sets = {name: set(map(str, lock.get(f"{name}_compounds", []))) for name in ("train", "valid", "test")}
    if not all(split_sets.values()):
        # The frozen lock uses either *_compounds or a compound list under a
        # split key.  Accept both spellings, but never infer a split silently.
        for name in split_sets:
            value = lock.get(name)
            if isinstance(value, list):
                split_sets[name] = set(map(str, value))
    if not all(split_sets.values()):
        raise BlockedEndpoint("split_lock_missing_compound_lists")
    if set.union(*(split_sets[name] for name in split_sets)) and sum(map(len, split_sets.values())) != len(set.union(*(split_sets[name] for name in split_sets))):
        raise BlockedEndpoint("split_lock_compound_overlap")
    for name in ("train", "valid", "test"):
        observed = set(compound[split == name])
        if observed != split_sets[name]:
            raise BlockedEndpoint(f"split_lock_observed_mismatch={name}:{len(observed)}:{len(split_sets[name])}")
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, key in enumerate(zip(compound, dose)):
        groups[(str(key[0]), str(key[1]))].append(i)
    for key in groups:
        groups[key].sort(key=lambda i: (str(plate[i]), str(well[i]), int(i)))
    if not all(np.any(split == name) for name in ("train", "valid", "test")):
        raise BlockedEndpoint("cp_artifact_missing_split_rows")
    train_rows = np.flatnonzero(split == "train")
    center = np.median(delta[train_rows].astype(np.float64), axis=0)
    mad = 1.4826 * np.median(np.abs(delta[train_rows].astype(np.float64) - center), axis=0)
    bad = ~np.isfinite(mad) | (mad < 1e-6)
    std = np.std(delta[train_rows].astype(np.float64), axis=0)
    mad[bad] = std[bad]
    bad = ~np.isfinite(mad) | (mad < 1e-6)
    mad[bad] = 1.0
    return {
        "compound": compound,
        "dose": dose,
        "plate": plate,
        "well": well,
        "split": split,
        "delta": delta,
        "baseline": baseline,
        "groups": dict(groups),
        "split_sets": split_sets,
        "train_center": center.astype(np.float32),
        "train_scale": mad.astype(np.float32),
        "feature_dim": int(delta.shape[1]),
    }


def _parse_rows(value: Any) -> tuple[int, ...]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return tuple()
    text = str(value).strip()
    if not text:
        return tuple()
    try:
        return tuple(int(part.strip()) for part in re.split(r"[|,; ]+", text) if part.strip())
    except ValueError as exc:
        raise BlockedEndpoint(f"invalid_reference_rows={text}") from exc


def load_bundle(profiles_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Load only the Raw/CFRA views from the explicit physical-view bundle.

    The upstream runner stores several candidate arrays in one NPZ for
    provenance (``imr``, ``lso``, ``imceb``, and ``reference`` are expected to
    be present).  This adapter never reads those arrays.  The identity arrays
    and the two requested profile arrays are selected by seed and aligned to
    the explicit manifest; an accidental unprefixed/legacy file is refused.
    """
    if not profiles_path.is_file() or not manifest_path.is_file():
        raise BlockedEndpoint(f"cfra_bundle_files_missing={manifest_path}:{profiles_path}")
    if "CFRA_PER_PHYSICAL_VIEW" not in profiles_path.name.upper() or "CFRA_PER_PHYSICAL_VIEW_MANIFEST" not in manifest_path.name.upper():
        raise BlockedEndpoint("profiles_manifest_are_not_named_cfra_per_physical_view")
    bundle_dir = profiles_path.parent
    audit_candidates = (bundle_dir / "bundle_audit.json", bundle_dir / "BUNDLE_AUDIT.json", bundle_dir / "AUDIT.json", bundle_dir / "CFRA_PER_PHYSICAL_VIEW_AUDIT.json", bundle_dir / "PREDICT_COMPLETE.json")
    audit_path = next((p for p in audit_candidates if p.is_file()), None)
    metadata = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path is not None else {"status": "audit_sidecar_not_found"}
    status = str(metadata.get("status", ""))
    if audit_path is not None and status and status != STATUS:
        raise BlockedEndpoint(f"cfra_bundle_status={status}")
    if audit_path is not None and metadata.get("test_used_for_fit_or_selection") not in (None, False):
        raise BlockedEndpoint("cfra_bundle_test_used_for_fit_or_selection")
    mf = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    require_columns(mf, ("seed",), "cfra_manifest")
    mf["seed"] = mf["seed"].astype(int)
    observed_seeds = tuple(sorted(set(mf["seed"].astype(int))))
    if observed_seeds != tuple(sorted(EXPECTED_SEEDS)):
        raise BlockedEndpoint(f"cfra_manifest_seeds={observed_seeds}:expected={tuple(sorted(EXPECTED_SEEDS))}")
    compound_col = first_column(mf, ("compound", "compound_id"), "cfra_manifest")
    dose_col = first_column(mf, ("dose", "nominal_dose"), "cfra_manifest")
    support_col = first_column(mf, ("support_row", "support_index", "support_physical_row"), "cfra_manifest")
    rotation_col = first_column(mf, ("rotation", "support_rotation", "rotation_rank"), "cfra_manifest")
    support_plate_col = first_column(mf, ("support_plate", "support_physical_plate"), "cfra_manifest")
    support_well_col = first_column(mf, ("support_well", "support_physical_well"), "cfra_manifest")
    profile_col = next((x for x in ("profile_index", "row_index", "bundle_row") if x in mf.columns), None)
    if profile_col is None:
        raise BlockedEndpoint("cfra_manifest_missing_profile_index")
    row_type_col = next((x for x in ("row_type", "record_type") if x in mf.columns), None)
    if row_type_col:
        values = mf[row_type_col].astype(str).str.lower()
        if values.isin({"aggregate", "condition_aggregate", "mean"}).any():
            raise BlockedEndpoint("cfra_manifest_contains_aggregate_rows")
    mf = mf.rename(columns={compound_col: "compound", dose_col: "dose", support_col: "support_row", rotation_col: "rotation", support_plate_col: "support_plate", support_well_col: "support_well"})
    mf["compound"] = mf["compound"].astype(str).str.strip()
    mf["dose"] = mf["dose"].map(canonical_dose)
    mf["support_plate"] = mf["support_plate"].astype(str).str.strip()
    mf["support_well"] = mf["support_well"].astype(str).str.strip()
    if (mf["support_plate"].eq("") | mf["support_well"].eq("")).any():
        raise BlockedEndpoint("cfra_manifest_missing_support_plate_or_well")
    try:
        mf["support_row"] = mf["support_row"].astype(int)
        mf["rotation"] = mf["rotation"].astype(int)
    except Exception as exc:
        raise BlockedEndpoint("cfra_manifest_identity_columns_not_integer") from exc
    mf["profile_index"] = mf[profile_col].astype(int)
    key_cols = ["seed", "compound", "dose", "support_row", "rotation"]
    if mf.duplicated(key_cols).any():
        raise BlockedEndpoint("cfra_manifest_rotation_key_duplicate")
    if mf.duplicated(["seed", "profile_index"]).any():
        raise BlockedEndpoint("cfra_manifest_profile_index_duplicate")
    if not len(mf):
        raise BlockedEndpoint("cfra_manifest_has_no_rotation_rows")
    with np.load(profiles_path, allow_pickle=False) as loaded:
        files = set(loaded.files)
        seeds = sorted(set(mf.seed.astype(int)))
        raw_parts: list[np.ndarray] = []
        cfra_parts: list[np.ndarray] = []
        # The main CFRA runner uses seed<id>_<field> keys.  An unprefixed
        # bundle is accepted only for a one-seed manifest and only when both
        # requested profile arrays are the sole profile names.
        prefixed = all(f"seed{seed}_raw" in files and f"seed{seed}_cfra" in files for seed in seeds)
        unprefixed = {"raw", "cfra"}.issubset(files) and len(seeds) == 1
        if not prefixed and not unprefixed:
            raise BlockedEndpoint(f"cfra_profile_npz_missing_seeded_raw_cfra={sorted(files)}")
        identity_arrays: dict[tuple[int, str], np.ndarray] = {}
        for seed in seeds:
            prefix = "" if unprefixed else f"seed{seed}_"
            raw_seed = np.asarray(loaded[f"{prefix}raw"], dtype=np.float32)
            cfra_seed = np.asarray(loaded[f"{prefix}cfra"], dtype=np.float32)
            group = mf[mf.seed.eq(seed)].sort_values("profile_index")
            if group.profile_index.to_numpy(dtype=np.int64).tolist() != list(range(len(group))):
                raise BlockedEndpoint(f"cfra_profile_index_not_contiguous={seed}")
            if raw_seed.ndim != 2 or cfra_seed.ndim != 2 or raw_seed.shape != cfra_seed.shape or raw_seed.shape[0] != len(group):
                raise BlockedEndpoint(f"cfra_seed_shape_or_manifest_mismatch={seed}:{raw_seed.shape}:{len(group)}")
            raw_parts.append(raw_seed)
            cfra_parts.append(cfra_seed)
            for field, aliases in (("compound", ("compound", "compound_id")), ("dose", ("dose",)), ("support_row", ("support_row", "support_index")), ("rotation", ("rotation", "rotation_rank"))):
                key = next((f"{prefix}{alias}" for alias in aliases if f"{prefix}{alias}" in files), None)
                if key is not None:
                    identity_arrays[(seed, field)] = np.asarray(loaded[key])
            for field in ("compound", "dose", "support_row", "rotation"):
                arr = identity_arrays.get((seed, field))
                if arr is None:
                    continue
                if len(arr) != len(group):
                    raise BlockedEndpoint(f"cfra_identity_length_mismatch={seed}:{field}")
                if field == "compound" and not np.array_equal(text_array(arr), group.compound.to_numpy(dtype=str)):
                    raise BlockedEndpoint(f"cfra_identity_compound_mismatch={seed}")
                if field == "dose" and not np.array_equal(np.asarray([canonical_dose(x) for x in arr], dtype=str), group.dose.to_numpy(dtype=str)):
                    raise BlockedEndpoint(f"cfra_identity_dose_mismatch={seed}")
                if field in {"support_row", "rotation"} and not np.array_equal(np.asarray(arr, dtype=np.int64), group[field].to_numpy(dtype=np.int64)):
                    raise BlockedEndpoint(f"cfra_identity_{field}_mismatch={seed}")
        raw = np.vstack(raw_parts).astype(np.float32)
        cfra = np.vstack(cfra_parts).astype(np.float32)
    if raw.ndim != 2 or cfra.ndim != 2 or raw.shape != cfra.shape or raw.shape[1] != EXPECTED_FEATURE_DIM or not np.isfinite(raw).all() or not np.isfinite(cfra).all():
        raise BlockedEndpoint(f"cfra_profile_shape_or_finite={raw.shape}:{cfra.shape}")
    if raw.shape[0] != len(mf):
        raise BlockedEndpoint(f"cfra_profile_manifest_rows={len(mf)}:npz_rows={raw.shape[0]}")
    if not (mf.groupby(["seed", "compound", "dose"]).rotation.nunique() == 2).all():
        raise BlockedEndpoint("cfra_requires_exactly_two_rotations_per_condition")
    if set(mf.rotation.astype(int)) != {0, 1}:
        raise BlockedEndpoint(f"cfra_rotation_values={sorted(set(mf.rotation.astype(int)))}:expected=[0,1]")
    if (mf.groupby(["seed", "compound", "dose"]).support_row.nunique() != 2).any():
        raise BlockedEndpoint("cfra_rotations_must_have_distinct_support_rows")
    if "reference_rows" in mf.columns:
        mf["reference_rows"] = mf["reference_rows"].map(_parse_rows)
    elif "reference_row_indices" in mf.columns:
        mf["reference_rows"] = mf["reference_row_indices"].map(_parse_rows)
    else:
        raise BlockedEndpoint("cfra_manifest_missing_reference_rows")
    if any(len(value) != 4 for value in mf["reference_rows"]):
        raise BlockedEndpoint("cfra_manifest_reference_rows_not_four")
    # Reorder arrays from seed blocks to manifest row order.  The manifest is
    # allowed to interleave seeds; profiles are stored as one block per seed.
    block_offsets = {}
    offset = 0
    for seed in seeds:
        block_offsets[seed] = offset
        offset += int(mf.seed.eq(seed).sum())
    order = []
    for seed in mf.seed.astype(int):
        group = mf[mf.seed.eq(seed)].sort_values("profile_index")
        local = int(np.flatnonzero(group.index.to_numpy() == len(order))[0]) if False else None
        # profile_index is the stable row identity.  Its value may not start
        # at zero, so resolve it directly against the sorted seed block.
        row_pos = int(mf.index[len(order)])
        sorted_indices = group.index.to_numpy()
        order.append(block_offsets[seed] + int(np.flatnonzero(sorted_indices == row_pos)[0]))
    raw = raw[np.asarray(order, dtype=np.int64)]
    cfra = cfra[np.asarray(order, dtype=np.int64)]
    return {"manifest": mf.reset_index(drop=True), "raw": raw, "cfra": cfra, "audit": metadata, "manifest_path": manifest_path, "profiles_path": profiles_path, "audit_path": audit_path}


def validate_bundle_physical(bundle: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    mf = bundle["manifest"].copy()
    errors: list[str] = []
    rows_checked = 0
    reference_checked = 0
    if set(mf.seed.astype(int)) != set(EXPECTED_SEEDS):
        raise BlockedEndpoint(f"cfra_bundle_seed_set={sorted(set(mf.seed.astype(int)))}")
    for i, row in mf.iterrows():
        key = (str(row.compound), canonical_dose(row.dose))
        rows = artifact["groups"].get(key, [])
        if str(row.compound) not in artifact["split_sets"]["test"]:
            errors.append(f"non_test_bundle_condition={key}")
            continue
        if len(rows) != 5:
            errors.append(f"condition_repeat_count={key}:{len(rows)}")
            continue
        support = int(row.support_row)
        if support not in rows:
            errors.append(f"support_row_not_physical={key}:{support}")
            continue
        if "support_plate" in mf.columns and str(row.support_plate) != str(artifact["plate"][support]):
            errors.append(f"support_plate_mismatch={key}:{support}:{row.support_plate}:{artifact['plate'][support]}")
            continue
        if "support_well" in mf.columns and str(row.support_well) != str(artifact["well"][support]):
            errors.append(f"support_well_mismatch={key}:{support}:{row.support_well}:{artifact['well'][support]}")
            continue
        refs = tuple(int(x) for x in row.reference_rows)
        if len(refs) != 4 or len(set(refs)) != 4 or support in refs or set(refs) != set(rows) - {support}:
            errors.append(f"reference_identity_mismatch={key}:{support}:{refs}")
            continue
        for ref in refs:
            if ref < 0 or ref >= len(artifact["compound"]):
                errors.append(f"reference_row_out_of_range={key}:{ref}")
                continue
            if (str(artifact["compound"][ref]), str(artifact["dose"][ref])) != key:
                errors.append(f"reference_condition_mismatch={key}:{ref}")
        rows_checked += 1
        reference_checked += len(refs)
    if errors:
        raise BlockedEndpoint(";".join(errors[:20]) + (f";n_errors={len(errors)}" if len(errors) > 20 else ""))
    per_seed_keys = {
        int(seed): set(zip(group.compound.astype(str), group.dose.astype(str), group.support_row.astype(int)))
        for seed, group in mf.groupby("seed", sort=True)
    }
    if any(per_seed_keys[int(seed)] != per_seed_keys[EXPECTED_SEEDS[0]] for seed in EXPECTED_SEEDS[1:]):
        raise BlockedEndpoint("cfra_cross_seed_physical_key_drift")
    bundle["manifest"] = mf
    return {"rotation_rows": int(len(mf)), "conditions": int(mf.groupby(["seed", "compound", "dose"]).ngroups), "references": int(reference_checked), "physical_rows_checked": int(rows_checked), "support_reference_overlap": 0, "cross_seed_key_count": int(len(per_seed_keys[EXPECTED_SEEDS[0]])), "seed_counts": {str(seed): int((mf.seed == seed).sum()) for seed in EXPECTED_SEEDS}}


def load_annotations(path: Path) -> dict[str, dict[str, set[str]]]:
    if not path.is_file():
        raise BlockedEndpoint(f"missing_curated_annotation_source={path}")
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    frame = pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False)
    compound_col = first_column(frame, ("compound_id", "broad_id"), "curated_annotations")
    moa_col = first_column(frame, ("moa", "Metadata_moa"), "curated_annotations")
    target_col = first_column(frame, ("target", "Metadata_target"), "curated_annotations")
    alt_moa = next((x for x in ("alternative_moa", "alternative_moa_labels") if x in frame.columns), None)
    alt_target = next((x for x in ("alternative_target", "alternative_target_labels") if x in frame.columns), None)
    sample_type_col = next((x for x in ("sample_type", "Metadata_broad_sample_type", "broad_sample_type") if x in frame.columns), None)
    if sample_type_col:
        frame = frame[frame[sample_type_col].astype(str).str.strip().str.lower().eq("trt")].copy()
    out: dict[str, dict[str, set[str]]] = {}
    for _, row in frame.iterrows():
        cid = str(row[compound_col]).strip()
        if not cid or cid in LEGACY_ANNOTATION_EXCLUSIONS:
            continue
        if cid in out:
            # A repeated Broad record is only safe when it yields the same
            # union labels; conflicting rows would alter the old estimand.
            current = out[cid]
        else:
            current = {"moa": set(), "target": set()}
            out[cid] = current
        current["moa"] |= split_labels(row[moa_col])
        current["target"] |= split_labels(row[target_col])
        if alt_moa:
            current["moa"] |= split_labels(row[alt_moa])
        if alt_target:
            current["target"] |= split_labels(row[alt_target])
    if not out:
        raise BlockedEndpoint("curated_annotation_source_empty")
    return out


def build_gallery(artifact: dict[str, Any], annotations: dict[str, dict[str, set[str]]], min_repeats: int) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[dict[str, Any]] = []
    profiles: list[np.ndarray] = []
    for (compound, dose), rows in sorted(artifact["groups"].items()):
        if compound not in artifact["split_sets"]["train"] | artifact["split_sets"]["valid"]:
            continue
        if len(rows) < min_repeats:
            continue
        labels = annotations.get(compound, {"moa": set(), "target": set()})
        profile = artifact["delta"][np.asarray(rows, dtype=np.int64)].mean(axis=0, dtype=np.float64).astype(np.float32)
        records.append({"gallery_id": f"{compound}|{dose}", "compound": compound, "dose": dose, "repeat_count": len(rows), "moa": "|".join(sorted(labels["moa"])), "target": "|".join(sorted(labels["target"])), "profile_index": len(profiles)})
        profiles.append(profile)
    if not records:
        raise BlockedEndpoint("gallery_empty_after_train_valid_repeat_gate")
    gallery = pd.DataFrame(records)
    values = np.vstack(profiles).astype(np.float32)
    if gallery.compound.isin(artifact["split_sets"]["test"]).any():
        raise BlockedEndpoint("test_compound_entered_gallery")
    return gallery, values


def load_activity_labels(path: Path | None, artifact: dict[str, Any], bundle: dict[str, Any]) -> pd.DataFrame:
    if path is None:
        raise BlockedEndpoint("confirmed_activity_labels_not_supplied_as_label_only_sidecar")
    if not path.is_file():
        raise BlockedEndpoint(f"missing_activity_label_sidecar={path}")
    # Use only label/identity columns.  This deliberately prevents a legacy
    # ELIGIBLE_SAMPLES file's IMR/GE score columns from entering the adapter.
    header = pd.read_csv(path, nrows=0)
    compound_col = first_column(header, ("compound", "compound_id"), "activity_labels")
    dose_col = first_column(header, ("dose", "nominal_dose"), "activity_labels")
    support_col = first_column(header, ("support_row", "support_index"), "activity_labels")
    label_col = first_column(header, ("confirmed_active", "activity_label", "label"), "activity_labels")
    usecols = [compound_col, dose_col, support_col, label_col]
    rotation_col = next((x for x in ("rotation", "rotation_rank", "support_rotation") if x in header.columns), None)
    seed_col = next((x for x in ("seed", "profile_seed") if x in header.columns), None)
    role_col = next((x for x in ("role", "record_role") if x in header.columns), None)
    split_col = next((x for x in ("split", "dataset_split") if x in header.columns), None)
    common_col = next((x for x in ("evaluation_common_all_methods", "common_all_methods") if x in header.columns), None)
    missing_filters = [name for name, column in (("role", role_col), ("split", split_col), ("evaluation_common_all_methods", common_col), ("seed", seed_col), ("rotation", rotation_col)) if column is None]
    if missing_filters:
        raise BlockedEndpoint(f"confirmed_activity_label_sidecar_missing_columns={missing_filters}")
    if rotation_col:
        usecols.append(rotation_col)
    if seed_col:
        usecols.append(seed_col)
    for optional in (role_col, split_col, common_col):
        if optional:
            usecols.append(optional)
    frame = pd.read_csv(path, dtype=str, usecols=usecols, keep_default_na=False)
    # The historical ELIGIBLE_SAMPLES.csv contains validation calibration rows,
    # test rows excluded from the common all-method set, and the rows used by
    # Figure 4's common endpoint.  Only the last group is a permissible label
    # source for this adapter; model score columns in that file are ignored.
    if role_col:
        frame = frame[frame[role_col].astype(str).str.strip().eq("test_evaluation")].copy()
    if split_col:
        frame = frame[frame[split_col].astype(str).str.strip().eq("test")].copy()
    if common_col:
        common_values = frame[common_col].astype(str).str.strip().str.lower()
        frame = frame[common_values.isin({"1", "true", "yes"})].copy()
    if frame.empty:
        raise BlockedEndpoint("confirmed_activity_label_sidecar_has_no_common_test_rows")
    frame = frame.rename(columns={compound_col: "compound", dose_col: "dose", support_col: "support_row", label_col: "label"})
    frame["compound"] = frame.compound.astype(str).str.strip(); frame["dose"] = frame.dose.map(canonical_dose); frame["support_row"] = frame.support_row.astype(int)
    frame["label"] = pd.to_numeric(frame.label, errors="coerce")
    if frame.label.isna().any() or ~frame.label.isin([0, 1]).all():
        raise BlockedEndpoint("activity_labels_not_binary")
    frame["label"] = frame.label.astype(int)
    frame["rotation"] = frame[rotation_col].astype(int) if rotation_col else -1
    frame["seed"] = frame[seed_col].astype(int) if seed_col else -1
    frame["query_key"] = [f"seed{seed}|{c}|{d}|support{s}" for seed, c, d, s in zip(frame.seed, frame.compound, frame.dose, frame.support_row)]
    if frame.duplicated("query_key").any():
        raise BlockedEndpoint("activity_label_key_duplicate")
    bundle_mf = bundle["manifest"]
    if not seed_col:
        # Confirmed activity is a physical-label property and is expected to
        # be identical across the three frozen profile seeds.  Broadcast only
        # when the sidecar contains one unique identity/label row and no
        # conflicting duplicate.
        base_keys = set(zip(frame.compound, frame.dose, frame.support_row))
        if len(base_keys) != len(frame):
            raise BlockedEndpoint("activity_label_duplicate_without_seed")
        frame = pd.concat([frame.assign(seed=int(seed)) for seed in sorted(bundle_mf.seed.unique())], ignore_index=True)
        frame["query_key"] = [f"seed{seed}|{c}|{d}|support{s}" for seed, c, d, s in zip(frame.seed, frame.compound, frame.dose, frame.support_row)]
    bundle_keys = set(zip(bundle_mf.seed.astype(int), bundle_mf.compound, bundle_mf.dose, bundle_mf.support_row))
    label_keys = set(zip(frame.seed.astype(int), frame.compound, frame.dose, frame.support_row))
    if not label_keys.issubset(bundle_keys):
        raise BlockedEndpoint(f"activity_label_key_not_in_cfra_bundle={len(label_keys - bundle_keys)}")
    if not label_keys:
        raise BlockedEndpoint("activity_label_bundle_intersection_empty")
    if rotation_col:
        bundle_rotation = {(int(seed), str(compound), canonical_dose(dose), int(support)): int(rotation) for seed, compound, dose, support, rotation in zip(bundle_mf.seed, bundle_mf.compound, bundle_mf.dose, bundle_mf.support_row, bundle_mf.rotation)}
        for row in frame.itertuples(index=False):
            key = (int(row.seed), str(row.compound), canonical_dose(row.dose), int(row.support_row))
            if int(row.rotation) != bundle_rotation[key]:
                raise BlockedEndpoint(f"activity_label_rotation_mismatch={key}")
    # Labels are endpoint properties, not seed-specific outcomes.  A conflicting
    # label for the same physical support across frozen seeds would make the
    # old three-seed paired estimand undefined.
    label_check = frame.groupby(["compound", "dose", "support_row"], sort=True).label.nunique()
    if (label_check > 1).any():
        raise BlockedEndpoint("activity_label_seed_conflict")
    if set(frame.compound) - artifact["split_sets"]["test"]:
        raise BlockedEndpoint("activity_label_non_test_compound")
    return frame


def query_frame(bundle: dict[str, Any], artifact: dict[str, Any], annotations: dict[str, dict[str, set[str]]]) -> pd.DataFrame:
    mf = bundle["manifest"]
    records: list[dict[str, Any]] = []
    for i, row in mf.iterrows():
        seed, compound, dose, support = int(row.seed), str(row.compound), canonical_dose(row.dose), int(row.support_row)
        labels = annotations.get(compound, {"moa": set(), "target": set()})
        records.append({"profile_index": int(i), "seed": seed, "query_id": f"seed{seed}|{compound}|{dose}|support{support}", "compound": compound, "dose": dose, "support_row": support, "rotation": int(row.rotation), "moa": "|".join(sorted(labels["moa"])), "target": "|".join(sorted(labels["target"]))})
    return pd.DataFrame(records)


def load_query_manifest(path: Path | None, bundle: dict[str, Any]) -> tuple[set[tuple[int, str, str, int]], dict[str, Any]]:
    """Load the old D query identity cohort without importing its scores.

    The historical retrieval artifacts intentionally expose a slightly larger
    physical CFRA bundle than the old D cohort.  The cohort manifest is an
    identity/eligibility lock, not a model output; requiring it prevents this
    adapter from silently changing the published test population.
    """
    if path is None:
        raise BlockedEndpoint("retrieval_query_manifest_not_supplied")
    if not path.is_file():
        raise BlockedEndpoint(f"missing_retrieval_query_manifest={path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    require_columns(frame, ("seed",), "retrieval_query_manifest")
    compound_col = first_column(frame, ("compound_id", "compound"), "retrieval_query_manifest")
    dose_col = first_column(frame, ("dose", "nominal_dose"), "retrieval_query_manifest")
    support_col = first_column(frame, ("support_row", "support_index"), "retrieval_query_manifest")
    frame["seed"] = frame["seed"].astype(int)
    frame["compound"] = frame[compound_col].astype(str).str.strip()
    frame["dose"] = frame[dose_col].map(canonical_dose)
    frame["support_row"] = frame[support_col].astype(int)
    if tuple(sorted(set(frame.seed))) != tuple(sorted(EXPECTED_SEEDS)):
        raise BlockedEndpoint(f"retrieval_query_manifest_seeds={tuple(sorted(set(frame.seed)))}:expected={tuple(sorted(EXPECTED_SEEDS))}")
    if frame.duplicated(["seed", "compound", "dose", "support_row"]).any():
        raise BlockedEndpoint("retrieval_query_manifest_key_duplicate")
    bundle_keys = set(zip(bundle["manifest"].seed.astype(int), bundle["manifest"].compound.astype(str), bundle["manifest"].dose.astype(str), bundle["manifest"].support_row.astype(int)))
    query_keys = set(zip(frame.seed, frame.compound, frame.dose, frame.support_row))
    if not query_keys.issubset(bundle_keys):
        raise BlockedEndpoint(f"retrieval_query_manifest_key_not_in_cfra_bundle={len(query_keys - bundle_keys)}")
    counts = frame.groupby(["seed", "compound", "dose"], sort=True).support_row.nunique()
    if not (counts == 2).all():
        raise BlockedEndpoint("retrieval_query_manifest_requires_two_support_rotations")
    return query_keys, {"path": str(path), "sha256": sha256(path), "rows": int(len(frame)), "conditions": int(frame.groupby(["seed", "compound", "dose"]).ngroups)}


def activity_scores(artifact: dict[str, Any], bundle: dict[str, Any], labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    center, scale = artifact["train_center"], artifact["train_scale"]
    mf = bundle["manifest"]
    label_map = labels.set_index("query_key")["label"].to_dict()
    all_keys = [f"seed{seed}|{compound}|{dose}|support{support}" for seed, compound, dose, support in zip(mf.seed, mf.compound, mf.dose, mf.support_row)]
    selected = np.asarray([key in label_map for key in all_keys], dtype=bool)
    if not selected.any():
        raise BlockedEndpoint("confirmed_activity_label_bundle_intersection_empty")
    positions = np.flatnonzero(selected)
    raw = np.linalg.norm((bundle["raw"].astype(np.float64)[positions] - center) / scale, axis=1)
    cfra = np.linalg.norm((bundle["cfra"].astype(np.float64)[positions] - center) / scale, axis=1)
    keys = [all_keys[int(position)] for position in positions]
    if set(keys) != set(label_map):
        raise BlockedEndpoint("confirmed_activity_label_profile_key_mismatch")
    selected_mf = mf.iloc[positions].reset_index(drop=True)
    q = pd.DataFrame({"seed": selected_mf.seed.astype(int).to_numpy(), "query_id": keys, "compound": selected_mf.compound.astype(str).to_numpy(), "dose": selected_mf.dose.astype(str).to_numpy(), "support_row": selected_mf.support_row.astype(int).to_numpy(), "rotation": selected_mf.rotation.astype(int).to_numpy(), "label": [int(label_map[key]) for key in keys]})
    q["raw"] = raw; q["cfra"] = cfra
    rows: list[dict[str, Any]] = []
    score_columns = {"Raw": "raw", "CFRA": "cfra"}
    for method in METHODS:
        order = np.argsort(-q[score_columns[method]].to_numpy(dtype=float), kind="mergesort")
        ap = average_precision(q.label.to_numpy(dtype=int)[order])
        rows.append({"endpoint": "confirmed_activity_ap", "method": method, "metric": "AP", "point": ap, "query_count": int(len(q)), "compound_count": int(q.compound.nunique())})
    result = q[["seed", "query_id", "compound", "dose", "support_row", "rotation", "label", "raw", "cfra"]].copy()
    result["endpoint"] = "confirmed_activity_ap"; result["metric"] = "AP"
    result["raw_metric"] = result["raw"]; result["cfra_metric"] = result["cfra"]
    # Store AP per row only for audit; arm-level AP is in PANEL_D_METRICS.
    return pd.DataFrame(rows), result


def retrieval_rows(
    endpoint: str,
    query: pd.DataFrame,
    bundle: dict[str, Any],
    gallery: pd.DataFrame,
    gallery_values: np.ndarray,
    label_field: str,
    min_class_size: int = 5,
    pathway: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed: set[str] = set()
    counts: dict[str, set[str]] = defaultdict(set)
    for _, row in gallery.iterrows():
        for value in split_labels(row[label_field]):
            counts[value].add(str(row.compound))
    for value, compounds in counts.items():
        if len(compounds) >= min_class_size:
            allowed.add(value)
    gallery_sets = [split_labels(x) & allowed for x in gallery[label_field]]
    g_by_dose = {str(dose): group.index.to_numpy(dtype=np.int64) for dose, group in gallery.groupby("dose", sort=False)}
    # Centering/norming a dose-matched gallery is independent of the query and
    # method.  Cache only that part: the query row remains a one-row PCC call,
    # so its arithmetic and the old stable/tie ordering are unchanged.
    gallery_pcc_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for dose, dose_indices in g_by_dose.items():
        gallery_block = np.asarray(gallery_values[dose_indices], dtype=np.float64)
        centered = gallery_block - gallery_block.mean(axis=1, keepdims=True)
        gallery_pcc_cache[dose] = (dose_indices, centered, np.linalg.norm(centered, axis=1))
    gallery_id_values = gallery["gallery_id"].to_numpy()
    pathway_rank_cache: dict[frozenset[str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    output: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    for _, qrow in query.iterrows():
        known = split_labels(qrow[label_field]) & allowed
        if endpoint == "reactome_target_neighbour_map":
            if pathway is None:
                raise BlockedEndpoint("pathway_state_missing")
            known_pathways = pathway["known_by_query"].get(str(qrow.query_id), set())
            known = known_pathways
        if not known:
            continue
        gi = g_by_dose.get(str(qrow.dose), np.empty(0, dtype=np.int64))
        if endpoint == "reactome_target_neighbour_map" and len(gi) < pathway["primary_k"]:
            continue
        if endpoint != "reactome_target_neighbour_map" and len(gi) == 0:
            continue
        cached_gallery = gallery_pcc_cache[str(qrow.dose)]
        qi = int(qrow.profile_index)
        for method, profiles in (("Raw", bundle["raw"]), ("CFRA", bundle["cfra"])):
            scores = pcc_matrix_with_cached_gallery(profiles[qi : qi + 1], cached_gallery[1], cached_gallery[2])[0]
            if endpoint == "reactome_target_neighbour_map":
                # Path-A's locked implementation uses gallery_id as the final
                # tie-break for pathway enrichment ranks.
                order_local = np.asarray(sorted(range(len(scores)), key=lambda j: (-float(scores[j]), str(gallery_id_values[int(gi[j])]))), dtype=np.int64)
            else:
                # MoA/target retrieval inherits the old stable mergesort over
                # the deterministic dose-sorted gallery; no new tie rule is
                # introduced for the CFRA arm.
                order_local = np.argsort(-scores, kind="mergesort")
            ranked = gi[order_local]
            if endpoint == "reactome_target_neighbour_map":
                selected = ranked[: pathway["primary_k"]]
                genes: set[str] = set()
                for j in selected:
                    genes |= pathway["target_by_gallery"].get(str(gallery_id_values[int(j)]), set())
                gene_key = frozenset(genes)
                cached_pathway = pathway_rank_cache.get(gene_key)
                if cached_pathway is None:
                    pvals, qvals, _ = pathway_enrichment(set(gene_key), pathway)
                    path_order = np.asarray(sorted(range(len(pathway["names"])), key=lambda j: (float(qvals[j]), float(pvals[j]), str(pathway["names"][j]))), dtype=np.int64)
                    cached_pathway = (pvals, qvals, path_order)
                    pathway_rank_cache[gene_key] = cached_pathway
                pvals, qvals, path_order = cached_pathway
                metric = average_precision(np.asarray([pathway["names"][j] in known for j in path_order], dtype=bool))
            else:
                relevant = np.asarray([bool(gallery_sets[int(j)] & known) for j in ranked], dtype=bool)
                metric = average_precision(relevant)
            if np.isfinite(metric):
                output.append({"endpoint": endpoint, "method": method, "metric": "mAP", "seed": int(qrow.seed), "query_id": str(qrow.query_id), "compound": str(qrow.compound), "dose": str(qrow.dose), "support_row": int(qrow.support_row), "rotation": int(qrow.rotation), "value": float(metric)})
                per_query.append(output[-1].copy())
    if not output:
        raise BlockedEndpoint(f"{endpoint}:no_eligible_query_rows")
    frame = pd.DataFrame(output)
    arms = frame.groupby("method", as_index=False).agg(point=("value", "mean"), query_count=("query_id", "nunique"), compound_count=("compound", "nunique"))
    arms.insert(0, "endpoint", endpoint); arms["metric"] = "mAP"
    return arms, pd.DataFrame(per_query)


def pathway_enrichment(genes: set[str], state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hypergeom is None:
        raise BlockedEndpoint("scipy_hypergeom_unavailable")
    n = len(genes); p = np.ones(len(state["names"]), dtype=np.float64); hits = np.zeros(len(state["names"]), dtype=np.int32)
    if n == 0 or state["background_size"] == 0:
        return p, p.copy(), hits
    for gene in genes:
        ix = state["gene_to_indices"].get(gene)
        if ix is not None:
            hits[ix] += 1
    positive = hits > 0
    p[positive] = hypergeom.sf(hits[positive] - 1, state["background_size"], state["pathway_sizes"][positive], n)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1, dtype=np.float64)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(p); q[order] = np.clip(ranked, 0.0, 1.0)
    return p, q, hits


def load_pathway_state(gmt_path: Path, annotations: dict[str, dict[str, set[str]]], gallery: pd.DataFrame, query: pd.DataFrame) -> dict[str, Any]:
    if not gmt_path.is_file():
        raise BlockedEndpoint(f"missing_reactome_gmt={gmt_path}")
    pathways: dict[str, set[str]] = {}
    by_gene: dict[str, set[str]] = defaultdict(set)
    for line in gmt_path.read_text(encoding="utf-8").splitlines():
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3:
            continue
        name = fields[0].strip(); genes = {x.strip().upper() for x in fields[2:] if x.strip()}
        if not name or not genes:
            continue
        pathways[name] = genes
        for gene in genes:
            by_gene[gene].add(name)
    if not pathways:
        raise BlockedEndpoint("reactome_gmt_empty")
    target_by_gallery: dict[str, set[str]] = {}
    all_genes: set[str] = set()
    for _, row in gallery.iterrows():
        genes = {x.upper() for x in annotations.get(str(row.compound), {"target": set()})["target"]}
        genes &= set(by_gene)
        target_by_gallery[str(row.gallery_id)] = genes
        all_genes |= genes
    if not all_genes:
        raise BlockedEndpoint("no_gallery_curated_targets_map_to_reactome")
    background = all_genes
    names = sorted(pathways)
    sizes = np.asarray([len(pathways[name] & background) for name in names], dtype=np.int64)
    gene_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, name in enumerate(names):
        for gene in pathways[name] & background:
            gene_to_indices[gene].append(i)
    known: dict[str, set[str]] = {}
    for _, row in query.iterrows():
        genes = {x.upper() for x in annotations.get(str(row.compound), {"target": set()})["target"]} & set(by_gene)
        known[str(row.query_id)] = set().union(*(by_gene[g] for g in genes)) if genes else set()
    if not any(known.values()):
        raise BlockedEndpoint("no_query_curated_targets_map_to_reactome")
    return {"names": names, "pathway_sizes": sizes, "gene_to_indices": {g: np.asarray(v, dtype=np.int64) for g, v in gene_to_indices.items()}, "background_size": len(background), "target_by_gallery": target_by_gallery, "known_by_query": known, "primary_k": 25, "gmt_sha256": sha256(gmt_path)}


def average_precision_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    """The frozen hit-recovery AP definition (stable descending score sort)."""
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s) & np.isfinite(y)
    y, s = y[finite], s[finite]
    positives = int(y.sum())
    if len(y) == 0 or positives == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    cumulative = np.cumsum(ys, dtype=np.float64)
    precision = cumulative / np.arange(1, len(ys) + 1, dtype=np.float64)
    return float(np.sum(precision[ys == 1]) / positives)


def _bootstrap_mean(values: np.ndarray, rounds: int, seed: int, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise BlockedEndpoint(f"{label}:no_finite_compound_values")
    rng = np.random.default_rng(stable_seed(seed, label))
    sims = np.empty(int(rounds), dtype=np.float64)
    chunk = 500
    for start in range(0, int(rounds), chunk):
        stop = min(int(rounds), start + chunk)
        draws = rng.integers(0, len(values), size=(stop - start, len(values)))
        sims[start:stop] = values[draws].mean(axis=1)
    return sims


def _activity_seed_bootstrap(frame: pd.DataFrame, seed: int, rounds: int) -> tuple[float, np.ndarray, int, int]:
    subset = frame[frame.seed.eq(int(seed))].copy()
    raw = subset[subset.method.eq("Raw")][["query_id", "compound", "label", "value"]].rename(columns={"value": "raw"})
    cfra = subset[subset.method.eq("CFRA")][["query_id", "compound", "label", "value"]].rename(columns={"value": "cfra", "label": "label_cfra"})
    merged = raw.merge(cfra, on=["query_id", "compound"], how="inner")
    if merged.empty or len(merged) != len(raw) or len(merged) != len(cfra):
        raise BlockedEndpoint(f"confirmed_activity_ap:seed{seed}:paired_profile_key_missing")
    if not np.array_equal(merged.label.to_numpy(dtype=int), merged.label_cfra.to_numpy(dtype=int)):
        raise BlockedEndpoint(f"confirmed_activity_ap:seed{seed}:label_pair_mismatch")
    merged = merged[np.isfinite(merged.raw) & np.isfinite(merged.cfra)].copy()
    if merged.empty:
        raise BlockedEndpoint(f"confirmed_activity_ap:seed{seed}:no_finite_pairs")
    raw_point = average_precision_scores(merged.label.to_numpy(dtype=int), merged.raw.to_numpy(dtype=float))
    cfra_point = average_precision_scores(merged.label.to_numpy(dtype=int), merged.cfra.to_numpy(dtype=float))
    if not np.isfinite(raw_point) or not np.isfinite(cfra_point):
        raise BlockedEndpoint(f"confirmed_activity_ap:seed{seed}:no_positive_labels")
    compounds = np.asarray(sorted(merged.compound.astype(str).unique()), dtype=str)
    groups = {compound: np.flatnonzero(merged.compound.astype(str).to_numpy() == compound) for compound in compounds}
    rng = np.random.default_rng(stable_seed(int(seed), "bootstrap|all"))
    sims = np.empty(int(rounds), dtype=np.float64)
    for iteration in range(int(rounds)):
        sampled = rng.integers(0, len(compounds), size=len(compounds))
        indices = np.concatenate([groups[compounds[int(position)]] for position in sampled])
        labels = merged.label.to_numpy(dtype=int)[indices]
        raw_ap = average_precision_scores(labels, merged.raw.to_numpy(dtype=float)[indices])
        cfra_ap = average_precision_scores(labels, merged.cfra.to_numpy(dtype=float)[indices])
        sims[iteration] = cfra_ap - raw_ap if np.isfinite(raw_ap) and np.isfinite(cfra_ap) else np.nan
    return float(cfra_point - raw_point), sims, int(len(compounds)), int(len(merged))


def _retrieval_seed_differences(frame: pd.DataFrame, endpoint: str, seed: int) -> tuple[pd.Series, int]:
    subset = frame[(frame.endpoint.eq(endpoint)) & (frame.seed.eq(int(seed)))].copy()
    if subset.empty or set(subset.method.astype(str)) != set(METHODS):
        raise BlockedEndpoint(f"{endpoint}:seed{seed}:paired_methods_missing")
    index_cols = ["query_id", "compound", "dose", "support_row", "rotation"]
    wide = subset.pivot_table(index=index_cols, columns="method", values="value", aggfunc="first").reset_index()
    if not {"Raw", "CFRA"}.issubset(wide.columns):
        raise BlockedEndpoint(f"{endpoint}:seed{seed}:paired_profile_key_missing")
    wide = wide[np.isfinite(wide["Raw"]) & np.isfinite(wide["CFRA"])].copy()
    if wide.empty:
        raise BlockedEndpoint(f"{endpoint}:seed{seed}:no_finite_pairs")
    wide["difference"] = wide["CFRA"] - wide["Raw"]
    return wide.groupby("compound", sort=True)["difference"].mean(), int(len(wide))


def build_arm_metrics(per_query: pd.DataFrame, endpoint: str) -> list[dict[str, Any]]:
    frame = per_query[per_query.endpoint.eq(endpoint)].copy()
    if frame.empty:
        raise BlockedEndpoint(f"{endpoint}:no_per_query_rows")
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        seed_points: list[float] = []
        seed_queries: list[int] = []
        seed_compounds: list[int] = []
        for seed in EXPECTED_SEEDS:
            subset = frame[(frame.seed == int(seed)) & (frame.method == method)].copy()
            if subset.empty:
                raise BlockedEndpoint(f"{endpoint}:seed{seed}:{method}:missing_arm")
            values = subset[np.isfinite(subset.value.to_numpy(dtype=float))].copy()
            if endpoint == "confirmed_activity_ap":
                point = average_precision_scores(values.label.to_numpy(dtype=int), values.value.to_numpy(dtype=float))
            else:
                point = float(values.value.mean()) if not values.empty else float("nan")
            if not np.isfinite(point):
                raise BlockedEndpoint(f"{endpoint}:seed{seed}:{method}:nonfinite_arm_metric")
            seed_points.append(float(point)); seed_queries.append(int(len(values))); seed_compounds.append(int(values.compound.nunique()))
        rows.append({"endpoint": endpoint, "method": method, "metric": "AP" if endpoint == "confirmed_activity_ap" else "mAP", "point": float(np.mean(seed_points)), "query_count": int(min(seed_queries)), "compound_count": int(min(seed_compounds)), "seed_aggregation": "mean_seeds", "status": "estimable"})
    return rows


def bootstrap_contrast(per_query: pd.DataFrame, endpoint: str, rounds: int, seed: int) -> dict[str, Any]:
    frame = per_query[per_query.endpoint.eq(endpoint)].copy()
    if frame.empty or set(frame.method.astype(str)) != set(METHODS):
        raise BlockedEndpoint(f"{endpoint}:paired_methods_missing")
    metric = "AP" if endpoint == "confirmed_activity_ap" else "mAP"
    per_seed: list[tuple[float, np.ndarray, int, int]] = []
    seed_points: dict[str, float] = {}
    seed_compounds: dict[str, int] = {}
    seed_queries: dict[str, int] = {}
    for current_seed in EXPECTED_SEEDS:
        if endpoint == "confirmed_activity_ap":
            result = _activity_seed_bootstrap(frame, int(current_seed), int(rounds))
        else:
            differences, n_queries = _retrieval_seed_differences(frame, endpoint, int(current_seed))
            values = differences.to_numpy(dtype=np.float64)
            sims = _bootstrap_mean(values, int(rounds), int(current_seed), f"{endpoint}|compound")
            result = (float(values.mean()), sims, int(len(values)), int(n_queries))
        point, sims, n_compounds, n_queries = result
        per_seed.append(result)
        seed_points[str(current_seed)] = float(point); seed_compounds[str(current_seed)] = int(n_compounds); seed_queries[str(current_seed)] = int(n_queries)
    mean_boot = np.nanmean(np.vstack([item[1] for item in per_seed]), axis=0)
    mean_boot = mean_boot[np.isfinite(mean_boot)]
    if not len(mean_boot):
        raise BlockedEndpoint(f"{endpoint}:no_finite_bootstrap_rounds")
    return {"endpoint": endpoint, "metric": metric, "comparison": "CFRA - Raw", "point": float(np.mean([item[0] for item in per_seed])), "ci_low": float(np.quantile(mean_boot, 0.025)), "ci_high": float(np.quantile(mean_boot, 0.975)), "rounds": int(rounds), "compound_count": int(min(item[2] for item in per_seed)), "query_count": int(min(item[3] for item in per_seed)) if endpoint == "confirmed_activity_ap" else int(sum(item[2] for item in per_seed)), "bootstrap_unit": "compound", "seed_aggregation": "mean_seeds", "per_seed_point": seed_points, "per_seed_compound_count": seed_compounds, "per_seed_query_count": seed_queries}


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True, help="CFRA_PER_PHYSICAL_VIEW.npz; only seed*_raw/seed*_cfra keys are read")
    parser.add_argument("--manifest", type=Path, required=True, help="CFRA_PER_PHYSICAL_VIEW_MANIFEST.csv with physical support/reference identity")
    parser.add_argument("--data", type=Path, required=True, help="Frozen cp_plate_rows.npz (delta + baseline, 242 features)")
    parser.add_argument("--cp-manifest", type=Path, required=True, help="Frozen CP physical-row manifest")
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="Curated Broad MoA/target sidecar (primary + alternative labels)")
    parser.add_argument("--reactome-gmt", type=Path, required=True, help="Frozen Reactome GMT used only by Pathway retrieval")
    parser.add_argument("--query-manifest", type=Path, default=None, help="Historical Panel-D query identity manifest; retrieval endpoints BLOCKED when absent")
    parser.add_argument("--activity-labels", type=Path, default=None, help="Label-only ELIGIBLE_SAMPLES.csv; common test rows are filtered, score columns are ignored")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--min-gallery-repeats", type=int, default=3)
    parser.add_argument("--min-class-size", type=int, default=5)
    parser.add_argument("--bootstrap-rounds", type=int, default=DEFAULT_BOOTSTRAP_ROUNDS)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_outputs(outdir: Path, metrics: list[dict[str, Any]], contrasts: list[dict[str, Any]], per_query: pd.DataFrame, audit: dict[str, Any], blocked: dict[str, str]) -> None:
    pd.DataFrame(metrics, columns=["endpoint", "method", "metric", "point", "query_count", "compound_count", "seed_aggregation", "status"]).to_csv(outdir / "PANEL_D_METRICS.csv", index=False)
    pd.DataFrame(contrasts, columns=["endpoint", "metric", "comparison", "point", "ci_low", "ci_high", "rounds", "compound_count", "query_count", "bootstrap_unit", "seed_aggregation"]).to_csv(outdir / "PANEL_D_CONTRASTS.csv", index=False)
    legacy_tasks = {"confirmed_activity_ap": "Hit recovery", "moa_map": "MoA retrieval", "target_map": "Target retrieval", "reactome_target_neighbour_map": "Pathway retrieval"}
    legacy_rows = []
    for row in contrasts:
        legacy_rows.append({"task": legacy_tasks[row["endpoint"]], "metric": row["metric"], "comparison": row["comparison"], "point": row["point"], "ci_low": row["ci_low"], "ci_high": row["ci_high"], "n_compounds": row["compound_count"], "interpretation": "Uncertain gain" if row["endpoint"] == "confirmed_activity_ap" else "No clear gain"})
    pd.DataFrame(legacy_rows, columns=["task", "metric", "comparison", "point", "ci_low", "ci_high", "n_compounds", "interpretation"]).to_csv(outdir / "figure4_D_downstream_boundaries.csv", index=False)
    per_query.to_csv(outdir / "PANEL_D_PER_QUERY.csv", index=False)
    pd.DataFrame([{"endpoint": endpoint, "status": "BLOCKED", "reason": blocked[endpoint]} for endpoint in PANEL_ENDPOINTS if endpoint in blocked], columns=["endpoint", "status", "reason"]).to_csv(outdir / "BLOCKED.csv", index=False)
    (outdir / "AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True, default=str), encoding="utf-8")
    lines = ["# Figure 4 Panel D CFRA adapter", "", f"version: `{VERSION}`", f"status: `{STATUS}`", "", "| endpoint | status |", "|---|---|"]
    for endpoint in PANEL_ENDPOINTS:
        status = "BLOCKED" if endpoint in blocked else "estimable"
        lines.append(f"| {endpoint} | {status} |")
    lines += ["", "Endpoint contract: `confirmed_activity_ap` = label-only confirmed-hit AP; `moa_map`/`target_map` = dose-matched curated-label mAP; `reactome_target_neighbour_map` = Reactome pathway mAP after top-25 neighbour target enrichment and per-query BH-FDR.", "", "Retrieval endpoints require the frozen Panel-D query identity manifest; its absence is BLOCKED rather than a bundle-wide cohort substitution.", "", "The adapter reads only Raw and CFRA arrays from the physical-support bundle. Missing labels/GMT or invalid physical identity stop the affected endpoint; no IMR/GE fallback is permitted."]
    (outdir / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = cli()
    if args.bootstrap_rounds < 1 or args.min_gallery_repeats < 1 or args.min_class_size < 1:
        raise ValueError("bootstrap/repeat/class thresholds must be positive")
    ensure_output_dir(args.outdir, args.force)
    audit: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "bootstrap_unit": "compound",
        "bootstrap_rounds": int(args.bootstrap_rounds),
        "loaded_arrays": ["raw", "cfra"],
        "forbidden_legacy_sources": list(FORBIDDEN_TOKENS),
        "input_schema": {
            "cp_artifact": ["compound_id", "dose", "plate", "well", "split", "delta", "baseline"],
            "cp_manifest": ["compound_id", "dose", "plate", "well", "split"],
            "cfra_manifest": ["seed", "compound", "dose", "rotation", "support_row", "support_plate", "support_well", "reference_rows", "profile_index"],
            "cfra_npz": ["seed{3407,42,2025}_raw", "seed{3407,42,2025}_cfra"],
            "activity_labels": ["compound", "dose", "support_row", "confirmed_active", "seed", "split", "role", "evaluation_common_all_methods"],
            "retrieval_query_manifest": ["seed", "compound_id", "dose", "support_row"],
            "curated_annotations": ["Metadata_broad_id", "Metadata_moa", "Metadata_target", "Metadata_alternative_moa", "Metadata_alternative_target"],
            "reactome_gmt": "name<TAB>description<TAB>gene...",
        },
        "output_schema": {
            "metrics": ["endpoint", "method", "metric", "point", "query_count", "compound_count", "seed_aggregation", "status"],
            "contrasts": ["endpoint", "metric", "comparison", "point", "ci_low", "ci_high", "rounds", "compound_count", "query_count", "bootstrap_unit", "seed_aggregation"],
            "figure4_D_downstream_boundaries.csv": ["task", "metric", "comparison", "point", "ci_low", "ci_high", "n_compounds", "interpretation"],
        },
        "inputs": {},
        "physical_audit": {},
        "endpoint_audit": {},
    }
    blocked: dict[str, str] = {}
    metrics: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    per_query_parts: list[pd.DataFrame] = []
    try:
        artifact = read_artifact(args.data, args.cp_manifest, args.split_lock)
        bundle = load_bundle(args.profiles, args.manifest)
        audit["inputs"] = {str(path): sha256(path) for path in (args.data, args.cp_manifest, args.split_lock, bundle["manifest_path"], bundle["profiles_path"]) if path.is_file()}
        if bundle["audit_path"] is not None:
            audit["inputs"][str(bundle["audit_path"])] = sha256(bundle["audit_path"])
        audit["bundle_audit_metadata"] = bundle["audit"]
        audit["physical_audit"] = validate_bundle_physical(bundle, artifact)
        annotations = load_annotations(args.annotations)
        audit["inputs"][str(args.annotations)] = sha256(args.annotations)
        query = query_frame(bundle, artifact, annotations)
        retrieval_query_reason: str | None = None
        try:
            retrieval_keys, retrieval_audit = load_query_manifest(args.query_manifest, bundle)
            query_keys = list(zip(query.seed.astype(int), query.compound.astype(str), query.dose.astype(str), query.support_row.astype(int)))
            query = query[np.asarray([key in retrieval_keys for key in query_keys], dtype=bool)].reset_index(drop=True)
            if query.empty:
                raise BlockedEndpoint("retrieval_query_manifest_bundle_intersection_empty")
            audit["inputs"][str(args.query_manifest)] = sha256(args.query_manifest)
            audit["query"] = {**retrieval_audit, "bundle_rows": int(len(bundle["manifest"])), "intersection_rows": int(len(query))}
        except BlockedEndpoint as exc:
            retrieval_query_reason = str(exc)
            audit["query"] = {"status": "BLOCKED", "reason": retrieval_query_reason, "bundle_rows": int(len(bundle["manifest"]))}
        gallery, gallery_values = build_gallery(artifact, annotations, args.min_gallery_repeats)
        audit["gallery"] = {"conditions": int(len(gallery)), "compounds": int(gallery.compound.nunique()), "min_repeats": int(args.min_gallery_repeats)}
        # Confirmed activity is independent of the profile estimators.  It is
        # the only endpoint whose labels cannot be reconstructed from curated
        # MoA/target annotations, so a missing sidecar blocks it explicitly.
        try:
            activity_labels = load_activity_labels(args.activity_labels, artifact, bundle)
            arm, activity_rows = activity_scores(artifact, bundle, activity_labels)
            activity_rows["value"] = activity_rows["cfra_metric"]
            activity_rows.loc[activity_rows["endpoint"].eq("confirmed_activity_ap"), "method"] = "CFRA"
            raw_rows = activity_rows.copy(); raw_rows["method"] = "Raw"; raw_rows["value"] = raw_rows["raw_metric"]
            activity_rows = pd.concat([raw_rows, activity_rows], ignore_index=True)
            per_query_parts.append(activity_rows[["endpoint", "method", "seed", "query_id", "compound", "dose", "support_row", "rotation", "label", "value"]])
            audit["endpoint_audit"]["confirmed_activity_ap"] = {"label_source": str(args.activity_labels), "label_rows": int(len(activity_labels)), "bundle_rows": int(len(bundle["manifest"])), "intersection_rows": int(len(activity_rows) // 2)}
        except BlockedEndpoint as exc:
            blocked["confirmed_activity_ap"] = str(exc)
        for endpoint, field in (("moa_map", "moa"), ("target_map", "target")):
            if retrieval_query_reason is not None:
                blocked[endpoint] = retrieval_query_reason
                continue
            try:
                arm, rows = retrieval_rows(endpoint, query, bundle, gallery, gallery_values, field, args.min_class_size)
                per_query_parts.append(rows[["endpoint", "method", "seed", "query_id", "compound", "dose", "support_row", "rotation", "value"]])
                audit["endpoint_audit"][endpoint] = {"label_field": field, "eligible_query_rows": int(rows.query_id.nunique()), "eligible_compounds": int(rows.compound.nunique()), "gallery_classes": int(sum(len(split_labels(x)) > 0 for x in gallery[field]))}
            except BlockedEndpoint as exc:
                blocked[endpoint] = str(exc)
        if retrieval_query_reason is not None:
            blocked["reactome_target_neighbour_map"] = retrieval_query_reason
        else:
            try:
                pathway_state = load_pathway_state(args.reactome_gmt, annotations, gallery, query)
                arm, rows = retrieval_rows("reactome_target_neighbour_map", query, bundle, gallery, gallery_values, "target", args.min_class_size, pathway_state)
                per_query_parts.append(rows[["endpoint", "method", "seed", "query_id", "compound", "dose", "support_row", "rotation", "value"]])
                audit["endpoint_audit"]["reactome_target_neighbour_map"] = {"gmt_sha256": pathway_state["gmt_sha256"], "pathway_count": len(pathway_state["names"]), "gallery_background_target_genes": int(pathway_state["background_size"]), "primary_k": int(pathway_state["primary_k"])}
            except BlockedEndpoint as exc:
                blocked["reactome_target_neighbour_map"] = str(exc)
    except BlockedEndpoint as exc:
        reason = str(exc)
        for endpoint in PANEL_ENDPOINTS:
            blocked[endpoint] = reason
        audit["global_block"] = reason
    per_query = pd.concat(per_query_parts, ignore_index=True) if per_query_parts else pd.DataFrame(columns=["endpoint", "method", "seed", "query_id", "compound", "dose", "support_row", "rotation", "value"])
    for endpoint in PANEL_ENDPOINTS:
        if endpoint in blocked:
            continue
        try:
            metrics.extend(build_arm_metrics(per_query, endpoint))
            contrasts.append(bootstrap_contrast(per_query, endpoint, args.bootstrap_rounds, args.bootstrap_seed))
        except BlockedEndpoint as exc:
            blocked[endpoint] = str(exc)
    for endpoint in PANEL_ENDPOINTS:
        if endpoint in blocked:
            # Remove any partial arm rows to prevent a partial result being
            # mistaken for a valid paired endpoint.
            metrics = [row for row in metrics if row.get("endpoint") != endpoint]
    audit["blocked"] = blocked
    audit["contrasts"] = contrasts
    audit["endpoint_status"] = {endpoint: ("BLOCKED" if endpoint in blocked else "estimable") for endpoint in PANEL_ENDPOINTS}
    write_outputs(args.outdir, metrics, contrasts, per_query, audit, blocked)
    print(json.dumps({"outdir": str(args.outdir), "endpoint_status": audit["endpoint_status"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
