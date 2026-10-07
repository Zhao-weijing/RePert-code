#!/usr/bin/env python3
"""Complete sci-Plex3 MLP-teacher matrix under the BBBC047 P0 contract.

The matrix is deliberately kept in one experiment root and has an explicit
two-stage barrier:

``fit_freeze``
    Fits one independent CFRA MLP teacher for every source cell and OOF fold,
    constructs balanced source-cell Student targets, selects Student
    checkpoints on the frozen P0 validation rule, and writes nine
    ``FIT_FREEZE_COMPLETE.json`` records.  The two unseen surfaces are run
    first and their target-cell Confirmation profiles never enter any model
    input.  A cross-cell-seen root has a root-specific target holdout, but its
    source-all183 physical rows may be reused by another root; this is not a
    global-blind claim.

``confirmation``
    Verifies every one of the nine freeze records and all recorded hashes,
    then opens target-cell treated profiles and emits per-cell-line,
    per-surface M0/M1/M2 values and paired contrasts.  The all-nine gate means
    every model refit is locked before evaluation starts; it does not claim
    that all cross-cell-seen target physical rows were globally unread during
    all preceding source roots.  No selection or refitting is performed in
    this stage.

The implementation uses two audited, explicitly hashed helper modules:
``run_p0.py`` supplies the sci-Plex3 reader and Student implementation, while
``run_formal_teacher.py`` supplies the MLP-teacher training primitives.  The
multi-cell adapter in this file is intentionally explicit: source-cell rows
are averaged within each ``(drug,dose)`` unit before Student fitting, and each
source cell keeps its own teacher normalizer, shrinkage and checkpoint.

The current ``evaluation-only-report-hotfix-r2-2026-09-16`` changes only the
final report index: wide summary rows are keyed by ``method`` and long
contrast rows by ``metric``.  It does not alter model loading, prediction,
metrics, bootstrap, selection or refit, and can recover a completed freeze
root whose prior report stopped at that output-only error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
P0_PATH = (Path(__file__).resolve().parents[3] / "analysis/predictor_supervision/sci_plex3/support/run_p0.py")
TEACHER_PATH = (Path(__file__).resolve().parents[3] / "analysis/predictor_supervision/sci_plex3/support/run_formal_teacher.py")
V3_DIR = HERE.parent / "sciplex3_rawagg_cfra_p0_20260915" / "formal_surface_isolated_v3"

CELL_LINES = ("A549", "K562", "MCF7")
SURFACES = ("own_cell_unseen", "cross_cell_unseen", "cross_cell_seen")
METHODS = ("M0_RawAggregate", "M1_CFRA1Aggregate", "M2_CFRA1Residual")
METHOD_KINDS = {
    "M0_RawAggregate": ("raw", "raw"),
    "M1_CFRA1Aggregate": ("cfra", "cfra"),
    "M2_CFRA1Residual": ("cfra", "residual"),
}
SEEDS = (3407, 42, 2025, 1337, 7331)
CAPACITY = "H512Z128"
CAPACITY_DIMS = (512, 128)
BOOTSTRAP_ROUNDS = 10_000
VERSION = "sciplex3-cfra-mlpteacher-complete-matrix-v2-2026-09-16"
# Report-only recovery hotfix.  The frozen v2 model artifacts remain the
# authority; this compatibility hash is the runner used to create the
# existing v5 freeze root.  It is accepted only by confirmation, after a
# complete FIT_FREEZE marker is present, and never by fit_freeze itself.
REPORT_ONLY_HOTFIX_VERSION = "evaluation-only-report-hotfix-r2-2026-09-16"
REPORT_ONLY_BASE_RUNNER_SHA256 = "2ade06c30d16531edcf529331c2afc4d0f98cd6022f66c8e70482163068abdd0"
STUDENT_PARAMETER_COUNT = 2_397_264
TEACHER_REGIME = "fit-compound-OOF"
REPEAT_STATUS = "DESCRIPTIVE_ACTUAL_TWO_REPEAT_CORRELATION_NONINDEPENDENT"
BOOTSTRAP_METRICS = ("A", "z_same_F", "z_foreign_F", "E", "held_pcc_A", "held_pcc_B", "delta_pcc", "full_target_pcc", "raw_target_mse")

DEFAULT_CONDITIONS = Path(
    "/path/to/data/AIDD/MVCPert_5_27/analysis/external_validation/sci_plex3/"
    "sciplex3_validation/eligible_conditions_n50.csv"
)
DEFAULT_GROUP_COUNTS = Path(
    "/path/to/data/AIDD/MVCPert_5_27/analysis/external_validation/sci_plex3/"
    "sciplex3_validation/group_counts.h5"
)
DEFAULT_SPLIT_LOCK = Path(
    "/path/to/data/AIDD/MVCPert_5_27/analysis/external_validation/sci_plex3/"
    "sciplex3_validation/cfra_confirmation/split_lock.json"
)
DEFAULT_STRUCTURE_MAP = Path(
    "/path/to/data/AIDD/MVCPert_5_27/analysis/external_validation/sci_plex3/"
    "sciplex3_student_crosscell/input_preflight/DRUG_ECFP4_MAPPING.npz"
)


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if not P0_PATH.is_file() or not TEACHER_PATH.is_file():
    # Allows --help and py_compile on a checkout where the historical helper
    # trees are not present, but fails clearly before any experiment stage.
    p0 = None
    teacher = None
else:
    p0 = _import(P0_PATH, "sciplex3_complete_matrix_p0_helper")
    teacher = _import(TEACHER_PATH, "sciplex3_complete_matrix_teacher_helper")
    # Only the pre-registered, previously selected capacity is available to
    # this matrix.  This prevents post-hoc capacity selection by target or
    # surface and makes the capacity choice visible in every artifact.
    teacher.CAPACITIES = {CAPACITY: CAPACITY_DIMS}
    teacher.CAPACITY_ORDER = (CAPACITY,)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "fit_freeze", "confirmation"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v3-dir", type=Path, default=V3_DIR)
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--group-counts", type=Path, default=DEFAULT_GROUP_COUNTS)
    parser.add_argument("--split-lock", type=Path, default=DEFAULT_SPLIT_LOCK)
    parser.add_argument("--structure-map", type=Path, default=DEFAULT_STRUCTURE_MAP)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--targets", nargs="*", choices=CELL_LINES, default=list(CELL_LINES), help="Subset only for fit_freeze; confirmation still requires all nine cells.")
    parser.add_argument("--surfaces", nargs="*", choices=SURFACES, default=list(SURFACES), help="Subset only for fit_freeze; confirmation still requires all nine cells.")
    parser.add_argument("--allow-partial-confirmation", action="store_true", help="Explicit diagnostic escape hatch; never use for the primary matrix.")
    return parser.parse_args()


def require_helpers() -> None:
    if p0 is None or teacher is None:
        raise RuntimeError(f"missing helper modules: {P0_PATH} and/or {TEACHER_PATH}")


def verified_student_parameter_count() -> int:
    """Count the imported sci-Plex3 Student package, not a BBBC package."""
    require_helpers()
    count = int(sum(int(parameter.numel()) for parameter in p0.old.Student().parameters()))
    if count != STUDENT_PARAMETER_COUNT:
        raise RuntimeError(f"sci-Plex3 Student parameter-count mismatch: measured={count}, registered={STUDENT_PARAMETER_COUNT}")
    return count


def _input_manifest(args: argparse.Namespace) -> dict[str, Any]:
    require_helpers()
    paths = {
        "conditions": args.conditions,
        "group_counts": args.group_counts,
        "split_lock": args.split_lock,
        "structure_map": args.structure_map,
        "old_p0_runner": P0_PATH,
        "mlp_teacher_runner": TEACHER_PATH,
    }
    out: dict[str, Any] = {}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        out[key] = {"path": str(path), "sha256": sha256(path)}
    return out


def _load_v3(args: argparse.Namespace, surface: str) -> dict[str, Any]:
    path = args.v3_dir / surface / "PLAN.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen v3 plan for {surface}: {path}")
    plan = read_json(path)
    if plan.get("confirmation_treated_loaded") is not False:
        raise RuntimeError(f"v3 plan has confirmation_treated_loaded=true: {path}")
    if plan.get("version") != p0.VERSION:
        raise RuntimeError(f"unexpected v3 version for {surface}: {plan.get('version')}")
    input_paths = {"conditions": args.conditions, "group_counts": args.group_counts, "split_lock": args.split_lock, "structure_map": args.structure_map}
    for name, path0 in input_paths.items():
        expected = plan.get("inputs", {}).get(name, {}).get("sha256")
        if expected and expected != sha256(path0):
            raise RuntimeError(f"v3 {surface} {name} SHA mismatch")
    return plan


def _units_from_plans(plans: dict[str, dict[str, Any]]) -> list[tuple[str, float]]:
    first = plans[SURFACES[0]].get("units", [])
    units = [(str(item[0]), float(item[1])) for item in first]
    if len(units) != 613 or len(set(units)) != len(units):
        raise RuntimeError(f"expected 613 unique common units, got {len(units)}")
    for surface, plan in plans.items():
        other = [(str(item[0]), float(item[1])) for item in plan.get("units", [])]
        if other != units:
            raise RuntimeError(f"unit order differs between v3 surfaces: {surface}")
    return units


def _check_split_spec(target: str, surface: str, spec: dict[str, Any], units: list[tuple[str, float]]) -> dict[str, Any]:
    expected_sources = [target] if surface == "own_cell_unseen" else [cell for cell in CELL_LINES if cell != target]
    if spec.get("source_cell_lines") != expected_sources:
        raise RuntimeError(f"{target}/{surface}: source cell lines drift: {spec.get('source_cell_lines')}")
    names = ("train_drugs", "valid_drugs", "refit_drugs", "test_drugs", "foreign_drugs")
    sets: dict[str, set[str]] = {}
    unit_drugs = {drug for drug, _dose in units}
    for name in names:
        value = spec.get(name)
        if not isinstance(value, list) or (name != "foreign_drugs" and not value):
            raise RuntimeError(f"{target}/{surface}: invalid {name}")
        sets[name] = {str(x) for x in value}
        if not sets[name] <= unit_drugs:
            raise RuntimeError(f"{target}/{surface}: {name} contains drug outside common units")
    # cross_cell_seen is a deliberately different estimand: source-cell
    # Fit/Validation/refit drugs and target-cell Confirmation drugs share
    # compound identities.  The overlap is legal because the source and
    # target cell identities differ.  Do not apply the unseen Confirmation
    # overlap gate here; leakage is decided from the target treated-profile
    # read audit below.
    # Fit and Validation are always disjoint, including the seen-transfer
    # surface.  The seen exception applies only to target-cell Confirmation
    # identities, which intentionally overlap the source-cell drug universe.
    if sets["train_drugs"] & sets["valid_drugs"]:
        raise RuntimeError(f"{target}/{surface}: Fit/Validation overlap")
    if surface != "cross_cell_seen":
        if sets["train_drugs"] & sets["test_drugs"] or sets["valid_drugs"] & sets["test_drugs"]:
            raise RuntimeError(f"{target}/{surface}: unseen Fit/Validation/Confirmation overlap")
    if not sets["train_drugs"] <= sets["refit_drugs"] or not sets["valid_drugs"] <= sets["refit_drugs"]:
        raise RuntimeError(f"{target}/{surface}: refit universe does not include Fit∪Validation")
    if surface == "cross_cell_seen":
        common183 = unit_drugs
        if len(common183) != 183:
            raise RuntimeError(f"{target}/{surface}: expected a 183-drug common universe, got {len(common183)}")
        for name in ("refit_drugs", "test_drugs", "foreign_drugs"):
            if sets[name] != common183:
                raise RuntimeError(f"{target}/{surface}: {name} must equal the common 183-drug universe")
    else:
        if sets["refit_drugs"] & sets["test_drugs"]:
            raise RuntimeError(f"{target}/{surface}: unseen surface has refit/test overlap")
        if sets["refit_drugs"] != sets["train_drugs"] | sets["valid_drugs"]:
            raise RuntimeError(f"{target}/{surface}: unseen refit must equal Fit union Validation")
        if sets["foreign_drugs"] != sets["refit_drugs"]:
            raise RuntimeError(f"{target}/{surface}: unseen foreign pool must equal refit universe")
    if surface != "cross_cell_seen" and len(sets["test_drugs"]) != 36:
        raise RuntimeError(f"{target}/{surface}: unseen confirmation expected 36 drugs")
    if len(sets["train_drugs"]) != 111 or len(sets["valid_drugs"]) != 36:
        raise RuntimeError(f"{target}/{surface}: expected Fit=111 and Validation=36")
    if surface == "cross_cell_seen":
        return {
            "split_overlap_policy": "allowed_source_refit_target_test_compound_overlap",
            "allowed_overlap_pairs": ["Fit-Confirmation", "Validation-Confirmation", "Fit-refit", "Validation-refit", "refit-Confirmation"],
            "leakage_decision": "for this root, target-cell treated-profile read audit is authoritative; other roots may reuse the same physical rows as source, and compound identity overlap alone is not leakage",
        }
    return {
        "split_overlap_policy": "strict_fit_validation_confirmation_drug_disjoint",
        "allowed_overlap_pairs": [],
        "leakage_decision": "unseen target-cell treated profiles must not be read before freeze; Fit/Validation/Confirmation drug sets are disjoint",
    }


def _matrix_plan(args: argparse.Namespace, plans: dict[str, dict[str, Any]], units: list[tuple[str, float]], inputs: dict[str, Any], split_audits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    student_parameter_count = verified_student_parameter_count()
    cells: list[dict[str, Any]] = []
    for target in CELL_LINES:
        for surface in SURFACES:
            spec = plans[surface]["folds"][target][surface]
            cells.append({
                "target_cell_line": target,
                "surface": surface,
                "source_cell_lines": list(spec["source_cell_lines"]),
                "status": spec.get("status"),
                "n_train_drugs": len(spec["train_drugs"]),
                "n_validation_drugs": len(spec["valid_drugs"]),
                "n_refit_drugs": len(spec["refit_drugs"]),
                "n_confirmation_drugs": len(spec["test_drugs"]),
                "n_foreign_drugs": len(spec["foreign_drugs"]),
                "seen_drug_transfer": surface == "cross_cell_seen",
                "target_profile_not_loaded_until_confirmation": True,
                "target_profile_holdout_scope": "this target/surface root only",
                "root_specific_target_holdout": True,
                "unseen_target_confirmation_never_model_input": surface != "cross_cell_seen",
                "cross_root_source_reuse_allowed": surface == "cross_cell_seen",
                "cross_root_hyperparameter_or_selection_sharing": False,
                "split_overlap_policy": split_audits[f"{target}|{surface}"]["split_overlap_policy"],
                "allowed_overlap_pairs": "|".join(split_audits[f"{target}|{surface}"]["allowed_overlap_pairs"]),
                "leakage_decision_audit": split_audits[f"{target}|{surface}"]["leakage_decision"],
            })
    return {
        "version": VERSION,
        "stage": "preflight",
        "dataset": "sci-Plex3 Zenodo 7041849",
        "cell_lines": list(CELL_LINES),
        "surfaces": list(SURFACES),
        "fit_freeze_order": {"surfaces": list(SURFACES), "unseen_surfaces_must_pass_before_seen": True, "unseen_surfaces": ["own_cell_unseen", "cross_cell_unseen"], "seen_surface": "cross_cell_seen", "scope": "execution/marker order; not a claim that cross-seen source rows are globally unread"},
        "units": [[drug, dose] for drug, dose in units],
        "n_units": len(units),
        "matrix_cells": cells,
        "capacity": {"name": CAPACITY, "hidden_dim": CAPACITY_DIMS[0], "latent_dim": CAPACITY_DIMS[1], "globally_fixed": True, "reason": "pre-registered H512Z128 selected in prior legal K562 compound-equal teacher selection; no per-cell/surface capacity search"},
        "student_package": {"parameter_count": student_parameter_count, "verification": "sum(parameter.numel() for parameter in imported sci-Plex3 Student parameters)"},
        "teacher_regime": TEACHER_REGIME,
        "teacher_selection": {"objective": "for each source cell independently: reciprocal direction -> (drug,dose) unit mean -> compounds equally weighted; no source-cell averaging is performed before selecting that cell's teacher epoch/simplex", "cross_cell_student_aggregation": "after each source-cell teacher is frozen, source-cell CFRA/Raw rows are averaged equally for the Student target", "checkpoint_search": "Validation only", "normalizer": "source-cell Fit support rows only", "simplex": "nonnegative three-component weights summing to one, selected on the same per-source-cell compound-equal Validation loss", "refit": "Fit∪Validation minus held OOF fold; fixed locked epoch; no Validation search after refit"},
        "student_contract": {"methods": list(METHODS), "M0": "two independent Raw-Aggregate branches averaged", "M1": "two independent CFRA-1R-Aggregate branches averaged; diagnostic", "M2": "CFRA-1R-Aggregate base + Raw-minus-CFRA residual", "source_cell_balance": "one row per (drug,dose), average source-cell aggregates equally before Student fitting", "checkpoint_rule": "original frozen P0 row-wise Validation loss for matching target kind", "seeds": list(SEEDS)},
        "split_contract": {"fit_validation_confirmation_barrier": True, "cross_cell_seen_exception": "root-specific target holdout: source refit and target Confirmation share compound identities by design (183); another root may reuse those physical rows as source, so this is not a global-blind claim", "cross_cell_seen_selection_scope": "each root uses only its own 111/36 Fit/Validation selection; source all183 is used only in the fixed refit; no cross-root hyperparameter/checkpoint/selection sharing", "unseen": "own_cell_unseen and cross_cell_unseen run before cross_cell_seen; their target Confirmation profiles never enter any model input and their refit/test drugs are disjoint"},
        "evaluation_contract": {"common_raw_target": ["delta_pcc", "full_target_pcc", "raw_target_mse"], "two_repeat_diagnostics": ["A", "z_same_F", "z_foreign_F", "E", "held_pcc_A", "held_pcc_B"], "repeat_status": REPEAT_STATUS, "bootstrap": "10,000 paired compound bootstrap; doses and five seeds averaged within drug first", "primary_contrast": "M2_CFRA1Residual - M0_RawAggregate"},
        "confirmation_gate": {"requires_all_nine_fit_freeze": True, "requires_all_nine_model_refits_locked_before_evaluation": True, "confirmation_profiles_loaded_before_gate": False, "checkpoint_search_after_gate": False, "global_blind_claim": False, "meaning": "all nine model refits are locked before evaluation; does not claim all cross-cell-seen target physical rows were globally unread during preceding source roots"},
        "inputs": inputs,
        "v3_plans": {surface: {"path": str(args.v3_dir / surface / "PLAN.json"), "sha256": sha256(args.v3_dir / surface / "PLAN.json"), "version": plans[surface].get("version")} for surface in SURFACES},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }


def preflight(args: argparse.Namespace) -> None:
    require_helpers()
    if args.root.exists() and any(args.root.iterdir()):
        raise FileExistsError(f"output root is not empty: {args.root}")
    inputs = _input_manifest(args)
    plans = {surface: _load_v3(args, surface) for surface in SURFACES}
    units = _units_from_plans(plans)
    split_audits: dict[str, dict[str, Any]] = {}
    for surface, plan in plans.items():
        for target in CELL_LINES:
            split_audits[f"{target}|{surface}"] = _check_split_spec(target, surface, plan["folds"][target][surface], units)
    args.root.mkdir(parents=True, exist_ok=True)
    matrix = _matrix_plan(args, plans, units, inputs, split_audits)
    # The just-constructed matrix is the source of the frozen v3 SHA fields;
    # validate those fields before writing the root PLAN.  (There is no
    # pre-existing ``plan`` variable at this point.)
    for surface in SURFACES:
        current_path = args.v3_dir / surface / "PLAN.json"
        current_sha = sha256(current_path) if current_path.is_file() else None
        declared_sha = matrix.get("v3_plans", {}).get(surface, {}).get("sha256")
        if not declared_sha or current_sha != declared_sha:
            raise RuntimeError(f"frozen v3 plan SHA mismatch for {surface}: current={current_sha} declared={declared_sha}")
    write_json(args.root / "PLAN.json", matrix)
    manifest_rows: list[dict[str, Any]] = []
    for row in matrix["matrix_cells"]:
        manifest_rows.append({**row, "n_units": len(units), "physical_repeats": 2, "teacher_capacity": CAPACITY, "confirmation_treated_loaded": False, "test_loaded": False})
    write_csv(args.root / "TARGET_VIEW_MANIFEST.csv", manifest_rows)
    write_json(args.root / "PREPARE_COMPLETE.json", {"status": "PASS", "version": VERSION, "matrix_cells": len(manifest_rows), "capacity": CAPACITY, "confirmation_treated_loaded": False, "test_loaded": False, "all_nine_required_before_confirmation": True, "all_nine_required_before_evaluation": True, "unseen_freezes_must_pass_before_seen_start": True, "fit_freeze_order": list(SURFACES), "global_blind_claim": False})
    print(json.dumps({"status": "PREFLIGHT_COMPLETE", "matrix_cells": len(manifest_rows), "n_units": len(units), "capacity": CAPACITY, "confirmation_treated_loaded": False}, sort_keys=True), flush=True)


def load_matrix_plan(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[tuple[str, float]]]:
    require_helpers()
    verified_student_parameter_count()
    plan_path = args.root / "PLAN.json"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = read_json(plan_path)
    gate = plan.get("confirmation_gate", {})
    if plan.get("version") != VERSION or gate.get("confirmation_profiles_loaded_before_gate") is not False or gate.get("global_blind_claim") is not False or plan.get("fit_freeze_order", {}).get("surfaces") != list(SURFACES) or plan.get("fit_freeze_order", {}).get("unseen_surfaces_must_pass_before_seen") is not True:
        raise RuntimeError("matrix PLAN version/leakage check failed")
    runner_meta = plan.get("runner", {})
    current_runner_sha = sha256(Path(__file__).resolve())
    declared_runner_sha = runner_meta.get("sha256")
    if declared_runner_sha != current_runner_sha:
        # The first v5 Confirmation reached the model/evaluation stage and
        # failed only while constructing the report (a contrast row was
        # incorrectly indexed by ``method``).  Permit this exact report-only
        # recovery against its unchanged freeze root, but never let the
        # compatibility path enter fit_freeze or rewrite model artifacts.
        legacy_report_recovery = (
            args.stage == "confirmation"
            and declared_runner_sha == REPORT_ONLY_BASE_RUNNER_SHA256
            and (args.root / "FIT_FREEZE_COMPLETE.json").is_file()
        )
        if not legacy_report_recovery:
            raise RuntimeError("matrix runner hash drift")
    current_inputs = _input_manifest(args)
    if plan.get("inputs") != current_inputs:
        raise RuntimeError("matrix input hash drift")
    plans = {surface: _load_v3(args, surface) for surface in SURFACES}
    units = _units_from_plans(plans)
    if [(str(x[0]), float(x[1])) for x in plan.get("units", [])] != units:
        raise RuntimeError("matrix unit order drift")
    return plan, plans, units


def _spec(plans: dict[str, dict[str, Any]], target: str, surface: str) -> dict[str, Any]:
    return plans[surface]["folds"][target][surface]


def _common_hvg(store, rows: dict, source_cells: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    hvg, audit = p0.old.fit_vehicle_hvg(store, rows, source_cells)
    if np.asarray(hvg).shape != (p0.old.N_HVG,):
        raise RuntimeError(f"invalid common source HGV shape {np.asarray(hvg).shape}")
    return np.asarray(hvg, dtype=np.int64), audit


def _balanced_rows(profile: dict, source_cells: list[str], drugs: set[str], units: list[tuple[str, float]]) -> p0.old.StudentRows:
    """Equal-source-cell, equal-repeat Raw-Aggregate rows."""
    rows: list[tuple[str, str, float, np.ndarray, np.ndarray]] = []
    for drug, dose in units:
        if drug not in drugs:
            continue
        source_raw: list[np.ndarray] = []
        source_base: list[np.ndarray] = []
        for cell in source_cells:
            r1 = p0.old.response(profile, cell, drug, dose, "rep1")
            r2 = p0.old.response(profile, cell, drug, dose, "rep2")
            b1 = p0.old.baseline(profile, cell, drug, dose, "rep1")
            b2 = p0.old.baseline(profile, cell, drug, dose, "rep2")
            source_raw.append((r1 + r2) / 2.0)
            source_base.append((b1 + b2) / 2.0)
        raw = np.mean(np.stack(source_raw), axis=0).astype(np.float32)
        base = np.mean(np.stack(source_base), axis=0).astype(np.float32)
        rows.append(("source_cell_balanced", drug, float(dose), base, raw))
    if not rows:
        raise RuntimeError("empty balanced source rows")
    return p0.old.StudentRows(
        np.asarray([x[0] for x in rows], dtype=str),
        np.asarray([x[1] for x in rows], dtype=str),
        np.asarray([x[2] for x in rows], dtype=np.float32),
        np.vstack([x[3] for x in rows]).astype(np.float32),
        np.vstack([x[4] for x in rows]).astype(np.float32),
        np.vstack([x[4] for x in rows]).astype(np.float32),
    )


def _teacher_plan(root: Path, source_cell: str, target: str, surface: str, v3: dict[str, Any], units: list[tuple[str, float]], spec0: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Synthetic adapter PLAN consumed by the audited teacher functions.

    The helper teacher is intentionally called only after this adapter writes
    a complete, hashable plan.  Its teacher ``TARGET`` is the source cell,
    while the parent matrix keeps the actual target cell and surface.
    """
    split = {"train_drugs": sorted(str(x) for x in spec0["train_drugs"]), "valid_drugs": sorted(str(x) for x in spec0["valid_drugs"]), "refit_drugs": sorted(str(x) for x in spec0["refit_drugs"]), "test_drugs": [], "foreign_drugs": sorted(str(x) for x in spec0["foreign_drugs"])}
    inputs = _input_manifest(args)
    plan = {
        "version": teacher.VERSION,
        "stage": "preflight",
        "dataset": "sci-Plex3 Zenodo 7041849",
        "target": source_cell,
        "parent_target": target,
        "parent_surface": surface,
        "surface": surface,
        "units": [[drug, dose] for drug, dose in units],
        "n_units": len(units),
        "split": split,
        "capacities": {CAPACITY: {"hidden": CAPACITY_DIMS[0], "latent": CAPACITY_DIMS[1], "architecture": f"{p0.old.N_HVG}-{CAPACITY_DIMS[0]}-{CAPACITY_DIMS[1]}-{CAPACITY_DIMS[0]}-{p0.old.N_HVG}-GELU"}},
        "capacity_globally_fixed": True,
        "teacher_regime": TEACHER_REGIME,
        "confirmation_treated_loaded": False,
        "has_test_stage": False,
        "inputs": {**inputs, "runner": {"path": str(TEACHER_PATH), "sha256": sha256(TEACHER_PATH)}},
        "parent_v3_plan": {"path": str(args.v3_dir / surface / "PLAN.json"), "sha256": sha256(args.v3_dir / surface / "PLAN.json")},
    }
    write_json(root / "PLAN.json", plan)
    # The historical teacher selection stage hashes these two frozen OOF
    # manifests as input artifacts.  They are written here explicitly instead
    # of being silently synthesized inside the helper.
    assignment = p0.old.teacher_fold_assignment(set(split["train_drugs"]))
    write_csv(root / "OOF_MANIFEST.csv", [{"role": "selection", "fold": fold, "fit_drugs": len(set(split["train_drugs"]) - {d for d, k in assignment.items() if k == fold}), "validation_drugs": len(split["valid_drugs"]), "held_drugs": sum(k == fold for k in assignment.values()), "fit_validation_overlap": 0, "fit_held_overlap": 0, "confirmation_in_fit": 0, "confirmation_in_validation": 0} for fold in range(5)])
    write_csv(root / "OOF_ASSIGNMENT_RELATION.csv", [{"drug": drug, "pool": "Fit", "selection_fold": int(fold), "refit_fold": int(p0.old.teacher_fold_assignment(set(split["refit_drugs"]))[drug]) if drug in p0.old.teacher_fold_assignment(set(split["refit_drugs"])) else "", "selection_assignment_scope": "Fit_only", "refit_assignment_scope": "Fit_union_Validation"} for drug, fold in sorted(assignment.items())])
    return plan


def _patch_teacher_context(source_cell: str, surface: str, v3: dict[str, Any], units: list[tuple[str, float]], teacher_split: dict[str, Any], hvg: np.ndarray, hvg_audit: dict[str, Any]):
    """Patch only helper module globals while invoking its public stages.

    The original teacher functions use module-level TARGET/SURFACE and call
    fit_vehicle_hvg internally.  Saving/restoring all patched values makes
    the adapter deterministic when nine roots are run in one process.
    """
    saved = {
        "target": teacher.TARGET,
        "surface": teacher.SURFACE,
        "capacities": teacher.CAPACITIES,
        "capacity_order": teacher.CAPACITY_ORDER,
        "v3": teacher._v3_and_split,
        "hvg": p0.old.fit_vehicle_hvg,
        "teacher_hvg": teacher.base.old.fit_vehicle_hvg,
    }
    teacher.TARGET = source_cell
    teacher.SURFACE = surface
    teacher.CAPACITIES = {CAPACITY: CAPACITY_DIMS}
    teacher.CAPACITY_ORDER = (CAPACITY,)
    teacher._v3_and_split = lambda _args: (v3, units, teacher_split)
    p0.old.fit_vehicle_hvg = lambda _store, _rows, _cells: (np.asarray(hvg, dtype=np.int64), dict(hvg_audit))
    teacher.base.old.fit_vehicle_hvg = lambda _store, _rows, _cells: (np.asarray(hvg, dtype=np.int64), dict(hvg_audit))
    return saved


def _restore_teacher_context(saved: dict[str, Any]) -> None:
    teacher.TARGET = saved["target"]
    teacher.SURFACE = saved["surface"]
    teacher.CAPACITIES = saved["capacities"]
    teacher.CAPACITY_ORDER = saved["capacity_order"]
    teacher._v3_and_split = saved["v3"]
    p0.old.fit_vehicle_hvg = saved["hvg"]
    teacher.base.old.fit_vehicle_hvg = saved["teacher_hvg"]


def _run_source_teacher(root: Path, source_cell: str, target: str, surface: str, v3: dict[str, Any], units: list[tuple[str, float]], spec0: dict[str, Any], args: argparse.Namespace, hvg: np.ndarray, hvg_audit: dict[str, Any]) -> dict[str, Any]:
    teacher_root = root / "teachers" / source_cell
    teacher_root.mkdir(parents=True, exist_ok=False)
    teacher_split = {"train_drugs": sorted(str(x) for x in spec0["train_drugs"]), "valid_drugs": sorted(str(x) for x in spec0["valid_drugs"]), "refit_drugs": sorted(str(x) for x in spec0["refit_drugs"]), "test_drugs": [], "foreign_drugs": sorted(str(x) for x in spec0["foreign_drugs"])}
    _teacher_plan(teacher_root, source_cell, target, surface, v3, units, spec0, args)
    # The helper reads target/source treated rows only through its source cell
    # argument (TARGET=source_cell).  The patched HGV is the surface-wide
    # source-cell-balanced coordinate system, so source teacher outputs are
    # aligned before averaging across source lines.
    saved = _patch_teacher_context(source_cell, surface, v3, units, teacher_split, hvg, hvg_audit)
    targs = argparse.Namespace(root=teacher_root, v3_plan=args.v3_dir / surface / "PLAN.json", conditions=args.conditions, group_counts=args.group_counts, split_lock=args.split_lock, structure_map=args.structure_map, device=args.device)
    try:
        teacher.select(targs)
        freeze = read_json(teacher_root / "SELECTION_FREEZE.json")
        if freeze.get("selected_capacity") != CAPACITY:
            raise RuntimeError(f"teacher selected capacity drift in {teacher_root}")
        teacher.refit(targs)
    finally:
        _restore_teacher_context(saved)
    refit_marker = read_json(teacher_root / "REFIT_COMPLETE.json")
    if refit_marker.get("status") != "PASS" or refit_marker.get("confirmation_treated_loaded") is not False:
        raise RuntimeError(f"invalid source teacher refit marker: {teacher_root}")
    return {
        "source_cell_line": source_cell,
        "root": str(teacher_root.relative_to(root).as_posix()),
        "selected_capacity": CAPACITY,
        "selection_freeze": "teachers/%s/SELECTION_FREEZE.json" % source_cell,
        "selection_freeze_sha256": sha256(teacher_root / "SELECTION_FREEZE.json"),
        "refit_complete": "teachers/%s/REFIT_COMPLETE.json" % source_cell,
        "refit_complete_sha256": sha256(teacher_root / "REFIT_COMPLETE.json"),
        "hvg_sha256": array_sha256(hvg),
        "confirmation_treated_loaded": False,
        "test_loaded": False,
    }


def _teacher_refit_map(teacher_root: Path) -> dict[int, Path]:
    marker = read_json(teacher_root / "REFIT_COMPLETE.json")
    result: dict[int, Path] = {}
    for item in marker.get("checkpoints", []):
        fold = int(item["fold"])
        path = teacher_root / str(item["path"])
        if not path.is_file() or sha256(path) != item.get("sha256"):
            raise RuntimeError(f"teacher refit checkpoint hash mismatch: {path}")
        result[fold] = path
    if set(result) != set(range(5)):
        raise RuntimeError(f"teacher refit lacks five folds: {teacher_root}")
    return result


def _teacher_selection_map(teacher_root: Path) -> dict[int, Path]:
    freeze = read_json(teacher_root / "SELECTION_FREEZE.json")
    result: dict[int, Path] = {}
    for item in freeze.get("selection_artifacts", []):
        if item.get("capacity") == CAPACITY and "fold" in item:
            path = teacher_root / str(item["path"])
            if not path.is_file() or sha256(path) != item.get("sha256"):
                raise RuntimeError(f"teacher selection checkpoint hash mismatch: {path}")
            result[int(item["fold"])] = path
    if set(result) != set(range(5)):
        raise RuntimeError(f"teacher selection lacks five folds: {teacher_root}")
    return result


def _predict_teacher_cell(teacher_objects: dict[int, dict[str, Any]], assignment: dict[str, int], profile: dict, cell: str, units: list[tuple[str, float]], drugs: set[str], *, selection: bool) -> np.ndarray:
    labels: list[np.ndarray] = []
    for drug, dose in units:
        if drug not in drugs:
            continue
        if drug not in assignment:
            raise RuntimeError(f"missing OOF assignment for {drug}")
        t = teacher_objects[int(assignment[drug])]
        repeats = np.vstack([p0.old.response(profile, cell, drug, dose, rep) for rep in p0.old.REPS]).astype(np.float32)
        labels.append(np.mean(teacher.predict_refit_teacher(t, repeats), axis=0).astype(np.float32))
    if not labels:
        raise RuntimeError(f"empty teacher labels for {cell}")
    out = np.vstack(labels).astype(np.float32)
    if out.shape[1] != p0.old.N_HVG or not np.isfinite(out).all():
        raise RuntimeError(f"invalid teacher labels: {out.shape}")
    return out


def _load_teacher_objects(paths: dict[int, Path], device: torch.device) -> dict[int, dict[str, Any]]:
    return {fold: teacher.load_refit_teacher(path, device) for fold, path in sorted(paths.items())}


def _validation_audit_labels(source_cell: str, profile: dict, fit_drugs: set[str], valid_drugs: set[str], units: list[tuple[str, float]], seed_token: str, device: torch.device) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Full-Fit teacher selected on Validation, per source cell.

    This is the only teacher used for Validation Student labels.  Capacity is
    already frozen globally; epoch/simplex selection remains source-cell
    specific and compound-equal as required by the teacher contract.
    """
    saved_target = teacher.TARGET
    teacher.TARGET = source_cell
    try:
        x, y, _fc, _fcond, _fd = teacher._pairs(profile, fit_drugs, units)
        vx, vy, vc, vcond, _vd = teacher._pairs(profile, valid_drugs, units)
        result, logs, compound_rows = teacher._fit_selection_teacher(
            CAPACITY, x, y, vx, vy, vc, vcond, fit_drugs, profile, units,
            teacher._seed(seed_token), device,
        )
        obj = {
            "model": teacher.MLPTeacher(p0.old.N_HVG, *CAPACITY_DIMS).to(device),
            "mean": np.asarray(result["support_mean"], dtype=np.float32),
            "scale": np.asarray(result["support_scale"], dtype=np.float32),
            "shrink": np.asarray(result["imceb_shrink"], dtype=np.float32),
            "weights": np.asarray(result["weights"], dtype=np.float32),
            "device": device,
            "capacity": CAPACITY,
        }
        obj["model"].load_state_dict(result["state_dict"])
        obj["model"].eval()
        labels = []
        for drug, dose in units:
            if drug not in valid_drugs:
                continue
            repeats = np.vstack([p0.old.response(profile, source_cell, drug, dose, rep) for rep in p0.old.REPS]).astype(np.float32)
            labels.append(np.mean(teacher.predict_refit_teacher(obj, repeats), axis=0).astype(np.float32))
        value = np.vstack(labels).astype(np.float32)
        metadata = {"source_cell_line": source_cell, "capacity": CAPACITY, "selected_epoch": int(result["selected_epoch"]), "selected_epoch_one_based": int(result["selected_epoch_one_based"]), "selected_validation_compound_equal_normalized_mse": float(result["selected_validation_compound_equal_normalized_mse"]), "weights": {"IMR_MLP": float(result["weights"][0]), "LSO_equals_IMR": float(result["weights"][1]), "IMCEB": float(result["weights"][2])}, "state_hash": result["state_hash"], "validation_objective": "compound_equal_normalized_mse", "confirmation_treated_loaded": False, "test_loaded": False}
        return value, {"result": result, "metadata": metadata}, logs, compound_rows
    finally:
        teacher.TARGET = saved_target


def _student_fit_fixed(train, label: np.ndarray, fingerprints: dict[str, np.ndarray], stats: dict[str, np.ndarray], seed: int, device: torch.device, epochs: int) -> tuple[Any, list[dict[str, Any]], str]:
    p0.old.set_seed(seed)
    tb, tf, td = p0.old.student_arrays(train, fingerprints, stats)
    target = ((label - stats["target_mean"]) / stats["target_scale"]).astype(np.float32)
    model = p0.old.Student().to(device)
    init_hash = p0.old.state_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=p0.old.LEARNING_RATE, weight_decay=p0.old.WEIGHT_DECAY)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(tb), torch.from_numpy(tf), torch.from_numpy(td), torch.from_numpy(target)), batch_size=p0.old.STUDENT_BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed), num_workers=0)
    logs: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        model.train()
        total = 0.0
        for baseline_value, fp, dose, y in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(baseline_value.to(device), fp.to(device), dose.to(device))
            loss = torch.mean((pred - y.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite fixed Student loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(y)
        logs.append({"epoch": epoch, "train_mse": total / len(train.drug), "validation_read": False, "checkpoint_criterion": "frozen_epoch_no_validation"})
    return model, logs, init_hash


def _fit_freeze_one(args: argparse.Namespace, plan: dict[str, Any], plans: dict[str, dict[str, Any]], units: list[tuple[str, float]], target: str, surface: str, device: torch.device, rows: dict, fingerprints: dict[str, np.ndarray], store) -> dict[str, Any]:
    spec0 = _spec(plans, target, surface)
    source_cells = list(spec0["source_cell_lines"])
    root = args.root / "fit_freeze" / surface / target
    root.mkdir(parents=True, exist_ok=False)
    source_drugs = set(str(x) for x in spec0["refit_drugs"])
    train_drugs = set(str(x) for x in spec0["train_drugs"])
    valid_drugs = set(str(x) for x in spec0["valid_drugs"])
    group_reads_before = set(int(x) for x in store.read_group_ids)
    control_reads_before = set(int(x) for x in store.read_control_ids)
    # The only treated rows opened in fit_freeze are source-cell rows in the
    # frozen Fit∪Validation/refit universe.  For own-cell unseen this is
    # disjoint from target Confirmation; for cross-cell it is a different cell.
    hvg, hvg_audit = _common_hvg(store, rows, source_cells)
    profile = p0.old.load_source_profiles(store, rows, source_cells, source_drugs, hvg, set(units))
    forbidden = {(target, drug) for drug in set(spec0["test_drugs"])}
    loaded_forbidden = [key for key in profile if (key[0], key[1]) in forbidden]
    if loaded_forbidden:
        raise RuntimeError(f"target Confirmation treated profile opened during fit_freeze: {loaded_forbidden[:3]}")
    v3 = plans[surface]
    teacher_records = []
    for cell in source_cells:
        teacher_records.append(_run_source_teacher(root, cell, target, surface, v3, units, spec0, args, hvg, hvg_audit))
    # Build OOF source labels and balanced rows.  Each cell's teacher is
    # independent; only the final all-cell mean is used as the Student row.
    fit_raw = _balanced_rows(profile, source_cells, train_drugs, units)
    valid_raw = _balanced_rows(profile, source_cells, valid_drugs, units)
    refit_raw = _balanced_rows(profile, source_cells, source_drugs, units)
    selection_assignment = p0.old.teacher_fold_assignment(train_drugs)
    refit_assignment = p0.old.teacher_fold_assignment(source_drugs)
    fit_cell_cfra: list[np.ndarray] = []
    valid_cell_cfra: list[np.ndarray] = []
    refit_cell_cfra: list[np.ndarray] = []
    validation_teacher_meta: list[dict[str, Any]] = []
    validation_logs: list[dict[str, Any]] = []
    validation_compound_rows: list[dict[str, Any]] = []
    for cell, record in zip(source_cells, teacher_records):
        troot = root / record["root"]
        selection_objs = _load_teacher_objects(_teacher_selection_map(troot), device)
        refit_objs = _load_teacher_objects(_teacher_refit_map(troot), device)
        cell_profile = {key: value for key, value in profile.items() if key[0] == cell}
        fit_cell_cfra.append(_predict_teacher_cell(selection_objs, selection_assignment, cell_profile, cell, units, train_drugs, selection=True))
        refit_cell_cfra.append(_predict_teacher_cell(refit_objs, refit_assignment, cell_profile, cell, units, source_drugs, selection=False))
        va, meta, logs, comp_rows = _validation_audit_labels(cell, cell_profile, train_drugs, valid_drugs, units, f"{target}|{surface}|validation-audit|{cell}|{CAPACITY}", device)
        valid_cell_cfra.append(va)
        validation_teacher_meta.append(meta["metadata"])
        validation_logs.extend([{**row, "source_cell_line": cell} for row in logs])
        validation_compound_rows.extend([{**row, "source_cell_line": cell} for row in comp_rows])
    # The order of rows is units filtered in the same frozen order for every
    # source cell, so equal-source-cell averaging is well-defined.
    fit_cfra = np.mean(np.stack(fit_cell_cfra), axis=0).astype(np.float32)
    valid_cfra = np.mean(np.stack(valid_cell_cfra), axis=0).astype(np.float32)
    refit_cfra = np.mean(np.stack(refit_cell_cfra), axis=0).astype(np.float32)
    labels = {"raw": fit_raw.raw_label, "cfra": fit_cfra, "residual": fit_raw.raw_label - fit_cfra}
    valid_labels = {"raw": valid_raw.raw_label, "cfra": valid_cfra, "residual": valid_raw.raw_label - valid_cfra}
    refit_labels = {"raw": refit_raw.raw_label, "cfra": refit_cfra, "residual": refit_raw.raw_label - refit_cfra}
    for name, value in {"fit_raw": fit_raw.raw_label, "fit_cfra": fit_cfra, "valid_raw": valid_raw.raw_label, "valid_cfra": valid_cfra, "refit_raw": refit_raw.raw_label, "refit_cfra": refit_cfra}.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"non-finite target {name}")
    np.savetxt(root / "source_vehicle_hvg2000.txt", hvg, fmt="%d")
    np.savez_compressed(root / "targets.npz", fit_raw=fit_raw.raw_label, fit_cfra=fit_cfra, fit_residual=labels["residual"], valid_raw=valid_raw.raw_label, valid_cfra=valid_cfra, valid_residual=valid_labels["residual"], refit_raw=refit_raw.raw_label, refit_cfra=refit_cfra, refit_residual=refit_labels["residual"])
    target_row_manifest: list[dict[str, Any]] = []
    for split_name, row_obj, array_names in (
        ("Fit_OOF", fit_raw, "fit_raw|fit_cfra|fit_residual"),
        ("Validation", valid_raw, "valid_raw|valid_cfra|valid_residual"),
        ("Refit", refit_raw, "refit_raw|refit_cfra|refit_residual"),
    ):
        for row_order, (drug, dose) in enumerate(zip(row_obj.drug, row_obj.dose)):
            target_row_manifest.append({
                "target_cell_line": target,
                "surface": surface,
                "split": split_name,
                "drug": str(drug),
                "dose_nM": float(dose),
                "source_cell_lines": "|".join(source_cells),
                "source_cell_balance": "equal_source_cell_mean",
                "physical_repeat_aggregation": "rep1+rep2 mean within source cell, then source-cell mean",
                "row_order": int(row_order),
                "row_order_contract": "frozen PLAN units filtered by split drug set",
                "target_arrays": array_names,
                "confirmation_treated_loaded": False,
                "test_loaded": False,
            })
    write_csv(root / "TARGET_ROW_MANIFEST.csv", target_row_manifest)
    np.savez_compressed(root / "fit_stats_raw.npz", **p0.old.fit_stats(fit_raw))
    for kind in ("cfra", "residual"):
        stats = p0.old.fit_stats(p0.old.StudentRows(fit_raw.cell, fit_raw.drug, fit_raw.dose, fit_raw.baseline, labels[kind], labels[kind]))
        np.savez_compressed(root / f"fit_stats_{kind}.npz", **stats)
    write_csv(root / "validation_teacher_epochs.csv", validation_logs)
    write_csv(root / "validation_teacher_compound_losses.csv", validation_compound_rows)
    write_json(root / "TEACHER_VALIDATION_AUDIT.json", {"source_cell_lines": source_cells, "capacity": CAPACITY, "teachers": validation_teacher_meta, "objective": "per_source_cell_compound_equal_normalized_mse", "source_cell_selection": "each source cell has its own Validation-selected epoch/simplex; no source-cell mean is used to select a single shared teacher", "cross_cell_student_aggregation": "only after source-cell teacher freezes, CFRA predictions are averaged equally for the Student row", "confirmation_treated_loaded": False, "test_loaded": False})
    # Student selection follows the original P0 row-wise Validation rule.  It
    # is intentionally separate from the teacher's compound-equal rule.
    fps = fingerprints
    stats = {kind: p0.old.fit_stats(p0.old.StudentRows(fit_raw.cell, fit_raw.drug, fit_raw.dose, fit_raw.baseline, labels[kind], labels[kind])) for kind in labels}
    selection_rows: list[dict[str, Any]] = []
    selection_values: dict[tuple[str, int, int], dict[str, Any]] = {}
    for method in METHODS:
        for seed in SEEDS:
            for branch, kind in enumerate(METHOD_KINDS[method]):
                branch_seed = int(seed + branch * 1_000_003)
                train = p0.old.StudentRows(fit_raw.cell, fit_raw.drug, fit_raw.dose, fit_raw.baseline, labels[kind], labels[kind])
                valid = p0.old.StudentRows(valid_raw.cell, valid_raw.drug, valid_raw.dose, valid_raw.baseline, valid_labels[kind], valid_labels[kind])
                ckpt = root / "student_selection" / method / f"seed_{seed}" / f"branch_{branch}" / "student.pt"
                best, init_hash = p0.fit_one(train, valid, labels[kind], valid_labels[kind], branch_seed, stats[kind], fps, device, ckpt)
                log = ckpt.parent / "training_log.csv"
                selection_rows.append({"target_cell_line": target, "surface": surface, "method": method, "seed": int(seed), "branch": branch, "target_kind": kind, "selected_epoch": int(best), "selected_epoch_one_based": int(best + 1), "fixed_refit_epochs": int(best + 1), "seed_used": branch_seed, "checkpoint": str(ckpt.relative_to(root).as_posix()), "checkpoint_sha256": sha256(ckpt), "training_log": str(log.relative_to(root).as_posix()), "training_log_sha256": sha256(log), "validation_target": "P0 row-wise Validation target of matching target kind", "confirmation_treated_loaded": False, "test_loaded": False})
                selection_values[(method, int(seed), branch)] = {"epoch": int(best), "branch_seed": branch_seed, "kind": kind}
    write_csv(root / "student_initial_epoch_selection.csv", selection_rows)
    refit_rows: list[dict[str, Any]] = []
    for row in selection_rows:
        method, seed, branch, kind = str(row["method"]), int(row["seed"]), int(row["branch"]), str(row["target_kind"])
        epochs = int(row["fixed_refit_epochs"])
        train = p0.old.StudentRows(refit_raw.cell, refit_raw.drug, refit_raw.dose, refit_raw.baseline, refit_labels[kind], refit_labels[kind])
        model, logs, init_hash = _student_fit_fixed(train, refit_labels[kind], fps, stats[kind], int(row["seed_used"]), device, epochs)
        ckpt = root / "student_refit" / method / f"seed_{seed}" / f"branch_{branch}" / "student.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": VERSION, "role": "student_refit", "target_cell_line": target, "surface": surface, "method": method, "target_kind": kind, "seed": seed, "branch": branch, "seed_used": int(row["seed_used"]), "selected_epoch": int(row["selected_epoch"]), "fixed_refit_epochs": epochs, "state_dict": model.state_dict(), "init_state_hash": init_hash, "label_sha256": array_sha256(refit_labels[kind]), "confirmation_treated_loaded": False, "test_loaded": False}, ckpt)
        log = ckpt.parent / "training_log.csv"
        write_csv(log, logs)
        refit_rows.append({"target_cell_line": target, "surface": surface, "method": method, "target_kind": kind, "seed": seed, "branch": branch, "selected_epoch": int(row["selected_epoch"]), "fixed_refit_epochs": epochs, "selection_checkpoint": row["checkpoint"], "selection_checkpoint_sha256": row["checkpoint_sha256"], "checkpoint": str(ckpt.relative_to(root).as_posix()), "checkpoint_sha256": sha256(ckpt), "training_log": str(log.relative_to(root).as_posix()), "training_log_sha256": sha256(log), "validation_read_mode": "not_read", "checkpoint_search": False, "confirmation_treated_loaded": False, "test_loaded": False})
    write_csv(root / "student_refit_manifest.csv", refit_rows)
    write_json(root / "STUDENT_SELECTION_FREEZE.json", {"status": "FROZEN", "version": VERSION, "target_cell_line": target, "surface": surface, "teacher_capacity": CAPACITY, "selection_metric": "original P0 row-wise Validation MSE", "selection_validation_only": True, "checkpoint_search_after_freeze": False, "selected_students": selection_rows, "confirmation_treated_loaded": False, "test_loaded": False})
    freeze_sha = sha256(root / "STUDENT_SELECTION_FREEZE.json")
    (root / "STUDENT_SELECTION_FREEZE.sha256").write_text(freeze_sha + "\n", encoding="utf-8")
    artifacts = [root / "source_vehicle_hvg2000.txt", root / "targets.npz", root / "TARGET_ROW_MANIFEST.csv", root / "fit_stats_raw.npz", root / "fit_stats_cfra.npz", root / "fit_stats_residual.npz", root / "student_initial_epoch_selection.csv", root / "student_refit_manifest.csv", root / "STUDENT_SELECTION_FREEZE.json", root / "TEACHER_VALIDATION_AUDIT.json", root / "validation_teacher_epochs.csv", root / "validation_teacher_compound_losses.csv"]
    artifacts.extend(root / str(row["checkpoint"]) for row in selection_rows)
    artifacts.extend(root / str(row["training_log"]) for row in selection_rows)
    artifacts.extend(root / str(row["checkpoint"]) for row in refit_rows)
    artifacts.extend(root / str(row["training_log"]) for row in refit_rows)
    artifacts.extend(path for path in (root / "teachers").rglob("*") if path.is_file())
    artifact_hashes = [{"path": str(path.relative_to(root).as_posix()), "sha256": sha256(path)} for path in artifacts]
    marker = {"status": "PASS", "version": VERSION, "target_cell_line": target, "surface": surface, "source_cell_lines": source_cells, "teacher_capacity": CAPACITY, "teacher_regime": TEACHER_REGIME, "student_parameter_count": STUDENT_PARAMETER_COUNT, "student_methods": list(METHODS), "student_checkpoint_count": len(refit_rows), "teacher_records": teacher_records, "hvg_audit": hvg_audit, "source_treated_group_reads": sorted(set(int(x) for x in store.read_group_ids) - group_reads_before), "source_control_group_reads": sorted(set(int(x) for x in store.read_control_ids) - control_reads_before), "confirmation_treated_group_reads": [], "target_confirmation_profile_loaded": False, "confirmation_treated_loaded": False, "test_loaded": False, "artifact_hashes": artifact_hashes, "student_selection_freeze_sha256": freeze_sha, "split_overlap_policy": "allowed_source_refit_target_test_compound_overlap" if surface == "cross_cell_seen" else "strict_fit_validation_confirmation_drug_disjoint", "leakage_decision_audit": "root-specific target-cell treated-profile read audit; another root may reuse these physical rows as source, so no global-blind claim" if surface == "cross_cell_seen" else "unseen target-cell treated profiles must not be read before freeze; Fit/Validation/Confirmation drug sets are disjoint", "target_overlap_rule": "cross_cell_seen allows source refit/target test compound overlap only because this root's target treated profiles are unopened; unseen target test is source-refit disjoint", "root_specific_target_holdout": True, "target_profile_holdout_scope": "this target/surface root only", "cross_root_source_reuse_allowed": surface == "cross_cell_seen", "cross_root_hyperparameter_or_selection_sharing": False, "unseen_target_confirmation_never_model_input": surface != "cross_cell_seen", "global_blind_claim": False}
    write_json(root / "FIT_FREEZE_COMPLETE.json", marker)
    marker["marker_sha256"] = sha256(root / "FIT_FREEZE_COMPLETE.json")
    write_json(root / "FIT_FREEZE_COMPLETE.json", marker)
    return {"version": VERSION, "target_cell_line": target, "surface": surface, "path": str(root.relative_to(args.root).as_posix()), "marker": str((root / "FIT_FREEZE_COMPLETE.json").relative_to(args.root).as_posix()), "marker_sha256": sha256(root / "FIT_FREEZE_COMPLETE.json"), "teacher_capacity": CAPACITY, "student_parameter_count": STUDENT_PARAMETER_COUNT, "source_cell_lines": source_cells, "confirmation_treated_loaded": False, "test_loaded": False}


def _assert_unseen_freezes_before_seen(args: argparse.Namespace, plan: dict[str, Any]) -> None:
    """Require every own/cross-unseen root to PASS before seen roots open data.

    This is an execution-order gate, not a claim of global blindness.  A
    cross-cell-seen root may legitimately reuse physical rows that were a
    source in another root; only the current root's target treated profiles
    are held out until its confirmation stage.
    """
    expected = {(target, surface) for target in CELL_LINES for surface in ("own_cell_unseen", "cross_cell_unseen")}
    failures: list[str] = []
    for target, surface in sorted(expected):
        marker_path = args.root / "fit_freeze" / surface / target / "FIT_FREEZE_COMPLETE.json"
        if not marker_path.is_file():
            failures.append(f"missing:{target}/{surface}")
            continue
        try:
            marker = read_json(marker_path)
        except Exception as exc:  # pragma: no cover - defensive audit path
            failures.append(f"unreadable:{target}/{surface}:{exc}")
            continue
        if marker.get("status") != "PASS":
            failures.append(f"status:{target}/{surface}:{marker.get('status')}")
        if marker.get("version") != VERSION or marker.get("target_cell_line") != target or marker.get("surface") != surface:
            failures.append(f"identity:{target}/{surface}")
        if marker.get("target_confirmation_profile_loaded") is not False or marker.get("confirmation_treated_loaded") is not False or marker.get("test_loaded") is not False:
            failures.append(f"target-read:{target}/{surface}")
        if marker.get("unseen_target_confirmation_never_model_input") is not True:
            failures.append(f"unseen-input-audit:{target}/{surface}")
    if failures:
        raise RuntimeError("cross_cell_seen cannot start before all own_cell_unseen and cross_cell_unseen freezes PASS: " + "; ".join(failures))
    if plan.get("fit_freeze_order", {}).get("unseen_surfaces_must_pass_before_seen") is not True:
        raise RuntimeError("frozen PLAN does not require unseen freezes before cross_cell_seen")
    print(json.dumps({"stage": "fit_freeze", "status": "UNSEEN_FREEZE_GATE_PASS", "required_cells": len(expected), "next_surface": "cross_cell_seen", "global_blind_claim": False}, sort_keys=True), flush=True)


def fit_freeze(args: argparse.Namespace) -> None:
    plan, plans, units = load_matrix_plan(args)
    if args.root.exists() and (args.root / "FIT_FREEZE_COMPLETE.json").is_file():
        raise FileExistsError("matrix FIT_FREEZE_COMPLETE already exists")
    device = p0.old.choose_device(args.device)
    rows = p0.old.build_row_map(p0.old.read_conditions(args.conditions))
    fingerprints, _canonical, structure_audit = p0.old.load_structure_map(args.structure_map)
    store = p0.old.ProfileStore(args.group_counts)
    results: list[dict[str, Any]] = []
    selected_targets = set(args.targets)
    selected_surfaces = set(args.surfaces)
    try:
        for surface in SURFACES:
            if surface not in selected_surfaces:
                continue
            if surface == "cross_cell_seen":
                # This check occurs immediately before the first seen-root
                # target is entered, so no cross-seen target profile can be
                # opened while either unseen surface is still unfinished.
                _assert_unseen_freezes_before_seen(args, plan)
            for target in CELL_LINES:
                if target not in selected_targets:
                    continue
                print(json.dumps({"stage": "fit_freeze", "target": target, "surface": surface, "status": "START", "confirmation_treated_loaded": False}), flush=True)
                results.append(_fit_freeze_one(args, plan, plans, units, target, surface, device, rows, fingerprints, store))
    finally:
        store.close()
    expected_subset = len(selected_targets) * len(selected_surfaces)
    if len(results) != expected_subset:
        raise RuntimeError(f"fit_freeze coverage mismatch {len(results)} != {expected_subset}")
    all_nine = len(results) == 9 and selected_targets == set(CELL_LINES) and selected_surfaces == set(SURFACES)
    if all_nine:
        result_rows = []
        for item in results:
            marker_path = args.root / str(item["marker"])
            result_rows.append({**item, "marker_sha256": sha256(marker_path)})
        write_json(args.root / "FIT_FREEZE_COMPLETE.json", {"status": "PASS", "version": VERSION, "matrix_cells": 9, "cells": result_rows, "capacity": CAPACITY, "confirmation_treated_loaded": False, "test_loaded": False, "all_nine_frozen": True, "structure_audit": structure_audit, "fit_freeze_order": list(SURFACES), "unseen_freezes_passed_before_seen": True, "unseen_confirmation_profiles_never_model_input": True, "root_specific_target_holdout": True, "cross_root_source_reuse_allowed": True, "cross_root_hyperparameter_or_selection_sharing": False, "global_blind_claim": False, "evaluation_after_all_nine_model_refits_locked": True, "confirmation_gate": "all nine model refits are locked before evaluation; this gate is not a global-blind claim because cross-cell-seen source rows may be reused across roots"})
        marker = read_json(args.root / "FIT_FREEZE_COMPLETE.json")
        marker["marker_sha256"] = sha256(args.root / "FIT_FREEZE_COMPLETE.json")
        write_json(args.root / "FIT_FREEZE_COMPLETE.json", marker)
        print(json.dumps({"status": "FIT_FREEZE_COMPLETE", "matrix_cells": 9, "capacity": CAPACITY, "confirmation_treated_loaded": False}, sort_keys=True), flush=True)
    else:
        write_json(args.root / "FIT_FREEZE_PARTIAL.json", {"status": "PARTIAL", "version": VERSION, "matrix_cells": len(results), "cells": results, "capacity": CAPACITY, "confirmation_treated_loaded": False, "test_loaded": False, "all_nine_frozen": False, "fit_freeze_order": list(SURFACES), "unseen_freezes_must_pass_before_seen_start": True, "global_blind_claim": False})
        print(json.dumps({"status": "FIT_FREEZE_PARTIAL", "matrix_cells": len(results), "capacity": CAPACITY, "confirmation_treated_loaded": False}, sort_keys=True), flush=True)


def _verify_all_freezes(args: argparse.Namespace, plan: dict[str, Any]) -> list[dict[str, Any]]:
    marker_path = args.root / "FIT_FREEZE_COMPLETE.json"
    if not marker_path.is_file():
        raise RuntimeError("global FIT_FREEZE_COMPLETE.json is missing; confirmation is blocked")
    marker = read_json(marker_path)
    if marker.get("status") != "PASS" or marker.get("all_nine_frozen") is not True or marker.get("matrix_cells") != 9 or marker.get("confirmation_treated_loaded") is not False or marker.get("global_blind_claim") is not False or marker.get("evaluation_after_all_nine_model_refits_locked") is not True:
        raise RuntimeError("global all-nine freeze gate failed")
    if marker.get("fit_freeze_order") != list(SURFACES) or marker.get("unseen_freezes_passed_before_seen") is not True:
        raise RuntimeError("global freeze order gate failed")
    cells = marker.get("cells", [])
    if len(cells) != 9:
        raise RuntimeError("global freeze has incomplete cell list")
    expected_plan_cells = {
        (str(row.get("target_cell_line")), str(row.get("surface"))): row
        for row in plan.get("matrix_cells", [])
    }
    if set(expected_plan_cells) != {(target, surface) for target in CELL_LINES for surface in SURFACES}:
        raise RuntimeError("frozen PLAN does not contain exactly nine target/surface cells")
    seen: set[tuple[str, str]] = set()
    for item in cells:
        key = (str(item["target_cell_line"]), str(item["surface"]))
        if key in seen or key[0] not in CELL_LINES or key[1] not in SURFACES:
            raise RuntimeError(f"duplicate/invalid frozen cell: {key}")
        seen.add(key)
        expected = expected_plan_cells[key]
        expected_sources = list(expected.get("source_cell_lines", []))
        expected_capacity = str(plan.get("capacity", {}).get("name", CAPACITY))
        if item.get("version") != VERSION or item.get("source_cell_lines") != expected_sources or item.get("teacher_capacity") != expected_capacity:
            raise RuntimeError(f"global freeze item disagrees with frozen PLAN for {key}")
        if str(item.get("path")) != f"fit_freeze/{key[1]}/{key[0]}":
            raise RuntimeError(f"unexpected child path for {key}: {item.get('path')}")
        child = args.root / str(item["marker"])
        if not child.is_file() or sha256(child) != item.get("marker_sha256"):
            raise RuntimeError(f"child freeze marker hash mismatch: {child}")
        child_data = read_json(child)
        if child_data.get("status") != "PASS" or child_data.get("target_confirmation_profile_loaded") is not False or child_data.get("confirmation_treated_loaded") is not False or child_data.get("test_loaded") is not False or child_data.get("global_blind_claim") is not False:
            raise RuntimeError(f"child freeze leakage marker failed: {child}")
        if child_data.get("version") != VERSION or child_data.get("target_cell_line") != key[0] or child_data.get("surface") != key[1] or child_data.get("source_cell_lines") != expected_sources or child_data.get("teacher_capacity") != expected_capacity or child_data.get("student_parameter_count") != STUDENT_PARAMETER_COUNT or child_data.get("root_specific_target_holdout") is not True or child_data.get("target_profile_holdout_scope") != "this target/surface root only" or child_data.get("cross_root_source_reuse_allowed") is not (key[1] == "cross_cell_seen") or child_data.get("cross_root_hyperparameter_or_selection_sharing") is not False or child_data.get("unseen_target_confirmation_never_model_input") is not (key[1] != "cross_cell_seen"):
            raise RuntimeError(f"child freeze marker disagrees with global/frozen PLAN for {key}")
        for artifact in child_data.get("artifact_hashes", []):
            path = child.parent / str(artifact["path"])
            if not path.is_file() or sha256(path) != artifact.get("sha256"):
                raise RuntimeError(f"child artifact hash mismatch: {path}")
    if seen != {(target, surface) for target in CELL_LINES for surface in SURFACES}:
        raise RuntimeError("global freeze does not cover all nine target/surface cells")
    # The global marker is itself hashed after its content is written.  Verify
    # using the sidecar field by rewriting a canonical temporary copy in
    # memory, rather than allowing a modified marker to pass silently.
    declared = marker.get("marker_sha256")
    if declared:
        copy = dict(marker)
        copy.pop("marker_sha256", None)
        canonical = json.dumps(copy, indent=2, sort_keys=True, default=str) + "\n"
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != declared:
            raise RuntimeError("global freeze marker self-hash mismatch")
    return cells


def _load_stats(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key], dtype=np.float32) for key in z.files}


def _load_student(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = p0.old.Student().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _target_profile(store, rows: dict, target: str, surface: str, drugs: set[str], units: list[tuple[str, float]], hvg: np.ndarray) -> tuple[dict, list[dict[str, Any]]]:
    profile: dict = {}
    reads: list[dict[str, Any]] = []
    for drug, dose in units:
        if drug not in drugs:
            continue
        for rep in p0.old.REPS:
            key = (target, drug, float(dose), rep)
            row = rows[key]
            profile[key] = store.profile(row, hvg, read_treated=True)
            reads.append({"target_cell_line": target, "surface": surface, "drug": drug, "dose_nM": float(dose), "replicate": rep, "group_id": int(row.group_id), "read_after_all_nine_model_refits_locked": True, "read_after_root_specific_target_holdout_gate": True, "root_specific_target_holdout": True, "cross_root_source_reuse_allowed": surface == "cross_cell_seen", "global_blind_claim": False, "confirmation_treated_loaded": drug in drugs})
    return profile, reads


def _mean_metric(values: list[dict[str, Any]], name: str) -> float:
    finite = [float(value[name]) for value in values if np.isfinite(float(value[name]))]
    return float(np.mean(finite)) if finite else float("nan")


def _paired_boot(raw: dict[str, float], other: dict[str, float], token: str) -> dict[str, Any]:
    keys = sorted(k for k in set(raw) & set(other) if np.isfinite(raw[k]) and np.isfinite(other[k]))
    if not keys:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_drugs": 0, "bootstrap_draws": BOOTSTRAP_ROUNDS, "bootstrap_unit": "paired_drug"}
    delta = np.asarray([other[key] - raw[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(p0.old.stable_int(3407, "complete-matrix", token))
    boot = delta[rng.integers(0, len(delta), size=(BOOTSTRAP_ROUNDS, len(delta)))].mean(axis=1)
    return {"estimate": float(delta.mean()), "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)), "n_drugs": len(keys), "bootstrap_draws": BOOTSTRAP_ROUNDS, "bootstrap_unit": "paired_drug"}


def _build_full_comparison_table(summary: list[dict[str, Any]], all_contrasts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the wide M0/M1/M2 table without mixing row schemas.

    ``summary`` is wide: one row per ``(target, surface, method)`` with all
    metric names as columns, so it must be indexed by ``method`` only.
    ``all_contrasts`` is long: one row per ``(target, surface, comparison,
    metric)``, so the M2-minus-M0 subset is indexed by ``metric``.  Keeping
    these two maps separate prevents report-only ``KeyError`` failures from
    being mistaken for a model/evaluation failure.
    """
    table: list[dict[str, Any]] = []
    comparison = "M2_CFRA1Residual - M0_RawAggregate"
    for target in CELL_LINES:
        for surface in SURFACES:
            contrast_by_metric = {
                str(row["metric"]): row
                for row in all_contrasts
                if row["target_cell_line"] == target
                and row["surface"] == surface
                and row["comparison"] == comparison
            }
            summary_by_method = {
                str(row["method"]): row
                for row in summary
                if row["stratum"] == "target_surface"
                and row["target_cell_line"] == target
                and row["surface"] == surface
            }
            for metric in BOOTSTRAP_METRICS:
                m0 = summary_by_method["M0_RawAggregate"]
                m1 = summary_by_method["M1_CFRA1Aggregate"]
                m2 = summary_by_method["M2_CFRA1Residual"]
                contrast = contrast_by_metric[metric]
                table.append({"target_cell_line": target, "surface": surface, "metric": metric, "n_compounds": m0["n_compounds"], "teacher_regime": "M0=N/A; M1/M2=" + TEACHER_REGIME, "capacity": CAPACITY, "parameter_count": STUDENT_PARAMETER_COUNT, "M0_raw_value": m0[metric], "M1_raw_value": m1[metric], "M2_raw_value": m2[metric], "M2_minus_M0_estimate": contrast["estimate"], "M2_minus_M0_ci_low": contrast["ci_low"], "M2_minus_M0_ci_high": contrast["ci_high"], "bootstrap_draws": BOOTSTRAP_ROUNDS, "bootstrap_unit": "paired_drug", "report_only_hotfix": REPORT_ONLY_HOTFIX_VERSION, "test_loaded": True})
    return table


def _evaluate_cell(args: argparse.Namespace, plans: dict[str, dict[str, Any]], units: list[tuple[str, float]], target: str, surface: str, device: torch.device, rows: dict, fingerprints: dict[str, np.ndarray], store, root_item: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    spec0 = _spec(plans, target, surface)
    child = args.root / str(root_item["path"])
    hvg = np.loadtxt(child / "source_vehicle_hvg2000.txt", dtype=np.int64)
    needed = set(str(x) for x in spec0["test_drugs"]) | set(str(x) for x in spec0["foreign_drugs"])
    profile, reads = _target_profile(store, rows, target, surface, needed, units, hvg)
    methods_models: dict[str, dict[tuple[int, int], Any]] = defaultdict(dict)
    stats = {kind: _load_stats(child / f"fit_stats_{kind}.npz") for kind in ("raw", "cfra", "residual")}
    for row in read_csv(child / "student_refit_manifest.csv"):
        key = (str(row["method"]), int(row["seed"]), int(row["branch"]))
        methods_models[str(row["method"])][(int(row["seed"]), int(row["branch"]))] = _load_student(child / str(row["checkpoint"]), device)
    condition_rows: list[dict[str, Any]] = []
    by_method: dict[str, defaultdict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    test_drugs = set(str(x) for x in spec0["test_drugs"])
    foreign_drugs = set(str(x) for x in spec0["foreign_drugs"])
    for drug, dose in units:
        if drug not in test_drugs:
            continue
        held = {rep: p0.old.response(profile, target, drug, dose, rep) for rep in p0.old.REPS}
        truth = ((held["rep1"] + held["rep2"]) / 2.0).astype(np.float32)
        base = ((p0.old.baseline(profile, target, drug, dose, "rep1") + p0.old.baseline(profile, target, drug, dose, "rep2")) / 2.0).astype(np.float32)
        foreign: list[dict[str, np.ndarray]] = []
        for donor in sorted(foreign_drugs - {drug}):
            if all((target, donor, float(dose), rep) in profile for rep in p0.old.REPS):
                foreign.append({rep: p0.old.response(profile, target, donor, dose, rep) for rep in p0.old.REPS})
        if len(foreign) < p0.old.N_FOREIGN:
            raise RuntimeError(f"foreign pool < {p0.old.N_FOREIGN} for {target}/{surface}/{drug}/{dose}")
        input_rows = p0.old.StudentRows(np.asarray([target]), np.asarray([drug]), np.asarray([dose], dtype=np.float32), base[None, :], np.zeros((1, p0.old.N_HVG), dtype=np.float32), truth[None, :])
        for method in METHODS:
            seed_preds: list[np.ndarray] = []
            for seed in SEEDS:
                branches: list[np.ndarray] = []
                for branch, kind in enumerate(METHOD_KINDS[method]):
                    pred = p0.old.predict_model(methods_models[method][(int(seed), branch)], input_rows, fingerprints, stats[kind], device)[0]
                    branches.append(pred)
                seed_preds.append((branches[0] + branches[1]) if method == "M2_CFRA1Residual" else np.mean(np.stack(branches), axis=0))
            pred = np.mean(np.stack(seed_preds), axis=0)
            repeat_metrics = p0.two_repeat_metrics(pred, held, foreign)
            item = {"target_cell_line": target, "surface": surface, "method": method, "drug": drug, "dose_nM": float(dose), "delta_pcc": p0.old.corr(pred, truth), "full_target_pcc": p0.old.corr(pred + base, truth + base), "raw_target_mse": float(np.mean((pred - truth) ** 2)), **repeat_metrics, "foreign_eligible": True, "held_metric_status": REPEAT_STATUS, "confirmation_treated_loaded": True, "test_loaded": True}
            condition_rows.append(item)
            by_method[method][drug].append(item)
    compound_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for drug, values in by_method[method].items():
            compound_rows.append({"target_cell_line": target, "surface": surface, "method": method, "drug": drug, "n_doses": len(values), "n_foreign_eligible_doses": len(values), **{metric: _mean_metric(values, metric) for metric in BOOTSTRAP_METRICS}, "teacher_regime": "N/A" if method == "M0_RawAggregate" else TEACHER_REGIME, "capacity": CAPACITY, "parameter_count": STUDENT_PARAMETER_COUNT, "held_metric_status": REPEAT_STATUS, "confirmation_treated_loaded": True, "test_loaded": True})
    contrasts: list[dict[str, Any]] = []
    comparisons = (("M2_CFRA1Residual", "M0_RawAggregate"), ("M1_CFRA1Aggregate", "M0_RawAggregate"), ("M2_CFRA1Residual", "M1_CFRA1Aggregate"))
    for metric in BOOTSTRAP_METRICS:
        values = {method: {str(row["drug"]): float(row[metric]) for row in compound_rows if row["method"] == method and np.isfinite(float(row[metric]))} for method in METHODS}
        for lhs, rhs in comparisons:
            contrasts.append({"target_cell_line": target, "surface": surface, "metric": metric, "comparison": lhs + " - " + rhs, **_paired_boot(values[rhs], values[lhs], f"{target}|{surface}|{metric}|{lhs}-{rhs}"), "teacher_regime": TEACHER_REGIME, "capacity": CAPACITY, "parameter_count": STUDENT_PARAMETER_COUNT, "confirmation_treated_loaded": True, "test_loaded": True})
    return compound_rows, condition_rows, [{**row, "target_cell_line": target, "surface": surface} for row in reads] + contrasts


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def confirmation(args: argparse.Namespace) -> None:
    plan, plans, units = load_matrix_plan(args)
    cells = _verify_all_freezes(args, plan)
    if not args.allow_partial_confirmation and len(cells) != 9:
        raise RuntimeError("primary confirmation requires all nine frozen cells")
    frozen_runner_sha = str(plan.get("runner", {}).get("sha256", ""))
    report_runner_sha = sha256(Path(__file__).resolve())
    report_only_recovery = frozen_runner_sha != report_runner_sha
    if report_only_recovery:
        print(json.dumps({"stage": "confirmation", "status": "REPORT_ONLY_HOTFIX_ACCEPTED", "report_only_hotfix": REPORT_ONLY_HOTFIX_VERSION, "frozen_plan_runner_sha256": frozen_runner_sha, "report_runner_sha256": report_runner_sha, "model_artifacts_unchanged": True, "selection_or_refit_performed": False}, sort_keys=True), flush=True)
    device = p0.old.choose_device(args.device)
    rows = p0.old.build_row_map(p0.old.read_conditions(args.conditions))
    fingerprints, _canonical, _audit = p0.old.load_structure_map(args.structure_map)
    store = p0.old.ProfileStore(args.group_counts)
    all_compound: list[dict[str, Any]] = []
    all_condition: list[dict[str, Any]] = []
    all_contrasts: list[dict[str, Any]] = []
    read_audit: list[dict[str, Any]] = []
    try:
        for item in sorted(cells, key=lambda x: (str(x["target_cell_line"]), str(x["surface"]))):
            target, surface = str(item["target_cell_line"]), str(item["surface"])
            print(json.dumps({"stage": "confirmation", "target": target, "surface": surface, "status": "READ_TARGET_AFTER_GATE"}), flush=True)
            compounds, conditions, mixed = _evaluate_cell(args, plans, units, target, surface, device, rows, fingerprints, store, item)
            all_compound.extend(compounds)
            all_condition.extend(conditions)
            for row in mixed:
                if "comparison" in row:
                    all_contrasts.append(row)
                else:
                    read_audit.append(row)
    finally:
        store.close()
    out = args.root / "confirmation"
    if out.exists() and not report_only_recovery:
        raise FileExistsError(f"confirmation output already exists: {out}")
    if (args.root / "CONFIRMATION_COMPLETE.json").is_file():
        raise FileExistsError("CONFIRMATION_COMPLETE already exists; refusing to overwrite a completed report")
    # The first report-only failure may have left the four intermediate CSVs
    # in place before the table assembly KeyError.  Reusing that directory is
    # permitted only for this exact frozen-runner recovery and overwrites no
    # model, selection, or refit artifact.
    out.mkdir(parents=True, exist_ok=report_only_recovery)
    write_csv(out / "CONFIRMATION_PER_COMPOUND.csv", all_compound)
    write_csv(out / "CONFIRMATION_PER_CONDITION.csv", all_condition)
    write_csv(out / "CONFIRMATION_CONTRASTS.csv", all_contrasts)
    write_csv(out / "CONFIRMATION_READ_AUDIT.csv", read_audit)
    # Method summaries are raw M0/M1/M2 values, not only contrasts.  A macro
    # is computed per surface across the three target lines; no global pooled
    # primary is emitted because cross_cell_seen has 183 test compounds while
    # both unseen surfaces have 36.
    summary: list[dict[str, Any]] = []
    for target in CELL_LINES:
        for surface in SURFACES:
            sub = [row for row in all_compound if row["target_cell_line"] == target and row["surface"] == surface]
            for method in METHODS:
                values = [row for row in sub if row["method"] == method]
                summary.append({"stratum": "target_surface", "target_cell_line": target, "surface": surface, "method": method, "n_compounds": len(values), **{metric: _mean_metric(values, metric) for metric in BOOTSTRAP_METRICS}, "teacher_regime": "N/A" if method == "M0_RawAggregate" else TEACHER_REGIME, "capacity": CAPACITY, "parameter_count": STUDENT_PARAMETER_COUNT, "confirmation_treated_loaded": True, "test_loaded": True})
    for surface in SURFACES:
        for method in METHODS:
            sub = [row for row in summary if row["stratum"] == "target_surface" and row["surface"] == surface and row["method"] == method]
            summary.append({"stratum": "surface_macro_mean", "target_cell_line": "ALL_3_MACRO", "surface": surface, "method": method, "n_compounds": "macro_not_pooled", **{metric: _mean_metric(sub, metric) for metric in BOOTSTRAP_METRICS}, "teacher_regime": "N/A" if method == "M0_RawAggregate" else TEACHER_REGIME, "capacity": CAPACITY, "parameter_count": STUDENT_PARAMETER_COUNT, "confirmation_treated_loaded": True, "test_loaded": True})
    write_csv(args.root / "CONFIRMATION_SUMMARY_ALL.csv", summary)
    write_csv(args.root / "CONFIRMATION_CONTRASTS_ALL.csv", all_contrasts)
    # Full comparison table places each raw method result and the M2-M0
    # paired effect in the same row, making it impossible to mistake a delta
    # for the underlying M0 or M2 value.
    table = _build_full_comparison_table(summary, all_contrasts)
    write_csv(args.root / "FULL_COMPARISON_TABLE.csv", table)
    report = ["# sci-Plex3 complete MLP-teacher matrix", "", "This table is stratified by target cell line and surface. M0/M1/M2 columns are the original method values; M2-minus-M0 columns are paired compound-level effects. Surface macro means are separate and are not pooled across the 36-drug unseen and 183-drug seen strata. Student parameter count is the verified sci-Plex3 package count (2,397,264); CFRA rows use the globally frozen H512Z128 teacher and fit-compound-OOF regime.", "", f"Report-only recovery: `{REPORT_ONLY_HOTFIX_VERSION}`. The first Confirmation stopped while assembling this report because contrast rows have `comparison` but no `method`; the first recovery then exposed that wide summary rows have metric columns but no `metric` field. The r2 builder keeps these schemas separate; model loading, prediction, metrics, bootstrap draws, and frozen artifacts are unchanged.", "", "| target | surface | metric | n | capacity | teacher regime | M0 | M1 | M2 | M2-M0 | 95% CI |", "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in table:
        if row["metric"] in ("delta_pcc", "full_target_pcc", "raw_target_mse", "A", "z_same_F", "z_foreign_F", "E", "held_pcc_A", "held_pcc_B"):
            report.append(f"| {row['target_cell_line']} | {row['surface']} | {row['metric']} | {row['n_compounds']} | {row['capacity']} | {row['teacher_regime']} | {float(row['M0_raw_value']):.6f} | {float(row['M1_raw_value']):.6f} | {float(row['M2_raw_value']):.6f} | {float(row['M2_minus_M0_estimate']):+.6f} | [{float(row['M2_minus_M0_ci_low']):+.6f}, {float(row['M2_minus_M0_ci_high']):+.6f}] |")
    (args.root / "FULL_COMPARISON_TABLE.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(args.root / "CONFIRMATION_COMPLETE.json", {"status": "PASS", "version": VERSION, "report_only_hotfix": REPORT_ONLY_HOTFIX_VERSION, "report_only_recovery": report_only_recovery, "matrix_cells": len(cells), "methods": list(METHODS), "capacity": CAPACITY, "confirmation_treated_loaded_after_all_nine_model_refits_locked": True, "confirmation_treated_loaded": True, "test_loaded": True, "bootstrap_draws": BOOTSTRAP_ROUNDS, "outputs": ["confirmation/CONFIRMATION_PER_COMPOUND.csv", "confirmation/CONFIRMATION_PER_CONDITION.csv", "confirmation/CONFIRMATION_CONTRASTS.csv", "FULL_COMPARISON_TABLE.csv", "FULL_COMPARISON_TABLE.md"], "evidence_status": "historical locked/retrieved Confirmation; not new blind prospective confirmation", "cross_cell_seen_label": "root-specific target holdout with historical source-row reuse; do not pool with unseen surfaces", "root_specific_target_holdout": True, "cross_root_source_reuse_allowed": True, "cross_root_hyperparameter_or_selection_sharing": False, "global_blind_claim": False, "model_artifacts_unchanged": True, "selection_or_refit_performed": False, "frozen_plan_runner_sha256": plan.get("runner", {}).get("sha256"), "report_runner_sha256": sha256(Path(__file__).resolve())})
    print(json.dumps({"status": "CONFIRMATION_COMPLETE", "matrix_cells": len(cells), "compound_rows": len(all_compound), "capacity": CAPACITY, "confirmation_treated_loaded": True}, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    if args.stage == "preflight":
        preflight(args)
    elif args.stage == "fit_freeze":
        fit_freeze(args)
    elif args.stage == "confirmation":
        confirmation(args)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
