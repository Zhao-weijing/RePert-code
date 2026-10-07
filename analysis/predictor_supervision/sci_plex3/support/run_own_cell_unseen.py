#!/usr/bin/env python3
"""Leakage-controlled sci-Plex3 own-cell unseen-drug Student experiment.

This experiment compares two otherwise identical Student models:

* ``raw`` uses one deterministic source-repeat response as its training label;
* ``cfra`` uses a same-cell, drug-OOF CFRA teacher prediction for that same
  support-repeat query.

The data set has only rep1/rep2.  A single support -> held repeat direction is
therefore selected for each drug-dose unit by a frozen hash.  The evaluation
uses the held repeat as truth and shares the target foreign-null donors between
the two Student arms.

For each cell line, base drugs are used for initial fit, calibration drugs for
checkpoint selection, and base+calibration for the fixed-epoch refit.  No
confirmation-drug treated profile is opened until ``test``, after every cell
line/arm/seed checkpoint has been frozen and hash-verified.  Each OOF CFRA
teacher excludes its held drug-fold from fit, normalizer, IMCEB shrinkage,
calibration, foreign donors, and validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import Ridge
from torch import nn


VERSION = "sciplex3-student-own-cell-unseen-v1-audit-2026-09-15"
SEED = 3407
SEEDS = (42, 3407, 2025)
OOF_FOLDS = 5
N_FEATURES = 110_983
N_HVG = 2_000
FP_DIM = 2_048
PARAMETER_COUNT = 2_397_264
N_FOREIGN = 20
RIDGE_ALPHA = 10.0
STUDENT_EPOCHS = 40
STUDENT_BATCH = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
BOOTSTRAP_ROUNDS = 10_000
FISHER_CLIP = 0.999999
CELL_LINES = ("A549", "K562", "MCF7")
REPS = ("rep1", "rep2")
ENDPOINTS = ("own_cell_unseen",)


@dataclass(frozen=True)
class MetaRow:
    drug: str
    cell_line: str
    dose: float
    replicate: str
    plate: str
    group_id: int
    control_group_id: int
    n_cells: int


@dataclass
class StudentRows:
    cell: np.ndarray
    drug: np.ndarray
    dose: np.ndarray
    baseline: np.ndarray
    raw_label: np.ndarray
    truth: np.ndarray


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def stable_int(*parts: Any) -> int:
    token = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big", signed=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("preflight", "dry_run", "fit", "test"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--group-counts", type=Path, required=True)
    p.add_argument("--conditions", type=Path, required=True)
    p.add_argument("--split-lock", type=Path, required=True)
    p.add_argument("--structure-map", type=Path, required=True)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p.parse_args()


def verify_frozen_inputs(plan: dict[str, Any], args: argparse.Namespace) -> None:
    current = {
        "group_counts": args.group_counts,
        "conditions": args.conditions,
        "split_lock": args.split_lock,
        "structure_map": args.structure_map,
        "runner": Path(__file__).resolve(),
    }
    for name, path in current.items():
        expected = plan.get("inputs", {}).get(name, {}).get("sha256")
        actual = sha256(path)
        if expected != actual:
            raise RuntimeError(f"{name} SHA mismatch against frozen PLAN: {actual} != {expected}")


def validate_plan_contract(plan: dict[str, Any], args: argparse.Namespace) -> None:
    """Rebuild all split/source semantics instead of trusting mutable PLAN fields."""
    if plan.get("version") != VERSION or plan.get("surface") != "own_cell_unseen":
        raise RuntimeError("PLAN version/surface mismatch")
    rows = read_conditions(args.conditions)
    expected_units, expected_drugs = common_units(rows)
    fingerprints, _canonical, _audit = load_structure_map(args.structure_map)
    expected_units = [unit for unit in expected_units if unit[0] in fingerprints]
    expected_drugs = sorted({drug for drug, _dose in expected_units})
    plan_units = [(str(x[0]), float(x[1])) for x in plan["common_universe"]["units"]]
    if plan_units != expected_units or list(plan["common_universe"]["drugs"]) != expected_drugs:
        raise RuntimeError("PLAN common universe does not match frozen inputs")
    split = read_split(args.split_lock)
    common_by_split = {name: sorted(set(values) & set(expected_drugs)) for name, values in split.items()}
    expected_spec = {
        "train_drugs": common_by_split["base"],
        "valid_drugs": common_by_split["calibration"],
        "test_drugs": common_by_split["confirmation"],
        "foreign_drugs": sorted(set(common_by_split["base"]) | set(common_by_split["calibration"])),
        "split_source": "same-cell frozen base/calibration/confirmation drug lock",
    }
    if set(expected_spec["train_drugs"]) & set(expected_spec["test_drugs"]) or set(expected_spec["valid_drugs"]) & set(expected_spec["test_drugs"]):
        raise RuntimeError("reconstructed confirmation split overlaps fit/validation")
    for target in CELL_LINES:
        fold = plan.get("folds", {}).get(target, {})
        if fold.get("source_cell_lines") != [target]:
            raise RuntimeError(f"PLAN is not own-cell for {target}")
        if set(fold.get("endpoints", {})) != {"own_cell_unseen"}:
            raise RuntimeError(f"unexpected endpoint surface for {target}")
        if fold["endpoints"]["own_cell_unseen"] != expected_spec:
            raise RuntimeError(f"PLAN split semantics drift for {target}")


def read_conditions(path: Path) -> list[MetaRow]:
    rows: list[MetaRow] = []
    with path.open(encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            rows.append(MetaRow(
                drug=str(raw["drug"]), cell_line=str(raw["cell_line"]),
                dose=float(raw["dose_nM"]), replicate=str(raw["replicate"]),
                plate=str(raw["plate"]), group_id=int(raw["group_id"]),
                control_group_id=int(raw["control_group_id"]),
                n_cells=int(raw["n_cells"]),
            ))
    if not rows:
        raise RuntimeError(f"empty conditions: {path}")
    return rows


def read_split(path: Path) -> dict[str, list[str]]:
    value = read_json(path)
    required = {"base", "calibration", "confirmation"}
    if required - set(value):
        raise RuntimeError(f"split lock missing {sorted(required - set(value))}")
    result = {key: [str(x) for x in value[key]] for key in sorted(required)}
    sets = {key: set(value[key]) for key in result}
    for a, b in (("base", "calibration"), ("base", "confirmation"), ("calibration", "confirmation")):
        if sets[a] & sets[b]:
            raise RuntimeError(f"split overlap {a}/{b}: {sorted(sets[a] & sets[b])}")
    return result


def build_row_map(rows: list[MetaRow]) -> dict[tuple[str, str, float, str], MetaRow]:
    out: dict[tuple[str, str, float, str], MetaRow] = {}
    for row in rows:
        key = (row.cell_line, row.drug, row.dose, row.replicate)
        if key in out:
            raise RuntimeError(f"duplicate condition identity: {key}")
        out[key] = row
    return out


def common_units(rows: list[MetaRow]) -> tuple[list[tuple[str, float]], list[str]]:
    by_cell: dict[str, dict[tuple[str, float], set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row.cell_line in CELL_LINES and row.replicate in REPS and row.n_cells >= 50:
            by_cell[row.cell_line][(row.drug, row.dose)].add(row.replicate)
    unit_sets = [{unit for unit, reps in by_cell[cell].items() if set(REPS).issubset(reps)} for cell in CELL_LINES]
    units = sorted(set.intersection(*unit_sets), key=lambda x: (x[0], x[1]))
    drugs = sorted({drug for drug, _ in units})
    if not units:
        raise RuntimeError("no common units")
    return units, drugs


def support_repeat(drug: str, dose: float) -> tuple[str, str]:
    token = stable_int(SEED, "support-repeat", drug, f"{dose:.9g}") % 2
    support = REPS[int(token)]
    held = REPS[1 - int(token)]
    return support, held


def load_structure_map(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    """Load the preflight mapping; generate fingerprints only as CSV fallback."""
    RDLogger.DisableLog("rdApp.warning")
    fingerprints: dict[str, np.ndarray] = {}
    canonical: dict[str, str] = {}
    audit: dict[str, Any] = {"path": str(path), "sha256": sha256(path), "format": path.suffix.lower()}
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
            drug_key = next((k for k in ("drug", "compound", "compound_id") if k in keys), None)
            fp_key = next((k for k in ("fingerprint", "ecfp4", "fp") if k in keys), None)
            if drug_key is None or fp_key is None:
                raise RuntimeError(f"structure map missing drug/fingerprint fields: {sorted(keys)}")
            drugs = z[drug_key].astype(str)
            fp = np.asarray(z[fp_key])
            if fp.ndim != 2 or fp.shape[1] != FP_DIM:
                raise RuntimeError(f"unexpected fingerprint shape: {fp.shape}")
            smiles_key = next((k for k in ("canonical_smiles", "smiles") if k in keys), None)
            smiles = z[smiles_key].astype(str) if smiles_key else np.full(len(drugs), "", dtype=str)
            status_key = next((k for k in ("mapping_status", "status") if k in keys), None)
            statuses = z[status_key].astype(str) if status_key else np.full(len(drugs), "resolved", dtype=str)
            excluded: list[str] = []
            for drug, row, smi, status in zip(drugs, fp, smiles, statuses):
                if str(status).startswith("exclude"):
                    excluded.append(str(drug))
                    continue
                if drug in fingerprints:
                    raise RuntimeError(f"duplicate structure map drug: {drug}")
                if not np.isfinite(row).all() or not np.isin(row, (0, 1)).all():
                    raise RuntimeError(f"non-binary/non-finite fingerprint: {drug}")
                fingerprints[str(drug)] = row.astype(np.float32)
                canonical[str(drug)] = str(smi)
    else:
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise RuntimeError(f"empty structure mapping: {path}")
        drug_key = next((k for k in ("drug", "compound", "compound_id") if k in rows[0]), None)
        smiles_key = next((k for k in ("canonical_smiles", "smiles", "SMILES") if k in rows[0]), None)
        if drug_key is None or smiles_key is None:
            raise RuntimeError("CSV structure map requires drug and canonical_smiles/smiles")
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_DIM)
        for raw in rows:
            drug = str(raw[drug_key])
            smi = str(raw[smiles_key]).strip()
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise RuntimeError(f"invalid SMILES for {drug}")
            fp = generator.GetFingerprint(mol)
            array = np.zeros(FP_DIM, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, array)
            if drug in fingerprints and not np.array_equal(fingerprints[drug], array):
                raise RuntimeError(f"fingerprint conflict for {drug}")
            fingerprints[drug] = array.astype(np.float32)
            canonical[drug] = Chem.MolToSmiles(mol, canonical=True)
    audit.update({"n_mapped": len(fingerprints), "n_excluded": len(excluded) if path.suffix.lower() == ".npz" else 0, "excluded_drugs": excluded if path.suffix.lower() == ".npz" else [], "fingerprint": "Morgan radius=2, 2048 bits"})
    if not fingerprints:
        raise RuntimeError("empty structure mapping")
    return fingerprints, canonical, audit


def stable_drug_split(drugs: list[str], fraction: float = 0.2) -> tuple[list[str], list[str]]:
    ordered = sorted(drugs, key=lambda x: (stable_int(SEED, "pure-valid", x), x))
    n_valid = max(1, min(len(ordered) - 1, int(round(len(ordered) * fraction))))
    valid = sorted(ordered[:n_valid])
    train = sorted(ordered[n_valid:])
    return train, valid


def preflight(args: argparse.Namespace) -> None:
    if args.root.exists() and any(p.is_file() for p in args.root.iterdir()):
        raise FileExistsError(f"output root has files: {args.root}")
    rows = read_conditions(args.conditions)
    row_map = build_row_map(rows)
    all_units, all_drugs = common_units(rows)
    eligible_units = {(row.drug, float(row.dose)) for row in rows if row.cell_line in CELL_LINES and row.replicate in REPS and row.n_cells >= 50}
    if len(all_drugs) < 100 or len(all_units) / max(len(eligible_units), 1) < 0.5:
        raise RuntimeError(f"Phase-0 coverage gate failed: {len(all_drugs)} drugs, {len(all_units)}/{len(eligible_units)} common units")
    plate_controls: dict[tuple[str, str, str], int] = {}
    for row in rows:
        if not row.plate or row.group_id < 0 or row.control_group_id < 0:
            raise RuntimeError(f"invalid physical identity for {row}")
        key = (row.cell_line, row.replicate, row.plate)
        prior = plate_controls.setdefault(key, row.control_group_id)
        if prior != row.control_group_id:
            raise RuntimeError(f"same plate/cell/repeat maps to multiple Vehicle controls: {key}")
    split = read_split(args.split_lock)
    fingerprints, canonical, structure_audit = load_structure_map(args.structure_map)
    missing = sorted(set(all_drugs) - set(fingerprints))
    # Structure exclusions are a priori eligibility exclusions.  They are
    # applied to every endpoint, source fold, and Student arm together; no
    # result-dependent rescue or per-arm filtering is permitted.
    if missing:
        excluded_drugs = missing
        units = [unit for unit in all_units if unit[0] in fingerprints]
        drugs = sorted({drug for drug, _dose in units})
    else:
        excluded_drugs = []
        units, drugs = all_units, all_drugs
    if not drugs:
        raise RuntimeError("structure mapping excludes every common drug")
    common_by_split = {name: sorted(set(values) & set(drugs)) for name, values in split.items()}
    if not common_by_split["base"] or not common_by_split["calibration"] or not common_by_split["confirmation"]:
        raise RuntimeError(f"frozen split has empty common component: { {k: len(v) for k,v in common_by_split.items()} }")
    folds: dict[str, Any] = {}
    manifest_rows: list[dict[str, Any]] = []
    for target in CELL_LINES:
        source = [target]
        endpoint_specs = {
            "own_cell_unseen": {
                "train_drugs": common_by_split["base"], "valid_drugs": common_by_split["calibration"],
                "test_drugs": common_by_split["confirmation"],
                "foreign_drugs": sorted(set(common_by_split["base"]) | set(common_by_split["calibration"])),
                "split_source": "same-cell frozen base/calibration/confirmation drug lock",
            },
        }
        folds[target] = {"source_cell_lines": source, "endpoints": endpoint_specs}
        for endpoint, spec in endpoint_specs.items():
            for drug, dose in units:
                support, held = support_repeat(drug, dose)
                if drug not in spec["test_drugs"]:
                    continue
                target_rows = [row_map[(target, drug, dose, rep)] for rep in REPS]
                for row in target_rows:
                    manifest_rows.append({
                        "target_cell_line": target, "endpoint": endpoint, "drug": drug,
                        "dose_nM": dose, "support_repeat": support, "held_repeat": held,
                        "support_group_id": row_map[(target, drug, dose, support)].group_id,
                        "held_group_id": row_map[(target, drug, dose, held)].group_id,
                        "role": "same_cell_confirmation_metadata_only",
                    })
    plan = {
        "version": VERSION, "stage": "preflight", "confirmation_treated_loaded": False,
        "dataset": "sci-Plex3 Zenodo 7041849", "surface": "own_cell_unseen",
        "inputs": {
            "group_counts": {"path": str(args.group_counts), "sha256": sha256(args.group_counts)},
            "conditions": {"path": str(args.conditions), "sha256": sha256(args.conditions)},
            "split_lock": {"path": str(args.split_lock), "sha256": sha256(args.split_lock)},
            "structure_map": structure_audit,
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        },
        "feature_map": {
            "annotation_to_h5": "GEO annotation data row i -> X[:,i], i=0..110982",
            "x_tail_column_excluded": 110_983, "n_features": N_FEATURES,
            "selection": "source Vehicle-only profiles; no treated profiles",
        },
        "dataset_policy": {"time_hours": 24, "nperts": 1, "n_min": 50, "replicates": list(REPS), "cell_lines": list(CELL_LINES)},
        "common_universe_raw": {"n_drug_dose_units": len(all_units), "n_drugs": len(all_drugs), "drugs": all_drugs},
        "phase0_gate": {"status": "PASS", "n_common_drugs": len(all_drugs), "common_unit_fraction": len(all_units) / len(eligible_units), "plate_control_keys": len(plate_controls)},
        "common_universe": {"n_drug_dose_units": len(units), "n_drugs": len(drugs), "drugs": drugs, "units": [[d, dose] for d, dose in units], "structure_excluded_drugs": excluded_drugs},
        "common_split_counts": {name: len(value) for name, value in common_by_split.items()},
        "folds": folds,
        "support_policy": "one global hash-selected support repeat per drug-dose, held is the other repeat",
        "student": {"input": "ECFP4 2048 + standardized log10 dose + same-cell held/query Vehicle baseline on frozen same-cell HGV2000", "output": "response HGV2000", "architecture": "4049 -> 512 -> 128 -> 2000", "parameter_count": PARAMETER_COUNT, "seeds": list(SEEDS)},
        "cfra_oof": {"folds": OOF_FOLDS, "fit": "same-cell drug-OOF on base/calibration only", "excluded_per_fold": ["teacher fit", "teacher normalizer", "IMCEB shrinkage", "weight calibration", "foreign donors", "teacher validation"], "ridge_alpha": RIDGE_ALPHA, "lso_policy": "LSO=IMR with two genuine repeats"},
        "n_metadata_rows": len(rows), "n_manifest_rows": len(manifest_rows),
    }
    args.root.mkdir(parents=True, exist_ok=True)
    write_json(args.root / "PLAN.json", plan)
    write_csv(args.root / "target_metadata_manifest.csv", manifest_rows)
    print(json.dumps({"status": "PREFLIGHT_COMPLETE", "root": str(args.root), "common_drugs": len(drugs), "common_units": len(units), "confirmation_treated_loaded": False}, sort_keys=True), flush=True)


class ProfileStore:
    """Lazy source/target pseudo-bulk profiles with explicit read accounting."""

    def __init__(self, counts_path: Path):
        self.file = h5py.File(counts_path, "r")
        if "counts" not in self.file:
            raise RuntimeError("group_counts.h5 lacks counts dataset")
        self.ds = self.file["counts"]
        if self.ds.ndim != 2 or self.ds.shape[1] < N_FEATURES:
            raise RuntimeError(f"unexpected counts shape {self.ds.shape}")
        self.control_cache: dict[int, np.ndarray] = {}
        self.response_cache: dict[int, np.ndarray] = {}
        self.read_group_ids: list[int] = []
        self.read_control_ids: list[int] = []

    def close(self) -> None:
        self.file.close()

    @staticmethod
    def log_cpm(counts: np.ndarray) -> np.ndarray:
        value = np.asarray(counts[:N_FEATURES], dtype=np.float64)
        total = float(value.sum())
        if total <= 0:
            raise RuntimeError("zero library size in pseudo-bulk")
        return np.log1p(value / total * 1e6).astype(np.float32)

    def control_group_profile(self, gid: int) -> np.ndarray:
        gid = int(gid)
        if gid not in self.control_cache:
            self.control_cache[gid] = self.log_cpm(self.ds[gid, :N_FEATURES])
            self.read_control_ids.append(gid)
        return self.control_cache[gid]

    def _control(self, row: MetaRow) -> np.ndarray:
        return self.control_group_profile(row.control_group_id)

    def profile(self, row: MetaRow, hvg: np.ndarray | None = None, *, read_treated: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if not read_treated:
            raise RuntimeError("profile response requested with read_treated=False")
        gid = int(row.group_id)
        if gid not in self.response_cache:
            treated = self.log_cpm(self.ds[gid, :N_FEATURES])
            control = self._control(row)
            self.response_cache[gid] = (treated - control).astype(np.float32)
            self.read_group_ids.append(gid)
        response = self.response_cache[gid]
        control = self._control(row)
        if hvg is None:
            return response.copy(), control.copy()
        return response[hvg].copy(), control[hvg].copy()

    def vehicle_profile(self, row: MetaRow) -> np.ndarray:
        return self._control(row).copy()


def fit_vehicle_hvg(store: ProfileStore, rows: dict[tuple[str, str, float, str], MetaRow], source_cells: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit HGV2000 from unique source-cell Vehicle groups only.

    The control-group IDs are collected independently of any drug list.  This
    is important for drug-OOF CFRA: a held label drug must not enter the outer
    HGV through repeated treated-condition membership, even indirectly.  Each
    source line contributes one equally weighted variance estimate; duplicate
    metadata rows pointing at the same Vehicle group are read only once.
    """
    variance_by_cell: list[np.ndarray] = []
    ids_by_cell: dict[str, list[int]] = {}
    counts_by_cell: dict[str, int] = {}
    for cell in source_cells:
        control_ids = sorted({
            int(row.control_group_id)
            for (row_cell, _drug, _dose, rep), row in rows.items()
            if row_cell == cell and rep in REPS
        })
        if len(control_ids) < 2:
            raise RuntimeError(f"too few unique Vehicle groups for HGV in {cell}: {len(control_ids)}")
        values = np.vstack([store.control_group_profile(gid) for gid in control_ids]).astype(np.float64)
        variance_by_cell.append(np.var(values, axis=0, ddof=1))
        ids_by_cell[cell] = control_ids
        counts_by_cell[cell] = len(control_ids)
    if not variance_by_cell:
        raise RuntimeError("no source cells for Vehicle-only HGV")
    # Equal source-cell weighting prevents a line with more metadata rows (or
    # more drugs/doses) from dominating the outer feature coordinates.
    variance = np.mean(np.vstack(variance_by_cell), axis=0)
    n = int(sum(counts_by_cell.values()))
    if n < 4:
        raise RuntimeError(f"too few unique Vehicle-only profiles for HGV: {n}")
    hvg = np.sort(np.argsort(variance, kind="mergesort")[-N_HVG:]).astype(np.int64)
    return hvg, {"n_vehicle_profiles": n, "n_unique_control_group_ids": n, "vehicle_profiles_by_cell": counts_by_cell, "vehicle_control_group_ids_by_cell": ids_by_cell, "fit_cell_equal_weight": True, "hvg_drug_membership_filter": None, "n_features_before_hvg": N_FEATURES, "n_hvg": int(len(hvg)), "hvg_fit_scope": "unique same-cell Vehicle-only control groups from all metadata; no treated profiles", "fit_cell_vehicle_loaded": True, "confirmation_treated_loaded": False}


def load_source_profiles(store: ProfileStore, rows: dict[tuple[str, str, float, str], MetaRow], source_cells: list[str], drugs: set[str], hvg: np.ndarray, units: set[tuple[str, float]] | None = None) -> dict[tuple[str, str, float, str], tuple[np.ndarray, np.ndarray]]:
    out: dict[tuple[str, str, float, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, row in sorted(rows.items()):
        cell, drug, dose, rep = key
        if cell in source_cells and drug in drugs and rep in REPS and (units is None or (drug, float(dose)) in units):
            response, baseline = store.profile(row, hvg, read_treated=True)
            if key in out:
                raise RuntimeError(f"duplicate profile: {key}")
            out[key] = (response, baseline)
    expected = sum(1 for key in rows if key[0] in source_cells and key[1] in drugs and key[3] in REPS and (units is None or (key[1], float(key[2])) in units))
    if len(out) != expected:
        raise RuntimeError(f"source profile count mismatch {len(out)} != {expected}")
    # Every profile unit entering a teacher or Student row must have both
    # support/held repeats in the frozen common universe.  This rejects
    # non-common doses that happen to be present for only one repeat.
    observed = {(cell, drug, float(dose)): set() for cell, drug, dose, rep in out}
    for cell, drug, dose, rep in out:
        observed[(cell, drug, float(dose))].add(rep)
    incomplete = [key for key, reps in observed.items() if reps != set(REPS)]
    if incomplete:
        raise RuntimeError(f"incomplete source repeats after common-unit filtering: {incomplete[:5]}")
    return out


def response(profile: dict[tuple[str, str, float, str], tuple[np.ndarray, np.ndarray]], cell: str, drug: str, dose: float, rep: str) -> np.ndarray:
    key = (cell, drug, float(dose), rep)
    if key not in profile:
        raise RuntimeError(f"missing profile {key}")
    return profile[key][0]


def baseline(profile: dict[tuple[str, str, float, str], tuple[np.ndarray, np.ndarray]], cell: str, drug: str, dose: float, rep: str) -> np.ndarray:
    key = (cell, drug, float(dose), rep)
    if key not in profile:
        raise RuntimeError(f"missing baseline {key}")
    return profile[key][1]


def source_examples(profile: dict[tuple[str, str, float, str], tuple[np.ndarray, np.ndarray]], source_cells: list[str], drugs: set[str], units: list[tuple[str, float]], fingerprints: dict[str, np.ndarray], *, include_truth: bool) -> StudentRows:
    cells: list[str] = []
    drug_values: list[str] = []
    doses: list[float] = []
    baselines: list[np.ndarray] = []
    raw_labels: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for cell in source_cells:
        for drug, dose in units:
            if drug not in drugs:
                continue
            support, held = support_repeat(drug, dose)
            cells.append(cell); drug_values.append(drug); doses.append(float(dose))
            # The Student query baseline is the same held/query-repeat
            # Vehicle baseline used for its held-repeat truth.  The support
            # repeat is used only for the Raw label; using its Vehicle would
            # give the Raw and CFRA arms a different, unfair input semantics.
            baselines.append(baseline(profile, cell, drug, dose, held))
            raw_labels.append(response(profile, cell, drug, dose, support))
            truths.append(response(profile, cell, drug, dose, held) if include_truth else np.zeros_like(raw_labels[-1]))
    if not cells:
        raise RuntimeError("empty Student examples")
    return StudentRows(np.asarray(cells, dtype=str), np.asarray(drug_values, dtype=str), np.asarray(doses, dtype=np.float32), np.vstack(baselines).astype(np.float32), np.vstack(raw_labels).astype(np.float32), np.vstack(truths).astype(np.float32))


def teacher_fold_assignment(drugs: Iterable[str]) -> dict[str, int]:
    return {drug: stable_int(SEED, "teacher-oof-fold", drug) % OOF_FOLDS for drug in sorted(set(drugs))}


def fisher(value: float) -> float:
    return float(np.arctanh(np.clip(value, -FISHER_CLIP, FISHER_CLIP))) if np.isfinite(value) else float("nan")


def corr(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64) - float(np.mean(left))
    b = np.asarray(right, dtype=np.float64) - float(np.mean(right))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")


def teacher_score(pred: np.ndarray, truth: np.ndarray, foreign: list[np.ndarray]) -> float:
    same = fisher(corr(pred, truth))
    foreign_z = [fisher(corr(pred, value)) for value in foreign]
    return same - float(np.mean(foreign_z))


def same_foreign_metrics(pred: np.ndarray, truth: np.ndarray, foreign: list[np.ndarray]) -> tuple[float, float, float, list[float]]:
    """Compute Same-Foreign using Fisher per donor, then average.

    The foreign Pearson correlations are transformed one at a time.  This is
    deliberately not ``fisher(mean(foreign_pcc))`` because Fisher's transform
    is nonlinear and the sci-Plex endpoint is defined on the donor-wise
    Fisher-z values.
    """
    same_pcc = corr(pred, truth)
    foreign_pcc = [corr(pred, value) for value in foreign]
    foreign_z = [fisher(value) for value in foreign_pcc]
    excess = fisher(same_pcc) - float(np.mean(foreign_z))
    return same_pcc, float(np.mean(foreign_pcc)), excess, foreign_z


def foreign_vectors(profile: dict[tuple[str, str, float, str], tuple[np.ndarray, np.ndarray]], cell: str, target_drug: str, dose: float, held: str, donor_drugs: set[str], token: str) -> tuple[list[str], list[np.ndarray]]:
    available = sorted({drug for (row_cell, drug, donor_dose, rep) in profile if row_cell == cell and float(donor_dose) == float(dose) and rep == held and drug in donor_drugs and drug != target_drug})
    if len(available) < N_FOREIGN:
        raise RuntimeError(f"foreign pool <{N_FOREIGN} for {cell}/{target_drug}/{dose}: {len(available)}")
    chosen = np.random.default_rng(stable_int(SEED, "foreign", token)).choice(np.asarray(available, dtype=str), N_FOREIGN, replace=False).tolist()
    return [str(x) for x in chosen], [response(profile, cell, str(x), dose, held) for x in chosen]


def fit_cfra_teacher_fold(profile: dict[tuple[str, str, float, str], tuple[np.ndarray, np.ndarray]], cell: str, fit_drugs: set[str], calibration_drugs: set[str], held_drugs: set[str], *, token: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit one strict OOF teacher for one source cell and held drug fold."""
    available = set(fit_drugs) - set(held_drugs)
    if len(available) < 8:
        raise RuntimeError(f"too few teacher fit drugs after OOF exclusion: {len(available)}")
    x: list[np.ndarray] = []
    y: list[np.ndarray] = []
    units = sorted({(drug, float(dose)) for (row_cell, drug, dose, rep) in profile if row_cell == cell and drug in available and rep in REPS})
    for drug, dose in units:
        for support, held in (("rep1", "rep2"), ("rep2", "rep1")):
            x.append(response(profile, cell, drug, dose, support))
            y.append(response(profile, cell, drug, dose, held))
    if len(x) < 8:
        raise RuntimeError("too few teacher repeat rotations")
    X = np.asarray(x, dtype=np.float32)
    Y = np.asarray(y, dtype=np.float32)
    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(X.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    model = Ridge(alpha=RIDGE_ALPHA).fit((X - mean) / scale, (Y - mean) / scale)
    means: list[np.ndarray] = []
    within_values: list[np.ndarray] = []
    for drug, dose in units:
        r1 = response(profile, cell, drug, dose, "rep1")
        r2 = response(profile, cell, drug, dose, "rep2")
        means.append((r1 + r2) / 2.0)
        within_values.append(np.var(np.vstack([r1, r2]), axis=0))
    between = np.var(np.vstack(means), axis=0)
    within = np.mean(np.vstack(within_values), axis=0)
    shrink = (between / (between + within + 1e-8)).astype(np.float32)
    held_set = set(held_drugs)
    # Calibration is defined by the requested calibration pool with the held
    # OOF drug fold removed.  In the initial pass this is disjoint from the
    # teacher fit pool (train vs valid); in all-source refit it may overlap by
    # design, but the held drug is always excluded.
    calibration_available = sorted(set(calibration_drugs) - held_set)
    if not calibration_available:
        raise RuntimeError(f"no teacher calibration drugs after excluding held fold {sorted(held_drugs)}")
    donor_drugs = (available | set(calibration_available)) - held_set
    calibration_rows = 0
    calibration_items: list[tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]] = []
    for drug in calibration_available:
        doses = sorted({float(dose) for (row_cell, row_drug, dose, rep) in profile if row_cell == cell and row_drug == drug and rep in REPS})
        for dose in doses:
            for support, held in (("rep1", "rep2"), ("rep2", "rep1")):
                p_imr = model.predict(((response(profile, cell, drug, dose, support)[None, :] - mean) / scale))[0] * scale + mean
                p_lso = p_imr.copy()
                p_eb = response(profile, cell, drug, dose, support) * shrink
                truth = response(profile, cell, drug, dose, held)
                _ids, foreign = foreign_vectors(profile, cell, drug, dose, held, donor_drugs, f"{token}|cal|{drug}|{dose}|{support}|{held}")
                calibration_items.append((np.asarray(p_imr, dtype=np.float32), np.asarray(p_eb, dtype=np.float32), np.asarray(truth, dtype=np.float32), foreign))
                calibration_rows += 1
    best_score = -np.inf
    best_w: np.ndarray | None = None
    for ia in range(21):
        for ib in range(21 - ia):
            w = np.asarray([ia / 20.0, ib / 20.0, 1.0 - ia / 20.0 - ib / 20.0], dtype=np.float64)
            values: list[float] = []
            # Reuse frozen calibration predictions/foreign IDs for every
            # simplex point; this keeps the 21x21 grid deterministic and fast.
            for p_imr, p_eb, truth, foreign in calibration_items:
                values.append(teacher_score((w[0] + w[1]) * p_imr + w[2] * p_eb, truth, foreign))
            score = float(np.mean(values)) if values else -np.inf
            if score > best_score + 1e-15:
                best_score, best_w = score, w
    if best_w is None:
        raise RuntimeError("teacher simplex selection failed")
    return {"mean": mean, "scale": scale, "shrink": shrink, "model_coef": np.asarray(model.coef_, dtype=np.float32), "model_intercept": np.asarray(model.intercept_, dtype=np.float32), "weights": best_w.astype(np.float32)}, {
        "cell_line": cell, "held_drugs": sorted(held_drugs), "fit_drugs": sorted(available), "calibration_drugs": sorted(calibration_available), "fit_calibration_overlap": sorted(set(available) & set(calibration_available)), "fit_calibration_disjoint": not bool(set(available) & set(calibration_available)), "n_fit_rotations": len(x), "n_calibration_rotations": calibration_rows, "ridge_alpha": RIDGE_ALPHA, "lso_equals_imr": True, "normalizer_fit_excludes_held_drugs": True, "imceb_fit_excludes_held_drugs": True, "weight_calibration_excludes_held_drugs": True, "foreign_pool_excludes_held_drugs": True, "teacher_validation": "none; fixed Ridge alpha and source-only calibration", "teacher_hvg": "outer fixed source Vehicle-only HGV; no treated profiles", "weights": {"IMR": float(best_w[0]), "LSO": float(best_w[1]), "IMCEB": float(best_w[2])}, "calibration_score": best_score}


def teacher_predict(teacher: dict[str, np.ndarray], value: np.ndarray) -> np.ndarray:
    mean = teacher["mean"]
    scale = teacher["scale"]
    coef = teacher["model_coef"]
    intercept = teacher["model_intercept"]
    normalized = (value - mean) / scale
    p_imr = normalized @ coef.T + intercept
    p_imr = p_imr * scale + mean
    p_eb = value * teacher["shrink"]
    weights = teacher["weights"]
    return (weights[0] * p_imr + weights[1] * p_imr + weights[2] * p_eb).astype(np.float32)


def build_oof_cfra(profile: dict[tuple[str, str, float, str], tuple[np.ndarray, np.ndarray]], source_cells: list[str], fit_drugs: set[str], calibration_drugs: set[str], units: list[tuple[str, float]], *, token: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    assignment = teacher_fold_assignment(fit_drugs)
    result: list[np.ndarray] = []
    rows_meta: list[dict[str, Any]] = []
    # Output order is source_examples order: source cell -> sorted unit.
    by_key: dict[tuple[str, str, float], np.ndarray] = {}
    for fold in range(OOF_FOLDS):
        held = {drug for drug, value in assignment.items() if value == fold}
        if not held:
            continue
        for cell in source_cells:
            teacher, meta = fit_cfra_teacher_fold(profile, cell, fit_drugs, calibration_drugs, held, token=f"{token}|fold{fold}|{cell}")
            for drug, dose in units:
                if drug not in held:
                    continue
                support, _held_rep = support_repeat(drug, dose)
                value = response(profile, cell, drug, dose, support)
                by_key[(cell, drug, float(dose))] = teacher_predict(teacher, value)
            rows_meta.append({"fold": fold, **meta})
    ordered: list[np.ndarray] = []
    for cell in source_cells:
        for drug, dose in units:
            if drug not in fit_drugs:
                continue
            key = (cell, drug, float(dose))
            if key not in by_key:
                raise RuntimeError(f"missing OOF CFRA label {key}")
            ordered.append(by_key[key])
    labels = np.vstack(ordered).astype(np.float32)
    return labels, rows_meta, assignment


class Student(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4_049, 512), nn.GELU(), nn.Linear(512, 128), nn.GELU(), nn.Linear(128, N_HVG))

    def forward(self, baseline_value: torch.Tensor, fingerprint: torch.Tensor, dose: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((baseline_value, fingerprint, dose), dim=1))


def dose_values(values: np.ndarray) -> np.ndarray:
    return np.log10(np.asarray(values, dtype=np.float32))[:, None]


def fit_stats(train: StudentRows) -> dict[str, np.ndarray]:
    dose = dose_values(train.dose)
    return {
        "baseline_mean": train.baseline.mean(axis=0, dtype=np.float64).astype(np.float32),
        "baseline_scale": np.maximum(train.baseline.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6),
        "dose_mean": dose.mean(axis=0, dtype=np.float64).astype(np.float32),
        "dose_scale": np.maximum(dose.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6),
        "target_mean": train.raw_label.mean(axis=0, dtype=np.float64).astype(np.float32),
        "target_scale": np.maximum(train.raw_label.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6),
    }


def student_arrays(rows: StudentRows, fingerprints: dict[str, np.ndarray], stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_value = (rows.baseline - stats["baseline_mean"]) / stats["baseline_scale"]
    fp = np.vstack([fingerprints[str(drug)] for drug in rows.drug]).astype(np.float32)
    dose = (dose_values(rows.dose) - stats["dose_mean"]) / stats["dose_scale"]
    return baseline_value.astype(np.float32), fp, dose.astype(np.float32)


def predict_model(model: Student, rows: StudentRows, fingerprints: dict[str, np.ndarray], stats: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    b, fp, d = student_arrays(rows, fingerprints, stats)
    out = np.empty((len(rows.drug), N_HVG), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(out), STUDENT_BATCH):
            value = model(torch.from_numpy(b[begin:begin + STUDENT_BATCH]).to(device), torch.from_numpy(fp[begin:begin + STUDENT_BATCH]).to(device), torch.from_numpy(d[begin:begin + STUDENT_BATCH]).to(device)).cpu().numpy()
            out[begin:begin + len(value)] = value * stats["target_scale"] + stats["target_mean"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite Student prediction")
    return out


def state_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode("utf-8")); h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def fit_student(train: StudentRows, valid: StudentRows, label: np.ndarray, fingerprints: dict[str, np.ndarray], stats: dict[str, np.ndarray], seed: int, device: torch.device, fixed_epochs: int | None = None) -> tuple[Student, list[dict[str, Any]], int, str]:
    set_seed(seed)
    tb, tf, td = student_arrays(train, fingerprints, stats)
    vb, vf, vd = student_arrays(valid, fingerprints, stats)
    train_target = (label - stats["target_mean"]) / stats["target_scale"]
    valid_target = (valid.truth - stats["target_mean"]) / stats["target_scale"]
    model = Student().to(device)
    init_hash = state_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(tb), torch.from_numpy(tf), torch.from_numpy(td), torch.from_numpy(train_target.astype(np.float32))), batch_size=STUDENT_BATCH, shuffle=True, generator=generator, num_workers=0)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = -1
    logs: list[dict[str, Any]] = []
    n_epochs = fixed_epochs if fixed_epochs is not None else STUDENT_EPOCHS
    for epoch in range(n_epochs):
        model.train(); train_total = 0.0
        for baseline_value, fp, dose, target in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(baseline_value.to(device), fp.to(device), dose.to(device))
            loss = torch.mean((pred - target.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Student loss")
            loss.backward(); optimizer.step(); train_total += float(loss.detach().cpu()) * len(target)
        model.eval(); valid_total = 0.0
        with torch.no_grad():
            for begin in range(0, len(valid.drug), STUDENT_BATCH):
                pred = model(torch.from_numpy(vb[begin:begin + STUDENT_BATCH]).to(device), torch.from_numpy(vf[begin:begin + STUDENT_BATCH]).to(device), torch.from_numpy(vd[begin:begin + STUDENT_BATCH]).to(device))
                truth = torch.from_numpy(valid_target[begin:begin + STUDENT_BATCH]).to(device)
                valid_total += float(torch.sum((pred - truth) ** 2).cpu())
        valid_mse = valid_total / valid_target.size
        logs.append({"epoch": epoch, "train_mse": train_total / len(train.drug), "validation_raw_held_mse": valid_mse, "checkpoint_criterion": "source validation held Raw truth"})
        if fixed_epochs is None and valid_mse < best_loss:
            best_loss = valid_mse; best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if fixed_epochs is None:
        if best_state is None:
            raise RuntimeError("Student checkpoint missing")
        model.load_state_dict(best_state)
    else:
        best_epoch = fixed_epochs - 1
    return model, logs, best_epoch, init_hash


def save_stats(path: Path, stats: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **stats)


def load_stats(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key], dtype=np.float32) for key in z.files}


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def dry_run(args: argparse.Namespace) -> None:
    """Exercise initial/refit OOF paths without confirmation-drug reads.

    Base/calibration profiles from the same cell line are legal fit inputs.
    This intentionally stops before any confirmation-drug treated profile is
    accessed.  It is a protocol smoke test:
    every endpoint/fold must produce finite initial and all-source-refit CFRA
    labels with the same source-only Vehicle HGV construction used by ``fit``.
    """
    plan = read_json(args.root / "PLAN.json")
    if plan.get("confirmation_treated_loaded") is not False:
        raise RuntimeError("invalid preflight leakage flags")
    verify_frozen_inputs(plan, args)
    validate_plan_contract(plan, args)
    if (args.root / "DRY_RUN_COMPLETE.json").exists():
        raise FileExistsError("dry-run already complete")
    fingerprints, _canonical, _structure_audit = load_structure_map(args.structure_map)
    rows = build_row_map(read_conditions(args.conditions))
    units = [(str(x[0]), float(x[1])) for x in plan["common_universe"]["units"]]
    result_root = args.root / "dry_run"
    result_root.mkdir(exist_ok=False)
    checks: list[dict[str, Any]] = []
    store = ProfileStore(args.group_counts)
    confirmation_group_ids = {
        row.group_id for row in rows.values()
        if row.drug in set(plan["folds"][row.cell_line]["endpoints"]["own_cell_unseen"]["test_drugs"])
    }
    try:
        for target in CELL_LINES:
            source_cells = list(plan["folds"][target]["source_cell_lines"])
            for endpoint in ENDPOINTS:
                spec = plan["folds"][target]["endpoints"][endpoint]
                train_drugs = set(spec["train_drugs"])
                valid_drugs = set(spec["valid_drugs"])
                all_source_drugs = train_drugs | valid_drugs
                hvg, hvg_audit = fit_vehicle_hvg(store, rows, source_cells)
                profile = load_source_profiles(store, rows, source_cells, all_source_drugs, hvg, set(units))
                # Constructing examples verifies that the held/query Vehicle
                # baseline and support/held response identities are available.
                train_rows = source_examples(profile, source_cells, train_drugs, units, fingerprints, include_truth=False)
                valid_rows = source_examples(profile, source_cells, valid_drugs, units, fingerprints, include_truth=True)
                if len(valid_rows.drug) == 0:
                    raise RuntimeError(f"empty source validation rows: {target}/{endpoint}")
                initial, initial_meta, initial_assign = build_oof_cfra(profile, source_cells, train_drugs, valid_drugs, units, token=f"{target}|{endpoint}|initial")
                refit, refit_meta, refit_assign = build_oof_cfra(profile, source_cells, all_source_drugs, valid_drugs, units, token=f"{target}|{endpoint}|refit")
                if initial.shape != train_rows.raw_label.shape:
                    raise RuntimeError(f"initial dry-run label shape mismatch {target}/{endpoint}: {initial.shape} != {train_rows.raw_label.shape}")
                refit_rows = source_examples(profile, source_cells, all_source_drugs, units, fingerprints, include_truth=False)
                if refit.shape != refit_rows.raw_label.shape:
                    raise RuntimeError(f"refit dry-run label shape mismatch {target}/{endpoint}: {refit.shape} != {refit_rows.raw_label.shape}")
                if not np.isfinite(initial).all() or not np.isfinite(refit).all():
                    raise RuntimeError(f"non-finite dry-run labels: {target}/{endpoint}")
                checks.append({
                    "target_cell_line": target, "endpoint": endpoint,
                    "source_cell_lines": source_cells,
                    "n_train_rows": int(len(train_rows.drug)), "n_valid_rows": int(len(valid_rows.drug)),
                    "n_refit_rows": int(len(refit_rows.drug)),
                    "initial_shape": list(initial.shape), "refit_shape": list(refit.shape),
                    "initial_sha256": array_sha256(initial), "refit_sha256": array_sha256(refit),
                    "n_initial_teacher_folds": len(initial_meta), "n_refit_teacher_folds": len(refit_meta),
                    "initial_fold_assignment": initial_assign, "refit_fold_assignment": refit_assign,
                    "hvg_audit": hvg_audit,
                    "confirmation_treated_loaded": False,
                })
                write_json(result_root / f"{endpoint}_{target}_teacher_audit.json", {"initial": initial_meta, "refit": refit_meta, "confirmation_treated_loaded": False})
    finally:
        source_groups = sorted(set(store.read_group_ids))
        source_controls = sorted(set(store.read_control_ids))
        store.close()
    leaked_confirmation = sorted(set(source_groups) & confirmation_group_ids)
    if leaked_confirmation:
        raise RuntimeError(f"dry-run read confirmation treated groups: {leaked_confirmation[:5]}")
    if not checks or len(checks) != len(CELL_LINES) * len(ENDPOINTS):
        raise RuntimeError(f"dry-run coverage incomplete: {len(checks)}")
    audit = {
        "version": VERSION, "stage": "dry_run", "status": "COMPLETE",
        "checks": checks, "fit_group_reads": source_groups, "vehicle_control_reads": source_controls,
        "confirmation_group_reads": [], "confirmation_treated_loaded": False,
        "label_paths": ["initial", "refit"], "endpoints": list(ENDPOINTS),
        "same_cell_base_calibration_only": True,
    }
    write_json(args.root / "DRY_RUN_COMPLETE.json", audit)
    print(json.dumps({"status": "DRY_RUN_COMPLETE", "root": str(args.root), "checks": len(checks), "confirmation_treated_loaded": False}, sort_keys=True), flush=True)


def fit(args: argparse.Namespace) -> None:
    observed_parameter_count = sum(p.numel() for p in Student().parameters())
    if observed_parameter_count != PARAMETER_COUNT:
        raise RuntimeError(f"Student parameter count drift: {observed_parameter_count} != {PARAMETER_COUNT}")
    plan = read_json(args.root / "PLAN.json")
    if plan.get("confirmation_treated_loaded") is not False:
        raise RuntimeError("invalid preflight leakage flags")
    dry_path = args.root / "DRY_RUN_COMPLETE.json"
    if not dry_path.exists() or read_json(dry_path).get("status") != "COMPLETE":
        raise RuntimeError("formal fit requires a completed source-only dry-run")
    verify_frozen_inputs(plan, args)
    validate_plan_contract(plan, args)
    if (args.root / "FIT_COMPLETE.json").exists():
        raise FileExistsError("fit already complete")
    fingerprints, canonical, structure_audit = load_structure_map(args.structure_map)
    rows_list = read_conditions(args.conditions)
    rows = build_row_map(rows_list)
    units = [(str(x[0]), float(x[1])) for x in plan["common_universe"]["units"]]
    device = choose_device(args.device)
    fit_root = args.root / "fit"; fit_root.mkdir(parents=True, exist_ok=False)
    store = ProfileStore(args.group_counts)
    audit: dict[str, Any] = {"version": VERSION, "stage": "fit", "status": "RUNNING", "surface": "own_cell_unseen", "confirmation_treated_loaded": False, "device": str(device), "checkpoints": [], "artifacts": [], "folds": {}}
    confirmation_group_ids = {
        row.group_id for row in rows.values()
        if row.drug in set(plan["folds"][row.cell_line]["endpoints"]["own_cell_unseen"]["test_drugs"])
    }
    try:
        for target in CELL_LINES:
            source_cells = list(plan["folds"][target]["source_cell_lines"])
            audit["folds"][target] = {}
            for endpoint in ENDPOINTS:
                spec = plan["folds"][target]["endpoints"][endpoint]
                train_drugs = set(spec["train_drugs"])
                valid_drugs = set(spec["valid_drugs"])
                all_source_drugs = train_drugs | valid_drugs
                # HGV and all response/baseline caches are built from source only.
                # The outer HGV is fixed from unique source Vehicle groups,
                # independent of the endpoint's train/validation drug list.
                hvg, hvg_audit = fit_vehicle_hvg(store, rows, source_cells)
                profile = load_source_profiles(store, rows, source_cells, all_source_drugs, hvg, set(units))
                endpoint_root = fit_root / endpoint / target; endpoint_root.mkdir(parents=True, exist_ok=False)
                hvg_path = endpoint_root / "source_vehicle_hvg2000.txt"
                np.savetxt(hvg_path, hvg, fmt="%d")
                train_rows = source_examples(profile, source_cells, train_drugs, units, fingerprints, include_truth=False)
                valid_rows = source_examples(profile, source_cells, valid_drugs, units, fingerprints, include_truth=True)
                if len(valid_rows.drug) == 0:
                    raise RuntimeError(f"empty source validation rows: {target}/{endpoint}")
                stats = fit_stats(train_rows)
                stats_path = endpoint_root / "normalization.npz"
                save_stats(stats_path, stats)
                initial_cfra, initial_teacher_meta, initial_assign = build_oof_cfra(profile, source_cells, train_drugs, set(spec["valid_drugs"]), units, token=f"{target}|{endpoint}|initial")
                if initial_cfra.shape != train_rows.raw_label.shape:
                    raise RuntimeError(f"OOF CFRA shape mismatch {initial_cfra.shape} != {train_rows.raw_label.shape}")
                initial_targets_path = endpoint_root / "initial_training_targets.npz"
                np.savez_compressed(initial_targets_path, raw=train_rows.raw_label, cfra=initial_cfra, drug=train_rows.drug, dose=train_rows.dose)
                initial_teacher_path = endpoint_root / "initial_teacher_audit.json"
                write_json(initial_teacher_path, {"fold_assignment": initial_assign, "folds": initial_teacher_meta, "confirmation_treated_loaded": False})
                selection_rows: list[dict[str, Any]] = []
                selected_epochs: dict[str, dict[str, int]] = {arm: {} for arm in ("raw", "cfra")}
                initial_dir = endpoint_root / "initial"; initial_dir.mkdir()
                for arm, label in (("raw", train_rows.raw_label), ("cfra", initial_cfra)):
                    for seed in SEEDS:
                        model, logs, best_epoch, init_hash = fit_student(train_rows, valid_rows, label, fingerprints, stats, seed, device)
                        seed_dir = initial_dir / arm / f"seed_{seed}"; seed_dir.mkdir(parents=True, exist_ok=False)
                        torch.save({"version": VERSION, "endpoint": endpoint, "target": target, "arm": arm, "seed": seed, "best_epoch": best_epoch, "state_dict": model.state_dict()}, seed_dir / "student.pt")
                        write_csv(seed_dir / "training_log.csv", logs)
                        selected_epochs[arm][str(seed)] = best_epoch
                        selection_rows.append({"arm": arm, "seed": seed, "best_epoch": best_epoch, "init_state_sha256": init_hash, "criterion": "same-cell calibration held Raw truth", "confirmation_treated_loaded": False})
                write_csv(endpoint_root / "initial_epoch_selection.csv", selection_rows)
                # Freeze the selected epochs, then fit both arms on all available source drugs.
                refit_drugs = all_source_drugs
                refit_cfra, refit_teacher_meta, refit_assign = build_oof_cfra(profile, source_cells, refit_drugs, set(spec["valid_drugs"]), units, token=f"{target}|{endpoint}|refit")
                refit_rows = source_examples(profile, source_cells, refit_drugs, units, fingerprints, include_truth=False)
                if refit_cfra.shape != refit_rows.raw_label.shape:
                    raise RuntimeError("refit OOF CFRA shape mismatch")
                refit_targets_path = endpoint_root / "refit_training_targets.npz"
                np.savez_compressed(refit_targets_path, raw=refit_rows.raw_label, cfra=refit_cfra, drug=refit_rows.drug, dose=refit_rows.dose)
                refit_teacher_path = endpoint_root / "refit_teacher_audit.json"
                write_json(refit_teacher_path, {"fold_assignment": refit_assign, "folds": refit_teacher_meta, "confirmation_treated_loaded": False, "refit_calibration_note": "calibration may overlap refit teacher fit by registered all-source-refit design; held drug fold remains excluded"})
                refit_dir = endpoint_root / "refit"; refit_dir.mkdir()
                for arm, label in (("raw", refit_rows.raw_label), ("cfra", refit_cfra)):
                    for seed in SEEDS:
                        epochs = int(selected_epochs[arm][str(seed)]) + 1
                        model, logs, fixed_epoch, init_hash = fit_student(refit_rows, valid_rows, label, fingerprints, stats, seed, device, fixed_epochs=epochs)
                        seed_dir = refit_dir / arm / f"seed_{seed}"; seed_dir.mkdir(parents=True, exist_ok=False)
                        ckpt = seed_dir / "student.pt"
                        torch.save({"version": VERSION, "endpoint": endpoint, "target": target, "arm": arm, "seed": seed, "fixed_epochs": epochs, "source_initial_best_epoch": epochs - 1, "state_dict": model.state_dict()}, ckpt)
                        write_csv(seed_dir / "training_log.csv", logs)
                        audit["checkpoints"].append({"target": target, "endpoint": endpoint, "arm": arm, "seed": seed, "path": str(ckpt.resolve()), "sha256": sha256(ckpt), "fixed_epochs": epochs, "confirmation_treated_loaded": False})
                for artifact_name, artifact_path in (("hvg", hvg_path), ("normalization", stats_path), ("initial_targets", initial_targets_path), ("refit_targets", refit_targets_path), ("initial_teacher_audit", initial_teacher_path), ("refit_teacher_audit", refit_teacher_path)):
                    audit["artifacts"].append({"target": target, "endpoint": endpoint, "kind": artifact_name, "path": str(artifact_path.resolve()), "sha256": sha256(artifact_path)})
                audit["folds"][target][endpoint] = {"source_cells": source_cells, "train_drugs_initial": sorted(train_drugs), "valid_drugs": sorted(valid_drugs), "refit_drugs": sorted(refit_drugs), "hvg": hvg_audit, "n_train_rows": len(train_rows.drug), "n_valid_rows": len(valid_rows.drug), "n_refit_rows": len(refit_rows.drug), "confirmation_treated_loaded": False}
    finally:
        audit["source_group_reads"] = sorted(set(store.read_group_ids))
        audit["source_control_reads"] = sorted(set(store.read_control_ids))
        store.close()
    leaked_confirmation = sorted(set(audit["source_group_reads"]) & confirmation_group_ids)
    if leaked_confirmation:
        raise RuntimeError(f"fit read confirmation treated groups: {leaked_confirmation[:5]}")
    audit["confirmation_group_reads"] = []
    audit["same_cell_base_calibration_only"] = True
    audit["confirmation_treated_loaded"] = False
    audit["normalizer_shared_between_arms"] = True
    audit["checkpoint_criterion_shared_between_arms"] = True
    audit["oof_label_exclusion"] = "held drug-fold excluded from teacher HVG input scope (Vehicle-only outer coordinates), fit, normalizer, IMCEB, calibration, foreign pool; no teacher validation"
    audit["status"] = "COMPLETE"
    write_json(args.root / "FIT_COMPLETE.json", audit)
    print(json.dumps({"status": "FIT_COMPLETE", "root": str(args.root), "device": str(device), "confirmation_treated_loaded": False}, sort_keys=True), flush=True)


def load_checkpoint(path: Path, device: torch.device, *, target: str, endpoint: str, arm: str, seed: int) -> Student:
    value = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"version": VERSION, "target": target, "endpoint": endpoint, "arm": arm, "seed": seed}
    mismatches = {key: (value.get(key), expected_value) for key, expected_value in expected.items() if value.get(key) != expected_value}
    if mismatches or "state_dict" not in value or "fixed_epochs" not in value:
        raise RuntimeError(f"checkpoint payload mismatch for {path}: {mismatches}")
    model = Student().to(device); model.load_state_dict(value["state_dict"]); model.eval(); return model


def bootstrap(values: dict[str, float], token: str) -> dict[str, Any]:
    keys = sorted(values)
    x = np.asarray([values[k] for k in keys], dtype=np.float64)
    if len(x) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_drugs": 0, "bootstrap_draws": BOOTSTRAP_ROUNDS, "bootstrap_unit": "drug"}
    rng = np.random.default_rng(stable_int(SEED, "bootstrap", token))
    draws = x[rng.integers(0, len(x), size=(BOOTSTRAP_ROUNDS, len(x)))].mean(axis=1)
    return {"estimate": float(np.mean(x)), "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)), "n_drugs": int(len(x)), "bootstrap_draws": BOOTSTRAP_ROUNDS, "bootstrap_unit": "drug"}


def paired_bootstrap(raw_values: dict[str, float], cfra_values: dict[str, float], token: str) -> dict[str, dict[str, Any]]:
    """One drug-index draw matrix is applied identically to Raw and CFRA."""
    keys = sorted(raw_values)
    if not keys or set(keys) != set(cfra_values):
        raise RuntimeError("paired bootstrap requires an identical, non-empty drug universe")
    raw = np.asarray([raw_values[key] for key in keys], dtype=np.float64)
    cfra = np.asarray([cfra_values[key] for key in keys], dtype=np.float64)
    if not np.isfinite(raw).all() or not np.isfinite(cfra).all():
        raise RuntimeError("paired bootstrap received non-finite values")
    draws = np.random.default_rng(stable_int(SEED, "paired-bootstrap", token)).integers(0, len(keys), size=(BOOTSTRAP_ROUNDS, len(keys)))
    def summarize(values: np.ndarray) -> dict[str, Any]:
        sampled = values[draws].mean(axis=1)
        return {"estimate": float(values.mean()), "ci_low": float(np.quantile(sampled, .025)), "ci_high": float(np.quantile(sampled, .975)), "n_drugs": len(keys), "bootstrap_draws": BOOTSTRAP_ROUNDS, "bootstrap_unit": "paired_drug"}
    return {"raw": summarize(raw), "cfra": summarize(cfra), "delta": summarize(cfra - raw)}


def hierarchical_bootstrap(values: dict[str, dict[str, float]], token: str) -> dict[str, Any]:
    targets = sorted(values)
    target_points = {target: float(np.mean(list(values[target].values()))) for target in targets}
    point = float(np.mean(list(target_points.values())))
    rng = np.random.default_rng(stable_int(SEED, "macro-bootstrap", token))
    draws: list[float] = []
    for _ in range(BOOTSTRAP_ROUNDS):
        chosen_targets = rng.integers(0, len(targets), size=len(targets))
        target_means = []
        for index in chosen_targets:
            target = targets[int(index)]; x = np.asarray(list(values[target].values()), dtype=np.float64)
            target_means.append(float(np.mean(x[rng.integers(0, len(x), size=len(x))])))
        draws.append(float(np.mean(target_means)))
    return {"estimate": point, "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)), "n_targets": len(targets), "target_lines": targets, "target_point_estimates": target_points, "bootstrap_draws": BOOTSTRAP_ROUNDS, "bootstrap_unit": "target_cell_line_then_drug"}


def test(args: argparse.Namespace) -> None:
    fit_audit = read_json(args.root / "FIT_COMPLETE.json")
    if fit_audit.get("status") != "COMPLETE" or fit_audit.get("surface") != "own_cell_unseen" or fit_audit.get("confirmation_treated_loaded") is not False:
        raise RuntimeError("fit audit is not confirmation-drug isolated")
    if (args.root / "TEST_COMPLETE.json").exists():
        raise FileExistsError("test already complete")
    plan = read_json(args.root / "PLAN.json")
    verify_frozen_inputs(plan, args)
    validate_plan_contract(plan, args)
    fingerprints, canonical, structure_audit = load_structure_map(args.structure_map)
    rows = build_row_map(read_conditions(args.conditions))
    units = [(str(x[0]), float(x[1])) for x in plan["common_universe"]["units"]]
    device = choose_device(args.device)
    # Verify the complete frozen model surface before opening the H5 treated
    # matrix.  A partially missing later fold must never cause an earlier
    # target profile to be read.
    all_checkpoint_paths: list[Path] = []
    for target_name in CELL_LINES:
        for endpoint_name in ENDPOINTS:
            endpoint_root = args.root / "fit" / endpoint_name / target_name
            for arm_name in ("raw", "cfra"):
                all_checkpoint_paths.extend(endpoint_root / "refit" / arm_name / f"seed_{seed}" / "student.pt" for seed in SEEDS)
    missing_checkpoints = [str(path) for path in all_checkpoint_paths if not path.exists()]
    if missing_checkpoints:
        raise RuntimeError(f"all checkpoint freeze verification failed before target read: {missing_checkpoints[:5]}")
    recorded = {str(item["path"]): str(item["sha256"]) for item in fit_audit.get("checkpoints", [])}
    resolved_checkpoint_paths = {str(path.resolve()) for path in all_checkpoint_paths}
    if set(recorded) != resolved_checkpoint_paths:
        raise RuntimeError("FIT_COMPLETE checkpoint manifest does not exactly cover the frozen surface")
    bad_hashes = [str(path) for path in all_checkpoint_paths if sha256(path) != recorded[str(path.resolve())]]
    if bad_hashes:
        raise RuntimeError(f"checkpoint hash mismatch before confirmation read: {bad_hashes[:5]}")
    artifact_records = {(str(item["target"]), str(item["endpoint"]), str(item["kind"])): item for item in fit_audit.get("artifacts", [])}
    expected_artifact_keys = {(target, endpoint, kind) for target in CELL_LINES for endpoint in ENDPOINTS for kind in ("hvg", "normalization", "initial_targets", "refit_targets", "initial_teacher_audit", "refit_teacher_audit")}
    if set(artifact_records) != expected_artifact_keys:
        raise RuntimeError("FIT_COMPLETE artifact manifest does not exactly cover the frozen surface")
    bad_artifacts = [item["path"] for item in artifact_records.values() if not Path(item["path"]).is_file() or sha256(Path(item["path"])) != item["sha256"]]
    if bad_artifacts:
        raise RuntimeError(f"fit artifact hash mismatch before confirmation read: {bad_artifacts[:3]}")
    store = ProfileStore(args.group_counts)
    test_root = args.root / "test"; test_root.mkdir(exist_ok=False)
    condition_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    drug_rows: list[dict[str, Any]] = []
    confirmation_read_events: list[dict[str, Any]] = []
    read_sequence = 0
    endpoint_summaries: dict[str, Any] = {}
    try:
        # No target access occurs before all frozen checkpoint paths are checked.
        for target in CELL_LINES:
            for endpoint in ENDPOINTS:
                spec = plan["folds"][target]["endpoints"][endpoint]
                endpoint_root = args.root / "fit" / endpoint / target
                hvg = np.loadtxt(endpoint_root / "source_vehicle_hvg2000.txt", dtype=np.int64)
                if hvg.shape != (N_HVG,):
                    raise RuntimeError(f"bad HGV shape {target}/{endpoint}: {hvg.shape}")
                stats = load_stats(endpoint_root / "normalization.npz")
                model_paths = {arm: [endpoint_root / "refit" / arm / f"seed_{seed}" / "student.pt" for seed in SEEDS] for arm in ("raw", "cfra")}
                for path in model_paths["raw"] + model_paths["cfra"]:
                    if not path.exists():
                        raise RuntimeError(f"missing frozen checkpoint before target read: {path}")
                source_cells = list(plan["folds"][target]["source_cell_lines"])
                test_drugs = set(spec["test_drugs"])
                foreign_drugs = set(spec["foreign_drugs"])
                # Target treated and Vehicle profiles are first read only here, after all checkpoint checks.
                target_profile: dict[tuple[str, str, float, str], tuple[np.ndarray, np.ndarray]] = {}
                for drug, dose in units:
                    # Cold-drug scoring uses base/calibration target-line
                    # profiles as the foreign-null donor pool, so those
                    # profiles are also opened only after the global freeze.
                    if drug not in (test_drugs | foreign_drugs):
                        continue
                    for rep in REPS:
                        row = rows[(target, drug, dose, rep)]
                        if drug in test_drugs:
                            read_sequence += 1
                            confirmation_read_events.append({"event_sequence": read_sequence, "stage": "test_after_global_checkpoint_hash_freeze", "target_cell_line": target, "drug": drug, "dose_nM": dose, "replicate": rep, "plate": row.plate, "treated_group_id": row.group_id, "vehicle_control_group_id": row.control_group_id})
                        target_profile[(target, drug, dose, rep)] = store.profile(row, hvg, read_treated=True)
                models = {arm: [load_checkpoint(path, device, target=target, endpoint=endpoint, arm=arm, seed=seed) for path, seed in zip(model_paths[arm], SEEDS)] for arm in ("raw", "cfra")}
                endpoint_condition_rows: list[dict[str, Any]] = []
                for drug, dose in units:
                    if drug not in test_drugs:
                        continue
                    support, held = support_repeat(drug, dose)
                    # Query/held-repeat Vehicle baseline is the sole Student
                    # input baseline; Raw and CFRA use this identical target
                    # input and differ only in their frozen training labels.
                    raw_input = StudentRows(np.asarray([target]), np.asarray([drug]), np.asarray([dose], dtype=np.float32), np.asarray([baseline(target_profile, target, drug, dose, held)], dtype=np.float32), np.zeros((1, N_HVG), dtype=np.float32), np.zeros((1, N_HVG), dtype=np.float32))
                    truth = response(target_profile, target, drug, dose, held)
                    donor_ids, foreign = foreign_vectors(target_profile, target, drug, dose, held, foreign_drugs, f"target|{target}|{endpoint}|{drug}|{dose}|{support}|{held}")
                    query_baseline = raw_input.baseline[0]
                    if not np.isfinite(truth).all() or not np.isfinite(query_baseline).all() or not np.isfinite(np.vstack(foreign)).all():
                        raise RuntimeError(f"non-finite truth/baseline/foreign vector: {target}/{drug}/{dose}")
                    predictions: dict[str, list[np.ndarray]] = {arm: [predict_model(model, raw_input, fingerprints, stats, device)[0] for model in models[arm]] for arm in ("raw", "cfra")}
                    for arm in ("raw", "cfra"):
                        for seed, pred in zip(SEEDS, predictions[arm]):
                            same, foreign_pcc, excess, foreign_z = same_foreign_metrics(pred, truth, foreign)
                            metric_values = (same, fisher(same), float(np.mean((pred - truth) ** 2)), foreign_pcc, float(np.mean(foreign_z)), excess)
                            if not np.isfinite(metric_values).all():
                                raise RuntimeError(f"non-finite seed metric: {target}/{drug}/{dose}/{arm}/seed{seed}")
                            seed_rows.append({"target_cell_line": target, "endpoint": endpoint, "drug": drug, "dose_nM": dose, "support_repeat": support, "held_repeat": held, "baseline_repeat": held, "baseline_role": "held/query Vehicle", "arm": arm, "seed": seed, "same_pcc": same, "z_same": fisher(same), "raw_target_mse": float(np.mean((pred - truth) ** 2)), "foreign_pcc": foreign_pcc, "foreign_fisher_z_mean": float(np.mean(foreign_z)), "foreign_fisher_z_by_donor": ";".join(f"{value:.9g}" for value in foreign_z), "same_foreign_excess_fisher_z": excess, "same_foreign_definition": "fisher(same_pcc) - mean_i fisher(foreign_pcc_i)", "prediction_norm": float(np.linalg.norm(pred)), "truth_norm": float(np.linalg.norm(truth)), "prediction_truth_norm_ratio": float(np.linalg.norm(pred) / max(np.linalg.norm(truth), 1e-12)), "foreign_drugs": ";".join(sorted(donor_ids))})
                        pred = np.mean(np.stack(predictions[arm]), axis=0)
                        same, foreign_pcc, excess, foreign_z = same_foreign_metrics(pred, truth, foreign)
                        metric_values = (same, fisher(same), float(np.mean((pred - truth) ** 2)), foreign_pcc, float(np.mean(foreign_z)), excess)
                        if not np.isfinite(metric_values).all():
                            raise RuntimeError(f"non-finite ensemble metric: {target}/{drug}/{dose}/{arm}")
                        endpoint_condition_rows.append({"target_cell_line": target, "endpoint": endpoint, "drug": drug, "dose_nM": dose, "support_repeat": support, "held_repeat": held, "baseline_repeat": held, "baseline_role": "held/query Vehicle", "arm": arm, "same_pcc": same, "z_same": fisher(same), "raw_target_mse": float(np.mean((pred - truth) ** 2)), "foreign_pcc": foreign_pcc, "foreign_fisher_z_mean": float(np.mean(foreign_z)), "foreign_fisher_z_by_donor": ";".join(f"{value:.9g}" for value in foreign_z), "same_foreign_excess_fisher_z": excess, "same_foreign_definition": "fisher(same_pcc) - mean_i fisher(foreign_pcc_i)", "prediction_norm": float(np.linalg.norm(pred)), "truth_norm": float(np.linalg.norm(truth)), "prediction_truth_norm_ratio": float(np.linalg.norm(pred) / max(np.linalg.norm(truth), 1e-12)), "foreign_drugs": ";".join(sorted(donor_ids)), "seed_aggregation": "mean prediction across 3 seeds before metric"})
                        endpoint_condition_rows[-1].update({"surface": "own_cell_unseen", "fit_cells": target, "evaluation_cell": target, "parameter_count_per_arm": PARAMETER_COUNT, "full_target_pcc": same, "delta_pcc": same, "delta_coordinate": "response = treated pseudo-bulk minus same plate/cell-line/repeat Vehicle", "z_foreign_F": float(np.mean(foreign_z)), "E_SF": excess, "truth_sha256": array_sha256(truth), "baseline_sha256": array_sha256(query_baseline), "foreign_vector_sha256": array_sha256(np.vstack(foreign))})
                condition_rows.extend(endpoint_condition_rows)
                for arm in ("raw", "cfra"):
                    for drug in sorted({str(row["drug"]) for row in endpoint_condition_rows if row["arm"] == arm}):
                        selected = [row for row in endpoint_condition_rows if row["arm"] == arm and row["drug"] == drug]
                        item = {key: float(np.mean([float(row[key]) for row in selected])) for key in ("same_pcc", "z_same", "raw_target_mse", "foreign_pcc", "foreign_fisher_z_mean", "same_foreign_excess_fisher_z", "prediction_norm", "truth_norm", "prediction_truth_norm_ratio")}
                        drug_rows.append({"surface": "own_cell_unseen", "fit_cells": target, "evaluation_cell": target, "parameter_count_per_arm": PARAMETER_COUNT, "target_cell_line": target, "endpoint": endpoint, "drug": drug, "arm": arm, "n_doses": len(selected), "aggregation": "3-seed mean prediction -> dose mean -> drug", **item})
                        drug_rows[-1].update({"full_target_pcc": drug_rows[-1]["same_pcc"], "delta_pcc": drug_rows[-1]["same_pcc"], "z_foreign_F": drug_rows[-1]["foreign_fisher_z_mean"], "E_SF": drug_rows[-1]["same_foreign_excess_fisher_z"]})
                endpoint_summaries.setdefault(endpoint, {})[target] = {arm: {"same_foreign_excess_fisher_z": bootstrap({row["drug"]: float(row["same_foreign_excess_fisher_z"]) for row in drug_rows if row["target_cell_line"] == target and row["endpoint"] == endpoint and row["arm"] == arm}, f"{endpoint}|{target}|{arm}|same") for arm in ("raw", "cfra")}}
    finally:
        read_groups = sorted(set(store.read_group_ids)); read_controls = sorted(set(store.read_control_ids)); store.close()
    confirmation_drugs = set(plan["folds"][CELL_LINES[0]]["endpoints"]["own_cell_unseen"]["test_drugs"])
    confirmation_read_rows = confirmation_read_events
    expected_confirmation_ids = {
        row.group_id for row in rows.values()
        if row.cell_line in CELL_LINES and row.drug in confirmation_drugs
        and (row.drug, float(row.dose)) in set(units) and row.replicate in REPS
    }
    observed_confirmation_ids = {int(row["treated_group_id"]) for row in confirmation_read_rows}
    if observed_confirmation_ids != expected_confirmation_ids:
        raise RuntimeError("confirmation read audit does not exactly cover the frozen test rows")
    if len(confirmation_read_rows) != len(expected_confirmation_ids):
        raise RuntimeError("duplicate or missing confirmation treated-read events")
    expected_test_read_ids = {
        row.group_id for row in rows.values()
        if row.cell_line in CELL_LINES and (row.drug, float(row.dose)) in set(units) and row.replicate in REPS
    }
    unexpected_test_reads = sorted(set(read_groups) - expected_test_read_ids)
    if unexpected_test_reads:
        raise RuntimeError(f"test opened unexpected treated group IDs: {unexpected_test_reads[:5]}")
    condition_index: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        condition_index[(str(row["target_cell_line"]), str(row["endpoint"]), str(row["drug"]), float(row["dose_nM"]))].append(row)
    for key, pair in condition_index.items():
        if len(pair) != 2 or {row["arm"] for row in pair} != {"raw", "cfra"}:
            raise RuntimeError(f"Raw/CFRA condition pairing failure: {key}")
        reference = pair[0]
        for field in ("support_repeat", "held_repeat", "baseline_repeat", "foreign_drugs", "truth_sha256", "baseline_sha256", "foreign_vector_sha256"):
            if any(row[field] != reference[field] for row in pair[1:]):
                raise RuntimeError(f"Raw/CFRA condition identity mismatch {field}: {key}")
        numeric_fields = ("same_pcc", "full_target_pcc", "delta_pcc", "z_same", "z_foreign_F", "E_SF", "raw_target_mse")
        if not np.isfinite(np.asarray([[float(row[field]) for field in numeric_fields] for row in pair], dtype=np.float64)).all():
            raise RuntimeError(f"non-finite paired condition metric: {key}")
    expected_seed_rows = len(condition_rows) * len(SEEDS)
    if len(seed_rows) != expected_seed_rows or len({(row["target_cell_line"], row["endpoint"], row["drug"], row["dose_nM"], row["arm"], row["seed"]) for row in seed_rows}) != expected_seed_rows:
        raise RuntimeError("seed metric coverage/uniqueness failure")
    write_csv(test_root / "condition_seed_metrics.csv", seed_rows)
    write_csv(test_root / "condition_metrics_seed_averaged.csv", condition_rows)
    write_csv(test_root / "drug_metrics.csv", drug_rows)
    write_csv(test_root / "CONFIRMATION_READ_AUDIT.csv", confirmation_read_rows)
    write_csv(test_root / "RAW_TARGET_PCC_TABLE.csv", [
        {key: row[key] for key in ("surface", "fit_cells", "evaluation_cell", "parameter_count_per_arm", "target_cell_line", "endpoint", "drug", "arm", "n_doses", "full_target_pcc", "delta_pcc", "z_same", "raw_target_mse")}
        for row in drug_rows
    ])
    summary_rows: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    macro_values: dict[str, dict[str, float]] = {endpoint: {} for endpoint in ENDPOINTS}
    target_summary: dict[str, Any] = {}
    for endpoint in ENDPOINTS:
        target_summary[endpoint] = {}
        for target in CELL_LINES:
            raw_values = {row["drug"]: float(row["E_SF"]) for row in drug_rows if row["target_cell_line"] == target and row["endpoint"] == endpoint and row["arm"] == "raw"}
            cfra_values = {row["drug"]: float(row["E_SF"]) for row in drug_rows if row["target_cell_line"] == target and row["endpoint"] == endpoint and row["arm"] == "cfra"}
            paired = paired_bootstrap(raw_values, cfra_values, f"{endpoint}|{target}|E_SF")
            raw_stat, cfra_stat, delta_stat = paired["raw"], paired["cfra"], paired["delta"]
            delta_values = {drug: cfra_values[drug] - raw_values[drug] for drug in sorted(raw_values)}
            target_summary[endpoint][target] = {"raw": raw_stat, "cfra": cfra_stat, "delta": delta_stat}
            summary_rows.extend([{"surface": "own_cell_unseen", "endpoint": endpoint, "target_cell_line": target, "arm": "raw", "metric": "E_SF", **raw_stat}, {"surface": "own_cell_unseen", "endpoint": endpoint, "target_cell_line": target, "arm": "cfra", "metric": "E_SF", **cfra_stat}, {"surface": "own_cell_unseen", "endpoint": endpoint, "target_cell_line": target, "arm": "cfra_minus_raw", "metric": "CFRA-Student minus Raw-Student E_SF", **delta_stat}])
            macro_values[endpoint][target] = delta_values
        macro = hierarchical_bootstrap(macro_values[endpoint], f"{endpoint}|macro|delta")
        contrasts.append({"endpoint": endpoint, "comparison": "CFRA-Student minus Raw-Student", "metric": "Same-Foreign excess Fisher-z", **macro})
    fidelity_contrasts: list[dict[str, Any]] = []
    for metric in ("full_target_pcc", "delta_pcc", "z_same", "raw_target_mse"):
        by_target: dict[str, dict[str, float]] = {}
        for target in CELL_LINES:
            raw_values = {row["drug"]: float(row[metric]) for row in drug_rows if row["target_cell_line"] == target and row["endpoint"] == "own_cell_unseen" and row["arm"] == "raw"}
            cfra_values = {row["drug"]: float(row[metric]) for row in drug_rows if row["target_cell_line"] == target and row["endpoint"] == "own_cell_unseen" and row["arm"] == "cfra"}
            delta_values = {drug: cfra_values[drug] - raw_values[drug] for drug in sorted(set(raw_values) & set(cfra_values))}
            if set(raw_values) != set(cfra_values):
                raise RuntimeError(f"Raw/CFRA drug pairing failure for {metric}/{target}")
            by_target[target] = delta_values
            fidelity_contrasts.append({"endpoint": "own_cell_unseen", "target_cell_line": target, "comparison": "CFRA-Student minus Raw-Student", "metric": metric, **bootstrap(delta_values, f"own|{target}|{metric}|delta")})
        fidelity_contrasts.append({"endpoint": "own_cell_unseen", "target_cell_line": "MACRO", "comparison": "CFRA-Student minus Raw-Student", "metric": metric, **hierarchical_bootstrap(by_target, f"own|macro|{metric}|delta")})
    write_csv(test_root / "target_summary.csv", summary_rows)
    write_csv(test_root / "target_macro_contrasts.csv", contrasts)
    write_csv(test_root / "RAW_TARGET_CONTRASTS.csv", fidelity_contrasts)
    gates: dict[str, str] = {}
    for endpoint in ENDPOINTS:
        macro = next(row for row in contrasts if row["endpoint"] == endpoint)
        fold_lows = [target_summary[endpoint][target]["delta"]["ci_low"] for target in CELL_LINES]
        gates[endpoint] = "RETROSPECTIVE_SUPPORTIVE_OWN_CELL" if macro["ci_low"] > 0 and all(value > 0 for value in fold_lows) else ("RETROSPECTIVE_NO_GO_OWN_CELL" if macro["ci_high"] < 0 else "RETROSPECTIVE_INCONCLUSIVE_OWN_CELL")
    audit = {"version": VERSION, "stage": "test", "status": "COMPLETE", "dataset": "sci-Plex3 Zenodo 7041849", "surface": "own_cell_unseen", "evidence_status": "leakage-controlled retrospective; confirmation profiles were viewed in prior project experiments", "confirmation_treated_loaded": True, "confirmation_read_after_checkpoint_hash_freeze": True, "input": "ECFP4 2048 + log10 dose + same-cell held/query Vehicle baseline projected to frozen same-cell Vehicle-only HGV2000", "output": "same-cell HGV2000 response", "architecture": "4049 -> 512 -> 128 -> 2000", "parameter_count_per_arm": PARAMETER_COUNT, "seeds": list(SEEDS), "same_foreign": {"definition": "fisher(same_pcc) - mean_i fisher(foreign_pcc_i)", "foreign_fisher_computed_per_donor_before_mean": True, "foreign_ids_shared_between_arms": True, "foreign_n": N_FOREIGN, "inference": "3-seed mean prediction -> dose mean -> drug bootstrap", "bootstrap_draws": BOOTSTRAP_ROUNDS}, "target_reads": {"group_ids": read_groups, "control_group_ids": read_controls, "confirmation_group_ids": sorted(observed_confirmation_ids)}, "target_summary": target_summary, "target_macro_contrasts": contrasts, "raw_target_contrasts": fidelity_contrasts, "gate": gates, "raw_phase0_gate_reference": "sciplex3_validation/raw_same_foreign_gate.json"}
    report = ["# sci-Plex3 Student own-cell unseen-drug transfer", "", "This compares Raw-label Student and same-cell drug-OOF CFRA-label Student. Confirmation-drug treated profiles were opened only after all 18 refit checkpoints were hash-verified. Because these profiles had been viewed in prior project experiments, the evidence is retrospective/supportive rather than a new blind confirmation.", "", "| endpoint | cell-line-macro CFRA-Student − Raw-Student Same-Foreign (95% CI) | gate |", "|---|---:|---|"]
    for row in contrasts:
        report.append(f"| {row['endpoint']} | {row['estimate']:.6f} [{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {gates[row['endpoint']]} |")
    report.extend(["", "Per-target results:", "", "| endpoint | target | Raw | CFRA | CFRA − Raw (95% CI) |", "|---|---|---:|---:|---:|"])
    for endpoint in ENDPOINTS:
        for target in CELL_LINES:
            s = target_summary[endpoint][target]
            d = s["delta"]
            report.append(f"| {endpoint} | {target} | {s['raw']['estimate']:.6f} | {s['cfra']['estimate']:.6f} | {d['estimate']:.6f} [{d['ci_low']:.6f}, {d['ci_high']:.6f}] |")
    report.extend(["", "## Leakage and fairness locks", "", "- Each cell line uses only its base drugs for initial fit, calibration drugs for checkpoint selection, and base+calibration for refit; confirmation drugs are excluded until test.", "- HGV2000 is fitted from unique same-cell Vehicle-only control groups; controls do not depend on drug membership and no treated profile is used for feature selection.", "- Each OOF teacher excludes its held drug fold from teacher fit, normalizer, IMCEB shrinkage, weight calibration, foreign donors, and teacher validation.", "- Student Raw/CFRA arms share rows, held/query Vehicle baseline input, input normalization, target normalization, seeds, optimizer, architecture, checkpoint criterion, target truth, and foreign donor IDs.", "- Same-Foreign is fisher(same_pcc) minus the mean of donor-wise fisher(foreign_pcc_i), not fisher(mean foreign PCC).", "- Inference averages the three seed predictions, then dose rows within drug; CIs bootstrap drugs and macro CIs bootstrap target cell lines then drugs.", ""])
    (args.root / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    fairness = ["# Final fairness audit", "", "- Surface: own_cell_unseen; evidence status: leakage-controlled retrospective/supportive.", "- Fit inputs: same-cell base drugs for initial fit, calibration drugs for selection, then base+calibration refit.", "- Confirmation treated profiles: read only in test after all 18 refit checkpoint and frozen artifact SHA-256 values verified.", "- Pairing: Raw and CFRA share every drug-dose key, support/held identity, query Vehicle baseline, Raw truth, and foreign donor vectors.", "- Statistics: three-seed prediction mean, dose mean within drug, then paired drug bootstrap; macro summaries resample cell lines then drugs.", "- Physical-repeat limit: two genuine repeats only; no 2R/three-repeat claim."]
    (args.root / "FINAL_FAIRNESS_AUDIT.md").write_text("\n".join(fairness) + "\n", encoding="utf-8")
    output_paths = [
        test_root / "condition_seed_metrics.csv", test_root / "condition_metrics_seed_averaged.csv", test_root / "drug_metrics.csv", test_root / "CONFIRMATION_READ_AUDIT.csv", test_root / "RAW_TARGET_PCC_TABLE.csv", test_root / "target_summary.csv", test_root / "target_macro_contrasts.csv", test_root / "RAW_TARGET_CONTRASTS.csv", args.root / "RESULTS.md", args.root / "FINAL_FAIRNESS_AUDIT.md",
    ]
    if any(not path.is_file() for path in output_paths):
        raise RuntimeError("test output-completeness gate failed")
    audit["output_hashes"] = {str(path.relative_to(args.root)): sha256(path) for path in output_paths}
    write_json(args.root / "TEST_COMPLETE.json", audit)
    print(json.dumps({"status": "TEST_COMPLETE", "root": str(args.root), "gates": gates, "confirmation_treated_loaded_after_freeze": True}, sort_keys=True), flush=True)


if __name__ == "__main__":
    opts = parse_args()
    {"preflight": preflight, "dry_run": dry_run, "fit": fit, "test": test}[opts.stage](opts)
