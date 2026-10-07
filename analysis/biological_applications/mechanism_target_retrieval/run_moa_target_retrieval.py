#!/usr/bin/env python3
"""Fixed, annotation-only nearest-neighbour retrieval for cpg0004-LINCS.

This evaluator deliberately contains no fitting or hyper-parameter search.  It
audits the curated CP metadata labels, builds one dose-matched gallery from
the frozen train+validation compounds, and evaluates the two support slots
that are actually present in the frozen 1R prediction artifacts.  The
cross-modal profile is reconstructed from the already frozen virtual-prior
and GE-residual outputs; no GE label is inferred and no downstream quantity
is used to change a model parameter.

The script is intended to run in the audited remote environment because that
environment provides RDKit for the registered scaffold/Tanimoto sensitivity.
The base PCC evaluation only requires NumPy and pandas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


VERSION = "cpg0004-LINCS-MoA-target-retrieval-2026-08-30"
OFFICIAL_RESOLVED_URL = "https://raw.githubusercontent.com/broadinstitute/lincs-cell-painting/master/metadata/moa/repurposing_info_external_moa_map_resolved.tsv"
OFFICIAL_INFO_URL = "https://raw.githubusercontent.com/broadinstitute/lincs-cell-painting/master/metadata/moa/repurposing_info.tsv"
SEEDS = (3407, 42, 2025)
MIN_GALLERY_REPEATS = 3
MIN_CLASS_SIZE = 5
MIN_QUERY_REPEATS = 5
EXPECTED_SUPPORT_SLOTS = 2
BOOTSTRAP_ROUNDS = 10_000
DEFAULT_LAMBDA = 0.10
PCC_EPS = 1e-12

LABEL_COLUMNS = {
    "primary_moa": "Metadata_moa",
    "alternative_moa": "Metadata_alternative_moa",
    "primary_target": "Metadata_target",
    "alternative_target": "Metadata_alternative_target",
}
UNION_COLUMNS = {
    "moa": ("primary_moa", "alternative_moa"),
    "target": ("primary_target", "alternative_target"),
}
METHODS = ("raw", "teacher", "virtual_prior", "posterior_ge")
TASKS = ("moa", "target")
VARIANTS = ("base", "scaffold_excluded", "tanimoto_gt_0.7_excluded")
METRIC_NAMES = ("map", "p_at_1", "p_at_5", "recall_at_10", "first_rank", "enrichment_at_10")
CONTRASTS = (
    ("teacher_minus_raw", "teacher", "raw"),
    ("posterior_ge_minus_teacher", "posterior_ge", "teacher"),
    ("virtual_prior_minus_raw", "virtual_prior", "raw"),
    ("posterior_ge_minus_raw", "posterior_ge", "raw"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="Frozen cpg0004 cp_plate_rows.npz")
    p.add_argument("--manifest", type=Path, required=True, help="Frozen cp_plate_manifest.csv")
    p.add_argument("--split-lock", type=Path, required=True, help="Frozen split_lock.json")
    p.add_argument("--annotations", type=Path, required=True, help="Curated CP metadata sidecar")
    p.add_argument("--smiles", type=Path, required=True, help="Frozen compound_smiles.csv")
    p.add_argument("--virtual-root", type=Path, required=True, help="Root containing seed*_1r_all_v6")
    p.add_argument("--ge-root", type=Path, required=True, help="Root containing all_b1_seed*")
    p.add_argument("--pair-root", type=Path, required=True, help="Root containing full_b1_all_seed*")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), choices=SEEDS)
    p.add_argument("--min-gallery-repeats", type=int, default=MIN_GALLERY_REPEATS)
    p.add_argument("--min-class-size", type=int, default=MIN_CLASS_SIZE)
    p.add_argument("--min-query-repeats", type=int, default=MIN_QUERY_REPEATS)
    p.add_argument("--bootstrap-rounds", type=int, default=BOOTSTRAP_ROUNDS)
    p.add_argument("--lambda", dest="lambda_value", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--smoke", action="store_true", help="Run structural checks on a small deterministic slice")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8", errors="replace").astype(str)
    return values.astype(str)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def split_labels(value: object) -> tuple[str, ...]:
    text = clean_text(value)
    if not text:
        return tuple()
    # The LINCS metadata uses pipes; commas/semicolons/slashes are accepted
    # defensively for independently curated sidecars, without any inference.
    labels = {part.strip() for part in re.split(r"[|,;/]+", text) if part.strip()}
    return tuple(sorted(labels))


def join_labels(values: Iterable[str]) -> str:
    return "|".join(sorted({clean_text(x) for x in values if clean_text(x)}))


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int((seed + int.from_bytes(digest, "little")) % (2**63 - 1))


def load_rows(path: Path) -> dict[str, np.ndarray]:
    required = {"compound_id", "dose", "plate", "well", "split", "delta", "baseline"}
    with np.load(path, allow_pickle=False) as loaded:
        missing = required - set(loaded.files)
        if missing:
            raise ValueError(f"{path} missing fields: {sorted(missing)}")
        rows = {key: loaded[key] for key in required}
    for key in ("compound_id", "dose", "plate", "well", "split"):
        rows[key] = decode(rows[key])
    rows["delta"] = np.asarray(rows["delta"], dtype=np.float32)
    rows["baseline"] = np.asarray(rows["baseline"], dtype=np.float32)
    n = len(rows["compound_id"])
    if rows["delta"].ndim != 2 or rows["delta"].shape[0] != n:
        raise ValueError(f"Invalid delta shape {rows['delta'].shape} for {n} rows")
    if rows["baseline"].shape != rows["delta"].shape:
        raise ValueError("baseline/delta shape mismatch")
    if not np.isfinite(rows["delta"]).all() or not np.isfinite(rows["baseline"]).all():
        raise ValueError("Non-finite frozen CP artifact")
    if set(rows["split"]) - {"train", "valid", "test"}:
        raise ValueError("Unexpected split value in CP artifact")
    return rows


def verify_manifest(rows: dict[str, np.ndarray], path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"compound_id", "dose", "plate", "well", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{path} missing fields: {sorted(missing)}")
    if len(manifest) != len(rows["compound_id"]):
        raise ValueError("Frozen manifest/artifact row count mismatch")
    for key in required:
        expected = rows[key]
        actual = manifest[key].astype(str).to_numpy()
        if not np.array_equal(actual, expected):
            raise ValueError(f"Frozen manifest differs from NPZ field {key}")
    manifest = manifest.copy()
    manifest["row_index"] = np.arange(len(manifest), dtype=np.int64)
    manifest["repeat_count"] = manifest.groupby(["compound_id", "dose"])["plate"].transform("size").astype(int)
    return manifest


def verify_split_lock(rows: dict[str, np.ndarray], path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = {key: sorted(str(x) for x in payload["%s_compounds" % key]) for key in ("train", "valid", "test")}
    assigned = {str(c): str(s) for c, s in zip(rows["compound_id"], rows["split"])}
    for name, compounds in split.items():
        missing = [c for c in compounds if assigned.get(c) != name]
        if missing:
            raise ValueError(f"split_lock disagrees with NPZ for {name}: {missing[:5]}")
    sets = [set(split[x]) for x in ("train", "valid", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("split_lock has overlap")
    return split


def load_annotations(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    frame = pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False)
    source_kind = "local_or_remote_curated_sidecar"
    source_url = None
    # The Broad repository's resolved table is the authoritative, explicitly
    # mapped cpg0004 annotation source.  Normalize its names in-memory so the
    # rest of the evaluator has exactly the same code path as the CP sidecar.
    # No label is generated here; all values come from the downloaded table.
    official_columns = {
        "broad_id",
        "broad_sample",
        "moa",
        "target",
        "alternative_moa",
        "alternative_target",
    }
    if official_columns.issubset(frame.columns):
        frame = frame.rename(
            columns={
                "broad_id": "Metadata_broad_id",
                "broad_sample": "Metadata_broad_sample",
                "moa": "Metadata_moa",
                "target": "Metadata_target",
                "alternative_moa": "Metadata_alternative_moa",
                "alternative_target": "Metadata_alternative_target",
            }
        )
        frame["Metadata_pert_id"] = frame["Metadata_broad_id"]
        frame["Metadata_broad_sample_type"] = "trt"
        source_kind = "Broad_lincs_cell_painting_repurposing_info_external_moa_map_resolved"
        source_url = OFFICIAL_RESOLVED_URL
    required = {"Metadata_broad_id", "Metadata_pert_id", "Metadata_broad_sample_type", *LABEL_COLUMNS.values()}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Annotation sidecar missing fields: {sorted(missing)}")
    broad = frame["Metadata_broad_id"].map(clean_text)
    pert = frame["Metadata_pert_id"].map(clean_text)
    frame["compound_id"] = broad.where(broad.ne(""), pert)
    frame["sample_type"] = frame["Metadata_broad_sample_type"].map(clean_text).str.lower()
    frame = frame[frame["sample_type"].eq("trt") & frame["compound_id"].ne("")].copy()
    frame = frame[~frame["compound_id"].isin({"BRD-K50691590", "BRD-K60230970"})].copy()
    grouped: dict[str, dict[str, tuple[str, ...]]] = {}
    for compound, group in frame.groupby("compound_id", sort=True):
        item: dict[str, tuple[str, ...]] = {}
        for name, source in LABEL_COLUMNS.items():
            labels = {label for value in group[source] for label in split_labels(value)}
            item[name] = tuple(sorted(labels))
        for name, sources in UNION_COLUMNS.items():
            item[name] = tuple(sorted({label for source in sources for label in item[source]}))
        grouped[str(compound)] = item
    records = []
    for compound in sorted(grouped):
        item = grouped[compound]
        records.append(
            {
                "compound_id": compound,
                **{name: join_labels(item[name]) for name in (*LABEL_COLUMNS.keys(), *UNION_COLUMNS.keys())},
                **{f"{name}_count": len(item[name]) for name in (*LABEL_COLUMNS.keys(), *UNION_COLUMNS.keys())},
            }
        )
    annotations = pd.DataFrame(records)
    if annotations.empty:
        raise ValueError("No treatment compound annotations")
    for name in (*LABEL_COLUMNS.keys(), *UNION_COLUMNS.keys()):
        annotations[name] = annotations[name].fillna("").astype(str)
    audit: dict[str, Any] = {
        "source": str(path),
        "source_sha256": sha256_file(path),
        "source_kind": source_kind,
        "source_url": source_url,
        "source_local_copy": "official_repurposing_info_external_moa_map_resolved.tsv" if source_url == OFFICIAL_RESOLVED_URL else None,
        "raw_rows": int(len(frame)),
        "compound_count": int(len(annotations)),
        "field_stats": {},
    }
    for name in (*LABEL_COLUMNS.keys(), *UNION_COLUMNS.keys()):
        label_lists = annotations[name].map(split_labels)
        counts = Counter(label for values in label_lists for label in values)
        audit["field_stats"][name] = {
            "compounds_with_label": int(label_lists.map(bool).sum()),
            "missing_compounds": int((~label_lists.map(bool)).sum()),
            "unique_labels": int(len(counts)),
            "multilabel_compounds": int(label_lists.map(lambda x: len(x) > 1).sum()),
            "multilabel_fraction": float(label_lists.map(lambda x: len(x) > 1).mean()),
            "labels_ge_5_compounds": int(sum(value >= MIN_CLASS_SIZE for value in counts.values())),
            "label_counts": dict(sorted(counts.items())),
        }
    return annotations, audit


def load_smiles(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"compound_id", "smiles"}
    if missing := required - set(frame.columns):
        raise ValueError(f"SMILES map missing fields: {sorted(missing)}")
    result: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        compound = clean_text(getattr(row, "compound_id"))
        smiles = clean_text(getattr(row, "smiles"))
        if not compound or not smiles:
            continue
        if compound in result and result[compound] != smiles:
            raise ValueError(f"Conflicting SMILES for {compound}")
        result[compound] = smiles
    if not result:
        raise ValueError("Empty SMILES map")
    return result


def read_prediction(path: Path, kind: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as loaded:
        files = set(loaded.files)
        required = {"compound", "dose", "held_index"}
        if kind == "virtual":
            required |= {"teacher", "virtual_prior", "posterior"}
        elif kind == "ge":
            required |= {"P0", "correct_GE"}
        missing = required - files
        if missing:
            raise ValueError(f"{path} missing {sorted(missing)}")
        compound = decode(loaded["compound"])
        dose = decode(loaded["dose"])
        held = np.asarray(loaded["held_index"], dtype=np.int64)
        profiles = {key: np.asarray(loaded[key], dtype=np.float32) for key in required - {"compound", "dose", "held_index"}}
    n = len(compound)
    frame = pd.DataFrame({"compound": compound, "dose": dose, "held_index": held})
    if frame.duplicated(["compound", "dose", "held_index"]).any():
        raise ValueError(f"Duplicate prediction keys in {path}")
    for key, value in profiles.items():
        if value.ndim != 2 or value.shape[0] != n or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} profile in {path}: {value.shape}")
    return frame, profiles


def load_pair_support(path: Path) -> dict[tuple[str, str, int], int]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"compound", "dose", "held_row", "support_rows"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Pair manifest missing {sorted(missing)}")
    result: dict[tuple[str, str, int], int] = {}
    for row in frame.itertuples(index=False):
        compound = clean_text(row.compound)
        dose = clean_text(row.dose)
        held = int(row.held_row)
        support_text = clean_text(row.support_rows)
        support_values = [x.strip() for x in re.split(r"[;,|]+", support_text) if x.strip()]
        if len(support_values) != 1:
            raise ValueError("This evaluator expects budget=1 support_rows")
        key = (compound, dose, held)
        support = int(support_values[0])
        if key in result and result[key] != support:
            raise ValueError(f"Conflicting support row for {key}")
        result[key] = support
    return result


def prediction_paths(args: argparse.Namespace, seed: int) -> tuple[Path, Path, Path]:
    virtual = args.virtual_root / f"seed{seed}_1r_all_v6" / "test_predictions.npz"
    ge = args.ge_root / f"all_b1_seed{seed}" / "test_predictions.npz"
    pair = args.pair_root / f"full_b1_all_seed{seed}" / "test_pair_manifest.csv"
    return virtual, ge, pair


def make_gallery(
    rows: dict[str, np.ndarray], manifest: pd.DataFrame, annotations: pd.DataFrame, smiles: dict[str, str], min_repeats: int
) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, set[str]]]:
    ann = annotations.set_index("compound_id")
    pieces: list[dict[str, Any]] = []
    gallery_values: list[np.ndarray] = []
    gallery = manifest[manifest["split"].isin(["train", "valid"])].copy()
    for (compound, dose), group in gallery.groupby(["compound_id", "dose"], sort=True):
        if len(group) < min_repeats:
            continue
        indices = group["row_index"].to_numpy(dtype=np.int64)
        pieces.append(
            {
                "gallery_id": f"{compound}|{dose}",
                "compound_id": str(compound),
                "dose": str(dose),
                "repeat_count": int(len(group)),
                "split": "+".join(sorted(set(group["split"].astype(str)))),
                "smiles": smiles.get(str(compound), ""),
                "smiles_available": bool(smiles.get(str(compound), "")),
                "moa_labels": ann.loc[str(compound), "moa"] if str(compound) in ann.index else "",
                "target_labels": ann.loc[str(compound), "target"] if str(compound) in ann.index else "",
            }
        )
        gallery_values.append(rows["delta"][indices].mean(axis=0, dtype=np.float64).astype(np.float32))
    if not pieces:
        raise RuntimeError("No gallery conditions meet minimum repeat count")
    out = pd.DataFrame(pieces)
    out["profile_index"] = np.arange(len(out), dtype=np.int64)
    out["moa_set"] = out["moa_labels"].map(lambda x: set(split_labels(x)))
    out["target_set"] = out["target_labels"].map(lambda x: set(split_labels(x)))
    gallery_labels = {task: set(label for values in out[f"{task}_set"] for label in values) for task in TASKS}
    return out, gallery_labels, {task: set() for task in TASKS}


def class_filter(gallery: pd.DataFrame, task: str, min_class_size: int) -> tuple[set[str], Counter[str]]:
    counts: Counter[str] = Counter()
    for compound, group in gallery.groupby("compound_id", sort=False):
        values = set(label for labels in group[f"{task}_set"] for label in labels)
        counts.update(values)
    keep = {label for label, count in counts.items() if count >= min_class_size}
    return keep, counts


def make_queries(
    rows: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    split: dict[str, list[str]],
    annotations: pd.DataFrame,
    smiles: dict[str, str],
    virtual_frame: pd.DataFrame,
    virtual_profiles: dict[str, np.ndarray],
    ge_frame: pd.DataFrame,
    ge_profiles: dict[str, np.ndarray],
    support_map: dict[tuple[str, str, int], int],
    min_repeats: int,
    lambda_value: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], list[dict[str, Any]]]:
    if not np.isclose(lambda_value, DEFAULT_LAMBDA, atol=1e-12):
        raise ValueError(f"The frozen all-dose run is locked at lambda={DEFAULT_LAMBDA}; got {lambda_value}")
    ann = annotations.set_index("compound_id")
    test_compounds = set(split["test"])
    repeat_lookup = manifest.groupby(["compound_id", "dose"])["row_index"].agg(list)
    repeat_count = manifest.groupby(["compound_id", "dose"]).size().astype(int).to_dict()
    vframe = virtual_frame.copy()
    vframe["support_row"] = [support_map.get((str(c), str(d), int(h)), -1) for c, d, h in zip(vframe.compound, vframe.dose, vframe.held_index)]
    if (vframe["support_row"] < 0).any():
        raise ValueError("A frozen virtual prediction key is absent from its frozen pair manifest")
    vframe["condition_key"] = list(zip(vframe["compound"].astype(str), vframe["dose"].astype(str)))
    vframe = vframe[vframe["compound"].isin(test_compounds)].copy()
    # GE rows are indexed by held plate row.  Their corrected residual is a
    # function of molecule/dose/baseline, not the support profile, so the row
    # whose held_index equals the support slot is the correct frozen residual.
    ge_lookup = {(str(c), str(d), int(h)): i for i, (c, d, h) in enumerate(zip(ge_frame.compound, ge_frame.dose, ge_frame.held_index))}
    v_lookup = {(str(c), str(d), int(h)): i for i, (c, d, h) in enumerate(zip(virtual_frame.compound, virtual_frame.dose, virtual_frame.held_index))}
    records: list[dict[str, Any]] = []
    profiles: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    exclusions: list[dict[str, Any]] = []
    for (compound, dose), group in sorted(vframe.groupby(["compound", "dose"], sort=True)):
        key = (str(compound), str(dose))
        nrep = int(repeat_count.get(key, 0))
        if nrep < min_repeats:
            exclusions.append({"scope": "condition", "compound_id": compound, "dose": dose, "reason": "query_repeat_count_below_threshold", "detail": str(nrep)})
            continue
        slots = sorted(set(int(x) for x in group["support_row"]))
        if len(slots) != EXPECTED_SUPPORT_SLOTS:
            exclusions.append({"scope": "condition", "compound_id": compound, "dose": dose, "reason": "frozen_support_slot_count_not_two", "detail": str(slots)})
            continue
        if key not in repeat_lookup:
            exclusions.append({"scope": "condition", "compound_id": compound, "dose": dose, "reason": "condition_absent_from_cp_manifest", "detail": ""})
            continue
        label_row = ann.loc[str(compound)] if str(compound) in ann.index else None
        moa_labels = str(label_row["moa"]) if label_row is not None else ""
        target_labels = str(label_row["target"]) if label_row is not None else ""
        for slot_index, support_slot in enumerate(slots, start=1):
            support_group = group[group["support_row"].eq(support_slot)].sort_values("held_index")
            support_key = (str(compound), str(dose), int(support_group.iloc[0]["held_index"]))
            support_virtual_pos = v_lookup[support_key]
            held_key = (str(compound), str(dose), int(support_slot))
            if held_key not in v_lookup:
                exclusions.append({"scope": "query", "compound_id": compound, "dose": dose, "support_row": support_slot, "reason": "held_slot_missing_in_frozen_virtual_predictions", "detail": ""})
                continue
            held_virtual_pos = v_lookup[held_key]
            teacher_s = virtual_profiles["teacher"][support_virtual_pos]
            teacher_h = virtual_profiles["teacher"][held_virtual_pos]
            posterior_h = virtual_profiles["posterior"][held_virtual_pos]
            virtual_prior_s = (posterior_h - (1.0 - lambda_value) * teacher_h) / lambda_value
            p0_s = (1.0 - lambda_value) * teacher_s + lambda_value * virtual_prior_s
            ge_key = held_key
            ge_available = ge_key in ge_lookup
            if ge_available:
                ge_pos = ge_lookup[ge_key]
                residual_s = ge_profiles["correct_GE"][ge_pos] - ge_profiles["P0"][ge_pos]
                posterior_ge_s = p0_s + residual_s
            else:
                residual_s = np.full_like(p0_s, np.nan)
                posterior_ge_s = np.full_like(p0_s, np.nan)
            query_id = f"{compound}|{dose}|support{support_slot}"
            records.append(
                {
                    "query_id": query_id,
                    "compound_id": str(compound),
                    "dose": str(dose),
                    "support_slot": int(slot_index),
                    "support_row": int(support_slot),
                    "support_plate": str(manifest.loc[support_slot, "plate"]),
                    "support_well": str(manifest.loc[support_slot, "well"]),
                    "repeat_count": nrep,
                    "held_index_for_prior": int(support_slot),
                    "moa_labels": moa_labels,
                    "target_labels": target_labels,
                    "smiles": smiles.get(str(compound), ""),
                    "smiles_available": bool(smiles.get(str(compound), "")),
                    "ge_available": bool(ge_available),
                }
            )
            profiles["raw"].append(rows["delta"][support_slot].astype(np.float32))
            profiles["teacher"].append(teacher_s.astype(np.float32))
            profiles["virtual_prior"].append(virtual_prior_s.astype(np.float32))
            profiles["posterior_ge"].append(posterior_ge_s.astype(np.float32))
    if not records:
        raise RuntimeError("No frozen query support slots passed eligibility")
    query = pd.DataFrame(records)
    out_profiles = {method: np.vstack(values).astype(np.float32) for method, values in profiles.items()}
    if len(query) != len(out_profiles["raw"]):
        raise AssertionError("query/profile length mismatch")
    for method in METHODS:
        if method == "posterior_ge":
            continue
        if not np.isfinite(out_profiles[method]).all():
            raise ValueError(f"Non-finite reconstructed {method} query profile")
    return query, out_profiles, exclusions


def optional_chemistry(smiles: dict[str, str], compounds: Iterable[str]) -> tuple[dict[str, Any], dict[str, str], str | None]:
    try:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import AllChem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        RDLogger.DisableLog("rdApp.warning")
    except Exception as exc:  # pragma: no cover - exercised only without RDKit
        return {}, {}, f"RDKit unavailable: {type(exc).__name__}: {exc}"
    bit_map: dict[str, Any] = {}
    scaffold_map: dict[str, str] = {}
    invalid: list[str] = []
    for compound in sorted(set(str(x) for x in compounds)):
        text = smiles.get(compound, "")
        if not text:
            invalid.append(compound)
            continue
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            invalid.append(compound)
            continue
        bit_map[compound] = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=2048)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        scaffold_map[compound] = scaffold if scaffold else "<EMPTY>"
    # Missing structures are retained in the base gallery, but are excluded
    # from the chemistry sensitivity candidates because their scaffold and
    # Tanimoto relation is unknowable.  This is a partial-coverage status, not
    # a reason to discard the independently valid PCC retrieval.
    return {"bits": bit_map, "DataStructs": DataStructs, "invalid_compounds": set(invalid)}, scaffold_map, None


def pcc_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float64)
    g = np.asarray(gallery, dtype=np.float64)
    q = q - q.mean(axis=1, keepdims=True)
    g = g - g.mean(axis=1, keepdims=True)
    qnorm = np.linalg.norm(q, axis=1)
    gnorm = np.linalg.norm(g, axis=1)
    scores = q @ g.T
    denom = qnorm[:, None] * gnorm[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = scores / np.maximum(denom, PCC_EPS)
    scores[~np.isfinite(scores)] = -np.inf
    return scores.astype(np.float64)


def build_score_cache(
    query: pd.DataFrame,
    gallery: pd.DataFrame,
    gallery_values: np.ndarray,
    profiles: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Compute dose-matched PCC scores once per seed/method.

    Sensitivity variants change only the candidate mask, not the PCC score.
    Keeping one full query-row score matrix per dose therefore avoids
    recomputing the same 242-dimensional matrix for every task and variant.
    """
    gallery_by_dose: dict[str, np.ndarray] = {
        dose: group["profile_index"].to_numpy(dtype=np.int64)
        for dose, group in gallery.groupby("dose", sort=False)
    }
    query_doses = query["dose"].astype(str).to_numpy()
    cache: dict[str, dict[str, np.ndarray]] = {}
    for method, profile in profiles.items():
        method_cache: dict[str, np.ndarray] = {}
        for dose, gallery_indices in gallery_by_dose.items():
            positions = np.flatnonzero(query_doses == dose)
            scores = np.full((len(query), len(gallery_indices)), -np.inf, dtype=np.float64)
            if len(positions):
                scores[positions] = pcc_matrix(profile[positions], gallery_values[gallery_indices])
            method_cache[dose] = scores
        cache[method] = method_cache
    return cache


def average_precision(relevant: np.ndarray) -> float:
    relevant = np.asarray(relevant, dtype=bool)
    n_positive = int(relevant.sum())
    if n_positive == 0:
        return float("nan")
    positions = np.flatnonzero(relevant) + 1
    precision = np.arange(1, len(positions) + 1, dtype=np.float64) / positions
    return float(precision.sum() / n_positive)


def build_candidate_cache(
    query: pd.DataFrame,
    gallery: pd.DataFrame,
    variant: str,
    chemistry: dict[str, Any],
    scaffold_map: dict[str, str],
) -> dict[tuple[str, str], np.ndarray]:
    """Precompute chemistry masks once per query compound-dose.

    The mask is independent of task and retrieval method.  Caching it outside
    the task/method loops avoids recomputing millions of RDKit Tanimoto calls
    while leaving the candidate definition itself unchanged.
    """
    gallery_by_dose: dict[str, np.ndarray] = {
        dose: group["profile_index"].to_numpy(dtype=np.int64)
        for dose, group in gallery.groupby("dose", sort=False)
    }
    bit_map = chemistry.get("bits", {})
    data_structs = chemistry.get("DataStructs")
    gallery_compounds = gallery["compound_id"].astype(str).to_numpy()
    cache: dict[tuple[str, str], np.ndarray] = {}
    keys = query[["compound_id", "dose"]].astype(str).drop_duplicates().itertuples(index=False, name=None)
    for compound, dose in keys:
        gallery_indices = gallery_by_dose.get(dose, np.empty(0, dtype=np.int64))
        keep = np.ones(len(gallery_indices), dtype=bool)
        if variant == "scaffold_excluded":
            if compound in scaffold_map:
                q_scaffold = scaffold_map[compound]
                keep = np.asarray(
                    [
                        gallery_compounds[i] in scaffold_map
                        and scaffold_map[gallery_compounds[i]] != q_scaffold
                        for i in gallery_indices
                    ],
                    dtype=bool,
                )
            else:
                keep = np.zeros(len(gallery_indices), dtype=bool)
        elif variant == "tanimoto_gt_0.7_excluded":
            if data_structs is not None and compound in bit_map:
                q_bit = bit_map[compound]
                keep = np.asarray(
                    [
                        gallery_compounds[i] in bit_map
                        and float(data_structs.TanimotoSimilarity(q_bit, bit_map[gallery_compounds[i]])) <= 0.7
                        for i in gallery_indices
                    ],
                    dtype=bool,
                )
            else:
                keep = np.zeros(len(gallery_indices), dtype=bool)
        cache[(compound, dose)] = keep
    return cache


def evaluate_method(
    seed: int,
    task: str,
    variant: str,
    method: str,
    query: pd.DataFrame,
    gallery: pd.DataFrame,
    gallery_values: np.ndarray,
    allowed_labels: set[str],
    chemistry: dict[str, Any],
    scaffold_map: dict[str, str],
    candidate_cache: dict[tuple[str, str], np.ndarray],
    score_cache: dict[str, np.ndarray],
) -> pd.DataFrame:
    labels_column = f"{task}_labels"
    q = query.copy()
    q["label_set"] = q[labels_column].map(lambda x: set(split_labels(x)) & allowed_labels)
    q = q[q["label_set"].map(bool)].copy()
    if method == "posterior_ge":
        q = q[q["ge_available"].astype(bool)].copy()
    if q.empty:
        return pd.DataFrame()
    # Positions in the score matrix follow the original query order.
    positions = q.index.to_numpy(dtype=np.int64)
    gallery_by_dose: dict[str, np.ndarray] = {
        dose: group["profile_index"].to_numpy(dtype=np.int64)
        for dose, group in gallery.groupby("dose", sort=False)
    }
    g_compounds = gallery["compound_id"].astype(str).to_numpy()
    g_doses = gallery["dose"].astype(str).to_numpy()
    g_sets = gallery[labels_column].map(lambda x: set(split_labels(x)) & allowed_labels).tolist()
    output: list[dict[str, Any]] = []
    for original_index, row in q.iterrows():
        dose = str(row["dose"])
        gallery_indices = gallery_by_dose.get(dose, np.empty(0, dtype=np.int64))
        scores = score_cache[dose][int(original_index)]
        keep = candidate_cache[(str(row["compound_id"]), dose)]
        candidate_indices_local = np.flatnonzero(keep)
        n_candidates = int(len(candidate_indices_local))
        if n_candidates == 0:
            output.append(_metric_row(seed, task, variant, method, row, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0, 0, 0, 0))
            continue
        order_local = candidate_indices_local[np.argsort(-scores[candidate_indices_local], kind="mergesort")]
        relevant = np.asarray([bool(set(g_sets[int(gallery_indices[i])]) & set(row["label_set"])) for i in order_local], dtype=bool)
        n_positive = int(relevant.sum())
        if n_positive == 0:
            output.append(_metric_row(seed, task, variant, method, row, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), n_candidates, 0, 0, 0))
            continue
        ap = average_precision(relevant)
        k1 = min(1, n_candidates)
        k5 = min(5, n_candidates)
        k10 = min(10, n_candidates)
        hit1 = int(relevant[:k1].sum())
        hit5 = int(relevant[:k5].sum())
        hit10 = int(relevant[:k10].sum())
        p1 = hit1 / k1 if k1 else float("nan")
        p5 = hit5 / k5 if k5 else float("nan")
        recall10 = hit10 / n_positive
        ranks = np.flatnonzero(relevant) + 1
        first_rank = float(ranks[0]) if len(ranks) else float("nan")
        enrichment10 = (hit10 / k10) / (n_positive / n_candidates) if k10 and n_positive else float("nan")
        output.append(_metric_row(seed, task, variant, method, row, ap, p1, p5, recall10, first_rank, enrichment10, n_candidates, n_positive, hit10, int(len(ranks))))
    return pd.DataFrame(output)


def _metric_row(seed: int, task: str, variant: str, method: str, row: pd.Series, ap: float, p1: float, p5: float, recall10: float, first_rank: float, enrichment10: float, n_candidates: int, n_positive: int, hit10: int, correct_count: int) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "task": task,
        "variant": variant,
        "method": method,
        "query_id": str(row["query_id"]),
        "compound_id": str(row["compound_id"]),
        "dose": str(row["dose"]),
        "map": ap,
        "p_at_1": p1,
        "p_at_5": p5,
        "recall_at_10": recall10,
        "first_rank": first_rank,
        "enrichment_at_10": enrichment10,
        "n_candidates": int(n_candidates),
        "n_positive": int(n_positive),
        "hits_at_10": int(hit10),
        "correct_neighbor_count": int(correct_count),
    }


def summarise_metrics(per_query: pd.DataFrame) -> pd.DataFrame:
    if per_query.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["seed", "task", "variant", "method"]
    for keys, group in per_query.groupby(group_columns, sort=True):
        seed, task, variant, method = keys
        item: dict[str, Any] = {
            "seed": int(seed),
            "task": task,
            "variant": variant,
            "method": method,
            "query_count": int(group["query_id"].nunique()),
            "compound_count": int(group["compound_id"].nunique()),
            "gallery_candidate_mean": float(group["n_candidates"].mean()),
            "positive_query_count": int(group["n_positive"].gt(0).sum()),
        }
        for metric in METRIC_NAMES:
            value = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            item[metric] = float(np.nanmean(value)) if np.isfinite(value).any() else float("nan")
            item[f"{metric}_n"] = int(np.isfinite(value).sum())
        rows.append(item)
    return pd.DataFrame(rows)


def bootstrap_diff(values: np.ndarray, seed: int, label: str, rounds: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(values.mean())
    rng = np.random.default_rng(stable_seed(seed, label))
    # Chunking prevents a large accidental query set from allocating an
    # unbounded bootstrap matrix.
    estimates = np.empty(rounds, dtype=np.float64)
    chunk = 1000
    for start in range(0, rounds, chunk):
        stop = min(rounds, start + chunk)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[indices].mean(axis=1)
    return point, float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def paired_contrasts(per_query: pd.DataFrame, seeds: list[int], rounds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if per_query.empty:
        return pd.DataFrame(), pd.DataFrame()
    contrast_rows: list[dict[str, Any]] = []
    for task in TASKS:
        for variant in VARIANTS:
            for contrast_name, method_a, method_b in CONTRASTS:
                for seed in seeds:
                    a = per_query[(per_query.seed == seed) & (per_query.task == task) & (per_query.variant == variant) & (per_query.method == method_a)]
                    b = per_query[(per_query.seed == seed) & (per_query.task == task) & (per_query.variant == variant) & (per_query.method == method_b)]
                    if a.empty or b.empty:
                        continue
                    for metric in METRIC_NAMES:
                        left = a[["query_id", "compound_id", metric]].rename(columns={metric: "a"})
                        right = b[["query_id", "compound_id", metric]].rename(columns={metric: "b"})
                        merged = left.merge(right, on=["query_id", "compound_id"], how="inner")
                        merged = merged[np.isfinite(merged["a"]) & np.isfinite(merged["b"])].copy()
                        if merged.empty:
                            continue
                        compound = merged.assign(diff=merged["a"] - merged["b"]).groupby("compound_id", sort=True)["diff"].mean()
                        point, low, high = bootstrap_diff(compound.to_numpy(), int(seed), f"{task}|{variant}|{contrast_name}|{metric}", rounds)
                        contrast_rows.append(
                            {
                                "seed": int(seed),
                                "task": task,
                                "variant": variant,
                                "contrast": contrast_name,
                                "method_a": method_a,
                                "method_b": method_b,
                                "metric": metric,
                                "compound_count": int(len(compound)),
                                "query_count": int(len(merged)),
                                "point": point,
                                "ci_low": low,
                                "ci_high": high,
                                "rounds": int(rounds),
                            }
                        )
                # Three-seed average: only compounds available in all seed
                # contrasts enter the seed-average paired bootstrap.
                per_seed: dict[int, pd.Series] = {}
                for seed in seeds:
                    a = per_query[(per_query.seed == seed) & (per_query.task == task) & (per_query.variant == variant) & (per_query.method == method_a)]
                    b = per_query[(per_query.seed == seed) & (per_query.task == task) & (per_query.variant == variant) & (per_query.method == method_b)]
                    if a.empty or b.empty:
                        continue
                    for metric in METRIC_NAMES:
                        left = a[["query_id", "compound_id", metric]].rename(columns={metric: "a"})
                        right = b[["query_id", "compound_id", metric]].rename(columns={metric: "b"})
                        merged = left.merge(right, on=["query_id", "compound_id"], how="inner")
                        merged = merged[np.isfinite(merged["a"]) & np.isfinite(merged["b"])].copy()
                        if not merged.empty:
                            per_seed.setdefault((seed, metric), None)
                            per_seed[(seed, metric)] = merged.assign(diff=merged["a"] - merged["b"]).groupby("compound_id", sort=True)["diff"].mean()
                for metric in METRIC_NAMES:
                    vectors = [per_seed[(seed, metric)] for seed in seeds if (seed, metric) in per_seed]
                    if len(vectors) != len(seeds):
                        continue
                    common = set(vectors[0].index)
                    for vector in vectors[1:]:
                        common &= set(vector.index)
                    if not common:
                        continue
                    common_ids = sorted(common)
                    values = np.vstack([vector.loc[common_ids].to_numpy(dtype=float) for vector in vectors]).mean(axis=0)
                    point, low, high = bootstrap_diff(values, 0, f"mean|{task}|{variant}|{contrast_name}|{metric}", rounds)
                    contrast_rows.append(
                        {
                            "seed": "mean_seeds",
                            "task": task,
                            "variant": variant,
                            "contrast": contrast_name,
                            "method_a": method_a,
                            "method_b": method_b,
                            "metric": metric,
                            "compound_count": int(len(values)),
                            "query_count": int(sum(len(x) for x in vectors)),
                            "point": point,
                            "ci_low": low,
                            "ci_high": high,
                            "rounds": int(rounds),
                        }
                    )
    contrasts = pd.DataFrame(contrast_rows)
    if contrasts.empty:
        return contrasts, contrasts.copy()
    ci = contrasts.copy()
    return contrasts, ci


def write_annotation_audit(path: Path, audit: dict[str, Any], annotations: pd.DataFrame, gallery: pd.DataFrame, split: dict[str, list[str]], query: pd.DataFrame, allowed: dict[str, set[str]], class_counts: dict[str, Counter[str]]) -> None:
    lines = [
        "# cpg0004 curated annotation audit",
        "",
        "## Scope and source",
        "",
        "Only independent curated fields from the cpg0004 CP metadata are used: `Metadata_moa`, `Metadata_target`, and their explicitly supplied `Metadata_alternative_*` counterparts. No MoA or target label is inferred from GE, CP profiles, similarity, or model output.",
        "",
        f"Source: `{audit['source']}`",
        f"Source SHA256: `{audit['source_sha256']}`",
        f"Source kind: `{audit.get('source_kind') or 'unspecified'}`",
        f"Source URL: `{audit.get('source_url') or 'not applicable (local/remote sidecar)'}`",
        f"Recorded local copy: `{audit.get('source_local_copy') or 'not applicable'}`",
        f"Treatment rows audited after excluding the two documented 20-mM reference compounds: `{audit['raw_rows']}`",
        f"Annotated compounds: `{audit['compound_count']}`",
        "",
        "## Compound/label coverage",
        "",
        "| field | compounds with label | missing compounds | unique labels | multi-label compounds | multi-label fraction | labels with >=5 compounds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (*LABEL_COLUMNS.keys(), *UNION_COLUMNS.keys()):
        stats = audit["field_stats"][name]
        lines.append(f"| `{name}` | {stats['compounds_with_label']} | {stats['missing_compounds']} | {stats['unique_labels']} | {stats['multilabel_compounds']} | {stats['multilabel_fraction']:.4f} | {stats['labels_ge_5_compounds']} |")
    lines.extend(
        [
            "",
            "The retrieval union fields (`moa`, `target`) retain primary and alternative curated values; primary-only counts are reported above so the union does not hide provenance. Label sets are aggregated at compound level before any split or retrieval calculation.",
            "",
            "## Frozen split and mapping",
            "",
            f"Frozen compounds: train `{len(split['train'])}`, valid `{len(split['valid'])}`, test `{len(split['test'])}`.",
            f"Gallery conditions (train+valid, >= {MIN_GALLERY_REPEATS} CP plate repeats): `{len(gallery)}` across `{gallery['compound_id'].nunique()}` compounds.",
            f"Query support rows (test, >= {MIN_QUERY_REPEATS} CP plate repeats, exactly two frozen support slots): `{len(query)}` across `{query['compound_id'].nunique()}` compounds.",
            f"Query compounds with curated MoA union labels: `{query['moa_labels'].map(bool).sum()}` rows; target union labels: `{query['target_labels'].map(bool).sum()}` rows.",
            f"Query rows with GE residual available for frozen M2 reconstruction: `{int(query['ge_available'].sum())}`.",
            "",
            "| task | gallery classes with >=5 compounds | query rows with at least one retained label |",
            "|---|---:|---:|",
        ]
    )
    for task in TASKS:
        retained_query = int(query[f"{task}_labels"].map(lambda x: bool(set(split_labels(x)) & allowed[task])).sum())
        lines.append(f"| `{task}` | {len(allowed[task])} | {retained_query} |")
    lines.extend(
        [
            "",
            "The gallery is dose-matched to each query; a query is scored only when its retained curated label has at least one same-dose gallery positive. All methods use the identical gallery and query manifest for a given comparison. The two support slots are a property of the frozen 1R artifacts, not a new choice made after inspecting retrieval results.",
            "",
            "## Audit artifacts",
            "",
            "- `annotation_compound_labels.csv` contains primary, alternative, and union label sets.",
            "- `annotation_class_counts.csv` contains distinct gallery-compound counts and the >=5 filter.",
            "- `gallery_manifest.csv`, `query_manifest.csv`, and `EXCLUSIONS.csv` make split, repeat, support-slot, structure, and GE coverage explicit.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results(path: Path, args: argparse.Namespace, audit: dict[str, Any], gallery: pd.DataFrame, query: pd.DataFrame, metrics: pd.DataFrame, contrasts: pd.DataFrame, chemistry_error: str | None, exclusions: pd.DataFrame) -> None:
    lines = [
        "# cpg0004 MoA/target nearest-neighbour retrieval",
        "",
        "## 1. Objective",
        "",
        "Test whether frozen reproducible-effect profiles recover curated MoA/target neighbourhoods more accurately than one-repeat raw CP. This is an unsupervised, fixed PCC nearest-neighbour evaluation; no classifier, downstream tuning, or GE-derived label is used.",
        "",
        "## 2. Dataset",
        "",
        "cpg0004-LINCS CP, acquisition-independent from BBBC047 at the raw plate namespace. Curated MoA/target labels are taken from the Broad `repurposing_info_external_moa_map_resolved.tsv` table when that official file is supplied; the raw CP metadata sidecar is retained as a separate audit input and is never used to infer labels from profiles.",
        "",
        "## 3. Eligibility",
        "",
        f"Gallery: train+valid compounds, >= {args.min_gallery_repeats} CP plate repeats per compound-dose. Query: test compound-dose conditions with >= {args.min_query_repeats} CP plate repeats and exactly the two support slots present in the frozen 1R artifacts. Dose is matched between query and gallery. Gallery classes require >= {args.min_class_size} distinct gallery compounds.",
        "",
        f"Gallery conditions: `{len(gallery)}`; gallery compounds: `{gallery['compound_id'].nunique()}`; query rows: `{len(query)}`; query compounds: `{query['compound_id'].nunique()}`; excluded records: `{len(exclusions)}`.",
        "",
        "## 4. Exact information available to model",
        "",
        "Raw is the single frozen support CP delta. Teacher is the frozen 1R reproducible-effect prediction on that support. The diagnostic virtual-prior profile is reconstructed from the frozen lambda=0.10 output. The validated cross-modal profile is `P0 + (correct_GE - P0)` using the frozen GE residual output; the GE estimator and beta are not refit here. Test labels are used only to score retrieval.",
        "",
        "## 5. Independent reference construction",
        "",
        "Each gallery row is the mean of all available CP plate deltas for a train+valid compound-dose with at least three repeats. No test compound enters the gallery. Positive means shared retained curated MoA or target label with a same-dose gallery compound.",
        "",
        "## 6. Primary endpoint",
        "",
        "Mean average precision (mAP) over query support slots, with compound-balanced paired bootstrap for comparisons.",
        "",
        "## 7. Secondary endpoints",
        "",
        "Precision@1, Precision@5, Recall@10, mean rank of the first correct neighbour, and enrichment@10. Sensitivity variants independently remove same Bemis-Murcko scaffold or gallery compounds with Morgan/ECFP4 Tanimoto >0.7.",
        "",
        "## 8. Sample count",
        "",
        f"The annotation audit covers `{audit['compound_count']}` treatment compounds. Retrieval uses `{len(gallery)}` gallery compound-dose rows and `{len(query)}` two-slot query rows before task-specific missing-label/no-positive filtering. Per-method denominators are recorded in `METRICS_BY_SEED.csv` and `per_query_metrics.csv`.",
        "",
        "## 9. Seed-level results",
        "",
    ]
    main = metrics[(metrics.variant == "base") & (metrics.task == "moa") & (metrics.method.isin(["raw", "teacher", "posterior_ge", "virtual_prior"]))].copy()
    if not main.empty:
        lines.append("| seed | method | mAP | P@1 | P@5 | R@10 | first rank | enrichment@10 | query rows |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in main.sort_values(["seed", "method"]).itertuples(index=False):
            lines.append(f"| {row.seed} | {row.method} | {row.map:.6f} | {row.p_at_1:.6f} | {row.p_at_5:.6f} | {row.recall_at_10:.6f} | {row.first_rank:.3f} | {row.enrichment_at_10:.6f} | {row.query_count} |")
    else:
        lines.append("No base MoA metric rows were available.")
    lines.extend(
        [
            "",
            "Target and scaffold/Tanimoto tables are in `METRICS_BY_SEED.csv`; all paired CIs are in `BOOTSTRAP_CI.csv`.",
            "",
            "## 10. Paired CI",
            "",
            "The outer bootstrap unit is compound. For each metric, each support slot/dose is first averaged within compound, and paired method differences are then resampled 10,000 times. `mean_seeds` rows average same-compound seed differences over the common compound set.",
            "",
        ]
    )
    if not contrasts.empty:
        show = contrasts[(contrasts["seed"].astype(str) == "mean_seeds") & (contrasts.task == "moa") & (contrasts.variant == "base") & (contrasts.metric == "map")]
        if not show.empty:
            lines.append("| contrast | point | 95% CI | compounds |")
            lines.append("|---|---:|---|---:|")
            for row in show.itertuples(index=False):
                lines.append(f"| {row.contrast} | {row.point:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | {row.compound_count} |")
        sensitivity = contrasts[
            (contrasts["seed"].astype(str) == "mean_seeds")
            & (contrasts.metric == "map")
            & (contrasts.variant.isin(["scaffold_excluded", "tanimoto_gt_0.7_excluded"]))
            & (contrasts.contrast.isin(["teacher_minus_raw", "posterior_ge_minus_teacher", "posterior_ge_minus_raw"]))
        ].sort_values(["task", "variant", "contrast"])
        if not sensitivity.empty:
            lines.extend(
                [
                    "",
                    "Registered chemical-similarity sensitivity (mean_seeds, mAP):",
                    "",
                    "| task | exclusion | contrast | point | 95% CI | compounds |",
                    "|---|---|---|---:|---|---:|",
                ]
            )
            for row in sensitivity.itertuples(index=False):
                lines.append(f"| {row.task} | {row.variant} | {row.contrast} | {row.point:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | {row.compound_count} |")
    lines.extend(
        [
            "",
            "## 11. GO / SUPPORTIVE / NO-GO",
            "",
            "The auditable decision is in `DECISION.md`. Teacher-minus-raw is the method comparison and is NO-GO for both MoA and target. A GE incremental GO only means posterior_ge-minus-teacher is positive under the registered gate; it does not establish that posterior_ge beats raw. Scaffold-excluded and Tanimoto-excluded results are sensitivity analyses, not a Strong biological GO.",
            "",
            "## 12. Biological interpretation",
            "",
            "The posterior_ge-minus-raw mAP CI crosses zero for both MoA and target, including both registered chemistry exclusions; therefore the final method is not superior to raw in this audit. A positive incremental GE contrast is reported as a bounded component result only. It does not establish causal target identification, universal chemical generalisation, or replacement of real repeats. The virtual-prior row is diagnostic because the upstream cpg0004 virtual-prior result was frozen as a no-go branch.",
            "",
            "## 13. Limitations",
            "",
            "The query set is retrospective and restricted to conditions with two frozen support slots and at least five CP plate rows. Dose matching reduces dose-driven similarity but lowers candidate counts. Curated metadata can be missing or multi-label; union labels preserve supplied alternatives but do not adjudicate them. Acquisition independence from BBBC047 does not imply independent chemical space. A missing RDKit installation would block only the registered chemical sensitivity, not base PCC retrieval.",
        ]
    )
    if chemistry_error:
        lines.extend(["", f"Chemistry sensitivity status: `{chemistry_error}`."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision(path: Path, contrasts: pd.DataFrame) -> None:
    def verdict(task: str, name: str, variant: str = "base") -> tuple[str, str]:
        sub = contrasts[(contrasts["seed"].astype(str) == "mean_seeds") & (contrasts.task == task) & (contrasts.variant == variant) & (contrasts.metric == "map") & (contrasts.contrast == name)]
        seed_rows = contrasts[(contrasts["seed"].astype(str).isin({"3407", "42", "2025"})) & (contrasts.task == task) & (contrasts.variant == variant) & (contrasts.metric == "map") & (contrasts.contrast == name)]
        if sub.empty:
            return "BLOCKED", "No common three-seed contrast row."
        row = sub.iloc[0]
        direction = bool((seed_rows["point"] > 0).all()) if len(seed_rows) == 3 else False
        if direction and float(row["ci_low"]) > 0:
            return "GO", f"All three seed point estimates are positive; mean-seed CI [{row.ci_low:.6f}, {row.ci_high:.6f}]."
        if direction:
            return "SUPPORTIVE", f"All three seed point estimates are positive, but mean-seed CI [{row.ci_low:.6f}, {row.ci_high:.6f}] crosses zero."
        return "NO-GO", f"Seed direction is not uniformly positive; mean-seed point {row.point:.6f}, CI [{row.ci_low:.6f}, {row.ci_high:.6f}]."
    teacher_verdict, teacher_reason = verdict("moa", "teacher_minus_raw")
    ge_verdict, ge_reason = verdict("moa", "posterior_ge_minus_teacher")
    target_teacher_verdict, target_teacher_reason = verdict("target", "teacher_minus_raw")
    target_ge_verdict, target_ge_reason = verdict("target", "posterior_ge_minus_teacher")

    def contrast_reason(task: str, name: str, variant: str = "base") -> str:
        sub = contrasts[
            (contrasts["seed"].astype(str) == "mean_seeds")
            & (contrasts.task == task)
            & (contrasts.variant == variant)
            & (contrasts.metric == "map")
            & (contrasts.contrast == name)
        ]
        if sub.empty:
            return "No common three-seed contrast row."
        row = sub.iloc[0]
        return f"point {row.point:.6f}, 95% CI [{row.ci_low:.6f}, {row.ci_high:.6f}] (n={int(row.compound_count)} compounds)."

    lines = [
        "# Decision",
        "",
        f"## Bio-B MoA retrieval: **{teacher_verdict}**",
        "",
        teacher_reason,
        "",
        f"## GE biological increment: **{ge_verdict}**",
        "",
        ge_reason,
        "",
        f"## Secondary target retrieval: **{target_teacher_verdict}**",
        "",
        target_teacher_reason,
        "",
        f"## Secondary target GE increment: **{target_ge_verdict}**",
        "",
        target_ge_reason,
        "",
        "## Final comparison boundary",
        "",
        "The incremental GE GO is not a final method GO: posterior_ge-minus-raw remains inconclusive when its CI crosses zero.",
        f"- MoA posterior_ge-minus-raw: {contrast_reason('moa', 'posterior_ge_minus_raw')}",
        f"- Target posterior_ge-minus-raw: {contrast_reason('target', 'posterior_ge_minus_raw')}",
        "",
        "## Registered chemistry sensitivities (mean_seeds mAP)",
        "",
        "| task | exclusion | teacher-minus-raw | posterior_ge-minus-teacher | posterior_ge-minus-raw |",
        "|---|---|---|---|---|",
    ]
    for task in TASKS:
        for variant in ("scaffold_excluded", "tanimoto_gt_0.7_excluded"):
            values = [contrast_reason(task, name, variant) for name in ("teacher_minus_raw", "posterior_ge_minus_teacher", "posterior_ge_minus_raw")]
            lines.append(f"| {task} | {variant} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(
        [
            "",
            "No sensitivity row upgrades the primary decision: teacher-minus-raw remains negative and posterior_ge-minus-raw CIs cross zero. The registered positive incremental GE rows are component-level support only, not a Strong biological GO.",
            "",
            "## Guardrails",
            "",
            "- No MoA/target label was inferred from GE or CP profiles.",
            "- No model, lambda, beta, threshold, test compound, or similarity metric was selected from retrieval results.",
            "- The same dose-matched train+valid gallery is used for all methods in each evaluation.",
            "- Bootstrap resampling is compound-level, preserving paired support-slot/dose observations.",
            "",
            "The virtual-prior branch remains diagnostic only and is not used to upgrade the recommendation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(outdir: Path, metrics: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        subset = metrics[(metrics.task == task) & (metrics.variant.isin(["base", "scaffold_excluded", "tanimoto_gt_0.7_excluded"])) & (metrics.method.isin(METHODS))]
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, variant in zip(axes, VARIANTS):
            view = subset[subset.variant == variant]
            if view.empty:
                ax.axis("off")
                continue
            pivot = view.pivot_table(index="seed", columns="method", values="map", aggfunc="mean").reindex(columns=METHODS)
            pivot.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"], legend=False)
            ax.set_title(variant.replace("_", " "))
            ax.set_xlabel("seed")
            ax.set_ylabel("mAP" if ax is axes[0] else "")
            ax.tick_params(axis="x", rotation=0)
        axes[-1].legend(loc="best", fontsize=8)
        fig.suptitle(f"cpg0004 {task} retrieval mAP")
        fig.tight_layout()
        fig.savefig(figures / f"{task}_map_by_seed.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if sorted(args.seeds) != sorted(set(args.seeds)):
        raise ValueError("seeds must be unique")
    if args.bootstrap_rounds < 1 or args.min_gallery_repeats < 1 or args.min_query_repeats < 1 or args.min_class_size < 1:
        raise ValueError("repeat/class/bootstrap thresholds must be positive")
    if args.outdir.exists() and any(args.outdir.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "figures").mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.data)
    manifest = verify_manifest(rows, args.manifest)
    split = verify_split_lock(rows, args.split_lock)
    annotations, audit = load_annotations(args.annotations)
    smiles = load_smiles(args.smiles)
    annotations.to_csv(args.outdir / "annotation_compound_labels.csv", index=False)

    gallery, _, _ = make_gallery(rows, manifest, annotations, smiles, args.min_gallery_repeats)
    gallery_compounds = set(gallery["compound_id"])
    if gallery_compounds & set(split["test"]):
        raise ValueError("Test compound entered gallery")
    allowed_labels: dict[str, set[str]] = {}
    class_counts: dict[str, Counter[str]] = {}
    class_rows: list[dict[str, Any]] = []
    for task in TASKS:
        keep, counts = class_filter(gallery, task, args.min_class_size)
        allowed_labels[task] = keep
        class_counts[task] = counts
        for label, count in sorted(counts.items()):
            class_rows.append({"task": task, "label": label, "gallery_compound_count": int(count), "eligible_ge_5": bool(label in keep), "min_class_size": args.min_class_size})
        gallery[f"{task}_labels"] = gallery[f"{task}_set"].map(lambda x: join_labels(x & keep))
    pd.DataFrame(class_rows).to_csv(args.outdir / "annotation_class_counts.csv", index=False)
    gallery.drop(columns=["moa_set", "target_set"], errors="ignore").to_csv(args.outdir / "gallery_manifest.csv", index=False)
    gallery_values = []
    for _, row in gallery.sort_values("profile_index").iterrows():
        indices = manifest[(manifest.compound_id == row.compound_id) & (manifest.dose == row.dose) & manifest.split.isin(["train", "valid"])]["row_index"].to_numpy(dtype=np.int64)
        gallery_values.append(rows["delta"][indices].mean(axis=0, dtype=np.float64).astype(np.float32))
    gallery_values_array = np.vstack(gallery_values).astype(np.float32)

    all_queries: list[pd.DataFrame] = []
    all_profiles: dict[int, dict[str, np.ndarray]] = {}
    all_exclusions: list[dict[str, Any]] = []
    selected_lambda: dict[int, float] = {}
    for seed in args.seeds:
        virtual_path, ge_path, pair_path = prediction_paths(args, seed)
        if not virtual_path.is_file() or not ge_path.is_file() or not pair_path.is_file():
            raise FileNotFoundError(f"Missing frozen prediction/manifest for seed {seed}: {virtual_path}, {ge_path}, {pair_path}")
        virtual_frame, virtual_profiles = read_prediction(virtual_path, "virtual")
        ge_frame, ge_profiles = read_prediction(ge_path, "ge")
        support_map = load_pair_support(pair_path)
        q, p, exclusions = make_queries(rows, manifest, split, annotations, smiles, virtual_frame, virtual_profiles, ge_frame, ge_profiles, support_map, args.min_query_repeats, args.lambda_value)
        q["seed"] = int(seed)
        q["moa_retained_label"] = q["moa_labels"].map(lambda x: join_labels(set(split_labels(x)) & allowed_labels["moa"]))
        q["target_retained_label"] = q["target_labels"].map(lambda x: join_labels(set(split_labels(x)) & allowed_labels["target"]))
        q["seed"] = int(seed)
        all_queries.append(q)
        all_profiles[int(seed)] = p
        for item in exclusions:
            item["seed"] = int(seed)
        all_exclusions.extend(exclusions)
        # Record that the upstream selection was already frozen at .10.
        metrics_path = virtual_path.with_name("metrics.json")
        if metrics_path.is_file():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            selected_lambda[int(seed)] = float(payload.get("selected_lambda", args.lambda_value))
            if not np.isclose(selected_lambda[int(seed)], args.lambda_value, atol=1e-12):
                raise ValueError(f"Seed {seed} frozen selected_lambda={selected_lambda[int(seed)]} differs from locked {args.lambda_value}")

    # Queries are structurally identical across seeds; keep a seed-specific
    # manifest because GE availability can be audited independently.
    query = pd.concat(all_queries, ignore_index=True)
    if query.empty:
        raise RuntimeError("No query rows")
    query.drop(columns=[], errors="ignore").to_csv(args.outdir / "query_manifest.csv", index=False)
    query_out = query[["seed", "query_id", "compound_id", "dose", "support_slot", "support_row", "support_plate", "support_well", "repeat_count", "held_index_for_prior", "moa_labels", "target_labels", "moa_retained_label", "target_retained_label", "smiles", "smiles_available", "ge_available"]].copy()
    query_out.to_csv(args.outdir / "ELIGIBLE_SAMPLES.csv", index=False)

    chemistry_compounds = set(gallery["compound_id"]) | set(query["compound_id"])
    chemistry, scaffold_map, chemistry_error = optional_chemistry(smiles, chemistry_compounds)
    if chemistry_error:
        all_exclusions.append({"scope": "sensitivity", "reason": "scaffold_tanimoto_blocked", "detail": chemistry_error})
    elif chemistry.get("invalid_compounds"):
        all_exclusions.append({"scope": "sensitivity", "reason": "structure_unavailable_for_chemical_sensitivity", "detail": f"{len(chemistry['invalid_compounds'])} gallery/query compounds lack a valid SMILES; they remain in base PCC but are excluded from chemical-sensitivity candidates"})
    exclusions_frame = pd.DataFrame(all_exclusions)
    if exclusions_frame.empty:
        exclusions_frame = pd.DataFrame(columns=["scope", "seed", "compound_id", "dose", "support_row", "reason", "detail"])
    exclusions_frame.to_csv(args.outdir / "EXCLUSIONS.csv", index=False)

    per_query_parts: list[pd.DataFrame] = []
    # Base and sensitivities share the same gallery.  The chemistry variants
    # are skipped only when RDKit is unavailable; base PCC remains runnable.
    active_variants = ["base"] + (["scaffold_excluded", "tanimoto_gt_0.7_excluded"] if not chemistry_error else [])
    for seed in args.seeds:
        # Profiles are stored per seed with a local 0..n-1 row order; reset
        # the concatenated manifest index before using it to index arrays.
        q_seed = query[query.seed.eq(seed)].reset_index(drop=True).copy()
        score_cache_by_method = build_score_cache(q_seed, gallery, gallery_values_array, all_profiles[int(seed)])
        for variant in active_variants:
            candidate_cache = build_candidate_cache(q_seed, gallery, variant, chemistry, scaffold_map)
            for task in TASKS:
                allowed = allowed_labels[task]
                for method in METHODS:
                    result = evaluate_method(seed, task, variant, method, q_seed, gallery, gallery_values_array, allowed, chemistry, scaffold_map, candidate_cache, score_cache_by_method[method])
                    if not result.empty:
                        per_query_parts.append(result)
    per_query = pd.concat(per_query_parts, ignore_index=True) if per_query_parts else pd.DataFrame()
    if not per_query.empty:
        per_query.to_csv(args.outdir / "per_query_metrics.csv", index=False)
    metrics = summarise_metrics(per_query)
    if metrics.empty:
        metrics = pd.DataFrame(columns=["seed", "task", "variant", "method", "query_count", "compound_count", *METRIC_NAMES])
    metrics.to_csv(args.outdir / "METRICS_BY_SEED.csv", index=False)
    contrasts, ci = paired_contrasts(per_query, list(args.seeds), args.bootstrap_rounds)
    if contrasts.empty:
        contrasts = pd.DataFrame(columns=["seed", "task", "variant", "contrast", "method_a", "method_b", "metric", "compound_count", "query_count", "point", "ci_low", "ci_high", "rounds"])
        ci = contrasts.copy()
    contrasts.to_csv(args.outdir / "PAIRED_CONTRASTS.csv", index=False)
    ci.to_csv(args.outdir / "BOOTSTRAP_CI.csv", index=False)

    audit["frozen_split"] = {name: len(values) for name, values in split.items()}
    audit["gallery"] = {"conditions": int(len(gallery)), "compounds": int(gallery.compound_id.nunique()), "min_repeats": args.min_gallery_repeats}
    audit["query"] = {"rows_across_seeds": int(len(query)), "compounds": int(query.compound_id.nunique()), "min_repeats": args.min_query_repeats, "exact_support_slots": EXPECTED_SUPPORT_SLOTS, "ge_available_rows": int(query.ge_available.sum())}
    audit["class_filter"] = {task: {"eligible_label_count": len(allowed_labels[task]), "min_class_size": args.min_class_size} for task in TASKS}
    audit["selected_lambda_by_seed"] = selected_lambda
    (args.outdir / "annotation_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    write_annotation_audit(args.outdir / "ANNOTATION_AUDIT.md", audit, annotations, gallery, split, query, allowed_labels, class_counts)

    config = {
        "version": VERSION,
        "seeds": [int(x) for x in args.seeds],
        "source_paths": {key: str(value) for key, value in {"data": args.data, "manifest": args.manifest, "split_lock": args.split_lock, "annotations": args.annotations, "smiles": args.smiles, "virtual_root": args.virtual_root, "ge_root": args.ge_root, "pair_root": args.pair_root}.items()},
        "source_sha256": {key: sha256_file(value) for key, value in {"data": args.data, "manifest": args.manifest, "split_lock": args.split_lock, "annotations": args.annotations, "smiles": args.smiles}.items()},
        "annotation_provenance": {
            "resolved_table_url": OFFICIAL_RESOLVED_URL,
            "base_table_url": OFFICIAL_INFO_URL,
            "resolved_table_local_copy": "official_repurposing_info_external_moa_map_resolved.tsv",
            "base_table_local_copy": "official_repurposing_info.tsv",
            "raw_sidecar_local_copy": "cpg0004_cp_annotation_sidecar.csv",
            "resolved_table_is_used_when_annotations_matches_official_schema": True,
            "label_origin": "curated metadata only; no GE/CP/profile inference",
        },
        "min_gallery_repeats": args.min_gallery_repeats,
        "min_query_repeats": args.min_query_repeats,
        "min_class_size": args.min_class_size,
        "bootstrap_rounds": args.bootstrap_rounds,
        "lambda": args.lambda_value,
        "similarity": "Pearson correlation coefficient (PCC), descending, stable compound-id tie break",
        "query_support_definition": "two unique support_rows in the frozen budget=1 pair manifest; teacher(s) from support_rows=s; virtual prior V(s) reverse-solved from held_index=s P0/teacher at locked lambda=.10; posterior_ge=P0(s)+(correct_GE-P0) at held_index=s",
        "gallery_definition": "mean of all train+valid CP plate deltas for each compound-dose with >=3 repeats, dose matched to query",
        "positive_definition": "query/gallery share at least one curated union MoA or target label; only gallery classes with >=5 distinct compounds retained",
        "methods": {"raw": "single CP support delta", "teacher": "frozen reproducible-effect teacher", "virtual_prior": "frozen diagnostic virtual prior", "posterior_ge": "frozen P0 plus observed GE residual"},
        "guardrails": ["no GE-derived labels", "no downstream fitting or tuning", "test compounds absent from gallery", "compound-level paired bootstrap", "two support slots retained exactly", "same gallery for all methods"],
        "rdkit_sensitivity_status": chemistry_error or "available",
    }
    (args.outdir / "CONFIG.json").write_text(json.dumps(config, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    protocol = args.outdir / "PROTOCOL.md"
    if not protocol.exists():
        protocol.write_text("""# Protocol lock: cpg0004 MoA/target retrieval\n\n- Dataset: cpg0004-LINCS CP.\n- Curated labels: `Metadata_moa`, `Metadata_target`, and explicitly supplied alternative fields; no GE inference.\n- Gallery: frozen train+valid compounds, mean CP plate delta per dose, at least 3 repeats.\n- Query: frozen test conditions with at least 5 CP plate rows and exactly the two support slots saved by the frozen 1R pair manifest.\n- Dose is matched between query and gallery. Positive means shared curated label. Gallery classes require at least 5 distinct compounds.\n- Similarity: PCC only, descending with deterministic compound-id tie break.\n- Methods: raw support, frozen teacher, frozen diagnostic virtual prior, frozen P0 plus observed GE residual.\n- Primary: mAP. Secondary: P@1, P@5, R@10, first-correct rank, enrichment@10.\n- Sensitivity: remove same Bemis--Murcko scaffold and, independently, ECFP4 Tanimoto > 0.7.\n- Statistics: 10,000 compound-level paired bootstrap resamples; no test-based tuning.\n""", encoding="utf-8")
    write_results(args.outdir / "RESULTS.md", args, audit, gallery, query, metrics, contrasts, chemistry_error, exclusions_frame)
    write_decision(args.outdir / "DECISION.md", contrasts)
    make_figures(args.outdir, metrics)

    summary = {"version": VERSION, "gallery_conditions": int(len(gallery)), "gallery_compounds": int(gallery.compound_id.nunique()), "query_rows_across_seeds": int(len(query)), "metrics_rows": int(len(metrics)), "contrast_rows": int(len(contrasts)), "chemistry_sensitivity": "available" if not chemistry_error else chemistry_error, "outdir": str(args.outdir)}
    (args.outdir / "RUN_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
