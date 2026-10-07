#!/usr/bin/env python3
"""Build the cpg0004-LINCS CP plate/condition table used by Phase 1--2.

The input is the replicate-level, variable-selected LINCS Pilot 1 CP CSV.  A
condition is a (compound, standard-dose, plate) key.  A plate is the
independent repeat unit; multiple wells with the same condition key are
technical duplicates and are averaged after subtracting that plate's DMSO
control median.  The script writes a split-locked HDF5 table, split NPZ files,
and metadata manifests.  It intentionally does not train a model.

The implementation is deliberately self-contained so the exact raw path,
column selection, exclusion rules, and split lock can be replayed on the data
host without importing experiment code from another directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd


VERSION = "cpg0004-LINCS-CP-PlateRows-Phase1-2-2026-08-30"
SPLIT_SEED = 3407
SPLIT_FRACTIONS = {"train": 0.60, "valid": 0.20, "test": 0.20}
STANDARD_DOSES = (0.04, 0.12, 0.37, 1.11, 3.33, 10.0)
STANDARD_DOSE_LABELS = tuple(f"{x:g}" for x in STANDARD_DOSES)
REFERENCE_PLATE_THRESHOLD = 100
TWENTY_MM_TOLERANCE = 0.05

DEFAULT_INPUT = (
    "/path/to/home/AIDD/baseline/data/MVC/LINCS-Pilot1/CellPainting/"
    "replicate_level_cp_normalized_variable_selected.csv.gz"
)

# The CSV contains 1,215 numerical CP features and a small number of legacy
# metadata columns without the Metadata_ prefix.  Those legacy columns must
# never silently enter a model matrix even when pandas infers them as floats
# because a sample happens to contain only missing values.
LEGACY_METADATA_COLUMNS = {
    "broad_id",
    "plate_map_name",
    "solvent",
    "Batch_Number",
    "Batch_Date",
    "pert_iname",
    "InChIKey14",
    "moa",
    "target",
    "broad_date",
    "clinical_phase",
    "alternative_moa",
    "alternative_target",
    "pert_type",
    "control_type",
}

REQUIRED_METADATA = (
    "Metadata_broad_sample_type",
    "Metadata_broad_id",
    "Metadata_pert_id",
    "Metadata_pert_mfc_id",
    "Metadata_mmoles_per_liter",
    "Metadata_dose_recode",
    "Metadata_pert_id_dose",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Batch_Number",
    "Metadata_Batch_Date",
    "Metadata_plate_map_name",
    "Metadata_cell_id",
    "Metadata_InChIKey14",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-existing", action="store_true",
                        help="Allow writing into an existing empty/non-empty output directory.")
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    return args


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def numeric(value: Any) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def compound_id(row: pd.Series) -> str:
    return clean(row.get("Metadata_broad_id")) or clean(row.get("Metadata_pert_id"))


def standard_dose(value: Any) -> tuple[str, float] | None:
    """Map Metadata_dose_recode to one of the six declared dose labels."""
    number = numeric(value)
    if number is None:
        return None
    distances = np.abs(np.asarray(STANDARD_DOSES, dtype=np.float64) - number)
    index = int(np.argmin(distances))
    # The recode field is already rounded by the LINCS preparation.  A modest
    # tolerance catches numeric serialization differences but not arbitrary
    # raw concentrations.
    tolerance = 0.06 if STANDARD_DOSES[index] < 1.0 else 0.08
    if float(distances[index]) > tolerance:
        return None
    return STANDARD_DOSE_LABELS[index], float(STANDARD_DOSES[index])


def sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_columns(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def choose_feature_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    missing = [name for name in REQUIRED_METADATA if name not in columns]
    if missing:
        raise ValueError(f"Required metadata columns missing: {missing}")
    metadata = [name for name in columns if name.startswith("Metadata_")]
    metadata.extend(name for name in LEGACY_METADATA_COLUMNS if name in columns)
    metadata = list(dict.fromkeys(metadata))
    features = [name for name in columns if name not in set(metadata)]
    if not features:
        raise ValueError("No numerical CP feature columns found")
    # Validate the feature contract on a small sample.  Do not coerce strings
    # to NaN: a malformed feature column should stop the build loudly.
    sample = pd.read_csv(path=path if False else "", nrows=0)  # pragma: no cover
    return features, metadata


def select_columns(path: Path) -> tuple[list[str], list[str], list[str]]:
    columns = read_columns(path)
    missing = [name for name in REQUIRED_METADATA if name not in columns]
    if missing:
        raise ValueError(f"Required metadata columns missing: {missing}")
    metadata = [name for name in columns if name.startswith("Metadata_")]
    metadata.extend(name for name in LEGACY_METADATA_COLUMNS if name in columns)
    metadata = list(dict.fromkeys(metadata))
    features = [name for name in columns if name not in set(metadata)]
    if not features:
        raise ValueError("No numerical CP feature columns found")
    sample = pd.read_csv(path, usecols=features, nrows=32, low_memory=False)
    bad = [name for name in features if not pd.api.types.is_numeric_dtype(sample[name])]
    if bad:
        raise TypeError(f"Non-numeric columns would enter CP matrix: {bad}")
    return columns, metadata, features


def iter_chunks(path: Path, usecols: list[str], chunksize: int) -> Iterable[tuple[int, pd.DataFrame]]:
    offset = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        yield offset, chunk
        offset += len(chunk)


def first_pass(path: Path, metadata: list[str], chunksize: int) -> dict[str, Any]:
    """Collect raw counts and identify the 136-plate reference compounds."""
    plate_set_by_compound: defaultdict[str, set[str]] = defaultdict(set)
    plate_counts: Counter[str] = Counter()
    sample_type_counts: Counter[str] = Counter()
    dose_recode_counts: Counter[str] = Counter()
    raw_dose_counts: Counter[str] = Counter()
    batch_counts: Counter[str] = Counter()
    cell_counts: Counter[str] = Counter()
    total_rows = 0
    treatment_rows = 0
    control_rows = 0
    duplicate_source_keys: Counter[tuple[str, str, str, str]] = Counter()
    usecols = list(dict.fromkeys(metadata))
    for offset, chunk in iter_chunks(path, usecols, chunksize):
        total_rows += len(chunk)
        for local_index, row in chunk.iterrows():
            typ = clean(row.get("Metadata_broad_sample_type"))
            sample_type_counts[typ] += 1
            plate = clean(row.get("Metadata_Plate"))
            if plate:
                plate_counts[plate] += 1
            batch = clean(row.get("Metadata_Batch_Number"))
            if batch:
                batch_counts[batch] += 1
            cell = clean(row.get("Metadata_cell_id"))
            if cell:
                cell_counts[cell] += 1
            raw = numeric(row.get("Metadata_mmoles_per_liter"))
            if raw is not None:
                raw_dose_counts[f"{raw:.12g}"] += 1
            recoded = standard_dose(row.get("Metadata_dose_recode"))
            if recoded is not None:
                dose_recode_counts[recoded[0]] += 1
            if typ != "trt":
                if typ == "control":
                    control_rows += 1
                continue
            treatment_rows += 1
            compound = compound_id(row)
            if not compound or not plate:
                continue
            plate_set_by_compound[compound].add(plate)
            # This key includes raw dose and well.  It is the row-level
            # duplicate identity; the condition key below intentionally uses
            # standard dose and will therefore reveal technical duplicates.
            duplicate_source_keys[(compound, clean(row.get("Metadata_pert_id_dose")), plate,
                                   clean(row.get("Metadata_Well")))] += 1
    unique_plates = sorted(plate_counts)
    reference_compounds = sorted(
        compound for compound, plates in plate_set_by_compound.items()
        if len(plates) == len(unique_plates) and len(unique_plates) >= REFERENCE_PLATE_THRESHOLD
    )
    return {
        "n_rows": int(total_rows),
        "treatment_rows": int(treatment_rows),
        "control_rows": int(control_rows),
        "sample_type_counts": dict(sample_type_counts),
        "unique_plates": len(unique_plates),
        "plates": unique_plates,
        "unique_compounds": len(plate_set_by_compound),
        "compound_plate_counts": {key: len(value) for key, value in sorted(plate_set_by_compound.items())},
        "reference_compounds": reference_compounds,
        "dose_recode_counts": dict(dose_recode_counts),
        "raw_dose_counts": dict(raw_dose_counts),
        "batch_counts": dict(batch_counts),
        "cell_counts": dict(cell_counts),
        "duplicate_source_key_groups": int(sum(1 for value in duplicate_source_keys.values() if value > 1)),
        "duplicate_source_key_extra_rows": int(sum(value - 1 for value in duplicate_source_keys.values() if value > 1)),
    }


def control_pass(path: Path, metadata: list[str], features: list[str], chunksize: int) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Compute an independent per-plate median over rows marked ``control``."""
    usecols = list(dict.fromkeys(metadata + features))
    controls: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for _offset, chunk in iter_chunks(path, usecols, chunksize):
        types = chunk["Metadata_broad_sample_type"].astype(str).str.strip()
        selected = chunk.loc[types.eq("control")]
        for plate, group in selected.groupby("Metadata_Plate", sort=False):
            plate_name = clean(plate)
            if not plate_name:
                continue
            values = group[features].to_numpy(dtype=np.float32, copy=True)
            controls[plate_name].extend(values)
    medians: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, Any]] = {}
    for plate in sorted(controls):
        matrix = np.vstack(controls[plate]).astype(np.float32, copy=False)
        with np.errstate(all="ignore"):
            median = np.nanmedian(matrix, axis=0).astype(np.float32)
        finite = np.isfinite(median)
        if not finite.all():
            raise ValueError(f"Plate {plate} control median has {int((~finite).sum())} non-finite features")
        medians[plate] = median
        stats[plate] = {
            "plate": plate,
            "control_rows": int(matrix.shape[0]),
            "feature_count": int(matrix.shape[1]),
            "finite_median_features": int(finite.sum()),
            "control_median_l2": float(np.linalg.norm(median.astype(np.float64))),
        }
    return medians, stats


def empty_group(compound: str, dose_label: str, dose_value: float, plate: str, row: pd.Series,
                source_row: int, delta: np.ndarray) -> dict[str, Any]:
    return {
        "compound": compound,
        "dose_standard": dose_label,
        "dose": float(dose_value),
        "plate": plate,
        "sum": delta.astype(np.float32, copy=True),
        "n": 1,
        "raw_doses": [float(numeric(row.get("Metadata_mmoles_per_liter")) or np.nan)],
        "wells": [clean(row.get("Metadata_Well"))],
        "source_rows": [int(source_row)],
        "batches": sorted({clean(row.get("Metadata_Batch_Number"))} - {""}),
        "batch_dates": sorted({clean(row.get("Metadata_Batch_Date"))} - {""}),
        "plate_map_names": sorted({clean(row.get("Metadata_plate_map_name"))} - {""}),
        "cell_ids": sorted({clean(row.get("Metadata_cell_id"))} - {""}),
        "inchikey14": sorted({clean(row.get("Metadata_InChIKey14"))} - {""}),
        "pert_id_dose": sorted({clean(row.get("Metadata_pert_id_dose"))} - {""}),
        "well_records": [{
            "source_row": int(source_row),
            "well": clean(row.get("Metadata_Well")),
            "raw_dose_mM": numeric(row.get("Metadata_mmoles_per_liter")),
            "pert_id_dose": clean(row.get("Metadata_pert_id_dose")),
        }],
    }


def treatment_pass(path: Path, metadata: list[str], features: list[str], chunksize: int,
                   medians: dict[str, np.ndarray], reference_compounds: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Subtract controls and aggregate only technical duplicates."""
    usecols = list(dict.fromkeys(metadata + features))
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    seen_source_keys: set[tuple[str, str, str, str]] = set()
    counters: Counter[str] = Counter()
    reference_set = set(reference_compounds)
    for offset, chunk in iter_chunks(path, usecols, chunksize):
        types = chunk["Metadata_broad_sample_type"].astype(str).str.strip()
        selected = chunk.loc[types.eq("trt")]
        for local_index, row in selected.iterrows():
            source_row = int(offset + int(local_index) + 2)  # one-based CSV line incl. header
            compound = compound_id(row)
            plate = clean(row.get("Metadata_Plate"))
            raw_dose = numeric(row.get("Metadata_mmoles_per_liter"))
            recoded = standard_dose(row.get("Metadata_dose_recode"))
            if not compound:
                counters["excluded_missing_compound"] += 1
                continue
            if not plate:
                counters["excluded_missing_plate"] += 1
                continue
            if compound in reference_set:
                counters["excluded_136_plate_reference"] += 1
                continue
            if raw_dose is not None and abs(raw_dose - 20.0) <= TWENTY_MM_TOLERANCE:
                counters["excluded_20mM"] += 1
                continue
            if recoded is None:
                counters["excluded_nonstandard_or_missing_dose"] += 1
                continue
            if plate not in medians:
                counters["excluded_no_plate_control_median"] += 1
                continue
            dose_label, dose_value = recoded
            source_key = (compound, clean(row.get("Metadata_pert_id_dose")), plate,
                          clean(row.get("Metadata_Well")))
            if source_key in seen_source_keys:
                raise ValueError(f"Duplicate raw treatment row key: {source_key}")
            seen_source_keys.add(source_key)
            values = row[features].to_numpy(dtype=np.float32, copy=True)
            delta = values - medians[plate]
            if not np.isfinite(delta).all():
                raise ValueError(f"Non-finite delta at CSV row {source_row}: {compound}, {plate}")
            key = (compound, dose_label, plate)
            if key not in groups:
                groups[key] = empty_group(compound, dose_label, dose_value, plate, row, source_row, delta)
            else:
                group = groups[key]
                group["sum"] += delta
                group["n"] += 1
                group["raw_doses"].append(float(raw_dose) if raw_dose is not None else np.nan)
                group["wells"].append(clean(row.get("Metadata_Well")))
                group["source_rows"].append(source_row)
                for field, metadata_name in (("batches", "Metadata_Batch_Number"),
                                              ("batch_dates", "Metadata_Batch_Date"),
                                              ("plate_map_names", "Metadata_plate_map_name"),
                                              ("cell_ids", "Metadata_cell_id"),
                                              ("inchikey14", "Metadata_InChIKey14"),
                                              ("pert_id_dose", "Metadata_pert_id_dose")):
                    value = clean(row.get(metadata_name))
                    if value and value not in group[field]:
                        group[field].append(value)
                group["well_records"].append({
                    "source_row": source_row,
                    "well": clean(row.get("Metadata_Well")),
                    "raw_dose_mM": raw_dose,
                    "pert_id_dose": clean(row.get("Metadata_pert_id_dose")),
                })
    counters["included_treatment_wells"] = int(sum(group["n"] for group in groups.values()))
    counters["included_condition_rows"] = int(len(groups))
    counters["technical_duplicate_extra_rows"] = int(sum(group["n"] - 1 for group in groups.values()))
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        count = int(group["n"])
        raw_doses = np.asarray(group["raw_doses"], dtype=np.float64)
        wells = sorted(set(group["wells"]))
        records = sorted(group["well_records"], key=lambda item: (float(item["raw_dose_mM"] or np.inf), item["well"], item["source_row"]))
        output.append({
            "compound": group["compound"],
            "dose_standard": group["dose_standard"],
            "dose": float(group["dose"]),
            "plate": group["plate"],
            "delta": (group["sum"] / count).astype(np.float32),
            "technical_duplicate_count": count,
            "raw_dose_min_mM": float(np.nanmin(raw_doses)),
            "raw_dose_max_mM": float(np.nanmax(raw_doses)),
            "raw_dose_mean_mM": float(np.nanmean(raw_doses)),
            "well": "|".join(wells),
            "wells": wells,
            "source_rows": sorted(int(value) for value in group["source_rows"]),
            "batch": "|".join(sorted(set(group["batches"]))),
            "batch_date": "|".join(sorted(set(group["batch_dates"]))),
            "plate_map_name": "|".join(sorted(set(group["plate_map_names"]))),
            "cell_id": "|".join(sorted(set(group["cell_ids"]))),
            "inchikey14": "|".join(sorted(set(group["inchikey14"]))),
            "pert_id_dose": "|".join(sorted(set(group["pert_id_dose"]))),
            "well_records": records,
        })
    return output, dict(counters)


def split_compounds(compounds: list[str], seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    ordered = np.asarray(sorted(set(compounds)), dtype=object)
    rng = np.random.default_rng(seed)
    permutation = ordered[rng.permutation(len(ordered))]
    n = len(permutation)
    train_end = int(np.floor(n * SPLIT_FRACTIONS["train"]))
    valid_end = int(np.floor(n * (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["valid"])))
    return {
        "train": sorted(str(x) for x in permutation[:train_end]),
        "valid": sorted(str(x) for x in permutation[train_end:valid_end]),
        "test": sorted(str(x) for x in permutation[valid_end:]),
    }


def encode_strings(values: list[str]) -> np.ndarray:
    width = max([1] + [len(str(value).encode("utf-8")) for value in values])
    return np.asarray(values, dtype=f"S{width}")


def arrays_for_rows(rows: list[dict[str, Any]], feature_dim: int, split_name: str,
                   split_by_compound: dict[str, str]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError(f"Refusing to write empty {split_name} split")
    delta = np.vstack([row["delta"] for row in rows]).astype(np.float32, copy=False)
    if delta.shape[1] != feature_dim or not np.isfinite(delta).all():
        raise ValueError(f"Invalid {split_name} delta shape/values: {delta.shape}")
    compounds = [str(row["compound"]) for row in rows]
    return {
        "compound": encode_strings(compounds),
        "compound_id": encode_strings(compounds),
        "smiles": encode_strings(compounds),
        "dose_standard": encode_strings([str(row["dose_standard"]) for row in rows]),
        "dose": np.asarray([row["dose"] for row in rows], dtype=np.float32),
        "plate": encode_strings([str(row["plate"]) for row in rows]),
        "batch": encode_strings([str(row["batch"]) for row in rows]),
        "batch_date": encode_strings([str(row["batch_date"]) for row in rows]),
        "well": encode_strings([str(row["well"]) for row in rows]),
        "technical_duplicate_count": np.asarray([row["technical_duplicate_count"] for row in rows], dtype=np.int16),
        "raw_dose_min_mM": np.asarray([row["raw_dose_min_mM"] for row in rows], dtype=np.float32),
        "raw_dose_max_mM": np.asarray([row["raw_dose_max_mM"] for row in rows], dtype=np.float32),
        "raw_dose_mean_mM": np.asarray([row["raw_dose_mean_mM"] for row in rows], dtype=np.float32),
        "condition_key": encode_strings([
            f"{row['compound']}|{row['dose_standard']}|{row['plate']}" for row in rows
        ]),
        "split": encode_strings([split_by_compound[str(row["compound"])] for row in rows]),
        "delta": delta,
    }


def write_h5(path: Path, rows: list[dict[str, Any]], split_by_compound: dict[str, str],
             feature_names: list[str], attrs: dict[str, Any]) -> None:
    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    for split in ("train", "valid", "test"):
        groups[split] = [row for row in rows if split_by_compound[str(row["compound"])] == split]
    string_keys = {"compound", "compound_id", "smiles", "dose_standard", "plate", "batch", "batch_date",
                   "well", "condition_key", "split"}
    with h5py.File(path, "w") as handle:
        for key, value in attrs.items():
            handle.attrs[key] = value
        handle.attrs["feature_dim"] = len(feature_names)
        handle.create_dataset("feature_names", data=encode_strings(feature_names), compression="gzip", compression_opts=4)
        for split, split_rows in groups.items():
            payload = arrays_for_rows(split_rows, len(feature_names), split, split_by_compound)
            group = handle.create_group(split)
            for name, value in payload.items():
                if name in string_keys:
                    group.create_dataset(name, data=value, compression="gzip", compression_opts=4)
                else:
                    group.create_dataset(name, data=value, compression="gzip", compression_opts=4)


def write_npz_files(outdir: Path, rows: list[dict[str, Any]], split_by_compound: dict[str, str], feature_dim: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        split_rows = [row for row in rows if split_by_compound[str(row["compound"])] == split]
        payload = arrays_for_rows(split_rows, feature_dim, split, split_by_compound)
        np.savez_compressed(outdir / f"{split}_cp_plate_rows.npz", **payload)
        counts[split] = len(split_rows)
    return counts


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_manifests(outdir: Path, rows: list[dict[str, Any]], split_by_compound: dict[str, str],
                    control_stats: dict[str, dict[str, Any]]) -> None:
    condition_rows = []
    well_rows = []
    for row in rows:
        condition_key = f"{row['compound']}|{row['dose_standard']}|{row['plate']}"
        split = split_by_compound[str(row["compound"])]
        condition_rows.append({
            "condition_key": condition_key,
            "compound": row["compound"],
            "dose_standard": row["dose_standard"],
            "dose_mM_standard": row["dose"],
            "plate": row["plate"],
            "batch": row["batch"],
            "batch_date": row["batch_date"],
            "well": row["well"],
            "technical_duplicate_count": row["technical_duplicate_count"],
            "independent_repeat_unit": "plate",
            "split": split,
            "raw_dose_min_mM": row["raw_dose_min_mM"],
            "raw_dose_max_mM": row["raw_dose_max_mM"],
            "raw_dose_mean_mM": row["raw_dose_mean_mM"],
            "source_rows": "|".join(map(str, row["source_rows"])),
            "control_rows_on_plate": control_stats[row["plate"]]["control_rows"],
        })
        for duplicate_index, record in enumerate(row["well_records"]):
            well_rows.append({
                "condition_key": condition_key,
                "compound": row["compound"],
                "dose_standard": row["dose_standard"],
                "plate": row["plate"],
                "batch": row["batch"],
                "batch_date": row["batch_date"],
                "well": record["well"],
                "raw_dose_mM": record["raw_dose_mM"],
                "pert_id_dose": record["pert_id_dose"],
                "technical_duplicate_index": duplicate_index,
                "technical_duplicate_count": row["technical_duplicate_count"],
                "split": split,
                "source_row": record["source_row"],
            })
    write_csv(outdir / "condition_manifest.csv", condition_rows, list(condition_rows[0]))
    write_csv(outdir / "well_manifest.csv", well_rows, list(well_rows[0]))
    write_csv(outdir / "control_plate_manifest.csv", list(control_stats.values()), list(next(iter(control_stats.values()))))


def write_split_lock(path: Path, split_by_compound: dict[str, str], source_sha256: str,
                     input_path: Path, row_count: int) -> None:
    payload = {
        "version": VERSION,
        "seed": SPLIT_SEED,
        "assignment_unit": "compound_id",
        "ordering": "sorted unique compound IDs, numpy.default_rng(seed).permutation",
        "fractions_requested": SPLIT_FRACTIONS,
        "source_path": str(input_path),
        "source_sha256": source_sha256,
        "included_condition_rows": int(row_count),
    }
    for split in ("train", "valid", "test"):
        payload[f"{split}_compounds"] = sorted(key for key, value in split_by_compound.items() if value == split)
        payload[f"{split}_compound_count"] = len(payload[f"{split}_compounds"])
    payload["compound_count"] = len(split_by_compound)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def verify_outputs(outdir: Path, rows: list[dict[str, Any]], split_by_compound: dict[str, str],
                   feature_dim: int, reference_compounds: list[str], source_sha256: str) -> dict[str, Any]:
    compounds = set(split_by_compound)
    split_sets = {split: {compound for compound, value in split_by_compound.items() if value == split}
                  for split in ("train", "valid", "test")}
    if not compounds or any(not value for value in split_sets.values()):
        raise ValueError(f"Invalid split compound counts: { {k: len(v) for k, v in split_sets.items()} }")
    if (split_sets["train"] & split_sets["valid"] or split_sets["train"] & split_sets["test"] or
            split_sets["valid"] & split_sets["test"]):
        raise ValueError("Compound overlap across split lock")
    condition_keys = [f"{row['compound']}|{row['dose_standard']}|{row['plate']}" for row in rows]
    if len(condition_keys) != len(set(condition_keys)):
        raise ValueError("Duplicate condition keys after aggregation")
    if any(row["compound"] in set(reference_compounds) for row in rows):
        raise ValueError("Reference compound leaked into included rows")
    if any(abs(float(row["dose"]) - 20.0) <= TWENTY_MM_TOLERANCE for row in rows):
        raise ValueError("20 mM row leaked into included rows")
    split_row_counts = {split: sum(1 for row in rows if split_by_compound[row["compound"]] == split)
                        for split in ("train", "valid", "test")}
    expected_npz = {split: outdir / f"{split}_cp_plate_rows.npz" for split in split_row_counts}
    for split, path in expected_npz.items():
        with np.load(path, allow_pickle=False) as loaded:
            if loaded["delta"].shape != (split_row_counts[split], feature_dim):
                raise ValueError(f"{split} NPZ shape mismatch: {loaded['delta'].shape}")
            if not np.isfinite(loaded["delta"]).all():
                raise ValueError(f"{split} NPZ contains non-finite delta")
    return {
        "included_condition_rows": len(rows),
        "included_compounds": len(compounds),
        "split_compound_counts": {key: len(value) for key, value in split_sets.items()},
        "split_condition_row_counts": split_row_counts,
        "feature_dim": feature_dim,
        "reference_compounds_excluded": list(reference_compounds),
        "source_sha256": source_sha256,
        "guards_passed": [
            "compound-disjoint train/valid/test",
            "no 20 mM conditions",
            "no 136-plate reference compounds",
            "unique (compound, standard dose, plate) condition keys",
            "all CP deltas finite",
            "per-plate control medians present",
        ],
    }


def dry_run(args: argparse.Namespace) -> None:
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    columns, metadata, features = select_columns(args.input_csv)
    sample = pd.read_csv(args.input_csv, usecols=metadata, nrows=8, low_memory=False)
    print(json.dumps({
        "version": VERSION,
        "input_path": str(args.input_csv),
        "input_size_bytes": args.input_csv.stat().st_size,
        "input_sha256": sha256(args.input_csv),
        "n_columns": len(columns),
        "n_metadata_columns": len(metadata),
        "n_feature_columns": len(features),
        "feature_head": features[:8],
        "required_metadata_present": all(name in columns for name in REQUIRED_METADATA),
        "sample_rows": len(sample),
        "split_seed": SPLIT_SEED,
        "split_fractions": SPLIT_FRACTIONS,
        "exclusion_rules": {
            "20mM_tolerance": TWENTY_MM_TOLERANCE,
            "reference_plate_threshold": REFERENCE_PLATE_THRESHOLD,
        },
    }, indent=2, sort_keys=True))


def run(args: argparse.Namespace) -> None:
    if not args.input_csv.is_file():
        raise FileNotFoundError(args.input_csv)
    if args.outdir.exists() and not args.allow_existing:
        raise FileExistsError(f"Refusing to write existing output directory: {args.outdir}; use --allow-existing")
    args.outdir.mkdir(parents=True, exist_ok=True)
    columns, metadata, features = select_columns(args.input_csv)
    source_sha256 = sha256(args.input_csv)
    raw_summary = first_pass(args.input_csv, metadata, args.chunksize)
    medians, control_stats = control_pass(args.input_csv, metadata, features, args.chunksize)
    rows, treatment_summary = treatment_pass(
        args.input_csv, metadata, features, args.chunksize, medians, raw_summary["reference_compounds"]
    )
    if not rows:
        raise RuntimeError("No included CP condition rows")
    split = split_compounds([row["compound"] for row in rows])
    split_by_compound = {compound: split_name for split_name, values in split.items() for compound in values}
    if set(split_by_compound) != {row["compound"] for row in rows}:
        raise ValueError("Split lock does not cover exactly the included compounds")
    npz_counts = write_npz_files(args.outdir, rows, split_by_compound, len(features))
    attrs = {
        "version": VERSION,
        "source_path": str(args.input_csv),
        "source_sha256": source_sha256,
        "split_seed": SPLIT_SEED,
        "condition_definition": "(compound_id, standard_dose, plate)",
        "independent_repeat_definition": "distinct plate within (compound_id, standard_dose)",
        "technical_duplicate_definition": "distinct treatment wells sharing (compound_id, standard_dose, plate)",
        "delta_definition": "treatment feature vector minus feature-wise median of all control rows on the same plate",
        "technical_duplicate_aggregation": "feature-wise arithmetic mean of per-well deltas",
        "exclusion_20mM_tolerance": TWENTY_MM_TOLERANCE,
        "reference_plate_threshold": REFERENCE_PLATE_THRESHOLD,
        "reference_compounds": json.dumps(raw_summary["reference_compounds"]),
    }
    write_h5(args.outdir / "cp_plate_rows.h5", rows, split_by_compound, features, attrs)
    np.savez_compressed(args.outdir / "control_plate_medians.npz",
                        plate=encode_strings(sorted(medians)),
                        median=np.vstack([medians[plate] for plate in sorted(medians)]).astype(np.float32))
    write_manifests(args.outdir, rows, split_by_compound, control_stats)
    write_split_lock(args.outdir / "split_lock_seed3407.json", split_by_compound, source_sha256,
                     args.input_csv, len(rows))
    summary = {
        "version": VERSION,
        "input": {"path": str(args.input_csv), "sha256": source_sha256, "size_bytes": args.input_csv.stat().st_size},
        "columns": {"n_total": len(columns), "n_metadata": len(metadata), "n_features": len(features),
                     "feature_columns": features},
        "raw_audit": raw_summary,
        "control_audit": {"plate_count": len(control_stats), "plates": control_stats},
        "treatment_audit": treatment_summary,
        "split": {"seed": SPLIT_SEED, "fractions": SPLIT_FRACTIONS,
                   "compound_counts": {key: len(value) for key, value in split.items()},
                   "condition_row_counts": npz_counts},
        "outputs": {
            "h5": str(args.outdir / "cp_plate_rows.h5"),
            "npz": {key: str(args.outdir / f"{key}_cp_plate_rows.npz") for key in npz_counts},
            "condition_manifest": str(args.outdir / "condition_manifest.csv"),
            "well_manifest": str(args.outdir / "well_manifest.csv"),
            "control_plate_manifest": str(args.outdir / "control_plate_manifest.csv"),
            "split_lock": str(args.outdir / "split_lock_seed3407.json"),
        },
    }
    summary["verification"] = verify_outputs(
        args.outdir, rows, split_by_compound, len(features), raw_summary["reference_compounds"], source_sha256
    )
    (args.outdir / "build_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    checksums = []
    for path in sorted(args.outdir.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            checksums.append(f"{sha256(path)}  {path.name}")
    (args.outdir / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps({
        "version": VERSION,
        "outdir": str(args.outdir),
        "feature_dim": len(features),
        "condition_rows": len(rows),
        "compound_counts": summary["split"]["compound_counts"],
        "condition_row_counts": npz_counts,
        "reference_compounds_excluded": raw_summary["reference_compounds"],
        "technical_duplicate_extra_rows": treatment_summary["technical_duplicate_extra_rows"],
        "guards": summary["verification"]["guards_passed"],
    }, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
