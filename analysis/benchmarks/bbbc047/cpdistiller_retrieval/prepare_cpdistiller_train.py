#!/usr/bin/env python3
"""Prepare strictly train-only BBBC047 inputs for official cpDistiller.

The source CellProfiler CSV is streamed twice.  The first pass reads only the
metadata prefix and picks one deterministic negative-control well for each
training batch.  The second pass parses CellProfiler values only for mapped
training treatments and those chosen controls.  Held-out source feature values
are never converted or retained by this script.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROLE_SEED = 3407
EXPECTED_CP_FEATURES = 1783
CP_PREFIXES = ("Cells_", "Cytoplasm_", "Nuclei_")
WELL_PATTERN = re.compile(r"^([A-Za-z]+)0*([0-9]+)$")


def stable_int(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def well_parts(value: str) -> tuple[str, str]:
    match = WELL_PATTERN.match(value.strip())
    if match is None:
        raise ValueError(f"unparseable well label: {value!r}")
    return match.group(1).upper(), str(int(match.group(2)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profiles", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def source_schema(source_path: Path) -> tuple[list[str], int, list[str], list[int], int]:
    with gzip.open(source_path, "rt", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    meta_count = next((idx for idx, name in enumerate(header) if not name.startswith("Metadata_")), None)
    if meta_count is None:
        raise RuntimeError("source table has no CellProfiler columns")
    metadata = header[:meta_count]
    payload = header[meta_count:]
    feature_offsets = [idx for idx, name in enumerate(payload) if name.startswith(CP_PREFIXES)]
    features = [payload[idx] for idx in feature_offsets]
    required = {
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Assay_Plate_Barcode",
        "Metadata_ASSAY_WELL_ROLE",
        "Metadata_pert_type",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise RuntimeError(f"source metadata columns missing: {missing}")
    if len(features) != EXPECTED_CP_FEATURES:
        raise RuntimeError(
            f"unexpected CellProfiler dimension {len(features)}; expected {EXPECTED_CP_FEATURES}"
        )
    if feature_offsets != list(range(EXPECTED_CP_FEATURES)):
        raise RuntimeError("CellProfiler columns are not the expected contiguous leading payload")
    return metadata, meta_count, features, feature_offsets, len(payload)


def load_training_mapping(path: Path) -> tuple[dict[int, dict[str, str]], set[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "source_row_index", "split", "compound", "dose", "plate", "well", "row", "column", "batch"
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"mapping missing fields: {sorted(missing)}")
        rows: dict[int, dict[str, str]] = {}
        for row in reader:
            if row["split"] != "train":
                continue
            source_idx = int(row["source_row_index"])
            if source_idx in rows:
                raise RuntimeError(f"duplicate source row in train mapping: {source_idx}")
            rows[source_idx] = row
    if not rows:
        raise RuntimeError("empty training mapping")
    return rows, {row["batch"] for row in rows.values()}


def metadata_prefix(line: str, meta_count: int) -> list[str]:
    # The source format has all metadata before CellProfiler values, with no
    # quoted commas in that metadata prefix.  This avoids parsing test features
    # while scanning for train-batch control wells.
    prefix = line.rstrip("\r\n").split(",", meta_count)
    if len(prefix) < meta_count:
        raise RuntimeError("malformed source line before CellProfiler payload")
    return prefix[:meta_count]


def choose_negative_controls(
    source_path: Path,
    metadata: list[str],
    meta_count: int,
    train_batches: set[str],
) -> tuple[set[int], Counter[str]]:
    index = {name: idx for idx, name in enumerate(metadata)}
    controls: set[int] = set()
    batches_with_controls: set[str] = set()
    roles: Counter[str] = Counter()
    with gzip.open(source_path, "rt", newline="", encoding="utf-8") as handle:
        next(handle)  # header
        for source_idx, line in enumerate(handle):
            values = metadata_prefix(line, meta_count)
            batch = values[index["Metadata_Assay_Plate_Barcode"]].strip()
            if batch not in train_batches:
                continue
            pert_type = values[index["Metadata_pert_type"]].strip().lower()
            if pert_type == "trt":
                continue
            role = values[index["Metadata_ASSAY_WELL_ROLE"]].strip().lower()
            roles[role] += 1
            controls.add(source_idx)
            batches_with_controls.add(batch)
    missing = sorted(train_batches - batches_with_controls)
    if missing:
        raise RuntimeError(f"training batches with no source negative control: {missing[:10]}")
    return controls, roles


def write_control_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["source_row_index", "batch", "plate", "well", "row", "column", "assay_well_role"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.outdir}")
    train_mapping, train_batches = load_training_mapping(args.mapping)
    metadata, meta_count, features, feature_offsets, payload_count = source_schema(args.source_profiles)
    control_source_rows, observed_non_treatment_roles = choose_negative_controls(
        args.source_profiles, metadata, meta_count, train_batches
    )
    source_indices = set(train_mapping) | control_source_rows
    expected_rows = len(train_mapping) + len(control_source_rows)
    if len(source_indices) != expected_rows:
        raise RuntimeError("training treatment and control source rows overlap")

    args.outdir.mkdir(parents=True)
    index = {name: idx for idx, name in enumerate(metadata)}
    source_order = sorted(source_indices)
    output_index = {source_idx: output_idx for output_idx, source_idx in enumerate(source_order)}
    X = np.empty((expected_rows, len(features)), dtype=np.float32)
    source_row_index = np.asarray(source_order, dtype=np.int64)
    row_labels = np.empty(expected_rows, dtype=f"U{max(2, 4)}")
    col_labels = np.empty(expected_rows, dtype="U3")
    batch_labels = np.empty(expected_rows, dtype="U16")
    control_labels = np.empty(expected_rows, dtype="U9")
    compounds = np.empty(expected_rows, dtype="U512")
    doses = np.empty(expected_rows, dtype="U32")
    plates = np.empty(expected_rows, dtype="U32")
    control_manifest: list[dict[str, Any]] = []
    found: set[int] = set()

    # Only selected training/negative-control source rows are fully CSV-parsed
    # and converted to floats.  Every other line, including all test rows, is
    # bypassed before its CellProfiler payload is parsed.
    with gzip.open(args.source_profiles, "rt", newline="", encoding="utf-8") as handle:
        next(handle)  # header
        for idx_source, line in enumerate(handle):
            if idx_source not in source_indices:
                continue
            values = next(csv.reader([line]))
            if len(values) != meta_count + payload_count:
                raise RuntimeError(f"source row {idx_source}: schema width mismatch")
            output_idx = output_index[idx_source]
            well = values[index["Metadata_Well"]].strip()
            row_label, col_label = well_parts(well)
            batch = values[index["Metadata_Assay_Plate_Barcode"]].strip()
            plate = values[index["Metadata_Plate"]].strip()
            if not batch:
                raise RuntimeError(f"source row {idx_source}: empty batch")
            try:
                X[output_idx] = np.asarray(
                    [values[meta_count + offset] for offset in feature_offsets], dtype=np.float32
                )
            except ValueError as error:
                raise RuntimeError(f"source row {idx_source}: nonnumeric CellProfiler value") from error
            row_labels[output_idx] = row_label
            col_labels[output_idx] = col_label
            batch_labels[output_idx] = batch
            plates[output_idx] = plate
            if idx_source in train_mapping:
                mapped = train_mapping[idx_source]
                if mapped["batch"] != batch or mapped["plate"] != plate:
                    raise RuntimeError(f"source row {idx_source}: mapping technical-label mismatch")
                control_labels[output_idx] = "treatment"
                compounds[output_idx] = mapped["compound"]
                doses[output_idx] = mapped["dose"]
            else:
                if batch not in train_batches:
                    raise RuntimeError(f"control row {idx_source}: non-training batch")
                if values[index["Metadata_pert_type"]].strip().lower() == "trt":
                    raise RuntimeError(f"control row {idx_source}: treatment selected as a control")
                control_labels[output_idx] = "negative"
                compounds[output_idx] = "__NEGATIVE_CONTROL__"
                doses[output_idx] = ""
                control_manifest.append(
                    {
                        "source_row_index": idx_source,
                        "batch": batch,
                        "plate": plate,
                        "well": well.upper(),
                        "row": row_label,
                        "column": col_label,
                        "assay_well_role": values[index["Metadata_ASSAY_WELL_ROLE"]].strip(),
                    }
                )
            found.add(idx_source)
    missing_source_rows = sorted(source_indices - found)
    if missing_source_rows:
        raise RuntimeError(f"selected source rows absent from profile table: {missing_source_rows[:10]}")

    # The official tutorial applies MAD normalization per plate using negative
    # controls, then Scanpy scaling per plate and once again globally.  These
    # statistics are estimated only from mapped training treatments and source
    # negative controls on batches represented in training.
    X_work = X.astype(np.float64)
    unique_plates = np.asarray(sorted(set(plates.tolist())))
    plate_control_center = np.empty((len(unique_plates), X.shape[1]), dtype=np.float64)
    plate_control_mad = np.empty_like(plate_control_center)
    plate_post_mean = np.empty_like(plate_control_center)
    plate_post_std = np.empty_like(plate_control_center)
    for plate_index, plate_label in enumerate(unique_plates):
        plate_mask = plates == plate_label
        control_mask = plate_mask & (control_labels == "negative")
        reference = X_work[control_mask] if control_mask.any() else X_work[plate_mask]
        center = np.nanmedian(reference, axis=0)
        center[~np.isfinite(center)] = 0.0
        mad = np.nanmedian(np.abs(reference - center), axis=0) * 1.4826
        mad[~np.isfinite(mad)] = 0.0
        block = X_work[plate_mask]
        invalid = ~np.isfinite(block)
        if invalid.any():
            block[invalid] = np.broadcast_to(center, block.shape)[invalid]
        block = (block - center) / (mad + 1e-18)
        post_mean = np.mean(block, axis=0)
        post_std = np.std(block, axis=0, ddof=1)
        post_mean[~np.isfinite(post_mean)] = 0.0
        post_std[~np.isfinite(post_std) | (post_std <= 1e-12)] = 1.0
        X_work[plate_mask] = (block - post_mean) / post_std
        plate_control_center[plate_index] = center
        plate_control_mad[plate_index] = mad
        plate_post_mean[plate_index] = post_mean
        plate_post_std[plate_index] = post_std
    global_mean = np.mean(X_work, axis=0)
    global_std = np.std(X_work, axis=0, ddof=1)
    global_mean[~np.isfinite(global_mean)] = 0.0
    global_std[~np.isfinite(global_std) | (global_std <= 1e-12)] = 1.0
    X = ((X_work - global_mean) / global_std).astype(np.float32)
    if not np.isfinite(X).all():
        raise RuntimeError("non-finite values remain after train-only preprocessing")

    feature_schema_path = args.outdir / "FEATURE_SCHEMA.json"
    feature_schema_path.write_text(
        json.dumps({"feature_count": len(features), "features": features}, indent=2) + "\n",
        encoding="utf-8",
    )
    control_manifest_path = args.outdir / "TRAIN_NEGATIVE_CONTROL_MANIFEST.csv"
    write_control_manifest(control_manifest_path, sorted(control_manifest, key=lambda row: int(row["source_row_index"])))
    np.savez_compressed(
        args.outdir / "TRAIN_INPUTS.npz",
        X=X,
        source_row_index=source_row_index,
        row=row_labels,
        col=col_labels,
        batch=batch_labels,
        control=control_labels,
        compound=compounds,
        dose=doses,
        plate=plates,
        preprocessing_plate=unique_plates,
        plate_control_center=plate_control_center.astype(np.float32),
        plate_control_mad=plate_control_mad.astype(np.float32),
        plate_post_mean=plate_post_mean.astype(np.float32),
        plate_post_std=plate_post_std.astype(np.float32),
        global_mean=global_mean.astype(np.float32),
        global_std=global_std.astype(np.float32),
    )
    audit = {
        "audit": "BBBC047-cpDistiller-train-inputs-v1-2026-09-19",
        "role_seed": ROLE_SEED,
        "source_profiles": str(args.source_profiles),
        "source_sha256": sha256_file(args.source_profiles),
        "mapping": str(args.mapping),
        "mapping_sha256": sha256_file(args.mapping),
        "expected_cp_feature_count": EXPECTED_CP_FEATURES,
        "feature_schema_sha256": sha256_file(feature_schema_path),
        "preprocessing": {
            "fit_rows": "mapped training treatment wells plus all source negative controls from batches represented in training",
            "feature_rule": "contiguous Cells_, Cytoplasm_, and Nuclei_ columns only",
            "plate_step": "official tutorial MAD convention using plate negative controls, followed by per-plate mean/std scaling",
            "global_step": "final train-only mean/std scaling as in the official tutorial",
            "held_out_transform": "frozen plate parameters where available; frozen global fallback for unseen plates",
        },
        "n_training_treatment_source_wells": len(train_mapping),
        "n_training_batches": len(train_batches),
        "n_selected_negative_controls": len(control_source_rows),
        "selected_control_role_counts": dict(sorted(observed_non_treatment_roles.items())),
        "total_fit_rows": expected_rows,
        "source_feature_values_materialized": "training treatments and selected training-batch negative controls only",
        "validation_source_feature_values_materialized": False,
        "test_source_feature_values_materialized": False,
        "test_used_for_fit_or_selection": False,
        "status": "PASS",
    }
    (args.outdir / "TRAIN_INPUT_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
