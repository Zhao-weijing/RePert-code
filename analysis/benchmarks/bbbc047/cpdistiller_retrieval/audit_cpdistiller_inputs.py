#!/usr/bin/env python3
"""Fail-closed input audit for the BBBC047 cpDistiller companion comparison.

The script reads only identity metadata from the existing NPZ archives.  It
never accesses ``delta`` values, fits a model, or produces retrieval metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROWS_ROOT = Path(
    "/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"
)
REQUIRED_NPZ_FIELDS = {"smiles", "dose", "plate", "delta"}
REQUIRED_MAPPING_COLUMNS = {
    "compound",
    "dose",
    "plate",
    "well",
    "row",
    "column",
    "batch",
}


def decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="strict")
    return str(value)


def dose_key(value: Any) -> str:
    text = decode(value).strip()
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-root", type=Path, default=DEFAULT_ROWS_ROOT)
    parser.add_argument(
        "--profile-mapping",
        type=Path,
        help=(
            "CSV joining every source well profile to compound,dose,plate,well,"
            "row,column,batch. Multiple wells per physical identity are allowed."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_npz_identities(path: Path) -> tuple[list[tuple[str, str, str]], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        fields = set(payload.files)
        missing = sorted(REQUIRED_NPZ_FIELDS - fields)
        if missing:
            raise RuntimeError(f"{path}: missing NPZ fields {missing}")
        smiles = np.asarray(payload["smiles"])
        dose = np.asarray(payload["dose"])
        plate = np.asarray(payload["plate"])
        if not (len(smiles) == len(dose) == len(plate)):
            raise RuntimeError(f"{path}: identity lengths disagree")
        identities = [
            (decode(compound), dose_key(concentration), decode(plate_id))
            for compound, concentration, plate_id in zip(smiles, dose, plate)
        ]
    counts = Counter(identities)
    return identities, {
        "path": str(path),
        "npz_fields": sorted(fields),
        "n_rows": len(identities),
        "n_unique_compound_dose_plate": len(counts),
        "n_duplicate_compound_dose_plate": sum(value > 1 for value in counts.values()),
        "contains_well_row_column_batch_fields": bool(
            {"well", "row", "column", "batch"} <= fields
        ),
        "delta_values_loaded": False,
    }


def read_mapping(path: Path) -> tuple[set[tuple[str, str, str]], dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_MAPPING_COLUMNS - fieldnames)
        if missing:
            raise RuntimeError(f"{path}: missing mapping columns {missing}")
        keys: list[tuple[str, str, str]] = []
        blank_rows = 0
        for row in reader:
            required_values = {column: (row.get(column) or "").strip() for column in REQUIRED_MAPPING_COLUMNS}
            if any(not value for value in required_values.values()):
                blank_rows += 1
                continue
            keys.append(
                (
                    required_values["compound"],
                    dose_key(required_values["dose"]),
                    required_values["plate"],
                )
            )
    counts = Counter(keys)
    return set(keys), {
        "path": str(path),
        "columns": sorted(fieldnames),
        "n_rows": len(keys),
        "n_blank_required_label_rows": blank_rows,
        "n_duplicate_compound_dose_plate": sum(value > 1 for value in counts.values()),
    }


def main() -> None:
    args = parse_args()
    split_identities: dict[str, set[tuple[str, str, str]]] = {}
    splits: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid", "test"):
        identities, summary = read_npz_identities(args.rows_root / f"{split}_cp_plate_rows.npz")
        split_identities[split] = set(identities)
        splits[split] = summary

    report: dict[str, Any] = {
        "audit": "BBBC047-cpDistiller-input-preflight-v1-2026-09-19",
        "rows_root": str(args.rows_root),
        "splits": splits,
        "test_profile_values_loaded": False,
        "test_used_for_fitting": False,
        "decision": None,
        "reasons": [],
    }

    if args.profile_mapping is None:
        report["decision"] = "BLOCKED_MISSING_PROFILE_TO_POSITION_MAPPING"
        report["reasons"].append(
            "The existing NPZ files contain no well, row, column, or batch fields. "
            "A complete source-well provenance mapping is required for faithful "
            "cpDistiller training."
        )
    else:
        mapping_keys, mapping_summary = read_mapping(args.profile_mapping)
        report["profile_mapping"] = mapping_summary
        missing_by_split = {
            split: len(keys - mapping_keys) for split, keys in split_identities.items()
        }
        report["unmapped_compound_dose_plate_by_split"] = missing_by_split
        if any(missing_by_split.values()):
            report["decision"] = "BLOCKED_INCOMPLETE_PROFILE_TO_POSITION_MAPPING"
            report["reasons"].append(
                "The mapping does not cover every retained physical profile identity."
            )
        elif mapping_summary["n_blank_required_label_rows"]:
            report["decision"] = "BLOCKED_BLANK_TECHNICAL_LABELS"
            report["reasons"].append(
                "The mapping contains blank well, row, column, or batch labels."
            )
        else:
            report["decision"] = "READY_FOR_UPSTREAM_CODE_AUDIT"
            report["reasons"].append(
                "Identity and required technical-label coverage passed. This does not "
                "authorize fitting until the official cpDistiller CellProfiler-only mode "
                "and a train-only implementation are verified."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not str(report["decision"]).startswith("READY_"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
