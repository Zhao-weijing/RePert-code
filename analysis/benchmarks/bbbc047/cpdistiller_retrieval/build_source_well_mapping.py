#!/usr/bin/env python3
"""Build and audit the BBBC047 source-well mapping for cpDistiller.

Only identity and technical-label columns are retained. CellProfiler feature
values are neither converted nor written. The output maps one or more source
wells to each retained physical ``(compound, dose, plate)`` identity.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem


DEFAULT_ROWS_ROOT = Path(
    "/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"
)
DEFAULT_SOURCE = Path(
    "/path/to/data/AIDD/CDRP-BBBC047-Bray/CellPainting/"
    "replicate_level_cp_augmented.csv.gz"
)
DEFAULT_L1000_PROFILES = Path(
    "/path/to/data/AIDD/CDRP-BBBC047-Bray/L1000/replicate_level_l1k.csv.gz"
)

SOURCE_COLUMNS = {
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Assay_Plate_Barcode",
    "Metadata_Plate_Map_Name",
    "Metadata_broad_sample",
    "Metadata_pert_id",
    "Metadata_Sample_Dose",
    "Metadata_mmoles_per_liter2",
    "Metadata_pert_type",
}
WELL_PATTERN = re.compile(r"^([A-Za-z]+)0*([0-9]+)$")


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


def canonical_smiles(value: str) -> str:
    """Apply the exact RDKit canonical/isomeric convention in the source notebook."""
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError(f"unparseable SMILES {value!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-root", type=Path, default=DEFAULT_ROWS_ROOT)
    parser.add_argument("--source-profiles", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--l1000-profiles", type=Path, default=DEFAULT_L1000_PROFILES)
    parser.add_argument("--mapping-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def read_npz_keys(rows_root: Path) -> tuple[dict[str, set[tuple[str, str, str]]], dict[str, Any]]:
    keys_by_split: dict[str, set[tuple[str, str, str]]] = {}
    summary: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        path = rows_root / f"{split}_cp_plate_rows.npz"
        with np.load(path, allow_pickle=False) as payload:
            required = {"smiles", "dose", "plate", "delta"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise RuntimeError(f"{path}: missing fields {missing}")
            smiles = np.asarray(payload["smiles"])
            dose = np.asarray(payload["dose"])
            plate = np.asarray(payload["plate"])
            keys = {
                (decode(compound), dose_key(concentration), decode(plate_id))
                for compound, concentration, plate_id in zip(smiles, dose, plate)
            }
        keys_by_split[split] = keys
        summary[split] = {
            "path": str(path),
            "n_physical_identities": len(keys),
            "delta_values_loaded": False,
        }
    return keys_by_split, summary


def read_smiles_by_parent_brd_id(path: Path) -> tuple[dict[str, str], int, int]:
    """Recover the source notebook's ``Metadata_pert_id -> CPD_SMILES`` map.

    The frozen CP table was built by mapping the source CP ``Treatment``
    (the parent BRD perturbation ID) to L1000 ``BROAD_CPD_ID``.  This differs
    from the publication's full sample-dose key and preserves the exact
    historical identity convention used to form the frozen SMILES split.
    No numerical L1000 measurements are converted or retained.
    """
    mapping: dict[str, str] = {}
    retained_rows = 0
    invalid_smiles = 0
    try:
        chunks = pd.read_csv(
            path,
            compression="gzip",
            usecols=["BROAD_CPD_ID", "CPD_SMILES"],
            dtype=str,
            keep_default_na=False,
            chunksize=16_384,
        )
        for chunk in chunks:
            for parent_brd_id, smiles in zip(chunk["BROAD_CPD_ID"], chunk["CPD_SMILES"]):
                parent_brd_id = parent_brd_id.strip()
                smiles = smiles.strip()
                if not parent_brd_id or not smiles:
                    continue
                try:
                    smiles = canonical_smiles(smiles)
                except ValueError:
                    invalid_smiles += 1
                    continue
                retained_rows += 1
                existing = mapping.setdefault(parent_brd_id, smiles)
                if existing != smiles:
                    raise RuntimeError(
                        f"{path}: conflicting SMILES for BROAD_CPD_ID={parent_brd_id!r}"
                    )
    except ValueError as error:
        raise RuntimeError(f"{path}: missing L1000 parent-ID identity columns") from error
    return mapping, retained_rows, invalid_smiles


def well_parts(value: str) -> tuple[str, str]:
    match = WELL_PATTERN.match(value.strip())
    if match is None:
        raise ValueError(f"unparseable well label {value!r}")
    return match.group(1).upper(), str(int(match.group(2)))


def main() -> None:
    args = parse_args()
    split_keys, split_summary = read_npz_keys(args.rows_root)
    all_keys = set().union(*split_keys.values())
    split_by_key = {key: split for split, keys in split_keys.items() for key in keys}
    (
        smiles_by_parent_brd_id,
        l1000_identity_rows,
        invalid_l1000_smiles,
    ) = read_smiles_by_parent_brd_id(args.l1000_profiles)

    args.mapping_output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[tuple[str, str, str]] = Counter()
    invalid_labels = 0
    matched_source_rows = 0
    unmapped_parent_brd_ids = 0
    treatment_rows = 0

    fieldnames = [
        "source_row_index", "split", "compound", "dose", "plate", "well",
        "row", "column", "batch", "plate_map_name", "source_profile_file",
    ]
    write_header = True
    try:
        chunks = pd.read_csv(
            args.source_profiles,
            compression="gzip",
            usecols=sorted(SOURCE_COLUMNS),
            dtype=str,
            keep_default_na=False,
            chunksize=4_096,
        )
        for chunk in chunks:
            treatments = chunk.loc[
                chunk["Metadata_pert_type"].str.strip().str.lower().eq("trt")
            ].copy()
            treatment_rows += len(treatments)
            if treatments.empty:
                continue
            treatments["_compound"] = treatments["Metadata_pert_id"].str.strip().map(
                smiles_by_parent_brd_id
            )
            unmapped_parent_brd_ids += int(treatments["_compound"].isna().sum())
            treatments = treatments.loc[treatments["_compound"].notna()].copy()
            treatments["_plate"] = treatments["Metadata_Plate"].str.strip()
            treatments["_dose"] = treatments["Metadata_mmoles_per_liter2"].map(dose_key)
            treatments["_key"] = list(
                zip(treatments["_compound"], treatments["_dose"], treatments["_plate"])
            )
            treatments = treatments.loc[treatments["_key"].isin(all_keys)].copy()
            if treatments.empty:
                continue
            well_parts_frame = treatments["Metadata_Well"].str.strip().str.extract(WELL_PATTERN)
            valid = well_parts_frame[0].notna() & treatments["Metadata_Assay_Plate_Barcode"].str.strip().ne("")
            invalid_labels += int((~valid).sum())
            treatments = treatments.loc[valid].copy()
            well_parts_frame = well_parts_frame.loc[valid]
            if treatments.empty:
                continue
            counts.update(treatments["_key"].tolist())
            matched_source_rows += len(treatments)
            output = pd.DataFrame(
                {
                    "source_row_index": treatments.index,
                    "split": treatments["_key"].map(split_by_key),
                    "compound": treatments["_compound"],
                    "dose": treatments["_dose"],
                    "plate": treatments["_plate"],
                    "well": treatments["Metadata_Well"].str.strip().str.upper(),
                    "row": well_parts_frame[0].str.upper(),
                    "column": well_parts_frame[1].astype(int).astype(str),
                    "batch": treatments["Metadata_Assay_Plate_Barcode"].str.strip(),
                    "plate_map_name": treatments["Metadata_Plate_Map_Name"].str.strip(),
                    "source_profile_file": str(args.source_profiles),
                }
            )
            output.to_csv(
                args.mapping_output,
                mode="w" if write_header else "a",
                header=write_header,
                index=False,
                columns=fieldnames,
            )
            write_header = False
    except ValueError as error:
        raise RuntimeError(f"{args.source_profiles}: missing source metadata columns") from error
    if write_header:
        pd.DataFrame(columns=fieldnames).to_csv(args.mapping_output, index=False)

    missing_by_split = {
        split: len(keys - set(counts)) for split, keys in split_keys.items()
    }
    rows_per_identity = list(counts.values())
    report: dict[str, Any] = {
        "audit": "BBBC047-cpDistiller-source-well-mapping-v4-2026-09-19",
        "source_profiles": str(args.source_profiles),
        "l1000_profiles": str(args.l1000_profiles),
        "historical_frozen_cp_join": "Metadata_pert_id == BROAD_CPD_ID",
        "l1000_identity_rows_seen": l1000_identity_rows,
        "n_invalid_l1000_smiles": invalid_l1000_smiles,
        "mapping_output": str(args.mapping_output),
        "npz_splits": split_summary,
        "n_source_treatment_rows_seen": treatment_rows,
        "n_source_rows_matched_to_retained_physical_identities": matched_source_rows,
        "n_retained_physical_identities_matched": len(counts),
        "n_unmapped_source_parent_brd_ids": unmapped_parent_brd_ids,
        "n_invalid_well_or_batch_labels": invalid_labels,
        "source_wells_per_physical_identity": {
            "min": min(rows_per_identity) if rows_per_identity else None,
            "median": median(rows_per_identity) if rows_per_identity else None,
            "max": max(rows_per_identity) if rows_per_identity else None,
        },
        "missing_physical_identities_by_split": missing_by_split,
        "source_cp_feature_values_loaded": False,
        "l1000_gene_feature_values_loaded": False,
        "test_used_for_fitting": False,
        "decision": None,
        "reasons": [],
    }
    if any(missing_by_split.values()):
        report["decision"] = "BLOCKED_INCOMPLETE_SOURCE_WELL_COVERAGE"
        report["reasons"].append(
            "At least one retained physical identity has no matched source-well profile."
        )
    elif invalid_labels:
        report["decision"] = "BLOCKED_INVALID_SOURCE_TECHNICAL_LABELS"
        report["reasons"].append(
            "At least one matched source profile has no valid well or batch label."
        )
    else:
        report["decision"] = "READY_FOR_UPSTREAM_CODE_AUDIT"
        report["reasons"].append(
            "Source well coverage and physical-identity mapping passed. This does not "
            "authorize model fitting until the official cpDistiller CellProfiler-only "
            "mode and train-only preprocessing are verified."
        )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["decision"] != "READY_FOR_UPSTREAM_CODE_AUDIT":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
