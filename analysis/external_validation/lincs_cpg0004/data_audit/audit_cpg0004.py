#!/usr/bin/env python3
"""Phase-0, metadata-only audit for the LINCS Pilot 1 / cpg0004 candidate.

The script deliberately reads only metadata columns.  It does not build model
inputs, aggregate profiles, or train a model.  It writes compact audit tables
that make provenance, repeat structure, CP--GE mapping, and raw acquisition
independence from BBBC047 inspectable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


LINCS_CP = (
    "/path/to/home/AIDD/baseline/data/MVC/LINCS-Pilot1/CellPainting/"
    "replicate_level_cp_normalized_variable_selected.csv.gz"
)
LINCS_GE = "/path/to/home/AIDD/baseline/data/MVC/LINCS-Pilot1/L1000/replicate_level_l1k.csv.gz"
BBBC047_CP = (
    "/path/to/data/AIDD/CDRP-BBBC047-Bray/CellPainting/"
    "replicate_level_cp_normalized_variable_selected.csv.gz"
)
BBBC047_GE = "/path/to/data/AIDD/CDRP-BBBC047-Bray/L1000/replicate_level_l1k.csv.gz"


def jsonable(value):
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    return str(value)


def clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def norm_dose(value) -> str:
    s = clean(value)
    if not s:
        return ""
    try:
        x = float(s)
        if math.isnan(x):
            return ""
        return f"{x:.12g}"
    except Exception:
        return s


STANDARD_DOSES = ("0.04", "0.12", "0.37", "1.11", "3.33", "10")


def standard_dose(value) -> str:
    """Return the six LINCS Pilot 1 dose labels, or an empty string.

    The CP table has a rounded ``Metadata_dose_recode`` field and the GE table
    stores the same labels inside ``pert_id_dose``.  Raw molar concentrations
    differ slightly between the two tables, so the recode is the auditable
    cross-modal key.
    """
    s = norm_dose(value)
    if not s:
        return ""
    try:
        x = float(s)
    except Exception:
        return ""
    best = min(STANDARD_DOSES, key=lambda d: abs(float(d) - x))
    return best if abs(float(best) - x) < 0.03 else ""


def standard_dose_from_pert_id_dose(value) -> str:
    s = clean(value)
    if "_" not in s:
        return ""
    return standard_dose(s.rsplit("_", 1)[-1])


def base_id(value) -> str:
    s = clean(value)
    if not s:
        return ""
    # GE pert_id_dose has the form BRD..._<dose>; the CP metadata contains the
    # base broad ID.  Do not split a base ID itself, only use the first token
    # when the prefix is an unmistakable BRD identifier.
    if s.startswith("BRD-") and "_" in s:
        return s.split("_", 1)[0]
    return s


def sha256(path: str, block: int = 1 << 20) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_header(path: str) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def value_counts(chunks: Iterable[pd.DataFrame], col: str) -> Counter:
    out: Counter = Counter()
    for chunk in chunks:
        if col in chunk:
            out.update(clean(x) for x in chunk[col].tolist())
    return Counter({k: v for k, v in out.items() if k})


def summarize_counter(counter: Counter, limit: int = 100) -> dict:
    return {str(k): int(v) for k, v in counter.most_common(limit)}


def write_csv(rows: list[Mapping], path: Path):
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def audit_lincs_cp(path: str, out: Path, chunksize: int = 50_000):
    cols = read_header(path)
    wanted = [
        "Metadata_plate_map_name",
        "Metadata_broad_sample",
        "Metadata_mg_per_ml",
        "Metadata_mmoles_per_liter",
        "Metadata_solvent",
        "Metadata_pert_id",
        "Metadata_pert_mfc_id",
        "Metadata_pert_well",
        "Metadata_pert_id_vendor",
        "Metadata_cell_id",
        "Metadata_broad_sample_type",
        "Metadata_pert_vehicle",
        "Metadata_pert_type",
        "Metadata_broad_id",
        "Metadata_InChIKey14",
        "Metadata_moa",
        "Metadata_target",
        "Metadata_broad_date",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Assay_Plate_Barcode",
        "Metadata_Plate_Map_Name",
        "Metadata_Batch_Number",
        "Metadata_Batch_Date",
        "Metadata_dose_recode",
        "Metadata_pert_id_dose",
    ]
    use = [c for c in wanted if c in cols]
    total = 0
    type_counts: Counter = Counter()
    cell_counts: Counter = Counter()
    plate_rows: Counter = Counter()
    plate_trt: Counter = Counter()
    plate_ctl: Counter = Counter()
    dose_counts: Counter = Counter()
    dose_recode_counts: Counter = Counter()
    batch_counts: Counter = Counter()
    date_counts: Counter = Counter()
    compound_to_plates: defaultdict[str, set[str]] = defaultdict(set)
    compound_dose_to_plates: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    compound_to_ikey: defaultdict[str, set[str]] = defaultdict(set)
    broad_ids: set[str] = set()
    treatment_rows: list[dict] = []
    chunk_iter = pd.read_csv(path, usecols=use, chunksize=chunksize, low_memory=False)
    for chunk in chunk_iter:
        total += len(chunk)
        if "Metadata_broad_sample_type" in chunk:
            type_counts.update(clean(x) for x in chunk["Metadata_broad_sample_type"])
        if "Metadata_cell_id" in chunk:
            cell_counts.update(clean(x) for x in chunk["Metadata_cell_id"])
        for _, row in chunk.iterrows():
            typ = clean(row.get("Metadata_broad_sample_type"))
            plate = clean(row.get("Metadata_Plate"))
            well = clean(row.get("Metadata_Well"))
            cid = base_id(row.get("Metadata_pert_id")) or base_id(row.get("Metadata_broad_id"))
            broad = base_id(row.get("Metadata_broad_id")) or cid
            dose = norm_dose(row.get("Metadata_mmoles_per_liter"))
            dose_recode = norm_dose(row.get("Metadata_dose_recode"))
            dose_key = dose_recode or standard_dose(dose)
            ikey = clean(row.get("Metadata_InChIKey14"))
            if plate:
                plate_rows[plate] += 1
                if typ == "trt":
                    plate_trt[plate] += 1
                elif typ == "control":
                    plate_ctl[plate] += 1
            if dose_key:
                dose_counts[dose_key] += 1
            if dose_recode:
                dose_recode_counts[dose_recode] += 1
            if clean(row.get("Metadata_Batch_Number")):
                batch_counts[clean(row.get("Metadata_Batch_Number"))] += 1
            if clean(row.get("Metadata_Batch_Date")):
                date_counts[clean(row.get("Metadata_Batch_Date"))] += 1
            if typ == "trt" and cid:
                broad_ids.add(cid)
                if plate:
                    compound_to_plates[cid].add(plate)
                    compound_dose_to_plates[(cid, dose_key)].add(plate)
                if ikey:
                    compound_to_ikey[cid].add(ikey)
                treatment_rows.append(
                    {
                        "compound_id": cid,
                        "broad_id": broad,
                        "inchikey14": ikey,
                        "dose_mM": dose,
                        "dose_recode": dose_recode,
                        "dose_key": dose_key,
                        "plate": plate,
                        "well": well,
                        "plate_map_name": clean(row.get("Metadata_plate_map_name")),
                        "batch_number": clean(row.get("Metadata_Batch_Number")),
                        "batch_date": clean(row.get("Metadata_Batch_Date")),
                        "cell_id": clean(row.get("Metadata_cell_id")),
                        "pert_id_dose": clean(row.get("Metadata_pert_id_dose")),
                    }
                )
    repeat_dist = Counter(len(v) for v in compound_to_plates.values())
    compound_dose_repeat_dist = Counter(len(v) for v in compound_dose_to_plates.values())
    plate_rows_out = []
    for p in sorted(plate_rows):
        plate_rows_out.append(
            {
                "plate": p,
                "rows": plate_rows[p],
                "treatment_rows": plate_trt[p],
                "control_rows": plate_ctl[p],
            }
        )
    dose_rows = [{"dose_mM": k, "rows": v} for k, v in sorted(dose_counts.items())]
    repeat_rows = [
        {"repeat_plates": int(k), "compound_count": int(v)}
        for k, v in sorted(repeat_dist.items())
    ]
    repeat_dose_rows = [
        {"repeat_plates": int(k), "compound_dose_count": int(v)}
        for k, v in sorted(compound_dose_repeat_dist.items())
    ]
    write_csv(treatment_rows, out / "cp_records.csv")
    write_csv(plate_rows_out, out / "plate_statistics.csv")
    write_csv(dose_rows, out / "dose_statistics.csv")
    write_csv(repeat_rows, out / "repeat_statistics.csv")
    write_csv(repeat_dose_rows, out / "compound_dose_repeat_statistics.csv")
    compound_rows = []
    for cid in sorted(broad_ids):
        compound_rows.append(
            {
                "compound_id": cid,
                "cp_plates": len(compound_to_plates[cid]),
                "inchikey14_count": len(compound_to_ikey[cid]),
                "inchikey14": ";".join(sorted(compound_to_ikey[cid])),
            }
        )
    write_csv(compound_rows, out / "compound_mapping.csv")
    return {
        "path": path,
        "file_size_bytes": os.path.getsize(path),
        "sha256": sha256(path),
        "n_columns": len(cols),
        "n_metadata_columns_used": len(use),
        "n_rows": total,
        "treatment_rows": int(sum(plate_trt.values())),
        "control_rows": int(sum(plate_ctl.values())),
        "sample_type_counts": summarize_counter(type_counts),
        "cell_id_counts": summarize_counter(cell_counts),
        "unique_plates": len(plate_rows),
        "unique_compounds": len(broad_ids),
        "unique_compound_plate_pairs": int(sum(len(v) for v in compound_to_plates.values())),
        "unique_compound_dose_plate_pairs": int(
            sum(len(v) for v in compound_dose_to_plates.values())
        ),
        "compound_repeat_plate_distribution": summarize_counter(repeat_dist, 1000),
        "compound_dose_repeat_plate_distribution": summarize_counter(
            compound_dose_repeat_dist, 1000
        ),
        "dose_counts": summarize_counter(dose_counts, 1000),
        "dose_recode_counts": summarize_counter(dose_recode_counts, 1000),
        "batch_counts": summarize_counter(batch_counts, 1000),
        "batch_date_counts": summarize_counter(date_counts, 1000),
        "missing_inchikey14_treatment_rows": int(
            sum(1 for r in treatment_rows if not r["inchikey14"])
        ),
        "metadata_columns": use,
    }


def audit_lincs_ge(path: str, out: Path, chunksize: int = 50_000):
    cols = read_header(path)
    wanted = [
        "cid",
        "cell_id",
        "det_plate",
        "det_well",
        "rna_plate",
        "rna_well",
        "pert_dose",
        "pert_id",
        "pert_idose",
        "pert_iname_x",
        "pert_iname_y",
        "pert_time",
        "pert_type_x",
        "pert_type_y",
        "pert_vehicle",
        "x_smiles",
        "pert_plate",
        "batch",
        "batch_date",
        "brew_prefix",
        "group_id",
        "pert_id_dose",
        "pert_type",
        "control_type",
    ]
    use = [c for c in wanted if c in cols]
    total = 0
    type_counts: Counter = Counter()
    cell_counts: Counter = Counter()
    plate_counts: Counter = Counter()
    batch_counts: Counter = Counter()
    date_counts: Counter = Counter()
    dose_counts: Counter = Counter()
    dose_recode_counts: Counter = Counter()
    compound_to_plates: defaultdict[str, set[str]] = defaultdict(set)
    compound_dose_to_plates: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    treatment_rows: list[dict] = []
    for chunk in pd.read_csv(path, usecols=use, chunksize=chunksize, low_memory=False):
        total += len(chunk)
        for _, row in chunk.iterrows():
            typ = clean(row.get("pert_type")) or clean(row.get("pert_type_y"))
            cid_raw = clean(row.get("pert_id"))
            cid = base_id(cid_raw)
            pidose = clean(row.get("pert_id_dose"))
            is_treatment = bool(cid and cid != "DMSO" and "control" not in typ.lower())
            # Keep the notebook's explicit BRD filter as a separate audit fact.
            is_brd = pidose.startswith("BRD-") or cid.startswith("BRD-")
            plate = clean(row.get("det_plate"))
            well = clean(row.get("det_well"))
            dose = norm_dose(row.get("pert_dose"))
            dose_key = standard_dose_from_pert_id_dose(pidose) or standard_dose(dose)
            if typ:
                type_counts[typ] += 1
            if clean(row.get("cell_id")):
                cell_counts[clean(row.get("cell_id"))] += 1
            if plate:
                plate_counts[plate] += 1
            if clean(row.get("batch")):
                batch_counts[clean(row.get("batch"))] += 1
            if clean(row.get("batch_date")):
                date_counts[clean(row.get("batch_date"))] += 1
            if dose:
                dose_counts[dose] += 1
            if dose_key:
                dose_recode_counts[dose_key] += 1
            if is_treatment and is_brd:
                compound_to_plates[cid].add(plate)
                compound_dose_to_plates[(cid, dose_key)].add(plate)
                treatment_rows.append(
                    {
                        "compound_id": cid,
                        "pert_id": cid_raw,
                        "inchikey_or_smiles": clean(row.get("x_smiles")),
                        "dose": dose,
                        "dose_recode": dose_key,
                        "det_plate": plate,
                        "det_well": well,
                        "rna_plate": clean(row.get("rna_plate")),
                        "rna_well": clean(row.get("rna_well")),
                        "pert_id_dose": pidose,
                        "pert_type": typ,
                        "cell_id": clean(row.get("cell_id")),
                        "batch": clean(row.get("batch")),
                        "batch_date": clean(row.get("batch_date")),
                    }
                )
    repeat_dist = Counter(len(v) for v in compound_to_plates)
    repeat_dose_dist = Counter(len(v) for v in compound_dose_to_plates)
    write_csv(treatment_rows, out / "ge_records.csv")
    write_csv(
        [{"det_plate": k, "rows": v} for k, v in sorted(plate_counts.items())],
        out / "ge_plate_statistics.csv",
    )
    write_csv(
        [{"dose": k, "rows": v} for k, v in sorted(dose_counts.items())],
        out / "ge_dose_statistics.csv",
    )
    write_csv(
        [{"repeat_plates": int(k), "compound_count": int(v)} for k, v in sorted(repeat_dist.items())],
        out / "ge_repeat_statistics.csv",
    )
    write_csv(
        [
            {"repeat_plates": int(k), "compound_dose_count": int(v)}
            for k, v in sorted(repeat_dose_dist.items())
        ],
        out / "ge_compound_dose_repeat_statistics.csv",
    )
    return {
        "path": path,
        "file_size_bytes": os.path.getsize(path),
        "sha256": sha256(path),
        "n_columns": len(cols),
        "n_metadata_columns_used": len(use),
        "n_rows": total,
        "metadata_type_counts": summarize_counter(type_counts, 1000),
        "cell_id_counts": summarize_counter(cell_counts, 1000),
        "unique_det_plates": len(plate_counts),
        "unique_brd_compounds": len(compound_to_plates),
        "unique_brd_compound_dose_pairs": len(compound_dose_to_plates),
        "brd_treatment_rows": len(treatment_rows),
        "compound_repeat_plate_distribution": summarize_counter(repeat_dist, 1000),
        "compound_dose_repeat_plate_distribution": summarize_counter(repeat_dose_dist, 1000),
        "dose_counts": summarize_counter(dose_counts, 1000),
        "dose_recode_counts": summarize_counter(dose_recode_counts, 1000),
        "batch_counts": summarize_counter(batch_counts, 1000),
        "batch_date_counts": summarize_counter(date_counts, 1000),
        "missing_smiles_or_x_smiles_brd_rows": int(
            sum(1 for r in treatment_rows if not r["inchikey_or_smiles"])
        ),
        "metadata_columns": use,
    }


def audit_bbbc047_plates(path: str, kind: str, chunksize: int = 100_000):
    cols = read_header(path)
    if kind == "cp":
        wanted = [
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_mmoles_per_liter",
            "Metadata_pert_id",
            "Metadata_broad_sample_type",
            "Metadata_pert_id_dose",
        ]
        use = [c for c in wanted if c in cols]
        plates: set[str] = set()
        compounds: set[str] = set()
        compound_plate: set[tuple[str, str]] = set()
        n = 0
        trt = 0
        for chunk in pd.read_csv(path, usecols=use, chunksize=chunksize, low_memory=False):
            n += len(chunk)
            for _, row in chunk.iterrows():
                typ = clean(row.get("Metadata_broad_sample_type"))
                plate = clean(row.get("Metadata_Plate"))
                cid = base_id(row.get("Metadata_pert_id"))
                if plate:
                    plates.add(plate)
                if typ == "trt" and cid:
                    trt += 1
                    compounds.add(cid)
                    compound_plate.add((cid, plate))
        return {
            "path": path,
            "kind": kind,
            "n_rows": n,
            "treatment_rows": trt,
            "unique_plates": len(plates),
            "unique_compounds": len(compounds),
            "unique_compound_plate_pairs": len(compound_plate),
            "plates": sorted(plates),
            "compounds": sorted(compounds),
        }
    wanted = ["det_plate", "pert_id", "BROAD_CPD_ID", "pert_id_dose", "pert_type", "control_type"]
    use = [c for c in wanted if c in cols]
    plates: set[str] = set()
    compounds: set[str] = set()
    n = 0
    trt = 0
    for chunk in pd.read_csv(path, usecols=use, chunksize=chunksize, low_memory=False):
        n += len(chunk)
        for _, row in chunk.iterrows():
            plate = clean(row.get("det_plate"))
            cid = base_id(row.get("BROAD_CPD_ID")) or base_id(row.get("pert_id"))
            typ = clean(row.get("pert_type")) or clean(row.get("control_type"))
            if plate:
                plates.add(plate)
            if cid and cid != "DMSO" and typ == "trt":
                trt += 1
                compounds.add(cid)
    return {
        "path": path,
        "kind": kind,
        "n_rows": n,
        "treatment_rows": trt,
        "unique_plates": len(plates),
        "unique_compounds": len(compounds),
        "plates": sorted(plates),
        "compounds": sorted(compounds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lincs-cp", default=LINCS_CP)
    ap.add_argument("--lincs-ge", default=LINCS_GE)
    ap.add_argument("--bbbc047-cp", default=BBBC047_CP)
    ap.add_argument("--bbbc047-ge", default=BBBC047_GE)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cp = audit_lincs_cp(args.lincs_cp, out)
    ge = audit_lincs_ge(args.lincs_ge, out)
    bcp = audit_bbbc047_plates(args.bbbc047_cp, "cp")
    bge = audit_bbbc047_plates(args.bbbc047_ge, "ge")
    cp_plates = set(cp.get("plates", []))
    # The LINCS CP/GE plate lists are recovered from the compact records rather
    # than kept in the large summary object.
    cp_rec = pd.read_csv(out / "cp_records.csv")
    ge_rec = pd.read_csv(out / "ge_records.csv")
    lincs_cp_plates = set(cp_rec["plate"].dropna().astype(str)) if len(cp_rec) else set()
    lincs_ge_plates = set(ge_rec["det_plate"].dropna().astype(str)) if len(ge_rec) else set()
    lincs_cp_ids = set(cp_rec["compound_id"].dropna().astype(str)) if len(cp_rec) else set()
    lincs_ge_ids = set(ge_rec["compound_id"].dropna().astype(str)) if len(ge_rec) else set()
    # CP and GE use slightly different raw concentration precision.  The
    # standard six-level dose recode is therefore the only dose key used for
    # the pair audit; the 20 mM plate-wide reference compounds are excluded.
    cp_rec["dose_recode"] = cp_rec["dose_recode"].map(standard_dose)
    ge_rec["dose_recode"] = ge_rec["dose_recode"].map(standard_dose)
    cp_std = cp_rec[cp_rec["dose_recode"].isin(STANDARD_DOSES)].copy()
    ge_std = ge_rec[ge_rec["dose_recode"].isin(STANDARD_DOSES)].copy()
    cp_group = (
        cp_std.groupby(["compound_id", "dose_recode"], dropna=False)
        .agg(cp_rows=("plate", "size"), cp_plates=("plate", "nunique"))
        .reset_index()
    )
    ge_group = (
        ge_std.groupby(["compound_id", "dose_recode"], dropna=False)
        .agg(ge_rows=("det_plate", "size"), ge_plates=("det_plate", "nunique"))
        .reset_index()
    )
    overlap = cp_group.merge(ge_group, on=["compound_id", "dose_recode"], how="outer", indicator=True)
    overlap["cp_rows"] = overlap["cp_rows"].fillna(0).astype(int)
    overlap["cp_plates"] = overlap["cp_plates"].fillna(0).astype(int)
    overlap["ge_rows"] = overlap["ge_rows"].fillna(0).astype(int)
    overlap["ge_plates"] = overlap["ge_plates"].fillna(0).astype(int)
    overlap["match_status"] = overlap["_merge"].map(
        {"both": "matched_compound_and_standard_dose", "left_only": "cp_only", "right_only": "ge_only"}
    )
    write_csv(overlap.drop(columns=["_merge"]).to_dict("records"), out / "cp_ge_overlap.csv")
    bcp_plates = set(bcp["plates"])
    bge_plates = set(bge["plates"])
    bcp_ids = set(bcp["compounds"])
    bge_ids = set(bge["compounds"])
    pair_cp_ge = sorted(lincs_cp_ids & lincs_ge_ids)
    # Because the raw acquisition plate is part of every record identity, a
    # disjoint plate set certifies zero exact raw (compound, plate, well)
    # overlap, regardless of compound overlap.
    independence = {
        "lincs_vs_bbbc047_cp_plate_intersection": sorted(lincs_cp_plates & bcp_plates),
        "lincs_vs_bbbc047_ge_plate_intersection": sorted(lincs_ge_plates & bge_plates),
        "lincs_vs_bbbc047_cp_compound_intersection": sorted(lincs_cp_ids & bcp_ids),
        "lincs_vs_bbbc047_ge_compound_intersection": sorted(lincs_ge_ids & bge_ids),
        "exact_cp_raw_record_overlap_status": (
            "zero_by_disjoint_plate_keys" if not (lincs_cp_plates & bcp_plates) else "requires_record_key_check"
        ),
        "exact_ge_raw_record_overlap_status": (
            "zero_by_disjoint_plate_keys" if not (lincs_ge_plates & bge_plates) else "requires_record_key_check"
        ),
        "lincs_cp_ge_compound_intersection_count": len(pair_cp_ge),
        "standard_cp_ge_pair_count": int((overlap["match_status"] == "matched_compound_and_standard_dose").sum()),
        "standard_cp_ge_compound_count": int(
            overlap.loc[overlap["match_status"] == "matched_compound_and_standard_dose", "compound_id"].nunique()
        ),
        "standard_cp_ge_dose_labels": sorted(
            overlap.loc[overlap["match_status"] == "matched_compound_and_standard_dose", "dose_recode"].astype(str).unique()
        ),
        "standard_cp_rows": int(len(cp_std)),
        "standard_ge_rows": int(len(ge_std)),
    }
    # Save only the identities needed to reproduce the comparisons.
    write_csv(
        [{"dataset": "lincs_cpg0004", "kind": "cp", "plate": x} for x in sorted(lincs_cp_plates)]
        + [{"dataset": "bbbc047", "kind": "cp", "plate": x} for x in sorted(bcp_plates)]
        + [{"dataset": "lincs_cpg0004", "kind": "ge", "plate": x} for x in sorted(lincs_ge_plates)]
        + [{"dataset": "bbbc047", "kind": "ge", "plate": x} for x in sorted(bge_plates)],
        out / "plate_identity_index.csv",
    )
    write_csv(
        [{"dataset": "lincs_cpg0004", "kind": "cp", "compound_id": x} for x in sorted(lincs_cp_ids)]
        + [{"dataset": "bbbc047", "kind": "cp", "compound_id": x} for x in sorted(bcp_ids)]
        + [{"dataset": "lincs_cpg0004", "kind": "ge", "compound_id": x} for x in sorted(lincs_ge_ids)]
        + [{"dataset": "bbbc047", "kind": "ge", "compound_id": x} for x in sorted(bge_ids)],
        out / "compound_identity_index.csv",
    )
    summary = {
        "lincs_cp": cp,
        "lincs_ge": ge,
        "bbbc047_cp": {k: v for k, v in bcp.items() if k not in {"plates", "compounds"}},
        "bbbc047_ge": {k: v for k, v in bge.items() if k not in {"plates", "compounds"}},
        "lincs_cp_plate_count_from_records": len(lincs_cp_plates),
        "lincs_ge_plate_count_from_records": len(lincs_ge_plates),
        "independence": independence,
        "raw_paths": {
            "lincs_cp": args.lincs_cp,
            "lincs_ge": args.lincs_ge,
            "bbbc047_cp": args.bbbc047_cp,
            "bbbc047_ge": args.bbbc047_ge,
        },
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
