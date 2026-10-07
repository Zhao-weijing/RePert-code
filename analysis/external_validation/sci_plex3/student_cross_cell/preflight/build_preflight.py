#!/usr/bin/env python3
"""Metadata-only sci-Plex3 Student cross-cell structure/preflight audit.

This script intentionally never opens group_counts.h5 (or any h5/h5ad file).
It maps the frozen 183-drug/613-unit common universe to structures using the
existing Broad repurposing metadata sidecar, with a PubChem fallback only for
names not represented there.  Ambiguous structure resolution is excluded,
never silently disambiguated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, inchi


ROOT = Path(__file__).resolve().parent
DEFAULT_VALIDATION = ROOT.parent.parent / "sciplex3_validation"
DEFAULT_COMMON = DEFAULT_VALIDATION / "cross_cell_generalization" / "LOCO_POOL_COMMON_DRUG_DOSE_UNITS.csv"
DEFAULT_ELIGIBLE = DEFAULT_VALIDATION / "eligible_conditions_n50.csv"
DEFAULT_SPLIT = DEFAULT_VALIDATION / "cfra_confirmation" / "split_lock.json"
DEFAULT_GROUP_META = DEFAULT_VALIDATION / "group_metadata.csv"
DEFAULT_SOURCE = ROOT / "input_cache" / "merged_repurposing.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def norm_exact(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    value = value.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    return re.sub(r"\s+", " ", value).strip().casefold()


def norm_alias(value: str) -> str:
    # Alias matching is only used when exact matching has no candidate and is
    # retained in the audit.  It removes formatting, not chemical words.
    return re.sub(r"[^\w]+", "", norm_exact(value), flags=re.UNICODE)


def clean_smiles(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    # Broad metadata may carry RDKit CXSMILES annotations after a space.
    if " |" in text:
        text = text.split(" |", 1)[0].strip()
    if "|" in text:
        text = text.split("|", 1)[0].strip()
    return text


def clean_cid(value: Any) -> str:
    """Render PubChem CIDs without pandas' trailing ``.0`` coercion."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def pubchem_queries(drug: str) -> list[str]:
    """Deterministic exact-name candidates for punctuation/alias variants."""
    values: list[str] = [str(drug).strip()]
    # sci-Plex has one visible typo-like question mark in a drug label.  The
    # text before a question mark/parenthesis is its primary name; parenthesis
    # aliases are also independent PubChem identities.
    primary = re.split(r"[?(]", str(drug), maxsplit=1)[0].strip()
    if primary and primary not in values:
        values.append(primary)
    m = re.search(r"\(([^)]*)\)", str(drug))
    if m:
        for alias in re.split(r"[,;/]", m.group(1)):
            alias = alias.strip()
            if alias and alias not in values:
                values.append(alias)
    return values


def canonicalize(smiles: str) -> tuple[str | None, str | None, str | None]:
    if not smiles:
        return None, None, "empty_smiles"
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, "invalid_smiles"
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        ik = inchi.MolToInchiKey(mol) if inchi.INCHI_AVAILABLE else None
        return canonical, ik, None
    except Exception as exc:  # pragma: no cover - defensive for odd vendor strings
        return None, None, f"rdkit_error:{type(exc).__name__}"


def query_pubchem(name: str, timeout: float = 20.0) -> dict[str, Any]:
    """Conservative PubChem exact-name fallback.

    A result is resolved only when all returned CIDs agree on the same
    canonical structure.  Multiple chemically distinct CIDs are ambiguous.
    """
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    cid_url = base + quote(name, safe="") + "/cids/JSON"
    req = Request(cid_url, headers={"User-Agent": "MVCPert-sciPlex3-preflight/1.0"})
    with urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    cids = [int(x) for x in payload.get("IdentifierList", {}).get("CID", [])]
    if not cids:
        return {"status": "no_pubchem_hit", "cids": []}
    # Avoid using an arbitrary first hit when PubChem returns a synonym set.
    prop_url = base + ",".join(str(x) for x in cids[:100]) + "/property/CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
    # The PUG endpoint accepts a list in the path only for /cid/; use the
    # documented comma-separated cid form and retain a bounded request.
    prop_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + ",".join(str(x) for x in cids[:100]) + "/property/CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
    req = Request(prop_url, headers={"User-Agent": "MVCPert-sciPlex3-preflight/1.0"})
    with urlopen(req, timeout=timeout) as response:
        props = json.loads(response.read().decode("utf-8")).get("PropertyTable", {}).get("Properties", [])
    rows = []
    for row in props:
        smi = row.get("IsomericSMILES") or row.get("ConnectivitySMILES") or row.get("CanonicalSMILES")
        can, ik, err = canonicalize(smi or "")
        rows.append({"cid": row.get("CID"), "smiles": smi, "canonical_smiles": can, "inchikey": ik, "error": err})
    structures = {(r["canonical_smiles"], r["inchikey"]) for r in rows if r["canonical_smiles"]}
    if len(structures) == 1:
        chosen = rows[0]
        return {"status": "resolved_pubchem", "cids": cids, **chosen, "candidate_rows": rows}
    return {"status": "ambiguous_pubchem", "cids": cids, "candidate_rows": rows}


def candidate_index(source: pd.DataFrame) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    exact: dict[str, list[int]] = defaultdict(list)
    alias: dict[str, list[int]] = defaultdict(list)
    for idx, row in source.iterrows():
        for field in ("pert_iname", "vendor_name"):
            value = row.get(field)
            if pd.isna(value) or not str(value).strip():
                continue
            exact[norm_exact(str(value))].append(int(idx))
            alias[norm_alias(str(value))].append(int(idx))
    return exact, alias


def contains_candidates(source: pd.DataFrame, drug: str) -> list[int]:
    """Return conservative alias-component matches for names with reordered
    or parenthesized aliases (e.g. ``Cerdulatinib (PRT062070, PRT2070)``).

    A candidate is considered only when a source name of at least six
    alphanumeric characters is a complete substring after punctuation
    normalization.  Chemical ambiguity is still resolved later from the
    resulting canonical structures; this function never picks a first hit.
    """
    query = norm_alias(drug)
    if len(query) < 6:
        return []
    hits: list[int] = []
    for idx, row in source.iterrows():
        for field in ("pert_iname", "vendor_name"):
            value = row.get(field)
            if pd.isna(value) or not str(value).strip():
                continue
            candidate = norm_alias(str(value))
            if len(candidate) < 6:
                continue
            if candidate in query or query in candidate:
                hits.append(int(idx))
                break
    return sorted(set(hits))


def distinct_structures(source: pd.DataFrame, indices: list[int]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    for idx in sorted(set(indices)):
        src = source.iloc[idx]
        raw_smiles = clean_smiles(src.get("smiles"))
        can, ik_rdkit, err = canonicalize(raw_smiles)
        ik_source = "" if pd.isna(src.get("InChIKey")) else str(src.get("InChIKey")).strip()
        ik = ik_rdkit or ik_source
        cid = clean_cid(src.get("pubchem_cid"))
        key = (can or "", ik or "")
        if can:
            keys.add(key)
        rows.append({
            "source_row": int(idx),
            "source_pert_iname": "" if pd.isna(src.get("pert_iname")) else str(src.get("pert_iname")),
            "source_vendor_name": "" if pd.isna(src.get("vendor_name")) else str(src.get("vendor_name")),
            "source_broad_id": "" if pd.isna(src.get("broad_id")) else str(src.get("broad_id")),
            "raw_smiles": raw_smiles,
            "canonical_smiles": can or "",
            "source_inchikey": ik_source,
            "inchikey": ik or "",
            "pubchem_cid": cid,
            "smiles_error": err or "",
        })
    return rows, sorted(keys)


def build_mapping(drugs: list[str], source: pd.DataFrame, do_pubchem: bool, rate_seconds: float) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    exact, alias = candidate_index(source)
    audits: list[dict[str, Any]] = []
    fp_rows: list[np.ndarray] = []
    for i, drug in enumerate(drugs):
        ekey = norm_exact(drug)
        akey = norm_alias(drug)
        indices = sorted(set(exact.get(ekey, [])))
        resolution = "exact"
        if not indices:
            indices = sorted(set(alias.get(akey, []))) if akey else []
            resolution = "alias" if indices else "none"
        if not indices:
            indices = contains_candidates(source, drug)
            resolution = "contains" if indices else "none"
        candidates, structures = distinct_structures(source, indices)
        status = "exclude_no_structure"
        canonical = ""
        ik = ""
        cid = ""
        pubchem_cids: list[int] = []
        pubchem_query_name = ""
        source_name = ""
        source_field = ""
        reason = ""
        if structures and len(structures) == 1:
            valid = [r for r in candidates if r["canonical_smiles"]]
            # Prefer the first deterministic source row; all structures have
            # already been proven identical, so this is not chemical choice.
            chosen = valid[0]
            canonical = chosen["canonical_smiles"]
            ik = chosen["inchikey"]
            cid = chosen["pubchem_cid"]
            source_name = chosen["source_pert_iname"] or chosen["source_vendor_name"]
            source_field = "pert_iname_or_vendor_name"
            if resolution == "exact":
                status = "resolved_merged_repurposing_exact"
            elif resolution == "contains":
                status = "resolved_merged_repurposing_contains"
            else:
                status = "resolved_merged_repurposing_alias"
            if len(candidates) > 1:
                reason = "multiple_rows_same_structure"
        elif structures and len(structures) > 1:
            # An alias-component match can legitimately collect a parent,
            # salt, or a similarly named compound.  Ask PubChem for the full
            # label and deterministic primary/parenthetical aliases before
            # excluding; do not choose among local candidates.
            status = "exclude_ambiguous_structure"
            reason = "multiple_distinct_structures_for_name"
            if do_pubchem:
                for query_name in pubchem_queries(drug):
                    try:
                        result = query_pubchem(query_name)
                    except Exception as exc:
                        result = {"status": "pubchem_error", "error": f"{type(exc).__name__}: {exc}"}
                    if result.get("status") == "resolved_pubchem":
                        canonical = result.get("canonical_smiles") or ""
                        ik = result.get("inchikey") or ""
                        cid = clean_cid(result.get("cid"))
                        pubchem_cids = [int(x) for x in result.get("cids", []) if str(x).isdigit()]
                        pubchem_query_name = query_name
                        source_name = query_name
                        source_field = "PubChem_PUG_exact_name_disambiguation"
                        status = "resolved_pubchem_disambiguated"
                        reason = "local_alias_candidates_conflicted; PubChem_unique_canonical_identity"
                        break
                    if rate_seconds > 0:
                        time.sleep(rate_seconds)
        elif do_pubchem:
            last_result: dict[str, Any] = {}
            for query_name in pubchem_queries(drug):
                try:
                    result = query_pubchem(query_name)
                except Exception as exc:
                    result = {"status": "pubchem_error", "error": f"{type(exc).__name__}: {exc}"}
                last_result = result
                if result.get("status") == "resolved_pubchem":
                    canonical = result.get("canonical_smiles") or ""
                    ik = result.get("inchikey") or ""
                    cid = clean_cid(result.get("cid"))
                    pubchem_cids = [int(x) for x in result.get("cids", []) if str(x).isdigit()]
                    pubchem_query_name = query_name
                    source_name = query_name
                    source_field = "PubChem_PUG_exact_name"
                    status = "resolved_pubchem"
                    reason = "merged_repurposing_no_candidate"
                    break
                if rate_seconds > 0:
                    time.sleep(rate_seconds)
            if not canonical:
                status = "exclude_" + str(last_result.get("status", "pubchem_error"))
                reason = json.dumps({"cids": last_result.get("cids", []), "candidate_rows": last_result.get("candidate_rows", []), "error": last_result.get("error", "")}, ensure_ascii=False, separators=(",", ":"))[:4000]
        if canonical:
            try:
                mol = Chem.MolFromSmiles(canonical)
                fp = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=False), dtype=np.uint8)
                if fp.shape != (2048,):
                    raise ValueError(f"unexpected fingerprint shape {fp.shape}")
            except Exception as exc:
                fp = np.zeros(2048, dtype=np.uint8)
                status = "exclude_ecfp_error"
                reason = f"{type(exc).__name__}: {exc}"
        else:
            fp = np.zeros(2048, dtype=np.uint8)
        fp_rows.append(fp)
        audits.append({
            "drug": drug,
            "drug_norm_exact": ekey,
            "drug_norm_alias": akey,
            "mapping_source": "PubChem PUG" if status.startswith("resolved_pubchem") else ("merged_repurposing.csv" if indices else ""),
            "name_resolution": resolution if not status.startswith("resolved_pubchem") else ("pubchem_exact_name_disambiguation" if status == "resolved_pubchem_disambiguated" else "pubchem_exact_name"),
            "source_field": source_field,
            "source_name": source_name,
            "pubchem_query_name": pubchem_query_name,
            "n_candidate_rows": len(candidates),
            "n_unique_structures": len(structures),
            "candidate_source_names": " || ".join(sorted({r["source_pert_iname"] or r["source_vendor_name"] for r in candidates})),
            "canonical_smiles": canonical,
            "inchikey": ik,
            "pubchem_cid": cid,
            "pubchem_cids": ";".join(str(x) for x in pubchem_cids),
            "mapping_status": status,
            "mapping_reason": reason,
        })
    audit_df = pd.DataFrame(audits)
    fp_array = np.stack(fp_rows, axis=0)
    fp_keys = {row.tobytes() for row in fp_array}
    return audit_df, fp_array, {
        "n_drugs": len(drugs),
        "n_resolved": int(sum(str(x).startswith("resolved_") for x in audit_df["mapping_status"])),
        "n_excluded": int(sum(str(x).startswith("exclude_") for x in audit_df["mapping_status"])),
        "n_unique_canonical_smiles": int(audit_df.loc[audit_df["canonical_smiles"].astype(str) != "", "canonical_smiles"].nunique()),
        "n_unique_inchikey": int(audit_df.loc[audit_df["inchikey"].astype(str) != "", "inchikey"].nunique()),
        "n_unique_ecfp4": int(len(fp_keys)),
        "ecfp4_collision_note": "Morgan radius-2, 2048-bit, no chirality; expected collisions include stereoisomer-insensitive ECFP and identical parent structures.",
    }


def read_drug_universe(common_path: Path) -> tuple[pd.DataFrame, list[str]]:
    common = pd.read_csv(common_path)
    required = {"drug", "dose_nM", "in_confirmation"}
    missing = required - set(common.columns)
    if missing:
        raise ValueError(f"common units missing columns {sorted(missing)}")
    common["drug"] = common["drug"].astype(str)
    drugs = sorted(common["drug"].drop_duplicates().tolist())
    if len(drugs) != 183 or len(common) != 613:
        raise ValueError(f"frozen common universe changed: {len(drugs)} drugs / {len(common)} units, expected 183 / 613")
    return common, drugs


def split_membership(split_path: Path, drugs: list[str]) -> dict[str, str]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    dset = set(drugs)
    membership: dict[str, str] = {}
    for name in ("base", "calibration", "confirmation"):
        for drug in split.get(name, []):
            if drug in dset:
                if drug in membership:
                    raise ValueError(f"drug appears in multiple split categories: {drug}")
                membership[drug] = name
    missing = sorted(dset - set(membership))
    if missing:
        raise ValueError(f"common drugs absent from split lock: {missing}")
    return membership


def foreign_feasibility(common: pd.DataFrame, eligible: pd.DataFrame, membership: dict[str, str]) -> dict[str, Any]:
    eligible = eligible.copy()
    eligible["drug"] = eligible["drug"].astype(str)
    eligible["cell_line"] = eligible["cell_line"].astype(str)
    eligible["dose_nM"] = eligible["dose_nM"].astype(float)
    eligible["replicate"] = eligible["replicate"].astype(str)
    # Only the frozen common base+calibration donor universe is eligible.
    donor_drugs = {d for d, s in membership.items() if s in {"base", "calibration"}}
    common_key = {(str(r.drug), float(r.dose_nM)) for r in common.itertuples()}
    eligible = eligible[eligible["drug"].isin(donor_drugs)]
    # n>=50 is already frozen in eligible_conditions_n50; keep a defensive
    # condition check if the column is present.
    if "n_cells" in eligible.columns:
        eligible = eligible[eligible["n_cells"].astype(float) >= 50]
    grouped = eligible.groupby(["cell_line", "drug", "dose_nM"], sort=True)["replicate"].agg(lambda x: set(x)).to_dict()
    result: dict[str, Any] = {}
    for target in ("A549", "K562", "MCF7"):
        conf = common[(common["in_confirmation"].astype(str).str.lower() == "true")]
        rows = []
        for r in conf.itertuples():
            key = (target, str(r.drug), float(r.dose_nM))
            # A donor is usable only if it has both real repeats for this
            # target line and dose.  The target drug is removed explicitly.
            counts = [drug for (line, drug, dose), reps in grouped.items() if line == target and dose == float(r.dose_nM) and len(reps) == 2 and drug != str(r.drug)]
            rows.append({"drug": str(r.drug), "dose_nM": float(r.dose_nM), "n_foreign_donors": len(set(counts)), "feasible_20": len(set(counts)) >= 20})
        counts = [x["n_foreign_donors"] for x in rows]
        result[target] = {
            "n_confirmation_units": len(rows),
            "n_units_with_20_donors": int(sum(x["feasible_20"] for x in rows)),
            "n_units_below_20_donors": int(sum(not x["feasible_20"] for x in rows)),
            "min_donors": int(min(counts)) if counts else 0,
            "median_donors": float(np.median(counts)) if counts else 0.0,
            "max_donors": int(max(counts)) if counts else 0,
            "all_units_feasible": bool(rows and all(x["feasible_20"] for x in rows)),
            "short_units": [x for x in rows if not x["feasible_20"]],
        }
    return result


def make_folds(common: pd.DataFrame, membership: dict[str, str], eligible: pd.DataFrame) -> dict[str, Any]:
    folds = {
        "K562+MCF7_to_A549": {"sources": ["K562", "MCF7"], "target": "A549"},
        "A549+MCF7_to_K562": {"sources": ["A549", "MCF7"], "target": "K562"},
        "A549+K562_to_MCF7": {"sources": ["A549", "K562"], "target": "MCF7"},
    }
    eligible = eligible.copy()
    eligible["drug"] = eligible["drug"].astype(str)
    eligible["cell_line"] = eligible["cell_line"].astype(str)
    eligible["dose_nM"] = eligible["dose_nM"].astype(float)
    common = common.copy()
    common["split"] = common["drug"].map(membership)
    common_by_split = common.groupby("split", sort=False)
    split_counts = {
        s: {"n_drugs": int(common.loc[common["split"] == s, "drug"].nunique()), "n_units": int((common["split"] == s).sum())}
        for s in ("base", "calibration", "confirmation")
    }
    foreign = foreign_feasibility(common, eligible, membership)
    out: dict[str, Any] = {"frozen_common_counts": {"n_drugs": int(common["drug"].nunique()), "n_units": int(len(common))}, "split_counts": split_counts, "folds": {}}
    # A single deterministic support->held rotation is the planned Student
    # row unit; both legal rotations are reported for reproducibility checks.
    n_base_units = split_counts["base"]["n_units"]
    n_cal_units = split_counts["calibration"]["n_units"]
    n_conf_units = split_counts["confirmation"]["n_units"]
    for fold_name, cfg in folds.items():
        n_source = len(cfg["sources"])
        out["folds"][fold_name] = {
            **cfg,
            "base": {"n_drugs": split_counts["base"]["n_drugs"], "n_units": n_base_units, "student_rows_one_fixed_rotation": int(n_base_units * n_source), "student_rows_both_ordered_rotations": int(n_base_units * n_source * 2)},
            "calibration": {"n_drugs": split_counts["calibration"]["n_drugs"], "n_units": n_cal_units, "student_rows_one_fixed_rotation": int(n_cal_units * n_source), "student_rows_both_ordered_rotations": int(n_cal_units * n_source * 2)},
            "confirmation": {"n_drugs": split_counts["confirmation"]["n_drugs"], "n_units": n_conf_units, "student_rows_one_fixed_rotation": n_conf_units, "student_rows_both_ordered_rotations": int(n_conf_units * 2), "treated_response_not_used_for_fit": True},
            "foreign_donor_feasibility": foreign[cfg["target"]],
            "student_total_rows_one_fixed_rotation": int((n_base_units + n_cal_units) * n_source + n_conf_units),
            "student_total_rows_both_ordered_rotations": int((n_base_units + n_cal_units) * n_source * 2 + n_conf_units * 2),
        }
    out["foreign_donor_feasibility"] = foreign
    return out


def write_npz(path: Path, audit: pd.DataFrame, fp: np.ndarray) -> None:
    np.savez_compressed(
        path,
        drug=np.asarray(audit["drug"].astype(str).tolist(), dtype="U"),
        canonical_smiles=np.asarray(audit["canonical_smiles"].astype(str).tolist(), dtype="U"),
        inchikey=np.asarray(audit["inchikey"].astype(str).tolist(), dtype="U"),
        pubchem_cid=np.asarray(audit["pubchem_cid"].astype(str).tolist(), dtype="U"),
        mapping_status=np.asarray(audit["mapping_status"].astype(str).tolist(), dtype="U"),
        ecfp4=np.asarray(fp, dtype=np.uint8),
        ecfp4_radius=np.int64(2),
        ecfp4_nbits=np.int64(2048),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common", type=Path, default=DEFAULT_COMMON)
    parser.add_argument("--eligible", type=Path, default=DEFAULT_ELIGIBLE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--group-metadata", type=Path, default=DEFAULT_GROUP_META)
    parser.add_argument("--structure-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--pubchem-fallback", action="store_true")
    parser.add_argument("--rate-seconds", type=float, default=0.25)
    args = parser.parse_args()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    forbidden = [args.common, args.eligible, args.split, args.group_metadata, args.structure_source]
    if any(p.suffix.lower() in {".h5", ".h5ad"} for p in forbidden):
        raise RuntimeError("metadata preflight refuses an H5/H5AD input")
    common, drugs = read_drug_universe(args.common)
    membership = split_membership(args.split, drugs)
    source = pd.read_csv(args.structure_source, low_memory=False)
    required_source = {"pert_iname", "vendor_name", "smiles", "InChIKey", "pubchem_cid"}
    if required_source - set(source.columns):
        raise ValueError(f"structure source missing {sorted(required_source - set(source.columns))}")
    audit, fp, mapping_summary = build_mapping(drugs, source, args.pubchem_fallback, args.rate_seconds)
    audit["split"] = audit["drug"].map(membership)
    audit["source_sha256"] = sha256(args.structure_source)
    audit["source_md5"] = md5(args.structure_source)
    audit_path = outdir / "DRUG_STRUCTURE_MAPPING_AUDIT.csv"
    audit.to_csv(audit_path, index=False, quoting=csv.QUOTE_MINIMAL)
    npz_path = outdir / "DRUG_ECFP4_MAPPING.npz"
    write_npz(npz_path, audit, fp)
    feasible = make_folds(common, membership, pd.read_csv(args.eligible, low_memory=False))
    resolved_drugs = set(audit.loc[audit["mapping_status"].astype(str).str.startswith("resolved_"), "drug"].astype(str))
    common_for_coverage = common.copy()
    common_for_coverage["split"] = common_for_coverage["drug"].map(membership)
    for fold_name, fold in feasible["folds"].items():
        fold["structure_mapping_coverage"] = {}
        for split_name in ("base", "calibration", "confirmation"):
            subset = common_for_coverage[common_for_coverage["split"] == split_name]
            available = subset[subset["drug"].isin(resolved_drugs)]
            fold["structure_mapping_coverage"][split_name] = {
                "n_available_drugs": int(available["drug"].nunique()),
                "n_available_units": int(len(available)),
                "n_excluded_drugs": int(subset["drug"].nunique() - available["drug"].nunique()),
                "n_excluded_units": int(len(subset) - len(available)),
            }
    source_rows = {
        "path": str(args.structure_source.resolve()),
        "sha256": sha256(args.structure_source),
        "md5": md5(args.structure_source),
        "n_rows": int(len(source)),
        "source_description": "existing project Broad repurposing metadata sidecar, with conservative exact/alias/contains name matching and PubChem PUG exact-name fallback/disambiguation enabled",
    }
    input_files = {}
    for label, path in (("common_units", args.common), ("eligible_conditions", args.eligible), ("split_lock", args.split), ("group_metadata", args.group_metadata), ("structure_source", args.structure_source), ("preflight_script", Path(__file__))):
        input_files[label] = {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}
    audit_json = {
        "dataset": "sci-Plex3 Zenodo 7041849",
        "analysis": "Student cross-cell structure and data-feasibility preflight",
        "status": "PASS" if mapping_summary["n_excluded"] == 0 else "PASS_WITH_FROZEN_EXCLUSIONS",
        "expression_access": {"group_counts_h5_opened": False, "h5ad_opened": False, "treated_or_target_expression_read": False, "guard": "metadata-only; no H5/H5AD path passed to any reader"},
        "frozen_universe": {"n_drugs": 183, "n_units": 613, "unit_list_sha256": sha256(args.common)},
        "mapping": {**mapping_summary, "rdkit_version": rdBase.rdkitVersion, "fingerprint": {"family": "Morgan/ECFP4", "radius": 2, "n_bits": 2048, "use_chirality": False, "use_bond_types": True}, "source": source_rows, "excluded_drugs": audit.loc[audit["mapping_status"].astype(str).str.startswith("exclude_"), "drug"].tolist()},
        "split_lock": {"seed": json.loads(args.split.read_text(encoding="utf-8")).get("seed"), "common_drug_counts": {s: int(sum(v == s for v in membership.values())) for s in ("base", "calibration", "confirmation")}, "no_overlap": True},
        "feasibility": feasible,
        "input_files": input_files,
        "outputs": {"mapping_npz": str(npz_path.resolve()), "mapping_audit_csv": str(audit_path.resolve())},
        "protocol_notes": [
            "The common 183-drug/613-unit universe is authoritative; split-lock names outside it are not included.",
            "A single fixed support->held rotation is the planned Student row unit; both legal ordered rotations are listed for audit only.",
            "Target treated responses are confirmation-only and are never used for Student fit, normalization, structure mapping, or early stopping.",
            "Foreign donors are restricted to target-line same-dose base+calibration drugs with both rep1 and rep2 n>=50, target drug excluded.",
        ],
    }
    json_path = outdir / "STUDENT_STRUCTURE_PREFLIGHT.json"
    json_path.write_text(json.dumps(audit_json, ensure_ascii=False, indent=2, default=json_safe) + "\n", encoding="utf-8")
    # Hash manifest intentionally excludes itself to avoid a self-referential hash.
    manifest_path = outdir / "HASH_MANIFEST.sha256"
    rows = []
    for path in sorted([audit_path, npz_path, json_path, args.common, args.eligible, args.split, args.group_metadata, args.structure_source, Path(__file__)]):
        rows.append(f"{sha256(path)}  {path.resolve()}")
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(outdir), "mapping": mapping_summary, "json": str(json_path), "npz": str(npz_path), "audit_csv": str(audit_path), "hash_manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
