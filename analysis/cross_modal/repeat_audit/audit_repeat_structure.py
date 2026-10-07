#!/usr/bin/env python3
"""Audit independent compound/dose/plate repeat structure for MVCPert.

The audit is deliberately data-only.  A repeat is a distinct plate within a
single (compound, dose) key; multiple wells on one plate are never counted as
independent repeats.  The script accepts BBBC047 split NPZ files and the
cpg0004 CP split plus its full GE plate-row artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SELECTIONS = ("train", "valid", "test")
BUDGETS = (1, 2, 3)


def decode_array(values: np.ndarray) -> list[str]:
    output: list[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            output.append(value.decode("utf-8"))
        else:
            output.append(str(value))
    return output


def choose_field(loaded: Any, names: tuple[str, ...]) -> str:
    for name in names:
        if name in loaded.files:
            return name
    raise KeyError(f"None of {names} is present; fields={loaded.files}")


def load_rows(path: Path, split: str, allowed_compounds: set[str] | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        compound_field = choose_field(loaded, ("smiles", "compound", "compound_id"))
        plate_field = choose_field(loaded, ("plate", "det_plate"))
        dose_field = choose_field(loaded, ("dose_standard", "dose"))
        well_field = next((x for x in ("well", "det_well") if x in loaded.files), None)
        batch_field = next((x for x in ("batch",) if x in loaded.files), None)
        feature_field = next((x for x in ("delta", "ge_effect") if x in loaded.files), None)
        compounds = decode_array(loaded[compound_field])
        plates = decode_array(loaded[plate_field])
        doses = decode_array(loaded[dose_field])
        wells = decode_array(loaded[well_field]) if well_field else [""] * len(compounds)
        batches = decode_array(loaded[batch_field]) if batch_field else [""] * len(compounds)
        feature_shape = tuple(loaded[feature_field].shape) if feature_field else ()
    if not (len(compounds) == len(plates) == len(doses) == len(wells) == len(batches)):
        raise ValueError(f"Length mismatch in {path}")
    if allowed_compounds is not None:
        keep = [i for i, compound in enumerate(compounds) if compound in allowed_compounds]
        compounds = [compounds[i] for i in keep]
        plates = [plates[i] for i in keep]
        doses = [doses[i] for i in keep]
        wells = [wells[i] for i in keep]
        batches = [batches[i] for i in keep]
    keys = list(zip(compounds, doses, plates))
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate (compound,dose,plate) rows in {path}")
    compound_dose: dict[tuple[str, str], set[str]] = defaultdict(set)
    for compound, dose, plate in keys:
        if not compound or not dose or not plate:
            raise ValueError(f"Empty compound/dose/plate in {path}")
        compound_dose[(compound, dose)].add(plate)
    return {
        "dataset": "",
        "modality": "",
        "split": split,
        "source": str(path),
        "rows": len(compounds),
        "features": feature_shape[1] if len(feature_shape) == 2 else None,
        "compounds": set(compounds),
        "plates": set(plates),
        "compound_dose": dict(compound_dose),
        "well_available": any(bool(x) for x in wells),
        "batch_available": any(bool(x) for x in batches),
        "site_available": False,
    }


def read_lock(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, set[str]] = {}
    for split in SELECTIONS:
        values = payload.get(f"{split}_compounds", payload.get(f"{split}_smiles"))
        if values is None:
            raise KeyError(f"Split lock has no compounds for {split}: {path}")
        result[split] = {str(x) for x in values}
    if any(result[a] & result[b] for a in SELECTIONS for b in SELECTIONS if a < b):
        raise ValueError(f"Compound overlap in split lock {path}")
    return result


def audit_pair(
    dataset: str,
    split: str,
    modality: str,
    rows: dict[str, Any],
    opposite: dict[str, Any],
    dose_definition: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    eligible_compounds: dict[int, set[str]] = {budget: set() for budget in BUDGETS}
    eligible_units: dict[int, int] = {budget: 0 for budget in BUDGETS}
    for (compound, _dose), plates in rows["compound_dose"].items():
        for budget in BUDGETS:
            if len(plates) >= budget + 1:
                eligible_units[budget] += 1
                eligible_compounds[budget].add(compound)
    for budget in BUDGETS:
        compounds = eligible_compounds[budget]
        opposite_compounds = rows["compounds"] & opposite["compounds"]
        records.append(
            {
                "dataset": dataset,
                "modality": modality,
                "split": split,
                "budget_support_repeats": budget,
                "required_total_plates": budget + 1,
                "source": rows["source"],
                "n_rows": rows["rows"],
                "feature_dim": rows["features"],
                "n_compounds": len(rows["compounds"]),
                "n_plates": len(rows["plates"]),
                "n_compound_dose_units": len(rows["compound_dose"]),
                "eligible_compound_dose_units": eligible_units[budget],
                "eligible_compounds": len(compounds),
                "eligible_compounds_with_opposite_modality": len(compounds & opposite_compounds),
                "opposite_modality_compounds": len(opposite_compounds),
                "plate_independence": "PASS_distinct_plates_by_definition",
                "well_metadata": "available" if rows["well_available"] else "not_in_artifact",
                "batch_metadata": "available" if rows["batch_available"] else "not_in_artifact",
                "site_metadata": "available" if rows["site_available"] else "not_available",
                "dose_definition": dose_definition,
                "threshold_class": (
                    "primary_ge_1000" if len(compounds) >= 1000
                    else "supportive_500_999" if len(compounds) >= 500
                    else "blocked_under_500"
                ),
            }
        )
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No audit rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_markdown(rows: list[dict[str, Any]], sources: dict[str, Any]) -> str:
    lines = [
        "# MVCPert multimodal repeat audit",
        "",
        "Version: `independent-compound-dose-plate-audit-2026-08-30`",
        "",
        "## Locked definition",
        "",
        "An independent repeat is a distinct acquisition plate within one `(compound, dose)` key. Technical wells on one plate are not counted as independent repeats. A support budget `k` requires at least `k + 1` distinct plates so that one plate can remain held out.",
        "",
        "The threshold is an engineering feasibility gate: at least 1000 eligible compounds is primary, 500–999 is supportive, and fewer than 500 is blocked for a primary quantitative claim.",
        "",
        "## Feasibility table",
        "",
        "| dataset | modality | split | support budget | eligible units | eligible compounds | with opposite modality | plate isolation | threshold |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {modality} | {split} | {budget_support_repeats}R | {eligible_compound_dose_units} | {eligible_compounds} | {eligible_compounds_with_opposite_modality} | {plate_independence} | {threshold_class} |".format(**row)
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- BBBC047 is represented by the existing plate-level NPZ artifacts and retains independent plate identity. The artifact has no batch/site fields; BBBC raw CP has well metadata, while the selected GE raw table has no well coordinate.",
        "- cpg0004 CP uses the locked compound split and six standard dose labels. Its GE plate-row artifact is filtered to the same split compounds and retains detection plate, dose, well and batch fields where present.",
        "- Counts are reported separately by split and modality; no aggregate profile is promoted to an independent repeat.",
        "- `BLOCKED` below 500 refers to this experiment's engineering gate, not to a claim that the modality has no biological information.",
        "",
        "## Source artifacts",
        "",
    ]
    for key, value in sources.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    bbbc_cp_root = Path(args.bbbc_cp_root)
    bbbc_ge_root = Path(args.bbbc_ge_root)
    cpg_cp_root = Path(args.cpg_cp_root)
    cpg_ge_file = Path(args.cpg_ge_file)
    cpg_lock = read_lock(Path(args.cpg_split_lock))

    rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for split in SELECTIONS:
        cp = load_rows(bbbc_cp_root / f"{split}_cp_plate_rows.npz", split)
        ge = load_rows(bbbc_ge_root / f"{split}_ge_plate_rows.npz", split)
        cp["dataset"], cp["modality"] = "bbbc047", "CP"
        ge["dataset"], ge["modality"] = "bbbc047", "GE"
        rows.extend(audit_pair("bbbc047", split, "CP", cp, ge, "artifact dose field; exact stored dose key"))
        rows.extend(audit_pair("bbbc047", split, "GE", ge, cp, "artifact dose field; exact stored dose key"))
    sources["bbbc047_cp_root"] = str(bbbc_cp_root)
    sources["bbbc047_ge_root"] = str(bbbc_ge_root)

    for split in SELECTIONS:
        cpg_cp_split_file = cpg_cp_root / f"{split}_cp_plate_rows.npz"
        cpg_cp_path = cpg_cp_split_file if cpg_cp_split_file.is_file() else cpg_cp_root / "cp_plate_rows.npz"
        cp = load_rows(cpg_cp_path, split, cpg_lock[split] if cpg_cp_path.name == "cp_plate_rows.npz" else None)
        ge = load_rows(cpg_ge_file, split, cpg_lock[split])
        cp["dataset"], cp["modality"] = "lincs_cpg0004", "CP"
        ge["dataset"], ge["modality"] = "lincs_cpg0004", "GE"
        rows.extend(audit_pair("lincs_cpg0004", split, "CP", cp, ge, "six standard dose labels from prepared CP artifact"))
        rows.extend(audit_pair("lincs_cpg0004", split, "GE", ge, cp, "six standard dose labels from prepared GE artifact"))
    sources["cpg0004_cp_root"] = str(cpg_cp_root)
    sources["cpg0004_ge_file"] = str(cpg_ge_file)
    sources["cpg0004_split_lock"] = str(args.cpg_split_lock)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "multimodal_repeat_audit.csv", rows)
    (outdir / "multimodal_repeat_audit.md").write_text(make_markdown(rows, sources), encoding="utf-8")
    summary = {"version": "independent-compound-dose-plate-audit-2026-08-30", "rows": rows, "sources": sources}
    (outdir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"version": summary["version"], "rows": len(rows), "outdir": str(outdir)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbbc-cp-root", required=True)
    parser.add_argument("--bbbc-ge-root", required=True)
    parser.add_argument("--cpg-cp-root", required=True)
    parser.add_argument("--cpg-ge-file", required=True)
    parser.add_argument("--cpg-split-lock", required=True)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
