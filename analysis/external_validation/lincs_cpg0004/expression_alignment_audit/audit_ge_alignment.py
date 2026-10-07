#!/usr/bin/env python3
"""Recompute the cpg0004 CP--GE condition alignment from compact audit tables."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cp-manifest", type=Path, required=True)
    p.add_argument("--ge-records", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)
    cp = pd.read_csv(args.cp_manifest); ge = pd.read_csv(args.ge_records)
    cp = cp[cp["dose"].isin([0.04, 0.12, 0.37, 1.11, 3.33, 10.0])].copy()
    ge = ge[ge["dose_recode"].isin([0.04, 0.12, 0.37, 1.11, 3.33, 10.0])].copy()
    cp_cond = cp.groupby(["compound_id", "dose"], as_index=False).agg(cp_plates=("plate", "nunique"), cp_rows=("plate", "size"))
    ge_cond = ge.groupby(["compound_id", "dose_recode"], as_index=False).agg(ge_plates=("det_plate", "nunique"), ge_rows=("det_plate", "size"))
    ge_cond = ge_cond.rename(columns={"dose_recode": "dose"})
    aligned = cp_cond.merge(ge_cond, on=["compound_id", "dose"], how="outer", indicator=True)
    aligned["status"] = aligned["_merge"].map({"both": "exact_compound_and_standard_dose", "left_only": "CP_only_condition", "right_only": "GE_only_condition"})
    aligned.drop(columns=["_merge"]).to_csv(args.outdir / "cp_ge_condition_alignment.csv", index=False)
    summary = {
        "cp_standard_conditions": int(len(cp_cond)), "ge_standard_conditions": int(len(ge_cond)),
        "exact_compound_dose_matches": int((aligned["status"] == "exact_compound_and_standard_dose").sum()),
        "cp_only_conditions": int((aligned["status"] == "CP_only_condition").sum()),
        "ge_only_conditions": int((aligned["status"] == "GE_only_condition").sum()),
        "compound_only_cp_to_ge": int(len(set(cp.compound_id) & set(ge.compound_id))),
        "cp_compounds": int(cp.compound_id.nunique()), "ge_compounds": int(ge.compound_id.nunique()),
        "ge_repeat_distribution": {str(k): int(v) for k, v in ge_cond.ge_plates.value_counts().sort_index().items()},
        "cp_repeat_distribution": {str(k): int(v) for k, v in cp_cond.cp_plates.value_counts().sort_index().items()},
        "dose_labels": sorted(set(cp.dose) & set(ge.dose_recode)),
        "ge_rows_with_smiles": int(ge["inchikey_or_smiles"].notna().sum()),
        "ge_rows": int(len(ge)),
    }
    (args.outdir / "alignment_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
