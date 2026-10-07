#!/usr/bin/env python3
"""Freeze the GE-derived compound -> SMILES map for the cpg0004 prior.

This reads only the compact metadata audit (not CP treatment values) and
rejects a compound if its GE rows disagree on canonical structure.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ge-records", type=Path, default=Path("../00_data_audit/ge_records.csv"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    ge = pd.read_csv(args.ge_records)
    ge = ge[ge["inchikey_or_smiles"].notna()].copy()
    ge["smiles"] = ge["inchikey_or_smiles"].astype(str).str.strip()
    rows = []
    for compound, group in ge.groupby("compound_id", sort=True):
        values = sorted(set(x for x in group["smiles"] if x and x.lower() not in {"nan", "none"}))
        if len(values) != 1:
            raise ValueError(f"Compound {compound} has {len(values)} GE structures")
        rows.append({"compound_id": str(compound), "smiles": values[0], "ge_rows": int(len(group)), "ge_doses": int(group["dose_recode"].nunique())})
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print({"compound_count": len(out), "out": str(args.out)})


if __name__ == "__main__":
    main()
