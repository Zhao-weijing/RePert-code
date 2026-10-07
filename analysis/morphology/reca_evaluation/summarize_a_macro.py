#!/usr/bin/env python3
"""Additive, compound-paired seven-module Figure 4A macro summary.

This read-only adapter consumes the physical-view metrics emitted by
``run_figure4_cfra.py``.  It averages the seven predefined module Fisher-z
values within each (seed, compound, dose, support rotation), then averages
those values within compound before the paired compound bootstrap.  It does
not open profile arrays, refit candidates, or select a cohort.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


STATUS = "HISTORICAL_LOCKED_RECOMPUTE"
EXPECTED_MODULES = {
    "A_module_fisher_z_DNA",
    "A_module_fisher_z_RNA",
    "A_module_fisher_z_ER",
    "A_module_fisher_z_Mito",
    "A_module_fisher_z_AGP",
    "A_module_fisher_z_Shape",
    "A_module_fisher_z_Cross-channel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--per-compound-out", required=True, type=Path)
    parser.add_argument("--bootstrap-rounds", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_rounds < 100:
        raise ValueError("bootstrap rounds must be at least 100")
    if not args.metrics.is_file() or args.summary_out.exists() or args.per_compound_out.exists():
        raise FileExistsError("inputs/outputs do not satisfy the no-overwrite contract")
    frame = pd.read_csv(args.metrics, low_memory=False)
    required = {"seed", "compound", "dose", "rotation", "endpoint", "raw", "cfra"}
    if required - set(frame.columns):
        raise RuntimeError(f"metrics missing {sorted(required - set(frame.columns))}")
    frame = frame[frame.endpoint.isin(EXPECTED_MODULES)].copy()
    counts = frame.groupby(["seed", "compound", "dose", "rotation"], sort=True).endpoint.nunique()
    if frame.empty or not (counts == len(EXPECTED_MODULES)).all():
        raise RuntimeError("incomplete seven-module profile encountered")
    views = frame.groupby(["seed", "compound", "dose", "rotation"], as_index=False, sort=True)[["raw", "cfra"]].mean()
    compounds = views.groupby("compound", as_index=False, sort=True)[["raw", "cfra"]].mean()
    compounds["cfra_minus_raw"] = compounds.cfra - compounds.raw
    if not np.isfinite(compounds[["raw", "cfra", "cfra_minus_raw"]].to_numpy(dtype=float)).all():
        raise FloatingPointError("non-finite compound macro score")
    digest = hashlib.sha256(b"cpg0004-figure4-cfra|A_macro_module_fisher_z").digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    draws = rng.integers(0, len(compounds), size=(args.bootstrap_rounds, len(compounds)))
    boot = compounds.cfra_minus_raw.to_numpy(dtype=float)[draws].mean(axis=1)
    summary = pd.DataFrame([{
        "endpoint": "A_macro_module_fisher_z",
        "raw": float(compounds.raw.mean()),
        "cfra": float(compounds.cfra.mean()),
        "cfra_minus_raw": float(compounds.cfra_minus_raw.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n_compounds": int(len(compounds)),
        "bootstrap_unit": "compound",
        "bootstrap_rounds": int(args.bootstrap_rounds),
        "status": STATUS,
        "aggregation": "mean seven modules within physical view, then mean dose/rotation/seed within compound",
    }])
    summary.to_csv(args.summary_out, index=False)
    compounds.to_csv(args.per_compound_out, index=False)


if __name__ == "__main__":
    main()
