#!/usr/bin/env python3
"""Post-test, read-only absolute summary for the frozen strict v2 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
WRAPPER = (Path(__file__).resolve().parents[3] / "analysis/predictor_supervision/bbbc047/run_strict_selected_repeat_m2_v2.py")
spec = importlib.util.spec_from_file_location("strict_selected_v2", WRAPPER)
if spec is None or spec.loader is None:
    raise RuntimeError(WRAPPER)
V2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = V2
spec.loader.exec_module(V2)
RUN = V2.RUN

METRICS = ("same_z", "delta_pcc", "full_target_pcc", "foreign_z", "E")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty summary")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-rounds", type=int, default=10_000)
    args = parser.parse_args()
    root = args.output_root.resolve()
    complete_path = root / "TEST_COMPLETE.json"
    per_path = root / "TEST_PER_COMPOUND.csv"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "PASS" or complete.get("phase") != "test":
        raise RuntimeError("the frozen test completion marker is not PASS")
    if complete.get("test_used_for_selection"):
        raise RuntimeError("a post-test audit cannot repair selection leakage")
    if not complete.get("test_loaded_after_authorization"):
        raise RuntimeError("test authorization marker missing")

    values: dict[tuple[int, str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    foreign_ok: dict[tuple[int, str], int] = {}
    with per_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"budget", "method", "seed", "compound_id", "foreign_ok", *METRICS}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"missing columns: {sorted(missing)}")
        for row in reader:
            budget, method, compound, seed = int(row["budget"]), row["method"], row["compound_id"], int(row["seed"])
            key = (budget, method, compound)
            if seed in values[key]:
                raise RuntimeError(f"duplicate seed row: {key}, {seed}")
            values[key][seed] = {metric: float(row[metric]) for metric in METRICS}
            foreign_key = (budget, compound)
            flag = int(row["foreign_ok"])
            if foreign_key in foreign_ok and foreign_ok[foreign_key] != flag:
                raise RuntimeError(f"inconsistent foreign flag: {foreign_key}")
            foreign_ok[foreign_key] = flag

    expected_seeds = set(RUN.SEEDS)
    by_arm: dict[tuple[int, str], list[str]] = defaultdict(list)
    for budget, method, compound in values:
        if set(values[(budget, method, compound)]) != expected_seeds:
            raise RuntimeError(f"seed coverage failure: {(budget, method, compound)}")
        by_arm[(budget, method)].append(compound)

    rows: list[dict[str, Any]] = []
    for (budget, method), compounds in sorted(by_arm.items()):
        compounds = sorted(compounds)
        row: dict[str, Any] = {
            "budget": budget,
            "method": method,
            "parameter_count": RUN.TOTAL_PARAMETERS,
            "n_compounds": len(compounds),
            "foreign_compounds": sum(foreign_ok[(budget, compound)] for compound in compounds),
            "seed_count": len(expected_seeds),
        }
        compound_array = np.asarray(compounds, dtype=str)
        for metric in METRICS:
            per_compound: list[float] = []
            for compound in compounds:
                seed_values = np.asarray(
                    [values[(budget, method, compound)][seed][metric] for seed in RUN.SEEDS],
                    dtype=np.float64,
                )
                per_compound.append(float(np.nanmean(seed_values)) if np.isfinite(seed_values).any() else float("nan"))
            absolute = np.asarray(per_compound, dtype=np.float64)
            point, low, high, n = RUN.BASE.bootstrap_pair(
                absolute,
                np.zeros(len(absolute), dtype=np.float64),
                compound_array,
                RUN.BASE.LEGACY.stable_int(RUN.SEED, f"{RUN.VERSION}|post-test-absolute|b{budget}|{method}|{metric}"),
                args.bootstrap_rounds,
            )
            row[f"{metric}_mean"] = point
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_n"] = n
        rows.append(row)

    expected_arms = {(budget, method) for budget in RUN.BUDGETS for method in RUN.METHODS}
    if set(by_arm) != expected_arms:
        raise RuntimeError(f"method/budget coverage failure: {set(by_arm)} != {expected_arms}")
    out_path = root / "TEST_METHOD_SUMMARY.csv"
    write_csv(out_path, rows)
    audit = {
        "version": RUN.VERSION,
        "status": "PASS",
        "phase": "post_test_absolute_summary",
        "post_test_read_only": True,
        "test_used_for_selection": False,
        "bootstrap_unit": "compound",
        "bootstrap_rounds": args.bootstrap_rounds,
        "seed_aggregation": "mean within compound over the five frozen master seeds before bootstrap",
        "source_test_complete_sha256": sha256(complete_path),
        "source_test_per_compound_sha256": sha256(per_path),
        "output_test_method_summary_sha256": sha256(out_path),
        "output_rows": len(rows),
    }
    (root / "TEST_METHOD_SUMMARY_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
