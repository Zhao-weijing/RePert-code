#!/usr/bin/env python3
"""Compute the three-cell surface macro table from frozen confirmation rows.

This is deliberately a report-only post-processing step.  It reads exactly one
``CONFIRMATION_PER_COMPOUND.csv`` and never opens HDF5 files, target profiles,
model checkpoints, or prediction artifacts.

For every surface, method values are first averaged over drugs within each
target cell line and then averaged equally over A549/K562/MCF7.  Paired
contrasts use a hierarchical bootstrap: each target line resamples its own
matched drugs with replacement, then the three target-line means are averaged
with equal weight.  One hash-derived RNG seed is used per surface and the same
sampled drug indices are reused for all metrics and method contrasts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


TARGET_LINES: Tuple[str, ...] = ("A549", "K562", "MCF7")
SURFACES: Tuple[str, ...] = (
    "own_cell_unseen",
    "cross_cell_unseen",
    "cross_cell_seen",
)
METHODS: Tuple[str, ...] = (
    "M0_RawAggregate",
    "M1_CFRA1Aggregate",
    "M2_CFRA1Residual",
)
METRICS: Tuple[str, ...] = (
    "delta_pcc",
    "full_target_pcc",
    "raw_target_mse",
    "A",
    "z_same_F",
    "z_foreign_F",
    "E",
    "held_pcc_A",
    "held_pcc_B",
)
CONTRASTS: Tuple[Tuple[str, str, str], ...] = (
    ("M1_minus_M0", "M1_CFRA1Aggregate", "M0_RawAggregate"),
    ("M2_minus_M0", "M2_CFRA1Residual", "M0_RawAggregate"),
    ("M2_minus_M1", "M2_CFRA1Residual", "M1_CFRA1Aggregate"),
)
REQUIRED_COLUMNS = {
    "target_cell_line",
    "surface",
    "method",
    "drug",
    "teacher_regime",
    "capacity",
    "parameter_count",
    "confirmation_treated_loaded",
    "test_loaded",
    *METRICS,
}
DEFAULT_DRAWS = 10_000


def fail(message: str) -> "NoReturn":
    raise ValueError(message)


def parse_bool(value: str, *, field: str, row_number: int) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    fail(f"row {row_number}: {field} is not boolean: {value!r}")


def parse_float(value: str, *, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {field} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        fail(f"row {row_number}: {field} is not finite: {value!r}")
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(input_sha256: str, surface: str, draws: int) -> int:
    payload = (
        f"sciplex3-surface-macro-v1|input={input_sha256}|surface={surface}|"
        f"draws={draws}|targets={','.join(TARGET_LINES)}"
    ).encode("utf-8")
    # The modulo makes the hash-derived value valid for NumPy's uint32 seed
    # range while retaining deterministic behavior across Python processes.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def read_rows(path: Path) -> List[dict]:
    if not path.is_file():
        fail(f"input CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            fail(f"input CSV has no header: {path}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            fail(f"input CSV is missing required columns: {sorted(missing)}")
        rows: List[dict] = []
        for row_number, row in enumerate(reader, start=2):
            if row.get(None) is not None:
                fail(f"row {row_number}: extra unnamed columns are present")
            target = str(row.get("target_cell_line", "")).strip()
            surface = str(row.get("surface", "")).strip()
            method = str(row.get("method", "")).strip()
            drug = str(row.get("drug", "")).strip()
            if not target or not surface or not method or not drug:
                fail(f"row {row_number}: target/surface/method/drug cannot be empty")
            if target not in TARGET_LINES:
                fail(f"row {row_number}: unexpected target cell line {target!r}")
            if surface not in SURFACES:
                fail(f"row {row_number}: unexpected surface {surface!r}")
            if method not in METHODS:
                fail(f"row {row_number}: unexpected method {method!r}")

            parsed = dict(row)
            parsed["target_cell_line"] = target
            parsed["surface"] = surface
            parsed["method"] = method
            parsed["drug"] = drug
            parsed["confirmation_treated_loaded_bool"] = parse_bool(
                row.get("confirmation_treated_loaded", ""),
                field="confirmation_treated_loaded",
                row_number=row_number,
            )
            parsed["test_loaded_bool"] = parse_bool(
                row.get("test_loaded", ""),
                field="test_loaded",
                row_number=row_number,
            )
            for metric in METRICS:
                parsed[f"{metric}_float"] = parse_float(
                    row.get(metric, ""), field=metric, row_number=row_number
                )
            rows.append(parsed)

    if not rows:
        fail("input CSV has no data rows")
    return rows


def format_values(values: Iterable[str]) -> str:
    unique = sorted({str(value).strip() for value in values})
    return "|".join(unique)


def build_index(rows: Sequence[dict]) -> Tuple[Dict[Tuple[str, str, str], Dict[str, dict]], Dict[str, str]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["target_cell_line"], row["surface"], row["method"])
        drug = row["drug"]
        if drug in grouped[key]:
            fail(f"duplicate target/surface/method/drug row: {key + (drug,)}")
        grouped[key][drug] = row

    for target in TARGET_LINES:
        for surface in SURFACES:
            sets = [set(grouped[(target, surface, method)]) for method in METHODS]
            if not all(drug_set == sets[0] for drug_set in sets[1:]):
                fail(f"method drug sets are not paired for {target}/{surface}")
            expected = 183 if surface == "cross_cell_seen" else 36
            if len(sets[0]) != expected:
                fail(
                    f"unexpected drug count for {target}/{surface}: "
                    f"got {len(sets[0])}, expected {expected}"
                )

    metadata: Dict[str, str] = {}
    for surface in SURFACES:
        surface_rows = [row for row in rows if row["surface"] == surface]
        metadata[f"{surface}:teacher_regime"] = format_values(
            row.get("teacher_regime", "") for row in surface_rows
        )
        metadata[f"{surface}:capacity"] = format_values(
            row.get("capacity", "") for row in surface_rows
        )
        metadata[f"{surface}:parameter_count"] = format_values(
            row.get("parameter_count", "") for row in surface_rows
        )
        if not all(row["confirmation_treated_loaded_bool"] for row in surface_rows):
            fail(f"confirmation_treated_loaded is not true for every {surface} row")
        if not all(row["test_loaded_bool"] for row in surface_rows):
            fail(f"test_loaded is not true for every {surface} row")
    return grouped, metadata


def percentile_ci(draws: np.ndarray) -> Tuple[float, float]:
    try:
        quantiles = np.percentile(draws, [2.5, 97.5], method="linear")
    except TypeError:  # NumPy < 1.22 compatibility on older remote environments.
        quantiles = np.percentile(draws, [2.5, 97.5], interpolation="linear")
    return float(quantiles[0]), float(quantiles[1])


def hierarchical_contrast(
    values_by_target: Mapping[str, np.ndarray],
    sampled_indices: Mapping[str, np.ndarray],
    draws: int,
) -> Tuple[float, float, float]:
    line_estimates = [float(values_by_target[target].mean()) for target in TARGET_LINES]
    estimate = float(np.mean(line_estimates))

    boot_line_means = []
    for target in TARGET_LINES:
        values = values_by_target[target]
        indices = sampled_indices[target]
        # Each row is one bootstrap draw and contains that target line's
        # matched drugs sampled with replacement.
        boot_line_means.append(values[indices].mean(axis=1))
    boot_macro = np.mean(np.stack(boot_line_means, axis=0), axis=0)
    ci_low, ci_high = percentile_ci(boot_macro)
    if boot_macro.shape != (draws,):
        fail(f"internal bootstrap shape error: {boot_macro.shape}")
    return estimate, ci_low, ci_high


def surface_label(surface: str) -> Tuple[str, str]:
    if surface == "cross_cell_seen":
        return "seen_drug_transfer", "seen_drugs_historical_locked_reuse"
    if surface == "cross_cell_unseen":
        return "unseen", "unseen_compounds_historical_confirmation"
    return "unseen", "unseen_compounds_historical_confirmation"


def make_output(rows: Sequence[dict], input_path: Path, script_path: Path, draws: int) -> List[dict]:
    if draws != DEFAULT_DRAWS:
        fail(f"the frozen report contract requires exactly {DEFAULT_DRAWS} bootstrap draws")
    grouped, metadata = build_index(rows)
    input_sha256 = sha256_file(input_path)
    script_sha256 = sha256_file(script_path)
    output_rows: List[dict] = []

    for surface in SURFACES:
        drugs_by_target: Dict[str, List[str]] = {}
        for target in TARGET_LINES:
            drugs_by_target[target] = sorted(
                grouped[(target, surface, METHODS[0])].keys()
            )
        n_by_target = [len(drugs_by_target[target]) for target in TARGET_LINES]
        n_by_target_text = "|".join(
            f"{target}:{len(drugs_by_target[target])}" for target in TARGET_LINES
        )
        seed = stable_seed(input_sha256, surface, draws)
        rng = np.random.default_rng(seed)
        sampled_indices = {
            target: rng.integers(
                low=0,
                high=len(drugs_by_target[target]),
                size=(draws, len(drugs_by_target[target])),
                dtype=np.int32,
            )
            for target in TARGET_LINES
        }
        class_label, historical_label = surface_label(surface)

        absolute_by_metric: Dict[str, Dict[str, float]] = {}
        for metric in METRICS:
            absolute_by_metric[metric] = {}
            for method in METHODS:
                line_means = []
                for target in TARGET_LINES:
                    values = np.asarray(
                        [
                            grouped[(target, surface, method)][drug][f"{metric}_float"]
                            for drug in drugs_by_target[target]
                        ],
                        dtype=np.float64,
                    )
                    line_means.append(float(values.mean()))
                absolute_by_metric[metric][method] = float(np.mean(line_means))

        for metric in METRICS:
            row = {
                "surface": surface,
                "surface_class": class_label,
                "historical_status": historical_label,
                "metric": metric,
                "target_line_macro_not_biological_ci": "true",
                "n_target_lines": str(len(TARGET_LINES)),
                "target_lines": "|".join(TARGET_LINES),
                "n_drugs_by_target_line": n_by_target_text,
                "n_drugs_macro_basis": str(min(n_by_target)),
                "absolute_aggregation": "drug_mean_then_equal_target_line_mean",
                "M0_RawAggregate_value": str(absolute_by_metric[metric][METHODS[0]]),
                "M1_CFRA1Aggregate_value": str(absolute_by_metric[metric][METHODS[1]]),
                "M2_CFRA1Residual_value": str(absolute_by_metric[metric][METHODS[2]]),
                "bootstrap_draws": str(draws),
                "bootstrap_unit": "hierarchical_paired_drug_within_target_line",
                "bootstrap_seed": str(seed),
                "ci_level": "0.95",
                "input_sha256": input_sha256,
                "script_sha256": script_sha256,
                "teacher_regime": metadata[f"{surface}:teacher_regime"],
                "capacity": metadata[f"{surface}:capacity"],
                "parameter_count": metadata[f"{surface}:parameter_count"],
                "confirmation_treated_loaded_all": "true",
                "test_loaded_all": "true",
            }
            for label, left, right in CONTRASTS:
                values_by_target = {}
                for target in TARGET_LINES:
                    drugs = drugs_by_target[target]
                    left_values = np.asarray(
                        [
                            grouped[(target, surface, left)][drug][f"{metric}_float"]
                            for drug in drugs
                        ],
                        dtype=np.float64,
                    )
                    right_values = np.asarray(
                        [
                            grouped[(target, surface, right)][drug][f"{metric}_float"]
                            for drug in drugs
                        ],
                        dtype=np.float64,
                    )
                    values_by_target[target] = left_values - right_values
                estimate, ci_low, ci_high = hierarchical_contrast(
                    values_by_target, sampled_indices, draws
                )
                row[f"{label}_estimate"] = str(estimate)
                row[f"{label}_ci_low"] = str(ci_low)
                row[f"{label}_ci_high"] = str(ci_high)
            output_rows.append(row)
    return output_rows


FIELDNAMES: Tuple[str, ...] = (
    "surface",
    "surface_class",
    "historical_status",
    "metric",
    "target_line_macro_not_biological_ci",
    "n_target_lines",
    "target_lines",
    "n_drugs_by_target_line",
    "n_drugs_macro_basis",
    "absolute_aggregation",
    "M0_RawAggregate_value",
    "M1_CFRA1Aggregate_value",
    "M2_CFRA1Residual_value",
    "M1_minus_M0_estimate",
    "M1_minus_M0_ci_low",
    "M1_minus_M0_ci_high",
    "M2_minus_M0_estimate",
    "M2_minus_M0_ci_low",
    "M2_minus_M0_ci_high",
    "M2_minus_M1_estimate",
    "M2_minus_M1_ci_low",
    "M2_minus_M1_ci_high",
    "bootstrap_draws",
    "bootstrap_unit",
    "bootstrap_seed",
    "ci_level",
    "input_sha256",
    "script_sha256",
    "teacher_regime",
    "capacity",
    "parameter_count",
    "confirmation_treated_loaded_all",
    "test_loaded_all",
)


def default_input(base: Path) -> Path:
    candidates = (
        base / "confirmation" / "CONFIRMATION_PER_COMPOUND.csv",
        base / "formal_v5_recovered_confirmation" / "CONFIRMATION_PER_COMPOUND.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def main(argv: Sequence[str] | None = None) -> int:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build the sci-Plex3 three-target-line surface macro comparison."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input(base),
        help="frozen confirmation/CONFIRMATION_PER_COMPOUND.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "surface_macro" / "SURFACE_MACRO_FULL_COMPARISON.csv",
        help="new report-only CSV output",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=DEFAULT_DRAWS,
        help=f"bootstrap draws; the contract fixes this at {DEFAULT_DRAWS}",
    )
    args = parser.parse_args(argv)

    try:
        input_path = args.input.resolve()
        output_path = args.output.resolve()
        rows = read_rows(input_path)
        output_rows = make_output(rows, input_path, Path(__file__).resolve(), args.draws)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="raise")
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"input={input_path}")
        print(f"output={output_path}")
        print(f"rows={len(output_rows)}")
        print(f"output_sha256={sha256_file(output_path)}")
        for row in output_rows:
            if row["metric"] in {"delta_pcc", "E", "raw_target_mse"}:
                print(
                    "key="
                    f"{row['surface']},{row['metric']},"
                    f"M0={row['M0_RawAggregate_value']},"
                    f"M1={row['M1_CFRA1Aggregate_value']},"
                    f"M2={row['M2_CFRA1Residual_value']},"
                    f"M2-M0={row['M2_minus_M0_estimate']} "
                    f"[{row['M2_minus_M0_ci_low']},{row['M2_minus_M0_ci_high']}]"
                )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
