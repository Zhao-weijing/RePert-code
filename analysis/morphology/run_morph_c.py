#!/usr/bin/env python3
"""Morph-C evaluator framework for the frozen cpg0004 dose experiment.

This file is intentionally a small, auditable entry point around the already
completed ``biological_applications/dose_response`` evaluator.  It accepts
the feature-audit module map as an input and exposes the frozen paths,
eligibility rule, endpoint definition, and output contract needed by the full
Morph-C run.  No model is trained or tuned here.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path
from typing import Any


VERSION = "cpg0004-LINCS-Morph-C-framework-2026-08-30"
DOSES = ("0.04", "0.12", "0.37", "1.11", "3.33", "10")
SEEDS = (3407, 42, 2025)
MIN_PRIMARY_FEATURES = 8
BOOTSTRAP_ROUNDS = 10_000
PRIMARY_MODULES = ("DNA", "RNA", "ER", "Mito", "AGP", "Shape", "Cross-channel")
OUTPUT_CONTRACT = (
    "PROTOCOL.md",
    "CONFIG.json",
    "FEATURE_MAPPING.csv",
    "ELIGIBLE_COMPOUNDS.csv",
    "EXCLUSIONS.csv",
    "METRICS_BY_COMPOUND.csv",
    "METRICS_BY_MODULE.csv",
    "METRICS_BY_SEED.csv",
    "PAIRED_CONTRASTS.csv",
    "BOOTSTRAP_CI.csv",
    "NEGATIVE_CONTROLS.csv",
    "RESULTS.md",
    "DECISION.md",
    "figures/",
)


def norm_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def norm_module(value: object) -> str:
    token = " ".join(str(value or "").strip().split())
    key = token.lower().replace("_", " ").replace("-", " ")
    aliases = {
        "dna": "DNA",
        "rna": "RNA",
        "er": "ER",
        "mito": "Mito",
        "mitochondria": "Mito",
        "agp": "AGP",
        "shape": "Shape",
        "area shape": "Shape",
        "cross channel": "Cross-channel",
        "crosschannel": "Cross-channel",
        "unmapped": "Unmapped",
        "unknown": "Unmapped",
        "": "Unmapped",
    }
    return aliases.get(key, token)


def pick_column(fieldnames: list[str], aliases: set[str]) -> str | None:
    normalized = {norm_header(name): name for name in fieldnames}
    for alias in aliases:
        if norm_header(alias) in normalized:
            return normalized[norm_header(alias)]
    return None


def read_feature_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"CP feature-name file not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        column = pick_column(fields, {"feature_name", "feature", "name"})
        if column is None:
            if not fields:
                raise ValueError(f"Feature-name file has no header: {path}")
            column = fields[0]
        names = [str(row.get(column, "")).strip() for row in reader]
    names = [name for name in names if name]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate CP feature names in {path}")
    return names


def load_module_map(path: Path, feature_names: list[str]) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """Read the external feature-audit CSV without redefining modules.

    Long-form audit maps with ``feature_name``/``feature_index`` and
    ``biological_module`` are supported, as are common header aliases.  Rows
    marked Unmapped/Unknown are retained in the audit table but excluded from
    primary modules.  No feature is assigned by a fallback substring rule.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Morph-C module map not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        feature_column = pick_column(fields, {"feature_name", "feature", "cp_feature", "name"})
        index_column = pick_column(fields, {"feature_index", "feature_idx", "index", "column_index"})
        module_column = pick_column(fields, {"biological_module", "module", "module_name", "module_label"})
        status_column = pick_column(fields, {"mapping_status", "annotation_status", "status"})
        if feature_column is None and index_column is None:
            raise ValueError(f"Module map needs a feature_name or feature_index column: {path}")
        if module_column is None:
            raise ValueError(f"Module map needs a biological_module/module column: {path}")
        rows = list(reader)

    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row_number, row in enumerate(rows, start=2):
        feature = str(row.get(feature_column, "")).strip() if feature_column else ""
        index: int | None = None
        if index_column and str(row.get(index_column, "")).strip():
            try:
                index = int(float(str(row[index_column]).strip()))
            except ValueError as exc:
                raise ValueError(f"Invalid feature index at {path}:{row_number}") from exc
        if index is None and feature:
            try:
                index = feature_names.index(feature)
            except ValueError:
                index = None
        if index is None or not 0 <= index < len(feature_names):
            raise ValueError(f"Module map feature cannot be aligned at {path}:{row_number}: {feature!r}")
        if feature and feature_names[index] != feature:
            raise ValueError(
                f"Module map feature/index mismatch at {path}:{row_number}: "
                f"index={index} names={feature_names[index]!r} row={feature!r}"
            )
        if index in seen:
            raise ValueError(f"Duplicate CP feature index in module map at {path}:{row_number}: {index}")
        seen.add(index)
        module = norm_module(row.get(module_column, ""))
        status = str(row.get(status_column, "")) if status_column else ""
        records.append(
            {
                "feature_index": index,
                "feature_name": feature_names[index],
                "biological_module": module,
                "mapping_status": status,
                "primary_eligible": False,
            }
        )

    groups: dict[str, list[int]] = {}
    for record in records:
        module = str(record["biological_module"])
        if module not in {"Unmapped", "Unknown"}:
            groups.setdefault(module, []).append(int(record["feature_index"]))
    eligible = {module: sorted(indices) for module, indices in groups.items() if len(indices) >= MIN_PRIMARY_FEATURES}
    for record in records:
        record["primary_eligible"] = str(record["biological_module"]) in eligible
    return records, eligible


def import_dose_response():
    module_path = (Path(__file__).resolve().parents[2] / "analysis/biological_applications/dose_response/run_dose_response.py")
    spec = importlib.util.spec_from_file_location("frozen_cpg0004_dose_response", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import frozen dose evaluator: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_argument_frame(args: argparse.Namespace) -> dict[str, Any]:
    """Return the exact frozen evaluator interface for the implementation stage."""
    return {
        "data": str(args.data.resolve()),
        "p0_root": str(args.p0_root.resolve()),
        "ge_root": str(args.ge_root.resolve()),
        "pair_root": str(args.pair_root.resolve()),
        "hit_dir": str(args.hit_dir.resolve()),
        "dose_response_module": str(
            ((Path(__file__).resolve().parents[2] / "analysis/biological_applications/dose_response/run_dose_response.py")).resolve()
        ),
        "methods": {"M0": "raw", "M1": "teacher", "M2": "posterior"},
        "seeds": list(SEEDS),
        "doses": list(DOSES),
        "strict_repeats_per_dose": 5,
        "frozen_support_rotations": 2,
        "reference_repeats": 4,
        "m1_pre_endpoint_expected_compounds": 260,
        "m2_pre_endpoint_expected_compounds": 258,
        "primary_endpoint": "unweighted MacroTC over modules with >=8 mapped features",
        "module_endpoint": "Spearman of 15 upper-triangular 1-PCC dose distances",
        "undefined_policy": "UNSCORABLE; no epsilon",
        "bootstrap": {"rounds": int(args.bootstrap_rounds), "unit": "compound", "paired": True},
    }


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    experiments = here.parent
    base = experiments / "external_validation" / "lincs_cpg0004"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-mapping", type=Path, default=here / "feature_annotation_audit" / "MODULE_MAPPING.csv")
    parser.add_argument("--feature-names", type=Path, default=base / "data_preparation" / "artifact" / "feature_names.csv")
    parser.add_argument("--data", type=Path, default=base / "data_preparation" / "artifact" / "cp_plate_rows.npz")
    parser.add_argument("--p0-root", type=Path, default=base / "virtual_prior" / "results" / "1r_all")
    parser.add_argument("--ge-root", type=Path, default=base / "single_repeat_expression_evidence" / "results" / "1r_all")
    parser.add_argument("--pair-root", type=Path, default=base / "cell_painting_repeat_benchmark" / "results" / "1r_all")
    parser.add_argument("--hit-dir", type=Path, default=experiments / "biological_applications" / "hit_recovery")
    parser.add_argument("--outdir", type=Path, default=here / "module_dose_trajectories")
    parser.add_argument("--bootstrap-rounds", type=int, default=BOOTSTRAP_ROUNDS)
    parser.add_argument("--framework-only", action="store_true", help="Write only the frozen parameter framework and audit mapping.")
    return parser.parse_args()


def write_framework(outdir: Path, frame: dict[str, Any], records: list[dict[str, Any]], eligible: dict[str, list[int]]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "CONFIG.json").write_text(
        json.dumps({**frame, "primary_modules": {k: len(v) for k, v in sorted(eligible.items())}, "version": VERSION, "output_contract": list(OUTPUT_CONTRACT)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (outdir / "FEATURE_MAPPING.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["feature_index", "feature_name", "biological_module", "mapping_status", "primary_eligible"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    protocol = "\n".join(
        [
            f"# Morph-C protocol framework ({VERSION})",
            "",
            "The evaluator imports the frozen cpg0004 six-dose evaluator and does not retrain or retune any model.",
            "",
            f"- Doses: `{', '.join(DOSES)}`; seeds: `{', '.join(map(str, SEEDS))}`.",
            "- M0/M1/M2 are frozen raw, teacher, and GE-updated posterior profiles.",
            "- Two recorded support rotations are scored separately; each reference is the mean of the other four physical repeats.",
            "- Module TC is Spearman correlation of the 15 upper-triangular `1-PCC` dose distances.",
            "- Primary MacroTC is the unweighted mean over mapped modules with at least eight features.",
            "- Undefined PCC/Spearman values are `UNSCORABLE`; no epsilon is added.",
            "- Bootstrap unit is the compound, paired within each comparison, with 10,000 rounds by default.",
            "",
            "Required full-run outputs:",
            "",
            *[f"- `{name}`" for name in OUTPUT_CONTRACT],
        ]
    )
    (outdir / "PROTOCOL.md").write_text(protocol + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_rounds <= 0:
        raise ValueError("--bootstrap-rounds must be positive")
    # Import is intentional: the full implementation must call the audited
    # six-dose/support/reference reconstruction rather than duplicate it.
    import_dose_response()
    feature_names = read_feature_names(args.feature_names)
    records, eligible = load_module_map(args.module_mapping, feature_names)
    frame = frozen_argument_frame(args)
    frame["feature_count"] = len(feature_names)
    frame["mapped_feature_count"] = len(records)
    frame["mapping_rate"] = len(records) / len(feature_names) if feature_names else 0.0
    frame["primary_module_feature_counts"] = {k: len(v) for k, v in sorted(eligible.items())}
    if args.framework_only:
        write_framework(args.outdir, frame, records, eligible)
    print(json.dumps({"version": VERSION, "framework_only": bool(args.framework_only), **frame}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
