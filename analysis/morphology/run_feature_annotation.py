#!/usr/bin/env python3
"""Freeze and audit the CellProfiler feature-to-biology mapping for cpg0004.

This stage is intentionally small and deterministic.  It reads the feature
order used by the frozen cpg0004 CP artifact and writes only annotation
artifacts; it never reads test outcomes and never changes the model feature
subset.  ``Batch_Number`` is retained because it is part of the frozen model
input, but is explicitly excluded from every biological endpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = (
    REPO_ROOT
    / "experiments"
    / "external_validation"
    / "lincs_cpg0004"
    / "data_preparation"
    / "artifact"
    / "feature_names.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "feature_annotation_audit"

EXPECTED_TOTAL = 242
EXPECTED_RELIABLE = 241
FROZEN_METADATA_FEATURES = {"Batch_Number"}
KNOWN_OBJECTS = ("Cells", "Cytoplasm", "Nuclei")
KNOWN_CHANNELS = ("DNA", "RNA", "ER", "Mito", "AGP")
KNOWN_FAMILIES = (
    "AreaShape",
    "Intensity",
    "Texture",
    "Granularity",
    "RadialDistribution",
    "Neighbors",
    "Correlation",
)
MODULE_ORDER = ("DNA", "RNA", "ER", "Mito", "AGP", "Shape", "Cross-channel")
FAMILY_ORDER = (*KNOWN_FAMILIES, "Others")
OBJECT_ORDER = (*KNOWN_OBJECTS, "Unmapped")

DICTIONARY_FIELDS = (
    "feature_index",
    "feature_name",
    "object",
    "channel",
    "channel_scope",
    "measurement_family",
    "biological_module",
    "mapping_status",
    "biological_endpoint",
    "mapping_reason",
)
MODULE_FIELDS = (
    "feature_index",
    "feature_name",
    "biological_module",
    "object",
    "channel",
    "measurement_family",
    "mapping_status",
    "primary_module",
    "biological_endpoint",
)
UNMAPPED_FIELDS = (
    "feature_index",
    "feature_name",
    "object",
    "channel",
    "channel_scope",
    "measurement_family",
    "biological_module",
    "mapping_status",
    "biological_endpoint",
    "mapping_reason",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Prefer a repository-relative path while keeping external paths valid."""

    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def input_record(path: Path, role: str) -> dict[str, Any]:
    record: dict[str, Any] = {"role": role, "path": display_path(path), "exists": path.exists()}
    if path.exists():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
    else:
        record["size_bytes"] = None
        record["sha256"] = None
    return record


def read_feature_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "feature" not in reader.fieldnames:
            raise ValueError(f"Feature list must contain a 'feature' column: {path}")
        values = [str(row.get("feature", "")).strip() for row in reader]
    if any(not value for value in values):
        bad = next(i for i, value in enumerate(values) if not value)
        raise ValueError(f"Empty feature name at input row {bad + 2}: {path}")
    return values


def unique_tokens(tokens: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for token in tokens:
        if token in KNOWN_CHANNELS and token not in seen:
            output.append(token)
            seen.add(token)
    return output


def parse_feature(feature_name: str, feature_index: int) -> dict[str, str | int]:
    """Apply the locked, conservative CellProfiler naming rules.

    Priority is deliberately explicit:
      1. frozen metadata is Unmapped/Excluded;
      2. multi-channel/correlation features are Cross-channel;
      3. single-channel measurement features use the channel module;
      4. AreaShape/Neighbors without a channel are Shape;
      5. anything else remains Unmapped.
    """

    if feature_name in FROZEN_METADATA_FEATURES:
        return {
            "feature_index": feature_index,
            "feature_name": feature_name,
            "object": "Unmapped",
            "channel": "",
            "channel_scope": "none",
            "measurement_family": "Others",
            "biological_module": "Unmapped",
            "mapping_status": "Unmapped",
            "biological_endpoint": "Excluded",
            "mapping_reason": (
                "Frozen metadata feature retained in model input; it is not a "
                "CellProfiler biological measurement and is excluded from all endpoints."
            ),
        }

    tokens = feature_name.split("_")
    object_name = tokens[0] if tokens and tokens[0] in KNOWN_OBJECTS else ""
    family = tokens[1] if len(tokens) > 1 and tokens[1] in KNOWN_FAMILIES else "Others"
    channels = unique_tokens(tokens)
    channel = "|".join(channels)

    base: dict[str, str | int] = {
        "feature_index": feature_index,
        "feature_name": feature_name,
        "object": object_name,
        "channel": channel,
        "channel_scope": "multi_channel" if len(channels) > 1 else ("single_channel" if channels else "none"),
        "measurement_family": family,
        "biological_module": "Unmapped",
        "mapping_status": "Unmapped",
        "biological_endpoint": "Excluded",
        "mapping_reason": "",
    }

    if not object_name:
        base["mapping_reason"] = "Object prefix is not one of Cells, Cytoplasm, or Nuclei."
    elif len(channels) > 1 or (family == "Correlation" and len(channels) >= 2):
        base.update(
            biological_module="Cross-channel",
            mapping_status="Mapped",
            biological_endpoint="Eligible",
            mapping_reason="Multiple explicit Cell Painting channels or Correlation measurement; Cross-channel has priority.",
        )
    elif len(channels) == 1 and family in KNOWN_FAMILIES:
        base.update(
            biological_module=channels[0],
            mapping_status="Mapped",
            biological_endpoint="Eligible",
            mapping_reason="Single explicit Cell Painting channel mapped to its stain module.",
        )
    elif not channels and family in {"AreaShape", "Neighbors"}:
        base.update(
            biological_module="Shape",
            mapping_status="Mapped",
            biological_endpoint="Eligible",
            mapping_reason="Channel-independent AreaShape/Neighbors measurement mapped to Shape.",
        )
    elif family == "Correlation" and len(channels) < 2:
        base["mapping_reason"] = "Correlation feature lacks two explicit known channels; conservative Unmapped."
    elif family == "Others":
        base["mapping_reason"] = "Measurement family is not in the locked CellProfiler family vocabulary."
    else:
        base["mapping_reason"] = "Feature does not satisfy a locked channel/module rule."

    return base


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def count_records(records: list[dict[str, str | int]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for field, order in (
        ("biological_module", (*MODULE_ORDER, "Unmapped")),
        ("object", OBJECT_ORDER),
        ("measurement_family", FAMILY_ORDER),
        ("mapping_status", ("Mapped", "Unmapped")),
        ("biological_endpoint", ("Eligible", "Excluded")),
    ):
        counter = Counter(str(record[field]) for record in records)
        counts[field] = {key: int(counter.get(key, 0)) for key in order if counter.get(key, 0)}
        for key in sorted(counter):
            if key not in counts[field]:
                counts[field][key] = int(counter[key])
    return counts


def module_summary(records: list[dict[str, str | int]]) -> list[dict[str, Any]]:
    by_module: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for record in records:
        module = str(record["biological_module"])
        if module != "Unmapped":
            by_module[module].append(record)
    total_reliable = sum(len(items) for items in by_module.values())
    rows: list[dict[str, Any]] = []
    for module in (*MODULE_ORDER, "Unmapped"):
        items = by_module.get(module, [])
        rows.append(
            {
                "module": module,
                "feature_count": len(items),
                "feature_fraction_reliable": (len(items) / total_reliable if total_reliable else 0.0),
                "primary_module": "Yes" if module in MODULE_ORDER and len(items) >= 8 else "No",
                "objects": ";".join(sorted({str(item["object"]) for item in items})),
                "measurement_families": ";".join(sorted({str(item["measurement_family"]) for item in items})),
                "channels": ";".join(sorted({str(item["channel"]) for item in items if str(item["channel"])})),
                "feature_names": ";".join(str(item["feature_name"]) for item in items),
            }
        )
    return rows


def write_protocol(path: Path, feature_path: Path, source_inputs: list[dict[str, Any]], records: list[dict[str, str | int]], counts: dict[str, dict[str, int]], output_hashes: dict[str, str]) -> None:
    total = len(records)
    reliable = sum(str(row["mapping_status"]) == "Mapped" for row in records)
    module_rows = module_summary(records)
    module_table = markdown_table(
        ["Module", "Features", "Primary (>=8)", "Objects", "Families"],
        [[row["module"], row["feature_count"], row["primary_module"], row["objects"] or "-", row["measurement_families"] or "-"] for row in module_rows],
    )
    object_rows = [[key, value] for key, value in counts["object"].items()]
    family_rows = [[key, value] for key, value in counts["measurement_family"].items()]
    hash_rows = [[item["role"], item["path"], item["sha256"] or "MISSING"] for item in source_inputs]
    output_rows = [[key, value] for key, value in output_hashes.items()]
    text = f"""# cpg0004 CellProfiler feature annotation audit

Status: **PASS — frozen annotation dictionary generated**  
Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`  
Dataset: `cpg0004-LINCS`  
Feature source: `{display_path(feature_path)}`

## Objective

Freeze the biological annotation of the exact CP feature order used by the cpg0004 model. The source contains `{total}` features: `{reliable}` reliable CellProfiler measurements and one frozen `Batch_Number` metadata column. The metadata column remains represented in the dictionary for input fidelity, but is `Unmapped` and `Excluded` from every biological endpoint.

## Locked parsing rules

1. `Cells`, `Cytoplasm`, and `Nuclei` are recognized as object prefixes.
2. `AreaShape`, `Intensity`, `Texture`, `Granularity`, `RadialDistribution`, `Neighbors`, and `Correlation` are recognized measurement families; unknown families are `Others`.
3. Explicit multi-channel names and `Correlation` names with at least two known channels (`DNA`, `RNA`, `ER`, `Mito`, `AGP`) map to `Cross-channel` with highest biological priority.
4. A single explicit channel maps to its corresponding stain module.
5. Channel-independent `AreaShape` and `Neighbors` map to `Shape`.
6. Anything not satisfying a rule is conservatively `Unmapped`; no feature is force-assigned.

## Gate 0

- Expected frozen feature count: `{EXPECTED_TOTAL}`; observed: `{total}`.
- Expected reliable CellProfiler measurements: `{EXPECTED_RELIABLE}`; observed: `{reliable}`.
- Reliability fraction: `{reliable / total:.6f}` (`{reliable / total:.2%}`), threshold `>=80%`: **PASS**.
- Primary module threshold: `>=8` features. Modules below the threshold are supplementary only; no current mapped module is promoted by test results.
- Duplicate feature names: `{len(records) - len({str(row['feature_name']) for row in records})}`.

## Module counts

{module_table}

## Object counts

{markdown_table(["Object", "Features"], object_rows)}

## Measurement-family counts

{markdown_table(["Measurement family", "Features"], family_rows)}

## Input hashes

{markdown_table(["Role", "Path", "SHA256"], hash_rows)}

## Output hashes

{markdown_table(["Output", "SHA256"], output_rows)}

## Reproducibility and boundaries

- The source feature order and model subset are not changed, expanded, or selected using held-out outcomes.
- No MoA, target, pathway, or signaling interpretation is assigned by this audit.
- `Cross-channel` denotes a Cell Painting measurement involving multiple stains; it is not a pathway label.
- The generated `FEATURE_DICTIONARY.csv` is the authoritative feature-level mapping for downstream Morph-A/B/C analyses. `MODULE_MAPPING.csv` is a compact feature-to-module table, and `UNMAPPED_FEATURES.csv` records all excluded rows, including `Batch_Number`.
"""
    path.write_text(text, encoding="utf-8")


def run(feature_path: Path, output_dir: Path) -> dict[str, Any]:
    if not feature_path.exists():
        raise FileNotFoundError(f"Frozen feature list does not exist: {feature_path}")
    features = read_feature_names(feature_path)
    records = [parse_feature(name, index) for index, name in enumerate(features)]
    duplicate_names = [name for name, count in Counter(features).items() if count > 1]
    if len(features) != EXPECTED_TOTAL:
        raise ValueError(f"Frozen cpg0004 feature count mismatch: expected {EXPECTED_TOTAL}, observed {len(features)}")
    if duplicate_names:
        raise ValueError(f"Frozen feature list contains duplicate names: {duplicate_names[:5]}")
    metadata_rows = [row for row in records if str(row["feature_name"]) in FROZEN_METADATA_FEATURES]
    reliable_rows = [row for row in records if str(row["mapping_status"]) == "Mapped"]
    if len(metadata_rows) != 1 or len(reliable_rows) != EXPECTED_RELIABLE:
        raise ValueError(
            "Frozen feature audit mismatch: "
            f"metadata_rows={len(metadata_rows)}, reliable_rows={len(reliable_rows)}, expected reliable={EXPECTED_RELIABLE}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary_path = output_dir / "FEATURE_DICTIONARY.csv"
    feature_mapping_path = output_dir / "FEATURE_MAPPING.csv"
    module_path = output_dir / "MODULE_MAPPING.csv"
    unmapped_path = output_dir / "UNMAPPED_FEATURES.csv"
    write_csv(dictionary_path, DICTIONARY_FIELDS, records)
    # Generic downstream name used by the Morph-A/B/C runners.  Keep it
    # byte-for-byte equivalent in schema and row order to the authoritative
    # dictionary so no second mapping can drift from the frozen annotation.
    write_csv(feature_mapping_path, DICTIONARY_FIELDS, records)
    write_csv(
        module_path,
        MODULE_FIELDS,
        (
            {
                **{field: row[field] for field in MODULE_FIELDS if field in row},
                "primary_module": "Yes" if str(row["biological_module"]) in MODULE_ORDER and sum(str(item["biological_module"]) == str(row["biological_module"]) for item in records) >= 8 else "No",
            }
            for row in records
        ),
    )
    write_csv(unmapped_path, UNMAPPED_FIELDS, (row for row in records if str(row["mapping_status"]) != "Mapped"))

    counts = count_records(records)
    source_inputs = [
        input_record(feature_path, "frozen CP feature order"),
        input_record(feature_path.parent / "cp_plate_rows.npz", "frozen CP plate rows"),
        input_record(feature_path.parent / "cp_plate_manifest.csv", "frozen CP plate metadata"),
        input_record(feature_path.parent / "split_lock.json", "frozen compound split"),
    ]
    output_hashes = {
        path.name: sha256_file(path) for path in (dictionary_path, feature_mapping_path, module_path, unmapped_path)
    }
    config: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "cpg0004-LINCS",
        "stage": "Morphological Biological Validation / Phase 1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_source": display_path(feature_path),
        "output_directory": display_path(output_dir),
        "frozen_feature_count": EXPECTED_TOTAL,
        "observed_feature_count": len(records),
        "expected_reliable_cellprofiler_features": EXPECTED_RELIABLE,
        "observed_reliable_cellprofiler_features": len(reliable_rows),
        "frozen_metadata_features": sorted(FROZEN_METADATA_FEATURES),
        "metadata_policy": "retained_in_model_input_but_Unmapped_and_Excluded_from_all_biological_endpoints",
        "mapping_rules": {
            "objects": list(KNOWN_OBJECTS),
            "channels": list(KNOWN_CHANNELS),
            "measurement_families": list(KNOWN_FAMILIES),
            "module_order": list(MODULE_ORDER),
            "primary_module_min_features": 8,
            "priority": ["metadata_exclusion", "multi_channel_or_correlation", "single_channel", "channel_independent_shape", "conservative_unmapped"],
        },
        "gate0": {
            "reliability_threshold": 0.8,
            "reliability_fraction": len(reliable_rows) / len(records),
            "reliability_status": "PASS" if len(reliable_rows) / len(records) >= 0.8 else "SUPPORTIVE",
            "module_primary_threshold": 8,
            "primary_modules": [row["module"] for row in module_summary(records) if row["module"] in MODULE_ORDER and row["feature_count"] >= 8],
            "supplementary_modules": [row["module"] for row in module_summary(records) if row["module"] in MODULE_ORDER and row["feature_count"] < 8],
            "duplicate_feature_names": len(duplicate_names),
        },
        "counts": counts,
        "module_summary": module_summary(records),
        "input_hashes": source_inputs,
        "output_hashes": output_hashes,
        "outputs": [dictionary_path.name, feature_mapping_path.name, module_path.name, unmapped_path.name, "AUDIT.md", "PROTOCOL.md", "CONFIG.json"],
    }
    (output_dir / "CONFIG.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_protocol(output_dir / "PROTOCOL.md", feature_path, source_inputs, records, counts, output_hashes)
    # AUDIT.md is the user-facing companion; keep it separate from the concise
    # protocol so downstream scripts can consume either file without parsing a
    # large narrative.
    audit_path = output_dir / "AUDIT.md"
    write_protocol(audit_path, feature_path, source_inputs, records, counts, output_hashes)
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-list", type=Path, default=DEFAULT_FEATURES, help="Frozen feature_names.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Feature annotation audit output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = run(args.feature_list.resolve(), args.output_dir.resolve())
    except Exception as exc:  # command-line audit should fail loudly and cleanly
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "PASS", "output_dir": display_path(args.output_dir), "counts": config["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
