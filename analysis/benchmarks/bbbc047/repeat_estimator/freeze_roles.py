#!/usr/bin/env python3
"""Freeze dose-aware BBBC047 CP support/held/foreign identities.

This preparation stage deliberately reads only ``smiles``, ``dose`` and
``plate`` from every split NPZ.  In particular, it never indexes ``delta`` in
the test archive and cannot calculate a profile, prediction or test metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "BBBC047-repeat-estimator-benchmark-v1-2026-09-16"
ROLE_SEED = 3407
BUDGETS = (1, 2, 3)
FOREIGN_DRAWS = 32
EXPECTED_COMPOUNDS = {"train": 12175, "valid": 4059, "test": 4060}
DEFAULT_ROWS = Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0")
DEFAULT_LOCK = Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json")


def stable_int(label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{ROLE_SEED}|{label}".encode("utf-8")).digest()[:8], "big")


def decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="strict")
    return str(value)


def dose_key(value: Any) -> str:
    """Use the project-wide BBBC047 nominal-dose canonicalization."""
    text = decode(value).strip()
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-root", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--split-lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("PROTOCOL.md"))
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def read_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read identity metadata only; do not access the ``delta`` array."""
    with np.load(path, allow_pickle=False) as payload:
        required = {"smiles", "dose", "plate", "delta"}
        missing = required - set(payload.files)
        if missing:
            raise RuntimeError(f"missing fields in {path}: {sorted(missing)}")
        smiles = np.asarray(payload["smiles"])
        dose = np.asarray(payload["dose"])
        plate = np.asarray(payload["plate"])
        if not (len(smiles) == len(dose) == len(plate)):
            raise RuntimeError(f"metadata length mismatch in {path}")
        # Do not read payload["delta"] here.  It remains an unopened NPZ member.
    return smiles, dose, plate


def split_metadata(rows_root: Path, split: str) -> tuple[dict[tuple[str, str], tuple[str, ...]], dict[str, Any]]:
    path = rows_root / f"{split}_cp_plate_rows.npz"
    smiles, dose, plate = read_metadata(path)
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    raw_rows: dict[tuple[str, str, str], int] = defaultdict(int)
    for compound_raw, dose_raw, plate_raw in zip(smiles, dose, plate):
        compound, canonical_dose, plate_id = decode(compound_raw), dose_key(dose_raw), decode(plate_raw)
        if not compound or not canonical_dose or not plate_id:
            raise RuntimeError(f"empty condition identity in {path}")
        grouped[(compound, canonical_dose)].add(plate_id)
        raw_rows[(compound, canonical_dose, plate_id)] += 1
    conditions = {key: tuple(sorted(value)) for key, value in grouped.items()}
    duplicate_plate_rows = sum(1 for count in raw_rows.values() if count > 1)
    return conditions, {
        "path": str(path),
        "sha256": sha256_file(path),
        "metadata_rows": int(len(smiles)),
        "n_compounds": int(len({key[0] for key in conditions})),
        "n_conditions": int(len(conditions)),
        "condition_plate_rows_with_multiple_technical_rows": int(duplicate_plate_rows),
        "test_profile_values_loaded": False,
    }


def assert_split_lock(conditions: dict[str, dict[tuple[str, str], tuple[str, ...]]], lock_path: Path) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    compound_sets = {split: {compound for compound, _ in items} for split, items in conditions.items()}
    for split, expected_n in EXPECTED_COMPOUNDS.items():
        expected = {str(value) for value in lock[f"{split}_smiles"]}
        observed = compound_sets[split]
        if len(observed) != expected_n or observed != expected:
            raise RuntimeError(f"official split mismatch for {split}: observed={len(observed)}, expected={len(expected)}")
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = compound_sets[left] & compound_sets[right]
        if overlap:
            raise RuntimeError(f"compound split leakage {left}/{right}: {len(overlap)}")


def ordered_plates(split: str, budget: int, compound: str, dose: str, plates: tuple[str, ...]) -> tuple[str, ...]:
    label = f"BBBC047|CP|{split}|b{budget}|{compound}|{dose}"
    return tuple(sorted(plates, key=lambda plate: (stable_int(f"plate|{label}|{plate}"), plate)))


def foreign_draws(
    split: str,
    budget: int,
    compound: str,
    dose: str,
    required_slots: tuple[str, ...],
    held_plate: str,
    conditions: dict[tuple[str, str], tuple[str, ...]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required = set(required_slots)
    # Match the actual left/right plate slots used by the query.  A donor may
    # have further physical plates: requiring its *complete* plate set to be
    # identical is an unintended eligibility filter, not nuisance matching.
    candidates = [
        key for key, donor_plates in conditions.items()
        if key[0] != compound and key[1] == dose and required.issubset(set(donor_plates))
    ]
    if not candidates:
        return (), ()
    condition_ids: list[str] = []
    held_ids: list[str] = []
    base = f"foreign-draw|{VERSION}|{split}|b{budget}|{compound}|{dose}|{held_plate}|{'|'.join(required_slots)}"
    for draw in range(FOREIGN_DRAWS):
        donor = candidates[stable_int(f"{base}|{draw}") % len(candidates)]
        condition_ids.append(f"{donor[0]}::{donor[1]}")
        held_ids.append(held_plate)
    return tuple(condition_ids), tuple(held_ids)


def build_manifest(conditions_by_split: dict[str, dict[tuple[str, str], tuple[str, ...]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for split, conditions in conditions_by_split.items():
        for budget in BUDGETS:
            counters = {"eligible_conditions": 0, "eligible_compounds": 0, "foreign_conditions": 0, "foreign_compounds": 0}
            eligible_compounds: set[str] = set()
            foreign_compounds: set[str] = set()
            for (compound, dose), all_plates in sorted(conditions.items()):
                if len(all_plates) < budget + 1:
                    continue
                ordered = ordered_plates(split, budget, compound, dose, all_plates)
                support, held = ordered[:budget], ordered[budget]
                foreign_conditions, foreign_held = foreign_draws(split, budget, compound, dose, tuple((*support, held)), held, conditions)
                counters["eligible_conditions"] += 1
                eligible_compounds.add(compound)
                foreign_ok = len(foreign_conditions) == FOREIGN_DRAWS
                if foreign_ok:
                    counters["foreign_conditions"] += 1
                    foreign_compounds.add(compound)
                rows.append({
                    "version": VERSION,
                    "dataset": "BBBC047",
                    "modality": "CP",
                    "split": split,
                    "budget": budget,
                    "role_seed": ROLE_SEED,
                    "compound_id": compound,
                    "dose": dose,
                    "condition_id": f"{compound}::{dose}",
                    "all_plate_ids": "|".join(all_plates),
                    "support_plate_ids": "|".join(support),
                    "held_plate_id": held,
                    "support_held_disjoint": True,
                    "foreign_draw_count": len(foreign_conditions),
                    "foreign_condition_ids": "|".join(foreign_conditions),
                    "foreign_held_plate_ids": "|".join(foreign_held),
                    "foreign_ok": foreign_ok,
                })
            counters["eligible_compounds"] = len(eligible_compounds)
            counters["foreign_compounds"] = len(foreign_compounds)
            counts[f"{split}_{budget}R"] = counters
    if not rows:
        raise RuntimeError("empty role manifest")
    return rows, counts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {args.outdir}")
    if not args.protocol.is_file() or not args.split_lock.is_file():
        raise FileNotFoundError("protocol and split lock are required")
    conditions: dict[str, dict[tuple[str, str], tuple[str, ...]]] = {}
    source: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        conditions[split], source[split] = split_metadata(args.rows_root, split)
    assert_split_lock(conditions, args.split_lock)
    manifest, counts = build_manifest(conditions)
    args.outdir.mkdir(parents=True)
    manifest_path = args.outdir / "FROZEN_ROLE_MANIFEST.csv"
    write_csv(manifest_path, manifest)
    audit = {
        "version": VERSION,
        "stage": "freeze_roles",
        "protocol_sha256": sha256_file(args.protocol),
        "split_lock_sha256": sha256_file(args.split_lock),
        "role_seed": ROLE_SEED,
        "budgets": list(BUDGETS),
        "foreign_draws": FOREIGN_DRAWS,
        "foreign_rule": "same canonical dose, held left-slot and all support right-slots; donor may have additional plates; different compound; 32 deterministic draws with replacement",
        "physical_unit": "(compound, dose, plate); technical rows sharing that key remain one physical plate response in later profile loading",
        "source": source,
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "rows": len(manifest)},
        "eligibility": counts,
        "test_metadata_opened": True,
        "test_profile_values_loaded": False,
        "test_predictions_loaded": False,
        "test_metrics_computed": False,
        "test_used_for_fit_or_selection": False,
        "status": "PASS",
    }
    (args.outdir / "ROLE_FREEZE_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": audit["stage"], "manifest_rows": len(manifest), "eligibility": counts, "test_profile_values_loaded": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
