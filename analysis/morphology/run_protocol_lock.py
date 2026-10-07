#!/usr/bin/env python3
"""Write the immutable input/analysis lock for cpg0004 morphological biology."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "external_validation" / "lincs_cpg0004"
OUTPUT = ROOT / "protocol"
SEEDS = (3407, 42, 2025)
DOSES = ("0.04", "0.12", "0.37", "1.11", "3.33", "10")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "cp_plate_rows": BASE / "data_preparation" / "artifact" / "cp_plate_rows.npz",
        "cp_plate_manifest": BASE / "data_preparation" / "artifact" / "cp_plate_manifest.csv",
        "split_lock": BASE / "data_preparation" / "artifact" / "split_lock.json",
        "feature_names": BASE / "data_preparation" / "artifact" / "feature_names.csv",
        "feature_mapping": ROOT / "feature_annotation_audit" / "FEATURE_MAPPING.csv",
    }
    for seed in SEEDS:
        paths[f"p0_seed{seed}"] = BASE / "virtual_prior" / "results" / "1r_all" / f"seed{seed}" / "test_predictions.npz"
        paths[f"ge_seed{seed}"] = BASE / "single_repeat_expression_evidence" / "results" / "1r_all" / f"seed{seed}" / "test_predictions.npz"
        paths[f"pair_manifest_seed{seed}"] = BASE / "cell_painting_repeat_benchmark" / "results" / "1r_all" / f"seed{seed}" / "test_pair_manifest.csv"
    entries = []
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Locked input missing: {role}: {path}")
        entries.append({"role": role, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest(path)})
    config = {
        "lock_id": "cpg0004-LINCS-morphological-biology-2026-08-30",
        "dataset": "cpg0004-LINCS",
        "frozen_methods": {"M0": "1R raw CP", "M1": "frozen reproducible-effect teacher", "M2": "frozen validated GE-updated CP posterior"},
        "seeds": list(SEEDS),
        "doses": list(DOSES),
        "feature_policy": {"frozen_dimensions": 242, "biological_dimensions": 241, "excluded_endpoint_feature": "Batch_Number", "mapping_gate": {"minimum_reliable_mapping_fraction": 0.80, "minimum_features_per_primary_module": 8}},
        "modules": ["DNA", "RNA", "ER", "Mito", "AGP", "Shape", "Cross-channel"],
        "support_reference": {"frozen_support_rotations": 2, "strict_repeats_per_condition": 5, "independent_reference": "mean of the four physical rows excluding the selected support"},
        "statistics": {"bootstrap_rounds": 10000, "unit": "compound", "paired": True, "ci": "percentile 95%", "go_rule": "all three locked seed directions > 0 and pooled CI lower bound > 0", "unscorable_policy": "zero-variance/no-PCC entries are reported and not imputed"},
        "morph_a_foreign_null": {"definition": "method-shared raw support-to-held matched foreign compound", "match": "exact dose + held/support plate slots + well row + test split", "draws": 32, "physical_well_status": "unavailable"},
        "morph_d_selection": {"basis": "PCA", "candidate_k": [8, 16, 24, 32], "selection": "minimum stable K within one validation reconstruction standard error of the best", "stability": "bootstrap train-condition PCA mean squared canonical correlation; lower 2.5% quantile >= 0.90", "basis_refit_after_selection": "train plus validation references only", "test_used_for_selection": False},
        "input_entries": entries,
    }
    (OUTPUT / "CONFIG.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    with (OUTPUT / "INPUT_HASHES.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader(); writer.writerows(entries)
    protocol = """# Morphological-biology protocol lock

- M0/M1/M2 are frozen arrays only; no model, lambda, beta, split, dose, support pairing, feature membership, or test-dependent selection is changed here.
- Biological endpoints use 241 annotated CellProfiler dimensions. `Batch_Number` remains in frozen arrays solely for dimensional fidelity and is excluded from all biological endpoint calculations.
- Primary uncertainty is a 10,000-round paired compound bootstrap. A GO requires positive effects in every locked seed and a positive pooled 95% CI lower bound.
- Any zero-variance PCC or undefined Spearman is unscorable: it is counted, reported, and never repaired with an epsilon.
- Morph-D may run because Morph-A or Morph-B has a preregistered GO. Its program basis and K are selected without test profiles according to `CONFIG.json`.
"""
    (OUTPUT / "PROTOCOL.md").write_text(protocol, encoding="utf-8")
    print(json.dumps({"outdir": str(OUTPUT), "locked_inputs": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
