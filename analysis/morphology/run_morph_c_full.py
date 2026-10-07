#!/usr/bin/env python3
"""Frozen cpg0004 Morph-C: module-level six-dose trajectory recovery.

The script imports the already audited Bio-C slot builder.  It changes only
the evaluation projection: each of the seven feature modules is evaluated
separately and then macro-averaged without feature-count weighting.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEEDS = (3407, 42, 2025)
DOSES = ("0.04", "0.12", "0.37", "1.11", "3.33", "10")
MODULES = ("DNA", "RNA", "ER", "Mito", "AGP", "Shape", "Cross-channel")
METHODS = ("raw", "teacher", "posterior")
LABELS = {"raw": "M0_1R_raw", "teacher": "M1_frozen_teacher", "posterior": "M2_frozen_GE_posterior"}
EXPECTED_M1, EXPECTED_M2 = 260, 258


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(("MorphC|" + label).encode()).digest()[:8], "little") % (2**32 - 1)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_bioc", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    base = here.parent / "external_validation" / "lincs_cpg0004"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, default=(Path(__file__).resolve().parents[2] / "analysis/biological_applications/dose_response/run_dose_response.py"))
    parser.add_argument("--data", type=Path, default=base / "data_preparation" / "artifact" / "cp_plate_rows.npz")
    parser.add_argument("--p0-root", type=Path, default=base / "virtual_prior" / "results" / "1r_all")
    parser.add_argument("--ge-root", type=Path, default=base / "single_repeat_expression_evidence" / "results" / "1r_all")
    parser.add_argument("--pair-root", type=Path, default=base / "cell_painting_repeat_benchmark" / "results" / "1r_all")
    parser.add_argument("--feature-mapping", type=Path, default=here / "feature_annotation_audit" / "FEATURE_MAPPING.csv")
    parser.add_argument("--outdir", type=Path, default=here / "module_dose_trajectories")
    parser.add_argument("--bootstrap-rounds", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_modules(path: Path, dim: int) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    need = {"feature_index", "biological_module", "biological_endpoint"}
    if not need.issubset(frame.columns) or len(frame) != dim:
        raise ValueError("Feature mapping is not aligned to frozen profile dimensions")
    active = frame[frame.biological_endpoint.astype(str).eq("Eligible")]
    out = {name: active.loc[active.biological_module.eq(name), "feature_index"].to_numpy(dtype=np.int64) for name in MODULES}
    if any(len(v) < 8 for v in out.values()):
        raise ValueError(f"Module map does not pass the >=8 feature gate: { {k:len(v) for k,v in out.items()} }")
    if int(active.feature_index.max()) >= dim or "Batch_Number" in set(active.get("feature_name", [])):
        raise ValueError("Biological mapping includes a frozen metadata feature")
    return out


def trajectory(M, profiles: list[np.ndarray], references: list[np.ndarray], indices: np.ndarray) -> float:
    distances, ref_distances = [], []
    for i, j in combinations(range(len(DOSES)), 2):
        left = M.pcc(profiles[i][indices], profiles[j][indices])
        right = M.pcc(references[i][indices], references[j][indices])
        if left is None or right is None:
            return math.nan
        distances.append(1.0 - float(left)); ref_distances.append(1.0 - float(right))
    value = M.spearman(np.asarray(distances), np.asarray(ref_distances))
    return float(value) if value is not None and np.isfinite(value) else math.nan


def compound_mean(frame: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for key, part in frame.groupby(["seed", "compound", "module", "method"], sort=True):
        valid = part.trajectory_tc.dropna()
        groups.append({"seed": key[0], "compound": key[1], "module": key[2], "method": key[3], "rotation_count": len(part), "valid_rotation_count": len(valid), "trajectory_tc": float(valid.mean()) if len(part) == 2 and len(valid) == 2 else math.nan})
    return pd.DataFrame(groups)


def bootstrap(values: np.ndarray, label: str, rounds: int) -> tuple[float, float, float, int]:
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return math.nan, math.nan, math.nan, -1
    seed = stable_seed(label); rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(rounds, len(values)), endpoint=False)
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975)), seed


def contrasts(frame: pd.DataFrame, rounds: int) -> pd.DataFrame:
    records = []
    for scope, scopes in (("module", MODULES), ("macro_module", ("Macro",))):
        source = frame if scope == "module" else frame.groupby(["seed", "compound", "method"], as_index=False).trajectory_tc.mean().assign(module="Macro")
        for module in scopes:
            for comparison, left, right in (("teacher_minus_raw", "teacher", "raw"), ("posterior_minus_teacher", "posterior", "teacher")):
                diffs: dict[int, pd.Series] = {}
                for seed in SEEDS:
                    l = source[(source.seed.eq(seed)) & (source.module.eq(module)) & (source.method.eq(left))].set_index("compound").trajectory_tc
                    r = source[(source.seed.eq(seed)) & (source.module.eq(module)) & (source.method.eq(right))].set_index("compound").trajectory_tc
                    both = sorted(set(l.index) & set(r.index))
                    diffs[seed] = pd.Series({c: float(l[c] - r[c]) for c in both if np.isfinite(l[c]) and np.isfinite(r[c])}, dtype=float)
                pre = sorted(set.intersection(*(set(x.index) for x in diffs.values())))
                pooled = np.asarray([np.mean([diffs[s][c] for s in SEEDS]) for c in pre], dtype=float)
                point, low, high, bs = bootstrap(pooled, f"{scope}|{module}|{comparison}", rounds)
                records.append({"seed": "mean", "analysis": scope, "module": module, "comparison": comparison, "endpoint": "trajectory_tc", "seed3407": float(diffs[3407].mean()) if len(diffs[3407]) else math.nan, "seed42": float(diffs[42].mean()) if len(diffs[42]) else math.nan, "seed2025": float(diffs[2025].mean()) if len(diffs[2025]) else math.nan, "n_compounds": len(pooled), "point": point, "ci_low": low, "ci_high": high, "rounds": rounds, "bootstrap_unit": "compound", "bootstrap_seed": bs})
    return pd.DataFrame(records)


def gate(row: pd.Series) -> str:
    directions = [row.seed3407, row.seed42, row.seed2025]
    return "GO" if all(np.isfinite(x) and x > 0 for x in directions) and np.isfinite(row.ci_low) and row.ci_low > 0 else "NO-GO"


def write_report(out: Path, config: dict[str, Any], primary: pd.DataFrame, decisions: pd.DataFrame) -> None:
    text = ["# Results — Morph-C module dose trajectory", "", "## 1. Objective", "Evaluate frozen-method recovery of independent six-dose morphology trajectories within predefined Cell Painting modules.", "", "## 2. Frozen methods", "M0 raw, M1 frozen teacher, and M2 frozen GE-updated posterior; no refitting or selection used test recovery.", "", "## 3. Dataset", "cpg0004-LINCS test conditions with six locked doses and five physical CP repeats per dose.", "", "## 4. Feature mapping", "241 biological CellProfiler features are grouped into seven >=8-feature modules; Batch_Number is excluded.", "", "## 5. Eligibility", f"M1 common set {config['counts']['m1_compounds']} compounds; M2 GE-aligned set {config['counts']['m2_compounds']} compounds.", "", "## 6. Support/reference isolation", "Two recorded frozen support rotations are scored separately; every reference averages the other four physical rows.", "", "## 7. Primary endpoint", "Module TC is Spearman correlation of the 15 six-dose 1-PCC distances; MacroTC is an unweighted mean of module TCs.", "", "## 8. Secondary endpoints", "Per-module TC is reported alongside MacroTC. No dose-label endpoint is selected here.", "", "## 9. Negative controls", "No Morph-C negative control was preregistered; NEGATIVE_CONTROLS.csv records this explicitly.", "", "## 10. Sample counts", json.dumps(config['counts'], sort_keys=True), "", "## 11. Seed-level results", "METRICS_BY_SEED.csv retains the locked three-seed summaries.", "", "## 12. Compound-paired bootstrap CI", primary.to_csv(index=False), "", "## 13. GO / SUPPORTIVE / NO-GO", decisions.to_csv(index=False), "", "## 14. Biological interpretation", "A positive gate supports improved recovery of repeat-defined trajectory geometry, not a pathway or repeat-replacement claim.", "", "## 15. What the result does NOT prove", "It does not establish a mechanism, dose potency, or equivalence to collecting a second physical repeat.", "", "## 16. Limitations", "Only two frozen support rotations are available; undefined PCC/Spearman values are unscorable and not repaired with an epsilon. See UNSCORABLE_COUNTS.csv."]
    (out / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args(); out = args.outdir.resolve()
    if args.bootstrap_rounds <= 0: raise ValueError("bootstrap rounds must be positive")
    if out.exists() and any(out.iterdir()) and not args.force: raise FileExistsError(f"Use --force to overwrite {out}")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    M = load_module(args.module)
    artifact = M.load_artifact(args.data)
    modules = read_modules(args.feature_mapping, artifact.feature_dim)
    mapping_errors: list[str] = []
    args.lambda_by_seed = {3407: .10, 42: .10, 2025: .10}; args.beta_by_seed = {3407: .10, 42: .15, 2025: .15}
    p0, ge, manifests, _ = M.load_pair_and_prediction_bundles(args, args.lambda_by_seed, args.beta_by_seed, mapping_errors)
    slots, exclusions, _, audit = M.build_slots(artifact, p0, ge, manifests, mapping_errors)
    m1, m2 = M.trim_to_common_sets(slots)
    if len(m1) != EXPECTED_M1 or len(m2) != EXPECTED_M2: raise RuntimeError(f"Unexpected frozen common sets: M1={len(m1)}, M2={len(m2)}")
    slot_records: list[dict[str, Any]] = []; eligible: list[dict[str, Any]] = []
    for compound in sorted(m1):
        eligible.append({"compound": compound, "split": "test", "six_dose": 1, "strict_repeats": 5, "frozen_support_rotations": 2, "m1_eligible": 1, "m2_eligible": int(compound in m2)})
        for seed in SEEDS:
            for rotation in range(2):
                for name, indices in modules.items():
                    for method in METHODS:
                        if method == "posterior" and compound not in m2: continue
                        by_dose = slots[seed][compound][rotation]
                        profiles = [by_dose[d].profiles[method] for d in DOSES]
                        references = [by_dose[d].reference for d in DOSES]
                        value = trajectory(M, profiles, references, indices)
                        slot_records.append({"seed": seed, "compound": compound, "rotation": rotation, "method": method, "method_label": LABELS[method], "module": name, "feature_count": len(indices), "trajectory_tc": value, "scorable": int(np.isfinite(value)), "dose_count": 6})
    slot_frame = pd.DataFrame(slot_records)
    comp = compound_mean(slot_frame)
    unscore = comp.assign(unscorable=~np.isfinite(comp.trajectory_tc)).groupby(["seed", "module", "method"], as_index=False).agg(compound_units=("compound", "size"), unscorable_units=("unscorable", "sum"))
    seed_table = comp.groupby(["seed", "module", "method"], as_index=False).trajectory_tc.mean()
    macro_seed = comp.groupby(["seed", "compound", "method"], as_index=False).trajectory_tc.mean().groupby(["seed", "method"], as_index=False).trajectory_tc.mean().assign(module="Macro")
    metrics_seed = pd.concat([seed_table, macro_seed], ignore_index=True)
    contrast = contrasts(comp, args.bootstrap_rounds)
    primary = contrast[(contrast.analysis.eq("macro_module")) & (contrast.endpoint.eq("trajectory_tc"))].copy(); primary["decision"] = primary.apply(gate, axis=1)
    decisions = primary[["comparison", "seed3407", "seed42", "seed2025", "n_compounds", "point", "ci_low", "ci_high", "decision"]].copy()
    input_paths = [args.data, args.feature_mapping, *(args.p0_root / f"seed{s}" / "test_predictions.npz" for s in SEEDS), *(args.ge_root / f"seed{s}" / "test_predictions.npz" for s in SEEDS), *(args.pair_root / f"seed{s}" / "test_pair_manifest.csv" for s in SEEDS)]
    config = {"version": "cpg0004-LINCS-Morph-C-2026-08-30", "dataset": "cpg0004-LINCS", "seeds": list(SEEDS), "doses": list(DOSES), "methods": LABELS, "feature_modules": {k: len(v) for k,v in modules.items()}, "excluded_metadata": ["Batch_Number"], "support_reference": "two frozen rotations; mean of four physical reference repeats excluding support", "endpoint": "Spearman of 15 pairwise six-dose 1-PCC distances", "macro_weighting": "unweighted modules", "bootstrap": {"rounds": args.bootstrap_rounds, "unit": "compound", "paired": True, "ci": "95% percentile"}, "unscorable": "zero-variance/no-PCC values not imputed", "counts": {"m1_compounds": len(m1), "m2_compounds": len(m2), "m1_condition_rotation_rows": len(m1)*2*6, "m2_condition_rotation_rows": len(m2)*2*6, "slot_module_records": len(slot_frame)}, "source_slot_audit": audit, "mapping_error_count": len(mapping_errors), "input_hashes": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in input_paths]}
    shutil.copyfile(args.feature_mapping, out / "FEATURE_MAPPING.csv")
    (out / "CONFIG.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (out / "PROTOCOL.md").write_text("# Morph-C protocol\n\nFrozen Bio-C slots only; seven pre-audited modules; two supports and four-repeat independent reference; MacroTC is unweighted. No model or test-dependent parameter is fitted.\n", encoding="utf-8")
    pd.DataFrame(eligible).to_csv(out / "ELIGIBLE_COMPOUNDS.csv", index=False); pd.DataFrame(exclusions + [{"scope":"audit", "reason":e} for e in mapping_errors]).to_csv(out / "EXCLUSIONS.csv", index=False)
    comp.to_csv(out / "METRICS_BY_COMPOUND.csv", index=False); seed_table.to_csv(out / "METRICS_BY_MODULE.csv", index=False); metrics_seed.to_csv(out / "METRICS_BY_SEED.csv", index=False); contrast.to_csv(out / "PAIRED_CONTRASTS.csv", index=False); contrast.to_csv(out / "BOOTSTRAP_CI.csv", index=False); unscore.to_csv(out / "UNSCORABLE_COUNTS.csv", index=False)
    pd.DataFrame([{"control": "not_applicable", "reason": "No Morph-C negative control was specified in the locked plan."}]).to_csv(out / "NEGATIVE_CONTROLS.csv", index=False)
    decisions.to_csv(out / "DECISION.csv", index=False); (out / "DECISION.md").write_text("# Morph-C decision\n\n```csv\n" + decisions.to_csv(index=False) + "```\n", encoding="utf-8")
    slot_frame.to_csv(out / "SLOT_METRICS.csv", index=False); write_report(out, config, primary, decisions)
    (out / "figures" / "summary.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg" width="1150" height="250"><rect width="100%" height="100%" fill="white"/><text x="20" y="35" font-size="20" font-family="Arial">Morph-C MacroTC paired contrasts</text><text x="20" y="65" font-size="12" font-family="monospace">' + primary.to_csv(index=False).replace("&","&amp;").replace("<","&lt;") + '</text></svg>', encoding="utf-8")
    print(json.dumps({"outdir": str(out), "m1":len(m1), "m2":len(m2), "decisions":decisions.to_dict("records"), "mapping_errors":mapping_errors}, indent=2, default=str))


if __name__ == "__main__": main()
