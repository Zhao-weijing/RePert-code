#!/usr/bin/env python3
"""Frozen cpg0004 Morph-A/B evaluator.

Evaluates compartment/stain modules and dominant Cell Painting features using
only frozen M0/M1/M2 arrays.  It deliberately keeps the original cpg0004
method-shared foreign raw support-to-held null: the null is a structural
baseline, while the method-specific quantity is same-condition recovery.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEEDS = (3407, 42, 2025)
MODULE_ORDER = ("DNA", "RNA", "ER", "Mito", "AGP", "Shape", "Cross-channel")
OBJECT_ORDER = ("Cells", "Cytoplasm", "Nuclei")
METHODS = ("raw", "teacher", "posterior_ge")
LABELS = {"raw": "M0_1R_raw", "teacher": "M1_reproducible_effect_teacher", "posterior_ge": "M2_GE_updated_posterior"}
K_VALUES = (10, 25, 50)
PRIMARY_K = 25
NULL_ROUNDS = 32
BOOTSTRAP_ROUNDS = 10_000


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    experiments = here.parent
    base = experiments / "external_validation" / "lincs_cpg0004"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, default=(Path(__file__).resolve().parents[2] / "analysis/biological_applications/mechanism_target_retrieval/run_moa_target_retrieval.py"))
    parser.add_argument("--data", type=Path, default=base / "data_preparation" / "artifact" / "cp_plate_rows.npz")
    parser.add_argument("--manifest", type=Path, default=base / "data_preparation" / "artifact" / "cp_plate_manifest.csv")
    parser.add_argument("--split-lock", type=Path, default=base / "data_preparation" / "artifact" / "split_lock.json")
    parser.add_argument("--annotations", type=Path, default=experiments / "pathway_validation" / "resources" / "official_repurposing_info_external_moa_map_resolved.tsv")
    parser.add_argument("--smiles", type=Path, default=base / "virtual_prior" / "compound_smiles.csv")
    parser.add_argument("--feature-mapping", type=Path, default=here / "feature_annotation_audit" / "FEATURE_MAPPING.csv")
    parser.add_argument("--virtual-root", type=Path, default=base / "virtual_prior" / "results" / "1r_all")
    parser.add_argument("--ge-root", type=Path, default=base / "single_repeat_expression_evidence" / "results" / "1r_all")
    parser.add_argument("--pair-root", type=Path, default=base / "cell_painting_repeat_benchmark" / "results" / "1r_all")
    parser.add_argument("--out-root", type=Path, default=here)
    parser.add_argument("--bootstrap-rounds", type=int, default=BOOTSTRAP_ROUNDS)
    parser.add_argument("--smoke", action="store_true", help="Use 200 bootstrap resamples only; never report as final evidence.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_moa_morph", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen evaluator: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(seed: int, label: str) -> int:
    h = hashlib.sha256(label.encode("utf-8")).digest()
    return (int(seed) + int.from_bytes(h[:8], "little")) % (2**32 - 1)


def pcc(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return math.nan
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 and np.isfinite(denominator) else math.nan


def fisher(value: float) -> float:
    return float(np.arctanh(np.clip(float(value), -0.999999, 0.999999))) if np.isfinite(value) else math.nan


def well_row(value: str) -> str:
    text = str(value).split(";", 1)[0]
    output = ""
    for char in text:
        if char.isalpha():
            output += char
        else:
            break
    return output


def read_mapping(path: Path, dim: int) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"feature_index", "feature_name", "object", "biological_module", "biological_endpoint"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature mapping lacks columns: {sorted(missing)}")
    frame["feature_index"] = frame.feature_index.astype(int)
    if len(frame) != dim or set(frame.feature_index) != set(range(dim)):
        raise ValueError("Feature mapping is not a one-to-one alignment to the frozen feature dimension")
    active = frame[frame.biological_endpoint.astype(str).eq("Eligible")].copy()
    modules = {name: np.asarray(active.loc[active.biological_module.eq(name), "feature_index"], dtype=np.int64) for name in MODULE_ORDER}
    modules = {name: ix for name, ix in modules.items() if len(ix) >= 8}
    if set(modules) != set(MODULE_ORDER):
        raise ValueError(f"Primary module gate failed or mapping changed: { {k: len(v) for k,v in modules.items()} }")
    objects = {name: np.asarray(active.loc[active.object.eq(name), "feature_index"], dtype=np.int64) for name in OBJECT_ORDER}
    if any(len(ix) < 8 for ix in objects.values()):
        raise ValueError("Object sensitivity mapping has fewer than eight features")
    return frame, modules, objects


def robust_standardizer(rows: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    train = np.flatnonzero(rows["split"].astype(str) == "train")
    values = np.asarray(rows["delta"][train], dtype=np.float64)
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    fallback = np.std(values, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = fallback[~np.isfinite(scale) | (scale < 1e-6)]
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return center, scale


def build_slot_map(rows: dict[str, np.ndarray]) -> dict[tuple[str, str, str], dict[str, int]]:
    result: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    for index, (compound, dose, plate, well) in enumerate(zip(rows["compound_id"].astype(str), rows["dose"].astype(str), rows["plate"].astype(str), rows["well"].astype(str))):
        key = (dose, plate, well_row(well))
        if compound in result[key]:
            raise RuntimeError(f"Duplicate compound in structural slot: {key}|{compound}")
        result[key][compound] = int(index)
    return result


def matched_foreign_rows(rows: dict[str, np.ndarray], slot_map: dict[tuple[str, str, str], dict[str, int]], compound: str, dose: str, support_row: int, held_row: int) -> list[tuple[int, int]]:
    slots = []
    for row in (held_row, support_row):
        slots.append((str(dose), str(rows["plate"][row]), well_row(str(rows["well"][row]))))
    candidates: set[str] | None = None
    for slot in slots:
        current = set(slot_map.get(slot, {}))
        candidates = current if candidates is None else candidates & current
    eligible = [] if candidates is None else sorted(c for c in candidates if c != compound and str(rows["split"][slot_map[slots[0]][c]]) == "test")
    return [(slot_map[slots[1]][c], slot_map[slots[0]][c]) for c in eligible]


def sampled_foreign(rows: dict[str, np.ndarray], donors: list[tuple[int, int]], seed: int, compound: str, dose: str, held_row: int) -> list[tuple[int, int]]:
    if not donors:
        return []
    label = f"null|{compound}|{dose}|{rows['plate'][held_row]}|{str(rows['well'][held_row]).split(';',1)[0]}"
    rng = np.random.default_rng(stable_seed(seed, label))
    positions = rng.integers(0, len(donors), size=NULL_ROUNDS, endpoint=False)
    return [donors[int(i)] for i in positions]


def relative_rmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    denom = float(np.sqrt(np.mean(np.square(reference))))
    if not np.isfinite(denom) or denom <= 0:
        return math.nan
    value = float(np.sqrt(np.mean(np.square(prediction - reference))) / denom)
    return value if np.isfinite(value) else math.nan


def top_indices(values: np.ndarray, k: int) -> np.ndarray:
    score = np.abs(np.asarray(values, dtype=np.float64))
    return np.lexsort((np.arange(len(score)), -score))[:k]


def feature_scores(prediction: np.ndarray, reference: np.ndarray, center: np.ndarray, scale: np.ndarray, biological: np.ndarray, k: int) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    pred = (prediction[biological] - center[biological]) / scale[biological]
    ref = (reference[biological] - center[biological]) / scale[biological]
    ref_rank = top_indices(ref, k)
    pred_rank = top_indices(pred, k)
    overlap = float(len(set(ref_rank.tolist()) & set(pred_rank.tolist())) / k)
    sign = float(np.mean(np.sign(pred[ref_rank]) == np.sign(ref[ref_rank])))
    return overlap, sign, pcc(pred[ref_rank], ref[ref_rank]), ref_rank, pred_rank


def js_module(ref_rank: np.ndarray, pred_rank: np.ndarray, local_to_global: np.ndarray, feature_to_module: dict[int, str]) -> float:
    names = ("DNA", "RNA", "ER", "Mito", "AGP", "Shape")
    def distribution(rank: np.ndarray) -> np.ndarray:
        vals = np.zeros(len(names), dtype=np.float64)
        for local in rank:
            name = feature_to_module.get(int(local_to_global[int(local)]), "")
            if name in names:
                vals[names.index(name)] += 1.0
        return vals / vals.sum() if vals.sum() > 0 else vals
    left, right = distribution(ref_rank), distribution(pred_rank)
    if left.sum() == 0 or right.sum() == 0:
        return math.nan
    mean = (left + right) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_left = np.sum(np.where(left > 0, left * np.log2(left / mean), 0.0))
        kl_right = np.sum(np.where(right > 0, right * np.log2(right / mean), 0.0))
    return float(0.5 * (kl_left + kl_right))


def bootstrap(seed: int, label: str, values: np.ndarray, rounds: int) -> tuple[float, float, float, int]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        return math.nan, math.nan, math.nan, -1
    rng_seed = stable_seed(seed, label)
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(values), size=(rounds, len(values)), endpoint=False)
    dist = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(dist, .025)), float(np.quantile(dist, .975)), int(rng_seed)


def contrast_table(frame: pd.DataFrame, comparison: str, left: str, right: str, endpoint: str, rounds: int, extra: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    by_seed: dict[int, pd.Series] = {}
    records: list[dict[str, Any]] = []
    for seed in SEEDS:
        a = frame[(frame.seed.eq(seed)) & (frame.method.eq(left))].set_index("compound")[endpoint]
        b = frame[(frame.seed.eq(seed)) & (frame.method.eq(right))].set_index("compound")[endpoint]
        common = sorted(set(a.index) & set(b.index))
        by_seed[seed] = pd.Series({compound: float(a[compound] - b[compound]) for compound in common}, dtype=float)
    common_before_finite = sorted(set.intersection(*(set(x.index) for x in by_seed.values()))) if by_seed else []
    # A Cell Painting subset can be structurally unscorable (for example, no
    # exact matched foreign donor or a zero-variance PCC).  Do not regularise
    # those values: retain only compound units valid in every frozen seed for
    # the paired bootstrap, and expose the count in the result table.
    common_all = [compound for compound in common_before_finite if all(np.isfinite(by_seed[seed][compound]) for seed in SEEDS)]
    pooled = np.asarray([np.mean([by_seed[seed][compound] for seed in SEEDS]) for compound in common_all], dtype=np.float64)
    point, low, high, rng_seed = bootstrap(3407, f"Morph|{comparison}|{endpoint}|{json.dumps(extra,sort_keys=True)}", pooled, rounds)
    records.append({"seed": "mean", "comparison": comparison, "endpoint": endpoint, **extra, "seed3407": float(by_seed[3407][common_all].mean()) if common_all else math.nan, "seed42": float(by_seed[42][common_all].mean()) if common_all else math.nan, "seed2025": float(by_seed[2025][common_all].mean()) if common_all else math.nan, "n_compounds": len(pooled), "n_common_before_finite_filter": len(common_before_finite), "n_excluded_unscorable": len(common_before_finite) - len(pooled), "point": point, "ci_low": low, "ci_high": high, "rounds": rounds, "bootstrap_unit": "compound", "bootstrap_seed": rng_seed})
    return records, by_seed


def decision(row: dict[str, Any], require_two: bool = False) -> str:
    points = [float(row[name]) for name in ("seed3407", "seed42", "seed2025")]
    if all(value > 0 for value in points) and float(row["ci_low"]) > 0:
        return "GO"
    if float(row["point"]) > 0 and sum(value > 0 for value in points) >= 2:
        return "SUPPORTIVE"
    return "NO-GO"


def csv_write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def copy_mapping(mapping: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mapping, out)


def protocol(title: str, details: list[str]) -> str:
    return "# " + title + "\n\n" + "\n".join("- " + item for item in details) + "\n"


def report(title: str, objective: str, config: dict[str, Any], primary: pd.DataFrame, decisions: pd.DataFrame, limits: list[str]) -> str:
    lines = [f"# Results — {title}", "", "## 1. Objective", objective, "", "## 2. Frozen methods", "M0 raw, M1 frozen teacher, and M2 frozen GE-updated posterior only; no fitting or retuning.", "", "## 3. Dataset", "Frozen cpg0004-LINCS test CP profiles with three locked seeds.", "", "## 4. Feature mapping", f"Mapped feature/module audit: `{config['feature_mapping']}`.", "", "## 5. Eligibility", "Five-repeat test conditions and two frozen support slots; M2 uses aligned GE keys only.", "", "## 6. Support/reference isolation", "Each reference is the mean of the four physical CP repeats excluding the selected support row.", "", "## 7. Primary endpoint", "See paired primary contrasts below.", "", "## 8. Secondary endpoints", "Raw PCC/RRMSE for Morph-A; top-k PCC, controls and module-composition JS for Morph-B.", "", "## 9. Negative controls", "Foreign structural matched and shuffled-reference controls are saved in `NEGATIVE_CONTROLS.csv`.", "", "## 10. Sample counts", json.dumps(config["counts"], sort_keys=True), "\nEndpoint-specific unscorable compound-unit counts (zero variance/no PCC; no imputation):\n", json.dumps(config["unscorable_summary"], sort_keys=True), "", "## 11. Seed-level results", "Saved in `METRICS_BY_SEED.csv`.", "", "## 12. Compound-paired bootstrap CI", primary.to_csv(index=False), "", "## 13. GO / SUPPORTIVE / NO-GO", decisions.to_csv(index=False), "", "## 14. Biological interpretation", "Only a positive preregistered paired gate supports module or dominant-feature recovery; no mechanism-label claim follows.", "", "## 15. What the result does NOT prove", "It does not establish pathway recovery, repeat replacement, or a causal biological mechanism.", "", "## 16. Limitations", *["- " + item for item in limits]]
    return "\n".join(lines) + "\n"


def make_query_data(M, args: argparse.Namespace, rows: dict[str, np.ndarray], manifest: pd.DataFrame, split: dict[str, list[str]], annotations: pd.DataFrame, smiles: dict[str, str]) -> tuple[dict[int, tuple[pd.DataFrame, dict[str, np.ndarray], dict[tuple[str,str,int],int]]], list[dict[str,Any]]]:
    all_data: dict[int, tuple[pd.DataFrame, dict[str, np.ndarray], dict[tuple[str,str,int],int]]] = {}
    exclusions: list[dict[str, Any]] = []
    for seed in SEEDS:
        vf, vp = M.read_prediction(args.virtual_root / f"seed{seed}" / "test_predictions.npz", "virtual")
        gf, gp = M.read_prediction(args.ge_root / f"seed{seed}" / "test_predictions.npz", "ge")
        supports = M.load_pair_support(args.pair_root / f"seed{seed}" / "test_pair_manifest.csv")
        query, profiles, dropped = M.make_queries(rows, manifest, split, annotations, smiles, vf, vp, gf, gp, supports, 5, 0.10)
        all_data[seed] = (query, profiles, supports)
        exclusions.extend([{"seed": seed, **item} for item in dropped])
    return all_data, exclusions


def main() -> None:
    args = parse_args()
    rounds = 200 if args.smoke else int(args.bootstrap_rounds)
    if rounds <= 0:
        raise ValueError("bootstrap rounds must be positive")
    root = args.out_root.resolve()
    out_a, out_b = root / "module_recovery", root / "dominant_feature_recovery"
    for out in (out_a, out_b):
        if out.exists() and any(out.iterdir()) and not args.force:
            raise FileExistsError(f"Refusing existing output without --force: {out}")
        out.mkdir(parents=True, exist_ok=True); (out / "figures").mkdir(exist_ok=True)
    M = load_module(args.module)
    rows = M.load_rows(args.data); manifest = M.verify_manifest(rows, args.manifest); split = M.verify_split_lock(rows, args.split_lock)
    annotations, _ = M.load_annotations(args.annotations); smiles = M.load_smiles(args.smiles)
    mapping, modules, objects = read_mapping(args.feature_mapping, rows["delta"].shape[1])
    center, scale = robust_standardizer(rows)
    biological = mapping.loc[mapping.biological_endpoint.astype(str).eq("Eligible"), "feature_index"].to_numpy(dtype=np.int64)
    module_for = dict(zip(mapping.feature_index.astype(int), mapping.biological_module.astype(str)))
    slot_map = build_slot_map(rows)
    data_by_seed, global_exclusions = make_query_data(M, args, rows, manifest, split, annotations, smiles)
    group_rows = manifest.groupby(["compound_id", "dose"])["row_index"].agg(list).to_dict()
    a_rows: list[dict[str, Any]] = []
    b_rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for seed, (query, profiles, support_map) in data_by_seed.items():
        qlookup = {(str(row.compound_id), str(row.dose), int(row.support_row)): pos for pos, row in enumerate(query.itertuples(index=False))}
        shuffled: dict[int, int] = {}
        for dose, positions in query.groupby("dose", sort=False).groups.items():
            positions = np.asarray(list(positions), dtype=np.int64)
            rng = np.random.default_rng(stable_seed(seed, f"MorphB|shuffle|{dose}"))
            permutation = rng.permutation(positions)
            if len(positions) > 1 and np.any(permutation == positions):
                permutation = np.roll(permutation, 1)
            shuffled.update({int(a): int(b) for a, b in zip(positions, permutation)})
        for pos, item in enumerate(query.itertuples(index=False)):
            compound, dose, support = str(item.compound_id), str(item.dose), int(item.support_row)
            condition_rows = [int(x) for x in group_rows[(compound, dose)]]
            ref_rows = [row for row in condition_rows if row != support]
            if len(condition_rows) != 5 or len(ref_rows) != 4:
                global_exclusions.append({"seed": seed, "scope": "condition", "compound": compound, "dose": dose, "support_row": support, "reason": "strict_5_repeat_reference_not_available"})
                continue
            held_sources = sorted(row for row in condition_rows if support_map.get((compound, dose, row)) == support)
            if not held_sources:
                global_exclusions.append({"seed": seed, "scope": "condition", "compound": compound, "dose": dose, "support_row": support, "reason": "frozen_pair_source_missing"})
                continue
            held_source = int(held_sources[0])
            reference = rows["delta"][np.asarray(ref_rows, dtype=np.int64)].mean(axis=0, dtype=np.float64).astype(np.float32)
            donors = sampled_foreign(rows, matched_foreign_rows(rows, slot_map, compound, dose, support, held_source), seed, compound, dose, held_source)
            base = {"seed": seed, "compound": compound, "dose": dose, "support_slot": int(item.support_slot), "support_row": support, "held_source_row": held_source, "reference_rows": "|".join(map(str, ref_rows)), "reference_excludes_support": 1, "ge_available": int(bool(item.ge_available)), "foreign_candidate_draws": len(donors)}
            sample_rows.append(base)
            for module_name, indices in modules.items():
                null_values = [fisher(pcc(rows["delta"][s, indices], rows["delta"][h, indices])) for s, h in donors]
                null_values = [value for value in null_values if np.isfinite(value)]
                null_mean = float(np.mean(null_values)) if null_values else math.nan
                for method in METHODS:
                    if method == "posterior_ge" and not bool(item.ge_available):
                        continue
                    prediction = profiles[method][pos]
                    same = pcc(prediction[indices], reference[indices])
                    a_rows.append({**base, "analysis": "module", "group": module_name, "feature_count": len(indices), "method": method, "method_label": LABELS[method], "same_pcc": same, "same_fisher_z": fisher(same), "null_fisher_z": null_mean, "excess_fisher_z": fisher(same) - null_mean if np.isfinite(fisher(same)) and np.isfinite(null_mean) else math.nan, "relative_rmse": relative_rmse(prediction[indices], reference[indices])})
                    if null_values:
                        controls.append({**base, "analysis": "MorphA_module_null", "group": module_name, "method": method, "control": "matched_foreign_raw_support_to_held", "value": null_mean, "draw_count": len(null_values)})
            for object_name, indices in objects.items():
                null_values = [fisher(pcc(rows["delta"][s, indices], rows["delta"][h, indices])) for s, h in donors]
                null_values = [value for value in null_values if np.isfinite(value)]
                null_mean = float(np.mean(null_values)) if null_values else math.nan
                for method in METHODS:
                    if method == "posterior_ge" and not bool(item.ge_available):
                        continue
                    same = pcc(profiles[method][pos][indices], reference[indices])
                    a_rows.append({**base, "analysis": "object_sensitivity", "group": object_name, "feature_count": len(indices), "method": method, "method_label": LABELS[method], "same_pcc": same, "same_fisher_z": fisher(same), "null_fisher_z": null_mean, "excess_fisher_z": fisher(same) - null_mean if np.isfinite(fisher(same)) and np.isfinite(null_mean) else math.nan, "relative_rmse": relative_rmse(profiles[method][pos][indices], reference[indices])})
            for method in METHODS:
                if method == "posterior_ge" and not bool(item.ge_available):
                    continue
                for k in K_VALUES:
                    overlap, sign, top_pcc, ref_rank, pred_rank = feature_scores(profiles[method][pos], reference, center, scale, biological, k)
                    b_rows.append({**base, "method": method, "method_label": LABELS[method], "k": k, "overlap_at_k": overlap, "sign_consistency_at_k": sign, "pcc_reference_topk": top_pcc, "module_js": js_module(ref_rank, pred_rank, biological, module_for) if k == PRIMARY_K else math.nan})
                    foreign_metrics = []
                    for _, foreign_held in donors:
                        foreign_metrics.append(feature_scores(profiles[method][pos], rows["delta"][foreign_held], center, scale, biological, k)[:3])
                    if foreign_metrics:
                        values = np.asarray(foreign_metrics, dtype=float).mean(axis=0)
                        controls.append({**base, "analysis": "MorphB", "method": method, "k": k, "control": "matched_foreign_held_reference", "overlap_at_k": values[0], "sign_consistency_at_k": values[1], "pcc_reference_topk": values[2]})
                    shuffle_pos = shuffled.get(pos)
                    if shuffle_pos is not None:
                        shuffled_compound, shuffled_dose, shuffled_support = str(query.iloc[shuffle_pos].compound_id), str(query.iloc[shuffle_pos].dose), int(query.iloc[shuffle_pos].support_row)
                        sr = [int(x) for x in group_rows[(shuffled_compound, shuffled_dose)] if int(x) != shuffled_support]
                        if len(sr) == 4:
                            shuffled_ref = rows["delta"][np.asarray(sr, dtype=np.int64)].mean(axis=0)
                            v = feature_scores(profiles[method][pos], shuffled_ref, center, scale, biological, k)
                            controls.append({**base, "analysis": "MorphB", "method": method, "k": k, "control": "shuffled_compound_reference", "overlap_at_k": v[0], "sign_consistency_at_k": v[1], "pcc_reference_topk": v[2]})
    a_frame, b_frame, control_frame, samples = pd.DataFrame(a_rows), pd.DataFrame(b_rows), pd.DataFrame(controls), pd.DataFrame(sample_rows)
    if a_frame.empty or b_frame.empty:
        raise RuntimeError("No eligible Morph-A/B records")
    # Compound means preserve all doses and two frozen slots as one bootstrap unit.
    a_comp = a_frame.groupby(["seed", "analysis", "group", "method", "compound"], as_index=False)[["same_pcc", "excess_fisher_z", "relative_rmse"]].mean()
    a_seed = a_comp.groupby(["seed", "analysis", "group", "method"], as_index=False)[["same_pcc", "excess_fisher_z", "relative_rmse"]].mean()
    a_unscorable = a_comp.assign(unscorable=~np.isfinite(a_comp.excess_fisher_z)).groupby(["seed", "analysis", "group", "method"], as_index=False).agg(compound_units=("compound", "size"), unscorable_units=("unscorable", "sum"))
    a_contrasts: list[dict[str, Any]] = []
    for analysis in ("module", "object_sensitivity"):
        groups = MODULE_ORDER if analysis == "module" else OBJECT_ORDER
        for group in groups:
            current = a_comp[(a_comp.analysis.eq(analysis)) & (a_comp.group.eq(group))]
            for comparison, left, right in (("teacher_minus_raw", "teacher", "raw"), ("posterior_ge_minus_teacher", "posterior_ge", "teacher")):
                for endpoint in ("excess_fisher_z", "same_pcc", "relative_rmse"):
                    records, _ = contrast_table(current, comparison, left, right, endpoint, rounds, {"analysis": analysis, "group": group})
                    a_contrasts.extend(records)
    # Macro modules: first average condition/slot within compound/module, then unweighted module mean.
    macro = a_comp[a_comp.analysis.eq("module")].groupby(["seed", "compound", "method"], as_index=False)[["same_pcc", "excess_fisher_z", "relative_rmse"]].mean()
    for comparison, left, right in (("teacher_minus_raw", "teacher", "raw"), ("posterior_ge_minus_teacher", "posterior_ge", "teacher")):
        for endpoint in ("excess_fisher_z", "same_pcc", "relative_rmse"):
            records, _ = contrast_table(macro, comparison, left, right, endpoint, rounds, {"analysis": "macro_module", "group": "Macro", "module_weighting": "unweighted"})
            a_contrasts.extend(records)
    a_contrast_frame = pd.DataFrame(a_contrasts)
    a_primary = a_contrast_frame[(a_contrast_frame.analysis.eq("macro_module")) & (a_contrast_frame.endpoint.eq("excess_fisher_z"))].copy()
    a_primary["decision"] = a_primary.apply(lambda row: decision(row.to_dict()), axis=1)
    # Strong teacher claim is a reporting label only, evaluated after fixed per-module output.
    teacher_modules = a_contrast_frame[(a_contrast_frame.analysis.eq("module")) & (a_contrast_frame.comparison.eq("teacher_minus_raw")) & (a_contrast_frame.endpoint.eq("excess_fisher_z"))]
    broad_count = int((teacher_modules.point > 0).sum())
    negative_module = bool((teacher_modules.ci_high < 0).any())
    a_primary["broad_status"] = np.where((a_primary.comparison.eq("teacher_minus_raw")) & (a_primary.decision.eq("GO")) & (broad_count >= 4) & ~negative_module, "STRONG_GO", a_primary.decision)
    b_comp = b_frame.groupby(["seed", "k", "method", "compound"], as_index=False)[["overlap_at_k", "sign_consistency_at_k", "pcc_reference_topk", "module_js"]].mean()
    b_seed = b_comp.groupby(["seed", "k", "method"], as_index=False)[["overlap_at_k", "sign_consistency_at_k", "pcc_reference_topk", "module_js"]].mean()
    b_unscorable = b_comp.assign(unscorable=~np.isfinite(b_comp.module_js)).groupby(["seed", "k", "method"], as_index=False).agg(compound_units=("compound", "size"), unscorable_units=("unscorable", "sum"))
    control_comp = control_frame[control_frame.analysis.eq("MorphB")].groupby(["seed", "k", "method", "compound", "control"], as_index=False)[["overlap_at_k", "sign_consistency_at_k", "pcc_reference_topk"]].mean()
    control_contrasts: list[dict[str, Any]] = []
    for method in METHODS:
        for k in K_VALUES:
            same_method = b_comp[(b_comp.method.eq(method)) & (b_comp.k.eq(k))]
            for control in ("matched_foreign_held_reference", "shuffled_compound_reference"):
                control_method = control_comp[(control_comp.method.eq(method)) & (control_comp.k.eq(k)) & (control_comp.control.eq(control))]
                for endpoint in ("overlap_at_k", "sign_consistency_at_k", "pcc_reference_topk"):
                    current = pd.concat([same_method[["seed", "compound", endpoint]].assign(method="same_condition_reference"), control_method[["seed", "compound", endpoint]].assign(method="negative_control_reference")], ignore_index=True)
                    records, _ = contrast_table(current, "same_minus_" + control, "same_condition_reference", "negative_control_reference", endpoint, rounds, {"method": method, "k": k, "control": control})
                    control_contrasts.extend(records)
    control_contrast_frame = pd.DataFrame(control_contrasts)
    b_contrasts: list[dict[str, Any]] = []
    for k in K_VALUES:
        current = b_comp[b_comp.k.eq(k)]
        for comparison, left, right in (("teacher_minus_raw", "teacher", "raw"), ("posterior_ge_minus_teacher", "posterior_ge", "teacher")):
            for endpoint in ("overlap_at_k", "sign_consistency_at_k", "pcc_reference_topk", "module_js"):
                records, _ = contrast_table(current, comparison, left, right, endpoint, rounds, {"k": k})
                b_contrasts.extend(records)
    b_contrast_frame = pd.DataFrame(b_contrasts)
    b_primary = b_contrast_frame[(b_contrast_frame.k.eq(PRIMARY_K)) & (b_contrast_frame.endpoint.isin(["overlap_at_k", "sign_consistency_at_k"]))].copy()
    decisions_b: list[dict[str, Any]] = []
    for comparison in ("teacher_minus_raw", "posterior_ge_minus_teacher"):
        x = b_primary[b_primary.comparison.eq(comparison)].set_index("endpoint")
        overlap, sign = x.loc["overlap_at_k"].to_dict(), x.loc["sign_consistency_at_k"].to_dict()
        go = decision(overlap) == "GO" and decision(sign) == "GO"
        supportive = decision(overlap) == "GO" or decision(sign) == "GO"
        decisions_b.append({"comparison": comparison, "decision": "GO" if go else ("SUPPORTIVE" if supportive else "NO-GO"), "overlap_point": overlap["point"], "overlap_ci_low": overlap["ci_low"], "overlap_ci_high": overlap["ci_high"], "sign_point": sign["point"], "sign_ci_low": sign["ci_low"], "sign_ci_high": sign["ci_high"], "n_overlap": overlap["n_compounds"], "n_sign": sign["n_compounds"]})
    b_decision_frame = pd.DataFrame(decisions_b)
    a_unscorable = a_unscorable.assign(endpoint="excess_fisher_z", k=pd.NA)
    b_unscorable = b_unscorable.assign(analysis="MorphB", group="dominant_feature_module_js", endpoint="module_js")
    unscorable_table = pd.concat([a_unscorable, b_unscorable], ignore_index=True, sort=False)
    unscorable_summary = json.loads(unscorable_table.to_json(orient="records"))
    input_hashes = {name: sha256(path) for name, path in {"cp_rows": args.data, "manifest": args.manifest, "split_lock": args.split_lock, "feature_mapping": args.feature_mapping}.items()}
    common_config = {"version": "cpg0004-LINCS-morphological-A-B-2026-08-30", "dataset": "cpg0004-LINCS", "seeds": list(SEEDS), "methods": LABELS, "feature_mapping": str(args.feature_mapping.resolve()), "feature_mapping_sha256": sha256(args.feature_mapping), "biological_feature_count": int(len(biological)), "excluded_frozen_metadata": ["Batch_Number"], "module_counts": {key: len(value) for key, value in modules.items()}, "object_counts": {key: len(value) for key, value in objects.items()}, "input_hashes": input_hashes, "support_reference": "two frozen support slots; four-repeat CP reference excluding support", "null": {"definition": "method-shared foreign raw support-to-held Fisher-z", "match": "exact dose + held/support plate slots + well row + test split", "rounds": NULL_ROUNDS, "exact_well_status": "structurally unavailable"}, "bootstrap": {"rounds": rounds, "unit": "compound", "paired": True, "ci": "percentile 95%"}, "counts": {"morphA_slot_records": len(a_frame), "morphB_slot_records": len(b_frame), "eligible_conditions": len(samples), "unique_compounds": int(samples.compound.nunique())}, "unscorable_summary": unscorable_summary}
    for out, title, kind in ((out_a, "Morph-A module recovery", "A"), (out_b, "Morph-B dominant feature recovery", "B")):
        copy_mapping(args.feature_mapping, out / "FEATURE_MAPPING.csv")
        (out / "PROTOCOL.md").write_text(protocol(title, ["Frozen M0/M1/M2 only; no retraining or parameter selection.", "Test support/reference and foreign-null structures are inherited from the cpg0004 lock.", "All primary uncertainty uses a 10,000-round compound-paired bootstrap (or smoke-only 200 rounds).", "Batch_Number is retained only for frozen-model dimensional fidelity and excluded from biological endpoints."]), encoding="utf-8")
        (out / "CONFIG.json").write_text(json.dumps({**common_config, "analysis": kind}, indent=2, sort_keys=True), encoding="utf-8")
        csv_write(out / "ELIGIBLE_COMPOUNDS.csv", samples)
        csv_write(out / "EXCLUSIONS.csv", pd.DataFrame(global_exclusions))
        csv_write(out / "NEGATIVE_CONTROLS.csv", control_frame[control_frame.analysis.str.startswith(f"Morph{kind}", na=False)])
        csv_write(out / "UNSCORABLE_COUNTS.csv", unscorable_table[unscorable_table.analysis.eq("MorphB")] if kind == "B" else unscorable_table[unscorable_table.analysis.ne("MorphB")])
    csv_write(out_a / "METRICS_BY_COMPOUND.csv", a_comp)
    csv_write(out_a / "METRICS_BY_MODULE.csv", a_seed)
    csv_write(out_a / "METRICS_BY_SEED.csv", a_seed)
    csv_write(out_a / "PAIRED_CONTRASTS.csv", a_contrast_frame)
    csv_write(out_a / "BOOTSTRAP_CI.csv", a_contrast_frame)
    csv_write(out_a / "DECISION.csv", a_primary)
    (out_a / "DECISION.md").write_text("# Morph-A decision\n\n```csv\n" + a_primary.to_csv(index=False) + "```\n", encoding="utf-8")
    (out_a / "RESULTS.md").write_text(report("Morph-A module recovery", "Test whether frozen methods recover independent repeat-supported cellular modules beyond the structural foreign null.", common_config, a_primary, a_primary, ["Foreign null is a method-shared structural baseline as in the frozen cpg0004 benchmark.", "Only the two recorded support slots are available; the other three are not inferred."]), encoding="utf-8")
    csv_write(out_b / "METRICS_BY_COMPOUND.csv", b_comp)
    csv_write(out_b / "METRICS_BY_MODULE.csv", b_seed)
    csv_write(out_b / "METRICS_BY_SEED.csv", b_seed)
    csv_write(out_b / "PAIRED_CONTRASTS.csv", b_contrast_frame)
    csv_write(out_b / "BOOTSTRAP_CI.csv", b_contrast_frame)
    csv_write(out_b / "CONTROL_CONTRASTS.csv", control_contrast_frame)
    csv_write(out_b / "DECISION.csv", b_decision_frame)
    (out_b / "DECISION.md").write_text("# Morph-B decision\n\n```csv\n" + b_decision_frame.to_csv(index=False) + "```\n", encoding="utf-8")
    b_report = report("Morph-B dominant feature recovery", "Test whether frozen methods recover independent-reference top changing Cell Painting features and their signs.", common_config, b_primary, b_decision_frame, ["Top-k uses the 241 biological features only; Batch_Number is excluded before ranking.", "Null controls validate correspondence but do not replace the teacher-minus-raw paired primary comparison."])
    b_report = b_report.replace("Foreign structural matched and shuffled-reference controls are saved in `NEGATIVE_CONTROLS.csv`.", "Foreign structural matched and shuffled-reference controls are saved in `NEGATIVE_CONTROLS.csv`; their paired contrasts are in `CONTROL_CONTRASTS.csv`.\n\n" + control_contrast_frame.to_csv(index=False))
    (out_b / "RESULTS.md").write_text(b_report, encoding="utf-8")
    # A compact editable SVG avoids a plotting dependency while retaining numeric provenance in CSV.
    for out, primary, label in ((out_a, a_primary, "Macro module excess-z"), (out_b, b_decision_frame, "Top-25 overlap/sign")):
        body = primary.to_csv(index=False).replace("&", "&amp;").replace("<", "&lt;")
        (out / "figures" / "summary.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="360"><rect width="100%" height="100%" fill="white"/><text x="30" y="42" font-family="Arial" font-size="22">{label}</text><text x="30" y="78" font-family="monospace" font-size="12" xml:space="preserve">{body}</text></svg>', encoding="utf-8")
    print(json.dumps({"out_root": str(root), "morphA_decisions": a_primary.to_dict("records"), "morphB_decisions": decisions_b, "rounds": rounds}, indent=2))


if __name__ == "__main__":
    main()
