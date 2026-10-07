#!/usr/bin/env python3
"""Frozen cpg0004 Path-C Reactome pathway-trajectory evaluator.

This evaluator contains no model fitting.  It reconstructs the same frozen
M0/M1/M2 profiles used by Path-A, obtains fixed-k Reactome pathway enrichment
score vectors from train+validation CP neighbours, and tests six-dose pathway
geometry against CP consensus profiles that exclude the support row.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEEDS = (3407, 42, 2025)
DOSES = ("0.04", "0.12", "0.37", "1.11", "3.33", "10")
METHODS = ("raw", "teacher", "posterior_ge")
METHOD_LABELS = {
    "raw": "M0_1R_raw",
    "teacher": "M1_reproducible_effect_teacher",
    "posterior_ge": "M2_validated_GE_updated_posterior",
}
PRIMARY_K = 25
REQUIRED_REPEATS = 5
EXPECTED_SUPPORT_SLOTS = 2
BOOTSTRAP_ROUNDS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True, help="Frozen cpg0004 retrieval implementation.")
    parser.add_argument("--pathway-module", type=Path, required=True, help="Path-A implementation supplying GMT/ORA helpers.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--smiles", type=Path, required=True)
    parser.add_argument("--reactome-gmt", type=Path, required=True)
    parser.add_argument("--virtual-root", type=Path, required=True)
    parser.add_argument("--ge-root", type=Path, required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--virtual-pattern", default="seed{seed}_1r_all_v6/test_predictions.npz", help="Relative frozen virtual-prediction path; must contain {seed}.")
    parser.add_argument("--ge-pattern", default="all_b1_seed{seed}/test_predictions.npz", help="Relative frozen GE-prediction path; must contain {seed}.")
    parser.add_argument("--pair-pattern", default="full_b1_all_seed{seed}/test_pair_manifest.csv", help="Relative frozen pair-manifest path; must contain {seed}.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bootstrap-rounds", type=int, default=BOOTSTRAP_ROUNDS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pcc(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x = x - x.mean()
    y = y - y.mean()
    norm = float(np.linalg.norm(x) * np.linalg.norm(y))
    if not np.isfinite(norm) or norm <= 0:
        return None
    value = float(np.dot(x, y) / norm)
    return value if np.isfinite(value) else None


def rankdata(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    rank = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        rank[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return rank


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return None
    return pcc(rankdata(x[valid]), rankdata(y[valid]))


def score_geometry(vectors: list[np.ndarray]) -> tuple[float, np.ndarray]:
    distances: list[float] = []
    for i, j in combinations(range(len(DOSES)), 2):
        value = pcc(vectors[i], vectors[j])
        if value is None:
            return math.nan, np.full(15, np.nan, dtype=np.float64)
        distances.append(1.0 - value)
    return 0.0, np.asarray(distances, dtype=np.float64)


def trajectory_concordance(method_vectors: list[np.ndarray], reference_vectors: list[np.ndarray]) -> tuple[float, np.ndarray, np.ndarray]:
    _, method_distances = score_geometry(method_vectors)
    _, reference_distances = score_geometry(reference_vectors)
    result = spearman(method_distances, reference_distances)
    return (float(result) if result is not None else math.nan), method_distances, reference_distances


def score_pathway_vector(
    M,
    E,
    profile: np.ndarray,
    dose: str,
    gallery: pd.DataFrame,
    gallery_values: np.ndarray,
    gallery_by_dose: dict[str, np.ndarray],
    targets_by_gallery: dict[str, set[str]],
    background_size: int,
    pathway_sizes: np.ndarray,
    gene_to_indices: dict[str, np.ndarray],
) -> tuple[np.ndarray, int, int]:
    gallery_indices = gallery_by_dose[str(dose)]
    scores = M.pcc_matrix(np.asarray(profile, dtype=np.float32)[None, :], gallery_values[gallery_indices])[0]
    names = gallery.iloc[gallery_indices].gallery_id.astype(str).to_numpy()
    order = np.lexsort((names, -scores))
    ranked = gallery_indices[order]
    if len(ranked) < PRIMARY_K:
        raise RuntimeError(f"Fewer than k={PRIMARY_K} gallery conditions at dose {dose}")
    genes = set().union(*(targets_by_gallery[str(gallery.iloc[index].gallery_id)] for index in ranked[:PRIMARY_K]))
    _, fdr, hits = E.enrich(genes, background_size, pathway_sizes, gene_to_indices)
    vector = -np.log10(np.clip(np.asarray(fdr, dtype=np.float64), 1e-300, 1.0))
    return vector, int(len(genes)), int(np.sum(hits > 0))


def bootstrap(values: np.ndarray, E, label: str, rounds: int) -> tuple[float, float, float, int]:
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0 or not np.isfinite(x).all():
        return math.nan, math.nan, math.nan, -1
    seed = int(E.stable(3407, label))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(x), size=(rounds, len(x)), endpoint=False)
    means = x[sampled].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), seed


def csv_block(frame: pd.DataFrame) -> str:
    return "```csv\n" + frame.to_csv(index=False) + "```"


def render_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    wanted = [row for row in rows if row.get("seed") == "mean"]
    width, height = 980, 430
    left, top, plot_w, plot_h = 95, 70, 800, 260
    values = [float(row["point"]) for row in wanted if np.isfinite(float(row["point"]))]
    lows = [float(row["ci_low"]) for row in wanted if np.isfinite(float(row["ci_low"]))]
    highs = [float(row["ci_high"]) for row in wanted if np.isfinite(float(row["ci_high"]))]
    lower = min([0.0, *lows, *values]) if values else -1.0
    upper = max([0.0, *highs, *values]) if values else 1.0
    padding = max((upper - lower) * 0.15, 0.02)
    lower, upper = lower - padding, upper + padding
    if upper <= lower:
        upper = lower + 1.0
    y = lambda value: top + (upper - value) / (upper - lower) * plot_h
    zero = y(0.0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.small{font-size:13px}.title{font-size:20px;font-weight:bold}</style>',
        '<text class="title" x="95" y="32">Path-C: Reactome pathway-trajectory concordance</text>',
        '<text class="small" x="95" y="53">point = three-seed mean paired difference; whisker = 95% compound-bootstrap CI</text>',
        f'<line x1="{left}" y1="{zero:.2f}" x2="{left+plot_w}" y2="{zero:.2f}" stroke="#777" stroke-width="1.5"/>',
    ]
    for tick in np.linspace(lower, upper, 5):
        yy = y(float(tick))
        lines.extend([f'<line x1="{left}" y1="{yy:.2f}" x2="{left+plot_w}" y2="{yy:.2f}" stroke="#e8e8e8"/>', f'<text class="small" x="18" y="{yy+4:.2f}">{tick:.3f}</text>'])
    labels = {"teacher_minus_raw": "M1 teacher − M0 raw", "posterior_ge_minus_teacher": "M2 posterior − M1 teacher"}
    for index, row in enumerate(wanted):
        cx = left + 210 + index * 350
        point, lo, hi = float(row["point"]), float(row["ci_low"]), float(row["ci_high"])
        color = "#2a9d8f" if lo > 0 else "#d1495b"
        lines.extend([
            f'<line x1="{cx}" y1="{y(lo):.2f}" x2="{cx}" y2="{y(hi):.2f}" stroke="{color}" stroke-width="4"/>',
            f'<line x1="{cx-12}" y1="{y(lo):.2f}" x2="{cx+12}" y2="{y(lo):.2f}" stroke="{color}" stroke-width="3"/>',
            f'<line x1="{cx-12}" y1="{y(hi):.2f}" x2="{cx+12}" y2="{y(hi):.2f}" stroke="{color}" stroke-width="3"/>',
            f'<circle cx="{cx}" cy="{y(point):.2f}" r="7" fill="{color}"/>',
            f'<text class="small" x="{cx-115}" y="365">{labels.get(row["comparison"], row["comparison"])}</text>',
            f'<text class="small" x="{cx-80}" y="388">{point:.4f} [{lo:.4f}, {hi:.4f}]</text>',
        ])
    lines.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = args.outdir
    if out.exists() and any(out.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing nonempty output without --force: {out}")
    out.mkdir(parents=True, exist_ok=True)
    M = load_module(args.module, "frozen_moa_pathc")
    E = load_module(args.pathway_module, "frozen_patha_pathc")
    rows = M.load_rows(args.data)
    manifest = M.verify_manifest(rows, args.manifest)
    split = M.verify_split_lock(rows, args.split_lock)
    annotations, annotation_audit = M.load_annotations(args.annotations)
    smiles = M.load_smiles(args.smiles)
    gallery, _, _ = M.make_gallery(rows, manifest, annotations, smiles, min_repeats=3)
    gallery_values = np.vstack([
        rows["delta"][manifest[(manifest.compound_id.eq(item.compound_id)) & (manifest.dose.eq(item.dose))].row_index.to_numpy(dtype=np.int64)].mean(axis=0, dtype=np.float64)
        for item in gallery.itertuples(index=False)
    ]).astype(np.float32)
    gallery_by_dose = {str(dose): group.profile_index.to_numpy(dtype=np.int64) for dose, group in gallery.groupby("dose", sort=False)}
    pathways, pathway_by_gene = E.parse_gmt(args.reactome_gmt)
    pathway_names = sorted(pathways)
    ann = annotations.set_index("compound_id")
    targets_by_gallery = {
        str(item.gallery_id): {str(gene).upper() for gene in M.split_labels(ann.loc[str(item.compound_id), "target"])} if str(item.compound_id) in ann.index else set()
        for item in gallery.itertuples(index=False)
    }
    background = set().union(*targets_by_gallery.values()) & set(pathway_by_gene)
    for key in targets_by_gallery:
        targets_by_gallery[key] &= background
    pathway_sizes = np.asarray([len(pathways[name] & background) for name in pathway_names], dtype=np.int64)
    gene_to_indices: dict[str, np.ndarray] = {}
    build_indices: dict[str, list[int]] = {}
    for index, name in enumerate(pathway_names):
        for gene in pathways[name] & background:
            build_indices.setdefault(gene, []).append(index)
    gene_to_indices = {gene: np.asarray(indices, dtype=np.int64) for gene, indices in build_indices.items()}

    (out / "protocol").mkdir(exist_ok=True)
    protocol = f"""# Path-C protocol lock

- Dataset: frozen cpg0004-LINCS CP plate rows, with `{', '.join(DOSES)}` as the only doses.
- Profiles: M0 raw one-support CP, M1 frozen reproducible-effect teacher, and M2 frozen GE-updated posterior; no fitting, parameter selection, or retraining.
- Gallery: train+validation compounds only; dose-matched CP consensus with at least three repeats.
- Pathway representation: fixed Reactome 2022 GMT; top `{PRIMARY_K}` PCC CP neighbours -> union of frozen curated target genes -> hypergeometric ORA against the gallery-target background; BH FDR per profile; score vector = `-log10(FDR)`.
- Eligibility: every test compound has exactly five CP repeats at all six doses and exactly two frozen support slots per condition.  M2 additionally requires GE availability for every dose/slot in all three seeds.
- Independent reference: mean of the four physical CP repeats other than the selected support row.  No support row appears in its own reference.
- Primary endpoint: Spearman correlation between the 15 upper-triangle pathway-score-vector distances `1-PCC` for a method and the corresponding held-out CP reference distances.
- Inference: score rotations first, average them at compound level, report three frozen seeds, then use 10,000 compound-paired bootstrap resamples.  GO requires all seed directions positive and the three-seed mean 95% CI > 0.

This is a target-annotation-derived pathway proxy, not an independently measured GE pathway truth.  Only two of five physical support rotations exist in the frozen artifacts.
"""
    (out / "protocol" / "PROTOCOL.md").write_text(protocol, encoding="utf-8")

    seed_queries: dict[int, tuple[pd.DataFrame, dict[str, np.ndarray]]] = {}
    all_exclusions: list[dict[str, Any]] = []
    m1_sets: dict[int, set[str]] = {}
    m2_sets: dict[int, set[str]] = {}
    support_signatures: dict[int, dict[tuple[str, str, int], int]] = {}
    for seed in SEEDS:
        vf, vp = M.read_prediction(args.virtual_root / args.virtual_pattern.format(seed=seed), "virtual")
        gf, gp = M.read_prediction(args.ge_root / args.ge_pattern.format(seed=seed), "ge")
        support = M.load_pair_support(args.pair_root / args.pair_pattern.format(seed=seed))
        query, profiles, exclusions = M.make_queries(rows, manifest, split, annotations, smiles, vf, vp, gf, gp, support, REQUIRED_REPEATS, 0.10)
        seed_queries[seed] = (query, profiles)
        all_exclusions.extend([{"seed": seed, **record} for record in exclusions])
        eligible_m1: set[str] = set()
        eligible_m2: set[str] = set()
        signatures: dict[tuple[str, str, int], int] = {}
        for compound in sorted(set(query.compound_id.astype(str))):
            strict = True
            rows_for_compound = manifest[manifest.compound_id.eq(compound)]
            for dose in DOSES:
                repeats = rows_for_compound[rows_for_compound.dose.eq(dose)].row_index.to_numpy(dtype=np.int64)
                slots = query[(query.compound_id.eq(compound)) & (query.dose.eq(dose))]
                if len(repeats) != REQUIRED_REPEATS or len(slots) != EXPECTED_SUPPORT_SLOTS or set(slots.support_slot.astype(int)) != {1, 2}:
                    strict = False
                    all_exclusions.append({"seed": seed, "scope": "six_dose_eligibility", "compound_id": compound, "dose": dose, "reason": "requires_5_repeats_and_2_frozen_slots", "detail": f"repeats={len(repeats)};slots={len(slots)}"})
                    continue
                for item in slots.itertuples(index=False):
                    signatures[(compound, str(dose), int(item.support_slot))] = int(item.support_row)
            if strict:
                eligible_m1.add(compound)
                q = query[query.compound_id.eq(compound)]
                if bool(q.ge_available.all()) and len(q) == len(DOSES) * EXPECTED_SUPPORT_SLOTS:
                    eligible_m2.add(compound)
                else:
                    all_exclusions.append({"seed": seed, "scope": "M2_common_set", "compound_id": compound, "dose": "all", "reason": "missing_GE_for_one_or_more_locked_slots", "detail": ""})
        m1_sets[seed], m2_sets[seed], support_signatures[seed] = eligible_m1, eligible_m2, signatures
    m1 = set.intersection(*(m1_sets[seed] for seed in SEEDS))
    m2 = set.intersection(*(m2_sets[seed] for seed in SEEDS))
    mapping_errors: list[str] = []
    base_sig = support_signatures[SEEDS[0]]
    for seed in SEEDS[1:]:
        if support_signatures[seed] != base_sig:
            mapping_errors.append(f"cross_seed_support_signature_mismatch:seed={seed};delta={len(set(base_sig.items()) ^ set(support_signatures[seed].items()))}")
    if mapping_errors:
        raise RuntimeError("; ".join(mapping_errors))

    reference_cache: dict[tuple[str, str, int], tuple[np.ndarray, int, int]] = {}
    raw_cache: dict[tuple[str, str, int], tuple[np.ndarray, int, int]] = {}
    slot_records: list[dict[str, Any]] = []
    compound_metric_rows: list[dict[str, Any]] = []
    per_seed_metric: dict[tuple[int, str, str, str], float] = {}
    for seed in SEEDS:
        query, profiles = seed_queries[seed]
        position = {(str(item.compound_id), str(item.dose), int(item.support_slot)): index for index, item in enumerate(query.itertuples(index=False))}
        for analysis_set, compounds, methods in (("M1_vs_M0_common", m1, ("raw", "teacher")), ("M2_vs_M1_common", m2, ("teacher", "posterior_ge"))):
            for compound in sorted(compounds):
                per_method_rotation: dict[str, list[float]] = {method: [] for method in methods}
                for rotation in (1, 2):
                    reference_vectors: list[np.ndarray] = []
                    vectors_by_method: dict[str, list[np.ndarray]] = {method: [] for method in methods}
                    slot_detail: list[dict[str, Any]] = []
                    for dose in DOSES:
                        key = (compound, dose, rotation)
                        if key not in position:
                            raise RuntimeError(f"Missing strict Path-C query key: seed={seed};key={key}")
                        pos = position[key]
                        row = query.iloc[pos]
                        support_row = int(row.support_row)
                        repeats = manifest[(manifest.compound_id.eq(compound)) & (manifest.dose.eq(dose))].row_index.to_numpy(dtype=np.int64)
                        reference_rows = tuple(int(x) for x in repeats if int(x) != support_row)
                        if len(repeats) != REQUIRED_REPEATS or len(reference_rows) != REQUIRED_REPEATS - 1 or support_row in reference_rows:
                            raise RuntimeError(f"Invalid held-out reference: {compound}|{dose}|{support_row}")
                        ref_key = (compound, dose, support_row)
                        if ref_key not in reference_cache:
                            reference = rows["delta"][np.asarray(reference_rows, dtype=np.int64)].mean(axis=0, dtype=np.float64).astype(np.float32)
                            reference_cache[ref_key] = score_pathway_vector(M, E, reference, dose, gallery, gallery_values, gallery_by_dose, targets_by_gallery, len(background), pathway_sizes, gene_to_indices)
                        ref_vector, ref_gene_count, ref_hit_pathways = reference_cache[ref_key]
                        reference_vectors.append(ref_vector)
                        detail = {"compound_id": compound, "dose": dose, "rotation": rotation, "support_row": support_row, "reference_rows": "|".join(map(str, reference_rows)), "reference_excludes_support": 1, "reference_neighbor_target_gene_count": ref_gene_count, "reference_hit_pathway_count": ref_hit_pathways}
                        for method in methods:
                            if method == "raw":
                                if ref_key not in raw_cache:
                                    raw_cache[ref_key] = score_pathway_vector(M, E, profiles["raw"][pos], dose, gallery, gallery_values, gallery_by_dose, targets_by_gallery, len(background), pathway_sizes, gene_to_indices)
                                vector, genes, hit_paths = raw_cache[ref_key]
                            else:
                                vector, genes, hit_paths = score_pathway_vector(M, E, profiles[method][pos], dose, gallery, gallery_values, gallery_by_dose, targets_by_gallery, len(background), pathway_sizes, gene_to_indices)
                            vectors_by_method[method].append(vector)
                            detail[f"{method}_neighbor_target_gene_count"] = genes
                            detail[f"{method}_hit_pathway_count"] = hit_paths
                        slot_detail.append(detail)
                    for method in methods:
                        score, method_distances, reference_distances = trajectory_concordance(vectors_by_method[method], reference_vectors)
                        status = "OK" if np.isfinite(score) else "UNDEFINED_PATHWAY_GEOMETRY"
                        if status == "OK":
                            per_method_rotation[method].append(score)
                        for item in slot_detail:
                            slot_records.append({"seed": seed, "analysis_set": analysis_set, "method": method, "method_label": METHOD_LABELS[method], "trajectory_concordance": score, "status": status, "pathway_count": len(pathway_names), "primary_k": PRIMARY_K, **item})
                        if status != "OK":
                            all_exclusions.append({"seed": seed, "scope": "endpoint_availability", "compound_id": compound, "dose": "all", "rotation": rotation, "method": method, "reason": status, "detail": "At least one fixed BH-FDR pathway score vector has zero variance, so PCC pathway geometry is undefined; no zero-distance imputation was used."})
                for method, scores in per_method_rotation.items():
                    if len(scores) != EXPECTED_SUPPORT_SLOTS:
                        continue
                    value = float(np.mean(scores))
                    per_seed_metric[(seed, analysis_set, method, compound)] = value
                    compound_metric_rows.append({"seed": seed, "analysis_set": analysis_set, "compound_id": compound, "method": method, "method_label": METHOD_LABELS[method], "trajectory_concordance": value, "support_rotations": EXPECTED_SUPPORT_SLOTS, "dose_count": len(DOSES), "primary_k": PRIMARY_K})

    metric_frame = pd.DataFrame(compound_metric_rows)
    seed_rows: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for analysis_set, comparison, left, right, compounds in (("M1_vs_M0_common", "teacher_minus_raw", "teacher", "raw", m1), ("M2_vs_M1_common", "posterior_ge_minus_teacher", "posterior_ge", "teacher", m2)):
        seed_values: dict[int, pd.Series] = {}
        for seed in SEEDS:
            left_values = metric_frame[(metric_frame.seed.eq(seed)) & (metric_frame.analysis_set.eq(analysis_set)) & (metric_frame.method.eq(left))].set_index("compound_id").trajectory_concordance
            right_values = metric_frame[(metric_frame.seed.eq(seed)) & (metric_frame.analysis_set.eq(analysis_set)) & (metric_frame.method.eq(right))].set_index("compound_id").trajectory_concordance
            common = sorted(set(left_values.index) & set(right_values.index))
            diffs = pd.Series({compound: float(left_values[compound] - right_values[compound]) for compound in common}, dtype=float)
            seed_values[seed] = diffs
            left_common = left_values.loc[common]
            right_common = right_values.loc[common]
            seed_rows.extend([
                {"seed": seed, "analysis_set": analysis_set, "method": left, "method_label": METHOD_LABELS[left], "endpoint": "pathway_trajectory_concordance", "value": float(left_common.mean()), "n_compounds": len(common), "support_rotations_per_compound": EXPECTED_SUPPORT_SLOTS, "dose_count": len(DOSES)},
                {"seed": seed, "analysis_set": analysis_set, "method": right, "method_label": METHOD_LABELS[right], "endpoint": "pathway_trajectory_concordance", "value": float(right_common.mean()), "n_compounds": len(common), "support_rotations_per_compound": EXPECTED_SUPPORT_SLOTS, "dose_count": len(DOSES)},
            ])
        common_all = sorted(set.intersection(*(set(values.index) for values in seed_values.values())))
        pooled = np.asarray([np.mean([seed_values[seed][compound] for seed in SEEDS]) for compound in common_all], dtype=np.float64)
        point, ci_low, ci_high, bootstrap_seed = bootstrap(pooled, E, f"PathC|{comparison}|trajectory", args.bootstrap_rounds)
        seed_points = {f"seed{seed}": float(seed_values[seed].mean()) for seed in SEEDS}
        contrasts.append({"seed": "mean", "analysis_set": analysis_set, "comparison": comparison, "endpoint": "pathway_trajectory_concordance", **seed_points, "compound_count": len(pooled), "point": point, "ci_low": ci_low, "ci_high": ci_high, "rounds": args.bootstrap_rounds, "bootstrap_unit": "compound", "bootstrap_seed": bootstrap_seed})
        go = all(seed_points[f"seed{seed}"] > 0 for seed in SEEDS) and ci_low > 0
        decisions.append({"comparison": comparison, "decision": "GO" if go else "NO-GO", "seed3407": seed_points["seed3407"], "seed42": seed_points["seed42"], "seed2025": seed_points["seed2025"], "point": point, "ci_low": ci_low, "ci_high": ci_high, "n_compounds": len(pooled), "reason": "all seed directions positive and pooled paired 95% CI above zero" if go else "predeclared all-positive-seed and positive-CI gate not met"})

    audit = {
        "version": "cpg0004-LINCS-PathC-Reactome-2026-08-30",
        "dataset": "cpg0004-LINCS",
        "library": "Reactome_2022",
        "reactome_gmt_sha256": digest(args.reactome_gmt),
        "reactome_pathway_count": len(pathway_names),
        "gallery_conditions": len(gallery),
        "gallery_compounds": int(gallery.compound_id.nunique()),
        "gallery_background_target_genes": len(background),
        "seeds": list(SEEDS),
        "doses": list(DOSES),
        "primary_k": PRIMARY_K,
        "strict_repeat_count": REQUIRED_REPEATS,
        "frozen_support_slots": EXPECTED_SUPPORT_SLOTS,
        "m1_common_compounds": len(m1),
        "m2_common_compounds": len(m2),
        "m1_condition_rotation_rows": len(m1) * len(DOSES) * EXPECTED_SUPPORT_SLOTS,
        "m2_condition_rotation_rows": len(m2) * len(DOSES) * EXPECTED_SUPPORT_SLOTS,
        "annotation_source_sha256": digest(args.annotations),
        "frozen_input_sha256": {"cp_rows": digest(args.data), "manifest": digest(args.manifest), "split_lock": digest(args.split_lock)},
        "guardrails": ["no model training or tuning", "test compounds absent from gallery", "fixed top-k target ORA", "BH FDR per profile", "independent CP reference excludes support", "rotation-before-compound aggregation", "compound-paired bootstrap"],
        "interpretation_limit": "Reactome scores are curated-target-neighbour pathway proxies, not an independent GE pathway measurement.",
    }
    (out / "protocol" / "CONFIG.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame([{ "compound_id": compound, "analysis_set": "M1_vs_M0_common", "m2_ge_available_all_seeds": int(compound in m2)} for compound in sorted(m1)]).to_csv(out / "ELIGIBLE_COMPOUNDS.csv", index=False)
    pd.DataFrame(all_exclusions).to_csv(out / "EXCLUSIONS.csv", index=False)
    pd.DataFrame(slot_records).to_csv(out / "SLOT_METRICS.csv", index=False)
    metric_frame.to_csv(out / "COMPOUND_METRICS.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "METRICS_BY_SEED.csv", index=False)
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(out / "PAIRED_CONTRASTS.csv", index=False)
    contrast_frame.to_csv(out / "BOOTSTRAP_CI.csv", index=False)
    decision_frame = pd.DataFrame(decisions)
    decision_frame.to_csv(out / "DECISION.csv", index=False)
    (out / "DECISION.md").write_text("# Path-C decision\n\n" + csv_block(decision_frame) + "\n", encoding="utf-8")
    (out / "RESULTS.md").write_text(
        "# Path-C Reactome pathway-trajectory result\n\n"
        "## Endpoint\n\n"
        "For every compound, frozen support rotation, and dose, CP profiles were converted to a fixed Reactome pathway-score vector via dose-matched train+validation CP neighbours and top-25 curated-target ORA. The endpoint is the Spearman correlation between the 15 pairwise `1-PCC` distances of those vectors and those from a four-repeat CP reference that excludes the support row.\n\n"
        "## Audit\n\n```json\n" + json.dumps(audit, indent=2, sort_keys=True) + "\n```\n\n"
        "## Primary paired contrasts\n\n" + csv_block(contrast_frame) + "\n\n"
        "## Decision\n\n" + csv_block(decision_frame) + "\n\n"
        "The result is limited to this target-annotation-derived Reactome proxy and two frozen support rotations. It does not establish replacement of physical repeats or independent GE pathway validation.\n",
        encoding="utf-8",
    )
    render_svg(out / "figures" / "pathway_trajectory_summary.svg", contrasts)
    print(json.dumps({"outdir": str(out), "m1_common_compounds": len(m1), "m2_common_compounds": len(m2), "decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
