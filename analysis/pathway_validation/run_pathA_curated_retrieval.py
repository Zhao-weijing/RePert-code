#!/usr/bin/env python3
"""Locked Path-A curated Reactome pathway-recovery evaluation for cpg0004."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

SEEDS = (3407, 42, 2025)
KS = (10, 25, 50)
PRIMARY_K = 25
BOOTSTRAP_ROUNDS = 10_000
METHODS = ("raw", "teacher", "posterior_ge")
VARIANTS = ("base", "scaffold_excluded", "tanimoto_gt_0.7_excluded")
METRICS = ("pathway_map", "mrr", "recall_at_10")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True, help="Frozen MoA retrieval implementation to reuse for profile reconstruction.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split-lock", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--smiles", type=Path, required=True)
    p.add_argument("--reactome-gmt", type=Path, required=True)
    p.add_argument("--library-name", default="Reactome_2022")
    p.add_argument("--virtual-root", type=Path, required=True)
    p.add_argument("--ge-root", type=Path, required=True)
    p.add_argument("--pair-root", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--min-gallery-repeats", type=int, default=3)
    p.add_argument("--min-query-repeats", type=int, default=5)
    p.add_argument("--bootstrap-rounds", type=int, default=BOOTSTRAP_ROUNDS)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_moa_retrieval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen retrieval module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_gmt(path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    pathways: dict[str, set[str]] = {}
    genes: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 3 or not fields[0]:
            continue
        name = fields[0].strip()
        members = {x.strip().upper() for x in fields[2:] if x.strip()}
        if not members:
            continue
        pathways[name] = members
        for gene in members:
            genes.setdefault(gene, set()).add(name)
    if not pathways:
        raise ValueError(f"No pathway gene sets read from {path}")
    return pathways, genes


def bh(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    ranked = values[order] * n / np.arange(1, n + 1, dtype=np.float64)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def ap(relevant: np.ndarray, total_positive: int) -> float:
    ix = np.flatnonzero(relevant) + 1
    if total_positive <= 0 or len(ix) == 0:
        return 0.0
    return float((np.arange(1, len(ix) + 1, dtype=np.float64) / ix).sum() / total_positive)


def enrich(
    neighbor_genes: set[str], background_size: int, pathway_sizes: np.ndarray, gene_to_indices: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(neighbor_genes)
    p = np.ones(len(pathway_sizes), dtype=np.float64)
    hit = np.zeros(len(pathway_sizes), dtype=np.int32)
    if n == 0 or background_size == 0:
        return p, p.copy(), hit
    for gene in neighbor_genes:
        ix = gene_to_indices.get(gene)
        if ix is not None:
            hit[ix] += 1
    positive = hit > 0
    p[positive] = hypergeom.sf(hit[positive] - 1, background_size, pathway_sizes[positive], n)
    return p, bh(p), hit


def path_metrics(order: np.ndarray, known: set[str], names: list[str]) -> tuple[float, float, float]:
    relevant = np.asarray([names[i] in known for i in order], dtype=bool)
    ranks = np.flatnonzero(relevant) + 1
    return ap(relevant, len(known)), (float(1.0 / ranks[0]) if len(ranks) else 0.0), float(relevant[:10].sum() / len(known))


def stable(seed: int, label: str) -> int:
    h = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int((seed + int.from_bytes(h, "little")) % (2**63 - 1))


def bootstrap(values: np.ndarray, seed: int, label: str, rounds: int) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64)
    point = float(x.mean())
    rng = np.random.default_rng(stable(seed, label))
    ids = rng.integers(0, len(x), size=(rounds, len(x)), endpoint=False)
    sims = x[ids].mean(axis=1)
    return point, float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))


def csv_block(frame: pd.DataFrame) -> str:
    return "```csv\n" + frame.to_csv(index=False) + "```"


def candidates_from_structure(M, query: pd.DataFrame, gallery: pd.DataFrame, chemistry: dict) -> dict[tuple[str, str], np.ndarray]:
    bits, ds = chemistry.get("bits", {}), chemistry.get("DataStructs")
    by_dose = {d: g.profile_index.to_numpy(dtype=np.int64) for d, g in gallery.groupby("dose", sort=False)}
    compounds = gallery.compound_id.astype(str).to_numpy()
    out: dict[tuple[str, str], np.ndarray] = {}
    for compound, dose in query[["compound_id", "dose"]].astype(str).drop_duplicates().itertuples(index=False, name=None):
        ix = by_dose.get(dose, np.empty(0, dtype=np.int64))
        if compound not in bits or ds is None:
            out[(compound, dose)] = np.empty(0, dtype=np.int64)
            continue
        values = np.asarray([float(ds.TanimotoSimilarity(bits[compound], bits[compounds[j]])) if compounds[j] in bits else -np.inf for j in ix])
        order = np.lexsort((gallery.iloc[ix].gallery_id.astype(str).to_numpy(), -values))
        out[(compound, dose)] = ix[order]
    return out


def write_protocol(outdir: Path, library_name: str) -> None:
    text = f"""# Path-A protocol lock\n\n- Dataset: frozen cpg0004-LINCS.\n- Methods: M0 raw one-CP-support, M1 frozen teacher, M2 frozen GE-updated posterior; no fitting or parameter selection.\n- Query: test compounds only, two saved support rotations, at least five CP repeats.\n- Gallery: train plus validation compounds only, dose-matched CP consensus with at least three repeats.\n- Knowledge source: `{library_name}`, fixed before testing.\n- Target labels: Broad curated `target` and explicitly supplied `alternative_target` fields; case normalization only, no target inference.\n- Pathway inference: top-k neighbours -> union of their curated target genes -> pathway hypergeometric enrichment; fixed background is all library-mapped gallery target genes; BH FDR per query.\n- Similarity: PCC; k=25 primary, k=10/50 sensitivity.\n- Controls: shuffled compound-pathway labels, matched-foreign CP, random gallery neighbours, and structure-only Morgan/ECFP4.\n- Chemistry sensitivity: same Bemis-Murcko scaffold excluded, then Tanimoto >0.7 excluded.\n- Inference: 10,000 compound-level paired bootstrap resamples, three frozen seeds.\n"""
    (outdir / "protocol").mkdir(parents=True, exist_ok=True)
    (outdir / "protocol" / "PROTOCOL.md").write_text(text, encoding="utf-8")


def evaluate(
    seed: int, method: str, variant: str, query: pd.DataFrame, profiles: np.ndarray, gallery: pd.DataFrame,
    values: np.ndarray, candidate_cache: dict, target_by_gallery: dict[str, set[str]], known_by_query: dict[str, set[str]],
    background_size: int, pathway_sizes: np.ndarray, gene_to_indices: dict[str, np.ndarray], names: list[str], score_cache: dict[str, tuple[np.ndarray, np.ndarray]], top_out: list[dict],
) -> list[dict]:
    by_dose = {d: g.profile_index.to_numpy(dtype=np.int64) for d, g in gallery.groupby("dose", sort=False)}
    records: list[dict] = []
    q_ids = query.query_id.astype(str).to_numpy()
    for dose, pos in query.groupby("dose", sort=False).groups.items():
        positions, scores = score_cache[str(dose)]
        local_gallery = by_dose[str(dose)]
        for row_pos, score in zip(positions, scores):
            row = query.iloc[int(row_pos)]
            known = known_by_query.get(str(row.query_id), set())
            if not known or (method == "posterior_ge" and not bool(row.ge_available)):
                continue
            keep = candidate_cache.get((str(row.compound_id), str(row.dose)), np.ones(len(local_gallery), dtype=bool))
            local = np.flatnonzero(keep)
            if len(local) < max(KS):
                continue
            ranked_local = local[np.lexsort((gallery.iloc[local_gallery[local]].gallery_id.astype(str).to_numpy(), -score[local]))]
            ranked = local_gallery[ranked_local]
            for k in KS:
                selected = ranked[:k]
                genes = set().union(*(target_by_gallery[str(gallery.iloc[i].gallery_id)] for i in selected))
                p, qv, hits = enrich(genes, background_size, pathway_sizes, gene_to_indices)
                order = np.lexsort((np.asarray(names, dtype=str), p, qv))
                m_ap, mrr, r10 = path_metrics(order, known, names)
                records.append({"seed": seed, "query_id": str(row.query_id), "compound_id": str(row.compound_id), "dose": str(row.dose), "support_slot": int(row.support_slot), "support_row": int(row.support_row), "method": method, "variant": variant, "k": k, "pathway_map": m_ap, "mrr": mrr, "recall_at_10": r10, "candidate_count": int(len(ranked)), "known_pathway_count": len(known)})
                if k == PRIMARY_K:
                    for rank, i in enumerate(order[:10], start=1):
                        top_out.append({"seed": seed, "query_id": str(row.query_id), "compound_id": str(row.compound_id), "dose": str(row.dose), "method": method, "variant": variant, "rank": rank, "pathway": names[i], "p_value": p[i], "fdr_bh": qv[i], "target_gene_hits": int(hits[i]), "is_known_pathway": names[i] in known})
    return records


def controls(
    seed: int, query: pd.DataFrame, gallery: pd.DataFrame, raw_profiles: np.ndarray, target_by_gallery: dict[str, set[str]],
    known_by_query: dict[str, set[str]], background_size: int, pathway_sizes: np.ndarray, gene_to_indices: dict[str, np.ndarray], names: list[str], M, chemistry: dict,
) -> list[dict]:
    base = M.build_candidate_cache(query, gallery, "base", chemistry, {})
    scores_by_dose = {}
    by_dose = {d: g.profile_index.to_numpy(dtype=np.int64) for d, g in gallery.groupby("dose", sort=False)}
    for dose, positions in query.groupby("dose", sort=False).groups.items():
        positions = np.asarray(list(positions), dtype=np.int64)
        scores_by_dose[str(dose)] = (positions, M.pcc_matrix(raw_profiles[positions], np.vstack([np.zeros(1)]))) if False else None
    # Compute one base PCC matrix per dose for reused raw/foreign rankings.
    profile_scores = {str(d): (np.asarray(list(pos), dtype=np.int64), M.pcc_matrix(raw_profiles[np.asarray(list(pos), dtype=np.int64)], np.asarray([]))) for d, pos in []}
    # The explicit loop below avoids storing an unnecessary all-dose dense matrix.
    raw_rank: dict[int, np.ndarray] = {}
    for dose, pos in query.groupby("dose", sort=False).groups.items():
        pos = np.asarray(list(pos), dtype=np.int64); gi = by_dose[str(dose)]
        score = M.pcc_matrix(raw_profiles[pos], np.asarray(gallery_values_global[gi], dtype=np.float32))
        for rp, sv in zip(pos, score):
            keep = base[(str(query.iloc[rp].compound_id), str(query.iloc[rp].dose))]
            loc = np.flatnonzero(keep)
            raw_rank[int(rp)] = gi[loc[np.lexsort((gallery.iloc[gi[loc]].gallery_id.astype(str).to_numpy(), -sv[loc]))]]
    structures = candidates_from_structure(M, query, gallery, chemistry) if chemistry else {}
    compounds = sorted({str(x) for x in query.compound_id})
    rng = np.random.default_rng(stable(seed, "pathway-controls"))
    perm = dict(zip(compounds, rng.permutation(compounds)))
    by_compound = {str(c): list(ix) for c, ix in query.groupby("compound_id", sort=False).groups.items()}
    out: list[dict] = []
    for i, row in query.iterrows():
        known = known_by_query.get(str(row.query_id), set())
        if not known:
            continue
        cand = raw_rank.get(int(i), np.empty(0, dtype=np.int64))
        if len(cand) < PRIMARY_K:
            continue
        foreign_candidates = [j for j in range(len(query)) if str(query.iloc[j].dose) == str(row.dose) and str(query.iloc[j].compound_id) != str(row.compound_id) and int(query.iloc[j].repeat_count) == int(row.repeat_count)]
        foreign = raw_rank[foreign_candidates[stable(seed, str(row.query_id)) % len(foreign_candidates)]] if foreign_candidates else np.empty(0, dtype=np.int64)
        random_rank = rng.choice(cand, size=PRIMARY_K, replace=False)
        structure_rank = structures.get((str(row.compound_id), str(row.dose)), np.empty(0, dtype=np.int64))
        shuffled_known = known_by_query.get(next((str(query.iloc[j].query_id) for j in by_compound.get(perm[str(row.compound_id)], []) if str(query.iloc[j].dose) == str(row.dose)), ""), set())
        cases = {"shuffled_compound_labels": (cand, shuffled_known), "matched_foreign_cp": (foreign, known), "random_gallery_neighbors": (random_rank, known), "structure_only": (structure_rank, known)}
        for control, (ranked, truth) in cases.items():
            if len(ranked) < PRIMARY_K or not truth:
                continue
            gs = set().union(*(target_by_gallery[str(gallery.iloc[j].gallery_id)] for j in ranked[:PRIMARY_K]))
            p, qv, _ = enrich(gs, background_size, pathway_sizes, gene_to_indices); order = np.lexsort((np.asarray(names, dtype=str), p, qv))
            a, r, recall = path_metrics(order, truth, names)
            out.append({"seed": seed, "query_id": str(row.query_id), "compound_id": str(row.compound_id), "control": control, "pathway_map": a, "mrr": r, "recall_at_10": recall})
    return out


gallery_values_global: np.ndarray


def main() -> None:
    global gallery_values_global
    a = parse_args(); out = a.outdir
    if out.exists() and any(out.iterdir()) and not a.force:
        raise FileExistsError(f"Refusing nonempty output directory without --force: {out}")
    out.mkdir(parents=True, exist_ok=True); write_protocol(out, a.library_name)
    M = load_module(a.module)
    rows = M.load_rows(a.data); manifest = M.verify_manifest(rows, a.manifest); split = M.verify_split_lock(rows, a.split_lock)
    annotations, annotation_audit = M.load_annotations(a.annotations); smiles = M.load_smiles(a.smiles)
    gallery, _, _ = M.make_gallery(rows, manifest, annotations, smiles, a.min_gallery_repeats)
    gallery_values_global = np.vstack([rows["delta"][manifest[(manifest.compound_id.eq(r.compound_id)) & (manifest.dose.eq(r.dose))].row_index.to_numpy(dtype=np.int64)].mean(axis=0) for r in gallery.itertuples(index=False)]).astype(np.float32)
    pathways, pathway_by_gene = parse_gmt(a.reactome_gmt); names = sorted(pathways)
    ann = annotations.set_index("compound_id")
    target_by_gallery = {str(r.gallery_id): set(M.split_labels(ann.loc[str(r.compound_id), "target"])) if str(r.compound_id) in ann.index else set() for r in gallery.itertuples(index=False)}
    all_gallery_genes = set().union(*target_by_gallery.values()) if target_by_gallery else set()
    background = all_gallery_genes & set(pathway_by_gene)
    for key in target_by_gallery:
        target_by_gallery[key] &= background
    pathway_sizes = np.asarray([len(pathways[name] & background) for name in names], dtype=np.int64)
    gene_to_indices: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        for gene in pathways[name] & background:
            gene_to_indices.setdefault(gene, []).append(i)
    gene_to_indices = {gene: np.asarray(indices, dtype=np.int64) for gene, indices in gene_to_indices.items()}
    pa = out / "01_annotation_audit"; pa.mkdir(exist_ok=True)
    compound_rows = []
    for compound, r in ann.iterrows():
        genes = set(M.split_labels(r.target)); mapped = genes & set(pathway_by_gene)
        compound_rows.append({"compound_id": compound, "curated_target_genes": "|".join(sorted(genes)), "direct_mapped_target_genes": "|".join(sorted(mapped)), "target_status": "direct_curated" if mapped else ("missing_target" if not genes else "unmapped_target"), "known_reactome_pathway_count": len(set().union(*(pathway_by_gene.get(g, set()) for g in mapped)) if mapped else set())})
    pd.DataFrame(compound_rows).to_csv(pa / "compound_annotations.csv", index=False)
    pd.DataFrame([{"target_gene": g, "reactome_pathway_count": len(pathway_by_gene.get(g, set())), "reactome_pathways": "|".join(sorted(pathway_by_gene.get(g, set()))) } for g in sorted(all_gallery_genes)]).to_csv(pa / "target_annotations.csv", index=False)
    pd.DataFrame([{"pathway": n, "gene_count_library": len(pathways[n]), "gene_count_gallery_background": len(pathways[n] & background)} for n in names]).to_csv(pa / "pathway_annotations.csv", index=False)
    ge_counts = pd.read_csv(a.ge_root.parent / "artifact" / "ge_condition_manifest.csv") if (a.ge_root.parent / "artifact" / "ge_condition_manifest.csv").is_file() else pd.DataFrame()
    ge_counts.to_csv(pa / "ge_repeat_audit.csv", index=False)
    all_metrics: list[dict] = []; top: list[dict] = []; all_controls: list[dict] = []; exclusions: list[dict] = []
    eligible_rows: list[dict] = []
    for seed in SEEDS:
        vf, vp = M.read_prediction(a.virtual_root / f"seed{seed}_1r_all_v6" / "test_predictions.npz", "virtual")
        gf, gp = M.read_prediction(a.ge_root / f"all_b1_seed{seed}" / "test_predictions.npz", "ge")
        support = M.load_pair_support(a.pair_root / f"full_b1_all_seed{seed}" / "test_pair_manifest.csv")
        query, profiles, ex = M.make_queries(rows, manifest, split, annotations, smiles, vf, vp, gf, gp, support, a.min_query_repeats, 0.10)
        query.insert(0, "seed", seed); exclusions.extend([{"seed": seed, **x} for x in ex])
        known = {}
        for r in query.itertuples(index=False):
            genes = set(M.split_labels(r.target_labels)) & background
            known[str(r.query_id)] = set().union(*(pathway_by_gene[g] for g in genes)) if genes else set()
            eligible_rows.append({"seed": seed, "query_id": r.query_id, "compound_id": r.compound_id, "dose": r.dose, "support_row": r.support_row, "ge_available": r.ge_available, "known_pathway_count": len(known[str(r.query_id)])})
        full_query, full_profiles = query.copy(), profiles
        query = query[query.query_id.map(lambda x: bool(known.get(str(x), set())))].reset_index(drop=True)
        pos = {str(x): i for i, x in enumerate(full_query.query_id.astype(str))}; take = np.asarray([pos[str(x)] for x in query.query_id.astype(str)], dtype=np.int64)
        profiles = {k: v[take] for k, v in full_profiles.items()}
        pcc_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for method in METHODS:
            by_dose_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for dose, positions0 in query.groupby("dose", sort=False).groups.items():
                positions = np.asarray(list(positions0), dtype=np.int64)
                gallery_ix = gallery.loc[gallery.dose.eq(str(dose)), "profile_index"].to_numpy(dtype=np.int64)
                by_dose_cache[str(dose)] = (positions, M.pcc_matrix(profiles[method][positions], gallery_values_global[gallery_ix]))
            pcc_cache[method] = by_dose_cache
        chemistry, scaffolds, chemistry_error = M.optional_chemistry(smiles, set(gallery.compound_id) | set(query.compound_id))
        variants = ["base"] + ([] if chemistry_error else ["scaffold_excluded", "tanimoto_gt_0.7_excluded"])
        for variant in variants:
            cache = M.build_candidate_cache(query, gallery, variant, chemistry, scaffolds)
            for method in METHODS:
                all_metrics.extend(evaluate(seed, method, variant, query, profiles[method], gallery, gallery_values_global, cache, target_by_gallery, known, len(background), pathway_sizes, gene_to_indices, names, pcc_cache[method], top))
        all_controls.extend(controls(seed, query, gallery, profiles["raw"], target_by_gallery, known, len(background), pathway_sizes, gene_to_indices, names, M, chemistry))
        if chemistry_error:
            exclusions.append({"seed": seed, "scope": "chemistry_sensitivity", "reason": chemistry_error})
    audit = {"pathway_library": a.library_name, "pathway_gmt": str(a.reactome_gmt), "pathway_gmt_sha256": digest(a.reactome_gmt), "pathway_count": len(names), "pathway_gene_count": len(pathway_by_gene), "gallery_conditions": len(gallery), "gallery_compounds": int(gallery.compound_id.nunique()), "gallery_background_target_genes": len(background), "gate0_eligible_test_compounds": int(pd.DataFrame(eligible_rows).query("known_pathway_count > 0").compound_id.nunique()), "gate0": "PASS" if pd.DataFrame(eligible_rows).query("known_pathway_count > 0").compound_id.nunique() >= 100 else "FAIL", "annotation_source_sha256": digest(a.annotations), "frozen_input_sha256": {"data": digest(a.data), "manifest": digest(a.manifest), "split_lock": digest(a.split_lock)}}
    (pa / "AUDIT.md").write_text("# Annotation audit\n\n```json\n" + json.dumps(audit, indent=2, sort_keys=True) + "\n```\n", encoding="utf-8")
    pd.DataFrame(eligible_rows).to_csv(pa / "eligible_compounds.csv", index=False)
    pb = out / "02_pathA_curated_retrieval"; pb.mkdir(exist_ok=True); (pb / "figures").mkdir(exist_ok=True)
    metrics = pd.DataFrame(all_metrics); control = pd.DataFrame(all_controls); exclusions_frame = pd.DataFrame(exclusions)
    metrics.to_csv(pb / "METRICS_BY_QUERY.csv", index=False); control.to_csv(pb / "NEGATIVE_CONTROLS.csv", index=False); pd.DataFrame(top).to_csv(pb / "TOP_PATHWAYS.csv", index=False); exclusions_frame.to_csv(pb / "EXCLUSIONS.csv", index=False); gallery.to_csv(pb / "GALLERY.csv", index=False)
    summary = metrics.groupby(["seed", "method", "variant", "k"], as_index=False)[list(METRICS)].mean(); summary.to_csv(pb / "METRICS_BY_SEED.csv", index=False)
    comp = metrics.groupby(["seed", "compound_id", "method", "variant", "k"], as_index=False)[list(METRICS)].mean()
    contrasts: list[dict] = []
    for variant in sorted(metrics.variant.unique()):
        for k in KS:
            for left, right, label in (("teacher", "raw", "teacher_minus_raw"), ("posterior_ge", "teacher", "posterior_ge_minus_teacher")):
                for metric in METRICS:
                    per_seed = []
                    maps = {}
                    for seed in SEEDS:
                        x = comp[(comp.seed.eq(seed)) & (comp.variant.eq(variant)) & (comp.k.eq(k)) & (comp.method.eq(left))].set_index("compound_id")[metric]
                        y = comp[(comp.seed.eq(seed)) & (comp.variant.eq(variant)) & (comp.k.eq(k)) & (comp.method.eq(right))].set_index("compound_id")[metric]
                        common = sorted(set(x.index) & set(y.index)); maps[seed] = pd.Series({c: float(x[c] - y[c]) for c in common}); per_seed.append(float(maps[seed].mean()) if common else float("nan"))
                    common_all = sorted(set.intersection(*(set(x.index) for x in maps.values())))
                    vals = np.asarray([np.mean([maps[s][c] for s in SEEDS]) for c in common_all], dtype=np.float64)
                    point, low, high = bootstrap(vals, 3407, f"{variant}|{k}|{label}|{metric}", a.bootstrap_rounds)
                    contrasts.append({"variant": variant, "k": k, "contrast": label, "metric": metric, "seed3407": per_seed[0], "seed42": per_seed[1], "seed2025": per_seed[2], "compound_count": len(vals), "point": point, "ci_low": low, "ci_high": high, "rounds": a.bootstrap_rounds})
    contrast = pd.DataFrame(contrasts); contrast.to_csv(pb / "PAIRED_CONTRASTS.csv", index=False); contrast.to_csv(pb / "BOOTSTRAP_CI.csv", index=False)
    primary = contrast[(contrast.variant.eq("base")) & (contrast.k.eq(PRIMARY_K)) & (contrast.metric.eq("pathway_map"))]
    decisions = []
    for label in ("teacher_minus_raw", "posterior_ge_minus_teacher"):
        x = primary[primary.contrast.eq(label)].iloc[0]
        go = all(float(x[c]) > 0 for c in ("seed3407", "seed42", "seed2025")) and float(x.ci_low) > 0
        decisions.append({"comparison": label, "decision": "GO" if go else "NO-GO", "point": float(x.point), "ci_low": float(x.ci_low), "ci_high": float(x.ci_high), "n_compounds": int(x.compound_count)})
    (pb / "DECISION.md").write_text("# Path-A decision\n\n" + csv_block(pd.DataFrame(decisions)) + "\n", encoding="utf-8")
    (pb / "RESULTS.md").write_text(f"# Path-A curated {a.library_name} retrieval\n\n## Objective\n\nRecover independently curated target-derived pathways from CP-neighbour target enrichment.\n\n## Eligibility and leakage audit\n\n```json\n" + json.dumps(audit, indent=2, sort_keys=True) + "\n```\n\nTest compounds never enter the gallery; M2 is reconstructed only from frozen arrays, and pathway annotations never enter model construction.\n\n## Primary results\n\n" + csv_block(primary) + "\n\n## Decisions\n\n" + csv_block(pd.DataFrame(decisions)) + "\n\nAll per-query, negative-control, seed-level, and paired-bootstrap details are saved alongside this report.\n", encoding="utf-8")
    config = {"version": f"cpg0004-PathA-{a.library_name}-2026-08-30", "library": a.library_name, "seeds": SEEDS, "k": KS, "primary_k": PRIMARY_K, "bootstrap_rounds": a.bootstrap_rounds, "pathway_gmt_sha256": digest(a.reactome_gmt), "guardrails": ["frozen profiles only", "no pathway/model tuning", "test absent from gallery", "PCC fixed", "BH FDR per query", "compound-level paired bootstrap"]}
    (out / "protocol" / "CONFIG.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (out / "FINAL_SUMMARY.md").write_text("# Pathway validation status\n\n| Level | Status |\n|---|---|\n| Path-A | completed; see `02_pathA_curated_retrieval/DECISION.md` |\n| Path-B | pending strict held-out GE implementation |\n| Path-C | pending pathway trajectory implementation |\n", encoding="utf-8")
    print(json.dumps({"outdir": str(out), "gate0": audit["gate0"], "eligible_compounds": audit["gate0_eligible_test_compounds"], "decisions": decisions}, indent=2))


if __name__ == "__main__":
    main()
