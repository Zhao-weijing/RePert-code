"""Frozen 1R CFRA confirmation for sci-Plex3.

This consumer starts from the already audited sci-Plex3 24 h pseudo-bulks
(``group_counts.h5``) and the GEO-corrected feature map made by
``run_raw_same_foreign.py``.  The only expression rows loaded before weight
freezing are base and calibration drugs.  Base-only per-cell-line HGV2000,
Ridge IMR/LSO and empirical-Bayes IMCEB are fit first; the CFRA simplex
weight is selected only from calibration drugs.  Confirmation profiles are
then read in a separate final pass and evaluated once.

Sci-Plex3 contains only two genuine biological repeats.  Consequently this
script emits a legal 1R confirmation only.  It deliberately does not create
or report any 2R number; ``SCI_2R_BLOCKED.md`` records the block.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SEED = 3407
BOOT = 10_000
REPS = ("rep1", "rep2")
N_HVG = 2000
N_FOREIGN = 20
METHODS = ("Raw", "IMR", "LSO", "IMCEB", "CFRA")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - float(a.mean())
    b = b - float(b.mean())
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den > 1e-12 else float("nan")


def fisher(r: float) -> float:
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999))) if np.isfinite(r) else float("nan")


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(x) for x in parts).encode("utf-8")
    return SEED + int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def norm_delta(t_counts: np.ndarray, c_counts: np.ndarray) -> np.ndarray:
    t = np.asarray(t_counts, dtype=np.float64)
    c = np.asarray(c_counts, dtype=np.float64)
    ts, cs = float(t.sum()), float(c.sum())
    if ts <= 0 or cs <= 0:
        raise RuntimeError("zero library size in pseudo-bulk")
    return (np.log1p(t / ts * 1e6) - np.log1p(c / cs * 1e6)).astype(np.float32)


def read_conditions(path: Path) -> list[dict[str, Any]]:
    rows = []
    for x in csv.DictReader(path.open(encoding="utf-8", newline="")):
        x["dose_nM"] = float(x["dose_nM"])
        x["group_id"] = int(x["group_id"])
        x["control_group_id"] = int(x["control_group_id"])
        x["n_cells"] = int(x["n_cells"])
        x["control_n_cells"] = int(x["control_n_cells"])
        rows.append(x)
    if not rows:
        raise RuntimeError("empty condition table")
    return rows


def split_drugs(drugs: list[str]) -> dict[str, list[str]]:
    drugs = sorted(set(drugs))
    rng = np.random.default_rng(SEED)
    shuffled = list(np.asarray(drugs)[rng.permutation(len(drugs))])
    nb = int(round(0.60 * len(shuffled)))
    nc = int(round(0.20 * len(shuffled)))
    nb = max(1, min(nb, len(shuffled) - 2))
    nc = max(1, min(nc, len(shuffled) - nb - 1))
    return {"base": sorted(shuffled[:nb]), "calibration": sorted(shuffled[nb : nb + nc]), "confirmation": sorted(shuffled[nb + nc :])}


def fit_hvg_and_profiles(
    group_counts,
    condition_rows: list[dict[str, Any]],
    base_drugs: set[str],
    cell_line: str,
) -> tuple[np.ndarray, dict[tuple[str, float, str], np.ndarray], dict[str, Any]]:
    """Choose HVGs and load only base profiles for one cell line."""
    base_rows = [r for r in condition_rows if r["drug"] in base_drugs and r["cell_line"] == cell_line]
    paired = {(r["drug"], r["dose_nM"]) for r in base_rows if r["replicate"] == "rep1"}
    paired &= {(r["drug"], r["dose_nM"]) for r in base_rows if r["replicate"] == "rep2"}
    base_rows = [r for r in base_rows if (r["drug"], r["dose_nM"]) in paired]
    if len(base_rows) < 10:
        raise RuntimeError(f"too few base rows for {cell_line}: {len(base_rows)}")
    n_vars = int(group_counts.shape[1])
    mean = np.zeros(n_vars, dtype=np.float64)
    m2 = np.zeros(n_vars, dtype=np.float64)
    n = 0
    for r in base_rows:
        d = norm_delta(group_counts[int(r["group_id"]), :], group_counts[int(r["control_group_id"]), :]).astype(np.float64)
        n += 1
        delta = d - mean
        mean += delta / n
        m2 += delta * (d - mean)
    var = m2 / max(n - 1, 1)
    hvg = np.sort(np.argsort(var, kind="mergesort")[-N_HVG:]).astype(np.int64)
    prof: dict[tuple[str, float, str], np.ndarray] = {}
    for r in base_rows:
        key = (r["drug"], float(r["dose_nM"]), r["replicate"])
        prof[key] = norm_delta(group_counts[int(r["group_id"]), :], group_counts[int(r["control_group_id"]), :])[hvg]
    return hvg, prof, {"n_base_condition_rows": len(base_rows), "n_base_paired_dose_units": len(paired), "variance_feature_count": n_vars}


def load_profiles(
    group_counts,
    condition_rows: list[dict[str, Any]],
    drugs: set[str],
    cell_line: str,
    hvg: np.ndarray,
) -> dict[tuple[str, float, str], np.ndarray]:
    out: dict[tuple[str, float, str], np.ndarray] = {}
    rows = [r for r in condition_rows if r["drug"] in drugs and r["cell_line"] == cell_line]
    for r in rows:
        key = (r["drug"], float(r["dose_nM"]), r["replicate"])
        if key in out:
            raise RuntimeError(f"duplicate condition {cell_line}/{key}")
        out[key] = norm_delta(group_counts[int(r["group_id"]), :], group_counts[int(r["control_group_id"]), :])[hvg]
    return out


def fit_estimators(base_profiles: dict[tuple[str, float, str], np.ndarray], hvg: np.ndarray):
    from sklearn.linear_model import Ridge

    units = sorted({(g, d) for g, d, _ in base_profiles})
    x, y = [], []
    for g, d in units:
        for s, t in (("rep1", "rep2"), ("rep2", "rep1")):
            if (g, d, s) in base_profiles and (g, d, t) in base_profiles:
                x.append(base_profiles[(g, d, s)])
                y.append(base_profiles[(g, d, t)])
    if len(x) < 4:
        raise RuntimeError("insufficient paired base examples")
    imr = Ridge(alpha=10.0).fit(np.asarray(x), np.asarray(y))
    # With exactly two true repeats, a leave-support-out target has no
    # independent second support repeat.  The protocol therefore freezes LSO
    # to the same two-repeat Ridge map as IMR, and records this degeneracy.
    lso = Ridge(alpha=10.0).fit(np.asarray(x), np.asarray(y))
    means = np.stack([(base_profiles[(g, d, "rep1")] + base_profiles[(g, d, "rep2")]) / 2.0 for g, d in units])
    within = np.mean(np.stack([np.var(np.stack([base_profiles[(g, d, "rep1")], base_profiles[(g, d, "rep2")]]), axis=0) for g, d in units]), axis=0)
    between = np.var(means, axis=0)
    shrink = between / (between + within + 1e-8)
    return imr, lso, shrink, {"n_paired_examples": len(x), "ridge_alpha": 10.0, "lso_equals_imr": True, "shrink_mean": float(np.mean(shrink))}


def foreign_vectors(
    profile_pool: dict[tuple[str, float, str], np.ndarray],
    target_drug: str,
    cell_line: str,
    dose: float,
    held: str,
    token: str,
) -> tuple[list[str], list[np.ndarray]]:
    donors = sorted({g for g, d, r in profile_pool if d == dose and r == held and g != target_drug})
    if len(donors) < N_FOREIGN:
        raise RuntimeError(f"foreign donor pool <{N_FOREIGN}: {cell_line} dose={dose} target={target_drug} n={len(donors)}")
    rng = np.random.default_rng(stable_seed("foreign", token))
    selected = [str(x) for x in rng.choice(np.asarray(donors, dtype=str), N_FOREIGN, replace=False)]
    return selected, [profile_pool[(g, dose, held)] for g in selected]


def score(pred: np.ndarray, true: np.ndarray, foreign: list[np.ndarray]) -> float:
    same = fisher(corr(pred, true))
    fz = [fisher(corr(pred, z)) for z in foreign]
    return same - float(np.mean(fz))


def select_weights(
    cal_profiles: dict[tuple[str, float, str], np.ndarray],
    all_cal_pool: dict[tuple[str, float, str], np.ndarray],
    cal_drugs: set[str],
    imr,
    lso,
    shrink: np.ndarray,
    hvg: np.ndarray,
    cell_line: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    # Grid exactly follows the existing Papalexi frozen development code.
    examples = []
    for drug, dose in sorted({(g, d) for g, d, _ in cal_profiles}):
        for support, held in (("rep1", "rep2"), ("rep2", "rep1")):
            if (drug, dose, support) not in cal_profiles or (drug, dose, held) not in cal_profiles:
                continue
            raw = cal_profiles[(drug, dose, support)].astype(np.float64)
            true = cal_profiles[(drug, dose, held)].astype(np.float64)
            donors, foreign = foreign_vectors(all_cal_pool, drug, cell_line, dose, held, f"cal|{drug}|{cell_line}|{dose}|{support}|{held}")
            p_imr = np.asarray(imr.predict(raw[None, :])[0], dtype=np.float64)
            p_lso = np.asarray(lso.predict(raw[None, :])[0], dtype=np.float64)
            p_eb = raw * shrink
            examples.append((p_imr, p_lso, p_eb, true, foreign))
    if not examples:
        raise RuntimeError(f"no calibration examples for weight selection {cell_line}")
    best_score, best_w = -np.inf, None
    for ia in range(21):
        a = ia / 20.0
        for ib in range(21 - ia):
            b = ib / 20.0
            c = 1.0 - a - b
            w = np.asarray([a, b, c], dtype=np.float64)
            vals = [score(w[0] * p[0] + w[1] * p[1] + w[2] * p[2], p[3], p[4]) for p in examples]
            s = float(np.mean(vals))
            if s > best_score + 1e-15:
                best_score, best_w = s, w
    if best_w is None:
        raise RuntimeError("simplex selection failed")
    return best_w, {"n_calibration_examples": len(examples), "calibration_mean_same_foreign": best_score, "grid_step": 0.05, "estimators_order": ["IMR", "LSO", "IMCEB"]}


def bootstrap(values: list[float], drugs: list[str], seed: int) -> dict[str, Any]:
    by: dict[str, list[float]] = defaultdict(list)
    for v, d in zip(values, drugs):
        if np.isfinite(v):
            by[str(d)].append(float(v))
    keys = sorted(by)
    x = np.asarray([np.mean(by[k]) for k in keys], dtype=np.float64)
    if not len(x):
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_drugs": 0, "bootstrap_draws": BOOT}
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(x), size=(BOOT, len(x)))
    means = x[draw].mean(axis=1)
    return {"estimate": float(x.mean()), "ci_low": float(np.quantile(means, .025)), "ci_high": float(np.quantile(means, .975)), "n_drugs": int(len(x)), "bootstrap_draws": BOOT}


def paired_bootstrap(cfra: list[float], raw: list[float], drugs: list[str], seed: int) -> dict[str, Any]:
    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for c, r, d in zip(cfra, raw, drugs):
        if np.isfinite(c) and np.isfinite(r):
            by[str(d)].append((float(c), float(r)))
    keys = sorted(by)
    c = np.asarray([np.mean([x[0] for x in by[k]]) for k in keys], dtype=np.float64)
    r = np.asarray([np.mean([x[1] for x in by[k]]) for k in keys], dtype=np.float64)
    d = c - r
    if not len(d):
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_drugs": 0, "bootstrap_draws": BOOT}
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(d), size=(BOOT, len(d)))
    vals = d[draw].mean(axis=1)
    return {"estimate": float(d.mean()), "ci_low": float(np.quantile(vals, .025)), "ci_high": float(np.quantile(vals, .975)), "n_drugs": int(len(d)), "bootstrap_draws": BOOT}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-counts", type=Path, required=True)
    ap.add_argument("--conditions", type=Path, required=True)
    ap.add_argument("--raw-gate", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    import h5py

    conditions = read_conditions(args.conditions)
    drugs = sorted({r["drug"] for r in conditions})
    split = split_drugs(drugs)
    base, cal, conf = map(set, (split["base"], split["calibration"], split["confirmation"]))
    if base & cal or base & conf or cal & conf:
        raise RuntimeError("drug split overlap")
    if len(base) + len(cal) + len(conf) != len(drugs):
        raise RuntimeError("drug split is not exhaustive")

    # Map condition identity exactly once and reject duplicates before any fit.
    cond_map: dict[tuple[str, str, float, str], dict[str, Any]] = {}
    for r in conditions:
        key = (r["drug"], r["cell_line"], float(r["dose_nM"]), r["replicate"])
        if key in cond_map:
            raise RuntimeError(f"duplicate condition unit {key}")
        cond_map[key] = r
    cell_lines = sorted({r["cell_line"] for r in conditions})
    models: dict[str, Any] = {}
    calibration_records: list[dict[str, Any]] = []
    hvg_records = []
    with h5py.File(args.group_counts, "r") as gh:
        counts = gh["counts"]
        for cell in cell_lines:
            hvg, base_profiles, hvg_audit = fit_hvg_and_profiles(counts, conditions, base, cell)
            cal_profiles = load_profiles(counts, conditions, cal, cell, hvg)
            # Calibration foreign donors are restricted to base+calibration.
            pool = dict(base_profiles)
            pool.update(cal_profiles)
            imr, lso, shrink, fit_audit = fit_estimators(base_profiles, hvg)
            w, w_audit = select_weights(cal_profiles, pool, cal, imr, lso, shrink, hvg, cell)
            models[cell] = {"hvg": hvg, "base_profiles": base_profiles, "cal_profiles": cal_profiles, "pool": pool, "imr": imr, "lso": lso, "shrink": shrink, "weights": w}
            np.savetxt(args.outdir / f"hvg2000_{cell}.txt", hvg, fmt="%d")
            hvg_records.append({"cell_line": cell, "n_hvg": len(hvg), **hvg_audit})
            calibration_records.append({"cell_line": cell, "weights_IMR": w[0], "weights_LSO": w[1], "weights_IMCEB": w[2], **w_audit, **fit_audit})

        # Confirmation is not loaded before this point.  This separate block
        # is the only expression read for confirmation evaluation.
        eval_rows: list[dict[str, Any]] = []
        confirmation_read_once = True
        for cell in cell_lines:
            model = models[cell]
            hvg = model["hvg"]
            conf_profiles = load_profiles(counts, conditions, conf, cell, hvg)
            for drug, dose in sorted({(g, d) for g, d, _ in conf_profiles}):
                for support, held in (("rep1", "rep2"), ("rep2", "rep1")):
                    key_s = (drug, float(dose), support)
                    key_h = (drug, float(dose), held)
                    if key_s not in conf_profiles or key_h not in conf_profiles:
                        continue
                    raw = np.asarray(conf_profiles[key_s], dtype=np.float64)
                    true = np.asarray(conf_profiles[key_h], dtype=np.float64)
                    donor_ids, foreign = foreign_vectors(model["pool"], drug, cell, float(dose), held, f"confirm|{drug}|{cell}|{dose}|{support}|{held}")
                    p_imr = np.asarray(model["imr"].predict(raw[None, :])[0], dtype=np.float64)
                    p_lso = np.asarray(model["lso"].predict(raw[None, :])[0], dtype=np.float64)
                    p_eb = raw * model["shrink"]
                    w = np.asarray(model["weights"], dtype=np.float64)
                    cfra = w[0] * p_imr + w[1] * p_lso + w[2] * p_eb
                    r_score = score(raw, true, foreign)
                    c_score = score(cfra, true, foreign)
                    eval_rows.append({
                        "drug": drug, "cell_line": cell, "dose_nM": dose,
                        "support_repeat": support, "held_repeat": held,
                        "n_support_cells": cond_map[(drug, cell, float(dose), support)]["n_cells"],
                        "n_held_cells": cond_map[(drug, cell, float(dose), held)]["n_cells"],
                        "foreign_n": len(foreign), "foreign_drugs": ";".join(sorted(donor_ids)),
                        "raw_target_fisher_z": fisher(corr(raw, true)),
                        "raw_foreign_mean_fisher_z": float(np.mean([fisher(corr(raw, z)) for z in foreign])),
                        "raw_same_foreign_excess_fisher_z": r_score,
                        "cfra_target_fisher_z": fisher(corr(cfra, true)),
                        "cfra_foreign_mean_fisher_z": float(np.mean([fisher(corr(cfra, z)) for z in foreign])),
                        "cfra_same_foreign_excess_fisher_z": c_score,
                        "cfra_minus_raw_same_foreign_excess_fisher_z": c_score - r_score,
                        "norm_ratio_cfra_over_raw": float(np.linalg.norm(cfra) / max(np.linalg.norm(raw), 1e-12)),
                    })
        if not eval_rows:
            raise RuntimeError("no confirmation 1R rows")

    write_csv(args.outdir / "confirmation_1R_same_foreign_per_condition.csv", eval_rows)
    summary_rows: list[dict[str, Any]] = []
    scopes = [("overall", "ALL", eval_rows)] + [("cell_line", c, [r for r in eval_rows if r["cell_line"] == c]) for c in cell_lines]
    abs_keys = {
        "raw_same_foreign_excess_fisher_z": "Raw Same-Foreign excess Fisher-z",
        "cfra_same_foreign_excess_fisher_z": "CFRA Same-Foreign excess Fisher-z",
        "norm_ratio_cfra_over_raw": "||CFRA||/||Raw||",
    }
    for scope, label, rows in scopes:
        for key, metric in abs_keys.items():
            summary_rows.append({"scope": scope, "cell_line": label, "metric": metric, "method": "Raw" if key.startswith("raw") else "CFRA" if key.startswith("cfra") else "CFRA", **bootstrap([float(r[key]) for r in rows], [r["drug"] for r in rows], stable_seed("bootstrap", scope, label, key))})
        summary_rows.append({"scope": scope, "cell_line": label, "metric": "CFRA-Raw Same-Foreign excess Fisher-z", "method": "CFRA-Raw", **paired_bootstrap([float(r["cfra_same_foreign_excess_fisher_z"]) for r in rows], [float(r["raw_same_foreign_excess_fisher_z"]) for r in rows], [r["drug"] for r in rows], stable_seed("paired_bootstrap", scope, label))})
    write_csv(args.outdir / "confirmation_1R_summary.csv", summary_rows)

    # Gate mirrors the requested final endpoint; Raw absolute result is also
    # reported, but only CFRA-Raw is a model comparison.
    overall_rows = [r for r in eval_rows]
    overall_delta = paired_bootstrap([r["cfra_same_foreign_excess_fisher_z"] for r in overall_rows], [r["raw_same_foreign_excess_fisher_z"] for r in overall_rows], [r["drug"] for r in overall_rows], stable_seed("paired_bootstrap", "overall"))
    by_cell_delta = {}
    for cell in cell_lines:
        rows = [r for r in eval_rows if r["cell_line"] == cell]
        by_cell_delta[cell] = paired_bootstrap([r["cfra_same_foreign_excess_fisher_z"] for r in rows], [r["raw_same_foreign_excess_fisher_z"] for r in rows], [r["drug"] for r in rows], stable_seed("paired_bootstrap", cell))

    raw_gate = json.loads(args.raw_gate.read_text(encoding="utf-8"))
    audit = {
        "dataset": "sci-Plex3 Zenodo 7041849",
        "input_md5_expected": "d1f51b9f8de35ca07638132539da9a99",
        "time_hours": 24,
        "nperts": 1,
        "n_min": 50,
        "split_seed": SEED,
        "split_drug_level": True,
        "split_counts": {k: len(v) for k, v in split.items()},
        "split_lists": split,
        "split_overlap": {"base_calibration": sorted(base & cal), "base_confirmation": sorted(base & conf), "calibration_confirmation": sorted(cal & conf)},
        "hvg": {"source": "base-only per cell line", "n": N_HVG, "files": [str(args.outdir / f"hvg2000_{c}.txt") for c in cell_lines], "records": hvg_records},
        "models": {"fit_source": "base-only", "estimators": ["Ridge IMR", "Ridge LSO", "IMCEB", "CFRA"], "ridge_alpha": 10.0, "lso_policy": "LSO=IMR because only two true biological repeats; independent leave-support-out target unavailable"},
        "weights": {"source": "calibration-only", "records": calibration_records},
        "confirmation": {"expression_read_after_freeze": True, "expression_read_once": confirmation_read_once, "n_condition_rows": len(eval_rows), "n_drugs": len({r["drug"] for r in eval_rows}), "n_cell_lines": len(cell_lines)},
        "fairness": {"same_support_held_for_raw_cfra": True, "same_foreign_pool_for_raw_cfra": True, "foreign_pool": "20 base+calibration donors matched cell line/dose/held, excluding target", "cell_level_ci": False, "bootstrap_group": "drug", "bootstrap_draws": BOOT},
        "control_policy": "same-plate same-cell-line same-repeat Vehicle; inherited from eligible_conditions_n50.csv",
        "feature_map": "GEO annotation row i -> X[:,i] for i=0..110982; X[:,110983] all-zero tail excluded",
        "raw_phase0_gate": raw_gate,
        "two_r": {"status": "BLOCKED", "reason": "only rep1 and rep2 are genuine biological repeats; no independent third held repeat"},
        "overall_cfra_minus_raw": overall_delta,
        "cell_line_cfra_minus_raw": by_cell_delta,
    }
    write_json(args.outdir / "SCI_CONFIRMATION_AUDIT.json", audit)
    write_json(args.outdir / "split_lock.json", {"seed": SEED, **split, "confirmation_loaded": True, "confirmation_loaded_only_after_freeze": True})
    write_csv(args.outdir / "calibration_weights.csv", calibration_records)

    block = [
        "# sci-Plex3 2R blocked",
        "",
        "Sci-Plex3 has only two genuine biological replicates (`rep1`, `rep2`). A legal 2R evaluation requires two support biological repeats and an independent held-out third repeat. That design is unavailable here.",
        "",
        "No two-repeat average was compared with either repeat, and no 2R value, CI, or gate was produced. The only model confirmation reported for sci-Plex3 is the legal 1R rep1→rep2 / rep2→rep1 evaluation.",
    ]
    (args.outdir / "SCI_2R_BLOCKED.md").write_text("\n".join(block) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = cell_lines + ["ALL"]
        raw_s = []
        cfra_s = []
        delta_s = []
        for c in cell_lines:
            rr = [r for r in eval_rows if r["cell_line"] == c]
            raw_s.append(bootstrap([r["raw_same_foreign_excess_fisher_z"] for r in rr], [r["drug"] for r in rr], stable_seed("plot", c, "raw")))
            cfra_s.append(bootstrap([r["cfra_same_foreign_excess_fisher_z"] for r in rr], [r["drug"] for r in rr], stable_seed("plot", c, "cfra")))
            delta_s.append(by_cell_delta[c])
        raw_s.append(bootstrap([r["raw_same_foreign_excess_fisher_z"] for r in eval_rows], [r["drug"] for r in eval_rows], stable_seed("plot", "all", "raw")))
        cfra_s.append(bootstrap([r["cfra_same_foreign_excess_fisher_z"] for r in eval_rows], [r["drug"] for r in eval_rows], stable_seed("plot", "all", "cfra")))
        delta_s.append(overall_delta)
        x = np.arange(len(labels)); width = .28
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for j, (name, stats, color) in enumerate((("Raw", raw_s, "#6b7280"), ("CFRA", cfra_s, "#b91c1c"), ("CFRA-Raw", delta_s, "#2563eb"))):
            y = np.asarray([s["estimate"] for s in stats]); lo = np.asarray([s["ci_low"] for s in stats]); hi = np.asarray([s["ci_high"] for s in stats])
            xx = x + (j - 1) * width
            ax.errorbar(xx, y, yerr=np.vstack([y - lo, hi - y]), fmt="o", color=color, label=name, capsize=3)
        ax.axhline(0, color="#374151", lw=.8); ax.set_xticks(x, labels); ax.set_ylabel("Same–Foreign excess Fisher-z"); ax.set_title("sci-Plex3 1R frozen confirmation"); ax.grid(axis="y", alpha=.2); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(args.outdir / "confirmation_1R_same_foreign.png", dpi=220); fig.savefig(args.outdir / "confirmation_1R_same_foreign.pdf"); plt.close(fig)
    except Exception as e:
        write_json(args.outdir / "figure_error.json", {"error": repr(e)})

    lines = [
        "# sci-Plex3 frozen 1R CFRA confirmation",
        "",
        "## Design audit",
        "",
        f"- 24 h only; nperts=1; treated condition n≥50; matched same-plate Vehicle; two ordered biological-repeat rotations.",
        f"- Drug split seed 3407: base/calibration/confirmation = {len(base)}/{len(cal)}/{len(conf)}; no overlap.",
        "- HGV2000 and Ridge estimators use base drugs only. CFRA simplex weights use calibration drugs only. Confirmation expression is read only after those quantities are frozen.",
        "- Raw and CFRA share each support/held pair and the same 20 base+calibration foreign donors; bootstrap unit is drug.",
        "",
        "## 1R CFRA−Raw",
        "",
        f"- Overall estimate={overall_delta['estimate']:.6f}, 95% CI [{overall_delta['ci_low']:.6f}, {overall_delta['ci_high']:.6f}], n_drugs={overall_delta['n_drugs']}.",
    ]
    for c in cell_lines:
        s = by_cell_delta[c]
        lines.append(f"- {c}: estimate={s['estimate']:.6f}, 95% CI [{s['ci_low']:.6f}, {s['ci_high']:.6f}], n_drugs={s['n_drugs']}.")
    lines += ["", "## 2R", "", "See `SCI_2R_BLOCKED.md`: no independent third biological repeat exists; no 2R number is reported."]
    (args.outdir / "SCI_CONFIRMATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
