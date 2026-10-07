"""Frozen sci-Plex3 cross-cell-line CFRA generalization.

This experiment asks whether a CFRA map fitted and calibrated in one cell
line can be applied, without refitting, to a different cell line.  There are
only two genuine biological repeats in sci-Plex3, so the endpoint is the
legal 1R repeat rotation (rep1 -> rep2 and rep2 -> rep1).  A source model is
fit on source-cell-line base drugs and its simplex weights are selected on
source-cell-line calibration drugs.  The model is then frozen and applied to
target-cell-line confirmation drugs.

Important leakage rules:

* the global drug split is read from the previously frozen split_lock.json;
* source HVGs, Ridge maps, EB shrinkage and CFRA weights use source base or
  source calibration data only;
* target confirmation expression is loaded only after every source model is
  frozen; target base/calibration profiles are used only as metric foreign
  donors and never to fit or calibrate a source model;
* Raw and CFRA for a source->target pair use the same target support/held
  rotation and the same foreign donor IDs.  Both are evaluated in the
  source-frozen HGV2000 feature space, which is the only feature space the
  frozen source model can legally consume.

Inference is at the drug level.  Dose and ordered-repeat rows are averaged
  within drug before 10,000-draw bootstrap.  The target-level summary first
  averages source models within target and then bootstraps drugs, avoiding
  pseudo-replication of the same target biology.  The overall macro summary
  gives each target cell line equal weight and uses a hierarchical bootstrap
  (target cell line, then drug).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SEED = 3407
BOOT = 10_000
REPS = ("rep1", "rep2")
N_HVG = 2_000
N_FOREIGN = 20
RIDGE_ALPHA = 10.0
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(x) for x in parts).encode("utf-8")
    return SEED + int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    x = x - float(x.mean())
    y = y - float(y.mean())
    den = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y / den) if den > 1e-12 else float("nan")


def fisher(r: float) -> float:
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999))) if np.isfinite(r) else float("nan")


def norm_delta(t_counts: np.ndarray, c_counts: np.ndarray) -> np.ndarray:
    t = np.asarray(t_counts, dtype=np.float64)
    c = np.asarray(c_counts, dtype=np.float64)
    tsum, csum = float(t.sum()), float(c.sum())
    if tsum <= 0 or csum <= 0:
        raise RuntimeError("zero library size in pseudo-bulk")
    return (np.log1p(t / tsum * 1e6) - np.log1p(c / csum * 1e6)).astype(np.float32)


def read_conditions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["dose_nM"] = float(row["dose_nM"])
            row["group_id"] = int(row["group_id"])
            row["control_group_id"] = int(row["control_group_id"])
            row["n_cells"] = int(row["n_cells"])
            row["control_n_cells"] = int(row["control_n_cells"])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty condition table: {path}")
    return rows


def read_split(path: Path, all_drugs: set[str]) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    split = {k: [str(x) for x in raw[k]] for k in ("base", "calibration", "confirmation")}
    sets = {k: set(v) for k, v in split.items()}
    if any(sets[a] & sets[b] for a, b in (("base", "calibration"), ("base", "confirmation"), ("calibration", "confirmation"))):
        raise RuntimeError("frozen split has overlap")
    union = sets["base"] | sets["calibration"] | sets["confirmation"]
    if union != set(all_drugs):
        missing, extra = sorted(set(all_drugs) - union), sorted(union - set(all_drugs))
        raise RuntimeError(f"frozen split does not match condition drugs; missing={missing}; extra={extra}")
    return {k: sorted(v) for k, v in split.items()}


def profile_from_row(counts: Any, row: dict[str, Any], hvg: np.ndarray | None = None) -> np.ndarray:
    x = norm_delta(counts[int(row["group_id"]), :], counts[int(row["control_group_id"]), :])
    return x if hvg is None else x[hvg]


def rows_for(condition_rows: list[dict[str, Any]], drugs: set[str], cell_line: str) -> list[dict[str, Any]]:
    return [r for r in condition_rows if r["drug"] in drugs and r["cell_line"] == cell_line]


def load_profiles(counts: Any, condition_rows: list[dict[str, Any]], drugs: set[str], cell_line: str, hvg: np.ndarray) -> dict[tuple[str, float, str], np.ndarray]:
    out: dict[tuple[str, float, str], np.ndarray] = {}
    for r in rows_for(condition_rows, drugs, cell_line):
        key = (r["drug"], float(r["dose_nM"]), r["replicate"])
        if key in out:
            raise RuntimeError(f"duplicate profile key {cell_line}/{key}")
        out[key] = profile_from_row(counts, r, hvg)
    return out


def fit_hvg_and_base(counts: Any, condition_rows: list[dict[str, Any]], base_drugs: set[str], cell_line: str) -> tuple[np.ndarray, dict[tuple[str, float, str], np.ndarray], dict[str, Any]]:
    base_rows = rows_for(condition_rows, base_drugs, cell_line)
    units = sorted({(r["drug"], float(r["dose_nM"])) for r in base_rows})
    paired = {u for u in units if all(any(r["drug"] == u[0] and float(r["dose_nM"]) == u[1] and r["replicate"] == rep for r in base_rows) for rep in REPS)}
    base_rows = [r for r in base_rows if (r["drug"], float(r["dose_nM"])) in paired]
    if len(base_rows) < 10 or len(paired) < 4:
        raise RuntimeError(f"too few source base profiles for {cell_line}: rows={len(base_rows)}, units={len(paired)}")
    n_vars = int(counts.shape[1])
    mean = np.zeros(n_vars, dtype=np.float64)
    m2 = np.zeros(n_vars, dtype=np.float64)
    n = 0
    for r in base_rows:
        d = profile_from_row(counts, r).astype(np.float64)
        n += 1
        delta = d - mean
        mean += delta / n
        m2 += delta * (d - mean)
    var = m2 / max(n - 1, 1)
    hvg = np.sort(np.argsort(var, kind="mergesort")[-N_HVG:]).astype(np.int64)
    base_profiles: dict[tuple[str, float, str], np.ndarray] = {}
    for r in base_rows:
        key = (r["drug"], float(r["dose_nM"]), r["replicate"])
        base_profiles[key] = profile_from_row(counts, r, hvg)
    audit = {
        "n_base_condition_rows": len(base_rows),
        "n_base_paired_dose_units": len(paired),
        "n_hvg": int(len(hvg)),
        "n_vars_before_hvg": n_vars,
        "hvg_is_source_only": True,
    }
    return hvg, base_profiles, audit


def fit_estimators(base_profiles: dict[tuple[str, float, str], np.ndarray]):
    from sklearn.linear_model import Ridge

    units = sorted({(drug, dose) for drug, dose, _ in base_profiles})
    x: list[np.ndarray] = []
    y: list[np.ndarray] = []
    for drug, dose in units:
        for support, held in (("rep1", "rep2"), ("rep2", "rep1")):
            if (drug, dose, support) in base_profiles and (drug, dose, held) in base_profiles:
                x.append(base_profiles[(drug, dose, support)])
                y.append(base_profiles[(drug, dose, held)])
    if len(x) < 4:
        raise RuntimeError("insufficient paired source base examples")
    X = np.asarray(x)
    Y = np.asarray(y)
    imr = Ridge(alpha=RIDGE_ALPHA).fit(X, Y)
    # Two real repeats do not permit a third independent support repeat.
    # The legal LSO proxy is therefore exactly the same frozen two-repeat map,
    # as in the within-cell-line confirmation protocol.
    lso = Ridge(alpha=RIDGE_ALPHA).fit(X, Y)
    means = np.stack([(base_profiles[(g, d, "rep1")] + base_profiles[(g, d, "rep2")]) / 2.0 for g, d in units])
    within = np.mean(np.stack([np.var(np.stack([base_profiles[(g, d, "rep1")], base_profiles[(g, d, "rep2")]]), axis=0) for g, d in units]), axis=0)
    between = np.var(means, axis=0)
    shrink = between / (between + within + 1e-8)
    return imr, lso, shrink, {
        "n_paired_examples": len(x),
        "ridge_alpha": RIDGE_ALPHA,
        "lso_equals_imr": True,
        "shrink_mean": float(np.mean(shrink)),
    }


def foreign_vectors(profile_pool: dict[tuple[str, float, str], np.ndarray], target_drug: str, dose: float, held: str, token: str) -> tuple[list[str], list[np.ndarray]]:
    donors = sorted({drug for drug, d, rep in profile_pool if d == dose and rep == held and drug != target_drug})
    if len(donors) < N_FOREIGN:
        raise RuntimeError(f"foreign donor pool <{N_FOREIGN}: dose={dose}, target={target_drug}, n={len(donors)}")
    rng = np.random.default_rng(stable_seed("foreign", token))
    chosen = [str(x) for x in rng.choice(np.asarray(donors, dtype=str), N_FOREIGN, replace=False)]
    return chosen, [profile_pool[(drug, dose, held)] for drug in chosen]


def score(pred: np.ndarray, truth: np.ndarray, foreign: list[np.ndarray]) -> float:
    same = fisher(corr(pred, truth))
    foreign_z = [fisher(corr(pred, x)) for x in foreign]
    return same - float(np.mean(foreign_z))


def select_weights(cal_profiles: dict[tuple[str, float, str], np.ndarray], pool: dict[tuple[str, float, str], np.ndarray], imr: Any, lso: Any, shrink: np.ndarray, cell_line: str) -> tuple[np.ndarray, dict[str, Any]]:
    examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]] = []
    for drug, dose in sorted({(g, d) for g, d, _ in cal_profiles}):
        for support, held in (("rep1", "rep2"), ("rep2", "rep1")):
            ks, kh = (drug, dose, support), (drug, dose, held)
            if ks not in cal_profiles or kh not in cal_profiles:
                continue
            raw = np.asarray(cal_profiles[ks], dtype=np.float64)
            truth = np.asarray(cal_profiles[kh], dtype=np.float64)
            donors, foreign = foreign_vectors(pool, drug, float(dose), held, f"cal|{cell_line}|{drug}|{dose}|{support}|{held}")
            del donors
            p_imr = np.asarray(imr.predict(raw[None, :])[0], dtype=np.float64)
            p_lso = np.asarray(lso.predict(raw[None, :])[0], dtype=np.float64)
            p_eb = raw * shrink
            examples.append((p_imr, p_lso, p_eb, truth, foreign))
    if not examples:
        raise RuntimeError(f"no source calibration examples for {cell_line}")
    best_score = -np.inf
    best_w: np.ndarray | None = None
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
    return best_w, {
        "n_calibration_examples": len(examples),
        "calibration_mean_same_foreign": best_score,
        "grid_step": 0.05,
        "estimators_order": ["IMR", "LSO", "IMCEB"],
        "calibration_source_only": True,
    }


def per_drug(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[key])
        if np.isfinite(value):
            values[str(row["drug"])].append(value)
    return {drug: float(np.mean(xs)) for drug, xs in sorted(values.items()) if xs}


def bootstrap_dict(values: dict[str, float], seed: int) -> dict[str, Any]:
    keys = sorted(values)
    x = np.asarray([values[k] for k in keys], dtype=np.float64)
    if not len(x):
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_drugs": 0, "bootstrap_draws": BOOT, "bootstrap_unit": "drug"}
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(x), size=(BOOT, len(x)))
    means = x[draw].mean(axis=1)
    return {
        "estimate": float(x.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n_drugs": int(len(x)),
        "bootstrap_draws": BOOT,
        "bootstrap_unit": "drug",
    }


def paired_delta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = per_drug(rows, "raw_same_foreign_excess_fisher_z")
    cfra = per_drug(rows, "cfra_same_foreign_excess_fisher_z")
    keys = sorted(set(raw) & set(cfra))
    delta = {k: cfra[k] - raw[k] for k in keys}
    return bootstrap_dict(delta, stable_seed("paired_bootstrap", *(sorted({str(r["source_cell_line"]) for r in rows})), *(sorted({str(r["target_cell_line"]) for r in rows}))))


def metric_summary(rows: list[dict[str, Any]], key: str, seed_token: str) -> dict[str, Any]:
    return bootstrap_dict(per_drug(rows, key), stable_seed("bootstrap", seed_token, key))


def hierarchical_target_bootstrap(target_drug_values: dict[str, dict[str, float]], seed: int) -> dict[str, Any]:
    targets = sorted(target_drug_values)
    if not targets or any(not target_drug_values[t] for t in targets):
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_targets": len(targets), "bootstrap_draws": BOOT, "bootstrap_unit": "target_cell_line_then_drug"}
    target_est = np.asarray([np.mean(list(target_drug_values[t].values())) for t in targets], dtype=np.float64)
    rng = np.random.default_rng(seed)
    # Resample target cell lines equally; within each selected target resample
    # its drugs.  This is a macro target-level uncertainty interval.
    draws = np.empty(BOOT, dtype=np.float64)
    for i in range(BOOT):
        sampled_targets = rng.integers(0, len(targets), size=len(targets))
        vals = []
        for j in sampled_targets:
            xs = np.asarray(list(target_drug_values[targets[int(j)]].values()), dtype=np.float64)
            idx = rng.integers(0, len(xs), size=len(xs))
            vals.append(float(xs[idx].mean()))
        draws[i] = float(np.mean(vals))
    return {
        "estimate": float(target_est.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_targets": len(targets),
        "target_lines": targets,
        "target_point_estimates": {t: float(target_est[i]) for i, t in enumerate(targets)},
        "bootstrap_draws": BOOT,
        "bootstrap_unit": "target_cell_line_then_drug",
    }


def build_condition_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, float, str], dict[str, Any]]:
    out: dict[tuple[str, str, float, str], dict[str, Any]] = {}
    for r in rows:
        key = (r["drug"], r["cell_line"], float(r["dose_nM"]), r["replicate"])
        if key in out:
            raise RuntimeError(f"duplicate condition identity {key}")
        out[key] = r
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-counts", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--raw-gate", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    import h5py

    conditions = read_conditions(args.conditions)
    condition_map = build_condition_map(conditions)
    all_drugs = {str(r["drug"]) for r in conditions}
    split = read_split(args.split_lock, all_drugs)
    base, calibration, confirmation = map(lambda k: set(split[k]), ("base", "calibration", "confirmation"))
    cell_lines = sorted({str(r["cell_line"]) for r in conditions})
    if len(cell_lines) < 2:
        raise RuntimeError("cross-cell evaluation requires at least two cell lines")

    models: dict[str, dict[str, Any]] = {}
    model_audit: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    with h5py.File(args.group_counts, "r") as h5:
        counts = h5["counts"]
        n_vars = int(counts.shape[1])
        # Freeze every source model before loading any target confirmation
        # profile.  Target base/cal profiles below are metric-only donors.
        for source in cell_lines:
            hvg, base_profiles, hvg_audit = fit_hvg_and_base(counts, conditions, base, source)
            source_cal = load_profiles(counts, conditions, calibration, source, hvg)
            source_pool = dict(base_profiles)
            source_pool.update(source_cal)
            imr, lso, shrink, fit_audit = fit_estimators(base_profiles)
            weights, weight_audit = select_weights(source_cal, source_pool, imr, lso, shrink, source)
            models[source] = {
                "hvg": hvg,
                "base_profiles": base_profiles,
                "cal_profiles": source_cal,
                "pool": source_pool,
                "imr": imr,
                "lso": lso,
                "shrink": shrink,
                "weights": weights,
            }
            hvg_path = args.outdir / f"source_hvg2000_{source}.txt"
            np.savetxt(hvg_path, hvg, fmt="%d")
            model_audit.append({"source_cell_line": source, **hvg_audit, **fit_audit, **weight_audit, "hvg_sha256": sha256_file(hvg_path)})
            weight_rows.append({"source_cell_line": source, "weights_IMR": float(weights[0]), "weights_LSO": float(weights[1]), "weights_IMCEB": float(weights[2]), **fit_audit, **weight_audit})

        eval_rows: list[dict[str, Any]] = []
        target_read_after_freeze = True
        # Target confirmation profiles are read only after all source models,
        # HVGs and source calibration weights have been frozen.
        for target in cell_lines:
            target_condition_rows = [r for r in conditions if r["cell_line"] == target]
            for source in cell_lines:
                if source == target:
                    continue
                model = models[source]
                hvg = model["hvg"]
                target_conf = load_profiles(counts, target_condition_rows, confirmation, target, hvg)
                target_metric_pool = load_profiles(counts, target_condition_rows, base | calibration, target, hvg)
                for drug, dose in sorted({(g, d) for g, d, _ in target_conf}):
                    for support, held in (("rep1", "rep2"), ("rep2", "rep1")):
                        ks, kh = (drug, dose, support), (drug, dose, held)
                        if ks not in target_conf or kh not in target_conf:
                            continue
                        raw = np.asarray(target_conf[ks], dtype=np.float64)
                        truth = np.asarray(target_conf[kh], dtype=np.float64)
                        # Token intentionally omits source: each target/dose/
                        # held endpoint gets identical donor IDs for all source
                        # models, preserving a common foreign benchmark.
                        donor_ids, foreign = foreign_vectors(target_metric_pool, drug, float(dose), held, f"target|{target}|{drug}|{dose}|{support}|{held}")
                        p_imr = np.asarray(model["imr"].predict(raw[None, :])[0], dtype=np.float64)
                        p_lso = np.asarray(model["lso"].predict(raw[None, :])[0], dtype=np.float64)
                        p_eb = raw * model["shrink"]
                        w = np.asarray(model["weights"], dtype=np.float64)
                        cfra = w[0] * p_imr + w[1] * p_lso + w[2] * p_eb
                        raw_score = score(raw, truth, foreign)
                        cfra_score = score(cfra, truth, foreign)
                        eval_rows.append({
                            "source_cell_line": source,
                            "target_cell_line": target,
                            "drug": drug,
                            "dose_nM": float(dose),
                            "support_repeat": support,
                            "held_repeat": held,
                            "n_support_cells": int(condition_map[(drug, target, float(dose), support)]["n_cells"]),
                            "n_held_cells": int(condition_map[(drug, target, float(dose), held)]["n_cells"]),
                            "foreign_n": len(foreign),
                            "foreign_drugs": ";".join(sorted(donor_ids)),
                            "raw_same_foreign_excess_fisher_z": float(raw_score),
                            "cfra_same_foreign_excess_fisher_z": float(cfra_score),
                            "cfra_minus_raw_same_foreign_excess_fisher_z": float(cfra_score - raw_score),
                            "norm_ratio_cfra_over_raw": float(np.linalg.norm(cfra) / max(np.linalg.norm(raw), 1e-12)),
                        })
    if not eval_rows:
        raise RuntimeError("no cross-cell confirmation rows")

    write_csv(args.outdir / "cross_cell_same_foreign_per_condition.csv", eval_rows)
    write_csv(args.outdir / "cross_cell_calibration_weights.csv", weight_rows)

    pair_rows: list[dict[str, Any]] = []
    pair_summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for source in cell_lines:
        for target in cell_lines:
            if source == target:
                continue
            rows = [r for r in eval_rows if r["source_cell_line"] == source and r["target_cell_line"] == target]
            if not rows:
                raise RuntimeError(f"no rows for source={source}, target={target}")
            raw_stat = metric_summary(rows, "raw_same_foreign_excess_fisher_z", f"pair|{source}|{target}")
            cfra_stat = metric_summary(rows, "cfra_same_foreign_excess_fisher_z", f"pair|{source}|{target}")
            delta_stat = paired_delta(rows)
            norm_stat = metric_summary(rows, "norm_ratio_cfra_over_raw", f"pair|{source}|{target}")
            row = {"source_cell_line": source, "target_cell_line": target, "metric": "Raw Same-Foreign excess Fisher-z", "method": "Raw", **raw_stat}
            pair_rows.append(row)
            pair_rows.append({"source_cell_line": source, "target_cell_line": target, "metric": "CFRA Same-Foreign excess Fisher-z", "method": "CFRA", **cfra_stat})
            pair_rows.append({"source_cell_line": source, "target_cell_line": target, "metric": "CFRA-Raw Same-Foreign excess Fisher-z", "method": "CFRA-Raw", **delta_stat})
            pair_rows.append({"source_cell_line": source, "target_cell_line": target, "metric": "||CFRA||/||Raw||", "method": "CFRA", **norm_stat})
            pair_summaries[(source, target)] = {"raw": raw_stat, "cfra": cfra_stat, "delta": delta_stat, "norm": norm_stat}
    write_csv(args.outdir / "cross_cell_source_target_summary.csv", pair_rows)

    # Target hierarchy: one mean per target/drug after averaging across the
    # two independent source models.  The Raw value is paired in each source
    # space, so the target-level baseline is the mean of those paired Raw
    # values, not a separately selected target feature set.
    target_rows: list[dict[str, Any]] = []
    target_drug_values: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for target in cell_lines:
        sources = [s for s in cell_lines if s != target]
        target_eval = [r for r in eval_rows if r["target_cell_line"] == target]
        drugs = sorted({str(r["drug"]) for r in target_eval})
        for drug in drugs:
            drug_rows = [r for r in target_eval if r["drug"] == drug]
            by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in drug_rows:
                by_source[str(r["source_cell_line"])].append(r)
            # Each source contributes its within-drug mean; then sources are
            # macro-averaged so one source cannot dominate due to missing rows.
            metrics: dict[str, float] = {}
            for key in ("raw_same_foreign_excess_fisher_z", "cfra_same_foreign_excess_fisher_z", "cfra_minus_raw_same_foreign_excess_fisher_z", "norm_ratio_cfra_over_raw"):
                source_means = [float(np.mean([float(r[key]) for r in by_source[s]])) for s in sources if by_source.get(s)]
                if source_means:
                    metrics[key] = float(np.mean(source_means))
            if len(metrics) != 4:
                continue
            target_rows.append({"target_cell_line": target, "drug": drug, **metrics})
            for key, value in metrics.items():
                target_drug_values[target][key][drug] = value
    target_summary_rows: list[dict[str, Any]] = []
    for target in cell_lines:
        rows = [r for r in target_rows if r["target_cell_line"] == target]
        for key, metric, method in (
            ("raw_same_foreign_excess_fisher_z", "Raw Same-Foreign excess Fisher-z", "Raw"),
            ("cfra_same_foreign_excess_fisher_z", "CFRA Same-Foreign excess Fisher-z", "CFRA"),
            ("cfra_minus_raw_same_foreign_excess_fisher_z", "CFRA-Raw Same-Foreign excess Fisher-z", "CFRA-Raw"),
            ("norm_ratio_cfra_over_raw", "||CFRA||/||Raw||", "CFRA"),
        ):
            target_summary_rows.append({"target_cell_line": target, "metric": metric, "method": method, **bootstrap_dict(target_drug_values[target][key], stable_seed("target", target, key))})
    # Overall target hierarchy is written as one row per metric with macro CI.
    for key, metric, method in (
        ("raw_same_foreign_excess_fisher_z", "Raw Same-Foreign excess Fisher-z", "Raw"),
        ("cfra_same_foreign_excess_fisher_z", "CFRA Same-Foreign excess Fisher-z", "CFRA"),
        ("cfra_minus_raw_same_foreign_excess_fisher_z", "CFRA-Raw Same-Foreign excess Fisher-z", "CFRA-Raw"),
        ("norm_ratio_cfra_over_raw", "||CFRA||/||Raw||", "CFRA"),
    ):
        target_summary_rows.append({"target_cell_line": "ALL_TARGETS_MACRO", "metric": metric, "method": method, **hierarchical_target_bootstrap({t: target_drug_values[t][key] for t in cell_lines}, stable_seed("macro", key))})
    write_csv(args.outdir / "cross_cell_target_summary.csv", target_summary_rows)
    write_csv(args.outdir / "cross_cell_target_drug_values.csv", target_rows)

    raw_gate = json.loads(args.raw_gate.read_text(encoding="utf-8"))
    split_hash = sha256_file(args.split_lock)
    audit = {
        "dataset": "sci-Plex3 Zenodo 7041849",
        "protocol": "source-cell-line fit/calibration, frozen direct target-cell-line application",
        "input_files": {
            "group_counts": {"path": str(args.group_counts), "sha256": sha256_file(args.group_counts), "md5": md5_file(args.group_counts), "shape": [int(counts.shape[0]), int(counts.shape[1])]},
            "conditions": {"path": str(args.conditions), "sha256": sha256_file(args.conditions)},
            "split_lock": {"path": str(args.split_lock), "sha256": split_hash},
            "raw_gate": {"path": str(args.raw_gate), "sha256": sha256_file(args.raw_gate)},
        },
        "input_md5_expected": "d1f51b9f8de35ca07638132539da9a99",
        "time_hours": 24,
        "nperts": 1,
        "n_min": 50,
        "replicates": list(REPS),
        "cell_lines": cell_lines,
        "source_target_pairs": [[s, t] for s in cell_lines for t in cell_lines if s != t],
        "drug_split": {"seed": SEED, "source": str(args.split_lock), "sha256": split_hash, "counts": {k: len(v) for k, v in split.items()}, "overlap": {"base_calibration": sorted(base & calibration), "base_confirmation": sorted(base & confirmation), "calibration_confirmation": sorted(calibration & confirmation)}},
        "feature_policy": {
            "feature_map": "GEO annotation row i -> H5 X[:,i] for i=0..110982; X[:,110983] all-zero tail excluded by upstream frozen group_counts",
            "source_hvg_n": N_HVG,
            "source_hvg_only": True,
            "target_hvg_fit": False,
            "raw_and_cfra_same_feature_space_per_pair": True,
            "space": "source-frozen HGV2000; target profiles are projected by source indices only",
        },
        "model_freeze": {
            "fit_data": "source cell line, base drugs, paired rep1/rep2 responses only",
            "calibration_data": "source cell line, calibration drugs only",
            "target_confirmation_used_for_fit": False,
            "target_base_cal_used_for_fit": False,
            "target_base_cal_role": "foreign metric donors only",
            "confirmation_read_after_all_source_models_frozen": True,
            "confirmation_read_once_per_source_target": True,
            "estimators": ["Ridge IMR", "Ridge LSO", "IMCEB", "CFRA"],
            "ridge_alpha": RIDGE_ALPHA,
            "lso_policy": "LSO=IMR because only two genuine biological repeats",
            "source_records": model_audit,
        },
        "endpoint": {
            "response": "log1p(CPM 1e6) treated pseudo-bulk minus same-plate/same-cell-line/same-repeat Vehicle pseudo-bulk",
            "ordered_rotations": ["rep1->rep2", "rep2->rep1"],
            "foreign_policy": "20 deterministic target-cell-line, same-dose, held-repeat donors from target base+calibration; target drug excluded",
            "same_foreign_ids_raw_cfra": True,
            "inference_unit": "drug",
            "dose_and_rotation_aggregation": "mean within drug before bootstrap",
            "bootstrap_draws": BOOT,
            "pair_ci": "drug-level bootstrap within source->target pair",
            "target_ci": "drug-level bootstrap after equal source macro-average within target",
            "overall_ci": "hierarchical target-cell-line then drug bootstrap",
        },
        "n_eval_rows": len(eval_rows),
        "n_eval_drugs_by_target": {target: len({r["drug"] for r in target_rows if r["target_cell_line"] == target}) for target in cell_lines},
        "raw_phase0_gate": raw_gate,
        "status": "PASS_WITH_FROZEN_SOURCE_TO_TARGET_EVALUATION",
    }
    write_json(args.outdir / "CROSS_CELL_GENERALIZATION_AUDIT.json", audit)

    report: list[str] = [
        "# sci-Plex3 cross-cell-line CFRA generalization",
        "",
        "## Question and frozen design",
        "",
        "This is a source→target generalization test. For each ordered pair, HVG2000, Ridge IMR/LSO, IMCEB shrinkage and CFRA weights are fit/calibrated in the source cell line only, then frozen and applied directly to target confirmation drugs.",
        "",
        "- Scope: 24 h, nperts=1, n≥50, matched same-plate/same-cell-line/same-repeat Vehicle; rep1→rep2 and rep2→rep1.",
        f"- Frozen drug split: base/calibration/confirmation={len(base)}/{len(calibration)}/{len(confirmation)}, seed={SEED}; no overlap.",
        "- Target base+calibration profiles are used only to choose the 20 target foreign donor IDs for the metric; they never fit, calibrate, scale or select a source model feature.",
        "- Raw and CFRA share the same target support/held rows and foreign donor IDs for each source→target endpoint. Both are scored in that source's frozen HVG2000 space.",
        "- Bootstrap unit: drug. Dose and repeat-rotation rows are averaged within drug first; 10,000 bootstrap draws. Target-level values macro-average sources before drug bootstrap; overall is hierarchical target-line→drug bootstrap.",
        "",
        "## Source→target results",
        "",
        "| Source → target | Raw Same–Foreign | CFRA Same–Foreign | CFRA − Raw (95% CI) | ||CFRA||/||Raw|| |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in cell_lines:
        for target in cell_lines:
            if source == target:
                continue
            s = pair_summaries[(source, target)]
            report.append(f"| {source} → {target} | {s['raw']['estimate']:.6f} | {s['cfra']['estimate']:.6f} | {s['delta']['estimate']:.6f} [{s['delta']['ci_low']:.6f}, {s['delta']['ci_high']:.6f}] | {s['norm']['estimate']:.6f} |")
    report.extend(["", "## Target-level and overall interpretation", ""])
    macro = {r["metric"]: r for r in target_summary_rows if r["target_cell_line"] == "ALL_TARGETS_MACRO"}
    report.append(f"- Overall target-macro CFRA−Raw: {macro['CFRA-Raw Same-Foreign excess Fisher-z']['estimate']:.6f}, 95% CI [{macro['CFRA-Raw Same-Foreign excess Fisher-z']['ci_low']:.6f}, {macro['CFRA-Raw Same-Foreign excess Fisher-z']['ci_high']:.6f}].")
    for target in cell_lines:
        row = next(r for r in target_summary_rows if r["target_cell_line"] == target and r["metric"] == "CFRA-Raw Same-Foreign excess Fisher-z")
        report.append(f"- {target} target macro across source models: CFRA−Raw={row['estimate']:.6f}, 95% CI [{row['ci_low']:.6f}, {row['ci_high']:.6f}], n_drugs={row['n_drugs']}.")
    report.extend(["", "## Audit conclusion", "", "The cross-cell result is considered a valid direct-application evaluation only because every source model quantity is frozen before target confirmation is read. A positive paired CI means CFRA preserves more same-repeat target concordance than Raw in the source-frozen feature space; it does not imply that target data were used to tune the model.", ""])
    (args.outdir / "CROSS_CELL_GENERALIZATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"pairs": pair_summaries, "macro": macro}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
