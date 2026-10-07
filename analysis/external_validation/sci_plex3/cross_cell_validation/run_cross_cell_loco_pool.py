"""sci-Plex3 leave-one-cell-line-out pooled CFRA generalization.

Formal cross-cell-line experiment:

    A549 + K562 -> MCF7
    A549 + MCF7 -> K562
    K562 + MCF7 -> A549

For each target, the two other cell lines are pooled for fitting and
calibration.  Only drug-dose units present with both biological-repeat rows
in all three cell lines are admitted (the frozen common universe is expected
to be 183 drugs/613 drug-dose units; its confirmation subset is expected to be
36 drugs/115 units).  The source model's HGV2000 features, Ridge IMR/LSO map,
IMCEB shrinkage and CFRA simplex weights are all fitted without reading target
expression.  Target expression is read only after all source quantities are
frozen, and target base/calibration profiles are used only as foreign-null
donors.

Raw and CFRA are scored on exactly the same target support/held repeat rows,
in the same source-pooled HGV2000 coordinates, with exactly the same target
foreign donor IDs.  Inference is at drug level: dose and ordered-repeat rows
are averaged within drug before 10,000-draw bootstrap.  Target-line macro CIs
use hierarchical target-cell-line then drug bootstrap.
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
N_HVG = 2_000
N_FOREIGN = 20
RIDGE_ALPHA = 10.0
N_FEATURES_WITH_GEO_MAP = 110_983  # X[:,110983] is the audited all-zero tail.
EXPECTED_COMMON_DRUGS = 183
EXPECTED_COMMON_UNITS = 613
EXPECTED_CONFIRMATION_DRUGS = 36
EXPECTED_CONFIRMATION_UNITS = 115


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


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
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
    x -= float(x.mean())
    y -= float(y.mean())
    den = math.sqrt(float(x @ x) * float(y @ y))
    return float(x @ y / den) if den > 1e-12 else float("nan")


def fisher(r: float) -> float:
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999))) if np.isfinite(r) else float("nan")


def norm_delta(t_counts: np.ndarray, c_counts: np.ndarray) -> np.ndarray:
    # The GEO annotation has 110,983 data rows.  Do not let the known
    # scPerturb extra zero tail enter HVG selection or any score.
    t = np.asarray(t_counts[:N_FEATURES_WITH_GEO_MAP], dtype=np.float64)
    c = np.asarray(c_counts[:N_FEATURES_WITH_GEO_MAP], dtype=np.float64)
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
    split = {key: [str(x) for x in raw[key]] for key in ("base", "calibration", "confirmation")}
    sets = {key: set(value) for key, value in split.items()}
    for left, right in (("base", "calibration"), ("base", "confirmation"), ("calibration", "confirmation")):
        if sets[left] & sets[right]:
            raise RuntimeError(f"frozen split overlap {left}/{right}")
    union = sets["base"] | sets["calibration"] | sets["confirmation"]
    if union != all_drugs:
        raise RuntimeError(f"frozen split does not cover condition drugs: missing={sorted(all_drugs-union)}, extra={sorted(union-all_drugs)}")
    return {key: sorted(value) for key, value in split.items()}


def build_condition_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, float, str], dict[str, Any]]:
    out: dict[tuple[str, str, float, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["drug"], row["cell_line"], float(row["dose_nM"]), row["replicate"])
        if key in out:
            raise RuntimeError(f"duplicate condition identity {key}")
        out[key] = row
    return out


def paired_units(rows: list[dict[str, Any]], cell_line: str, allowed_drugs: set[str] | None = None) -> set[tuple[str, float]]:
    by: dict[tuple[str, float], set[str]] = defaultdict(set)
    for row in rows:
        if row["cell_line"] != cell_line or (allowed_drugs is not None and row["drug"] not in allowed_drugs):
            continue
        by[(row["drug"], float(row["dose_nM"]))].add(row["replicate"])
    return {unit for unit, reps in by.items() if set(REPS).issubset(reps)}


def profile_from_row(counts: Any, row: dict[str, Any], hvg: np.ndarray | None = None) -> np.ndarray:
    vector = norm_delta(counts[int(row["group_id"]), :], counts[int(row["control_group_id"]), :])
    return vector if hvg is None else vector[hvg]


def rows_for_units(rows: list[dict[str, Any]], cell_line: str, units: set[tuple[str, float]], drugs: set[str] | None = None) -> list[dict[str, Any]]:
    return [row for row in rows if row["cell_line"] == cell_line and (row["drug"], float(row["dose_nM"])) in units and (drugs is None or row["drug"] in drugs)]


def load_profiles(counts: Any, rows: list[dict[str, Any]], units: set[tuple[str, float]], cell_line: str, hvg: np.ndarray) -> dict[tuple[str, float, str], np.ndarray]:
    out: dict[tuple[str, float, str], np.ndarray] = {}
    for row in rows_for_units(rows, cell_line, units):
        key = (row["drug"], float(row["dose_nM"]), row["replicate"])
        if key in out:
            raise RuntimeError(f"duplicate profile key {cell_line}/{key}")
        out[key] = profile_from_row(counts, row, hvg)
    return out


def load_source_profiles(counts: Any, rows: list[dict[str, Any]], sources: list[str], units: set[tuple[str, float]], drugs: set[str], hvg: np.ndarray | None = None) -> dict[tuple[str, str, float, str], np.ndarray]:
    out: dict[tuple[str, str, float, str], np.ndarray] = {}
    for source in sources:
        for row in rows_for_units(rows, source, units, drugs):
            key = (source, row["drug"], float(row["dose_nM"]), row["replicate"])
            if key in out:
                raise RuntimeError(f"duplicate source profile key {key}")
            out[key] = profile_from_row(counts, row, hvg)
    return out


def fit_joint_hvg(counts: Any, rows: list[dict[str, Any]], sources: list[str], base_units: set[tuple[str, float]], n_features: int) -> tuple[np.ndarray, dict[tuple[str, str, float, str], np.ndarray], dict[str, Any]]:
    source_rows = [row for source in sources for row in rows_for_units(rows, source, base_units)]
    if len(source_rows) < 10:
        raise RuntimeError(f"too few joint source base rows: {len(source_rows)}")
    # Online variance avoids retaining the 110k-dimensional profile matrix.
    mean = np.zeros(n_features, dtype=np.float64)
    m2 = np.zeros(n_features, dtype=np.float64)
    n = 0
    for row in source_rows:
        value = profile_from_row(counts, row).astype(np.float64)
        n += 1
        delta = value - mean
        mean += delta / n
        m2 += delta * (value - mean)
    variance = m2 / max(n - 1, 1)
    hvg = np.sort(np.argsort(variance, kind="mergesort")[-N_HVG:]).astype(np.int64)
    profiles = load_source_profiles(counts, rows, sources, base_units, {drug for drug, _ in base_units}, hvg)
    source_counts = {source: sum(1 for key in profiles if key[0] == source) for source in sources}
    audit = {
        "source_lines": sources,
        "source_base_rows": len(source_rows),
        "source_base_rows_by_line": source_counts,
        "source_base_rows_equal": len(set(source_counts.values())) == 1,
        "n_features_before_hvg": n_features,
        "n_hvg": int(len(hvg)),
        "hvg_fit_scope": "joint source base only",
        "target_expression_used": False,
    }
    return hvg, profiles, audit


def fit_estimators(base_profiles: dict[tuple[str, str, float, str], np.ndarray], sources: list[str]) -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import Ridge

    units = sorted({(drug, dose) for _, drug, dose, _ in base_profiles})
    x: list[np.ndarray] = []
    y: list[np.ndarray] = []
    train_keys: list[tuple[str, str, float, str]] = []
    source_row_count: dict[str, int] = {source: 0 for source in sources}
    for source in sources:
        for drug, dose in units:
            for support, held in (("rep1", "rep2"), ("rep2", "rep1")):
                ks = (source, drug, dose, support)
                kh = (source, drug, dose, held)
                if ks not in base_profiles or kh not in base_profiles:
                    continue
                x.append(base_profiles[ks])
                y.append(base_profiles[kh])
                train_keys.append(ks)
                source_row_count[source] += 1
    if len(x) < 8 or set(source_row_count.values()) != {source_row_count[sources[0]]}:
        raise RuntimeError(f"unbalanced or insufficient pooled training rows: {source_row_count}")
    X, Y = np.asarray(x), np.asarray(y)
    # Equal total weight per source, even though the common-unit construction
    # already gives equal rows.  This prevents source abundance from deciding
    # the pooled map if the upstream table changes.
    sample_weight = np.asarray([1.0 / source_row_count[key[0]] for key in train_keys], dtype=np.float64)
    if len(sample_weight) != len(X):
        raise RuntimeError("training sample-weight alignment failure")
    imr = Ridge(alpha=RIDGE_ALPHA).fit(X, Y, sample_weight=sample_weight)
    # With two biological repeats there is no independent third support repeat;
    # preserve the audited legal convention LSO=IMR.
    lso = Ridge(alpha=RIDGE_ALPHA).fit(X, Y, sample_weight=sample_weight)
    means = np.stack([(base_profiles[(source, drug, dose, "rep1")] + base_profiles[(source, drug, dose, "rep2")]) / 2.0 for source in sources for drug, dose in units])
    within = np.mean(np.stack([np.var(np.stack([base_profiles[(source, drug, dose, "rep1")], base_profiles[(source, drug, dose, "rep2")]]), axis=0) for source in sources for drug, dose in units]), axis=0)
    between = np.var(means, axis=0)
    shrink = between / (between + within + 1e-8)
    return imr, lso, shrink, {
        "n_paired_training_examples": len(X),
        "training_rows_by_source": source_row_count,
        "equal_source_training_weight": True,
        "ridge_alpha": RIDGE_ALPHA,
        "lso_equals_imr": True,
        "shrink_mean": float(np.mean(shrink)),
    }


def foreign_vectors(profile_pool: dict[tuple[str, float, str], np.ndarray], target_drug: str, dose: float, held: str, token: str) -> tuple[list[str], list[np.ndarray]]:
    donors = sorted({drug for drug, donor_dose, replicate in profile_pool if donor_dose == dose and replicate == held and drug != target_drug})
    if len(donors) < N_FOREIGN:
        raise RuntimeError(f"target foreign pool <{N_FOREIGN}: dose={dose}, target={target_drug}, available={len(donors)}")
    rng = np.random.default_rng(stable_seed("foreign", token))
    selected = [str(x) for x in rng.choice(np.asarray(donors, dtype=str), N_FOREIGN, replace=False)]
    return selected, [profile_pool[(drug, dose, held)] for drug in selected]


def score(pred: np.ndarray, truth: np.ndarray, foreign: list[np.ndarray]) -> float:
    same = fisher(corr(pred, truth))
    foreign_z = [fisher(corr(pred, value)) for value in foreign]
    return same - float(np.mean(foreign_z))


def select_pooled_weights(cal_profiles: dict[tuple[str, str, float, str], np.ndarray], source_pool: dict[tuple[str, str, float, str], np.ndarray], sources: list[str], calibration_units: set[tuple[str, float]], imr: Any, lso: Any, shrink: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    examples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]] = []
    source_pools: dict[str, dict[tuple[str, float, str], np.ndarray]] = {
        source: {(drug, dose, rep): value for (source_key, drug, dose, rep), value in source_pool.items() if source_key == source}
        for source in sources
    }
    for source in sources:
        for drug, dose in sorted(calibration_units):
            for support, held in (("rep1", "rep2"), ("rep2", "rep1")):
                ks = (source, drug, dose, support)
                kh = (source, drug, dose, held)
                if ks not in cal_profiles or kh not in cal_profiles:
                    continue
                raw = np.asarray(cal_profiles[ks], dtype=np.float64)
                truth = np.asarray(cal_profiles[kh], dtype=np.float64)
                donor_ids, foreign = foreign_vectors(source_pools[source], drug, float(dose), held, f"cal|{source}|{drug}|{dose}|{support}|{held}")
                del donor_ids
                p_imr = np.asarray(imr.predict(raw[None, :])[0], dtype=np.float64)
                p_lso = np.asarray(lso.predict(raw[None, :])[0], dtype=np.float64)
                p_eb = raw * shrink
                examples.append((source, p_imr, p_lso, p_eb, truth, foreign))
    if not examples:
        raise RuntimeError("no pooled source calibration examples")
    best_score, best_w = -np.inf, None
    for ia in range(21):
        a = ia / 20.0
        for ib in range(21 - ia):
            b = ib / 20.0
            c = 1.0 - a - b
            w = np.asarray([a, b, c], dtype=np.float64)
            # First average calibration scores within source line, then give
            # each source line equal weight.  This remains correct if a future
            # eligibility table has unequal source condition counts.
            by_source: dict[str, list[float]] = defaultdict(list)
            for source, p_imr, p_lso, p_eb, truth, foreign in examples:
                by_source[source].append(score(w[0] * p_imr + w[1] * p_lso + w[2] * p_eb, truth, foreign))
            value = float(np.mean([np.mean(by_source[source]) for source in sources]))
            if value > best_score + 1e-15:
                best_score, best_w = value, w
    if best_w is None:
        raise RuntimeError("pooled simplex selection failed")
    return best_w, {
        "n_calibration_examples": len(examples),
        "calibration_examples_by_source": {source: sum(1 for source_key, *_ in examples if source_key == source) for source in sources},
        "calibration_source_equal_weight": True,
        "calibration_mean_same_foreign": best_score,
        "calibration_source_only": True,
        "grid_step": 0.05,
        "estimators_order": ["IMR", "LSO", "IMCEB"],
    }


def per_drug(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[key])
        if np.isfinite(value):
            values[str(row["drug"])].append(value)
    return {drug: float(np.mean(xs)) for drug, xs in sorted(values.items()) if xs}


def bootstrap_values(values: dict[str, float], seed: int) -> dict[str, Any]:
    keys = sorted(values)
    x = np.asarray([values[key] for key in keys], dtype=np.float64)
    if not len(x):
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_drugs": 0, "bootstrap_draws": BOOT, "bootstrap_unit": "drug"}
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(x), size=(BOOT, len(x)))
    means = x[draw].mean(axis=1)
    return {"estimate": float(x.mean()), "ci_low": float(np.quantile(means, .025)), "ci_high": float(np.quantile(means, .975)), "n_drugs": int(len(x)), "bootstrap_draws": BOOT, "bootstrap_unit": "drug"}


def summary_for(rows: list[dict[str, Any]], token: str) -> dict[str, Any]:
    raw = bootstrap_values(per_drug(rows, "raw_same_foreign_excess_fisher_z"), stable_seed(token, "raw"))
    cfra = bootstrap_values(per_drug(rows, "cfra_same_foreign_excess_fisher_z"), stable_seed(token, "cfra"))
    norm = bootstrap_values(per_drug(rows, "norm_ratio_cfra_over_raw"), stable_seed(token, "norm"))
    raw_by = per_drug(rows, "raw_same_foreign_excess_fisher_z")
    cfra_by = per_drug(rows, "cfra_same_foreign_excess_fisher_z")
    delta = {drug: cfra_by[drug] - raw_by[drug] for drug in sorted(set(raw_by) & set(cfra_by))}
    delta_stat = bootstrap_values(delta, stable_seed(token, "paired_delta"))
    return {"raw": raw, "cfra": cfra, "delta": delta_stat, "norm": norm}


def hierarchical_bootstrap(target_values: dict[str, dict[str, float]], seed: int) -> dict[str, Any]:
    targets = sorted(target_values)
    if not targets or any(not target_values[target] for target in targets):
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_targets": len(targets), "bootstrap_draws": BOOT, "bootstrap_unit": "target_cell_line_then_drug"}
    target_points = {target: float(np.mean(list(target_values[target].values()))) for target in targets}
    point = float(np.mean(list(target_points.values())))
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOT, dtype=np.float64)
    for i in range(BOOT):
        target_indices = rng.integers(0, len(targets), size=len(targets))
        values = []
        for index in target_indices:
            x = np.asarray(list(target_values[targets[int(index)]].values()), dtype=np.float64)
            values.append(float(x[rng.integers(0, len(x), size=len(x))].mean()))
        draws[i] = float(np.mean(values))
    worst_target = min(target_points, key=target_points.get)
    return {
        "estimate": point,
        "ci_low": float(np.quantile(draws, .025)),
        "ci_high": float(np.quantile(draws, .975)),
        "n_targets": len(targets),
        "target_lines": targets,
        "target_point_estimates": target_points,
        "worst_target": worst_target,
        "worst_target_estimate": target_points[worst_target],
        "bootstrap_draws": BOOT,
        "bootstrap_unit": "target_cell_line_then_drug",
    }


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
    all_drugs = {str(row["drug"]) for row in conditions}
    split = read_split(args.split_lock, all_drugs)
    base_drugs, calibration_drugs, confirmation_drugs = (set(split[key]) for key in ("base", "calibration", "confirmation"))
    cell_lines = sorted({str(row["cell_line"]) for row in conditions})
    if set(cell_lines) != {"A549", "K562", "MCF7"}:
        raise RuntimeError(f"unexpected sci-Plex3 cell lines: {cell_lines}")

    paired_by_line = {cell: paired_units(conditions, cell) for cell in cell_lines}
    common_units = set.intersection(*(paired_by_line[cell] for cell in cell_lines))
    common_drugs = {drug for drug, _ in common_units}
    common_confirmation_units = {unit for unit in common_units if unit[0] in confirmation_drugs}
    common_confirmation_drugs = {drug for drug, _ in common_confirmation_units}
    if (len(common_drugs), len(common_units), len(common_confirmation_drugs), len(common_confirmation_units)) != (EXPECTED_COMMON_DRUGS, EXPECTED_COMMON_UNITS, EXPECTED_CONFIRMATION_DRUGS, EXPECTED_CONFIRMATION_UNITS):
        raise RuntimeError(f"frozen common-universe mismatch: drugs={len(common_drugs)}, units={len(common_units)}, confirmation_drugs={len(common_confirmation_drugs)}, confirmation_units={len(common_confirmation_units)}")
    common_rows = [row for row in conditions if (row["drug"], float(row["dose_nM"])) in common_units]
    unit_rows = [{"drug": drug, "dose_nM": dose, "in_confirmation": drug in confirmation_drugs} for drug, dose in sorted(common_units)]
    write_csv(args.outdir / "LOCO_POOL_COMMON_DRUG_DOSE_UNITS.csv", unit_rows)

    models: dict[str, dict[str, Any]] = {}
    model_audit: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    with h5py.File(args.group_counts, "r") as h5:
        counts = h5["counts"]
        shape = [int(counts.shape[0]), int(counts.shape[1])]
        if shape[1] < N_FEATURES_WITH_GEO_MAP:
            raise RuntimeError(f"group-count feature dimension too small: {shape}")
        # Leave one cell line out.  Everything needed to make the model is
        # loaded from the two source lines before target confirmation is read.
        for target in cell_lines:
            sources = [cell for cell in cell_lines if cell != target]
            base_units = common_units & {(drug, dose) for drug, dose in common_units if drug in base_drugs}
            calibration_units = common_units & {(drug, dose) for drug, dose in common_units if drug in calibration_drugs}
            hvg, base_profiles, hvg_audit = fit_joint_hvg(counts, common_rows, sources, base_units, N_FEATURES_WITH_GEO_MAP)
            cal_profiles = load_source_profiles(counts, common_rows, sources, calibration_units, calibration_drugs, hvg)
            source_pool = dict(base_profiles)
            source_pool.update(cal_profiles)
            imr, lso, shrink, fit_audit = fit_estimators(base_profiles, sources)
            weights, weight_audit = select_pooled_weights(cal_profiles, source_pool, sources, calibration_units, imr, lso, shrink)
            models[target] = {"target": target, "sources": sources, "hvg": hvg, "base_profiles": base_profiles, "cal_profiles": cal_profiles, "pool": source_pool, "imr": imr, "lso": lso, "shrink": shrink, "weights": weights}
            hvg_path = args.outdir / f"LOCO_POOL_HVG2000_target_{target}.txt"
            np.savetxt(hvg_path, hvg, fmt="%d")
            model_audit.append({"target_cell_line": target, "source_cell_lines": sources, **hvg_audit, **fit_audit, **weight_audit, "hvg_sha256": digest(hvg_path)})
            weight_rows.append({"target_cell_line": target, "source_cell_lines": "+".join(sources), "weights_IMR": float(weights[0]), "weights_LSO": float(weights[1]), "weights_IMCEB": float(weights[2]), **fit_audit, **weight_audit})

        eval_rows: list[dict[str, Any]] = []
        # This is the first point at which any target expression is loaded.
        target_confirmation_read_after_freeze = True
        for target in cell_lines:
            model = models[target]
            hvg = model["hvg"]
            target_rows = [row for row in common_rows if row["cell_line"] == target]
            target_conf = load_profiles(counts, target_rows, common_confirmation_units, target, hvg)
            target_metric_pool = load_profiles(counts, target_rows, common_units - common_confirmation_units, target, hvg)
            for drug, dose in sorted(common_confirmation_units):
                for support, held in (("rep1", "rep2"), ("rep2", "rep1")):
                    key_support, key_held = (drug, dose, support), (drug, dose, held)
                    if key_support not in target_conf or key_held not in target_conf:
                        raise RuntimeError(f"missing target confirmation repeat {target}/{drug}/{dose}")
                    raw = np.asarray(target_conf[key_support], dtype=np.float64)
                    truth = np.asarray(target_conf[key_held], dtype=np.float64)
                    # Token omits method and is shared by Raw/CFRA.  It also
                    # omits source because this LOCO target has one pooled map.
                    donor_ids, foreign = foreign_vectors(target_metric_pool, drug, float(dose), held, f"target|{target}|{drug}|{dose}|{support}|{held}")
                    p_imr = np.asarray(model["imr"].predict(raw[None, :])[0], dtype=np.float64)
                    p_lso = np.asarray(model["lso"].predict(raw[None, :])[0], dtype=np.float64)
                    p_eb = raw * model["shrink"]
                    w = np.asarray(model["weights"], dtype=np.float64)
                    cfra = w[0] * p_imr + w[1] * p_lso + w[2] * p_eb
                    raw_score = score(raw, truth, foreign)
                    cfra_score = score(cfra, truth, foreign)
                    condition = condition_map[(drug, target, float(dose), support)]
                    held_condition = condition_map[(drug, target, float(dose), held)]
                    eval_rows.append({
                        "source_cell_lines": "+".join(model["sources"]),
                        "target_cell_line": target,
                        "drug": drug,
                        "dose_nM": float(dose),
                        "support_repeat": support,
                        "held_repeat": held,
                        "n_support_cells": int(condition["n_cells"]),
                        "n_held_cells": int(held_condition["n_cells"]),
                        "foreign_n": len(foreign),
                        "foreign_drugs": ";".join(sorted(donor_ids)),
                        "raw_same_foreign_excess_fisher_z": float(raw_score),
                        "cfra_same_foreign_excess_fisher_z": float(cfra_score),
                        "cfra_minus_raw_same_foreign_excess_fisher_z": float(cfra_score - raw_score),
                        "norm_ratio_cfra_over_raw": float(np.linalg.norm(cfra) / max(np.linalg.norm(raw), 1e-12)),
                    })
    if len(eval_rows) != len(cell_lines) * len(common_confirmation_units) * 2:
        raise RuntimeError(f"unexpected eval row count: {len(eval_rows)}")
    write_csv(args.outdir / "LOCO_POOL_same_foreign_per_condition.csv", eval_rows)
    write_csv(args.outdir / "LOCO_POOL_calibration_weights.csv", weight_rows)

    pair_summary_rows: list[dict[str, Any]] = []
    target_summaries: dict[str, dict[str, Any]] = {}
    for target in cell_lines:
        rows = [row for row in eval_rows if row["target_cell_line"] == target]
        summary = summary_for(rows, f"target|{target}")
        target_summaries[target] = summary
        source_label = "+".join(models[target]["sources"])
        pair_summary_rows.extend([
            {"source_cell_lines": source_label, "target_cell_line": target, "metric": "Raw Same-Foreign excess Fisher-z", "method": "Raw", **summary["raw"]},
            {"source_cell_lines": source_label, "target_cell_line": target, "metric": "CFRA Same-Foreign excess Fisher-z", "method": "CFRA", **summary["cfra"]},
            {"source_cell_lines": source_label, "target_cell_line": target, "metric": "CFRA-Raw Same-Foreign excess Fisher-z", "method": "CFRA-Raw", **summary["delta"]},
            {"source_cell_lines": source_label, "target_cell_line": target, "metric": "||CFRA||/||Raw||", "method": "CFRA", **summary["norm"]},
        ])
    write_csv(args.outdir / "LOCO_POOL_source_target_summary.csv", pair_summary_rows)

    # Dose-stratified summaries are required to expose a potential dose shift
    # that could be hidden by averaging the four dose levels.
    dose_rows: list[dict[str, Any]] = []
    doses = sorted({float(unit[1]) for unit in common_confirmation_units})
    dose_target_values: dict[str, dict[float, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for target in cell_lines:
        for dose in doses:
            rows = [row for row in eval_rows if row["target_cell_line"] == target and float(row["dose_nM"]) == dose]
            summary = summary_for(rows, f"dose|{target}|{dose}")
            source_label = "+".join(models[target]["sources"])
            dose_rows.extend([
                {"source_cell_lines": source_label, "target_cell_line": target, "dose_nM": dose, "metric": "Raw Same-Foreign excess Fisher-z", "method": "Raw", **summary["raw"]},
                {"source_cell_lines": source_label, "target_cell_line": target, "dose_nM": dose, "metric": "CFRA Same-Foreign excess Fisher-z", "method": "CFRA", **summary["cfra"]},
                {"source_cell_lines": source_label, "target_cell_line": target, "dose_nM": dose, "metric": "CFRA-Raw Same-Foreign excess Fisher-z", "method": "CFRA-Raw", **summary["delta"]},
                {"source_cell_lines": source_label, "target_cell_line": target, "dose_nM": dose, "metric": "||CFRA||/||Raw||", "method": "CFRA", **summary["norm"]},
            ])
            delta_by_drug = per_drug(rows, "cfra_minus_raw_same_foreign_excess_fisher_z")
            dose_target_values[target][dose] = delta_by_drug
    write_csv(args.outdir / "LOCO_POOL_dose_summary.csv", dose_rows)

    macro_rows: list[dict[str, Any]] = []
    for key, metric, method in (("raw", "Raw Same-Foreign excess Fisher-z", "Raw"), ("cfra", "CFRA Same-Foreign excess Fisher-z", "CFRA"), ("delta", "CFRA-Raw Same-Foreign excess Fisher-z", "CFRA-Raw"), ("norm", "||CFRA||/||Raw||", "CFRA")):
        target_values = {target: per_drug([row for row in eval_rows if row["target_cell_line"] == target], {"raw": "raw_same_foreign_excess_fisher_z", "cfra": "cfra_same_foreign_excess_fisher_z", "delta": "cfra_minus_raw_same_foreign_excess_fisher_z", "norm": "norm_ratio_cfra_over_raw"}[key]) for target in cell_lines}
        stat = hierarchical_bootstrap(target_values, stable_seed("macro", key))
        macro_rows.append({"scope": "ALL_TARGETS_MACRO", "dose_nM": None, "metric": metric, "method": method, **stat})
    for dose in doses:
        stat = hierarchical_bootstrap({target: dose_target_values[target][dose] for target in cell_lines}, stable_seed("dose_macro", dose))
        macro_rows.append({"scope": "ALL_TARGETS_MACRO", "dose_nM": dose, "metric": "CFRA-Raw Same-Foreign excess Fisher-z", "method": "CFRA-Raw", **stat})
    write_csv(args.outdir / "LOCO_POOL_target_macro_summary.csv", macro_rows)

    raw_gate = json.loads(args.raw_gate.read_text(encoding="utf-8"))
    # Resolve the overall target-macro paired endpoint before writing the
    # audit, because the audit stores this value as a frozen result record.
    macro_delta = next(row for row in macro_rows if row["metric"] == "CFRA-Raw Same-Foreign excess Fisher-z" and row.get("dose_nM") in (None, ""))
    fold_delta_cis = {target: target_summaries[target]["delta"] for target in cell_lines}
    fold_ci_low_positive = {target: bool(fold_delta_cis[target]["ci_low"] > 0) for target in cell_lines}
    if macro_delta["ci_low"] > 0 and all(fold_ci_low_positive.values()):
        gate_status = "GO"
    elif macro_delta["ci_high"] < 0:
        gate_status = "NO_GO"
    else:
        gate_status = "INCONCLUSIVE"
    audit = {
        "dataset": "sci-Plex3 Zenodo 7041849",
        "protocol": "leave-one-cell-line-out pooled source fit/calibration and direct target application",
        "formal_main_analysis": True,
        "single_source_six_direction_analysis": "not_run_not_main",
        "input_files": {
            "group_counts": {"path": str(args.group_counts), "sha256": digest(args.group_counts), "md5": digest(args.group_counts, "md5"), "shape": shape},
            "conditions": {"path": str(args.conditions), "sha256": digest(args.conditions)},
            "split_lock": {"path": str(args.split_lock), "sha256": digest(args.split_lock)},
            "raw_gate": {"path": str(args.raw_gate), "sha256": digest(args.raw_gate)},
            "script": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__).resolve())},
        },
        "raw_h5_md5_expected": "d1f51b9f8de35ca07638132539da9a99",
        "raw_h5_md5_note": "the current group_counts.h5 is the upstream pseudo-bulk derivative; its actual checksum is recorded under input_files.group_counts",
        "time_hours": 24,
        "nperts": 1,
        "n_min": 50,
        "replicates": list(REPS),
        "cell_lines": cell_lines,
        "leave_one_out_design": [{"sources": [cell for cell in cell_lines if cell != target], "target": target} for target in cell_lines],
        "common_universe": {
            "definition": "drug-dose unit present with both rep1 and rep2 n>=50 rows in all three cell lines",
            "n_drugs": len(common_drugs),
            "n_drug_dose_units": len(common_units),
            "n_confirmation_drugs": len(common_confirmation_drugs),
            "n_confirmation_units": len(common_confirmation_units),
            "expected": {"n_drugs": EXPECTED_COMMON_DRUGS, "n_drug_dose_units": EXPECTED_COMMON_UNITS, "n_confirmation_drugs": EXPECTED_CONFIRMATION_DRUGS, "n_confirmation_units": EXPECTED_CONFIRMATION_UNITS},
            "unit_list_sha256": digest(args.outdir / "LOCO_POOL_COMMON_DRUG_DOSE_UNITS.csv"),
        },
        "drug_split": {"seed": SEED, "base_n": len(base_drugs), "calibration_n": len(calibration_drugs), "confirmation_n": len(confirmation_drugs), "overlap": {"base_calibration": sorted(base_drugs & calibration_drugs), "base_confirmation": sorted(base_drugs & confirmation_drugs), "calibration_confirmation": sorted(calibration_drugs & confirmation_drugs)}},
        "feature_policy": {
            "feature_map": "GEO annotation row i -> H5 X[:,i] for i=0..110982; audited zero tail X[:,110983] excluded",
            "n_features_used": N_FEATURES_WITH_GEO_MAP,
            "hvg_n": N_HVG,
            "hvg_fit": "joint two-source base only, equal source rows",
            "target_hvg_fit": False,
            "target_expression_used_for_hvg": False,
        },
        "model_freeze": {
            "fit_scope": "two source cell lines, common base drug-dose units only",
            "calibration_scope": "two source cell lines, common calibration drug-dose units only",
            "source_training_unit": "source x drug-dose unit x ordered repeat rotation",
            "equal_source_training_weight": True,
            "target_confirmation_used_for_fit": False,
            "target_expression_used_for_fit_or_weight": False,
            "target_base_cal_role": "foreign null donors only, after model freeze",
            "estimators": ["Ridge IMR", "Ridge LSO", "IMCEB", "CFRA"],
            "lso_policy": "LSO=IMR because only two genuine biological repeats",
            "source_records": model_audit,
        },
        "endpoint": {
            "response": "log1p(CPM 1e6) treated pseudo-bulk minus same-plate/same-cell-line/same-repeat Vehicle pseudo-bulk",
            "ordered_rotations": ["rep1->rep2", "rep2->rep1"],
            "foreign_policy": "20 deterministic target-line same-dose held-repeat donors from target common base+calibration units, target drug excluded",
            "raw_cfra_same_target_rows": True,
            "raw_cfra_same_foreign_ids": True,
            "score_feature_space": "target profiles projected onto frozen joint-source HGV2000",
            "inference_unit": "drug",
            "dose_rotation_aggregation": "mean within drug before bootstrap",
            "bootstrap_draws": BOOT,
            "pair_ci": "drug-level bootstrap within each target (one pooled source->target map)",
            "overall_ci": "hierarchical target-cell-line then drug bootstrap",
            "dose_stratified": True,
        },
        "limitations": {
            "cross_cell_line_plate_block": "source and target cell lines occupy different plate blocks; same-plate controls remove within-target plate nuisance, but direct transfer cannot separate cell-line shift from cross-cell-line/plate-block shift",
            "cell_level_inference": False,
            "two_repeats": "only legal 1R repeat rotations; no independent third biological repeat for 2R",
        },
        "n_eval_rows": len(eval_rows),
        "n_eval_rows_per_target": {target: sum(1 for row in eval_rows if row["target_cell_line"] == target) for target in cell_lines},
        "results": {
            "target_fold_delta": fold_delta_cis,
            "target_fold_norm": {target: target_summaries[target]["norm"] for target in cell_lines},
            "overall_target_macro_delta": macro_delta,
        },
        "gate": {
            "status": gate_status,
            "rule": "GO iff overall target-macro paired delta CI low > 0 and every target fold paired delta CI low > 0; NO_GO iff overall CI high < 0; otherwise INCONCLUSIVE",
            "fold_ci_low_positive": fold_ci_low_positive,
        },
        "raw_phase0_gate": raw_gate,
        "status": "COMPLETED",
    }
    write_json(args.outdir / "LOCO_POOL_AUDIT.json", audit)

    report: list[str] = [
        "# sci-Plex3 leave-one-cell-line-out pooled CFRA generalization",
        "",
        "## Main question and design",
        "",
        "The formal main analysis pools the two non-target cell lines for fitting/calibration and transfers the frozen model directly to the held-out target:",
        "",
        "- A549 + K562 → MCF7",
        "- A549 + MCF7 → K562",
        "- K562 + MCF7 → A549",
        "",
        f"Only the three-line common paired drug-dose universe is used: {len(common_drugs)} drugs and {len(common_units)} units; the frozen confirmation subset contains {len(common_confirmation_drugs)} drugs and {len(common_confirmation_units)} units.",
        f"Scope is 24 h, nperts=1, n≥50, same-plate/same-cell-line/same-repeat Vehicle, with rep1→rep2 and rep2→rep1. The frozen drug split is base/calibration/confirmation={len(base_drugs)}/{len(calibration_drugs)}/{len(confirmation_drugs)}, seed={SEED}.",
        "",
        "## Leakage and fairness lock",
        "",
        "- Joint HGV2000, Ridge IMR/LSO and IMCEB shrinkage use only the two source lines' common base units. Training rows are source×drug-dose×rotation and have equal total source weight.",
        "- CFRA weights are selected only on the two source lines' common calibration units. Target confirmation expression is not used for HVG, fit, scale or weight selection.",
        "- Target base/calibration profiles are loaded only after the model is frozen and serve only as the target foreign-null donor pool.",
        "- Raw and CFRA use identical target support/held rows, identical 20 target foreign donor IDs, and identical frozen joint-source HGV2000 coordinates.",
        "- Dose and rotation rows are averaged within drug before 10,000-drug-bootstrap CIs. Overall CI is hierarchical target-cell-line then drug.",
        "",
        "## Source-pool → target results",
        "",
        "| Source pool → target | Raw | CFRA | CFRA − Raw (95% CI) | ||CFRA||/||Raw|| (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for target in cell_lines:
        s = target_summaries[target]
        report.append(f"| {' + '.join(models[target]['sources'])} → {target} | {s['raw']['estimate']:.6f} | {s['cfra']['estimate']:.6f} | {s['delta']['estimate']:.6f} [{s['delta']['ci_low']:.6f}, {s['delta']['ci_high']:.6f}] | {s['norm']['estimate']:.6f} [{s['norm']['ci_low']:.6f}, {s['norm']['ci_high']:.6f}] |")
    report.extend(["", "## Overall target-macro result", "", f"- Overall target-macro CFRA−Raw = {macro_delta['estimate']:.6f}, 95% CI [{macro_delta['ci_low']:.6f}, {macro_delta['ci_high']:.6f}], with equal target-cell-line weighting.", f"- Worst target fold by point estimate: {macro_delta['worst_target']} ({macro_delta['worst_target_estimate']:.6f}).", f"- Gate status: {gate_status}. GO requires the overall macro CI low and every target-fold CI low to be positive; NO_GO requires overall CI high < 0; otherwise INCONCLUSIVE."])
    report.extend(["", "## Dose-stratified CFRA−Raw", "", "| Dose (nM) | Target | Estimate | 95% CI | n drugs |", "|---:|---|---:|---:|---:|"])
    for row in dose_rows:
        if row["metric"] == "CFRA-Raw Same-Foreign excess Fisher-z":
            report.append(f"| {row['dose_nM']:.0f} | {row['target_cell_line']} | {row['estimate']:.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {row['n_drugs']} |")
    report.extend(["", "## Interpretation and limitation", "", "A positive paired CFRA−Raw interval indicates that the frozen pooled source model improves same-repeat target concordance over the paired Raw baseline in the source-pooled HGV space. This is a direct cross-cell-line transfer test, not a randomized separation of cell-line and plate-block effects: source and target lines occupy different plate blocks, so that shift remains a documented limitation. Only legal 1R rotations are available because the dataset has two genuine biological repeats.", ""])
    (args.outdir / "LOCO_POOL_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"target_summaries": target_summaries, "macro_delta": macro_delta}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
