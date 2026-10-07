#!/usr/bin/env python3
"""Leakage-free Stage A/B screen for the replicate-calibrated EB estimator.

RCEB fits its prior and noise covariance only from training CP plate means.
Every condition is a (compound, dose) key and every distinct acquisition plate
contributes one profile.  Technical wells are averaged within plate before any
covariance calculation.  Validation is never read while fitting the model.

The Stage-B PCA comparator is intentionally allowed to select its rank on the
same validation split, exactly as a screening-strength comparator.  Thus a
successful RCEB screen is conservative with respect to PCA, but Stage B is not
a final confirmatory result and never reads the test split.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE.parent / "baselines"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from ablation_common import (  # noqa: E402
    BOOTSTRAP_N,
    PCA_CANDIDATES,
    Rows,
    bootstrap_difference,
    example_manifest,
    make_examples,
    metric_arrays,
    stable_int,
    summarize_metrics,
    write_csv,
    write_json,
)


VERSION = "rceb-cp-screen-v1-2026-08-31"
PAIR_SEED = 3407
DEFAULT_BBBC_ROOT = Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0")
DEFAULT_CPG_CP = Path("/path/to/data/AIDD/MVCPert_5_27/runs/cpg0004_lincs_cp_prepared_20260830/artifact/cp_plate_rows.npz")
DEFAULT_CPG_GE = Path("/path/to/data/AIDD/MVCPert_5_27/runs/cpg0004_lincs_ge_prepared_20260830/artifact/ge_plate_rows.npz")
DEFAULT_CPG_LOCK = Path("/path/to/data/AIDD/MVCPert_5_27/runs/cpg0004_lincs_cp_prepared_20260830/artifact/split_lock.json")
DEFAULT_OUTPUT = Path("/path/to/data/AIDD/MVCPert_5_27/runs/rceb_repeat_calibrated_20260831")


@dataclass(frozen=True)
class RCEBModel:
    mean: np.ndarray
    noise_raw: np.ndarray
    signal_raw: np.ndarray
    noise_regularized: np.ndarray
    cholesky_noise: np.ndarray
    generalized_eigenvalues: np.ndarray
    generalized_eigenvectors: np.ndarray
    ridge_relative: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--datasets", nargs="+", choices=("BBBC047", "cpg0004-LINCS"), default=("BBBC047", "cpg0004-LINCS"))
    parser.add_argument("--budgets", nargs="+", type=int, choices=(1, 2, 3), default=(1, 3))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bbbc-root", type=Path, default=DEFAULT_BBBC_ROOT)
    parser.add_argument("--cpg-cp", type=Path, default=DEFAULT_CPG_CP)
    parser.add_argument("--cpg-ge", type=Path, default=DEFAULT_CPG_GE)
    parser.add_argument("--cpg-lock", type=Path, default=DEFAULT_CPG_LOCK)
    parser.add_argument("--ridge-relative", type=float, default=1e-6, help="fixed numerical ridge relative to mean diagonal of Sigma_noise; never selected on validation")
    parser.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_N)
    return parser.parse_args(argv)


def _symmetric(values: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float64) + np.asarray(values, dtype=np.float64).T) / 2.0).astype(np.float64, copy=False)


def load_cp_rows(dataset: str, split: str, bbbc_root: Path, cpg_cp_path: Path) -> Rows:
    """Load exactly one CP split, never materialising another split by accident."""
    if dataset == "BBBC047":
        path = bbbc_root / f"{split}_cp_plate_rows.npz"
        data = np.load(path, allow_pickle=True)
        return Rows(dataset, "CP", split, data["smiles"], data["dose"], data["plate"], data["delta"])
    if dataset == "cpg0004-LINCS":
        data = np.load(cpg_cp_path, allow_pickle=True)
        labels = np.asarray([str(x.decode("utf-8", errors="replace")) if isinstance(x, (bytes, np.bytes_)) else str(x) for x in data["split"]], dtype=str)
        mask = labels == split
        return Rows(dataset, "CP", split, data["compound_id"][mask], data["dose"][mask], data["plate"][mask], data["delta"][mask])
    raise ValueError(f"unsupported dataset: {dataset}")


def _spectrum_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    quantile_points = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "positive_n": int(np.sum(values > 1e-10)),
        "nonnegative_n": int(np.sum(values >= -1e-10)),
        "quantiles": {str(q): float(np.quantile(values, q)) for q in quantile_points},
        "count_lambda_gt": {str(x): int(np.sum(values > x)) for x in (0.0, 0.01, 0.1, 0.25, 1.0)},
    }


def _generalized_spectrum(cholesky_noise: np.ndarray, cross_covariance: np.ndarray) -> np.ndarray:
    """Return the untruncated generalized spectrum for an audit covariance."""
    whitened = np.linalg.solve(cholesky_noise, _symmetric(cross_covariance))
    whitened = _symmetric(np.linalg.solve(cholesky_noise, whitened.T).T)
    return np.linalg.eigvalsh(whitened)


def fit_rceb(rows: Rows, ridge_relative: float) -> tuple[RCEBModel, dict[str, Any]]:
    """Estimate replicate signal/noise covariances using only train plate means.

    For condition c, let m_c be its mean plate profile and N_c its sample
    within-condition covariance.  The two requested pair estimators are

      Sigma_noise = mean_c N_c
      Sigma_signal = mean_c [(m_c-mu0)(m_c-mu0)^T - N_c/n_c]

    The second expression is exactly the average ordered cross-repeat
    covariance after centering at mu0, but avoids materialising O(n_c^2)
    plate pairs.  Conditions, rather than individual rows/pairs, are equally
    weighted so high-repeat compounds cannot dominate the estimator.
    """
    if not (np.isfinite(ridge_relative) and ridge_relative > 0):
        raise ValueError("ridge_relative must be positive and finite")
    group_values: list[np.ndarray] = []
    group_keys: list[tuple[str, str]] = []
    repeat_histogram: dict[int, int] = {}
    for key, by_plate in rows.plate_means().items():
        values = np.stack([by_plate[plate] for plate in sorted(by_plate)], axis=0).astype(np.float64, copy=False)
        if len(values) < 2:
            continue
        group_values.append(values)
        group_keys.append(key)
        repeat_histogram[len(values)] = repeat_histogram.get(len(values), 0) + 1
    if not group_values:
        raise RuntimeError("RCEB requires at least one training condition with two independent plates")

    means = np.vstack([values.mean(axis=0) for values in group_values])
    center = means.mean(axis=0)
    # D_c / sqrt(n_c-1), stacked by condition.  Its cross-product is sum N_c.
    within = np.vstack([
        (values - values.mean(axis=0, keepdims=True)) / math.sqrt(len(values) - 1)
        for values in group_values
    ])
    pair_weights = np.concatenate([np.full(len(values), 1.0 / len(values), dtype=np.float64) for values in group_values])
    n_conditions = len(group_values)
    noise_raw = _symmetric((within.T @ within) / n_conditions)
    # Reuse the same residual matrix: after row scaling, its cross-product is
    # sum_c N_c / n_c, the finite-repeat correction in cross covariance.
    within *= np.sqrt(pair_weights)[:, None]
    cross_repeat_correction = _symmetric((within.T @ within) / n_conditions)
    centered_means = means - center
    signal_raw = _symmetric((centered_means.T @ centered_means) / n_conditions - cross_repeat_correction)

    dimension = rows.dim
    diagonal_scale = float(np.trace(noise_raw) / dimension)
    if not (np.isfinite(diagonal_scale) and diagonal_scale > 0):
        raise RuntimeError(f"invalid Sigma_noise diagonal scale: {diagonal_scale}")
    noise_regularized = _symmetric(noise_raw + np.eye(dimension, dtype=np.float64) * (ridge_relative * diagonal_scale))
    try:
        chol = np.linalg.cholesky(noise_regularized)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Sigma_noise remained non-positive-definite after fixed ridge") from exc
    whitened = np.linalg.solve(chol, signal_raw)
    whitened = _symmetric(np.linalg.solve(chol, whitened.T).T)
    generalized_values, generalized_vectors = np.linalg.eigh(whitened)
    model = RCEBModel(
        mean=center.astype(np.float64),
        noise_raw=noise_raw,
        signal_raw=signal_raw,
        noise_regularized=noise_regularized,
        cholesky_noise=chol,
        generalized_eigenvalues=generalized_values,
        generalized_eigenvectors=generalized_vectors,
        ridge_relative=float(ridge_relative),
    )
    raw_signal_values = np.linalg.eigvalsh(signal_raw)
    gain_1r = np.clip(generalized_values, 0.0, None) / (np.clip(generalized_values, 0.0, None) + 1.0)
    gain_3r = np.clip(generalized_values, 0.0, None) / (np.clip(generalized_values, 0.0, None) + 1.0 / 3.0)
    # Controls are audit-only.  They are built from the same condition means
    # but cross different compounds, so their spectrum detects shared plate or
    # control structure that would otherwise masquerade as repeat signal.
    shuffled_order = sorted(range(n_conditions), key=lambda i: (stable_int(PAIR_SEED, f"rceb-shuffled|{group_keys[i][0]}|{group_keys[i][1]}"), group_keys[i]))
    shuffled_donor = {index: shuffled_order[(position + 1) % n_conditions] for position, index in enumerate(shuffled_order)}
    shuffled_cross = _symmetric(sum(
        np.outer(centered_means[index], centered_means[donor])
        for index, donor in shuffled_donor.items()
    ) / n_conditions)
    signature_index: dict[tuple[str, tuple[str, ...]], list[int]] = {}
    for index, (key, values) in enumerate(zip(group_keys, group_values)):
        signature_index.setdefault((key[1], tuple(sorted(rows.plate_means()[key]))), []).append(index)
    foreign_left: list[int] = []
    foreign_right: list[int] = []
    for index, key in enumerate(group_keys):
        signature = (key[1], tuple(sorted(rows.plate_means()[key])))
        candidates = [candidate for candidate in signature_index[signature] if group_keys[candidate][0] != key[0]]
        if not candidates:
            continue
        donor = candidates[stable_int(PAIR_SEED, f"rceb-matched-foreign|{key[0]}|{key[1]}|{signature}") % len(candidates)]
        foreign_left.append(index)
        foreign_right.append(donor)
    if foreign_left:
        foreign_cross = _symmetric(sum(
            np.outer(centered_means[left], centered_means[right])
            for left, right in zip(foreign_left, foreign_right)
        ) / len(foreign_left))
        foreign_values = _generalized_spectrum(chol, foreign_cross)
    else:
        foreign_cross = np.full_like(signal_raw, np.nan)
        foreign_values = np.full(dimension, np.nan, dtype=np.float64)
    shuffled_values = _generalized_spectrum(chol, shuffled_cross)
    strong_minimum = max(10, int(math.ceil(0.02 * dimension)))
    strong_directions = int(np.sum(generalized_values > 0.1))
    correct_tail = float(np.quantile(generalized_values, 0.95))
    foreign_tail = float(np.quantile(foreign_values, 0.95)) if np.all(np.isfinite(foreign_values)) else float("nan")
    spectrum_pass = bool(
        np.all(np.isfinite(generalized_values))
        and strong_directions >= strong_minimum
        and correct_tail > 0.1
        and float(np.max(generalized_values)) > 0.25
        and np.isfinite(foreign_tail)
        and correct_tail > foreign_tail
    )
    metadata: dict[str, Any] = {
        "version": VERSION,
        "fit_split": "train",
        "modality": "CP",
        "unit": "one mean profile per distinct acquisition plate; technical wells averaged within plate",
        "condition_key": "(compound, nominal dose)",
        "condition_weighting": "equal condition weight; not row-weighted or pair-weighted",
        "n_conditions_with_at_least_2_plates": n_conditions,
        "n_plate_profiles_used": int(sum(len(values) for values in group_values)),
        "feature_dim": dimension,
        "repeat_histogram": {str(k): int(v) for k, v in sorted(repeat_histogram.items())},
        "mu0": {"norm": float(np.linalg.norm(center)), "mean": float(np.mean(center)), "std": float(np.std(center))},
        "noise_ridge_relative": float(ridge_relative),
        "noise_diagonal_scale": diagonal_scale,
        "signal_covariance_eigenvalues": _spectrum_summary(raw_signal_values),
        "noise_whitened_generalized_eigenvalues": _spectrum_summary(generalized_values),
        "control_spectra": {
            "shuffled_condition_cross_covariance": {
                "rule": "deterministic cross-condition cyclic permutation; all conditions; no same-condition pair",
                "n_pairs": n_conditions,
                "noise_whitened_generalized_eigenvalues": _spectrum_summary(shuffled_values),
            },
            "matched_foreign_cross_covariance": {
                "rule": "deterministic different-compound donor with identical nominal dose and full acquisition-plate signature",
                "n_pairs": len(foreign_left),
                "coverage": float(len(foreign_left) / n_conditions),
                "noise_whitened_generalized_eigenvalues": _spectrum_summary(foreign_values) if len(foreign_left) else {"status": "NO_ELIGIBLE_MATCHED_FOREIGN"},
                "correct_minus_matched_foreign_q95": float(correct_tail - foreign_tail) if np.isfinite(foreign_tail) else float("nan"),
            },
        },
        "implied_gain": {
            "1R": _spectrum_summary(gain_1r),
            "3R": _spectrum_summary(gain_3r),
        },
        "stage_a_rule": {
            "minimum_lambda_gt_0_1": strong_minimum,
            "requires": [
                "finite generalized spectrum",
                "at least minimum_lambda_gt_0_1 directions with lambda > 0.1",
                "95th percentile lambda > 0.1",
                "maximum lambda > 0.25",
                "correct 95th-percentile generalized eigenvalue exceeds matched-foreign 95th percentile",
            ],
        },
        "stage_a_pass": spectrum_pass,
        "stage_a_decision": "GO_TO_STAGE_B" if spectrum_pass else "STOP_SPECTRUM",
    }
    return model, metadata


def save_model(path: Path, model: RCEBModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean=model.mean,
        noise_raw=model.noise_raw,
        signal_raw=model.signal_raw,
        noise_regularized=model.noise_regularized,
        cholesky_noise=model.cholesky_noise,
        generalized_eigenvalues=model.generalized_eigenvalues,
        generalized_eigenvectors=model.generalized_eigenvectors,
        ridge_relative=np.asarray([model.ridge_relative], dtype=np.float64),
    )


def load_model(path: Path) -> RCEBModel:
    data = np.load(path, allow_pickle=False)
    return RCEBModel(
        mean=np.asarray(data["mean"], dtype=np.float64),
        noise_raw=np.asarray(data["noise_raw"], dtype=np.float64),
        signal_raw=np.asarray(data["signal_raw"], dtype=np.float64),
        noise_regularized=np.asarray(data["noise_regularized"], dtype=np.float64),
        cholesky_noise=np.asarray(data["cholesky_noise"], dtype=np.float64),
        generalized_eigenvalues=np.asarray(data["generalized_eigenvalues"], dtype=np.float64),
        generalized_eigenvectors=np.asarray(data["generalized_eigenvectors"], dtype=np.float64),
        ridge_relative=float(np.asarray(data["ridge_relative"], dtype=np.float64).reshape(-1)[0]),
    )


def rceb_predict(model: RCEBModel, support: np.ndarray, budget: int) -> np.ndarray:
    if budget < 1:
        raise ValueError("budget must be at least one")
    values = np.asarray(support, dtype=np.float64)
    whitened = np.linalg.solve(model.cholesky_noise, (values - model.mean).T)
    rotated = model.generalized_eigenvectors.T @ whitened
    positive = np.clip(model.generalized_eigenvalues, 0.0, None)
    gains = positive / (positive + 1.0 / budget)
    corrected = model.mean[:, None] + model.cholesky_noise @ (model.generalized_eigenvectors @ (gains[:, None] * rotated))
    return corrected.T.astype(np.float32)


def pca_predictions(train_support: np.ndarray, valid_support: np.ndarray, valid_examples: Any, pair_seed: int) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
    from sklearn.decomposition import PCA

    maximum = min(max(PCA_CANDIDATES), train_support.shape[0], train_support.shape[1])
    ranks = sorted({min(int(rank), int(maximum)) for rank in PCA_CANDIDATES if min(int(rank), int(maximum)) >= 1})
    if not ranks:
        ranks = [maximum]
    pca = PCA(n_components=maximum, svd_solver="randomized", random_state=pair_seed)
    pca.fit(train_support)

    def reconstruct(values: np.ndarray, rank: int) -> np.ndarray:
        scores = (values - pca.mean_) @ pca.components_[:rank].T
        return (pca.mean_ + scores @ pca.components_[:rank]).astype(np.float32)

    curve: list[dict[str, Any]] = []
    candidate_predictions: dict[int, np.ndarray] = {}
    for rank in ranks:
        prediction = reconstruct(valid_support, rank)
        candidate_predictions[rank] = prediction
        metrics = metric_arrays(prediction, valid_examples)
        curve.append({
            "rank": rank,
            "validation_excess_z_mean": float(np.nanmean(metrics["excess_z"])),
            "validation_pcc_mean": float(np.nanmean(metrics["pcc"])),
            "validation_foreign_n": int(np.sum(valid_examples.foreign_ok)),
        })
    selected = sorted(curve, key=lambda row: (-row["validation_excess_z_mean"], row["rank"]))[0]["rank"]
    return candidate_predictions[int(selected)], int(selected), curve


def _contrast_rows(rceb: np.ndarray, references: dict[str, np.ndarray], examples: Any, bootstrap_n: int) -> list[dict[str, Any]]:
    rceb_metrics = metric_arrays(rceb, examples)
    rows: list[dict[str, Any]] = []
    for reference_name, prediction in references.items():
        reference_metrics = metric_arrays(prediction, examples)
        for metric in ("excess_z", "pcc", "delta_pcc", "excess_pcc"):
            point, low, high, n = bootstrap_difference(
                rceb_metrics[metric], reference_metrics[metric], examples.compound,
                PAIR_SEED + int(sum(ord(ch) for ch in f"{reference_name}|{metric}")), bootstrap_n,
            )
            rows.append({
                "comparison": f"rceb_minus_{reference_name}",
                "metric": metric,
                "mean_difference": point,
                "ci_low": low,
                "ci_high": high,
                "n_compounds": n,
            })
    return rows


def run_stage_a(args: argparse.Namespace) -> None:
    for dataset in args.datasets:
        outdir = args.output_root / "stage_a" / dataset / "CP"
        outdir.mkdir(parents=True, exist_ok=True)
        model, summary = fit_rceb(load_cp_rows(dataset, "train", args.bbbc_root, args.cpg_cp), float(args.ridge_relative))
        summary["dataset"] = dataset
        summary["source_paths"] = {
            "bbbc_root": str(args.bbbc_root),
            "cpg_cp": str(args.cpg_cp),
            "test_loaded": False,
        }
        save_model(outdir / "rceb_model.npz", model)
        write_json(outdir / "spectrum.json", summary)
        text = (
            f"# RCEB Stage A — {dataset} CP\n\n"
            f"Decision: `{summary['stage_a_decision']}`.\n\n"
            f"Training-only conditions with at least two independent plates: `{summary['n_conditions_with_at_least_2_plates']}`. "
            f"Plate profiles: `{summary['n_plate_profiles_used']}`. Feature dimension: `{summary['feature_dim']}`.\n\n"
            "The spectrum is the generalized eigenvalue lambda of signal relative to replicate noise. "
            "For R support plates, its implied shrinkage is lambda/(lambda+1/R). "
            "All thresholds and counts are recorded in `spectrum.json`; no validation or test profile was used here.\n"
        )
        (outdir / "STAGE_A_SUMMARY.md").write_text(text, encoding="utf-8")
        print(f"[stage-a] {dataset} {summary['stage_a_decision']}", flush=True)


def run_one_stage_b(args: argparse.Namespace, dataset: str, budget: int) -> dict[str, Any]:
    stage_a = args.output_root / "stage_a" / dataset / "CP"
    spectrum_path = stage_a / "spectrum.json"
    model_path = stage_a / "rceb_model.npz"
    if not spectrum_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Stage A artifacts are required before Stage B: {stage_a}")
    spectrum = json.loads(spectrum_path.read_text(encoding="utf-8"))
    if not bool(spectrum.get("stage_a_pass")):
        raise RuntimeError(f"Stage B is forbidden after Stage A STOP: {dataset}")
    model = load_model(model_path)
    train_rows = load_cp_rows(dataset, "train", args.bbbc_root, args.cpg_cp)
    valid_rows = load_cp_rows(dataset, "valid", args.bbbc_root, args.cpg_cp)
    # Strict CP-only evaluation: no GE-common filtering is used in this screen.
    train = make_examples(train_rows, None, budget, PAIR_SEED, all_conditions=True, require_source=False)
    valid = make_examples(valid_rows, None, budget, PAIR_SEED, all_conditions=False, require_source=False)
    raw = valid.support.astype(np.float32)
    rceb = rceb_predict(model, valid.support, budget)
    pca, selected_rank, pca_curve = pca_predictions(train.support, valid.support, valid, PAIR_SEED)
    outdir = args.output_root / "stage_b" / dataset / "CP" / f"budget{budget}"
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, values in (("raw_mean", raw), ("pca", pca), ("rceb", rceb)):
        row, _ = summarize_metrics(name, values, valid, PAIR_SEED, "valid", bootstrap_n=int(args.bootstrap_n))
        summary_rows.append(row)
    contrasts = _contrast_rows(rceb, {"pca": pca, "raw_mean": raw}, valid, int(args.bootstrap_n))
    primary = {(row["comparison"], row["metric"]): row for row in contrasts}
    rceb_pca = primary[("rceb_minus_pca", "excess_z")]
    rceb_raw = primary[("rceb_minus_raw_mean", "excess_z")]
    decision = {
        "dataset": dataset,
        "modality": "CP",
        "budget": budget,
        "split": "valid",
        "version": VERSION,
        "pair_seed": PAIR_SEED,
        "evaluation_population": "CP-only; one deterministic compound-dose condition per validation compound; independent held plate",
        "fit_population": "official training compounds only; all plate means from each eligible (compound,dose) condition",
        "pca": {
            "fit": "training CP support means",
            "rank_selection": "selected on this validation split for an intentionally screening-strength PCA comparator",
            "selected_rank": selected_rank,
        },
        "rceb": {
            "validation_selected_parameters": False,
            "noise_ridge_relative": model.ridge_relative,
            "stage_a_model": str(model_path),
        },
        "primary_contrasts": {
            "rceb_minus_pca_excess_z": rceb_pca,
            "rceb_minus_raw_mean_excess_z": rceb_raw,
        },
        "screen_flags": {
            "better_than_pca": bool(float(rceb_pca["mean_difference"]) > 0.0),
            "significantly_lower_than_raw": bool(float(rceb_raw["ci_high"]) < 0.0),
        },
    }
    write_csv(outdir / "summary.csv", summary_rows)
    write_csv(outdir / "contrasts.csv", contrasts)
    write_csv(outdir / "pca_validation_rank_curve.csv", pca_curve)
    write_csv(outdir / "valid_pair_manifest.csv", example_manifest(valid))
    np.savez_compressed(
        outdir / "valid_predictions.npz",
        compound=valid.compound,
        dose=valid.dose,
        support=valid.support,
        target=valid.target,
        foreign=valid.foreign,
        foreign_ok=valid.foreign_ok.astype(np.uint8),
        raw_mean=raw,
        pca=pca,
        rceb=rceb,
    )
    write_json(outdir / "decision.json", decision)
    print(f"[stage-b] {dataset} CP {budget}R: RCEB-PCA excess-z={rceb_pca['mean_difference']:.6f}; RCEB-Raw={rceb_raw['mean_difference']:.6f}", flush=True)
    return decision


def run_stage_b(args: argparse.Namespace) -> None:
    decisions = [run_one_stage_b(args, dataset, budget) for dataset in args.datasets for budget in args.budgets]
    if len(decisions) != 4 or set(args.datasets) != {"BBBC047", "cpg0004-LINCS"} or set(args.budgets) != {1, 3}:
        raise RuntimeError("The Stage-B gate is defined only for BBBC047/cpg0004-LINCS × CP 1R/3R; run precisely those four conditions")
    pca_differences = [float(row["primary_contrasts"]["rceb_minus_pca_excess_z"]["mean_difference"]) for row in decisions]
    raw_failures = [row for row in decisions if bool(row["screen_flags"]["significantly_lower_than_raw"])]
    pass_pca = int(sum(delta > 0.0 for delta in pca_differences)) >= 3
    pass_raw = not raw_failures
    pass_macro = float(np.mean(pca_differences)) > 0.0
    passed = bool(pass_pca and pass_raw and pass_macro)
    gate = {
        "version": VERSION,
        "stage": "B",
        "conditions": decisions,
        "rule": {
            "pca": "RCEB primary excess Fisher-z is numerically greater than PCA in at least 3 of 4 conditions",
            "raw": "no condition has a paired compound-bootstrap 95% CI wholly below zero for RCEB minus Raw primary excess Fisher-z",
            "macro": "unweighted mean of the four RCEB-minus-PCA primary excess-z differences is greater than zero",
        },
        "observed": {
            "rceb_better_than_pca_n": int(sum(delta > 0.0 for delta in pca_differences)),
            "rceb_minus_pca_macro_excess_z": float(np.mean(pca_differences)),
            "significantly_below_raw_conditions": [{"dataset": row["dataset"], "budget": row["budget"]} for row in raw_failures],
        },
        "passed": passed,
        "decision": "GO_TO_STAGE_C" if passed else "STOP_AFTER_STAGE_B",
        "limits": "Stage B is validation-only. PCA rank was selected on validation and is therefore a deliberately strong screen comparator. No test profile was loaded or scored.",
    }
    root = args.output_root / "stage_b"
    write_json(root / "STAGE_B_DECISION.json", gate)
    lines = [
        "# RCEB Stage B decision",
        "",
        f"Decision: `{gate['decision']}`.",
        "",
        f"RCEB exceeded PCA in `{gate['observed']['rceb_better_than_pca_n']}/4` validation conditions; macro RCEB−PCA excess Fisher-z was `{gate['observed']['rceb_minus_pca_macro_excess_z']:.6f}`.",
        f"Conditions significantly below Raw: `{len(raw_failures)}`.",
        "",
        "This is CP-only and validation-only; it is a screening gate, not a test claim.",
    ]
    (root / "STAGE_B_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[stage-b] {gate['decision']}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.stage == "a":
        run_stage_a(args)
    else:
        run_stage_b(args)


if __name__ == "__main__":
    main()
