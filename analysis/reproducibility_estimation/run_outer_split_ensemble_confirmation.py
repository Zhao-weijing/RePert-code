#!/usr/bin/env python3
"""One frozen compound-disjoint outer confirmation for the unified ensemble.

For each dataset, the pre-existing physical training split is deterministically
partitioned by compound into 60% base fitting, 20% ensemble calibration, and
20% untouched outer confirmation.  Teacher, LSO, RCEB, Raw and PCA are built
without access to the calibration/outer compounds.  Only the fixed
non-negative simplex ensemble weights see calibration targets; the outer
compounds are never used for fitting, epoch selection, rank selection, or
ensemble weighting.  The run covers BBBC/cpg CP 1R and 3R only, and never
opens a test file or test prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASELINE_DIR = ROOT / "baselines"
RCEB_DIR = ROOT / "repeat_calibration"
for folder in (HERE, BASELINE_DIR, RCEB_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from ablation_common import (  # noqa: E402
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
from run_phase1_prototype import DATASETS, PAIR_SEED, SEED, load_rows, select_device  # noqa: E402
from run_rceb_screen import fit_rceb, rceb_predict  # noqa: E402
from run_stage_c_strong_baselines import fit_baseline, lso_labels, predict, torch_modules  # noqa: E402
from run_validation_calibrated_ensemble import CANDIDATES, shrunken_weight, simplex_mse_weights  # noqa: E402


VERSION = "outer-compound-split-unified-ensemble-confirmation-v1-2026-08-31"
OUTER_SPLIT_SEED = 3407
CONDITIONS = (("BBBC047", 1), ("BBBC047", 3), ("cpg0004-LINCS", 1), ("cpg0004-LINCS", 3))
DEFAULT_OUTPUT = Path("/path/to/data/AIDD/MVCPert_5_27/runs/outer_ensemble_confirmation_20260831")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bbbc-root", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"))
    parser.add_argument("--cpg-train", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/cpg0004_lincs_cp_train_valid_only_20260831/train_cp_plate_rows.npz"))
    parser.add_argument("--cpg-valid", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/cpg0004_lincs_cp_train_valid_only_20260831/valid_cp_plate_rows.npz"))
    parser.add_argument("--bootstrap-n", type=int, default=10_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def compound_partition(rows: Rows, dataset: str) -> dict[str, set[str]]:
    compounds = sorted(set(np.asarray(rows.compound, dtype=str).tolist()), key=lambda compound: (stable_int(OUTER_SPLIT_SEED, f"outer-ensemble-confirmation-v1|{dataset}|{compound}"), compound))
    if len(compounds) < 15:
        raise RuntimeError(f"too few compounds for 60/20/20 outer partition: {dataset}")
    base_end = int(np.floor(len(compounds) * 0.60))
    calibration_end = base_end + int(np.floor(len(compounds) * 0.20))
    partitions = {
        "base": set(compounds[:base_end]),
        "calibration": set(compounds[base_end:calibration_end]),
        "outer": set(compounds[calibration_end:]),
    }
    if any(not values for values in partitions.values()) or sum(map(len, partitions.values())) != len(compounds):
        raise RuntimeError(f"invalid outer partition: {dataset}")
    if partitions["base"] & partitions["calibration"] or partitions["base"] & partitions["outer"] or partitions["calibration"] & partitions["outer"]:
        raise RuntimeError("compound partition overlaps")
    return partitions


def subset_rows(rows: Rows, allowed: set[str], split: str) -> Rows:
    mask = np.isin(rows.compound, np.asarray(sorted(allowed), dtype=str))
    if not np.any(mask):
        raise RuntimeError(f"empty {split} rows")
    return Rows(rows.dataset, rows.modality, split, rows.compound[mask], rows.dose[mask], rows.plate[mask], rows.delta[mask])


def fit_pca(base_support: np.ndarray, calibration_support: np.ndarray, calibration: Any) -> tuple[Any, int, list[dict[str, Any]]]:
    from sklearn.decomposition import PCA

    maximum = min(max(PCA_CANDIDATES), base_support.shape[0], base_support.shape[1])
    ranks = sorted({min(int(rank), int(maximum)) for rank in PCA_CANDIDATES if min(int(rank), int(maximum)) >= 1})
    if not ranks:
        raise RuntimeError("no legal PCA rank")
    pca = PCA(n_components=maximum, svd_solver="randomized", random_state=PAIR_SEED)
    pca.fit(base_support)

    def reconstruct(values: np.ndarray, rank: int) -> np.ndarray:
        score = (values - pca.mean_) @ pca.components_[:rank].T
        return (pca.mean_ + score @ pca.components_[:rank]).astype(np.float32)

    curve: list[dict[str, Any]] = []
    for rank in ranks:
        prediction = reconstruct(calibration_support, rank)
        curve.append({"rank": rank, "calibration_excess_z_mean": float(np.nanmean(metric_arrays(prediction, calibration)["excess_z"]))})
    selected = sorted(curve, key=lambda row: (-row["calibration_excess_z_mean"], row["rank"]))[0]["rank"]
    return pca, int(selected), curve


def pca_reconstruct(pca: Any, values: np.ndarray, rank: int) -> np.ndarray:
    score = (values - pca.mean_) @ pca.components_[:rank].T
    return (pca.mean_ + score @ pca.components_[:rank]).astype(np.float32)


def metric_contrast(label: str, left: np.ndarray, right: np.ndarray, examples: Any, bootstrap_n: int) -> dict[str, Any]:
    point, low, high, n = bootstrap_difference(
        metric_arrays(left, examples)["excess_z"], metric_arrays(right, examples)["excess_z"], examples.compound,
        stable_int(SEED, f"outer-ensemble|{label}"), bootstrap_n,
    )
    return {"comparison": label, "metric": "excess_z", "mean_difference": float(point), "ci_low": float(low), "ci_high": float(high), "n_compounds": int(n)}


def macro_bootstrap(deltas: Sequence[np.ndarray], compounds: Sequence[np.ndarray], bootstrap_n: int) -> dict[str, float]:
    if len(deltas) != 4:
        raise ValueError("outer confirmation is frozen to four conditions")
    per_condition: list[np.ndarray] = []
    for values, ids in zip(deltas, compounds):
        grouped: dict[str, list[float]] = {}
        for compound, value in zip(np.asarray(ids, dtype=str), np.asarray(values, dtype=float)):
            if np.isfinite(value):
                grouped.setdefault(str(compound), []).append(float(value))
        if not grouped:
            raise RuntimeError("empty outer metric")
        per_condition.append(np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=float))
    point = float(np.mean([np.mean(values) for values in per_condition]))
    rng = np.random.default_rng(stable_int(SEED, "outer-ensemble|macro") % (2**63 - 1))
    draws = np.zeros(bootstrap_n, dtype=float)
    for values in per_condition:
        sampled = rng.integers(0, len(values), size=(bootstrap_n, len(values)))
        draws += values[sampled].mean(axis=1) / len(per_condition)
    return {"mean_difference": point, "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)), "bootstrap_n": int(bootstrap_n)}


def dataset_context(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    physical_train = load_rows(args, dataset, "train")
    partitions = compound_partition(physical_train, dataset)
    base_rows = subset_rows(physical_train, partitions["base"], "outer_base")
    calibration_rows = subset_rows(physical_train, partitions["calibration"], "outer_calibration")
    outer_rows = subset_rows(physical_train, partitions["outer"], "outer_confirmation")
    all_sets = [set(base_rows.compound), set(calibration_rows.compound), set(outer_rows.compound)]
    if all_sets[0] & all_sets[1] or all_sets[0] & all_sets[2] or all_sets[1] & all_sets[2]:
        raise RuntimeError(f"row subsets not compound-disjoint: {dataset}")
    return {"physical_train": physical_train, "partitions": partitions, "base_rows": base_rows, "calibration_rows": calibration_rows, "outer_rows": outer_rows}


def run_condition(args: argparse.Namespace, torch: Any, nn: Any, device: Any, dataset: str, budget: int, context: dict[str, Any], rceb_model: Any) -> dict[str, Any]:
    base = make_examples(context["base_rows"], None, budget, PAIR_SEED, all_conditions=True, require_source=False)
    calibration = make_examples(context["calibration_rows"], None, budget, PAIR_SEED, all_conditions=False, require_source=False)
    outer = make_examples(context["outer_rows"], None, budget, PAIR_SEED, all_conditions=False, require_source=False)
    lso_target, lso_meta = lso_labels(context["base_rows"], base)
    teacher_fit, teacher_meta = fit_baseline(
        torch, nn, base.support, base.target, base.target, base.compound,
        name=f"outer-teacher|{dataset}|b{budget}", epochs=80, batch_size=256, learning_rate=3e-4, weight_decay=1e-5, device=device,
    )
    lso_fit, lso_fit_meta = fit_baseline(
        torch, nn, base.support, lso_target, base.target, base.compound,
        name=f"outer-lso|{dataset}|b{budget}", epochs=80, batch_size=256, learning_rate=3e-4, weight_decay=1e-5, device=device,
    )
    calibration_predictions = {
        "historical_teacher_traininternal": predict(torch, teacher_fit, calibration.support, 256, device),
        "lso_traininternal": predict(torch, lso_fit, calibration.support, 256, device),
        "rceb": rceb_predict(rceb_model, calibration.support, budget),
    }
    outer_predictions = {
        "raw_mean": outer.support.astype(np.float32),
        "historical_teacher_traininternal": predict(torch, teacher_fit, outer.support, 256, device),
        "lso_traininternal": predict(torch, lso_fit, outer.support, 256, device),
        "rceb": rceb_predict(rceb_model, outer.support, budget),
    }
    candidate_calibration = np.stack([calibration_predictions[name] for name in CANDIDATES], axis=1)
    raw_weight = simplex_mse_weights(candidate_calibration, calibration.target)
    weight, gamma = shrunken_weight(raw_weight, len(set(np.asarray(calibration.compound, dtype=str).tolist())))
    outer_predictions["validation_calibrated_ensemble"] = np.einsum("nkd,k->nd", np.stack([outer_predictions[name] for name in CANDIDATES], axis=1), weight, optimize=True).astype(np.float32)
    pca, selected_rank, pca_curve = fit_pca(base.support, calibration.support, calibration)
    outer_predictions["pca"] = pca_reconstruct(pca, outer.support, selected_rank)
    candidate_scores = {name: float(np.nanmean(metric_arrays(outer_predictions[name], outer)["excess_z"])) for name in CANDIDATES}
    best_name = sorted(CANDIDATES, key=lambda name: (-candidate_scores[name], name))[0]
    ensemble = outer_predictions["validation_calibrated_ensemble"]
    comparisons = [
        metric_contrast("ensemble_minus_best_single_posthoc", ensemble, outer_predictions[best_name], outer, args.bootstrap_n),
        metric_contrast("ensemble_minus_raw_mean", ensemble, outer_predictions["raw_mean"], outer, args.bootstrap_n),
        metric_contrast("ensemble_minus_pca", ensemble, outer_predictions["pca"], outer, args.bootstrap_n),
    ]
    output = args.outdir / dataset / "CP" / f"budget{budget}"
    output.mkdir(parents=True)
    summary = []
    for name, values in outer_predictions.items():
        row, _ = summarize_metrics(name, values, outer, SEED, "outer_confirmation", bootstrap_n=args.bootstrap_n)
        summary.append(row)
    write_csv(output / "outer_summary.csv", summary)
    write_csv(output / "outer_contrasts.csv", comparisons)
    write_csv(output / "pca_calibration_rank_curve.csv", pca_curve)
    write_csv(output / "outer_pair_manifest.csv", example_manifest(outer))
    np.savez_compressed(output / "outer_predictions.npz", compound=outer.compound, dose=outer.dose, support=outer.support, target=outer.target, foreign=outer.foreign, foreign_ok=outer.foreign_ok.astype(np.uint8), **outer_predictions)
    return {
        "dataset": dataset,
        "budget": budget,
        "base_examples": int(len(base)),
        "calibration_examples": int(len(calibration)),
        "outer_examples": int(len(outer)),
        "outer_compounds": int(len(set(outer.compound.tolist()))),
        "best_single_posthoc": best_name,
        "candidate_scores": candidate_scores,
        "ensemble_weight_raw": {name: float(raw_weight[index]) for index, name in enumerate(CANDIDATES)},
        "ensemble_weight": {name: float(weight[index]) for index, name in enumerate(CANDIDATES)},
        "ensemble_gamma_uniform_prior": float(gamma),
        "pca_rank_selected_on_calibration": int(selected_rank),
        "comparisons": {row["comparison"]: row for row in comparisons},
        "outer": outer,
        "ensemble": ensemble,
        "best_prediction": outer_predictions[best_name],
        "raw": outer_predictions["raw_mean"],
        "pca": outer_predictions["pca"],
        "teacher_meta": teacher_meta,
        "lso_meta": lso_fit_meta | lso_meta,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite frozen outer confirmation output: {args.outdir}")
    if args.bootstrap_n < 100:
        raise ValueError("bootstrap-n must be at least 100")
    torch, nn = torch_modules()
    device = select_device(torch, args.device)
    contexts = {dataset: dataset_context(args, dataset) for dataset in DATASETS}
    args.outdir.mkdir(parents=True)
    partition_manifest = []
    for dataset, context in contexts.items():
        for partition, compounds in context["partitions"].items():
            partition_manifest.extend({"dataset": dataset, "partition": partition, "compound": compound} for compound in sorted(compounds))
    write_csv(args.outdir / "OUTER_COMPOUND_PARTITION.csv", partition_manifest)
    outcomes: list[dict[str, Any]] = []
    rceb_audit: dict[str, Any] = {}
    for dataset in DATASETS:
        rceb_model, rceb_audit[dataset] = fit_rceb(contexts[dataset]["base_rows"], ridge_relative=1e-6)
        for selected_dataset, budget in CONDITIONS:
            if selected_dataset == dataset:
                outcomes.append(run_condition(args, torch, nn, device, dataset, budget, contexts[dataset], rceb_model))
    condition_rows = []
    bstar_deltas = []
    bstar_compounds = []
    gates_per_condition = []
    for outcome in outcomes:
        best = outcome["comparisons"]["ensemble_minus_best_single_posthoc"]
        raw = outcome["comparisons"]["ensemble_minus_raw_mean"]
        pca = outcome["comparisons"]["ensemble_minus_pca"]
        condition_rows.append({
            "dataset": outcome["dataset"], "budget": outcome["budget"], "outer_examples": outcome["outer_examples"], "outer_compounds": outcome["outer_compounds"], "best_single_posthoc": outcome["best_single_posthoc"],
            "ensemble_minus_best_single": best["mean_difference"], "best_ci_low": best["ci_low"], "best_ci_high": best["ci_high"],
            "ensemble_minus_raw": raw["mean_difference"], "ensemble_minus_pca": pca["mean_difference"],
            **{f"weight_{name}": value for name, value in outcome["ensemble_weight"].items()}, "ensemble_gamma": outcome["ensemble_gamma_uniform_prior"], "pca_rank": outcome["pca_rank_selected_on_calibration"],
        })
        bstar_deltas.append(metric_arrays(outcome["ensemble"], outcome["outer"])["excess_z"] - metric_arrays(outcome["best_prediction"], outcome["outer"])["excess_z"])
        bstar_compounds.append(outcome["outer"].compound)
        gates_per_condition.append({"best_gt_neg_0_01": best["mean_difference"] > -0.01, "best_nonnegative": best["mean_difference"] >= 0.0, "beats_raw": raw["mean_difference"] > 0.0, "beats_pca": pca["mean_difference"] > 0.0})
    macro = macro_bootstrap(bstar_deltas, bstar_compounds, args.bootstrap_n)
    gates = {
        "all_4_ensemble_minus_best_single_gt_neg_0_01": bool(all(item["best_gt_neg_0_01"] for item in gates_per_condition)),
        "at_least_3_of_4_ensemble_minus_best_single_nonnegative": bool(sum(item["best_nonnegative"] for item in gates_per_condition) >= 3),
        "four_condition_macro_ensemble_minus_best_single_positive": bool(macro["mean_difference"] > 0.0),
        "all_4_ensemble_point_beats_raw": bool(all(item["beats_raw"] for item in gates_per_condition)),
        "all_4_ensemble_point_beats_pca": bool(all(item["beats_pca"] for item in gates_per_condition)),
    }
    passed = bool(all(gates.values()))
    decision = {
        "version": VERSION,
        "scope": "One deterministic 60/20/20 compound-disjoint outer confirmation, BBBC/cpg CP 1R/3R only",
        "test_loaded": False,
        "outer_split_seed": OUTER_SPLIT_SEED,
        "partition_rule": "unique physical-training compounds sorted by stable hash; first 60% base fitting, next 20% ensemble/PCA calibration, remaining 20% untouched outer confirmation",
        "candidate_rule": "Teacher/LSO fit only on base, with their fixed compound-disjoint internal epoch selection; RCEB covariance fit only on base; no candidate accesses calibration or outer profiles during fitting",
        "ensemble_rule": "only fixed exact simplex reconstruction weights on calibration profiles, followed by fixed Dirichlet(1,1,1) pseudo-compound shrinkage; one frozen weight vector per dataset×budget",
        "pca_rule": "PCA fit on base supports; fixed candidate rank selected only on calibration; PCA is not an ensemble candidate",
        "requirements": {"safety": "all four ensemble-B* > -0.01", "coverage": "at least three of four ensemble-B* point estimates >= 0", "macro": "equal-condition outer macro ensemble-B* > 0", "foundation": "ensemble point estimate > Raw and > PCA in every condition"},
        "rceb_base_audits": rceb_audit,
        "conditions": condition_rows,
        "macro_ensemble_minus_best_single": macro,
        "gates": gates,
        "passed": passed,
        "decision": "OUTER-ENSEMBLE-CONFIRMED" if passed else "OUTER-ENSEMBLE-NO-GO",
        "next_step": "CONFIRMED supports a pre-registered final six-condition confirmation after refitting candidates on base+calibration with outer-calibrated weights frozen. NO-GO ends the unified-estimator claim; do not test alternative ensemble features or search candidates/weights.",
    }
    write_csv(args.outdir / "OUTER_CONDITION_SUMMARY.csv", condition_rows)
    write_json(args.outdir / "OUTER_DECISION.json", decision)
    (args.outdir / "README.md").write_text("# Outer ensemble confirmation\n\nOne frozen compound-disjoint outer confirmation. No test input/prediction is opened. See `OUTER_DECISION.json`.\n", encoding="utf-8")
    print(json.dumps({"decision": decision["decision"], "gates": gates, "outdir": str(args.outdir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
