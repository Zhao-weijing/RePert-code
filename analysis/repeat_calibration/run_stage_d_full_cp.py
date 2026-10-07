#!/usr/bin/env python3
"""Stage-D full CP test evaluation after passed RCEB Stage A/B/C gates.

For every BBBC047/cpg0004-LINCS CP budget (1R, 2R, 3R), this driver fits PCA
on training support means, locks its rank on the official validation split, and
only then opens the official test split.  RCEB uses the unchanged Stage-A
training-only covariance model.  The primary result is compound-balanced
held-out recovery excess Fisher-z with 10,000 paired bootstrap draws.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE.parent / "baselines"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ablation_common import (  # noqa: E402
    bootstrap_difference,
    example_manifest,
    metric_arrays,
    summarize_metrics,
    write_csv,
    write_json,
)
from run_rceb_screen import (  # noqa: E402
    DEFAULT_BBBC_ROOT,
    DEFAULT_CPG_CP,
    PAIR_SEED,
    VERSION as RCEB_VERSION,
    load_cp_rows,
    load_model,
    make_examples,
    pca_predictions,
    rceb_predict,
)


VERSION = "rceb-stage-d-full-cp-v1-2026-08-31"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=("BBBC047", "cpg0004-LINCS"), default=("BBBC047", "cpg0004-LINCS"))
    parser.add_argument("--budgets", nargs="+", type=int, choices=(1, 2, 3), default=(1, 2, 3))
    parser.add_argument("--bbbc-root", type=Path, default=DEFAULT_BBBC_ROOT)
    parser.add_argument("--cpg-cp", type=Path, default=DEFAULT_CPG_CP)
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--aggregate-only", action="store_true", help="repair/rebuild the two full-matrix CSVs from existing Stage-D condition artifacts without opening data files")
    return parser.parse_args(argv)


def primary_contrast(left: np.ndarray, right: np.ndarray, examples: Any, label: str, bootstrap_n: int) -> dict[str, Any]:
    l = metric_arrays(left, examples)
    r = metric_arrays(right, examples)
    point, low, high, n = bootstrap_difference(l["excess_z"], r["excess_z"], examples.compound, PAIR_SEED + sum(ord(x) for x in label), bootstrap_n)
    return {"comparison": label, "metric": "excess_z", "mean_difference": point, "ci_low": low, "ci_high": high, "n_compounds": n}


def run_condition(args: argparse.Namespace, dataset: str, budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stage_a = args.output_root / "stage_a" / dataset / "CP"
    if not (stage_a / "rceb_model.npz").exists():
        raise FileNotFoundError(f"missing Stage-A model for {dataset}")
    train_rows = load_cp_rows(dataset, "train", args.bbbc_root, args.cpg_cp)
    valid_rows = load_cp_rows(dataset, "valid", args.bbbc_root, args.cpg_cp)
    train = make_examples(train_rows, None, budget, PAIR_SEED, all_conditions=True, require_source=False)
    valid = make_examples(valid_rows, None, budget, PAIR_SEED, all_conditions=False, require_source=False)
    # Validation selection is completed before the test NPZ is loaded below.
    _, selected_rank, rank_curve = pca_predictions(train.support, valid.support, valid, PAIR_SEED)
    test_rows = load_cp_rows(dataset, "test", args.bbbc_root, args.cpg_cp)
    test = make_examples(test_rows, None, budget, PAIR_SEED, all_conditions=False, require_source=False)
    # Refit the already-selected PCA rank on train only; no test values enter
    # rank choice.  Reusing pca_predictions would select on test, so explicitly
    # reconstruct at the frozen rank here.
    from sklearn.decomposition import PCA
    pca = PCA(n_components=selected_rank, svd_solver="randomized", random_state=PAIR_SEED)
    pca.fit(train.support)
    pca_prediction = (pca.mean_ + ((test.support - pca.mean_) @ pca.components_.T) @ pca.components_).astype(np.float32)
    rceb = rceb_predict(load_model(stage_a / "rceb_model.npz"), test.support, budget)
    raw = test.support.astype(np.float32)
    summaries: list[dict[str, Any]] = []
    for name, prediction in (("raw_mean", raw), ("pca", pca_prediction), ("rceb", rceb)):
        row, _ = summarize_metrics(name, prediction, test, PAIR_SEED, "test", bootstrap_n=args.bootstrap_n)
        summaries.append(row)
    contrasts = [
        primary_contrast(rceb, raw, test, "rceb_minus_raw_mean", args.bootstrap_n),
        primary_contrast(rceb, pca_prediction, test, "rceb_minus_pca", args.bootstrap_n),
    ]
    for row in contrasts:
        row.update({"dataset": dataset, "target_modality": "CP", "budget": budget, "split": "test", "pair_seed": PAIR_SEED})
    outdir = args.output_root / "stage_d" / dataset / "CP" / f"budget{budget}"
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "summary.csv", summaries)
    write_csv(outdir / "contrasts.csv", contrasts)
    write_csv(outdir / "pca_validation_rank_curve.csv", rank_curve)
    write_csv(outdir / "test_pair_manifest.csv", example_manifest(test))
    np.savez_compressed(outdir / "test_predictions.npz", compound=test.compound, dose=test.dose, support=test.support, target=test.target, foreign=test.foreign, foreign_ok=test.foreign_ok.astype(np.uint8), raw_mean=raw, pca=pca_prediction, rceb=rceb)
    decision = {
        "version": VERSION,
        "rceb_version": RCEB_VERSION,
        "dataset": dataset,
        "modality": "CP",
        "budget": budget,
        "pair_seed": PAIR_SEED,
        "model_fit": "RCEB Stage-A model fit on official train CP plate means only",
        "pca": {"fit": "training CP support means only", "selected_rank_validation_only": selected_rank},
        "test_access": "test CP opened only after PCA rank was fixed on validation",
        "primary_contrasts": contrasts,
        "n_test_pairs": len(test),
        "n_test_foreign": int(np.sum(test.foreign_ok)),
    }
    write_json(outdir / "decision.json", decision)
    print(f"[stage-d] {dataset} CP {budget}R: RCEB-PCA excess-z={contrasts[1]['mean_difference']:.6f}", flush=True)
    return summaries, contrasts, decision


def rebuild_aggregate_only(args: argparse.Namespace) -> None:
    """Rebuild consolidated CSVs from completed condition artifacts only."""
    root = args.output_root / "stage_d"
    all_summaries: list[dict[str, Any]] = []
    all_contrasts: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for budget in args.budgets:
            outdir = root / dataset / "CP" / f"budget{budget}"
            decision_path, summary_path, contrast_path = outdir / "decision.json", outdir / "summary.csv", outdir / "contrasts.csv"
            if not all(path.exists() for path in (decision_path, summary_path, contrast_path)):
                raise FileNotFoundError(f"missing completed Stage-D condition artifact: {outdir}")
            decisions.append(json.loads(decision_path.read_text(encoding="utf-8")))
            with summary_path.open(newline="", encoding="utf-8") as handle:
                all_summaries.extend(dict(row) for row in csv.DictReader(handle))
            with contrast_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row.update({"dataset": dataset, "target_modality": "CP", "budget": budget, "split": "test", "pair_seed": PAIR_SEED})
                    all_contrasts.append(row)
    write_csv(root / "FULL_CP_SUMMARY.csv", all_summaries)
    write_csv(root / "FULL_CP_PRIMARY_CONTRASTS.csv", all_contrasts)
    payload = {
        "version": VERSION,
        "stage": "D",
        "conditions": decisions,
        "bootstrap": {"unit": "compound; repeated condition rows mean-aggregated", "n": args.bootstrap_n, "interval": "95% percentile"},
        "primary_endpoint": "held-out recovery excess Fisher-z against strict matched foreign",
        "status": "COMPLETED",
        "aggregate_rebuilt_without_data_access": True,
        "limit": "Aggregate-only rebuild reads completed Stage-D CSV/JSON artifacts; it does not open CP NPZ data or recompute a prediction/metric.",
    }
    write_json(root / "STAGE_D_COMPLETED.json", payload)
    print("[stage-d] aggregate-only COMPLETED", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if set(args.datasets) != {"BBBC047", "cpg0004-LINCS"} or set(args.budgets) != {1, 2, 3}:
        raise SystemExit("Stage D is preregistered only for BBBC047/cpg0004-LINCS × CP 1R/2R/3R")
    stage_c = args.output_root / "stage_c" / "STAGE_C_DECISION.json"
    if not stage_c.exists() or not bool(json.loads(stage_c.read_text(encoding="utf-8")).get("passed")):
        raise SystemExit("Stage D requires a passed Stage C gate")
    if args.aggregate_only:
        rebuild_aggregate_only(args)
        return
    all_summaries: list[dict[str, Any]] = []
    all_contrasts: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for budget in args.budgets:
            summaries, contrasts, decision = run_condition(args, dataset, budget)
            all_summaries.extend(summaries)
            all_contrasts.extend(contrasts)
            decisions.append(decision)
    root = args.output_root / "stage_d"
    write_csv(root / "FULL_CP_SUMMARY.csv", all_summaries)
    write_csv(root / "FULL_CP_PRIMARY_CONTRASTS.csv", all_contrasts)
    payload = {
        "version": VERSION,
        "stage": "D",
        "conditions": decisions,
        "bootstrap": {"unit": "compound; repeated condition rows mean-aggregated", "n": args.bootstrap_n, "interval": "95% percentile"},
        "primary_endpoint": "held-out recovery excess Fisher-z against strict matched foreign",
        "status": "COMPLETED",
        "limit": "RCEB is deterministic conditional on the frozen training data and pair manifest; no multi-seed neural retraining is applicable. Stage-D strong-baseline refits are intentionally not extrapolated beyond the Stage-C validation gate.",
    }
    write_json(root / "STAGE_D_COMPLETED.json", payload)
    (root / "STAGE_D_COMPLETED.md").write_text("# RCEB Stage D complete\n\nSix CP test conditions completed with frozen validation PCA ranks and compound bootstrap. See `FULL_CP_SUMMARY.csv` and `FULL_CP_PRIMARY_CONTRASTS.csv`.\n", encoding="utf-8")
    print("[stage-d] COMPLETED", flush=True)


if __name__ == "__main__":
    main()
