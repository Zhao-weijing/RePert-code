#!/usr/bin/env python3
"""No-training RCEB comparison on the immutable historical CP pair bundles.

This script reads only already-completed Historical Teacher/LSO artifacts and
the frozen RCEB covariance models.  RCEB is projected onto the exact saved
support vectors from each historical test bundle.  It neither fits a model nor
loads any Stage-C/D RCEB predictions.

Each learned method has historical seeds 3407, 42, and 2025.  The bootstrap
unit is compound: values are first averaged across those fixed seeds within a
compound, then bootstrapped.  B* is selected only after Teacher, LSO, and RCEB
scores are complete for a condition; it never affects a prediction, parameter,
or model choice.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from run_rceb_screen import PAIR_SEED, load_model, rceb_predict  # noqa: E402


VERSION = "rceb-canonical-historical-six-condition-v1-2026-08-31"
SEEDS = (3407, 42, 2025)
DEFAULT_RCEB_ROOT = Path("/path/to/data/AIDD/MVCPert_5_27/runs/rceb_repeat_calibrated_20260831_strict")
DEFAULT_HISTORY_ROOT = Path("/path/to/data/AIDD/MVCPert_5_27/runs/historical_single_modality_ablation_20260831")
DEFAULT_BBBC_SOURCE = DEFAULT_HISTORY_ROOT / "source"
DEFAULT_BBBC_HISTORY = Path("/path/to/data/AIDD/MVCPert_5_27/runs/repro_effect_information_budget_20260829")
DEFAULT_BBBC_ROWS = Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0")
DEFAULT_BBBC_LOCK = Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json")
DEFAULT_CPG_TEACHER = (Path(__file__).resolve().parents[2] / "analysis/external_validation/lincs_cpg0004/cell_painting_repeat_benchmark/run_cp_repeat_benchmark.py")
DEFAULT_CPG_DATA = Path("/path/to/data/AIDD/MVCPert_5_27/runs/cpg0004_lincs_cp_prepared_20260830/artifact/cp_plate_rows.npz")
DEFAULT_CPG_HISTORY = Path("/path/to/data/AIDD/MVCPert_5_27/runs/cpg0004_lincs_cp_repeat_20260830")


DECISION_RULES = {
    "significant_lower": "paired compound-bootstrap 95% CI upper bound < 0",
    "case_a": "no one of six RCEB-B* contrasts significantly lower; six-condition macro point estimate > -0.005; and at least one RCEB-Teacher or RCEB-LSO CI lower bound > 0",
    "case_b": "-0.02 < macro RCEB-B* < -0.005; no condition has RCEB-B* CI upper bound < -0.02; both dataset-level macro point estimates are negative",
    "case_c": "macro RCEB-B* < -0.02 or at least two conditions are significantly lower than B*",
    "otherwise": "INDETERMINATE_STOP; no nonlinear residual is authorized",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rceb-root", type=Path, default=DEFAULT_RCEB_ROOT)
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY_ROOT)
    parser.add_argument("--bbbc-source", type=Path, default=DEFAULT_BBBC_SOURCE)
    parser.add_argument("--bbbc-history", type=Path, default=DEFAULT_BBBC_HISTORY)
    parser.add_argument("--bbbc-rows", type=Path, default=DEFAULT_BBBC_ROWS)
    parser.add_argument("--bbbc-lock", type=Path, default=DEFAULT_BBBC_LOCK)
    parser.add_argument("--cpg-teacher", type=Path, default=DEFAULT_CPG_TEACHER)
    parser.add_argument("--cpg-data", type=Path, default=DEFAULT_CPG_DATA)
    parser.add_argument("--cpg-history", type=Path, default=DEFAULT_CPG_HISTORY)
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.outdir is None:
        args.outdir = args.rceb_root / "canonical_historical_comparison"
    return args


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_equal(label: str, left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        maximum = float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))) if left.shape == right.shape and np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number) else float("nan")
        raise RuntimeError(f"canonical bundle mismatch: {label}; max_abs={maximum}")


def dict_values(scores: Mapping[str, Mapping[str, Any]], field: str = "model_excess_z") -> dict[str, float]:
    return {str(key): float(value[field]) for key, value in scores.items()}


def pool_seed_scores(seed_scores: Mapping[int, Mapping[str, Mapping[str, float]]]) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    methods = ("rceb", "teacher", "lso")
    common: set[str] | None = None
    for seed in SEEDS:
        for method in methods:
            keys = set(seed_scores[seed][method])
            common = keys if common is None else common & keys
    if not common:
        raise RuntimeError("no common compounds over method/seed score maps")
    compounds = np.asarray(sorted(common), dtype=str)
    pooled = {
        method: np.asarray([
            np.mean([seed_scores[seed][method][compound] for seed in SEEDS])
            for compound in compounds
        ], dtype=float)
        for method in methods
    }
    coverage = {
        "common_compounds_all_method_seed": int(len(compounds)),
        "per_seed_method_compounds": {str(seed): {method: int(len(seed_scores[seed][method])) for method in methods} for seed in SEEDS},
    }
    return pooled, compounds, coverage


def bootstrap_difference(values: np.ndarray, compounds: np.ndarray, seed: int, n_boot: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if len(values) != len(compounds) or not len(values):
        raise ValueError("invalid bootstrap values")
    point = float(np.mean(values))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    offset = 0
    while offset < n_boot:
        count = min(256, n_boot - offset)
        index = rng.integers(0, len(values), size=(count, len(values)))
        boot[offset : offset + count] = values[index].mean(axis=1)
        offset += count
    return {"mean_difference": point, "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)), "n_compounds": int(len(values))}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty csv {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bbbc_condition(args: argparse.Namespace, runner: ModuleType, budget: int) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    C, B = runner.modules(args.bbbc_source)
    rows = C.BASE.load_splits(args.bbbc_rows, args.bbbc_lock)
    test, support_rows, manifest = runner.budget_pairs(C, B, rows["test"], "test", budget)
    model = load_model(args.rceb_root / "stage_a" / "BBBC047" / "CP" / "rceb_model.npz")
    seed_scores: dict[int, dict[str, dict[str, float]]] = {}
    canonical_checks: list[dict[str, Any]] = []
    for seed in SEEDS:
        artifact = args.history_root / f"bbbc_lso_formal_b{budget}_seed{seed}" / "test_predictions.npz"
        saved = np.load(artifact, allow_pickle=True)
        assert_equal(f"BBBC{budget} seed{seed} smiles", np.asarray(test.smiles, dtype=str), np.asarray(saved["smiles"], dtype=str))
        assert_equal(f"BBBC{budget} seed{seed} held rows", test.held_rows, saved["held_rows"])
        assert_equal(f"BBBC{budget} seed{seed} support", test.support, saved["support"])
        assert_equal(f"BBBC{budget} seed{seed} target", test.target, saved["target"])
        reference = runner.reference(C, rows["test"], test, support_rows, 32, seed)
        rceb_values = rceb_predict(model, saved["support"], budget)
        rceb_scores = C.BASE.method_stats(runner.prediction_map(test, rceb_values), reference, rows["test"])
        lso_scores = C.BASE.method_stats(runner.prediction_map(test, saved["aggregate_target_teacher"]), reference, rows["test"])
        history = args.bbbc_history / f"seed{seed}" / f"budget{budget}_per_molecule_scores.csv"
        teacher_scores = runner.historical_scores(history, f"{budget}R_teacher")
        seed_scores[seed] = {
            "rceb": dict_values(rceb_scores),
            "teacher": dict_values(teacher_scores),
            "lso": dict_values(lso_scores),
        }
        canonical_checks.append({"seed": seed, "pair_count": len(test.smiles), "reference_compounds": len(reference), "teacher_score_compounds": len(teacher_scores), "lso_score_compounds": len(lso_scores), "rceb_score_compounds": len(rceb_scores)})
    pooled, compounds, coverage = pool_seed_scores(seed_scores)
    return pooled, compounds, {"protocol": "BBBC047 historical canonical pair bundle; fixed pair manifest seed 3407; 32-draw strict plate-slot foreign", "canonical_checks": canonical_checks, **coverage}


def cpg_condition(args: argparse.Namespace, runner: ModuleType, budget: int) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    C = runner.load_teacher_module(args.cpg_teacher)
    rows = C.load_rows(args.cpg_data)
    pairs = C.make_pairs(rows, budget, "all")
    test_compounds = set(rows.compound[rows.split == "test"])
    test = C.subset_pairs(pairs, test_compounds)
    model = load_model(args.rceb_root / "stage_a" / "cpg0004-LINCS" / "CP" / "rceb_model.npz")
    seed_scores: dict[int, dict[str, dict[str, float]]] = {}
    canonical_checks: list[dict[str, Any]] = []
    for seed in SEEDS:
        artifact = args.history_root / f"cpg_lso_formal_b{budget}_seed{seed}" / "test_predictions.npz"
        saved = np.load(artifact, allow_pickle=True)
        assert_equal(f"cpg{budget} seed{seed} compound", np.asarray(test.compound, dtype=str), np.asarray(saved["compound"], dtype=str))
        assert_equal(f"cpg{budget} seed{seed} dose", np.asarray(test.dose, dtype=str), np.asarray(saved["dose"], dtype=str))
        assert_equal(f"cpg{budget} seed{seed} held index", test.held_index, saved["held_index"])
        assert_equal(f"cpg{budget} seed{seed} support", test.support, saved["support"])
        assert_equal(f"cpg{budget} seed{seed} target", test.target, saved["target"])
        usable = np.asarray(saved["usable"], dtype=bool)
        null_z = np.asarray(saved["null_z"], dtype=float)
        history = args.cpg_history / f"full_b{budget}_all_seed{seed}" / "per_pair_scores.csv"
        teacher_excess, _, teacher_pair = runner.historical_scores(history, test, usable)
        for index in np.flatnonzero(usable):
            if abs(float(teacher_pair[int(index)]["null_z"]) - float(null_z[index])) > 1e-9:
                raise RuntimeError(f"cpg{budget} seed{seed} historical null mismatch at pair {index}")
        lso_excess, _, _ = runner.score_by_compound(C, saved["aggregate_target_teacher"], test, null_z, usable)
        rceb_values = rceb_predict(model, saved["support"], budget)
        rceb_excess, _, _ = runner.score_by_compound(C, rceb_values, test, null_z, usable)
        seed_scores[seed] = {"rceb": rceb_excess, "teacher": teacher_excess, "lso": lso_excess}
        canonical_checks.append({"seed": seed, "pair_count": len(test.compound), "usable_pairs": int(usable.sum()), "teacher_score_compounds": len(teacher_excess), "lso_score_compounds": len(lso_excess), "rceb_score_compounds": len(rceb_excess)})
    pooled, compounds, coverage = pool_seed_scores(seed_scores)
    return pooled, compounds, {"protocol": "cpg0004 historical all-dose/every-legal-held-plate canonical pair bundle; exact stored matched null", "canonical_checks": canonical_checks, **coverage}


def condition_results(dataset: str, budget: int, pooled: Mapping[str, np.ndarray], compounds: np.ndarray, bootstrap_n: int) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    means = {method: float(np.mean(values)) for method, values in pooled.items()}
    learned_best = max(("teacher", "lso"), key=lambda method: means[method])
    per_compound: list[dict[str, Any]] = []
    # Preserve individual Teacher/LSO rows exactly once plus a descriptive B*
    # summary.  B* is picked from completed condition means only.
    comparisons = [
        {"dataset": dataset, "budget": budget, "comparison": f"rceb_minus_{method}", "baseline": method, **bootstrap_difference(pooled["rceb"] - pooled[method], compounds, PAIR_SEED + budget * 1009 + sum(ord(x) for x in f"{dataset}|{method}"), bootstrap_n)}
        for method in ("teacher", "lso")
    ] + [{"dataset": dataset, "budget": budget, "comparison": "rceb_minus_bstar_learned", "baseline": learned_best, **bootstrap_difference(pooled["rceb"] - pooled[learned_best], compounds, PAIR_SEED + budget * 1009 + sum(ord(x) for x in f"{dataset}|bstar"), bootstrap_n)}]
    for index, compound in enumerate(compounds):
        per_compound.append({"dataset": dataset, "budget": budget, "compound": compound, "rceb_excess_z_pooled": pooled["rceb"][index], "teacher_excess_z_pooled": pooled["teacher"][index], "lso_excess_z_pooled": pooled["lso"][index], "bstar_learned": learned_best, "rceb_minus_bstar": pooled["rceb"][index] - pooled[learned_best][index]})
    summary = {"dataset": dataset, "budget": budget, "means": means, "bstar_learned": learned_best, "n_compounds": int(len(compounds))}
    return comparisons, summary, per_compound


def macro_bootstrap(condition_deltas: Sequence[np.ndarray], n_boot: int) -> dict[str, float]:
    point = float(np.mean([np.mean(values) for values in condition_deltas]))
    rng = np.random.default_rng(PAIR_SEED + 99173)
    boot = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        boot[draw] = float(np.mean([values[rng.integers(0, len(values), size=len(values))].mean() for values in condition_deltas]))
    return {"point": point, "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975))}


def decide(condition_rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]], macro: Mapping[str, float]) -> dict[str, Any]:
    bstar = [row for row in condition_rows if row["comparison"] == "rceb_minus_bstar_learned"]
    teacher_lso = [row for row in condition_rows if row["comparison"] in ("rceb_minus_teacher", "rceb_minus_lso")]
    significant_lower = [row for row in bstar if float(row["ci_high"]) < 0.0]
    significant_win = [row for row in teacher_lso if float(row["ci_low"]) > 0.0]
    by_dataset: dict[str, list[float]] = {}
    for row in bstar:
        by_dataset.setdefault(str(row["dataset"]), []).append(float(row["mean_difference"]))
    dataset_macro = {dataset: float(np.mean(values)) for dataset, values in by_dataset.items()}
    values = [float(row["mean_difference"]) for row in bstar]
    macro_point = float(macro["point"])
    case_a = not significant_lower and macro_point > -0.005 and bool(significant_win)
    case_b = (-0.02 < macro_point < -0.005 and not any(float(row["ci_high"]) < -0.02 for row in bstar) and all(value < 0 for value in dataset_macro.values()))
    case_c = macro_point < -0.02 or len(significant_lower) >= 2
    if case_a:
        decision, authorization = "A_FREEZE_RCEB", "Do not add a neural model. Freeze RCEB as the final estimator candidate."
    elif case_b:
        decision, authorization = "B_AUTHORIZE_LIGHT_NONLINEAR_RESIDUAL", "A small nonlinear residual is authorized; no feature-wise gate or architecture search."
    elif case_c:
        decision, authorization = "C_STOP_RCEB_AS_STATISTICAL_COMPONENT", "Do not add a neural residual. Keep RCEB only as a reproducibility-aware statistical baseline/component."
    else:
        decision, authorization = "INDETERMINATE_STOP", "No nonlinear residual is authorized because the frozen A/B/C criteria are not met."
    return {"rules": DECISION_RULES, "macro_rceb_minus_bstar": macro, "dataset_macro_rceb_minus_bstar": dataset_macro, "n_bstar_significantly_lower": len(significant_lower), "significantly_lower_conditions": [{"dataset": row["dataset"], "budget": row["budget"]} for row in significant_lower], "n_rceb_significantly_exceeds_teacher_or_lso": len(significant_win), "decision": decision, "authorization": authorization, "six_condition_bstar_differences": values}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite canonical comparison output: {args.outdir}")
    # Use the immutable runner copies archived beside the historical artifacts;
    # the remote project mirror intentionally does not contain the local
    # experiment directory itself.
    bbbc_runner = load_module("canonical_bbbc_historical_runner", args.history_root / "scripts" / "run_bbbc_historical_ablation.py")
    cpg_runner = load_module("canonical_cpg_historical_runner", args.history_root / "scripts" / "run_cpg_historical_ablation.py")
    condition_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    compound_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    deltas: list[np.ndarray] = []
    for dataset, build in (("BBBC047", bbbc_condition), ("cpg0004-LINCS", cpg_condition)):
        for budget in (1, 2, 3):
            pooled, compounds, audit = build(args, bbbc_runner if dataset == "BBBC047" else cpg_runner, budget)
            comparisons, summary, rows = condition_results(dataset, budget, pooled, compounds, args.bootstrap_n)
            bstar = summary["bstar_learned"]
            deltas.append(pooled["rceb"] - pooled[bstar])
            condition_rows.extend(comparisons)
            summaries.append(summary)
            compound_rows.extend(rows)
            audits.append({"dataset": dataset, "budget": budget, **audit})
            print(f"[canonical] {dataset} CP {budget}R B*={bstar}", flush=True)
    macro = macro_bootstrap(deltas, args.bootstrap_n)
    decision = decide(condition_rows, summaries, macro)
    args.outdir.mkdir(parents=True)
    write_csv(args.outdir / "CANONICAL_CONTRASTS.csv", condition_rows)
    write_csv(args.outdir / "CANONICAL_PER_COMPOUND_SCORES.csv", compound_rows)
    write_csv(args.outdir / "CANONICAL_CONDITION_SUMMARY.csv", summaries)
    (args.outdir / "CANONICAL_AUDIT.json").write_text(json.dumps({"version": VERSION, "seeds": list(SEEDS), "rceb_models": {"BBBC047": str(args.rceb_root / "stage_a" / "BBBC047" / "CP" / "rceb_model.npz"), "cpg0004-LINCS": str(args.rceb_root / "stage_a" / "cpg0004-LINCS" / "CP" / "rceb_model.npz")}, "audits": audits, "no_training": True, "no_stage_cd_predictions_loaded": True}, indent=2, sort_keys=True), encoding="utf-8")
    (args.outdir / "CANONICAL_DECISION.json").write_text(json.dumps({"version": VERSION, **decision}, indent=2, sort_keys=True), encoding="utf-8")
    (args.outdir / "CANONICAL_DECISION.md").write_text(f"# Canonical historical RCEB decision\n\nDecision: `{decision['decision']}`.\n\nSix-condition macro RCEB−B*: `{macro['point']:.6f}` (95% CI [{macro['ci_low']:.6f}, {macro['ci_high']:.6f}]).\n\n{decision['authorization']}\n", encoding="utf-8")
    print(json.dumps({"decision": decision["decision"], "macro": macro, "authorization": decision["authorization"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
