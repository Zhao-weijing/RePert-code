#!/usr/bin/env python3
"""Cross-fitted validation-calibrated Teacher/LSO/RCEB ensemble screen.

This is one fixed unified-algorithm screen, not model selection by hand:
  1. load the three already frozen candidate predictions on validation pairs;
  2. fit a non-negative, sum-to-one reconstruction ensemble on independent
     validation compounds only;
  3. use deterministic compound-level five-fold cross-fitting to score the
     ensemble on validation compounds not used for its weights.

The profile reconstruction objective, simplex constraint, fold rule and
analytic uniform-prior shrinkage are frozen below.  No model is trained, no
candidate is changed, and no test artifact is read.  If this screen passes,
the separately saved full-validation weights are deployment weights only; this
script never applies them to test profiles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE.parent / "baselines"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from ablation_common import (  # noqa: E402
    bootstrap_difference,
    example_manifest,
    metric_arrays,
    stable_int,
    summarize_metrics,
    write_csv,
    write_json,
)


VERSION = "validation-calibrated-convex-ensemble-v1-2026-08-31"
PAIR_SEED = 3407
FOLD_COUNT = 5
CANDIDATES = ("historical_teacher_traininternal", "lso_traininternal", "rceb")
UNIFORM = np.full(len(CANDIDATES), 1.0 / len(CANDIDATES), dtype=np.float64)
DEFAULT_STRICT_ROOT = Path("/path/to/data/AIDD/MVCPert_5_27/runs/rceb_repeat_calibrated_20260831_strict")
DEFAULT_OUTPUT = DEFAULT_STRICT_ROOT / "validation_calibrated_ensemble_20260831"
CONDITIONS = (("BBBC047", 1), ("BBBC047", 3), ("cpg0004-LINCS", 1), ("cpg0004-LINCS", 3))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-root", type=Path, default=DEFAULT_STRICT_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-n", type=int, default=10_000)
    return parser.parse_args(argv)


def assert_identical(name: str, left: np.ndarray, right: np.ndarray) -> None:
    left, right = np.asarray(left), np.asarray(right)
    equal = np.array_equal(left, right, equal_nan=True) if np.issubdtype(left.dtype, np.inexact) else np.array_equal(left, right)
    if left.shape != right.shape or not equal:
        raise RuntimeError(f"canonical validation bundle mismatch: {name}")


def simplex_mse_weights(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Exact non-negative least-squares solution on a three-candidate simplex.

    The objective is the unweighted sum of per-feature squared reconstruction
    errors across calibration profiles.  Enumerating the simplex interior,
    edges and vertices avoids optimizer tolerances and makes the rule fully
    deterministic.
    """
    values = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != len(CANDIDATES) or values.shape[0] != target.shape[0] or values.shape[2] != target.shape[1]:
        raise ValueError(f"invalid prediction/target shapes: {values.shape}, {target.shape}")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(target)):
        raise FloatingPointError("non-finite calibration profiles")

    def loss(weight: np.ndarray) -> float:
        residual = np.einsum("nkd,k->nd", values, weight, optimize=True) - target
        return float(np.einsum("nd,nd->", residual, residual, optimize=True))

    candidates: list[np.ndarray] = [np.eye(len(CANDIDATES), dtype=np.float64)[index] for index in range(len(CANDIDATES))]
    # Each edge has one scalar least-squares optimum, projected onto [0, 1].
    for left in range(len(CANDIDATES)):
        for right in range(left + 1, len(CANDIDATES)):
            base = values[:, right, :]
            direction = values[:, left, :] - base
            denominator = float(np.einsum("nd,nd->", direction, direction, optimize=True))
            coefficient = 0.0 if denominator <= 0.0 else float(np.einsum("nd,nd->", target - base, direction, optimize=True) / denominator)
            coefficient = float(np.clip(coefficient, 0.0, 1.0))
            weight = np.zeros(len(CANDIDATES), dtype=np.float64)
            weight[left] = coefficient
            weight[right] = 1.0 - coefficient
            candidates.append(weight)
    # Solve the equality-constrained interior candidate using candidates 0/1
    # relative to candidate 2.  It is retained only when it is in the simplex.
    base = values[:, 2, :]
    directions = np.stack((values[:, 0, :] - base, values[:, 1, :] - base), axis=-1)
    gram = np.einsum("ndk,ndl->kl", directions, directions, optimize=True)
    rhs = np.einsum("ndk,nd->k", directions, target - base, optimize=True)
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.full(2, np.nan)
    interior = np.asarray([coefficients[0], coefficients[1], 1.0 - coefficients.sum()], dtype=np.float64)
    if np.all(np.isfinite(interior)) and np.all(interior >= -1e-12):
        candidates.append(np.maximum(interior, 0.0) / np.maximum(interior.sum(), 1e-12))
    # Stable tie order: lexicographic tuple after objective value.
    return min(candidates, key=lambda weight: (loss(weight), tuple(float(x) for x in weight))).copy()


def shrunken_weight(fit_weight: np.ndarray, n_calibration_compounds: int) -> tuple[np.ndarray, float]:
    """Fixed Dirichlet(1,1,1) pseudo-compound shrinkage toward uniform.

    gamma=n/(n+3) is analytic from the three unit prior counts and number of
    calibration compounds, so it introduces neither a validation-tuned gamma
    nor a candidate-specific hyperparameter.
    """
    if n_calibration_compounds < 1:
        raise ValueError("at least one calibration compound is required")
    gamma = float(n_calibration_compounds / (n_calibration_compounds + len(CANDIDATES)))
    weight = (1.0 - gamma) * UNIFORM + gamma * np.asarray(fit_weight, dtype=np.float64)
    if not (np.all(np.isfinite(weight)) and np.all(weight >= -1e-12) and abs(float(weight.sum()) - 1.0) < 1e-9):
        raise RuntimeError("invalid shrunken simplex weight")
    return weight, gamma


def compound_folds(compounds: np.ndarray) -> np.ndarray:
    compounds = np.asarray(compounds, dtype=str)
    result = np.asarray([
        stable_int(PAIR_SEED, f"validation-calibrated-ensemble-fold-v1|{compound}") % FOLD_COUNT
        for compound in compounds
    ], dtype=int)
    if len(set(result.tolist())) != FOLD_COUNT:
        raise RuntimeError("deterministic five-fold partition unexpectedly has an empty fold")
    return result


def prediction_stack(stage_b: Mapping[str, np.ndarray], stage_c: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.stack((
        np.asarray(stage_c["historical_teacher_traininternal"], dtype=np.float32),
        np.asarray(stage_c["lso_traininternal"], dtype=np.float32),
        np.asarray(stage_b["rceb"], dtype=np.float32),
    ), axis=1)


def load_condition(strict_root: Path, dataset: str, budget: int) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    stage_b_dir = strict_root / "stage_b" / dataset / "CP" / f"budget{budget}"
    stage_c_dir = strict_root / "stage_c" / dataset / "CP" / f"budget{budget}"
    stage_b = np.load(stage_b_dir / "valid_predictions.npz", allow_pickle=False)
    stage_c = np.load(stage_c_dir / "valid_predictions.npz", allow_pickle=False)
    decision = json.loads((stage_c_dir / "decision.json").read_text(encoding="utf-8"))
    if decision.get("test_loaded") is not False:
        raise RuntimeError(f"{dataset} {budget}R Stage-C artifact lacks test_loaded=false audit")
    fields = ("compound", "dose", "support", "target", "foreign", "foreign_ok")
    for field in fields:
        assert_identical(f"{dataset}|{budget}R|{field}", np.asarray(stage_b[field]), np.asarray(stage_c[field]))
    for field in CANDIDATES:
        source = stage_b if field == "rceb" else stage_c
        if field not in source.files:
            raise RuntimeError(f"missing frozen candidate {field}: {dataset} {budget}R")
    values = {
        "compound": np.asarray(stage_b["compound"], dtype=str),
        "dose": np.asarray(stage_b["dose"], dtype=str),
        "support": np.asarray(stage_b["support"], dtype=np.float32),
        "target": np.asarray(stage_b["target"], dtype=np.float32),
        "foreign": np.asarray(stage_b["foreign"], dtype=np.float32),
        "foreign_ok": np.asarray(stage_b["foreign_ok"], dtype=bool),
        "historical_teacher_traininternal": np.asarray(stage_c["historical_teacher_traininternal"], dtype=np.float32),
        "lso_traininternal": np.asarray(stage_c["lso_traininternal"], dtype=np.float32),
        "rceb": np.asarray(stage_b["rceb"], dtype=np.float32),
    }
    metadata = {
        "dataset": dataset,
        "budget": budget,
        "stage_b_validation_predictions": str(stage_b_dir / "valid_predictions.npz"),
        "stage_c_validation_predictions": str(stage_c_dir / "valid_predictions.npz"),
        "stage_c_lso": "frozen train-internal LSO; loaded only, no retraining",
        "stage_c_teacher": "frozen train-internal Teacher; loaded only, no retraining",
        "canonical_fields_asserted": list(fields),
        "n_examples": int(len(values["compound"])),
        "n_compounds": int(len(set(values["compound"].tolist()))),
        "test_loaded": False,
    }
    return metadata, prediction_stack(stage_b, stage_c), values


def example_proxy(values: Mapping[str, np.ndarray], dataset: str, budget: int) -> Any:
    return type("ValidationExamples", (), {
        "__len__": lambda self: len(self.compound),
        "dataset": dataset,
        "modality": "CP",
        "split": "valid",
        "budget": budget,
        "seed": PAIR_SEED,
        "compound": values["compound"],
        "dose": values["dose"],
        "condition": np.asarray([f"{compound}::{dose}" for compound, dose in zip(values["compound"], values["dose"])], dtype=str),
        "support": values["support"],
        "target": values["target"],
        "aggregate_target": values["target"],
        "source": None,
        "foreign": values["foreign"],
        "foreign_ok": values["foreign_ok"],
        "support_plates": [tuple() for _ in values["compound"]],
        "held_plates": [tuple() for _ in values["compound"]],
        "foreign_compound": np.asarray(["" for _ in values["compound"]], dtype=str),
        "foreign_condition": np.asarray(["" for _ in values["compound"]], dtype=str),
    })()


def crossfit_ensemble(predictions: np.ndarray, target: np.ndarray, compounds: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, float]:
    folds = compound_folds(compounds)
    output = np.empty((len(predictions), predictions.shape[2]), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        evaluation = folds == fold
        calibration = ~evaluation
        calibration_compounds = len(set(np.asarray(compounds, dtype=str)[calibration].tolist()))
        raw_weight = simplex_mse_weights(predictions[calibration], target[calibration])
        weight, gamma = shrunken_weight(raw_weight, calibration_compounds)
        output[evaluation] = np.einsum("nkd,k->nd", predictions[evaluation], weight, optimize=True).astype(np.float32)
        rows.append({
            "fold": fold,
            "n_calibration_examples": int(calibration.sum()),
            "n_calibration_compounds": calibration_compounds,
            "n_evaluation_examples": int(evaluation.sum()),
            "n_evaluation_compounds": int(len(set(np.asarray(compounds, dtype=str)[evaluation].tolist())),),
            "gamma_uniform_prior": gamma,
            **{f"raw_weight_{name}": float(raw_weight[index]) for index, name in enumerate(CANDIDATES)},
            **{f"weight_{name}": float(weight[index]) for index, name in enumerate(CANDIDATES)},
        })
    full_raw = simplex_mse_weights(predictions, target)
    full_weight, full_gamma = shrunken_weight(full_raw, len(set(np.asarray(compounds, dtype=str).tolist())))
    return output, rows, full_weight, full_gamma


def contrast_from_metrics(left_name: str, left_metric: np.ndarray, right_name: str, right_metric: np.ndarray, compounds: np.ndarray, bootstrap_n: int, label: str) -> dict[str, Any]:
    point, low, high, n = bootstrap_difference(
        left_metric,
        right_metric,
        compounds,
        stable_int(PAIR_SEED, f"validation-calibrated-ensemble|{label}|{left_name}|{right_name}"),
        bootstrap_n,
    )
    return {
        "comparison": f"{left_name}_minus_{right_name}",
        "metric": "excess_z",
        "mean_difference": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_compounds": int(n),
    }


def metric_summary(name: str, prediction: np.ndarray, examples: Any, bootstrap_n: int) -> dict[str, Any]:
    row, _ = summarize_metrics(name, prediction, examples, PAIR_SEED, "valid", bootstrap_n=bootstrap_n)
    return row


def compound_mean(values: np.ndarray, compounds: np.ndarray) -> np.ndarray:
    grouped: dict[str, list[float]] = {}
    for compound, value in zip(np.asarray(compounds, dtype=str), np.asarray(values, dtype=float)):
        if np.isfinite(value):
            grouped.setdefault(str(compound), []).append(float(value))
    return np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=float)


def macro_bootstrap(deltas: Sequence[np.ndarray], bootstrap_n: int) -> dict[str, float]:
    if not deltas or any(len(values) == 0 for values in deltas):
        raise RuntimeError("empty condition in macro bootstrap")
    point = float(np.mean([np.mean(values) for values in deltas]))
    rng = np.random.default_rng(stable_int(PAIR_SEED, "validation-calibrated-ensemble|macro-bootstrap") % (2**63 - 1))
    draws = np.zeros(bootstrap_n, dtype=float)
    for values in deltas:
        choices = rng.integers(0, len(values), size=(bootstrap_n, len(values)))
        draws += values[choices].mean(axis=1) / len(deltas)
    return {"mean_difference": point, "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)), "bootstrap_n": int(bootstrap_n)}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite ensemble screen output: {args.outdir}")
    if args.bootstrap_n < 100:
        raise ValueError("bootstrap-n must be at least 100")

    records: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    macro_deltas: list[np.ndarray] = []
    output_payload: dict[tuple[str, int], tuple[dict[str, Any], Mapping[str, np.ndarray], np.ndarray, list[dict[str, Any]], np.ndarray, float]] = {}
    for dataset, budget in CONDITIONS:
        metadata, predictions, values = load_condition(args.strict_root, dataset, budget)
        ensemble_oof, weight_rows, full_weight, full_gamma = crossfit_ensemble(predictions, values["target"], values["compound"])
        examples = example_proxy(values, dataset, budget)
        candidate_predictions = {
            "historical_teacher_traininternal": values["historical_teacher_traininternal"],
            "lso_traininternal": values["lso_traininternal"],
            "rceb": values["rceb"],
            "validation_calibrated_ensemble_oof": ensemble_oof,
        }
        metrics = {name: metric_arrays(prediction, examples)["excess_z"] for name, prediction in candidate_predictions.items()}
        candidate_mean = {name: float(np.nanmean(metrics[name])) for name in CANDIDATES}
        best_single = sorted(CANDIDATES, key=lambda name: (-candidate_mean[name], name))[0]
        contrast = contrast_from_metrics(
            "validation_calibrated_ensemble_oof",
            metrics["validation_calibrated_ensemble_oof"],
            best_single,
            metrics[best_single],
            values["compound"],
            args.bootstrap_n,
            f"{dataset}|b{budget}",
        )
        nonlower = bool(contrast["mean_difference"] >= 0.0)
        condition_rows.append({
            "dataset": dataset,
            "budget": budget,
            "best_single_posthoc": best_single,
            "best_single_excess_z_mean": candidate_mean[best_single],
            "ensemble_oof_excess_z_mean": float(np.nanmean(metrics["validation_calibrated_ensemble_oof"])),
            "ensemble_minus_best_single": contrast["mean_difference"],
            "ci_low": contrast["ci_low"],
            "ci_high": contrast["ci_high"],
            "n_compounds": contrast["n_compounds"],
            "nonlower_point_estimate": nonlower,
            **{f"full_weight_{name}": float(full_weight[index]) for index, name in enumerate(CANDIDATES)},
            "full_weight_gamma_uniform_prior": full_gamma,
        })
        macro_deltas.append(compound_mean(metrics["validation_calibrated_ensemble_oof"] - metrics[best_single], values["compound"]))
        for name, prediction in candidate_predictions.items():
            records.append(metric_summary(name, prediction, examples, args.bootstrap_n) | {"dataset": dataset, "budget": budget})
        for row in weight_rows:
            row.update({"dataset": dataset, "budget": budget})
        output_payload[(dataset, budget)] = (metadata, values, ensemble_oof, weight_rows, full_weight, full_gamma)

    macro = macro_bootstrap(macro_deltas, args.bootstrap_n)
    nonlower_count = int(sum(bool(row["nonlower_point_estimate"]) for row in condition_rows))
    passed = bool(nonlower_count >= 3 and macro["mean_difference"] > 0.0)

    args.outdir.mkdir(parents=True)
    write_csv(args.outdir / "VALIDATION_SUMMARY.csv", records)
    write_csv(args.outdir / "CONDITION_CONTRASTS.csv", condition_rows)
    all_weights = [row for (_, _), (_, _, _, rows, _, _) in output_payload.items() for row in rows]
    write_csv(args.outdir / "OOF_FOLD_WEIGHTS.csv", all_weights)
    deployment_rows = []
    for (dataset, budget), (metadata, _, _, _, full_weight, full_gamma) in output_payload.items():
        deployment_rows.append({
            "dataset": dataset,
            "budget": budget,
            "n_validation_compounds": metadata["n_compounds"],
            "gamma_uniform_prior": full_gamma,
            **{f"weight_{name}": float(full_weight[index]) for index, name in enumerate(CANDIDATES)},
            "status": "saved_for_a_future frozen deployment only; not applied to test in this screen",
        })
    write_csv(args.outdir / "FULL_VALIDATION_DEPLOYMENT_WEIGHTS.csv", deployment_rows)
    for (dataset, budget), (metadata, values, ensemble_oof, _, _, _) in output_payload.items():
        condition_dir = args.outdir / dataset / "CP" / f"budget{budget}"
        condition_dir.mkdir(parents=True)
        write_csv(condition_dir / "valid_pair_manifest.csv", example_manifest(example_proxy(values, dataset, budget)))
        np.savez_compressed(
            condition_dir / "valid_predictions.npz",
            compound=values["compound"], dose=values["dose"], support=values["support"], target=values["target"], foreign=values["foreign"], foreign_ok=values["foreign_ok"].astype(np.uint8),
            historical_teacher_traininternal=values["historical_teacher_traininternal"], lso_traininternal=values["lso_traininternal"], rceb=values["rceb"], validation_calibrated_ensemble_oof=ensemble_oof,
        )
        write_json(condition_dir / "ARTIFACT_AUDIT.json", metadata)
    decision = {
        "version": VERSION,
        "scope": "four-condition validation-only cross-fitted convex ensemble screen",
        "candidates": list(CANDIDATES),
        "candidate_construction": "loaded frozen Stage-B/C validation predictions; no candidate retraining or modification",
        "calibration": {
            "objective": "non-negative sum-to-one least squares on independent held-repeat profile reconstruction",
            "cross_fit": f"compound-level deterministic {FOLD_COUNT}-fold; each out-of-fold prediction uses weights fit without that compound",
            "uniform_shrinkage": "Dirichlet(1,1,1) pseudo-compound prior: gamma=n_calibration_compounds/(n_calibration_compounds+3)",
            "validation_tuned_hyperparameters": False,
        },
        "test_loaded": False,
        "test_predictions_loaded": False,
        "candidate_provenance_caveat": "This screen opens only saved validation predictions. Their Stage-C decisions declare test_loaded=false (no test scoring/use); because the historical cpg preparation NPZ physically co-locates splits, this is not a claim of byte-level test isolation during the earlier candidate fit.",
        "conditions": condition_rows,
        "macro_ensemble_minus_best_single": macro,
        "frozen_gate": {
            "condition_rule": "at least 3 of 4 cross-fitted ensemble minus post-hoc best single candidate point estimates must be >= 0",
            "macro_rule": "unweighted four-condition macro cross-fitted ensemble minus post-hoc best single candidate point estimate must be > 0",
        },
        "observed": {"nonlower_condition_count": nonlower_count, "macro_point_positive": bool(macro["mean_difference"] > 0.0)},
        "passed": passed,
        "decision": "VALIDATION-GO" if passed else "STOP-UNIFIED-ESTIMATOR",
        "guard": "This validation screen cannot turn any previously inspected test into confirmatory evidence. If it passes, confirmation requires a new CP resource or unused outer split.",
    }
    write_json(args.outdir / "ENSEMBLE_DECISION.json", decision)
    (args.outdir / "README.md").write_text(
        "# Cross-fitted validation-calibrated ensemble\n\n"
        "This is a no-training, validation-only screen. `validation_calibrated_ensemble_oof` is the only outcome used for the gate; full-validation weights are saved but are not applied to any test profile.\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision["decision"], "nonlower_condition_count": nonlower_count, "macro": macro, "outdir": str(args.outdir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
