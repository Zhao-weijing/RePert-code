#!/usr/bin/env python3
"""Frozen Phase-1 ReproPoE prototype screen: BBBC/cpg CP 1R validation only.

ReproPoE is deliberately the minimal product-of-evidence model specified in
the protocol: each repeat is encoded independently into a 32-dimensional
Gaussian evidence factor; diagonal precisions add analytically with one global
learned diagonal prior; only the fused latent mean is decoded.  There are no
condition IDs, attention, gates, candidate-estimator inputs, foreign losses,
architecture searches, or validation-selected hyperparameters.

Exactly one fixed-seed checkpoint is fit per dataset.  Each checkpoint trains
on randomly constructed 1R/2R/3R legal support sets from training compounds,
but this Phase-1 screen evaluates only 1R validation.  Other baselines are
loaded frozen from the prior strict validation artifacts; no baseline is
retrained.  The script never opens a test data file or test prediction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASELINE_DIR = ROOT / "baselines"
RCEB_DIR = ROOT / "repeat_calibration"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))
if str(RCEB_DIR) not in sys.path:
    sys.path.insert(0, str(RCEB_DIR))

from ablation_common import (  # noqa: E402
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
from run_rceb_screen import DEFAULT_BBBC_ROOT, PAIR_SEED  # noqa: E402


VERSION = "repropoe-phase1-minimal-v1-2026-08-31"
SEED = 3407
LATENT_DIM = 32
ENCODER_HIDDEN = 256
ENCODER_BOTTLENECK = 64
DECODER_HIDDEN = 256
EPOCHS = 60
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
LOG_VARIANCE_REGULARIZATION = 1e-3
VARIANCE_FLOOR = 1e-4
STRICT_ROOT = Path("/path/to/data/AIDD/MVCPert_5_27/runs/rceb_repeat_calibrated_20260831_strict")
CPG_SPLIT_ROOT = Path("/path/to/data/AIDD/MVCPert_5_27/runs/cpg0004_lincs_cp_train_valid_only_20260831")
DEFAULT_OUTPUT = Path("/path/to/data/AIDD/MVCPert_5_27/runs/repropoe_phase1_20260831")
DATASETS = ("BBBC047", "cpg0004-LINCS")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strict-root", type=Path, default=STRICT_ROOT)
    parser.add_argument("--bbbc-root", type=Path, default=DEFAULT_BBBC_ROOT)
    parser.add_argument("--cpg-train", type=Path, default=CPG_SPLIT_ROOT / "train_cp_plate_rows.npz")
    parser.add_argument("--cpg-valid", type=Path, default=CPG_SPLIT_ROOT / "valid_cp_plate_rows.npz")
    parser.add_argument("--bootstrap-n", type=int, default=10_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def torch_modules() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ReproPoE requires PyTorch") from exc
    return torch, nn, functional


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(torch: Any, requested: str) -> Any:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


def assert_identical(name: str, left: np.ndarray, right: np.ndarray) -> None:
    left, right = np.asarray(left), np.asarray(right)
    equal = np.array_equal(left, right, equal_nan=True) if np.issubdtype(left.dtype, np.inexact) else np.array_equal(left, right)
    if left.shape != right.shape or not equal:
        raise RuntimeError(f"canonical validation bundle mismatch: {name}")


def load_cpg_split(path: Path, expected_split: str) -> Rows:
    data = np.load(path, allow_pickle=False)
    required = ("compound_id", "dose", "plate", "delta", "split")
    missing = [field for field in required if field not in data.files]
    if missing:
        raise RuntimeError(f"missing split-only cpg fields in {path}: {missing}")
    labels = np.asarray(data["split"], dtype=str)
    if len(labels) == 0 or not np.all(labels == expected_split):
        raise RuntimeError(f"cpg path is not physically restricted to {expected_split}: {path}")
    return Rows("cpg0004-LINCS", "CP", expected_split, np.asarray(data["compound_id"], dtype=str), np.asarray(data["dose"], dtype=str), np.asarray(data["plate"], dtype=str), np.asarray(data["delta"], dtype=np.float32))


def load_rows(args: argparse.Namespace, dataset: str, split: str) -> Rows:
    if dataset == "BBBC047":
        path = args.bbbc_root / f"{split}_cp_plate_rows.npz"
        data = np.load(path, allow_pickle=False)
        return Rows(dataset, "CP", split, np.asarray(data["smiles"], dtype=str), np.asarray(data["dose"], dtype=str), np.asarray(data["plate"], dtype=str), np.asarray(data["delta"], dtype=np.float32))
    if dataset == "cpg0004-LINCS":
        return load_cpg_split(args.cpg_train if split == "train" else args.cpg_valid, split)
    raise ValueError(dataset)


@dataclass(frozen=True)
class ConditionRecord:
    compound: str
    dose: str
    plates: np.ndarray


def condition_records(rows: Rows) -> list[ConditionRecord]:
    records: list[ConditionRecord] = []
    for (compound, dose), by_plate in rows.plate_means().items():
        values = np.stack([by_plate[plate] for plate in sorted(by_plate)], axis=0).astype(np.float32, copy=False)
        if len(values) >= 2:
            records.append(ConditionRecord(str(compound), str(dose), values))
    if not records:
        raise RuntimeError("ReproPoE requires training conditions with at least two independent plates")
    return sorted(records, key=lambda record: (record.compound, record.dose))


def epoch_examples(records: Sequence[ConditionRecord], epoch: int) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, int]]:
    """One random legal support/held construction per training condition.

    Randomness is deterministic from seed+epoch.  A condition can contribute
    1R always, 2R with >=3 plates, and 3R with >=4 plates.  Held plate is never
    part of its support set.
    """
    rng = np.random.default_rng(stable_int(SEED, f"repropoe|epoch|{epoch}") % (2**63 - 1))
    supports: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    targets: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    counts = {1: 0, 2: 0, 3: 0}
    for record in records:
        legal = [budget for budget in (1, 2, 3) if len(record.plates) >= budget + 1]
        budget = legal[int(rng.integers(0, len(legal)))]
        order = rng.permutation(len(record.plates))
        supports[budget].append(record.plates[order[:budget]])
        targets[budget].append(record.plates[int(order[budget])])
        counts[budget] += 1
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for budget in (1, 2, 3):
        if supports[budget]:
            output[budget] = (np.stack(supports[budget]), np.stack(targets[budget]))
    return output, counts


def make_model(nn: Any, functional: Any, dimension: int) -> Any:
    class ReproPoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(dimension, ENCODER_HIDDEN), nn.GELU(),
                nn.Linear(ENCODER_HIDDEN, ENCODER_BOTTLENECK), nn.GELU(),
            )
            self.evidence_mean = nn.Linear(ENCODER_BOTTLENECK, LATENT_DIM)
            self.evidence_log_variance = nn.Linear(ENCODER_BOTTLENECK, LATENT_DIM)
            self.prior_mean = nn.Parameter(np_to_tensor(np.zeros(LATENT_DIM, dtype=np.float32)))
            self.prior_log_variance = nn.Parameter(np_to_tensor(np.zeros(LATENT_DIM, dtype=np.float32)))
            self.decoder_hidden = nn.Sequential(nn.Linear(LATENT_DIM, DECODER_HIDDEN), nn.GELU())
            self.decoder_mean = nn.Linear(DECODER_HIDDEN, dimension)
            self.decoder_log_observation_variance = nn.Linear(DECODER_HIDDEN, 1)
            # softplus(0.5413) is approximately one: neutral initial scalar variance.
            nn.init.constant_(self.decoder_log_observation_variance.bias, 0.54132485)

        def forward(self, support: Any) -> tuple[Any, Any, Any, Any]:
            if support.ndim != 3:
                raise ValueError(f"support must have [batch,repeats,features], got {tuple(support.shape)}")
            batch, repeats, _ = support.shape
            encoded = self.encoder(support.reshape(batch * repeats, -1))
            evidence_mean = self.evidence_mean(encoded).reshape(batch, repeats, LATENT_DIM)
            evidence_log_variance = self.evidence_log_variance(encoded).reshape(batch, repeats, LATENT_DIM)
            evidence_variance = functional.softplus(evidence_log_variance) + VARIANCE_FLOOR
            evidence_precision = evidence_variance.reciprocal()
            prior_variance = functional.softplus(self.prior_log_variance) + VARIANCE_FLOOR
            prior_precision = prior_variance.reciprocal()
            fused_precision = prior_precision.unsqueeze(0) + evidence_precision.sum(dim=1)
            fused_mean = (prior_precision.unsqueeze(0) * self.prior_mean.unsqueeze(0) + (evidence_precision * evidence_mean).sum(dim=1)) / fused_precision
            hidden = self.decoder_hidden(fused_mean)
            predicted_mean = self.decoder_mean(hidden)
            observation_variance = functional.softplus(self.decoder_log_observation_variance(hidden)).squeeze(-1) + VARIANCE_FLOOR
            return predicted_mean, observation_variance, evidence_log_variance, fused_precision

    def np_to_tensor(values: np.ndarray) -> Any:
        # Captured torch is intentionally avoided by returning a plain Tensor
        # through nn.Parameter's accepted array conversion below.
        import torch
        return torch.from_numpy(values)

    return ReproPoE()


def train_model(torch: Any, model: Any, records: Sequence[ConditionRecord], device: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history: list[dict[str, Any]] = []
    total_budgets = {1: 0, 2: 0, 3: 0}
    for epoch in range(EPOCHS):
        model.train()
        groups, counts = epoch_examples(records, epoch)
        for budget in total_budgets:
            total_budgets[budget] += counts[budget]
        epoch_nll: list[float] = []
        epoch_reg: list[float] = []
        for budget in sorted(groups):
            supports, targets = groups[budget]
            order = np.random.default_rng(stable_int(SEED, f"repropoe|epoch|{epoch}|budget|{budget}") % (2**63 - 1)).permutation(len(supports))
            for begin in range(0, len(order), BATCH_SIZE):
                index = order[begin : begin + BATCH_SIZE]
                support = torch.from_numpy(supports[index]).to(device)
                target = torch.from_numpy(targets[index]).to(device)
                optimizer.zero_grad(set_to_none=True)
                predicted, observation_variance, evidence_log_variance, _ = model(support)
                residual_squared = torch.mean((target - predicted) ** 2, dim=1)
                predictive_nll = 0.5 * torch.mean(residual_squared / observation_variance + torch.log(observation_variance))
                precision_regularizer = torch.mean(evidence_log_variance**2) + torch.mean(model.prior_log_variance**2)
                loss = predictive_nll + LOG_VARIANCE_REGULARIZATION * precision_regularizer
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite ReproPoE loss at epoch {epoch}, budget {budget}")
                loss.backward()
                optimizer.step()
                epoch_nll.append(float(predictive_nll.detach().cpu()))
                epoch_reg.append(float(precision_regularizer.detach().cpu()))
        history.append({
            "epoch_one_based": epoch + 1,
            "predictive_nll": float(np.mean(epoch_nll)),
            "precision_regularizer": float(np.mean(epoch_reg)),
            **{f"sampled_budget{budget}": counts[budget] for budget in (1, 2, 3)},
        })
    return history, total_budgets


def predict(torch: Any, model: Any, support: np.ndarray, device: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predicted = np.empty((len(support), support.shape[-1]), dtype=np.float32)
    observation_variance = np.empty(len(support), dtype=np.float32)
    fused_precision = np.empty((len(support), LATENT_DIM), dtype=np.float32)
    with torch.no_grad():
        for begin in range(0, len(support), BATCH_SIZE):
            values = torch.from_numpy(np.asarray(support[begin : begin + BATCH_SIZE], dtype=np.float32)).to(device)
            mean, variance, _, precision = model(values)
            length = len(values)
            predicted[begin : begin + length] = mean.detach().cpu().numpy()
            observation_variance[begin : begin + length] = variance.detach().cpu().numpy()
            fused_precision[begin : begin + length] = precision.detach().cpu().numpy()
    return predicted, observation_variance, fused_precision


def load_baselines(args: argparse.Namespace, dataset: str, valid: Any) -> dict[str, np.ndarray]:
    stage_b_dir = args.strict_root / "stage_b" / dataset / "CP" / "budget1"
    stage_c_dir = args.strict_root / "stage_c" / dataset / "CP" / "budget1"
    ensemble_dir = args.strict_root / "validation_calibrated_ensemble_20260831" / dataset / "CP" / "budget1"
    stage_b = np.load(stage_b_dir / "valid_predictions.npz", allow_pickle=False)
    stage_c = np.load(stage_c_dir / "valid_predictions.npz", allow_pickle=False)
    ensemble = np.load(ensemble_dir / "valid_predictions.npz", allow_pickle=False)
    fields = ("compound", "dose", "support", "target", "foreign", "foreign_ok")
    for source_name, source in (("stage_b", stage_b), ("stage_c", stage_c), ("ensemble", ensemble)):
        for field in fields:
            assert_identical(f"{dataset}|{source_name}|{field}", np.asarray(getattr(valid, field)), np.asarray(source[field]))
    decision = json.loads((stage_c_dir / "decision.json").read_text(encoding="utf-8"))
    if decision.get("test_loaded") is not False:
        raise RuntimeError(f"Stage-C baseline has no test_loaded=false audit: {dataset}")
    return {
        "raw_mean": np.asarray(stage_b["raw_mean"], dtype=np.float32),
        "pca": np.asarray(stage_b["pca"], dtype=np.float32),
        "historical_teacher_traininternal": np.asarray(stage_c["historical_teacher_traininternal"], dtype=np.float32),
        "lso_traininternal": np.asarray(stage_c["lso_traininternal"], dtype=np.float32),
        "rceb": np.asarray(stage_b["rceb"], dtype=np.float32),
        "validation_calibrated_ensemble_oof": np.asarray(ensemble["validation_calibrated_ensemble_oof"], dtype=np.float32),
    }


def contrast(name: str, prediction: np.ndarray, reference_name: str, reference: np.ndarray, valid: Any, bootstrap_n: int) -> dict[str, Any]:
    left = metric_arrays(prediction, valid)["excess_z"]
    right = metric_arrays(reference, valid)["excess_z"]
    point, low, high, n = bootstrap_difference(left, right, valid.compound, stable_int(SEED, f"repropoe|{name}|{reference_name}"), bootstrap_n)
    return {"comparison": f"{name}_minus_{reference_name}", "metric": "excess_z", "mean_difference": float(point), "ci_low": float(low), "ci_high": float(high), "n_compounds": int(n)}


def macro_bootstrap(deltas: Sequence[np.ndarray], compound_ids: Sequence[np.ndarray], bootstrap_n: int) -> dict[str, float]:
    if len(deltas) != 2:
        raise ValueError("Phase-1 macro needs exactly two datasets")
    means: list[np.ndarray] = []
    for values, compounds in zip(deltas, compound_ids):
        group: dict[str, list[float]] = {}
        for compound, value in zip(np.asarray(compounds, dtype=str), np.asarray(values, dtype=float)):
            if np.isfinite(value):
                group.setdefault(str(compound), []).append(float(value))
        means.append(np.asarray([np.mean(group[key]) for key in sorted(group)], dtype=float))
    rng = np.random.default_rng(stable_int(SEED, "repropoe|phase1|macro") % (2**63 - 1))
    point = float(np.mean([np.mean(values) for values in means]))
    draws = np.zeros(bootstrap_n, dtype=float)
    for values in means:
        choices = rng.integers(0, len(values), size=(bootstrap_n, len(values)))
        draws += values[choices].mean(axis=1) / len(means)
    return {"mean_difference": point, "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)), "bootstrap_n": int(bootstrap_n)}


def run_dataset(args: argparse.Namespace, torch: Any, nn: Any, functional: Any, device: Any, dataset: str) -> dict[str, Any]:
    train_rows = load_rows(args, dataset, "train")
    valid_rows = load_rows(args, dataset, "valid")
    valid = make_examples(valid_rows, None, 1, PAIR_SEED, all_conditions=False, require_source=False)
    records = condition_records(train_rows)
    set_seed(torch, SEED)
    model = make_model(nn, functional, train_rows.dim)
    history, budget_counts = train_model(torch, model, records, device)
    repro_prediction, observation_variance, fused_precision = predict(torch, model, valid.support[:, None, :], device)
    baselines = load_baselines(args, dataset, valid)
    candidates = ("historical_teacher_traininternal", "lso_traininternal", "rceb")
    candidate_scores = {name: float(np.nanmean(metric_arrays(baselines[name], valid)["excess_z"])) for name in candidates}
    best_single = sorted(candidates, key=lambda name: (-candidate_scores[name], name))[0]
    methods = {**baselines, "repropoe": repro_prediction}
    summary_rows = []
    for name, values in methods.items():
        row, _ = summarize_metrics(name, values, valid, SEED, "valid", bootstrap_n=args.bootstrap_n)
        summary_rows.append(row)
    comparisons = [
        contrast("repropoe", repro_prediction, best_single, baselines[best_single], valid, args.bootstrap_n),
        contrast("repropoe", repro_prediction, "validation_calibrated_ensemble_oof", baselines["validation_calibrated_ensemble_oof"], valid, args.bootstrap_n),
        contrast("repropoe", repro_prediction, "raw_mean", baselines["raw_mean"], valid, args.bootstrap_n),
        contrast("repropoe", repro_prediction, "pca", baselines["pca"], valid, args.bootstrap_n),
    ]
    values = {row["comparison"]: row for row in comparisons}
    output = args.outdir / dataset / "CP" / "budget1"
    output.mkdir(parents=True)
    torch.save({
        "state_dict": model.state_dict(),
        "architecture": {"input_dim": train_rows.dim, "encoder": [ENCODER_HIDDEN, ENCODER_BOTTLENECK], "latent_dim": LATENT_DIM, "decoder_hidden": DECODER_HIDDEN},
        "training": {"epochs": EPOCHS, "seed": SEED, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "log_variance_regularization": LOG_VARIANCE_REGULARIZATION, "variance_floor": VARIANCE_FLOOR},
    }, output / "repropoe_seed3407.pt")
    write_csv(output / "training_history.csv", history)
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "contrasts.csv", comparisons)
    write_csv(output / "valid_pair_manifest.csv", example_manifest(valid))
    np.savez_compressed(
        output / "valid_predictions.npz",
        compound=valid.compound, dose=valid.dose, support=valid.support, target=valid.target, foreign=valid.foreign, foreign_ok=valid.foreign_ok.astype(np.uint8),
        repropoe=repro_prediction, observation_variance=observation_variance, fused_precision=fused_precision,
    )
    audit = {
        "version": VERSION,
        "dataset": dataset,
        "modality": "CP",
        "budget_evaluated": 1,
        "seed": SEED,
        "fit_split": "train only",
        "evaluation_split": "valid only",
        "test_loaded": False,
        "architecture": {"input": f"d={train_rows.dim}", "shared_encoder": f"{train_rows.dim}->256->64 GELU", "evidence_heads": "64->32 mean and 64->32 log-variance", "fusion": "global diagonal learnable Gaussian prior plus analytic precision addition", "decoder": f"32->256->d={train_rows.dim} GELU; scalar observation variance"},
        "prohibited_inputs": ["dataset ID", "dose ID", "repeat-budget label", "plate ID", "Teacher/LSO/RCEB results", "molecular structure", "GE"],
        "prohibited_components": ["attention", "gate", "residual branch", "Set Transformer", "VAE sampling", "feature-wise reliability", "foreign-negative loss"],
        "training_support_budgets": "random legal 1R/2R/3R per condition/epoch; held repeat excluded from support",
        "training_sample_counts_across_epochs": budget_counts,
        "n_train_conditions_with_at_least_two_plates": len(records),
        "n_validation_examples": int(len(valid.compound)),
        "baseline_sources": {
            "stage_b": str(args.strict_root / "stage_b" / dataset / "CP" / "budget1" / "valid_predictions.npz"),
            "stage_c": str(args.strict_root / "stage_c" / dataset / "CP" / "budget1" / "valid_predictions.npz"),
            "ensemble": str(args.strict_root / "validation_calibrated_ensemble_20260831" / dataset / "CP" / "budget1" / "valid_predictions.npz"),
            "note": "existing frozen validation predictions only; no baseline is retrained",
        },
        "best_single_posthoc": best_single,
        "contrasts": values,
    }
    write_json(output / "AUDIT.json", audit)
    return {"dataset": dataset, "valid": valid, "repro": repro_prediction, "baselines": baselines, "best_single": best_single, "comparisons": values, "audit": audit}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if tuple(args.datasets) != DATASETS:
        raise SystemExit("Phase-1 is frozen to BBBC047 and cpg0004-LINCS together")
    if args.outdir.exists():
        raise FileExistsError(f"refusing to overwrite Phase-1 output: {args.outdir}")
    if args.bootstrap_n < 100:
        raise ValueError("bootstrap-n must be at least 100")
    torch, nn, functional = torch_modules()
    device = select_device(torch, args.device)
    args.outdir.mkdir(parents=True)
    outcomes = [run_dataset(args, torch, nn, functional, device, dataset) for dataset in DATASETS]
    by_dataset = {outcome["dataset"]: outcome for outcome in outcomes}
    bstar_deltas = [metric_arrays(outcome["repro"], outcome["valid"])["excess_z"] - metric_arrays(outcome["baselines"][outcome["best_single"]], outcome["valid"])["excess_z"] for outcome in outcomes]
    ensemble_deltas = [metric_arrays(outcome["repro"], outcome["valid"])["excess_z"] - metric_arrays(outcome["baselines"]["validation_calibrated_ensemble_oof"], outcome["valid"])["excess_z"] for outcome in outcomes]
    bstar_macro = macro_bootstrap(bstar_deltas, [outcome["valid"].compound for outcome in outcomes], args.bootstrap_n)
    ensemble_macro = macro_bootstrap(ensemble_deltas, [outcome["valid"].compound for outcome in outcomes], args.bootstrap_n)
    gates: dict[str, Any] = {}
    per_dataset: list[dict[str, Any]] = []
    for outcome in outcomes:
        dataset = outcome["dataset"]
        bstar = outcome["comparisons"][f"repropoe_minus_{outcome['best_single']}"]
        ensemble = outcome["comparisons"]["repropoe_minus_validation_calibrated_ensemble_oof"]
        raw = outcome["comparisons"]["repropoe_minus_raw_mean"]
        pca = outcome["comparisons"]["repropoe_minus_pca"]
        per_dataset.append({
            "dataset": dataset,
            "best_single": outcome["best_single"],
            "repropoe_minus_best_single": bstar["mean_difference"],
            "repropoe_minus_ensemble": ensemble["mean_difference"],
            "repropoe_minus_raw": raw["mean_difference"],
            "repropoe_minus_pca": pca["mean_difference"],
        })
        gates[f"{dataset}_no_failure_vs_best_single_gt_neg_0_01"] = bool(bstar["mean_difference"] > -0.01)
        gates[f"{dataset}_no_failure_vs_ensemble_gt_neg_0_01"] = bool(ensemble["mean_difference"] > -0.01)
        gates[f"{dataset}_point_beats_raw"] = bool(raw["mean_difference"] > 0.0)
        gates[f"{dataset}_point_beats_pca"] = bool(pca["mean_difference"] > 0.0)
    gates["macro_vs_best_single_positive"] = bool(bstar_macro["mean_difference"] > 0.0)
    gates["macro_vs_ensemble_at_least_neg_0_005"] = bool(ensemble_macro["mean_difference"] >= -0.005)
    passed = bool(all(gates.values()))
    decision = {
        "version": VERSION,
        "scope": "Phase-1 minimal pressure test: BBBC047/cpg0004 CP 1R validation, one seed 3407",
        "test_loaded": False,
        "no_baseline_retraining": True,
        "frozen_config": {"latent_dim": LATENT_DIM, "encoder": "d->256->64 GELU", "decoder": "32->256->d GELU", "epochs": EPOCHS, "seed": SEED, "support_budgets_trained": [1, 2, 3], "evaluation_budget": 1, "loss": "scalar-variance Gaussian predictive NLL + fixed 1e-3 squared log-variance regularization"},
        "phase1_requirements": {
            "A": "both datasets ReproPoE-B*_single > -0.01",
            "B": "equal-dataset macro ReproPoE-B*_single > 0",
            "C": "both datasets ReproPoE-Ensemble > -0.01 and macro >= -0.005",
            "foundation": "both datasets ReproPoE-Raw > 0 and ReproPoE-PCA > 0 point estimates",
        },
        "per_dataset": per_dataset,
        "macro_repropoe_minus_best_single": bstar_macro,
        "macro_repropoe_minus_ensemble": ensemble_macro,
        "gates": gates,
        "passed": passed,
        "decision": "PROTOTYPE-GO" if passed else "PROTOTYPE-NO-GO",
        "next_step": "Only PROTOTYPE-GO permits Phase-2 six-condition validation. PROTOTYPE-NO-GO forbids architecture/loss/latent-size sweeps and all 2R/3R runs.",
    }
    write_csv(args.outdir / "PHASE1_CONDITION_SUMMARY.csv", per_dataset)
    write_json(args.outdir / "PHASE1_DECISION.json", decision)
    (args.outdir / "README.md").write_text(
        "# ReproPoE Phase-1 prototype\n\n"
        "This is the frozen two-condition 1R validation pressure test. No test split or test prediction is read. See `PHASE1_DECISION.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision["decision"], "gates": gates, "outdir": str(args.outdir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
