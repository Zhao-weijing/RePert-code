#!/usr/bin/env python3
"""Stage-C validation-only comparison of RCEB against two CP-only MLP baselines.

The baselines use the historical Teacher architectures and the two historical
supervision targets: independent held repeat (Teacher) and leave-support-out
plate aggregate (LSO).  They are deliberately re-fit rather than loading the
old formal checkpoints, because those old checkpoints selected epochs against
their historical validation split.  Here every epoch decision is made on a
compound-disjoint, train-internal split; the selected epoch is then used for a
fresh all-training fit before the official validation split is read.

This driver never opens a test data file.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
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
    BOOTSTRAP_N,
    bootstrap_difference,
    example_manifest,
    metric_arrays,
    stable_int,
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
    rceb_predict,
)


VERSION = "rceb-stage-c-train-internal-historical-mlp-v1-2026-08-31"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=("BBBC047", "cpg0004-LINCS"), default=("BBBC047", "cpg0004-LINCS"))
    parser.add_argument("--budgets", nargs="+", type=int, choices=(1, 2, 3), default=(1, 3))
    parser.add_argument("--bbbc-root", type=Path, default=DEFAULT_BBBC_ROOT)
    parser.add_argument("--cpg-cp", type=Path, default=DEFAULT_CPG_CP)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Stage C requires PyTorch") from exc
    return torch, nn


def select_device(torch: Any, requested: str) -> Any:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inner_train_mask(compounds: np.ndarray) -> np.ndarray:
    """Fixed compound-disjoint 80/20 split, never based on CP values."""
    result = np.asarray([
        stable_int(PAIR_SEED, f"rceb-stage-c-inner|{compound}") % 5 != 0
        for compound in np.asarray(compounds, dtype=str)
    ], dtype=bool)
    if result.sum() == 0 or (~result).sum() == 0:
        raise RuntimeError("empty train-internal split")
    return result


def architecture(dataset: str, dimension: int) -> tuple[int, int]:
    if dataset == "BBBC047":
        if dimension != 775:
            raise RuntimeError(f"unexpected BBBC047 CP dimension: {dimension}")
        return 256, 32
    if dataset == "cpg0004-LINCS":
        if dimension != 242:
            raise RuntimeError(f"unexpected cpg0004 CP dimension: {dimension}")
        return 128, 32
    raise ValueError(dataset)


def lso_labels(rows: Any, examples: Any) -> tuple[np.ndarray, dict[str, Any]]:
    output: list[np.ndarray] = []
    counts: list[int] = []
    overlaps = 0
    by_plate = rows.plate_means()
    for compound, dose, support_plates in zip(examples.compound, examples.dose, examples.support_plates):
        key = (str(compound), str(dose))
        support_set = set(support_plates)
        legal = [plate for plate in by_plate[key] if plate not in support_set]
        overlap = support_set & set(legal)
        if overlap:
            raise RuntimeError(f"LSO support/label overlap for {key}: {sorted(overlap)}")
        if not legal:
            raise RuntimeError(f"no LSO label plates for {key}")
        output.append(rows.condition_mean(key, legal))
        counts.append(len(legal))
        overlaps += len(overlap)
    return np.vstack(output).astype(np.float32), {
        "definition": "mean of all train condition plate means excluding every support plate",
        "input_label_overlap_rows": overlaps,
        "label_plate_count_min": int(min(counts)),
        "label_plate_count_mean": float(np.mean(counts)),
        "label_plate_count_max": int(max(counts)),
    }


@dataclass
class FitResult:
    model: Any
    input_mean: np.ndarray
    input_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    selected_epoch: int
    selected_inner_loss: float
    final_epochs: int


def _stats(inputs: np.ndarray, direct_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_mean = inputs.mean(axis=0, dtype=np.float64).astype(np.float32)
    input_scale = np.maximum(inputs.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    target_mean = direct_target.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = np.maximum(direct_target.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    return input_mean, input_scale, target_mean, target_scale


def _make_model(nn: Any, dimension: int, hidden: int, latent: int) -> Any:
    return nn.Sequential(
        nn.Linear(dimension, hidden), nn.GELU(),
        nn.Linear(hidden, latent), nn.GELU(),
        nn.Linear(latent, hidden), nn.GELU(),
        nn.Linear(hidden, dimension),
    )


def _fit_epochs(
    torch: Any,
    nn: Any,
    inputs: np.ndarray,
    labels: np.ndarray,
    direct_target: np.ndarray,
    train_mask: np.ndarray,
    epoch_count: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: Any,
    select_epoch: bool,
) -> FitResult:
    if epoch_count < 1:
        raise ValueError("epoch_count must be positive")
    stats_mask = train_mask if select_epoch else np.ones(len(inputs), dtype=bool)
    input_mean, input_scale, target_mean, target_scale = _stats(inputs[stats_mask], direct_target[stats_mask])
    x = ((inputs - input_mean) / input_scale).astype(np.float32)
    y = ((labels - target_mean) / target_scale).astype(np.float32)
    direct_y = ((direct_target - target_mean) / target_scale).astype(np.float32)
    hidden, latent = architecture("BBBC047" if inputs.shape[1] == 775 else "cpg0004-LINCS", inputs.shape[1])
    set_seed(torch, seed)
    model = _make_model(nn, inputs.shape[1], hidden, latent).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x[train_mask]), torch.from_numpy(y[train_mask])),
        batch_size=batch_size, shuffle=True, generator=generator, num_workers=0,
    )
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    best_epoch = -1
    validation_mask = ~train_mask
    for epoch in range(epoch_count):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(batch_x.to(device)) - batch_y.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Stage-C loss")
            loss.backward()
            optimizer.step()
        if select_epoch:
            model.eval()
            with torch.no_grad():
                prediction = model(torch.from_numpy(x[validation_mask]).to(device))
                loss = float(torch.mean((prediction - torch.from_numpy(direct_y[validation_mask]).to(device)) ** 2).detach().cpu())
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if select_epoch:
        if best_state is None:
            raise RuntimeError("no train-internal checkpoint selected")
        model.load_state_dict(best_state)
        return FitResult(model, input_mean, input_scale, target_mean, target_scale, best_epoch, best_loss, best_epoch + 1)
    return FitResult(model, input_mean, input_scale, target_mean, target_scale, epoch_count - 1, float("nan"), epoch_count)


def fit_baseline(
    torch: Any,
    nn: Any,
    inputs: np.ndarray,
    label: np.ndarray,
    direct_target: np.ndarray,
    compounds: np.ndarray,
    *,
    name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: Any,
) -> tuple[FitResult, dict[str, Any]]:
    train_mask = inner_train_mask(compounds)
    seed = stable_int(PAIR_SEED, f"stage-c-model|{name}") % (2**32 - 1)
    selected = _fit_epochs(torch, nn, inputs, label, direct_target, train_mask, epochs, batch_size, learning_rate, weight_decay, seed, device, True)
    final = _fit_epochs(torch, nn, inputs, label, direct_target, np.ones(len(inputs), dtype=bool), selected.final_epochs, batch_size, learning_rate, weight_decay, seed, device, False)
    meta = {
        "model": name,
        "architecture": f"{inputs.shape[1]}->{architecture('BBBC047' if inputs.shape[1] == 775 else 'cpg0004-LINCS', inputs.shape[1])[0]}->32->{architecture('BBBC047' if inputs.shape[1] == 775 else 'cpg0004-LINCS', inputs.shape[1])[0]}->{inputs.shape[1]} GELU",
        "optimizer": "AdamW(lr=3e-4, weight_decay=1e-5)",
        "selection": "compound-disjoint train-internal direct-held-repeat MSE",
        "inner_train_conditions": int(train_mask.sum()),
        "inner_validation_conditions": int((~train_mask).sum()),
        "selected_epoch_zero_based": selected.selected_epoch,
        "selected_inner_loss": selected.selected_inner_loss,
        "final_epochs": selected.final_epochs,
        "official_validation_used_for_selection": False,
    }
    return final, meta


def predict(torch: Any, fit: FitResult, inputs: np.ndarray, batch_size: int, device: Any) -> np.ndarray:
    normalized = ((inputs - fit.input_mean) / fit.input_scale).astype(np.float32)
    output = np.empty_like(inputs, dtype=np.float32)
    fit.model.eval()
    with torch.no_grad():
        for begin in range(0, len(inputs), batch_size):
            values = fit.model(torch.from_numpy(normalized[begin : begin + batch_size]).to(device)).detach().cpu().numpy()
            output[begin : begin + len(values)] = values * fit.target_scale + fit.target_mean
    return output


def contrast(method: np.ndarray, rceb: np.ndarray, examples: Any, method_name: str, bootstrap_n: int) -> dict[str, Any]:
    left = metric_arrays(rceb, examples)
    right = metric_arrays(method, examples)
    point, low, high, n = bootstrap_difference(left["excess_z"], right["excess_z"], examples.compound, stable_int(PAIR_SEED, f"rceb-minus-{method_name}"), bootstrap_n)
    return {"comparison": f"rceb_minus_{method_name}", "metric": "excess_z", "mean_difference": point, "ci_low": low, "ci_high": high, "n_compounds": n}


def run_condition(args: argparse.Namespace, torch: Any, nn: Any, device: Any, dataset: str, budget: int) -> dict[str, Any]:
    stage_a = args.output_root / "stage_a" / dataset / "CP"
    stage_b = args.output_root / "stage_b" / dataset / "CP" / f"budget{budget}"
    if not (stage_a / "rceb_model.npz").exists() or not (stage_b / "decision.json").exists():
        raise FileNotFoundError(f"Stage A/B artifacts required for {dataset} {budget}R")
    train_rows = load_cp_rows(dataset, "train", args.bbbc_root, args.cpg_cp)
    valid_rows = load_cp_rows(dataset, "valid", args.bbbc_root, args.cpg_cp)
    train = make_examples(train_rows, None, budget, PAIR_SEED, all_conditions=True, require_source=False)
    valid = make_examples(valid_rows, None, budget, PAIR_SEED, all_conditions=False, require_source=False)
    lso, lso_meta = lso_labels(train_rows, train)
    teacher_fit, teacher_meta = fit_baseline(
        torch, nn, train.support, train.target, train.target, train.compound, name=f"teacher|{dataset}|b{budget}",
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, device=device,
    )
    lso_fit, lso_fit_meta = fit_baseline(
        torch, nn, train.support, lso, train.target, train.compound, name=f"lso|{dataset}|b{budget}",
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, device=device,
    )
    teacher = predict(torch, teacher_fit, valid.support, args.batch_size, device)
    lso_prediction = predict(torch, lso_fit, valid.support, args.batch_size, device)
    rceb = rceb_predict(load_model(stage_a / "rceb_model.npz"), valid.support, budget)
    rows = []
    summaries: dict[str, dict[str, Any]] = {}
    for name, values in (("rceb", rceb), ("historical_teacher_traininternal", teacher), ("lso_traininternal", lso_prediction)):
        row, _ = summarize_metrics(name, values, valid, PAIR_SEED, "valid", bootstrap_n=args.bootstrap_n)
        rows.append(row)
        summaries[name] = row
    winner = max(("historical_teacher_traininternal", "lso_traininternal"), key=lambda name: float(summaries[name]["excess_z_mean"]))
    best = teacher if winner == "historical_teacher_traininternal" else lso_prediction
    contrasts = [contrast(teacher, rceb, valid, "historical_teacher_traininternal", args.bootstrap_n), contrast(lso_prediction, rceb, valid, "lso_traininternal", args.bootstrap_n), contrast(best, rceb, valid, "best_strong_baseline", args.bootstrap_n)]
    best_contrast = contrasts[-1]
    outdir = args.output_root / "stage_c" / dataset / "CP" / f"budget{budget}"
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": teacher_fit.model.state_dict(), "input_mean": teacher_fit.input_mean, "input_scale": teacher_fit.input_scale, "target_mean": teacher_fit.target_mean, "target_scale": teacher_fit.target_scale, "meta": teacher_meta}, outdir / "historical_teacher_traininternal.pt")
    torch.save({"state_dict": lso_fit.model.state_dict(), "input_mean": lso_fit.input_mean, "input_scale": lso_fit.input_scale, "target_mean": lso_fit.target_mean, "target_scale": lso_fit.target_scale, "meta": lso_fit_meta}, outdir / "lso_traininternal.pt")
    write_csv(outdir / "summary.csv", rows)
    write_csv(outdir / "contrasts.csv", contrasts)
    write_csv(outdir / "valid_pair_manifest.csv", example_manifest(valid))
    np.savez_compressed(outdir / "valid_predictions.npz", compound=valid.compound, dose=valid.dose, support=valid.support, target=valid.target, foreign=valid.foreign, foreign_ok=valid.foreign_ok.astype(np.uint8), rceb=rceb, historical_teacher_traininternal=teacher, lso_traininternal=lso_prediction)
    decision = {
        "version": VERSION,
        "rceb_version": RCEB_VERSION,
        "dataset": dataset,
        "modality": "CP",
        "budget": budget,
        "split": "valid",
        "pair_seed": PAIR_SEED,
        "test_loaded": False,
        "teacher": teacher_meta,
        "lso": lso_fit_meta | lso_meta,
        "best_strong_baseline": winner,
        "rceb_minus_best_strong_baseline_excess_z": best_contrast,
        "not_clearly_worse_than_best_by_0_01": bool(float(best_contrast["mean_difference"]) > -0.01),
        "limit": "Screening-only Stage C. Historical architectures/targets are reproduced with a train-internal epoch split; results are not interchangeable with old formal historical test artifacts.",
    }
    write_json(outdir / "decision.json", decision)
    print(f"[stage-c] {dataset} CP {budget}R: best={winner}; RCEB-best excess-z={best_contrast['mean_difference']:.6f}", flush=True)
    return decision


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if set(args.datasets) != {"BBBC047", "cpg0004-LINCS"} or set(args.budgets) != {1, 3}:
        raise SystemExit("Stage C is preregistered only for BBBC047/cpg0004-LINCS × CP 1R/3R")
    gate_path = args.output_root / "stage_b" / "STAGE_B_DECISION.json"
    if not gate_path.exists() or not bool(json.loads(gate_path.read_text(encoding="utf-8")).get("passed")):
        raise SystemExit("Stage C requires a passed Stage B gate")
    torch, nn = torch_modules()
    device = select_device(torch, args.device)
    decisions = [run_condition(args, torch, nn, device, dataset, budget) for dataset in args.datasets for budget in args.budgets]
    differences = [float(row["rceb_minus_best_strong_baseline_excess_z"]["mean_difference"]) for row in decisions]
    condition_passes = [bool(row["not_clearly_worse_than_best_by_0_01"]) for row in decisions]
    macro = float(np.mean(differences))
    passed = bool(all(condition_passes) and macro >= 0.0)
    gate = {
        "version": VERSION,
        "stage": "C",
        "conditions": decisions,
        "rule": {
            "per_condition": "RCEB minus better of Teacher/LSO must be > -0.01 in all four conditions",
            "macro": "unweighted mean of four RCEB-minus-best-strong-baseline excess Fisher-z differences must be >= 0",
        },
        "observed": {"rceb_minus_best_macro_excess_z": macro, "per_condition_passes": condition_passes},
        "passed": passed,
        "decision": "GO_TO_STAGE_D" if passed else "STOP_AFTER_STAGE_C",
        "limit": "Validation-only screen; no test profile was opened. These are train-internal refits of historical architectures and labels, not substitutions for the immutable historical formal test artifacts.",
    }
    root = args.output_root / "stage_c"
    write_json(root / "STAGE_C_DECISION.json", gate)
    (root / "STAGE_C_DECISION.md").write_text(
        f"# RCEB Stage C decision\n\nDecision: `{gate['decision']}`.\n\nMacro RCEB−B*: `{macro:.6f}`.\n\nNo test profile was opened.\n",
        encoding="utf-8",
    )
    print(f"[stage-c] {gate['decision']}", flush=True)


if __name__ == "__main__":
    main()
