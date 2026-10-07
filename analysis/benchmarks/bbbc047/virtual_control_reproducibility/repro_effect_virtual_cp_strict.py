#!/usr/bin/env python3
"""Strict CP-only virtual prediction: C0, compound-OOF C1, and validation-selected C2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch

import repro_effect_virtual_cp_mvp as C


VERSION = "BBBC047-Reproducible-Effect-Virtual-CP-Strict-C0-C1-C2-2026-08-29"
LAMBDA_GRID = (0.1, 0.25, 0.5, 1.0)


def inner_holdout_pairs(rows: Any, seed: int, label: str) -> Any:
    """One deterministic evaluation plate per compound, excluded from its support.

    Unlike a full leave-one-out average, this construct makes one plate a pure
    within-compound audit target: it cannot appear in the support used to form
    that compound's teacher label.
    """
    support: list[np.ndarray] = []
    target: list[np.ndarray] = []
    smiles: list[str] = []
    held_rows: list[int] = []
    for compound in sorted(rows.groups):
        group = rows.groups[compound]
        if group.size < 3:
            continue
        local = C.BASE.stable_seed(seed, f"{label}|{compound}") % group.size
        held = int(group[int(local)])
        others = group[group != held]
        support.append(rows.delta[others].mean(axis=0, dtype=np.float64).astype(np.float32))
        target.append(rows.delta[held])
        smiles.append(compound)
        held_rows.append(held)
    if not support:
        raise RuntimeError("No P>=3 compounds for inner held-plate protocol")
    return C.BASE.LooPairs(np.vstack(support), np.vstack(target), smiles, np.asarray(held_rows, dtype=np.int64))


def teacher_inner_audit(pairs: Any, prediction: np.ndarray) -> dict[str, Any]:
    predicted = [C.BASE.pcc(value, target) for value, target in zip(prediction, pairs.target)]
    support = [C.BASE.pcc(value, target) for value, target in zip(pairs.support, pairs.target)]
    predicted = [value for value in predicted if value is not None]
    support = [value for value in support if value is not None]
    if not predicted or not support:
        raise RuntimeError("No finite inner teacher audit PCC")
    return {
        "definition": "Audit only: each compound's deterministic inner held plate is excluded from that compound's support and from the compound-OOF teacher fit; it is compared after label generation and never used to select labels or lambda.",
        "compound_count": len(predicted),
        "teacher_vs_reserved_plate_pcc_mean": float(np.mean(predicted)),
        "support_mean_vs_reserved_plate_pcc_mean": float(np.mean(support)),
    }


def build_crossfitted_inner_teacher_labels(
    cp_splits: dict[str, Any], virtual_train: C.VirtualSplit, args: argparse.Namespace, device: torch.device
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build Delta_rep without allowing a compound's reserved plate into its label."""
    train_pairs = inner_holdout_pairs(cp_splits["train"], args.seed, "train-inner-held-plate")
    valid_pairs = inner_holdout_pairs(cp_splits["valid"], args.seed, "valid-inner-held-plate")
    eligible = set(train_pairs.smiles)
    assignment = C.fold_map(virtual_train.smiles, args.seed, args.teacher_folds)
    labels: dict[str, np.ndarray] = {}
    inner_prediction: list[np.ndarray] = []
    inner_pairs: list[Any] = []
    metadata: list[dict[str, Any]] = []
    for fold in range(args.teacher_folds):
        held = {value for value, assigned in assignment.items() if assigned == fold and value in eligible}
        fitted = eligible - held
        if not held or held & fitted:
            raise RuntimeError("Teacher compounds are not disjoint")
        teacher_seed = C.BASE.stable_seed(args.seed, f"strict-inner-teacher-fold{fold}") % (2**32 - 1)
        teacher_args = SimpleNamespace(seed=teacher_seed, hidden_dim=256, latent_dim=32, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, epochs=args.teacher_epochs, smoke=args.smoke)
        model, mean, scale, best_epoch, validation_mse = C.BASE.fit_encoder(C.restrict_pairs(train_pairs, fitted), valid_pairs, teacher_args, device)
        held_pairs = C.restrict_pairs(train_pairs, held)
        prediction = C.effect_array(model, held_pairs, mean, scale, args.batch_size, device)
        for value, predicted_delta in zip(held_pairs.smiles, prediction):
            if value in labels:
                raise RuntimeError(f"Duplicate compound-OOF label: {value}")
            labels[value] = predicted_delta.copy()
        inner_prediction.extend(prediction)
        inner_pairs.extend((held_pairs,))
        metadata.append({"fold": fold, "fit_compounds": len(fitted), "held_compounds": len(held), "best_epoch": best_epoch, "validation_mse": validation_mse})
    ordered_pairs = C.BASE.LooPairs(
        np.vstack([pair.support for pair in inner_pairs]),
        np.vstack([pair.target for pair in inner_pairs]),
        [value for pair in inner_pairs for value in pair.smiles],
        np.concatenate([pair.held_rows for pair in inner_pairs]),
    )
    if set(labels) != eligible:
        raise RuntimeError("Incomplete compound-OOF teacher labels")
    target = virtual_train.delta.copy()
    train_index = {value: index for index, value in enumerate(virtual_train.smiles)}
    for value, predicted_delta in labels.items():
        target[train_index[value]] = predicted_delta
    return target, {
        "fold_count": args.teacher_folds,
        "eligible_p_ge_3": len(eligible),
        "raw_fallback_p_lt_3": len(virtual_train.smiles) - len(eligible),
        "folds": metadata,
        "definition": "For each P>=3 training compound, one deterministic inner held CP plate is excluded from its support. An Experiment-A-style teacher trained on other compound folds predicts that held plate; this prediction is the direct Delta_rep label.",
        "inner_teacher_audit": teacher_inner_audit(ordered_pairs, np.vstack(inner_prediction)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp-rows-root", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"))
    parser.add_argument("--aggregate-npz", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0/molecule_aggregates.npz"))
    parser.add_argument("--model-h5", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/Paired_CP_GE_PlateMedian_v1_model_compat.h5"))
    parser.add_argument("--split-lock", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json"))
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--aggregate-root", type=Path)
    parser.add_argument("--seed", type=int, choices=C.SEEDS)
    parser.add_argument("--teacher-folds", type=int, default=5)
    parser.add_argument("--teacher-epochs", type=int, default=80)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--null-rounds", type=int, default=32)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.aggregate_root:
        if args.seed is not None or args.outdir is not None or args.dry_run or args.smoke:
            parser.error("--aggregate-root cannot be combined with run options")
        return args
    if args.seed is None and not args.dry_run:
        parser.error("--seed is required unless --dry-run or --aggregate-root is used")
    if not args.dry_run and args.outdir is None:
        parser.error("--outdir is required for a run")
    if (args.teacher_folds, args.teacher_epochs, args.student_epochs, args.batch_size) != (5, 80, 40, 256):
        parser.error("Strict C locks folds=5, teacher_epochs=80, student_epochs=40, batch_size=256")
    if args.null_rounds < 1 or args.bootstrap_rounds < 1:
        parser.error("null and bootstrap rounds must be positive")
    return args


def fit_student(
    train: C.VirtualSplit,
    valid: C.VirtualSplit,
    reproducible_label: np.ndarray,
    mean_loss_weight: float,
    args: argparse.Namespace,
    device: torch.device,
    name: str,
) -> tuple[C.Student, dict[str, np.ndarray], int, float]:
    """Fit C0/C1/C2 with identical initialization and minibatch order.

    C2 uses the requested loss exactly: MSE(pred, reproducible) + lambda MSE(pred, mean).
    All deltas share the C0 mean/scale, so lambda is not altered by target normalization.
    """
    C.set_seed(args.seed)
    control_mean = train.control.mean(axis=0, dtype=np.float64).astype(np.float32)
    control_scale = np.maximum(train.control.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    target_mean = train.delta.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = np.maximum(train.delta.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    train_control = (train.control - control_mean) / control_scale
    valid_control = (valid.control - control_mean) / control_scale
    train_reproducible = (reproducible_label - target_mean) / target_scale
    train_mean = (train.delta - target_mean) / target_scale
    valid_mean = (valid.delta - target_mean) / target_scale
    model = C.Student().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(); generator.manual_seed(args.seed)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_control),
        torch.from_numpy(train.fingerprint),
        torch.from_numpy(train_reproducible),
        torch.from_numpy(train_mean),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    best_state: dict[str, torch.Tensor] | None = None
    best_value, best_epoch = float("inf"), -1
    for epoch in range(1 if args.smoke else args.student_epochs):
        model.train()
        for control, feature, reproducible, mean_target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(control.to(device), feature.to(device))
            loss = torch.mean((prediction - reproducible.to(device)) ** 2)
            if mean_loss_weight:
                loss = loss + mean_loss_weight * torch.mean((prediction - mean_target.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {name} loss")
            loss.backward(); optimizer.step()
        model.eval(); total, count = 0.0, 0
        with torch.no_grad():
            for begin in range(0, valid_control.shape[0], args.batch_size):
                control = torch.from_numpy(valid_control[begin:begin + args.batch_size]).to(device)
                feature = torch.from_numpy(valid.fingerprint[begin:begin + args.batch_size]).to(device)
                target = torch.from_numpy(valid_mean[begin:begin + args.batch_size]).to(device)
                value = torch.mean((model(control, feature) - target) ** 2)
                total += float(value.cpu()) * len(control); count += len(control)
        validation_mse = total / max(count, 1)
        if validation_mse < best_value:
            best_value, best_epoch = validation_mse, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"[{name} seed {args.seed} epoch {epoch}] validation_mse={validation_mse:.7f}", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"control_mean": control_mean, "control_scale": control_scale, "target_mean": target_mean, "target_scale": target_scale}, best_epoch, best_value


def score(prediction: dict[str, np.ndarray], pairs: Any, reference: dict[str, dict[str, Any]], rows: Any) -> dict[str, dict[str, Any]]:
    return C.BASE.method_stats(C.held_predictions(prediction, pairs, rows), reference, rows)


def run(args: argparse.Namespace) -> None:
    if not args.dry_run and args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    cp = C.BASE.load_splits(args.cp_rows_root, args.split_lock)
    virtual = C.load_virtual(args.model_h5, args.aggregate_npz, args.split_lock)
    if args.dry_run:
        pairs = {name: C.BASE.loo_pairs(rows) for name, rows in cp.items()}
        print(json.dumps({"version": VERSION, "split_counts": {name: len(rows.smiles) for name, rows in virtual.items()}, "cp_loo_pairs_p_ge_3": {name: int(rows.held_rows.size) for name, rows in pairs.items()}, "lambda_grid": list(LAMBDA_GRID), "guards": ["CP only", "molecule+baseline inference only", "compound-OOF teacher", "lambda selected on validation only"]}, indent=2, sort_keys=True))
        return
    args.outdir.mkdir(parents=True)
    device = C.select_device(args.device)
    reproducible_target, teacher_meta = build_crossfitted_inner_teacher_labels(cp, virtual["train"], args, device)
    with h5py.File(args.outdir / "oof_effect_teacher.h5", "w") as handle:
        handle.create_dataset("canonical_smiles", data=np.asarray(virtual["train"].smiles, dtype="S256"))
        handle.create_dataset("cp_oof_effect_delta", data=reproducible_target)
        handle.attrs["metadata_json"] = json.dumps(teacher_meta, sort_keys=True)

    fitted: dict[str, tuple[C.Student, dict[str, np.ndarray], int, float]] = {}
    fitted["c0_raw_aggregate"] = fit_student(virtual["train"], virtual["valid"], virtual["train"].delta, 0.0, args, device, "c0_raw_aggregate")
    fitted["c1_oof_repro_effect"] = fit_student(virtual["train"], virtual["valid"], reproducible_target, 0.0, args, device, "c1_oof_repro_effect")

    valid_pairs = C.BASE.loo_pairs(cp["valid"])
    valid_reference = C.BASE.reference_stats(cp["valid"], args.null_rounds, args.seed)
    c2_candidates: dict[float, dict[str, Any]] = {}
    for value in LAMBDA_GRID:
        name = f"c2_lambda_{str(value).replace('.', 'p')}"
        model, stats, epoch, validation_mse = fit_student(virtual["train"], virtual["valid"], reproducible_target, value, args, device, name)
        valid_prediction = C.predict_student(model, virtual["valid"], stats, args.batch_size, device)
        valid_records = score(valid_prediction, valid_pairs, valid_reference, cp["valid"])
        c2_candidates[value] = {"model": model, "stats": stats, "best_epoch": epoch, "validation_mse": validation_mse, "validation_rsf": C.BASE.summarize(valid_records)["rsf"], "validation_molecule_count": len(valid_records)}
    selected_lambda = max(LAMBDA_GRID, key=lambda value: (c2_candidates[value]["validation_rsf"], -value))
    selected = c2_candidates[selected_lambda]
    fitted["c2_repro_plus_mean"] = (selected["model"], selected["stats"], selected["best_epoch"], selected["validation_mse"])

    predictions = {name: C.predict_student(model, virtual["test"], stats, args.batch_size, device) for name, (model, stats, _, _) in fitted.items()}
    test_pairs = C.BASE.loo_pairs(cp["test"])
    test_reference = C.BASE.reference_stats(cp["test"], args.null_rounds, args.seed)
    records = {name: score(prediction, test_pairs, test_reference, cp["test"]) for name, prediction in predictions.items()}
    contrasts = {
        "c1_minus_c0": C.BASE.paired_bootstrap(records["c1_oof_repro_effect"], records["c0_raw_aggregate"], args.bootstrap_rounds, C.BASE.stable_seed(args.seed, "c1-minus-c0")),
        "c2_minus_c0": C.BASE.paired_bootstrap(records["c2_repro_plus_mean"], records["c0_raw_aggregate"], args.bootstrap_rounds, C.BASE.stable_seed(args.seed, "c2-minus-c0")),
        "c2_minus_c1": C.BASE.paired_bootstrap(records["c2_repro_plus_mean"], records["c1_oof_repro_effect"], args.bootstrap_rounds, C.BASE.stable_seed(args.seed, "c2-minus-c1")),
    }
    payload = {
        "version": VERSION,
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "device": str(device),
        "inputs": {"cp_rows_root": str(args.cp_rows_root), "aggregate_npz": str(args.aggregate_npz), "model_h5": str(args.model_h5), "split_lock": str(args.split_lock)},
        "locked_hyperparameters": {"teacher_folds": 5, "teacher_architecture": "775-256-32-256-775", "teacher_epochs_max": 80, "student_architecture": "(CP775+ECFP4-2048)-512-128-CP775", "student_epochs_max": 40, "batch_size": 256, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "null_rounds": args.null_rounds, "bootstrap_rounds": args.bootstrap_rounds, "c2_lambda_grid": list(LAMBDA_GRID)},
        "teacher": teacher_meta,
        "c2_selection": {"definition": "C2 minimizes MSE(pred, OOF reproducible delta) + lambda MSE(pred, raw mean delta); lambda is selected before test using validation held-out-plate RSF only.", "selected_lambda": selected_lambda, "candidates": {str(value): {key: candidate[key] for key in ("best_epoch", "validation_mse", "validation_rsf", "validation_molecule_count")} for value, candidate in c2_candidates.items()}},
        "student_checkpoints": {name: {"best_epoch": epoch, "validation_mse": validation_mse} for name, (_, _, epoch, validation_mse) in fitted.items()},
        "test_reference_molecule_count": len(test_reference),
        "summaries": {name: C.BASE.summarize(value) for name, value in records.items()},
        "aggregate_delta_pcc": {name: C.mean_pcc(prediction, virtual["test"]) for name, prediction in predictions.items()},
        "paired_bootstrap": contrasts,
        "definition": "C0/C1/C2 virtual predictors are evaluated on unseen compounds against independent held-out CP plates with an exact plate-slot matched foreign null. Final predictor inputs are molecule fingerprint plus pre-treatment CP baseline only.",
        "guardrails": ["no GE data, feature, target, or loss", "student sees molecule plus baseline CP only", "teacher never trains on the compound it labels", "each compound's deterministic inner held plate is excluded from its teacher-label support", "teacher is frozen before student fitting", "teacher/student fit train only", "test target stays raw and is not read before C2 lambda selection", "C2 lambda selection uses validation held-out-plate RSF only", "P<3 uses declared raw fallback", "strict held-out RSF is primary", "smoke output is not evidence"],
    }
    C.write_scores(args.outdir / "per_molecule_scores.csv", records)
    (args.outdir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def aggregate(root: Path) -> None:
    paths = [root / f"seed{seed}" / "metrics.json" for seed in C.SEEDS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing seed metrics: " + ", ".join(missing))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    methods = sorted(payloads[0]["summaries"])
    summary = {method: {metric: float(np.mean([payload["summaries"][method][metric] for payload in payloads])) for metric in ("model_excess_z_mean", "replicate_excess_z_mean", "rsf")} | {"molecule_count": int(payloads[0]["summaries"][method]["molecule_count"])} for method in methods}
    output = {"version": VERSION, "seeds": list(C.SEEDS), "seed_metric_paths": [str(path) for path in paths], "seed_mean_summaries": summary, "selected_lambdas": {str(payload["seed"]): payload["c2_selection"]["selected_lambda"] for payload in payloads}, "note": "Inspect every seed-specific paired bootstrap contrast. C2 lambda selection uses validation only and is not a test comparison."}
    path = root / "aggregate_index.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.aggregate_root:
        aggregate(args.aggregate_root)
    else:
        run(args)


if __name__ == "__main__":
    main()
