#!/usr/bin/env python3
"""Diagnostic: can a frozen C0 use the molecule-predictable part of A's correction?"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch

import repro_effect_virtual_cp_mvp as C
import repro_effect_virtual_cp_strict as S


VERSION = "BBBC047-Reproducible-Effect-Correction-Diagnostic-2026-08-29"
ALPHA_GRID = (0.0, 0.25, 0.5, 1.0)


@dataclass
class CorrectionSplit:
    smiles: list[str]
    control: np.ndarray
    fingerprint: np.ndarray
    correction: np.ndarray


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
        parser.error("Correction diagnostic locks folds=5, teacher_epochs=80, student_epochs=40, batch_size=256")
    if args.null_rounds < 1 or args.bootstrap_rounds < 1:
        parser.error("null and bootstrap rounds must be positive")
    return args


def correction_labels(virtual_train: C.VirtualSplit, pairs: Any, teacher_target: np.ndarray) -> CorrectionSplit:
    support = {smiles: value for smiles, value in zip(pairs.smiles, pairs.support)}
    indices = [index for index, smiles in enumerate(virtual_train.smiles) if smiles in support]
    if len(indices) != len(support):
        raise RuntimeError("Mismatch between compound-OOF teacher targets and inner support means")
    residual = np.vstack([teacher_target[index] - support[virtual_train.smiles[index]] for index in indices]).astype(np.float32)
    return CorrectionSplit([virtual_train.smiles[index] for index in indices], virtual_train.control[indices], virtual_train.fingerprint[indices], residual)


def split_correction(data: CorrectionSplit, seed: int) -> tuple[CorrectionSplit, CorrectionSplit]:
    valid_indices = np.asarray([index for index, value in enumerate(data.smiles) if C.BASE.stable_seed(seed, f"correction-checkpoint|{value}") % 10 == 0], dtype=np.int64)
    if valid_indices.size < 100:
        raise RuntimeError("Correction checkpoint split unexpectedly small")
    valid_set = set(valid_indices.tolist())
    fit_indices = np.asarray([index for index in range(len(data.smiles)) if index not in valid_set], dtype=np.int64)
    subset = lambda indices: CorrectionSplit([data.smiles[index] for index in indices.tolist()], data.control[indices], data.fingerprint[indices], data.correction[indices])
    return subset(fit_indices), subset(valid_indices)


def fit_correction(train: CorrectionSplit, valid: CorrectionSplit | None, epochs: int, args: argparse.Namespace, device: torch.device, name: str) -> tuple[C.Student, dict[str, np.ndarray], int, float | None]:
    C.set_seed(args.seed)
    control_mean = train.control.mean(axis=0, dtype=np.float64).astype(np.float32)
    control_scale = np.maximum(train.control.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    correction_mean = train.correction.mean(axis=0, dtype=np.float64).astype(np.float32)
    correction_scale = np.maximum(train.correction.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    train_control = (train.control - control_mean) / control_scale
    train_target = (train.correction - correction_mean) / correction_scale
    if valid is not None:
        valid_control = (valid.control - control_mean) / control_scale
        valid_target = (valid.correction - correction_mean) / correction_scale
    model = C.Student().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(); generator.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(train_control), torch.from_numpy(train.fingerprint), torch.from_numpy(train_target)), batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch, best_value = -1, float("inf")
    for epoch in range(epochs):
        model.train()
        for control, feature, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(control.to(device), feature.to(device)) - target.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {name} loss")
            loss.backward(); optimizer.step()
        if valid is None:
            continue
        model.eval(); total, count = 0.0, 0
        with torch.no_grad():
            for begin in range(0, valid_control.shape[0], args.batch_size):
                control = torch.from_numpy(valid_control[begin:begin + args.batch_size]).to(device)
                feature = torch.from_numpy(valid.fingerprint[begin:begin + args.batch_size]).to(device)
                target = torch.from_numpy(valid_target[begin:begin + args.batch_size]).to(device)
                value = torch.mean((model(control, feature) - target) ** 2)
                total += float(value.cpu()) * len(control); count += len(control)
        value = total / max(count, 1)
        if value < best_value:
            best_value, best_epoch = value, epoch
            best_state = {key: parameter.detach().cpu().clone() for key, parameter in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"[{name} seed {args.seed} epoch {epoch}] validation_correction_mse={value:.7f}", flush=True)
    if valid is not None:
        assert best_state is not None
        model.load_state_dict(best_state)
        return model, {"control_mean": control_mean, "control_scale": control_scale, "target_mean": correction_mean, "target_scale": correction_scale}, best_epoch, best_value
    return model, {"control_mean": control_mean, "control_scale": control_scale, "target_mean": correction_mean, "target_scale": correction_scale}, epochs - 1, None


def unseen_teacher_correction(cp: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[Any, np.ndarray, dict[str, Any]]:
    train_pairs = S.inner_holdout_pairs(cp["train"], args.seed, "unseen-correction-teacher-train")
    valid_pairs = S.inner_holdout_pairs(cp["valid"], args.seed, "unseen-correction-teacher-valid")
    test_pairs = S.inner_holdout_pairs(cp["test"], args.seed, "unseen-correction-teacher-test")
    teacher_args = SimpleNamespace(seed=C.BASE.stable_seed(args.seed, "unseen-correction-teacher") % (2**32 - 1), hidden_dim=256, latent_dim=32, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, epochs=args.teacher_epochs, smoke=args.smoke)
    model, mean, scale, epoch, validation_mse = C.BASE.fit_encoder(train_pairs, valid_pairs, teacher_args, device)
    prediction = C.effect_array(model, test_pairs, mean, scale, args.batch_size, device)
    correction = prediction - test_pairs.support
    return test_pairs, correction.astype(np.float32), {"best_epoch": epoch, "validation_mse": validation_mse, "inner_teacher_audit": S.teacher_inner_audit(test_pairs, prediction)}


def vector_pcc_summary(prediction: dict[str, np.ndarray], pairs: Any, correction: np.ndarray) -> dict[str, Any]:
    values = [C.BASE.pcc(prediction[smiles], target) for smiles, target in zip(pairs.smiles, correction)]
    values = [value for value in values if value is not None]
    if not values:
        raise RuntimeError("No finite test correction PCC")
    return {"definition": "Post-hoc diagnostic only: PCC between g(molecule, baseline) and train-only-teacher R_test = T_test - support_mean. R_test is never fed to a final predictor.", "molecule_count": len(values), "mean_vector_pcc": float(np.mean(values)), "median_vector_pcc": float(np.median(values)), "prediction_rms": float(np.sqrt(np.mean(np.vstack([prediction[value] for value in pairs.smiles]) ** 2))), "target_rms": float(np.sqrt(np.mean(correction ** 2)))}


def score(prediction: dict[str, np.ndarray], pairs: Any, reference: dict[str, dict[str, Any]], rows: Any) -> dict[str, dict[str, Any]]:
    return C.BASE.method_stats(C.held_predictions(prediction, pairs, rows), reference, rows)


def run(args: argparse.Namespace) -> None:
    if not args.dry_run and args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    cp = C.BASE.load_splits(args.cp_rows_root, args.split_lock)
    virtual = C.load_virtual(args.model_h5, args.aggregate_npz, args.split_lock)
    if args.dry_run:
        print(json.dumps({"version": VERSION, "split_counts": {name: len(value.smiles) for name, value in virtual.items()}, "alpha_grid": list(ALPHA_GRID), "guards": ["CP only", "frozen C0", "compound-OOF train correction labels", "alpha selected on validation only", "test R is post-hoc only"]}, indent=2, sort_keys=True))
        return
    args.outdir.mkdir(parents=True)
    device = C.select_device(args.device)

    # C0 is fitted once to raw aggregate deltas and then receives no gradients.
    c0_model, c0_stats, c0_epoch, c0_validation = C.fit_student(virtual["train"], virtual["valid"], virtual["train"].delta, args, device, "c0_frozen_raw_aggregate")

    teacher_target, teacher_meta = S.build_crossfitted_inner_teacher_labels(cp, virtual["train"], args, device)
    train_pairs = S.inner_holdout_pairs(cp["train"], args.seed, "train-inner-held-plate")
    correction_all = correction_labels(virtual["train"], train_pairs, teacher_target)
    correction_fit, correction_valid = split_correction(correction_all, args.seed)
    checkpoint_model, _, correction_epoch, correction_validation = fit_correction(correction_fit, correction_valid, 1 if args.smoke else args.student_epochs, args, device, "g_checkpoint")
    del checkpoint_model
    correction_model, correction_stats, _, _ = fit_correction(correction_all, None, correction_epoch + 1, args, device, "g_final")

    valid_c0 = C.predict_student(c0_model, virtual["valid"], c0_stats, args.batch_size, device)
    valid_g = C.predict_student(correction_model, virtual["valid"], correction_stats, args.batch_size, device)
    valid_pairs, valid_reference = C.BASE.loo_pairs(cp["valid"]), C.BASE.reference_stats(cp["valid"], args.null_rounds, args.seed)
    alpha_validation: dict[float, dict[str, Any]] = {}
    for alpha in ALPHA_GRID:
        candidate = {smiles: valid_c0[smiles] + alpha * valid_g[smiles] for smiles in valid_c0}
        candidate_records = score(candidate, valid_pairs, valid_reference, cp["valid"])
        alpha_validation[alpha] = C.BASE.summarize(candidate_records)
    selected_alpha = max(ALPHA_GRID, key=lambda value: (alpha_validation[value]["rsf"], -value))

    test_c0 = C.predict_student(c0_model, virtual["test"], c0_stats, args.batch_size, device)
    test_g = C.predict_student(correction_model, virtual["test"], correction_stats, args.batch_size, device)
    test_prediction = {smiles: test_c0[smiles] + selected_alpha * test_g[smiles] for smiles in test_c0}
    test_pairs, test_reference = C.BASE.loo_pairs(cp["test"]), C.BASE.reference_stats(cp["test"], args.null_rounds, args.seed)
    records = {"c0_frozen": score(test_c0, test_pairs, test_reference, cp["test"]), "c0_plus_predictable_correction": score(test_prediction, test_pairs, test_reference, cp["test"])}
    contrast = C.BASE.paired_bootstrap(records["c0_plus_predictable_correction"], records["c0_frozen"], args.bootstrap_rounds, C.BASE.stable_seed(args.seed, "correction-minus-c0"))

    diagnostic_pairs, diagnostic_target, diagnostic_teacher = unseen_teacher_correction(cp, args, device)
    direct_correction = {smiles: test_g[smiles] for smiles in diagnostic_pairs.smiles}
    payload = {
        "version": VERSION,
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "device": str(device),
        "inputs": {"cp_rows_root": str(args.cp_rows_root), "aggregate_npz": str(args.aggregate_npz), "model_h5": str(args.model_h5), "split_lock": str(args.split_lock)},
        "locked_hyperparameters": {"teacher_folds": 5, "teacher_epochs_max": 80, "student_epochs_max": 40, "student_architecture": "(CP775+ECFP4-2048)-512-128-CP775", "batch_size": 256, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "null_rounds": args.null_rounds, "bootstrap_rounds": args.bootstrap_rounds, "alpha_grid": list(ALPHA_GRID)},
        "c0": {"definition": "Raw aggregate virtual predictor is trained first and frozen before g is fitted.", "best_epoch": c0_epoch, "validation_mse": c0_validation},
        "train_teacher": teacher_meta,
        "g": {"definition": "g(molecule, pre-treatment baseline) predicts R = T - support_mean. g receives no post-treatment CP.", "eligible_p_ge_3": len(correction_all.smiles), "checkpoint_fit_count": len(correction_fit.smiles), "checkpoint_valid_count": len(correction_valid.smiles), "best_epoch": correction_epoch, "validation_correction_mse": correction_validation},
        "alpha_selection": {"definition": "alpha is selected before test by validation strict held-out-plate RSF only.", "selected_alpha": selected_alpha, "candidates": {str(value): summary for value, summary in alpha_validation.items()}},
        "test_reference_molecule_count": len(test_reference),
        "summaries": {name: C.BASE.summarize(value) for name, value in records.items()},
        "aggregate_delta_pcc": {"c0_frozen": C.mean_pcc(test_c0, virtual["test"]), "c0_plus_predictable_correction": C.mean_pcc(test_prediction, virtual["test"])},
        "paired_bootstrap": {"correction_minus_c0": contrast},
        "test_correction_predictability": vector_pcc_summary(direct_correction, diagnostic_pairs, diagnostic_target),
        "unseen_teacher_audit": diagnostic_teacher,
        "definition": "The final predictor is C0(molecule, baseline) + alpha g(molecule, baseline). All final test predictions are compared only to independent held-out CP plates; test post-treatment CP is read only after alpha selection for evaluation and post-hoc R diagnosis.",
        "guardrails": ["no GE data, feature, target, or loss", "C0 is frozen before g training", "g input is molecule plus baseline CP only", "train R labels use a compound-OOF teacher", "a compound's inner held plate is excluded from its teacher-label support", "alpha selection uses validation held-out-plate RSF only", "test R is post-hoc and never enters prediction or alpha selection", "P<3 compounds are excluded from g supervision rather than silently imputed", "strict held-out RSF is primary", "smoke output is not evidence"],
    }
    with h5py.File(args.outdir / "train_oof_correction.h5", "w") as handle:
        handle.create_dataset("canonical_smiles", data=np.asarray(correction_all.smiles, dtype="S256"))
        handle.create_dataset("cp_oof_reproducible_correction", data=correction_all.correction)
        handle.attrs["teacher_metadata_json"] = json.dumps(teacher_meta, sort_keys=True)
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
    output = {"version": VERSION, "seeds": list(C.SEEDS), "seed_metric_paths": [str(path) for path in paths], "seed_mean_summaries": summary, "selected_alphas": {str(payload["seed"]): payload["alpha_selection"]["selected_alpha"] for payload in payloads}, "test_correction_predictability": {str(payload["seed"]): payload["test_correction_predictability"] for payload in payloads}, "note": "Inspect all seed-specific paired bootstrap correction-minus-C0 contrasts before a conclusion."}
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
