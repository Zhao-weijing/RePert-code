#!/usr/bin/env python3
"""Strict CP posterior controls with train-internal checkpoints and matched nulls."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

import repro_effect_virtual_cp_mvp as C
import repro_effect_prior_posterior as B


FIXED_GRID = tuple(round(value, 3) for value in np.arange(0.0, 0.3001, 0.025))
RELATION_BOOTSTRAP_ROUNDS = 400
VERSION = "BBBC047-Prior-Evidence-Posterior-Strict-Controls-2026-08-29"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("--cp-rows-root", "/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"),
        ("--aggregate-npz", "/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0/molecule_aggregates.npz"),
        ("--model-h5", "/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/Paired_CP_GE_PlateMedian_v1_model_compat.h5"),
        ("--split-lock", "/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json"),
    ):
        parser.add_argument(name, type=Path, default=Path(default))
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--aggregate-root", type=Path)
    parser.add_argument("--seed", type=int, choices=C.SEEDS)
    parser.add_argument("--teacher-folds", type=int, default=5)
    parser.add_argument("--teacher-epochs", type=int, default=80)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--gate-epochs", type=int, default=40)
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
        parser.error("--outdir is required")
    if (args.teacher_folds, args.teacher_epochs, args.student_epochs, args.gate_epochs, args.batch_size) != (5, 80, 40, 40, 256):
        parser.error("Strict controls lock 5 folds, 80/40/40 epochs, batch 256")
    return args


def teacher_members(states: list[B.TeacherState], pairs: B.EvidencePairs, args: argparse.Namespace,
                    device: Any) -> np.ndarray:
    return np.stack([
        C.effect_array(state.model, B.as_loo(pairs), state.mean, state.scale, args.batch_size, device)
        for state in states
    ])


def prior_map(pairs: B.EvidencePairs, values: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    return {int(held): values[smiles] for held, smiles in zip(pairs.held_rows.tolist(), pairs.smiles)}


def fixed_predictions(pairs: B.EvidencePairs, teacher: np.ndarray, prior: dict[str, np.ndarray],
                      lam: float) -> dict[int, np.ndarray]:
    return {
        int(held): (1.0 - lam) * teacher[index] + lam * prior[smiles]
        for index, (held, smiles) in enumerate(zip(pairs.held_rows.tolist(), pairs.smiles))
    }


def select_fixed(valid: B.EvidencePairs, teacher: np.ndarray, prior: dict[str, np.ndarray],
                 reference: dict[str, Any], rows: Any) -> tuple[float, dict[str, Any]]:
    scores = {
        lam: C.BASE.summarize(C.BASE.method_stats(fixed_predictions(valid, teacher, prior, lam), reference, rows))
        for lam in FIXED_GRID
    }
    best = max(FIXED_GRID, key=lambda value: (scores[value]["rsf"], -value))
    return best, {str(key): value for key, value in scores.items()}


def permuted_prior(prior: dict[str, np.ndarray], seed: int, label: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    keys = sorted(prior)
    rng = np.random.default_rng(C.BASE.stable_seed(seed, label))
    donors = keys.copy()
    rng.shuffle(donors)
    if any(left == right for left, right in zip(keys, donors)):
        donors = keys[1:] + keys[:1]
    fixed_points = sum(left == right for left, right in zip(keys, donors))
    if fixed_points:
        raise RuntimeError("Shuffled-prior derangement retained a fixed point")
    return {key: prior[donor] for key, donor in zip(keys, donors)}, {
        "mapping_seed": int(C.BASE.stable_seed(seed, label)), "fixed_points": fixed_points, "molecule_count": len(keys),
    }


def matched_foreign_prior(prior: dict[str, np.ndarray], pairs: B.EvidencePairs, rows: Any,
                          seed: int) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    """Return a pair-specific prior whose donor occupies every target plate slot."""
    values: dict[int, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    prior_compounds = set(prior)
    for held, smiles in zip(pairs.held_rows.tolist(), pairs.smiles):
        held = int(held)
        group = rows.groups[smiles]
        support_rows = [int(row) for row in group.tolist() if int(row) != held]
        held_plate = str(rows.plate[held])
        support_plates = [str(rows.plate[row]) for row in support_rows]
        candidates: set[str] = set(rows.by_plate[held_plate])
        for plate in support_plates:
            candidates &= rows.by_plate[plate]
        choices = sorted((candidates & prior_compounds) - {smiles})
        donor = None
        if choices:
            token = f"matched-foreign|{smiles}|{held_plate}|{'|'.join(sorted(support_plates))}"
            donor = choices[C.BASE.stable_seed(seed, token) % len(choices)]
            values[held] = prior[donor]
        manifest.append({
            "canonical_smiles": smiles,
            "held_row": held,
            "held_plate": held_plate,
            "support_plates": "|".join(sorted(support_plates)),
            "mapping_type": "plate_slot_matched_foreign_prior",
            "candidate_count": len(choices),
            "donor_canonical_smiles": donor or "",
            "eligible": int(donor is not None),
            "mapping_seed": int(C.BASE.stable_seed(seed, "matched-foreign")),
        })
    return values, manifest


def foreign_common_reference(reference: dict[str, dict[str, Any]], foreign_prior: dict[int, np.ndarray]) -> dict[str, dict[str, Any]]:
    common = {
        smiles: record for smiles, record in reference.items()
        if all(int(held) in foreign_prior for held in record["held_rows"])
    }
    if not common:
        raise RuntimeError("No molecule has an exact plate-slot matched foreign prior for every scored held plate")
    return common


def pair_prediction(gate: B.EvidenceGate, pairs: B.EvidencePairs, teacher: np.ndarray,
                    prior_by_held: dict[int, np.ndarray], stats: dict[str, np.ndarray],
                    device: Any) -> dict[int, np.ndarray]:
    evidence = B.gate_features(pairs, stats["feature_mean"], stats["feature_scale"])
    gate.eval()
    with torch.no_grad():
        lam = gate(torch.from_numpy(evidence).to(device)).cpu().numpy().reshape(-1)
    output: dict[int, np.ndarray] = {}
    for index, held in enumerate(pairs.held_rows.tolist()):
        held = int(held)
        if held in prior_by_held:
            output[held] = (1.0 - lam[index]) * teacher[index] + lam[index] * prior_by_held[held]
    return output


def held_delta_pcc(predictions: dict[int, np.ndarray], reference: dict[str, dict[str, Any]],
                   rows: Any) -> dict[str, float]:
    output: dict[str, float] = {}
    for smiles, record in reference.items():
        values = [C.BASE.pcc(predictions[int(held)], rows.delta[int(held)]) for held in record["held_rows"]]
        valid = [value for value in values if value is not None]
        if valid:
            output[smiles] = float(np.mean(valid))
    return output


def scalar_bootstrap(left: dict[str, float], right: dict[str, float], rounds: int, seed: int) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    if not common:
        raise RuntimeError("No common molecule for scalar paired bootstrap")
    diff = np.asarray([left[key] - right[key] for key in common], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(rounds, dtype=np.float64)
    for begin in range(0, rounds, 100):
        end = min(rounds, begin + 100)
        draw = rng.integers(0, len(diff), size=(end - begin, len(diff)))
        boot[begin:end] = diff[draw].mean(axis=1)
    return {
        "molecule_count": len(common),
        "candidate_minus_reference_delta_pcc": float(diff.mean()),
        "candidate_minus_reference_delta_pcc_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }


def score_bundle(predictions: dict[int, np.ndarray], reference: dict[str, dict[str, Any]], rows: Any) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, Any]]:
    scores = C.BASE.method_stats(predictions, reference, rows)
    pcc = held_delta_pcc(predictions, reference, rows)
    summary = C.BASE.summarize(scores)
    common = sorted(set(scores) & set(pcc))
    if not common:
        raise RuntimeError("No common molecule for held-plate PCC summary")
    summary["held_delta_pcc_mean"] = float(np.mean([pcc[key] for key in common]))
    summary["held_delta_pcc_molecule_count"] = len(common)
    return scores, pcc, summary


def contrast(left_scores: dict[str, dict[str, Any]], right_scores: dict[str, dict[str, Any]],
             left_pcc: dict[str, float], right_pcc: dict[str, float], args: argparse.Namespace,
             label: str) -> dict[str, Any]:
    return {
        "excess_z": C.BASE.paired_bootstrap(left_scores, right_scores, args.bootstrap_rounds, C.BASE.stable_seed(args.seed, f"{label}|excess-z")),
        "delta_pcc": scalar_bootstrap(left_pcc, right_pcc, args.bootstrap_rounds, C.BASE.stable_seed(args.seed, f"{label}|delta-pcc")),
    }


def average_tie_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranked[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranked


def spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return None
    return C.BASE.pcc(average_tie_rank(left[finite]), average_tie_rank(right[finite]))


def relation_summary(records: list[dict[str, Any]], split: str, seed: int) -> dict[str, Any]:
    selected = [record for record in records if record["split"] == split]
    metric_names = (
        "support_pairwise_disagreement", "support_dispersion_rms_norm", "teacher_ensemble_sd_rms",
        "support_prior_distance_rms", "teacher_prior_distance_rms", "support_effect_rms", "teacher_effect_rms",
        "prior_effect_rms", "n_plates",
    )
    compounds = sorted({record["canonical_smiles"] for record in selected})
    by_compound = {compound: [i for i, record in enumerate(selected) if record["canonical_smiles"] == compound] for compound in compounds}
    lam = np.asarray([record["lambda_teacher_gate"] for record in selected], dtype=np.float64)
    output: dict[str, Any] = {"split": split, "pair_count": len(selected), "molecule_count": len(compounds), "metrics": {}}
    rng = np.random.default_rng(C.BASE.stable_seed(seed, f"gate-relation-bootstrap|{split}"))
    for name in metric_names:
        values = np.asarray([record[name] for record in selected], dtype=np.float64)
        observed = spearman(lam, values)
        boot: list[float] = []
        if observed is not None:
            for _ in range(RELATION_BOOTSTRAP_ROUNDS):
                draw = rng.integers(0, len(compounds), size=len(compounds))
                indices = np.concatenate([np.asarray(by_compound[compounds[index]], dtype=np.int64) for index in draw])
                value = spearman(lam[indices], values[indices])
                if value is not None:
                    boot.append(value)
        finite = np.isfinite(values)
        quartiles: list[float | None] = []
        if int(finite.sum()) >= 4:
            edges = np.quantile(values[finite], [0.0, 0.25, 0.5, 0.75, 1.0])
            for bin_index in range(4):
                lower, upper = edges[bin_index], edges[bin_index + 1]
                mask = finite & ((values >= lower) if bin_index == 0 else (values > lower)) & (values <= upper)
                quartiles.append(float(np.median(lam[mask])) if mask.any() else None)
        else:
            quartiles = [None, None, None, None]
        output["metrics"][name] = {
            "spearman_rho": observed,
            "cluster_bootstrap_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if boot else None,
            "finite_pair_count": int(finite.sum()),
            "lambda_median_by_metric_quartile": quartiles,
        }
    return output


def pairwise_disagreement(rows: Any, smiles: str, held: int) -> float:
    support = [int(row) for row in rows.groups[smiles].tolist() if int(row) != held]
    values = []
    for left, right in combinations(support, 2):
        correlation = C.BASE.pcc(rows.delta[left], rows.delta[right])
        if correlation is not None:
            values.append(1.0 - correlation)
    return float(np.mean(values)) if values else float("nan")


def evidence_records(split: str, pairs: B.EvidencePairs, rows: Any, teacher_members_value: np.ndarray,
                     teacher: np.ndarray, prior: dict[str, np.ndarray], lam_teacher: np.ndarray,
                     lam_mean: np.ndarray, gate_stats: dict[str, np.ndarray],
                     reference_held_rows: set[int]) -> list[dict[str, Any]]:
    if (teacher_members_value.ndim != 3 or teacher_members_value.shape[1:] != pairs.support.shape
            or teacher.shape != pairs.support.shape or len(lam_teacher) != len(pairs.smiles)
            or len(lam_mean) != len(pairs.smiles)):
        raise ValueError("Evidence/gate arrays are not pair-aligned")
    if not (np.isfinite(lam_teacher).all() and np.isfinite(lam_mean).all()
            and ((0.0 <= lam_teacher) & (lam_teacher <= 1.0)).all()
            and ((0.0 <= lam_mean) & (lam_mean <= 1.0)).all()):
        raise FloatingPointError("Gate lambda must be finite and within [0, 1]")
    scale = gate_stats["target_scale"]
    values: list[dict[str, Any]] = []
    for index, (held, smiles) in enumerate(zip(pairs.held_rows.tolist(), pairs.smiles)):
        held = int(held)
        prior_value = prior[smiles]
        members = teacher_members_value[:, index, :]
        support_plates = [str(rows.plate[row]) for row in rows.groups[smiles].tolist() if int(row) != held]
        rms = lambda value: float(np.sqrt(np.mean(np.square(value / scale))))
        values.append({
            "split": split, "canonical_smiles": smiles, "held_row": held, "held_plate": str(rows.plate[held]),
            "support_plates": "|".join(sorted(support_plates)), "reference_used": int(held in reference_held_rows),
            "n_plates": int(rows.groups[smiles].size), "n_support": int(rows.groups[smiles].size - 1),
            "lambda_teacher_gate": float(lam_teacher[index]), "lambda_mean_gate": float(lam_mean[index]),
            "support_pairwise_disagreement": pairwise_disagreement(rows, smiles, held),
            "support_dispersion_rms_norm": rms(pairs.dispersion[index]),
            "teacher_ensemble_sd_rms": rms(np.std(members, axis=0)),
            "support_prior_distance_rms": rms(pairs.support[index] - prior_value),
            "teacher_prior_distance_rms": rms(teacher[index] - prior_value),
            "support_effect_rms": rms(pairs.support[index]), "teacher_effect_rms": rms(teacher[index]),
            "prior_effect_rms": rms(prior_value),
        })
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    if not args.dry_run and args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    cp = C.BASE.load_splits(args.cp_rows_root, args.split_lock)
    virtual = C.load_virtual(args.model_h5, args.aggregate_npz, args.split_lock)
    if args.dry_run:
        print(json.dumps({
            "version": VERSION, "fixed_lambda_grid": list(FIXED_GRID),
            "guards": [
                "CP only", "official validation selects fixed lambda only", "teacher/C0/gate checkpoints are train-internal",
                "matched foreign donor occupies held and every support plate slot", "evidence-only gates", "test target never enters a model",
            ], "split_counts": {key: len(value.smiles) for key, value in virtual.items()},
        }, indent=2, sort_keys=True))
        return
    args.outdir.mkdir(parents=True)
    device = C.select_device(args.device)
    train, valid, test = (B.evidence_pairs(cp[key]) for key in ("train", "valid", "test"))
    teacher_train, states, teacher_meta = B.fit_oof_teachers(train, virtual["train"], args, device)
    valid_members = teacher_members(states, valid, args, device)
    test_members = teacher_members(states, test, args, device)
    teacher_valid = valid_members.mean(axis=0).astype(np.float32)
    teacher_test = test_members.mean(axis=0).astype(np.float32)

    prior_train, prior_oof_meta = B.oof_prior(virtual["train"], args, device)
    prior_model, prior_stats, prior_full_meta = B.fit_internal_prior(
        virtual["train"], set(virtual["train"].smiles), args, device, "c0-controls-full"
    )
    prior_valid = C.predict_student(prior_model, virtual["valid"], prior_stats, args.batch_size, device)
    prior_test = C.predict_student(prior_model, virtual["test"], prior_stats, args.batch_size, device)

    gate_teacher, gate_teacher_stats, gate_teacher_meta = B.fit_gate_internal(
        train, teacher_train, prior_train, args, device, "posterior-teacher-gate"
    )
    gate_mean, gate_mean_stats, gate_mean_meta = B.fit_gate_internal(
        train, train.support, prior_train, args, device, "posterior-mean-gate"
    )

    valid_reference = C.BASE.reference_stats(cp["valid"], args.null_rounds, args.seed)
    fixed_lambda, fixed_validation = select_fixed(valid, teacher_valid, prior_valid, valid_reference, cp["valid"])
    posterior_teacher, lambda_teacher = B.gated_prediction(
        gate_teacher, test, teacher_test, prior_test, gate_teacher_stats, device
    )
    posterior_mean, lambda_mean = B.gated_prediction(
        gate_mean, test, test.support, prior_test, gate_mean_stats, device
    )
    _, valid_lambda_teacher = B.gated_prediction(gate_teacher, valid, teacher_valid, prior_valid, gate_teacher_stats, device)
    _, valid_lambda_mean = B.gated_prediction(gate_mean, valid, valid.support, prior_valid, gate_mean_stats, device)
    shuffled_prior, shuffled_meta = permuted_prior(prior_test, args.seed, "shuffled-prior")
    posterior_shuffled, _ = B.gated_prediction(gate_teacher, test, teacher_test, shuffled_prior, gate_teacher_stats, device)
    foreign_prior, foreign_manifest = matched_foreign_prior(prior_test, test, cp["test"], args.seed)
    posterior_foreign = pair_prediction(gate_teacher, test, teacher_test, foreign_prior, gate_teacher_stats, device)

    full_methods = {
        "support_mean": {int(held): test.support[index] for index, held in enumerate(test.held_rows.tolist())},
        "a_teacher": {int(held): teacher_test[index] for index, held in enumerate(test.held_rows.tolist())},
        "c0_prior": prior_map(test, prior_test),
        "posterior_adaptive_teacher": posterior_teacher,
        "posterior_fixed_teacher": fixed_predictions(test, teacher_test, prior_test, fixed_lambda),
        "posterior_adaptive_mean": posterior_mean,
        "posterior_shuffled_prior": posterior_shuffled,
    }
    reference = C.BASE.reference_stats(cp["test"], args.null_rounds, args.seed)
    common_foreign_reference = foreign_common_reference(reference, foreign_prior)
    reference_held_rows = {int(held) for record in reference.values() for held in record["held_rows"]}
    common_foreign_compounds = set(common_foreign_reference)
    for record in foreign_manifest:
        record["reference_used"] = int(record["held_row"] in reference_held_rows)
        record["common_molecule_used"] = int(record["canonical_smiles"] in common_foreign_compounds)
    common_methods = {**full_methods, "posterior_matched_foreign_prior": posterior_foreign}
    full_bundles = {name: score_bundle(prediction, reference, cp["test"]) for name, prediction in full_methods.items()}
    foreign_bundles = {name: score_bundle(prediction, common_foreign_reference, cp["test"]) for name, prediction in common_methods.items()}

    paired = {
        "adaptive_minus_fixed": contrast(full_bundles["posterior_adaptive_teacher"][0], full_bundles["posterior_fixed_teacher"][0], full_bundles["posterior_adaptive_teacher"][1], full_bundles["posterior_fixed_teacher"][1], args, "adaptive-minus-fixed"),
        "adaptive_minus_teacher": contrast(full_bundles["posterior_adaptive_teacher"][0], full_bundles["a_teacher"][0], full_bundles["posterior_adaptive_teacher"][1], full_bundles["a_teacher"][1], args, "adaptive-minus-teacher"),
        "fixed_minus_teacher": contrast(full_bundles["posterior_fixed_teacher"][0], full_bundles["a_teacher"][0], full_bundles["posterior_fixed_teacher"][1], full_bundles["a_teacher"][1], args, "fixed-minus-teacher"),
        "teacher_prior_minus_mean_prior": contrast(full_bundles["posterior_adaptive_teacher"][0], full_bundles["posterior_adaptive_mean"][0], full_bundles["posterior_adaptive_teacher"][1], full_bundles["posterior_adaptive_mean"][1], args, "teacher-prior-minus-mean-prior"),
        "correct_minus_shuffled": contrast(full_bundles["posterior_adaptive_teacher"][0], full_bundles["posterior_shuffled_prior"][0], full_bundles["posterior_adaptive_teacher"][1], full_bundles["posterior_shuffled_prior"][1], args, "correct-minus-shuffled"),
        "teacher_minus_support": contrast(full_bundles["a_teacher"][0], full_bundles["support_mean"][0], full_bundles["a_teacher"][1], full_bundles["support_mean"][1], args, "teacher-minus-support"),
        "correct_minus_matched_foreign_common": contrast(foreign_bundles["posterior_adaptive_teacher"][0], foreign_bundles["posterior_matched_foreign_prior"][0], foreign_bundles["posterior_adaptive_teacher"][1], foreign_bundles["posterior_matched_foreign_prior"][1], args, "correct-minus-matched-foreign"),
    }

    valid_reference_held_rows = {int(held) for record in valid_reference.values() for held in record["held_rows"]}
    evidence = evidence_records("validation", valid, cp["valid"], valid_members, teacher_valid, prior_valid, valid_lambda_teacher, valid_lambda_mean, gate_teacher_stats, valid_reference_held_rows)
    evidence += evidence_records("test", test, cp["test"], test_members, teacher_test, prior_test, lambda_teacher, lambda_mean, gate_teacher_stats, reference_held_rows)
    write_csv(args.outdir / "evidence_lambda.csv", evidence)
    write_csv(args.outdir / "prior_mapping.csv", foreign_manifest)
    C.write_scores(args.outdir / "per_molecule_scores.csv", {name: bundle[0] for name, bundle in full_bundles.items()})
    C.write_scores(args.outdir / "matched_foreign_common_per_molecule_scores.csv", {name: bundle[0] for name, bundle in foreign_bundles.items()})

    eligible_pairs = sum(int(record["eligible"]) for record in foreign_manifest)
    reference_pair_count = len(reference_held_rows)
    eligible_reference_pairs = sum(int(record["eligible"]) for record in foreign_manifest if record["held_row"] in reference_held_rows)
    payload = {
        "version": VERSION, "seed": args.seed, "smoke": bool(args.smoke), "device": str(device),
        "fixed_lambda": {"grid": list(FIXED_GRID), "selected_on_validation_rsf": fixed_lambda, "validation_scores": fixed_validation},
        "teacher": teacher_meta,
        "prior": {"oof": prior_oof_meta, "full_train_internal_checkpoint": prior_full_meta},
        "gates": {"adaptive_teacher": gate_teacher_meta, "adaptive_mean": gate_mean_meta},
        "foreign_mapping": {
            "definition": "For each (compound, held plate), donor is a different molecule present on the held plate and every support plate. No CP values select the donor.",
            "pair_count": len(foreign_manifest), "eligible_pair_count": eligible_pairs,
            "pair_coverage": float(eligible_pairs / len(foreign_manifest)),
            "reference_pair_count": reference_pair_count, "eligible_reference_pair_count": eligible_reference_pairs,
            "reference_pair_coverage": float(eligible_reference_pairs / reference_pair_count),
            "full_reference_molecule_count": len(reference), "common_molecule_count": len(common_foreign_reference),
            "common_molecule_coverage": float(len(common_foreign_reference) / len(reference)),
        },
        "shuffled_prior": shuffled_meta,
        "lambda_summary": {
            "validation_mean_teacher": float(valid_lambda_teacher.mean()), "validation_mean_mean": float(valid_lambda_mean.mean()),
            "test_mean_teacher": float(lambda_teacher.mean()), "test_mean_mean": float(lambda_mean.mean()),
        },
        "evidence_lambda": {
            "definition": "Post-hoc evidence description after all models are frozen; no held target enters any evidence covariate.",
            "validation": relation_summary(evidence, "validation", args.seed), "test": relation_summary(evidence, "test", args.seed),
        },
        "test_reference_molecule_count": len(reference),
        "summaries": {name: bundle[2] for name, bundle in full_bundles.items()},
        "matched_foreign_common_summaries": {name: bundle[2] for name, bundle in foreign_bundles.items()},
        "paired_bootstrap": paired,
        "definition": "Strict controls test whether a compound-specific C0 prior provides a posterior correction to repeat-based A teacher under independent held-plate evaluation.",
        "guardrails": [
            "no GE data, feature, target, or loss", "teacher/C0/gate checkpoints are train-internal", "official validation selects fixed lambda only",
            "gate sees support mean and support dispersion only", "shuffled and exact plate-slot matched foreign priors never select models",
            "test held target is never an input", "strict held-out RSF and held delta PCC are reported", "smoke is not evidence",
        ],
    }
    (args.outdir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def aggregate(root: Path) -> None:
    paths = [root / f"seed{seed}" / "metrics.json" for seed in C.SEEDS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing seed metrics: " + ", ".join(missing))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    names = sorted(payloads[0]["summaries"])
    metrics = ("model_excess_z_mean", "replicate_excess_z_mean", "rsf", "held_delta_pcc_mean")
    summary = {
        name: {metric: float(np.mean([payload["summaries"][name][metric] for payload in payloads])) for metric in metrics}
        | {"molecule_count": int(payloads[0]["summaries"][name]["molecule_count"])}
        for name in names
    }
    output = {
        "version": VERSION, "seeds": list(C.SEEDS), "seed_mean_summaries": summary,
        "selected_fixed_lambda": {str(payload["seed"]): payload["fixed_lambda"]["selected_on_validation_rsf"] for payload in payloads},
        "foreign_mapping": {str(payload["seed"]): payload["foreign_mapping"] for payload in payloads},
        "note": "Read each seed's paired bootstrap before a decision; matched-foreign comparisons use the explicitly reported common molecule set.",
    }
    (root / "aggregate_index.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    aggregate(args.aggregate_root) if args.aggregate_root else run(args)


if __name__ == "__main__":
    main()
