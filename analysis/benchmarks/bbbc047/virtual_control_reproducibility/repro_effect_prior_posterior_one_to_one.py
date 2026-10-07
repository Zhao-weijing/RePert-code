#!/usr/bin/env python3
"""Independent CP 1->1 posterior confirmation: one support plate predicts one held plate."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import repro_effect_virtual_cp_mvp as C
import repro_effect_prior_posterior as B
import repro_effect_prior_posterior_controls as S


PAIR_MANIFEST_SEED = 3407
EXPECTED_PAIR_COUNTS = {"train": 12121, "valid": 4041, "test": 4040}
VERSION = "BBBC047-Prior-Evidence-Posterior-OneToOne-2026-08-29"


@dataclass
class OneToOnePairs:
    support: np.ndarray
    dispersion: np.ndarray
    target: np.ndarray
    smiles: list[str]
    held_rows: np.ndarray
    support_rows: np.ndarray
    extra_support_rows: np.ndarray
    two_support: np.ndarray


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
        parser.error("One-to-one confirmation locks 5 folds, 80/40/40 epochs, batch 256")
    return args


def choose(rows: list[int], label: str) -> int:
    if not rows:
        raise RuntimeError(f"Cannot choose an empty plate list for {label}")
    ordered = list(rows)
    return ordered[C.BASE.stable_seed(PAIR_MANIFEST_SEED, label) % len(ordered)]


def one_to_one_pairs(rows: Any, split: str) -> tuple[OneToOnePairs, list[dict[str, Any]]]:
    support: list[np.ndarray] = []
    target: list[np.ndarray] = []
    smiles: list[str] = []
    held_rows: list[int] = []
    support_rows: list[int] = []
    extra_rows: list[int] = []
    two_support: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    for smiles_value in sorted(rows.groups):
        group = sorted((int(row) for row in rows.groups[smiles_value].tolist()), key=lambda row: (str(rows.plate[row]), row))
        if len(group) < 2:
            continue
        support_row = choose(group, f"one-to-one-support|{smiles_value}")
        held_row = choose([row for row in group if row != support_row], f"one-to-one-held|{smiles_value}")
        available = [row for row in group if row not in (support_row, held_row)]
        extra_row = choose(available, f"one-to-one-extra|{smiles_value}") if available else -1
        support.append(rows.delta[support_row])
        target.append(rows.delta[held_row])
        smiles.append(smiles_value)
        held_rows.append(held_row)
        support_rows.append(support_row)
        extra_rows.append(extra_row)
        two_support.append(((rows.delta[support_row] + rows.delta[extra_row]) / 2.0).astype(np.float32) if extra_row >= 0 else np.full(C.CP_DIM, np.nan, dtype=np.float32))
        manifest.append({
            "split": split, "canonical_smiles": smiles_value, "n_plates": len(group),
            "support_row": support_row, "support_plate": str(rows.plate[support_row]),
            "held_row": held_row, "held_plate": str(rows.plate[held_row]),
            "extra_support_row": extra_row,
            "extra_support_plate": str(rows.plate[extra_row]) if extra_row >= 0 else "",
            "pair_manifest_seed": PAIR_MANIFEST_SEED,
            "selection_rule": "stable support, then distinct held, then optional distinct extra support; CP values never select rows",
        })
    if len(smiles) != EXPECTED_PAIR_COUNTS[split]:
        raise RuntimeError(f"{split} one-to-one pair count {len(smiles)} != locked {EXPECTED_PAIR_COUNTS[split]}")
    pairs = OneToOnePairs(
        support=np.vstack(support).astype(np.float32),
        dispersion=np.zeros((len(support), C.CP_DIM), dtype=np.float32),
        target=np.vstack(target).astype(np.float32), smiles=smiles,
        held_rows=np.asarray(held_rows, dtype=np.int64), support_rows=np.asarray(support_rows, dtype=np.int64),
        extra_support_rows=np.asarray(extra_rows, dtype=np.int64), two_support=np.vstack(two_support).astype(np.float32),
    )
    if not np.array_equal(pairs.dispersion, np.zeros_like(pairs.dispersion)):
        raise AssertionError("One-to-one dispersion must be exactly zero")
    if np.any(pairs.support_rows == pairs.held_rows):
        raise AssertionError("One-to-one support and held rows must differ")
    if np.any((pairs.extra_support_rows >= 0) & ((pairs.extra_support_rows == pairs.support_rows) | (pairs.extra_support_rows == pairs.held_rows))):
        raise AssertionError("Two-support auxiliary row must differ from support and held")
    return pairs, manifest


def as_evidence(pairs: OneToOnePairs) -> B.EvidencePairs:
    return B.EvidencePairs(pairs.support, pairs.dispersion, pairs.target, pairs.smiles, pairs.held_rows)


def selected_reference_stats(rows: Any, pairs: OneToOnePairs, null_rounds: int, seed: int) -> dict[str, dict[str, Any]]:
    """One held/support slot per compound with a two-compound exact plate-slot null."""
    output: dict[str, dict[str, Any]] = {}
    for index, smiles_value in enumerate(pairs.smiles):
        held = int(pairs.held_rows[index])
        support = int(pairs.support_rows[index])
        held_plate, support_plate = str(rows.plate[held]), str(rows.plate[support])
        replicate = C.BASE.pcc(rows.delta[held], rows.delta[support])
        left_choices = sorted(rows.by_plate[held_plate] - {smiles_value})
        right_choices = sorted(rows.by_plate[support_plate] - {smiles_value})
        null_values: list[float] = []
        if replicate is not None and left_choices and right_choices:
            rng = np.random.default_rng(C.BASE.stable_seed(seed, f"one-to-one-null|{smiles_value}|{held_plate}|{support_plate}"))
            for _ in range(null_rounds):
                for _attempt in range(128):
                    left = left_choices[int(rng.integers(len(left_choices)))]
                    right = right_choices[int(rng.integers(len(right_choices)))]
                    if left != right:
                        break
                else:
                    continue
                value = C.BASE.pcc(rows.delta[rows.index[(held_plate, left)]], rows.delta[rows.index[(support_plate, right)]])
                if value is not None:
                    null_values.append(C.BASE.fisher_z(value))
        if null_values:
            replicate_z = C.BASE.fisher_z(replicate)
            output[smiles_value] = {
                "n_plates": int(rows.groups[smiles_value].size), "used_holdout_plates": 1,
                "held_rows": [held], "support_row": support, "null_z": float(np.mean(null_values)),
                "replicate_z": replicate_z, "replicate_excess_z": float(replicate_z - np.mean(null_values)),
                "null_draw_count": len(null_values),
            }
    if not output:
        raise RuntimeError("No 1->1 pair had an exact plate-slot matched null")
    return output


def matched_foreign_prior(prior: dict[str, np.ndarray], pairs: OneToOnePairs, rows: Any,
                          seed: int) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    values: dict[int, np.ndarray] = {}
    manifest: list[dict[str, Any]] = []
    for index, smiles_value in enumerate(pairs.smiles):
        held, support = int(pairs.held_rows[index]), int(pairs.support_rows[index])
        held_plate, support_plate = str(rows.plate[held]), str(rows.plate[support])
        choices = sorted((set(rows.by_plate[held_plate]) & rows.by_plate[support_plate] & set(prior)) - {smiles_value})
        donor = None
        if choices:
            donor = choices[C.BASE.stable_seed(seed, f"one-to-one-foreign|{smiles_value}|{held_plate}|{support_plate}") % len(choices)]
            values[held] = prior[donor]
        manifest.append({
            "canonical_smiles": smiles_value, "held_row": held, "held_plate": held_plate,
            "support_row": support, "support_plate": support_plate,
            "mapping_type": "one_to_one_same_donor_plate_slot_matched_prior",
            "candidate_count": len(choices), "donor_canonical_smiles": donor or "", "eligible": int(donor is not None),
            "mapping_seed": int(C.BASE.stable_seed(seed, "one-to-one-foreign")),
        })
    return values, manifest


def teacher_members(states: list[B.TeacherState], pairs: OneToOnePairs, args: argparse.Namespace,
                    device: Any) -> np.ndarray:
    evidence = as_evidence(pairs)
    return np.stack([
        C.effect_array(state.model, B.as_loo(evidence), state.mean, state.scale, args.batch_size, device)
        for state in states
    ])


def evidence_rows(split: str, pairs: OneToOnePairs, rows: Any, teacher: np.ndarray, prior: dict[str, np.ndarray],
                  lambda_adaptive: np.ndarray, gate_stats: dict[str, np.ndarray], reference: dict[str, Any]) -> list[dict[str, Any]]:
    if not (np.isfinite(lambda_adaptive).all() and ((0.0 <= lambda_adaptive) & (lambda_adaptive <= 1.0)).all()):
        raise FloatingPointError("One-to-one gate lambda must be finite and within [0, 1]")
    scale = gate_stats["target_scale"]
    reference_rows = {int(held) for record in reference.values() for held in record["held_rows"]}
    result = []
    for index, smiles_value in enumerate(pairs.smiles):
        held, support = int(pairs.held_rows[index]), int(pairs.support_rows[index])
        prior_value = prior[smiles_value]
        rms = lambda value: float(np.sqrt(np.mean(np.square(value / scale))))
        result.append({
            "split": split, "canonical_smiles": smiles_value, "held_row": held, "held_plate": str(rows.plate[held]),
            "support_row": support, "support_plate": str(rows.plate[support]),
            "n_plates": int(rows.groups[smiles_value].size), "lambda_teacher_gate": float(lambda_adaptive[index]),
            "support_effect_rms": rms(pairs.support[index]), "support_prior_distance_rms": rms(pairs.support[index] - prior_value),
            "teacher_prior_distance_rms": rms(teacher[index] - prior_value), "reference_used": int(held in reference_rows),
        })
    return result


def auxiliary_two_support(pairs: OneToOnePairs, rows: Any, posterior_pcc: dict[str, float],
                          args: argparse.Namespace) -> dict[str, Any]:
    two_support: dict[str, float] = {}
    for index, smiles_value in enumerate(pairs.smiles):
        if int(pairs.extra_support_rows[index]) < 0:
            continue
        value = C.BASE.pcc(pairs.two_support[index], rows.delta[int(pairs.held_rows[index])])
        if value is not None:
            two_support[smiles_value] = value
    common = sorted(set(two_support) & set(posterior_pcc))
    if not common:
        raise RuntimeError("No P>=3 molecule for the synchronized 2->1 auxiliary comparison")
    one = {smiles_value: posterior_pcc[smiles_value] for smiles_value in common}
    two = {smiles_value: two_support[smiles_value] for smiles_value in common}
    return {
        "definition": "Auxiliary raw held-plate PCC comparison on P>=3 molecules sharing the identical support S and held H; two-support mean uses S plus the locked extra support R.",
        "molecule_count": len(common), "one_posterior_fixed_held_delta_pcc_mean": float(np.mean(list(one.values()))),
        "two_support_mean_held_delta_pcc_mean": float(np.mean(list(two.values()))),
        "one_posterior_fixed_minus_two_support_mean_delta_pcc": S.scalar_bootstrap(one, two, args.bootstrap_rounds, C.BASE.stable_seed(args.seed, "one-to-one-fixed-minus-two-support")),
    }


def save_states(outdir: Path, states: list[B.TeacherState], teacher_train: np.ndarray, train: OneToOnePairs,
                prior_model: Any, prior_stats: dict[str, np.ndarray], gate: Any, gate_stats: dict[str, np.ndarray]) -> None:
    np.savez_compressed(outdir / "one_to_one_oof_teacher.npz", canonical_smiles=np.asarray(train.smiles), oof_teacher=teacher_train)
    torch.save({"teacher_states": [{"state_dict": state.model.state_dict(), "mean": state.mean, "scale": state.scale} for state in states]}, outdir / "one_to_one_teacher_states.pt")
    torch.save({"state_dict": prior_model.state_dict(), "stats": prior_stats}, outdir / "c0_train_only.pt")
    torch.save({"state_dict": gate.state_dict(), "stats": gate_stats}, outdir / "gate_one_to_one.pt")


def run(args: argparse.Namespace) -> None:
    if not args.dry_run and args.outdir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    cp = C.BASE.load_splits(args.cp_rows_root, args.split_lock)
    virtual = C.load_virtual(args.model_h5, args.aggregate_npz, args.split_lock)
    pairs_and_manifests = {name: one_to_one_pairs(cp[name], name) for name in ("train", "valid", "test")}
    pairs = {name: value[0] for name, value in pairs_and_manifests.items()}
    manifests = [row for name in ("train", "valid", "test") for row in pairs_and_manifests[name][1]]
    if args.dry_run:
        print(json.dumps({
            "version": VERSION, "pair_manifest_seed": PAIR_MANIFEST_SEED,
            "pair_counts": {name: len(value.smiles) for name, value in pairs.items()},
            "two_support_p_ge_3_counts": {name: int(np.sum(value.extra_support_rows >= 0)) for name, value in pairs.items()},
            "guards": ["one real support plate and a distinct held plate", "C0/teacher/gate checkpoints train-internal", "validation 1->1 RSF selects fixed lambda only", "no GE"],
        }, indent=2, sort_keys=True))
        return
    args.outdir.mkdir(parents=True)
    S.write_csv(args.outdir / "pair_manifest.csv", manifests)
    device = C.select_device(args.device)
    train, valid, test = (pairs[name] for name in ("train", "valid", "test"))
    train_evidence, valid_evidence, test_evidence = (as_evidence(value) for value in (train, valid, test))
    teacher_train, states, teacher_meta = B.fit_oof_teachers(train_evidence, virtual["train"], args, device)
    teacher_valid_members = teacher_members(states, valid, args, device)
    teacher_test_members = teacher_members(states, test, args, device)
    teacher_valid = teacher_valid_members.mean(axis=0).astype(np.float32)
    teacher_test = teacher_test_members.mean(axis=0).astype(np.float32)
    prior_train, prior_oof_meta = B.oof_prior(virtual["train"], args, device)
    prior_model, prior_stats, prior_full_meta = B.fit_internal_prior(virtual["train"], set(virtual["train"].smiles), args, device, "one-to-one-c0-full")
    prior_valid = C.predict_student(prior_model, virtual["valid"], prior_stats, args.batch_size, device)
    prior_test = C.predict_student(prior_model, virtual["test"], prior_stats, args.batch_size, device)
    gate, gate_stats, gate_meta = B.fit_gate_internal(train_evidence, teacher_train, prior_train, args, device, "one-to-one-gate")
    valid_reference = selected_reference_stats(cp["valid"], valid, args.null_rounds, args.seed)
    fixed_lambda, fixed_validation = S.select_fixed(valid_evidence, teacher_valid, prior_valid, valid_reference, cp["valid"])
    posterior_adaptive, lambda_adaptive = B.gated_prediction(gate, test_evidence, teacher_test, prior_test, gate_stats, device)
    posterior_fixed = S.fixed_predictions(test_evidence, teacher_test, prior_test, fixed_lambda)
    shuffled_prior, shuffled_meta = S.permuted_prior(prior_test, args.seed, "one-to-one-shuffled-prior")
    posterior_shuffled, _ = B.gated_prediction(gate, test_evidence, teacher_test, shuffled_prior, gate_stats, device)
    foreign_prior, foreign_manifest = matched_foreign_prior(prior_test, test, cp["test"], args.seed)
    posterior_foreign = S.pair_prediction(gate, test_evidence, teacher_test, foreign_prior, gate_stats, device)
    reference = selected_reference_stats(cp["test"], test, args.null_rounds, args.seed)
    common_foreign_reference = S.foreign_common_reference(reference, foreign_prior)
    reference_rows = {int(held) for record in reference.values() for held in record["held_rows"]}
    common_foreign_compounds = set(common_foreign_reference)
    for row in foreign_manifest:
        row["reference_used"] = int(row["held_row"] in reference_rows)
        row["common_molecule_used"] = int(row["canonical_smiles"] in common_foreign_compounds)
    methods = {
        "one_support": {int(held): test.support[index] for index, held in enumerate(test.held_rows.tolist())},
        "one_teacher": {int(held): teacher_test[index] for index, held in enumerate(test.held_rows.tolist())},
        "c0_prior": S.prior_map(test_evidence, prior_test),
        "one_posterior_adaptive": posterior_adaptive,
        "one_posterior_fixed": posterior_fixed,
        "one_posterior_shuffled_prior": posterior_shuffled,
    }
    full = {name: S.score_bundle(prediction, reference, cp["test"]) for name, prediction in methods.items()}
    common = {**methods, "one_posterior_matched_foreign_prior": posterior_foreign}
    foreign = {name: S.score_bundle(prediction, common_foreign_reference, cp["test"]) for name, prediction in common.items()}
    paired = {
        "adaptive_minus_fixed": S.contrast(full["one_posterior_adaptive"][0], full["one_posterior_fixed"][0], full["one_posterior_adaptive"][1], full["one_posterior_fixed"][1], args, "one-to-one-adaptive-minus-fixed"),
        "adaptive_minus_teacher": S.contrast(full["one_posterior_adaptive"][0], full["one_teacher"][0], full["one_posterior_adaptive"][1], full["one_teacher"][1], args, "one-to-one-adaptive-minus-teacher"),
        "fixed_minus_teacher": S.contrast(full["one_posterior_fixed"][0], full["one_teacher"][0], full["one_posterior_fixed"][1], full["one_teacher"][1], args, "one-to-one-fixed-minus-teacher"),
        "correct_minus_shuffled": S.contrast(full["one_posterior_adaptive"][0], full["one_posterior_shuffled_prior"][0], full["one_posterior_adaptive"][1], full["one_posterior_shuffled_prior"][1], args, "one-to-one-correct-minus-shuffled"),
        "teacher_minus_support": S.contrast(full["one_teacher"][0], full["one_support"][0], full["one_teacher"][1], full["one_support"][1], args, "one-to-one-teacher-minus-support"),
        "correct_minus_matched_foreign_common": S.contrast(foreign["one_posterior_adaptive"][0], foreign["one_posterior_matched_foreign_prior"][0], foreign["one_posterior_adaptive"][1], foreign["one_posterior_matched_foreign_prior"][1], args, "one-to-one-correct-minus-foreign"),
    }
    _, valid_lambda = B.gated_prediction(gate, valid_evidence, teacher_valid, prior_valid, gate_stats, device)
    evidence = evidence_rows("validation", valid, cp["valid"], teacher_valid, prior_valid, valid_lambda, gate_stats, valid_reference)
    evidence += evidence_rows("test", test, cp["test"], teacher_test, prior_test, lambda_adaptive, gate_stats, reference)
    S.write_csv(args.outdir / "evidence_lambda.csv", evidence)
    S.write_csv(args.outdir / "prior_mapping.csv", foreign_manifest)
    C.write_scores(args.outdir / "per_molecule_scores.csv", {name: value[0] for name, value in full.items()})
    C.write_scores(args.outdir / "matched_foreign_common_per_molecule_scores.csv", {name: value[0] for name, value in foreign.items()})
    save_states(args.outdir, states, teacher_train, train, prior_model, prior_stats, gate, gate_stats)
    reference_pair_count = len(reference_rows)
    eligible_reference_pairs = sum(int(row["eligible"]) for row in foreign_manifest if row["held_row"] in reference_rows)
    payload = {
        "version": VERSION, "seed": args.seed, "smoke": bool(args.smoke), "device": str(device),
        "pair_manifest": {"seed": PAIR_MANIFEST_SEED, "pair_counts": {name: len(value.smiles) for name, value in pairs.items()}, "two_support_p_ge_3_counts": {name: int(np.sum(value.extra_support_rows >= 0)) for name, value in pairs.items()}},
        "teacher": teacher_meta, "prior": {"oof": prior_oof_meta, "full_train_internal_checkpoint": prior_full_meta}, "gate": gate_meta,
        "fixed_lambda": {"grid": list(S.FIXED_GRID), "selected_on_validation_one_to_one_rsf": fixed_lambda, "validation_scores": fixed_validation},
        "foreign_mapping": {"pair_count": len(foreign_manifest), "eligible_pair_count": sum(int(row["eligible"]) for row in foreign_manifest), "reference_pair_count": reference_pair_count, "eligible_reference_pair_count": eligible_reference_pairs, "reference_pair_coverage": float(eligible_reference_pairs / reference_pair_count), "common_molecule_count": len(common_foreign_reference), "common_molecule_coverage": float(len(common_foreign_reference) / len(reference))},
        "shuffled_prior": shuffled_meta, "lambda_summary": {"validation_mean": float(valid_lambda.mean()), "test_mean": float(lambda_adaptive.mean())},
        "test_reference_molecule_count": len(reference), "summaries": {name: value[2] for name, value in full.items()}, "matched_foreign_common_summaries": {name: value[2] for name, value in foreign.items()},
        "paired_bootstrap": paired, "two_support_auxiliary": auxiliary_two_support(test, cp["test"], full["one_posterior_fixed"][1], args),
        "definition": "Primary 1->1: exactly one real support plate and one distinct held plate per compound. The virtual C0 prior is only a train-only molecule-plus-baseline correction after one support is observed.",
        "guardrails": ["no GE", "single support and held plate are deterministic and distinct", "teacher/C0/gate checkpoints are train-internal", "fixed lambda uses 1->1 validation RSF only", "test target never enters model fitting", "strict selected-pair RSF plus held delta PCC are reported", "smoke is not evidence"],
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
    output = {
        "version": VERSION, "seeds": list(C.SEEDS),
        "seed_mean_summaries": {name: {metric: float(np.mean([payload["summaries"][name][metric] for payload in payloads])) for metric in metrics} | {"molecule_count": int(payloads[0]["summaries"][name]["molecule_count"])} for name in names},
        "selected_fixed_lambda": {str(payload["seed"]): payload["fixed_lambda"]["selected_on_validation_one_to_one_rsf"] for payload in payloads},
        "foreign_mapping": {str(payload["seed"]): payload["foreign_mapping"] for payload in payloads},
        "note": "Read every seed-specific paired bootstrap before a conclusion; one-to-one is an independent confirmation, not a zero-repeat virtual-prediction result.",
    }
    (root / "aggregate_index.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    aggregate(args.aggregate_root) if args.aggregate_root else run(args)


if __name__ == "__main__":
    main()
