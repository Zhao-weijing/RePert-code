#!/usr/bin/env python3
"""Frozen 1R CP posterior updated by observed-GE residual evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import torch

import repro_effect_virtual_cp_mvp as C
import repro_effect_prior_posterior as B
import repro_effect_prior_posterior_controls as S
import repro_effect_prior_posterior_one_to_one as O


VERSION = "BBBC047-1R-GE-Residual-Evidence-2026-08-29"
BETA_GRID = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
PROFILE_RAW_ROUNDTRIP_ATOL = 5e-4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cp-rows-root", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"))
    p.add_argument("--aggregate-npz", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0/molecule_aggregates.npz"))
    p.add_argument("--model-h5", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/Paired_CP_GE_PlateMedian_v1_model_compat.h5"))
    p.add_argument("--split-lock", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json"))
    p.add_argument("--posterior-root", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/repro_effect_prior_posterior_one_to_one_20260829"))
    p.add_argument("--d2-test-root", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/cm_d2b_ge_to_cp_20260828/true"))
    p.add_argument("--d2-valid-root", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/cm_d2b_ge_to_cp_validprobe_20260828/true"))
    p.add_argument("--outdir", type=Path)
    p.add_argument("--aggregate-root", type=Path)
    p.add_argument("--seed", type=int, choices=C.SEEDS)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--null-rounds", type=int, default=32)
    p.add_argument("--bootstrap-rounds", type=int, default=10000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--fixed-beta", type=float, choices=BETA_GRID, help="Common beta selected from the mean three-seed validation curve")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.aggregate_root:
        if a.seed is not None or a.outdir is not None or a.dry_run:
            p.error("--aggregate-root cannot be combined with run options")
    elif a.seed is None:
        p.error("--seed is required")
    elif not a.dry_run and a.outdir is None:
        p.error("--outdir is required")
    return a


def decode(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]


def locate_profile(root: Path, seed: int, valid: bool) -> Path:
    middle = "cmd2b_true_observed_ge_validprobe" if valid else "cmd2b_true_observed_ge"
    expected = f"MVC_{middle}_seed{seed}_BBBC047_smiles_split"
    found = list(root.glob(f"*/{expected}/ECFP4_Default/predict/test_prediction_profile.h5"))
    if len(found) != 1:
        raise RuntimeError(f"Expected one D2 profile for {expected}; found {found}")
    return found[0]


def cp_normalization(model_h5: Path, split_lock: Path) -> dict[str, np.ndarray]:
    lock = json.loads(split_lock.read_text(encoding="utf-8"))
    with h5py.File(model_h5, "r") as h:
        all_smiles = decode(h["canonical_smiles"][:]); index = {s: i for i, s in enumerate(all_smiles)}
        rows = np.sort(np.asarray([index[str(s)] for s in lock["train_smiles"]], dtype=np.int64))
        control, target = h["control_CP"][rows].astype(np.float64), h["target_CP"][rows].astype(np.float64)
    output = {"control_mean": control.mean(0), "control_std": control.std(0), "target_mean": target.mean(0), "target_std": target.std(0)}
    output["control_std"][output["control_std"] < 1e-8] = 1.0
    output["target_std"][output["target_std"] < 1e-8] = 1.0
    return output


def load_g(path: Path, order: list[str], stats: dict[str, np.ndarray], expected_target: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h:
        required = ("smiles", "control_cp", "target_cp", "cp_pred")
        missing = [key for key in required if key not in h]
        if missing: raise KeyError(f"{path} missing {missing}")
        smiles = decode(h["smiles"][:]); index = {s: i for i, s in enumerate(smiles)}
        if not set(order).issubset(index) or len(index) != len(smiles):
            raise ValueError(f"D2 membership mismatch: {path}")
        rows = np.asarray([index[s] for s in order], dtype=np.int64)
        control_n, target_n, pred_n = h["control_cp"][:][rows], h["target_cp"][:][rows], h["cp_pred"][:][rows]
    control = control_n * stats["control_std"] + stats["control_mean"]
    target = target_n * stats["target_std"] + stats["target_mean"]
    pred = pred_n * stats["target_std"] + stats["target_mean"]
    target_delta = target - control
    expected = np.vstack([expected_target[s] for s in order])
    error = float(np.max(np.abs(target_delta - expected)))
    if error > PROFILE_RAW_ROUNDTRIP_ATOL:
        raise ValueError(f"D2 raw target roundtrip mismatch {error:.8g}: {path}")
    return {s: (pred[i] - control[i]).astype(np.float32) for i, s in enumerate(order)}


def load_states(root: Path, seed: int, device: torch.device) -> tuple[list[B.TeacherState], Any, dict[str, np.ndarray], float, dict[str, Any]]:
    run = root / f"seed{seed}"
    teacher_payload = torch.load(run / "one_to_one_teacher_states.pt", map_location="cpu")
    states: list[B.TeacherState] = []
    for record in teacher_payload["teacher_states"]:
        model = C.BASE.CrossFitEffectNet(C.CP_DIM, 256, 32).to(device)
        model.load_state_dict(record["state_dict"])
        states.append(B.TeacherState(model, np.asarray(record["mean"], dtype=np.float32), np.asarray(record["scale"], dtype=np.float32)))
    prior_payload = torch.load(run / "c0_train_only.pt", map_location="cpu")
    prior_model = C.Student().to(device); prior_model.load_state_dict(prior_payload["state_dict"])
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    lam = float(metrics["fixed_lambda"]["selected_on_validation_one_to_one_rsf"])
    return states, prior_model, prior_payload["stats"], lam, metrics


def teacher_prediction(states: list[B.TeacherState], pairs: O.OneToOnePairs, batch: int, device: torch.device) -> np.ndarray:
    evidence = O.as_evidence(pairs)
    return np.mean(np.stack([C.effect_array(s.model, B.as_loo(evidence), s.mean, s.scale, batch, device) for s in states]), axis=0).astype(np.float32)


def p0_and_residual(pairs: O.OneToOnePairs, teacher: np.ndarray, prior: dict[str, np.ndarray], g: dict[str, np.ndarray], lam: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    v = np.vstack([prior[s] for s in pairs.smiles])
    p0 = ((1.0 - lam) * teacher + lam * v).astype(np.float32)
    residual = {s: (g[s] - prior[s]).astype(np.float32) for s in pairs.smiles}
    return p0, residual


def prediction_map(pairs: O.OneToOnePairs, p0: np.ndarray, residual: dict[str, np.ndarray], beta: float, donors: dict[str, str] | None = None) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for i, (held, smiles) in enumerate(zip(pairs.held_rows.tolist(), pairs.smiles)):
        donor = smiles if donors is None else donors.get(smiles)
        if donor is not None:
            output[int(held)] = (p0[i] + beta * residual[donor]).astype(np.float32)
    return output


def derangement(keys: list[str], seed: int, label: str) -> dict[str, str]:
    ordered = sorted(keys); rng = np.random.default_rng(C.BASE.stable_seed(seed, label)); rng.shuffle(ordered)
    donors = ordered[1:] + ordered[:1]
    mapping = dict(zip(ordered, donors))
    if any(key == donor for key, donor in mapping.items()): raise RuntimeError("Derangement has a fixed point")
    return mapping


def quartile(values: np.ndarray) -> np.ndarray:
    cuts = np.quantile(values.astype(np.float64), (0.25, 0.5, 0.75))
    return np.searchsorted(cuts, values, side="right").astype(np.int64)


def ge_metadata(model_h5: Path, ge_rows_path: Path, smiles: list[str]) -> dict[str, dict[str, Any]]:
    with h5py.File(model_h5, "r") as h:
        all_smiles = decode(h["canonical_smiles"][:]); index = {s: i for i, s in enumerate(all_smiles)}
        rows = np.asarray([index[s] for s in smiles], dtype=np.int64)
        delta = h["target_GE"][:][rows].astype(np.float64) - h["control_GE"][:][rows].astype(np.float64)
    rms = np.sqrt(np.mean(delta * delta, axis=1))
    ge = np.load(ge_rows_path, allow_pickle=False)
    grouped: dict[str, list[int]] = {}
    for row, value in enumerate(decode(ge["smiles"])): grouped.setdefault(value, []).append(row)
    max_dose = np.asarray([np.max(ge["dose"][grouped[s]]) for s in smiles], dtype=np.float64)
    dose_q, rms_q = quartile(max_dose), quartile(rms)
    output: dict[str, dict[str, Any]] = {}
    for i, s in enumerate(smiles):
        ix = sorted(grouped[s], key=lambda row: (str(ge["plate"][row]), row))
        plates = tuple(str(ge["plate"][row]) for row in ix)
        doses = tuple(f"{float(ge['dose'][row]):.8g}" for row in ix)
        output[s] = {"ge_plate_set": plates, "ge_dose_signature": tuple(zip(plates, doses)), "ge_max_dose": float(max_dose[i]), "ge_max_dose_quartile": int(dose_q[i]), "ge_effect_rms": float(rms[i]), "ge_effect_rms_quartile": int(rms_q[i])}
    return output


def matched_foreign(pairs: O.OneToOnePairs, rows: Any, metadata: dict[str, dict[str, Any]], seed: int, exact_dose: bool) -> tuple[dict[str, str], list[dict[str, Any]]]:
    mapping: dict[str, str] = {}; manifest: list[dict[str, Any]] = []
    available = set(metadata)
    for i, smiles in enumerate(pairs.smiles):
        held, support = int(pairs.held_rows[i]), int(pairs.support_rows[i])
        hp, sp = str(rows.plate[held]), str(rows.plate[support])
        candidates = sorted((rows.by_plate[hp] & rows.by_plate[sp] & available) - {smiles})
        matched = [d for d in candidates if metadata[d]["ge_plate_set"] == metadata[smiles]["ge_plate_set"] and metadata[d]["ge_max_dose_quartile"] == metadata[smiles]["ge_max_dose_quartile"] and metadata[d]["ge_effect_rms_quartile"] == metadata[smiles]["ge_effect_rms_quartile"]]
        if exact_dose:
            matched = [d for d in matched if metadata[d]["ge_dose_signature"] == metadata[smiles]["ge_dose_signature"]]
        donor = None
        if matched:
            donor = matched[C.BASE.stable_seed(seed, f"ge-matched-foreign|{exact_dose}|{smiles}|{hp}|{sp}") % len(matched)]
            mapping[smiles] = donor
        manifest.append({"canonical_smiles": smiles, "held_plate": hp, "support_plate": sp, "slot_candidate_count": len(candidates), "matched_candidate_count": len(matched), "donor_canonical_smiles": donor or "", "eligible": int(donor is not None), "ge_plate_set": "|".join(metadata[smiles]["ge_plate_set"]), "ge_max_dose": metadata[smiles]["ge_max_dose"], "ge_max_dose_quartile": metadata[smiles]["ge_max_dose_quartile"], "ge_effect_rms": metadata[smiles]["ge_effect_rms"], "ge_effect_rms_quartile": metadata[smiles]["ge_effect_rms_quartile"], "exact_per_plate_dose": int(exact_dose)})
    return mapping, manifest


def select_beta(pairs: O.OneToOnePairs, p0: np.ndarray, residual: dict[str, np.ndarray], reference: dict[str, Any], rows: Any) -> tuple[float, dict[str, Any]]:
    curve: dict[str, Any] = {}
    for beta in BETA_GRID:
        scores = C.BASE.method_stats(prediction_map(pairs, p0, residual, beta), reference, rows)
        curve[str(beta)] = C.BASE.summarize(scores)
    selected = max(BETA_GRID, key=lambda beta: (curve[str(beta)]["rsf"], -beta))
    return selected, curve


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    if not args.dry_run and args.outdir.exists(): raise FileExistsError(args.outdir)
    cp = C.BASE.load_splits(args.cp_rows_root, args.split_lock)
    virtual = C.load_virtual(args.model_h5, args.aggregate_npz, args.split_lock)
    pairs = {split: O.one_to_one_pairs(cp[split], split)[0] for split in ("valid", "test")}
    profiles = {"valid": locate_profile(args.d2_valid_root, args.seed, True), "test": locate_profile(args.d2_test_root, args.seed, False)}
    state_dir = args.posterior_root / f"seed{args.seed}"
    for path in (*profiles.values(), state_dir / "one_to_one_teacher_states.pt", state_dir / "c0_train_only.pt", state_dir / "metrics.json"):
        if not path.is_file(): raise FileNotFoundError(path)
    metadata = {split: ge_metadata(args.model_h5, args.cp_rows_root / f"{split}_ge_plate_rows.npz", pairs[split].smiles) for split in ("valid", "test")}
    matched = {split: matched_foreign(pairs[split], cp[split], metadata[split], args.seed, False) for split in ("valid", "test")}
    exact_dose = {split: matched_foreign(pairs[split], cp[split], metadata[split], args.seed, True) for split in ("valid", "test")}
    if args.dry_run:
        print(json.dumps({"version": VERSION, "seed": args.seed, "beta_grid": BETA_GRID, "pair_counts": {k: len(v.smiles) for k, v in pairs.items()}, "matched_foreign_eligible": {k: len(v[0]) for k, v in matched.items()}, "exact_dose_foreign_eligible": {k: len(v[0]) for k, v in exact_dose.items()}, "profiles": {k: str(v) for k, v in profiles.items()}}, indent=2))
        return
    device = C.select_device(args.device); stats = cp_normalization(args.model_h5, args.split_lock)
    states, prior_model, prior_stats, lam, frozen_metrics = load_states(args.posterior_root, args.seed, device)
    prior = {split: C.predict_student(prior_model, virtual[split], prior_stats, args.batch_size, device) for split in ("valid", "test")}
    teacher = {split: teacher_prediction(states, pairs[split], args.batch_size, device) for split in ("valid", "test")}
    # D2 target identity is checked against the official aggregate CP delta.
    expected = {split: {s: virtual[split].delta[i] for i, s in enumerate(virtual[split].smiles)} for split in ("valid", "test")}
    g = {split: load_g(profiles[split], pairs[split].smiles, stats, expected[split]) for split in ("valid", "test")}
    values = {split: p0_and_residual(pairs[split], teacher[split], prior[split], g[split], lam) for split in ("valid", "test")}
    valid_reference = O.selected_reference_stats(cp["valid"], pairs["valid"], args.null_rounds, args.seed)
    per_seed_beta, curve = select_beta(pairs["valid"], values["valid"][0], values["valid"][1], valid_reference, cp["valid"])
    beta = per_seed_beta if args.fixed_beta is None else float(args.fixed_beta)
    test_reference = O.selected_reference_stats(cp["test"], pairs["test"], args.null_rounds, args.seed)
    p0, residual = values["test"]
    shuffled = derangement(pairs["test"].smiles, args.seed, "ge-residual-shuffled-test")
    foreign, foreign_manifest = matched["test"]
    exact_foreign, exact_foreign_manifest = exact_dose["test"]
    methods = {
        "p0": prediction_map(pairs["test"], p0, residual, 0.0),
        "correct_ge": prediction_map(pairs["test"], p0, residual, beta),
        "shuffled_ge": prediction_map(pairs["test"], p0, residual, beta, shuffled),
        "matched_foreign_ge": prediction_map(pairs["test"], p0, residual, beta, foreign),
        "exact_dose_foreign_ge": prediction_map(pairs["test"], p0, residual, beta, exact_foreign),
    }
    common_foreign = {s: record for s, record in test_reference.items() if s in foreign}
    common_exact = {s: record for s, record in test_reference.items() if s in exact_foreign}
    full = {name: S.score_bundle(pred, test_reference, cp["test"]) for name, pred in methods.items() if name not in ("matched_foreign_ge", "exact_dose_foreign_ge")}
    foreign_scores = {name: S.score_bundle(pred, common_foreign, cp["test"]) for name, pred in methods.items() if name != "exact_dose_foreign_ge"}
    exact_scores = {name: S.score_bundle(pred, common_exact, cp["test"]) for name, pred in methods.items() if name != "matched_foreign_ge"}
    ns = SimpleNamespace(bootstrap_rounds=args.bootstrap_rounds, seed=args.seed)
    contrasts = {
        "correct_minus_p0": S.contrast(full["correct_ge"][0], full["p0"][0], full["correct_ge"][1], full["p0"][1], ns, "ge-correct-minus-p0"),
        "correct_minus_shuffled": S.contrast(full["correct_ge"][0], full["shuffled_ge"][0], full["correct_ge"][1], full["shuffled_ge"][1], ns, "ge-correct-minus-shuffled"),
        "correct_minus_matched_foreign_common": S.contrast(foreign_scores["correct_ge"][0], foreign_scores["matched_foreign_ge"][0], foreign_scores["correct_ge"][1], foreign_scores["matched_foreign_ge"][1], ns, "ge-correct-minus-matched-foreign"),
        "correct_minus_exact_dose_foreign_common": S.contrast(exact_scores["correct_ge"][0], exact_scores["exact_dose_foreign_ge"][0], exact_scores["correct_ge"][1], exact_scores["exact_dose_foreign_ge"][1], ns, "ge-correct-minus-exact-dose-foreign"),
    }
    args.outdir.mkdir(parents=True)
    C.write_scores(args.outdir / "per_molecule_scores.csv", {name: result[0] for name, result in full.items()})
    C.write_scores(args.outdir / "matched_foreign_common_scores.csv", {name: result[0] for name, result in foreign_scores.items()})
    C.write_scores(args.outdir / "exact_dose_foreign_common_scores.csv", {name: result[0] for name, result in exact_scores.items()})
    write_csv(args.outdir / "matched_foreign_mapping.csv", foreign_manifest)
    write_csv(args.outdir / "exact_dose_foreign_mapping.csv", exact_foreign_manifest)
    write_csv(args.outdir / "shuffled_mapping.csv", [{"canonical_smiles": s, "donor_canonical_smiles": d} for s, d in sorted(shuffled.items())])
    payload = {"version": VERSION, "seed": args.seed, "device": str(device), "frozen_cp_lambda": lam, "selected_beta": beta, "per_seed_validation_argmax_beta": per_seed_beta, "common_beta_override": args.fixed_beta, "beta_grid": BETA_GRID, "validation_beta_curve": curve, "summaries": {name: value[2] for name, value in full.items()}, "matched_foreign_common_summaries": {name: value[2] for name, value in foreign_scores.items()}, "exact_dose_foreign_common_summaries": {name: value[2] for name, value in exact_scores.items()}, "contrasts": contrasts, "matched_foreign": {"eligible": len(foreign), "test_pairs": len(pairs["test"].smiles), "reference_common": len(common_foreign), "rule": "exact held/support CP slots and observed-GE plate set; same GE max-dose and effect-RMS quartiles"}, "exact_dose_foreign": {"eligible": len(exact_foreign), "reference_common": len(common_exact), "rule": "primary matched rule plus identical per-GE-plate maximum-dose signature"}, "guards": ["frozen one-to-one CP teacher and C0 prior", "same C0 prior defines P0 and G-V", "validation alone selects beta", "controls reuse selected beta", "test is evaluated once"], "frozen_cp_version": frozen_metrics.get("version")}
    (args.outdir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    records = []
    for seed in C.SEEDS:
        path = args.aggregate_root / f"seed{seed}" / "metrics.json"
        if not path.is_file(): raise FileNotFoundError(path)
        records.append(json.loads(path.read_text(encoding="utf-8")))
    beta = [record["selected_beta"] for record in records]
    def pooled(label: str, common_file: str) -> dict[str, Any]:
        per_seed = []
        left_name, right_name = {"correct_minus_p0": ("correct_ge", "p0"), "correct_minus_shuffled": ("correct_ge", "shuffled_ge"), "correct_minus_matched_foreign_common": ("correct_ge", "matched_foreign_ge"), "correct_minus_exact_dose_foreign_common": ("correct_ge", "exact_dose_foreign_ge")}[label]
        for seed in C.SEEDS:
            rows: dict[str, dict[str, float]] = {}
            with (args.aggregate_root / f"seed{seed}" / common_file).open(newline="", encoding="utf-8") as h:
                for row in csv.DictReader(h): rows.setdefault(row["method"], {})[row["canonical_smiles"]] = float(row["model_excess_z"])
            common = sorted(set(rows[left_name]) & set(rows[right_name])); per_seed.append({s: rows[left_name][s] - rows[right_name][s] for s in common})
        common = sorted(set.intersection(*(set(values) for values in per_seed)))
        diff = np.asarray([np.mean([values[s] for values in per_seed]) for s in common])
        rng = np.random.default_rng(C.BASE.stable_seed(20260829, f"pooled|{label}")); boot = np.empty(10000)
        for begin in range(0, 10000, 100):
            draw = rng.integers(0, len(diff), size=(100, len(diff))); boot[begin:begin+100] = diff[draw].mean(1)
        return {"molecule_count": len(common), "mean_excess_z_gain": float(diff.mean()), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    pooled_values = {"correct_minus_p0": pooled("correct_minus_p0", "per_molecule_scores.csv"), "correct_minus_shuffled": pooled("correct_minus_shuffled", "per_molecule_scores.csv"), "correct_minus_matched_foreign_common": pooled("correct_minus_matched_foreign_common", "matched_foreign_common_scores.csv"), "correct_minus_exact_dose_foreign_common": pooled("correct_minus_exact_dose_foreign_common", "exact_dose_foreign_common_scores.csv")}
    seed_directions = [record["contrasts"]["correct_minus_p0"]["excess_z"]["candidate_minus_reference_excess_z"] for record in records]
    go = all(value > 0 for value in seed_directions) and pooled_values["correct_minus_p0"]["ci95"][0] > 0 and pooled_values["correct_minus_shuffled"]["ci95"][0] > 0
    payload = {"version": VERSION, "selected_beta_by_seed": dict(zip(map(str, C.SEEDS), beta)), "correct_minus_p0_by_seed": dict(zip(map(str, C.SEEDS), seed_directions)), "pooled": pooled_values, "decision": "GO" if go else "NO_GO", "stop_condition_all_beta_zero": all(value == 0 for value in beta), "matched_foreign_supporting_pass": pooled_values["correct_minus_matched_foreign_common"]["ci95"][0] > 0}
    (args.aggregate_root / "aggregate_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    arguments = parse_args()
    aggregate(arguments) if arguments.aggregate_root else run(arguments)
