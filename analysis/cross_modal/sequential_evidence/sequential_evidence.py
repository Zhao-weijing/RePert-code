#!/usr/bin/env python3
"""Evaluate CP1/CP2/GE evidence accumulation on one fixed CP held plate.

The common query set is the already locked same-held-plate cross-budget set.
P1 and P2 use frozen CP teacher-plus-prior states; GE is the frozen observed-GE
residual update.  No model is fitted or selected on the final target here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve()
CB_DIR = HERE.parent.parent.parent / "MVCPert_5_27" / "experiments" / "bbbc047_cross_budget_1r_ai_vs_2r_mean"
if not CB_DIR.exists():
    CB_DIR = Path("/path/to/data/AIDD/MVCPert_5_27/analysis/bbbc047_cross_budget_1r_ai_vs_2r_mean")
sys.path.insert(0, str(CB_DIR))
import evaluate_cross_budget as X  # noqa: E402


VERSION = "BBBC047-Sequential-CP1-CP2-GE-2026-08-30"
SEEDS = X.C.SEEDS
PAIR_MANIFEST_SEED = X.PAIR_MANIFEST_SEED
GE_BETA = 0.3
CP2_LAMBDA = 0.25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cp-rows-root", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"))
    p.add_argument("--aggregate-npz", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0/molecule_aggregates.npz"))
    p.add_argument("--model-h5", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/Paired_CP_GE_PlateMedian_v1_model_compat.h5"))
    p.add_argument("--split-lock", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json"))
    p.add_argument("--posterior-root", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/repro_effect_prior_posterior_one_to_one_20260829"))
    p.add_argument("--information-budget-root", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/2r3r_ge_residual_evidence_20260829"))
    p.add_argument("--d2-valid-root", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/cm_d2b_ge_to_cp_validprobe_20260828/true"))
    p.add_argument("--d2-test-root", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/cm_d2b_ge_to_cp_20260828/true"))
    p.add_argument("--outdir", type=Path)
    p.add_argument("--aggregate-root", type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--null-rounds", type=int, default=32)
    p.add_argument("--bootstrap-rounds", type=int, default=10000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.aggregate_root:
        if a.seed is not None or a.outdir is not None or a.dry_run:
            p.error("--aggregate-root cannot be combined with run options")
    elif not a.dry_run and (a.seed is None or a.outdir is None):
        p.error("--seed and --outdir are required")
    return a


def load_budget2_states(root: Path, seed: int, device: torch.device) -> list[Any]:
    path = root / f"seed{seed}" / "budget2_teacher_states.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    states = []
    for record in payload["teacher_states"]:
        model = X.C.BASE.CrossFitEffectNet(X.C.CP_DIM, 256, 32).to(device)
        model.load_state_dict(record["state_dict"])
        states.append(X.B.TeacherState(model, np.asarray(record["mean"], dtype=np.float32), np.asarray(record["scale"], dtype=np.float32)))
    return states


def teacher_on_support(states: list[Any], support: np.ndarray, target: np.ndarray, smiles: list[str], held_rows: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    evidence = X.B.EvidencePairs(support, np.zeros_like(support), target, smiles, held_rows)
    return np.mean(np.stack([X.C.effect_array(state.model, X.B.as_loo(evidence), state.mean, state.scale, batch_size, device) for state in states]), axis=0).astype(np.float32)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def flatten_contrast(name: str, value: dict[str, Any], scope: str, seed: Any) -> dict[str, Any]:
    excess = value["excess_z"]
    pcc = value["delta_pcc"]
    return {
        "scope": scope,
        "seed": seed,
        "contrast": name,
        "molecule_count": excess["molecule_count"],
        "excess_z_point": excess["candidate_minus_reference_excess_z"],
        "excess_z_ci95": json.dumps(excess["candidate_minus_reference_excess_z_ci95"]),
        "held_delta_pcc_point": pcc["candidate_minus_reference_delta_pcc"],
        "held_delta_pcc_ci95": json.dumps(pcc["candidate_minus_reference_delta_pcc_ci95"]),
    }


def run_one(args: argparse.Namespace) -> None:
    if args.outdir is None or args.seed is None:
        raise ValueError("--seed and --outdir are required")
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    args.outdir.mkdir(parents=True)
    cp = X.C.BASE.load_splits(args.cp_rows_root, args.split_lock)
    virtual = X.C.load_virtual(args.model_h5, args.aggregate_npz, args.split_lock)
    pair_data = {split: X.cross_pairs(cp[split], split) for split in ("valid", "test")}
    pairs = {split: pair_data[split][0] for split in pair_data}
    device = X.C.select_device(args.device)
    one_states, prior_model, prior_stats, lambda1, _ = X.E.load_states(args.posterior_root, args.seed, device)
    two_states = load_budget2_states(args.information_budget_root, args.seed, device)
    prior = {split: X.C.predict_student(prior_model, virtual[split], prior_stats, args.batch_size, device) for split in ("valid", "test")}
    teacher1 = {split: X.teacher_on_support(one_states, pairs[split], args.batch_size, device) for split in pairs}
    support2 = {split: ((pairs[split].support1 + pairs[split].support2) / 2.0).astype(np.float32) for split in pairs}
    teacher2 = {split: teacher_on_support(two_states, support2[split], pairs[split].target, pairs[split].smiles, pairs[split].held_rows, args.batch_size, device) for split in pairs}
    stats = X.E.cp_normalization(args.model_h5, args.split_lock)
    expected = {split: {s: virtual[split].delta[i] for i, s in enumerate(virtual[split].smiles)} for split in ("valid", "test")}
    profiles = {"valid": X.E.locate_profile(args.d2_valid_root, args.seed, True), "test": X.E.locate_profile(args.d2_test_root, args.seed, False)}
    g = {split: X.E.load_g(profiles[split], pairs[split].smiles, stats, expected[split]) for split in pairs}
    methods = {}
    references = {}
    scores = {}
    for split in ("valid", "test"):
        bundle = pairs[split]
        v = np.vstack([prior[split][s] for s in bundle.smiles]).astype(np.float32)
        p1 = ((1.0 - lambda1) * teacher1[split] + lambda1 * v).astype(np.float32)
        p2 = ((1.0 - CP2_LAMBDA) * teacher2[split] + CP2_LAMBDA * v).astype(np.float32)
        ge_residual = np.vstack([g[split][s] for s in bundle.smiles]).astype(np.float32) - v
        methods[split] = {
            "CP1_raw": {int(h): bundle.support1[i] for i, h in enumerate(bundle.held_rows.tolist())},
            "P1": {int(h): p1[i] for i, h in enumerate(bundle.held_rows.tolist())},
            "P2": {int(h): p2[i] for i, h in enumerate(bundle.held_rows.tolist())},
            "P1_GE": {int(h): (p1[i] + GE_BETA * ge_residual[i]).astype(np.float32) for i, h in enumerate(bundle.held_rows.tolist())},
            "P2_GE": {int(h): (p2[i] + GE_BETA * ge_residual[i]).astype(np.float32) for i, h in enumerate(bundle.held_rows.tolist())},
        }
        references[split] = X.common_reference(cp[split], bundle, args.null_rounds, args.seed)
        scores[split] = {name: X.S.score_bundle(pred, references[split], cp[split]) for name, pred in methods[split].items()}
    test_scores = scores["test"]
    ns = SimpleNamespace(bootstrap_rounds=args.bootstrap_rounds, seed=args.seed)
    contrast_specs = (
        ("CP2|CP1", "P2", "P1"),
        ("GE|CP1", "P1_GE", "P1"),
        ("GE|CP1+CP2", "P2_GE", "P2"),
        ("CP2|CP1+GE", "P2_GE", "P1_GE"),
    )
    contrasts = {}
    for name, left, right in contrast_specs:
        contrasts[name] = X.S.contrast(test_scores[left][0], test_scores[right][0], test_scores[left][1], test_scores[right][1], ns, f"sequential-{name}")
    rows = []
    for method, scored in test_scores.items():
        for compound, record in scored[0].items():
            rows.append({"method": method, "canonical_smiles": compound, "model_excess_z": record["model_excess_z"], "model_z": record["model_z"], "held_delta_pcc": float(np.tanh(record["model_z"])), "replicate_excess_z": record["replicate_excess_z"]})
    write_csv(args.outdir / "per_molecule_scores.csv", rows)
    write_csv(args.outdir / "pair_manifest.csv", pair_data["test"][1])
    write_csv(args.outdir / "conditional_marginal_gain.csv", [flatten_contrast(name, value, "seed", args.seed) for name, value in contrasts.items()])
    write_csv(args.outdir / "evidence_trajectory.csv", [{"scope": "seed", "seed": args.seed, "stage": method, **summary} for method, summary in ((name, value[2]) for name, value in test_scores.items())])
    payload = {
        "version": VERSION,
        "seed": args.seed,
        "device": str(device),
        "pair_manifest_seed": PAIR_MANIFEST_SEED,
        "pair_counts": {split: len(pairs[split].smiles) for split in pairs},
        "common_reference_test_molecule_count": len(references["test"]),
        "lambda1": lambda1,
        "lambda2": CP2_LAMBDA,
        "beta_ge": GE_BETA,
        "contrasts": contrasts,
        "guards": ["same held CP row and same CP1/CP2 support slots for every method", "P1/P2 teachers and prior are frozen artifacts", "GE profile is frozen observed-GE evidence", "no final-test value selects a method or hyperparameter", "this is an evidence-accumulation analysis, not a replicate-replacement claim"],
    }
    (args.outdir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def read_method_scores(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.setdefault(row["method"], {})[row["canonical_smiles"]] = {"model_excess_z": float(row["model_excess_z"]), "held_delta_pcc": float(row["held_delta_pcc"])}
    return output


def pooled_contrast(root: Path, left: str, right: str, rounds: int, label: str) -> dict[str, Any]:
    tables = [read_method_scores(root / f"seed{seed}" / "per_molecule_scores.csv") for seed in SEEDS]
    common = set(tables[0][left]) & set(tables[0][right])
    for table in tables[1:]:
        common &= set(table[left]) & set(table[right])
    common = sorted(common)
    diff_z = np.asarray([np.mean([table[left][c]["model_excess_z"] - table[right][c]["model_excess_z"] for table in tables]) for c in common])
    diff_pcc = np.asarray([np.mean([table[left][c]["held_delta_pcc"] - table[right][c]["held_delta_pcc"] for table in tables]) for c in common])
    rng = np.random.default_rng(X.C.BASE.stable_seed(20260830, f"sequential-pooled|{label}"))
    boot_z = np.empty(rounds); boot_pcc = np.empty(rounds)
    for begin in range(0, rounds, 100):
        end = min(rounds, begin + 100); draw = rng.integers(0, len(common), size=(end - begin, len(common)))
        boot_z[begin:end] = diff_z[draw].mean(axis=1); boot_pcc[begin:end] = diff_pcc[draw].mean(axis=1)
    return {"scope": "pooled_three_seed", "contrast": label, "molecule_count": len(common), "excess_z_point": float(diff_z.mean()), "excess_z_ci95": [float(np.quantile(boot_z, .025)), float(np.quantile(boot_z, .975))], "held_delta_pcc_point": float(diff_pcc.mean()), "held_delta_pcc_ci95": [float(np.quantile(boot_pcc, .025)), float(np.quantile(boot_pcc, .975))], "bootstrap_rounds": rounds}


def aggregate(args: argparse.Namespace) -> None:
    if args.aggregate_root is None:
        raise ValueError("--aggregate-root is required")
    root = args.aggregate_root
    trajectory = []
    gains = []
    for seed in SEEDS:
        payload = json.loads((root / f"seed{seed}" / "metrics.json").read_text(encoding="utf-8"))
        for name, value in payload["contrasts"].items():
            gains.append(flatten_contrast(name, value, "seed", seed))
        with (root / f"seed{seed}" / "evidence_trajectory.csv").open(newline="", encoding="utf-8") as handle:
            trajectory.extend(dict(row) for row in csv.DictReader(handle))
    for label, left, right in (("CP2|CP1", "P2", "P1"), ("GE|CP1", "P1_GE", "P1"), ("GE|CP1+CP2", "P2_GE", "P2"), ("CP2|CP1+GE", "P2_GE", "P1_GE")):
        gains.append(pooled_contrast(root, left, right, args.bootstrap_rounds or 10000, label))
    write_csv(root / "evidence_trajectory.csv", trajectory)
    write_csv(root / "conditional_marginal_gain.csv", gains)
    payload = {"version": VERSION, "seeds": list(SEEDS), "evidence_trajectory": str(root / "evidence_trajectory.csv"), "conditional_marginal_gain": str(root / "conditional_marginal_gain.csv"), "guard": "same-held-CP, same-compound sequential evidence analysis; no replicate-replacement claim"}
    (root / "aggregate_index.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def dry_run(args: argparse.Namespace) -> None:
    cp = X.C.BASE.load_splits(args.cp_rows_root, args.split_lock)
    pair_data = {split: X.cross_pairs(cp[split], split)[0] for split in ("valid", "test")}
    print(json.dumps({"version": VERSION, "pair_counts": {split: len(bundle.smiles) for split, bundle in pair_data.items()}, "guard": "dry-run only; fixed same-held CP pair set"}, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.aggregate_root:
        aggregate(args)
    elif args.dry_run:
        dry_run(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
