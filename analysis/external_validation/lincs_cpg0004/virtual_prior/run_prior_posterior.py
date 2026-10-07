#!/usr/bin/env python3
"""Fit a molecule+baseline virtual CP prior and freeze a posterior update.

The teacher is the CP-only held-out-plate model from Phase 1.  The prior sees
only a GE-derived molecular structure fingerprint, the pre-treatment CP
control vector for the held-out plate, and dose.  It never sees post-treatment
CP or GE values.  A single lambda is selected on validation and then applied
once to test:

    P0 = (1-lambda) * teacher + lambda * virtual_prior
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from torch import nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell_painting_repeat_benchmark"))
import run_cp_repeat_benchmark as B


VERSION = "cpg0004-LINCS-virtual-prior-2026-08-30"
SEEDS = B.SEEDS
RDLogger.DisableLog("rdApp.warning")


class Prior(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.Linear(512, 128), nn.GELU(), nn.Linear(128, output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--smiles", type=Path, required=True)
    p.add_argument("--outdir", type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--budget", type=int, choices=B.BUDGETS, default=1)
    p.add_argument("--dose", default="all")
    p.add_argument("--teacher-epochs", type=int, default=80)
    p.add_argument("--prior-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--null-rounds", type=int, default=32)
    p.add_argument("--bootstrap-rounds", type=int, default=2000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def fingerprint(smiles: str, n_bits: int = 2048) -> np.ndarray:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    bitvector = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=n_bits)
    result = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvector, result)
    return result


def load_smiles(path: Path) -> dict[str, np.ndarray]:
    import csv
    mapping: dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            compound = str(row["compound_id"]).strip(); smiles = str(row["smiles"]).strip()
            if compound in mapping and not np.array_equal(mapping[compound], fingerprint(smiles)):
                raise ValueError(f"Conflicting structure for {compound}")
            mapping[compound] = fingerprint(smiles)
    if not mapping:
        raise ValueError("Empty compound structure map")
    return mapping


def subset(pairs: B.Pairs, mask: np.ndarray) -> B.Pairs:
    ix = np.flatnonzero(mask).astype(np.int64)
    if not len(ix):
        raise RuntimeError("Empty pair subset")
    return B.Pairs(pairs.support[ix], pairs.target[ix], pairs.compound[ix], pairs.dose[ix], pairs.held_index[ix], pairs.held_plate[ix], pairs.held_well[ix], [pairs.support_indices[int(i)] for i in ix])


def prior_features(compounds: np.ndarray, doses: np.ndarray, baseline: np.ndarray, fingerprints: dict[str, np.ndarray]) -> np.ndarray:
    dose_values = ["0.04", "0.12", "0.37", "1.11", "3.33", "10"]
    out = np.zeros((len(compounds), 2048 + baseline.shape[1] + 6), dtype=np.float32)
    for i, (compound, dose) in enumerate(zip(compounds, doses)):
        if compound not in fingerprints:
            raise KeyError(compound)
        out[i, :2048] = fingerprints[compound]
        out[i, 2048:2048 + baseline.shape[1]] = baseline[i]
        out[i, 2048 + dose_values.index(str(dose))] = 1.0
    return out


def fit_prior(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray, args: argparse.Namespace, device: torch.device) -> tuple[Prior, dict[str, np.ndarray], dict[str, float]]:
    B.set_seed(args.seed + 101)
    xmean = train_x[:, 2048:2048 + train_y.shape[1]].mean(axis=0, dtype=np.float64).astype(np.float32)
    xscale = np.maximum(train_x[:, 2048:2048 + train_y.shape[1]].std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    ymean = train_y.mean(axis=0, dtype=np.float64).astype(np.float32)
    yscale = np.maximum(train_y.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    train_x = train_x.copy(); valid_x = valid_x.copy()
    train_x[:, 2048:2048 + train_y.shape[1]] = (train_x[:, 2048:2048 + train_y.shape[1]] - xmean) / xscale
    valid_x[:, 2048:2048 + train_y.shape[1]] = (valid_x[:, 2048:2048 + train_y.shape[1]] - xmean) / xscale
    train_y = (train_y - ymean) / yscale; valid_y = (valid_y - ymean) / yscale
    model = Prior(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(); generator.manual_seed(B.stable_seed(args.seed, "prior-loader"))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    best = None; best_value = float("inf"); best_epoch = -1
    for epoch in range(args.prior_epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True); loss = torch.mean((model(x.to(device)) - y.to(device)) ** 2)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite prior loss")
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): value = float(torch.mean((model(torch.from_numpy(valid_x).to(device)) - torch.from_numpy(valid_y).to(device)) ** 2).cpu())
        if value < best_value:
            best_value = value; best_epoch = epoch; best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"[cpg0004 prior seed={args.seed} budget={args.budget} epoch={epoch}] valid_mse={value:.7f}", flush=True)
    if best is None: raise RuntimeError("No prior checkpoint")
    model.load_state_dict(best)
    return model, {"xmean": xmean, "xscale": xscale, "ymean": ymean, "yscale": yscale}, {"best_epoch": best_epoch, "best_validation_mse": best_value}


def predict_prior(model: Prior, x: np.ndarray, stats: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    x = x.copy(); d = len(stats["xmean"]); x[:, 2048:2048 + d] = (x[:, 2048:2048 + d] - stats["xmean"]) / stats["xscale"]
    model.eval()
    with torch.no_grad(): out = model(torch.from_numpy(x).to(device)).cpu().numpy()
    return out * stats["yscale"] + stats["ymean"]


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def run(args: argparse.Namespace) -> None:
    if args.outdir is None and not args.dry_run: raise ValueError("--outdir required")
    rows = B.load_rows(args.data)
    with np.load(args.data, allow_pickle=False) as loaded:
        baseline = loaded["baseline"].astype(np.float32)
    fingerprints = load_smiles(args.smiles)
    pairs = B.make_pairs(rows, args.budget, args.dose)
    split_sets = {name: set(rows.compound[rows.split == name]) for name in ("train", "valid", "test")}
    train_pairs = B.subset_pairs(pairs, split_sets["train"]); valid_pairs = B.subset_pairs(pairs, split_sets["valid"]); test_pairs = B.subset_pairs(pairs, split_sets["test"])
    mapped_mask = np.asarray([c in fingerprints for c in test_pairs.compound], dtype=bool)
    mapped_valid_mask = np.asarray([c in fingerprints for c in valid_pairs.compound], dtype=bool)
    mapped_train_mask = np.asarray([c in fingerprints for c in train_pairs.compound], dtype=bool)
    train_rows_mask = np.asarray([c in fingerprints and s == "train" for c, s in zip(rows.compound, rows.split)], dtype=bool)
    valid_rows_mask = np.asarray([c in fingerprints and s == "valid" for c, s in zip(rows.compound, rows.split)], dtype=bool)
    if args.dry_run:
        print(json.dumps({"version": VERSION, "feature_dim": int(rows.delta.shape[1]), "fingerprint_compounds": len(fingerprints), "pair_counts": {"all": len(pairs.compound), "train": len(train_pairs.compound), "valid": len(valid_pairs.compound), "test": len(test_pairs.compound), "mapped_test": int(mapped_mask.sum())}, "dose": args.dose, "budget": args.budget}, indent=2, sort_keys=True)); return
    args.outdir.mkdir(parents=True)
    device = B.select_device(args.device)
    teacher_args = SimpleNamespace(seed=args.seed, hidden_dim=128, latent_dim=32, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, epochs=args.teacher_epochs, smoke=False, budget=args.budget)
    teacher, teacher_stats, teacher_meta = B.fit_teacher(train_pairs, valid_pairs, teacher_args, device)
    teacher_valid = B.predict(teacher, valid_pairs, teacher_stats, device); teacher_test = B.predict(teacher, test_pairs, teacher_stats, device)
    # Prior targets are all mapped plate rows in train, with validation used
    # only for checkpoint selection.  The held-out plate baseline is allowed
    # because it is measured before treatment and is present in the artifact.
    tr_ix = np.flatnonzero(train_rows_mask); va_ix = np.flatnonzero(valid_rows_mask)
    tx = prior_features(rows.compound[tr_ix], rows.dose[tr_ix], baseline[tr_ix], fingerprints); vx = prior_features(rows.compound[va_ix], rows.dose[va_ix], baseline[va_ix], fingerprints)
    prior_model, prior_stats, prior_meta = fit_prior(tx, rows.delta[tr_ix], vx, rows.delta[va_ix], args, device)
    valid_x = prior_features(valid_pairs.compound[mapped_valid_mask], valid_pairs.dose[mapped_valid_mask], baseline[valid_pairs.held_index[mapped_valid_mask]], fingerprints)
    test_x = prior_features(test_pairs.compound[mapped_mask], test_pairs.dose[mapped_mask], baseline[test_pairs.held_index[mapped_mask]], fingerprints)
    prior_valid_mapped = predict_prior(prior_model, valid_x, prior_stats, device); prior_test_mapped = predict_prior(prior_model, test_x, prior_stats, device)
    valid_mapped = subset(valid_pairs, mapped_valid_mask); test_mapped = subset(test_pairs, mapped_mask)
    # One fixed lambda, selected on mapped validation compounds only.
    lambdas = np.arange(0.0, 0.5001, 0.05)
    lambda_mse = {f"{float(lam):.2f}": mse((1 - lam) * teacher_valid[mapped_valid_mask] + lam * prior_valid_mapped, valid_mapped.target) for lam in lambdas}
    best_lambda = float(lambdas[int(np.argmin([lambda_mse[f"{float(lam):.2f}"] for lam in lambdas]))])
    posterior_test = (1 - best_lambda) * teacher_test[mapped_mask] + best_lambda * prior_test_mapped
    null_z, usable, exact = B.make_matched_null(rows, test_mapped, args.null_rounds, args.seed)
    # Prior controls use the same frozen prior checkpoint and target baseline
    # but replace the target molecule with a foreign test molecule.  The
    # matched-foreign pool preserves dose and the target condition's repeat
    # count whenever possible; the shuffled control is an independent random
    # permutation of the correct prior predictions.
    condition_repeat = {(c, d): int(np.sum((rows.compound == c) & (rows.dose == d))) for c, d in zip(rows.compound, rows.dose)}
    test_compounds = sorted(set(test_mapped.compound))
    foreign_labels_list: list[str] = []
    for compound, dose_label in zip(test_mapped.compound, test_mapped.dose):
        candidates = [x for x in test_compounds if x != compound and condition_repeat.get((x, dose_label), 0) == condition_repeat.get((compound, dose_label), 0)]
        foreign_labels_list.append(candidates[0] if candidates else next(x for x in test_compounds if x != compound))
    foreign_labels = np.asarray(foreign_labels_list, dtype=str)
    foreign_x = prior_features(foreign_labels, test_mapped.dose, baseline[test_mapped.held_index], fingerprints)
    foreign_prior = predict_prior(prior_model, foreign_x, prior_stats, device)
    rng = np.random.default_rng(B.stable_seed(args.seed, "prior-shuffle"))
    shuffled_prior = prior_test_mapped[rng.permutation(len(prior_test_mapped))]
    methods = {"support_mean": test_mapped.support, "teacher": teacher_test[mapped_mask], "virtual_prior": prior_test_mapped, "posterior": posterior_test, "shuffled_prior": shuffled_prior, "matched_foreign_prior": foreign_prior}
    summaries, vectors, raw_vectors = B.score_predictions(methods, test_mapped, null_z, usable)
    contrasts = {}
    for method in ("virtual_prior", "posterior", "shuffled_prior", "matched_foreign_prior"):
        contrasts[f"{method}_minus_teacher_excess_z"] = B.bootstrap_difference(vectors[method], vectors["teacher"], args.bootstrap_rounds, B.stable_seed(args.seed, f"boot|{method}|z"))
        contrasts[f"{method}_minus_teacher_pcc"] = B.bootstrap_difference(raw_vectors[method], raw_vectors["teacher"], args.bootstrap_rounds, B.stable_seed(args.seed, f"boot|{method}|pcc"))
    for method in ("virtual_prior", "posterior"):
        for control in ("shuffled_prior", "matched_foreign_prior"):
            contrasts[f"{method}_minus_{control}_excess_z"] = B.bootstrap_difference(vectors[method], vectors[control], args.bootstrap_rounds, B.stable_seed(args.seed, f"boot|{method}|{control}|z"))
            contrasts[f"{method}_minus_{control}_pcc"] = B.bootstrap_difference(raw_vectors[method], raw_vectors[control], args.bootstrap_rounds, B.stable_seed(args.seed, f"boot|{method}|{control}|pcc"))
    np.savez_compressed(
        args.outdir / "test_predictions.npz",
        compound=test_mapped.compound.astype(str), dose=test_mapped.dose.astype(str),
        held_index=test_mapped.held_index.astype(np.int64),
        support_mean=test_mapped.support.astype(np.float32),
        teacher=teacher_test[mapped_mask].astype(np.float32),
        virtual_prior=prior_test_mapped.astype(np.float32),
        posterior=posterior_test.astype(np.float32),
        usable=usable.astype(bool), null_z=null_z.astype(np.float64),
    )
    np.savez_compressed(
        args.outdir / "valid_predictions.npz",
        compound=valid_mapped.compound.astype(str), dose=valid_mapped.dose.astype(str),
        held_index=valid_mapped.held_index.astype(np.int64),
        support_mean=valid_mapped.support.astype(np.float32),
        teacher=teacher_valid[mapped_valid_mask].astype(np.float32),
        virtual_prior=prior_valid_mapped.astype(np.float32),
        posterior=((1 - best_lambda) * teacher_valid[mapped_valid_mask] + best_lambda * prior_valid_mapped).astype(np.float32),
        target=valid_mapped.target.astype(np.float32),
    )
    payload = {"version": VERSION, "seed": args.seed, "budget": args.budget, "dose": args.dose, "device": str(device), "data": str(args.data), "smiles": str(args.smiles), "feature_dim": int(rows.delta.shape[1]), "fingerprint_compound_count": len(fingerprints), "mapped_pair_counts": {"train": int(mapped_train_mask.sum()), "valid": int(mapped_valid_mask.sum()), "test": int(mapped_mask.sum()), "test_row_matched_null_usable": int(usable.sum()), "test_exact_well_null_usable": int(exact.sum())}, "teacher": teacher_meta, "prior": prior_meta, "lambda_grid": [float(x) for x in lambdas], "validation_lambda_mse": lambda_mse, "selected_lambda": best_lambda, "locked_hyperparameters": {"teacher_epochs": args.teacher_epochs, "prior_epochs": args.prior_epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "null_rounds": args.null_rounds, "bootstrap_rounds": args.bootstrap_rounds, "prior_input": "ECFP4-2048 + held-plate pre-treatment CP control median + six-dose one-hot"}, "controls": {"foreign_selection": "same test split, same CP condition repeat count when available, same target dose/baseline", "shuffled_definition": "random permutation of correct prior predictions within mapped test pairs"}, "summaries": summaries, "paired_bootstrap": contrasts, "definition": "P0=(1-lambda) teacher + lambda virtual prior; lambda is selected on mapped validation pairs only and frozen for test. Prior has no post-treatment CP or GE input.", "guardrails": ["compound-level locked split", "teacher and prior fit only training compounds", "held-out baseline is pre-treatment control only", "GE treatment profiles are not read by this script", "test used once after lambda selection", "row-matched foreign null and exact-well availability are recorded"]}
    (args.outdir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    run(parse_args())
