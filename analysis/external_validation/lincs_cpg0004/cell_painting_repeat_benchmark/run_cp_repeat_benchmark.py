#!/usr/bin/env python3
"""CP-only held-out-plate repeat benchmark for the cpg0004 external audit.

The independent repeat unit is a plate.  A condition is a compound at one of
the six standard doses.  For budget R, the first R plates (stable plate order)
are support and the next plate is held out, with every plate used once as the
held-out target whenever at least R+1 plates exist.  The same target/support
definition is used for the raw support mean and the learned teacher.

The strict primary score is the mean, over test compounds, of Fisher-z
held-plate PCC minus a foreign null matched on dose, held/support plate slots,
and the well row.  The LINCS Pilot 1 layout assigns one compound to each exact
well slot across plates, so an exact-well foreign null is structurally
impossible; the audit records that zero-availability fact and uses the
strongest feasible row-matched null.  The null is built only from test
compounds and is never used for fitting or checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


VERSION = "cpg0004-LINCS-CP-repeat-benchmark-2026-08-30"
SEEDS = (3407, 42, 2025)
BUDGETS = (1, 2, 3)


@dataclass
class Rows:
    compound: np.ndarray
    dose: np.ndarray
    plate: np.ndarray
    well: np.ndarray
    split: np.ndarray
    delta: np.ndarray


@dataclass
class Pairs:
    support: np.ndarray
    target: np.ndarray
    compound: np.ndarray
    dose: np.ndarray
    held_index: np.ndarray
    held_plate: np.ndarray
    held_well: np.ndarray
    support_indices: list[tuple[int, ...]]


class Teacher(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, latent: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, latent),
            nn.GELU(), nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--outdir", type=Path)
    p.add_argument("--aggregate-root", type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--budget", type=int, choices=BUDGETS, default=1)
    p.add_argument("--dose", default="all", help="standard dose label or all")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--null-rounds", type=int, default=32)
    p.add_argument("--bootstrap-rounds", type=int, default=2000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.aggregate_root:
        if args.seed is not None or args.outdir is not None:
            p.error("--aggregate-root cannot be combined with --seed/--outdir")
        return args
    if args.seed is None and not (args.dry_run or args.smoke):
        p.error("--seed is required for an evidence run")
    if not args.dry_run and not args.smoke and args.outdir is None:
        p.error("--outdir is required")
    if args.null_rounds < 1 or args.bootstrap_rounds < 1:
        p.error("null/bootstrap rounds must be positive")
    return args


def decode(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8", errors="replace").astype(str)
    return values.astype(str)


def load_rows(path: Path) -> Rows:
    with np.load(path, allow_pickle=False) as loaded:
        required = {"compound_id", "dose", "plate", "well", "split", "delta"}
        missing = required - set(loaded.files)
        if missing:
            raise ValueError(f"{path} missing fields: {sorted(missing)}")
        rows = Rows(
            compound=decode(loaded["compound_id"]), dose=decode(loaded["dose"]),
            plate=decode(loaded["plate"]), well=decode(loaded["well"]),
            split=decode(loaded["split"]), delta=loaded["delta"].astype(np.float32),
        )
    n = len(rows.compound)
    if rows.delta.ndim != 2 or rows.delta.shape[0] != n or not np.isfinite(rows.delta).all():
        raise ValueError(f"Invalid delta matrix in {path}: {rows.delta.shape}")
    if any(len(x) != n for x in (rows.dose, rows.plate, rows.well, rows.split)):
        raise ValueError("Metadata/vector length mismatch")
    if set(rows.split) - {"train", "valid", "test"}:
        raise ValueError(f"Unexpected split values: {sorted(set(rows.split))}")
    for split in ("train", "valid", "test"):
        compounds = set(rows.compound[rows.split == split])
        for other in ("train", "valid", "test"):
            if other != split and compounds & set(rows.compound[rows.split == other]):
                raise ValueError("Compound overlap between locked splits")
    return rows


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return (seed + int.from_bytes(digest, "little")) % (2**63 - 1)


def set_seed(seed: int) -> None:
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed % (2**32 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed % (2**32 - 1))


def fisher(value: float) -> float:
    return float(np.arctanh(np.clip(value, -0.999999, 0.999999)))


def pcc(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not np.isfinite(denom) or denom <= 0:
        return None
    value = float(np.dot(left, right) / denom)
    return value if np.isfinite(value) else None


def make_pairs(rows: Rows, budget: int, dose: str) -> Pairs:
    # Plate order is a property of the frozen manifest, never tuned per seed.
    groups: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (compound, label) in enumerate(zip(rows.compound, rows.dose)):
        if dose != "all" and label != dose:
            continue
        groups[(compound, label)].append(i)
    supports: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    compounds: list[str] = []
    doses: list[str] = []
    held: list[int] = []
    held_plates: list[str] = []
    held_wells: list[str] = []
    support_indices: list[tuple[int, ...]] = []
    for (compound, label), indices in sorted(groups.items()):
        indices = sorted(indices, key=lambda i: (rows.plate[i], rows.well[i], int(i)))
        if len(indices) < budget + 1:
            continue
        for held_position, held_index in enumerate(indices):
            others = [i for i in indices if i != held_index]
            selected = tuple(others[:budget])
            supports.append(rows.delta[np.asarray(selected)].mean(axis=0, dtype=np.float64).astype(np.float32))
            targets.append(rows.delta[held_index])
            compounds.append(compound); doses.append(label); held.append(held_index)
            held_plates.append(rows.plate[held_index]); held_wells.append(rows.well[held_index])
            support_indices.append(selected)
    if not supports:
        raise RuntimeError(f"No pairs for budget={budget}, dose={dose}")
    return Pairs(
        support=np.vstack(supports), target=np.vstack(targets),
        compound=np.asarray(compounds, dtype=str), dose=np.asarray(doses, dtype=str),
        held_index=np.asarray(held, dtype=np.int64), held_plate=np.asarray(held_plates, dtype=str),
        held_well=np.asarray(held_wells, dtype=str), support_indices=support_indices,
    )


def subset_pairs(pairs: Pairs, compounds: set[str]) -> Pairs:
    ix = np.asarray([i for i, value in enumerate(pairs.compound) if value in compounds], dtype=np.int64)
    if ix.size == 0:
        raise RuntimeError("Empty pair subset")
    return Pairs(
        support=pairs.support[ix], target=pairs.target[ix], compound=pairs.compound[ix],
        dose=pairs.dose[ix], held_index=pairs.held_index[ix], held_plate=pairs.held_plate[ix],
        held_well=pairs.held_well[ix], support_indices=[pairs.support_indices[int(i)] for i in ix],
    )


def select_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if value in ("auto", "cuda") and torch.cuda.is_available() else "cpu")


def fit_teacher(train: Pairs, valid: Pairs, args: argparse.Namespace, device: torch.device) -> tuple[Teacher, dict[str, np.ndarray], dict[str, float]]:
    set_seed(args.seed)
    support_mean = train.support.mean(axis=0, dtype=np.float64).astype(np.float32)
    support_scale = np.maximum(train.support.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    target_mean = train.target.mean(axis=0, dtype=np.float64).astype(np.float32)
    target_scale = np.maximum(train.target.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6)
    x = (train.support - support_mean) / support_scale
    y = (train.target - target_mean) / target_scale
    vx = (valid.support - support_mean) / support_scale
    vy = (valid.target - target_mean) / target_scale
    model = Teacher(train.support.shape[1], args.hidden_dim, args.latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(); generator.manual_seed(stable_seed(args.seed, "loader"))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0,
    )
    best_state = None; best_value = float("inf"); best_epoch = -1
    epochs = 1 if args.smoke else args.epochs
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((model(batch_x.to(device)) - batch_y.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite teacher loss")
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            value = float(torch.mean((model(torch.from_numpy(vx).to(device)) - torch.from_numpy(vy).to(device)) ** 2).cpu())
        if value < best_value:
            best_value = value; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"[cpg0004 teacher seed={args.seed} budget={args.budget} epoch={epoch}] valid_mse={value:.7f}", flush=True)
    if best_state is None:
        raise RuntimeError("No teacher checkpoint")
    model.load_state_dict(best_state)
    return model, {"support_mean": support_mean, "support_scale": support_scale, "target_mean": target_mean, "target_scale": target_scale}, {"best_epoch": best_epoch, "best_validation_mse": best_value}


def predict(model: Teacher, pairs: Pairs, stats: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    x = (pairs.support - stats["support_mean"]) / stats["support_scale"]
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(x).to(device)).cpu().numpy()
    return out * stats["target_scale"] + stats["target_mean"]


def well_row(value: str) -> str:
    token = str(value).split(";")[0]
    letters = ""
    for char in token:
        if char.isalpha():
            letters += char
        else:
            break
    return letters


def make_matched_null(rows: Rows, pairs: Pairs, null_rounds: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Exact well slots are compound-unique in this resource (the exact-well
    # availability is therefore reported separately and expected to be zero).
    # Use exact dose + plate + well-row as the strongest feasible foreign
    # match, while preserving the same support/held plate structure.
    exact_slot_map: defaultdict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    row_slot_map: defaultdict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    for i in range(len(rows.compound)):
        exact_slot_map[(rows.dose[i], rows.plate[i], rows.well[i])][rows.compound[i]] = i
        row_slot_map[(rows.dose[i], rows.plate[i], well_row(rows.well[i]))][rows.compound[i]] = i
    slot_map: defaultdict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    slot_map = row_slot_map
    null_z = np.full(len(pairs.compound), np.nan, dtype=np.float64)
    usable = np.zeros(len(pairs.compound), dtype=bool)
    exact_usable = np.zeros(len(pairs.compound), dtype=bool)
    for position in range(len(pairs.compound)):
        target_compound = pairs.compound[position]
        held = int(pairs.held_index[position])
        exact_slots = [(rows.dose[held], rows.plate[held], rows.well[held])]
        exact_slots.extend((rows.dose[i], rows.plate[i], rows.well[i]) for i in pairs.support_indices[position])
        slots = [(dose, plate, well_row(well)) for dose, plate, well in exact_slots]
        exact_candidate: set[str] | None = None
        for slot in exact_slots:
            current = set(exact_slot_map.get(slot, {}))
            exact_candidate = current if exact_candidate is None else exact_candidate & current
        if exact_candidate is not None:
            exact_candidate.discard(target_compound)
            exact_candidate = {c for c in exact_candidate if rows.split[exact_slot_map[exact_slots[0]][c]] == "test"}
            exact_usable[position] = bool(exact_candidate)
        candidate: set[str] | None = None
        for slot in slots:
            current = set(slot_map.get(slot, {}))
            candidate = current if candidate is None else candidate & current
        if candidate is None:
            continue
        candidate.discard(target_compound)
        candidate = {c for c in candidate if rows.split[slot_map[slots[0]][c]] == "test"}
        if not candidate:
            continue
        choices = sorted(candidate)
        rng = np.random.default_rng(stable_seed(seed, f"null|{target_compound}|{pairs.dose[position]}|{pairs.held_plate[position]}|{','.join(pairs.held_well[position].split('|'))}"))
        values: list[float] = []
        for _ in range(null_rounds):
            foreign = choices[int(rng.integers(len(choices)))]
            foreign_held = slot_map[slots[0]][foreign]
            foreign_support = [slot_map[slot][foreign] for slot in slots[1:]]
            score = pcc(rows.delta[foreign_held], rows.delta[np.asarray(foreign_support)].mean(axis=0))
            if score is not None:
                values.append(fisher(score))
        if values:
            null_z[position] = float(np.mean(values)); usable[position] = True
    return null_z, usable, exact_usable


def score_predictions(predictions: dict[str, np.ndarray], pairs: Pairs, null_z: np.ndarray, usable: np.ndarray) -> tuple[dict[str, dict[str, float]], dict[str, list[float]], dict[str, list[float]]]:
    records: dict[str, dict[str, float]] = {}
    vectors: dict[str, list[float]] = {}
    raw_vectors: dict[str, list[float]] = {}
    for method, values in predictions.items():
        by_compound: defaultdict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for i in np.flatnonzero(usable):
            score = pcc(values[i], pairs.target[i])
            if score is None:
                continue
            z = fisher(score)
            by_compound[pairs.compound[i]].append((z, score, float(null_z[i])))
        excess: list[float] = []; raw_z: list[float] = []; raw_pcc: list[float] = []
        for compound, values_for_compound in sorted(by_compound.items()):
            mz = float(np.mean([x[0] for x in values_for_compound]))
            mp = float(np.mean([x[1] for x in values_for_compound]))
            nz = float(np.mean([x[2] for x in values_for_compound]))
            raw_z.append(mz); raw_pcc.append(mp); excess.append(mz - nz)
        vectors[method] = excess
        raw_vectors[method] = raw_pcc
        records[method] = {
            "compound_count": int(len(excess)),
            "pair_count": int(sum(len(v) for v in by_compound.values())),
            "raw_pcc_mean": float(np.mean(raw_pcc)) if raw_pcc else float("nan"),
            "raw_fisher_z_mean": float(np.mean(raw_z)) if raw_z else float("nan"),
            "null_fisher_z_mean": float(np.mean([x for vals in by_compound.values() for _, _, x in vals])) if by_compound else float("nan"),
            "excess_fisher_z_mean": float(np.mean(excess)) if excess else float("nan"),
        }
    denominator = records.get("support_mean", {}).get("excess_fisher_z_mean", float("nan"))
    for value in records.values():
        value["rsf_vs_support_mean"] = float(value["excess_fisher_z_mean"] / denominator) if np.isfinite(denominator) and denominator != 0 else float("nan")
    return records, vectors, raw_vectors


def bootstrap_difference(left: list[float], right: list[float], rounds: int, seed: int) -> dict[str, float | int]:
    n = min(len(left), len(right))
    if n == 0:
        return {"n": 0, "point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "rounds": rounds}
    a = np.asarray(left[:n], dtype=np.float64); b = np.asarray(right[:n], dtype=np.float64)
    diff = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(rounds, n))
    samples = diff[indices].mean(axis=1)
    return {"n": int(n), "point": float(diff.mean()), "ci_low": float(np.quantile(samples, .025)), "ci_high": float(np.quantile(samples, .975)), "rounds": int(rounds)}


def write_pair_manifest(path: Path, rows: Rows, pairs: Pairs, null_z: np.ndarray, usable: np.ndarray, exact_usable: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_index", "compound", "dose", "held_row", "held_plate", "held_well", "support_rows", "support_plates", "support_wells", "row_matched_null_usable", "exact_well_null_usable", "null_z"])
        for i in range(len(pairs.compound)):
            support = pairs.support_indices[i]
            writer.writerow([i, pairs.compound[i], pairs.dose[i], int(pairs.held_index[i]), pairs.held_plate[i], pairs.held_well[i], "|".join(map(str, support)), "|".join(rows.plate[j] for j in support), "|".join(rows.well[j] for j in support), int(usable[i]), int(exact_usable[i]), "" if not np.isfinite(null_z[i]) else float(null_z[i])])


def run(args: argparse.Namespace) -> None:
    if args.outdir is None and not (args.dry_run or args.smoke):
        raise ValueError("--outdir required")
    if args.outdir is not None and args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.outdir}")
    rows = load_rows(args.data)
    pairs = make_pairs(rows, args.budget, args.dose)
    split_sets = {name: set(rows.compound[rows.split == name]) for name in ("train", "valid", "test")}
    train = subset_pairs(pairs, split_sets["train"])
    valid = subset_pairs(pairs, split_sets["valid"])
    test = subset_pairs(pairs, split_sets["test"])
    if args.dry_run or args.smoke:
        payload = {"version": VERSION, "dry_run": bool(args.dry_run), "smoke": bool(args.smoke), "feature_dim": int(rows.delta.shape[1]), "row_count": len(rows.compound), "budget": args.budget, "dose": args.dose, "pair_counts": {"all": len(pairs.compound), "train": len(train.compound), "valid": len(valid.compound), "test": len(test.compound)}, "compound_counts": {k: len(v) for k, v in split_sets.items()}, "guardrails": ["compound-level locked split", "plate is independent repeat unit", "teacher sees only support mean", "no GE input", "dose/plate/well-row foreign null; exact-well availability audited"]}
        print(json.dumps(payload, indent=2, sort_keys=True)); return
    args.outdir.mkdir(parents=True)
    device = select_device(args.device)
    model, stats, training = fit_teacher(train, valid, args, device)
    null_z, usable, exact_usable = make_matched_null(rows, test, args.null_rounds, args.seed)
    predictions = {"support_mean": test.support, "teacher": predict(model, test, stats, device)}
    summaries, vectors, raw_vectors = score_predictions(predictions, test, null_z, usable)
    contrast = bootstrap_difference(vectors["teacher"], vectors["support_mean"], args.bootstrap_rounds, stable_seed(args.seed, "bootstrap|teacher-minus-mean"))
    contrast_pcc = bootstrap_difference(raw_vectors["teacher"], raw_vectors["support_mean"], args.bootstrap_rounds, stable_seed(args.seed, "bootstrap|teacher-minus-mean-pcc"))
    write_pair_manifest(args.outdir / "test_pair_manifest.csv", rows, test, null_z, usable, exact_usable)
    per_condition = []
    for i in np.flatnonzero(usable):
        teacher_pcc = pcc(predictions["teacher"][i], test.target[i]); mean_pcc = pcc(test.support[i], test.target[i])
        per_condition.append({"pair_index": int(i), "compound": test.compound[i], "dose": test.dose[i], "held_plate": test.held_plate[i], "held_well": test.held_well[i], "teacher_pcc": teacher_pcc, "support_mean_pcc": mean_pcc, "null_z": float(null_z[i])})
    with (args.outdir / "per_pair_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_condition[0]) if per_condition else ["pair_index"]); writer.writeheader(); writer.writerows(per_condition)
    payload = {
        "version": VERSION, "seed": args.seed, "budget": args.budget, "dose": args.dose,
        "device": str(device), "data": str(args.data), "feature_dim": int(rows.delta.shape[1]),
        "row_count": int(len(rows.compound)), "pair_counts": {"all": len(pairs.compound), "train": len(train.compound), "valid": len(valid.compound), "test": len(test.compound), "row_matched_null_usable": int(usable.sum()), "exact_well_null_usable": int(exact_usable.sum())},
        "test_compounds_with_row_matched_null": int(len(set(test.compound[usable]))),
        "row_matched_null_usable_pairs": int(usable.sum()),
        "exact_well_null_usable_pairs": int(exact_usable.sum()),
        "training": training, "locked_hyperparameters": {"epochs": args.epochs, "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "null_rounds": args.null_rounds, "bootstrap_rounds": args.bootstrap_rounds},
        "summaries": summaries, "paired_bootstrap": {"teacher_minus_support_mean_excess_z": contrast, "teacher_minus_support_mean_pcc": contrast_pcc},
        "definition": "Strict held-out plate target; teacher and support mean use exactly the same R support plates; foreign null matches dose, held/support plate slots, well row, and test split. Exact-well availability is reported separately because LINCS assigns each exact well slot to one compound.",
        "guardrails": ["no GE data or post-treatment cross-modal input", "test compounds never fit the teacher", "validation only selects checkpoint", "row-matched null is independent of model fitting", "technical wells were collapsed during data preparation", "exact-well foreign null availability is explicitly audited"]
    }
    (args.outdir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def aggregate(root: Path) -> None:
    paths = [root / f"seed{s}" / "metrics.json" for s in SEEDS]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("Missing seed metrics: " + ", ".join(str(p) for p in paths if not p.is_file()))
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    methods = sorted(payloads[0]["summaries"])
    summary = {}
    for method in methods:
        summary[method] = {key: float(np.mean([p["summaries"][method][key] for p in payloads])) for key in ("raw_pcc_mean", "excess_fisher_z_mean", "rsf_vs_support_mean")}
        summary[method]["compound_count"] = int(min(p["summaries"][method]["compound_count"] for p in payloads))
    contrast = [p["paired_bootstrap"]["teacher_minus_support_mean_excess_z"] for p in payloads]
    out = {"version": VERSION, "seeds": list(SEEDS), "seed_metric_paths": [str(p) for p in paths], "seed_mean_summaries": summary, "per_seed_contrasts": contrast, "note": "Seed means are descriptive; inspect each seed and pooled paired intervals before a gate decision."}
    path = root / "aggregate_metrics.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.aggregate_root:
        aggregate(args.aggregate_root)
    else:
        run(args)


if __name__ == "__main__":
    main()
