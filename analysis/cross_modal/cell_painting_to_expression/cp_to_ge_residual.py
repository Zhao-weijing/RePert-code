#!/usr/bin/env python3
"""Evaluate frozen GE teacher plus CP residual evidence.

This is the locked CP -> GE direction.  The GE teacher from Experiment 1 is
reloaded and never retrained.  A train-only CP-to-GE mapper supplies a
candidate GE vector; its residual from the train-only target mean is added to
the frozen teacher with a validation-selected beta. Correct, shuffled, and
metadata-matched foreign CP inputs reuse the same beta and the same frozen
teacher.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "expression_teacher"))
import ge_teacher_experiment as T  # noqa: E402


VERSION = "MVCPert-CP-to-GE-Frozen-Teacher-Residual-2026-08-30"
SEEDS = T.SEEDS
BETA_GRID = (0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
CP_DIM = 775
GE_DIM = 977


class CPToGENet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(CP_DIM, 256), nn.GELU(),
            nn.Linear(256, 32), nn.GELU(),
            nn.Linear(32, 256), nn.GELU(),
            nn.Linear(256, GE_DIM),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cm0-npz", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0/molecule_aggregates.npz"))
    p.add_argument("--ge-data-root", type=Path, default=Path("/path/to/data/AIDD/MVC_BBBC047/PlateMedian_v1_training/crossmodal_cm0"))
    p.add_argument("--split-lock", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/source/baseline/artifacts/split_locks/BBBC047_smiles_split_seed3407_official_v1.json"))
    p.add_argument("--teacher-root", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/runs/multimodal_expansion_20260830/01_GE_teacher/formal_v2"))
    p.add_argument("--outdir", type=Path)
    p.add_argument("--aggregate-root", type=Path)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--null-rounds", type=int, default=32)
    p.add_argument("--bootstrap-rounds", type=int, default=10000)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def decode(values: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in np.asarray(values).reshape(-1)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_aggregates(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    output: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as loaded:
        for split in ("train", "valid", "test"):
            smiles = decode(loaded[f"{split}_smiles"])
            cp = loaded[f"{split}_cp"].astype(np.float32)
            if cp.shape != (len(smiles), CP_DIM):
                raise ValueError(f"Unexpected CP aggregate shape for {split}: {cp.shape}")
            output[split] = {
                "smiles": smiles,
                "cp": cp,
                "cp_plates": loaded[f"{split}_cp_plates"].astype(np.float64),
                "cp_rms": loaded[f"{split}_cp_rms"].astype(np.float64),
                "cp_max_dose": loaded[f"{split}_max_dose"].astype(np.float64),
            }
    return output


def aggregate_map(data: dict[str, dict[str, Any]], split: str) -> dict[str, np.ndarray]:
    return {compound: data[split]["cp"][i] for i, compound in enumerate(data[split]["smiles"])}


def load_teacher(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        return {"compound": np.asarray(decode(loaded["compound"])), "teacher": loaded["teacher"].astype(np.float32)}


def fit_mapper(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[CPToGENet, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    set_seed(seed)
    x_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_scale = np.maximum(x_train.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    y_mean = y_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    y_scale = np.maximum(y_train.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    tx = (x_train - x_mean) / x_scale
    ty = (y_train - y_mean) / y_scale
    vx = (x_valid - x_mean) / x_scale
    vy = (y_valid - y_mean) / y_scale
    model = CPToGENet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(); generator.manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss, best_epoch = float("inf"), -1
    limit = 1 if args.smoke else args.epochs
    for epoch in range(limit):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x.to(device)) - y.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CP-to-GE mapper loss")
            loss.backward(); optimizer.step()
        model.eval(); total = 0.0; count = 0
        with torch.no_grad():
            for start in range(0, len(vx), args.batch_size):
                x = torch.from_numpy(vx[start:start + args.batch_size]).to(device)
                y = torch.from_numpy(vy[start:start + args.batch_size]).to(device)
                value = torch.mean((model(x) - y) ** 2)
                total += float(value.detach().cpu()) * len(x); count += len(x)
        val_loss = total / max(count, 1)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 0 or epoch % 10 == 0:
            print(f"[seed {seed}] CP-to-GE epoch={epoch} validation_mse={val_loss:.7f}", flush=True)
    if best_state is None:
        raise RuntimeError("CP-to-GE mapper did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, x_mean, x_scale, y_mean, y_scale, best_epoch, best_loss


def predict_mapper(model: CPToGENet, x: np.ndarray, x_mean: np.ndarray, x_scale: np.ndarray, y_mean: np.ndarray, y_scale: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval(); output = np.empty((len(x), GE_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            value = (x[start:start + batch_size] - x_mean) / x_scale
            pred = model(torch.from_numpy(value).to(device)).detach().cpu().numpy()
            output[start:start + len(pred)] = pred * y_scale + y_mean
    return output


def derangement(compounds: list[str], seed: int, label: str) -> dict[str, str]:
    ordered = sorted(compounds, key=lambda x: T.stable_seed(seed, f"shuffle|{label}|{x}"))
    if len(ordered) < 2:
        return {}
    return {compound: ordered[(i + 1) % len(ordered)] for i, compound in enumerate(ordered)}


def quartile(value: float, cuts: np.ndarray) -> int:
    return int(np.searchsorted(cuts, value, side="right"))


def matched_foreign(
    pair_manifest: list[dict[str, Any]],
    split_data: dict[str, Any],
    train_data: dict[str, Any],
    seed: int,
    label: str,
) -> dict[str, str]:
    train_rms = np.asarray(train_data["cp_rms"], dtype=np.float64)
    train_max = np.asarray(train_data["cp_max_dose"], dtype=np.float64)
    rms_cuts = np.quantile(train_rms[np.isfinite(train_rms)], [0.25, 0.5, 0.75])
    max_cuts = np.quantile(train_max[np.isfinite(train_max)], [0.25, 0.5, 0.75])
    aggregate_index = {str(c): i for i, c in enumerate(split_data["smiles"])}
    meta = {}
    for row in pair_manifest:
        c = str(row["canonical_compound"])
        if c not in aggregate_index:
            continue
        i = aggregate_index[c]
        meta[c] = {
            "plates": int(split_data["cp_plates"][i]),
            "rms_bin": quartile(float(split_data["cp_rms"][i]), rms_cuts),
            "max_bin": quartile(float(split_data["cp_max_dose"][i]), max_cuts),
            "max_dose": float(split_data["cp_max_dose"][i]),
            "dose": float(row["dose"]),
        }
    output: dict[str, str] = {}
    for row in pair_manifest:
        compound = str(row["canonical_compound"])
        if compound not in meta:
            continue
        target = meta[compound]
        candidates = []
        for donor in meta:
            if donor == compound or donor not in meta:
                continue
            value = meta[donor]
            if value["plates"] != target["plates"] or value["rms_bin"] != target["rms_bin"] or value["max_bin"] != target["max_bin"]:
                continue
            tolerance = max(T.DOSE_TOLERANCE_MILLIMOLAR, 0.01 * max(abs(target["dose"]), 1.0))
            if abs(value["max_dose"] - target["dose"]) > tolerance:
                continue
            distance = (abs(value["max_dose"] - target["dose"]), abs(value["max_dose"] - target["max_dose"]), donor)
            candidates.append((distance, donor))
        if candidates:
            output[compound] = min(candidates, key=lambda item: (item[0], T.stable_seed(seed, f"foreign|{label}|{compound}|{item[1]}")))[1]
    return output


def prediction_map(pairs: T.BudgetPairs, p0: np.ndarray, mapper: dict[str, np.ndarray], v_ge: np.ndarray, beta: float, donors: dict[str, str] | None, cp_map: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for i, compound in enumerate(pairs.compound):
        donor = donors.get(compound, compound) if donors is not None else compound
        if donor not in cp_map:
            continue
        output[compound] = (p0[i] + beta * (mapper[donor] - v_ge)).astype(np.float32)
    return output


def select_beta(pairs: T.BudgetPairs, p0: np.ndarray, mapper: dict[str, np.ndarray], v_ge: np.ndarray, reference: dict[str, dict[str, Any]], cp_map: dict[str, np.ndarray]) -> tuple[float, dict[str, Any]]:
    curve: dict[str, Any] = {}
    for beta in BETA_GRID:
        pred = prediction_map(pairs, p0, mapper, v_ge, beta, None, cp_map)
        records = T.method_records(np.vstack([pred[c] for c in pairs.compound if c in pred]), pairs, reference, f"correct_beta_{beta}")
        curve[str(beta)] = T.summarize(records)
    selected = max(BETA_GRID, key=lambda beta: (curve[str(beta)]["rsf"], -beta))
    return selected, curve


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def records_for(predictions: dict[str, np.ndarray], pairs: T.BudgetPairs, reference: dict[str, dict[str, Any]], method: str) -> dict[str, dict[str, Any]]:
    index = {compound: i for i, compound in enumerate(pairs.compound)}
    output: dict[str, dict[str, Any]] = {}
    for compound, ref in reference.items():
        if compound not in predictions:
            continue
        score = T.pcc(predictions[compound], pairs.target[index[compound]])
        if score is None:
            continue
        output[compound] = {"method": method, "canonical_compound": compound, "dose": pairs.dose[index[compound]], "raw_pcc": score, "model_z": T.fisher_z(score), "null_z": ref["null_z"], "model_excess_z": T.fisher_z(score) - ref["null_z"], "replicate_excess_z": ref["replicate_excess_z"]}
    return output


def contrast(left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], rounds: int, seed: int) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    if not common:
        raise RuntimeError("No common molecules for paired contrast")
    diff_excess = np.asarray([left[c]["model_excess_z"] - right[c]["model_excess_z"] for c in common], dtype=np.float64)
    diff_raw = np.asarray([left[c]["raw_pcc"] - right[c]["raw_pcc"] for c in common], dtype=np.float64)
    rng = np.random.default_rng(seed); boot_excess = np.empty(rounds); boot_raw = np.empty(rounds)
    for begin in range(0, rounds, 100):
        end = min(rounds, begin + 100); draw = rng.integers(0, len(common), size=(end - begin, len(common)))
        boot_excess[begin:end] = diff_excess[draw].mean(axis=1); boot_raw[begin:end] = diff_raw[draw].mean(axis=1)
    return {"molecule_count": len(common), "excess_z_point": float(diff_excess.mean()), "excess_z_ci95": [float(np.quantile(boot_excess, .025)), float(np.quantile(boot_excess, .975))], "raw_pcc_point": float(diff_raw.mean()), "raw_pcc_ci95": [float(np.quantile(boot_raw, .025)), float(np.quantile(boot_raw, .975))], "bootstrap_rounds": rounds, "bootstrap_seed": seed}


def run_one(args: argparse.Namespace) -> None:
    if args.seed is None or args.outdir is None:
        raise ValueError("--seed and --outdir are required")
    if args.outdir.exists():
        raise FileExistsError(args.outdir)
    args.outdir.mkdir(parents=True)
    if args.smoke:
        args.epochs = 1; args.bootstrap_rounds = min(args.bootstrap_rounds, 100)
    data = load_aggregates(args.cm0_npz)
    rows = {split: T.load_split_rows(args.ge_data_root, split) for split in ("train", "valid", "test")}
    pairs = {split: T.build_pairs(rows[split], split, 1) for split in ("train", "valid", "test")}
    teacher_valid = load_teacher(args.teacher_root / f"seed_{args.seed}" / "budget1_valid_predictions.npz")
    teacher_test = load_teacher(args.teacher_root / f"seed_{args.seed}" / "budget1_predictions.npz")
    p0 = {"valid": teacher_valid["teacher"], "test": teacher_test["teacher"]}
    for split in ("valid", "test"):
        expected = pairs[split].compound
        if list(teacher_valid["compound"] if split == "valid" else teacher_test["compound"]) != expected:
            raise ValueError(f"Teacher prediction order mismatch for {split}")
    cp_maps = {split: aggregate_map(data, split) for split in ("train", "valid", "test")}
    x_train = np.vstack([cp_maps["train"][c] for c in pairs["train"].compound])
    y_train = pairs["train"].target
    x_valid = np.vstack([cp_maps["valid"][c] for c in pairs["valid"].compound])
    y_valid = pairs["valid"].target
    device = T.choose_device(args.device)
    model, x_mean, x_scale, y_mean, y_scale, best_epoch, best_loss = fit_mapper(x_train, y_train, x_valid, y_valid, args.seed, args, device)
    mapped = {
        split: predict_mapper(
            model,
            np.vstack([cp_maps[split][c] for c in data[split]["smiles"]]),
            x_mean,
            x_scale,
            y_mean,
            y_scale,
            args.batch_size,
            device,
        )
        for split in ("valid", "test")
    }
    mapper_maps = {split: {str(c): mapped[split][i] for i, c in enumerate(data[split]["smiles"])} for split in ("valid", "test")}
    # The target-space mean is the frozen, train-only virtual prior used to define the residual.
    v_ge = y_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    reference_valid = T.strict_reference(rows["valid"], pairs["valid"], args.null_rounds, args.seed)
    reference_test = T.strict_reference(rows["test"], pairs["test"], args.null_rounds, args.seed)
    beta, beta_curve = select_beta(pairs["valid"], p0["valid"], mapper_maps["valid"], v_ge, reference_valid, cp_maps["valid"])
    shuffled = {split: derangement(pairs[split].compound, args.seed, split) for split in ("valid", "test")}
    foreign = {split: matched_foreign(pairs[split].manifest, data[split], data["train"], args.seed, split) for split in ("valid", "test")}
    outputs: dict[str, dict[str, dict[str, Any]]] = {}
    for split, ref, p0_values in (("valid", reference_valid, p0["valid"]), ("test", reference_test, p0["test"])):
        p0_map = {c: p0_values[i] for i, c in enumerate(pairs[split].compound)}
        mapper_map = mapper_maps[split]
        predictions = {
            "frozen_GE_teacher": p0_map,
            "correct_CP": prediction_map(pairs[split], p0_values, mapper_map, v_ge, beta, None, cp_maps[split]),
            "shuffled_CP": prediction_map(pairs[split], p0_values, mapper_map, v_ge, beta, shuffled[split], cp_maps[split]),
            "matched_foreign_CP": prediction_map(pairs[split], p0_values, mapper_map, v_ge, beta, foreign[split], cp_maps[split]),
        }
        outputs[split] = {name: records_for(pred, pairs[split], ref, name) for name, pred in predictions.items()}
    test = outputs["test"]
    contrasts = {
        "correct_minus_p0": contrast(test["correct_CP"], test["frozen_GE_teacher"], args.bootstrap_rounds, T.stable_seed(args.seed, "correct-minus-p0")),
        "correct_minus_shuffled": contrast(test["correct_CP"], test["shuffled_CP"], args.bootstrap_rounds, T.stable_seed(args.seed, "correct-minus-shuffled")),
    }
    if outputs["test"]["matched_foreign_CP"]:
        contrasts["correct_minus_matched_foreign"] = contrast(test["correct_CP"], test["matched_foreign_CP"], args.bootstrap_rounds, T.stable_seed(args.seed, "correct-minus-foreign"))
    for name in ("correct_CP", "shuffled_CP", "matched_foreign_CP"):
        if outputs["test"][name]:
            write_csv(args.outdir / ("correct.csv" if name == "correct_CP" else "shuffled.csv" if name == "shuffled_CP" else "matched_foreign.csv"), list(outputs["test"][name].values()))
    write_csv(args.outdir / "frozen.csv", list(outputs["test"]["frozen_GE_teacher"].values()))
    write_csv(args.outdir / "paired_contrasts.csv", [{"contrast": name, **value} for name, value in contrasts.items()])
    write_csv(args.outdir / "matched_foreign_mapping.csv", [{"split": split, "canonical_compound": c, "donor_compound": d} for split in ("valid", "test") for c, d in sorted(foreign[split].items())])
    payload = {"version": VERSION, "seed": args.seed, "device": str(device), "selected_beta": beta, "beta_grid": BETA_GRID, "validation_beta_curve": beta_curve, "mapper_checkpoint": {"best_epoch": best_epoch, "best_validation_mse": best_loss}, "pair_counts": {split: len(pairs[split].compound) for split in ("train", "valid", "test")}, "reference_counts": {"valid": len(reference_valid), "test": len(reference_test)}, "matched_foreign_counts": {split: len(foreign[split]) for split in ("valid", "test")}, "test_summaries": {name: T.summarize(value) for name, value in test.items()}, "test_contrasts": contrasts, "guards": ["GE teacher P0 is frozen and loaded from Experiment 1", "CP-to-GE mapper fits training compounds only", "V_GE is the train-only held-GE target mean", "beta is selected on validation only and reused for all test controls", "test held-out GE rows are used only by the final evaluator", "foreign matching uses CP plate-count and train-fitted CP RMS/max-dose quartiles plus fixed dose tolerance"]}
    (args.outdir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def read_metrics(root: Path) -> list[dict[str, Any]]:
    output = []
    for seed in SEEDS:
        path = root / f"seed_{seed}" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append(json.loads(path.read_text(encoding="utf-8")))
    return output


def aggregate(args: argparse.Namespace) -> None:
    root = args.aggregate_root
    if root is None:
        raise ValueError("--aggregate-root required")
    metrics = read_metrics(root)
    contrasts: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for payload in metrics:
        for name, value in payload["test_contrasts"].items():
            contrasts.append({"scope": "seed", "seed": payload["seed"], "contrast": name, **value})
        for name, value in payload["test_summaries"].items():
            summary.append({"scope": "seed", "seed": payload["seed"], "method": name, **value})
    # Pool per-molecule differences after averaging the three fixed seeds.
    args.bootstrap_rounds = args.bootstrap_rounds or 10000
    # Reconstruct score dictionaries from the persisted per-method CSVs.
    method_files = {"correct_minus_p0": ("correct.csv", None), "correct_minus_shuffled": ("correct.csv", "shuffled.csv"), "correct_minus_matched_foreign": ("correct.csv", "matched_foreign.csv")}
    for contrast_name, (left_name, right_name) in method_files.items():
        per_seed: list[tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]] = []
        for seed in SEEDS:
            def read_method(filename: str) -> dict[str, dict[str, float]]:
                if filename is None:
                    return {"__none__": {}}
                path = root / f"seed_{seed}" / filename
                if not path.is_file():
                    return {}
                result = {}
                with path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        result[row["canonical_compound"]] = {"model_excess_z": float(row["model_excess_z"]), "raw_pcc": float(row["raw_pcc"])}
                return result
            per_seed.append((read_method("correct.csv"), read_method("frozen.csv") if right_name is None else read_method(right_name)))
        common = set(per_seed[0][0])
        for left, right in per_seed:
            common &= set(left) & set(right)
        if not common:
            continue
        common = sorted(common)
        diff_excess = np.asarray([np.mean([left[c]["model_excess_z"] - right[c]["model_excess_z"] for left, right in per_seed]) for c in common], dtype=np.float64)
        diff_raw = np.asarray([np.mean([left[c]["raw_pcc"] - right[c]["raw_pcc"] for left, right in per_seed]) for c in common], dtype=np.float64)
        rng = np.random.default_rng(T.stable_seed(20260830, f"pooled|{contrast_name}")); rounds = args.bootstrap_rounds; bx = np.empty(rounds); br = np.empty(rounds)
        for begin in range(0, rounds, 100):
            end = min(rounds, begin + 100); draw = rng.integers(0, len(common), size=(end - begin, len(common)))
            bx[begin:end] = diff_excess[draw].mean(axis=1); br[begin:end] = diff_raw[draw].mean(axis=1)
        contrasts.append({"scope": "pooled_three_seed", "seed": "all", "contrast": contrast_name, "molecule_count": len(common), "excess_z_point": float(diff_excess.mean()), "excess_z_ci95": [float(np.quantile(bx, .025)), float(np.quantile(bx, .975))], "raw_pcc_point": float(diff_raw.mean()), "raw_pcc_ci95": [float(np.quantile(br, .025)), float(np.quantile(br, .975))], "bootstrap_rounds": rounds})
    write_csv(root / "paired_contrasts.csv", contrasts)
    write_csv(root / "bootstrap_summary.csv", contrasts)
    pooled = [row for row in contrasts if row["scope"] == "pooled_three_seed"]
    decisions = {row["contrast"]: {"point": row["excess_z_point"], "ci95": row["excess_z_ci95"], "go_pooled": row["excess_z_ci95"][0] > 0} for row in pooled}
    payload = {"version": VERSION, "seeds": list(SEEDS), "decisions": decisions, "files": {"paired_contrasts": str(root / "paired_contrasts.csv"), "bootstrap_summary": str(root / "bootstrap_summary.csv")}}
    (root / "aggregate_index.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def dry_run(args: argparse.Namespace) -> None:
    data = load_aggregates(args.cm0_npz)
    rows = {split: T.load_split_rows(args.ge_data_root, split) for split in ("train", "valid", "test")}
    pairs = {split: T.build_pairs(rows[split], split, 1) for split in ("train", "valid", "test")}
    print(json.dumps({"version": VERSION, "pair_counts": {split: len(pairs[split].compound) for split in pairs}, "cp_aggregate_counts": {split: len(data[split]["smiles"]) for split in data}, "guard": "dry-run only; no mapper training or test evaluation"}, indent=2, sort_keys=True))


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
