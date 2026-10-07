#!/usr/bin/env python3
"""Fit and evaluate post-hoc capacity controls for the v2 decomposition run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
FIT_SCRIPT = (Path(__file__).resolve().parents[4] / "analysis/predictor_supervision/bbbc047/support/run_fit.py")
CONFIRM_SCRIPT = (Path(__file__).resolve().parents[4] / "analysis/predictor_supervision/bbbc047/support/run_confirm.py")
PROTOCOL = HERE / "CAPACITY_PROTOCOL.md"
spec = importlib.util.spec_from_file_location("bbbc047_capacity_fit", FIT_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"unable to import {FIT_SCRIPT}")
RF = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = RF
spec.loader.exec_module(RF)

SRC = RF.SRC
LEGACY = RF.LEGACY
CP_DIM = RF.CP_DIM
FP_DIM = RF.FP_DIM
BUDGETS = RF.BUDGETS
SEEDS = RF.MASTER_SEEDS

VERSION = "BBBC047-CFRA-Decomposition-capacity-audit-v1-2026-09-13"
BASE_DIMS = (512, 128)
WIDE_DIMS = (988, 245)
SMALL_DIMS = (256, 80)
METHODS = ("RawWide", "RawTwoBranch", "CFRADecompSmall")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()} for row in rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "confirm"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-fit-root", type=Path, required=True)
    parser.add_argument("--prepare-root", type=Path, required=True)
    parser.add_argument("--rows-root", type=Path, default=RF.DEFAULT_ROWS)
    parser.add_argument("--model-h5", type=Path, default=RF.DEFAULT_H5)
    parser.add_argument("--aggregate-npz", type=Path, default=RF.DEFAULT_AGG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--student-epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    args = parser.parse_args()
    if args.batch_size < 1 or args.student_epochs < 1 or args.bootstrap_rounds < 1:
        parser.error("batch-size, student-epochs, and bootstrap-rounds must be positive")
    return args


def device_for(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


class StudentWidth(nn.Module):
    def __init__(self, hidden1: int, hidden2: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CP_DIM + FP_DIM, hidden1), nn.GELU(),
            nn.Linear(hidden1, hidden2), nn.GELU(),
            nn.Linear(hidden2, CP_DIM),
        )

    def forward(self, control: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((control, feature), dim=1))


def parameter_count(dims: tuple[int, int]) -> int:
    h1, h2 = dims
    return (CP_DIM + FP_DIM) * h1 + h1 + h1 * h2 + h2 + h2 * CP_DIM + CP_DIM


def seed_for(master: int, budget: int, label: str) -> int:
    return int(LEGACY.stable_int(master, f"{VERSION}|budget{budget}|{label}") % (2**32 - 1))


def predict_single(model: nn.Module, split: Any, stats: Mapping[str, np.ndarray], batch: int, device: torch.device) -> np.ndarray:
    control = (split.control - stats["control_mean"]) / stats["control_scale"]
    output = np.empty((len(split.compound), CP_DIM), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(split.compound), batch):
            value = model(
                torch.from_numpy(control[begin:begin + batch]).to(device),
                torch.from_numpy(split.fingerprint[begin:begin + batch]).to(device),
            ).cpu().numpy()
            output[begin:begin + len(value)] = value * stats["target_scale"] + stats["target_mean"]
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite prediction")
    return output


def predict_joint(models: tuple[nn.Module, nn.Module], split: Any, stats: Mapping[str, np.ndarray], batch: int, device: torch.device) -> np.ndarray:
    control = (split.control - stats["control_mean"]) / stats["control_scale"]
    output = np.empty((len(split.compound), CP_DIM), dtype=np.float32)
    for model in models:
        model.eval()
    with torch.no_grad():
        for begin in range(0, len(split.compound), batch):
            control_t = torch.from_numpy(control[begin:begin + batch]).to(device)
            feature_t = torch.from_numpy(split.fingerprint[begin:begin + batch]).to(device)
            value = (models[0](control_t, feature_t) + models[1](control_t, feature_t)).cpu().numpy()
            output[begin:begin + len(value)] = value * stats["target_scale"] + stats["target_mean"]
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite joint prediction")
    return output


def fit_single(train: Any, valid: Any, train_target: np.ndarray, valid_target: np.ndarray, dims: tuple[int, int], seed: int, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], np.ndarray, float, int]:
    LEGACY.set_seed(seed)
    stats = RF.branch_stats(train, train_target)
    model = StudentWidth(*dims).to(device)
    train_control = (train.control - stats["control_mean"]) / stats["control_scale"]
    train_z = (np.asarray(train_target, dtype=np.float32) - stats["target_mean"]) / stats["target_scale"]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(train_control), torch.from_numpy(train.fingerprint), torch.from_numpy(train_z)),
        batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_pred = None
    best_epoch = -1
    for epoch in range(args.student_epochs):
        model.train()
        for control, feature, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(control.to(device), feature.to(device)) - target.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite single-branch loss")
            loss.backward(); optimizer.step()
        prediction = predict_single(model, valid, stats, args.batch_size, device)
        valid_loss = float(np.mean((prediction.astype(np.float64) - valid_target.astype(np.float64)) ** 2))
        if valid_loss < best_loss:
            best_loss = valid_loss; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_pred = prediction.copy()
    if best_state is None or best_pred is None:
        raise RuntimeError("single-branch checkpoint missing")
    return {"state_dict": best_state, "stats": stats}, best_pred, best_loss, best_epoch


def fit_joint_raw(train: Any, valid: Any, train_target: np.ndarray, valid_target: np.ndarray, seed: int, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], np.ndarray, float, int]:
    LEGACY.set_seed(seed)
    stats = RF.branch_stats(train, train_target)
    models = (StudentWidth(*BASE_DIMS).to(device), StudentWidth(*BASE_DIMS).to(device))
    train_control = (train.control - stats["control_mean"]) / stats["control_scale"]
    train_z = (np.asarray(train_target, dtype=np.float32) - stats["target_mean"]) / stats["target_scale"]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(train_control), torch.from_numpy(train.fingerprint), torch.from_numpy(train_z)),
        batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(list(models[0].parameters()) + list(models[1].parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_states = None; best_loss = float("inf"); best_pred = None; best_epoch = -1
    for epoch in range(args.student_epochs):
        models[0].train(); models[1].train()
        for control, feature, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = models[0](control.to(device), feature.to(device)) + models[1](control.to(device), feature.to(device))
            loss = torch.mean((prediction - target.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite joint loss")
            loss.backward(); optimizer.step()
        prediction = predict_joint(models, valid, stats, args.batch_size, device)
        valid_loss = float(np.mean((prediction.astype(np.float64) - valid_target.astype(np.float64)) ** 2))
        if valid_loss < best_loss:
            best_loss = valid_loss; best_epoch = epoch; best_pred = prediction.copy()
            best_states = tuple({k: v.detach().cpu().clone() for k, v in model.state_dict().items()} for model in models)
    if best_states is None or best_pred is None:
        raise RuntimeError("joint checkpoint missing")
    return {"state_dict_a": best_states[0], "state_dict_b": best_states[1], "stats": stats}, best_pred, best_loss, best_epoch


def load_targets(formal_fit_root: Path, budget: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(formal_fit_root / f"budget{budget}" / "training_targets.npz", allow_pickle=False) as train_z, np.load(formal_fit_root / f"budget{budget}" / "validation_targets.npz", allow_pickle=False) as valid_z:
        train = {"compound": np.asarray(train_z["compound"], dtype=str)}
        train.update({key: np.asarray(train_z[key], dtype=np.float32) for key in ("raw", "cfra", "residual")})
        valid = {"compound": np.asarray(valid_z["compound"], dtype=str)}
        valid.update({key: np.asarray(valid_z[key], dtype=np.float32) for key in ("raw_held_mean", "cfra_teacher", "residual")})
    return train, valid


def fit_phase(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        if any(args.output_root.iterdir()):
            raise FileExistsError(args.output_root)
    else:
        args.output_root.mkdir(parents=True)
    formal = args.formal_fit_root
    if not (formal / "FIT_COMPLETE.json").is_file() or not (formal / "MODEL_ARTIFACTS.json").is_file():
        raise FileNotFoundError("formal v2 fit artifacts are required")
    fit_marker = json.loads((formal / "FIT_COMPLETE.json").read_text(encoding="utf-8"))
    if fit_marker.get("status") != "PASS" or fit_marker.get("confirmation_used_for_selection") is not False:
        raise RuntimeError("formal fit marker is not sealed")
    args_rf = SimpleNamespace(prepare_root=args.prepare_root, protocol_file=RF.HERE / "PROTOCOL.md")
    prep = RF.require_prepare(args_rf)
    rows = RF.load_filtered_rows(args.prepare_root, prep["ids"], prep["marker"])
    virtual = RF.load_virtual_filtered(args.model_h5, args.aggregate_npz, prep["ids"])
    bundles = {b: {s: SRC.build_dual_examples(rows[s], b, include_foreign=False) for s in ("fit", "validation")} for b in BUDGETS}
    device = device_for(args.device)
    root = args.output_root
    rows_out: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    for budget in BUDGETS:
        train_targets, valid_targets = load_targets(formal, budget)
        fit_ex = bundles[budget]["fit"]; val_ex = bundles[budget]["validation"]
        if not np.array_equal(train_targets["compound"].astype(str), fit_ex.compound.astype(str)):
            raise RuntimeError(f"training target row order mismatch at budget {budget}")
        if not np.array_equal(valid_targets["compound"].astype(str), val_ex.compound.astype(str)):
            raise RuntimeError(f"validation target row order mismatch at budget {budget}")
        bdir = root / f"budget{budget}"; bdir.mkdir()
        artifacts[str(budget)] = {}
        for method in METHODS:
            artifacts[str(budget)][method] = {}
        for master in SEEDS:
            train_raw = RF.subset_virtual(virtual["fit"], fit_ex, train_targets["raw"])
            valid_raw = RF.subset_virtual(virtual["validation"], val_ex, valid_targets["raw_held_mean"])
            seed = seed_for(master, budget, "rawwide")
            ck, pred, loss, epoch = fit_single(train_raw, valid_raw, train_targets["raw"], valid_targets["raw_held_mean"], WIDE_DIMS, seed, args, device)
            path = bdir / "RawWide" / f"seed{master}.pt"; path.parent.mkdir(exist_ok=True)
            torch.save({"version": VERSION, "method": "RawWide", "master_seed": master, "hidden_dims": WIDE_DIMS, "parameter_count": parameter_count(WIDE_DIMS), **ck}, path)
            artifacts[str(budget)]["RawWide"][str(master)] = [str(path.relative_to(root))]
            rows_out.append({"budget": budget, "method": "RawWide", "master_seed": master, "parameter_count": parameter_count(WIDE_DIMS), "hidden_dims": str(WIDE_DIMS), "best_epoch": epoch, "validation_mse_original": loss, "validation_pcc_mean": float(np.nanmean(LEGACY.rowwise_corr(pred, valid_targets["raw_held_mean"])))})
            seed = seed_for(master, budget, "rawtwobranch")
            ck, pred, loss, epoch = fit_joint_raw(train_raw, valid_raw, train_targets["raw"], valid_targets["raw_held_mean"], seed, args, device)
            path = bdir / "RawTwoBranch" / f"seed{master}.pt"; path.parent.mkdir(exist_ok=True)
            torch.save({"version": VERSION, "method": "RawTwoBranch", "master_seed": master, "hidden_dims": BASE_DIMS, "parameter_count": 2 * parameter_count(BASE_DIMS), **ck}, path)
            artifacts[str(budget)]["RawTwoBranch"][str(master)] = [str(path.relative_to(root))]
            rows_out.append({"budget": budget, "method": "RawTwoBranch", "master_seed": master, "parameter_count": 2 * parameter_count(BASE_DIMS), "hidden_dims": str(BASE_DIMS), "best_epoch": epoch, "validation_mse_original": loss, "validation_pcc_mean": float(np.nanmean(LEGACY.rowwise_corr(pred, valid_targets["raw_held_mean"])))})
            cfra_train = RF.subset_virtual(virtual["fit"], fit_ex, train_targets["cfra"])
            cfra_valid = RF.subset_virtual(virtual["validation"], val_ex, valid_targets["cfra_teacher"])
            residual_train = RF.subset_virtual(virtual["fit"], fit_ex, train_targets["residual"])
            residual_valid = RF.subset_virtual(virtual["validation"], val_ex, valid_targets["residual"])
            small_paths = []
            for branch, tr, va, target, objective in (("cfra", cfra_train, cfra_valid, train_targets["cfra"], valid_targets["cfra_teacher"]), ("residual", residual_train, residual_valid, train_targets["residual"], valid_targets["residual"])):
                seed = seed_for(master, budget, f"small-{branch}")
                ck, pred, loss, epoch = fit_single(tr, va, target, objective, SMALL_DIMS, seed, args, device)
                path = bdir / "CFRADecompSmall" / f"{branch}_seed{master}.pt"; path.parent.mkdir(exist_ok=True)
                torch.save({"version": VERSION, "method": "CFRADecompSmall", "branch": branch, "master_seed": master, "hidden_dims": SMALL_DIMS, "parameter_count": parameter_count(SMALL_DIMS), **ck}, path)
                small_paths.append(str(path.relative_to(root)))
                rows_out.append({"budget": budget, "method": "CFRADecompSmall", "branch": branch, "master_seed": master, "parameter_count": parameter_count(SMALL_DIMS), "hidden_dims": str(SMALL_DIMS), "best_epoch": epoch, "validation_mse_original": loss, "validation_pcc_mean": float(np.nanmean(LEGACY.rowwise_corr(pred, objective)))})
            artifacts[str(budget)]["CFRADecompSmall"][str(master)] = small_paths
            del train_raw, valid_raw, cfra_train, cfra_valid, residual_train, residual_valid
            if device.type == "cuda": torch.cuda.empty_cache()
    write_csv(root / "CAPACITY_VALIDATION.csv", rows_out)
    dump(root / "MODEL_ARTIFACTS.json", artifacts)
    hashes = {}
    for budget in BUDGETS:
        for method in METHODS:
            for seed, paths in artifacts[str(budget)][method].items():
                for path in paths:
                    hashes[path] = sha256_file(root / path)
    dump(root / "CHECKPOINT_HASHES.json", hashes)
    audit = {"version": VERSION, "phase": "fit", "status": "PASS", "confirmation_loaded": False, "confirmation_values_opened": False, "confirmation_used_for_selection": False, "formal_fit_root": str(formal), "prepare_root": str(args.prepare_root), "formal_fit_marker_sha256": sha256_file(formal / "FIT_COMPLETE.json"), "protocol_sha256": sha256_file(PROTOCOL), "methods": list(METHODS), "budgets": list(BUDGETS), "master_seeds": list(SEEDS), "parameter_counts": {"base": parameter_count(BASE_DIMS), "wide": parameter_count(WIDE_DIMS), "small_branch": parameter_count(SMALL_DIMS)}, "checkpoint_count": len(hashes)}
    dump(root / "FIT_COMPLETE.json", audit)
    return audit


def load_custom_prediction(path: Path, kind: str, virtual: Any, names: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    ck = torch.load(path, map_location=device, weights_only=False)
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in ck["stats"].items()}
    if kind == "RawTwoBranch":
        a = StudentWidth(*BASE_DIMS).to(device); b = StudentWidth(*BASE_DIMS).to(device)
        a.load_state_dict(ck["state_dict_a"]); b.load_state_dict(ck["state_dict_b"])
        return predict_joint((a, b), virtual, stats, batch, device)
    dims = tuple(int(x) for x in ck["hidden_dims"])
    model = StudentWidth(*dims).to(device); model.load_state_dict(ck["state_dict"])
    return predict_single(model, virtual, stats, batch, device)


def compound_average(values: list[np.ndarray]) -> np.ndarray:
    return np.nanmean(np.stack(values, axis=0).astype(np.float64), axis=0)


def bootstrap(left: np.ndarray, right: np.ndarray, seed: int, rounds: int) -> dict[str, Any]:
    finite = np.isfinite(left) & np.isfinite(right); left = np.asarray(left)[finite]; right = np.asarray(right)[finite]
    diff = left - right
    rng = np.random.default_rng(seed); draws = np.mean(diff[rng.integers(0, len(diff), size=(rounds, len(diff)))], axis=1)
    return {"estimate": float(np.mean(diff)), "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)), "n_compounds": int(len(diff)), "rounds": rounds}


def confirm_phase(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root
    fit_marker = json.loads((root / "FIT_COMPLETE.json").read_text(encoding="utf-8"))
    if fit_marker.get("status") != "PASS" or fit_marker.get("confirmation_loaded"):
        raise RuntimeError("capacity fit marker is not sealed")
    spec2 = importlib.util.spec_from_file_location("bbbc047_capacity_confirm", CONFIRM_SCRIPT)
    if spec2 is None or spec2.loader is None: raise RuntimeError("cannot import confirmation evaluator")
    RC = importlib.util.module_from_spec(spec2); sys.modules[spec2.name] = RC; spec2.loader.exec_module(RC)
    formal = args.formal_fit_root; names_set = RC._confirmation_compounds(formal)
    ns = SimpleNamespace(rows_root=args.rows_root, smoke=False)
    source = RC._combine_source_rows(ns, True); confirmation_rows = RC.PREP.subset(source, names_set, "confirmation")
    bundles = {b: SRC.build_dual_examples(confirmation_rows, b, include_foreign=True) for b in BUDGETS}
    all_names = np.asarray(sorted(names_set), dtype=str)
    virtual_all = RC._load_virtual_confirmation(args.model_h5, args.aggregate_npz, all_names)
    vi = {str(n): i for i, n in enumerate(virtual_all.smiles)}
    device = device_for(args.device); formal_index = RC._artifact_index(formal)
    methods = ("M0", "M2", "M4", "RawWide", "RawTwoBranch", "CFRADecompSmall")
    metrics: dict[tuple[int, str, int], dict[str, np.ndarray]] = {}
    index = json.loads((root / "MODEL_ARTIFACTS.json").read_text(encoding="utf-8"))
    for budget in BUDGETS:
        ex = bundles[budget]; names = np.asarray(ex.compound, dtype=str); ix = np.asarray([vi[str(n)] for n in names]); virtual = type("V", (), {"smiles": names, "compound": names, "control": virtual_all.control[ix], "fingerprint": virtual_all.fingerprint[ix]})()
        for method in methods:
            for seed in SEEDS:
                if method in ("M0", "M2", "M4"):
                    pred, _ = RC._method_predictions(formal_index, budget, method, seed, virtual, names, device, args.batch_size)
                elif method == "RawWide":
                    pred = load_custom_prediction(root / index[str(budget)][method][str(seed)][0], method, virtual, names, device, args.batch_size)
                elif method == "RawTwoBranch":
                    pred = load_custom_prediction(root / index[str(budget)][method][str(seed)][0], method, virtual, names, device, args.batch_size)
                else:
                    paths = index[str(budget)][method][str(seed)]
                    p0 = load_custom_prediction(root / paths[0], method, virtual, names, device, args.batch_size)
                    p1 = load_custom_prediction(root / paths[1], method, virtual, names, device, args.batch_size)
                    pred = p0 + p1
                metrics[(budget, method, seed)] = RC._metrics(pred, ex)
    summary=[]; arrays={}
    for b in BUDGETS:
        for method in methods:
            A_A=compound_average([metrics[(b,method,s)]["A_A"] for s in SEEDS]); A_B=compound_average([metrics[(b,method,s)]["A_B"] for s in SEEDS]); E=compound_average([metrics[(b,method,s)]["E_dual"] for s in SEEDS]);
            arrays[(b,method,"A")] = (A_A + A_B)/2; arrays[(b,method,"E")] = E
            arrays[(b,method,"E_A")] = compound_average([metrics[(b,method,s)]["E_A"] for s in SEEDS]); arrays[(b,method,"E_B")] = compound_average([metrics[(b,method,s)]["E_B"] for s in SEEDS])
            summary.append({"budget":b,"method":method,"A_A_mean":float(np.nanmean(A_A)),"A_B_mean":float(np.nanmean(A_B)),"A_mean":float(np.nanmean((A_A+A_B)/2)),"E_mean":float(np.nanmean(E)),"E_A_mean":float(np.nanmean(arrays[(b,method,"E_A")])),"E_B_mean":float(np.nanmean(arrays[(b,method,"E_B")])),"held_pcc_A_mean":float(np.nanmean(compound_average([metrics[(b,method,s)]["held_pcc_A"] for s in SEEDS]))),"held_pcc_B_mean":float(np.nanmean(compound_average([metrics[(b,method,s)]["held_pcc_B"] for s in SEEDS])))})
    contrasts=[]; pairs=(("RawWide-M0","RawWide","M0"),("RawTwoBranch-M0","RawTwoBranch","M0"),("CFRADecompSmall-M0","CFRADecompSmall","M0"),("M4-RawWide","M4","RawWide"),("M4-RawTwoBranch","M4","RawTwoBranch"),("M4-CFRADecompSmall","M4","CFRADecompSmall"))
    for b in BUDGETS:
        for label,left,right in pairs:
            for metric in ("A","E","E_A","E_B"):
                seed=LEGACY.stable_int(3407,f"{VERSION}|b{b}|{label}|{metric}")
                result=bootstrap(arrays[(b,left,metric)],arrays[(b,right,metric)],seed,args.bootstrap_rounds)
                contrasts.append({"budget":b,"comparison":label,"metric":metric,**result})
    write_csv(root/"CAPACITY_SUMMARY.csv",summary); write_csv(root/"CAPACITY_CONTRASTS.csv",contrasts)
    audit={"version":VERSION,"phase":"confirm","status":"PASS","exploratory_posthoc":True,"confirmation_loaded":True,"confirmation_values_opened":True,"confirmation_used_for_selection":False,"formal_fit_root":str(formal),"fit_marker_sha256":sha256_file(root/"FIT_COMPLETE.json"),"protocol_sha256":sha256_file(PROTOCOL),"methods":list(methods),"budgets":list(BUDGETS),"bootstrap_rounds":args.bootstrap_rounds}
    dump(root/"CONFIRMATION_COMPLETE.json",audit); return audit


def main() -> None:
    args = parse_args()
    result = fit_phase(args) if args.phase == "fit" else confirm_phase(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
