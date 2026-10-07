#!/usr/bin/env python3
"""sci-Plex3 adaptation of the BBBC047 P0 contract.

The data have two genuine repeats, so Raw-Aggregate and CFRA-1R-Aggregate are
formed at each (drug,dose) from both repeat views.  M0 is a two-branch
Raw-Aggregate Student, M2 is CFRA-Aggregate base plus Raw-space residual, and
M1 is the two-branch direct CFRA-Aggregate diagnostic.  The implementation
reuses the audited sci-Plex3 data reader and OOF Ridge teacher.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OLD_PATH = (Path(__file__).resolve().parents[4] / "analysis/predictor_supervision/sci_plex3/support/run_own_cell_unseen.py")
spec = importlib.util.spec_from_file_location("sciplex_old", OLD_PATH)
old = importlib.util.module_from_spec(spec); sys.modules["sciplex_old"] = old; assert spec and spec.loader; spec.loader.exec_module(old)

VERSION = "sciplex3-bbbc047-p0-m0-m1-m2-v3-surface-isolated-2026-09-15"
SEEDS = (3407, 42, 2025, 1337, 7331)
SURFACES = ("cross_cell_seen", "cross_cell_unseen", "own_cell_unseen")
METHODS = ("M0_RawAggregate", "M1_CFRA1Aggregate", "M2_CFRA1Residual")
METHOD_KINDS = {
    "M0_RawAggregate": ("raw", "raw"),
    "M1_CFRA1Aggregate": ("cfra", "cfra"),
    "M2_CFRA1Residual": ("cfra", "residual"),
}
BOOTSTRAP_ROUNDS = 10000
# The sci-Plex3 input has exactly two physical repeats.  We retain the
# protocol's held-A/B names for the repeat-wise descriptive diagnostics, but
# never present them as two independent validation estimands.
REPEAT_LABELS = (("A", "rep1"), ("B", "rep2"))
HELD_METRIC_STATUS = "DESCRIPTIVE_ACTUAL_TWO_REPEAT_CORRELATION_NONINDEPENDENT"
BOOTSTRAP_METRICS = (
    # Aggregate-view held diagnostics.
    "A", "z_same_F", "z_foreign_F", "E", "held_pcc_A", "held_pcc_B",
    # Existing common Raw-Aggregate endpoints (and the existing MSE audit).
    "delta_pcc", "full_target_pcc", "raw_target_mse",
)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text("\n", encoding="utf-8"); return
    keys = list(rows[0]);
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("preflight", "dry_run", "fit", "test"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--surface", choices=SURFACES, required=True, help="One surface per root; prevents cross-surface confirmation exposure before its checkpoint freeze.")
    p.add_argument("--conditions", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/experiments/external_validation_sciplex3_papalexi_20260903/sciplex3_validation/eligible_conditions_n50.csv"))
    p.add_argument("--group-counts", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/experiments/external_validation_sciplex3_papalexi_20260903/sciplex3_validation/group_counts.h5"))
    p.add_argument("--split-lock", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/experiments/external_validation_sciplex3_papalexi_20260903/sciplex3_validation/cfra_confirmation/split_lock.json"))
    p.add_argument("--structure-map", type=Path, default=Path("/path/to/data/AIDD/MVCPert_5_27/experiments/external_validation_sciplex3_papalexi_20260903/sciplex3_student_crosscell/input_preflight/DRUG_ECFP4_MAPPING.npz"))
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p.parse_args()

def device(name: str) -> torch.device:
    return old.choose_device(name)

def surface_spec(surface: str, target: str, drugs: list[str], split: dict[str, list[str]]) -> dict[str, Any]:
    if surface == "cross_cell_seen":
        return {"source_cell_lines": [x for x in old.CELL_LINES if x != target], "train_drugs": sorted(set(split["base"]) & set(drugs)), "valid_drugs": sorted(set(split["calibration"]) & set(drugs)), "refit_drugs": drugs, "test_drugs": drugs, "foreign_drugs": drugs, "status": "HISTORICAL_LOCKED_REUSE"}
    if surface == "cross_cell_unseen":
        base_cal = sorted((set(split["base"]) | set(split["calibration"])) & set(drugs))
        return {"source_cell_lines": [x for x in old.CELL_LINES if x != target], "train_drugs": sorted(set(split["base"]) & set(drugs)), "valid_drugs": sorted(set(split["calibration"]) & set(drugs)), "refit_drugs": base_cal, "test_drugs": sorted(set(split["confirmation"]) & set(drugs)), "foreign_drugs": base_cal, "status": "HISTORICAL_LOCKED_REUSE"}
    base_cal = sorted((set(split["base"]) | set(split["calibration"])) & set(drugs))
    return {"source_cell_lines": [target], "train_drugs": sorted(set(split["base"]) & set(drugs)), "valid_drugs": sorted(set(split["calibration"]) & set(drugs)), "refit_drugs": base_cal, "test_drugs": sorted(set(split["confirmation"]) & set(drugs)), "foreign_drugs": base_cal, "status": "RETROSPECTIVE_SUPPORTIVE"}

def preflight(a: argparse.Namespace) -> None:
    if a.root.exists() and any(a.root.iterdir()): raise FileExistsError(a.root)
    rows = old.read_conditions(a.conditions); row_map = old.build_row_map(rows)
    units, drugs = old.common_units(rows); split = old.read_split(a.split_lock)
    fps, canonical, smeta = old.load_structure_map(a.structure_map)
    drugs = sorted(set(drugs) & set(fps)); units = [(d, float(x)) for d, x in units if d in drugs]
    if len(drugs) < 100: raise RuntimeError("coverage gate failed")
    for key, row in row_map.items():
        if row.group_id < 0 or row.control_group_id < 0 or not row.plate: raise RuntimeError(f"bad physical identity {key}")
    surfaces = (a.surface,)
    folds = {t: {s: surface_spec(s, t, drugs, split) for s in surfaces} for t in old.CELL_LINES}
    plan = {"version": VERSION, "stage": "preflight", "dataset": "sci-Plex3 Zenodo 7041849", "cell_lines": list(old.CELL_LINES), "repeats": list(old.REPS), "n_common_drugs": len(drugs), "units": [[d, x] for d, x in units], "split_lock": split, "surfaces": list(surfaces), "surface_isolation": "one surface per root; a target confirmation profile cannot be opened by another surface before this root freezes its checkpoints", "folds": folds, "methods": list(METHODS), "seeds": list(SEEDS), "teacher_regime": "fit-compound/drug-OOF Ridge support-to-held, applied separately to both repeats then averaged", "student_input": "ECFP4-2048 + log10 dose + mean of the two same-condition Vehicle controls", "raw_aggregate": "mean(response(rep1), response(rep2)) at each drug-dose", "cfra_aggregate": "mean(teacher(response(rep1)), teacher(response(rep2))) at each drug-dose", "m2": "base(CFRA1Aggregate) + correction(RawAggregate-CFRA1Aggregate)", "m0": "mean of two RawAggregate branches", "m1": "mean of two CFRA1Aggregate branches", "evaluation_contract": {"common_raw_target": ["delta_pcc", "full_target_pcc", "raw_target_mse"], "two_repeat_diagnostics": ["A", "z_same_F", "z_foreign_F", "E", "held_pcc_A", "held_pcc_B"], "repeat_status": HELD_METRIC_STATUS, "foreign_rule": "same target cell-line and dose; exclude query drug; calculate Fisher-z separately for every foreign donor and repeat, then mean"}, "confirmation_treated_loaded": False, "inputs": {"conditions": {"path": str(a.conditions), "sha256": sha256(a.conditions)}, "group_counts": {"path": str(a.group_counts), "sha256": sha256(a.group_counts)}, "split_lock": {"path": str(a.split_lock), "sha256": sha256(a.split_lock)}, "structure_map": {"path": str(a.structure_map), "sha256": sha256(a.structure_map), **smeta}, "old_runner": {"path": str(OLD_PATH.resolve()), "sha256": sha256(OLD_PATH)}, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}}}
    a.root.mkdir(parents=True); write_json(a.root / "PLAN.json", plan)
    write_csv(a.root / "TARGET_VIEW_MANIFEST.csv", [{"target": t, "surface": s, "n_units": len(units), "repeat_views": 2, "raw_aggregation": "rep1+rep2 mean", "cfra_aggregation": "teacher(rep1)+teacher(rep2) mean", "status": folds[t][s]["status"]} for t in old.CELL_LINES for s in surfaces])
    write_json(a.root / "PREPARE_COMPLETE.json", {"status": "PASS", "confirmation_treated_loaded": False, "n_drugs": len(drugs), "n_units": len(units), "surfaces": list(surfaces), "surface_isolation": True, "evaluation_metrics": list(BOOTSTRAP_METRICS), "repeat_status": HELD_METRIC_STATUS})
    print(json.dumps({"status": "PREFLIGHT_COMPLETE", "n_drugs": len(drugs), "n_units": len(units), "confirmation_treated_loaded": False}))

def load_plan(a: argparse.Namespace) -> dict[str, Any]:
    p = json.loads((a.root / "PLAN.json").read_text(encoding="utf-8"))
    if p.get("version") != VERSION or p.get("confirmation_treated_loaded") is not False: raise RuntimeError("plan drift/leakage")
    if p["inputs"]["runner"]["sha256"] != sha256(Path(__file__).resolve()): raise RuntimeError("runner hash drift")
    if p["inputs"]["old_runner"]["sha256"] != sha256(OLD_PATH): raise RuntimeError("imported helper hash drift")
    for name, path in (("conditions", a.conditions), ("group_counts", a.group_counts), ("split_lock", a.split_lock), ("structure_map", a.structure_map)):
        if p["inputs"][name]["sha256"] != sha256(path): raise RuntimeError(f"{name} hash drift")
    return p

def aggregate_rows(profile: dict, cells: list[str], drugs: set[str], units: list[tuple[str, float]], *, truth: bool = True):
    out = []
    for cell in cells:
        for drug, dose in units:
            if drug not in drugs: continue
            r1 = old.response(profile, cell, drug, dose, "rep1"); r2 = old.response(profile, cell, drug, dose, "rep2")
            b1 = old.baseline(profile, cell, drug, dose, "rep1"); b2 = old.baseline(profile, cell, drug, dose, "rep2")
            raw = (r1 + r2) / 2.0; base = (b1 + b2) / 2.0
            out.append((cell, drug, float(dose), base, raw if truth else np.zeros_like(raw), raw))
    if not out: raise RuntimeError("empty aggregate rows")
    return old.StudentRows(np.asarray([x[0] for x in out], str), np.asarray([x[1] for x in out], str), np.asarray([x[2] for x in out], np.float32), np.vstack([x[3] for x in out]).astype(np.float32), np.vstack([x[4] for x in out]).astype(np.float32), np.vstack([x[5] for x in out]).astype(np.float32))

def cfra_agg(profile: dict, cells: list[str], fit_drugs: set[str], cal_drugs: set[str], units: list[tuple[str, float]], token: str, oof: bool) -> tuple[old.StudentRows, dict[str, Any]]:
    labels = {}; audits = []
    if oof:
        assignment = old.teacher_fold_assignment(fit_drugs)
        for fold in sorted(set(assignment.values())):
            held = {d for d, k in assignment.items() if k == fold}
            for cell in cells:
                teacher, meta = old.fit_cfra_teacher_fold(profile, cell, fit_drugs, cal_drugs, held, token=f"{token}|{cell}|fold{fold}"); audits.append({"fold": fold, **meta})
                for drug, dose in units:
                    if drug not in held: continue
                    vals = [old.teacher_predict(teacher, old.response(profile, cell, drug, dose, rep)) for rep in old.REPS]
                    labels[(cell, drug, float(dose))] = np.mean(np.stack(vals), axis=0).astype(np.float32)
    else:
        for cell in cells:
            teacher, meta = old.fit_cfra_teacher_fold(profile, cell, fit_drugs, cal_drugs, set(), token=f"{token}|{cell}|validation"); audits.append({"fold": "validation", **meta})
            for drug, dose in units:
                if drug not in cal_drugs: continue
                vals = [old.teacher_predict(teacher, old.response(profile, cell, drug, dose, rep)) for rep in old.REPS]
                labels[(cell, drug, float(dose))] = np.mean(np.stack(vals), axis=0).astype(np.float32)
    rows = aggregate_rows(profile, cells, set(fit_drugs if oof else cal_drugs), units, truth=False)
    ordered = [labels[(c, d, float(q))] for c, d, q in zip(rows.cell, rows.drug, rows.dose)]
    return old.StudentRows(rows.cell, rows.drug, rows.dose, rows.baseline, np.vstack(ordered).astype(np.float32), rows.truth), {"folds": audits, "oof": oof}

def make_stats(rows, label: np.ndarray) -> dict[str, np.ndarray]:
    d = old.dose_values(rows.dose)
    return {"baseline_mean": rows.baseline.mean(0).astype(np.float32), "baseline_scale": np.maximum(rows.baseline.std(0), 1e-6).astype(np.float32), "dose_mean": d.mean(0).astype(np.float32), "dose_scale": np.maximum(d.std(0), 1e-6).astype(np.float32), "target_mean": label.mean(0).astype(np.float32), "target_scale": np.maximum(label.std(0), 1e-6).astype(np.float32)}

def effective_rank(values: np.ndarray) -> float:
    """Deterministic 256-row entropy effective-rank audit via row Gram."""
    x = np.asarray(values, dtype=np.float64)
    if x.shape[0] > 256:
        x = x[np.linspace(0, x.shape[0] - 1, 256, dtype=np.int64)]
    x = x - x.mean(axis=0, keepdims=True)
    eig = np.linalg.eigvalsh(x @ x.T)
    eig = np.maximum(eig, 0.0)
    total = float(eig.sum())
    if total <= 1e-12: return 0.0
    prob = eig[eig > total * 1e-14] / total
    return float(np.exp(-np.sum(prob * np.log(prob))))

def target_quality(scope: str, rows, cfra: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = np.asarray(rows.raw_label, dtype=np.float32)
    per_row = []
    for cell, drug, dose, r, q in zip(rows.cell, rows.drug, rows.dose, raw, cfra):
        per_row.append({"scope":scope,"cell_line":str(cell),"drug":str(drug),"dose_nM":float(dose),"raw_cfra_pcc":old.corr(r,q),"cfra_to_raw_effect_norm_ratio":float(np.linalg.norm(q) / max(np.linalg.norm(r),1e-12))})
    return per_row, {"scope":scope,"n_rows":len(per_row),"effective_rank_sample_rows":min(len(per_row),256),"mean_raw_cfra_pcc":float(np.mean([x["raw_cfra_pcc"] for x in per_row])),"mean_cfra_to_raw_effect_norm_ratio":float(np.mean([x["cfra_to_raw_effect_norm_ratio"] for x in per_row])),"raw_entropy_effective_rank":effective_rank(raw),"cfra_entropy_effective_rank":effective_rank(cfra)}

def fit_one(train, valid, label, valid_label, seed, stats, fps, dev, path: Path, epochs: int | None = None) -> tuple[int, str]:
    v = old.StudentRows(valid.cell, valid.drug, valid.dose, valid.baseline, valid.raw_label, valid_label)
    model, logs, best, init = old.fit_student(train, v, label, fps, stats, seed, dev, fixed_epochs=epochs)
    payload = {"version": VERSION, "seed": seed, "fixed_epochs": epochs, "best_epoch": best, "state_dict": model.state_dict()}
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, path); old.write_csv(path.parent / "training_log.csv", logs)
    return best, init

def model(path: Path, dev: torch.device) -> old.Student:
    z = torch.load(path, map_location="cpu", weights_only=False); m = old.Student().to(dev); m.load_state_dict(z["state_dict"]); m.eval(); return m

def fit(a: argparse.Namespace) -> None:
    p = load_plan(a); surfaces = tuple(p["surfaces"]); dev = device(a.device); rows = old.build_row_map(old.read_conditions(a.conditions)); units = [(str(x[0]), float(x[1])) for x in p["units"]]; fps, _, _ = old.load_structure_map(a.structure_map); store = old.ProfileStore(a.group_counts); audit = {"status": "RUNNING", "methods": list(METHODS), "surfaces": list(surfaces), "seeds": list(SEEDS), "checkpoints": [], "artifacts": [], "confirmation_treated_loaded": False, "surface_isolation": True}
    try:
        for target in old.CELL_LINES:
            for surface in surfaces:
                spec0 = p["folds"][target][surface]; cells = list(spec0["source_cell_lines"]); train_drugs, cal_drugs, refit_drugs = set(spec0["train_drugs"]), set(spec0["valid_drugs"]), set(spec0["refit_drugs"])
                hvg, hvg_audit = old.fit_vehicle_hvg(store, rows, cells); profile = old.load_source_profiles(store, rows, cells, refit_drugs, hvg, set(units)); tr_raw = aggregate_rows(profile, cells, train_drugs, units); va_raw = aggregate_rows(profile, cells, cal_drugs, units)
                tr_cfra, tmeta = cfra_agg(profile, cells, train_drugs, cal_drugs, units, f"{target}|{surface}|initial", True); va_cfra, _ = cfra_agg(profile, cells, train_drugs, cal_drugs, units, f"{target}|{surface}|valid", False)
                labels = {"raw": tr_raw.raw_label, "cfra": tr_cfra.raw_label, "residual": tr_raw.raw_label - tr_cfra.raw_label}; vlabels = {"raw": va_raw.raw_label, "cfra": va_cfra.raw_label, "residual": va_raw.raw_label - va_cfra.raw_label}; stats = {k: make_stats(tr_raw, v) for k, v in labels.items()}
                refit_profile = profile; refit_raw = aggregate_rows(refit_profile, cells, refit_drugs, units); refit_cfra, refmeta = cfra_agg(refit_profile, cells, refit_drugs, cal_drugs, units, f"{target}|{surface}|refit", True); refit_labels = {"raw": refit_raw.raw_label, "cfra": refit_cfra.raw_label, "residual": refit_raw.raw_label - refit_cfra.raw_label}
                ep = a.root / "fit" / surface / target; ep.mkdir(parents=True, exist_ok=True); np.savetxt(ep / "source_vehicle_hvg2000.txt", hvg, fmt="%d"); np.savez_compressed(ep / "targets.npz", raw=refit_labels["raw"], cfra=refit_labels["cfra"], residual=refit_labels["residual"]); [np.savez_compressed(ep / f"stats_{k}.npz", **stats[k]) for k in stats]; write_json(ep / "teacher_audit.json", {"initial": tmeta, "refit": refmeta, "aggregation": "both repeats independently teacher-predicted then averaged", "confirmation_treated_loaded": False})
                quality_rows, quality_fit = target_quality("initial_fit", tr_raw, tr_cfra.raw_label)
                quality_valid_rows, quality_valid = target_quality("validation", va_raw, va_cfra.raw_label)
                quality_refit_rows, quality_refit = target_quality("refit", refit_raw, refit_cfra.raw_label)
                write_csv(ep / "TARGET_QUALITY.csv", quality_rows + quality_valid_rows + quality_refit_rows)
                write_json(ep / "TARGET_AUDIT.json", {"initial_fit":quality_fit,"validation":quality_valid,"refit":quality_refit,"teacher_regime":"drug-OOF for fit/refit; fit-only teacher for validation audit","confirmation_treated_loaded":False})
                selected = {}
                selection_rows = []
                for method in METHODS:
                    kinds = METHOD_KINDS[method]
                    for seed in SEEDS:
                        branch_epochs = []
                        for branch, kind in enumerate(kinds):
                            s = seed + branch * 1000003; bpath = ep / "initial" / method / f"seed_{seed}" / f"branch_{branch}.pt"; best, init = fit_one(old.StudentRows(tr_raw.cell, tr_raw.drug, tr_raw.dose, tr_raw.baseline, labels[kind], tr_raw.truth), old.StudentRows(va_raw.cell, va_raw.drug, va_raw.dose, va_raw.baseline, vlabels[kind], va_raw.truth), labels[kind], vlabels[kind], s, stats[kind], fps, dev, bpath); branch_epochs.append(best)
                            initial_model = model(bpath, dev); pred = old.predict_model(initial_model, va_raw, fps, stats[kind], dev)
                            selection_rows.append({"target":target,"surface":surface,"method":method,"seed":seed,"branch":branch,"target_kind":kind,"selected_epoch":best,"fixed_refit_epochs":best+1,"selection_target_mse":float(np.mean((pred-vlabels[kind])**2)),"raw_aggregate_validation_mse":float(np.mean((pred-va_raw.raw_label)**2)),"checkpoint_sha256":sha256(bpath),"confirmation_treated_loaded":False})
                        selected[f"{method}|{seed}"] = branch_epochs
                for method in METHODS:
                    kinds = METHOD_KINDS[method]
                    for seed in SEEDS:
                        for branch, kind in enumerate(kinds):
                            epochs = selected[f"{method}|{seed}"][branch] + 1; s = seed + branch * 1000003; bpath = ep / "refit" / method / f"seed_{seed}" / f"branch_{branch}.pt"; fit_one(old.StudentRows(refit_raw.cell, refit_raw.drug, refit_raw.dose, refit_raw.baseline, refit_labels[kind], refit_raw.truth), va_raw, refit_labels[kind], vlabels[kind], s, stats[kind], fps, dev, bpath, epochs=epochs); audit["checkpoints"].append({"target": target, "surface": surface, "method": method, "seed": seed, "branch": branch, "path": str(bpath.resolve()), "sha256": sha256(bpath), "epochs": epochs, "confirmation_treated_loaded": False})
                write_csv(ep / "initial_epoch_selection.csv", selection_rows)
                artifact_paths = [ep / "source_vehicle_hvg2000.txt", ep / "targets.npz", ep / "teacher_audit.json", ep / "TARGET_QUALITY.csv", ep / "TARGET_AUDIT.json", ep / "initial_epoch_selection.csv", *(ep / f"stats_{k}.npz" for k in ("raw", "cfra", "residual"))]
                audit["artifacts"].extend({"target":target,"surface":surface,"path":str(q.resolve()),"sha256":sha256(q)} for q in artifact_paths)
    finally: audit["source_group_reads"] = sorted(set(store.read_group_ids)); audit["source_control_reads"] = sorted(set(store.read_control_ids)); store.close()
    if len(audit["checkpoints"]) != len(old.CELL_LINES) * len(surfaces) * len(METHODS) * len(SEEDS) * 2: raise RuntimeError("checkpoint coverage incomplete")
    if len(audit["artifacts"]) != len(old.CELL_LINES) * len(surfaces) * 9: raise RuntimeError("artifact coverage incomplete")
    audit["status"] = "PASS"; write_json(a.root / "FIT_COMPLETE.json", audit); print(json.dumps({"status": "FIT_COMPLETE", "checkpoints": len(audit["checkpoints"]), "confirmation_treated_loaded": False}))

def paired_boot(raw: dict[str, float], other: dict[str, float], token: str) -> dict[str, Any]:
    keys = sorted(k for k in (set(raw) & set(other)) if np.isfinite(raw[k]) and np.isfinite(other[k]))
    if not keys:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_drugs": 0, "bootstrap_draws": BOOTSTRAP_ROUNDS, "bootstrap_unit": "paired_drug"}
    x = np.asarray([other[k] - raw[k] for k in keys], dtype=np.float64)
    rng = np.random.default_rng(old.stable_int(3407, "p0", token))
    d = x[rng.integers(0, len(x), (BOOTSTRAP_ROUNDS, len(x)))].mean(1)
    return {"estimate": float(x.mean()), "ci_low": float(np.quantile(d, .025)), "ci_high": float(np.quantile(d, .975)), "n_drugs": len(keys), "bootstrap_draws": BOOTSTRAP_ROUNDS, "bootstrap_unit": "paired_drug"}

def two_repeat_metrics(pred: np.ndarray, held: dict[str, np.ndarray], foreign: list[dict[str, np.ndarray]]) -> dict[str, float]:
    """Contracted dual-repeat diagnostics for a Raw-Aggregate prediction.

    The two sci-Plex3 views are real `(drug,dose,plate)` repeats.  They are
    reported separately as held A/B, but must not be treated as independent
    confirmation cohorts.  Foreign scores use exactly the same query/dose
    units as `z_same_F`: per donor and per repeat Fisher-z first, then mean.
    """
    held_pcc = {label: old.corr(pred, held[rep]) for label, rep in REPEAT_LABELS}
    held_z = {label: float(np.arctanh(np.clip(held_pcc[label], -0.999999, 0.999999))) for label, _ in REPEAT_LABELS}
    answer = {
        "held_pcc_A": held_pcc["A"], "held_pcc_B": held_pcc["B"],
        "z_same_A": held_z["A"], "z_same_B": held_z["B"],
        "A": float(np.mean([held_z["A"], held_z["B"]])),
        "z_same_F": float("nan"), "z_foreign_F": float("nan"), "E": float("nan"),
        "foreign_donor_count": float(len(foreign)),
    }
    if not foreign:
        return answer
    foreign_z = []
    for donor in foreign:
        foreign_z.append(float(np.mean([
            np.arctanh(np.clip(old.corr(pred, donor[rep]), -0.999999, 0.999999))
            for _, rep in REPEAT_LABELS
        ])))
    same_f = answer["A"]
    foreign_f = float(np.mean(foreign_z))
    answer.update({"z_same_F": same_f, "z_foreign_F": foreign_f, "E": same_f - foreign_f})
    return answer

def mean_metric(values: list[dict[str, Any]], name: str) -> float:
    finite = [float(v[name]) for v in values if np.isfinite(float(v[name]))]
    return float(np.mean(finite)) if finite else float("nan")

def test(a: argparse.Namespace) -> None:
    p = load_plan(a); surfaces = tuple(p["surfaces"]); fa = json.loads((a.root / "FIT_COMPLETE.json").read_text(encoding="utf-8"));
    if fa.get("status") != "PASS" or fa.get("confirmation_treated_loaded") is not False: raise RuntimeError("fit marker invalid")
    for item in fa["checkpoints"]:
        q = Path(item["path"])
        if not q.is_file() or sha256(q) != item["sha256"]: raise RuntimeError(f"checkpoint hash mismatch {q}")
    for item in fa.get("artifacts", []):
        q = Path(item["path"])
        if not q.is_file() or sha256(q) != item["sha256"]: raise RuntimeError(f"artifact hash mismatch {q}")
    dev = device(a.device); rows = old.build_row_map(old.read_conditions(a.conditions)); units = [(str(x[0]), float(x[1])) for x in p["units"]]; fps, _, _ = old.load_structure_map(a.structure_map); store = old.ProfileStore(a.group_counts); all_rows = []; condition_rows = []; read_events = []
    try:
        for target in old.CELL_LINES:
            for surface in surfaces:
                spec0 = p["folds"][target][surface]; test_drugs, foreign_drugs = set(spec0["test_drugs"]), set(spec0["foreign_drugs"]); ep = a.root / "fit" / surface / target; hvg = np.loadtxt(ep / "source_vehicle_hvg2000.txt", dtype=np.int64) if (ep / "source_vehicle_hvg2000.txt").exists() else old.fit_vehicle_hvg(store, rows, list(spec0["source_cell_lines"]))[0]
                needed = test_drugs | foreign_drugs; profile = {}
                for drug, dose in units:
                    if drug not in needed: continue
                    for rep in old.REPS:
                        row = rows[(target, drug, dose, rep)]; read_events.append({"target": target, "surface": surface, "drug": drug, "dose": dose, "replicate": rep, "group_id": row.group_id, "read_after_fit_checkpoint_and_artifact_hash_freeze": True}); profile[(target, drug, dose, rep)] = store.profile(row, hvg, read_treated=True)
                stats = {}
                for kind in ("raw", "cfra", "residual"):
                    with np.load(ep / f"stats_{kind}.npz", allow_pickle=False) as z: stats[kind] = {k: np.asarray(z[k], dtype=np.float32) for k in z.files}
                by_method = defaultdict(lambda: defaultdict(list))
                for drug, dose in units:
                    if drug not in test_drugs: continue
                    held = {rep: old.response(profile, target, drug, dose, rep) for rep in old.REPS}
                    truth = (held["rep1"] + held["rep2"]) / 2.0; base = (old.baseline(profile, target, drug, dose, "rep1") + old.baseline(profile, target, drug, dose, "rep2")) / 2.0
                    foreign = []
                    for fd in sorted(foreign_drugs - {drug}):
                        if all((target, fd, dose, rep) in profile for rep in old.REPS):
                            foreign.append({rep: old.response(profile, target, fd, dose, rep) for rep in old.REPS})
                    inp = old.StudentRows(np.asarray([target]), np.asarray([drug]), np.asarray([dose], np.float32), base[None, :], np.zeros((1, old.N_HVG), np.float32), truth[None, :]);
                    for method in METHODS:
                        kinds = METHOD_KINDS[method]
                        preds = []
                        for seed in SEEDS:
                            branch = []
                            for bi, kind in enumerate(kinds):
                                m = model(ep / "refit" / method / f"seed_{seed}" / f"branch_{bi}.pt", dev); branch.append(old.predict_model(m, inp, fps, stats[kind], dev)[0])
                            pred = (branch[0] + branch[1]) / 2.0 if method != "M2_CFRA1Residual" else branch[0] + branch[1]; preds.append(pred)
                        pred = np.mean(np.stack(preds), 0)
                        repeat_metrics = two_repeat_metrics(pred, held, foreign)
                        full = old.corr(pred + base, truth + base)
                        item = {"target": target, "surface": surface, "method": method, "drug": drug, "dose_nM": dose, "delta_pcc": old.corr(pred, truth), "full_target_pcc": full, "raw_target_mse": float(np.mean((pred-truth)**2)), **repeat_metrics, "foreign_eligible": bool(foreign), "same_foreign_definition": "mean(A/B Fisher same) - mean(donor, A/B Fisher foreign), matched target-cell-line+dose and excluding query drug", "held_metric_status": HELD_METRIC_STATUS}
                        condition_rows.append(item)
                        by_method[method][drug].append(item)
                for method in METHODS:
                    for drug, vals in by_method[method].items():
                        all_rows.append({"target": target, "surface": surface, "method": method, "drug": drug, "n_doses": len(vals), "n_foreign_eligible_doses": int(sum(x["foreign_eligible"] for x in vals)), **{metric: mean_metric(vals, metric) for metric in BOOTSTRAP_METRICS}, "held_metric_status": HELD_METRIC_STATUS})
    finally: store.close()
    out = a.root / "test"; out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "CONFIRMATION_PER_CONDITION.csv", condition_rows)
    write_csv(out / "CONFIRMATION_PER_COMPOUND.csv", all_rows)
    write_csv(out / "CONFIRMATION_READ_AUDIT.csv", read_events)
    contrasts = []
    for target in old.CELL_LINES:
        for surface in surfaces:
            sub = [x for x in all_rows if x["target"] == target and x["surface"] == surface]
            for metric in BOOTSTRAP_METRICS:
                mm = {m: {x["drug"]: float(x[metric]) for x in sub if x["method"] == m and np.isfinite(float(x[metric]))} for m in METHODS}
                pairs = [("M2_CFRA1Residual", "M0_RawAggregate")]
                if "M1_CFRA1Aggregate" in METHODS:
                    pairs.extend((("M1_CFRA1Aggregate", "M0_RawAggregate"), ("M2_CFRA1Residual", "M1_CFRA1Aggregate")))
                for lhs, rhs in pairs:
                    q = paired_boot(mm[rhs], mm[lhs], f"{target}|{surface}|{metric}|{lhs}-{rhs}"); contrasts.append({"target": target, "surface": surface, "metric": metric, "comparison": lhs + " - " + rhs, **q})
    write_csv(out / "CONFIRMATION_CONTRASTS.csv", contrasts)
    report = ["# sci-Plex3 BBBC047-adapted P0", "", "Raw-Aggregate is the mean of the two real `(drug,dose,plate)` responses. CFRA-1R-Aggregate applies the frozen 1R drug-OOF teacher independently to each repeat before averaging. M0 averages two Raw-Aggregate branches; M2 sums a CFRA-1R-Aggregate base and a Raw-minus-CFRA residual branch; M1 is the direct CFRA target diagnostic.", "", "Primary P0 is `M2_CFRA1Residual - M0_RawAggregate` on common Raw-Aggregate `delta_pcc` and `full_target_pcc`; no conclusion is drawn from M1 alone.", "", "`A`, `z_same_F`, `z_foreign_F`, `E`, `held_pcc_A`, and `held_pcc_B` use the two actual sci-Plex3 repeat views. They are descriptive within the same two-repeat data, not independent held-plate confirmation endpoints. For foreign-eligible units, foreign scores use the same target-cell-line/dose signature, omit the query drug, take Fisher-z separately for each donor and repeat, then average.", "", "| target | surface | metric | comparison | estimate | 95% CI |", "|---|---|---|---|---:|---:|"]
    for x in contrasts: report.append(f"| {x['target']} | {x['surface']} | {x['metric']} | {x['comparison']} | {x['estimate']:.6f} | [{x['ci_low']:.6f}, {x['ci_high']:.6f}] |")
    (a.root / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(a.root / "TEST_COMPLETE.json", {"status": "PASS", "surface_count": len(surfaces), "surfaces": list(surfaces), "surface_isolation": True, "methods": list(METHODS), "confirmation_treated_loaded_after_freeze": True, "physical_repeats": 2, "repeat_metric_status": HELD_METRIC_STATUS, "evaluation_metrics": list(BOOTSTRAP_METRICS), "evidence_status": "historical locked/retrieved confirmation; not new blind confirmatory", "outputs": [str((out / "CONFIRMATION_PER_CONDITION.csv").resolve()), str((out / "CONFIRMATION_PER_COMPOUND.csv").resolve()), str((out / "CONFIRMATION_CONTRASTS.csv").resolve()), str((out / "CONFIRMATION_READ_AUDIT.csv").resolve())]})
    print(json.dumps({"status": "TEST_COMPLETE", "rows": len(all_rows)}))

if __name__ == "__main__":
    a = parse_args(); {"preflight": preflight, "dry_run": lambda a: (load_plan(a), write_json(a.root / "DRY_RUN_COMPLETE.json", {"status":"PASS","confirmation_treated_loaded":False})), "fit": fit, "test": test}[a.stage](a)
