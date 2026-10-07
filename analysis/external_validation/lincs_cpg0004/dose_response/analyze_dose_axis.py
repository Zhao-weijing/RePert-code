#!/usr/bin/env python3
"""Summarize the frozen cpg0004 dose axis without refitting any model.

The primary score is the same compound-level held-out-plate metric used by the
repeat benchmark: mean Fisher-z(PCC) minus the row-matched foreign null.  Raw
PCC is retained as a secondary diagnostic.  The script only reads saved
predictions/score tables and the frozen CP target artifact.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CP_ROOT = ROOT / "cell_painting_repeat_benchmark" / "results"
PRIOR_ROOT = ROOT / "virtual_prior" / "results"
GE_ROOT = ROOT / "single_repeat_expression_evidence" / "results"
CP_DATA = ROOT / "data_preparation" / "artifact" / "cp_plate_rows.npz"
SEEDS = (3407, 42, 2025)
DOSES = ("0.04", "0.12", "0.37", "1.11", "3.33", "10")


def decode(a: np.ndarray) -> np.ndarray:
    return np.char.decode(a, "utf-8", errors="replace").astype(str) if a.dtype.kind == "S" else a.astype(str)


def fisher(x: float) -> float:
    return float(np.arctanh(np.clip(x, -0.999999, 0.999999)))


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64) - float(np.mean(a))
    bb = b.astype(np.float64) - float(np.mean(b))
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / den) if den > 0 else float("nan")


def summarize_rows(rows: list[tuple[str, float, float]]) -> dict[str, float | int]:
    """Rows are (compound, raw_pcc, null_z), one held target per row."""
    by_comp: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    for compound, raw, nz in rows:
        if np.isfinite(raw) and np.isfinite(nz):
            by_comp[compound].append((raw, nz))
    raw_means: list[float] = []
    excess: list[float] = []
    for values in by_comp.values():
        raw_means.append(float(np.mean([x[0] for x in values])))
        excess.append(float(np.mean([fisher(x[0]) - x[1] for x in values])))
    return {
        "compound_count": len(by_comp),
        "pair_count": sum(len(v) for v in by_comp.values()),
        "raw_pcc_mean": float(np.mean(raw_means)) if raw_means else float("nan"),
        "excess_fisher_z_mean": float(np.mean(excess)) if excess else float("nan"),
    }


def load_targets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(CP_DATA, allow_pickle=False) as d:
        return d["delta"].astype(np.float32), decode(d["compound_id"]), decode(d["dose"])


def add_record(out: list[dict], dataset: str, seed: int, budget: int, dose: str, method: str, score: dict):
    out.append({"dataset": dataset, "seed": seed, "budget": budget, "dose": dose, "method": method, **score})


def analyze_cp(out: list[dict]) -> None:
    for budget in (1, 2, 3):
        for seed in SEEDS:
            path = CP_ROOT / f"{budget}r_all" / f"seed{seed}" / "per_pair_scores.csv"
            grouped: defaultdict[tuple[str, str], list[tuple[str, float, float]]] = defaultdict(list)
            with path.open(encoding="utf-8", newline="") as h:
                for row in csv.DictReader(h):
                    dose = str(row["dose"])
                    grouped[(dose, "teacher")].append((row["compound"], float(row["teacher_pcc"]), float(row["null_z"])))
                    grouped[(dose, "support_mean")].append((row["compound"], float(row["support_mean_pcc"]), float(row["null_z"])))
            for dose in DOSES:
                for method in ("teacher", "support_mean"):
                    add_record(out, "cp_repeat", seed, budget, dose, method, summarize_rows(grouped[(dose, method)]))


def analyze_prior(out: list[dict], delta: np.ndarray, compounds: np.ndarray, row_doses: np.ndarray) -> None:
    for seed in SEEDS:
        path = PRIOR_ROOT / "1r_all" / f"seed{seed}" / "test_predictions.npz"
        with np.load(path, allow_pickle=False) as d:
            c = decode(d["compound"]); dose = decode(d["dose"]); held = d["held_index"].astype(np.int64)
            usable = d["usable"].astype(bool); nz = d["null_z"].astype(np.float64)
            values = {k: d[k].astype(np.float32) for k in ("support_mean", "teacher", "virtual_prior", "posterior")}
        grouped: dict[tuple[str, str], list[tuple[str, float, float]]] = defaultdict(list)
        for i in np.flatnonzero(usable):
            target = delta[held[i]]
            for method, pred in values.items():
                grouped[(dose[i], method)].append((c[i], pcc(pred[i], target), nz[i]))
        for dose_label in DOSES:
            for method in values:
                add_record(out, "virtual_prior", seed, 1, dose_label, method, summarize_rows(grouped[(dose_label, method)]))


def analyze_ge(out: list[dict], delta: np.ndarray) -> None:
    for seed in SEEDS:
        path = GE_ROOT / "1r_all" / f"seed{seed}" / "test_predictions.npz"
        with np.load(path, allow_pickle=False) as d:
            c = decode(d["compound"]); dose = decode(d["dose"]); held = d["held_index"].astype(np.int64)
            usable = d["usable"].astype(bool); nz = d["null_z"].astype(np.float64)
            values = {k: d[k].astype(np.float32) for k in ("P0", "correct_GE", "shuffled_GE", "matched_foreign_GE")}
        grouped: dict[tuple[str, str], list[tuple[str, float, float]]] = defaultdict(list)
        for i in np.flatnonzero(usable):
            target = delta[held[i]]
            for method, pred in values.items():
                grouped[(dose[i], method)].append((c[i], pcc(pred[i], target), nz[i]))
        for dose_label in DOSES:
            for method in values:
                add_record(out, "ge_residual", seed, 1, dose_label, method, summarize_rows(grouped[(dose_label, method)]))


def main() -> None:
    out: list[dict] = []
    delta, compounds, row_doses = load_targets()
    analyze_cp(out)
    analyze_prior(out, delta, compounds, row_doses)
    analyze_ge(out, delta)
    outdir = Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "dose_axis_summary.csv").open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(out[0]))
        writer.writeheader(); writer.writerows(out)
    # Human-readable differences use the same per-dose summaries and are
    # deliberately secondary to the saved per-seed records.
    by = {(r["dataset"], r["seed"], r["budget"], r["dose"], r["method"]): r for r in out}
    contrasts = []
    pairs = {
        "teacher_minus_support": ("cp_repeat", "teacher", "support_mean"),
        "posterior_minus_teacher": ("virtual_prior", "posterior", "teacher"),
        "correct_GE_minus_P0": ("ge_residual", "correct_GE", "P0"),
        "correct_GE_minus_shuffled_GE": ("ge_residual", "correct_GE", "shuffled_GE"),
        "correct_GE_minus_matched_foreign_GE": ("ge_residual", "correct_GE", "matched_foreign_GE"),
    }
    for contrast, (dataset, left, right) in pairs.items():
        for seed in SEEDS:
            budget = 1 if dataset != "cp_repeat" else None
            for dose in DOSES:
                budgets = (1, 2, 3) if dataset == "cp_repeat" else (1,)
                for b in budgets:
                    lk = (dataset, seed, b, dose, left); rk = (dataset, seed, b, dose, right)
                    if lk not in by or rk not in by: continue
                    l, r = by[lk], by[rk]
                    contrasts.append({"contrast": contrast, "dataset": dataset, "seed": seed, "budget": b, "dose": dose,
                                      "delta_excess_fisher_z": l["excess_fisher_z_mean"] - r["excess_fisher_z_mean"],
                                      "delta_raw_pcc": l["raw_pcc_mean"] - r["raw_pcc_mean"],
                                      "left_compounds": l["compound_count"], "right_compounds": r["compound_count"]})
    with (outdir / "dose_axis_contrasts.csv").open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(contrasts[0])); writer.writeheader(); writer.writerows(contrasts)
    (outdir / "DOSE_AXIS.md").write_text(
        "# cpg0004 dose axis\n\n"
        "This is a post-hoc summary of frozen predictions. It uses the held-out-plate, compound-level excess Fisher-z metric as primary and raw PCC as secondary. No model, split, or hyperparameter was changed. Exact-well null availability remains zero; the underlying row-matched null is inherited from each run.\n",
        encoding="utf-8",
    )
    print(json.dumps({"records": len(out), "contrasts": len(contrasts), "outdir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
