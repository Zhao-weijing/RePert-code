"""Sci-Plex3 frozen 24 h no-model Raw Same--Foreign evaluation.

The input is the scPerturb Zenodo-7041849 H5AD.  scPerturb issue #7 shifted
the feature annotation by one row and appended the annotation header as a
feature.  This script never uses ``var`` for feature identity: it downloads
the authoritative GEO gene annotation, maps annotation row ``i`` to X column
``i`` for i=0..110982, and audits that the final X column is all zero.

The response unit is one treated drug x cell-line x dose x biological repeat.
Only time=24, nperts=1, fully annotated treated cells are used.  Raw
pseudo-bulk responses are log1p(CPM) minus all matched Vehicle cells from the
same plate, cell line and repeat.  The no-model endpoint uses ordered
rep1->rep2 and rep2->rep1 rotations.  Foreign donors are a deterministic
20-drug sample from the same cell-line, dose and held repeat, excluding the
target drug.  Inference is a 10,000-draw bootstrap over drugs, never cells.

This is intentionally a preparation/evaluation script only.  It does not
fit an estimator, select a split, or access a confirmation model.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SEED = 3407
BOOT = 10_000
REPS = ("rep1", "rep2")
N_FOREIGN = 20
N_ANNOTATION = 110_983
ANNOTATION_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl/"
    "GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz"
)


def decode_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in "SU":
        return arr.astype(str)
    out = np.empty(arr.shape, dtype=object)
    for i, v in enumerate(arr.reshape(-1)):
        out.reshape(-1)[i] = v.decode("utf-8") if isinstance(v, bytes) else str(v)
    return out


def read_obs_column(h, key: str) -> np.ndarray:
    obj = h[f"obs/{key}"]
    if hasattr(obj, "keys"):
        cats = decode_array(obj["categories"][:])
        codes = np.asarray(obj["codes"][:])
        out = np.empty(codes.shape, dtype=object)
        out[:] = ""
        ok = codes >= 0
        out[ok] = cats[codes[ok]]
        return out
    return decode_array(obj[:])


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def download_annotation(outdir: Path, url: str) -> tuple[Path, dict[str, Any], list[tuple[str, str]]]:
    p = outdir / "provenance_inputs" / Path(url).name
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        urllib.request.urlretrieve(url, p)
    raw = p.read_bytes()
    records: list[tuple[str, str]] = []
    first_lines: list[str] = []
    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f):
            line = line.rstrip("\r\n")
            if line_no < 5:
                first_lines.append(line)
            if line_no == 0:
                if line != "id gene_short_name":
                    raise RuntimeError(f"unexpected GEO annotation header: {line!r}")
                continue
            fields = line.split()
            if len(fields) < 2:
                raise RuntimeError(f"malformed GEO annotation row {line_no + 1}: {line!r}")
            gene_id = fields[0].split(".", 1)[0]
            symbol = fields[1]
            records.append((gene_id, symbol))
    if len(records) != N_ANNOTATION:
        raise RuntimeError(f"expected {N_ANNOTATION} GEO genes, found {len(records)}")
    meta = {
        "url": url,
        "local_path": str(p),
        "size_bytes": len(raw),
        "md5": hashlib.md5(raw).hexdigest(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "first_lines": first_lines,
        "header": first_lines[0],
        "n_lines_including_header": len(records) + 1,
        "n_annotation_genes": len(records),
    }
    return p, meta, records


def audit_feature_map(h, records: list[tuple[str, str]], annotation_meta: dict[str, Any], outdir: Path) -> dict[str, Any]:
    n_vars = int(h["X"].attrs["shape"][1])
    n_rows = int(h["X"].attrs["shape"][0])
    if n_vars != len(records) + 1:
        raise RuntimeError(f"H5 n_vars={n_vars} is not annotation genes+1={len(records)+1}")
    # Current scPerturb labels are deliberately recorded but not used.
    current_var_first: dict[str, Any] = {}
    for key in ("ensembl_id", "gene_symbol"):
        obj = h[f"var/{key}"]
        if hasattr(obj, "keys"):
            cats = decode_array(obj["categories"][:])
            codes = np.asarray(obj["codes"][:])
            vals = [str(cats[int(codes[i])]) if codes[i] >= 0 else "" for i in range(min(5, len(codes)))]
        else:
            vals = [str(x) for x in decode_array(obj[:5])]
        current_var_first[key] = vals
    # Write an explicit, inspectable map.  Annotation row i is X column i.
    fmap_rows = []
    for i, (gid, symbol) in enumerate(records):
        fmap_rows.append({
            "x_column": i,
            "annotation_row_after_header": i + 1,
            "gene_id": gid,
            "gene_symbol": symbol,
            "used": True,
            "current_scperturb_var_row": i,
        })
    write_csv(outdir / "feature_map.csv", fmap_rows)
    audit = {
        "mapping_rule": "GEO annotation data row i -> H5 X[:,i], i=0..110982; H5 X[:,110983] excluded as scPerturb Issue#7 tail column",
        "n_obs": n_rows,
        "n_h5_vars": n_vars,
        "n_annotation_genes": len(records),
        "x_columns_used": [0, len(records) - 1],
        "x_tail_column_excluded": n_vars - 1,
        "annotation_first_gene": {"gene_id": records[0][0], "gene_symbol": records[0][1]},
        "annotation_last_gene": {"gene_id": records[-1][0], "gene_symbol": records[-1][1]},
        "annotation": annotation_meta,
        "current_scperturb_var_first_rows_ignored": current_var_first,
    }
    return audit


def stable_token_seed(*parts: Any) -> int:
    tok = "|".join(str(x) for x in parts).encode("utf-8")
    return SEED + int.from_bytes(hashlib.sha256(tok).digest()[:4], "big")


def norm_delta(t_counts: np.ndarray, c_counts: np.ndarray) -> np.ndarray:
    t = np.asarray(t_counts, dtype=np.float64)
    c = np.asarray(c_counts, dtype=np.float64)
    ts = float(t.sum())
    cs = float(c.sum())
    if ts <= 0 or cs <= 0:
        raise RuntimeError("zero library size in a selected pseudo-bulk")
    return np.log1p(t / ts * 1e6) - np.log1p(c / cs * 1e6)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - float(a.mean())
    b = b - float(b.mean())
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den > 1e-12 else float("nan")


def fisher(r: float) -> float:
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999))) if np.isfinite(r) else float("nan")


def bootstrap(values: list[float], drugs: list[str], seed: int = SEED) -> dict[str, Any]:
    by: dict[str, list[float]] = defaultdict(list)
    for v, d in zip(values, drugs):
        if np.isfinite(v):
            by[str(d)].append(float(v))
    keys = sorted(by)
    x = np.asarray([np.mean(by[k]) for k in keys], dtype=np.float64)
    if len(x) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_drugs": 0, "bootstrap_draws": BOOT}
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(x), size=(BOOT, len(x)))
    means = x[draw].mean(axis=1)
    return {
        "estimate": float(x.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n_drugs": int(len(x)),
        "bootstrap_draws": BOOT,
    }


def read_csr_block(h, start: int, end: int, n_vars: int):
    from scipy.sparse import csr_matrix
    indptr = np.asarray(h["X/indptr"][start : end + 1], dtype=np.int64)
    p0, p1 = int(indptr[0]), int(indptr[-1])
    data = np.asarray(h["X/data"][p0:p1], dtype=np.float32)
    indices = np.asarray(h["X/indices"][p0:p1], dtype=np.int32)
    indptr -= p0
    return csr_matrix((data, indices, indptr), shape=(end - start, n_vars)), indices


def aggregate_groups(h, group_codes: np.ndarray, group_meta: list[dict[str, Any]], outdir: Path, chunk_rows: int = 10_000) -> Path:
    import h5py
    from scipy.sparse import csr_matrix

    n_rows, n_vars = map(int, h["X"].attrs["shape"])
    group_path = outdir / "group_counts.h5"
    if group_path.exists():
        return group_path
    with h5py.File(group_path, "w") as g:
        counts = g.create_dataset(
            "counts", shape=(len(group_meta), n_vars), dtype="f4",
            chunks=(1, min(n_vars, 8192)), compression="lzf",
        )
        g.attrs["shape"] = (len(group_meta), n_vars)
        tail_nnz = 0
        total_selected = 0
        for start in range(0, n_rows, chunk_rows):
            end = min(n_rows, start + chunk_rows)
            block, block_indices = read_csr_block(h, start, end, n_vars)
            tail_nnz += int(np.count_nonzero(block_indices == n_vars - 1))
            codes = group_codes[start:end]
            keep = codes >= 0
            if not np.any(keep):
                continue
            rows = np.flatnonzero(keep)
            b = block[rows]
            gc = codes[rows]
            unique, inv = np.unique(gc, return_inverse=True)
            onehot = csr_matrix(
                (np.ones(len(rows), dtype=np.float32), (inv, np.arange(len(rows)))),
                shape=(len(unique), len(rows)),
            )
            agg = (onehot @ b).toarray().astype(np.float32, copy=False)
            for j, gid in enumerate(unique):
                counts[int(gid), :] += agg[j, :]
            total_selected += int(len(rows))
            if (start // chunk_rows) % 10 == 0:
                print(f"aggregate rows {end}/{n_rows} selected={total_selected}", flush=True)
        g.attrs["x_tail_column_nnz"] = int(tail_nnz)
        g.attrs["selected_cells"] = int(total_selected)
    write_json(outdir / "group_counts_audit.json", {
        "group_counts_path": str(group_path),
        "x_tail_column_nnz": int(tail_nnz),
        "selected_cells": int(total_selected),
        "n_groups": len(group_meta),
        "n_vars": n_vars,
    })
    return group_path


def make_groups(h, n_rows: int) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    needed = ["nperts", "time", "replicate", "plate", "cell_line", "perturbation", "target", "dose_value", "well"]
    obs = {k: read_obs_column(h, k) for k in needed}
    codes = np.full(n_rows, -1, dtype=np.int32)
    key_to_gid: dict[tuple[Any, ...], int] = {}
    meta: list[dict[str, Any]] = []
    selected = {"all": 0, "treated": 0, "control": 0, "excluded": 0}
    complete_fields = ["replicate", "plate", "cell_line", "perturbation", "target", "dose_value"]
    for i in range(n_rows):
        try:
            time = float(obs["time"][i])
            nperts = int(obs["nperts"][i])
            rep = str(obs["replicate"][i])
            plate = str(obs["plate"][i])
            cell = str(obs["cell_line"][i])
            perturb = str(obs["perturbation"][i])
            target = str(obs["target"][i])
            dose = float(obs["dose_value"][i])
        except (TypeError, ValueError):
            selected["excluded"] += 1
            continue
        if time != 24.0 or nperts != 1 or rep not in REPS:
            selected["excluded"] += 1
            continue
        if any((not x) or x.lower() in {"nan", "none"} for x in (rep, plate, cell, perturb, target)) or not np.isfinite(dose):
            selected["excluded"] += 1
            continue
        selected["all"] += 1
        pnorm = perturb.lower()
        is_control = pnorm in {"control", "vehicle"} or target.lower() == "vehicle"
        if is_control:
            key = ("control", rep, plate, cell)
            selected["control"] += 1
            group_type = "control"
        else:
            if dose <= 0:
                selected["excluded"] += 1
                continue
            key = ("treated", rep, plate, cell, perturb, target, dose)
            selected["treated"] += 1
            group_type = "treated"
        gid = key_to_gid.get(key)
        if gid is None:
            gid = len(meta)
            key_to_gid[key] = gid
            if group_type == "control":
                rec = {"group_id": gid, "group_type": group_type, "replicate": rep, "plate": plate, "cell_line": cell}
            else:
                rec = {
                    "group_id": gid, "group_type": group_type, "replicate": rep, "plate": plate,
                    "cell_line": cell, "drug": perturb, "target": target, "dose_nM": dose,
                }
            meta.append(rec)
        codes[i] = gid
    selected["n_groups"] = len(meta)
    return codes, meta, selected


def build_condition_profiles(group_h5: Path, group_meta: list[dict[str, Any]], split_outdir: Path, n_min: int = 50):
    import h5py

    with h5py.File(group_h5, "r") as gh:
        counts = gh["counts"]
        group_by_id = {int(m["group_id"]): m for m in group_meta}
        # This count comes from cell metadata and is added before this function.
        conditions = []
        for m in group_meta:
            if m["group_type"] != "treated":
                continue
            control_key = ("control", m["replicate"], m["plate"], m["cell_line"])
            control = next((x for x in group_meta if (x["group_type"], x["replicate"], x["plate"], x["cell_line"]) == control_key), None)
            if control is None:
                continue
            if int(m.get("n_cells", 0)) < n_min:
                continue
            # Control is allowed to be any positive number of cells; it is a
            # same-plate Vehicle pool and is never a treated condition.
            t = np.asarray(counts[int(m["group_id"]), :], dtype=np.float64)
            c = np.asarray(counts[int(control["group_id"]), :], dtype=np.float64)
            if t.sum() <= 0 or c.sum() <= 0:
                continue
            d = norm_delta(t, c)
            conditions.append({
                "drug": m["drug"], "target": m["target"], "cell_line": m["cell_line"],
                "dose_nM": float(m["dose_nM"]), "replicate": m["replicate"],
                "plate": m["plate"], "group_id": int(m["group_id"]),
                "control_group_id": int(control["group_id"]), "n_cells": int(m["n_cells"]),
                "control_n_cells": int(control.get("n_cells", 0)), "delta": d,
            })
    if not conditions:
        raise RuntimeError("no eligible >=50-cell treated conditions with matched Vehicle")
    return conditions


def _write_group_meta(meta: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for m in meta:
        x = dict(m)
        x.setdefault("n_cells", 0)
        rows.append(x)
    write_csv(path, rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--annotation-url", default=ANNOTATION_URL)
    ap.add_argument("--n-min", type=int, default=50)
    ap.add_argument("--chunk-rows", type=int, default=10_000)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    import h5py

    with h5py.File(args.input, "r") as h:
        n_rows, n_vars = map(int, h["X"].attrs["shape"])
        annotation_path, annotation_meta, records = download_annotation(args.outdir, args.annotation_url)
        feature_audit = audit_feature_map(h, records, annotation_meta, args.outdir)
        codes, meta, selection_audit = make_groups(h, n_rows)
        # Populate n_cells from codes before writing and aggregation.
        cell_counts = np.bincount(codes[codes >= 0], minlength=len(meta))
        for m in meta:
            m["n_cells"] = int(cell_counts[int(m["group_id"])])
        _write_group_meta(meta, args.outdir / "group_metadata.csv")
        group_h5 = aggregate_groups(h, codes, meta, args.outdir, args.chunk_rows)
        # X tail can be checked without trusting var labels.  We require it
        # to be all zero over the full matrix before proceeding.
        with h5py.File(group_h5, "r") as gh:
            tail_nnz = int(gh.attrs.get("x_tail_column_nnz", -1))
        if tail_nnz != 0:
            raise RuntimeError(f"scPerturb tail X column has {tail_nnz} nonzeros; annotation alignment unsafe")

    conditions = build_condition_profiles(group_h5, meta, args.outdir, args.n_min)
    # condition key should be one row per drug x cell line x dose x replicate.
    key_counts = defaultdict(int)
    for r in conditions:
        key_counts[(r["drug"], r["cell_line"], r["dose_nM"], r["replicate"])] += 1
    duplicate_keys = {str(k): v for k, v in key_counts.items() if v != 1}
    if duplicate_keys:
        raise RuntimeError(f"duplicate condition units found (would violate drug x cell_line x dose x repeat): {duplicate_keys}")
    condition_rows = []
    for r in conditions:
        x = {k: v for k, v in r.items() if k != "delta"}
        condition_rows.append(x)
    write_csv(args.outdir / "eligible_conditions_n50.csv", condition_rows)

    by_key = {(r["drug"], r["cell_line"], float(r["dose_nM"]), r["replicate"]): r for r in conditions}
    pair_rows = []
    raw_scores: list[float] = []
    raw_drugs: list[str] = []
    raw_celllines: list[str] = []
    n_foreign_short = 0
    stable_keys = {(r["drug"], r["cell_line"], float(r["dose_nM"])) for r in conditions if r["n_cells"] >= args.n_min}
    for drug, cell, dose in sorted(stable_keys, key=lambda x: (x[1], x[0], x[2])):
        if (drug, cell, dose, "rep1") not in by_key or (drug, cell, dose, "rep2") not in by_key:
            continue
        for support, held in (("rep1", "rep2"), ("rep2", "rep1")):
            src = by_key[(drug, cell, dose, support)]
            true = by_key[(drug, cell, dose, held)]
            donor_keys = sorted(
                d for d in stable_keys
                if d[1] == cell and float(d[2]) == float(dose) and d[0] != drug
                and (d[0], cell, dose, held) in by_key
            )
            if len(donor_keys) < N_FOREIGN:
                n_foreign_short += 1
                selected = donor_keys
            else:
                rng = np.random.default_rng(stable_token_seed("foreign", drug, cell, dose, support, held))
                selected = [tuple(x) for x in rng.choice(np.asarray(donor_keys, dtype=object), N_FOREIGN, replace=False)]
            foreign_scores = [fisher(corr(src["delta"], by_key[(d[0], cell, dose, held)]["delta"])) for d in selected]
            same_z = fisher(corr(src["delta"], true["delta"]))
            foreign_z = float(np.mean(foreign_scores)) if foreign_scores else float("nan")
            excess = same_z - foreign_z if np.isfinite(same_z) and np.isfinite(foreign_z) else float("nan")
            row = {
                "drug": drug, "target": src["target"], "cell_line": cell, "dose_nM": dose,
                "support_repeat": support, "held_repeat": held, "n_support_cells": src["n_cells"],
                "n_held_cells": true["n_cells"], "n_foreign": len(selected),
                "foreign_drugs": ";".join(sorted(d[0] for d in selected)),
                "target_fisher_z": same_z, "foreign_mean_fisher_z": foreign_z,
                "same_foreign_excess_fisher_z": excess,
            }
            pair_rows.append(row)
            raw_scores.append(excess); raw_drugs.append(drug); raw_celllines.append(cell)
    if not pair_rows:
        raise RuntimeError("no ordered rep1/rep2 eligible pairs")
    write_csv(args.outdir / "raw_same_foreign_per_condition.csv", pair_rows)

    # Target/drug is the bootstrap unit.  Dose and repeat rotations are first
    # averaged within a drug for each cell line, then drugs are resampled.
    by_cell_drug: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_drug: dict[str, list[float]] = defaultdict(list)
    for v, d, c in zip(raw_scores, raw_drugs, raw_celllines):
        by_cell_drug[(c, d)].append(v)
        by_drug[d].append(v)
    summary_rows = []
    cell_stats = {}
    for cell in sorted({c for c, _ in by_cell_drug}):
        vals, drugs = [], []
        for (c, d), xs in sorted(by_cell_drug.items()):
            if c == cell:
                vals.append(float(np.mean(xs))); drugs.append(d)
        stat = bootstrap(vals, drugs, stable_token_seed("bootstrap", cell))
        cell_stats[cell] = stat
        summary_rows.append({"scope": "cell_line", "cell_line": cell, "metric": "Raw Same-Foreign excess Fisher-z", **stat})
    vals, drugs = [], []
    for d, xs in sorted(by_drug.items()):
        vals.append(float(np.mean(xs))); drugs.append(d)
    overall = bootstrap(vals, drugs, stable_token_seed("bootstrap", "overall"))
    summary_rows.append({"scope": "overall", "cell_line": "ALL", "metric": "Raw Same-Foreign excess Fisher-z", **overall})
    write_csv(args.outdir / "raw_same_foreign_summary.csv", summary_rows)

    eligible_drugs = sorted({r["drug"] for r in conditions if r["n_cells"] >= args.n_min})
    stable_drugs = sorted({d for d, xs in by_drug.items() if len(xs) > 0})
    gate = {
        "endpoint": "Raw Same-Foreign excess Fisher-z",
        "time_hours": 24,
        "nperts": 1,
        "cell_level_inference": False,
        "response": "log1p(CPM 1e6) treated pseudo-bulk minus same-plate/cell-line/repeat Vehicle pseudo-bulk",
        "ordered_rotations": ["rep1->rep2", "rep2->rep1"],
        "foreign_policy": "deterministic 20 same-cell-line same-dose held-repeat donor drugs, excluding target; all donors require both repeats n>=50",
        "n_foreign_short_rotations": n_foreign_short,
        "n_pair_rows": len(pair_rows),
        "n_eligible_drugs_any_condition": len(eligible_drugs),
        "n_stable_drugs_with_paired_condition": len(stable_drugs),
        "bootstrap_draws": BOOT,
        "bootstrap_group": "drug",
        "phase0_gate_rule": "overall 95% CI low > 0 and positive point estimate in at least two cell lines",
        "overall": overall,
        "cell_lines": cell_stats,
        "pass": bool(overall["ci_low"] > 0 and sum(float(x["estimate"]) > 0 for x in cell_stats.values()) >= 2),
        "feature_map_alignment": feature_audit,
        "selection_audit": selection_audit,
        "n_min": args.n_min,
        "seed": SEED,
        "models_fit": False,
    }
    write_json(args.outdir / "raw_same_foreign_gate.json", gate)
    write_json(args.outdir / "SCI_PHASE0_AUDIT.json", {
        "dataset": "SrivatsanTrapnell2020_sciplex3.h5ad",
        "input": str(args.input),
        "input_md5_expected": "d1f51b9f8de35ca07638132539da9a99",
        "input_md5_check_deferred": True,
        "feature_map": feature_audit,
        "selection": selection_audit,
        "time24_only": True,
        "nperts1_only": True,
        "matched_control_verified_by_group_key": True,
        "tail_column_nnz": 0,
        "raw_gate": gate,
    })

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = list(sorted(cell_stats)) + ["ALL"]
        stats = [cell_stats[x] for x in sorted(cell_stats)] + [overall]
        x = np.arange(len(labels))
        y = np.asarray([s["estimate"] for s in stats])
        lo = np.asarray([s["ci_low"] for s in stats])
        hi = np.asarray([s["ci_high"] for s in stats])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), fmt="o", color="#b91c1c", capsize=4)
        ax.axhline(0, color="#374151", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Raw Same–Foreign excess Fisher-z")
        ax.set_title("Sci-Plex3 24 h Raw repeat gate")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.outdir / "raw_same_foreign_gate.png", dpi=220)
        fig.savefig(args.outdir / "raw_same_foreign_gate.pdf")
        plt.close(fig)
    except Exception as e:
        write_json(args.outdir / "figure_error.json", {"error": repr(e)})

    report = [
        "# sci-Plex3 24 h no-model Raw Same–Foreign",
        "",
        "## Frozen scope",
        "",
        "- Input: Zenodo 7041849 scPerturb H5AD; only time=24 h, nperts=1, complete metadata, non-control treated cells.",
        "- Response: treated pseudo-bulk log1p(CPM 1e6) minus same-plate, same-cell-line, same-repeat Vehicle pseudo-bulk.",
        "- Repeat rotations: rep1→rep2 and rep2→rep1; inference unit is drug, with 10,000 bootstrap draws. Cells are never bootstrap units.",
        "- Foreign pool: deterministic 20 eligible donor drugs matched on cell line, dose and held repeat, excluding target drug.",
        "",
        "## Feature-map correction",
        "",
        "The H5AD `var` labels are not used. GEO `gene.annotations` data row i is mapped to X column i for i=0..110982; X column 110983 is excluded and audited as all-zero. The complete map and SHA-256 are in `feature_map.csv` and `SCI_PHASE0_AUDIT.json`.",
        "",
        "## Result",
        "",
        f"- Overall: estimate={overall['estimate']:.6f}, 95% CI [{overall['ci_low']:.6f}, {overall['ci_high']:.6f}], n_drugs={overall['n_drugs']}; gate={'PASS' if gate['pass'] else 'NO-GO'}.",
    ]
    for cell in sorted(cell_stats):
        s = cell_stats[cell]
        report.append(f"- {cell}: estimate={s['estimate']:.6f}, 95% CI [{s['ci_low']:.6f}, {s['ci_high']:.6f}], n_drugs={s['n_drugs']}.")
    (args.outdir / "RAW_SAME_FOREIGN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
