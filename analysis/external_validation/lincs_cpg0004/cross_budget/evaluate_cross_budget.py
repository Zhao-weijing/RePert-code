#!/usr/bin/env python3
"""Matched cpg0004 contrast: 1R + frozen AI posterior vs 2R raw mean."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cell_painting_repeat_benchmark"))
import run_cp_repeat_benchmark as B


def paired_vectors(compounds, values, pairs, null_z, usable):
    grouped = {}
    for i in np.flatnonzero(usable):
        score = B.pcc(values[i], pairs.target[i])
        if score is None: continue
        grouped.setdefault(compounds[i], []).append((B.fisher(score), score, float(null_z[i])))
    out = {}
    for c, vals in grouped.items():
        out[c] = {"excess_z": float(np.mean([v[0]-v[2] for v in vals])), "pcc": float(np.mean([v[1] for v in vals])), "n_pairs": len(vals)}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--one-r-predictions", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--null-rounds", type=int, default=32)
    p.add_argument("--bootstrap-rounds", type=int, default=5000)
    args = p.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)
    rows = B.load_rows(args.data)
    one = np.load(args.one_r_predictions, allow_pickle=False)
    one_comp = one["compound"].astype(str); one_dose = one["dose"].astype(str); one_held = one["held_index"].astype(np.int64)
    one_key = {(c, d, int(h)): i for i, (c, d, h) in enumerate(zip(one_comp, one_dose, one_held))}
    two_all = B.make_pairs(rows, 2, "all")
    test_compounds = set(rows.compound[rows.split == "test"])
    two = B.subset_pairs(two_all, test_compounds)
    two_null, two_usable, two_exact = B.make_matched_null(rows, two, args.null_rounds, args.seed)
    keep_one=[]; keep_two=[]
    for j, (c, d, h) in enumerate(zip(two.compound, two.dose, two.held_index)):
        i = one_key.get((str(c), str(d), int(h)))
        if i is not None and bool(one["usable"][i]) and bool(two_usable[j]): keep_one.append(i); keep_two.append(j)
    if not keep_one: raise RuntimeError("No common usable 1R/2R held-out targets")
    keep_one=np.asarray(keep_one,dtype=np.int64); keep_two=np.asarray(keep_two,dtype=np.int64)
    common_comp=one_comp[keep_one]
    # Compare compound-level means so compounds with more plates do not get
    # extra weight in the paired bootstrap.
    p1 = paired_vectors(common_comp, one["posterior"][keep_one], B.Pairs(one["support_mean"][keep_one], rows.delta[one_held[keep_one]], common_comp, one_dose[keep_one], one_held[keep_one], rows.plate[one_held[keep_one]], rows.well[one_held[keep_one]], [tuple()] * len(keep_one)), one["null_z"][keep_one], np.ones(len(keep_one),dtype=bool))
    p2 = paired_vectors(two.compound[keep_two], two.support[keep_two], B.Pairs(two.support[keep_two], two.target[keep_two], two.compound[keep_two], two.dose[keep_two], two.held_index[keep_two], two.held_plate[keep_two], two.held_well[keep_two], [two.support_indices[int(j)] for j in keep_two]), two_null[keep_two], np.ones(len(keep_two),dtype=bool))
    compounds=sorted(set(p1)&set(p2)); z1=np.asarray([p1[c]["excess_z"] for c in compounds]); z2=np.asarray([p2[c]["excess_z"] for c in compounds]); q1=np.asarray([p1[c]["pcc"] for c in compounds]); q2=np.asarray([p2[c]["pcc"] for c in compounds])
    rng=np.random.default_rng(B.stable_seed(args.seed,"cross-budget-bootstrap")); idx=rng.integers(0,len(compounds),size=(args.bootstrap_rounds,len(compounds))); dz=(z1-z2); dq=(q1-q2); bz=dz[idx].mean(axis=1); bq=dq[idx].mean(axis=1)
    payload={"version":"cpg0004-LINCS-cross-budget-2026-08-30","seed":args.seed,"data":str(args.data),"one_r_predictions":str(args.one_r_predictions),"contrast":"1R frozen teacher+prior posterior vs 2R raw support mean","common_held_target_count":int(len(keep_one)),"paired_compound_count":int(len(compounds)),"row_matched_null_usable":{"one_r":int(one["usable"].sum()),"two_r":int(two_usable.sum()),"intersection":int(len(keep_one))},"exact_well_null_usable":{"one_r":0,"two_r":int(two_exact.sum())},"one_r_excess_z_mean":float(z1.mean()),"two_r_excess_z_mean":float(z2.mean()),"one_r_pcc_mean":float(q1.mean()),"two_r_pcc_mean":float(q2.mean()),"one_r_minus_two_r_excess_z":{"point":float(dz.mean()),"ci_low":float(np.quantile(bz,.025)),"ci_high":float(np.quantile(bz,.975)),"rounds":args.bootstrap_rounds},"one_r_minus_two_r_pcc":{"point":float(dq.mean()),"ci_low":float(np.quantile(bq,.025)),"ci_high":float(np.quantile(bq,.975)),"rounds":args.bootstrap_rounds},"guardrails":["same compound+dose+held plate target", "nested support construction from the frozen plate order", "1R posterior checkpoint and lambda are frozen before this contrast", "compound-level paired bootstrap", "row-matched null; exact-well availability reported"]}
    (args.outdir/"metrics.json").write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
    with (args.outdir/"paired_manifest.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.writer(h); w.writerow(["compound","dose","held_row","one_r_index","two_r_index"]); w.writerows([[c,d,int(hh),int(i),int(j)] for c,d,hh,i,j in zip(one_comp[keep_one],one_dose[keep_one],one_held[keep_one],keep_one,keep_two)])
    print(json.dumps(payload,indent=2,sort_keys=True))


if __name__ == "__main__": main()
