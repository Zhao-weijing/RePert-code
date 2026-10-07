#!/usr/bin/env python3
"""Assemble the locked Morph-A–D decisions without recomputing any endpoint."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def row(path: Path, comparison: str) -> dict:
    table = pd.read_csv(path)
    hit = table[table.comparison.astype(str).eq(comparison)]
    if len(hit) != 1:
        raise RuntimeError(f"Expected one {comparison} row in {path}")
    return hit.iloc[0].to_dict()


def effect(item: dict, prefix: str = "") -> str:
    point = item.get(prefix + "point", item.get("point"))
    low = item.get(prefix + "ci_low", item.get("ci_low")); high = item.get(prefix + "ci_high", item.get("ci_high"))
    return f"{float(point):+.4f} (95% CI {float(low):+.4f} to {float(high):+.4f})"


def main() -> None:
    a = ROOT / "module_recovery" / "DECISION.csv"
    b = ROOT / "dominant_feature_recovery" / "DECISION.csv"
    c = ROOT / "module_dose_trajectories" / "DECISION.csv"
    d = ROOT / "response_programs" / "DECISION.csv"
    a_t, a_g = row(a, "teacher_minus_raw"), row(a, "posterior_ge_minus_teacher")
    b_t, b_g = row(b, "teacher_minus_raw"), row(b, "posterior_ge_minus_teacher")
    c_t, c_g = row(c, "teacher_minus_raw"), row(c, "posterior_minus_teacher")
    d_rows = pd.read_csv(d)
    d_status = "; ".join(f"{x.comparison}: {x.decision}" for x in d_rows.itertuples(index=False))
    k = pd.read_csv(ROOT / "response_programs" / "K_SELECTION.csv")
    final = ROOT / "final_synthesis"; final.mkdir(parents=True, exist_ok=True)
    table = [
        ("Morph-A", "teacher 是否改善细胞结构模块恢复", a_t["broad_status"], effect(a_t), int(a_t["n_compounds"])),
        ("Morph-A-GE", "GE 是否进一步改善模块恢复", a_g["decision"], effect(a_g), int(a_g["n_compounds"])),
        ("Morph-B", "teacher 是否改善主要形态变化及方向恢复", b_t["decision"], f"overlap {float(b_t['overlap_point']):+.4f} [{float(b_t['overlap_ci_low']):+.4f}, {float(b_t['overlap_ci_high']):+.4f}]; sign {float(b_t['sign_point']):+.4f} [{float(b_t['sign_ci_low']):+.4f}, {float(b_t['sign_ci_high']):+.4f}]", int(b_t["n_overlap"])),
        ("Morph-B-GE", "GE 是否进一步改善主要形态变化及方向恢复", b_g["decision"], f"overlap {float(b_g['overlap_point']):+.4f} [{float(b_g['overlap_ci_low']):+.4f}, {float(b_g['overlap_ci_high']):+.4f}]; sign {float(b_g['sign_point']):+.4f} [{float(b_g['sign_ci_low']):+.4f}, {float(b_g['sign_ci_high']):+.4f}]", int(b_g["n_overlap"])),
        ("Morph-C", "teacher 是否改善模块级剂量轨迹", c_t["decision"], effect(c_t), int(c_t["n_compounds"])),
        ("Morph-C-GE", "GE 是否进一步改善模块级剂量轨迹", c_g["decision"], effect(c_g), int(c_g["n_compounds"])),
        ("Morph-D", "数据驱动 response programs", "NOT RUN", d_status, 0),
    ]
    lines = ["# Final decision — cpg0004-LINCS morphological biology", "", "| Level | Core question | Decision | Locked effect / reason | Compound units |", "|---|---|---|---|---:|"]
    lines += [f"| {level} | {question} | **{decision}** | {detail} | {n} |" for level, question, decision, detail, n in table]
    lines += ["", "## Main conclusion", "", "Frozen reproducible-effect learning improved recovery of independent-repeat-supported major Cell Painting changes and multiple predefined cellular-structure modules. The same direction persisted for the GE-updated posterior. The improvement also extended to module-level six-dose morphological trajectory organization.", "", "## Scope and boundary", "", "This is morphology recovery evidence only: it does not assert pathway, target, MoA, potency, or physical-repeat replacement. `Batch_Number` was excluded from every biological endpoint. All reported positive gates use the locked three-seed direction plus 10,000-round paired compound-bootstrap CI rule.", "", "## Morph-D boundary", "", "Morph-D was triggered by the A/B/C GO results, but did not score test profiles. Under the precommitted validation-only compact/stability rule, K=8 was stable but missed the one-standard-error reconstruction criterion; K=32 met reconstruction but its stability lower quantile was below 0.90. The rule was not relaxed after observing validation data.", "", "## Auditable source artifacts", "", "- `protocol/CONFIG.json` and `INPUT_HASHES.csv`", "- `feature_annotation_audit/AUDIT.md`", "- `module_recovery/DECISION.md`", "- `dominant_feature_recovery/DECISION.md`", "- `module_dose_trajectories/DECISION.md`", "- `response_programs/DECISION.md` and `K_SELECTION.csv`", ""]
    decision = "\n".join(lines)
    (final / "FINAL_DECISION.md").write_text(decision, encoding="utf-8")
    (ROOT / "FINAL_SUMMARY.md").write_text(decision, encoding="utf-8")
    (final / "SUMMARY.json").write_text(json.dumps({"rows": table, "morph_d_k_selection": k.to_dict("records")}, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"outdir": str(final), "final_summary": str(ROOT / 'FINAL_SUMMARY.md')}, indent=2))


if __name__ == "__main__":
    main()
