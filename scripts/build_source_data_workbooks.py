"""Build Supplementary Data 1 (figures) and 2 (tables) from source files in this repository.

Each sheet is a verbatim copy of one CSV file or of the ``rows`` list of one JSON file.
The only derived sheets are the Figure 3b per-condition and per-seed tables, computed
exactly as in figures/scripts/fig3_crossmodal.py. Row-level files larger than MAX_SHEET_ROWS are listed
in the INDEX sheet and remain available as CSV files in the repository.

Usage (from the repository root):
    python scripts/build_source_data_workbooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "source_data"
F2 = "figures/data/figure_2"
F3 = "figures/data/figure_3"
F4E = "figures/data/figure_4"
SD = "data/source_data"
MAX_SHEET_ROWS = 20_000
NOTES = {f"{F2}/figure2_panelE.csv": "Methods shown in Fig. 2d; other Supplementary Table S4-S5 rows are not in this file"}

FIGURES = [
    ("Fig. 2a", f"{F2}/figure2_panelA.csv"),
    ("Fig. 2a", f"{F2}/figure2_panelA_compounds.csv"),
    ("Fig. 2b", f"{F2}/figure2_panelB.csv"),
    ("Fig. 2b", f"{F2}/figure2_panelB_seed.csv"),
    ("Fig. 2b", f"{F2}/figure2_panelB_compounds.csv"),
    ("Fig. 2b", f"{F2}/figure2_panelC.csv"),
    ("Fig. 2b", f"{F2}/figure2_panelC_seed.csv"),
    ("Fig. 2c", f"{F2}/figure2_panelD.csv"),
    ("Fig. 2c", f"{F2}/figure2_panelD_all_metrics.csv"),
    ("Fig. 2d", f"{F2}/figure2_panelE.csv"),
    ("Fig. 2e", f"{F2}/figure2_panelF.csv"),
    ("Fig. 3a-c", f"{F3}/figure3_summary.csv"),
    ("Fig. 3b", "DERIVED:figure3b"),
    ("Fig. 3b", "DERIVED:figure3b_seed"),
    ("Fig. 3b", f"{F3}/figure3_paired_ge_scores.csv"),
    ("Fig. 3c", f"{F3}/inputs/cross_modal/sequential_evidence/conditional_marginal_gain.csv"),
    ("Fig. 4a-c", f"{SD}/figure4_cfra_summary.csv"),
    ("Fig. 4a", f"{SD}/figure4_cfra_module_macro.csv"),
    ("Fig. 4a-c", f"{SD}/figure4_cfra_per_compound.csv"),
    ("Fig. 5a-b", f"{SD}/predictor_main_source.csv"),
    ("Supplementary Fig. S1", f"{F3}/figure3_p0_conditioned_report_results.csv"),
    ("Supplementary Fig. S2", f"{F4E}/figure4_E_target_summary.csv"),
    ("Supplementary Fig. S2", f"{F4E}/figure4_E_target_macro_contrasts.csv"),
    ("Supplementary Fig. S2", f"{F4E}/figure4_E_direct_estimator_transfer_by_target.csv"),
    ("Supplementary Fig. S2", f"{F4E}/figure4_E_direct_estimator_transfer_boundary.csv"),
    ("Supplementary Fig. S2", f"{F4E}/figure4_E_drug_metrics.csv"),
    ("Supplementary Fig. S3", f"{SD}/figure4_supp_downstream_arm_metrics.csv"),
    ("Supplementary Fig. S3", f"{SD}/figure4_supp_downstream_contrasts.csv"),
    ("Supplementary Fig. S3", f"{SD}/figure4_supp_downstream_per_query.csv"),
]

TABLES = [
    ("Table 1", "TABLE1"),
    ("Supplementary Table S1", f"{F2}/figure2_panelB.csv"),
    ("Supplementary Table S1", f"{F2}/figure2_panelC.csv"),
    ("Supplementary Table S1", f"{F2}/figure2_panelD_all_metrics.csv"),
    ("Supplementary Table S1", f"{F2}/figure2_panelE_outer.csv"),
    ("Supplementary Table S1", f"{F2}/figure2_panelD_weights.csv"),
    ("Supplementary Table S2", f"{F3}/figure3_p0_conditioned_report_results.csv"),
    ("Supplementary Table S2", "tables/inputs/prior_rows.json"),
    ("Supplementary Table S3", f"{SD}/figure4_supp_downstream_contrasts.csv"),
    ("Supplementary Table S3", f"{F4E}/figure4_E_target_summary.csv"),
    ("Supplementary Table S3", f"{F4E}/figure4_E_direct_estimator_transfer_by_target.csv"),
    ("Supplementary Tables S4-S5", f"{F2}/figure2_panelE.csv"),
    ("Supplementary Table S6", f"{SD}/sciplex3_target_matrix_source.csv"),
]

# Table 1 is descriptive (datasets and representations), not a result.
TABLE1 = pd.DataFrame([
    ("BBBC047 (Rosetta)", "U2OS", "CP", "CellProfiler features", 775),
    ("BBBC047 (Rosetta)", "U2OS", "GE", "L1000 profiles", 977),
    ("cpg0004-LINCS", "A549", "CP", "CellProfiler features (241 biological + 1 batch coordinate)", 242),
    ("sci-Plex3", "A549, K562, MCF7", "GE", "RNA-seq pseudobulk", 2000),
], columns=["Dataset", "Cell line(s)", "Assay", "Representation", "Dimensions"])


def figure3b() -> pd.DataFrame:
    paired = pd.read_csv(ROOT / F3 / "figure3_paired_ge_scores.csv")
    g = paired.groupby(["canonical_compound", "dose"])[["p0", "fusion", "update"]].mean().reset_index()
    g["joint_prediction_gain"] = g["fusion"] - g["p0"]
    g["conditional_update_gain"] = g["update"] - g["p0"]
    return g.rename(columns={"p0": "target_only_agreement_seed_mean", "fusion": "joint_agreement_seed_mean",
                             "update": "conditional_agreement_seed_mean"})


def figure3b_seed() -> pd.DataFrame:
    """Per-training-run means drawn as open circles in Fig. 3b (figures/scripts/fig3_crossmodal.py)."""
    paired = pd.read_csv(ROOT / F3 / "figure3_paired_ge_scores.csv")
    rows = []
    for seed, g in paired.groupby("seed"):
        rows.append({"seed": seed, "n_conditions": len(g),
                     "joint_minus_target_only": (g["fusion"] - g["p0"]).mean(),
                     "conditional_minus_target_only": (g["update"] - g["p0"]).mean(),
                     "conditional_minus_joint": (g["update"] - g["fusion"]).mean()})
    return pd.DataFrame(rows)


def load(source: str) -> pd.DataFrame:
    if source == "TABLE1":
        return TABLE1
    if source == "DERIVED:figure3b":
        return figure3b()
    if source == "DERIVED:figure3b_seed":
        return figure3b_seed()
    path = ROOT / source
    if path.suffix == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["rows"])
    return pd.read_csv(path)


def describe(source: str) -> str:
    if source == "TABLE1":
        return "Manuscript Table 1 (descriptive)"
    if source == "DERIVED:figure3b":
        return f"{F3}/figure3_paired_ge_scores.csv, seed mean per compound and dose"
    if source == "DERIVED:figure3b_seed":
        return f"{F3}/figure3_paired_ge_scores.csv, mean over conditions within each training seed"
    return source


def style_sheet(ws) -> None:
    for c in ws[1]:
        c.font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        c.fill = PatternFill("solid", fgColor="3C786F")
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    for col in ws.columns:
        sample = list(col)[:200]
        width = min(60, max(12, max(len(str(c.value or "")) for c in sample) + 2))
        ws.column_dimensions[col[0].column_letter].width = width


def build(items, prefix: str, title: str, path: Path) -> None:
    wb = Workbook()
    index = wb.active
    index.title = "INDEX"
    index.append(["Sheet", "Display item", "Source", "Rows", "Note"])
    counters: dict[str, int] = {}
    for item, source in items:
        frame = load(source)
        if len(frame) > MAX_SHEET_ROWS:
            index.append(["(CSV only)", item, describe(source), len(frame), "Row-level file provided as CSV"])
            continue
        key = item.replace("Supplementary ", "S").replace("Fig. ", "F").replace("Table ", "T").replace("Tables ", "T")
        key = key.replace(" ", "").replace(",", "")
        counters[key] = counters.get(key, 0) + 1
        name = f"{key}_{counters[key]}"[:31]
        ws = wb.create_sheet(name)
        ws.append(list(frame.columns))
        for row in frame.itertuples(index=False, name=None):
            ws.append([None if (isinstance(x, float) and pd.isna(x)) else x for x in row])
        style_sheet(ws)
        note = NOTES.get(source, "") if item.startswith("Supplementary Tables S4") else ""
        index.append([name, item, describe(source), len(frame), note])
    style_sheet(index)
    index.insert_rows(1)
    index["A1"] = title
    index["A1"].font = Font(name="Arial", bold=True, size=11)
    wb.save(path)
    print(f"{path.name}: {len(wb.sheetnames) - 1} data sheets")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build(FIGURES, "F", "Supplementary Data 1 | Source data for Figures 2-5 and Supplementary Figures S1-S3",
          OUT / "Supplementary_Data_1.xlsx")
    build(TABLES, "T", "Supplementary Data 2 | Source data for Table 1 and Supplementary Tables S1-S6",
          OUT / "Supplementary_Data_2.xlsx")


if __name__ == "__main__":
    main()
