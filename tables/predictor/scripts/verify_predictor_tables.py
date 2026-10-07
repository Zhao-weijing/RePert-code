"""Check frozen values, reference-style cell placement and the compiled PDF."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/"tables/predictor/generated"
METRICS={"full_target_pcc","delta_pcc","raw_target_rmse"}
NUMS=("raw_mean","raw_ci_low","raw_ci_high","comparator_mean","comparator_ci_low",
      "comparator_ci_high","difference","difference_ci_low","difference_ci_high")


def read(p):
    with p.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(r):
    return tuple(r[k] for k in ("dataset","setting","target_line","metric"))


def main():
    pdf=OUT/"predictor_tables_preview.pdf"
    reader=PdfReader(pdf)
    assert len(reader.pages)==2
    texts=[p.extract_text() for p in reader.pages]
    checks={}
    for i,(name,expected) in enumerate((("predictor_main_source.csv",15),("predictor_celllines_source.csv",18))):
        now=read(ROOT/"tables/predictor/results"/name);old=read(HERE/"archive_pre_rmse"/name)
        assert len(now)==expected and set(r["metric"] for r in now)==METRICS
        assert all(r["comparator_method"]=="CFRA" for r in now if r["dataset"]=="sci-Plex3")
        current={key(r):r for r in now}
        style_baseline=HERE/"archive_pre_reference_style"/name
        assert (ROOT/"tables/predictor/results"/name).read_bytes()==style_baseline.read_bytes(), "style revision changed source data"
        for r in old:
            if r["metric"]=="E":
                continue
            assert all(float(r[k])==float(current[key(r)][k]) for k in NUMS), "original correlation changed"
        for r in now:
            assert math.isclose(float(r["comparator_mean"])-float(r["raw_mean"]),float(r["difference"]),abs_tol=2e-10)
            assert r["preferred_direction"]==("lower" if r["metric"]=="raw_target_rmse" else "higher")
            for prefix,keys in (("raw",NUMS[:3]),("comparator",NUMS[3:6]),("difference",NUMS[6:9])):
                if i==0 and prefix=="difference":
                    continue  # Main table intentionally displays only the two methods.
                digits=4
                while any(float(r[k])!=0 and round(float(r[k]),digits)==0 for k in keys):
                    digits+=1
                for k in keys:
                    assert f"{abs(float(r[k])):.{digits}f}" in texts[i], f"PDF value missing {r} {k}"
        checks[name]=dict(rows=len(now),original_correlations_exactly_unchanged=True,
            all_data_byte_identical_to_pre_style=True,all_displayed_numbers_found_in_pdf=True)
    assert "CFRA+Res" not in texts[1]
    assert not re.search(r"\bDifference\b",texts[0])
    assert (OUT/"table_predictor_celllines.tex").read_bytes()==(
        HERE/"archive_pre_main_difference_removal/table_predictor_celllines.tex").read_bytes()
    for t in texts:
        assert "RMSE" in t and "↑" in t and "↓" in t
        assert "zsame" not in t and not re.search(r"\bE\b",t)
    for name,csv_name,count in (("table_predictor_supervision.tex","predictor_main_source.csv",5),
                                ("table_predictor_celllines.tex","predictor_celllines_source.csv",6)):
        source=(OUT/name).read_text(encoding="utf-8")
        show_difference=name=="table_predictor_celllines.tex"
        prefixes=("raw","comparator","difference") if show_difference else ("raw","comparator")
        assert source.count(r"PCC $\uparrow$")==4  # two metric headers, twice
        assert source.count(r"RMSE $\downarrow$")==2
        assert source.count(r"\CFRATableCell{")==3*len(prefixes)*count
        assert source.count(r"\CFRATableCell{\textbf{")==3*count
        assert source.count(r"\textit{Difference}")==(count if show_difference else 0)
        # Check every physical row/metric column, including which mean is bold.
        data_rows=[s for s in source.splitlines() if r"\CFRATableCell{" in s]
        assert len(data_rows)==len(prefixes)*count
        groups={}
        for r in read(ROOT/"tables/predictor/results"/csv_name):
            groups.setdefault((r["dataset"],r["setting"],r["target_line"]),{})[r["metric"]]=r
        for g_index,metric_map in enumerate(groups.values()):
            for arm_index,prefix in enumerate(prefixes):
                physical_row=data_rows[g_index*len(prefixes)+arm_index].split(" & ")
                assert len(physical_row)==6
                for m_index,metric in enumerate(("full_target_pcc","delta_pcc","raw_target_rmse")):
                    r=metric_map[metric]
                    keys=NUMS[6:9] if prefix=="difference" else (
                        prefix+"_mean",prefix+"_ci_low",prefix+"_ci_high")
                    digits=4
                    while any(float(r[k])!=0 and round(float(r[k]),digits)==0 for k in keys):
                        digits+=1
                    expected=[]
                    for j,k in enumerate(keys):
                        value=float(r[k])
                        formatted=f"{value:+.{digits}f}" if prefix=="difference" and j==0 else f"{value:.{digits}f}"
                        expected.append(formatted.replace("-",r"\ensuremath{-}"))
                    comp_better=(float(r["comparator_mean"]) < float(r["raw_mean"]) if metric=="raw_target_rmse"
                                 else float(r["comparator_mean"]) > float(r["raw_mean"]))
                    is_bold=prefix!="difference" and ((prefix=="comparator")==comp_better)
                    if is_bold:
                        expected[0]=r"\textbf{"+expected[0]+"}"
                    expected_cell=r"\CFRATableCell{"+"}{".join(expected)+"}"
                    assert physical_row[3+m_index].startswith(expected_cell), "misplaced or incorrectly bolded cell"
    assert "0.00005" in texts[1]
    for log in (OUT/"predictor_tables_preview.log",ROOT/"tmp/pdfs/predictor_table_20260922/integration_width_qa.log"):
        s=log.read_text(errors="replace")
        assert not re.search(r"Overfull|Underfull|Float too large",s)
    audit=dict(status="PASS",pages=2,checks=checks,arrows_verified=True,no_E_rows=True,
        sciPlex3_M2_absent=True,near_zero_ci_preserved=True,no_layout_overflow=True,
        all_numeric_cells_in_correct_row_and_column=True,better_mean_bolding_verified=True,
        main_display_rows=10,main_difference_rows_absent=True,supplement_tex_byte_identical=True,
        original_paired_statistics_preserved_in_source_data=True,
        pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
    (HERE/"final_qa.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
