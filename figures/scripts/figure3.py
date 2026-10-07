"""Figure 3: auditable cross-modal evidence comparisons; never trains models.

Run with the bundled Python. Inputs remain in the project experiment tree; all
used compact tables are copied to data/figure_3/inputs for portability.
If data/figure_3/conditioned_residual/seed_*/per_molecule_scores.csv is supplied,
the newer update is used only after exact paired-baseline and null checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import figure_style as STYLE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
EXP = PROJECT / "analysis"
MULTI = EXP / "cross_modal"
DATA = ROOT / "data" / "figure_3"
INPUTS = DATA / "inputs"
NOTES = ROOT / "notes"
SEEDS = (3407, 42, 2025)
BOOTSTRAP_ROUNDS = 10000
METHOD_VERSION = "initial"
SOURCES = []
AUDIT = []
BLUE = STYLE.COLORS["cp"]
TEAL = STYLE.COLORS["ge"]
ORANGE = STYLE.COLORS["update"]
GRAY = STYLE.COLORS["control"]
NOTE = "#59616B"
INK = STYLE.COLORS["text"]


def copy_source(path: Path) -> Path:
    relative = path.relative_to(EXP)
    target = INPUTS / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    original_available = path.is_file()
    if original_available:
        shutil.copy2(path, target)
    elif not target.is_file():
        raise FileNotFoundError(f"Neither original source nor packaged input is available: {path}")
    SOURCES.append({"source": str(path), "copy": str(target.relative_to(ROOT)),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "original_available_at_render": original_available})
    return target


def read_source(path: Path) -> pd.DataFrame:
    return pd.read_csv(copy_source(path))


def paired_stats(values: np.ndarray, key: str) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid paired sample for {key}")
    # Common draws across all comparisons preserve identical intervals for an
    # estimate repeated in panels b and c and retain paired covariation.
    bootstrap_seed = int.from_bytes(hashlib.sha256(b"figure3-common-compound-draws-20260910").digest()[:8], "little")
    rng = np.random.default_rng(bootstrap_seed)
    sampled = np.empty(BOOTSTRAP_ROUNDS)
    for start in range(0, BOOTSTRAP_ROUNDS, 100):
        stop = min(start + 100, BOOTSTRAP_ROUNDS)
        draw = rng.integers(0, len(values), (stop - start, len(values)))
        sampled[start:stop] = values[draw].mean(axis=1)
    return float(values.mean()), *map(float, np.quantile(sampled, [0.025, 0.975]))


def row(panel, contrast, estimate, low, high, n, version, source, **extra):
    return dict(panel=panel, contrast=contrast, estimate=float(estimate), ci_low=float(low),
                ci_high=float(high), n_compounds=int(n), estimator_version=version,
                metric="paired_change_repeat_agreement_fisher_z_fixed_null_cancels", source=source, **extra)


def from_saved(panel, table, key, version, source):
    found = table.loc[table.contrast == key]
    if len(found) != 1:
        raise ValueError(f"Expected one saved row: {key}, got {len(found)}")
    r = found.iloc[0]
    lo, hi = json.loads(r.excess_z_ci95)
    return row(panel, key, r.excess_z_point, lo, hi, r.molecule_count, version, source,
               interval_source="saved compound-paired bootstrap")


def method_index(table, method, column="canonical_compound"):
    out = table.loc[table.method == method].copy().set_index(column)
    if out.index.has_duplicates or not len(out):
        raise ValueError(f"Missing or duplicate compound rows: {method}")
    return out.sort_index()


def verify_equal(left, right, label):
    if set(left.index) != set(right.index):
        raise ValueError(f"Compound sets differ: {label}")
    columns = ["raw_pcc", "model_z", "null_z", "model_excess_z"]
    errors = (left[columns] - right[columns]).abs().max()
    if not (errors <= 1e-12).all():
        raise ValueError(f"Baseline/null mismatch: {label}: {errors.to_dict()}")
    if "dose" in left and "dose" in right:
        if not np.allclose(left.dose.astype(float), right.dose.astype(float), atol=1e-10, rtol=0):
            raise ValueError(f"Dose mismatch: {label}")
    AUDIT.append(dict(check=label, n=len(left), max_abs_error=errors.to_dict(), passed=True))


def prepare_data():
    global METHOD_VERSION
    for folder in (DATA, NOTES, INPUTS):
        folder.mkdir(parents=True, exist_ok=True)
    rows = []
    # A uses report values to their published precision; its matched control has
    # a smaller intersection. Thus these are paired contrasts, not three arms.
    a_path = copy_source(EXP / "bbbc047_1r_ge_residual_evidence" / "RESULTS.md")
    a_source = str(a_path.relative_to(ROOT))
    for key, point, lo, hi, n in [
        ("correct_minus_p0", .02150, .01937, .02366, 4023),
        ("correct_minus_shuffled", .03742, .03488, .03995, 4023),
        ("correct_minus_matched_foreign", .01081, .00903, .01261, 3538),
    ]:
        rows.append(row("A", key, point, lo, hi, n, "GE_to_CP_residual_vector_control", a_source,
                        interval_source="saved report, rounded to five decimals"))
    a_sensitivity = row("supplement", "correct_minus_exact_plate_dose_foreign", .01854,
                        .01421, .02293, 715, "GE_to_CP_residual_vector_control", a_source,
                        interval_source="saved report, rounded to five decimals")

    legacy_path = MULTI / "cell_painting_to_expression" / "paired_contrasts.csv"
    legacy_summary = read_source(legacy_path)
    legacy_pooled = legacy_summary.loc[legacy_summary.scope == "pooled_three_seed"]
    fusion_summary = read_source(MULTI / "direct_fusion" / "bootstrap_summary.csv")
    fusion_pooled = fusion_summary.loc[fusion_summary.scope == "pooled_three_seed"]
    for filename in ["CONDITIONAL_RESIDUAL_AUDIT_20260902.md"]:
        copy_source(MULTI / "cell_painting_to_expression" / filename)
    copy_source(MULTI / "control_conditioned_residual" / "RESULTS_20260902.md")
    copy_source(MULTI / "control_conditioned_residual" / "run_p0_conditioned_cp_residual.py")
    copy_source(MULTI / "expression_teacher" / "ge_teacher_experiment.py")
    copy_source(EXP / "bbbc047_repro_effect_mvp" / "repro_effect_mvp.py")
    copy_source(EXP / "bbbc047_repro_effect_virtual_cp_mvp" / "repro_effect_prior_posterior_controls.py")
    copy_source(EXP / "bbbc047_1r_ge_residual_evidence" / "evaluate_1r_ge_residual.py")
    copy_source(MULTI / "sequential_evidence" / "sequential_evidence.py")

    by_seed = {}
    new_paths = {seed: DATA / "conditioned_residual" / f"seed_{seed}" / "per_molecule_scores.csv"
                 for seed in SEEDS}
    use_new = all(path.is_file() for path in new_paths.values())
    if use_new:
        METHOD_VERSION = "P0_conditioned"
    pairs = []
    for seed in SEEDS:
        direct = read_source(MULTI / "direct_fusion" / "formal" / f"seed_{seed}" / "per_molecule_scores.csv")
        baseline = method_index(direct, "GE_teacher")
        fusion = method_index(direct, "direct_fusion")
        legacy_frozen = read_source(MULTI / "cell_painting_to_expression" / f"seed_{seed}" / "frozen.csv").set_index("canonical_compound").sort_index()
        legacy_correct = read_source(MULTI / "cell_painting_to_expression" / f"seed_{seed}" / "correct.csv").set_index("canonical_compound").sort_index()
        verify_equal(baseline, legacy_frozen, f"fusion vs initial-update baseline, seed {seed}")
        if not np.allclose(fusion.null_z, legacy_correct.null_z, atol=1e-12, rtol=0):
            raise ValueError("Initial update and fusion null differ")
        if use_new:
            new = pd.read_csv(new_paths[seed])
            SOURCES.append({"source": f"user@compute-host:/path/to/data/AIDD/MVCPert_5_27/runs/multimodal_expansion_p0_conditioned_20260902/formal/seed_{seed}/per_molecule_scores.csv", "copy": str(new_paths[seed].relative_to(ROOT)),
                            "sha256": hashlib.sha256(new_paths[seed].read_bytes()).hexdigest()})
            frozen = method_index(new, "frozen_P0")
            verify_equal(baseline, frozen, f"fusion vs P0-conditioned-update baseline, seed {seed}")
            correct = method_index(new, "correct_CP")
            shuffled = method_index(new, "shuffled_CP")
            foreign = method_index(new, "matched_foreign_CP")
            for label, arm in [("correct", correct), ("shuffled", shuffled), ("foreign", foreign)]:
                if set(arm.index) != set(baseline.index) or not np.allclose(arm.null_z, baseline.null_z, atol=1e-12, rtol=0):
                    raise ValueError(f"New arm/null mismatch: {seed} {label}")
        else:
            correct = legacy_correct
            shuffled = read_source(MULTI / "cell_painting_to_expression" / f"seed_{seed}" / "shuffled.csv").set_index("canonical_compound").sort_index()
            foreign = read_source(MULTI / "cell_painting_to_expression" / f"seed_{seed}" / "matched_foreign.csv").set_index("canonical_compound").sort_index()
        by_seed[seed] = {"p0": baseline, "fusion": fusion, "correct": correct,
                         "shuffled": shuffled, "foreign": foreign}
        for compound in baseline.index:
            pairs.append(dict(seed=seed, canonical_compound=compound, dose=baseline.loc[compound, "dose"],
                              p0=float(baseline.loc[compound, "model_excess_z"]),
                              fusion=float(fusion.loc[compound, "model_excess_z"]),
                              update=float(correct.loc[compound, "model_excess_z"]),
                              shuffled=float(shuffled.loc[compound, "model_excess_z"]),
                              matched_control=float(foreign.loc[compound, "model_excess_z"]),
                              version=METHOD_VERSION))
    pair_table = pd.DataFrame(pairs)
    pair_table.to_csv(DATA / "figure3_paired_ge_scores.csv", index=False)
    pooled = pair_table.groupby("canonical_compound", sort=True)[["p0", "fusion", "update", "shuffled", "matched_control"]].mean()
    if not pair_table.groupby("canonical_compound").seed.nunique().eq(3).all():
        raise ValueError("Not all compounds occur in all three seeds")
    contrast_cols = {"correct_minus_p0": ("update", "p0"),
                     "correct_minus_shuffled": ("update", "shuffled"),
                     "correct_minus_matched_foreign": ("update", "matched_control")}
    for key, (left, right) in contrast_cols.items():
        if use_new:
            point, lo, hi = paired_stats(pooled[left] - pooled[right], "figure3-new-" + key)
            rows.append(row("B", key, point, lo, hi, len(pooled), METHOD_VERSION,
                            "data/figure_3/figure3_paired_ge_scores.csv", interval_source="recomputed: seed-average then compound bootstrap"))
        else:
            rows.append(from_saved("B", legacy_pooled, key, METHOD_VERSION,
                                   str((INPUTS / legacy_path.relative_to(EXP)).relative_to(ROOT))))
    for key, left, right in [("fusion_minus_p0", "fusion", "p0"),
                              ("update_minus_p0", "update", "p0"),
                              ("update_minus_fusion", "update", "fusion")]:
        values = pooled[left] - pooled[right]
        if use_new:
            point, lo, hi = paired_stats(values, "figure3-new-" + key)
        else:
            if key == "fusion_minus_p0":
                rr = fusion_pooled.loc[(fusion_pooled.left == "direct_fusion") & (fusion_pooled.right == "GE_teacher")].iloc[0]
                point = float(rr.excess_z_point); lo, hi = json.loads(rr.excess_z_ci95)
            elif key == "update_minus_p0":
                rr = legacy_pooled.loc[legacy_pooled.contrast == "correct_minus_p0"].iloc[0]
                point = float(rr.excess_z_point); lo, hi = json.loads(rr.excess_z_ci95)
            else:
                rr = fusion_pooled.loc[(fusion_pooled.left == "direct_fusion") & (fusion_pooled.right == "CP_residual")].iloc[0]
                point = -float(rr.excess_z_point); neg_lo, neg_hi = json.loads(rr.excess_z_ci95)
                lo, hi = -neg_hi, -neg_lo
            if not np.isclose(values.mean(), point, atol=1e-12, rtol=0):
                raise ValueError(f"Saved figure3 C point disagrees with paired rows: {key}")
        rows.append(row("C", key, point, lo, hi, len(pooled), METHOD_VERSION,
                        "data/figure_3/figure3_paired_ge_scores.csv",
                        interval_source="recomputed compound bootstrap" if use_new else "saved paired CI; point verified from per-compound rows"))

    seq_path = MULTI / "sequential_evidence" / "conditional_marginal_gain.csv"
    seq_summary = read_source(seq_path)
    seq_pooled = seq_summary.loc[seq_summary.scope == "pooled_three_seed"]
    for key in ("GE|CP1", "GE|CP1+CP2", "CP2|CP1", "CP2|CP1+GE"):
        rows.append(from_saved("D" if key.startswith("GE") else "supplement", seq_pooled, key,
                               "sequential_fixed_CP_held_reference", str((INPUTS / seq_path.relative_to(EXP)).relative_to(ROOT))))
    seq_parts = []
    for seed in SEEDS:
        table = read_source(MULTI / "sequential_evidence" / "formal" / f"seed{seed}" / "per_molecule_scores.csv")
        table["seed"] = seed
        copy_source(MULTI / "sequential_evidence" / "formal" / f"seed{seed}" / "pair_manifest.csv")
        seq_parts.append(table)
    pd.concat(seq_parts, ignore_index=True).to_csv(DATA / "figure3_sequential_per_compound.csv", index=False)
    # Seed-specific newer results retained even when remote row exports are unavailable.
    new_report_rows = []
    report_values = {
        3407: [("correct_minus_p0", .011512, .007037, .015997), ("correct_minus_shuffled", .016996, .011512, .022518), ("correct_minus_matched_foreign", .014753, .009347, .020229)],
        42: [("correct_minus_p0", .008020, .003556, .012363), ("correct_minus_shuffled", .015020, .010034, .020039), ("correct_minus_matched_foreign", .010200, .005578, .014917)],
        2025: [("correct_minus_p0", .008585, .004423, .012695), ("correct_minus_shuffled", .010503, .006469, .014635), ("correct_minus_matched_foreign", .009225, .005360, .013028)],
    }
    for seed, values in report_values.items():
        for key, point, lo, hi in values:
            if use_new:
                left, right = contrast_cols[key]
                actual = float((pair_table.loc[pair_table.seed == seed, left] -
                                pair_table.loc[pair_table.seed == seed, right]).mean())
                if abs(actual - point) > .00000051:
                    raise ValueError(f"New source mean does not match report precision: {seed} {key}")
                AUDIT.append(dict(check=f"New seed mean matches saved report: {seed} {key}",
                                  actual_mean=actual, report_rounded_mean=point, passed=True))
            new_report_rows.append(row("supplement", key, point, lo, hi, 3691, "P0_conditioned",
                                       "analysis/cross_modal/control_conditioned_residual/RESULTS_20260902.md",
                                       seed=seed, interval_source="saved report rounded to six decimals"))
    pd.DataFrame(new_report_rows).to_csv(DATA / "figure3_p0_conditioned_report_results.csv", index=False)
    rows.append(a_sensitivity)
    summary = pd.DataFrame(rows)
    summary.to_csv(DATA / "figure3_summary.csv", index=False)
    pd.DataFrame(SOURCES).drop_duplicates("copy").to_csv(DATA / "figure3_source_manifest.csv", index=False)
    audit = dict(selected_update_version=METHOD_VERSION, available_new_per_compound_exports=use_new,
                 n_ge_compounds=len(pooled), seeds=list(SEEDS), checks=AUDIT,
                 bootstrap_rounds=BOOTSTRAP_ROUNDS,
                 bootstrap_rng="numpy.default_rng", bootstrap_seed=int.from_bytes(hashlib.sha256(b"figure3-common-compound-draws-20260910").digest()[:8], "little"),
                 inference="Seed-average paired differences are bootstrapped by compound; seeds are not independent biological samples.",
                 verification_only=True, no_model_fitting=True)
    (NOTES / "figure3_data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return summary, pd.DataFrame(new_report_rows)


def apply_style():
    STYLE.setup()


def axes_clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def panel_title(ax, letter, title):
    ax.text(-.18, 1.25, letter, transform=ax.transAxes, fontsize=10, weight="bold", va="bottom")
    ax.text(0, 1.25, title, transform=ax.transAxes, fontsize=8.5, va="bottom")


def forest(ax, table, color, target):
    ordered = table.set_index("contrast").loc[["correct_minus_p0", "correct_minus_shuffled", "correct_minus_matched_foreign"]]
    for y, (_, r) in zip([2, 1, 0], ordered.iterrows()):
        ax.errorbar(r.estimate, y, xerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]],
                    fmt="D" if target == "GE" else "o", color=color, markersize=4.2, elinewidth=1.15, capsize=2.5, capthick=.9)
        if target == "CP":
            ax.text(.99, y + .22, f"n = {int(r.n_compounds):,}", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=8, color=NOTE)
    ax.set_yticks([2, 1, 0], ["Target-only\nestimate", "Shuffled\npairing", "Matched\ncontrol"])
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_ylim(-.55, 2.62)
    ax.set_xlim(-.002, .045 if target == "CP" else .038)
    ax.axvline(0, color=GRAY, linewidth=.7, dashes=(2, 2))
    ax.set_xticks([0, .01, .02, .03, .04] if target == "CP" else [0, .01, .02, .03])
    ax.set_xlabel("Change in repeat agreement\n(Fisher z)")
    ax.text(0, 1.07, "Gain over each comparator" + ("\nn = 3,691" if target == "GE" else ""),
            transform=ax.transAxes, fontsize=8, color=NOTE)
    axes_clean(ax)
    ax.spines["left"].set_visible(False)


def save_figure(fig, stem):
    STYLE.save_figure(fig, stem)


def plot_main(summary):
    apply_style()
    fig = STYLE.new_figure(height_mm=174, width_mm=180)
    grid = fig.add_gridspec(2, 2, left=.176, right=.985, bottom=.115, top=.845, wspace=.78, hspace=1.05)
    axa = fig.add_subplot(grid[0, 0]); axb = fig.add_subplot(grid[0, 1])
    axc = fig.add_subplot(grid[1, 0]); axd = fig.add_subplot(grid[1, 1])
    forest(axa, summary[summary.panel == "A"], BLUE, "CP")
    forest(axb, summary[summary.panel == "B"], ORANGE, "GE")
    panel_title(axa, "a", "GE → CP")
    panel_title(axb, "b", "CP → GE")

    c = summary[summary.panel == "C"].set_index("contrast")
    for x, key, color in [(0, "fusion_minus_p0", GRAY), (1, "update_minus_p0", ORANGE)]:
        r = c.loc[key]
        axc.errorbar(x, r.estimate, yerr=[[r.estimate-r.ci_low], [r.ci_high-r.estimate]],
                    fmt="D" if key == "update_minus_p0" else "o", color=color, markersize=5, elinewidth=1.2, capsize=3)
    axc.axhline(0, color=GRAY, linewidth=.7, dashes=(2, 2))
    axc.set_xticks([0, 1], ["Joint\nprediction", "Conditional\nupdate" if METHOD_VERSION == "P0_conditioned" else "Residual\nupdate"])
    axc.set_xlim(-.45, 1.45)
    axc.set_ylim(-.001, .020)
    axc.set_yticks([0, .005, .010, .015, .020])
    axc.set_ylabel("Change in repeat agreement\n(Fisher z)")
    delta = c.loc["update_minus_fusion"]
    axc.plot([0, 0, 1, 1], [.0168, .0173, .0173, .0168], color=INK, linewidth=.65)
    axc.text(.5, .0180, f"Δ = +{delta.estimate:.4f}\n95% CI {delta.ci_low:.4f}–{delta.ci_high:.4f}", ha="center", fontsize=8)
    axc.text(0, 1.06, "Gain over target-only GE\nn = 3,691", transform=axc.transAxes, fontsize=8, color=NOTE)
    panel_title(axc, "c", "Multimodal integration")
    axes_clean(axc)

    d = summary[summary.panel == "D"].set_index("contrast")
    estimates = [d.loc[k].estimate for k in ["GE|CP1", "GE|CP1+CP2"]]
    axd.plot([0, 1], estimates, color=BLUE, linewidth=1)
    for x, key in [(0, "GE|CP1"), (1, "GE|CP1+CP2")]:
        r = d.loc[key]
        axd.errorbar(x, r.estimate, yerr=[[r.estimate-r.ci_low], [r.ci_high-r.estimate]],
                    fmt="o", color=BLUE, markersize=4.8, elinewidth=1.2, capsize=3)
        axd.text(x, r.ci_high + .0011, f"+{r.estimate:.4f}", ha="center", fontsize=8)
    axd.set_xticks([0, 1], ["1 CP\nmeasurement", "2 CP\nmeasurements"])
    axd.set_xlim(-.4, 1.4)
    axd.set_ylim(0, .026)
    axd.set_yticks([0, .01, .02])
    axd.set_ylabel("Increment from GE\n(Fisher z)")
    axd.text(0, 1.06, "Same held-out CP measurement\nn = 3,983", transform=axd.transAxes, fontsize=8, color=NOTE)
    panel_title(axd, "d", "Value of GE evidence")
    axes_clean(axd)
    save_figure(fig, "figure3")
    plt.close(fig)


def plot_p0_supplement(table):
    apply_style()
    fig, ax = plt.subplots(figsize=(180 / 25.4, 92 / 25.4))
    fig.subplots_adjust(left=.34, right=.97, top=.81, bottom=.18)
    labels = {"correct_minus_p0": "Target-only estimate", "correct_minus_shuffled": "Shuffled CP pairing",
              "correct_minus_matched_foreign": "Matched CP control"}
    colors = [TEAL, BLUE, ORANGE]
    for group, key in enumerate(labels):
        for offset, seed, color, marker in zip([.23, 0, -.23], SEEDS, colors, ["o", "s", "^"]):
            r = table[(table.contrast == key) & (table.seed == seed)].iloc[0]
            ax.errorbar(r.estimate, 2-group+offset, xerr=[[r.estimate-r.ci_low], [r.ci_high-r.estimate]],
                        fmt=marker, color=color, markersize=3.6, capsize=2,
                        label=f"Run {SEEDS.index(seed)+1}" if group == 0 else None)
    ax.set_yticks([2, 1, 0], list(labels.values()))
    ax.set_ylim(-.55, 2.65); ax.set_xlim(-.001, .025)
    ax.axvline(0, color=GRAY, linewidth=.7, dashes=(2, 2))
    ax.set_xlabel("Change in repeat agreement (Fisher z)")
    ax.text(0, 1.23, "CP → GE: conditional update", transform=ax.transAxes,
            fontsize=8.5)
    ax.text(0, 1.08, "Gain from same-compound CP · n = 3,691", transform=ax.transAxes, fontsize=8, color=NOTE)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    axes_clean(ax); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save_figure(fig, "figure3_supplement_p0_conditioned")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="plot frozen source_data CSVs")
    args = parser.parse_args()
    if args.plot_only:
        summary = pd.read_csv(DATA / "figure3_summary.csv")
        newer = pd.read_csv(DATA / "figure3_p0_conditioned_report_results.csv")
    else:
        summary, newer = prepare_data()
    plot_main(summary)
    plot_p0_supplement(newer)
    print(json.dumps({"figure": "figure3", "update_version": METHOD_VERSION,
                      "summary_rows": len(summary), "output": str(ROOT)}, indent=2))


if __name__ == "__main__":
    main()
