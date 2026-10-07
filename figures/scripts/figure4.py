"""Figure 4: frozen morphology, downstream boundaries and Student transfer.

Read original artifacts, export the exact plotted numbers, and render a 180-mm
figure. No model fitting, tuning, new case selection or outcome-driven filtering.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
DATA = OUT / "data" / "figure_4"
NOTES = OUT / "notes"
MORPH = ROOT / "analysis/morphology"
BIO = ROOT / "analysis/biological_applications"
STUDENT = ROOT / "analysis/external_validation/sci_plex3/student_cross_cell/results/frozen_evaluation"
DIRECT = ROOT / "analysis/external_validation/sci_plex3/cross_cell_validation"
MODULES = ["DNA", "RNA", "ER", "Mito", "AGP", "Shape"]
ALL_MODULES = MODULES + ["Cross-channel"]
DOSES = ["0.04", "0.12", "0.37", "1.11", "3.33", "10"]
COLORS = {"raw": "#85898F", "imr": "#7563A5", "update": "#C47A37", "cfra": "#C47A37", "reference": "#252A31"}
SOURCE_FILES: set[Path] = set()


def read(path: Path) -> pd.DataFrame:
    SOURCE_FILES.add(path)
    return pd.read_csv(path)


def save(frame: pd.DataFrame, stem: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA / f"figure4_{stem}.csv", index=False, float_format="%.12g")


def one(frame: pd.DataFrame, **criteria) -> pd.Series:
    selected = frame
    for key, value in criteria.items():
        selected = selected[selected[key].astype(str).eq(str(value))]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {criteria}, found {len(selected)}")
    return selected.iloc[0]


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_data() -> pd.DataFrame:
    """Mean of within-profile dose distances, never distance of pooled profiles.

    All 258 complete Raw/IMR/update compounds are included. Distances are
    computed separately for each physical support rotation and each module;
    average modules, rotations and seeds inside compound, then compounds.
    This descriptive geometry is not substituted for the frozen MacroTC CI.
    """
    config_path = MORPH / "module_dose_trajectories/CONFIG.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    SOURCE_FILES.add(config_path)
    hash_checks = []
    for entry in config["input_hashes"]:
        path = Path(entry["path"])
        actual = hash_file(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"Frozen morphology input hash changed: {path}")
        SOURCE_FILES.add(path)
        hash_checks.append({"path": str(path), "sha256": actual, "matches_frozen": True})
    helper = BIO / "dose_response/run_dose_response.py"
    SOURCE_FILES.add(helper)
    spec = importlib.util.spec_from_file_location("figure4_frozen_bioc", helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    base = ROOT / "analysis/external_validation/lincs_cpg0004"
    args = SimpleNamespace(
        p0_root=base / "virtual_prior/results/1r_all",
        ge_root=base / "single_repeat_expression_evidence/results/1r_all",
        pair_root=base / "cell_painting_repeat_benchmark/results/1r_all",
    )
    artifact = module.load_artifact(base / "data_preparation/artifact/cp_plate_rows.npz")
    errors = []
    p0, ge, manifests, _ = module.load_pair_and_prediction_bundles(args, {3407: .1, 42: .1, 2025: .1}, {3407: .1, 42: .15, 2025: .15}, errors)
    slots, _, _, audit = module.build_slots(artifact, p0, ge, manifests, errors)
    m1, m2 = module.trim_to_common_sets(slots)
    if errors or len(m1) != 260 or len(m2) != 258:
        raise RuntimeError(f"Frozen slot audit failed: {errors}; common counts {len(m1)}, {len(m2)}")
    mapping = read(MORPH / "feature_annotation_audit/FEATURE_MAPPING.csv")
    mapping = mapping[mapping.biological_endpoint.eq("Eligible")]
    indices = {name: mapping.loc[mapping.biological_module.eq(name), "feature_index"].to_numpy(int) for name in ALL_MODULES}
    if any(len(index) < 8 for index in indices.values()):
        raise RuntimeError("Missing biological module")
    compound_rows = []
    for compound in sorted(m2):
        for method in ("reference", "raw", "teacher", "posterior"):
            distances = []
            for seed in (3407, 42, 2025):
                for rotation in range(2):
                    by_dose = slots[seed][compound][rotation]
                    values = np.vstack([by_dose[d].reference if method == "reference" else by_dose[d].profiles[method] for d in DOSES])
                    for index in indices.values():
                        projected = values[:, index].astype(float)
                        centered = projected - projected.mean(axis=1, keepdims=True)
                        lengths = np.linalg.norm(centered, axis=1)
                        if np.any(lengths <= 0) or not np.isfinite(projected).all():
                            raise RuntimeError(f"Undefined dose distance: {compound}, {method}")
                        corr = (centered @ centered.T) / np.outer(lengths, lengths)
                        distance = 1 - np.clip(corr, -1, 1)
                        np.fill_diagonal(distance, 0)
                        distances.append(distance)
            averaged = np.mean(distances, axis=0)
            for i, dose1 in enumerate(DOSES):
                for j, dose2 in enumerate(DOSES):
                    compound_rows.append({"compound": compound, "method": method, "dose_i_uM": dose1, "dose_j_uM": dose2, "mean_1_minus_pcc": averaged[i, j]})
    compound_frame = pd.DataFrame(compound_rows)
    save(compound_frame, "C_dose_geometry_by_compound")
    matrix = compound_frame.groupby(["method", "dose_i_uM", "dose_j_uM"], sort=False, as_index=False).mean_1_minus_pcc.mean()
    matrix["n_compounds"] = 258
    matrix["aggregation"] = "distance within module/rotation/seed -> equal module/rotation/seed mean -> compound mean"
    save(matrix, "C_dose_geometry")
    # Locked illustration rule, independent of all model outputs: mitochondria
    # is specified in advance, and choose the compound nearest the median
    # range of the held-reference RMS-z curve (ties broken by compound ID).
    mito = indices["Mito"]
    reference_curves = {}
    for compound in sorted(m2):
        repeat_curves = []
        for seed in (3407, 42, 2025):
            for rotation in range(2):
                by_dose = slots[seed][compound][rotation]
                values = np.vstack([by_dose[d].reference for d in DOSES])[:, mito]
                z = (values - artifact.train_center[mito]) / artifact.train_scale[mito]
                repeat_curves.append(np.sqrt(np.mean(z * z, axis=1)))
        reference_curves[compound] = np.mean(repeat_curves, axis=0)
    amplitudes = {compound: float(np.ptp(values)) for compound, values in reference_curves.items()}
    median_amplitude = float(np.median(list(amplitudes.values())))
    selected_compound = min(amplitudes, key=lambda compound: (abs(amplitudes[compound]-median_amplitude), compound))
    selection_rows = [{"compound": compound, "reference_RMS_z_range": value, "distance_to_median": abs(value-median_amplitude), "selected": compound == selected_compound} for compound, value in amplitudes.items()]
    save(pd.DataFrame(selection_rows).sort_values(["distance_to_median", "compound"]), "C_example_selection")
    curve_rows = []
    distance_rows = []
    for method in ("reference", "raw", "teacher", "posterior"):
        for seed in (3407, 42, 2025):
            for rotation in range(2):
                by_dose = slots[seed][selected_compound][rotation]
                values = np.vstack([by_dose[d].reference if method == "reference" else by_dose[d].profiles[method] for d in DOSES])[:, mito]
                z = (values - artifact.train_center[mito]) / artifact.train_scale[mito]
                magnitude = np.sqrt(np.mean(z * z, axis=1))
                for dose, value in zip(DOSES, magnitude):
                    curve_rows.append({"compound": selected_compound, "method": method, "seed": seed, "support_rotation": rotation, "dose_uM": dose, "mitochondrial_RMS_z": value})
                centered = values.astype(float) - values.mean(axis=1, keepdims=True)
                centered /= np.linalg.norm(centered, axis=1, keepdims=True)
                distances = 1 - np.clip(centered @ centered[0], -1, 1)
                distances[0] = 0
                for dose, value in zip(DOSES, distances):
                    distance_rows.append({"compound": selected_compound, "method": method, "seed": seed, "support_rotation": rotation, "dose_uM": dose, "distance_1_minus_PCC_to_lowest": value})
    curves = pd.DataFrame(curve_rows)
    save(curves, "C_example_curve_by_seed_rotation")
    save(curves.groupby(["compound", "method", "dose_uM"], sort=False, as_index=False).mitochondrial_RMS_z.mean(), "C_example_curve")
    distance_frame = pd.DataFrame(distance_rows)
    save(distance_frame, "C_example_distance_by_seed_rotation")
    relative = distance_frame.groupby(["compound", "method", "dose_uM"], sort=False, as_index=False).distance_1_minus_PCC_to_lowest.mean()
    relative["normalization_max"] = relative.groupby("method").distance_1_minus_PCC_to_lowest.transform("max")
    if (relative.normalization_max <= 0).any():
        raise RuntimeError("The selected example has a constant distance curve")
    relative["relative_morphological_distance"] = relative.distance_1_minus_PCC_to_lowest / relative.normalization_max
    save(relative, "C_example_relative_distance")
    (NOTES / "figure4_geometry_audit.json").write_text(json.dumps({"selection_rule": "Mito module specified before extraction; select nearest median reference-only six-dose RMS-z range among all 258 complete compounds, lexicographic compound tie-break; model outputs never used for selection", "selected_compound": selected_compound, "median_reference_range": median_amplitude, "selected_reference_range": amplitudes[selected_compound], "reference_curve_definition": "RMS of (reference-train_center)/train_scale over 17 Mito features; magnitude within rotation/seed before averaging", "modules": ALL_MODULES, "physical_support_slots": 2, "reference": "four physical rows excluding current support", "hash_checks": hash_checks, "slot_audit": audit, "mapping_errors": errors, "purpose": "Descriptive illustration only; formal inference uses saved compound-paired MacroTC differences"}, indent=2), encoding="utf-8")
    return matrix


def prepare() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    a = read(MORPH / "module_recovery/PAIRED_CONTRASTS.csv")
    a = a[a.endpoint.eq("excess_fisher_z") & ((a.analysis.eq("module") & a.group.isin(ALL_MODULES)) | a.analysis.eq("macro_module"))].copy()
    save(a, "A_module_effects")
    b = read(MORPH / "dominant_feature_recovery/PAIRED_CONTRASTS.csv")
    b = b[b.k.eq(25) & b.endpoint.isin(["overlap_at_k", "sign_consistency_at_k"])].copy()
    save(b, "B_dominant_features")
    c = read(MORPH / "module_dose_trajectories/PAIRED_CONTRASTS.csv")
    save(c, "C_trajectory_effects")
    geometry_data()
    h = read(BIO / "hit_recovery/BOOTSTRAP_CI.csv")
    r = read(BIO / "mechanism_target_retrieval/BOOTSTRAP_CI.csv")
    p = read(ROOT / "analysis/pathway_validation/curated_retrieval/BOOTSTRAP_CI.csv")
    records = []
    hit = one(h, seed="mean", rotation="all", analysis_set="common_all_methods", left_method="teacher", right_method="raw", metric="AP")
    records.append({"task": "Hit recovery", "metric": "AP", "comparison": "IMR - Raw", "point": hit.point_difference, "ci_low": hit.ci_low, "ci_high": hit.ci_high, "n_compounds": hit.n_compounds, "interpretation": "Uncertain gain"})
    for task, label in (("moa", "MoA retrieval"), ("target", "Target retrieval")):
        row = one(r, seed="mean_seeds", task=task, variant="base", contrast="teacher_minus_raw", metric="map")
        records.append({"task": label, "metric": "mAP", "comparison": "IMR - Raw", "point": row.point, "ci_low": row.ci_low, "ci_high": row.ci_high, "n_compounds": row.compound_count, "interpretation": "No clear gain"})
    row = one(p, variant="base", k=25, contrast="teacher_minus_raw", metric="pathway_map")
    records.append({"task": "Pathway retrieval", "metric": "mAP", "comparison": "IMR - Raw", "point": row.point, "ci_low": row.ci_low, "ci_high": row.ci_high, "n_compounds": row.compound_count, "interpretation": "No clear gain"})
    save(pd.DataFrame(records), "D_downstream_boundaries")
    for name in ("target_summary", "target_macro_contrasts", "drug_metrics"):
        save(read(STUDENT / "test" / f"{name}.csv"), f"E_{name}")
    save(read(DIRECT / "LOCO_POOL_target_macro_summary.csv"), "E_direct_estimator_transfer_boundary")
    save(read(DIRECT / "LOCO_POOL_source_target_summary.csv"), "E_direct_estimator_transfer_by_target")
    source_manifest = [{"path": str(path), "sha256": hash_file(path), "bytes": path.stat().st_size} for path in sorted(SOURCE_FILES)]
    save(pd.DataFrame(source_manifest), "source_manifest")
    caption = CAPTION.replace("Blue denotes", "Purple denotes")
    start = caption.index("**c**,")
    end = caption.index(" **d**,", start)
    caption = caption[:start] + """**c**, A descriptive six-dose mitochondrial response curve and population-level trajectory recovery. The illustrative compound is selected without reference to model performance: the Mito module is specified in advance, the RMS-z amplitude range of the four-repeat independent reference is calculated for every one of the 258 complete common compounds, and the compound nearest the median range is selected (lexicographic compound-ID tie-break). RMS z is the root mean square of the 17 mitochondrial features after centring/scaling with the frozen training statistics. Curves average the separately computed magnitudes across two support rotations and three seeds; no error bars or inference are assigned to this single example. The reference averages four physical CP rows excluding the current support. The right-hand summary shows the formal seven-module MacroTC endpoint, the mean of module-specific Spearman correlations between the 15 predicted and reference dose distances: IMR minus Raw, +0.0521 [0.0384, 0.0659] (260 compounds); update minus IMR, +0.0115 [0.0078, 0.0153] (258 compounds). This seven-module macro includes the cross-channel module, which is not among the six structural modules shown in a. Doses are in micromolar.""" + caption[end:]
    selection = json.loads((NOTES / "figure4_geometry_audit.json").read_text(encoding="utf-8"))
    caption += f"\nIllustrative compound: `{selection['selected_compound']}`; selected reference RMS-z range {selection['selected_reference_range']:.6f}, cohort median {selection['median_reference_range']:.6f}. This post hoc visualization rule was specified before curve extraction and is not a preregistered efficacy endpoint.\n"
    caption += "\nDisplay terminology: Observed denotes the saved Raw profile; Update denotes the full frozen cross-modal update; specificity denotes Same-Foreign excess Fisher-z. In e, calibrated and observed-response targets denote CFRA and Raw training labels for otherwise matched Students.\n"
    caption = f"""# Figure 4. Morphological organization and downstream transfer

**a**, Changes in module agreement for six cpg0004-LINCS structural feature groups: DNA, RNA, endoplasmic reticulum (ER), mitochondria (Mito), actin/Golgi/plasma membrane (AGP), and shape (110 compounds). The historical fixed-null correction cancels in each paired difference; these contrasts measure Fisher-z agreement, not prediction-specific foreign discrimination. Purple circles denote IMR minus Observed; orange diamonds denote the frozen cross-modal update minus IMR. Observed denotes a single measured profile; the historical teacher is IMR, not CFRA. **b**, Reference-defined top-25 feature overlap and direction agreement (262 compounds). **c**, Dose-dependent mitochondrial organization for {selection['selected_compound']}, selected solely by proximity to the median reference RMS-z range among 258 shared compounds. For each method, distances (1−PCC across 17 mitochondrial features) from the lowest dose are computed within seed and support rotation, averaged, and divided by that method's maximum. Normalization preserves within-method distance ranks but removes absolute magnitude. This descriptive example is not selected for model improvement. At right, population-level dose-response concordance uses all 15 dose-pair distances and seven equally weighted feature groups, including cross-channel features (260 IMR–Observed and 258 update–IMR compounds). References average four physical rows excluding the current support; only the two frozen support rotations are used. **d**, Endpoint-specific changes in confirmed activity AP and mechanism, target, and curated-pathway mAP; none establishes a consistent gain over Observed. **e**, sci-Plex3 held-out-cell-line/plate-block transfer of matched predictors trained on CFRA-calibrated versus observed-response targets (183 previously seen compounds per target). Inputs include chemical structure, dose, and target Vehicle baseline. All models were frozen before target reads. The unseen-compound confirmation remains inconclusive because the MCF7 interval crosses zero; direct CFRA-estimator transfer was negative (Supplementary Figure). Cell-line and plate-block effects cannot be separated. Points and bars show paired effects and 95% percentile intervals from 10,000 compound/drug bootstrap resamples; the mean in e resamples targets then drugs.\n"""
    caption = caption.replace("**d**, Endpoint-specific changes in confirmed activity AP and mechanism, target, and curated-pathway mAP; none establishes a consistent gain over Observed.", "**d**, Activity AP and mechanism, target, and curated Reactome target-neighbour mAP (262/140/146/184 compounds, respectively); none establishes a consistent gain. Pathway retrieval is not RNA pathway-expression recovery.")
    (NOTES / "figure4_caption.md").write_text(caption, encoding="utf-8")
    boundaries = BOUNDARIES.replace("- C includes every one of 258 complete common compounds. No favorable compound, dose or module is selected. The heatmap is descriptive average geometry; the caption separately gives the already-frozen MacroTC paired inference on 260/258 compounds. Distances are calculated before averaging, and support never enters its four-row reference.", "- C selects a descriptive Mito curve by a pre-extraction reference-only median-range rule among all 258 complete common compounds; no model output enters selection. It is not a pre-registered efficacy endpoint. Population inference uses the already-frozen MacroTC results on 260/258 compounds. The supplementary source tables retain dose-distance matrices for all compounds. Support never enters its four-row reference.")
    boundaries += "\n- The main example plots 1−PCC to the lowest dose across 17 Mito features, averaged across seeds/rotations before normalization by each method's maximum. This retains within-method dose-distance ranks and suppresses amplitude; RMS-z magnitude curves are retained in source_data. The compound selected before viewing model distances is unchanged.\n- Morph-A uses a shared method-independent raw-support/held foreign correction. It cancels in paired method contrasts, so panel a displays change in module agreement (Fisher z), not prediction-specific perturbation specificity. The saved finite common set of 110 compounds is preserved.\n"
    (NOTES / "figure4_boundaries.md").write_text(boundaries, encoding="utf-8")
    (NOTES / "figure4_supp_transfer_caption.md").write_text("""# Supplementary Figure. Boundaries of cross-environment transfer

**a**, Unseen-compound Student evaluation, with 36 confirmation compounds and 115 drug-dose units per target. Calibrated-target minus observed-target Students have a positive target-mean specificity difference, but MCF7 has a confidence interval crossing zero; the prespecified all-target confirmation criterion is therefore not met. **b**, A separate direct-estimator transfer experiment, in which the source-pooled CFRA estimator acts on a target support profile rather than supplying training labels to a Student. All three target folds and their equally weighted mean favor the observed support profile. For both panels, specificity is Same-Foreign excess Fisher-z, error bars are 95% paired percentile bootstrap intervals (10,000 resamples), target intervals resample drugs, and mean intervals resample targets then drugs. Both tests confound cell-line and plate-block shift. They answer different questions and their effect sizes are not pooled.\n""", encoding="utf-8")


CAPTION = """# Figure 4. Recovery of morphological organization and transfer to a downstream prediction task

**a**, Paired improvement in module-specific Same-Foreign excess Fisher-z in cpg0004-LINCS. Six structural modules are displayed: DNA, RNA, endoplasmic reticulum (ER), mitochondria (Mito), actin/Golgi/plasma membrane (AGP) and shape. The historical frozen teacher is labelled IMR; it is not CFRA. Blue denotes IMR minus Raw; orange denotes the frozen cross-modal update minus IMR. Points and bars show the three-seed mean paired effect and its 95% compound-bootstrap interval (110 compounds). **b**, Recovery of the top 25 reference features and their signs (262 compounds). Overlap and sign consistency are separate endpoints. **c**, Six-dose morphology geometry, shown as the mean pairwise 1-PCC distance within the frozen module coordinates. All 258 compounds available for all methods are used; distances are first calculated within each module, seed and physical support rotation, then equally averaged within compound and across compounds. The reference averages four physical CP rows excluding the current support. These descriptive heatmaps are not estimates of trajectory concordance. The formal seven-module MacroTC endpoint is the mean of module-specific Spearman correlations between the 15 predicted and reference dose distances: IMR minus Raw, +0.0521 [0.0384, 0.0659] (260 compounds); update minus IMR, +0.0115 [0.0078, 0.0153] (258 compounds). Seven-module macro results include the cross-channel module, which is not one of the six structural modules shown in a. Doses are in micromolar. **d**, Endpoint-specific downstream results for IMR minus Raw. Values are absolute paired changes with 95% intervals; AP and mAP values are not pooled. Hit recovery includes 262 compounds but only seven confirmed-active compounds. No clear MoA, target or curated-pathway retrieval gain is established. **e**, sci-Plex3 leave-one-cell-line/plate-block-out transfer of otherwise matched Students trained with Raw or source-only drug-OOF CFRA labels. Shown are the already-seen-drug setting (183 drugs and 613 drug-dose units per target) and its equally weighted target macro result. Inputs are chemical structure, dose and the target query Vehicle baseline. Each target remains excluded from fitting, feature construction and model selection; target reads occur only after all 36 checkpoints are frozen. Error bars bootstrap drugs within target, and target cell lines followed by drugs for the macro. The unseen-drug confirmation result is positive overall (+0.00479 [0.00167, 0.00726]) but does not meet the all-target confirmation rule because the MCF7 interval crosses zero. Direct transfer of the CFRA estimator, a different experiment, was negative overall (−0.01087 [−0.01896, −0.00499]); the Student result therefore supports a bounded training-label application, not universal cross-cell-line transfer. All intervals are paired 95% percentile bootstrap intervals from 10,000 resamples. Morphology panels retain the two support slots available in the historical frozen output; they are not a new rotation-first confirmation.
"""

BOUNDARIES = """# Figure 4 data and interpretation notes

- A-D use historical cpg0004 morphology/application artifacts, with M0=Raw, M1=frozen Teacher (displayed as IMR), M2=frozen GE-updated estimate. They are not CFRA measurements and must not be cited as direct CFRA confirmation.
- The original M2 reconstruction contains the locked cpg0004 prior/P0 component and the saved GE residual. The displayed M2−IMR contrast is the effect of the full frozen update, not an isolated causal GE-only ablation.
- Morph-A formal Same-Foreign excludes unscorable entries: 110 of 262 common compounds. The six structural panels do not include cross-channel, whereas the saved macro is an equal mean of seven modules. The macro CI must not be attributed to only six modules.
- C includes every one of 258 complete common compounds. No favorable compound, dose or module is selected. The heatmap is descriptive average geometry; the caption separately gives the already-frozen MacroTC paired inference on 260/258 compounds. Distances are calculated before averaging, and support never enters its four-row reference.
- D uses distinct endpoint units and deliberately presents a numerical evidence table rather than one shared axis for AP and mAP. CI overlap with zero does not prove equivalence. Pathway retrieval is the fixed Reactome target-neighbour proxy (k=25), not direct RNA pathway validation. Missing mapping prevents the latter; it is not shown as a zero effect.
- E tests Student training-label replacement. The seen-drug setting meets the frozen gate. Unseen-drug confirmation remains inconclusive because MCF7 has CI crossing zero. Cell-line and plate-block effects are confounded; target Vehicle baseline is a permitted inference input. This is not target-free zero-shot prediction.
- E measures perturbation specificity, not uniform expression fidelity. MCF7 same-PCC decreases despite higher Same-Foreign. Absolute effects are small and Raw is near zero; percentage improvements are inappropriate. The distinct direct-CFRA-transfer experiment was negative and is retained in source_data.
- Fitting, tuning, dataset construction and new experiments were not performed. Every rendered numerical mark comes from frozen outputs or a documented descriptive extraction of those outputs.
"""


def plot() -> None:
    import figure_style as style
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = style.new_figure(height_mm=200, width_mm=180)
    def panel(letter, title, x, y):
        fig.text(x, y, letter, fontsize=10, fontweight="bold", va="bottom")
        fig.text(x + .030, y + .001, title, fontsize=8.5, va="bottom")
    def clean(ax, zero=True):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2.5, pad=2)
        if zero:
            ax.axvline(0, color="#B7BDC3", lw=.6, zorder=0)
        ax.set_axisbelow(True)
    panel("a", "Morphological modules", .025, .945)
    a = pd.read_csv(DATA / "figure4_A_module_effects.csv")
    ax = fig.add_axes([.112, .718, .365, .205])
    for j, (comp, color, offset) in enumerate((("teacher_minus_raw", COLORS["imr"], -.12), ("posterior_ge_minus_teacher", COLORS["update"], .12))):
        for i, name in enumerate(MODULES):
            row = one(a, comparison=comp, analysis="module", group=name)
            ax.errorbar(row.point, 5-i + offset, xerr=[[row.point-row.ci_low], [row.ci_high-row.point]], fmt="o" if j == 0 else "D", ms=3.1, color=color, elinewidth=.9, capsize=1.5)
    ax.set_yticks(range(6), MODULES[::-1]); ax.set_ylim(-.6, 5.6); ax.set_xlim(-.02, .30)
    ax.set_xlabel("Change in module agreement (Fisher z)")
    clean(ax)
    ax.text(1, 1.025, "n = 110", transform=ax.transAxes, ha="right", fontsize=8, color="#59616B")
    handles = [Line2D([0], [0], marker="o", lw=0, color=COLORS["imr"], ms=3, label="IMR − Observed"), Line2D([0], [0], marker="D", lw=0, color=COLORS["update"], ms=3, label="Update − IMR")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.085, .679), ncol=2, frameon=False, fontsize=8, handletextpad=.3, columnspacing=1.0)
    panel("b", "Dominant morphological features", .535, .945)
    b = pd.read_csv(DATA / "figure4_B_dominant_features.csv")
    for x, endpoint, title in ((.576, "overlap_at_k", "Top-25 overlap"), (.800, "sign_consistency_at_k", "Direction agreement")):
        ax = fig.add_axes([x, .747, .153, .176])
        for i, comp in enumerate(("teacher_minus_raw", "posterior_ge_minus_teacher")):
            row = one(b, comparison=comp, endpoint=endpoint)
            color = COLORS["imr"] if i == 0 else COLORS["update"]
            ax.errorbar(i, row.point, yerr=[[row.point-row.ci_low], [row.ci_high-row.point]], fmt="o" if i == 0 else "D", color=color, ms=4, capsize=2, elinewidth=1)
        ax.axhline(0, color="#B7BDC3", lw=.6); ax.set_xlim(-.55, 1.55); ax.set_ylim(-.004, .092)
        ax.set_xticks([0, 1], ["IMR\n−\nObserved", "Update\n−\nIMR"]); ax.set_title(title, fontsize=8, pad=5)
        ax.set_ylabel("Paired change" if x < .7 else "")
        clean(ax, zero=False)
    fig.text(.762, .666, "Reference-defined features; n = 262", ha="center", fontsize=8, color="#59616B")
    panel("c", "Dose-dependent morphology", .025, .615)
    curves = pd.read_csv(DATA / "figure4_C_example_relative_distance.csv")
    ax = fig.add_axes([.115, .401, .380, .147])
    methods = (("reference", "Reference", "reference", "s"), ("raw", "Observed", "raw", "o"), ("teacher", "IMR", "imr", "o"), ("posterior", "Update", "update", "D"))
    for method, label, color, marker in methods:
        selected = curves[curves.method.eq(method)].sort_values("dose_uM")
        ax.plot(selected.dose_uM, selected.relative_morphological_distance, color=COLORS[color], marker=marker, ms=3.3, lw=1.0, label=label, ls="--" if method == "reference" else "-")
    ax.set_xscale("log"); ax.set_xticks([float(d) for d in DOSES], DOSES)
    ax.tick_params(axis="x", labelrotation=35)
    ax.set_xlabel("Dose (µM)"); ax.set_ylabel("Relative morphological distance")
    clean(ax, zero=False)
    fig.text(.115, .589, str(curves.compound.iloc[0]) + "  ·  Mitochondria", fontsize=8, va="bottom")
    curve_handles, curve_labels = ax.get_legend_handles_labels()
    fig.legend(curve_handles, curve_labels, loc="upper left", bbox_to_anchor=(.115, .582), fontsize=8, ncol=4, handlelength=1.2, columnspacing=.8, handletextpad=.4)
    c = pd.read_csv(DATA / "figure4_C_trajectory_effects.csv")
    ax = fig.add_axes([.724, .417, .229, .139])
    for i, (comp, label, color) in enumerate((("teacher_minus_raw", "IMR − Raw", "imr"), ("posterior_minus_teacher", "Update − IMR", "update"))):
        row = one(c, analysis="macro_module", comparison=comp)
        ax.errorbar(row.point, 1-i, xerr=[[row.point-row.ci_low], [row.ci_high-row.point]], fmt="o", color=COLORS[color], ms=4, capsize=2)
    ax.set_yticks([1, 0], ["IMR − Observed", "Update − IMR"]); ax.set_ylim(-.55, 1.55); ax.set_xlim(0, .075)
    ax.set_xticks([0, .025, .05, .075]); ax.set_xlabel("Change in concordance")
    ax.set_title("All compounds", fontsize=8, pad=9)
    clean(ax)
    fig.text(.674, .350, "Seven modules; n = 260 / 258", fontsize=8, color="#59616B", va="top")
    panel("d", "Downstream tasks", .025, .295)
    d = pd.read_csv(DATA / "figure4_D_downstream_boundaries.csv")
    ax = fig.add_axes([.033, .090, .476, .184]); ax.set_axis_off()
    cells = [[row.task.replace(" retrieval", "").replace("Pathway", "Pathways").replace("MoA", "Mechanism") + f" ({row.metric})", f"{row.point:+.4f}\n[{row.ci_low:+.4f}, {row.ci_high:+.4f}]"] for row in d.itertuples()]
    table = ax.table(cellText=cells, colLabels=["Endpoint", "IMR − Observed [95% CI]"], colWidths=[.39, .61], cellLoc="left", colLoc="left", loc="upper left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(8.0)
    for (r, c), cell in table.get_celld().items():
        cell.visible_edges=""; cell.PAD=.05; cell.set_facecolor("white")
        if r == 0: cell.set_text_props(weight="bold")
    for y in (1, .8, 0):
        ax.plot([0, 1], [y, y], transform=ax.transAxes, color="#59616B", lw=.65, clip_on=False)
    panel("e", "Held-out cell lines", .565, .295)
    es = pd.read_csv(DATA / "figure4_E_target_summary.csv")
    em = pd.read_csv(DATA / "figure4_E_target_macro_contrasts.csv")
    ax = fig.add_axes([.660, .112, .292, .152])
    for i, target in enumerate(("A549", "K562", "MCF7", "Macro")):
        row = one(em, endpoint="pure_seen") if target == "Macro" else one(es, endpoint="pure_seen", target_cell_line=target, arm="cfra_minus_raw")
        ax.errorbar(row.estimate, 3-i, xerr=[[row.estimate-row.ci_low], [row.ci_high-row.estimate]], fmt="D" if target == "Macro" else "o", color=COLORS["cfra"], ms=3.8, elinewidth=1, capsize=2)
    ax.set_yticks(range(4), ["Mean", "MCF7", "K562", "A549"]); ax.set_ylim(-.5, 3.5); ax.set_xlim(-.0007, .012)
    ax.set_xticks([0, .004, .008, .012]); ax.set_xlabel("Calibrated − observed targets\nChange in specificity (Fisher z)", fontsize=8)
    clean(ax)
    ax.set_title("Seen compounds; n = 183 / target", fontsize=8, pad=6)
    fig.text(.572, .035, "Unseen-compound confirmation: inconclusive", fontsize=8, color="#59616B", va="top")
    OUT.mkdir(parents=True, exist_ok=True)
    style.save_figure(fig, "figure4")
    plt.close(fig)
    # Retain the predeclared unresolved and negative transfer results visibly.
    fig = style.new_figure(height_mm=100, width_mm=180)
    direct_s = pd.read_csv(DATA / "figure4_E_direct_estimator_transfer_by_target.csv")
    direct_m = pd.read_csv(DATA / "figure4_E_direct_estimator_transfer_boundary.csv")
    for j, (title, source, macro_source) in enumerate((("Unseen compounds", es, em), ("Direct estimator transfer", direct_s, direct_m))):
        ax = fig.add_axes([.135 + j * .48, .28, .32, .54])
        style.panel_label(ax, "a" if j == 0 else "b", x=-.26, y=1.10)
        style.panel_heading(ax, title)
        for i, target in enumerate(("A549", "K562", "MCF7", "Mean")):
            if j == 0:
                row = one(macro_source, endpoint="cold_confirmation") if target == "Mean" else one(source, endpoint="cold_confirmation", target_cell_line=target, arm="cfra_minus_raw")
            else:
                row = one(macro_source[macro_source.dose_nM.isna()], scope="ALL_TARGETS_MACRO", method="CFRA-Raw") if target == "Mean" else one(source, target_cell_line=target, method="CFRA-Raw")
            style.errorbar(ax, [row.estimate], [3-i], [row.ci_low], [row.ci_high], color="cfra", marker="D" if target == "Mean" else "o")
        ax.set_yticks(range(4), ["Mean", "MCF7", "K562", "A549"]); ax.set_ylim(-.55, 3.55)
        ax.set_xlim((-.0045, .0115) if j == 0 else (-.034, .004))
        style.clean_axis(ax, horizontal=True)
        ax.set_xlabel("Calibrated − observed targets\nChange in specificity (Fisher z)" if j == 0 else "ReCA − Observed\nChange in specificity (Fisher z)")
        ax.text(.5, -.33, "Confirmation remains inconclusive" if j == 0 else "Lower specificity after direct transfer", transform=ax.transAxes, ha="center", fontsize=8)
    style.save_figure(fig, "figure4_supp_transfer")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only: prepare()
    if not args.prepare_only: plot()
    print(json.dumps({"figure": 4, "output": str(OUT), "prepare_only": args.prepare_only, "plot_only": args.plot_only}))
