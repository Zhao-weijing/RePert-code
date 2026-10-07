# Reproduction instructions

## Verified scope

The package supplies frozen numerical results and the current plot/table generators. Full training reruns need processed assay inputs, frozen split/configuration records, trained artifacts and third-party software that are not all included here. Syntax and a synthetic example do not validate the reported scientific results.

## Main figures

Run `python figures/scripts/build_main_figures.py` from the repository root. Individual entry points are `fig2_estimation.py`, `fig3_crossmodal.py`, `fig4_morphology.py` and `fig5_predictor.py`. Outputs are written under `figures/` and copied to `outputs/figures/`. The current manuscript Figure 3 reference PDF includes local editorial changes; confirm visual parity before publication.

Figure 1 is a conceptual schematic, not a computed result. Its original drawing is supplied in `figures/data/figure_1/`; `fig1_schematic.py` requires PyMuPDF and applies text edits. Create `outputs/figures/` before running it.

## Supplementary figures

`python figures/scripts/figure3.py --plot-only` renders the supplementary cross-modal plot from supplied scores. `python figures/scripts/figure4.py --plot-only` renders the sci-Plex3 transfer plot. `python figures/scripts/fig_s3_downstream.py` reuses the original downstream panel function to reconstruct Supplementary Fig. S3 from its frozen contrasts. Arm-level and per-query CSVs are also supplied in `data/source_data/`. Compare outputs with `figures/reference/` when checking the final presentation.

## Tables

Run `python tables/predictor/scripts/build_predictor_table.py`. It rebuilds the predictor source tables and LaTeX table fragments from frozen results; it does not open raw assay targets or train models.

`data/reported_tables/` records all current displayed rows, preserving their LaTeX cell text. These are transcriptions of the displayed values, not recovered unrounded observations. Do not use their rounded values to recalculate confidence intervals. Full available source records remain in the figure and predictor directories.

## Source-data workbooks

Run `python scripts/build_source_data_workbooks.py`. It generates Supplementary Data 1 and 2 under `outputs/source_data/`. The output includes an index of large row-level CSVs that exceed the workbook sheet limit. Full displayed S4-S5 values are also in `data/reported_tables/`; the older benchmark CSV contains only a subset of those methods.

## Upstream analysis map

| Result | Code |
| --- | --- |
| Figure 2a | `analysis/benchmarks/bbbc047/reproducibility_effect/` |
| Figure 2b-c and independent calibration | `analysis/baselines/`, `analysis/repeat_calibration/`, `analysis/reproducibility_estimation/` |
| Figure 2d-e, Tables S4-S5 | `analysis/benchmarks/bbbc047/repeat_estimator/`, `architecture_sensitivity/`, `native_feature_size/`, `cpdistiller_retrieval/` |
| Figure 3 and Table S2 | `analysis/cross_modal/`, `analysis/benchmarks/bbbc047/single_repeat_expression_residual/` |
| Figure 4 and Supplementary Fig. S3 | `analysis/morphology/reca_evaluation/` (current ReCA evaluation); remaining morphology, biological-application and pathway modules provide reused helpers and historical context. |
| Figure 5 | `analysis/predictor_supervision/bbbc047/` |
| Supplementary Fig. S2 and Table S6 | `analysis/external_validation/sci_plex3/`, `analysis/predictor_supervision/sci_plex3/` |

Inspect each runner's `--help`, stage restrictions and required inputs. Do not replace historical frozen settings with current defaults. Generic `/path/to/...` defaults require local overrides. `requirements-analysis.txt` is an import inventory, not a recovered environment lock.

## Release gates

Before claiming complete reproduction: recover and hash the processed inputs and split/configuration records; verify external code versions and terms; run the intended entry points in a clean scientific environment; compare reconstructed statistics with frozen records; inspect the current display assets; approve the licence; verify public access and freeze an archival version.
