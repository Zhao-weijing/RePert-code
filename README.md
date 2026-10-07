# RePert

RePert constructs reproducible estimates of cellular perturbation responses from independent measurements and uses matched cross-modal evidence for conditional updates.

## Contents

| Directory | Contents |
| --- | --- |
| `analysis/` | Copied estimation, calibration, benchmark, cross-modal, morphology and predictor workflows. |
| `figures/scripts/` | Current main-figure and supplementary-figure rendering code. |
| `figures/data/` | Frozen figure summaries and available compound-, condition- and seed-level results. |
| `data/source_data/` | Current Figure 4, Figure 5 and supplementary numerical sources. |
| `data/reported_tables/` | Displayed values of Table 1 and Supplementary Tables S1-S6, including rows absent from older source summaries. |
| `tables/predictor/` | Frozen predictor inputs, source CSVs and table-generation code. |
| `examples/` | Small synthetic estimator example and expected structural output. |
| `scripts/` | Example, source-data workbook and package-integrity commands. |
| `docs/` | Reproduction instructions, source-data mapping and data access. |
| `metadata/` | File checksums and release status. |

## Run the small example

Python 3.10 or later and NumPy are required. Run from this directory:

```sh
python scripts/run_estimator_example.py
python scripts/verify_package.py
```

The example calls the copied empirical-Bayes estimator and convex calibration functions on synthetic inputs. It writes predictions and calibration weights under `outputs/example/`. It is a software demonstration, not an experimental result.

## Reconstruct figures and source-data workbooks

```sh
python -m pip install -r requirements-publication.txt
python figures/scripts/build_main_figures.py
python tables/predictor/scripts/build_predictor_table.py
python scripts/build_source_data_workbooks.py
```

See `docs/reproduction.md` for supplementary plots and upstream requirements. Reconstruction from frozen results and upstream model training are different workflows. Main-figure redraws do not fit models.

## Data and versions

`docs/source_data_map.csv` maps each display item to supplied files. Public assay data access is documented in `docs/data_access.md`. Preserve compound, dose and acquisition-plate identity when preparing inputs. Training seeds are model realizations, not independent biological measurements.

## Release status and licence

The publication code package is hosted at https://github.com/Zhao-weijing/RePert-publication. RePert code is distributed under the MIT licence; see `LICENSE`. Third-party software and assay data retain their own terms, as described in `docs/third_party_software.md` and `docs/data_access.md`.

The numerical reconstruction and synthetic example have been checked locally. Full upstream training reproduction remains unverified. An archival DOI has not yet been assigned. See `metadata/release_status.json` for the verification scope.
