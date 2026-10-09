<a id="top"></a>
<div align="center">
<h1>RePert</h1>
<p><strong>Reproducible perturbation responses from independent measurements</strong></p>
<p>
  <a href="README_zh-CN.md"><img src="https://img.shields.io/badge/-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-0B7285?style=flat-square" alt="简体中文"></a>
  <a href="https://zhao-weijing.github.io/RePert-code/"><img src="https://img.shields.io/badge/-Project%20website-0B7285?style=flat-square" alt="Project website"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/-Quick%20start-0B7285?style=flat-square" alt="Quick start"></a>
  <img src="https://img.shields.io/badge/-Cell%20Painting-0B7285?style=flat-square" alt="Cell Painting">
  <img src="https://img.shields.io/badge/-Gene%20expression-0B7285?style=flat-square" alt="Gene expression">
  <img src="https://img.shields.io/badge/-Independent%20biological%20measurements-0B7285?style=flat-square" alt="Independent biological measurements">
</p>
</div>

## 🧬 Abstract

Repeated measurements of the same cellular perturbation can disagree, making it difficult to distinguish genuine biological responses from experimental variation. **RePert** addresses this challenge by combining complementary repeat-aware estimators into calibrated response profiles and using matched Cell Painting (CP) and gene-expression (GE) evidence for conditional updates. Across the evaluated settings, the framework improves recovery of reproducible, compound-specific signals and explores their value for downstream characterization and perturbation prediction. These benefits remain task- and context-dependent.

<p align="center">
  <a href="figures/data/figure_1/figure1_authors.pdf">
    <img src="assets/repert_figure1.png" width="960" alt="RePert Figure 1: independent measurements, calibrated response estimation and conditional cross-modal updates">
  </a>
  <br>
  <sub>Figure 1 · RePert overview. <a href="figures/data/figure_1/figure1_authors.pdf">View original high-resolution PDF ↗</a></sub>
</p>

## ✨ Framework at a glance

<table>
<tr>
<td width="33%" valign="top">
<strong>🔁 Recover reliable responses</strong><br><br>
Learn from independent measurements and combine complementary estimators to suppress unstable signals.
</td>
<td width="33%" valign="top">
<strong>🧩 Integrate complementary evidence</strong><br><br>
Update an existing response estimate with matched CP or GE information that contributes additional signal.
</td>
<td width="33%" valign="top">
<strong>🧠 Support downstream modeling</strong><br><br>
Evaluate whether more reproducible profiles improve morphological analysis and perturbation-response prediction.
</td>
</tr>
</table>

**Main finding.** RePert improves perturbation-specific repeat agreement in the tested settings, with additional gains from correctly matched cross-modal evidence. Improvements in other biological tasks are not guaranteed. [Explore the frozen results ↗](figures/data/)

<a id="quick-start"></a>
## 🚀 Quick start

**New to RePert?** Imagine a compound tested at a fixed dose on several independent acquisition plates. Each plate yields a high-dimensional Cell Painting (CP) or gene-expression (GE) response, but the measurements need not agree perfectly. RePert uses the available *support measurements* to estimate the response that should persist in an **independent held-out measurement**. When a matched second modality is available, it can provide an additional conditional update.

**The workflow in one line:** observed repeats → calibrated response estimate → optional matched CP/GE update → biological analysis or response-predictor supervision.

### 1. Run the example in minutes

The bundled example needs **Python 3.10+ and NumPy**. It generates no downloads and does not require a GPU:

```bash
git clone https://github.com/Zhao-weijing/RePert-code.git
cd RePert-code
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
python -m pip install numpy

python scripts/run_estimator_example.py
python scripts/verify_package.py
```

The synthetic data contain **32 fitting compounds**, **12 separate calibration compounds**, and **5 query compounds**, with 16 artificial features. The script fits an empirical-Bayes estimator using independent simulated plates, computes convex weights on a separate toy calibration set, predicts query profiles, and checks that the saved estimator reloads correctly. All three compound roles are disjoint.

| Output | What to look for |
| :--- | :--- |
| `outputs/example/result.json` | `status: PASS`, role counts, fitted/shrunken calibration weights, and checks on the predictions |
| `outputs/example/predictions.csv` | One estimated 16-feature profile per toy query compound (5 rows) |
| `outputs/example/synthetic_eb_model.npz` | Saved, reloadable empirical-Bayes estimator |
| `outputs/package_verification.json` | Manifest SHA-256 checks, Python syntax checks, and static release-integrity scan |

> [!NOTE]
> **This is a software example, not the full ReCA experiment.** It exercises the copied empirical-Bayes and weight-calibration functions; it does **not** train IMR/LSO, run the complete ensemble, or reproduce manuscript biology. Read [the example specification](examples/README.md) if you want to inspect the synthetic input format.

### 2. Explore the reported results

For readers interested in the scientific evidence before running a model, the repository includes **frozen numerical source data** and scripts that reconstruct the main figures and tables:

```bash
python -m pip install -r requirements-publication.txt
python figures/scripts/build_main_figures.py
python tables/predictor/scripts/build_predictor_table.py
python scripts/build_source_data_workbooks.py
```

Start with the [original Figure 1](figures/data/figure_1/figure1_authors.pdf) for the scientific overview, browse [the reference figures](figures/reference/) alongside [figure source values](figures/data/), or use [the source-data mapping](docs/source_data_map.csv) to find the CSV behind a particular panel. The figure builders read saved results; **redrawing a plot is not rerunning its upstream model**. For supplementary plots and output locations, see [reproduction instructions](docs/reproduction.md).

## 🗂️ Find your way around the code

The repository contains several analysis pipelines, so there is **no single command that trains every RePert experiment**. The quickest way in is to choose what you want to understand:

| If your goal is to… | Start here | What you will find |
| :--- | :--- | :--- |
| **Understand response estimation** | [`analysis/repeat_calibration/`](analysis/repeat_calibration/) | Empirical-Bayes signal/noise estimation, validation calibration, and comparisons with learned repeat estimators |
| **Understand the evaluation** | [`analysis/baselines/`](analysis/baselines/) | Support/held-out construction, condition matching, specificity metrics, and baseline helpers |
| **Study how CP and GE work together** | [`analysis/cross_modal/`](analysis/cross_modal/) | Conditional residual updates, direct-fusion comparisons, and sequential evidence experiments |
| **Study predictor training** | [`analysis/predictor_supervision/`](analysis/predictor_supervision/) | Prediction with Raw, estimated, and decomposed response targets |
| **Inspect external settings** | [`analysis/external_validation/`](analysis/external_validation/) | LINCS Cell Painting and sci-Plex3 analyses |
| **Find a figure or table's numbers** | [`figures/data/`](figures/data/), [`data/source_data/`](data/source_data/), [`data/reported_tables/`](data/reported_tables/) | Saved panel values, numerical sources, and displayed table rows |

**A useful reading order:** [Figure 1](figures/data/figure_1/figure1_authors.pdf) → [synthetic estimator example](examples/README.md) → [repeat estimation](analysis/repeat_calibration/) → [cross-modal evidence](analysis/cross_modal/) → [predictor supervision](analysis/predictor_supervision/). This follows the scientific argument without requiring an immediate upstream training run.

## 🧫 Data and full reproduction

The analyses draw on several different assay settings:

| Dataset | Cells / assay | Why it appears here |
| :--- | :--- | :--- |
| **BBBC047 / Rosetta** | U2OS · CP and L1000 GE | Matched cross-modal observations and repeat-aware estimation |
| **cpg0004-LINCS** | A549 · CP | Additional Cell Painting repeat measurements |
| **sci-Plex3** | A549, K562, MCF7 · GE pseudobulk | Gene-expression evidence and context-transfer analyses |

The public repository provides **analysis scripts, numerical source files, figure/table builders, and the synthetic demo**. It does **not** include every processed assay matrix, frozen split/configuration file, trained artifact, or external dependency needed for a clean end-to-end training run. Begin with [data access](docs/data_access.md) and [reproduction instructions](docs/reproduction.md), preserve compound–dose–acquisition-plate identities, and check each analysis runner's required inputs rather than relying on placeholder paths.

**What has been checked?** The release records local checks for the toy example and reconstruction from frozen results. **Full upstream model-training reproduction remains unverified.** See [release status](metadata/release_status.json) and [the integrity manifest](metadata/FILE_MANIFEST.csv) for the precise scope. Code is distributed under the [MIT licence](LICENSE); original assay datasets and third-party tools retain their own terms. No archival DOI is currently assigned.

---

<p align="center"><sub><a href="#top">↑ Back to top</a> · <a href="https://zhao-weijing.github.io/RePert-code/">Project website ↗</a></sub></p>
