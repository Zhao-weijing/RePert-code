<a id="top"></a>

<div align="center">

# RePert
### Reproducible Perturbation Responses from Independent Measurements

<p>
  <a href="https://zhao-weijing.github.io/RePert-code/"><img src="https://img.shields.io/badge/🌐_Project-Website-073947?style=for-the-badge" alt="Project website"></a>
  <a href="figures/data/figure_1/figure1_authors.pdf"><img src="https://img.shields.io/badge/🧬_Framework-Figure_1-3A6D9A?style=for-the-badge" alt="Authors' Figure 1 PDF"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-516B74?style=for-the-badge" alt="MIT licence"></a>
</p>
<p>
  <img src="https://img.shields.io/badge/Assay-Cell%20Painting-0C8F91?logo=microscope&logoColor=white" alt="Cell Painting">
  <img src="https://img.shields.io/badge/Modality-Gene%20Expression-7753B6?logo=dna&logoColor=white" alt="Gene expression">
  <img src="https://img.shields.io/badge/Principle-Independent%20Repeats-2A697A?logo=checkmarx&logoColor=white" alt="Independent biological measurements">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10 or later">
</p>

**Measure what repeats. Update with what complements.**

**English** · [简体中文](README_zh-CN.md)

[🌐 Project website](https://zhao-weijing.github.io/RePert-code/) · [🧭 Framework](#-framework) · [📊 Results](#-selected-findings) · [🚀 Quick start](#-quick-start) · [🧪 Reproduction](#-reproduction--scope) · [📁 Source data](docs/source_data_map.csv)

</div>

## 🧬 Abstract

**RePert** investigates how to construct reliable cellular perturbation-response estimates when experimental repeats are limited and measurements contain substantial variation. Independent acquisition-plate measurements are used to distinguish repeatable, compound-specific responses from background agreement. A **repeat-aware estimation framework (ReCA)** combines independent-measurement regression, leave-support-out supervision, and empirical-Bayes shrinkage through validation-calibrated fusion. Matched cross-modal experiments then ask when Cell Painting (CP) and gene expression (GE) provide *additional evidence* beyond an existing response estimate. Finally, supervised prediction experiments test whether these estimates improve predictions for held-out perturbations. The results support more reproducible response estimates in several settings, while also identifying boundaries: improvements in repeat agreement do **not** imply uniform gains in downstream biological retrieval or cross-cell-line transfer.

<p align="center">
  <a href="figures/data/figure_1/figure1_authors.pdf">
    <img alt="RePert conceptual roadmap: replicate-aware estimation, cross-modal conditional evidence and downstream prediction" src="assets/repert_overview.svg" width="980">
  </a>
  <br>
  <sub>Conceptual web overview (not a reproduction of the authors' illustration). <a href="figures/data/figure_1/figure1_authors.pdf"><b>Open the original authors' Figure 1 (PDF) ↗</b></a> · <a href="https://zhao-weijing.github.io/RePert-code/#figure1">View Figure 1 on the project website ↗</a></sub>
</p>

<a id="-framework"></a>
## ✨ Framework at a glance

<table>
<tr>
<td width="33%" valign="top">
<strong>🔁 Repeat-aware estimation</strong><br><br>
Infer a reliable perturbation profile from one or a few independent acquisition-plate measurements. Separate the same-compound signal from matched foreign/background structure.
</td>
<td width="33%" valign="top">
<strong>🧩 Conditional cross-modal updating</strong><br><br>
Start with an estimate in the target modality. Add CP or GE residual evidence only where it contributes compound-matched information beyond what is already known.
</td>
<td width="33%" valign="top">
<strong>🧠 Prediction with improved targets</strong><br><br>
Study whether repeat-aware response targets and residual decomposition improve unseen-compound response prediction, rather than assuming better measurements ensure better generalization.
</td>
</tr>
</table>

### 🔬 How ReCA estimates responses

ReCA combines three complementary routes:

| Component | Key idea | Role |
| :--- | :--- | :--- |
| **IMR** | Predict an independent held-out repeat from observed support measurements | Learned repeat recovery |
| **LSO** | Learn from averages of plates excluded from the support set | Lower-noise supervision |
| **IMCEB** | Shrink response directions according to repeat-derived signal and noise covariance | Statistical regularization |
| **ReCA / CFRA** | Fit nonnegative ensemble weights on independent validation targets | Calibrated estimator fusion |

The empirical-Bayes component uses an explicit replicate-budget-dependent shrinkage rule. The final fusion weights are nonnegative and sum to one; they are fitted on validation data, not selected using test measurements.

**Interpretation:** ReCA estimates an already *observed* intervention's response. It is not itself an inverse intervention-design model.

<a id="-selected-findings"></a>
## 📊 Selected findings

**Figure 2 · BBBC047 repeat-aware specificity.** The frozen `E` metric is same-compound minus matched-foreign agreement on the Fisher-z scale; higher is better. Results below are *absolute endpoint values*, not gains.

| Independent support measurements | Raw support `E` | ReCA `E` | Raw mAP@33 | ReCA mAP@33 |
| :---: | ---: | ---: | ---: | ---: |
| **1** | 0.1895 | **0.2212** | 0.4733 | **0.4845** |
| **2** | 0.2294 | **0.2542** | 0.4974 | **0.5113** |
| **3** | 0.2692 | **0.2821** | 0.5174 | **0.5252** |

Source: [Figure 2 frozen values](figures/data/figure_2/) (specificity and retrieval rows are separate endpoints and condition subsets).

**Figure 3 · Cross-modal evidence is conditional.** In the fixed held-out-CP protocol, adding observed GE evidence to one CP measurement improved repeat agreement by **+0.0194 Fisher-z** (95% CI [0.0173, 0.0216]); with two CP measurements already available, GE still contributed **+0.0149** [0.0129, 0.0168]. These are conditional *measurement* gains, not proof that GE replaces a CP repeat.

**Figures 4–5 · Benefits have limits.** Morphology-module and dose-trajectory agreement improved in reported comparisons, and some supervised CP prediction metrics improved modestly. However, hit-recovery confidence intervals overlap zero, MoA/target/pathway retrieval shows no clear general improvement, and direct estimator transfer across cell lines can underperform Raw. See [downstream boundaries](figures/data/figure_4/figure4_D_downstream_boundaries.csv), [cross-cell transfer](figures/data/figure_4/figure4_E_direct_estimator_transfer_boundary.csv), and [predictor results](data/source_data/predictor_main_source.csv).

> [!IMPORTANT]
> **Better repeat agreement ≠ universally better biology or experimental decisions.** RePert reports task-specific boundaries and retains independent held-out measurements as the evaluation reference.

## 🧫 Datasets and modalities

| Dataset | Cellular context | Assay | Published representation in this package |
| :--- | :--- | :--- | :--- |
| **BBBC047 / Rosetta** | U2OS | Cell Painting + L1000 | 775 CP + 977 GE features |
| **cpg0004-LINCS** | A549 | Cell Painting | 242 CP features |
| **sci-Plex3** | A549, K562, MCF7 | RNA-seq pseudobulk | 2,000 GE features |

The release contains numerical source results, analysis code, and a toy example—not the full upstream processed assay inputs. See [data access & provenance](docs/data_access.md). Preserve **compound, nominal dose, acquisition plate, and train/validation/test role** when preparing data. Training seeds are model realizations, not independent biological repeats.

<a id="-quick-start"></a>
## 🚀 Quick start

**Python 3.10+ · NumPy · synthetic input only.** This quick example exercises the copied empirical-Bayes estimator and convex fusion weights, without downloading biological assays or training an experimental model.

```bash
git clone https://github.com/Zhao-weijing/RePert-code.git
cd RePert-code
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install numpy
python scripts/run_estimator_example.py
python scripts/verify_package.py
```

The example writes `outputs/example/predictions.csv`, fusion weights, and a reloadable toy estimator. See [examples/README.md](examples/README.md) for the synthetic input and expected output contract.

<a id="-reproduction--scope"></a>
## 🧪 Reproduction & scope

There are **two distinct workflows**, and this repository deliberately does not conflate them.

| Goal | Entry point | Data requirement |
| :--- | :--- | :--- |
| ✅ Run toy estimator & verify release manifest | `scripts/run_estimator_example.py` · `scripts/verify_package.py` | Bundled synthetic data only |
| 📈 Rebuild figure and table outputs | `figures/scripts/build_main_figures.py` · `tables/predictor/scripts/build_predictor_table.py` · `scripts/build_source_data_workbooks.py` | Bundled frozen numerical outputs |
| 🧬 Fully rerun upstream experiments or models | `analysis/` | Additional processed assays, frozen splits, checkpoints, and versioned dependencies |

For numerical reconstruction:

```bash
python -m pip install -r requirements-publication.txt
python figures/scripts/build_main_figures.py
python tables/predictor/scripts/build_predictor_table.py
python scripts/build_source_data_workbooks.py
```

**Original Figure 1:** [authors' PDF](figures/data/figure_1/figure1_authors.pdf). For a GitHub-friendly PNG preview, run `python scripts/render_public_figure.py` after installing `pymupdf`; this renders the supplied PDF without redrawing it.

**Verification boundary:** [release_status.json](metadata/release_status.json) reports local checks of frozen figure/table reconstruction and a synthetic example; **full upstream model-training reproduction remains unverified**. The manuscript is not included, and no archival DOI has been assigned. See the [complete reproduction instructions](docs/reproduction.md).

## 🗂️ Find the code you need

| What you want to inspect | Where |
| :--- | :--- |
| 🔁 Repeat-aware estimators and calibration | [`analysis/repeat_calibration/`](analysis/repeat_calibration/) |
| 📏 Baselines, matching and reproducibility metrics | [`analysis/baselines/`](analysis/baselines/) |
| 🧩 CP↔GE residual and sequential-evidence analyses | [`analysis/cross_modal/`](analysis/cross_modal/) |
| 🧫 BBBC047/cpg0004/sci-Plex3 validation | [`analysis/benchmarks/`](analysis/benchmarks/) · [`analysis/external_validation/`](analysis/external_validation/) |
| 🧠 Predictor supervision and target decomposition | [`analysis/predictor_supervision/`](analysis/predictor_supervision/) |
| 📊 Frozen figures and table builders | [`figures/`](figures/) · [`tables/`](tables/) |
| 📁 Numerical source data, mappings and integrity | [`data/`](data/) · [`docs/source_data_map.csv`](docs/source_data_map.csv) · [`metadata/FILE_MANIFEST.csv`](metadata/FILE_MANIFEST.csv) |

## 📃 Licence & citation

Code is available under the [MIT licence](LICENSE). Original assay datasets and third-party software remain under their own terms; see [third-party notices](docs/third_party_software.md) and [data access](docs/data_access.md). This is a research code package. **Please do not infer a paper DOI, journal acceptance or full reproducibility from the presence of code and frozen plots.**

---

<div align="center">
  <strong>Reliable perturbation responses begin with independent evidence.</strong><br>
  <sub><a href="#top">↑ Back to top</a> · <a href="https://zhao-weijing.github.io/RePert-code/">Project website ↗</a> · <a href="figures/data/figure_1/figure1_authors.pdf">Original Figure 1 ↗</a></sub>
</div>
