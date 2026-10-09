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

Run the lightweight **synthetic example** (Python 3.10+; no biological data or GPU required):

```bash
git clone https://github.com/Zhao-weijing/RePert-code.git
cd RePert-code
python -m pip install numpy
python scripts/run_estimator_example.py
python scripts/verify_package.py
```

> The example demonstrates estimator and calibration code; it does not reproduce the biological experiments.

## 🗂️ Code, data & reproduction

| Resource | Description |
| :--- | :--- |
| [🔁 Response estimators](analysis/repeat_calibration/) | Repeat-aware estimation and calibration |
| [🧩 Cross-modal analysis](analysis/cross_modal/) | Conditional CP ↔ GE updates |
| [🧠 Predictor supervision](analysis/predictor_supervision/) | Learning from estimated response targets |
| [📊 Figures & source data](figures/data/) | Frozen results and original figure assets |
| [🧫 Data access](docs/data_access.md) | BBBC047/Rosetta, cpg0004-LINCS and sci-Plex3 |
| [📖 Reproduction guide](docs/reproduction.md) | Figure/table reconstruction and upstream requirements |

The package provides analysis code, frozen numerical results and a synthetic example. **Full upstream model-training reproduction has not been verified**; see [release status](metadata/release_status.json). Code is released under the [MIT licence](LICENSE); third-party data retain their original terms.

---
<p align="center"><sub><a href="#top">↑ Back to top</a> · <a href="https://zhao-weijing.github.io/RePert-code/">Project website ↗</a></sub></p>
