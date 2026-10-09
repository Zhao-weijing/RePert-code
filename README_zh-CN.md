<a id="top"></a>
<div align="center">
<h1>RePert</h1>
<p><strong>基于独立测量构建可重复的细胞扰动响应</strong></p>
<p>
  <a href="README.md">English</a> ·
  <a href="https://zhao-weijing.github.io/RePert-code/">🌐 项目主页</a> ·
  <a href="#quick-start">🚀 快速开始</a> ·
  <img src="https://img.shields.io/badge/Assay-Cell%20Painting-0C8F91?logo=microscope&logoColor=white" alt="Cell Painting"> <img src="https://img.shields.io/badge/Modality-Gene%20Expression-7753B6?logo=dna&logoColor=white" alt="Gene expression"> <img src="https://img.shields.io/badge/Principle-Independent%20Repeats-2A697A?logo=checkmarx&logoColor=white" alt="Independent biological measurements">
</p>
</div>

## 🧬 摘要

同一细胞扰动的不同实验重复可能存在明显差异，尤其在独立测量次数有限时，很难区分真实生物学响应与实验噪声。**RePert** 通过整合互补的重复测量估计方法，构建校准后的扰动响应，并利用正确匹配的 Cell Painting（CP）与基因表达（GE）证据进行条件更新。在所评估的实验设置中，该框架能够更好地恢复可重复、具有扰动特异性的信号，并探索这些估计对下游表型分析和扰动预测的价值；具体收益仍取决于任务与细胞背景。

<p align="center">
  <a href="figures/data/figure_1/figure1_authors.pdf">
    <img src="assets/repert_figure1.png" width="960" alt="RePert 图 1：独立测量、校准响应估计与跨模态条件更新">
  </a>
  <br>
  <sub>图 1 · RePert 整体框架。<a href="figures/data/figure_1/figure1_authors.pdf">查看原始高清 PDF ↗</a></sub>
</p>

## ✨ 方法概览

<table>
<tr>
<td width="33%" valign="top">
<strong>🔁 恢复可靠响应</strong><br><br>
利用独立实验重复，整合互补估计器，减少不稳定测量信号。
</td>
<td width="33%" valign="top">
<strong>🧩 整合互补模态</strong><br><br>
通过匹配的 CP 或 GE 信息，对已有扰动响应进行条件更新。
</td>
<td width="33%" valign="top">
<strong>🧠 支持下游建模</strong><br><br>
检验更可靠的响应是否改善形态学分析和扰动效应预测。
</td>
</tr>
</table>

**主要发现：** RePert 在测试设置中改善了扰动特异性的重复一致性，正确配对的跨模态信息还能提供增益；但下游生物学任务不一定同步受益。[查看冻结实验结果 ↗](figures/data/)

<a id="quick-start"></a>
## 🚀 快速开始

运行仓库自带的**合成数据示例**（Python 3.10+，无需下载真实实验数据或使用 GPU）：

```bash
git clone https://github.com/Zhao-weijing/RePert-code.git
cd RePert-code
python -m pip install numpy
python scripts/run_estimator_example.py
python scripts/verify_package.py
```

> 该示例只演示估计与校准代码的基本流程，并非真实生物学实验复现。

## 🗂️ 代码、数据与复现

| 资源 | 内容 |
| :--- | :--- |
| [🔁 扰动响应估计](analysis/repeat_calibration/) | 重复测量估计与校准 |
| [🧩 跨模态分析](analysis/cross_modal/) | CP ↔ GE 条件更新 |
| [🧠 预测模型监督](analysis/predictor_supervision/) | 基于响应估计训练预测器 |
| [📊 图表与源数据](figures/data/) | 冻结实验结果与原始图示 |
| [🧫 数据获取说明](docs/data_access.md) | BBBC047/Rosetta、cpg0004-LINCS、sci-Plex3 |
| [📖 复现指南](docs/reproduction.md) | 图表重建与上游训练依赖 |

仓库提供分析代码、冻结数值和合成示例，**目前尚未验证能够从头完整复现上游模型训练**，具体见[发布状态](metadata/release_status.json)。代码采用 [MIT 协议](LICENSE)，第三方数据保留各自原始许可。

---
<p align="center"><sub><a href="#top">↑ 返回顶部</a> · <a href="https://zhao-weijing.github.io/RePert-code/">项目主页 ↗</a></sub></p>
