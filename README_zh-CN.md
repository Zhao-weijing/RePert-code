<a id="top"></a>
<div align="center">
<h1>RePert</h1>
<p><strong>基于独立测量构建可重复的细胞扰动响应</strong></p>
<p>
  <a href="README.md"><img src="https://img.shields.io/badge/-English-0B7285?style=flat-square&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMS44IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik00IDZoMTJNMTAgM3YzTTYgMTBjMSAzIDMgNSA3IDdNMTQgOWMtMiA0LTUgNy05IDlNMTUgMjBsMy04IDMgOE0xNiAxN2g0Ii8%2BPC9zdmc%2B" alt="English"></a>
  <a href="https://zhao-weijing.github.io/RePert-code/"><img src="https://img.shields.io/badge/-%E9%A1%B9%E7%9B%AE%E4%B8%BB%E9%A1%B5-0B7285?style=flat-square&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMS44IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjkiLz48cGF0aCBkPSJNMyAxMmgxOE0xMiAzYzQgNSA0IDEzIDAgMThNMTIgM2MtNCA1LTQgMTMgMCAxOCIvPjwvc3ZnPg%3D%3D" alt="项目主页"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/-%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B-0B7285?style=flat-square&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMS44IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Im0xMyAyLTkgMTJoN2wtMSA4IDEwLTEyaC03bDAtOFoiLz48L3N2Zz4%3D" alt="快速开始"></a>
  <img src="https://img.shields.io/badge/-Cell%20Painting-0B7285?style=flat-square&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMS44IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxyZWN0IHg9IjMiIHk9IjQiIHdpZHRoPSIxOCIgaGVpZ2h0PSIxNiIgcng9IjMiLz48Y2lyY2xlIGN4PSI5IiBjeT0iMTAiIHI9IjIiLz48Y2lyY2xlIGN4PSIxNiIgY3k9IjkiIHI9IjIiLz48Y2lyY2xlIGN4PSIxMyIgY3k9IjE1IiByPSIyIi8%2BPC9zdmc%2B" alt="Cell Painting">
  <img src="https://img.shields.io/badge/-Gene%20expression-0B7285?style=flat-square&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMS44IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik01IDJjMCAxMCAxNCAxMCAxNCAyME0xOSAyYzAgMTAtMTQgMTAtMTQgMjBNNyA2aDEwTTYgMTBoMTJNNiAxNGgxMk03IDE4aDEwIi8%2BPC9zdmc%2B" alt="Gene expression">
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

**第一次了解 RePert？** 可以先想象一个实验：同一种药物、同一剂量，在多个独立采集板（acquisition plates）上重复测量，得到 Cell Painting（CP）或基因表达（GE）特征。由于实验变异，不同重复的响应并不完全一致。RePert 利用已经观测到的**部分重复（support）**估计更可靠的扰动响应，再用**独立留出的实验重复**进行评价。如果还存在正确匹配的另一模态，可以进一步利用它进行条件更新。

**一句话理解流程：** 有限的重复观测 → 校准后的响应估计 → 可选的 CP/GE 互补信息更新 → 生物学表型分析或扰动预测模型训练。

### 1. 几分钟运行一个最小示例

只需 **Python 3.10+ 和 NumPy**，不需要 GPU，也不用下载真实生物实验数据：

```bash
git clone https://github.com/Zhao-weijing/RePert-code.git
cd RePert-code
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
python -m pip install numpy

python scripts/run_estimator_example.py
python scripts/verify_package.py
```

仓库提供的合成数据包含 **32 个拟合药物、12 个独立校准药物和 5 个待估计药物**，每个观测有 16 个人工特征。脚本利用模拟的独立采集板拟合经验贝叶斯估计器，在单独的合成校准集上计算融合权重，为待估计药物生成响应，并验证模型保存和重新加载的一致性。三个药物集合彼此不重叠。

| 输出文件 | 可以从中看到什么 |
| :--- | :--- |
| `outputs/example/result.json` | `status: PASS`、不同集合的数量、校准权重与预测检查结果 |
| `outputs/example/predictions.csv` | 5 个待估计药物的响应，每行包含 16 个特征 |
| `outputs/example/synthetic_eb_model.npz` | 保存后的经验贝叶斯估计器，可重新加载 |
| `outputs/package_verification.json` | 文件 SHA-256 完整性、Python 语法及静态发布检查 |

> [!NOTE]
> **这是软件示例，不是完整的 ReCA 实验。** 它只调用经验贝叶斯与权重校准函数，**不会**训练 IMR/LSO、运行完整融合模型或复现论文的生物学结果。合成数据格式和示例边界见 [examples/README.md](examples/README.md)。

### 2. 先看论文结果，不必立即训练模型

如果你的目标是理解研究证据，可以先使用仓库附带的**冻结数值**重建主要图表和表格：

```bash
python -m pip install -r requirements-publication.txt
python figures/scripts/build_main_figures.py
python tables/predictor/scripts/build_predictor_table.py
python scripts/build_source_data_workbooks.py
```

建议从[原始 Figure 1](figures/data/figure_1/figure1_authors.pdf) 理解整体框架，再对照[参考图片](figures/reference/)与[图表源数据](figures/data/)阅读结果。如果想定位某个面板具体使用哪个 CSV，可以查看[源数据索引](docs/source_data_map.csv)。**这些命令基于已保存的结果生成图表，并不重新训练上游模型。** 补充图和输出目录见[复现指南](docs/reproduction.md)。

## 🗂️ 如何阅读仓库代码？

仓库包含多个实验流水线，**不存在一条命令就能从头训练所有 RePert 实验**。更合理的入门方式是按自己最关心的问题找入口：

| 你希望了解的内容 | 推荐入口 | 主要包含什么 |
| :--- | :--- | :--- |
| **扰动响应是如何估计的？** | [`analysis/repeat_calibration/`](analysis/repeat_calibration/) | 信号与噪声协方差估计、经验贝叶斯收缩、独立验证集校准和估计器比较 |
| **可靠性怎么评价？** | [`analysis/baselines/`](analysis/baselines/) | 输入与独立留出观测的划分、条件匹配、扰动特异性指标和基线 |
| **CP 与 GE 怎么互补？** | [`analysis/cross_modal/`](analysis/cross_modal/) | 条件残差更新、直接融合对照和顺序证据分析 |
| **如何用于扰动预测？** | [`analysis/predictor_supervision/`](analysis/predictor_supervision/) | 原始响应、估计响应及残差分解监督的预测实验 |
| **还有哪些数据集？** | [`analysis/external_validation/`](analysis/external_validation/) | LINCS Cell Painting 和 sci-Plex3 的外部分析 |
| **图和表的原始数值在哪里？** | [`figures/data/`](figures/data/)、[`data/source_data/`](data/source_data/)、[`data/reported_tables/`](data/reported_tables/) | 冻结面板数据、数值源数据与论文展示表格 |

**推荐阅读顺序：** [Figure 1](figures/data/figure_1/figure1_authors.pdf) → [合成示例](examples/README.md) → [响应估计](analysis/repeat_calibration/) → [跨模态证据](analysis/cross_modal/) → [预测监督](analysis/predictor_supervision/)。这样可以先理解科学逻辑，再深入实现。

## 🧫 真实数据与完整复现

项目分析使用了不同类型的细胞扰动实验数据：

| 数据集 | 细胞与模态 | 在项目中的作用 |
| :--- | :--- | :--- |
| **BBBC047 / Rosetta** | U2OS · CP + L1000 GE | 匹配的跨模态实验及重复响应估计 |
| **cpg0004-LINCS** | A549 · CP | 额外的 Cell Painting 重复测量 |
| **sci-Plex3** | A549、K562、MCF7 · GE pseudobulk | 基因表达证据与跨细胞背景分析 |

公开仓库提供**分析脚本、数值源数据、图表生成器和合成示例**，但没有包含完整上游训练所需的全部处理后实验矩阵、冻结划分和配置、模型权重及外部依赖。准备真实数据时，应先阅读[数据获取说明](docs/data_access.md)和[复现指南](docs/reproduction.md)，保留药物、剂量和独立采集板的身份信息，并根据对应脚本补齐所需输入，而不是直接使用代码中的占位路径。

**目前验证到什么程度？** 发布记录表明，合成示例和基于冻结数据的图表重建已经完成本地检查；**从头完整复现上游模型训练仍未验证**。具体见[发布状态](metadata/release_status.json)和[文件完整性清单](metadata/FILE_MANIFEST.csv)。代码采用 [MIT 许可](LICENSE)；原始数据和第三方工具遵循各自许可。目前尚未分配归档 DOI。

---

<p align="center"><sub><a href="#top">↑ 返回顶部</a> · <a href="https://zhao-weijing.github.io/RePert-code/">项目主页 ↗</a></sub></p>
