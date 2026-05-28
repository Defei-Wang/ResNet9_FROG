```markdown
[English Version](#english-version) | [中文版](#中文版)

---

# 中文版

## ResNet9-FROG 脉冲反演系统

本项目为基于 ResNet9 与双通道空间注意力机制的飞秒激光脉冲频率分辨光学门控（FROG）相位恢复系统。实现从二维非线性时频迹线到一维复电场（实虚部）的正交回归。

### 1. 核心模块
系统底层逻辑由以下三个基准脚本构成：
* `Model.py`: 定义深度学习网络拓扑（包含 ResNet9 主干、双通道注意力分流、直角坐标回归头）。
* `Dataholder.py`: 负责张量转换、动态相位锚定及 DataLoader 批处理管线。
* `FROG_Utils.py`: 物理计算引擎，包含脉冲生成、高阶啁啾调制、及 DFT 抗混叠加窗算子。

### 2. 标准运行流水线
请严格按照以下顺序执行脚本以复现系统功能：

**Step 1: 物理数据生成**
```bash
python generate_20k_dataset.py

```

> 生成包含多峰碎裂、高阶色散的 20,000 组训练样本（输出至 `dataset_20k/`）。

**Step 2: 模型训练**

```bash
python trainDual_20k.py

```

> 执行双通道（SHG+THG）融合训练。单通道基线对比可运行 `train_SHG_only_20k.py`。（权重保存至 `logs_Dual_20k/`）。

**Step 3: 模型评估与推断**

```bash
python evaluate_Dual_20k.py

```

> 测算测试集 MSE 分布，并可视化反演电场对比图。

**Step 4: 故障诊断与鲁棒性验证（可选）**

* `test_robustness.py`: 注入高斯白噪声测试模型抗噪边界。
* `diagnose_dual_errors.py`: 提取并分析 MSE 离群样本（长尾/极端色散脉冲）。
* `plot_final_ablation.py`: 绘制消融实验对比图表。

### 3. 项目结构摘要

```text
E:\FROG_DL_Project
├── dataset_20k/            # 训练与验证数据集 (SHG/THG/CSV)
├── dataset_clean_test/     # 纯净测试集
├── logs_Dual_20k/          # 双通道模型权重 (.pt) 及训练日志
├── logs_SHG_only_20k/      # 单通道对比模型权重及日志
├── Model.py                # [核心] 网络架构
├── Dataholder.py           # [核心] 数据加载管线
├── FROG_Utils.py           # [核心] 物理算子库
├── generate_*.py           # 数据集生成脚本
├── train*.py               # 训练脚本
├── evaluate_*.py           # 评估脚本
├── diagnose_*.py           # 误差诊断脚本
└── *_plots.py              # 学位论文图表绘制脚本

```

---

# English Version

[中文版](https://www.google.com/search?q=%23%E4%B8%AD%E6%96%87%E7%89%88) | [English Version](https://www.google.com/search?q=%23english-version)

---

## ResNet9-FROG Pulse Retrieval System

This project is a deep learning-based phase retrieval system for femtosecond laser pulses using Frequency-Resolved Optical Gating (FROG). It employs a ResNet9 architecture with a dual-channel spatial attention mechanism to directly map 2D time-frequency traces to 1D complex electric fields.

### 1. Core Modules

The underlying logic is encapsulated in three foundational scripts:

* `Model.py`: Defines the neural network topology (ResNet9 backbone, dual-channel attention fusion, and Cartesian regression head).
* `Dataholder.py`: Handles tensor transformations, dynamic phase anchoring, and DataLoader pipelines.
* `FROG_Utils.py`: Physics computational engine, including pulse generation, high-order chirp modulation, and anti-aliasing DFT windowing.

### 2. Standard Execution Pipeline

Execute the scripts in the following order to reproduce the workflow:

**Step 1: Dataset Generation**

```bash
python generate_20k_dataset.py

```

> Generates 20,000 training samples with physical constraints and high-order dispersion (Outputs to `dataset_20k/`).

**Step 2: Model Training**

```bash
python trainDual_20k.py

```

> Initiates dual-channel (SHG+THG) training. For baseline comparison, run `train_SHG_only_20k.py`. (Weights are saved to `logs_Dual_20k/`).

**Step 3: Evaluation & Inference**

```bash
python evaluate_Dual_20k.py

```

> Calculates MSE distributions on the test set and visualizes the retrieved electric fields.

**Step 4: Diagnostics & Robustness (Optional)**

* `test_robustness.py`: Injects AWGN to evaluate the model's noise resilience.
* `diagnose_dual_errors.py`: Extracts and analyzes MSE outlier samples (extreme dispersion/multi-peak pulses).
* `plot_final_ablation.py`: Generates ablation study charts for thesis documentation.

### 3. Directory Structure Summary

```text
E:\FROG_DL_Project
├── dataset_20k/            # Training/Validation Dataset (SHG/THG/CSV)
├── dataset_clean_test/     # Hold-out Test Dataset
├── logs_Dual_20k/          # Dual-channel weights (.pt) & logs
├── logs_SHG_only_20k/      # Single-channel baseline weights & logs
├── Model.py                # [Core] Network Architecture
├── Dataholder.py           # [Core] Data Loading Pipeline
├── FROG_Utils.py           # [Core] Physics Operations Library
├── generate_*.py           # Dataset Generation Scripts
├── train*.py               # Training Pipelines
├── evaluate_*.py           # Evaluation Scripts
├── diagnose_*.py           # Outlier Diagnostic Scripts
└── *_plots.py              # Thesis Visualization Scripts

```

```

```
