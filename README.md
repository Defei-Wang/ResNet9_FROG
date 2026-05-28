<a id="chinese"></a>
[English Version](#english) | [中文版](#chinese)

---

# 中文版

## ResNet9-FROG 脉冲反演系统

本项目为基于 ResNet9 与双通道空间注意力机制的飞秒激光脉冲频率分辨光学门控（FROG）相位恢复系统。核心目标是实现从二维非线性时频迹线到一维复电场（实部与虚部）的端到端正交回归，从而规避传统相位包裹带来的不连续性问题。

### 1. 核心代码架构
系统底层逻辑由以下三个基准脚本构成：
* `Model.py`: 深度学习网络拓扑定义（包含 ResNet9 主干、双通道空间注意力分流机制，以及 8192 维展平对接的直角坐标回归头）。
* `DataLoader.py`: 数据加载管线，负责张量转换、动态相位锚定（峰值对齐与全局相位归一化）及多线程批处理。
* `FROG_Utils.py`: 物理计算引擎，包含超快脉冲基底混合、高阶啁啾调制、及用于抑制离散傅里叶变换混叠的超高斯加窗算子。

### 2. 标准执行流水线
请严格按照以下顺序执行脚本，以完成从物理数据仿真到模型推断的完整工程闭环：

**Step 1: 物理数据集生成**
> 运行: `python generate_20k_dataset.py`
> 
> **功能**: 生成包含多峰碎裂、高阶色散的 20,000 组强物理约束训练样本。生成的图像及标签将自动落盘至 `dataset_20k/` 目录。

**Step 2: 模型训练**
> 运行: `python trainDual_20k.py`
> 
> **功能**: 执行 SHG 与 THG 双通道融合训练。如需进行单通道基线对比，请运行 `train_SHG_only_20k.py`。训练日志与最优模型权重（.pt）将归档至 `logs_Dual_20k/`。

**Step 3: 模型评估与推断**
> 运行: `python evaluate_Dual_20k.py`
> 
> **功能**: 测算模型在独立测试集下的均方误差（MSE）分布，并输出反演复电场（实部、虚部及绝对包络）的可视化对齐曲线。

**Step 4: 误差诊断与消融分析（可选）**
* `test_robustness.py`: 注入高斯白噪声（AWGN），测试模型在极端信噪比下的抗噪边界。
* `diagnose_dual_errors.py`: 提取并分析 MSE 离群样本（如强色散长尾脉冲），诊断结果输出至 `BadSamples_Gallery/`。
* `plot_final_ablation.py`: 提取多组训练日志，绘制消融实验对比图表，供学位论文直接调用。

### 3. 项目拓扑结构
```text
E:\FROG_DL_Project
├── dataset_20k/            # 训练与验证数据集 (包含双谐波图像及 CSV 时域标签)
├── dataset_clean_test/     # 独立泛化测试集
├── logs_Dual_20k/          # 双通道模型权重 (.pt) 及训练损失日志
├── logs_SHG_only_20k/      # 单通道对比模型权重及日志
├── Model.py                # [Core] 网络架构
├── DataLoader.py           # [Core] 数据流控与相位锚定
├── FROG_Utils.py           # [Core] 物理算子库
├── generate_*.py           # 数据生成管线
├── train*.py               # 训练管线
├── evaluate_*.py           # 评估推断管线
├── diagnose_*.py           # 误差诊断模块
└── *_plots.py              # 论文图表绘制模块
```

---

<br>

<a id="english"></a>
# English Version

[中文版](#chinese) | [English Version](#english)

---

## ResNet9-FROG Pulse Retrieval System

This project implements a deep learning-based phase retrieval system for femtosecond laser pulses using Frequency-Resolved Optical Gating (FROG). It utilizes a ResNet9 architecture with a dual-channel spatial attention mechanism to directly map 2D non-linear time-frequency traces to 1D complex electric fields (real and imaginary components), effectively resolving the phase-wrapping discontinuity issue.

### 1. Core Architecture
The system is built upon three foundational modules:
* `Model.py`: Defines the neural network topology, including the ResNet9 backbone, dual-channel attention fusion, and a Cartesian regression head flattened to 8192 dimensions.
* `DataLoader.py`: Handles tensor transformations, dynamic phase anchoring (peak centering and phase normalization), and batched data loading.
* `FROG_Utils.py`: The physics computational engine, executing pulse envelope blending, high-order chirp modulation, and anti-aliasing super-Gaussian windowing.

### 2. Standard Execution Pipeline
Execute the following scripts sequentially to reproduce the end-to-end workflow:

**Step 1: Dataset Generation**
> Command: `python generate_20k_dataset.py`
> 
> **Function**: Generates 20,000 training samples subject to strict physical constraints, injecting multi-peak fragmentation and extreme dispersion. Outputs are routed to the `dataset_20k/` directory.

**Step 2: Model Training**
> Command: `python trainDual_20k.py`
> 
> **Function**: Initiates dual-channel (SHG+THG) representation learning. For a single-channel baseline comparison, execute `train_SHG_only_20k.py`. Training logs and optimal weights (.pt) are saved in `logs_Dual_20k/`.

**Step 3: Evaluation & Inference**
> Command: `python evaluate_Dual_20k.py`
> 
> **Function**: Calculates the Mean Squared Error (MSE) distribution on the hold-out test set and visualizes the retrieved electric fields against ground truth values.

**Step 4: Diagnostics & Ablation Analysis (Optional)**
* `test_robustness.py`: Injects Additive White Gaussian Noise (AWGN) to benchmark the model's noise resilience.
* `diagnose_dual_errors.py`: Extracts and analyzes MSE outlier samples (e.g., heavy-tailed pulses), exporting diagnostics to `BadSamples_Gallery/`.
* `plot_final_ablation.py`: Compares multiple training logs to generate ablation study charts for academic publications.

### 3. Directory Structure
```text
E:\FROG_DL_Project
├── dataset_20k/            # Training/Validation Dataset (Images & CSV labels)
├── dataset_clean_test/     # Independent Test Dataset
├── logs_Dual_20k/          # Dual-channel weights (.pt) & logs
├── logs_SHG_only_20k/      # Single-channel baseline weights & logs
├── Model.py                # [Core] Network Architecture
├── DataLoader.py           # [Core] Data Streaming & Phase Anchoring
├── FROG_Utils.py           # [Core] Physics Operations Library
├── generate_*.py           # Dataset Generation Scripts
├── train*.py               # Training Pipelines
├── evaluate_*.py           # Evaluation & Inference Scripts
├── diagnose_*.py           # Outlier Diagnostic Scripts
└── *_plots.py              # Thesis Visualization Scripts
```
