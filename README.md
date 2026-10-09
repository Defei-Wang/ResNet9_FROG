# ResNet9-FROG Pulse Retrieval System / 脉冲反演系统

[English Version](#english-version) | [中文版](#中文版)

---

## 中文版

### 1. 系统概述
本项目实现了一个基于 ResNet9 骨干网络与前置双分支空间注意力机制的飞秒激光脉冲频率分辨光学门控（FROG）相位恢复系统。系统采用数据驱动范式，直接输入二维非线性时频迹线，端到端回归一维复电场（直角坐标系实部与虚部序列），在底层数学表征上彻底规避了极坐标回归固有的相位包裹（Phase Wrapping）不连续问题。

### 2. 核心模块架构
整个工程由以下三个底层脚本提供核心支撑：
* `Model.py`: 网络拓扑定义模块。构建了包含双分支空间注意力分流机制（独立加权二阶 SHG 与三阶 THG 迹线的高频干涉条纹）、ResNet9 特征编码器以及 8192 维全连接层对接的一维直角坐标正交回归头。
* `DataLoader.py`: 数据流控与预处理模块。实现张量变换、自动寻找能量主峰平移对齐至网格正中心（128偏置校准）、以及基于瞬时相位的相干归一化锚定管线。
* `FROG_Utils.py`: 物理特征提取与后处理模块。提供标准的时域坐标轴生成，并执行一维复电场的标准相位解包（Unwrap）与最高峰零点锚定；在归一化电场模长极低（<8%）的弱信号区部署强力掩膜，将失去物理实在意义的噪声相位强制置为 `NaN`（非数），以确保可视化图表的物理洁净度。

### 3. 标准执行流水线
请严格按照以下工程阶段顺序运行脚本以复现完整闭环：

**第一阶段：数据仿真与边界防线固化**
```bash
python generate_20k_dataset.py
```
> **功能描述**: 基于多元脉冲基底（高斯、孤子、洛伦兹）注入最高 15π 强高阶啁啾及 5 阶多项式复调幅，并通过 10 阶超高斯窗与 Pre-roll 边界验证算子阻断离散傅里叶变换产生的边缘环绕伪影，生成 20,000 组具备严格物理约束的对抗性训练样本，输出至 `dataset_20k/`。

**第二阶段：数据流控验证（可选）**
```bash
python visualize_pulses.py
```
> **功能描述**: 抽样解析时域 CSV 标签及对应的双谐波图像，验证 `DataLoader.py` 动态平移对齐和归一化管线的数学正确性。

**第三阶段：模型表征训练**
```bash
python trainDual_20k.py
```
> **功能描述**: 启动双通道（SHG+THG）融合表征训练。若需执行消融对照，可运行单通道基线训练脚本 `train_SHG_only_20k.py`。最优权重（.pt）及训练损失日志将自动归档至相应的 `logs_*/` 文件夹。

**第四阶段：工程评估、测速与故障诊断**
* `evaluate_Dual_20k.py`: 在测试集上测算全局均方误差（MSE）分布，验证无显卡加速移动端 CPU 的前向推理延迟（55ms/张）与吞吐量（18 FPS）。
* `test_robustness.py`: 注入加性高斯白噪声（AWGN），测试模型在低信噪比恶劣工况下的泛化抗噪边界。
* `diagnose_dual_errors.py`: 自动提取并分析 MSE 残差偏高的离群异常样本（如极端色散长尾脉冲），诊断结果可视化输出至 `BadSamples_Gallery/`。
* `plot_final_ablation.py`: 聚合多组对比日志，绘制消融实验对比图表，供学位论文直接调用。

### 4. 项目项目目录结构
```text
E:\FROG_DL_Project
├── dataset_20k/            # 训练与验证数据集 (包含双谐波图像及 CSV 时域标签)
├── dataset_clean_test/     # 独立泛化测试集
├── logs_Dual_20k/          # 双通道模型权重 (bestnet_Dual.pt) 训练日志与诊断画廊
├── logs_SHG_only_20k/      # 单通道对比模型权重及日志
├── final_thesis_plots/     # 论文图表输出目录
├── Model.py                # [核心] 网络架构拓扑定义
├── DataLoader.py           # [核心] 数据流控、时域平移与相位锚定
├── FROG_Utils.py           # [核心] 后处理物理特征解包与弱信号 NaN 掩膜工具
├── generate_*.py           # 数据生成与仿真流水线
├── train*.py               # 训练与优化管线
├── evaluate_*.py           # 推理速度测算与评估管线
├── diagnose_*.py           # 边界误差诊断模块
└── *_plots.py              # 论文图表绘制模块
```
### 5. 参考文献与学术归属 (References & Acknowledgments)
本项目双谐波（SHG + THG）联合输入表征与反演策略的思想受到了以下前沿文献及开源工作的启发：
- **Paper**: Jaffray W, Guo Z, Di Falco A, et al. Machine-learning-assisted dual harmonic generation FROG for enhanced ultrafast pulse recovery[J/OL]. Machine Learning: Science and Technology, 2024, 5(4): 045074. http://dx.doi.org/10.1088/2632-2153/ad 9f21.
- **Related Project**: Reference concept from `https://github.com/ZooBeasts/Machine-learning-for-ultrafast-laser-characterization-FROG`.

**本项目的主要独立实现与工程增量 (Our Specific Contributions)**:
1. **网络与表征重构**: 采用轻量化双分支空间注意力 ResNet9 骨干，在直角坐标系下直接端到端回归复电场实部与虚部序列，消除极坐标回归的相位不连续性。
2. **物理边界防御**: 自研 10 阶超高斯滤波窗与 Pre-roll 边界校验算法，有效消除离散傅里叶变换的周期卷积边界伪影。
3. **低算力边缘部署**: 实现 CPU 端 55ms 低延迟推理，并开发了针对极弱信号区的物理置信度 NaN 动态掩膜后处理工具。
---

## English Version

### 1. System Overview
This project implements a deep learning-based phase retrieval system for femtosecond laser pulses using Frequency-Resolved Optical Gating (FROG). Driven by a data-driven paradigm, the system takes 2D non-linear time-frequency traces as input and performs end-to-end regression to map them to 1D complex electric fields in a Cartesian coordinate system (real and imaginary sequences). This representation intrinsically resolves the phase-wrapping discontinuity problem associated with polar coordinate regression.

### 2. Core Architecture
The underlying engineering framework is sustained by three core modules:
* `Model.py`: Network Topology Module. Defines the dual-branch spatial attention mechanism (which independently weights high-frequency interference fringes of the SHG and THG traces), the ResNet9 feature encoder, and a Cartesian regression head flattened to 8192 dimensions.
* `DataLoader.py`: Data Streaming Module. Executes tensor transformations, automated energy peak centering (128 index calibration shift), and coherent phase normalization anchoring.
* `FROG_Utils.py`: Physical Post-Processing Module. Generates standard linear time axes and performs phase unwrapping and peak alignment on the 1D complex electric field. It deploys an intensity mask filter that forces noisy phases in weak-signal regions (<8% normalized amplitude) to `NaN` (Not a Number), preserving the physical fidelity of visualizations.

### 3. Standard Execution Pipeline
Execute the following scripts sequentially to fully reproduce the engineering pipeline from simulation to inference:

**Phase 1: Physical Data Simulation**
```bash
python generate_20k_dataset.py
```
> **Function**: Blends diverse pulse bases (Gaussian, Sech², Lorentzian) with high-order chirps (up to 15π) and 5th-order polynomial amplitude modulations. It suppresses DFT aliasing artifacts using a 10th-order super-Gaussian window and a pre-roll boundary check, outputting 20,000 highly constrained samples to `dataset_20k/`.

**Phase 2: Stream Flow Verification (Optional)**
```bash
python visualize_pulses.py
```
> **Function**: Decodes temporal CSV labels and their corresponding harmonic traces to verify the mathematical correctness of the peak alignment and phase anchoring routines in `DataLoader.py`.

**Phase 3: Representation Learning**
```bash
python trainDual_20k.py
```
> **Function**: Initiates dual-channel (SHG+THG) representation training. For ablation analysis, run the single-channel baseline pipeline via `train_SHG_only_20k.py`. Optimal weights (.pt) and logs are routed to the respective `logs_*/` directory.

**Phase 4: Diagnostics & Performance Benchmarking**
* `evaluate_Dual_20k.py`: Computes global Mean Squared Error (MSE) metrics on the test set and profiles forward inference on CPU (achieving 55ms latency and 18 FPS throughput).
* `test_robustness.py`: Injects Additive White Gaussian Noise (AWGN) to benchmark generalization boundaries under low SNR environments.
* `diagnose_dual_errors.py`: Extracts and isolates MSE outlier samples (e.g., heavy-tailed pulses), exporting diagnostic curves to `BadSamples_Gallery/`.
* `plot_final_ablation.py`: Aggregates disparate training logs to generate comparative ablation study charts for thesis documentation.

### 4. Directory Structure Summary
```text
E:\FROG_DL_Project
├── dataset_20k/            # Training/Validation Dataset (Harmonic images & CSV labels)
├── dataset_clean_test/     # Independent Evaluation Dataset
├── logs_Dual_20k/          # Dual-channel weights (bestnet_Dual.pt), logs, and diagnostics
├── logs_SHG_only_20k/      # Single-channel baseline weights and log tracking
├── final_thesis_plots/     # Output path for publication-ready figures
├── Model.py                # [Core] Network Architecture Specification
├── DataLoader.py           # [Core] Data Streaming, Temporal Shifting, and Phase Anchoring
├── FROG_Utils.py           # [Core] Post-processing Phase Unwrapping & NaN Masking Tool
├── generate_*.py           # Physics Simulation and Data Pipelines
├── train*.py               # Optimization and Training Pipelines
├── evaluate_*.py           # Inference Profiling and Benchmarking Pipelines
├── diagnose_*.py           # Boundary Error Diagnostic Modules
└── *_plots.py              # Thesis Visualization Scripts
```
### 5. References & Acknowledgments
The concept behind the dual-harmonic (SHG + THG) combined input characterization and inversion strategy for this project was inspired by the following cutting-edge literature and open-source work：
- **Paper**: Jaffray W, Guo Z, Di Falco A, et al. Machine-learning-assisted dual harmonic generation FROG for enhanced ultrafast pulse recovery[J/OL]. Machine Learning: Science and Technology, 2024, 5(4): 045074. http://dx.doi.org/10.1088/2632-2153/ad 9f21.
- **Related Project**: Reference concept from `https://github.com/ZooBeasts/Machine-learning-for-ultrafast-laser-characterization-FROG`.
