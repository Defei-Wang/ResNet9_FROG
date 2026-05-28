# ResNet9_FROG
这是一个符合资深程序员规范、高信息密度且完全移除Emoji的 `README.md` 源码。针对 GitHub Markdown 的特性，顶端配置了中英双语的跳转锚点链接，方便一键切换阅读。

```markdown
# FROG_DL_Project

[中文说明](#中文说明) | [English Description](#english-description)

---

## 中文说明

本项目基于深度学习（ResNet9 与注意力机制复合架构）实现超快激光脉冲二阶（SHG）与三阶（THG）频率分辨光学门控（FROG）迹线的非迭代、实时时域复电场快速反演。

### 核心代码结构

| 文件名 | 模块类型 | 核心功能与物理/工程意义 |
| :--- | :--- | :--- |
| `Model.py` | 模型架构 | 核心网络定义。包含前置双分支独立空间注意力机制与深度降维的 ResNet9 主干，末端采用实虚部正交分解回归头，规避相位包裹。 |
| `Dataholder.py` | 数据管线 | 继承 PyTorch Dataset。内嵌动态相位锚定与能量主峰时域中心对准算子，消除平移二义性，保障训练收敛。 |
| `FROG_Utils.py` | 工具函数 | 包含物理仿真基础算子、信号流向转换及损失函数评估模块。 |
| `generate_20k_dataset.py` | 数据生成 | 注入高阶色散与多项式复振幅调制的脉冲仿真管线。包含 10 阶超高斯窗截断与 Pre-roll 边界验证，清除 DFT 环绕伪影。 |
| `trainDual_20k.py` | 算法训练 | 双通道（SHG + THG）联合反演模型训练脚本，引入带温重启的余弦退火学习率调度器。 |
| `evaluate_Dual_20k.py` | 性能评估 | 评估测试集推理延迟、吞吐量（FPS）及均方误差（MSE）分布，支持 CPU 极限基准性能测试。 |
| `plot_final_ablation.py` | 论文绘图 | 全自动读取日志，生成双通道与单通道（SHG-only）对比的消融实验科学图表。 |

### 执行工作流与运行顺序

请严格按照以下顺序执行脚本：

1. **生成训练数据**
   ```bash
   python generate_20k_dataset.py
   python generate_test_dataset.py

```

*说明：将在本地构建包含严格物理约束的 `dataset_20k` 与测试集。*

2. **执行模型训练**
```bash
python trainDual_20k.py
python train_SHG_only_20k.py

```


*说明：启动双通道网络与消融对照组的训练，权重及损失曲线自动输出至各自的 `logs` 文件夹。*
3. **系统性能评估与量化测速**
```bash
python evaluate_Dual_20k.py
python evaluate_SHG_only.py

```


*说明：测试纯 CPU 或 GPU 环境下的单张推理延迟及 FPS。*
4. **误差诊断与论文绘图**
```bash
python diagnose_dual_errors.py
python plot_final_ablation.py

```


*说明：捕获并归档离群坏样本（Bad Samples），同时全自动编译输出毕业论文所需的高清实验图表至 `final_thesis_plots`。*

---

## English Description

This project implements a non-iterative, real-time retrieval system for ultra-fast laser pulses from Second-Harmonic Generation (SHG) and Third-Harmonic Generation (THG) Frequency-Resolved Optical Gating (FROG) traces based on a deep learning architecture (ResNet9 combined with Spatial Attention).

### Core Architecture

| File Name | Module Type | Key Functionality & Significance |
| --- | --- | --- |
| `Model.py` | Architecture | Core network definition. Features dual-branch independent spatial attention modules, a ResNet9 backbone, and an orthogonal regression head splitting real/imaginary components to bypass phase-wrapping. |
| `Dataholder.py` | Data Pipeline | PyTorch Dataset implementation. Integrates dynamic phase anchoring and peak-centering operators to eliminate temporal shift ambiguities and stabilize convergence. |
| `FROG_Utils.py` | Utilities | Contains non-linear optics simulation primitives, signal tensor transforms, and customized loss evaluation modules. |
| `generate_20k_dataset.py` | Data Engine | Pulse simulation engine injecting high-order dispersion and polynomial amplitude modulation. Deploys 10th-order super-Gaussian windowing and pre-roll boundary verification to clear DFT wrap-around artifacts. |
| `trainDual_20k.py` | Optimization | Training script for the dual-channel retrival model, regulated by a Cosine Annealing with Warm Restarts learning rate scheduler. |
| `evaluate_Dual_20k.py` | Evaluation | Assesses inference latency, throughput (FPS), and MSE distribution over test sets, supporting pure CPU performance benchmarking. |
| `plot_final_ablation.py` | Visualization | Automatically processes logs to generate publication-grade charts for ablation analysis (Dual-channel vs. SHG-only). |

### Execution Workflow

Execute scripts strictly in the following sequential order:

1. **Synthesize Physical Datasets**
```bash
python generate_20k_dataset.py
python generate_test_dataset.py

```


*Note: Establishes physics-constrained training samples under `dataset_20k` and clean evaluation sets.*
2. **Launch Model Training**
```bash
python trainDual_20k.py
python train_SHG_only_20k.py

```


*Note: Runs optimization for both the main model and the ablation baseline. Weights and tracking metrics are directed to respective `logs` directories.*
3. **Evaluate and Benchmark Performance**
```bash
python evaluate_Dual_20k.py
python evaluate_SHG_only.py

```


*Note: Measures single-shot inference latency and FPS throughput on local execution targets (CPU/GPU).*
4. **Run Diagnostics & Compile Figures**
```bash
python diagnose_dual_errors.py
python plot_final_ablation.py

```


*Note: Captures and isolates worst-performing statistical outliers while generating vector plots into `final_thesis_plots`.*

```

```
