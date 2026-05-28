import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
import random
from Model import resnet9_with_attention 
from DataLoader import FROGdata 
from FROG_Utils import get_phys_props, get_time_axis

# ==========================================
# 0. 锁定环境
# ==========================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
set_seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset_20k")
WEIGHTS_SHG = os.path.join(BASE_DIR, 'logs_SHG_only_20k', 'bestnet_SHG_only.pt')
WEIGHTS_DUAL = os.path.join(BASE_DIR, 'logs_Dual_20k', 'bestnet_Dual.pt')
SAVE_FIG_DIR = os.path.join(BASE_DIR, 'final_thesis_plots')
os.makedirs(SAVE_FIG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 加载模型
# ==========================================
model_shg = resnet9_with_attention(6, 64, 512).to(device)
model_shg.load_state_dict(torch.load(WEIGHTS_SHG, map_location=device, weights_only=True))
model_shg.eval()

model_dual = resnet9_with_attention(6, 64, 512).to(device)
model_dual.load_state_dict(torch.load(WEIGHTS_DUAL, map_location=device, weights_only=True))
model_dual.eval()

test_dataset = FROGdata(csv_file=os.path.join(DATA_DIR, "Ere_20k.csv"), 
                        img_dir_shg=os.path.join(DATA_DIR, "SHG"), 
                        img_dir_thg=os.path.join(DATA_DIR, "THG"), train=False)

# ==========================================
# 2. 绘图配置 (Nature 风格)
# ==========================================
COLOR_TRUTH_E = '#222222'   # 振幅真值：深黑
COLOR_TRUTH_P = '#888888'   # 相位真值：灰色
COLOR_DUAL = '#0072B2'      # 双通道：深蓝
COLOR_SHG = '#D55E00'       # 单通道：朱红

time_axis = get_time_axis()
NUM_PLOTS = 20 
sample_indices = random.sample(range(len(test_dataset)), NUM_PLOTS)

print(f"🎨 正在生成 {NUM_PLOTS} 张带迹线的终极消融对比图...")

for idx in sample_indices:
    inputs, targets, filename = test_dataset[idx]
    inputs_tensor = inputs.unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_dual = model_dual(inputs_tensor).cpu().squeeze().numpy()
        
        # 掩蔽 THG 通道 (通道 3, 4, 5)，模拟单 SHG 输入
        inputs_shg = inputs_tensor.clone()
        inputs_shg[:, 3:6, :, :] = 0.0 
        out_shg = model_shg(inputs_shg).cpu().squeeze().numpy()
        
    truth = targets.cpu().numpy()
    
    # 提取 SHG 和 THG 迹线图像 (提取单通道即可可视化)
    # 注意：假设 inputs shape 为 [6, H, W]，0 为 SHG, 3 为 THG
    shg_trace = inputs[0].cpu().numpy()
    thg_trace = inputs[3].cpu().numpy()
    
    # ==========================================
    # 物理量提取与相位对齐
    # ==========================================
    A_true, P_true = get_phys_props(truth[:256], truth[256:])
    A_dual, P_dual = get_phys_props(out_dual[:256], out_dual[256:])
    A_shg, P_shg = get_phys_props(out_shg[:256], out_shg[256:])

    MASK_THRES = 0.05
    P_true[A_true < MASK_THRES] = np.nan
    P_dual[A_dual < MASK_THRES] = np.nan
    P_shg[A_shg < MASK_THRES] = np.nan
    
    peak_idx = np.argmax(A_true)
    if not np.isnan(P_true[peak_idx]):
        offset_true = P_true[peak_idx]
        P_true -= offset_true
        if not np.isnan(P_dual[peak_idx]): P_dual -= P_dual[peak_idx]
        if not np.isnan(P_shg[peak_idx]): P_shg -= P_shg[peak_idx]

    # ==========================================
    # 开始绘制画板 (3行，上排迹线，下两排波形)
    # ==========================================
    fig = plt.figure(figsize=(10, 9), dpi=300)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1])
    gs.update(hspace=0.25, wspace=0.15)
    
    # --- 1. SHG 迹线 ---
    ax_shg = fig.add_subplot(gs[0, 0])
    im1 = ax_shg.imshow(shg_trace, cmap='inferno', origin='lower', aspect='auto')
    ax_shg.set_title('(a) Input SHG Trace', fontweight='bold', fontsize=12)
    ax_shg.set_xticks([]) # 隐藏像素刻度
    ax_shg.set_yticks([])
    
    # --- 2. THG 迹线 ---
    ax_thg = fig.add_subplot(gs[0, 1])
    im2 = ax_thg.imshow(thg_trace, cmap='inferno', origin='lower', aspect='auto')
    ax_thg.set_title('(b) Input THG Trace', fontweight='bold', fontsize=12)
    ax_thg.set_xticks([])
    ax_thg.set_yticks([])

    # --- 3. 振幅对比图 (横跨两列) ---
    ax_amp = fig.add_subplot(gs[1, :])
    ax_amp.plot(time_axis, A_true, color=COLOR_TRUTH_E, linestyle='-', linewidth=2.5, label='Ground Truth')
    ax_amp.plot(time_axis, A_dual, color=COLOR_DUAL, linestyle='--', linewidth=2, label='Dual-Channel (Proposed)')
    ax_amp.plot(time_axis, A_shg, color=COLOR_SHG, linestyle=':', linewidth=2, label='SHG-Only (Baseline)')
    
    ax_amp.set_title('(c) Amplitude |E| Reconstruction', fontweight='bold', fontsize=12, loc='left')
    ax_amp.set_ylabel('Amplitude (a.u.)', fontweight='bold', fontsize=11)
    ax_amp.set_ylim([-0.05, 1.15])
    ax_amp.legend(loc='upper right', frameon=False, fontsize=11)
    ax_amp.tick_params(direction='in', top=True, right=True)
    ax_amp.set_xticklabels([]) # 隐藏 X 轴刻度标签，留给最底下的图

    # --- 4. 相位对比图 (横跨两列) ---
    ax_phase = fig.add_subplot(gs[2, :])
    ax_phase.plot(time_axis, P_true, color=COLOR_TRUTH_P, linestyle='-', linewidth=3.5, alpha=0.5)
    ax_phase.plot(time_axis, P_dual, color=COLOR_DUAL, linestyle='--', linewidth=2)
    ax_phase.plot(time_axis, P_shg, color=COLOR_SHG, linestyle=':', linewidth=2)
    
    ax_phase.set_title('(d) Phase Reconstruction', fontweight='bold', fontsize=12, loc='left')
    ax_phase.set_xlabel('Time (fs)', fontweight='bold', fontsize=12)
    ax_phase.set_ylabel('Phase (rad)', fontweight='bold', fontsize=11)
    ax_phase.set_xlim([-200, 200])
    ax_phase.tick_params(direction='in', top=True, right=True)
    
    # 动态调整相位的 Y 轴范围
    valid_p = P_true[~np.isnan(P_true)]
    if len(valid_p) > 0:
        ax_phase.set_ylim([np.min(valid_p)-1.5, np.max(valid_p)+1.5])

    # 导出保存
    save_path = os.path.join(SAVE_FIG_DIR, f"Ablation_FullPanel_{filename}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

print(f"\n🎉 20张完整消融分析面板生成完毕！请前往 {SAVE_FIG_DIR} 文件夹验收。")