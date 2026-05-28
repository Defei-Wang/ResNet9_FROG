import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random # 确保导入了 random 模块
from torch.utils.data import DataLoader

from Model import resnet9_with_attention 
from DataLoader import FROGdata 
from FROG_Utils import get_phys_props, get_time_axis
# ==========================================
# 0. 锁定随机种子 
# ==========================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
set_seed(42)

# ==========================================
# 1. 路径配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset_20k")
CSV_PATH = os.path.join(DATA_DIR, "Ere_20k.csv")
SHG_DIR = os.path.join(DATA_DIR, "SHG")
THG_DIR = os.path.join(DATA_DIR, "THG")

LOG_DIR = os.path.join(BASE_DIR, 'logs_SHG_only_20k')
WEIGHTS_PATH = os.path.join(LOG_DIR, 'bestnet_SHG_only.pt')
SAVE_FIG_DIR = os.path.join(LOG_DIR, 'evaluation_plots')
os.makedirs(SAVE_FIG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 初始化模型
# ==========================================
model = resnet9_with_attention(6, 64, 512).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

# ==========================================
# 3. 加载测试集数据
# ==========================================
print("⏳ 正在加载 4000 组测试集数据...")
test_dataset = FROGdata(csv_file=CSV_PATH, img_dir_shg=SHG_DIR, img_dir_thg=THG_DIR, train=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# ==========================================
# 4. 计算全局平均 MSE 
# ==========================================
criterion = torch.nn.MSELoss()
all_mses = []

print("📊 正在计算单通道模型(SHG)全局测试误差 (MSE)...")
with torch.no_grad():
    for inputs, targets, _ in test_loader:
        # 【单通道核心】：强行掩膜抹除 THG 信号
        inputs[:, 3:6, :, :] = 0.0
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        
        for i in range(outputs.size(0)):
            loss = criterion(outputs[i], targets[i])
            all_mses.append(loss.item())

mean_mse = np.mean(all_mses)
print("\n" + "="*40)
print("🏆 单通道模型 (SHG Only) 测试集评估报告")
print("="*40)
print(f"评估样本数: {len(all_mses)}")
print(f"平均均方误差 (Average MSE): {mean_mse:.8e}")
print(f"最大误差 (Max MSE): {np.max(all_mses):.8e}")
print("="*40 + "\n")

mse_df = pd.DataFrame({"Sample_Index": range(len(all_mses)), "MSE": all_mses})
mse_df.to_csv(os.path.join(LOG_DIR, 'test_mse_distribution_SHG.csv'), index=False)



# ==========================================
# 6. 生成对比图
# ==========================================


# 从 4000 个验证集样本中，随机抽取 30 个样本进行画图
# (如果你想画 50 张，就把 30 改成 50)
sample_indices = random.sample(range(4000), 30)

# 如果你不想随机，只想按顺序看前 30 个，可以改成：
# sample_indices = range(30)

time_axis = np.linspace(-400, 400, 256) 

print(f"🎨 开始生成 {len(sample_indices)} 张单通道反演物理对比图 (|E| & Phase)...")

# ... 下方的 for 循环绘图代码保持完全不变 ...

for idx in sample_indices:
    inputs, targets, filename = test_dataset[idx]
    
    # 预测前必须掩膜
    inputs_shg = inputs.clone().unsqueeze(0)
    inputs_shg[:, 3:6, :, :] = 0.0
    inputs_shg = inputs_shg.to(device)
    
    with torch.no_grad():
        outputs = model(inputs_shg)
    
    pred_array = outputs.cpu().squeeze().numpy()
    true_array = targets.cpu().squeeze().numpy()
    
    A_true, P_true = get_phys_props(true_array[:256], true_array[256:])
    A_pred, P_pred = get_phys_props(pred_array[:256], pred_array[256:])
    
    peak_idx = np.argmax(A_true)
    P_true -= P_true[peak_idx]
    P_pred -= P_pred[peak_idx]
    
    P_true[A_true < 0.05] = np.nan
    P_pred[A_pred < 0.05] = np.nan
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150)
    
    # 子图 1: 绝对电场 |E|
    ax1.plot(time_axis, A_true, 'k-', label='Ground Truth |E|', linewidth=3)
    ax1.plot(time_axis, A_pred, 'b-.', label='SHG-Only Predict', linewidth=2)
    ax1.set_title(f"Sample ID: {filename} - Absolute Electric Field (SHG-Only)", fontsize=14)
    ax1.set_ylabel("Amplitude |E| (a.u.)", fontsize=12)
    ax1.set_xlim([-250, 250])
    ax1.set_ylim([-0.05, 1.1])
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='upper right')
    
    # 子图 2: 连续相位
    ax2.plot(time_axis, P_true, 'k-', label='Ground Truth Phase', linewidth=3, alpha=0.3)
    ax2.plot(time_axis, P_pred, 'b-.', label='Predicted Phase', linewidth=2)
    ax2.set_title("Temporal Phase", fontsize=14)
    ax2.set_xlabel("Time (fs)", fontsize=12)
    ax2.set_ylabel("Phase (rad)", fontsize=12)
    ax2.set_xlim([-250, 250])
    
    valid_p = P_true[~np.isnan(P_true)]
    if len(valid_p) > 0:
        ax2.set_ylim([np.min(valid_p)-2, np.max(valid_p)+2])
        
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_FIG_DIR, f"SHG_Phys_{filename}.png")
    plt.savefig(save_path)
    plt.close()
    
    print(f"   => ✅ 样本 {filename} 图像已保存至 {save_path}")

print("\n🎉 单通道评估圆满完成！")