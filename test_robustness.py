import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from DataLoader import FROGdata
from Model import resnet9_with_attention
from tqdm import tqdm
from FROG_Utils import get_phys_props, get_time_axis

# ==========================================
# 1. 核心配置区
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset_20k')
SHG_DIR = os.path.join(DATA_DIR, 'SHG')
THG_DIR = os.path.join(DATA_DIR, 'THG')
CSV_PATH = os.path.join(DATA_DIR, 'Ere_20k.csv')

# 权重路径 (这里测试的是双通道的终极鲁棒性)
WEIGHTS_PATH = os.path.join(BASE_DIR, 'logs_Dual_20k', 'bestnet_Dual.pt')
LOG_DIR = os.path.join(BASE_DIR, 'logs_Dual_20k')

# 定义我们要扫描的噪声强度 (标准差): 0%, 1%, 3%, 5%, 8%, 12%, 15%
NOISE_LEVELS = [0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.15]

# ==========================================
# 2. 噪声注入函数
# ==========================================
def add_gaussian_noise(tensor, noise_std):
    """
    向归一化到 [0,1] 的 FROG 迹线张量中注入高斯白噪声
    """
    if noise_std <= 0.0:
        return tensor
    
    noise = torch.randn_like(tensor) * noise_std
    noisy_tensor = tensor + noise
    
    # 物理约束：相机的光强读数不可能是负数，且这里做过归一化，上限约为 1.0 (容许微小溢出)
    noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.1)
    return noisy_tensor

# ==========================================
# 3. 加载模型与数据集
# ==========================================
print("⏳ 正在准备鲁棒性压力测试...")
model = resnet9_with_attention(6, 64, 512).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
model.eval()

# 使用更大的 batch_size 加速测试集的全局评估
test_set = FROGdata(CSV_PATH, SHG_DIR, THG_DIR, train=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
criterion = nn.MSELoss()

# ==========================================
# 4. 压力测试主循环
# ==========================================
mse_results = []
example_saved = False

print(f"📊 开始扫描不同的噪声等级...")

with torch.no_grad():
    for nl in NOISE_LEVELS:
        total_mse = 0.0
        
        # 遍历整个测试集
        for i, (inputs, targets, filenames) in enumerate(tqdm(test_loader, desc=f"Noise Level: {nl:.0%}")):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 注入噪声！
            noisy_inputs = add_gaussian_noise(inputs, nl)
            
            # 模型预测
            outputs = model(noisy_inputs)
            loss = criterion(outputs, targets)
            total_mse += loss.item() * inputs.size(0)
            
            # 【提取素材】：保存一张 10% 噪声下的直观对比图供论文使用
            if nl == 0.10 and not example_saved and i == 0:
                example_saved = True
                
                fig = plt.figure(figsize=(12, 5))
                # 原始无噪声 SHG
                ax1 = plt.subplot(1, 3, 1)
                ax1.imshow(inputs[0, 0, :, :].cpu().numpy(), cmap='jet', aspect='auto')
                ax1.set_title("Clean SHG Trace")
                # 加了 10% 噪声的 SHG
                ax2 = plt.subplot(1, 3, 2)
                ax2.imshow(noisy_inputs[0, 0, :, :].cpu().numpy(), cmap='jet', aspect='auto')
                ax2.set_title("Noisy SHG Trace (10% Noise)")
                # 原始无噪声 THG
                ax3 = plt.subplot(1, 3, 3)
                ax3.imshow(noisy_inputs[0, 3, :, :].cpu().numpy(), cmap='jet', aspect='auto')
                ax3.set_title("Noisy THG Trace (10% Noise)")
                
                plt.tight_layout()
                example_path = os.path.join(LOG_DIR, 'Noise_10_Percent_Visual.png')
                plt.savefig(example_path)
                plt.close(fig)
                print(f"\n   📸 已保存 10% 噪声视觉示例图: {example_path}")
                
        # 计算该噪声等级下的平均 MSE
        avg_mse = total_mse / len(test_set)
        mse_results.append(avg_mse)
        print(f"   => 平均 MSE: {avg_mse:.6f}\n")

# ==========================================
# 5. 绘制论文抗噪折线图
# ==========================================
# 将结果保存为 CSV 方便日后重新画图
df_results = pd.DataFrame({'Noise_Level_Std': NOISE_LEVELS, 'Average_MSE': mse_results})
csv_save_path = os.path.join(LOG_DIR, 'Robustness_Noise_Results.csv')
df_results.to_csv(csv_save_path, index=False)

# 画图
plt.figure(figsize=(8, 6), dpi=150)
noise_percentages = [nl * 100 for nl in NOISE_LEVELS]
plt.plot(noise_percentages, mse_results, marker='o', markersize=8, linestyle='-', linewidth=2.5, color='darkred')

plt.title('Model Robustness against Gaussian Noise (Dual-Channel)', fontsize=14)
plt.xlabel('Gaussian Noise Level (%)', fontsize=12)
plt.ylabel('Average MSE', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加 0.05 阈值的参考线（证明即使在强噪声下，误差也没有失控）
plt.axhline(y=0.05, color='gray', linestyle=':', label='High Error Threshold (0.05)')
plt.legend()

plot_save_path = os.path.join(LOG_DIR, 'Robustness_Curve.png')
plt.savefig(plot_save_path)
print(f"🎉 鲁棒性压力测试完成！")
print(f"📊 数据已保存至: {csv_save_path}")
print(f"📈 曲线图已保存至: {plot_save_path}")