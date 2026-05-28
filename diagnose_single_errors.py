import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from DataLoader import FROGdata
from Model import resnet9_with_attention
from sklearn.metrics import mean_squared_error
from FROG_Utils import get_phys_props, get_time_axis 

# ==========================================
# 1. 核心配置区
# ==========================================
# 设定诊断阈值：SHG-only 翻车会非常多，0.05 是个很好的观察起点
ERROR_THRESHOLD = 0.05
MAX_PLOTS = 66            

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset_20k')
SHG_DIR = os.path.join(DATA_DIR, 'SHG')
THG_DIR = os.path.join(DATA_DIR, 'THG')
CSV_PATH = os.path.join(DATA_DIR, 'Ere_20k.csv')

# 【修改点 1】：指向单通道的日志和权重
LOG_DIR = os.path.join(BASE_DIR, 'logs_SHG_only_20k')
WEIGHTS_PATH = os.path.join(LOG_DIR, 'bestnet_SHG_only.pt')

# 定义坏样本专门的存放文件夹 (隔离存放)
BAD_SAMPLES_DIR = os.path.join(LOG_DIR, 'BadSamples_Gallery_SHG')
os.makedirs(BAD_SAMPLES_DIR, exist_ok=True) 

# ==========================================
# 2. 加载模型与数据集
# ==========================================
print(f"正在准备单通道 (SHG-Only) 诊断工具... (阈值: MSE > {ERROR_THRESHOLD})")
model = resnet9_with_attention(6, 64, 512).to(device)
# 加载单通道权重
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
model.eval()

# 使用测试集
test_set = FROGdata(CSV_PATH, SHG_DIR, THG_DIR, train=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False) 

# ==========================================
# 3. 诊断主循环
# ==========================================
plots_generated = 0
found_bad_samples = 0

print("开始扫描测试集寻找 SHG-Only 翻车样本...")

with torch.no_grad():
    for index, (inputs, targets, filename) in enumerate(test_loader):
        # 【修改点 2】：强行掩膜抹除 THG 信号！！！
        inputs[:, 3:6, :, :] = 0.0
        
        inputs_device = inputs.to(device)
        targets_device = targets.to(device)
        
        # 模型预测
        outputs = model(inputs_device)
        
        # 提取数据用于计算
        y_true = targets_device.cpu().numpy()[0]
        y_pred = outputs.cpu().numpy()[0]
        
        # 计算当前样本的 MSE
        mse = mean_squared_error(y_true, y_pred)
        
        # 判断是否超过了“坏样本”阈值
        if mse > ERROR_THRESHOLD:
            found_bad_samples += 1
            print(f"⚠️ 发现坏样本! 文件名: {filename[0]}, MSE: {mse:.6f}")
            
            if plots_generated < MAX_PLOTS:
                # --- 开始绘制诊断面板 ---
                fig = plt.figure(figsize=(15, 10))
                fig.suptitle(f'SHG-Only Failure Mode - Sample: {filename[0]} | MSE: {mse:.6f}', fontsize=16, fontweight='bold')
                
                # 获取图像数据用于显示 (因为 THG 被抹零了，所以显示全黑是正常的)
                shg_img = inputs[0, 0, :, :].numpy()
                thg_img = inputs[0, 3, :, :].numpy()
                
                # 子图 1: FROG 迹线图
                ax1 = plt.subplot(2, 2, 1)
                ax1.imshow(shg_img, cmap='jet', aspect='auto')
                ax1.set_title("Input: SHG Trace")
                ax1.axis('off')
                
                ax2 = plt.subplot(2, 2, 2)
                ax2.imshow(thg_img, cmap='gray', aspect='auto', vmin=0, vmax=1)
                ax2.set_title("Input: THG Trace (Masked/Zeroed)")
                ax2.axis('off')
                
                # 物理量提取
                A_true, P_true = get_phys_props(y_true[:256], y_true[256:])
                A_pred, P_pred = get_phys_props(y_pred[:256], y_pred[256:])
                time_axis = get_time_axis()
                
                # 子图 3: 电场绝对振幅 |E| 对比
                ax3 = plt.subplot(2, 2, 3)
                ax3.plot(time_axis, A_true, label='Ground Truth |E|', color='black', linewidth=3)
                ax3.plot(time_axis, A_pred, label='SHG-Only Predict |E|', color='blue', linestyle='-.', linewidth=2)
                ax3.set_title('Electric Field: Amplitude |E|')
                ax3.set_xlabel('Time (fs)')
                ax3.set_ylabel('Amplitude (a.u.)')
                ax3.legend()
                ax3.grid(True, linestyle=':', alpha=0.6)
                
                # 子图 4: 电场连续相位对比
                ax4 = plt.subplot(2, 2, 4)
                ax4.plot(time_axis, P_true, label='Ground Truth Phase', color='black', linewidth=4, alpha=0.2)
                ax4.plot(time_axis, P_pred, label='SHG-Only Predict Phase', color='cyan', linestyle='-.', linewidth=2)
                ax4.set_title('Electric Field: Phase')
                ax4.set_xlabel('Time (fs)')
                ax4.set_ylabel('Phase (rad)')
                
                # 动态 Y 轴
                valid_p = P_true[~np.isnan(P_true)]
                if len(valid_p) > 0:
                    ax4.set_ylim([np.min(valid_p)-2, np.max(valid_p)+2])
                
                ax4.legend()
                ax4.grid(True, linestyle=':', alpha=0.6)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                # 【修改点 3】：保存到 SHG 专属文件夹
                save_name = os.path.join(BAD_SAMPLES_DIR, f'BadSample_SHG_MSE_{mse:.5f}_{filename[0]}.png')
                plt.savefig(save_name)
                print(f"   📸 诊断图已保存: {save_name}")
                
                plt.close(fig) 
                plots_generated += 1

print("\n" + "="*40)
print(f"扫描完毕。单通道网络共找到 {found_bad_samples} 个 MSE > {ERROR_THRESHOLD} 的翻车样本")