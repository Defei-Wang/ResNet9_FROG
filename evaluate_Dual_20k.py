import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader

from Model import resnet9_with_attention 
from DataLoader import FROGdata 
from FROG_Utils import get_phys_props, get_time_axis 

# ==========================================
# 0. 锁定随机种子 
# ==========================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    set_seed(42)

    # ==========================================
    # 1. 路径配置
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "dataset_20k")
    CSV_PATH = os.path.join(DATA_DIR, "Ere_20k.csv")
    SHG_DIR = os.path.join(DATA_DIR, "SHG")
    THG_DIR = os.path.join(DATA_DIR, "THG")

    LOG_DIR = os.path.join(BASE_DIR, 'logs_Dual_20k')
    WEIGHTS_PATH = os.path.join(LOG_DIR, 'bestnet_Dual.pt')
    SAVE_FIG_DIR = os.path.join(LOG_DIR, 'evaluation_plots')
    os.makedirs(SAVE_FIG_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 2. 初始化模型
    # ==========================================
    model = resnet9_with_attention(2, 64, 512).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    # ==========================================
    # 3. 加载测试集数据 (暴力提速版)
    # ==========================================
    print("⏳ 正在加载测试集数据...")
    test_dataset = FROGdata(csv_file=CSV_PATH, img_dir_shg=SHG_DIR, img_dir_thg=THG_DIR, train=False)
    # 评估阶段无梯度要求，调大 Batch Size，拉满 CPU 多核读取
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # ==========================================
    # 3.5 测算单次前向推理延迟 (FPS Benchmark)
    # ==========================================
    print("⏱️ 正在进行 CPU/GPU 前向推理测速基准测试...")
    dummy_input, _, _ = test_dataset[0]
    dummy_input = dummy_input.unsqueeze(0).to(device)

    # 预热 (Warm-up)，消除 CUDA/CPU 冷启动耗时
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 连续执行 100 次以获取精准的平均耗时
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    end_time = time.perf_counter()

    avg_time_ms = ((end_time - start_time) / 100) * 1000
    fps = 1000 / avg_time_ms
    print(f"⚡ 单张图像推理延迟: {avg_time_ms:.2f} ms")
    print(f"⚡ 理论重构吞吐量 (FPS): {fps:.2f} 帧/秒\n")

    # ==========================================
    # 4. 计算全局平均 MSE (张量化无 for 循环版)
    # ==========================================
    # reduction='none' 使得 loss 保持 batch 维度，方便一次性求平均
    criterion = torch.nn.MSELoss(reduction='none')
    all_mses = []

    print("📊 正在计算双通道模型全局测试误差 (MSE)...")
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs = inputs.to(device) 
            targets = targets.to(device)
            outputs = model(inputs)
            
            # 张量化计算：直接得到 [Batch_Size] 长度的损失向量
            batch_losses = criterion(outputs, targets).mean(dim=1)
            all_mses.extend(batch_losses.cpu().tolist())

    mean_mse = np.mean(all_mses)
    print("="*40)
    print("🏆 双通道基准模型 测试集评估报告")
    print("="*40)
    print(f"评估样本数: {len(all_mses)}")
    print(f"平均均方误差 (Average MSE): {mean_mse:.8e}")
    print("="*40 + "\n")

    mse_df = pd.DataFrame({"Sample_Index": range(len(all_mses)), "MSE": all_mses})
    mse_df.to_csv(os.path.join(LOG_DIR, 'test_mse_distribution_Dual.csv'), index=False)

    # ==========================================
    # 5. 生成物理对比图 (调用 Utils)
    # ==========================================
    num_random_plots = 50  
    sample_indices = random.sample(range(len(test_dataset)), num_random_plots)
    time_axis = get_time_axis() 

    print(f"🎨 随机选择了 {num_random_plots} 个样本，开始生成物理对比图...")

    for idx in sample_indices:
        inputs, targets, filename = test_dataset[idx]
        inputs = inputs.unsqueeze(0).to(device) 
        
        with torch.no_grad():
            outputs = model(inputs)
        
        pred_array = outputs.cpu().squeeze().numpy()
        true_array = targets.cpu().squeeze().numpy()
        
        # 直接调用工具箱，自动执行相位解包与 NaN 掩膜防断崖
        A_true, P_true = get_phys_props(true_array[:256], true_array[256:])
        A_pred, P_pred = get_phys_props(pred_array[:256], pred_array[256:])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150)
        
        # 子图 1: 绝对电场 |E|
        ax1.plot(time_axis, A_true, 'k-', label='Ground Truth |E|', linewidth=3)
        ax1.plot(time_axis, A_pred, 'r--', label='Dual-Channel Predict', linewidth=2)
        ax1.set_title(f"Sample ID: {filename} - Absolute Electric Field (Dual-Channel)", fontsize=14)
        ax1.set_ylabel("Amplitude |E| (a.u.)", fontsize=12)
        ax1.set_xlim([-250, 250])
        ax1.set_ylim([-0.05, 1.1])
        ax1.grid(True, linestyle=':', alpha=0.7)
        ax1.legend(loc='upper right')
        
        # 子图 2: 时域相位 (已物理掩膜)
        ax2.plot(time_axis, P_true, 'k-', label='Ground Truth Phase', linewidth=3, alpha=0.3)
        ax2.plot(time_axis, P_pred, 'r--', label='Predicted Phase', linewidth=2)
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
        save_path = os.path.join(SAVE_FIG_DIR, f"Dual_Phys_{filename}.png")
        plt.savefig(save_path)
        plt.close()

    print("🎉 双通道评估与可视化圆满完成！")
