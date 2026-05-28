import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import csv
import numpy as np

# 锁定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 开启底层卷积算法自动优化（显著提速）
cudnn.benchmark = True

# 导入网络与数据集类
from Model import resnet9_with_attention 
from DataLoader import FROGdata 

# ==========================================
# 0. 全局随机种子对齐
# ==========================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

if __name__ == '__main__':
    # 严格在 main 函数内启动，保障 Windows 下的多进程 DataLoader 安全
    set_seed(42) 

    # ==========================================
    # 1. 路径配置与安全检查
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "dataset_20k", "Ere_20k.csv")
    SHG_DIR = os.path.join(BASE_DIR, "dataset_20k", "SHG")
    THG_DIR = os.path.join(BASE_DIR, "dataset_20k", "THG")

    LOG_DIR = os.path.join(BASE_DIR, "logs_Dual_20k")
    # 自动创建日志文件夹，防止训练途中因路径不存在导致崩溃
    os.makedirs(LOG_DIR, exist_ok=True) 

    WEIGHTS_SAVE_PATH = os.path.join(LOG_DIR, 'bestnet_Dual.pt')
    FINAL_WEIGHTS_PATH = os.path.join(LOG_DIR, 'final_Dual.pt')
    TRAIN_LOSS_CSV = os.path.join(LOG_DIR, 'training_loss_Dual.csv')
    VAL_LOSS_CSV = os.path.join(LOG_DIR, 'validation_loss_Dual.csv')

    # ==========================================
    # 2. 超参数配置
    # ==========================================
    batch_size = 32     
    num_epochs = 150 
    learning_rate = 1e-3
    
    early_stopping_patience = 35  # 延长早停容忍度，给学习率衰减留下挽救空间
    epochs_no_improve = 0         

    # ==========================================
    # 3. 加载数据集 (多核加速)
    # ==========================================
    print("⏳ 正在加载 20000 组双通道数据集...")
    train_dataset = FROGdata(csv_file=CSV_PATH, img_dir_shg=SHG_DIR, img_dir_thg=THG_DIR, train=True)
    val_dataset = FROGdata(csv_file=CSV_PATH, img_dir_shg=SHG_DIR, img_dir_thg=THG_DIR, train=False)

    # 开启 num_workers=4 压榨多核 CPU 并发读取，极大缓解 I/O 瓶颈
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"✅ 训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)}")

    # ==========================================
    # 4. 初始化模型 (2 通道: SHG灰度 + THG灰度)
    # ==========================================
    model = resnet9_with_attention(2, 64, 512).to(device) 

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器：连续 15 个 epoch 不降则减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    # ==========================================
    # 5. 训练循环 (含实时日志写入)
    # ==========================================
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    print(f"\n🚀 开始 双通道 (SHG+THG) 基准模型训练 (Batch: {batch_size})...")

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_train_loss = 0.0
        
        # DataLoader 传出的 targets 已经是 512 维的一维张量
        for i, (inputs, targets, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
            if (i + 1) % 100 == 0:
                print(f"   Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4e}")
            
        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # --- 验证阶段 ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)
        
        print(f"--- Epoch [{epoch+1:03d}] 完成! Train Loss: {epoch_train_loss:.4e} | Val Loss: {epoch_val_loss:.4e}")
        
        # ====== 🌟 核心防崩溃机制：实时写入 CSV ======
        with open(TRAIN_LOSS_CSV, 'w', newline='') as f:
            csv.writer(f).writerows([["Epoch", "Loss"]] + [[idx+1, l] for idx, l in enumerate(train_losses)])
        with open(VAL_LOSS_CSV, 'w', newline='') as f:
            csv.writer(f).writerows([["Epoch", "Loss"]] + [[idx+1, l] for idx, l in enumerate(val_losses)])
        
        # --- 早停判断与权重保存 ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0  
            torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
            print(f"   => ✅ 最优权重已保存至: {WEIGHTS_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            print(f"   => ⚠️ 验证集未提升 ({epochs_no_improve}/{early_stopping_patience})")
            if epochs_no_improve >= early_stopping_patience:
                print(f"\n🛑 触发早停！双通道模型已收敛。")
                break

    # ==========================================
    # 6. 保存最终轮次权重
    # ==========================================
    torch.save(model.state_dict(), FINAL_WEIGHTS_PATH)
    print(f"\n💾 最终 Epoch 权重已额外保存至: {FINAL_WEIGHTS_PATH}")
    print("🎉 双通道基准模型训练圆满结束！")
