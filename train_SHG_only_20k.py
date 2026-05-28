import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import csv
import numpy as np

from Model import resnet9_with_attention 
from DataLoader import FROGdata 

# ==========================================
# 0. 全局随机种子对齐 (保证公平对决)
# ==========================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    set_seed(42) 
    
    # ==========================================
    # 1. 路径配置
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "dataset_20k", "Ere_20k.csv")
    SHG_DIR = os.path.join(BASE_DIR, "dataset_20k", "SHG")
    THG_DIR = os.path.join(BASE_DIR, "dataset_20k", "THG")
    LOG_DIR = os.path.join(BASE_DIR, "logs_SHG_only_20k") 
    os.makedirs(LOG_DIR, exist_ok=True)

    WEIGHTS_SAVE_PATH = os.path.join(LOG_DIR, 'bestnet_SHG_only.pt')
    FINAL_WEIGHTS_PATH = os.path.join(LOG_DIR, 'final_SHG_only.pt')
    TRAIN_LOSS_CSV = os.path.join(LOG_DIR, 'training_loss_SHG.csv')
    VAL_LOSS_CSV = os.path.join(LOG_DIR, 'validation_loss_SHG.csv')

    # ==========================================
    # 2. 超参数对齐 (与 Dual 严格一致)
    # ==========================================
    batch_size = 32 
    num_epochs = 150 
    learning_rate = 1e-3
    
    early_stopping_patience = 15  # 统一改为 15
    epochs_no_improve = 0         

    # ==========================================
    # 3. 加载数据集
    # ==========================================
    print("⏳ 正在加载 20000 组数据集索引...")
    train_dataset = FROGdata(csv_file=CSV_PATH, img_dir_shg=SHG_DIR, img_dir_thg=THG_DIR, train=True)
    val_dataset = FROGdata(csv_file=CSV_PATH, img_dir_shg=SHG_DIR, img_dir_thg=THG_DIR, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print(f"✅ 训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)}")

    # ==========================================
    # 4. 初始化模型
    # ==========================================
    model = resnet9_with_attention(2, 64, 512).to(device) 

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    # ==========================================
    # 🌟 5. 断点续训核心逻辑 (防崩溃机制)
    # ==========================================
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(WEIGHTS_SAVE_PATH) and os.path.exists(TRAIN_LOSS_CSV):
        print("\n⚡ 检测到历史训练记录，启动断点续训！")
        # 加载历史最优权重
        model.load_state_dict(torch.load(WEIGHTS_SAVE_PATH, map_location=device, weights_only=True))
        
        # 尝试从 CSV 解析已完成的 Epoch
        try:
            with open(TRAIN_LOSS_CSV, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_epoch = int(lines[-1].split(',')[0])
                    start_epoch = last_epoch
            print(f"   => 成功读取历史进度，将从第 {start_epoch + 1} 轮继续训练。")
            
            # 恢复 best_val_loss 防止优秀权重被覆盖
            if os.path.exists(VAL_LOSS_CSV):
                with open(VAL_LOSS_CSV, 'r') as f:
                    reader = csv.reader(f)
                    next(reader) # 跳过表头
                    losses = [float(row[1]) for row in reader if len(row) > 1]
                    if losses:
                        best_val_loss = min(losses)
                        print(f"   => 成功恢复历史最优 Val Loss: {best_val_loss:.4e}")
        except Exception as e:
            print(f"   => ⚠️ 解析日志失败，将重新开始训练。错误: {e}")
            start_epoch = 0

    # ==========================================
    # 6. 训练循环
    # ==========================================
    print(f"\n🚀 开始 单通道(SHG-Only)消融训练 (Batch: {batch_size})...")

    for epoch in range(start_epoch, num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_train_loss = 0.0
        for i, (inputs, targets, _) in enumerate(train_loader):
            # 【核心物理约束】：将 THG 通道（索引3,4,5）强行抹零
            inputs[:, 3:6, :, :] = 0.0  
            
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
        
        # --- 验证阶段 ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs[:, 3:6, :, :] = 0.0  
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        scheduler.step(epoch_val_loss)
        
        print(f"--- Epoch [{epoch+1:03d}] 完成! Train Loss: {epoch_train_loss:.4e} | Val Loss: {epoch_val_loss:.4e}")
        
        # ====== 🌟 实时日志追加写入 ======
        file_mode = 'w' if (epoch == 0 and start_epoch == 0) else 'a'
        
        with open(TRAIN_LOSS_CSV, file_mode, newline='') as f:
            writer = csv.writer(f)
            if file_mode == 'w': writer.writerow(["Epoch", "Loss"])
            writer.writerow([epoch + 1, epoch_train_loss])
            
        with open(VAL_LOSS_CSV, file_mode, newline='') as f:
            writer = csv.writer(f)
            if file_mode == 'w': writer.writerow(["Epoch", "Loss"])
            writer.writerow([epoch + 1, epoch_val_loss])
        
        # --- 早停与最优状态管理 ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0  
            torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
            print(f"   => ✅ 最优权重已保存至: {WEIGHTS_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            print(f"   => ⚠️ 验证集未提升 ({epochs_no_improve}/{early_stopping_patience})")
            if epochs_no_improve >= early_stopping_patience:
                print(f"\n🛑 触发早停。模型受制于 SHG 物理模糊性已达收敛极限。")
                break

    # 保存最终 Epoch 权重
    torch.save(model.state_dict(), FINAL_WEIGHTS_PATH)
    print(f"\n💾 最终 Epoch 权重已额外保存至: {FINAL_WEIGHTS_PATH}")
    print("🎉 单通道模型训练圆满结束！")
