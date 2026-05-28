import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

seeding = 42

# 图像预处理：灰度图输入后会自动变为 [1, H, W] 的张量
transforms_config = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class FROGdata(Dataset):
    def __init__(self, csv_file, img_dir_shg, img_dir_thg, transforms=transforms_config, train=True, val_size=0.2):
        self.data = pd.read_csv(csv_file, header=0) 
        self.img_dir_shg = img_dir_shg
        self.img_dir_thg = img_dir_thg
        self.transforms = transforms

        train_indices, val_indices = train_test_split(
            range(len(self.data)), test_size=val_size, random_state=seeding
        )

        if train:
            self.data = self.data.iloc[train_indices]
        else:
            self.data = self.data.iloc[val_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # ==========================================
        # 1. 图像加载与灰度降维 (彻底剔除 RGB 冗余)
        # ==========================================
        raw_id = str(self.data.iloc[index, 0]).zfill(5)
        img_path_shg = os.path.join(self.img_dir_shg, f"shg_{raw_id}.png")
        img_path_thg = os.path.join(self.img_dir_thg, f"thg_{raw_id}.png")

        try:
            # 强制转换为 'L' (8-bit 单通道灰度)
            image_shg = Image.open(img_path_shg).convert('L')
            image_thg = Image.open(img_path_thg).convert('L')
        except Exception as e:
            print(f"⚠️ 无法读取图像: {img_path_shg}, 跳过. 错误: {e}")
            return self.__getitem__((index + 1) % len(self.data))

        if self.transforms:
            image_shg = self.transforms(image_shg)
            image_thg = self.transforms(image_thg)

        # 拼接后张量维度为 [2, 256, 256]
        dual_channel_image = torch.cat((image_shg, image_thg), dim=0)

        # ==========================================
        # 2. 提取物理 CSV 数据
        # ==========================================
        row = self.data.iloc[index]
        
        real = row[[f'real_{i}' for i in range(256)]].values.astype(np.float64)
        imag = row[[f'imag_{i}' for i in range(256)]].values.astype(np.float64)
        
        complex_E = real + 1j * imag
        
        # ==========================================
        # 3. 刚性时域平移与相位锚定
        # ==========================================
        abs_E = np.abs(complex_E)
        peak_idx = np.argmax(abs_E)

        # 消除时间平移歧义：对齐到索引 128
        shift_amount = 128 - peak_idx
        complex_E = np.roll(complex_E, shift_amount)

        # 消除载波相位歧义：中心点虚部强制归零
        global_phase = np.angle(complex_E[128])
        anchored_E = complex_E * np.exp(-1j * global_phase)

        # ==========================================
        # 4. 封装回归目标张量
        # ==========================================
        # 自动拼接为 512 维的一维向量 (实部256 + 虚部256)
        final_points = np.concatenate([np.real(anchored_E), np.imag(anchored_E)])
        points = torch.tensor(final_points, dtype=torch.float32)

        return dual_channel_image, points, raw_id
