import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

seeding = 42

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
        # 1. Image Loading and Preprocessing
        raw_id = str(self.data.iloc[index, 0]).zfill(5)
        img_path_shg = os.path.join(self.img_dir_shg, f"shg_{raw_id}.png")
        img_path_thg = os.path.join(self.img_dir_thg, f"thg_{raw_id}.png")

        try:
            image_shg = Image.open(img_path_shg).convert('RGB')
            image_thg = Image.open(img_path_thg).convert('RGB')
        except Exception as e:
            print(f"⚠️ 无法读取图像: {img_path_shg}, 跳过. 错误: {e}")
            return self.__getitem__((index + 1) % len(self.data))

        if self.transforms:
            image_shg = self.transforms(image_shg)
            image_thg = self.transforms(image_thg)

        dual_channel_image = torch.cat((image_shg, image_thg), dim=0)

        # 2. Extract Real and Imaginary Data from CSV
        row = self.data.iloc[index]
        
        real = row[[f'real_{i}' for i in range(256)]].values.astype(np.float64)
        imag = row[[f'imag_{i}' for i in range(256)]].values.astype(np.float64)
        
        complex_E = real + 1j * imag
        
        # 3. Dynamic Phase Anchoring and Centering
        abs_E = np.abs(complex_E)
        peak_idx = np.argmax(abs_E)

        # Center the peak to index 128
        shift_amount = 128 - peak_idx
        complex_E = np.roll(complex_E, shift_amount)

        # Anchor phase at index 128
        global_phase = np.angle(complex_E[128])
        anchored_E = complex_E * np.exp(-1j * global_phase)

        # 4. Repackage into Target Tensor
        final_points = np.concatenate([np.real(anchored_E), np.imag(anchored_E)])
        points = torch.tensor(final_points, dtype=torch.float32)

        return dual_channel_image, points, raw_id