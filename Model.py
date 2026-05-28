import torch
import torch.nn as nn

# ==========================================
# 1. 单通道基准网络 (纯净物理映射基线)
# ==========================================
class resnet9(nn.Module):
    def __init__(self, in_channels, feature_size, output_size):
        super().__init__()
        self.conv1 = self._block(in_channels, feature_size, pool=False)
        self.conv2 = self._block(feature_size, feature_size * 2, pool=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.res_block = nn.Sequential(
            self._block(feature_size * 2, feature_size * 2, pool=False),
            self._block(feature_size * 2, feature_size * 2, pool=False)
        )
        self.shortcut = nn.Conv2d(feature_size * 2, feature_size * 2, kernel_size=1, stride=1, padding=0)

        self.conv3 = self._block(feature_size * 2, feature_size * 4, pool=True)
        self.conv4 = self._block(feature_size * 4, feature_size * 8, pool=True)

        self.res_block2 = nn.Sequential(
            self._block(feature_size * 8, feature_size * 8, pool=False),
            self._block(feature_size * 8, feature_size * 8, pool=False)
        )
        self.shortcut2 = nn.Conv2d(feature_size * 8, feature_size * 8, kernel_size=1, stride=1, padding=0)

        # 剥离 Dropout 和 ReLU，对齐 4x4 空间拓扑 (输出 512 维)
        self.regression = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(feature_size * 8 * 16, output_size) 
        )

    def _block(self, in_channels, feature_size, pool=False):
        # 移除了 Dropout，确保 MSE 拟合精度
        block_layers = [nn.Conv2d(in_channels, feature_size, 3, 2 if pool else 1, 1),
                        nn.BatchNorm2d(feature_size),
                        nn.ReLU(inplace=True)]
        return nn.Sequential(*block_layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.res_block(x) + self.shortcut(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res_block2(x) + self.shortcut2(x)
        x = self.regression(x)
        return x


# ==========================================
# 2. 空间注意力机制 (独立权重提取)
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return torch.sigmoid(attn)


# ==========================================
# 3. 双通道注意力前馈网络 (终极核心架构)
# ==========================================
class resnet9_with_attention(nn.Module):
    def __init__(self, in_channels, feature_size, num_classes):
        super().__init__()

        self.attention = SpatialAttention()

        # 注意：这里的 self.conv1 接收的是 combined_input，其通道数会等于 in_channels 
        self.conv1 = self._block(in_channels, feature_size)
        self.conv2 = self._block(feature_size, feature_size * 2, pool=True)

        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.res_block = nn.Sequential(self._block(feature_size * 2, feature_size * 2),
                                       self._block(feature_size * 2, feature_size * 2))
        self.shortcut1 = nn.Conv2d(feature_size * 2, feature_size * 2, 1, 1, 0)

        self.conv3 = self._block(feature_size * 2, feature_size * 4, pool=True)
        self.conv4 = self._block(feature_size * 4, feature_size * 8, pool=True)

        self.res_block_2 = nn.Sequential(self._block(feature_size * 8, feature_size * 8),
                                         self._block(feature_size * 8, feature_size * 8))
        self.shortcut2 = nn.Conv2d(feature_size * 8, feature_size * 8, 1, 1, 0)

        # 严格保留 4x4 (16个分区) 时频空间特征矩阵
        self.regression = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), 
            nn.Flatten(),
            nn.Linear(feature_size * 8 * 16, num_classes)
        )

    def _block(self, in_channels, feature_size, pool=False):
        block_layers = [nn.Conv2d(in_channels, feature_size, 3, 2 if pool else 1, 1),
                        nn.BatchNorm2d(feature_size),
                        nn.ReLU(inplace=True)]
        return nn.Sequential(*block_layers)

    def forward(self, img):
        # 1. 物理层解耦：精准切分张量第0通道(SHG灰度)和第1通道(THG灰度)
        img1 = img[:, 0:1, :, :]  
        img2 = img[:, 1:2, :, :]  
        
        # 2. 并行独立空间注意力加权 (互不干扰)
        attn_map_1 = self.attention(img1)
        img1_attended = img1 * attn_map_1
        
        attn_map_2 = self.attention(img2)
        img2_attended = img2 * attn_map_2
        
        # 3. 级联进入主干网络 [B, 2, H, W]
        combined_input = torch.cat((img1_attended, img2_attended), dim=1)

        # 4. 主干网络特征凝聚
        x = self.conv1(combined_input)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.res_block(x) + self.shortcut1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res_block_2(x) + self.shortcut2(x)

        # 5. 回归至欧氏空间实/虚部向量
        out = self.regression(x)
        return out

if __name__ == "__main__":
    pass
