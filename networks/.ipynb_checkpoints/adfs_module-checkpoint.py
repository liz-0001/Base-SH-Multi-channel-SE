import torch
import torch.nn as nn
import torch.nn.functional as F

class SHC_ADFS(nn.Module):
    def __init__(self, num_sh_channels, num_freq_bins):
        """
        Args:
            num_sh_channels: 球谐通道数，通常是 (L+1)^2
            num_freq_bins: 频率点数，通常是 n_fft // 2 + 1
        """
        super(SHC_ADFS, self).__init__()
        
        # --- 维度 A：频率自适应选择 (Frequency-dependent Selection) ---
        # 作用：学习哪些频段对当前的语音信息最重要
        self.freq_net = nn.Sequential(
            nn.Linear(num_freq_bins, num_freq_bins // 4),
            nn.ReLU(),
            nn.Linear(num_freq_bins // 4, num_freq_bins),
            nn.Sigmoid()
        )
        
        # --- 维度 B：空间层级权重 (Layer-wise Weights) ---
        # 作用：学习不同阶数的球谐系数（0阶、1阶等）的重要程度
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 将时间维度和频率维度压缩，只剩下通道信息
            nn.Conv2d(num_sh_channels, num_sh_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
            """
            输入 x 的形状: (Batch, Channels, Time, Freq) 
            例如: (1, 50, 126, 257)
            """
            # 如果输入是复数，取模值计算权重
            if torch.is_complex(x):
                x_input = x.abs()
            else:
                x_input = x

            # --- 1. 计算频率维度的权重 ---
            # 注意：这里 dim 改为 2，表示对 Time 维度取平均，保留 Freq 维度
            # 结果形状从 (B, C, T, F) 变为 (B, C, F)
            freq_info = torch.mean(x_input, dim=2) 
            
            # 经过全连接层得到频率权重 (B, C, F)
            f_weights = self.freq_net(freq_info)    
            
            # 将权重作用于原始数据。我们需要把 (B, C, F) 扩充为 (B, C, 1, F)
            # 这样才能和原始的 (B, C, T, F) 进行广播相乘
            x = x * f_weights.unsqueeze(2) 
            
            # --- 2. 计算通道（球谐层）维度的权重 ---
            # channel_gate 使用的是 AdaptiveAvgPool2d(1)，它会自动处理 T 和 F 维度
            c_weights = self.channel_gate(x_input) # 得到 (B, C, 1, 1)
            x = x * c_weights
            
            return x, c_weights