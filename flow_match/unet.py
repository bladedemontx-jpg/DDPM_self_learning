#! -*- coding: utf-8 -*-
# ============================================================================
# Flow Matching的U-Net模型定义
# ============================================================================
# 【与DDPM的根本区别】
#   DDPM: 离散时间 t ∈ {0,1,...,T-1}，预测噪声 ε_θ(x_t, t)
#   Flow Matching: 连续时间 t ∈ [0,1]，预测速度场 v_θ(x_t, t)
#
# 【Flow Matching框架】
#   插值路径 (Conditional OT): x_t = (1-t)·x_0 + t·ε,  ε~N(0,I)
#   速度场目标: v = dx_t/dt = ε - x_0 (线性路径的导数)
#   ODE采样: dx/dt = v_θ(x, t),  从 t=1 积分到 t=0
#
# 【模型改进】相对ddpm2/UNet2:
#   1. SinusoidalTimeEmbedding: 支持连续时间输入（而非离散查表）
#      - t乘以1000后编码，增加编码分辨率
#   2. forward(x, t): t是(B,)连续向量而非离散索引
#   3. 其余UNet结构与ddpm2基本相同 (Pre-Norm, Concatenate风格)
#
# 参考: Flow Matching for Generative Modeling (Lipman et al., 2022)
# ============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 模型配置
img_size = 128
embedding_size = 128
channels = [1, 1, 2, 2, 4, 4]
blocks = 2


class GroupNorm(nn.Module):
    """定义GroupNorm，默认groups=32
    """
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        assert num_channels % 32 == 0
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        B, C, H, W = x.shape
        groups = 32
        x = x.view(B, groups, C // groups, H, W)
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        var = x.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
        x = (x - mean) / (var + 1e-6).sqrt()
        x = x.view(B, C, H, W)
        x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, init_scale=1):
        super().__init__()
        init_scale = max(init_scale, 1e-10)
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        fan_avg = (in_dim + out_dim) / 2
        limit = math.sqrt(3 * init_scale / fan_avg)
        nn.init.uniform_(self.linear.weight, -limit, limit)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation == 'swish':
            x = F.silu(x)
        return x


class Conv2dLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, init_scale=1):
        super().__init__()
        init_scale = max(init_scale, 1e-10)
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False)
        fan_in = in_dim * 9
        fan_out = out_dim * 9
        fan_avg = (fan_in + fan_out) / 2
        limit = math.sqrt(3 * init_scale / fan_avg)
        nn.init.uniform_(self.conv.weight, -limit, limit)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation == 'swish':
            x = F.silu(x)
        return x


class SinusoidalTimeEmbedding(nn.Module):
    """【Flow Matching特有】连续时间的正弦位置编码
    vs DDPM的SinusoidalEmbedding(查表):
      - DDPM: 离散索引 → 查预计算的编码表
      - Flow Matching: 连续值 t · 1000 → 实时计算正弦编码
        乘以1000是为了让t∈[0,1]的编码分辨率与DDPM的t∈[0,1000]可比
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: (B,) 连续时间 ∈ [0, 1]"""
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None] * emb[None, :] * 1000  # 乘以1000使得编码更丰富
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """残差block (Pre Norm风格)
    """
    def __init__(self, in_dim, ch, t_dim):
        super().__init__()
        out_dim = ch * embedding_size
        if in_dim != out_dim:
            self.proj = DenseLayer(in_dim, out_dim)
        else:
            self.proj = None
        self.norm1 = GroupNorm(in_dim)
        self.conv1 = Conv2dLayer(in_dim, out_dim)
        self.t_proj = DenseLayer(t_dim, out_dim)
        self.norm2 = GroupNorm(out_dim)
        self.conv2 = Conv2dLayer(out_dim, out_dim, init_scale=0)

    def forward(self, x, t):
        if self.proj is not None:
            xi = self.proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            xi = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        t_emb = self.t_proj(t)  # (B, 1, out_dim)
        x = x + t_emb.permute(0, 2, 1).unsqueeze(-1)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = x + xi
        return x


class FlowMatchUNet(nn.Module):
    """【Flow Matching】U-Net速度场模型
    输入: x_t (B,3,H,W) + 连续时间 t (B,)
    输出: 速度场 v_θ(x_t, t) ≈ ε - x_0
    
    结构与ddpm2/UNet2相同: Pre-Norm + Concatenate跳跃连接
    差异仅在时间编码:
      ddpm2: SinusoidalEmbedding(离散查表) + MLP
      FM:    SinusoidalTimeEmbedding(连续计算) + MLP
    """
    def __init__(self):
        super().__init__()
        t_dim = embedding_size * 4
        self.t_sinusoidal = SinusoidalTimeEmbedding(embedding_size)
        self.t_dense1 = DenseLayer(embedding_size, t_dim, 'swish')
        self.t_dense2 = DenseLayer(t_dim, t_dim, 'swish')

        self.init_conv = Conv2dLayer(3, embedding_size)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        in_ch = embedding_size
        self.encoder_channels = []

        for i, ch in enumerate(channels):
            block_list = nn.ModuleList()
            for j in range(blocks):
                block_list.append(ResidualBlock(in_ch, ch, t_dim))
                in_ch = ch * embedding_size
                self.encoder_channels.append(in_ch)
            self.down_blocks.append(block_list)
            if i != len(channels) - 1:
                self.down_pools.append(nn.AvgPool2d(2))
                self.encoder_channels.append(in_ch)
            else:
                self.down_pools.append(None)

        self.mid_block = ResidualBlock(in_ch, channels[-1], t_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        enc_chs = list(self.encoder_channels)

        for i, ch in enumerate(channels[::-1]):
            block_list = nn.ModuleList()
            for j in range(blocks + 1):
                skip_ch = enc_chs.pop()
                block_list.append(ResidualBlock(in_ch + skip_ch, ch, t_dim))
                in_ch = ch * embedding_size
            self.up_blocks.append(block_list)
            if i != len(channels) - 1:
                self.up_samples.append(nn.Upsample(scale_factor=2))
            else:
                self.up_samples.append(None)

        self.final_norm = GroupNorm(in_ch)
        self.final_conv = Conv2dLayer(in_ch, 3)

    def forward(self, x, t):
        """
        x: (B, 3, H, W)
        t: (B,) 连续时间
        """
        t_emb = self.t_sinusoidal(t)
        t_emb = self.t_dense1(t_emb)
        t_emb = self.t_dense2(t_emb)
        t_emb = t_emb.unsqueeze(1)  # (B, 1, D)

        x = self.init_conv(x)
        skips = [x]

        for block_list, pool in zip(self.down_blocks, self.down_pools):
            for block in block_list:
                x = block(x, t_emb)
                skips.append(x)
            if pool is not None:
                x = pool(x)
                skips.append(x)

        x = self.mid_block(x, t_emb)

        for block_list, up in zip(self.up_blocks, self.up_samples):
            for block in block_list:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, t_emb)
            if up is not None:
                x = up(x)

        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        return x
