#! -*- coding: utf-8 -*-
# ============================================================================
# DDPM的U-Net模型定义 (简化版 / 基线版本)
# ============================================================================
# 这是所有DDPM变体的基线(baseline)，使用简化的U-Net结构。
#
# 【模型特点】(对比ddpm2改进版)
#   1. 时间编码: 可学习的 nn.Embedding（查表），而非固定正弦编码
#   2. 跳跃连接: Addition风格 (x = x + skip)，信息融合较简单
#   3. 残差块: Post-Norm风格 (先卷积再归一化)
#   4. 初始化: 按 1/num_layers^0.5 缩放，而非 init_scale=0
#   5. 下采样控制: 引入 min_pixel 参数避免feature map过小
#
# 原始Keras版本博客：https://kexue.fm/archives/9152
# ============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 模型配置
img_size = 128  # 如果只想快速实验，可以改为64
embedding_size = 128
channels = [1, 1, 2, 2, 4, 4]
num_layers = len(channels) * 2 + 1
blocks = 2  # 如果显存不够，可以降低为1
min_pixel = 4  # 不建议降低，显存足够可以增加到8
T = 1000


class GroupNorm(nn.Module):
    """定义GroupNorm，默认groups=32，带可学习的scale和offset
    """
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        assert num_channels % 32 == 0
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x: (B, C, H, W)
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
    """Dense包装（线性层，无bias）
    """
    def __init__(self, in_dim, out_dim, activation=None, init_scale=1):
        super().__init__()
        init_scale = max(init_scale, 1e-10)
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        # fan_avg uniform init
        fan_in = in_dim
        fan_out = out_dim
        fan_avg = (fan_in + fan_out) / 2
        limit = math.sqrt(3 * init_scale / fan_avg)
        nn.init.uniform_(self.linear.weight, -limit, limit)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation == 'swish':
            x = F.silu(x)
        return x


class Conv2dLayer(nn.Module):
    """Conv2D包装（3x3, same padding, 无bias）
    """
    def __init__(self, in_dim, out_dim, activation=None, init_scale=1):
        super().__init__()
        init_scale = max(init_scale, 1e-10)
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False)
        # fan_avg uniform init
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


class ResidualBlock(nn.Module):
    """残差block (Post-Norm风格)
    结构: x → [+t_emb] → Conv1(swish) → Conv2(swish) → [+shortcut] → GroupNorm
    
    【对比ddpm2的ResidualBlock2 (Pre-Norm风格)】
      ddpm:  x + t → conv1 → conv2 → (+skip) → norm   (Post-Norm)
      ddpm2: norm → silu → conv1 → (+t) → norm → silu → conv2 → (+skip)  (Pre-Norm)
    Pre-Norm训练更稳定，是目前主流做法。
    
    【init_scale差异】
      ddpm:  conv层用 init_scale = 1/num_layers^0.5 (缩小初始化)
      ddpm2: conv2用 init_scale = 0 (零初始化，训练初期残差块≈恒等映射)
    """
    def __init__(self, in_dim, ch, t_dim):
        super().__init__()
        out_dim = ch * embedding_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        if in_dim != out_dim:
            self.proj = DenseLayer(in_dim, out_dim)
        else:
            self.proj = None
        # 时间条件投影: 将时间embedding投影到与输入同维度，然后相加
        self.t_proj = DenseLayer(t_dim, in_dim)
        # 两层卷积，init_scale = 1/√num_layers (ddpm2中conv2使用init_scale=0)
        self.conv1 = Conv2dLayer(in_dim, out_dim, 'swish', 1 / num_layers**0.5)
        self.conv2 = Conv2dLayer(out_dim, out_dim, 'swish', 1 / num_layers**0.5)
        # Post-Norm: 归一化放在最后 (ddpm2使用Pre-Norm: 归一化放在卷积前)
        self.norm = GroupNorm(out_dim)

    def forward(self, x, t):
        # x: (B, C, H, W), t: (B, 1, D)
        if self.proj is not None:
            xi = self.proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            xi = x
        # add t
        t_emb = self.t_proj(t)  # (B, 1, in_dim)
        x = x + t_emb.permute(0, 2, 1).unsqueeze(-1)  # broadcast to (B, C, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + xi
        x = self.norm(x)
        return x


class UNet(nn.Module):
    """U-Net去噪模型 (简化版，基线)
    
    【DDPM核心思想】
    前向扩散过程: q(x_t | x_0) = N(x_t; ᾱ_t · x_0, β̄_t² · I)
    其中 ᾱ_t = ∏_{s=1}^{t} α_s,  β̄_t = √(1 - ᾱ_t²)
    即 x_t = ᾱ_t · x_0 + β̄_t · ε,  ε ~ N(0, I)
    
    U-Net的目标: 预测噪声 ε_θ(x_t, t) ≈ ε
    
    【与ddpm2的UNet2对比】
      1. 时间编码: nn.Embedding(T, D) 可学习查表 vs SinusoidalEmbedding + MLP
      2. 跳跃连接: Addition (x = x + skip) vs Concatenation (x = cat[x, skip])
         - Addition简单轻量，但信息混合受限
         - Concatenation保留更多信息，但参数量更大
      3. Decoder结构: 先上采样+skip相加，再过ResBlock
         vs 先pop skip做concat，再过ResBlock (blocks+1个block消化skip)
      4. min_pixel: 控制最小feature map尺寸，避免过度下采样
    """
    def __init__(self):
        super().__init__()
        # 【时间编码】可学习的embedding查表
        # 优点: 简单直接，可以学到任意的时间表示
        # 缺点: 无法泛化到未见过的时间步，且缺乏归纳偏置
        # ddpm2改进: 使用固定的SinusoidalEmbedding + MLP，
        #           先用正弦编码提供平滑的位置信息，再用MLP学习非线性变换
        self.t_embed = nn.Embedding(T, embedding_size)

        self.init_conv = Conv2dLayer(3, embedding_size)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        in_ch = embedding_size
        self.encoder_channels = []  # 记录每层输出通道数
        skip_pooling = 0
        cur_size = img_size

        for i, ch in enumerate(channels):
            block_list = nn.ModuleList()
            for j in range(blocks):
                block_list.append(ResidualBlock(in_ch, ch, embedding_size))
                in_ch = ch * embedding_size
                self.encoder_channels.append(in_ch)
            self.down_blocks.append(block_list)
            if cur_size > min_pixel:
                self.down_pools.append(nn.AvgPool2d(2))
                self.encoder_channels.append(in_ch)
                cur_size = cur_size // 2
            else:
                self.down_pools.append(None)
                skip_pooling += 1

        self.mid_block = ResidualBlock(in_ch, channels[-1], embedding_size)
        self.skip_pooling = skip_pooling

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i, ch in enumerate(channels[::-1]):
            if i >= skip_pooling:
                self.up_samples.append(nn.Upsample(scale_factor=2))
            else:
                self.up_samples.append(None)
            block_list = nn.ModuleList()
            for j in range(blocks):
                skip_ch = self.encoder_channels.pop()
                block_list.append(ResidualBlock(in_ch, skip_ch // embedding_size, embedding_size))
                in_ch = skip_ch
            self.up_blocks.append(block_list)

        self.final_norm = GroupNorm(in_ch)
        self.final_conv = Conv2dLayer(in_ch, 3)

    def forward(self, x, t_idx):
        # x: (B, 3, H, W), t_idx: (B, 1)
        t = self.t_embed(t_idx.squeeze(-1))  # (B, D)
        t = t.unsqueeze(1)  # (B, 1, D)

        x = self.init_conv(x)
        skips = [x]

        # Encoder
        for i, (block_list, pool) in enumerate(zip(self.down_blocks, self.down_pools)):
            for block in block_list:
                x = block(x, t)
                skips.append(x)
            if pool is not None:
                x = pool(x)
                skips.append(x)

        x = self.mid_block(x, t)
        skips.pop()  # 去掉最后多余的

        # Decoder
        # 【Addition风格跳跃连接】: x = x + skip
        # 简单相加，参数量小，但decoder无法区分encoder信息和当前层信息
        # ddpm2改进为Concatenation风格: x = cat[x, skip]，更强的信息保留
        for i, (block_list, up) in enumerate(zip(self.up_blocks, self.up_samples)):
            if up is not None:
                x = up(x)
                x = x + skips.pop()  # Addition skip connection
            for block in block_list:
                xi = skips.pop()
                x = block(x, t)
                x = x + xi  # Addition skip connection

        x = self.final_norm(x)
        x = self.final_conv(x)
        return x
