#! -*- coding: utf-8 -*-
# ============================================================================
# DDPM2的U-Net模型定义 (Concatenate风格 / 改进版)
# ============================================================================
# 【相对ddpm基线的改进项】
#   1. 时间编码: 固定SinusoidalEmbedding + MLP (更平滑的表示)
#      - ddpm用nn.Embedding查表，缺乏时间步之间的平滑性
#      - SinusoidalEmbedding用不同频率正弦波编码，时间步越相近表示越相似
#   2. 跳跃连接: Concatenation风格 (x = cat[x, skip])
#      - ddpm用Addition (x = x + skip)，信息混合受限
#      - Concat保留了完整的encoder和decoder特征，由网络学习如何融合
#   3. 残差块: Pre-Norm风格 + conv2零初始化
#      - ddpm用Post-Norm，训练不如Pre-Norm稳定
#      - conv2的init_scale=0使初始时残差块≈恒等映射，训练更平稳
#   4. Decoder每层blocks+1个block: 额外一个用于消化concat后的维度变化
#
# 这版U-Net结构尽量保持跟原版一致（除了没加Attention），效果相对更好，计算量也更大
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
blocks = 2  # 如果显存不够，可以降低为1
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
    """Dense包装
    """
    def __init__(self, in_dim, out_dim, activation=None, init_scale=1):
        super().__init__()
        init_scale = max(init_scale, 1e-10)
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
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
    """Conv2D包装
    """
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


class SinusoidalEmbedding(nn.Module):
    """【改进1】正弦位置编码（不可训练）
    公式: PE(pos, 2i) = sin(pos / 10000^{2i/d})
           PE(pos, 2i+1) = cos(pos / 10000^{2i/d})
    
    vs ddpm的nn.Embedding:
      - 固定编码而非可学习，提供平滑的时间信号
      - 相邻时间步的编码相似度高，有利于模型泛化
      - 后接MLP可以学到非线性的时间表示
    """
    def __init__(self, num_embeddings, dim):
        super().__init__()
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        pos = torch.arange(num_embeddings, dtype=torch.float32)
        emb = pos[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        self.register_buffer('embedding', emb)

    def forward(self, idx):
        return self.embedding[idx]


class ResidualBlock2(nn.Module):
    """【改进3】残差block (Pre-Norm风格)
    结构: norm1 → silu → conv1 → [+t_emb] → norm2 → silu → conv2 → [+shortcut]
    
    vs ddpm的ResidualBlock (Post-Norm风格):
      ddpm:  x + t → conv1(swish) → conv2(swish) → (+skip) → norm
      ddpm2: norm → silu → conv1 → (+t) → norm → silu → conv2 → (+skip)
    
    改进要点:
      1. Pre-Norm: 先归一化再卷积，训练更稳定（参考Pre-LN Transformer）
      2. conv2的init_scale=0: 初始时残差块输出≈零，整个残差块≈恒等映射
         这种初始化策略来自"Fixup Initialization"，有利于深层网络训练
      3. 时间条件注入位置: 在两层卷积之间注入（而非ddpm在卷积前注入）
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
        self.norm1 = GroupNorm(in_dim)
        self.conv1 = Conv2dLayer(in_dim, out_dim)
        self.t_proj = DenseLayer(t_dim, out_dim)
        self.norm2 = GroupNorm(out_dim)
        # 【改进】conv2零初始化: 训练初期残差块输出≈零，整体≈恒等映射
        self.conv2 = Conv2dLayer(out_dim, out_dim, init_scale=0)

    def forward(self, x, t):
        if self.proj is not None:
            xi = self.proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            xi = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        # add t
        t_emb = self.t_proj(t)  # (B, 1, out_dim)
        x = x + t_emb.permute(0, 2, 1).unsqueeze(-1)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = x + xi
        return x


class UNet2(nn.Module):
    """【改进版】U-Net去噪模型 (Concatenate风格)
    
    核心改进总结:
      1. 时间编码: SinusoidalEmbedding(T, D) → MLP(D→4D→4D)
         —— 固定编码提供平滑基底，MLP学习非线性变换
      2. 跳跃连接: x = torch.cat([x, skip], dim=1)
         —— 保留encoder和decoder完整信息，由后续卷积学习融合
      3. Decoder每层blocks+1个残差块
         —— 额外一个block处理concat后维度增加的特征
      4. Pre-Norm + init_scale=0 —— 更稳定的训练
    """
    def __init__(self):
        super().__init__()
        # 【改进1】时间编码: SinusoidalEmbedding + MLP
        # step 1: 固定正弦编码 t → (D,)
        # step 2: MLP(D → 4D → 4D) 学习非线性时间表示
        self.t_sinusoidal = SinusoidalEmbedding(T, embedding_size)
        self.t_dense1 = DenseLayer(embedding_size, embedding_size * 4, 'swish')
        self.t_dense2 = DenseLayer(embedding_size * 4, embedding_size * 4, 'swish')

        self.init_conv = Conv2dLayer(3, embedding_size)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        in_ch = embedding_size
        self.encoder_channels = []
        t_dim = embedding_size * 4

        for i, ch in enumerate(channels):
            block_list = nn.ModuleList()
            for j in range(blocks):
                block_list.append(ResidualBlock2(in_ch, ch, t_dim))
                in_ch = ch * embedding_size
                self.encoder_channels.append(in_ch)
            self.down_blocks.append(block_list)
            if i != len(channels) - 1:
                self.down_pools.append(nn.AvgPool2d(2))
                self.encoder_channels.append(in_ch)
            else:
                self.down_pools.append(None)

        self.mid_block = ResidualBlock2(in_ch, channels[-1], t_dim)

        # 【改进2】Decoder使用Concatenate跳跃连接
        # 每层有blocks+1个残差块，比encoder多一个
        # 额外的block负责处理concat带来的维度增加
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        # 需要记录encoder_channels，用于concat后的输入维度
        enc_chs = list(self.encoder_channels)  # 拷贝一份用于计算

        for i, ch in enumerate(channels[::-1]):
            block_list = nn.ModuleList()
            for j in range(blocks + 1):
                skip_ch = enc_chs.pop()
                block_list.append(ResidualBlock2(in_ch + skip_ch, ch, t_dim))
                in_ch = ch * embedding_size
            self.up_blocks.append(block_list)
            if i != len(channels) - 1:
                self.up_samples.append(nn.Upsample(scale_factor=2))
            else:
                self.up_samples.append(None)

        self.final_norm = GroupNorm(in_ch)
        self.final_conv = Conv2dLayer(in_ch, 3)

    def forward(self, x, t_idx):
        t = self.t_sinusoidal(t_idx.squeeze(-1))
        t = self.t_dense1(t)
        t = self.t_dense2(t)
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

        # Decoder
        for i, (block_list, up) in enumerate(zip(self.up_blocks, self.up_samples)):
            for block in block_list:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)  # 【改进2】Concatenate跳跃连接
                x = block(x, t)
            if up is not None:
                x = up(x)

        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        return x
