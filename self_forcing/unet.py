#! -*- coding: utf-8 -*-
# ============================================================================
# Self-Forcing的U-Net模型定义
# ============================================================================
# 【模型结构】
#   与ddpm2/UNet2基本相同: Pre-Norm + Concatenation跳跃连接
#   使用SinusoidalEmbedding + MLP作为时间编码
#   改进仅在训练策略（见train.py），模型结构本身无变化
#
# 参考: Self-Forcing: Bridging the Train-Test Gap
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
T = 1000


class GroupNorm(nn.Module):
    """定义GroupNorm，默认groups=32
    """
    def __init__(self, num_channels):
        super().__init__()
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


class SinusoidalEmbedding(nn.Module):
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


class ResidualBlock(nn.Module):
    """残差block (Pre Norm)
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
        t_emb = self.t_proj(t)
        x = x + t_emb.permute(0, 2, 1).unsqueeze(-1)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = x + xi
        return x


class SelfForcingUNet(nn.Module):
    """U-Net去噪模型
    """
    def __init__(self):
        super().__init__()
        t_dim = embedding_size * 4
        self.t_sinusoidal = SinusoidalEmbedding(T, embedding_size)
        self.t_dense1 = DenseLayer(embedding_size, t_dim, 'swish')
        self.t_dense2 = DenseLayer(t_dim, t_dim, 'swish')

        self.init_conv = Conv2dLayer(3, embedding_size)

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

    def forward(self, x, t_idx):
        t = self.t_sinusoidal(t_idx.squeeze(-1))
        t = self.t_dense1(t)
        t = self.t_dense2(t)
        t = t.unsqueeze(1)

        x = self.init_conv(x)
        skips = [x]

        for block_list, pool in zip(self.down_blocks, self.down_pools):
            for block in block_list:
                x = block(x, t)
                skips.append(x)
            if pool is not None:
                x = pool(x)
                skips.append(x)

        x = self.mid_block(x, t)

        for block_list, up in zip(self.up_blocks, self.up_samples):
            for block in block_list:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, t)
            if up is not None:
                x = up(x)

        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        return x
