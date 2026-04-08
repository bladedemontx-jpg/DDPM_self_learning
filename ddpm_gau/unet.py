#! -*- coding: utf-8 -*-
# ============================================================================
# DDPM的GAU去噪模型定义 - 完全不同的架构
# ============================================================================
# 【核心改进】用GAU (Gated Attention Unit) 替代U-Net
#
# 【与U-Net系列的根本差异】
#   U-Net (ddpm/ddpm2/flow_match等):
#     - 卷积网络，多层编码器-解码器结构
#     - 跳跃连接保留多尺度信息
#     - 无注意力机制（本项目的实现）
#
#   GAU (ddpm_gau):
#     - 图像分到88的patch，展平为序列
#     - 用Gated Attention处理序列，具有全局感受野
#     - 2D RoPE (Rotary Position Embedding) 编码空间位置
#     - 类似ViT的思路，但用GAU替代标准Self-Attention
#
# 【GAU的优势】
#   - 全局感受野: 每个patch都能看到所有patch (而U-Net依赖池化扩大感受野)
#   - 门控机制: u*Attn(v) 提供更精细的信息控制
#   - 无多尺度结构: 更简洁，无需上/下采样
#
# 原始Keras版本参考：https://kexue.fm/archives/9984
# ============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 模型配置
img_size = 128  # 如果只想快速实验，可以改为64
hidden_size = 768
num_layers = 24
T = 1000


def sinusoidal_embeddings(pos, dim, base=10000):
    """手动实现正弦位置编码
    pos: (L,) 位置序列
    dim: 编码维度
    返回: (L, dim)
    """
    half_dim = dim // 2
    emb = math.log(base) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=pos.device) * -emb)
    emb = pos[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb


def apply_rotary_pos_emb(x, pos_emb):
    """【RoPE (Rotary Position Embedding)】旋转位置编码
    公式: f(x, pos) = R(pos) · x
    其中 R 是2D旋转矩阵:
      [cosθ  -sinθ] [x1]   [x1·cosθ - x2·sinθ]
      [sinθ   cosθ] [x2] = [x2·cosθ + x1·sinθ]
    
    2D RoPE: 将位置分为行+列两组，分别编码、拼接
    vs 绝对位置编码:
      - RoPE是相对位置编码，<q·R(θ_m), k·R(θ_n)> 只依赖m-n
      - 保持平移不变性，更适合图像的空间结构
    """
    dim = pos_emb.shape[-1]
    # 将pos_emb拆成cos和sin
    cos = pos_emb[..., :dim // 2].repeat(1, 1, 2)
    sin = pos_emb[..., :dim // 2].repeat(1, 1, 2)
    # 更直接地实现: 将x分成前后两半
    x1, x2 = x[..., :dim // 2], x[..., dim // 2:dim]
    # 使用sinusoidal编码做旋转
    sin_emb = pos_emb[..., :dim // 2]
    cos_emb = pos_emb[..., dim // 2:dim]
    out1 = x1 * cos_emb - x2 * sin_emb
    out2 = x2 * cos_emb + x1 * sin_emb
    if x.shape[-1] > dim:
        return torch.cat([out1, out2, x[..., dim:]], dim=-1)
    return torch.cat([out1, out2], dim=-1)


class RMSNorm(nn.Module):
    """【归一化改进】RMS归一化（不减均值，无offset）
    公式: RMSNorm(x) = x / RMS(x) · γ
    其中 RMS(x) = √(mean(x²))
    
    vs GroupNorm (U-Net中使用):
      - GroupNorm: 减均值 + 除标准差，有偏置
      - RMSNorm: 只除RMS，更简单高效，在Transformer中更流行
    """
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
        return x / (rms + 1e-6) * self.weight


class GatedAttentionUnit(nn.Module):
    """【GAU】Gated Attention Unit
    公式: GAU(x) = u ⊙ Attention(v, q, k) = u ⊙ softmax(q·k^T/√d) · v
    其中 u, v, q, k 都由x经线性变换得到，u和v后接SiLU门控
    
    vs 标准Multi-Head Self-Attention:
      - 标准MHSA: Q,K,V + MultiHead + Linear
      - GAU: 用门控u替代多头，参数更少，效果相当
      - 门控机制允许模型动态控制信息流
    
    参考：https://kexue.fm/archives/8934
    """
    def __init__(self, hidden_size, key_size, normalization='softmax'):
        super().__init__()
        self.hidden_size = hidden_size
        self.key_size = key_size
        self.normalization = normalization

        # UV门控 + q,k
        self.to_uv = nn.Linear(hidden_size // 2, hidden_size * 2 + key_size * 2, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size // 2, bias=False)

        # 缩放因子
        self.scale = key_size ** -0.5

    def forward(self, x, pos_emb):
        """
        x: (B, L, D/2)  输入（经过LayerNorm）
        pos_emb: (1, L, key_size*2) 2D RoPE
        """
        B, L, _ = x.shape

        # 计算u, v(SiLU门控), q, k
        uvqk = self.to_uv(x)
        u, v, q, k = uvqk.split([self.hidden_size, self.hidden_size, self.key_size, self.key_size], dim=-1)
        u = F.silu(u)  # 门控信号 u = σ(Wx)
        v = F.silu(v)  # 值信号 v = σ(Wx)

        # 应用RoPE到q, k，使注意力具有相对位置感知
        q = apply_rotary_pos_emb(q, pos_emb[:, :L, :self.key_size])
        k = apply_rotary_pos_emb(k, pos_emb[:, :L, :self.key_size])

        # 注意力: softmax(q·k^T / √d)
        qk = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, L, L)
        if self.normalization == 'softmax':
            attn = F.softmax(qk, dim=-1)
        else:
            attn = F.relu(qk) ** 2

        # 【GAU门控输出】out = u ⊙ Attention(v)
        out = torch.bmm(attn, v)  # Attention(v) = softmax(QK^T/√d) · V
        out = u * out              # 门控: u ⊙ Attn(v)
        out = self.to_out(out)
        return out


class GAUBlock(nn.Module):
    """【GAU块】Pre-Norm + GAU + 残差连接
    结构: x + GAU(RMSNorm(x))
    类似Transformer的Pre-LN结构，但用GAU替代MHSA+FFN
    """
    def __init__(self, hidden_size, key_size):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.gau = GatedAttentionUnit(hidden_size, key_size, 'softmax')

    def forward(self, x, pos_emb):
        xi = x
        x = self.norm(x)
        x = self.gau(x, pos_emb)
        return xi + x


class GAUDenoisingModel(nn.Module):
    """【GAU去噪模型】基于Patch + GAU的去噪模型
    
    处理流程:
      1. Patchify: 图像(3,128,128) → 8×8 patch → 展平为(256, 192)序列
      2. Linear: (256, 192) → (256, 768) 映射到隐藏层
      3. +时间编码: 可学习Embedding，加到所有patch
      4. 24层GAU Block: 全局注意力处理
      5. Unpatchify: 还原为图像
    
    vs U-Net:
      - 无多尺度结构 (无编码器/解码器，无上/下采样)
      - 全局感受野 (GAU attention vs 局部卷积)
      - Patch粒度处理 (8×8 patch vs 像素级卷积)
    """
    def __init__(self):
        super().__init__()
        self.patch_size = 8
        patch_dim = self.patch_size * self.patch_size * 3  # 192
        self.num_patches = (img_size // self.patch_size) ** 2

        self.patch_proj = nn.Linear(patch_dim, hidden_size, bias=False)
        # 【时间编码】与ddpm基线相同，使用可学习的Embedding查表
        # 因为GAU模型尺寸较大，查表开销可忽略
        self.t_embed = nn.Embedding(T, hidden_size)

        self.blocks = nn.ModuleList([
            GAUBlock(hidden_size, 128)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(hidden_size)
        self.unpatch_proj = nn.Linear(hidden_size, patch_dim, bias=False)

    def _make_rope_2d(self, device):
        """【生成2D RoPE位置编码】
        将patch位置(row, col)分别编码为64维正弦向量，拼接得128维
        这样GAU的注意力就能感知patch之间的相对空间位置
        —— 比U-Net的剧变卷积更灵活，全局位置关系
        """
        w = img_size // self.patch_size
        pos = torch.arange(w * w, device=device)
        pos1 = pos // w
        pos2 = pos % w
        emb1 = sinusoidal_embeddings(pos1, 64, 1000)
        emb2 = sinusoidal_embeddings(pos2, 64, 1000)
        return torch.cat([emb1, emb2], dim=-1).unsqueeze(0)  # (1, L, 128)

    def _patchify(self, x):
        """将图像转为patch序列
        x: (B, 3, H, W) -> (B, num_patches, patch_dim)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, H//p, W//p, p, p, C)
        x = x.reshape(B, self.num_patches, -1)
        return x

    def _unpatchify(self, x):
        """将patch序列还原为图像
        x: (B, num_patches, patch_dim) -> (B, 3, H, W)
        """
        B = x.shape[0]
        p = self.patch_size
        h = w = img_size // p
        x = x.view(B, h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        x = x.reshape(B, 3, img_size, img_size)
        return x

    def forward(self, x, t_idx):
        """
        x: (B, 3, H, W)
        t_idx: (B, 1)
        """
        # Patchify
        x = self._patchify(x)  # (B, L, patch_dim)
        x = self.patch_proj(x)  # (B, L, hidden)

        # 时间编码
        t = self.t_embed(t_idx.squeeze(-1))  # (B, hidden)
        x = x + t.unsqueeze(1)

        # 2D RoPE
        pos_emb = self._make_rope_2d(x.device)

        # GAU blocks
        for block in self.blocks:
            x = block(x, pos_emb)

        x = self.final_norm(x)
        x = self.unpatch_proj(x)  # (B, L, patch_dim)

        # Unpatchify
        x = self._unpatchify(x)  # (B, 3, H, W)
        return x
