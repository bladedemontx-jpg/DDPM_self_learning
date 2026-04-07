#! -*- coding: utf-8 -*-
# 生成扩散模型DDPM参考代码 (PyTorch版)
# 用了Pre Norm GAU架构
# 原始Keras版本参考：https://kexue.fm/archives/9984

import os
import cv2
import glob
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('samples'):
    os.mkdir('samples')

# 基本配置
def list_pictures(directory, ext='png'):
    return sorted(glob.glob(os.path.join(directory, f'*.{ext}')))

imgs = list_pictures('/root/CelebA-HQ/train/', 'png')
imgs += list_pictures('/root/CelebA-HQ/valid/', 'png')
np.random.shuffle(imgs)
img_size = 128  # 如果只想快速实验，可以改为64
batch_size = 64  # 如果显存不够，可以降低为32、16，但不建议低于16
hidden_size = 768
num_layers = 24

# 超参数选择
T = 1000
alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
beta = np.sqrt(1 - alpha**2)
bar_alpha = np.cumprod(alpha)
bar_beta = np.sqrt(1 - bar_alpha**2)
sigma = beta.copy()


def imread(f, crop_size=None):
    """读取图片
    """
    x = cv2.imread(f)
    height, width = x.shape[:2]
    if crop_size is None:
        crop_size = min([height, width])
    else:
        crop_size = min([crop_size, height, width])
    height_x = (height - crop_size + 1) // 2
    width_x = (width - crop_size + 1) // 2
    x = x[height_x:height_x + crop_size, width_x:width_x + crop_size]
    if x.shape[:2] != (img_size, img_size):
        x = cv2.resize(x, (img_size, img_size))
    x = x.astype('float32')
    x = x / 255 * 2 - 1
    return x


def imwrite(path, figure):
    """归一化到了[-1, 1]的图片矩阵保存为图片
    """
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    cv2.imwrite(path, figure)


class ImageDataset(Dataset):
    """图片数据集
    """
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = imread(self.img_paths[idx])
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)


def collate_fn(batch):
    """自定义collate
    """
    batch_imgs = torch.stack(batch, dim=0)
    batch_steps = torch.randint(0, T, (batch_imgs.shape[0],))
    batch_bar_alpha = torch.from_numpy(bar_alpha[batch_steps.numpy()]).float()[:, None, None, None]
    batch_bar_beta = torch.from_numpy(bar_beta[batch_steps.numpy()]).float()[:, None, None, None]
    batch_noise = torch.randn_like(batch_imgs)
    batch_noisy_imgs = batch_imgs * batch_bar_alpha + batch_noise * batch_bar_beta
    return batch_noisy_imgs, batch_steps[:, None], batch_noise


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
    """应用2D旋转位置编码
    x: (B, L, D)
    pos_emb: (1, L, D)
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
    """RMS归一化（不减均值，无offset）
    """
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = (x ** 2).mean(dim=-1, keepdim=True).sqrt()
        return x / (rms + 1e-6) * self.weight


class GatedAttentionUnit(nn.Module):
    """Gated Attention Unit (GAU)
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

        # 计算u, v, q, k
        uvqk = self.to_uv(x)
        u, v, q, k = uvqk.split([self.hidden_size, self.hidden_size, self.key_size, self.key_size], dim=-1)
        u = F.silu(u)
        v = F.silu(v)

        # 应用RoPE到q, k
        q = apply_rotary_pos_emb(q, pos_emb[:, :L, :self.key_size])
        k = apply_rotary_pos_emb(k, pos_emb[:, :L, :self.key_size])

        # 注意力
        qk = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, L, L)
        if self.normalization == 'softmax':
            attn = F.softmax(qk, dim=-1)
        else:
            attn = F.relu(qk) ** 2

        # 门控输出
        out = torch.bmm(attn, v)  # (B, L, hidden)
        out = u * out
        out = self.to_out(out)
        return out


class GAUBlock(nn.Module):
    """GAU块: LayerNorm + GAU + 残差
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
    """基于GAU的去噪模型
    将图像打成8x8的patch，然后用GAU处理
    """
    def __init__(self):
        super().__init__()
        self.patch_size = 8
        patch_dim = self.patch_size * self.patch_size * 3  # 192
        self.num_patches = (img_size // self.patch_size) ** 2

        self.patch_proj = nn.Linear(patch_dim, hidden_size, bias=False)
        self.t_embed = nn.Embedding(T, hidden_size)

        self.blocks = nn.ModuleList([
            GAUBlock(hidden_size, 128)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(hidden_size)
        self.unpatch_proj = nn.Linear(hidden_size, patch_dim, bias=False)

    def _make_rope_2d(self, device):
        """生成2D RoPE位置编码
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


def l2_loss(y_true, y_pred):
    """用l2距离为损失，不能用mse代替
    """
    return ((y_true - y_pred)**2).sum(dim=[1, 2, 3])


# 构建模型
model = GAUDenoisingModel().to(device)
print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')

# EMA模型
ema_model = copy.deepcopy(model)
ema_momentum = 0.9999


def update_ema():
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(ema_momentum).add_(p.data, alpha=1 - ema_momentum)


# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)

def get_lr_scale(step):
    if step < 4000:
        return step / 4000
    elif step < 20000:
        return 1.0
    elif step < 40000:
        return 0.5
    else:
        return 0.1

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)


@torch.no_grad()
def sample(path=None, n=4, z_samples=None, t0=0, use_ema=False):
    """随机采样函数
    """
    net = ema_model if use_ema else model
    net.eval()
    if z_samples is None:
        z_samples = np.random.randn(n**2, 3, img_size, img_size).astype('float32')
    else:
        z_samples = z_samples.copy()
    for t_step in tqdm(range(t0, T), ncols=0):
        t_step = T - t_step - 1
        bt = torch.full((z_samples.shape[0], 1), t_step, dtype=torch.long, device=device)
        z_tensor = torch.from_numpy(z_samples).to(device)
        pred = net(z_tensor, bt).cpu().numpy()
        z_samples -= beta[t_step]**2 / bar_beta[t_step] * pred
        z_samples /= alpha[t_step]
        z_samples += np.random.randn(*z_samples.shape).astype('float32') * sigma[t_step]
    x_samples = np.clip(z_samples, -1, 1)
    x_samples_hwc = x_samples.transpose(0, 2, 3, 1)
    if path is None:
        return x_samples
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            digit = x_samples_hwc[i * n + j]
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = digit
    imwrite(path, figure)
    return x_samples


@torch.no_grad()
def sample_inter(path, n=4, k=8, sep=10, t0=500, use_ema=False):
    """随机采样插值函数
    """
    figure = np.ones((img_size * n, img_size * (k + 2) + sep * 2, 3))
    x_samples = [imread(f) for f in np.random.choice(imgs, n * 2)]
    X = []
    for i in range(n):
        figure[i * img_size:(i + 1) * img_size, :img_size] = x_samples[2 * i]
        figure[i * img_size:(i + 1) * img_size,
               -img_size:] = x_samples[2 * i + 1]
        for j in range(k):
            lamb = 1. * j / (k - 1)
            x = x_samples[2 * i] * (1 - lamb) + x_samples[2 * i + 1] * lamb
            X.append(x)
    x_np = np.array(X).transpose(0, 3, 1, 2).astype('float32')
    x_np = x_np * bar_alpha[t0]
    x_np += np.random.randn(*x_np.shape).astype('float32') * bar_beta[t0]
    x_rec_samples = sample(z_samples=x_np, t0=t0, use_ema=use_ema)
    x_rec_hwc = x_rec_samples.transpose(0, 2, 3, 1)
    for i in range(n):
        for j in range(k):
            ij = i * k + j
            figure[i * img_size:(i + 1) * img_size, img_size * (j + 1) +
                   sep:img_size * (j + 2) + sep] = np.clip(x_rec_hwc[ij], -1, 1)
    imwrite(path, figure)


def train_one_epoch(epoch, dataloader, steps_per_epoch=2000):
    """训练一个epoch
    """
    model.train()
    total_loss = 0
    step = 0
    for batch_noisy, batch_steps, batch_noise in dataloader:
        batch_noisy = batch_noisy.to(device)
        batch_steps = batch_steps.to(device)
        batch_noise = batch_noise.to(device)

        pred = model(batch_noisy, batch_steps)
        loss = l2_loss(batch_noise, pred).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        update_ema()

        total_loss += loss.item()
        step += 1
        if step >= steps_per_epoch:
            break

    avg_loss = total_loss / step
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

    torch.save(model.state_dict(), 'model.pth')
    sample(f'samples/{epoch + 1:05d}.png')
    torch.save(ema_model.state_dict(), 'model_ema.pth')
    sample(f'samples/{epoch + 1:05d}_ema.png', use_ema=True)


if __name__ == '__main__':
    dataset = ImageDataset(imgs)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, drop_last=True
    )

    for epoch in range(10000):
        train_one_epoch(epoch, dataloader)

else:
    ema_model.load_state_dict(torch.load('model_ema.pth', map_location=device))
    ema_model.eval()
