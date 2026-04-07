#! -*- coding: utf-8 -*-
# Self-Forcing 改进版 (PyTorch)
# 参考：Self-Forcing: Bridging the Train-Test Gap in Autoregressive Diffusion Models
# 核心思想：训练时不仅用ground truth加噪数据，还用模型自身的预测结果进行下一步预测，
#          从而减少训练与推理之间的分布偏移（exposure bias）。
# 基于DDPM框架，在训练中加入self-forcing机制：
#   1. 正常的teacher forcing步：用ground truth加噪作为输入
#   2. self-forcing步：用模型自身在上一步的预测结果重新加噪作为输入
#   3. 两种loss加权组合

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
img_size = 128
batch_size = 32
embedding_size = 128
channels = [1, 1, 2, 2, 4, 4]
blocks = 2

# 超参数选择
T = 1000
alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
beta = np.sqrt(1 - alpha**2)
bar_alpha = np.cumprod(alpha)
bar_beta = np.sqrt(1 - bar_alpha**2)
sigma = beta.copy()

# Self-Forcing超参数
sf_lambda = 0.5  # self-forcing loss的权重
sf_warmup_steps = 5000  # 在此步数之前只用teacher forcing


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
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = imread(self.img_paths[idx])
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)


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


def l2_loss(y_true, y_pred):
    return ((y_true - y_pred)**2).sum(dim=[1, 2, 3])


# 构建模型
model = SelfForcingUNet().to(device)
print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')

# EMA模型
ema_model = copy.deepcopy(model)
ema_momentum = 0.9999


def update_ema():
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(ema_momentum).add_(p.data, alpha=1 - ema_momentum)


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
global_step = 0


def self_forcing_train_step(batch_imgs):
    """Self-Forcing训练步骤
    1. Teacher forcing: 正常的DDPM训练
    2. Self-forcing: 用模型预测结果重新加噪，再让模型预测
    两个loss加权组合
    """
    global global_step
    B = batch_imgs.shape[0]
    batch_imgs = batch_imgs.to(device)

    # 随机选择两个连续的时间步 t1 > t2（t1更接近纯噪声）
    batch_steps_t1 = torch.randint(1, T, (B,), device=device)
    batch_steps_t2 = batch_steps_t1 - 1  # t2 = t1 - 1

    # Teacher forcing loss
    ba1 = torch.from_numpy(bar_alpha[batch_steps_t1.cpu().numpy()]).float().to(device)[:, None, None, None]
    bb1 = torch.from_numpy(bar_beta[batch_steps_t1.cpu().numpy()]).float().to(device)[:, None, None, None]
    noise1 = torch.randn_like(batch_imgs)
    noisy1 = batch_imgs * ba1 + noise1 * bb1
    pred_noise1 = model(noisy1, batch_steps_t1[:, None])
    tf_loss = l2_loss(noise1, pred_noise1).mean()

    # Self-forcing loss（在warmup阶段跳过）
    sf_weight = min(1.0, max(0.0, (global_step - sf_warmup_steps) / sf_warmup_steps)) * sf_lambda

    if sf_weight > 0:
        # 用模型在t1时刻的预测来估计x_0
        with torch.no_grad():
            pred_x0 = (noisy1 - bb1 * pred_noise1.detach()) / ba1
            pred_x0 = pred_x0.clamp(-1, 1)

        # 用预测的x0重新加噪到t2
        ba2 = torch.from_numpy(bar_alpha[batch_steps_t2.cpu().numpy()]).float().to(device)[:, None, None, None]
        bb2 = torch.from_numpy(bar_beta[batch_steps_t2.cpu().numpy()]).float().to(device)[:, None, None, None]
        noise2 = torch.randn_like(batch_imgs)
        # 用预测的x0（而非真实x0）来构造t2时刻的加噪图像
        noisy2_sf = pred_x0 * ba2 + noise2 * bb2
        pred_noise2_sf = model(noisy2_sf, batch_steps_t2[:, None])
        sf_loss = l2_loss(noise2, pred_noise2_sf).mean()

        total_loss = tf_loss + sf_weight * sf_loss
    else:
        total_loss = tf_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step()
    update_ema()
    global_step += 1

    return total_loss.item()


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
    for batch_imgs in dataloader:
        if isinstance(batch_imgs, (list, tuple)):
            batch_imgs = batch_imgs[0]
        loss = self_forcing_train_step(batch_imgs)
        total_loss += loss
        step += 1
        if step >= steps_per_epoch:
            break

    avg_loss = total_loss / step
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Global Step: {global_step}')

    torch.save(model.state_dict(), 'model_sf.pth')
    sample(f'samples/{epoch + 1:05d}_sf.png')
    torch.save(ema_model.state_dict(), 'model_sf_ema.pth')
    sample(f'samples/{epoch + 1:05d}_sf_ema.png', use_ema=True)


if __name__ == '__main__':
    dataset = ImageDataset(imgs)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True
    )

    for epoch in range(10000):
        train_one_epoch(epoch, dataloader)

else:
    ema_model.load_state_dict(torch.load('model_sf_ema.pth', map_location=device))
    ema_model.eval()
