#! -*- coding: utf-8 -*-
# 生成扩散模型DDPM参考代码 (PyTorch版)
# U-Net结构经过简化，降低了计算量
# 原始Keras版本博客：https://kexue.fm/archives/9152

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
embedding_size = 128
channels = [1, 1, 2, 2, 4, 4]
num_layers = len(channels) * 2 + 1
blocks = 2  # 如果显存不够，可以降低为1
min_pixel = 4  # 不建议降低，显存足够可以增加到8

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
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)


def collate_fn(batch):
    """自定义collate，加噪并返回 [noisy_imgs, steps], noise
    """
    batch_imgs = torch.stack(batch, dim=0)  # (B, 3, H, W)
    batch_steps = torch.randint(0, T, (batch_imgs.shape[0],))
    batch_bar_alpha = torch.from_numpy(bar_alpha[batch_steps.numpy()]).float()[:, None, None, None]
    batch_bar_beta = torch.from_numpy(bar_beta[batch_steps.numpy()]).float()[:, None, None, None]
    batch_noise = torch.randn_like(batch_imgs)
    batch_noisy_imgs = batch_imgs * batch_bar_alpha + batch_noise * batch_bar_beta
    return batch_noisy_imgs, batch_steps[:, None], batch_noise


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
    """残差block
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
        self.t_proj = DenseLayer(t_dim, in_dim)
        self.conv1 = Conv2dLayer(in_dim, out_dim, 'swish', 1 / num_layers**0.5)
        self.conv2 = Conv2dLayer(out_dim, out_dim, 'swish', 1 / num_layers**0.5)
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
    """U-Net去噪模型 (简化版)
    """
    def __init__(self):
        super().__init__()
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
        for i, (block_list, up) in enumerate(zip(self.up_blocks, self.up_samples)):
            if up is not None:
                x = up(x)
                x = x + skips.pop()
            for block in block_list:
                xi = skips.pop()
                x = block(x, t)
                x = x + xi

        x = self.final_norm(x)
        x = self.final_conv(x)
        return x


def l2_loss(y_true, y_pred):
    """用l2距离为损失，不能用mse代替
    """
    return ((y_true - y_pred)**2).sum(dim=[1, 2, 3])


# 构建模型
model = UNet().to(device)
print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')

# EMA模型
ema_model = copy.deepcopy(model)
ema_momentum = 0.9999


def update_ema():
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(ema_momentum).add_(p.data, alpha=1 - ema_momentum)


# 优化器 (LAMB风格 + warmup + lr schedule)
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
    # CHW -> HWC for saving
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
    # HWC -> CHW
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

    # 保存权重
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
