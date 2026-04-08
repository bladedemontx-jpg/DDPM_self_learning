#! -*- coding: utf-8 -*-
# ============================================================================
# 生成扩散模型DDPM参考代码2 (PyTorch版) - 改进版
# ============================================================================
# 【相对ddpm基线的训练层面改进】
#   1. 使用UNet2（Pre-Norm + Concatenate风格）代替UNet
#   2. batch_size从64降为32（因为Concatenate风格参数量更大）
# 训练理论和噪声调度与ddpm完全相同
#
# 【DDPM训练目标】
#   L = E_{t, x_0, ε} [ ||ε - ε_θ(ᾱ_t x_0 + β̄_t ε, t)||_2² ]
#
# 原始Keras版本博客：https://kexue.fm/archives/9152
# ============================================================================

import os
import cv2
import glob
import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 导入UNet模型
from .unet import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('samples'):
    os.mkdir('samples')

# 基本配置
def list_pictures(directory, ext='png'):
    return sorted(glob.glob(os.path.join(directory, f'*.{ext}')))

imgs = list_pictures('/root/CelebA-HQ/train/', 'png')
imgs += list_pictures('/root/CelebA-HQ/valid/', 'png')
np.random.shuffle(imgs)
batch_size = 32  # 比ddpm的64小，因为Concatenate风格的UNet2参数量更大，显存占用更多

# 【噪声调度】与ddpm基线完全相同
# α_t = √(1 - 0.02t/T), β_t = √(1-α_t²)
# ᾱ_t = ∏α_s, β̄_t = √(1-ᾱ_t²), σ_t = β_t

# 超参数选择
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
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return torch.from_numpy(img)


def collate_fn(batch):
    """自定义collate，加噪并返回 noisy_imgs, steps, noise
    """
    batch_imgs = torch.stack(batch, dim=0)
    batch_steps = torch.randint(0, T, (batch_imgs.shape[0],))
    batch_bar_alpha = torch.from_numpy(bar_alpha[batch_steps.numpy()]).float()[:, None, None, None]
    batch_bar_beta = torch.from_numpy(bar_beta[batch_steps.numpy()]).float()[:, None, None, None]
    batch_noise = torch.randn_like(batch_imgs)
    batch_noisy_imgs = batch_imgs * batch_bar_alpha + batch_noise * batch_bar_beta
    return batch_noisy_imgs, batch_steps[:, None], batch_noise


def l2_loss(y_true, y_pred):
    """用l2距离为损失，不能用mse代替
    """
    return ((y_true - y_pred)**2).sum(dim=[1, 2, 3])


# 构建模型
model = UNet2().to(device)
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
