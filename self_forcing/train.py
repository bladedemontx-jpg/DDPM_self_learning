#! -*- coding: utf-8 -*-
# ============================================================================
# Self-Forcing 改进版 (PyTorch) - 基于DDPM框架
# ============================================================================
# 【问题: Exposure Bias (训练-推理分布偏移)】
#   DDPM训练时: x_t是用真实x_0加噪得到的 (teacher forcing)
#   DDPM推理时: x_t是用模型自己的预测结果逐步迭代得到的
#   → 训练时看到的分布和推理时不同，误差会累积
#
# 【Self-Forcing解决方案】
#   在训练中混合两种输入:
#   1. Teacher Forcing: 正常DDPM训练，x_t用真实x_0加噪
#   2. Self-Forcing: 用模型在t步的预测估计x̂_0，
#      再用x̂_0重新加噪到t-1步，让模型学习处理自己的误差
#   总损失: L = L_tf + λ · L_sf
#
# 【Self-Forcing过程的数学描述】
#   1. 正常前向: x_t1 = ᾱ_t1·x_0 + β̄_t1·ε_1,  ε̂_1 = ε_θ(x_t1, t1)
#   2. 估计x_0:  x̂_0 = (x_t1 - β̄_t1·ε̂_1) / ᾱ_t1
#   3. 重新加噪: x_t2 = ᾱ_t2·x̂_0 + β̄_t2·ε_2 (使用模型预测的x̂_0而非真实x_0)
#   4. Self-Forcing损失: L_sf = ||ε_2 - ε_θ(x_t2, t2)||_2²
#
# 参考: Self-Forcing: Bridging the Train-Test Gap
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
batch_size = 32

# 超参数选择
alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
beta = np.sqrt(1 - alpha**2)
bar_alpha = np.cumprod(alpha)
bar_beta = np.sqrt(1 - bar_alpha**2)
sigma = beta.copy()

# Self-Forcing超参数
sf_lambda = 0.5        # self-forcing loss的权重 λ
sf_warmup_steps = 5000  # warmup期间只用teacher forcing，等模型基本收敛后再加入SF


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
    """【Self-Forcing训练步骤】
    包含两个部分:
    1. Teacher Forcing Loss: 标准DDPM训练 L_tf = ||ε - ε_θ(x_t1, t1)||_2²
    2. Self-Forcing Loss: 用模型预测的x̂_0重新加噪，再让模型预测
       L_sf = ||ε_2 - ε_θ(ᾱ_{t2}·x̂_0 + β̄_{t2}·ε_2, t2)||_2²
    总损失: L = L_tf + λ(step) · L_sf
    """
    global global_step
    B = batch_imgs.shape[0]
    batch_imgs = batch_imgs.to(device)

    # 随机选择两个连续时间步 t1 > t2（t1更接近纯噪声）
    # t2 = t1 - 1 确保是相邻时间步
    batch_steps_t1 = torch.randint(1, T, (B,), device=device)
    batch_steps_t2 = batch_steps_t1 - 1  # t2 = t1 - 1

    # === 【Step 1】Teacher Forcing Loss ===
    # 标准DDPM前向过程: x_t1 = ᾱ_t1 · x_0 + β̄_t1 · ε_1
    ba1 = torch.from_numpy(bar_alpha[batch_steps_t1.cpu().numpy()]).float().to(device)[:, None, None, None]
    bb1 = torch.from_numpy(bar_beta[batch_steps_t1.cpu().numpy()]).float().to(device)[:, None, None, None]
    noise1 = torch.randn_like(batch_imgs)
    noisy1 = batch_imgs * ba1 + noise1 * bb1   # x_t1 = ᾱ_t1 · x_0 + β̄_t1 · ε_1
    pred_noise1 = model(noisy1, batch_steps_t1[:, None])  # ε̂_1 = ε_θ(x_t1, t1)
    tf_loss = l2_loss(noise1, pred_noise1).mean()  # L_tf = ||ε_1 - ε̂_1||_2²

    # === 【Step 2】Self-Forcing Loss ===
    # λ的warmup调度: 前N步为0，然后线性增加到sf_lambda
    sf_weight = min(1.0, max(0.0, (global_step - sf_warmup_steps) / sf_warmup_steps)) * sf_lambda

    if sf_weight > 0:
        # 【Self-Forcing核心】用模型预测估计x_0
        # x̂_0 = (x_t1 - β̄_t1 · ε̂_1) / ᾱ_t1
        with torch.no_grad():
            pred_x0 = (noisy1 - bb1 * pred_noise1.detach()) / ba1
            pred_x0 = pred_x0.clamp(-1, 1)  # 裁剪到有效范围

        # 用预测的x̂_0（而非真实x_0）重新加噪到t2
        # x_t2 = ᾱ_t2 · x̂_0 + β̄_t2 · ε_2
        # 这就模拟了推理时的情景：输入来自模型自己的预测而非真实数据
        ba2 = torch.from_numpy(bar_alpha[batch_steps_t2.cpu().numpy()]).float().to(device)[:, None, None, None]
        bb2 = torch.from_numpy(bar_beta[batch_steps_t2.cpu().numpy()]).float().to(device)[:, None, None, None]
        noise2 = torch.randn_like(batch_imgs)
        # 【关键差异】用预测的x̂_0而非真实x_0来构造t2时刻的加噪图
        noisy2_sf = pred_x0 * ba2 + noise2 * bb2    # x_t2_sf = ᾱ_t2·x̂_0 + β̄_t2·ε_2
        pred_noise2_sf = model(noisy2_sf, batch_steps_t2[:, None])  # ε̂_2 = ε_θ(x_t2_sf, t2)
        sf_loss = l2_loss(noise2, pred_noise2_sf).mean()  # L_sf = ||ε_2 - ε̂_2||_2²

        total_loss = tf_loss + sf_weight * sf_loss  # L = L_tf + λ · L_sf
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
