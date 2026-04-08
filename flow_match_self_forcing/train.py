#! -*- coding: utf-8 -*-
# ============================================================================
# Flow Matching + Self-Forcing 改进版 (PyTorch)
# ============================================================================
# 【结合两种改进】
#   Flow Matching: 连续ODE框架，不需要噪声调度
#   Self-Forcing:  训练时用模型自身输出做多步展开，减少exposure bias
#
# 【与self_forcing(基于DDPM)的对比】
#   self_forcing:     离散时间，预测x̂_0后重新加噪到t-1，只展开1步
#   flow_match_sf:    连续时间，用Euler积分展开sf_unroll_steps步
#
# 【训练流程】
#   1. Teacher Forcing: 随机采样t，计算x_t，预测v_θ(x_t,t)
#      L_tf = ||v_θ(x_t,t) - (ε-x_0)||_2²
#   2. Self-Forcing多步展开: 从t_start开始
#      for step in range(sf_unroll_steps):
#        计算 L_sf += ||v_θ(x_t, t) - (ε-x_0)||_2²
#        x_{t-dt} = x_t - dt · v_θ(x_t, t)   ← 用模型自己的输出做下一步输入
#   3. 总损失: L = L_tf + λ · L_sf
#
# 【与self_forcing的具体差异】
#   - 空间: DDPM语境下预测噪声ε vs FM语境下预测速度场v
#   - 展开: self_forcing只做1步(t1→t2=t1-1)
#            flow_match_sf做sf_unroll_steps步(默认3步)
#   - 时间: 离散 vs 连续
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

# Self-Forcing超参数
sf_lambda = 0.5          # self-forcing loss权重 λ
sf_warmup_steps = 5000   # warmup步数，前期只用teacher forcing
sf_unroll_steps = 3      # 【关键参数】self-forcing展开步数，比self_forcing的单1步更多


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
model = FlowMatchSFUNet().to(device)
print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')

# EMA
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


def fm_sf_train_step(batch_imgs):
    """【Flow Matching + Self-Forcing 训练步骤】
    分两部分:
    1. Teacher Forcing: L_tf = ||v_θ(x_t,t) - (ε-x_0)||_2²
    2. Self-Forcing (sf_unroll_steps步Euler展开):
       每步计算loss并用模型自己的预测做下一步输入
       L_sf = (1/K) ∑_{k=0}^{K-1} ||v_θ(x_{t_k}, t_k) - v_target||_2²
    总损失: L = L_tf + λ · L_sf
    """
    global global_step
    B = batch_imgs.shape[0]
    batch_imgs = batch_imgs.to(device)

    # === 【Teacher Forcing】 ===
    t_tf = torch.rand(B, device=device)  # t ~ Uniform(0,1)
    noise_tf = torch.randn_like(batch_imgs)
    # 插值路径: x_t = (1-t)·x_0 + t·ε
    x_t_tf = (1 - t_tf[:, None, None, None]) * batch_imgs + t_tf[:, None, None, None] * noise_tf
    v_target_tf = noise_tf - batch_imgs  # 目标速度场 v = ε - x_0
    v_pred_tf = model(x_t_tf, t_tf)
    tf_loss = l2_loss(v_target_tf, v_pred_tf).mean()  # L_tf

    # === Self-forcing loss ===
    sf_weight = min(1.0, max(0.0, (global_step - sf_warmup_steps) / sf_warmup_steps)) * sf_lambda

    if sf_weight > 0:
        # 【Self-Forcing多步展开】
        # 随机选起始时间 t_start ∈ [0.2, 1.0]，从高噪声开始
        t_start = torch.rand(B, device=device) * 0.8 + 0.2
        dt = t_start / sf_unroll_steps  # 每步的时间步长

        # 构造起始点: x_{t_start} = (1-t)·x_0 + t·ε
        noise_sf = torch.randn_like(batch_imgs)
        x_t = (1 - t_start[:, None, None, None]) * batch_imgs + t_start[:, None, None, None] * noise_sf

        sf_loss = torch.tensor(0.0, device=device)
        for step_i in range(sf_unroll_steps):
            t_cur = t_start - step_i * dt  # 当前时间
            # 速度场目标: v = ε - x_0 (线性路径下为常数，与t无关)
            v_target = noise_sf - batch_imgs
            v_pred = model(x_t, t_cur)
            sf_loss = sf_loss + l2_loss(v_target, v_pred).mean()

            # 【Self-Forcing核心】用模型自己的预测做Euler积分
            # x_{t-dt} = x_t - dt · v_θ(x_t, t)  ← 而非用真实轨迹
            if step_i < sf_unroll_steps - 1:
                with torch.no_grad():
                    x_t = x_t - dt[:, None, None, None] * v_pred.detach()

        sf_loss = sf_loss / sf_unroll_steps
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
def sample(path=None, n=4, z_samples=None, num_steps=100, use_ema=False):
    """采样函数：通过Euler方法积分ODE
    """
    net = ema_model if use_ema else model
    net.eval()
    if z_samples is None:
        z_samples = np.random.randn(n**2, 3, img_size, img_size).astype('float32')
    else:
        z_samples = z_samples.copy()

    dt = 1.0 / num_steps
    for i in tqdm(range(num_steps), ncols=0):
        t_val = 1.0 - i * dt
        t_tensor = torch.full((z_samples.shape[0],), t_val, dtype=torch.float32, device=device)
        z_tensor = torch.from_numpy(z_samples).to(device)
        v_pred = net(z_tensor, t_tensor).cpu().numpy()
        z_samples -= dt * v_pred

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
def sample_inter(path, n=4, k=8, num_steps=100, use_ema=False):
    """随机采样插值函数（球面插值）
    """
    figure = np.ones((img_size * n, img_size * k, 3))
    Z = np.random.randn(n * 2, 3, img_size, img_size).astype('float32')
    z_samples = []
    for i in range(n):
        for j in range(k):
            theta = np.pi / 2 * j / (k - 1)
            z = Z[2 * i] * np.sin(theta) + Z[2 * i + 1] * np.cos(theta)
            z_samples.append(z)
    x_samples = sample(z_samples=np.array(z_samples), num_steps=num_steps, use_ema=use_ema)
    x_samples_hwc = x_samples.transpose(0, 2, 3, 1)
    for i in range(n):
        for j in range(k):
            ij = i * k + j
            figure[i * img_size:(i + 1) * img_size,
                   img_size * j:img_size * (j + 1)] = np.clip(x_samples_hwc[ij], -1, 1)
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
        loss = fm_sf_train_step(batch_imgs)
        total_loss += loss
        step += 1
        if step >= steps_per_epoch:
            break

    avg_loss = total_loss / step
    print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Global Step: {global_step}')

    torch.save(model.state_dict(), 'model_fmsf.pth')
    sample(f'samples/{epoch + 1:05d}_fmsf.png')
    torch.save(ema_model.state_dict(), 'model_fmsf_ema.pth')
    sample(f'samples/{epoch + 1:05d}_fmsf_ema.png', use_ema=True)


if __name__ == '__main__':
    dataset = ImageDataset(imgs)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True
    )

    for epoch in range(10000):
        train_one_epoch(epoch, dataloader)

else:
    ema_model.load_state_dict(torch.load('model_fmsf_ema.pth', map_location=device))
    ema_model.eval()
