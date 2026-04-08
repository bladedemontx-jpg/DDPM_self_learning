#! -*- coding: utf-8 -*-
# ============================================================================
# Flow Matching 生成模型 (PyTorch版)
# ============================================================================
# 【与DDPM的根本范式差异】
#   DDPM: 离散SDE框架，t∈{0,...,T-1}，预测噪声ε，需要T步采样
#   FM:   连续ODE框架，t∈[0,1]，预测速度场v，任意步数采样
#
# 【Flow Matching理论】
#   插值路径 (Conditional OT): x_t = (1-t) · x_0 + t · ε
#   速度场目标: v = ε - x_0 (x_t对t的导数，线性路径下为常数)
#   训练目标: L = E_{t,x_0,ε} [ ||v_θ(x_t, t) - (ε - x_0)||_2² ]
#   采样: 从x_1=ε出发，用Euler方法求解ODE: dx/dt = v_θ(x,t)
#          x_{t-dt} = x_t - dt · v_θ(x_t, t)
#
# 【改进优势】相对DDPM:
#   1. 更简洁: 无需设计噪声调度(α_t, β_t等)
#   2. 更灵活: 采样步数可任意调整（100步、甚至10步）
#   3. 更直观: 学习从噪声到数据的传输映射
#
# 参考: Flow Matching for Generative Modeling (Lipman et al., 2022)
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

# 【注意】Flow Matching不需要离散的T步和噪声调度
# 使用连续的时间 t ∈ [0, 1]
# 插值路径：x_t = (1 - t) * x_0 + t * ε
# 目标速度场：v = ε - x_0  (线性路径的导数，与t无关)


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


def collate_fn(batch):
    """【Flow Matching数据生成】
    插值路径: x_t = (1 - t) · x_0 + t · ε
    目标:     v = ε - x_0 (x_t对t的导数)
    
    vs DDPM的collate_fn:
      DDPM: x_t = ᾱ_t·x_0 + β̄_t·ε, 目标=ε
      FM:   x_t = (1-t)·x_0 + t·ε,  目标=ε-x_0
    """
    batch_imgs = torch.stack(batch, dim=0)  # (B, 3, H, W)
    # 均匀采样时间 t ∈ (0, 1)
    # 均匀采样时间 t ~ Uniform(0, 1)
    batch_t = torch.rand(batch_imgs.shape[0], 1, 1, 1)  # (B, 1, 1, 1)
    batch_noise = torch.randn_like(batch_imgs)
    # 【插值路径】x_t = (1-t)·x_0 + t·ε  (Conditional Optimal Transport)
    batch_x_t = (1 - batch_t) * batch_imgs + batch_t * batch_noise
    # 【目标速度场】v = dx_t/dt = ε - x_0  (线性路径的导数为常数)
    batch_v = batch_noise - batch_imgs
    # t用于网络输入，取标量
    batch_t_scalar = batch_t[:, 0, 0, 0]  # (B,)
    return batch_x_t, batch_t_scalar, batch_v


def l2_loss(y_true, y_pred):
    """用l2距离为损失
    """
    return ((y_true - y_pred)**2).sum(dim=[1, 2, 3])


# 构建模型
model = FlowMatchUNet().to(device)
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
def sample(path=None, n=4, z_samples=None, num_steps=100, use_ema=False):
    """【ODE采样】通过Euler方法求解ODE
    dx/dt = v_θ(x_t, t),  从 t=1 积分到 t=0
    离散化: x_{t-dt} = x_t - dt · v_θ(x_t, t)
    
    vs DDPM采样:
      DDPM: 必须T=1000步，随机性采样
      FM:   任意步数(100步甚至10步)，确定性ODE采样
    """
    net = ema_model if use_ema else model
    net.eval()
    if z_samples is None:
        z_samples = np.random.randn(n**2, 3, img_size, img_size).astype('float32')
    else:
        z_samples = z_samples.copy()

    dt = 1.0 / num_steps
    for i in tqdm(range(num_steps), ncols=0):
        t_val = 1.0 - i * dt  # 从1到0
        t_tensor = torch.full((z_samples.shape[0],), t_val, dtype=torch.float32, device=device)
        z_tensor = torch.from_numpy(z_samples).to(device)
        v_pred = net(z_tensor, t_tensor).cpu().numpy()  # 预测速度场 v_θ(x_t, t)
        # Euler积分: x_{t-dt} = x_t - dt · v_θ(x_t, t)
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
    for batch_x_t, batch_t, batch_v in dataloader:
        batch_x_t = batch_x_t.to(device)
        batch_t = batch_t.to(device)
        batch_v = batch_v.to(device)

        # 【FM训练目标】L = E[ ||v_θ(x_t, t) - (ε - x_0)||_2² ]
        v_pred = model(batch_x_t, batch_t)
        loss = l2_loss(batch_v, v_pred).mean()

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

    torch.save(model.state_dict(), 'model_fm.pth')
    sample(f'samples/{epoch + 1:05d}_fm.png')
    torch.save(ema_model.state_dict(), 'model_fm_ema.pth')
    sample(f'samples/{epoch + 1:05d}_fm_ema.png', use_ema=True)


if __name__ == '__main__':
    dataset = ImageDataset(imgs)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, drop_last=True
    )

    for epoch in range(10000):
        train_one_epoch(epoch, dataloader)

else:
    ema_model.load_state_dict(torch.load('model_fm_ema.pth', map_location=device))
    ema_model.eval()
