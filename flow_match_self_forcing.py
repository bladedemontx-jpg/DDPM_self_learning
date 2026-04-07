#! -*- coding: utf-8 -*-
# Flow Matching + Self-Forcing 改进版 (PyTorch)
# 结合Flow Matching的连续ODE框架与Self-Forcing的训练策略
# 核心思想：
#   - Flow Matching：学习速度场 v(x_t, t)，通过ODE从噪声到数据
#   - Self-Forcing：训练时用模型自身预测进行多步展开，减少train-test gap
#   具体地，在训练中执行若干步Euler积分，对每步都计算loss，
#   同时用模型自身的输出（而非ground truth）作为下一步的输入

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

# Self-Forcing超参数
sf_lambda = 0.5  # self-forcing loss权重
sf_warmup_steps = 5000  # warmup步数
sf_unroll_steps = 3  # self-forcing展开步数


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


class SinusoidalTimeEmbedding(nn.Module):
    """连续时间的正弦位置编码
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None] * emb[None, :] * 1000
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


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


class FlowMatchSFUNet(nn.Module):
    """U-Net速度场模型，用于Flow Matching + Self-Forcing
    """
    def __init__(self):
        super().__init__()
        t_dim = embedding_size * 4
        self.t_sinusoidal = SinusoidalTimeEmbedding(embedding_size)
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

    def forward(self, x, t):
        """
        x: (B, 3, H, W)
        t: (B,) 连续时间
        """
        t_emb = self.t_sinusoidal(t)
        t_emb = self.t_dense1(t_emb)
        t_emb = self.t_dense2(t_emb)
        t_emb = t_emb.unsqueeze(1)

        x = self.init_conv(x)
        skips = [x]

        for block_list, pool in zip(self.down_blocks, self.down_pools):
            for block in block_list:
                x = block(x, t_emb)
                skips.append(x)
            if pool is not None:
                x = pool(x)
                skips.append(x)

        x = self.mid_block(x, t_emb)

        for block_list, up in zip(self.up_blocks, self.up_samples):
            for block in block_list:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, t_emb)
            if up is not None:
                x = up(x)

        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        return x


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
    """Flow Matching + Self-Forcing 训练步骤
    1. Teacher forcing: 在随机时间t采样x_t，预测速度场v
    2. Self-forcing: 从随机时间t开始，用模型自身的预测进行多步Euler积分，
       每步都计算对应的flow matching loss
    """
    global global_step
    B = batch_imgs.shape[0]
    batch_imgs = batch_imgs.to(device)

    # === Teacher forcing loss ===
    t_tf = torch.rand(B, device=device)  # (B,) ∈ (0, 1)
    noise_tf = torch.randn_like(batch_imgs)
    x_t_tf = (1 - t_tf[:, None, None, None]) * batch_imgs + t_tf[:, None, None, None] * noise_tf
    v_target_tf = noise_tf - batch_imgs  # 目标速度场
    v_pred_tf = model(x_t_tf, t_tf)
    tf_loss = l2_loss(v_target_tf, v_pred_tf).mean()

    # === Self-forcing loss ===
    sf_weight = min(1.0, max(0.0, (global_step - sf_warmup_steps) / sf_warmup_steps)) * sf_lambda

    if sf_weight > 0:
        # 随机选起始时间，从高噪声开始
        t_start = torch.rand(B, device=device) * 0.8 + 0.2  # ∈ [0.2, 1.0]
        dt = t_start / sf_unroll_steps  # 每步的时间步长

        # 构造起始点
        noise_sf = torch.randn_like(batch_imgs)
        x_t = (1 - t_start[:, None, None, None]) * batch_imgs + t_start[:, None, None, None] * noise_sf

        sf_loss = torch.tensor(0.0, device=device)
        for step_i in range(sf_unroll_steps):
            t_cur = t_start - step_i * dt  # 当前时间
            # 计算当前时刻的ground truth速度场
            # x_t = (1 - t) * x_0 + t * eps → v = eps - x_0
            v_target = noise_sf - batch_imgs  # 速度场是常数（线性路径）
            v_pred = model(x_t, t_cur)
            sf_loss = sf_loss + l2_loss(v_target, v_pred).mean()

            # 用模型预测进行Euler积分到下一步（self-forcing的核心）
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
