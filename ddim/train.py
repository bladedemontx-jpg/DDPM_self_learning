#! -*- coding: utf-8 -*-
# ============================================================================
# 生成扩散模型DDIM参考代码 (PyTorch版)
# ============================================================================
# 【核心改进】仅修改采样过程，不改变训练（直接复用ddpm2的训练代码）
#
# 【DDIM vs DDPM 采样公式对比】
#   DDPM:  x_{t-1} = (x_t - β_t²/β̄_t · ε_θ) / α_t + σ_t · z
#     - 随机性来自 σ_t = β_t，必须走完所有T步
#
#   DDIM:  x_{t-1} = ᾱ_{t-1}/ᾱ_t · (x_t - ε_t · ε_θ) + ε_{t-1} · ε_θ + σ_t · z
#     其中 ε_t = β̄_t - α_t · √(β̄_{t-1}² - σ_t²)
#     - eta=0时完全确定性，同一噪声总是生成同一图像
#     - eta=1时退化为DDPM
#     - stride参数允许跳步采样，大幅加速（如T=1000, stride=100只需走10步）
#
# 【改进优势】
#   1. 可控的随机性: eta∈[0,1]控制采样方差
#   2. 加速采样: stride跳步可以减少采样步数
#   3. 确定性采样: eta=0时可用于图像插值等应用
#
# 原始Keras版本博客：https://kexue.fm/archives/9181
# ============================================================================


# 导入ddpm2的训练代码（包含模型和预训练权重）
from ddpm2.train import *


@torch.no_grad()
def sample_ddim(path=None, n=4, z_samples=None, stride=1, eta=1, use_ema=True):
    """【DDIM采样】
    eta控制方差的相对大小; stride空间跳跃加速采样
    
    DDIM采样公式（重参数化形式）:
      σ_t = β̄_{t-1}/β̄_t · √(1-α_t²) · eta   (eta控制随机性)
      ε_t = β̄_t - α_t · √(β̄_{t-1}² - σ_t²)  (确定性部分的系数)
      x_{t-1} = (x_t - ε_t · ε_θ) / α_t + σ_t · z
    
    特殊情况:
      eta=0 → σ=0, 完全确定性采样 (DDIM)
      eta=1 → 退化为DDPM采样
    """
    net = ema_model if use_ema else model
    net.eval()
    # 【DDIM采样参数计算】
    # stride跳步: 从原始T步中等间距抽取子序列
    bar_alpha_ = bar_alpha[::stride]        # 跳步后的ᾱ_t
    bar_alpha_pre_ = np.pad(bar_alpha_[:-1], [1, 0], constant_values=1)  # ᾱ_{t-1}, 第0步为1
    bar_beta_ = np.sqrt(1 - bar_alpha_**2)  # β̄_t = √(1-ᾱ_t²)
    bar_beta_pre_ = np.sqrt(1 - bar_alpha_pre_**2)  # β̄_{t-1}
    alpha_ = bar_alpha_ / bar_alpha_pre_     # α_t = ᾱ_t / ᾱ_{t-1}
    # DDIM方差: σ_t = β̄_{t-1}/β̄_t · √(1-α_t²) · eta
    sigma_ = bar_beta_pre_ / bar_beta_ * np.sqrt(1 - alpha_**2) * eta
    # DDIM确定性部分系数: ε_t = β̄_t - α_t · √(β̄_{t-1}² - σ_t²)
    epsilon_ = bar_beta_ - alpha_ * np.sqrt(bar_beta_pre_**2 - sigma_**2)
    T_ = len(bar_alpha_)  # 跳步后的实际采样步数
    # 采样过程
    if z_samples is None:
        z_samples = np.random.randn(n**2, 3, img_size, img_size).astype('float32')
    else:
        z_samples = z_samples.copy()
    for t in tqdm(range(T_), ncols=0):
        t = T_ - t - 1
        bt = torch.full((z_samples.shape[0], 1), t * stride, dtype=torch.long, device=device)
        z_tensor = torch.from_numpy(z_samples).to(device)
        pred = net(z_tensor, bt).cpu().numpy()  # 预测噪声 ε_θ(x_t, t)
        # DDIM采样公式: x_{t-1} = (x_t - ε_t · ε_θ) / α_t + σ_t · z
        z_samples -= epsilon_[t] * pred   # 减去确定性部分
        z_samples /= alpha_[t]            # 除以α_t
        z_samples += np.random.randn(*z_samples.shape).astype('float32') * sigma_[t]  # 加随机噪声
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
def sample_inter_ddim(path, n=4, k=8, stride=1, use_ema=True):
    """【DDIM球面插值】
    利用DDIM的确定性采样(eta=0)实现比统噪声空间插值。
    球面插值: z = z_1 * sin(θ) + z_2 * cos(θ),  θ ∈ [0, π/2]
    相比线性插值，球面插值保持噪声向量的范数不变，插值过渡更自然。
    """
    figure = np.ones((img_size * n, img_size * k, 3))
    Z = np.random.randn(n * 2, 3, img_size, img_size).astype('float32')
    z_samples = []
    for i in range(n):
        for j in range(k):
            theta = np.pi / 2 * j / (k - 1)
            z = Z[2 * i] * np.sin(theta) + Z[2 * i + 1] * np.cos(theta)
            z_samples.append(z)
    x_samples = sample_ddim(z_samples=np.array(z_samples), stride=stride, eta=0, use_ema=use_ema)
    x_samples_hwc = x_samples.transpose(0, 2, 3, 1)
    for i in range(n):
        for j in range(k):
            ij = i * k + j
            figure[i * img_size:(i + 1) * img_size,
                   img_size * j:img_size * (j + 1)] = np.clip(x_samples_hwc[ij], -1, 1)
    imwrite(path, figure)


if __name__ == '__main__':
    sample_ddim('test.png', n=4, stride=100, eta=0)
    sample_inter_ddim('test_inter.png', n=8, k=15, stride=20)
