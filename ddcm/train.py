#! -*- coding: utf-8 -*-
# ============================================================================
# DDCM（Denoising Diffusion Codebook Models）参考代码 (PyTorch版)
# ============================================================================
# 【核心改进】用Codebook替换采样中的随机噪声，不改变训练
#
# 【DDCM理论】
#   DDPM采样: x_{t-1} = (x_t - β_t²/β̄_t · ε_θ) / α_t + σ_t · z,  z ~ N(0,I)
#   DDCM采样: x_{t-1} = (x_t - β_t²/β̄_t · ε_θ) / α_t + σ_t · c_t^(k)
#   其中 c_t^(k) 是从Codebook中随机选取或按相似度选取的码字
#
# 【改进优势】
#   1. 采样：用码本替代随机噪声，可以控制采样多样性
#   2. 编码：可以将图像编码为码本索引序列，实现图像压缩
#   3. 不需要额外训练，直接使用预训练的DDPM模型
#
# 原始Keras版本博客：https://kexue.fm/archives/9245
# ============================================================================


# 导入ddpm2的训练代码（包含模型和预训练权重）
from ddpm2.train import *

# 【Codebook】每个时间步预分配K个噪声向量，代替采样时的随机噪声
K_codebook = 64  # 每步的Codebook大小
# codebook[t][k] 是时间步t的第k个噪声向量，形状同图像
codebook = np.random.randn(T + 1, K_codebook, 3, img_size, img_size).astype('float32')


@torch.no_grad()
def sample_ddcm(path, n=4, use_ema=True):
    """【DDCM采样】用Codebook中随机选取的码字替代随机噪声
    DDPM:   + σ_t · z          (z ~ N(0,I))
    DDCM:   + σ_t · c_t^(k)    (k随机选取)
    """
    net = ema_model if use_ema else model
    net.eval()
    # 【起始点】从codebook[T]中随机选取初始噪声（而非DDPM的z~N(0,I)）
    z_samples = codebook[T][np.random.choice(K_codebook, size=n**2)].copy()
    for t_step in tqdm(range(T), ncols=0):
        t_step = T - t_step - 1
        bt = torch.full((z_samples.shape[0], 1), t_step, dtype=torch.long, device=device)
        z_tensor = torch.from_numpy(z_samples).to(device)
        pred = net(z_tensor, bt).cpu().numpy()  # 预测噪声 ε_θ(x_t, t)
        # DDPM去噪步骤: x = (x - β²/β̄ · ε_θ) / α
        z_samples -= beta[t_step]**2 / bar_beta[t_step] * pred
        z_samples /= alpha[t_step]
        # 【DDCM核心】用codebook中随机选取的码字替代随机噪声
        # DDPM: + σ_t · N(0,I)     DDCM: + σ_t · codebook[t][k]
        z_samples += codebook[t_step][np.random.choice(K_codebook, size=n**2)] * sigma[t_step]
    x_samples = np.clip(z_samples, -1, 1)
    x_samples_hwc = x_samples.transpose(0, 2, 3, 1)
    figure = np.zeros((img_size * n, img_size * n, 3))
    for i in range(n):
        for j in range(n):
            digit = x_samples_hwc[i * n + j]
            figure[i * img_size:(i + 1) * img_size,
                   j * img_size:(j + 1) * img_size] = digit
    imwrite(path, figure)


@torch.no_grad()
def encode_ddcm(path, n=4, use_ema=True):
    """【DDCM编码】将真实图像编码为码本索引序列，并重构
    编码过程:
      1. 在每个时间步t，用模型估计x_0: x̂_0 = (x_t - β̄_t · ε_θ) / ᾱ_t
      2. 计算每个码字与残差(x_real - x̂_0)的相似度
      3. 选择最相似的码字，使重构更接近原图
    这样可以把一张图像压缩为T个码本索引（每个log2(K)bit）
    """
    net = ema_model if use_ema else model
    net.eval()
    x_samples = [imread(f) for f in np.random.choice(imgs, n**2)]
    # HWC -> CHW
    x_samples_chw = [x.transpose(2, 0, 1) for x in x_samples]
    z_samples = np.repeat(codebook[T][:1], n**2, axis=0).copy()
    for t_step in tqdm(range(T), ncols=0):
        t_step = T - t_step - 1
        bt = torch.full((z_samples.shape[0], 1), t_step, dtype=torch.long, device=device)
        z_tensor = torch.from_numpy(z_samples).to(device)
        pred = net(z_tensor, bt).cpu().numpy()  # 预测噪声 ε_θ(x_t, t)
        # 估计x_0: x̂_0 = (x_t - β̄_t · ε_θ) / ᾱ_t
        x0 = (z_samples - bar_beta[t_step] * pred) / bar_alpha[t_step]
        # 【编码核心】计算码字与残差的相似度，选择最佳码字
        # sim(k,b) = <codebook[t][k], x_real[b] - x̂_0[b]>
        x_diff = np.array(x_samples_chw) - x0  # (B, 3, H, W)
        sims = np.einsum('kcwh,bcwh->kb', codebook[t_step], x_diff)  # 向量内积作为相似度
        idxs = sims.argmax(0)  # 选取与残差最相似的码字索引
        z_samples -= beta[t_step]**2 / bar_beta[t_step] * pred
        z_samples /= alpha[t_step]
        z_samples += codebook[t_step][idxs] * sigma[t_step]
    z_samples = np.clip(z_samples, -1, 1)
    z_samples_hwc = z_samples.transpose(0, 2, 3, 1)
    figure = np.zeros((img_size * n, img_size * n * 2, 3))
    for i in range(n):
        for j in range(n):
            digit = x_samples[i * n + j]
            figure[i * img_size:(i + 1) * img_size,
                   2 * j * img_size:(2 * j + 1) * img_size] = digit
            digit = z_samples_hwc[i * n + j]
            figure[i * img_size:(i + 1) * img_size,
                   (2 * j + 1) * img_size:(2 * j + 2) * img_size] = digit
    imwrite(path, figure)


if __name__ == '__main__':
    sample_ddcm('test1.png')
    encode_ddcm('test2.png')
