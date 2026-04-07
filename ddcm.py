#! -*- coding: utf-8 -*-
# DDCM（Denoising Diffusion Codebook Models）参考代码 (PyTorch版)
# 在DDPM上修改，不用改变训练，只修改采样过程
# 原始Keras版本博客：https://kexue.fm/archives/9245

from ddpm2 import *  # 加载训练好的模型

K_codebook = 64  # 每步的Codebook大小
codebook = np.random.randn(T + 1, K_codebook, 3, img_size, img_size).astype('float32')


@torch.no_grad()
def sample_ddcm(path, n=4, use_ema=True):
    """随机采样函数
    """
    net = ema_model if use_ema else model
    net.eval()
    z_samples = codebook[T][np.random.choice(K_codebook, size=n**2)].copy()
    for t_step in tqdm(range(T), ncols=0):
        t_step = T - t_step - 1
        bt = torch.full((z_samples.shape[0], 1), t_step, dtype=torch.long, device=device)
        z_tensor = torch.from_numpy(z_samples).to(device)
        pred = net(z_tensor, bt).cpu().numpy()
        z_samples -= beta[t_step]**2 / bar_beta[t_step] * pred
        z_samples /= alpha[t_step]
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
    """随机选一些图片，进行编码和重构
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
        pred = net(z_tensor, bt).cpu().numpy()
        x0 = (z_samples - bar_beta[t_step] * pred) / bar_alpha[t_step]
        # 计算相似度: codebook[t_step] shape: (K, 3, H, W), x_samples - x0 shape: (B, 3, H, W)
        # sims[k, b] = sum over (c, h, w) of codebook[t_step][k] * (x_samples_chw[b] - x0[b])
        x_diff = np.array(x_samples_chw) - x0  # (B, 3, H, W)
        sims = np.einsum('kcwh,bcwh->kb', codebook[t_step], x_diff)
        idxs = sims.argmax(0)
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
