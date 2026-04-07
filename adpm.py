#! -*- coding: utf-8 -*-
# 生成扩散模型Analytic-DPM参考代码 (PyTorch版)
# 在DDIM上修改，不用改变训练，只修改采样过程的方差
# 原始Keras版本博客：https://kexue.fm/archives/9245

# from ddpm import *  # 加载训练好的模型
from ddpm2 import *  # 加载训练好的模型


def estimate_factors():
    """估计方差修正项
    用(batch_size * steps)个样本去估计
    """
    dataset_est = ImageDataset(imgs)
    dataloader_est = DataLoader(
        dataset_est, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=True
    )
    net = ema_model
    net.eval()

    factors_list = []
    for t in tqdm(range(T), ncols=0):
        preds_sq_sum = 0
        count = 0
        steps = 5
        data_iter = iter(dataloader_est)
        for _ in range(steps):
            try:
                batch_imgs = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader_est)
                batch_imgs = next(data_iter)
            batch_imgs = batch_imgs.to(device)
            batch_steps_t = torch.full((batch_imgs.shape[0], 1), t, dtype=torch.long, device=device)
            batch_bar_alpha_t = bar_alpha[t]
            batch_bar_beta_t = bar_beta[t]
            batch_noise = torch.randn_like(batch_imgs)
            batch_noisy = batch_imgs * batch_bar_alpha_t + batch_noise * batch_bar_beta_t
            with torch.no_grad():
                pred = net(batch_noisy, batch_steps_t)
            preds_sq_sum += (pred ** 2).mean().item()
            count += 1
        factors_list.append(preds_sq_sum / count)

    factors = np.clip(1 - np.array(factors_list), 0, 1)
    return factors


# 估计factors（需要预训练好的模型）
print("Estimating variance correction factors...")
factors = estimate_factors()


@torch.no_grad()
def sample_adpm(path=None, n=4, z_samples=None, stride=1, eta=1, use_ema=True):
    """随机采样函数 (Analytic-DPM)
    注：eta控制方差的相对大小；stride空间跳跃
    """
    net = ema_model if use_ema else model
    net.eval()
    # 采样参数
    bar_alpha_ = bar_alpha[::stride]
    bar_alpha_pre_ = np.pad(bar_alpha_[:-1], [1, 0], constant_values=1)
    bar_beta_ = np.sqrt(1 - bar_alpha_**2)
    bar_beta_pre_ = np.sqrt(1 - bar_alpha_pre_**2)
    alpha_ = bar_alpha_ / bar_alpha_pre_
    sigma_ = bar_beta_pre_ / bar_beta_ * np.sqrt(1 - alpha_**2) * eta
    epsilon_ = bar_beta_ - alpha_ * np.sqrt(bar_beta_pre_**2 - sigma_**2)
    gamma_ = epsilon_ * bar_alpha_pre_ / bar_alpha_  # 增加代码
    sigma_ = np.sqrt(sigma_**2 + gamma_**2 * factors[::stride])  # 增加代码
    T_ = len(bar_alpha_)
    # 采样过程
    if z_samples is None:
        z_samples = np.random.randn(n**2, 3, img_size, img_size).astype('float32')
    else:
        z_samples = z_samples.copy()
    for t in tqdm(range(T_), ncols=0):
        t = T_ - t - 1
        bt = torch.full((z_samples.shape[0], 1), t * stride, dtype=torch.long, device=device)
        z_tensor = torch.from_numpy(z_samples).to(device)
        pred = net(z_tensor, bt).cpu().numpy()
        z_samples -= epsilon_[t] * pred
        z_samples /= alpha_[t]
        z_samples += np.random.randn(*z_samples.shape).astype('float32') * sigma_[t]
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


if __name__ == '__main__':
    sample_adpm('test.png', n=8, stride=100, eta=1)
