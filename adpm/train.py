#! -*- coding: utf-8 -*-
# ============================================================================
# 生成扩散模型Analytic-DPM参考代码 (PyTorch版)
# ============================================================================
# 【核心改进】在DDIM基础上修正采样方差，不改变训练
#
# 【Analytic-DPM理论】
#   DDPM/DDIM的采样方差σ_t是启发式选择的，并非最优。
#   Analytic-DPM用解析公式计算最优方差:
#     σ²_opt = σ²_ddim + γ²_t · factors_t
#   其中:
#     γ_t = ε_t · ᾱ_{t-1} / ᾱ_t   (ε_t是DDIM的确定性系数)
#     factors_t = clip(1 - E[ε_θ(x_t,t)²], 0, 1)
#                通过采样估计，反映该时间步预测的不确定性
#
# 【改进优势】
#   - 无需额外训练，只需预训练模型 + 估计factors
#   - 采样质量更高，特别是少采样步数时
#
# 原始Keras版本博客：https://kexue.fm/archives/9245
# ============================================================================


# 导入ddpm2的训练代码（包含模型和预训练权重）
from ddpm2.train import *


def estimate_factors():
    """【Analytic-DPM核心】估计方差修正项 factors_t
    公式: factors_t = clip(1 - E[||ε_θ(x_t, t)||_2² / d], 0, 1)
    其中 d 是数据维度，这里用mean代替了sum/d
    
    直觉理解:
      - 如果模型预测的噪声很大 (E[ε²]≈ 1)，说明该时间步噪声很重，
        factors≈ 0，方差修正小
      - 如果模型预测的噪声很小 (E[ε²]≈0)，说明该时间步几乎没噪声，
        factors≈1，方差修正大
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
                pred = net(batch_noisy, batch_steps_t)  # ε_θ(x_t, t)
            preds_sq_sum += (pred ** 2).mean().item()  # ||ε_θ||_2² 的均值
            count += 1
        factors_list.append(preds_sq_sum / count)

    # factors_t = clip(1 - E[||ε_θ||_2²], 0, 1)
    factors = np.clip(1 - np.array(factors_list), 0, 1)
    return factors


# 估计factors（需要预训练好的模型）
print("Estimating variance correction factors...")
factors = estimate_factors()


@torch.no_grad()
def sample_adpm(path=None, n=4, z_samples=None, stride=1, eta=1, use_ema=True):
    """【Analytic-DPM采样】
    相比DDIM, 增加了方差修正项:
      γ_t = ε_t · ᾱ_{t-1} / ᾱ_t
      σ²_opt = σ²_ddim + γ_t² · factors_t
    这种修正使得采样方差更接近理论最优值
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
    # 【Analytic-DPM方差修正】这两行是相对DDIM新增的核心代码
    gamma_ = epsilon_ * bar_alpha_pre_ / bar_alpha_  # γ_t = ε_t · ᾱ_{t-1} / ᾱ_t
    sigma_ = np.sqrt(sigma_**2 + gamma_**2 * factors[::stride])  # σ²_opt = σ²_ddim + γ² · factors
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
