"""图内自参考一致性检测 -- Training-Free 基线

本模块实现方案文档 2.4-2.5 节的核心方法:
  - 多路密集特征融合 (DINOv2 + 高通残差 + VAE重建残差)
  - 迭代式条件参考估计
  - 内容条件归一化 (按局部纹理能量分桶)
  - Top-k 池化输出
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def weighted_robust_mean(x: torch.Tensor, weights: torch.Tensor,
                          trim_ratio: float = 0.1) -> torch.Tensor:
    """加权截尾均值"""
    if weights.sum() < 2:
        return x.mean(dim=0)
    w = weights / weights.sum()
    sorted_vals, _ = torch.sort(x, dim=0)
    n = x.shape[0]
    trim_n = max(1, int(n * trim_ratio))
    trimmed = sorted_vals[trim_n:n - trim_n]
    return trimmed.mean(dim=0)


def weighted_robust_cov(x: torch.Tensor, weights: torch.Tensor,
                         reg: float = 1e-4) -> torch.Tensor:
    """加权协方差矩阵 (带正则化)"""
    if weights.sum() < 2:
        return torch.eye(x.shape[-1], device=x.device) * reg + torch.eye(x.shape[-1], device=x.device)
    w = weights / weights.sum()
    centered = x - weighted_robust_mean(x, weights)
    cov = (centered.T @ (centered * w.unsqueeze(-1)))
    cov += torch.eye(cov.shape[0], device=cov.device) * reg
    return cov


def mahalanobis(x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """马氏距离"""
    centered = x - mean
    try:
        L = torch.linalg.cholesky(cov)
        solved = torch.cholesky_solve(centered.T, L).T
    except RuntimeError:
        cov_inv = torch.pinverse(cov)
        solved = centered @ cov_inv
    dist = (centered * solved).sum(dim=-1)
    return dist


@torch.no_grad()
def self_reference_detect(
    features: torch.Tensor,
    texture_energy: torch.Tensor,
    k: float = 0.05,
    n_iter: int = 3,
    n_bins: int = 5,
) -> Tuple[torch.Tensor, float]:
    """
    图内自参考检测 -- training-free 版 (v2: L2距离 + 空间邻居一致性)

    核心改进:
      - 用L2距离代替马氏距离(高维下协方差矩阵不稳定)
      - 加入空间邻居比较,检测局部不一致性
      - 图像级分数 = 最异常patch/中位异常 的对数比

    Args:
        features: [N, D] 多路融合后的 patch 特征 (N=grid_size^2)
        texture_energy: [N] 各 patch 的纹理能量
        k: top-k 池化比例
        n_iter: 迭代精化轮数
        n_bins: 纹理能量分桶数

    Returns:
        heatmap: [N] 各 patch 的相对偏离度
        image_score: float 图像级检测分数
    """
    device = features.device
    n = features.shape[0]
    grid_size = int(n ** 0.5)

    # ---- 内容条件归一化: 按纹理能量分桶 ----
    q = torch.quantile(texture_energy, torch.linspace(0, 1, n_bins + 1, device=device))
    bins = torch.bucketize(texture_energy, q[1:-1])

    # ---- 桶内 L2 偏离度 ----
    dev = torch.zeros(n, device=device)
    weights = torch.ones(n, device=device)

    for _ in range(n_iter):
        dev = torch.zeros(n, device=device)
        for b in bins.unique():
            mask = bins == b
            if mask.sum() < 3:
                continue
            x_bin = features[mask]
            # 桶内参考 = 加权中位数 (robust)
            ref = weighted_robust_mean(x_bin, weights[mask])
            # L2 距离
            dev[mask] = ((x_bin - ref) ** 2).sum(dim=-1)

        thr = torch.quantile(dev, 0.90)
        weights = (dev < thr).float()

    # ---- 空间邻居一致性 ----
    dev_2d = dev.reshape(grid_size, grid_size)
    spatial_dev = torch.zeros_like(dev_2d)
    # 上下左右4邻居的最大偏离差
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        shifted = torch.roll(dev_2d, shifts=(dx, dy), dims=(0, 1))
        spatial_dev = torch.maximum(spatial_dev, (dev_2d - shifted).abs())
    # 边界处理
    spatial_dev = spatial_dev.clamp(min=0)

    # 混合: 桶内偏离 + 空间不一致
    dev_combined = dev_2d + 0.5 * spatial_dev
    dev_flat = dev_combined.reshape(-1)

    # ---- 图像级分数: top-k / 中位数的比率 ----
    n_top = max(1, int(k * n))
    top_mean = dev_flat.topk(n_top).values.mean()
    median = dev_flat.median()
    # 异常区域vs背景的比率, 使用log避免量级差异
    image_score = (top_mean / (median + 1e-8)).log().item()

    heatmap = dev_combined.cpu().numpy()

    return heatmap, image_score


def compute_map(heatmap: np.ndarray, mask_gt: np.ndarray,
                thresholds: np.ndarray = None) -> float:
    """计算像素级 mAP"""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)[1:-1]

    heatmap_flat = heatmap.ravel()
    mask_flat = mask_gt.ravel()

    aps = []
    for th in thresholds:
        pred = (heatmap_flat >= th).astype(np.float32)
        tp = (pred * mask_flat).sum()
        fp = (pred * (1 - mask_flat)).sum()
        fn = ((1 - pred) * mask_flat).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        aps.append(prec * rec)

    return np.trapz(aps, thresholds) if len(aps) > 1 else 0.0


@torch.no_grad()
def self_reference_detect_online(
    img: torch.Tensor,
    backbone,
    vae=None,
    k: float = 0.05,
    n_iter: int = 3,
    patch_size: int = 14,
) -> Tuple[np.ndarray, float]:
    """
    在线版自参考检测 -- 直接从图像到输出。
    不依赖预存特征缓存。

    Args:
        img: [1, 3, H, W] 归一化后的图像
        backbone: 冻结的 DINOv2
        vae: 可选, SD VAE

    Returns:
        heatmap: [grid_h, grid_w]
        image_score: float
    """
    from ..utils.highpass import cross_diff_highpass, pool_to_patch_grid, local_texture_energy

    _, _, h, w = img.shape
    feats = []

    # (a) DINOv2
    tok = backbone.forward_features(img)["x_norm_patchtokens"]  # [1, N, C]
    feats.append(F.normalize(tok, dim=-1))

    n_patches = tok.shape[1]
    grid_size = int(n_patches ** 0.5)

    # (b) 高通残差
    hp = cross_diff_highpass(img)
    hp = F.pad(hp, (0, 1, 0, 1))
    hp_pooled = pool_to_patch_grid(hp, n_patches)
    feats.append(hp_pooled)

    # (c) VAE 重建残差
    if vae is not None:
        with torch.autocast("cuda", dtype=torch.float16):
            posterior = vae.encode(img.half()).latent_dist
            rec = vae.decode(posterior.mean).sample
        rec = rec.float()
        vres = (img - rec).abs().mean(dim=1, keepdim=True)
        vres_pooled = pool_to_patch_grid(vres, n_patches)
        feats.append(vres_pooled)

    X = torch.cat(feats, dim=-1).squeeze(0)  # [N, D]

    tex = local_texture_energy(img, patch_size=patch_size).squeeze(0)

    heatmap_flat, image_score = self_reference_detect(X, tex, k=k, n_iter=n_iter)
    heatmap = heatmap_flat.reshape(grid_size, grid_size)

    return heatmap, image_score
