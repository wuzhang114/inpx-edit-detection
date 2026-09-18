"""高通残差滤波与纹理能量估计"""
import torch
import torch.nn.functional as F


def cross_diff_highpass(img: torch.Tensor) -> torch.Tensor:
    """
    交叉差分高通滤波，来自 INP-X 论文的频谱分析滤波器。
    CD[i,j] = |I[i,j] - I[i+1,j] - I[i,j+1] + I[i+1,j+1]|

    Args:
        img: [B, C, H, W] 或 [C, H, W]，值域任意

    Returns:
        [B, 1, H-1, W-1] 或 [1, H-1, W-1]
    """
    no_batch = img.dim() == 3
    if no_batch:
        img = img.unsqueeze(0)

    b, c, h, w = img.shape
    cd = (img[:, :, :-1, :-1] - img[:, :, 1:, :-1]
          - img[:, :, :-1, 1:] + img[:, :, 1:, 1:]).abs()

    hp = cd.mean(dim=1, keepdim=True)

    if no_batch:
        hp = hp.squeeze(0)
    return hp


def local_texture_energy(img: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    计算图像局部纹理能量，用于内容条件归一化中的分桶。

    Args:
        img: [B, 3, H, W]
        patch_size: patch 大小

    Returns:
        tex: [B, num_patches] 每个 patch 的纹理能量
    """
    b, _, h, w = img.shape
    hp = cross_diff_highpass(img)  # [B, 1, H-1, W-1]
    hp = F.pad(hp, (0, 1, 0, 1))  # 恢复原始尺寸

    num_h = h // patch_size
    num_w = w // patch_size
    hp = hp[:, :, :num_h * patch_size, :num_w * patch_size]
    hp_patches = hp.reshape(b, 1, num_h, patch_size, num_w, patch_size)
    hp_patches = hp_patches.permute(0, 1, 2, 4, 3, 5).reshape(b, 1, num_h * num_w, patch_size * patch_size)

    tex = hp_patches.mean(dim=-1).reshape(b, num_h * num_w)
    return tex


def pool_to_patch_grid(feature_map: torch.Tensor, num_patches: int) -> torch.Tensor:
    """
    将任意分辨率特征图池化到 patch 网格。

    Args:
        feature_map: [B, C, H, W] 或 [C, H, W]
        num_patches: 目标 patch 数 (N = H'/P * W'/P)

    Returns:
        [N, C]
    """
    no_batch = feature_map.dim() == 3
    if no_batch:
        feature_map = feature_map.unsqueeze(0)

    b, c, h, w = feature_map.shape
    grid_size = int(num_patches ** 0.5)
    pooled = F.adaptive_avg_pool2d(feature_map, (grid_size, grid_size))
    pooled = pooled.reshape(b, c, -1).permute(0, 2, 1)
    if no_batch:
        pooled = pooled.squeeze(0)
    return pooled
