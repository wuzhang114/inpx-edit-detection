"""像素级定位评测: 方向性VAE热力图 vs 弱监督头热力图 vs 真实mask

数据已就绪 (INP-X 图片 + mask), 特征用缓存 (37x37 网格对齐)。
指标: mIoU / F1 (阈值扫描), 按 mask 面积分档 + 按 kind 分。
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.directional_vae import slide_bins, directional_anomaly  # noqa: E402
from baselines.weakly_supervised_v2 import MLPHead  # noqa: E402

CACHE = "D:/lunwen/data/features_cache"
IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
GRID = 37  # 518/14


def load_mask(mask_path: str, grid: int = GRID) -> np.ndarray:
    """读取 mask → 二值 → 最近邻 resize 到 grid"""
    img = Image.open(IMG_ROOT / mask_path).convert("L")
    img = img.resize((grid, grid), Image.NEAREST)
    m = np.asarray(img)
    return (m > 127).astype(np.float32)


def localization_metrics(heatmap: np.ndarray, mask: np.ndarray,
                         thresholds=None) -> dict:
    """mIoU/F1 阈值扫描 (同 evaluation/metrics.py)"""
    if thresholds is None:
        thresholds = np.linspace(heatmap.min(), heatmap.max(), 256)
    best_f1, best_iou = 0.0, 0.0
    for th in thresholds:
        pred = (heatmap >= th).astype(np.float32)
        tp = (pred * mask).sum()
        fp = (pred * (1 - mask)).sum()
        fn = ((1 - pred) * mask).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        best_f1 = max(best_f1, f1)
        best_iou = max(best_iou, iou)
    return {"mIoU": best_iou, "F1": best_f1}


def directional_heatmap(vae, tex, n_bins=48):
    """方向性 VAE 高尾热力图 (最优配置)"""
    bins = slide_bins(tex, n_bins=n_bins)
    anomaly = directional_anomaly(vae, bins, mode="high", n_iter=3)
    return anomaly.numpy().reshape(GRID, GRID)


def weaksup_heatmap(model, dino, hp):
    """弱监督头 patch 分数热力图"""
    model.eval()
    with torch.no_grad():
        X = torch.cat([torch.from_numpy(dino.astype(np.float32)),
                       torch.from_numpy(hp.astype(np.float32))], dim=-1)
        logits = model(X.unsqueeze(0).cuda()).squeeze(0)
        scores = logits.sigmoid().cpu().numpy()
    return scores.reshape(GRID, GRID)


def evaluate(n_samples=300, n_bins=48):
    with open(f"{CACHE}/labels.json") as f:
        labels = json.load(f)
    with open(f"{CACHE}/metadata.json") as f:
        meta = json.load(f)

    n_total, n_patches, dino_dim = meta["n_total"], meta["n_patches"], meta["dino_dim"]
    vae_mm = np.memmap(f"{CACHE}/vae_residual.npy", dtype=np.float32, mode="r",
                       shape=(n_total, n_patches, 1))
    tex_mm = np.memmap(f"{CACHE}/texture_energy.npy", dtype=np.float32, mode="r",
                       shape=(n_total, n_patches))
    dino_mm = np.memmap(f"{CACHE}/dino_tokens.npy", dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, dino_dim))
    hp_mm = np.memmap(f"{CACHE}/highpass.npy", dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))

    # 弱监督头
    head = MLPHead(dino_dim + 1).cuda()
    head.load_state_dict(torch.load("D:/lunwen/outputs/weak_sup_head.pt"))
    head.eval()
    print("弱监督头加载完成")

    # 采样: standard + exchanged (需有 mask)
    by_kind = defaultdict(list)
    for item in labels:
        if item["kind"] in ("standard", "exchanged") and item.get("mask_path"):
            by_kind[item["kind"]].append(item["idx"])

    rng = np.random.RandomState(42)
    eval_idxs = []
    for k in ["standard", "exchanged"]:
        idxs = by_kind.get(k, [])
        if len(idxs) > n_samples:
            idxs = sorted(rng.choice(idxs, n_samples, replace=False))
        eval_idxs.extend(idxs)
    eval_idxs = sorted(eval_idxs)
    print(f"定位评测样本: {len(eval_idxs)} (standard+exchanged)")

    rows = []
    t0 = time.time()
    for idx in tqdm(eval_idxs, desc="定位评测"):
        item = labels[idx]
        mask = load_mask(item["mask_path"])
        if mask.sum() == 0:
            continue
        vae = torch.from_numpy(vae_mm[idx].astype(np.float32)).squeeze(-1)
        tex = torch.from_numpy(tex_mm[idx].astype(np.float32))
        dino = dino_mm[idx].astype(np.float32)
        hp = hp_mm[idx].astype(np.float32)

        hm_dir = directional_heatmap(vae, tex, n_bins=n_bins)
        hm_ws = weaksup_heatmap(head, dino, hp)

        m_dir = localization_metrics(hm_dir, mask)
        m_ws = localization_metrics(hm_ws, mask)
        rows.append({
            "idx": int(idx),
            "kind": item["kind"],
            "cat": item.get("cat", "unknown"),
            "mask_ratio": float(item.get("mask_ratio") or 0.0),
            "size_class": item.get("size_class") or "unknown",
            "dir_mIoU": float(m_dir["mIoU"]), "dir_F1": float(m_dir["F1"]),
            "ws_mIoU": float(m_ws["mIoU"]), "ws_F1": float(m_ws["F1"]),
        })
    print(f"完成! {time.time()-t0:.1f}s")

    # ---- 汇总 ----
    def _agg(rows_sub, name):
        if not rows_sub:
            return
        d_iou = np.mean([r["dir_mIoU"] for r in rows_sub])
        d_f1 = np.mean([r["dir_F1"] for r in rows_sub])
        w_iou = np.mean([r["ws_mIoU"] for r in rows_sub])
        w_f1 = np.mean([r["ws_F1"] for r in rows_sub])
        print(f"{name} (n={len(rows_sub)}): "
              f"方向VAE mIoU={d_iou:.4f} F1={d_f1:.4f} | "
              f"弱监督头 mIoU={w_iou:.4f} F1={w_f1:.4f}")

    print("\n" + "=" * 70)
    _agg(rows, "全部")
    for k in ["standard", "exchanged"]:
        _agg([r for r in rows if r["kind"] == k], f"kind={k}")
    for c in ["CelebAHQ", "CityScapes", "OpenImages", "SUN_RGBD"]:
        _agg([r for r in rows if r["cat"] == c], f"cat={c}")
    print("按mask分档:")
    for sc in ["tiny", "small", "medium", "large"]:
        _agg([r for r in rows if r["size_class"] == sc], f"  {sc}")

    out = Path("D:/lunwen/outputs")
    with open(out / "localization_results.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n已保存: {out / 'localization_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=300)
    p.add_argument("--n_bins", type=int, default=48)
    args = p.parse_args()
    evaluate(n_samples=args.n_samples, n_bins=args.n_bins)
