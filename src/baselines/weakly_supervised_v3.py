"""弱监督 v3: 像素级 mask 监督训练 (冻结骨干 + 轻量 MLP 解码头)

与 v2 的区别: v2 用图像级 max-pool 弱监督 (loss 只在 max 分数上),
v3 对每个 patch 用真实 mask 做 BCE+Dice 监督 → patch 分数直接对应编辑区。

- 输入: DINOv2 token + 高通残差 (缓存, 冻结)
- 输出: [B, 1369] patch logits
- 监督: mask (37x37 网格, 预加载到内存); real 图像用全零 mask
- 划分: 与 v2 相同 (train = CityScapes/OpenImages/SUN, test = CelebAHQ 跨域)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

CACHE = "D:/lunwen/data/features_cache"
IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
OUT = Path("D:/lunwen/outputs")
GRID = 37
N_PATCHES = 1369


class MLPHead(nn.Module):
    """同 v2 的轻量解码头 (约30K参数)"""

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        b, n, c = x.shape
        return self.net(x.reshape(b * n, c)).reshape(b, n)


def load_mask(mask_path: str) -> np.ndarray:
    img = Image.open(IMG_ROOT / mask_path).convert("L")
    img = img.resize((GRID, GRID), Image.NEAREST)
    return (np.asarray(img) > 127).astype(np.float32)


def build_dataset(labels_data, meta, n_patches, dino_dim):
    """预加载所有训练/测试样本的 mask (real 用全零)"""
    n_total = meta["n_total"]
    masks = np.zeros((n_total, n_patches), dtype=np.float32)
    n_mask = 0
    for item in labels_data:
        mp = item.get("mask_path")
        if mp and (IMG_ROOT / mp).exists():
            masks[item["idx"]] = load_mask(mp).ravel()
            n_mask += 1
    print(f"mask 预加载完成: {n_mask}/{n_total} 样本有 mask (real 用全零)")
    return masks


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1) + eps
    return 1 - (2 * inter / denom).mean()


def train_v3(epochs=20, batch_size=64, lr=1e-3, hidden=64):
    with open(f"{CACHE}/labels.json") as f:
        labels_data = json.load(f)
    with open(f"{CACHE}/metadata.json") as f:
        meta = json.load(f)
    n_total, n_patches, dino_dim = meta["n_total"], meta["n_patches"], meta["dino_dim"]
    assert n_patches == N_PATCHES

    dino_mm = np.memmap(f"{CACHE}/dino_tokens.npy", dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, dino_dim))
    hp_mm = np.memmap(f"{CACHE}/highpass.npy", dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))

    cats = np.array([l.get("cat", "unknown") for l in labels_data])
    labels_arr = np.array([l["label"] for l in labels_data])
    all_idx = np.arange(n_total)
    test_mask = cats == "CelebAHQ"
    train_idx = all_idx[~test_mask]
    test_idx = all_idx[test_mask]

    rng = np.random.RandomState(42)
    rng.shuffle(train_idx)
    n_val = int(len(train_idx) * 0.1)
    val_idx = train_idx[:n_val]
    train_idx = train_idx[n_val:]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test(CelebAHQ): {len(test_idx)}")

    masks = build_dataset(labels_data, meta, n_patches, dino_dim)

    model = MLPHead(dino_dim + 1, hidden=hidden).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,} ({n_params*4/1024:.0f} KB)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def load_batch(idxs):
        dino = torch.from_numpy(dino_mm[idxs].astype(np.float32))
        hp = torch.from_numpy(hp_mm[idxs].astype(np.float32))
        X = torch.cat([dino, hp], dim=-1).cuda()
        y = torch.from_numpy(masks[idxs]).cuda()
        return X, y

    def predict_patches(idxs):
        model.eval()
        all_scores, all_masks = [], []
        with torch.no_grad():
            for i in range(0, len(idxs), batch_size):
                b_idx = idxs[i:i + batch_size]
                X, y = load_batch(b_idx)
                logits = model(X)
                all_scores.append(torch.sigmoid(logits).cpu().numpy())
                all_masks.append(y.cpu().numpy())
        return np.concatenate(all_scores), np.concatenate(all_masks)

    best_val = 0
    best_state = None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_idx)
        losses = []
        pbar = tqdm(range(0, len(train_idx), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for start in pbar:
            b_idx = train_idx[start:start + batch_size]
            X, y = load_batch(b_idx)
            logits = model(X)
            bce = F.binary_cross_entropy_with_logits(logits, y)
            dice = dice_loss(logits, y)
            loss = bce + 0.5 * dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{np.mean(losses[-20:]):.4f}"})
        sched.step()

        # val: patch mIoU + 图像级 AUC (max池化)
        scores, masks_v = predict_patches(val_idx)
        img_scores = scores.max(axis=1)
        img_auc = roc_auc_score(labels_arr[val_idx], img_scores)
        # patch 级 mIoU (阈值扫描)
        best_miou = 0
        for th in np.linspace(0.1, 0.9, 17):
            pred = (scores >= th).astype(np.float32)
            tp = (pred * masks_v).sum()
            fp = (pred * (1 - masks_v)).sum()
            fn = ((1 - pred) * masks_v).sum()
            iou = tp / (tp + fp + fn + 1e-8)
            best_miou = max(best_miou, iou)
        print(f"  ep{epoch+1}: loss={np.mean(losses):.4f} "
              f"val imgAUC={img_auc:.4f} patch mIoU={best_miou:.4f}")
        if img_auc > best_val:
            best_val = img_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\n训练完成! 耗时 {(time.time()-t0)/60:.1f} 分钟")
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT / "weak_sup_v3_head.pt")
    print(f"模型: {OUT / 'weak_sup_v3_head.pt'}")

    # ---- 测试 (CelebAHQ 跨域) ----
    print("\n" + "=" * 60)
    print("测试 (CelebAHQ 跨域)")
    scores_t, masks_t = predict_patches(test_idx)
    img_scores_t = scores_t.max(axis=1)
    y_t = labels_arr[test_idx]

    auc = roc_auc_score(y_t, img_scores_t)
    thr = np.median(img_scores_t)
    yp = (img_scores_t >= thr).astype(int)
    print(f"图像级: Acc={accuracy_score(y_t, yp):.4f} AUC={auc:.4f} "
          f"R.Acc={accuracy_score(y_t[y_t==0], yp[y_t==0]):.4f} "
          f"F.Acc={accuracy_score(y_t[y_t==1], yp[y_t==1]):.4f}")

    # 定位 (仅 fake)
    fake_mask = y_t == 1
    best_miou_all, best_f1_all = 0, 0
    for th in np.linspace(0.05, 0.95, 91):
        pred = (scores_t[fake_mask] >= th).astype(np.float32)
        m = masks_t[fake_mask]
        tp = (pred * m).sum(); fp = (pred * (1 - m)).sum(); fn = ((1 - pred) * m).sum()
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        best_miou_all = max(best_miou_all, iou)
        best_f1_all = max(best_f1_all, f1)
    print(f"定位(仅fake): mIoU={best_miou_all:.4f} F1={best_f1_all:.4f}")

    # 分档 (fake)
    sizes = np.array([l.get("size_class") or "unknown" for l in labels_data])[test_idx][fake_mask]
    for sc in ["tiny", "small", "medium", "large"]:
        mm = sizes == sc
        if mm.sum() < 5:
            continue
        best_iou = 0
        for th in np.linspace(0.05, 0.95, 91):
            pred = (scores_t[fake_mask][mm] >= th).astype(np.float32)
            m = masks_t[fake_mask][mm]
            tp = (pred * m).sum(); fp = (pred * (1 - m)).sum(); fn = ((1 - pred) * m).sum()
            iou = tp / (tp + fp + fn + 1e-8)
            best_iou = max(best_iou, iou)
        print(f"  {sc} (n={mm.sum()}): mIoU={best_iou:.4f}")

    with open(OUT / "weak_sup_v3_results.json", "w") as f:
        json.dump({
            "img_auc": float(auc),
            "img_acc": float(accuracy_score(y_t, yp)),
            "r_acc": float(accuracy_score(y_t[y_t == 0], yp[y_t == 0])),
            "f_acc": float(accuracy_score(y_t[y_t == 1], yp[y_t == 1])),
            "loc_miou": float(best_miou_all),
            "loc_f1": float(best_f1_all),
        }, f, indent=1)
    print(f"结果: {OUT / 'weak_sup_v3_results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    args = p.parse_args()
    train_v3(epochs=args.epochs, batch_size=args.batch_size,
             lr=args.lr, hidden=args.hidden)
