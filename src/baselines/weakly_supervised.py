"""弱监督版: 冻结特征 + 轻量解码头, mask监督训练

使用缓存的 per-patch 特征, 训练一个轻量 MLP 头做 patch 级二分类。
骨干完全冻结, 可训练参数仅几MB, 单次训练几分钟。
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import classification_metrics


class PatchesDataset(Dataset):
    """从缓存的 per-patch 特征构建训练集"""

    def __init__(self, features_dir: str, idxs: list, labels_data: list,
                 use_dino: bool = True):
        self.features_dir = Path(features_dir)
        self.idxs = idxs
        self.use_dino = use_dino

        with open(self.features_dir / "metadata.json") as f:
            self.meta = json.load(f)

        self.n_total = self.meta["n_total"]
        self.n_patches = self.meta["n_patches"]
        self.dino_dim = self.meta["dino_dim"]

        self.dino = np.memmap(
            str(self.features_dir / "dino_tokens.npy"),
            dtype=np.float16, mode="r",
            shape=(self.n_total, self.n_patches, self.dino_dim),
        )
        self.hp = np.memmap(
            str(self.features_dir / "highpass.npy"),
            dtype=np.float32, mode="r",
            shape=(self.n_total, self.n_patches, 1),
        )

        # 构建 patch 级标签
        # 从 labels 获取 mask_ratio, 但完整 mask 需要原始 mask 文件
        # 替代方案: 用 mask_ratio 做弱标签 (仅知道编辑占比, 不知道具体位置)
        # 或者: 用 inpx_index.json 中的 mask_path 读原 mask
        # 当前策略: 读原始 mask 文件 (需要图片可用)
        self.labels = labels_data

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        idx = self.idxs[i]
        item = self.labels[idx]

        # 特征
        feat_list = []
        if self.use_dino:
            d = self.dino[idx].astype(np.float32)
            feat_list.append(d)
        h = self.hp[idx].astype(np.float32)
        feat_list.append(h)

        feats = np.concatenate(feat_list, axis=-1)  # [N, C]
        label = item["label"]  # 0=real, 1=fake

        return {
            "feats": torch.from_numpy(feats),
            "label": label,
            "idx": int(idx),
            "mask_ratio": float(item.get("mask_ratio") or 0.0),
            "kind": str(item.get("kind", "real")),
            "cat": str(item.get("cat", "")),
            "size_class": str(item.get("size_class", "unknown")),
        }


class LightweightHead(nn.Module):
    """轻量解码头: MLP + 空间一致性"""

    def __init__(self, in_dim: int, hidden: int = 64, n_patches: int = 1369):
        super().__init__()
        grid = int(n_patches ** 0.5)
        self.grid = grid
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        """
        x: [B, N, C] 或 [N, C]
        返回: [B, N] 或 [N] patch 级 logits
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        b, n, c = x.shape
        flat = x.reshape(b * n, c)
        logits = self.net(flat).reshape(b, n)
        return logits.squeeze(0) if b == 1 else logits


class LightweightConvHead(nn.Module):
    """卷积解码头: 1x1 Conv + 空间平滑"""

    def __init__(self, in_dim: int, hidden: int = 32, n_patches: int = 1369):
        super().__init__()
        grid = int(n_patches ** 0.5)
        self.grid = grid
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        b, n, c = x.shape
        spatial = x.reshape(b, self.grid, self.grid, c).permute(0, 3, 1, 2)
        logits = self.conv(spatial).reshape(b, -1)
        return logits.squeeze(0) if b == 1 else logits


def train_weakly_supervised(
    features_dir: str,
    output_dir: str,
    use_dino: bool = True,
    use_conv: bool = False,
    hidden: int = 64,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """训练弱监督 patch 级分类器"""
    features_dir = Path(features_dir)

    with open(features_dir / "labels.json") as f:
        labels_data = json.load(f)
    with open(features_dir / "metadata.json") as f:
        meta = json.load(f)

    # 划分 (用和 D4 linear probe 相同的划分)
    # 跨数据集: train on CityScapes+OpenImages+SUN, test on CelebAHQ
    cats = np.array([l.get("cat", "unknown") for l in labels_data])
    train_cats = ["CityScapes", "OpenImages", "SUN_RGBD"]
    test_cats = ["CelebAHQ"]

    train_mask = np.isin(cats, train_cats)
    test_mask = np.isin(cats, test_cats)

    all_idxs = np.arange(len(labels_data))
    train_idxs = all_idxs[train_mask]
    test_idxs = all_idxs[test_mask]

    rng = np.random.RandomState(42)
    rng.shuffle(train_idxs)
    n_val = int(len(train_idxs) * 0.1)
    val_idxs = train_idxs[:n_val]
    train_idxs = train_idxs[n_val:]

    print(f"Train: {len(train_idxs)}, Val: {len(val_idxs)}, Test: {len(test_idxs)}")

    # 数据集
    train_ds = PatchesDataset(features_dir, train_idxs, labels_data, use_dino=use_dino)
    val_ds = PatchesDataset(features_dir, val_idxs, labels_data, use_dino=use_dino)
    test_ds = PatchesDataset(features_dir, test_idxs, labels_data, use_dino=use_dino)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # 模型
    in_dim = (meta["dino_dim"] if use_dino else 0) + 1  # +1 for hp
    n_patches = meta["n_patches"]
    if use_conv:
        model = LightweightConvHead(in_dim, hidden=hidden, n_patches=n_patches)
    else:
        model = LightweightHead(in_dim, hidden=hidden, n_patches=n_patches)

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {param_count:,} ({param_count*4/1e6:.1f} MB)")
    print(f"输入维度: {in_dim}, patches: {n_patches}")

    # 损失: 图像级 BCE (没有 patch 级 mask 时的替代方案)
    # 实际上我们需要 mask 来做 patch 级监督
    # 临时方案: 用图像级标签做弱监督 (pooling后二分类)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            feats = batch["feats"].to(device)  # [B, N, C]
            labels = batch["label"].float().to(device)  # [B]

            # forward
            patch_logits = model(feats)  # [B, N]

            # 图像级预测: max pool over patches
            img_scores = patch_logits.sigmoid().max(dim=1).values  # [B]

            # BCE loss
            loss = F.binary_cross_entropy(img_scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(img_scores.detach().cpu().tolist())
            train_targets.extend(labels.cpu().tolist())

        scheduler.step()

        # 验证
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                feats = batch["feats"].to(device)
                labels = batch["label"].float()

                patch_logits = model(feats)
                img_scores = patch_logits.sigmoid().max(dim=1).values

                val_preds.extend(img_scores.cpu().tolist())
                val_targets.extend(labels.tolist())

        val_metrics = classification_metrics(
            np.array(val_targets), np.array(val_preds)
        )

        if val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: loss={train_loss/len(train_loader):.4f}, "
                  f"val Acc={val_metrics['Acc']:.4f}, AUC={val_metrics['AUC']:.4f}, "
                  f"R.Acc={val_metrics['R.Acc']:.4f}, F.Acc={val_metrics['F.Acc']:.4f}")

    # 加载最佳模型
    if best_state:
        model.load_state_dict(best_state)

    # 测试
    print(f"\n{'='*60}")
    print("测试集结果")

    def evaluate_split(loader, name):
        model.eval()
        all_preds = []
        all_targets = []
        all_cats = []
        all_sizes = []

        with torch.no_grad():
            for batch in loader:
                feats = batch["feats"].to(device)
                labels = batch["label"].float()

                patch_logits = model(feats)
                img_scores = patch_logits.sigmoid().max(dim=1).values

                all_preds.extend(img_scores.cpu().tolist())
                all_targets.extend(labels.tolist())
                all_cats.extend(batch["cat"])
                all_sizes.extend(batch["size_class"])

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        metrics = classification_metrics(y_true, y_pred)

        print(f"\n{name} (n={len(y_true)}):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # 按数据集
        for cat in sorted(set(all_cats)):
            cmask = np.array([c == cat for c in all_cats])
            if cmask.sum() < 2:
                continue
            cm = classification_metrics(y_true[cmask], y_pred[cmask])
            print(f"    {cat}: Acc={cm['Acc']:.4f}, AUC={cm['AUC']:.4f}, "
                  f"R.Acc={cm['R.Acc']:.4f}, F.Acc={cm['F.Acc']:.4f}")

        # 按mask面积
        mask_real = y_true == 1
        for sc in ["tiny", "small", "medium", "large"]:
            name_map = {"tiny": "极小(<2%)", "small": "小(2-5%)",
                       "medium": "中(5-15%)", "large": "大(>15%)"}
            smask = np.array([s == sc for s in all_sizes]) & mask_real
            if smask.sum() < 5:
                continue
            facc = (y_pred[smask] >= 0.5).mean()
            print(f"    {name_map[sc]}: n={smask.sum()}, F.Acc={facc:.4f}")

        return metrics

    evaluate_split(test_loader, "Test (CelebAHQ)")
    val_metrics = evaluate_split(val_loader, "Val (in-domain)")

    # 保存
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "weak_sup_head.pt")
        print(f"\n模型已保存: {output_dir / 'weak_sup_head.pt'}")

    return val_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str,
                        default="D:/lunwen/data/features_cache")
    parser.add_argument("--output_dir", type=str,
                        default="D:/lunwen/outputs")
    parser.add_argument("--use_dino", action="store_true", default=True)
    parser.add_argument("--use_conv", action="store_true")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_weakly_supervised(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        use_dino=args.use_dino,
        use_conv=args.use_conv,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
