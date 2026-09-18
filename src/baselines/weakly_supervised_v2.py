"""弱监督版 v2: 高效批训练

优化: 用memmap顺序批量加载代替DataLoader随机访问。
"""
import json, sys, argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import classification_metrics


def load_batch_features(dino_memmap, hp_memmap, indices, device, use_dino=True):
    """顺序加载一批样本的特征"""
    dino_batch = dino_memmap[indices]  # [B, N, C] float16
    hp_batch = hp_memmap[indices]      # [B, N, 1] float32

    feats = [torch.from_numpy(dino_batch.astype(np.float32))] if use_dino else []
    feats.append(torch.from_numpy(hp_batch.astype(np.float32)))
    X = torch.cat(feats, dim=-1).to(device)  # [B, N, C]
    return X


class MLPHead(nn.Module):
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


def train_weakly_supervised_v2(
    features_dir="D:/lunwen/data/features_cache",
    output_dir="D:/lunwen/outputs",
    epochs=15,
    batch_size=64,
    lr=1e-3,
    hidden=64,
    max_train=10000,
    device="cuda",
):
    features_dir = Path(features_dir)
    with open(features_dir / "labels.json") as f:
        labels_data = json.load(f)
    with open(features_dir / "metadata.json") as f:
        meta = json.load(f)

    n_total = meta["n_total"]
    n_patches = meta["n_patches"]
    dino_dim = meta["dino_dim"]

    dino_mm = np.memmap(str(features_dir / "dino_tokens.npy"),
                        dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, dino_dim))
    hp_mm = np.memmap(str(features_dir / "highpass.npy"),
                      dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))

    # 划分
    cats = np.array([l.get("cat", "unknown") for l in labels_data])
    labels_arr = np.array([l["label"] for l in labels_data])
    sizes = np.array([l.get("size_class", "unknown") for l in labels_data])
    kinds = np.array([l["kind"] for l in labels_data])

    all_idx = np.arange(n_total)
    test_mask = np.isin(cats, ["CelebAHQ"])
    train_mask = ~test_mask
    train_idx = all_idx[train_mask]
    test_idx = all_idx[test_mask]

    rng = np.random.RandomState(42)
    rng.shuffle(train_idx)
    n_val = int(len(train_idx) * 0.1)
    val_idx = train_idx[:n_val]
    train_idx = train_idx[n_val:]

    if max_train and len(train_idx) > max_train:
        train_idx = train_idx[:max_train]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Input dim: {dino_dim + 1}, Patches: {n_patches}, Batch: {batch_size}")

    in_dim = dino_dim + 1
    model = MLPHead(in_dim, hidden=hidden).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,} "
          f"({sum(p.numel() for p in model.parameters())*4/1024:.0f} KB)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def predict(idxs):
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for i in range(0, len(idxs), batch_size):
                b_idx = idxs[i:i + batch_size]
                X = load_batch_features(dino_mm, hp_mm, b_idx, device)
                logits = model(X)
                scores = logits.sigmoid().max(dim=1).values
                preds.extend(scores.cpu().tolist())
                targets.extend(labels_arr[b_idx].tolist())
        return np.array(targets), np.array(preds)

    best_val_auc = 0
    best_state = None
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_idx)
        losses = []

        pbar = tqdm(range(0, len(train_idx), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for start in pbar:
            b_idx = train_idx[start:start + batch_size]
            X = load_batch_features(dino_mm, hp_mm, b_idx, device)
            y = torch.from_numpy(labels_arr[b_idx]).float().to(device)

            logits = model(X)
            scores = logits.sigmoid().max(dim=1).values
            loss = F.binary_cross_entropy(scores, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{np.mean(losses[-20:]):.4f}"})

        sched.step()

        # Val metrics every epoch
        yt, yp = predict(val_idx)
        vm = classification_metrics(yt, yp)
        if vm["AUC"] > best_val_auc:
            best_val_auc = vm["AUC"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  train_loss={np.mean(losses):.4f}, "
              f"val Acc={vm['Acc']:.4f} AUC={vm['AUC']:.4f} "
              f"R.Acc={vm['R.Acc']:.4f} F.Acc={vm['F.Acc']:.4f}")

    elapsed = time.time() - t0
    print(f"\n训练完成, 耗时: {elapsed:.1f}s")

    if best_state:
        model.load_state_dict(best_state)

    # 测试
    print(f"\n{'='*50}")
    print(" 测试结果")
    print(f"{'='*50}")

    def evaluate(name, idxs):
        yt, yp = predict(idxs)
        m = classification_metrics(yt, yp)
        print(f"\n{name} (n={len(yt)}): Acc={m['Acc']:.4f} AUC={m['AUC']:.4f} "
              f"R.Acc={m['R.Acc']:.4f} F.Acc={m['F.Acc']:.4f}")

        cats_test = cats[idxs]
        sizes_test = sizes[idxs]
        kinds_test = kinds[idxs]

        for c in sorted(set(cats_test)):
            cm = np.array([c2 == c for c2 in cats_test])
            if cm.sum() < 2:
                continue
            cmt = classification_metrics(yt[cm], yp[cm])
            print(f"  {c}: Acc={cmt['Acc']:.4f} AUC={cmt['AUC']:.4f} "
                  f"R.Acc={cmt['R.Acc']:.4f} F.Acc={cmt['F.Acc']:.4f}")

        for sc in ["tiny", "small", "medium", "large"]:
            nm = {"tiny": "极小", "small": "小", "medium": "中", "large": "大"}
            sm = (np.array([s == sc for s in sizes_test])) & (yt == 1)
            if sm.sum() < 5:
                continue
            facc = (yp[sm] >= 0.5).mean()
            print(f"  {nm[sc]}mask: n={sm.sum()} F.Acc={facc:.4f}")

        return m

    evaluate("Test (CelebAHQ)", test_idx)
    evaluate("Val (in-domain)", val_idx)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_dir / "weak_sup_head.pt")
        print(f"\n模型: {output_dir / 'weak_sup_head.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max_train", type=int, default=10000)
    args = parser.parse_args()

    train_weakly_supervised_v2(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        max_train=args.max_train,
    )
