"""D4: 冻结 VFM 线性探针基线

在预抽取的 DINOv2 特征上训练线性二分类器。
用 sklearn SGDClassifier + partial_fit 高效处理 59GB memmap。

输出:
  - 图像级 Acc / AUC / R.Acc / F.Acc
  - 按 mask 面积分档报告
  - 跨数据集/跨编辑器泛化
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluation.metrics import classification_metrics


def load_labels(label_path: str):
    with open(label_path) as f:
        return json.load(f)


def build_split(labels, train_cats=None, test_cats=None, val_ratio=0.1):
    """构建数据集划分。

    默认策略:
      - 跨数据集: train on CityScapes+OpenImages+SUN-RGBD, test on CelebAHQ
      - 跨编辑器: train on 2 inpainter, test on 3rd (需 model 字段)
      - 随机划分 (fallback)
    """
    idxs = np.arange(len(labels))
    cats = np.array([l.get("cat", "unknown") for l in labels])
    kinds = np.array([l.get("kind", "real") for l in labels])
    models = np.array([l.get("model", "unknown") for l in labels])
    labels_arr = np.array([l["label"] for l in labels])
    mask_ratios = np.array([l.get("mask_ratio", 0) for l in labels])
    size_classes = np.array([l.get("size_class", "unknown") for l in labels])

    if train_cats is None:
        train_cats = ["CityScapes", "OpenImages", "SUN_RGBD"]
    if test_cats is None:
        test_cats = ["CelebAHQ"]

    train_mask = np.isin(cats, train_cats)
    test_mask = np.isin(cats, test_cats)
    unused = ~(train_mask | test_mask)

    # val from train
    rng = np.random.RandomState(42)
    train_idxs = idxs[train_mask]
    rng.shuffle(train_idxs)
    n_val = int(len(train_idxs) * val_ratio)
    val_idxs = train_idxs[:n_val]
    train_idxs = train_idxs[n_val:]

    test_idxs = idxs[test_mask]

    return {
        "train": train_idxs,
        "val": val_idxs,
        "test": test_idxs,
        "labels": labels_arr,
        "cats": cats,
        "kinds": kinds,
        "models": models,
        "mask_ratios": mask_ratios,
        "size_classes": size_classes,
    }


def load_batched(dino_memmap, idxs, batch_size=5000):
    """按 batch 加载 memmap 切片，做 mean pooling -> [n, dino_dim]"""
    all_pooled = []
    for start in range(0, len(idxs), batch_size):
        batch_idxs = idxs[start:start + batch_size]
        tokens = dino_memmap[batch_idxs]  # [B, N_patches, C]
        pooled = tokens.mean(axis=1).astype(np.float32)
        all_pooled.append(pooled)
    return np.concatenate(all_pooled, axis=0)


def train_linear_probe(features_dir: str, output_dir: str = None):
    """训练线性探针并评测"""
    features_dir = Path(features_dir)

    print("加载标签...")
    labels_data = load_labels(features_dir / "labels.json")
    print(f"  总样本: {len(labels_data)}")

    print("加载特征 memmap...")
    dino_memmap = np.memmap(
        str(features_dir / "dino_tokens.npy"),
        dtype=np.float16,
        mode="r",
        shape=(len(labels_data), 1369, 384),
    )

    # ---- 划分 ----
    print("\n构建数据划分...")
    split = build_split(labels_data)
    for name in ["train", "val", "test"]:
        idxs = split[name]
        lbs = split["labels"][idxs]
        cats = split["cats"][idxs]
        print(f"  {name}: {len(idxs)} 样本, real={sum(lbs==0)}, fake={sum(lbs==1)}")
        for c in np.unique(cats):
            print(f"    {c}: {sum(cats==c)}")

    # ---- 加载特征 ----
    print("\n加载训练特征...")
    X_train = load_batched(dino_memmap, split["train"])
    y_train = split["labels"][split["train"]]
    print(f"  X_train: {X_train.shape}, {X_train.nbytes/1e9:.2f} GB")

    print("加载验证特征...")
    X_val = load_batched(dino_memmap, split["val"])
    y_val = split["labels"][split["val"]]

    print("加载测试特征...")
    X_test = load_batched(dino_memmap, split["test"])
    y_test = split["labels"][split["test"]]

    # ---- 训练 ----
    print("\n标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("训练 SGDClassifier (linear probe)...")
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        verbose=1,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    # ---- 评测: 图像级 ----
    def evaluate(name, X, y, mask_ratios, size_classes, kinds):
        y_proba = clf.predict_proba(X)[:, 1]
        y_pred = clf.predict(X)

        metrics_all = classification_metrics(y, y_proba)
        print(f"\n{'='*50}")
        print(f"  {name} (n={len(y)})")
        for k, v in metrics_all.items():
            print(f"    {k}: {v:.4f}")

        # 按 mask 面积分档 (仅 fake 样本)
        print(f"\n  按 mask 面积分档 (仅 edited 样本, n={sum(y==1)}):")
        fake_mask = y == 1
        for bin_name in ["tiny", "small", "medium", "large"]:
            bin_map = {"tiny": "极小(<2%)", "small": "小(2-5%)", "medium": "中(5-15%)", "large": "大(>15%)"}
            bin_show = bin_map.get(bin_name, bin_name)
            bin_mask = fake_mask & (size_classes == bin_name) if isinstance(size_classes[0], str) else fake_mask
            if bin_mask.sum() == 0:
                continue
            bin_acc = accuracy_score(y[bin_mask], y_pred[bin_mask])
            bin_fake_acc = accuracy_score(
                y[bin_mask], y_pred[bin_mask]
            )
            print(f"    {bin_show}: n={bin_mask.sum()}, Acc={bin_fake_acc:.4f}")

        # 按 kind 分
        print(f"\n  按图像类型:")
        for k in np.unique(kinds):
            kmask = kinds == k
            if kmask.sum() == 0:
                continue
            kacc = accuracy_score(y[kmask], y_pred[kmask])
            kreal_acc = accuracy_score(y[kmask & (y==0)], y_pred[kmask & (y==0)]) if sum(kmask & (y==0))>0 else 0
            kfake_acc = accuracy_score(y[kmask & (y==1)], y_pred[kmask & (y==1)]) if sum(kmask & (y==1))>0 else 0
            print(f"    {k}: n={kmask.sum()}, Acc={kacc:.4f}, R.Acc={kreal_acc:.4f}, F.Acc={kfake_acc:.4f}")

        return metrics_all

    evaluate("Train", X_train_scaled, y_train, split["mask_ratios"][split["train"]],
             split["size_classes"][split["train"]], split["kinds"][split["train"]])
    evaluate("Val", X_val_scaled, y_val, split["mask_ratios"][split["val"]],
             split["size_classes"][split["val"]], split["kinds"][split["val"]])
    test_metrics = evaluate("Test", X_test_scaled, y_test, split["mask_ratios"][split["test"]],
                             split["size_classes"][split["test"]], split["kinds"][split["test"]])

    # ---- 保存结果 ----
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "linear_probe_results.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\n结果已保存到 {output_dir}")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str,
                        default="D:/lunwen/data/features_cache")
    parser.add_argument("--output_dir", type=str,
                        default="D:/lunwen/outputs")
    args = parser.parse_args()

    train_linear_probe(args.features_dir, args.output_dir)
