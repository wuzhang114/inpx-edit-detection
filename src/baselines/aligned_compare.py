"""D5 对比: 格式对齐前后, 同一批样本上线性探针/弱监督的表现差异

流程:
1. 从 index_aligned.json (对齐子集 800 对) 通过 standard_path 匹配 features_cache 的 labels
2. 用旧特征 (features_cache) 在子集上训线性探针, 测试同子集
3. 用对齐特征 (features_cache_v2, 需先抽取) 训同一探针, 对比
4. 输出: Acc/AUC/R.Acc/F.Acc + 样本数 (对齐前后)

注意: 本脚本只做"旧特征"部分与匹配逻辑; 对齐特征需先运行
extract_features.py --index_file INP-X_aligned/index_aligned.json --data_root INP-X_aligned
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

CACHE_OLD = "D:/lunwen/data/features_cache"
CACHE_NEW = "D:/lunwen/data/features_cache_v2"
ALIGNED_INDEX = "D:/lunwen/data/INP-X_aligned/index_aligned.json"


def build_subset_indices():
    """返回: 旧缓存中与对齐子集对应的 idx 列表 (编辑图 + 真实图)"""
    aligned_root = Path("D:/lunwen/data/INP-X_aligned")
    with open(ALIGNED_INDEX) as f:
        aligned = json.load(f)
    with open(f"{CACHE_OLD}/labels.json") as f:
        labels = json.load(f)

    path2idx = {l["path"]: i for i, l in enumerate(labels)}
    subset = []
    missing = 0
    # 编辑图: 只取 train-data (旧缓存只有 train-data)
    for s in aligned:
        if s.get("split") != "train-data":
            continue
        std = s.get("standard_path")
        if not std:
            continue
        if std in path2idx:
            subset.append(path2idx[std])
        else:
            missing += 1
    # 真实图: 扫描 aligned originals 目录 (路径与旧缓存一致)
    for cat in ["CelebAHQ", "CityScapes", "OpenImages", "SUN_RGBD"]:
        d = aligned_root / "train-data" / "data" / "originals" / cat
        if not d.is_dir():
            continue
        for p in d.glob("*.jpg"):
            rel = str(Path("train-data") / "data" / "originals" / cat / p.name)
            if rel in path2idx:
                subset.append(path2idx[rel])
    n_train = sum(1 for s in aligned if s.get("split") == "train-data")
    print(f"对齐子集 train-data {n_train} 条, 旧缓存匹配编辑图 {len(subset)} (缺失 {missing})")
    subset = sorted(set(subset))
    return subset


def linear_probe_eval(cache_dir, idxs, labels_data, n_patches, dino_dim):
    """在给定子集上训练+测试线性探针 (同分布内 5-fold 简单划分)"""
    dino = np.memmap(f"{cache_dir}/dino_tokens.npy", dtype=np.float16, mode="r",
                     shape=(len(labels_data), n_patches, dino_dim))
    hp = np.memmap(f"{cache_dir}/highpass.npy", dtype=np.float32, mode="r",
                   shape=(len(labels_data), n_patches, 1))

    X, y = [], []
    for i in idxs:
        d = dino[i].astype(np.float32)  # [N, C]
        h = hp[i].astype(np.float32)    # [N, 1]
        # 池化到图像级: mean
        X.append(np.concatenate([d.mean(0), h.mean(0)]))
        y.append(labels_data[i]["label"])
    X = np.stack(X)
    y = np.array(y)
    print(f"  [{cache_dir.split('/')[-1]}] 样本: {len(y)}, 正例: {y.sum()}")

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(y))
    n_test = len(y) // 5
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    clf = LogisticRegression(max_iter=500)
    clf.fit(X[train_idx], y[train_idx])
    proba = clf.predict_proba(X[test_idx])[:, 1]
    yt = y[test_idx]
    auc = roc_auc_score(yt, proba)
    thr = np.median(proba)
    yp = (proba >= thr).astype(int)
    acc = accuracy_score(yt, yp)
    r_acc = accuracy_score(yt[yt == 0], yp[yt == 0]) if (yt == 0).sum() else -1
    f_acc = accuracy_score(yt[yt == 1], yp[yt == 1]) if (yt == 1).sum() else -1
    print(f"    线性探针: Acc={acc:.4f} AUC={auc:.4f} R.Acc={r_acc:.4f} F.Acc={f_acc:.4f}")
    return {"Acc": acc, "AUC": auc, "R.Acc": r_acc, "F.Acc": f_acc}


if __name__ == "__main__":
    subset = build_subset_indices()
    with open(f"{CACHE_OLD}/metadata.json") as f:
        meta = json.load(f)
    n_total, n_patches, dino_dim = meta["n_total"], meta["n_patches"], meta["dino_dim"]
    with open(f"{CACHE_OLD}/labels.json") as f:
        labels_data = json.load(f)

    print("\n=== 旧特征 (未对齐) ===")
    r_old = linear_probe_eval(CACHE_OLD, subset, labels_data, n_patches, dino_dim)

    print("\n=== 对齐特征 (需先抽取) ===")
    if Path(f"{CACHE_NEW}/labels.json").exists():
        with open(f"{CACHE_NEW}/metadata.json") as f:
            meta_new = json.load(f)
        # 对齐缓存 labels 的 path 指向 INP-X_aligned
        with open(f"{CACHE_NEW}/labels.json") as f:
            labels_new = json.load(f)
        path2idx_new = {l["path"]: i for i, l in enumerate(labels_new)}
        aligned_root = Path("D:/lunwen/data/INP-X_aligned")
        subset_new = []
        with open(ALIGNED_INDEX) as f:
            aligned = json.load(f)
        for s in aligned:
            if s.get("split") != "train-data":
                continue
            std = s.get("standard_path")
            if std and std in path2idx_new:
                subset_new.append(path2idx_new[std])
        for cat in ["CelebAHQ", "CityScapes", "OpenImages", "SUN_RGBD"]:
            d = aligned_root / "train-data" / "data" / "originals" / cat
            if not d.is_dir():
                continue
            for p in d.glob("*.jpg"):
                rel = str(Path("train-data") / "data" / "originals" / cat / p.name)
                if rel in path2idx_new:
                    subset_new.append(path2idx_new[rel])
        print(f"对齐缓存匹配: {len(set(subset_new))}")
        r_new = linear_probe_eval(CACHE_NEW, sorted(set(subset_new)),
                                  labels_new, meta_new["n_patches"], meta_new["dino_dim"])
        print("\n=== 对比 ===")
        for k in r_old:
            print(f"  {k}: 旧={r_old[k]:.4f} 新={r_new[k]:.4f} Δ={r_new[k]-r_old[k]:+.4f}")
    else:
        print("features_cache_v2 不存在, 跳过 (先运行 extract_features.py)")
