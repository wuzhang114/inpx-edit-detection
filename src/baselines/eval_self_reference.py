"""自参考检测评测 -- 在缓存patch特征上运行training-free方法

不依赖原始图片，直接用预缓存的 per-patch 特征。
关键对比：自参考方法 vs 线性探针，尤其是跨域泛化。
"""
import json
import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.self_reference import self_reference_detect
from evaluation.metrics import mask_area_bin


def evaluate_self_reference(
    features_dir: str,
    output_dir: str = None,
    n_samples: int = None,
    k: float = 0.05,
    n_iter: int = 3,
):
    """在缓存特征上运行自参考检测"""
    features_dir = Path(features_dir)

    # 加载元数据
    with open(features_dir / "labels.json") as f:
        labels_data = json.load(f)
    with open(features_dir / "metadata.json") as f:
        metadata = json.load(f)

    n_total = metadata["n_total"]
    n_patches = metadata["n_patches"]
    dino_dim = metadata["dino_dim"]
    use_vae = metadata.get("use_vae", False)

    print(f"样本数: {n_total}, patches: {n_patches}, dino_dim: {dino_dim}")
    print(f"VAE残差: {use_vae}")

    # 按split组织
    # labels中的 kind: real(0), standard(1), exchanged(1)
    by_kind = defaultdict(list)
    by_cat = defaultdict(list)
    by_size = defaultdict(list)

    for item in labels_data:
        kind = item["kind"]
        cat = item.get("cat", "unknown")
        sc = item.get("size_class", "unknown")
        by_kind[kind].append(item["idx"])
        by_cat[cat].append(item["idx"])
        by_size[sc].append(item["idx"])

    print(f"\n按类型: real={len(by_kind['real'])}, standard={len(by_kind['standard'])}, "
          f"exchanged={len(by_kind['exchanged'])}")
    for cat, idxs in sorted(by_cat.items()):
        print(f"  {cat}: {len(idxs)}")
    print(f"按mask面积: {dict((k, len(v)) for k, v in by_size.items())}")

    if n_samples:
        # 随机采样用于快速测试
        rng = np.random.RandomState(42)
        all_idxs = []
        for kind_key in ["real", "standard", "exchanged"]:
            idxs = by_kind.get(kind_key, [])
            if len(idxs) > n_samples:
                idxs = sorted(rng.choice(idxs, n_samples, replace=False))
            all_idxs.extend(idxs)
        eval_idxs = sorted(all_idxs)
        print(f"\n采样 {len(eval_idxs)} 样本用于评测")
    else:
        eval_idxs = sorted(set().union(*by_kind.values()))

    # 加载 memmap
    dino_memmap = np.memmap(
        str(features_dir / "dino_tokens.npy"),
        dtype=np.float16, mode="r",
        shape=(n_total, n_patches, dino_dim),
    )
    hp_memmap = np.memmap(
        str(features_dir / "highpass.npy"),
        dtype=np.float32, mode="r",
        shape=(n_total, n_patches, 1),
    )
    tex_memmap = np.memmap(
        str(features_dir / "texture_energy.npy"),
        dtype=np.float32, mode="r",
        shape=(n_total, n_patches),
    )
    if use_vae:
        vae_memmap = np.memmap(
            str(features_dir / "vae_residual.npy"),
            dtype=np.float32, mode="r",
            shape=(n_total, n_patches, 1),
        )

    # 结果收集
    results = []

    print(f"\n评测 {len(eval_idxs)} 张图像...")
    t0 = time.time()

    for batch_start in tqdm(range(0, len(eval_idxs), 1), desc="自参考检测"):
        idx = eval_idxs[batch_start]

        item = labels_data[idx]
        label = item["label"]
        kind = item["kind"]
        cat = item.get("cat", "unknown")
        mask_ratio = item.get("mask_ratio", 0)
        size_class = item.get("size_class", "unknown")

        # 构建 per-patch 特征 (只用低层: hp + vae残差; DINOv2语义太强会压住编辑痕迹)
        hp = torch.from_numpy(hp_memmap[idx].astype(np.float32))       # [N, 1]
        tex = torch.from_numpy(tex_memmap[idx].astype(np.float32))      # [N]

        feats = [hp]
        if use_vae:
            vae = torch.from_numpy(vae_memmap[idx].astype(np.float32))
            feats.append(vae)

        X = torch.cat(feats, dim=-1)  # [N, 1 or 2]

        # 运行自参考检测
        try:
            heatmap, image_score = self_reference_detect(
                X.cuda() if torch.cuda.is_available() else X,
                tex.cuda() if torch.cuda.is_available() else tex,
                k=k, n_iter=n_iter,
            )
            heatmap = heatmap.cpu() if isinstance(heatmap, torch.Tensor) else heatmap
        except Exception as e:
            print(f"\nError on idx {idx}: {e}")
            heatmap = np.zeros(n_patches)
            image_score = 0.0

        results.append({
            "idx": int(idx),
            "label": int(label),
            "kind": str(kind),
            "cat": str(cat),
            "mask_ratio": float(mask_ratio) if mask_ratio is not None else 0.0,
            "size_class": str(size_class) if size_class else "unknown",
            "image_score": round(float(image_score) if isinstance(image_score, (float, int, np.floating, np.integer)) else 0.0, 5),
            "heatmap_mean": round(float(heatmap.mean()) if heatmap is not None else 0.0, 5),
            "heatmap_max": round(float(heatmap.max()) if heatmap is not None else 0.0, 5),
        })

    elapsed = time.time() - t0
    print(f"\n完成! 耗时: {elapsed:.1f}s ({len(eval_idxs)/elapsed:.1f} img/s)")

    # ---- 汇总指标 ----
    compute_and_print_metrics(results, by_kind.keys())

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "self_reference_results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存: {out_file}")

    return results


def compute_and_print_metrics(results, kinds):
    """汇总并打印各类指标"""
    import numpy as np
    from sklearn.metrics import roc_auc_score, accuracy_score

    labels = np.array([r["label"] for r in results])
    scores = np.array([r["image_score"] for r in results])
    cats = np.array([r["cat"] for r in results])
    kinds_arr = np.array([r["kind"] for r in results])
    size_classes = np.array([r["size_class"] for r in results])

    def _metrics(mask):
        y = labels[mask]
        s = scores[mask]
        if len(np.unique(y)) < 2:
            return {"Acc": -1, "AUC": -1, "R.Acc": -1, "F.Acc": -1}
        y_pred = (s >= np.median(s)).astype(int)
        acc = accuracy_score(y, y_pred)
        try:
            auc = roc_auc_score(y, s)
        except:
            auc = -1
        r_acc = accuracy_score(y[y==0], y_pred[y==0]) if sum(y==0) > 0 else -1
        f_acc = accuracy_score(y[y==1], y_pred[y==1]) if sum(y==1) > 0 else -1
        return {"Acc": acc, "AUC": auc, "R.Acc": r_acc, "F.Acc": f_acc}

    print(f"\n{'='*60}")
    print("自参考检测 -- 图像级分类结果")
    print(f"{'='*60}")

    # 全局
    m = _metrics(slice(None))
    print(f"\n全局 (n={len(labels)}):")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")

    # 按图像类型
    print(f"\n按图像类型:")
    for kind in ["real", "standard", "exchanged"]:
        mask = kinds_arr == kind
        if mask.sum() == 0:
            continue
        m = _metrics(mask)
        print(f"  {kind}: n={mask.sum()}, Acc={m['Acc']:.4f}, AUC={m['AUC']:.4f}, "
              f"R.Acc={m['R.Acc']:.4f}, F.Acc={m['F.Acc']:.4f}")

    # 按数据集
    print(f"\n按数据集:")
    for cat in sorted(np.unique(cats)):
        mask = cats == cat
        m = _metrics(mask)
        if m["Acc"] < 0:
            continue
        print(f"  {cat}: n={mask.sum()}, Acc={m['Acc']:.4f}, AUC={m['AUC']:.4f}, "
              f"R.Acc={m['R.Acc']:.4f}, F.Acc={m['F.Acc']:.4f}")

    # 按mask面积分档
    print(f"\n按mask面积分档:")
    for sc in ["tiny", "small", "medium", "large"]:
        name_map = {"tiny": "极小(<2%)", "small": "小(2-5%)", "medium": "中(5-15%)", "large": "大(>15%)"}
        mask = (size_classes == sc) & (labels == 1)
        if mask.sum() < 5:
            continue
        fake_acc = (scores[mask] >= np.median(scores)).mean()
        print(f"  {name_map[sc]}: n={mask.sum()}, F.Acc={fake_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str,
                        default="D:/lunwen/data/features_cache")
    parser.add_argument("--output_dir", type=str,
                        default="D:/lunwen/outputs")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="每类采样数 (-1 = 全部)")
    parser.add_argument("--k", type=float, default=0.05,
                        help="top-k 池化比例")
    parser.add_argument("--n_iter", type=int, default=3,
                        help="迭代精化轮数")
    args = parser.parse_args()

    evaluate_self_reference(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples if args.n_samples > 0 else None,
        k=args.k,
        n_iter=args.n_iter,
    )
