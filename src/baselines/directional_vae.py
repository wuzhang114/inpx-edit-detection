"""方向性 VAE 残差信号实验 -- training-free 自参考 v3

核心假设 (来自 INP-X 论文观察, r≈0.94):
  - 真实图像过 SD VAE 的重建残差与高频内容强相关
  - 被 VAE 处理过的编辑区域, 其重建残差异常地低
→ 图内自参考: 在内容匹配(纹理分桶 / DINO 语义分桶)条件下,
  找"残差异常低"的 patch (单尾方向性异常), 而非对称 L2 偏离。

变体:
  vae_low_tex : VAE残差低尾 + 纹理分桶   (主假设)
  vae_high_tex: VAE残差高尾 + 纹理分桶   (方向对照)
  vae_abs_tex : |z| 双侧异常 + 纹理分桶  (对称对照)
  vae_low_dino: VAE残差低尾 + DINO语义聚类分桶 (内容条件方案 ii)
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE = "D:/lunwen/data/features_cache"


def robust_z(x: torch.Tensor, n_iter: int = 3, thr: float = 3.0):
    """迭代式鲁棒 z-score: 排除离群值后重估 median/MAD (对污染鲁棒)"""
    x = x.float()
    w = torch.ones_like(x)
    for _ in range(n_iter):
        m = w > 0.5
        if m.sum() < 5:
            break
        med = x[m].median()
        mad = (x[m] - med).abs().median()
        if mad < 1e-8:
            mad = x[m].std() + 1e-8
        z = (x - med) / (1.4826 * mad + 1e-8)
        w = (z.abs() <= thr).float()
    return z


@torch.no_grad()
def directional_anomaly(vae: torch.Tensor, bins: torch.Tensor,
                        mode: str = "low", n_iter: int = 3) -> torch.Tensor:
    """桶内方向性异常分数 (per-patch)"""
    n = vae.shape[0]
    anomaly = torch.zeros(n)
    for b in bins.unique():
        m = bins == b
        if m.sum() < 10:
            continue
        z = robust_z(vae[m], n_iter=n_iter)
        if mode == "low":
            anomaly[m] = torch.clamp(-z, min=0)   # 残差异常低 → 疑似编辑区
        elif mode == "high":
            anomaly[m] = torch.clamp(z, min=0)    # 残差异常高
        elif mode == "abs":
            anomaly[m] = z.abs()                  # 双侧
    return anomaly


def tex_bins(tex: torch.Tensor, n_bins: int = 5) -> torch.Tensor:
    q = torch.quantile(tex, torch.linspace(0, 1, n_bins + 1))
    q = torch.unique(q)  # 去重, 避免空桶
    return torch.bucketize(tex, q[1:-1])


def dino_bins(dino: torch.Tensor, n_clusters: int = 6) -> torch.Tensor:
    """DINO token 语义聚类 (图内内容条件)"""
    dn = torch.nn.functional.normalize(dino, dim=-1).numpy()
    km = KMeans(n_clusters=n_clusters, n_init=2, random_state=0,
                max_iter=50).fit(dn)
    return torch.from_numpy(km.labels_)


def slide_bins(tex: torch.Tensor, n_bins: int = 24) -> torch.Tensor:
    """按纹理能量排序后的等分桶 (连续条件化的离散近似)"""
    order = torch.argsort(tex)
    n = tex.shape[0]
    bins = torch.empty(n, dtype=torch.long)
    per = max(1, n // n_bins)
    for i, idx in enumerate(order):
        bins[idx] = min(i // per, n_bins - 1)
    return bins


def evaluate(features_dir: str, output_dir: str, n_samples: int = 300,
             variant: str = "vae_low_tex", k: float = 0.05,
             n_bins: int = 5, n_clusters: int = 6, n_iter: int = 3):
    features_dir = Path(features_dir)
    with open(features_dir / "labels.json") as f:
        labels_data = json.load(f)
    with open(features_dir / "metadata.json") as f:
        meta = json.load(f)

    n_total, n_patches, dino_dim = meta["n_total"], meta["n_patches"], meta["dino_dim"]
    vae_mm = np.memmap(str(features_dir / "vae_residual.npy"), dtype=np.float32,
                       mode="r", shape=(n_total, n_patches, 1))
    hp_mm = np.memmap(str(features_dir / "highpass.npy"), dtype=np.float32,
                      mode="r", shape=(n_total, n_patches, 1))
    tex_mm = np.memmap(str(features_dir / "texture_energy.npy"), dtype=np.float32,
                       mode="r", shape=(n_total, n_patches))
    dino_mm = np.memmap(str(features_dir / "dino_tokens.npy"), dtype=np.float16,
                        mode="r", shape=(n_total, n_patches, dino_dim))

    by_kind = defaultdict(list)
    for item in labels_data:
        by_kind[item["kind"]].append(item["idx"])

    rng = np.random.RandomState(42)
    all_idxs = []
    for kind_key in ["real", "standard", "exchanged"]:
        idxs = by_kind.get(kind_key, [])
        if len(idxs) > n_samples:
            idxs = sorted(rng.choice(idxs, n_samples, replace=False))
        all_idxs.extend(idxs)
    eval_idxs = sorted(all_idxs)
    print(f"变体={variant}, k={k}, n_bins={n_bins}, n_clusters={n_clusters}, "
          f"样本={len(eval_idxs)}")

    use_dino_bins = variant.endswith("_dino")
    use_slide_bins = "_slide" in variant
    feature = "hp" if variant.startswith("hp") else "vae"
    mode = "low" if "_low_" in variant else ("high" if "_high_" in variant else "abs")

    results = []
    t0 = time.time()
    for idx in tqdm(eval_idxs, desc=variant):
        item = labels_data[idx]
        if feature == "vae":
            feat = torch.from_numpy(vae_mm[idx].astype(np.float32)).squeeze(-1)
        else:
            feat = torch.from_numpy(hp_mm[idx].astype(np.float32)).squeeze(-1)
        tex = torch.from_numpy(tex_mm[idx].astype(np.float32))

        if use_dino_bins:
            dino = torch.from_numpy(dino_mm[idx].astype(np.float32))
            bins = dino_bins(dino, n_clusters=n_clusters)
        elif use_slide_bins:
            bins = slide_bins(tex, n_bins=n_bins)
        else:
            bins = tex_bins(tex, n_bins=n_bins)

        anomaly = directional_anomaly(feat, bins, mode=mode, n_iter=n_iter)
        n_top = max(1, int(k * n_patches))
        score = anomaly.topk(n_top).values.mean().item()
        results.append({
            "idx": int(idx),
            "label": int(item["label"]),
            "kind": item["kind"],
            "cat": item.get("cat", "unknown"),
            "mask_ratio": item.get("mask_ratio", 0.0) or 0.0,
            "size_class": item.get("size_class") or "unknown",
            "score": round(score, 6),
        })

    print(f"完成! {time.time()-t0:.1f}s")

    # ---- 指标 ----
    labels = np.array([r["label"] for r in results])
    scores = np.array([r["score"] for r in results])
    kinds = np.array([r["kind"] for r in results])
    cats = np.array([r["cat"] for r in results])
    sizes = np.array([r["size_class"] for r in results])

    def _m(mask):
        y, s = labels[mask], scores[mask]
        if len(np.unique(y)) < 2 or np.all(s == s[0]):
            return None
        auc = roc_auc_score(y, s)
        thr = np.median(s)
        yp = (s >= thr).astype(int)
        return {
            "AUC": auc,
            "Acc": accuracy_score(y, yp),
            "R.Acc": accuracy_score(y[y == 0], yp[y == 0]) if (y == 0).sum() else -1,
            "F.Acc": accuracy_score(y[y == 1], yp[y == 1]) if (y == 1).sum() else -1,
        }

    print("\n" + "=" * 60)
    m = _m(slice(None))
    if m:
        print(f"全局 (n={len(labels)}): AUC={m['AUC']:.4f} Acc={m['Acc']:.4f} "
              f"R.Acc={m['R.Acc']:.4f} F.Acc={m['F.Acc']:.4f}")
    # 两两对比 (单类无法算AUC, 用子集)
    for a, b in [("real", "standard"), ("real", "exchanged"),
                 ("standard", "exchanged")]:
        ma, mb = kinds == a, kinds == b
        y = np.concatenate([np.zeros(ma.sum()), np.ones(mb.sum())])
        s = np.concatenate([scores[ma], scores[mb]])
        print(f"  {a} vs {b}: AUC={roc_auc_score(y, s):.4f}")
    print("按mask面积 (仅fake):")
    for sc in ["tiny", "small", "medium", "large"]:
        mm = (sizes == sc) & (labels == 1)
        if mm.sum() < 5:
            continue
        thr = np.median(scores)
        facc = (scores[mm] >= thr).mean()
        print(f"  {sc}: n={mm.sum()} F.Acc={facc:.4f}")

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"directional_{variant}_k{k}.json", "w") as f:
            json.dump(results, f, indent=1)
        print(f"已保存: {out / f'directional_{variant}_k{k}.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="vae_low_tex",
                   choices=["vae_low_tex", "vae_high_tex", "vae_abs_tex",
                            "vae_low_dino", "vae_high_dino",
                            "vae_high_slide", "vae_low_slide",
                            "hp_low_dino", "hp_high_dino"])
    p.add_argument("--n_samples", type=int, default=300)
    p.add_argument("--k", type=float, default=0.05)
    p.add_argument("--n_bins", type=int, default=5)
    p.add_argument("--n_clusters", type=int, default=6)
    p.add_argument("--output_dir", default="D:/lunwen/outputs")
    args = p.parse_args()
    evaluate(CACHE, args.output_dir, n_samples=args.n_samples,
             variant=args.variant, k=args.k, n_bins=args.n_bins,
             n_clusters=args.n_clusters)
