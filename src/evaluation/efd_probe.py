"""Exploratory evidence-subspace localization probe.

Training masks define edited-region minus background feature vectors.
Normalized vectors form an SVD basis. At inference, patch features are
centered by the image-wide feature mean and projected onto this basis.
Projection norms provide localization scores; masks are used for evaluation.
This probe is separate from the manuscript's main trained readout."""
import argparse
import json
import os
import sys
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CACHE = "D:/lunwen/data/features_cache"
IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
OUT = Path("D:/lunwen/outputs")
GRID = 37
NP = GRID * GRID

KEY_RE = r"^(.*?)_(?:CelebAHQ|CityScapes|SUN_RGBD|OpenImages)_(?:Kandinsky_2_2|StableDiffusion_v4|OpenJourney)(?:_simple)?\.jpg$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="INP-X", choices=["INP-X", "external"])
    ap.add_argument("--n_anns", type=int, default=5000, help="用多少注记建基")
    ap.add_argument("--k", type=int, default=8, help="证据基维数")
    ap.add_argument("--no_hp", action="store_true")
    ap.add_argument("--local_rad", type=int, default=2, help="局部均值半径 (25x25 窗口)")
    args = ap.parse_args()

    labels = json.load(open(f"{CACHE}/labels.json", encoding="utf-8"))
    meta = json.load(open(f"{CACHE}/metadata.json", encoding="utf-8"))
    n_total, n_patches, d = meta["n_total"], meta["n_patches"], meta["dino_dim"]
    dino_mm = np.memmap(f"{CACHE}/dino_tokens.npy", dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, d))
    hp_mm = np.memmap(f"{CACHE}/highpass.npy", dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))
    split = json.load(open(OUT / "split_budget_master.json"))
    train_idx = set(split["train_idx"])

    man = json.load(open(OUT / "mask_budget_manifest.json", encoding="utf-8"))
    anns = man["ordered_annotations"]
    # 只取训练池注记
    anns = [a for a in anns if a["rep_idx"] in train_idx]
    anns = anns[: args.n_anns]
    print(f"建基注记: {len(anns)}", flush=True)

    def mask_of(idx):
        mp = labels[idx].get("mask_path")
        if mp and (IMG_ROOT / mp).exists():
            m = Image.open(IMG_ROOT / mp).convert("L").resize((GRID, GRID), Image.NEAREST)
            return (np.asarray(m) > 127).astype(np.float32).ravel()
        return np.zeros(NP, np.float32)

    def feat(i):
        Fx = torch.from_numpy(dino_mm[i].copy()).float()
        if not args.no_hp:
            Fx = torch.cat([Fx, torch.from_numpy(hp_mm[i].copy()).float()], -1)
        return Fx

    # ---- Step 1: 建扰动矢量 ----
    vecs = []
    rng = np.random.RandomState(0)
    anns = rng.permutation(anns)
    n_used = 0
    for a in anns:
        idx = a["rep_idx"]
        mk = mask_of(idx)
        if mk.sum() < 4 or mk.sum() > NP * 0.9:
            continue
        F = feat(idx)  # (1369, dim)
        e = F[mk > 0].mean(0)
        if (mk == 0).sum() == 0:
            continue
        bg = F[mk == 0].mean(0)
        v = e - bg
        v = v / (v.norm() + 1e-8)
        vecs.append(v.numpy())
        n_used += 1
        if n_used >= args.n_anns:
            break
    V = np.stack(vecs)  # (n, dim)
    print(f"Normalized basis vectors: {V.shape}", flush=True)
    np.save(OUT / "efd_vecs.npy", V)

    # ---- Step 2: SVD 证据基 ----
    # V 是 (n_sample, dim)。扰动矢量 d 在特征空间; 特征的证据基 = 右奇异矢量 Vt 的行。
    # 对每张图的扰动 d_i ~ U S Vt; 特征空间的证据轴 = Vt^T 的行 (dim 维)。
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    Ek = Vt[: args.k].T  # (dim, k)
    np.save(OUT / "efd_basis.npy", Ek)
    print(f"SVD 奇异值前10: {np.round(S[:10], 3)}", flush=True)
    print(f"能量占比: {np.round(np.cumsum(S[:args.k]**2) / (S**2).sum(), 3)}", flush=True)

    # ---- Step 3: 推理 (盲) —— 用训练池图测试投影谱 ----
    from sklearn.metrics import roc_auc_score, average_precision_score

    def blind_score(idx):
        F = feat(idx).cuda()  # (1369, dim)
        Ekc = torch.from_numpy(Ek).float().cuda()
        # 扰动 = patch 特征 - 图内全局均值 (与建基定义一致: 编辑区 vs 整图)
        perturb = F - F.mean(dim=0, keepdim=True)  # (1369, dim)
        z = perturb @ Ekc  # (1369, k)
        score = z.norm(dim=1)  # (1369,)
        return torch.sigmoid(score).cpu().numpy()

    # 抽样 200 个训练图 + mask, 评估: 预测 vs GT -> AUROC/AP/mIoU
    eval_pool = [(a["rep_idx"], mask_of(a["rep_idx"])) for a in anns[:200]]
    scores, masks = [], []
    for idx, mk in eval_pool:
        if mk.sum() == 0:
            continue
        s = blind_score(idx)
        scores.append(s)
        masks.append(mk)
    scores = np.stack(scores)
    masks = np.stack(masks)
    flat_s, flat_m = scores.ravel(), masks.ravel()
    auroc = roc_auc_score(flat_m, flat_s)
    ap = average_precision_score(flat_m, flat_s)
    # best-thr mIoU
    best = 0.0
    for th in np.linspace(0.05, 0.95, 91):
        pred = (scores >= th).astype(np.float32)
        tp = (pred * masks).sum(); fp = (pred * (1 - masks)).sum(); fn = ((1 - pred) * masks).sum()
        best = max(best, tp / (tp + fp + fn + 1e-8))
    print(f"\n[盲定位, 训练池 200 张] AUROC={auroc:.4f} AP={ap:.4f} best-mIoU={best:.4f}", flush=True)

    res = {"n_basis": int(V.shape[0]), "k": args.k, "svals": [float(s) for s in S[:10]],
           "energy_topk": float(np.cumsum(S[:args.k] ** 2).sum() / (S ** 2).sum()),
           "blind_auroc": float(auroc), "blind_ap": float(ap), "blind_miou": float(best)}
    with open(OUT / "efd_probe.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("saved:", OUT / "efd_probe.json", flush=True)


if __name__ == "__main__":
    main()
