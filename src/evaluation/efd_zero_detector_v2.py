"""Exploratory patch-direction similarity diagnostic.

Compares normalized patch deviations from the image mean with a training
direction library and averages the ten highest patch similarities.
This historical probe is separate from the main manuscript evaluation."""
import json
import os
import sys
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
CACHE = "D:/lunwen/data/features_cache"
IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
DATA_JSON = "D:/lunwen/data/imdl_inpx_test.json"
OUT = Path("D:/lunwen/outputs")
GRID = 37
NP = 1369


def main():
    labels = json.load(open(f"{CACHE}/labels.json", encoding="utf-8"))
    meta = json.load(open(f"{CACHE}/metadata.json", encoding="utf-8"))
    n_total, n_patches, d = meta["n_total"], meta["n_patches"], meta["dino_dim"]
    dino_mm = np.memmap(f"{CACHE}/dino_tokens.npy", dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, d))
    hp_mm = np.memmap(f"{CACHE}/highpass.npy", dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))
    path2idx = {l["path"].replace("\\", "/"): i for i, l in enumerate(labels)}

    V = np.load(OUT / "efd_vecs.npy")
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)

    recs = json.load(open(DATA_JSON, encoding="utf-8"))
    edit_idx, real_idx = [], []
    for img_path, mask_path in recs:
        rel = str(Path(img_path).relative_to(IMG_ROOT)).replace("\\", "/")
        idx = path2idx.get(rel)
        if idx is None:
            continue
        (real_idx if mask_path == "Negative" else edit_idx).append(idx)
    edit_idx = np.array(edit_idx)
    real_idx = np.array(real_idx)
    print(f"测试: 编辑 {len(edit_idx)}, real {len(real_idx)}", flush=True)

    def mask_of(idx):
        mp = labels[idx].get("mask_path")
        if mp and (IMG_ROOT / mp).exists():
            m = Image.open(IMG_ROOT / mp).convert("L").resize((GRID, GRID), Image.NEAREST)
            return (np.asarray(m) > 127).astype(np.float32).ravel()
        return np.zeros(NP, np.float32)

    def dir_from_mask(idx):
        mk = mask_of(idx)
        F = torch.from_numpy(dino_mm[idx].copy()).float()
        if mk.sum() < 4 or mk.sum() > NP * 0.9:
            return None
        e = F[mk > 0].mean(0)
        bg = F[mk == 0].mean(0) if (mk == 0).sum() > 0 else F.mean(0)
        v = e - bg
        hp = torch.from_numpy(hp_mm[idx].copy()).float().mean(0)  # (1,)
        v = torch.cat([v, hp])
        n = v.norm()
        return (v / n).numpy() if n > 1e-8 else None

    def direction_scores(idx):
        """该图所有 patch 与图均值的偏差方向 (1370 个方向), 与库 V 的相似度."""
        F = torch.from_numpy(dino_mm[idx].copy()).float()  # (1369, dim)
        hp = torch.from_numpy(hp_mm[idx].copy()).float()   # (1369, 1)
        F = torch.cat([F, hp], -1)
        Fn = F - F.mean(0, keepdim=True)
        Fn = Fn / (Fn.norm(dim=1, keepdim=True) + 1e-8)   # (1369, dim) 归一化
        sims = Fn @ V.T  # (1369, 4844) 每 patch 与每库方向的 cos
        # 每个 patch 与库的 max 相似度, 取所有 patch 的 top-k 平均
        patch_max = sims.max(dim=1).values  # (1369,) 每 patch 最匹配方向
        top_patches = np.sort(patch_max.cpu().numpy())[-10:]  # 最编辑样 patch
        return float(top_patches.mean())

    # ---- 评估: 均用盲推理 (edit 和 real 都用整图 top-patch 方向, 公平) ----
    edit_scores, real_scores = [], []
    for i, idx in enumerate(edit_idx):
        edit_scores.append(direction_scores(idx))
        if (i + 1) % 100 == 0:
            print(f"  edit {i+1}/{len(edit_idx)}", flush=True)
    for i, idx in enumerate(real_idx):
        real_scores.append(direction_scores(idx))
        if (i + 1) % 100 == 0:
            print(f"  real {i+1}/{len(real_idx)}", flush=True)
    edit_scores = np.array(edit_scores)
    real_scores = np.array(real_scores)
    auc = roc_auc_score(np.concatenate([np.ones(len(edit_scores)),
                                        np.zeros(len(real_scores))]),
                        np.concatenate([edit_scores, real_scores]))
    print(f"\n=== 零训练证据场检测器 v2 (编辑用 mask 方向; real 用整图方向) ===")
    print(f"edit: n={len(edit_scores)} mean={edit_scores.mean():.4f} "
          f"p10={np.percentile(edit_scores,10):.4f} p90={np.percentile(edit_scores,90):.4f}")
    print(f"real: n={len(real_scores)} mean={real_scores.mean():.4f} "
          f"p10={np.percentile(real_scores,10):.4f} p90={np.percentile(real_scores,90):.4f}")
    print(f"检测 AUC = {auc:.4f}")
    print(f"(对照: 有监督 g1=0.928; 零监督 30K=0.946)")
    print(f"判定: {'√ 强 (>0.85)' if auc > 0.85 else ('√ 中 (0.7-0.85)' if auc > 0.7 else '× 低 (<0.7)')}")

    with open(OUT / "efd_zero_detector_v2.json", "w", encoding="utf-8") as f:
        json.dump({"n_edit": int(len(edit_scores)), "n_real": int(len(real_scores)),
                   "edit_mean": float(edit_scores.mean()), "real_mean": float(real_scores.mean()),
                   "auc": float(auc)}, f, indent=1)
    print("saved:", OUT / "efd_zero_detector_v2.json", flush=True)


if __name__ == "__main__":
    main()
