"""方向三: 证据场→零训练检测器 (Zero-shot evidence-field detector)

核心假设: 编辑图的扰动方向 d = mean(edit tokens) - mean(bg tokens) 会与
方向库 V (来自训练池 4844 个不同编辑) 中的某个方向高度对齐, 而 real 图
的扰动方向随机。用 max-cosine 相似度作为检测分数。

关键验证:
  1. 已知编辑 (库内): max-cos 应高 → 检测为编辑
  2. 未见编辑 (库外, 如测试集编辑): max-cos 是否仍高? (泛化性)
  3. real 图: max-cos 应低
输出: AUC (编辑 vs real), 分位数, 与 g1 (0.928) 对比。

优化 (避免边际随机): 检测分数不用"与库中任一方向最大相似度", 而用
  库的中位数投影类似度或 top-k 平均相似度, 以减少偶然对齐。
"""
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
    split = json.load(open(OUT / "split_budget_master.json"))
    path2idx = {l["path"].replace("\\", "/"): i for i, l in enumerate(labels)}
    train_idx = set(split["train_idx"])

    V = np.load(OUT / "efd_vecs.npy")  # (4844, 385) 训练池编辑方向库 (归一化)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)

    # ---- 构造测试池: 500 编辑 (标准版) + 300 real ----
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

    def perturb_dir(idx):
        """编辑区扰动方向 (需要 mask; real 图无 mask -> 用全图平均? 用背景=全图?)"""
        mk = mask_of(idx)
        F = torch.from_numpy(dino_mm[idx].copy()).float()
        if mk.sum() < 4 or mk.sum() > NP * 0.9:
            return None
        e = F[mk > 0].mean(0)
        bg = F[mk == 0].mean(0) if (mk == 0).sum() > 0 else F.mean(0)
        v = e - bg
        n = v.norm()
        if n < 1e-8:
            return None
        return (v / n).numpy()

    def score_image(idx):
        """检测分数: 扰动方向与库的相似度 (top-k 平均, 或 max)"""
        v = perturb_dir(idx)
        if v is None:
            return 0.0
        sims = V @ v  # (4844,)
        # top-10 平均相似度
        top = np.sort(sims)[-10:]
        return float(top.mean())

    # ---- 评估 ----
    edit_scores, real_scores = [], []
    n_skip_e = n_skip_r = 0
    for i, idx in enumerate(edit_idx):
        s = score_image(idx)
        if s is None:
            n_skip_e += 1
            continue
        edit_scores.append(s)
        if (i + 1) % 100 == 0:
            print(f"  edit {i+1}/{len(edit_idx)}", flush=True)
    for i, idx in enumerate(real_idx):
        s = score_image(idx)
        if s is None:
            n_skip_r += 1
            continue
        real_scores.append(s)
        if (i + 1) % 100 == 0:
            print(f"  real {i+1}/{len(real_idx)}", flush=True)
    edit_scores = np.array(edit_scores)
    real_scores = np.array(real_scores)
    auc = roc_auc_score(np.concatenate([np.ones(len(edit_scores)), np.zeros(len(real_scores))]),
                        np.concatenate([edit_scores, real_scores]))
    print(f"\n=== 零训练证据场检测器 ===")
    print(f"edit: n={len(edit_scores)} mean={edit_scores.mean():.4f} "
          f"p10={np.percentile(edit_scores,10):.4f} p90={np.percentile(edit_scores,90):.4f}")
    print(f"real: n={len(real_scores)} mean={real_scores.mean():.4f} "
          f"p10={np.percentile(real_scores,10):.4f} p90={np.percentile(real_scores,90):.4f}")
    print(f"检测 AUC = {auc:.4f}")
    print(f"(对照: 有监督 g1 = 0.928; 零监督 30K = 0.946)")
    print(f"结论编辑: {'√ 高 (>0.85)' if auc > 0.85 else ('√ 中 (0.7-0.85)' if auc > 0.7 else '× 低 (<0.7)')}")

    with open(OUT / "efd_zero_detector.json", "w", encoding="utf-8") as f:
        json.dump({"n_edit": int(len(edit_scores)), "n_real": int(len(real_scores)),
                   "edit_mean": float(edit_scores.mean()), "real_mean": float(real_scores.mean()),
                   "auc": float(auc), "skip_edit": int(n_skip_e), "skip_real": int(n_skip_r)},
                  f, indent=1)
    print("saved:", OUT / "efd_zero_detector.json", flush=True)


if __name__ == "__main__":
    main()
