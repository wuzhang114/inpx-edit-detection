"""loo 补充评测: 被排除 inpainter 的编辑图 (真实"未见 inpainter"数字)

测试集 = 全部 real + 目标 inpainter 的全部编辑图 (训练域内, 但该 inpainter
在 loo 训练中被排除 → 对模型是"未见 inpainter")。

用法: python src/evaluation/eval_loo_subset.py --head weak_sup_v4_loo_kandinsky_head.pt --model Kandinsky_2_2
      python src/evaluation/eval_loo_subset.py --head weak_sup_v4_loo_sdv4_head.pt --model StableDiffusion_v4
      python src/evaluation/eval_loo_subset.py --head weak_sup_v4_head.pt --model Kandinsky_2_2  (对照: 全量头)
评测: 检测 AUC (real + 目标 inpainter 编辑图) + 定位 mIoU (仅目标 inpainter 编辑图)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE = "D:/lunwen/data/features_cache"
OUT = "D:/lunwen/outputs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True)
    p.add_argument("--model", required=True, help="目标 inpainter, 如 Kandinsky_2_2")
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    from baselines.weakly_supervised_v4 import MLPHead  # 与 v4 同构
    with open(f"{CACHE}/labels.json") as f:
        labels_data = json.load(f)
    with open(f"{CACHE}/metadata.json") as f:
        meta = json.load(f)
    n_total, n_patches, dino_dim = meta["n_total"], meta["n_patches"], meta["dino_dim"]

    dino_mm = np.memmap(f"{CACHE}/dino_tokens.npy", dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, dino_dim))
    hp_mm = np.memmap(f"{CACHE}/highpass.npy", dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))

    labels_arr = np.array([l["label"] for l in labels_data])
    models_arr = np.array([l.get("model") for l in labels_data])

    real_idx = np.where(labels_arr == 0)[0]
    tgt_fake = np.where((labels_arr == 1) & (models_arr == args.model))[0]
    eval_idx = np.concatenate([real_idx, tgt_fake])
    print(f"real={len(real_idx)} 目标inpainter编辑图={len(tgt_fake)} 共{len(eval_idx)}")

    head = MLPHead(dino_dim + 1).cuda()
    head.load_state_dict(torch.load(f"{OUT}/{args.head}", map_location="cpu"))
    head.eval()

    scores = []
    with torch.no_grad():
        for i in range(0, len(eval_idx), args.batch_size):
            b = eval_idx[i:i + args.batch_size]
            X = torch.cat([
                torch.from_numpy(dino_mm[b].astype(np.float32)),
                torch.from_numpy(hp_mm[b].astype(np.float32)),
            ], dim=-1).cuda()
            logits = head(X)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    scores = np.concatenate(scores)  # [n, N]
    img_scores = scores.max(axis=1)

    y = np.concatenate([np.zeros(len(real_idx)), np.ones(len(tgt_fake))])
    auc = roc_auc_score(y, img_scores)
    print(f"检测 AUC (real+{args.model}编辑图) = {auc:.4f}")

    # 定位: 仅目标编辑图
    from baselines.weakly_supervised_v4 import build_masks
    masks = build_masks(labels_data, n_total, n_patches)
    m_tgt = masks[tgt_fake]
    s_tgt = scores[len(real_idx):]
    best_miou, best_f1 = 0.0, 0.0
    for th in np.linspace(0.05, 0.95, 91):
        pred = (s_tgt >= th).astype(np.float32)
        tp = (pred * m_tgt).sum(); fp = (pred * (1 - m_tgt)).sum(); fn = ((1 - pred) * m_tgt).sum()
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        best_miou = max(best_miou, tp / (tp + fp + fn + 1e-8))
        best_f1 = max(best_f1, 2 * prec * rec / (prec + rec + 1e-8))
    print(f"定位 mIoU (仅{args.model}) = {best_miou:.4f} F1 = {best_f1:.4f}")

    tag = args.head.replace("weak_sup_v4_", "").replace("_head.pt", "")
    res = {"head": args.head, "target_model": args.model, "n_real": int(len(real_idx)),
           "n_fake": int(len(tgt_fake)), "img_auc": float(auc),
           "loc_miou": float(best_miou), "loc_f1": float(best_f1)}
    path = f"{OUT}/loo_subset_{tag}_{args.model}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"结果: {path}")


if __name__ == "__main__":
    main()
