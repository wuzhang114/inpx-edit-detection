"""跨域 CelebAHQ 按 kind 拆分评测: standard(未交换, 现实场景) vs exchanged(压力测试)

用法: python src/evaluation/eval_kind_split.py [--head weak_sup_v4_head.pt]
产出: outputs/kind_split_<head>.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.weakly_supervised_v4 import MLPHead, build_masks

CACHE = "D:/lunwen/data/features_cache"
OUT = "D:/lunwen/outputs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="weak_sup_v4_head.pt")
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    with open(f"{CACHE}/labels.json") as f:
        labels_data = json.load(f)
    with open(f"{CACHE}/metadata.json") as f:
        meta = json.load(f)
    n_total, n_patches, dino_dim = meta["n_total"], meta["n_patches"], meta["dino_dim"]
    dino_mm = np.memmap(f"{CACHE}/dino_tokens.npy", dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, dino_dim))
    hp_mm = np.memmap(f"{CACHE}/highpass.npy", dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))

    cats = np.array([l.get("cat", "unknown") for l in labels_data])
    labels_arr = np.array([l["label"] for l in labels_data])
    kinds = np.array([l.get("kind", "standard") for l in labels_data])

    celeb = np.where(cats == "CelebAHQ")[0]
    real_idx = celeb[labels_arr[celeb] == 0]
    masks = build_masks(labels_data, n_total, n_patches)

    head = MLPHead(dino_dim + 1).cuda()
    head.load_state_dict(torch.load(f"{OUT}/{args.head}", map_location="cpu"))
    head.eval()

    def evaluate(fake_idx, name):
        eval_idx = np.concatenate([real_idx, fake_idx])
        scores = []
        with torch.no_grad():
            for i in range(0, len(eval_idx), args.batch_size):
                b = eval_idx[i:i + args.batch_size]
                X = torch.cat([torch.from_numpy(dino_mm[b].astype(np.float32)),
                               torch.from_numpy(hp_mm[b].astype(np.float32))], dim=-1).cuda()
                logits = head(X)
                scores.append(torch.sigmoid(logits).cpu().numpy())
        scores = np.concatenate(scores)
        img_scores = scores.max(axis=1)
        y = np.concatenate([np.zeros(len(real_idx)), np.ones(len(fake_idx))])
        auc = roc_auc_score(y, img_scores)
        m_tgt = masks[fake_idx]
        s_tgt = scores[len(real_idx):]
        best_miou, best_f1 = 0.0, 0.0
        for th in np.linspace(0.05, 0.95, 91):
            pred = (s_tgt >= th).astype(np.float32)
            tp = (pred * m_tgt).sum(); fp = (pred * (1 - m_tgt)).sum(); fn = ((1 - pred) * m_tgt).sum()
            prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
            best_miou = max(best_miou, tp / (tp + fp + fn + 1e-8))
            best_f1 = max(best_f1, 2 * prec * rec / (prec + rec + 1e-8))
        print(f"{name}: n_fake={len(fake_idx)} AUC={auc:.4f} mIoU={best_miou:.4f} F1={best_f1:.4f}")
        return {"name": name, "n_real": int(len(real_idx)), "n_fake": int(len(fake_idx)),
                "img_auc": float(auc), "loc_miou": float(best_miou), "loc_f1": float(best_f1)}

    results = [
        evaluate(celeb[(labels_arr[celeb] == 1) & (kinds[celeb] == "standard")], "standard(unexchanged)"),
        evaluate(celeb[(labels_arr[celeb] == 1) & (kinds[celeb] == "exchanged")], "exchanged"),
    ]
    tag = args.head.replace("_head.pt", "")
    path = f"{OUT}/kind_split_{tag}.json"
    with open(path, "w") as f:
        json.dump({"head": args.head, "results": results}, f, indent=1)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
