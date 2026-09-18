"""SDXL 现代编辑集评测: v4 头检测 AUC + 定位 mIoU

统一协议 (顶级 AI 决策): INP-X-only 训练 → INP-X validation threshold (--thr) → zero-shot
用法: python src/evaluation/eval_sdxl.py [--head weak_sup_v4_head.pt] [--thr 0.45] [--edit_root D:/lunwen/data/sdxl_edits_500]
  不传 --thr 时保留 best-thr (oracle) 口径作参考
产出: outputs/sdxl_eval_<head>_<tag>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.weakly_supervised_v2 import MLPHead
from utils.highpass import cross_diff_highpass, pool_to_patch_grid

GRID = 37
IMAGE_SIZE = 518


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="weak_sup_v4_head.pt")
    p.add_argument("--thr", type=float, default=None,
                   help="INP-X 验证集固定阈值 (统一协议); 不传则报 best-thr")
    p.add_argument("--edit_root", default="D:/lunwen/data/sdxl_edits")
    args = p.parse_args()
    edit_root = Path(args.edit_root)

    meta = json.load(open(edit_root / "meta.json"))
    n = len(meta)
    print(f"评测 {n} 张 SDXL 编辑图 from {edit_root}", flush=True)

    import torch.hub as hub
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
    model = model.cuda().eval()
    head = MLPHead(385).cuda()
    head.load_state_dict(torch.load(f"D:/lunwen/outputs/{args.head}", map_location="cpu"))
    head.eval()

    def extract_feat(path):
        img = np.asarray(Image.open(path).convert("RGB"))
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        t = F.interpolate(t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
        t = (t - mean) / std
        tok = model.forward_features(t)["x_norm_patchtokens"]
        hp = F.pad(cross_diff_highpass(t), (0, 1, 0, 1))
        hp_pooled = pool_to_patch_grid(hp, GRID * GRID)
        return torch.cat([tok, hp_pooled], dim=-1).cuda()

    t0 = time.time()
    fake_scores, fake_masks = [], []
    real_scores = []
    with torch.no_grad():
        for i, m in enumerate(meta):
            sid = m["id"]
            fe = extract_feat(edit_root / sid / "edit.jpg")
            logits = head(fe).squeeze(0)
            scores = torch.sigmoid(logits).cpu().numpy().reshape(GRID, GRID)
            fake_scores.append(scores)
            mk = Image.open(edit_root / sid / "mask.png").convert("L").resize((GRID, GRID), Image.NEAREST)
            fake_masks.append((np.asarray(mk) > 127).astype(np.float32))
            fe_r = extract_feat(edit_root / sid / "src.jpg")
            logits_r = head(fe_r).squeeze(0)
            real_scores.append(torch.sigmoid(logits_r).cpu().numpy().reshape(GRID, GRID))
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{n}", flush=True)

    fake_scores = np.stack(fake_scores)
    real_scores = np.stack(real_scores)
    y = np.concatenate([np.zeros(n), np.ones(n)])
    img_scores = np.concatenate([real_scores.max(axis=(1, 2)), fake_scores.max(axis=(1, 2))])
    auc = roc_auc_score(y, img_scores)
    print(f"检测 AUC (src vs SDXL编辑) = {auc:.4f}")

    fake_masks = np.stack(fake_masks)
    flat_s, flat_m = fake_scores.ravel(), fake_masks.ravel()
    pixel_auroc = roc_auc_score(flat_m, flat_s)
    pixel_ap = average_precision_score(flat_m, flat_s)
    print(f"pixel AUROC={pixel_auroc:.4f} AP={pixel_ap:.4f}")

    best_miou, best_f1 = 0.0, 0.0
    for th in np.linspace(0.05, 0.95, 91):
        pred = (fake_scores >= th).astype(np.float32)
        m = fake_masks
        tp = (pred * m).sum(); fp = (pred * (1 - m)).sum(); fn = ((1 - pred) * m).sum()
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        best_miou = max(best_miou, tp / (tp + fp + fn + 1e-8))
        best_f1 = max(best_f1, 2 * prec * rec / (prec + rec + 1e-8))
    print(f"定位 mIoU (best-thr) = {best_miou:.4f} F1 = {best_f1:.4f} 耗时{(time.time()-t0)/60:.1f}min")

    res = {"n": n, "img_auc": float(auc), "loc_miou": float(best_miou),
           "loc_f1": float(best_f1), "pixel_auroc": float(pixel_auroc),
           "pixel_ap": float(pixel_ap), "head": args.head}
    if args.thr is not None:
        pred = (fake_scores >= args.thr).astype(np.float32)
        m = fake_masks
        tp = (pred * m).sum(); fp = (pred * (1 - m)).sum(); fn = ((1 - pred) * m).sum()
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        fixed_miou = tp / (tp + fp + fn + 1e-8)
        fixed_f1 = 2 * prec * rec / (prec + rec + 1e-8)
        res["fixed_thr"] = args.thr
        res["fixed_loc_miou"] = float(fixed_miou)
        res["fixed_loc_f1"] = float(fixed_f1)
        print(f"[统一协议 thr={args.thr}] mIoU={fixed_miou:.4f} F1={fixed_f1:.4f}")

    tag = f"{args.thr}" if args.thr is not None else "bestthr"
    path = f"D:/lunwen/outputs/sdxl_eval_{args.head.replace('_head.pt','')}_thr{tag}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"结果: {path}")


if __name__ == "__main__":
    main()

