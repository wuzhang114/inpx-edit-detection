"""v4 head 在 CocoGlide (TRAIL 官方测试集) 上的零训练迁移评测

统一协议 (顶级 AI 决策): INP-X-only 训练 → INP-X validation threshold (--thr) → zero-shot
指标: patch AUROC/AP (TRAIL 报告协议) + fixed-thr mIoU/F1 + best-thr (参考) + 检测 AUC
用法: python src/evaluation/eval_cocoglide.py [--head weak_sup_v4_head.pt] [--thr 0.45]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, "D:/lunwen/src")
from baselines.weakly_supervised_v2 import MLPHead
from utils.highpass import cross_diff_highpass, pool_to_patch_grid

ASSETS = Path("D:/lunwen/data/cocoglide_400/assets")
GRID = 37
SIZE = 518
NATIVE = 448

p = argparse.ArgumentParser()
p.add_argument("--head", default="weak_sup_v4_head.pt", help="outputs/ 下的 head 文件名")
p.add_argument("--thr", type=float, default=None,
               help="INP-X 验证集固定阈值 (统一协议); 不传则报 best-thr")
args = p.parse_args()

import torch.hub as hub
model = hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).cuda().eval()
head = MLPHead(385).cuda()
head.load_state_dict(torch.load(f"D:/lunwen/outputs/{args.head}", map_location="cpu"))
head.eval()


def scores_of(path):
    img = np.asarray(Image.open(path).convert("RGB"))
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    t = (t - mean) / std
    with torch.no_grad():
        tok = model.forward_features(t)["x_norm_patchtokens"]
        hp = F.pad(cross_diff_highpass(t), (0, 1, 0, 1))
        hp_p = pool_to_patch_grid(hp, GRID * GRID)
        X = torch.cat([tok, hp_p], dim=-1).cuda()
        logits = head(X).squeeze(0)
    return torch.sigmoid(logits).cpu().numpy().reshape(GRID, GRID)


ids = sorted({p.name.split("_")[0] + "_" + p.name.split("_")[1] for p in ASSETS.glob("*_fake.png")})
print(f"n={len(ids)}", flush=True)

fake_img_scores, real_img_scores = [], []
auroc_sum = ap_sum = miou_sum = dice5_sum = n = 0
smap_cache, m37_cache = [], []
with torch.no_grad():
    for i, sid in enumerate(ids):
        smap = scores_of(ASSETS / f"{sid}_fake.png")
        mask = np.asarray(Image.open(ASSETS / f"{sid}_mask.png").convert("L")) > 127
        # patch AUROC/AP: 分数上采样到原生分辨率 vs GT
        smap_up = np.asarray(Image.fromarray(smap).resize((NATIVE, NATIVE), Image.BILINEAR))
        flat_s = smap_up.ravel(); flat_m = mask.ravel().astype(int)
        auroc_sum += roc_auc_score(flat_m, flat_s)
        ap_sum += average_precision_score(flat_m, flat_s)
        # mIoU (37x37 网格, best-thr 91 档, 与主协议一致)
        m37 = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((GRID, GRID), Image.NEAREST)) > 127
        m37 = m37.astype(np.float32)
        smap_cache.append(smap); m37_cache.append(m37)
        best = 0.0
        for th in np.linspace(0.05, 0.95, 91):
            pr = (smap >= th).astype(np.float32)
            tp = (pr * m37).sum(); fp = (pr * (1 - m37)).sum(); fn = ((1 - pr) * m37).sum()
            best = max(best, tp / (tp + fp + fn + 1e-8))
        miou_sum += best
        # fixed-0.5 dice
        pr5 = (smap >= 0.5).astype(np.float32)
        tp = (pr5 * m37).sum(); fp = (pr5 * (1 - m37)).sum(); fn = ((1 - pr5) * m37).sum()
        dice5_sum += 2 * tp / (2 * tp + fp + fn + 1e-8)
        fake_img_scores.append(float(smap.max()))
        real_img_scores.append(float(scores_of(ASSETS / f"{sid}_real.png").max()))
        n += 1
        if i % 100 == 99:
            print(f"  {i+1}/{len(ids)}", flush=True)

y = np.concatenate([np.zeros(len(real_img_scores)), np.ones(len(fake_img_scores))])
img_scores = np.concatenate([real_img_scores, fake_img_scores])
auc_det = roc_auc_score(y, img_scores)
res = {"dataset": "CocoGlide (TRAIL official test, 400 sources)", "n": n,
       "patch_auroc": float(auroc_sum / n),
       "patch_ap": float(ap_sum / n),
       "loc_miou_best37": float(miou_sum / n),
       "fixed05_dice": float(dice5_sum / n),
       "det_auc": float(auc_det), "head": args.head}
if args.thr is not None:
    # 固定阈值全局 mIoU/F1 (所有图合并像素, 复用缓存不重复推理)
    smap_all = np.concatenate([s.ravel() for s in smap_cache])
    m37_all = np.concatenate([m.ravel() for m in m37_cache])
    pred = (smap_all >= args.thr).astype(np.float32)
    tp = (pred * m37_all).sum(); fp = (pred * (1 - m37_all)).sum(); fn = ((1 - pred) * m37_all).sum()
    prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
    res["fixed_thr"] = args.thr
    res["fixed_loc_miou"] = float(tp / (tp + fp + fn + 1e-8))
    res["fixed_loc_f1"] = float(2 * prec * rec / (prec + rec + 1e-8))
    # 逐图 fixed-thr mIoU + size 分层 (supplementary)
    per_img = []
    ratios = []
    for s, m in zip(smap_cache, m37_cache):
        p = (s >= args.thr).astype(np.float32)
        tpi = (p * m).sum(); fpi = (p * (1 - m)).sum(); fni = ((1 - p) * m).sum()
        per_img.append(tpi / (tpi + fpi + fni + 1e-8))
        ratios.append(float(m.mean()))
    res["per_image_fixed_miou"] = [round(float(v), 6) for v in per_img]
    res["per_image_ratio"] = [round(float(v), 6) for v in ratios]
    # size 分层 IoU (逐图平均)
    strata = {"tiny": [], "small": [], "medium": [], "large": []}
    for r, iou in zip(ratios, per_img):
        sc = "tiny" if r < 0.02 else ("small" if r < 0.05
              else ("medium" if r < 0.15 else "large"))
        strata[sc].append(iou)
    res["size_stratified_iou"] = {k: {"n": len(v),
        "mIoU": round(float(np.mean(v)), 4) if v else None} for k, v in strata.items()}
    print(f"[统一协议 thr={args.thr}] mIoU={res['fixed_loc_miou']:.4f} "
          f"size分层IoU={res['size_stratified_iou']}")
import json
print(json.dumps(res, indent=1))
tag = args.head.replace("_head.pt", "")
thr_tag = f"thr{args.thr}" if args.thr is not None else "bestthr"
with open(f"D:/lunwen/outputs/cocoglide_{tag}_{thr_tag}_result.json", "w") as f:
    json.dump(res, f, indent=1)
