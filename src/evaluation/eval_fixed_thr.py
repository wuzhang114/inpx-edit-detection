"""Fixed-threshold localization of the v4 readout on 500 edited images.

协议: 图像 → DINOv2+hp 特征 → v4 头 → sigmoid 分数 (37×37) → 阈值 0.5 → mIoU/F1
对比: DinoLizer 0.260 (固定0.5, 全分辨率) — 口径差异: 本脚本 37×37 网格
用法: python eval_fixed_thr.py [--head weak_sup_v4_head.pt] [--thr 0.5]
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.weakly_supervised_v2 import MLPHead
from utils.highpass import cross_diff_highpass, pool_to_patch_grid

IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
DATA_JSON = "D:/lunwen/data/imdl_inpx_test.json"
GRID = 37
IMAGE_SIZE = 518


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="weak_sup_v4_head.pt", help="outputs/ 下的 head 文件名")
    p.add_argument("--thr", type=float, default=0.5)
    args = p.parse_args()

    import torch.hub as hub
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
    model = model.cuda().eval()
    head = MLPHead(385).cuda()
    head.load_state_dict(torch.load(f"D:/lunwen/outputs/{args.head}", map_location="cpu"))
    head.eval()

    records = json.load(open(DATA_JSON))
    miou_sum = f1_sum = n = 0
    per_img_list = []
    t0 = time.time()
    with torch.no_grad():
        for img_path, mask_path in records:
            if mask_path == "Negative":
                continue
            img = np.asarray(Image.open(img_path).convert("RGB"))
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
            t = F.interpolate(t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
            t = (t - mean) / std
            tok = model.forward_features(t)["x_norm_patchtokens"]
            hp = F.pad(cross_diff_highpass(t), (0, 1, 0, 1))
            hp_pooled = pool_to_patch_grid(hp, GRID * GRID)
            X = torch.cat([tok, hp_pooled], dim=-1).cuda()
            logits = head(X).squeeze(0)
            scores = torch.sigmoid(logits).cpu().numpy().reshape(GRID, GRID)

            m = Image.open(mask_path).convert("L").resize((GRID, GRID), Image.NEAREST)
            gt = (np.asarray(m) > 127).astype(np.float32)
            pr = (scores >= args.thr).astype(np.float32)
            tp = (pr * gt).sum(); fp = (pr * (1 - gt)).sum(); fn = ((1 - pr) * gt).sum()
            iou = tp / (tp + fp + fn + 1e-8)
            prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            miou_sum += iou; f1_sum += f1; n += 1
            per_img_list.append(float(iou))
            if n % 100 == 0:
                print(f"{n}/500", flush=True)

    miou = miou_sum / n; f1 = f1_sum / n
    print(f"{args.head} fixed-{args.thr}: mIoU={miou:.4f} F1={f1:.4f} n={n} 耗时{(time.time()-t0)/60:.1f}min")
    tag = args.head.replace("_head.pt", "")
    with open(f"D:/lunwen/outputs/fixed_thr{args.thr}_{tag}.json", "w") as f:
        json.dump({"thr": args.thr, "loc_miou": float(miou), "loc_f1": float(f1),
                   "n": n, "grid": GRID, "head": args.head,
                   "per_image_miou": per_img_list}, f, indent=1)


if __name__ == "__main__":
    main()
