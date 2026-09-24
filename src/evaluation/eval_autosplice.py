"""AutoSplice (WMF@CVPR2023) 外部验证: DALL-E 2 文本引导编辑, 跨编辑器家族迁移

协议: 只用 JPEG-100; 固定阈值 (INP-X 验证集阈值 0.19); 报告检测 AUC + 定位 IoU;
smoke test (--smoke 20) 先验证结构与 mask 对齐; 结果进 supplementary "out-of-family transfer"。

用法: python eval_autosplice.py [--smoke 20] [--head weak_sup_v4_sd_seed42_head.pt] [--thr 0.19]
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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.weakly_supervised_v2 import MLPHead
from utils.highpass import cross_diff_highpass, pool_to_patch_grid

OUT = Path("D:/lunwen/outputs")
GRID = 37
SIZE = 518


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="D:/lunwen/data/autosplice/jpeg100")
    p.add_argument("--head", default="weak_sup_v4_sd_seed42_head.pt")
    p.add_argument("--thr", type=float, default=0.19)
    p.add_argument("--smoke", type=int, default=0, help=">0 时只测前 N 张 (结构/mask 对齐验证)")
    args = p.parse_args()
    root = Path(args.root)
    print(f"AutoSplice root: {root}")
    print("目录结构:", [str(x.relative_to(root)) for x in sorted(root.rglob('*'))[:20]])

    # 按官方结构: Forged_JPEG100/ + Authentic/ + Mask/ + Caption/
    mani_dir = root / "Forged_JPEG100"
    auth_dir = root / "Authentic"
    mask_dir = root / "Mask"
    if not mani_dir.exists():
        print("!! 未找到 Forged_JPEG100, 请检查路径")
        return
    print(f"manipulated={mani_dir.name} authentic={auth_dir.name} masks={mask_dir.name}")

    mani = sorted(mani_dir.glob("*.jpg")) + sorted(mani_dir.glob("*.png"))
    print(f"manipulated 图: {len(mani)}")

    import torch.hub as hub
    model = hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).cuda().eval()
    head = MLPHead(385).cuda()
    head.load_state_dict(torch.load(OUT / args.head, map_location="cpu"))
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

    # mask 匹配: Forged {id}_{index}.jpg → Mask/{id}_mask.png (同 id 多编辑共享 mask)
    def find_mask(img_path):
        if mask_dir is None:
            return None
        fid = img_path.stem.split("_")[0]  # 取 {id}
        cand = mask_dir / f"{fid}_mask.png"
        return cand if cand.exists() else None

    n = min(len(mani), args.smoke) if args.smoke else len(mani)
    fake_scores, real_scores = [], []
    miou_sum, n_mask = 0, 0
    per_img = []
    tp_list, fp_list, fn_list, src_ids = [], [], [], []
    t0 = time.time()
    for i, fp in enumerate(mani[:n]):
        smap = scores_of(fp)
        fake_scores.append(float(smap.max()))
        src_ids.append(fp.stem.split("_")[0])          # source id (cluster bootstrap 单位)
        mp = find_mask(fp)
        if mp is not None:
            m = np.asarray(Image.open(mp).convert("L"))
            m37 = (np.asarray(Image.fromarray(m).resize((GRID, GRID), Image.NEAREST)) > 127).astype(np.float32)
            pr = (smap >= args.thr).astype(np.float32)
            tp = (pr * m37).sum(); fpp = (pr * (1 - m37)).sum(); fn = ((1 - pr) * m37).sum()
            iou = tp / (tp + fpp + fn + 1e-8)
            miou_sum += iou; n_mask += 1
            per_img.append(iou)
            tp_list.append(float(tp)); fp_list.append(float(fpp)); fn_list.append(float(fn))
        if auth_dir is not None:
            ap = auth_dir / f"{fp.stem.split('_')[0]}.jpg"   # Forged {id}_{index} → Authentic {id}.jpg
            if ap.exists():
                real_scores.append(float(scores_of(ap).max()))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n} 有mask={n_mask} 耗时{(time.time()-t0)/60:.1f}min", flush=True)

    res = {"dataset": "AutoSplice (DALL-E2, WMF@CVPR2023)", "n": n, "thr": args.thr,
           "n_with_mask": n_mask, "head": args.head,
           "n_authentic": len(real_scores), "bootstrap_unit": "source_id"}
    if n_mask:
        # global micro IoU (合并像素) + per-image mean IoU
        g_tp, g_fp, g_fn = sum(tp_list), sum(fp_list), sum(fn_list)
        res["loc_miou_fixed"] = float(np.mean(per_img))            # per-image mean
        res["loc_miou_global_micro"] = float(g_tp / (g_tp + g_fp + g_fn + 1e-8))
        res["per_image_fixed_miou"] = [round(float(v), 6) for v in per_img]
        res["per_image_tp"] = tp_list; res["per_image_fp"] = fp_list; res["per_image_fn"] = fn_list
        res["source_ids"] = src_ids
        print(f"[定位 fixed-thr={args.thr}] per-image mIoU={res['loc_miou_fixed']:.4f} "
              f"global micro={res['loc_miou_global_micro']:.4f} (n={n_mask})")
    if real_scores:
        y = np.concatenate([np.zeros(len(real_scores)), np.ones(len(fake_scores))])
        sc = np.concatenate([real_scores, fake_scores])
        res["det_auc"] = float(roc_auc_score(y, sc))
        res["fake_scores"] = [round(float(v), 6) for v in fake_scores]
        res["real_scores"] = [round(float(v), 6) for v in real_scores]
        print(f"[检测] AUC={res['det_auc']:.4f} (fake={len(fake_scores)} real={len(real_scores)})")
    print(json.dumps(res, indent=1))
    tag = "smoke" if args.smoke else "full"
    with open(OUT / f"autosplice_{tag}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
