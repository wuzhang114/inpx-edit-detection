"""统一口径的外部数据集评估: 全部使用主实验的全量读出 (g2_s42 head, 385 维)。

修正此前外部家族使用 g1/sd_seed42 等不同 head 版本导致的口径不一致。
数据集: cocoglide / magicbrush / sdxl / autosplice
输出: outputs/external_uniform_g2.json
"""
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
from sklearn.metrics import roc_auc_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, "D:/lunwen/src")
from baselines.weakly_supervised_v2 import MLPHead  # noqa: E402
from utils.highpass import cross_diff_highpass, pool_to_patch_grid  # noqa: E402

OUT = Path("D:/lunwen/outputs")
GRID, SIZE = 37, 518
HEAD = OUT / "weak_sup_v4_g2_s42_head.pt"
VAL_THR = json.load(open(OUT / "eval500_eval_g2_s42.json", encoding="utf-8"))["val_thr"]
_HUB = os.path.expanduser("~/.cache/torch/hub")


def pairs_of(ds):
    if ds == "cocoglide":
        root = Path("D:/lunwen/data/flame_cocoglide")
        return [(f, root / "Mask" / f.name) for f in sorted((root / "Forged").glob("*.png"))]
    if ds == "magicbrush":
        root = Path("D:/lunwen/data/magicbrush_eval")
        return [(root / "images" / f"{n}.png", root / "masks" / f"{n}.png")
                for n in sorted(p.stem for p in (root / "images").glob("*.png"))]
    if ds == "sdxl":
        root = Path("D:/lunwen/data/sdxl_edits_500")
        return [(d / "edit.jpg", d / "mask.png") for d in sorted(root.iterdir())
                if (d / "edit.jpg").exists()]
    if ds == "autosplice":
        root = Path("D:/lunwen/data/autosplice_extracted/AutoSplice")
        out = []
        for f in sorted((root / "Forged_JPEG100").glob("*.jpg")):
            m = root / "Mask" / f"{f.stem.split('_')[0]}_mask.png"
            if m.exists():
                out.append((f, m))
        return out
    raise ValueError(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="magicbrush,sdxl,autosplice")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    repo = Path(_HUB) / "facebookresearch_dinov2_main"
    if not repo.exists():
        cands = list(Path(_HUB).glob("facebookresearch_dinov2_*"))
        repo = cands[0] if cands else repo
    backbone = torch.hub.load(str(repo), "dinov2_vits14", source="local").cuda().eval()
    head = MLPHead(385, hidden=64).cuda()
    head.load_state_dict(torch.load(HEAD, map_location="cpu"))
    head.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    all_res = {}
    for ds in args.datasets.split(","):
        pairs = pairs_of(ds)
        print(f"[{ds}] {len(pairs)} 对, head=g2(val_thr={VAL_THR:.2f})", flush=True)
        probs, gts = [], []
        with torch.no_grad():
            for i in range(0, len(pairs), args.batch):
                chunk = pairs[i:i + args.batch]
                imgs = []
                for f, _ in chunk:
                    im = Image.open(f).convert("RGB")
                    t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
                    t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
                    imgs.append(t)
                t = torch.cat(imgs, 0)
                t = (t - mean) / std
                tok = backbone.forward_features(t)["x_norm_patchtokens"]
                hp = F.pad(cross_diff_highpass(t), (0, 1, 0, 1))
                hp_p = pool_to_patch_grid(hp, GRID * GRID)
                X = torch.cat([tok, hp_p], dim=-1)
                lo = head(X)
                if lo.dim() == 3:
                    lo = lo.squeeze(-1)
                probs.append(torch.sigmoid(lo).cpu().numpy())
                for _, m in chunk:
                    gm = Image.open(m).convert("L").resize((GRID, GRID), Image.NEAREST)
                    gts.append((np.asarray(gm) > 127).astype(np.float32).ravel())
                if (i // args.batch + 1) % 20 == 0:
                    print(f"   {min(i+args.batch, len(pairs))}/{len(pairs)}", flush=True)
        pm = np.concatenate(probs, 0)
        gt = np.stack(gts)

        def miou(th):
            pred = (pm >= th).astype(np.float32)
            inter = (pred * gt).sum(1)
            union = ((pred + gt) > 0).sum(1)
            return float(np.mean(inter / (union + 1e-8)))

        best_iou, best_thr = 0.0, 0.5
        for th in np.linspace(0.05, 0.95, 91):
            v = miou(float(th))
            if v > best_iou:
                best_iou, best_thr = v, float(th)
        all_res[ds] = {"n": len(pairs), "head": "g2_s42",
                       "miou_at_val_thr": round(miou(float(VAL_THR)), 4),
                       "miou_best37": round(best_iou, 4), "best_thr": round(best_thr, 2),
                       "pixel_auroc": round(float(roc_auc_score(gt.ravel(), pm.ravel())), 4)}
        print(f"   -> mIoU@val_thr={all_res[ds]['miou_at_val_thr']:.4f} "
              f"best37={best_iou:.4f} (thr {best_thr:.2f}) pAUROC={all_res[ds]['pixel_auroc']:.4f}",
              flush=True)

    json.dump(all_res, open(OUT / "external_uniform_g2.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("saved: external_uniform_g2.json", flush=True)


if __name__ == "__main__":
    main()
