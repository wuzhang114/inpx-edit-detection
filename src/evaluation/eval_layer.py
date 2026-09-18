"""层相变专用评估: 在子集内 split 的 test_idx 上评估 (fixed-thr mIoU)。

与 eval_seen500 相同协议, 但测试图直接用 layer 子集 split 的 test_idx,
而不是 imdl_inpx_test.json (后者与层子集不重叠)。
用法: python src/evaluation/eval_layer.py --head ... --layer 6 --budget 200
输出: outputs/eval_layer_lay{L}_b{B}.json
"""
import argparse
import json
import os
import sys
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.weakly_supervised_v2 import MLPHead  # noqa: E402

IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
OUT = Path("D:/lunwen/outputs")
GRID = 37
NP = GRID * GRID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--budget", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache = f"D:/lunwen/data/features_cache_layer{args.layer}"
    sub = json.load(open(f"{cache}/labels.json", encoding="utf-8"))
    split = json.load(open(OUT / f"layer_split_L{args.layer}.json"))
    dd = np.memmap(f"{cache}/dino_tokens.npy", dtype="float16", mode="r",
                   shape=(len(sub), 1369, 384))

    head = MLPHead(384, hidden=64).cuda()
    head.load_state_dict(torch.load(OUT / args.head, map_location="cpu"))
    head.eval()

    def gt_mask(idx):
        mp = sub[idx].get("mask_path")
        if mp and (IMG_ROOT / mp).exists():
            m = Image.open(IMG_ROOT / mp).convert("L").resize((GRID, GRID), Image.NEAREST)
            return (np.asarray(m) > 127).astype(np.float32).ravel()
        return np.zeros(NP, np.float32)

    # test 编辑图 + real (用 label==0 的 test 图)
    test_edit = [i for i in split["test_idx"] if sub[i].get("label") == 1
                 and sub[i].get("mask_path")]
    test_real = [i for i in split["test_idx"] if sub[i].get("label") == 0]
    print(f"test edit={len(test_edit)} real={len(test_real)}", flush=True)

    def predict(idxs):
        scores = []
        with torch.no_grad():
            for i in idxs:
                X = torch.from_numpy(dd[i].astype(np.float32)).unsqueeze(0).cuda()
                lo = head(X)
                if lo.dim() == 3:
                    lo = lo.squeeze(-1)
                scores.append(torch.sigmoid(lo.squeeze(0)).cpu().numpy())
        return np.stack(scores)

    se = predict(test_edit)
    sr = predict(test_real)
    me = np.stack([gt_mask(i) for i in test_edit])

    # 固定阈值: 用子集 val 编辑图选 (与主协议一致)
    val_edit = [i for i in split["val_idx"] if sub[i].get("label") == 1
                and sub[i].get("mask_path")]
    sv = predict(val_edit)
    mv = np.stack([gt_mask(i) for i in val_edit])
    best_thr, best_vm = 0.5, 0.0
    for th in np.linspace(0.05, 0.95, 91):
        p = (sv >= th).astype(np.float32)
        tp = (p * mv).sum(); fp = (p * (1 - mv)).sum(); fn = ((1 - p) * mv).sum()
        iou = tp / (tp + fp + fn + 1e-8)
        if iou > best_vm:
            best_vm, best_thr = iou, th
    # fixed-thr mIoU
    pred = (se >= best_thr).astype(np.float32)
    tp = (pred * me).sum(); fp = (pred * (1 - me)).sum(); fn = ((1 - pred) * me).sum()
    fixed_iou = tp / (tp + fp + fn + 1e-8)
    # per-image fixed
    per_img = []
    for k in range(len(test_edit)):
        p = (se[k] >= best_thr).astype(np.float32); m = me[k]
        tpi = (p * m).sum(); fpi = (p * (1 - m)).sum(); fni = ((1 - p) * m).sum()
        per_img.append(tpi / (tpi + fpi + fni + 1e-8))
    # det AUC (test edit vs test real)
    det_auc = roc_auc_score(np.concatenate([np.ones(len(test_edit)),
                                            np.zeros(len(test_real))]),
                            np.concatenate([se.max(1), sr.max(1)]))
    px_auroc = roc_auc_score(me.ravel(), se.ravel())

    tag = f"lay{args.layer}_b{args.budget}_s{args.seed}"
    res = {"layer": args.layer, "budget": args.budget, "val_thr": float(best_thr),
           "fixed_miou": float(fixed_iou), "det_auc": float(det_auc),
           "pixel_auroc": float(px_auroc), "n_test_edit": int(len(test_edit)),
           "per_image_fixed_miou": [round(float(v), 6) for v in per_img]}
    with open(OUT / f"eval_layer_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"layer{args.layer} b{args.budget}: thr={best_thr:.2f} mIoU={fixed_iou:.4f} "
          f"det={det_auc:.4f} AUROC={px_auroc:.4f}", flush=True)


if __name__ == "__main__":
    main()
