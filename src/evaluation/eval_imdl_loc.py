"""评测 IMDLBenCo 外部模型 (MVSS/CAT/IML-ViT...) 在 INP-X 子集上的定位表现

输入: imdl_ckpt 权重 + imdl_inpx_test.json (编辑对) + pred 目录 (模型预测 mask)
输出: mIoU / F1, 按 mask 面积分档
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
DATA_JSON = "D:/lunwen/data/imdl_inpx_test.json"
GRID = 512  # IMDLBenCo 输出 512x512


def load_imdl_mask(p: Path, grid=GRID) -> np.ndarray:
    """IMDLBenCo 保存的预测 mask (0-255 或 0-1)"""
    img = Image.open(p).convert("L")
    img = img.resize((grid, grid), Image.NEAREST)
    a = np.asarray(img).astype(np.float32)
    if a.max() > 1.0:
        a /= 255.0
    return a


def load_gt_mask(mask_path: str, grid=GRID) -> np.ndarray:
    img = Image.open(IMG_ROOT / mask_path).convert("L")
    img = img.resize((grid, grid), Image.NEAREST)
    return (np.asarray(img) > 127).astype(np.float32)


def loc_metrics(pred: np.ndarray, gt: np.ndarray, thresholds=None):
    if thresholds is None:
        lo, hi = pred.min(), pred.max()
        if hi - lo < 1e-6:
            hi = lo + 1.0
        thresholds = np.linspace(lo, hi, 101)
    best_f1, best_iou = 0.0, 0.0
    for th in thresholds:
        p = (pred >= th).astype(np.float32)
        tp = (p * gt).sum(); fp = (p * (1 - gt)).sum(); fn = ((1 - p) * gt).sum()
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        best_f1 = max(best_f1, f1)
        best_iou = max(best_iou, iou)
    return best_iou, best_f1


def main():
    pred_dir = sys.argv[1] if len(sys.argv) > 1 else "D:/lunwen/outputs/imdl_pred_mvss/pred"
    name = Path(pred_dir).parent.name
    records = json.load(open(DATA_JSON))

    # pred 文件名 → 路径
    pred_files = {p.name: p for p in Path(pred_dir).glob("*.png")}
    pred_files.update({p.name: p for p in Path(pred_dir).glob("*.jpg")})
    print(f"pred 文件数: {len(pred_files)}")

    rows = []
    n_miss = 0
    for img_path, mask_path in records:
        if mask_path == "Negative":
            continue  # 只评编辑图
        img_name = Path(img_path).name
        # pred 命名可能带后缀
        key = img_name
        p = pred_files.get(key)
        if p is None:
            # 尝试去掉扩展名的匹配
            stem = Path(img_name).stem
            cand = [v for k, v in pred_files.items() if k.startswith(stem)]
            p = cand[0] if cand else None
        if p is None:
            n_miss += 1
            continue
        pred = load_imdl_mask(p)
        gt = load_gt_mask(mask_path)
        iou, f1 = loc_metrics(pred, gt)
        # mask 占比
        ratio = gt.mean()
        if ratio < 0.02:
            sc = "tiny"
        elif ratio < 0.05:
            sc = "small"
        elif ratio < 0.15:
            sc = "medium"
        else:
            sc = "large"
        rows.append({"img": img_name, "mIoU": float(iou), "F1": float(f1),
                     "size": sc, "ratio": float(ratio)})

    print(f"评测样本: {len(rows)}, 缺失 pred: {n_miss}")
    if not rows:
        return
    print(f"\n=== {name} 在 INP-X 子集上的定位 ===")
    ious = np.array([r["mIoU"] for r in rows])
    f1s = np.array([r["F1"] for r in rows])
    print(f"全部 (n={len(rows)}): mIoU={ious.mean():.4f} F1={f1s.mean():.4f}")
    for sc in ["tiny", "small", "medium", "large"]:
        sub = [r for r in rows if r["size"] == sc]
        if len(sub) < 5:
            continue
        print(f"  {sc} (n={len(sub)}): mIoU={np.mean([r['mIoU'] for r in sub]):.4f} "
              f"F1={np.mean([r['F1'] for r in sub]):.4f}")

    out = Path("D:/lunwen/outputs") / f"imdl_{name}_loc.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
