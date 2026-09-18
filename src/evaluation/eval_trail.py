"""TRAIL (frozen DINO patch-drift) 在 500 集上的定位评测, 与主协议一致"""
import glob
import json
from pathlib import Path

import numpy as np
from PIL import Image

OUT = "D:/lunwen/outputs/trail_out"
MASKS = Path("D:/lunwen/data/dinolizer_test/masks")
MANIFEST = "D:/lunwen/data/dinolizer_test/manifest.csv"
GRID = 37

import csv
with open(MANIFEST) as f:
    rows = list(csv.DictReader(f))
miou_best_sum = f1_best_sum = miou_f05_sum = n = 0
npzs = sorted(glob.glob(f"{OUT}/*.npz"))
for idx, npz in enumerate(npzs):
    d = np.load(npz)
    scores = d["scores"]  # [L, 32, 32]
    smap = scores.mean(axis=0)  # 层平均 (无偏)
    img = Image.fromarray(smap).resize((GRID, GRID), Image.BILINEAR)
    s = np.asarray(img, dtype=np.float32)
    mask_rel = rows[idx]["mask"]
    m = Image.open(MASKS / mask_rel.split("/")[-1])
    gt = (np.asarray(m.convert("L").resize((GRID, GRID), Image.NEAREST)) > 127).astype(np.float32)
    best_iou = best_f1 = 0.0
    for th in np.linspace(0.05, 0.95, 91):
        pr = (s >= th).astype(np.float32)
        tp = (pr * gt).sum(); fp = (pr * (1 - gt)).sum(); fn = ((1 - pr) * gt).sum()
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        best_iou = max(best_iou, tp / (tp + fp + fn + 1e-8))
        best_f1 = max(best_f1, 2 * prec * rec / (prec + rec + 1e-8))
    pr5 = (s >= 0.5).astype(np.float32)
    tp = (pr5 * gt).sum(); fp = (pr5 * (1 - gt)).sum(); fn = ((1 - pr5) * gt).sum()
    miou_f05 = tp / (tp + fp + fn + 1e-8)
    miou_best_sum += best_iou; f1_best_sum += best_f1; miou_f05_sum += miou_f05
    n += 1

res = {"method": "TRAIL (dinov2_vits14, all layers averaged)", "n": n,
       "loc_miou_best": float(miou_best_sum / n), "loc_f1_best": float(f1_best_sum / n),
       "loc_miou_fixed05": float(miou_f05_sum / n)}
print(json.dumps(res, indent=1))
with open("D:/lunwen/outputs/trail_result.json", "w") as f:
    json.dump(res, f, indent=1)
print("saved -> outputs/trail_result.json")
