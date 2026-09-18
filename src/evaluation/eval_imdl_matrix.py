"""统一评测 IMDLBenCo 批量输出: 定位 mIoU (best-thr 512) + 检测 AUC (pred_labels)
用法: python src/evaluation/eval_imdl_matrix.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

DATASETS = {
    'inpx': 'D:/lunwen/data/imdl_inpx_test.json',
    'cocoglide': 'D:/lunwen/data/imdl_cocoglide_list.json',
    'magicbrush': 'D:/lunwen/data/imdl_magicbrush_list.json',
    'sdxl': 'D:/lunwen/data/imdl_sdxl_list.json',
    'celeba': 'D:/lunwen/data/imdl_celeba_list.json',
}
METHODS = ['objectformer', 'mvss', 'iml_vit']


def loc_miou(pred_dir, records):
    preds = {}
    for p in Path(pred_dir).iterdir():
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            preds[p.stem] = p
    miou_sum = n = 0
    for img_path, mask_path in records:
        if mask_path == 'Negative':
            continue
        stem = Path(img_path).stem
        p = preds.get(stem)
        if p is None:
            cand = [v for k, v in preds.items() if k.startswith(stem)]
            p = cand[0] if cand else None
        if p is None:
            continue
        pr = np.asarray(Image.open(p).convert('L').resize((128, 128), Image.BILINEAR), dtype=np.float32) / 255.0
        gt = (np.asarray(Image.open(mask_path).convert('L').resize((128, 128), Image.NEAREST)) > 127).astype(np.float32)
        best = 0.0
        lo, hi = pr.min(), pr.max()
        if hi - lo < 1e-6:
            hi = lo + 1.0
        for th in np.linspace(lo, hi, 101):
            pbin = (pr >= th).astype(np.float32)
            tp = (pbin * gt).sum(); fp = (pbin * (1 - gt)).sum(); fn = ((1 - pbin) * gt).sum()
            best = max(best, tp / (tp + fp + fn + 1e-8))
        miou_sum += best; n += 1
    return miou_sum / n if n else None, n


def det_auc(labels_path, records):
    if not Path(labels_path).exists():
        return None
    labels = json.load(open(labels_path))
    ys, ss = [], []
    for img_path, mask_path in records:
        stem = Path(img_path).stem
        if stem not in labels:
            cand = [k for k in labels if k.startswith(stem)]
            stem = cand[0] if cand else None
        if stem is None:
            continue
        v = labels[stem]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ys.append(0 if mask_path == 'Negative' else 1)
        ss.append(v)
    if len(set(ys)) < 2 or len(ss) < 10:
        return None
    return float(roc_auc_score(ys, ss))


results = {}
for method in METHODS:
    for ds, dj in DATASETS.items():
        records = json.load(open(dj))
        pred_dir = f'D:/lunwen/outputs/imdl_{ds}_{method}/pred'
        labels_path = f'D:/lunwen/outputs/imdl_{ds}_{method}/pred_labels.json'
        miou, n = loc_miou(pred_dir, records)
        auc = det_auc(labels_path, records)
        results[f'{ds}_{method}'] = {'loc_miou': miou, 'n': n, 'det_auc': auc}
        print(f'{ds:10s} {method:12s} mIoU={miou if miou is None else round(miou,4)} (n={n}) detAUC={auc if auc is None else round(auc,4)}')

json.dump(results, open('D:/lunwen/outputs/imdl_matrix_results.json', 'w'), indent=1)
