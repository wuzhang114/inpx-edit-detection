"""B: 标注成本口径审计 (P1-1) —— 区分 training-mask budget 与固定验证注记成本
输出: outputs/mask_cost_audit.json
"""
import json
from pathlib import Path
import numpy as np

OUT = Path("D:/lunwen/outputs")
CACHE = Path("D:/lunwen/data/features_cache")

man = json.load(open(OUT / "mask_budget_manifest.json", encoding="utf-8"))
split = json.load(open(OUT / "split_budget_master.json", encoding="utf-8"))
labels = json.load(open(CACHE / "labels.json", encoding="utf-8"))
labels_g = np.array([l["label"] for l in labels])

n_ann = int(man["n_annotations"])
val_idx = split["val_idx"]
val_edit = [i for i in val_idx if labels_g[i] == 1]

# 验证集编辑图的唯一 (canonical source, mask) 注记
uniq_val = set()
for i in val_edit:
    l = labels[i]
    mp = l.get("mask_path")
    if mp:
        uniq_val.add(mp)
n_val_ann = len(uniq_val)

res = {
    "n_training_annotations": n_ann,
    "n_val_edit_rows": len(val_edit),
    "n_val_unique_annotations": n_val_ann,
    "n_total_annotations_training_plus_val": n_ann + n_val_ann,
    "note": "阈值选择使用验证编辑集 mask (k 之外固定成本); k 应称为 training-mask budget",
    "per_budget": {},
}
for k in [0, 10, 50, 200, 1000, 5000]:
    tot = k + n_val_ann
    res["per_budget"][str(k)] = {
        "k": k, "train_masks": k, "total_masks_incl_val": tot,
        "share_of_all_annotations_pct": round(100.0 * tot / (n_ann + n_val_ann), 2),
        "train_share_pct": round(100.0 * k / (n_ann + n_val_ann), 2),
    }
res["per_budget"]["ALL"] = {
    "k": n_ann, "train_masks": n_ann, "total_masks_incl_val": n_ann + n_val_ann,
    "share_of_all_annotations_pct": 100.0, "train_share_pct": round(100.0 * n_ann / (n_ann + n_val_ann), 2)}

json.dump(res, open(OUT / "mask_cost_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:900])
