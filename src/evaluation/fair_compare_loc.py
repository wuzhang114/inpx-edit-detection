"""Compare v2/v3/v4 readouts on the same 500 INP-X edits used by IMDLBenCo.

与 eval_imdl_loc.py (MVSS) 使用同一测试集, 保证可比性。
用法: python fair_compare_loc.py [--head <head.pt> --tag <tag>]  (指定单头)
      默认评测 v2/v3/v4 三个头。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.weakly_supervised_v2 import MLPHead

CACHE = "D:/lunwen/data/features_cache"
IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
DATA_JSON = "D:/lunwen/data/imdl_inpx_test.json"
OUT = Path("D:/lunwen/outputs")
GRID = 37  # 特征网格


def eval_head(head_path: Path, tag: str, eval_items, dino_mm, hp_mm, dino_dim):
    head = MLPHead(dino_dim + 1).cuda()
    head.load_state_dict(torch.load(head_path))
    head.eval()

    rows = []
    with torch.no_grad():
        for idx, mask_path in eval_items:
            dino = torch.from_numpy(dino_mm[idx].astype(np.float32))
            hp = torch.from_numpy(hp_mm[idx].astype(np.float32))
            X = torch.cat([dino, hp], dim=-1).unsqueeze(0).cuda()
            logits = head(X).squeeze(0)
            scores = torch.sigmoid(logits).cpu().numpy().reshape(GRID, GRID)

            # GT mask → 37x37
            m = Image.open(IMG_ROOT / mask_path).convert("L")
            m = m.resize((GRID, GRID), Image.NEAREST)
            gt = (np.asarray(m) > 127).astype(np.float32)

            # mIoU 阈值扫描
            best_iou, best_f1 = 0.0, 0.0
            for th in np.linspace(0.05, 0.95, 91):
                p = (scores >= th).astype(np.float32)
                tp = (p * gt).sum(); fp = (p * (1 - gt)).sum(); fn = ((1 - p) * gt).sum()
                prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
                iou = tp / (tp + fp + fn + 1e-8)
                f1 = 2 * prec * rec / (prec + rec + 1e-8)
                best_iou = max(best_iou, iou)
                best_f1 = max(best_f1, f1)
            ratio = gt.mean()
            sc = "tiny" if ratio < 0.02 else ("small" if ratio < 0.05
                  else ("medium" if ratio < 0.15 else "large"))
            rows.append({"mIoU": float(best_iou), "F1": float(best_f1),
                         "size": sc, "ratio": float(ratio)})

    ious = np.array([r["mIoU"] for r in rows])
    print(f"\n=== {tag} (同 MVSS 测试集, 500 编辑图) ===")
    print(f"全部 (n={len(rows)}): mIoU={ious.mean():.4f}")
    for sc in ["tiny", "small", "medium", "large"]:
        sub = [r for r in rows if r["size"] == sc]
        if len(sub) >= 5:
            print(f"  {sc} (n={len(sub)}): mIoU={np.mean([r['mIoU'] for r in sub]):.4f}")
    with open(OUT / f"fairloc_{tag}.json", "w") as f:
        json.dump(rows, f, indent=1)
    return float(ious.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", type=str, default=None, help="单头评测: head.pt 文件名")
    p.add_argument("--tag", type=str, default=None, help="单头评测: 输出 tag (fairloc_<tag>.json)")
    args = p.parse_args()

    with open(f"{CACHE}/labels.json") as f:
        labels = json.load(f)
    with open(f"{CACHE}/metadata.json") as f:
        meta = json.load(f)
    n_total, n_patches, dino_dim = meta["n_total"], meta["n_patches"], meta["dino_dim"]

    dino_mm = np.memmap(f"{CACHE}/dino_tokens.npy", dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, dino_dim))
    hp_mm = np.memmap(f"{CACHE}/highpass.npy", dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))

    # path → idx
    path2idx = {l["path"]: i for i, l in enumerate(labels)}

    records = json.load(open(DATA_JSON))
    # 找到 500 编辑对在缓存中的 idx
    eval_items = []
    for img_path, mask_path in records:
        if mask_path == "Negative":
            continue
        rel = str(Path(img_path).relative_to(IMG_ROOT))  # 保持与 labels.json 相同的分隔符
        idx = path2idx.get(rel)
        if idx is None:
            print(f"缓存缺失: {rel}")
            continue
        eval_items.append((idx, mask_path))
    print(f"匹配缓存: {len(eval_items)} 张编辑图")

    if args.head:
        assert args.tag, "--tag 必须与 --head 同时给出"
        head_path = OUT / args.head
        assert head_path.exists(), f"head 不存在: {head_path}"
        eval_head(head_path, args.tag, eval_items, dino_mm, hp_mm, dino_dim)
        return

    for head_name, tag in [("weak_sup_head", "v2"), ("weak_sup_v3_head", "v3"),
                           ("weak_sup_v4_head", "v4")]:
        hp_path = OUT / f"{head_name}.pt"
        if not hp_path.exists():
            print(f"跳过 {head_name}")
            continue
        eval_head(hp_path, tag, eval_items, dino_mm, hp_mm, dino_dim)


if __name__ == "__main__":
    main()
