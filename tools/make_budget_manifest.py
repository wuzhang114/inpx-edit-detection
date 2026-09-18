"""生成 mask budget manifest(注记 = 唯一 mask 标注;层内 round-robin;组成统计)。

用法: python tools/make_budget_manifest.py
输出: outputs/mask_budget_manifest.json
  - annotations : 唯一注记列表 (canonical, mask_path) — std/exchange 共 mask 计 1 次(已验证 20,000 对 mask_path 相同)
  - budgets     : {10/50/200/1000/5000/ALL: [ann_id...]} — 嵌套 10⊂50⊂200⊂1000⊂5000⊂ALL
  - composition : 每个 k 点的 (domain, editor, size_class) 组成比例(样本=注记), 与 ALL 的偏差 >±5pp 标注
  - rep_idx     : 每注记的代表样本(优先 standard 版)
"""
import argparse
import json
import os
import sys
import io
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CACHE = "D:/lunwen/data/features_cache"
OUT = "D:/lunwen/outputs"
MASTER = f"{OUT}/split_budget_master.json"
MANIFEST_SEED = 20260831
BUDGETS = [10, 50, 200, 1000, 5000, None]  # None = ALL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=MANIFEST_SEED)
    args = ap.parse_args()

    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from data.canonical_source import canonical_source_id

    labels = json.load(open(f"{CACHE}/labels.json", encoding="utf-8"))
    master = json.load(open(MASTER, encoding="utf-8"))
    train_set = set(master["train_groups"])

    # ---- 注记 = (canonical, mask_path) 唯一;范围 = master train 组的编辑样本 ----
    ann_map = {}  # key -> {"canonical", "mask_path", "rep_idx", "n_env", "domain", "editor", "size_class"}
    for l in labels:
        if l["path"].replace("\\", "/") in {p.replace("\\", "/") for p in []}:
            continue
        cid = canonical_source_id(l["path"], l.get("cat"))
        if cid not in train_set:
            continue
        if l.get("label") != 1:
            continue
        mp = (l.get("mask_path") or "").replace("\\", "/")
        key = (cid, mp)
        if key not in ann_map:
            ann_map[key] = {
                "canonical": cid, "mask_path": mp,
                "rep_idx": l["idx"],
                "domain": l.get("cat"), "editor": l.get("model"), "size_class": l.get("size_class"),
                "n_matrix_versions": 1,
            }
        else:
            ann_map[key]["n_matrix_versions"] += 1  # std/exc 版本数

    anns = list(ann_map.values())
    print(f"master train 组注记: {len(anns)}")

    # ---- 分层 (domain, editor, size_class) → 层内 shuffle → 按全量比例的平滑加权调度 ----
    layers = defaultdict(list)
    n_total = 0
    for a in anns:
        layers[(a["domain"], a["editor"] or "real", a["size_class"] or "real")].append(a)
        n_total += 1
    rng = np.random.RandomState(args.seed)
    for k, v in layers.items():
        rng.shuffle(v)
    layer_keys = sorted(layers.keys())
    p = {k: len(v) / n_total for k, v in layers.items()}
    # 平滑加权调度: 每步取 (count_L + 0.5)/p_L 最小的层;任意前缀组成 ≈ 全量联合分布;嵌套稳定
    counts = {k: 0 for k in layer_keys}
    merged = []
    for _ in range(n_total):
        best = min(layer_keys, key=lambda k: (counts[k] + 0.5) / p[k])
        merged.append((best, layers[best][counts[best]]))
        counts[best] += 1
    ordered = [a for _, a in merged]

    # ---- 预算与组成 ----
    n_all = len(ordered)
    budgets = {}
    composition = {}
    for b in BUDGETS:
        k = n_all if b is None else min(b, n_all)
        sub = ordered[:k]
        budgets["ALL" if b is None else str(b)] = [a["canonical"] + "||" + a["mask_path"] for a in sub]
        comp = defaultdict(int)
        for a in sub:
            comp[f"{a['domain']}/{a['editor']}/{a['size_class']}"] += 1
        composition["ALL" if b is None else str(b)] = {
            "n": len(sub), "compose": dict(comp)}

    # 组成对比: 每点 vs ALL(偏差 >±5pp 标注)
    all_comp = composition["ALL"]["compose"]
    all_total = sum(all_comp.values())
    dev_notes = {}
    for key in budgets:
        if key == "ALL":
            continue
        c = composition[key]["compose"]
        tot = composition[key]["n"]
        devs = {}
        for kk, vv in all_comp.items():
            p_all = vv / all_total
            p_k = c.get(kk, 0) / tot
            dev = p_k - p_all
            if abs(dev) > 0.05:
                devs[kk] = {"p_all": round(p_all, 4), "p_k": round(p_k, 4), "dev": round(dev, 4)}
        dev_notes[key] = devs

    # ---- 嵌套性验证 ----
    nested = True
    seq = None
    for key in ["10", "50", "200", "1000", "5000", "ALL"]:
        cur = set(budgets[key])
        if seq is not None and not seq.issubset(cur):
            nested = False
            print(f"  [FAIL] 嵌套性违反: {key} 不包含前序")
        seq = cur
    print("嵌套性 (10⊂50⊂200⊂1000⊂5000⊂ALL):", "OK" if nested else "FAIL")

    # ---- 保存 ----
    doc = {
        "manifest_seed": args.seed,
        "n_annotations": n_all,
        "ordered_annotations": [
            {**{k2: v2 for k2, v2 in a.items()}, "ann_id": i}
            for i, a in enumerate(ordered)],
        "budgets": budgets,
        "composition": composition,
        "deviation_gt5pp": dev_notes,
        "nested_check": nested,
    }
    outp = f"{OUT}/mask_budget_manifest.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"注记 {n_all} | 已保存: {outp}")

    # ---- Swap manifest(层内轮转,无固定点;记录偶然 IoU;swap_manifest_seed=20260832) ----
    make_swap_manifest(ordered, args.seed + 1)

    print(json.dumps({"n": n_all, "nested": nested,
                      "dev_gt5pp_keys": list(dev_notes.keys())}, ensure_ascii=False))


def make_swap_manifest(ordered, swap_seed):
    """层内轮转置换(rotate by 1 ⇒ 无固定点);排除同 canonical 互换由层内同 canonical 处理:
    层内再按 canonical 分桶,桶间轮转(桶大小 ≥2),避免同源互换。记录 swapped 与真实 mask 的偶然 IoU。"""
    import numpy as np
    import re
    from PIL import Image
    from pathlib import Path

    IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
    GRID = 37
    ordered = [dict(a, ann_id=i) for i, a in enumerate(ordered)]

    # 层 = (domain, editor)(size_class 不参与 swap 分层:同源多属性常按 size 拆层,层内单桶会导致无法交换);
    # 桶 = canonical(禁止同源互换)
    bucket_map = {}
    for a in ordered:
        layer = (a["domain"], a["editor"] or "real")
        bucket_map.setdefault(layer, {}).setdefault(a["canonical"], []).append(a["ann_id"])
    rng = np.random.RandomState(swap_seed)
    swap = {}
    iou_vals = []

    def ann_mask(a):
        mp = a["mask_path"]
        if not mp:
            return None
        p = IMG_ROOT / mp
        if not p.exists():
            return None
        m = Image.open(p).convert("L").resize((GRID, GRID), Image.NEAREST)
        return (np.asarray(m) > 127).astype(np.float32).ravel()

    by_id = {a["ann_id"]: a for a in ordered}
    for layer, buckets in bucket_map.items():
        bucket_ids = sorted(buckets.keys())
        for bi, cid in enumerate(bucket_ids):
            ids = buckets[cid]
            if len(ids) < 2:
                # 单桶:层内跨桶轮转(层内至少 2 桶才可行;否则该注记不交换)
                continue
            # 桶间轮转:每个 id 映射到下一桶的同位元素
            nxt = bucket_ids[(bi + 1) % len(bucket_ids)]
            nxt_ids = buckets[nxt]
            if len(nxt_ids) < 2:
                continue
            for j, ann_id in enumerate(ids):
                tgt = nxt_ids[j % len(nxt_ids)]
                swap[ann_id] = tgt
                try:
                    m1 = ann_mask(by_id[ann_id])
                    m2 = ann_mask(by_id[tgt])
                    if m1 is not None and m2 is not None:
                        inter = float((m1 * m2).sum())
                        union = float(((m1 + m2) > 0).sum())
                        if union > 0:
                            iou_vals.append(inter / union)
                except Exception:
                    pass
    # 无固定点校验
    fixed = [a for a, b in swap.items() if a == b]
    swap_doc = {
        "swap_manifest_seed": swap_seed,
        "n_swapped": len(swap),
        "fixed_points": fixed,
        "derangement_ok": len(fixed) == 0,
        "accidental_iou": {"n": len(iou_vals),
                           "mean": float(np.mean(iou_vals)) if iou_vals else 0.0,
                           "p95": float(np.percentile(iou_vals, 95)) if iou_vals else 0.0,
                           "max": float(np.max(iou_vals)) if iou_vals else 0.0},
        "swap": dict(sorted(swap.items(), key=lambda kv: kv[0])),
    }
    with open(f"{OUT}/mask_swap_manifest.json", "w", encoding="utf-8") as f:
        json.dump(swap_doc, f, ensure_ascii=False, indent=1)
    print(f"swap manifest: {len(swap)} 注记, 无固定点={swap_doc['derangement_ok']}, "
          f"偶然IoU p95={swap_doc['accidental_iou']['p95']:.4f} (n={len(iou_vals)})")


if __name__ == "__main__":
    main()
