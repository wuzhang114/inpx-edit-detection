"""Evaluate held-out edits with validation-selected localization thresholds.

流程:
1. 从 split JSON 取排除后的验证集 val_idx (不含评测集样本)
2. 在 val 的 fake 子集上选全局最优固定阈值 (91 档, 像素级 mIoU)
3. 在 500 编辑对上 (训练/验证从未见过) 报告:
   - 固定阈值全局 mIoU / F1 (+ size 分档)
   - pixel AUROC / AP (sklearn, 全 patch)
   - 补充口径: oracle 全局 best-thr; per-image oracle (原 fair_compare_loc 口径)
4. 检测: 800 对 (500 edited + 300 real) 图像级 AUC (max-pool 分数)

用法: python eval_seen500.py --head weak_sup_v4_fix_seed42_head.pt --split split_fix_seed42.json --tag fix_seed42
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.weakly_supervised_v2 import MLPHead

CACHE = "D:/lunwen/data/features_cache"
IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
DATA_JSON = "D:/lunwen/data/imdl_inpx_test.json"
OUT = Path("D:/lunwen/outputs")
GRID = 37


def load_cache(cache_dir=None):
    cache_dir = cache_dir or CACHE
    labels = json.load(open(f"{cache_dir}/labels.json"))
    meta = json.load(open(f"{cache_dir}/metadata.json"))
    n_total, n_patches, dino_dim = meta["n_total"], meta["n_patches"], meta["dino_dim"]
    dino_mm = np.memmap(f"{cache_dir}/dino_tokens.npy", dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, dino_dim))
    hp_mm = np.memmap(f"{cache_dir}/highpass.npy", dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))
    labels_arr = np.array([l["label"] for l in labels])
    path2idx = {l["path"]: i for i, l in enumerate(labels)}
    return labels, meta, dino_mm, hp_mm, labels_arr, path2idx


def predict_scores(head, idxs, dino_mm, hp_mm, use_hp=True):
    scores = []
    with torch.no_grad():
        for idx in idxs:
            dino = torch.from_numpy(dino_mm[idx].astype(np.float32))
            X = [dino]
            if use_hp:
                X.append(torch.from_numpy(hp_mm[idx].astype(np.float32)))
            X = torch.cat(X, dim=-1).unsqueeze(0).cuda()
            logits = head(X)
            if logits.dim() == 3:
                logits = logits.squeeze(-1)  # linear probe 输出 (b,1369,1)
            logits = logits.squeeze(0)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.stack(scores)  # (n, 1369)


def gt_masks(idxs, path2idx, labels):
    """返回 (n, 1369) GT mask; 无 mask 的样本全 0"""
    out = np.zeros((len(idxs), GRID * GRID), dtype=np.float32)
    for k, idx in enumerate(idxs):
        mp = labels[idx].get("mask_path")
        if mp and (IMG_ROOT / mp).exists():
            m = Image.open(IMG_ROOT / mp).convert("L").resize((GRID, GRID), Image.NEAREST)
            out[k] = (np.asarray(m) > 127).astype(np.float32).ravel()
    return out


def pixel_miou(scores, masks, thr):
    pred = (scores >= thr).astype(np.float32)
    tp = (pred * masks).sum(); fp = (pred * (1 - masks)).sum(); fn = ((1 - pred) * masks).sum()
    return tp / (tp + fp + fn + 1e-8), tp, fp, fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", required=True, help="head 文件名 (在 outputs/ 下)")
    p.add_argument("--split", required=True, help="split JSON (split_<tag>.json)")
    p.add_argument("--tag", required=True, help="输出 tag → outputs/eval500_<tag>.json")
    p.add_argument("--no_hp", action="store_true", help="纯 DINO (384 维输入) 头, 不加高通残差")
    p.add_argument("--group_json", type=str, default=None,
                   help="可选: source-overlap 分组 JSON(默认不分组; 旧 source_overlap.json 基于旧 split, 勿用于 master)")
    p.add_argument("--linear", action="store_true", help="linear probe 头(Linear(in_dim,1), 无 BN)")
    p.add_argument("--hidden", type=int, default=64, help="MLPHead hidden 维度(如 4K 头为 8)")
    p.add_argument("--cache", type=str, default=CACHE, help="特征缓存目录(跨 backbone 时指向其他缓存)")
    args = p.parse_args()

    labels, meta, dino_mm, hp_mm, labels_arr, path2idx = load_cache(args.cache)
    dino_dim = meta["dino_dim"]
    split = json.load(open(OUT / args.split))
    val_idx = np.array(split["val_idx"])

    in_dim = dino_dim if args.no_hp else dino_dim + 1
    if args.linear:
        head = torch.nn.Linear(in_dim, 1).cuda()
    else:
        head = MLPHead(in_dim, hidden=args.hidden).cuda()
    head.load_state_dict(torch.load(OUT / args.head))
    head.eval()

    # ---- 验证集选阈值 ----
    val_fake = val_idx[labels_arr[val_idx] == 1]
    sv = predict_scores(head, val_fake, dino_mm, hp_mm, use_hp=not args.no_hp)
    mv = gt_masks(val_fake, path2idx, labels)
    best_thr, best_vm = 0.5, 0.0
    for th in np.linspace(0.05, 0.95, 91):
        iou, *_ = pixel_miou(sv, mv, th)
        if iou > best_vm:
            best_vm, best_thr = iou, th
    print(f"验证集 (n={len(val_fake)}) 选阈值: thr={best_thr:.3f} mIoU={best_vm:.4f}")

    # ---- 500 编辑对 (完全未见) ----
    records = json.load(open(DATA_JSON))
    eval_idx, real_idx = [], []
    for img_path, mask_path in records:
        rel = str(Path(img_path).relative_to(IMG_ROOT))
        idx = path2idx.get(rel)
        if idx is None:
            continue
        (real_idx if mask_path == "Negative" else eval_idx).append(idx)
    eval_idx, real_idx = np.array(eval_idx), np.array(real_idx)
    print(f"未见测试: 编辑图 {len(eval_idx)}, real {len(real_idx)}")

    # ---- source-overlap 分组(可选; 默认不做, 主协议以 source-cluster 敏感指标为准) ----
    group = None  # None=未知, 'clean'=源图完全未见, 'overlap'=同源图其他编辑在训练候选
    if args.group_json is not None:
        so = json.load(open(args.group_json))
        so_idx = list(so["eval_edited_idx"])
        if so_idx == list(eval_idx):
            group = ["clean" if (not a and not b) else "overlap"
                     for a, b in zip(so["same_key_overlap"], so["src_overlap"])]
            n_clean = group.count("clean")
            print(f"source-overlap 分组: clean={n_clean}, overlap={len(group) - n_clean} "
                  f"(edit-disjoint 已达成; 分组仅反映图级残余)")

    se = predict_scores(head, eval_idx, dino_mm, hp_mm, use_hp=not args.no_hp)
    me = gt_masks(eval_idx, path2idx, labels)

    # 固定阈值 (验证集选定的阈值)
    fixed_iou, tp, fp, fn = pixel_miou(se, me, best_thr)
    prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
    fixed_f1 = 2 * prec * rec / (prec + rec + 1e-8)
    print(f"[固定阈值 thr={best_thr:.3f}] mIoU={fixed_iou:.4f} F1={fixed_f1:.4f}")

    # pixel AUROC / AP (编辑图所有 patch vs GT)
    fake_pix = me.ravel()
    auroc = roc_auc_score(fake_pix, se.ravel())
    ap = average_precision_score(fake_pix, se.ravel())
    print(f"[pixel] AUROC={auroc:.4f} AP={ap:.4f}")

    # 补充: oracle 全局 best-thr (测试集上选)
    best_o, best_of = 0.0, 0.0
    for th in np.linspace(0.05, 0.95, 91):
        iou, tp, fp, fn = pixel_miou(se, me, th)
        p_ = tp / (tp + fp + 1e-8); r_ = tp / (tp + fn + 1e-8)
        best_o = max(best_o, iou)
        best_of = max(best_of, 2 * p_ * r_ / (p_ + r_ + 1e-8))
    print(f"[oracle 全局] mIoU={best_o:.4f} F1={best_of:.4f}")

    # 补充: per-image oracle (原 fair_compare_loc 口径)
    per_img = []
    for k in range(len(eval_idx)):
        bi = 0.0
        for th in np.linspace(0.05, 0.95, 91):
            p = (se[k] >= th).astype(np.float32); m = me[k]
            tp = (p * m).sum(); fp = (p * (1 - m)).sum(); fn = ((1 - p) * m).sum()
            bi = max(bi, tp / (tp + fp + fn + 1e-8))
        per_img.append(bi)
    per_img_miou = float(np.mean(per_img))
    print(f"[per-image oracle] mIoU={per_img_miou:.4f}")

    # 逐图固定阈值 mIoU (bootstrap CI 用)
    per_img_fixed = []
    for k in range(len(eval_idx)):
        p = (se[k] >= best_thr).astype(np.float32); m = me[k]
        tp = (p * m).sum(); fp = (p * (1 - m)).sum(); fn = ((1 - p) * m).sum()
        per_img_fixed.append(tp / (tp + fp + fn + 1e-8))

    # size 分档 (固定阈值)
    sizes = {}
    for k in range(len(eval_idx)):
        ratio = me[k].mean()
        sc = "tiny" if ratio < 0.02 else ("small" if ratio < 0.05
              else ("medium" if ratio < 0.15 else "large"))
        sizes.setdefault(sc, []).append(k)
    size_tbl = {}
    for sc, ks in sizes.items():
        if len(ks) < 5:
            continue
        iou, *_ = pixel_miou(se[ks], me[ks], best_thr)
        size_tbl[sc] = {"n": len(ks), "mIoU": float(iou)}
    print(f"size 分档: {size_tbl}")

    # ---- 图级残余分组分析 (clean 60 vs overlap 440; confidence intervals reported separately) ----
    group_tbl = {}
    if group is not None:
        from collections import Counter as _C
        grp_arr = np.array(group)
        print(f"\n[source-overlap 分组, fixed thr={best_thr:.3f}]")
        for g in ["clean", "overlap"]:
            ks = np.where(grp_arr == g)[0]
            if len(ks) < 5:
                continue
            sub = [per_img_fixed[k] for k in ks]
            m_s = float(np.mean(sub))
            # 子组 bootstrap CI (逐图)
            lo_s, hi_s = 0.0, 0.0
            if len(sub) >= 10:
                rng = np.random.RandomState(0)
                bs = np.array([np.mean(rng.choice(sub, len(sub), replace=True))
                               for _ in range(2000)])
                lo_s, hi_s = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            # domain / inpainter 分层
            cats = [labels[eval_idx[k]].get("cat") for k in ks]
            models = [labels[eval_idx[k]].get("model") for k in ks]
            dom_tbl = dict(_C(cats))
            mod_tbl = dict(_C(models))
            print(f"  {g:7s} n={len(ks):3d}  mIoU={m_s:.4f}  95%CI=[{lo_s:.4f},{hi_s:.4f}]"
                  f"  域={dom_tbl}  模型={mod_tbl}")
            group_tbl[g] = {"n": int(len(ks)), "mIoU": float(m_s),
                            "ci95": [lo_s, hi_s], "domains": dom_tbl, "models": mod_tbl}
        if "clean" in group_tbl and "overlap" in group_tbl:
            delta = group_tbl["clean"]["mIoU"] - group_tbl["overlap"]["mIoU"]
            group_tbl["delta_clean_minus_overlap"] = float(delta)
            print(f"  Delta (clean - overlap) = {delta:+.4f}")

    # 检测: 800 对图像级 AUC (max-pool)
    sr = predict_scores(head, real_idx, dino_mm, hp_mm, use_hp=not args.no_hp)
    img_scores = np.concatenate([se.max(1), sr.max(1)])
    y_all = np.concatenate([np.ones(len(eval_idx)), np.zeros(len(real_idx))])
    det_auc = roc_auc_score(y_all, img_scores)
    print(f"[检测 800 对] imgAUC={det_auc:.4f}")

    out = {
        "head": args.head, "split": args.split, "seed": split.get("seed"),
        "val_thr": float(best_thr), "val_miou": float(best_vm),
        "n_edited": int(len(eval_idx)), "n_real": int(len(real_idx)),
        "fixed_miou": float(fixed_iou), "fixed_f1": float(fixed_f1),
        "pixel_auroc": float(auroc), "pixel_ap": float(ap),
        "oracle_global_miou": float(best_o), "oracle_global_f1": float(best_of),
        "per_image_oracle_miou": per_img_miou,
        "per_image_fixed_miou": [round(float(v), 6) for v in per_img_fixed],
        "det_auc_800": float(det_auc),
        "size_fixed_miou": size_tbl,
        "source_overlap_groups": group_tbl,
    }
    with open(OUT / f"eval500_{args.tag}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"结果: {OUT / f'eval500_{args.tag}.json'}")


if __name__ == "__main__":
    main()
