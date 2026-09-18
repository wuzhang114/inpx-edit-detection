"""v5 双分支头评测: 标准 500 + exchange 498 + 检测(global 分支) + true-vs-placebo。

与 eval_seen500/eval_exchange_500 同协议, 但:
- loc  : local 分支输出 (37x37 分数图), 阈值在 val 图 91 档固定 (local 图)
- det  : global 分支输出 (图像级标量), 不再取 local max-pool
- 输出: outputs/eval500_v5_{tag}.json (含 standard / exchange / paired / placebo 四节)
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
from baselines.weakly_supervised_v5 import (  # noqa: E402
    DualBranchHead, load_mask, make_placebo)

CACHE = "D:/lunwen/data/features_cache"
IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
DATA_JSON = "D:/lunwen/data/imdl_inpx_test.json"
PAIR_MANIFEST = "D:/lunwen/outputs/placebo_pair_manifest.json"
OUT = Path("D:/lunwen/outputs")
GRID = 37
N_PATCHES = GRID * GRID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="v5 tag (weak_sup_v5_<tag>_head.pt)")
    ap.add_argument("--split", default="split_budget_master.json")
    args = ap.parse_args()

    meta = json.load(open(OUT / f"weak_sup_v5_{args.tag}_meta.json", encoding="utf-8"))
    cfg = meta["head_config"]
    labels = json.load(open(f"{CACHE}/labels.json", encoding="utf-8"))
    meta2 = json.load(open(f"{CACHE}/metadata.json", encoding="utf-8"))
    n_total, n_patches, dino_dim = meta2["n_total"], meta2["n_patches"], meta2["dino_dim"]
    dino_mm = np.memmap(f"{CACHE}/dino_tokens.npy", dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, dino_dim))
    hp_mm = np.memmap(f"{CACHE}/highpass.npy", dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))
    in_dim = dino_dim + (1 if meta.get("use_hp", True) else 0)
    head = DualBranchHead(in_dim, local_hidden=cfg["local_hidden"],
                          global_hidden=cfg["global_hidden"]).cuda()
    head.load_state_dict(torch.load(OUT / f"weak_sup_v5_{args.tag}_head.pt", map_location="cpu"))
    head.eval()
    labels_arr = np.array([l["label"] for l in labels])
    path2idx = {l["path"].replace("\\", "/"): i for i, l in enumerate(labels)}

    label_by_base = {}
    for l in labels:
        base = os.path.basename(l["path"].replace("\\", "/")).replace("_simple.jpg", ".jpg")
        label_by_base.setdefault(base, []).append(l["idx"])

    def predict(idxs):
        gs, ls = [], []
        with torch.no_grad():
            for i in range(0, len(idxs), 128):
                b = idxs[i:i + 128]
                dino = torch.from_numpy(dino_mm[b])
                X = [dino]
                if meta.get("use_hp", True):
                    X.append(torch.from_numpy(hp_mm[b]))
                X = torch.cat(X, -1).cuda().float()
                out = head(X)
                gs.append(torch.sigmoid(out["global"]).cpu().numpy())
                ls.append(torch.sigmoid(out["local"]).cpu().numpy())
        return np.concatenate(gs), np.concatenate(ls)

    def gt(idx):
        mp = labels[idx].get("mask_path")
        if mp and (IMG_ROOT / mp).exists():
            return load_mask(mp).ravel()
        return np.zeros(n_patches, np.float32)

    # ---- 1. val 集固定阈值 (local 图, 91 档) ----
    split = json.load(open(OUT / args.split))
    val_idx = np.array(split["val_idx"])
    val_fake = val_idx[labels_arr[val_idx] == 1]
    _, sv = predict(val_fake)
    mv = np.stack([gt(i) for i in val_fake])
    best_thr, best_vm = 0.5, 0.0
    for th in np.linspace(0.05, 0.95, 91):
        pred = (sv >= th).astype(np.float32)
        tp = (pred * mv).sum(); fp = (pred * (1 - mv)).sum(); fn = ((1 - pred) * mv).sum()
        iou = tp / (tp + fp + fn + 1e-8)
        if iou > best_vm:
            best_vm, best_thr = iou, th
    print(f"val 阈值: thr={best_thr:.3f} mIoU={best_vm:.4f}", flush=True)

    # ---- 2. 标准 500 + 检测 800 ----
    recs = json.load(open(DATA_JSON, encoding="utf-8"))
    std_idx, real_idx = [], []
    for img_path, mask_path in recs:
        rel = str(Path(img_path).relative_to(IMG_ROOT)).replace("\\", "/")
        idx = path2idx.get(rel)
        if idx is None:
            continue
        (real_idx if mask_path == "Negative" else std_idx).append(idx)
    std_idx, real_idx = np.array(std_idx), np.array(real_idx)
    g_s, s_s = predict(std_idx)
    g_r, s_r = predict(real_idx)
    m_s = np.stack([gt(i) for i in std_idx])
    pred = (s_s >= best_thr).astype(np.float32)
    tp = (pred * m_s).sum(); fp = (pred * (1 - m_s)).sum(); fn = ((1 - pred) * m_s).sum()
    fixed_iou = tp / (tp + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
    fixed_f1 = 2 * prec * rec / (prec + rec + 1e-8)
    per_img_std = []
    for k in range(len(std_idx)):
        p = (s_s[k] >= best_thr).astype(np.float32); m = m_s[k]
        tpi = (p * m).sum(); fpi = (p * (1 - m)).sum(); fni = ((1 - p) * m).sum()
        per_img_std.append(tpi / (tpi + fpi + fni + 1e-8))
    det_auc = roc_auc_score(np.concatenate([np.ones(len(std_idx)), np.zeros(len(real_idx))]),
                            np.concatenate([g_s, g_r]))
    px_auroc = roc_auc_score(m_s.ravel(), s_s.ravel())
    px_ap = average_precision_score(m_s.ravel(), s_s.ravel())
    print(f"standard: fixed_mIoU={fixed_iou:.4f} F1={fixed_f1:.4f} det_auc={det_auc:.4f} "
          f"AUROC={px_auroc:.4f}", flush=True)

    # ---- 3. exchange 498 ----
    std_ex_pairs = []
    for k in range(len(std_idx)):
        base = os.path.basename(labels[std_idx[k]]["path"].replace("\\", "/"))
        base_no = base[:-4]
        cand = label_by_base.get(base_no + ".jpg", [])
        ex_i = [i for i in cand if labels[i].get("kind") == "exchanged"
                and "inpainting_exchange" in labels[i]["path"].replace("\\", "/")]
        if ex_i:
            std_ex_pairs.append((int(std_idx[k]), ex_i[0]))
    ex_std = np.array([p[0] for p in std_ex_pairs])
    ex_idx = np.array([p[1] for p in std_ex_pairs])
    g_e, s_e = predict(ex_idx)
    m_e = np.stack([gt(i) for i in ex_idx])
    pred = (s_e >= best_thr).astype(np.float32)
    tp = (pred * m_e).sum(); fp = (pred * (1 - m_e)).sum(); fn = ((1 - pred) * m_e).sum()
    ex_fixed_iou = tp / (tp + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
    ex_fixed_f1 = 2 * prec * rec / (prec + rec + 1e-8)
    per_img_ex = []
    for k in range(len(ex_idx)):
        p = (s_e[k] >= best_thr).astype(np.float32); m = m_e[k]
        tpi = (p * m).sum(); fpi = (p * (1 - m)).sum(); fni = ((1 - p) * m).sum()
        per_img_ex.append(tpi / (tpi + fpi + fni + 1e-8))
    det_ex = roc_auc_score(np.concatenate([np.ones(len(ex_idx)), np.zeros(len(real_idx))]),
                           np.concatenate([g_e, g_r]))
    # matched det (444 对): 交换版 vs 同源 source, global 分支
    mat = {"n": 0, "auc": None, "ex_mu": None, "src_mu": None}
    if Path(PAIR_MANIFEST).exists():
        man = json.load(open(PAIR_MANIFEST, encoding="utf-8"))
        pairs = man.get("pairs") or man.get("manifest") or (man if isinstance(man, list) else [])
        ex_scores, src_scores = [], []
        for rec in pairs:
            ep = rec.get("edited_path"); sp = rec.get("source_path")
            if not ep or not sp:
                continue
            ep = ep.replace("\\", "/")
            base = os.path.basename(ep)
            base_no = base[:-4].replace("_simple", "")
            cand = label_by_base.get(base_no + ".jpg", [])
            eidx = next((i for i in cand if labels[i].get("kind") == "exchanged"
                         and "inpainting_exchange" in labels[i]["path"].replace("\\", "/")), None)
            try:
                sidx = path2idx.get(str(Path(sp).relative_to(IMG_ROOT)).replace("\\", "/"))
            except Exception:
                sidx = None
            if eidx is None or sidx is None:
                continue
            gau, _ = predict(np.array([eidx]))
            gaub, _ = predict(np.array([sidx]))
            ex_scores.append(float(gau[0]))
            src_scores.append(float(gaub[0]))
        if len(ex_scores) >= 50:
            mat = {"n": len(ex_scores),
                   "auc": float(roc_auc_score(np.concatenate([np.ones(len(ex_scores)),
                                                              np.zeros(len(src_scores))]),
                                              np.concatenate([ex_scores, src_scores]))),
                   "ex_mu": float(np.mean(ex_scores)), "src_mu": float(np.mean(src_scores))}
    print(f"exchange: fixed_mIoU={ex_fixed_iou:.4f} det={det_ex:.4f} matched={mat}", flush=True)

    # ---- 4. 配对指标: 逐对 std-exc + 编辑区一致性 + true-vs-placebo ----
    # std_ex_pairs 按 std_idx 顺序; per_img_std 同序 → 直接索引
    pair_diffs, cons_in, pl_gap = [], [], []
    rng = np.random.RandomState(7)
    idx_of_std = {int(s): k for k, s in enumerate(std_idx)}
    for t, (si, ei) in enumerate(std_ex_pairs):
        m = m_e[t]
        if m.sum() == 0:
            continue
        gs1, s1 = predict(np.array([si]))
        gs2, s2 = predict(np.array([ei]))
        cons_in.append(float(np.abs(s1[0] - s2[0])[m > 0].mean()))
        pair_diffs.append(per_img_std[idx_of_std[si]] - per_img_ex[t])
        pbo = make_placebo(m.astype(np.float32), rng)
        if pbo.sum() > 0:
            z1 = s1[0].astype(np.float32); z2 = s2[0].astype(np.float32)
            em1 = float(z1[m > 0].mean()); em2 = float(z2[m > 0].mean())
            ep1 = float(z1[pbo > 0].mean()); ep2 = float(z2[pbo > 0].mean())
            pl_gap.append(float(min(em1, em2) - max(ep1, ep2)))
    res = {
        "tag": args.tag, "variant": meta["variant"], "budget": meta["budget"],
        "val_thr": float(best_thr), "val_miou": float(best_vm),
        "standard": {
            "n": int(len(std_idx)), "fixed_miou": float(fixed_iou),
            "fixed_f1": float(fixed_f1), "det_auc_800": float(det_auc),
            "pixel_auroc": float(px_auroc), "pixel_ap": float(px_ap),
            "per_image_fixed_miou": [round(float(v), 6) for v in per_img_std],
        },
        "exchange": {
            "n": int(len(ex_idx)), "fixed_miou": float(ex_fixed_iou),
            "fixed_f1": float(ex_fixed_f1), "det_auc": float(det_ex),
            "per_image_fixed_miou": [round(float(v), 6) for v in per_img_ex],
            "matched": mat,
        },
        "paired": {
            "n": int(len(pair_diffs)),
            "std_minus_ex_miou_mean": float(np.mean(pair_diffs)),
            "edited_region_consistency_mean": float(np.mean(cons_in)),
        },
        "placebo_gap": {"n": int(len(pl_gap)), "mean": float(np.mean(pl_gap))},
    }
    with open(OUT / f"eval500_v5_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"结果: {OUT / f'eval500_v5_{args.tag}.json'}", flush=True)
    print(f"paired: std-exc mIoU 差均值={np.mean(pair_diffs):+.4f} "
          f"编辑区一致性={np.mean(cons_in):.4f} placebo_gap={np.mean(pl_gap):.4f}", flush=True)


if __name__ == "__main__":
    main()
