"""因果隔离实验 v6: v4 单头结构 + 绑定损失 (L_exc / L_placebo), 其余完全不变。

This exploratory variant tests binding losses with a single detection head.
- 保留 v4 原版: 单头 MLPHead(385→64), max(local map) detection, 20ep/128batch/
  wd=1e-4, master split, val image-AUC(max-pool) checkpoint —— 与 g1_s42 完全同规
- 只加两个损失 (可分别开关):
  L_exc    = (1/|M_core|) Σ_{p∈M_core} Huber(z^std_p − z^exc_p), δ=1.0
             (M_core = 腐蚀 1 的核心区; 边缘不强求)
  L_placebo= [m − min(E_M(z^std), E_M(z^exc)) + max(E_M−(z^std), E_M−(z^exc))]_+
             M^- = 等面积合法平移 (bbox 平移后完全在图内, 无回卷)
- 与 v5 的区别: detection 仍是 max-pool local map; 不加 global 分支
- 变体: exc(只 L_exc) / expl(L_exc+L_placebo)

用法: python weakly_supervised_v6.py --tag v6_exc_bALL_s42 --variant exc --mask_budget ALL ...
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

CACHE = "D:/lunwen/data/features_cache"
IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
OUT = Path("D:/lunwen/outputs")
GRID = 37
N_PATCHES = 1369

KEY_RE = re.compile(
    r"^(.*?)_(?:CelebAHQ|CityScapes|SUN_RGBD|OpenImages)_"
    r"(?:Kandinsky_2_2|StableDiffusion_v4|OpenJourney)(?:_simple)?\.jpg$")


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        b, n, c = x.shape
        return self.net(x.reshape(b * n, c)).reshape(b, n)


def load_mask(mask_path: str) -> np.ndarray:
    img = Image.open(IMG_ROOT / mask_path).convert("L")
    img = img.resize((GRID, GRID), Image.NEAREST)
    return (np.asarray(img) > 127).astype(np.float32)


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1) + eps
    return 1 - (2 * inter / denom).mean()


def erode_mask(mask, size=1):
    m = (mask.reshape(GRID, GRID) > 0.5).astype(np.uint8)
    e = m.copy()
    for dx in range(-size, size + 1):
        for dy in range(-size, size + 1):
            e = e & np.roll(np.roll(m, dx, 1), dy, 0)
    e = e.astype(np.float32)
    if e.sum() == 0:
        return mask.ravel()
    return e.ravel()


def make_placebo_valid(mask, rng, max_try=30, iou_thr=0.35):
    """等面积合法平移 (bbox 平移后完全在图内, 无回卷); IoU < iou_thr 接受。

    局部 bbox 平移: 从 mask 的 bbox 出发, 采样 (dy,dx) 使 bbox 完全留在图内,
    其余区域置 0 → 形状与面积 (bbox 内) 完全一致, 无 np.roll 回卷伪影。
    找不到 (bbox 接近全图) 则取 IoU 最小的可行平移 (训练循环由 ok_pl 排除)。
    """
    m = mask.reshape(GRID, GRID)
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        return m.ravel()
    y0, x0 = ys.min(), xs.min()
    h, w = ys.max() - y0 + 1, xs.max() - x0 + 1
    dy_hi = GRID - (y0 + h)
    dx_hi = GRID - (x0 + w)
    if dy_hi + dx_hi < 2:
        return m.ravel()  # bbox 几乎占满, 无可行平移
    best, best_iou = m.copy(), 1.0
    for _ in range(max_try):
        dy = int(rng.randint(-y0, dy_hi + 1))
        dx = int(rng.randint(-x0, dx_hi + 1))
        if dy == 0 and dx == 0:
            continue
        sh = np.zeros_like(m)
        sh[y0 + dy:y0 + dy + h, x0 + dx:x0 + dx + w] = m[y0:y0 + h, x0:x0 + w]
        inter = float((sh * m).sum())
        iou = inter / max(m.sum() + sh.sum() - inter, 1e-6)
        if iou < best_iou:
            best_iou, best = iou, sh
        if best_iou < iou_thr:
            break
    return best.ravel()


def train_v6(tag, epochs=20, batch_size=128, lr=1e-3, seed=42, mask_budget="ALL",
             variant="exc", lam_cons=0.25, lam_pl=0.25, pl_margin=0.5,
             hidden=64, master_split=None, mask_manifest=None, wd=1e-4,
             cache=None, use_hp=True):
    cache = cache or CACHE
    labels = json.load(open(f"{cache}/labels.json", encoding="utf-8"))
    meta = json.load(open(f"{cache}/metadata.json", encoding="utf-8"))
    n_total, n_patches, dino_dim = meta["n_total"], meta["n_patches"], meta["dino_dim"]
    dino_mm = np.memmap(f"{cache}/dino_tokens.npy", dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, dino_dim))
    hp_mm = np.memmap(f"{cache}/highpass.npy", dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))

    split = json.load(open(master_split))
    train_idx = np.array(split["train_idx"], dtype=np.int64)
    val_idx = np.array(split["val_idx"], dtype=np.int64)
    print(f"[v6 {variant}] master split: train={len(train_idx)} val={len(val_idx)} "
          f"seed={seed} budget={mask_budget}", flush=True)
    labels_arr = np.array([l["label"] for l in labels])

    # ---- mask 全表 (v4 同构) ----
    masks = np.zeros((n_total, n_patches), dtype=np.float32)
    for l in labels:
        mp = l.get("mask_path")
        if mp and (IMG_ROOT / mp).exists():
            masks[l["idx"]] = load_mask(mp).ravel()

    # ---- 预算注记 (v4 同构) ----
    man = json.load(open(mask_manifest))
    all_anns = man["ordered_annotations"]
    k = len(all_anns) if str(mask_budget).upper() == "ALL" else int(mask_budget)
    k = min(k, len(all_anns))
    anns = all_anns[:k]
    ann_rep = np.array([a["rep_idx"] for a in anns], dtype=np.int64)
    ann_masks = np.zeros((len(anns), n_patches), dtype=np.float32)
    for j, a in enumerate(anns):
        idx = a["rep_idx"]
        mp = labels[idx].get("mask_path")
        if mp and (IMG_ROOT / mp).exists():
            ann_masks[j] = load_mask(mp).ravel()
    ann_pos = np.array([a["ann_id"] for a in anns], dtype=np.int64)

    # ---- std/exc 配对 ----
    exc_of = {}
    for l in labels:
        p = l["path"].replace("\\", "/")
        base = os.path.basename(p).replace("_simple.jpg", ".jpg")
        if l["kind"] == "exchanged":
            exc_of.setdefault(base, l["idx"])
    ann_exc = np.full(len(anns), -1, dtype=np.int64)
    for j, a in enumerate(anns):
        idx = a["rep_idx"]
        base = os.path.basename(labels[idx]["path"].replace("\\", "/")).replace("_simple.jpg", ".jpg")
        if base in exc_of and exc_of[base] != idx:
            ann_exc[j] = exc_of[base]
    n_pairable = int((ann_exc >= 0).sum())
    print(f"[pairs] 预算内注记有 exc 配对: {n_pairable}/{len(anns)}", flush=True)

    # ---- 预计算: 腐蚀核心区 + 4 个合法平移 placebo ----
    t_pre = time.time()
    ann_erode = np.zeros((len(anns), n_patches), dtype=np.float32)
    ann_placebo = np.zeros((len(anns), 4, n_patches), dtype=np.float32)
    rng_pre = np.random.RandomState(seed + 1000)
    n_valid_pbo = 0
    for jj in range(len(anns)):
        m = ann_masks[jj]
        if m.sum() == 0:
            continue
        ann_erode[jj] = erode_mask(m)
        for c in range(4):
            pb = make_placebo_valid(m, rng_pre)
            if pb.sum() > 0 and not np.array_equal(pb, m):
                ann_placebo[jj, c] = pb
                n_valid_pbo += 1
    print(f"[precalc] erode+placebo done in {(time.time()-t_pre):.1f}s "
          f"(placebo 有效 {n_valid_pbo}/{len(anns)*4})", flush=True)

    def feat_of(idxs):
        dino = torch.from_numpy(dino_mm[idxs])
        parts = [dino]
        if use_hp:
            parts.append(torch.from_numpy(hp_mm[idxs]))
        return torch.cat(parts, dim=-1).float().cuda()

    model = MLPHead(dino_dim + (1 if use_hp else 0), hidden=hidden).cuda()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,} "
          f"(hidden={hidden}, 与 v4 g1 一致)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    best_val = -1.0
    best_state = None
    t0 = time.time()
    n_train = len(train_idx)
    U_img = int(np.ceil(n_train / batch_size))
    U_pix = U_img if k > 0 else 0
    print(f"[loader] U_img={U_img} U_pix={U_pix} 像素曝光={U_pix*batch_size}/epoch", flush=True)
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_train)
        img_l, pix_l, cons_l, pl_l = [], [], [], []
        step_i = 0
        for b0 in range(0, n_train, batch_size):
            bi = train_idx[perm[b0:b0 + batch_size]]
            j = rng.choice(k, size=len(bi), replace=True)
            pidx = ann_rep[j]
            feat_bi = feat_of(bi)
            feat_pix = feat_of(pidx)
            valid = ann_exc[j] >= 0
            exc_idx = ann_exc[j][valid]
            if len(exc_idx):
                feat_exc = feat_of(exc_idx)
                X = torch.cat([feat_bi, feat_pix, feat_exc], 0)
            else:
                feat_exc = None
                X = torch.cat([feat_bi, feat_pix], 0)
            logits = model(X)
            if logits.dim() == 3:
                logits = logits.squeeze(-1)
            # ---- 与 v4 完全一致的基础损失: 图像批+像素批 max-pool 图像级 BCE ----
            y_img = torch.cat([
                torch.from_numpy(labels_arr[bi]).float().cuda(),
                torch.ones(len(bi), device="cuda"),   # pidx 全为编辑图
                torch.ones(len(exc_idx), device="cuda") if len(exc_idx) else
                torch.zeros(0, device="cuda"),
            ])
            img_score = torch.sigmoid(logits).max(dim=1).values
            bce_img = F.binary_cross_entropy(img_score, y_img)
            pix_logits = logits[len(bi):len(bi) + len(bi)]
            tgt = torch.from_numpy(ann_masks[j]).cuda()
            bce_pix = F.binary_cross_entropy_with_logits(pix_logits, tgt)
            dice = dice_loss(pix_logits, tgt)
            loss = bce_img + bce_pix + 0.5 * dice
            # ---- L_exc (单头; 编辑区腐蚀核心一致性) ----
            if variant in ("exc", "expl") and feat_exc is not None:
                m_cons = torch.from_numpy(ann_erode[j][valid]).cuda()
                diff = F.huber_loss(pix_logits[valid], logits[len(bi) + len(bi):],
                                    delta=1.0, reduction="none")
                l_cons = (diff * m_cons).sum() / m_cons.sum().clamp(min=1)
                loss = loss + lam_cons * l_cons
                cons_l.append(float(l_cons.item()))
            # ---- L_placebo (合法平移区 hinge) ----
            if variant == "expl" and feat_exc is not None:
                ms = torch.from_numpy(ann_masks[j][valid]).cuda()
                pbo = torch.from_numpy(ann_placebo[j[valid], step_i % 4]).cuda()
                exc_logits = logits[len(bi) + len(bi):]
                em_s = (pix_logits[valid] * ms).sum(1) / ms.sum(1).clamp(min=1)
                em_e = (exc_logits * ms).sum(1) / ms.sum(1).clamp(min=1)
                ep_s = (pix_logits[valid] * pbo).sum(1) / pbo.sum(1).clamp(min=1)
                ep_e = (exc_logits * pbo).sum(1) / pbo.sum(1).clamp(min=1)
                h = F.relu(pl_margin - torch.minimum(em_s, em_e) + torch.maximum(ep_s, ep_e))
                ok_pl = (ms.sum(1) > 0) & (pbo.sum(1) > 0)
                if ok_pl.any():
                    l_pl = h[ok_pl].mean()
                    loss = loss + lam_pl * l_pl
                    pl_l.append(float(l_pl.item()))
            step_i += 1
            opt.zero_grad()
            loss.backward()
            opt.step()
            del X, feat_bi, feat_pix
            img_l.append(float(bce_img.item()))
            pix_l.append(float((bce_pix + 0.5 * dice).item()))
        sched.step()

        # ---- val: max-pool image AUC (与 v4 完全一致) ----
        model.eval()
        sv = []
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                b = val_idx[i:i + batch_size]
                lo = model(feat_of(b)).squeeze(-1)
                sv.append(torch.sigmoid(lo).max(dim=1).values.cpu().numpy())
        sv = np.concatenate(sv)
        img_auc = roc_auc_score(labels_arr[val_idx], sv)
        mean_img = float(np.mean(img_l))
        mean_pix = float(np.mean(pix_l))
        mean_cons = float(np.mean(cons_l)) if cons_l else 0.0
        mean_pl = float(np.mean(pl_l)) if pl_l else 0.0
        print(f"ep{epoch+1}: img={mean_img:.4f} pix={mean_pix:.4f} cons={mean_cons:.4f} "
              f"pl={mean_pl:.4f} val_imgAUC={img_auc:.4f} "
              f"({(time.time()-t0)/60:.1f}min) peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB",
              flush=True)
        if img_auc > best_val:
            best_val = img_auc
            best_state = {kk: vv.clone() for kk, vv in model.state_dict().items()}

    print(f"\n训练完成 {(time.time()-t0)/60:.1f} 分钟; best val imgAUC={best_val:.4f}", flush=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT / f"weak_sup_v6_{tag}_head.pt")
    side = {
        "tag": tag, "variant": variant, "seed": int(seed), "budget": str(mask_budget),
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "wd": wd,
        "lam_cons": float(lam_cons), "lam_pl": float(lam_pl), "pl_margin": float(pl_margin),
        "use_hp": bool(use_hp), "hidden": int(hidden), "n_pairable_anns": int(n_pairable),
        "head_config": {"in_dim": dino_dim + (1 if use_hp else 0), "hidden": int(hidden)},
        "val_img_auc": float(best_val),
        "protocol": "A_20_128 master split dual-loader, checkpoint=val max-pool AUC (v4 同规)",
        "detection": "max(local map) (v4 同规, 无 global 分支)",
    }
    with open(OUT / f"weak_sup_v6_{tag}_meta.json", "w", encoding="utf-8") as f:
        json.dump(side, f, indent=1)
    print(f"模型: {OUT / f'weak_sup_v6_{tag}_head.pt'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_budget", default="ALL")
    p.add_argument("--variant", default="exc", choices=["exc", "expl"])
    p.add_argument("--lam_cons", type=float, default=0.25)
    p.add_argument("--lam_pl", type=float, default=0.25)
    p.add_argument("--pl_margin", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--master_split", default=str(OUT / "split_budget_master.json"))
    p.add_argument("--mask_manifest", default=str(OUT / "mask_budget_manifest.json"))
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--no_hp", action="store_true")
    p.add_argument("--cache", type=str, default=None)
    args = p.parse_args()
    train_v6(tag=args.tag, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
             seed=args.seed, mask_budget=args.mask_budget, variant=args.variant,
             lam_cons=args.lam_cons, lam_pl=args.lam_pl, pl_margin=args.pl_margin,
             hidden=args.hidden, master_split=args.master_split,
             mask_manifest=args.mask_manifest, wd=args.wd, cache=args.cache,
             use_hp=not args.no_hp)
