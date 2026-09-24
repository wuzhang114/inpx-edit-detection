"""弱监督 v5: 双分支 (global/local) + Exchange-Paired Spatial Binding

Exploratory dual-branch architecture:
- local branch : 输出定位图, 使用 pixel BCE+Dice + L_cons + L_placebo
- global branch: 独立 head, 输出图像级检测分, 单独监督图像级 BCE
                  推理时 detection 不再取 local map 的 max-pool
- L_cons    = (1/|M|) Σ_{p∈M} ρ(z^std_p − z^exc_p),  ρ=Huber(δ=1)
              编辑区响应对全局伪影剥离保持稳定 (std/exc 同 mask、同源图;
              像素验证: 编辑区差异 0.98/255, 背景差异 5.84/255)
- L_placebo = [m − min(E_M(z^std), E_M(z^exc)) + max(E_M−(z^std), E_M−(z^exc))]_+
              真实区平均 logit 应高于等面积随机位移区 (M^- ~ IoU<0.3 shift)
- 变体: dec(双分支基线) / cons(+L_cons) / conspl(+L_cons+L_placebo)

损失: L = bce_img + bce_pix + 0.5*dice + λ_cons*L_cons + λ_pl*L_placebo
与 v4 主协议一致: master split / 双 loader (像素曝光恒定) / 20 ep / 128 batch /
wd=1e-4 / checkpoint = val image-AUC (global 分支) best epoch。
head 保存为 DualBranchHead state_dict (weak_sup_v5_{tag}_head.pt)。
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


def source_key(fname):
    f = fname.replace("_simple.jpg", ".jpg")
    m = KEY_RE.match(f)
    if m:
        return m.group(1)
    if f.endswith(".jpg"):
        return f[:-4]
    return fname


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


class GlobalHead(nn.Module):
    """图像级分支: 1369 patch 特征 mean-pool → MLP → 标量 logit"""

    def __init__(self, in_dim, hidden=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x).squeeze(-1)  # (b,)


class DualBranchHead(nn.Module):
    def __init__(self, in_dim, local_hidden=48, global_hidden=20):
        super().__init__()
        self.local = MLPHead(in_dim, hidden=local_hidden)
        self.global_net = GlobalHead(in_dim, hidden=global_hidden)
        self.local_hidden = local_hidden
        self.global_hidden = global_hidden
        self.in_dim = in_dim

    def forward(self, x):
        return {"local": self.local(x), "global": self.global_net(x)}

    def save_meta(self):
        return {"in_dim": int(self.in_dim), "local_hidden": int(self.local_hidden),
                "global_hidden": int(self.global_hidden)}


def load_mask(mask_path: str) -> np.ndarray:
    img = Image.open(IMG_ROOT / mask_path).convert("L")
    img = img.resize((GRID, GRID), Image.NEAREST)
    return (np.asarray(img) > 127).astype(np.float32)


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1) + eps
    return 1 - (2 * inter / denom).mean()


def huber_diff(logits_a, logits_b, delta=1.0):
    """逐 patch Huber(z_a - z_b), 只用于 mask 内点"""
    d = logits_a - logits_b
    abs_d = d.abs()
    quad = 0.5 * d * d
    lin = delta * (abs_d - 0.5 * delta)
    loss = torch.where(abs_d <= delta, quad, lin)
    return loss.mean()


def make_placebo(mask, rng):
    """等面积随机位移 mask: roll(dx,dy), IoU<0.35 接受, 20 次重采样取最优"""
    m = mask.reshape(GRID, GRID)
    best = m
    best_iou = 1.0
    for _ in range(20):
        dx = int(rng.randint(-14, 15))
        dy = int(rng.randint(-14, 15))
        if dx == 0 and dy == 0:
            dx = 3
        sh = np.roll(np.roll(m, dx, axis=1), dy, axis=0)
        inter = float((sh * m).sum())
        iou = inter / max(m.sum() + sh.sum() - inter, 1e-6)
        if iou < best_iou:
            best_iou, best = iou, sh
        if best_iou < 0.35:
            break
    return best.ravel()


def erode_mask(mask, size=1):
    """3x3x... 腐蚀: 核心区 (边缘 patch 不强求一致; 空则回退原 mask)"""
    m = (mask.reshape(GRID, GRID) > 0.5).astype(np.uint8)
    e = m.copy()
    for dx in range(-size, size + 1):
        for dy in range(-size, size + 1):
            e = e & np.roll(np.roll(m, dx, 1), dy, 0)
    e = e.astype(np.float32)
    if e.sum() == 0:
        return mask.ravel()
    return e.ravel()


def train_v5(tag, epochs=20, batch_size=128, lr=1e-3, seed=42, mask_budget="ALL",
             variant="dec", lam_cons=0.25, lam_pl=0.25, pl_margin=0.5,
             local_hidden=48, global_hidden=20, master_split=None,
             mask_manifest=None, wd=1e-4, cache=None, use_hp=True):
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
    print(f"[v5 {variant}] master split: train={len(train_idx)} val={len(val_idx)} "
          f"seed={seed} budget={mask_budget}", flush=True)
    labels_arr = np.array([l["label"] for l in labels])

    # ---- mask 全表 + 预算注记 ----
    masks = np.zeros((n_total, n_patches), dtype=np.float32)
    for l in labels:
        mp = l.get("mask_path")
        if mp and (IMG_ROOT / mp).exists():
            masks[l["idx"]] = load_mask(mp).ravel()

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

    # ---- 预计算: 腐蚀核心区 + 4 个 placebo 候选 (静态, 每 step 免循环) ----
    t_pre = time.time()
    ann_erode = np.zeros((len(anns), n_patches), dtype=np.float32)
    ann_placebo = np.zeros((len(anns), 4, n_patches), dtype=np.float32)
    rng_pre = np.random.RandomState(seed + 1000)
    for jj in range(len(anns)):
        m = ann_masks[jj]
        if m.sum() == 0:
            continue
        ann_erode[jj] = erode_mask(m)
        for c in range(4):
            ann_placebo[jj, c] = make_placebo(m, rng_pre)
    print(f"[precalc] erode+placebo done in {(time.time()-t_pre):.1f}s", flush=True)

    def feat_of(idxs):
        # 纯 fp32 (与 v4 baseline 完全同精度, 保证可比性; 不用 autocast)
        dino = torch.from_numpy(dino_mm[idxs])
        parts = [dino]
        if use_hp:
            parts.append(torch.from_numpy(hp_mm[idxs]))
        return torch.cat(parts, dim=-1).float().cuda()

    model = DualBranchHead(dino_dim + (1 if use_hp else 0),
                           local_hidden=local_hidden, global_hidden=global_hidden).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,} (local={sum(p.numel() for p in model.local.parameters()):,}, "
          f"global={sum(p.numel() for p in model.global_net.parameters()):,})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def predict(idxs):
        model.eval()
        g_out, l_out = [], []
        with torch.no_grad():
            for i in range(0, len(idxs), batch_size):
                b = idxs[i:i + batch_size]
                out = model(feat_of(b))
                g_out.append(torch.sigmoid(out["global"]).cpu().numpy())
                l_out.append(torch.sigmoid(out["local"]).squeeze(-1).cpu().numpy())
        g = np.concatenate(g_out) if g_out else np.zeros((0,))
        l = np.concatenate(l_out, 0) if l_out else np.zeros((0, 0))
        return g, l

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
            feat_pix = feat_of(pidx)
            valid = ann_exc[j] >= 0
            exc_idx = ann_exc[j][valid]
            if len(exc_idx):
                feat_exc = feat_of(exc_idx)
                X_loc = torch.cat([feat_pix, feat_exc], 0)
            else:
                feat_exc = None
                X_loc = feat_pix
            logits_loc = model.local(X_loc).squeeze(-1)
            pix_logits = logits_loc[:len(bi)]
            # ---- global 分支: mean-pool 后拼接 (省 ~800MB 显存) ----
            g_feats = [feat_of(bi).mean(dim=1), feat_pix.mean(dim=1)]
            if feat_exc is not None:
                g_feats.append(feat_exc.mean(dim=1))
            g_logits = model.global_net(torch.cat(g_feats, 0))
            y_g = torch.cat([
                torch.from_numpy(labels_arr[bi]).float().cuda(),
                torch.ones(len(bi), device="cuda"),            # pidx 全为编辑图
                torch.ones(len(exc_idx), device="cuda") if len(exc_idx) else
                torch.zeros(0, device="cuda"),
            ])
            bce_img = F.binary_cross_entropy(torch.sigmoid(g_logits), y_g)
            # ---- 像素损失 ----
            tgt = torch.from_numpy(ann_masks[j]).cuda()
            bce_pix = F.binary_cross_entropy_with_logits(pix_logits, tgt)
            dice = dice_loss(pix_logits, tgt)
            loss = bce_img + bce_pix + 0.5 * dice
            # ---- L_cons (全向量化) ----
            if variant in ("cons", "conspl") and feat_exc is not None:
                m_cons = torch.from_numpy(ann_erode[j][valid]).cuda()
                diff = F.huber_loss(pix_logits[valid], logits_loc[len(bi):],
                                    delta=1.0, reduction="none")
                l_cons = (diff * m_cons).sum() / m_cons.sum().clamp(min=1)
                loss = loss + lam_cons * l_cons
                cons_l.append(float(l_cons.item()))
            # ---- L_placebo (全向量化, 候选轮换) ----
            if variant == "conspl" and feat_exc is not None:
                ms = torch.from_numpy(ann_masks[j][valid]).cuda()
                pbo = torch.from_numpy(ann_placebo[j[valid], step_i % 4]).cuda()
                em_s = (pix_logits[valid] * ms).sum(1) / ms.sum(1).clamp(min=1)
                em_e = (logits_loc[len(bi):] * ms).sum(1) / ms.sum(1).clamp(min=1)
                ep_s = (pix_logits[valid] * pbo).sum(1) / pbo.sum(1).clamp(min=1)
                ep_e = (logits_loc[len(bi):] * pbo).sum(1) / pbo.sum(1).clamp(min=1)
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
            del X_loc, feat_pix
            img_l.append(float(bce_img.item()))
            pix_l.append(float((bce_pix + 0.5 * dice).item()))
        sched.step()

        # ---- val: global 分支图像 AUC 选 best (与 v4 协议一致) ----
        gv, lv = predict(val_idx)
        yv = labels_arr[val_idx]
        img_auc = roc_auc_score(yv, gv)
        mean_img = float(np.mean(img_l))
        mean_pix = float(np.mean(pix_l))
        mean_cons = float(np.mean(cons_l)) if cons_l else 0.0
        mean_pl = float(np.mean(pl_l)) if pl_l else 0.0
        print(f"ep{epoch+1}: img={mean_img:.4f} pix={mean_pix:.4f} "
              f"cons={mean_cons:.4f} pl={mean_pl:.4f} val_imgAUC={img_auc:.4f} "
              f"({(time.time()-t0)/60:.1f}min) peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB",
              flush=True)
        if img_auc > best_val:
            best_val = img_auc
            best_state = {kk: vv.clone() for kk, vv in model.state_dict().items()}

    print(f"\n训练完成 {(time.time()-t0)/60:.1f} 分钟; best val imgAUC={best_val:.4f}", flush=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT / f"weak_sup_v5_{tag}_head.pt")
    side = {
        "tag": tag, "variant": variant, "seed": int(seed), "budget": str(mask_budget),
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "wd": wd,
        "lam_cons": float(lam_cons), "lam_pl": float(lam_pl), "pl_margin": float(pl_margin),
        "use_hp": bool(use_hp), "n_pairable_anns": int(n_pairable),
        "head_config": model.save_meta(),
        "val_img_auc": float(best_val),
        "protocol": "A_20_128 master split dual-loader",
    }
    with open(OUT / f"weak_sup_v5_{tag}_meta.json", "w", encoding="utf-8") as f:
        json.dump(side, f, indent=1)
    print(f"模型: {OUT / f'weak_sup_v5_{tag}_head.pt'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_budget", default="ALL")
    p.add_argument("--variant", default="dec", choices=["dec", "cons", "conspl"])
    p.add_argument("--lam_cons", type=float, default=0.25)
    p.add_argument("--lam_pl", type=float, default=0.25)
    p.add_argument("--pl_margin", type=float, default=0.5)
    p.add_argument("--local_hidden", type=int, default=48)
    p.add_argument("--global_hidden", type=int, default=20)
    p.add_argument("--master_split", default=str(OUT / "split_budget_master.json"))
    p.add_argument("--mask_manifest", default=str(OUT / "mask_budget_manifest.json"))
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--no_hp", action="store_true")
    p.add_argument("--cache", type=str, default=None)
    args = p.parse_args()
    train_v5(tag=args.tag, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
             seed=args.seed, mask_budget=args.mask_budget, variant=args.variant,
             lam_cons=args.lam_cons, lam_pl=args.lam_pl, pl_margin=args.pl_margin,
             local_hidden=args.local_hidden, global_hidden=args.global_hidden,
             master_split=args.master_split, mask_manifest=args.mask_manifest,
             wd=args.wd, cache=args.cache, use_hp=not args.no_hp)
