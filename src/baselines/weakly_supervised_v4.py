"""Mixed-supervision patch readout and mask-budget experiments.

Combines image-level BCE on max-pooled patch scores with pixel BCE and Dice
losses. The master-split training path supports nested mask budgets and a
matched pixel-loss-off control. The legacy path retains its original loss
weighting; use the master-split arguments for the manuscript protocol."""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

CACHE = "D:/lunwen/data/features_cache"
IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
OUT = Path("D:/lunwen/outputs")
GRID = 37
N_PATCHES = 1369

KEY_RE = re.compile(r"^(.*?)_(?:CelebAHQ|CityScapes|SUN_RGBD|OpenImages)_"
                    r"(?:Kandinsky_2_2|StableDiffusion_v4|OpenJourney)"
                    r"(?:_simple)?\.jpg$")


def source_key(fname):
    """编辑图 {src}_{attr}_{DOMAIN}_{MODEL}[_simple].jpg / real {src}_{attr}.jpg -> {src}_{attr}"""
    f = fname.replace("_simple.jpg", ".jpg")
    m = KEY_RE.match(f)
    if m:
        return m.group(1)
    if f.endswith(".jpg"):
        return f[:-4]
    return fname


def src_prefix(fname):
    """{src}_{attr} -> {src} (源图级)"""
    return source_key(fname).rsplit("_", 1)[0]


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


def build_masks(labels_data, n_total, n_patches):
    masks = np.zeros((n_total, n_patches), dtype=np.float32)
    n_mask = 0
    for item in labels_data:
        mp = item.get("mask_path")
        if mp and (IMG_ROOT / mp).exists():
            masks[item["idx"]] = load_mask(mp).ravel()
            n_mask += 1
    print(f"mask 预加载: {n_mask}/{n_total}")
    return masks


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1) + eps
    return 1 - (2 * inter / denom).mean()


def _feat_of(idx, dino_mm, hp_mm, use_hp):
    """按样本 idx 取缓存特征 -> (1, 1369, 385) cuda tensor (float16 direct)"""
    dino = torch.from_numpy(dino_mm[idx])
    parts = [dino]
    if use_hp:
        parts.append(torch.from_numpy(hp_mm[idx]))
    return torch.cat(parts, dim=-1).unsqueeze(0).cuda().float()


def train_v4(epochs=20, batch_size=128, lr=1e-3, hidden=64, lam_img=0.5,
             exclude_model=None, tag="", use_hp=True, seed=42, exclude_json=None,
             exclude_src_level=False, use_trace=False, alpha=0.1, beta=0.1,
             trace_margin=0.5, pair_manifest=None, trace_balance=False,
             cache=None):
    cache = cache or CACHE
    with open(f"{cache}/labels.json") as f:
        labels_data = json.load(f)
    with open(f"{cache}/metadata.json") as f:
        meta = json.load(f)
    n_total, n_patches, dino_dim = meta["n_total"], meta["n_patches"], meta["dino_dim"]

    dino_mm = np.memmap(f"{cache}/dino_tokens.npy", dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, dino_dim))
    hp_mm = np.memmap(f"{cache}/highpass.npy", dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))

    cats = np.array([l.get("cat", "unknown") for l in labels_data])
    labels_arr = np.array([l["label"] for l in labels_data])
    models_arr = np.array([l.get("model") for l in labels_data])
    all_idx = np.arange(n_total)
    test_mask = cats == "CelebAHQ"
    train_idx = all_idx[~test_mask]
    test_idx = all_idx[test_mask]

    # leave-one-out: 训练时排除指定 inpainter 的编辑图 (仍保留其 real)
    if exclude_model:
        excl = (models_arr == exclude_model) & (labels_arr == 1)
        train_idx = train_idx[~excl[train_idx]]
        print(f"leave-one-out: 排除 inpainter={exclude_model} 的编辑图 "
              f"({excl.sum()} 张), 训练集剩 {len(train_idx)}")

    # Exclude all 800 evaluation records in imdl_inpx_test.json from the split.
    # 按 source key 排除 (standard + exchange 双版本 + real), 杜绝同编辑孪生版本泄漏
    # exclude_src_level=True 时升级为 source-disjoint: 按 {src} 前缀排除同源图所有属性编辑
    if exclude_json:
        import json as _json
        excl_records = _json.load(open(exclude_json))
        path2idx = {l["path"]: i for i, l in enumerate(labels_data)}

        excl_keys = set()
        for img_path, mask_path in excl_records:
            rel = str(Path(img_path).relative_to(IMG_ROOT))
            idx = path2idx.get(rel)
            if idx is None:
                continue
            name = rel.replace("\\", "/").split("/")[-1]
            excl_keys.add(src_prefix(name) if exclude_src_level else source_key(name))

        key_fn = src_prefix if exclude_src_level else source_key
        kept = [i for i in train_idx
                if key_fn(labels_data[i]["path"].replace("\\", "/").split("/")[-1])
                not in excl_keys]
        n_excl = len(train_idx) - len(kept)
        lvl = "source-disjoint" if exclude_src_level else "edit-disjoint"
        print(f"评测集按 {lvl} 排除: {len(excl_keys)} 个 key, "
              f"训练候选排除 {n_excl} 个样本 → 剩 {len(kept)}")
        train_idx = np.array(kept)

    # 随机性: seed 控制 train/val 划分、epoch 内 shuffle 与参数初始化
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng.shuffle(train_idx)
    n_val = int(len(train_idx) * 0.1)
    val_idx = train_idx[:n_val]
    train_idx = train_idx[n_val:]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test(CelebAHQ): {len(test_idx)}")

    # 保存划分 (评测脚本据此在未见测试集上报固定阈值)
    split_name = f"split_{tag}" if tag else "split_main"
    with open(OUT / f"{split_name}.json", "w") as f:
        json.dump({"seed": int(seed), "exclude_json": exclude_json,
                   "train_idx": train_idx.tolist(), "val_idx": val_idx.tolist(),
                   "test_idx": test_idx.tolist()}, f)
    print(f"划分已保存: {OUT / f'{split_name}.json'}")

    masks = build_masks(labels_data, n_total, n_patches)

    # ---- Content-Controlled Edit-Trace Supervision 配对表 ----
    # 负样本 (clean) 来源: ① sibling edit: 同 src 前缀的不同 key 编辑, 其未编辑位置 = 同场景 clean
    #                        ② real 文件 ({src}.jpg, 仅部分 key 可用)
    # safe_region: target_mask ∩ ~dilate(sibling_mask), 排除 sibling 编辑区及其邻域
    fake_of_key = real_by_src = sibling_key = None
    if use_trace:
        from collections import defaultdict
        manifest_name = f"pairs_{tag}.json" if tag else "pairs_main.json"
        manifest_path = OUT / manifest_name
        if pair_manifest:
            manifest_path = Path(pair_manifest)
        if manifest_path.exists():
            _m = json.load(open(manifest_path))
            fake_of_key = defaultdict(list, {k: list(v) for k, v in _m["fake_of_key"].items()})
            real_by_src = {k: v for k, v in _m["real_by_src"].items()}
            sibling_key = {k: v for k, v in _m["sibling_key"].items()}
            print(f"配对 manifest 加载: {manifest_path} ({len(fake_of_key)} keys)")
        else:
            fake_of_key = defaultdict(list)   # key({src}_{attr}) -> [样本 idx...] (fake 编辑, 含 std/exc)
            real_by_src = {}                  # {src} -> real 样本 idx (存在时)
            src_keys = defaultdict(list)      # {src} -> [key...] (同源不同属性编辑)
            for i in train_idx:
                name = labels_data[i]["path"].replace("\\", "/").split("/")[-1]
                if labels_arr[i] == 1:
                    k = source_key(name)
                    fake_of_key[k].append(int(i))
                    src_keys[src_prefix(name)].append(k)
                elif name.endswith(".jpg"):
                    real_by_src.setdefault(name[:-4], int(i))
            sibling_key = {}
            for src, keys in src_keys.items():
                uniq = sorted(set(keys))
                if len(uniq) >= 2:
                    for k in uniq:
                        sibling_key[k] = [u for u in uniq if u != k][0]
            n_fake = sum(len(v) for v in fake_of_key.values())
            n_sib = sum(1 for k in fake_of_key if k in sibling_key)
            n_real = sum(1 for k in fake_of_key if src_prefix(k) in real_by_src)
            n_clean = sum(1 for k in fake_of_key
                          if k in sibling_key or src_prefix(k) in real_by_src)
            n_pair_keys = sum(1 for v in fake_of_key.values() if len(v) >= 2)
            print(f"L_trace 配对: fake 编辑 {n_fake} 个 (key {len(fake_of_key)}); "
                  f"sibling 负样本 {n_sib} ({n_sib/max(len(fake_of_key),1)*100:.1f}%), "
                  f"real 负样本 {n_real} ({n_real/max(len(fake_of_key),1)*100:.1f}%), "
                  f"至少一种 clean 来源 {n_clean} ({n_clean/max(len(fake_of_key),1)*100:.1f}%); "
                  f"L_pair (std<->exc) 可用 {n_pair_keys} 个 key")
            # 固定配对 manifest: 保存映射摘要, 供各变体共用同一配对
            manifest = {"fake_of_key": {k: v for k, v in fake_of_key.items()},
                        "real_by_src": real_by_src,
                        "sibling_key": sibling_key}
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)
            print(f"配对 manifest 已保存: {manifest_path}")
        # safe_region 覆盖率统计: 每域有效配对/空区域/平均面积比
        from collections import defaultdict as _dd
        safe_stats = _dd(lambda: {"n_eff": 0, "n_empty": 0, "ratios": []})
        for k, v in fake_of_key.items():
            idx0 = v[0]
            dom = labels_data[idx0].get("cat")
            em = masks[idx0] > 0.5
            if em.sum() == 0:
                continue
            sk = sibling_key.get(k)
            used = False
            if sk is not None and fake_of_key.get(sk):
                c_idx = fake_of_key[sk][0]
                if c_idx != idx0:
                    m_dil = (F.max_pool2d(
                        torch.from_numpy((masks[c_idx] > 0.5).astype(np.float32)).view(1, 1, GRID, GRID),
                        kernel_size=3, stride=1, padding=1).view(-1) > 0.5).cpu().numpy()
                    safe = em & ~m_dil
                    safe_stats[dom]["n_eff"] += 1
                    if safe.sum() == 0:
                        safe_stats[dom]["n_empty"] += 1
                    else:
                        safe_stats[dom]["ratios"].append(float(safe.sum() / em.sum()))
                    used = True
            if not used:
                r_idx = real_by_src.get(src_prefix(labels_data[idx0]["path"].replace("\\", "/").split("/")[-1]))
                if r_idx is not None and r_idx != idx0:
                    safe_stats[dom]["n_eff"] += 1
                    safe_stats[dom]["ratios"].append(1.0)
        dom_cov_s = {}
        for d, st in sorted(safe_stats.items()):
            ratios = st["ratios"]
            mean_ratio = float(np.mean(ratios)) if ratios else 0.0
            dom_cov_s[d] = {"n_eff": st["n_eff"], "n_empty": st["n_empty"],
                            "mean_ratio": round(mean_ratio, 3)}
            print(f"  safe-region {d:12s}: 有效配对 {st['n_eff']}, 空区域 {st['n_empty']}, "
                  f"平均面积比 {mean_ratio:.2f}")
        # 域加权: trace 项按域反比加权 + 均值归一化 + 上限 3x
        # 归一化保证: sum(w_d * n_eff_d) == total_eff (加权前后总 trace 权重一致)
        dom_weights = {}
        total_eff = sum(st["n_eff"] for st in safe_stats.values())
        if total_eff > 0 and trace_balance:
            for d, st in safe_stats.items():
                w = total_eff / max(st["n_eff"], 1) / len(safe_stats)
                dom_weights[d] = min(w, 3.0)   # 上限: 防 OpenImages 少量配对梯度爆炸
            wsum = sum(dom_weights.get(d, 1.0) * st["n_eff"]
                       for d, st in safe_stats.items())
            print(f"域加权 (trace_balance, 均值归一化, 总权重 {wsum:.0f} vs 未加权 {total_eff}): "
                  f"{ {d: str(round(w, 3)) for d, w in dom_weights.items()} }")

    model = MLPHead(dino_dim + (1 if use_hp else 0), hidden=hidden).cuda()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,} "
          f"(dino_dim={dino_dim}, hp={'on' if use_hp else 'off'}, hidden={hidden})")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def load_batch(idxs):
        # float16 direct path (benchmarked ~4.5x faster than float32 convert;
        # no extra RAM): dino stays fp16 until cat converts to fp32 on GPU
        dino = torch.from_numpy(dino_mm[idxs])
        parts = [dino]
        if use_hp:
            hp = torch.from_numpy(hp_mm[idxs])
            parts.append(hp)
        X = torch.cat(parts, dim=-1).cuda().float()
        y = torch.from_numpy(masks[idxs]).cuda()
        y_img = torch.from_numpy(labels_arr[idxs]).float().cuda()
        return X, y, y_img

    def predict_patches(idxs):
        model.eval()
        all_scores, all_masks = [], []
        with torch.no_grad():
            for i in range(0, len(idxs), batch_size):
                b_idx = idxs[i:i + batch_size]
                X, y, _ = load_batch(b_idx)
                logits = model(X)
                all_scores.append(torch.sigmoid(logits).cpu().numpy())
                all_masks.append(y.cpu().numpy())
        return np.concatenate(all_scores), np.concatenate(all_masks)

    best_val = 0
    best_state = None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        rng.shuffle(train_idx)
        losses = []
        pbar = tqdm(range(0, len(train_idx), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for start in pbar:
            b_idx = train_idx[start:start + batch_size]
            X, y, y_img = load_batch(b_idx)
            logits = model(X)
            bce_pix = F.binary_cross_entropy_with_logits(logits, y)
            dice = dice_loss(logits, y)
            # 图像级: max-pool 分数 vs 图像标签 (real=0, fake=1)
            img_score = torch.sigmoid(logits).max(dim=1).values
            bce_img = F.binary_cross_entropy(img_score, y_img)
            loss = bce_pix + 0.5 * dice + lam_img * bce_img

            # ---- Content-Controlled Edit-Trace Supervision ----
            trace_loss = pair_loss = torch.zeros((), device=logits.device)
            trace_n = pair_n = 0
            if use_trace:
                t_terms, p_terms = [], []
                dom_trace_acc = defaultdict(float)   # 各域 trace loss 累计
                dom_trace_cnt = defaultdict(int)
                for j, idx in enumerate(b_idx):
                    if labels_arr[idx] != 1:
                        continue
                    name = labels_data[idx]["path"].replace("\\", "/").split("/")[-1]
                    k = source_key(name)
                    edit_mask = y[j] > 0.5
                    if edit_mask.sum() == 0:
                        continue
                    # L_trace: safe_region = target_mask ∩ ~dilate(sibling_mask)
                    clean_idx = None
                    clean_mask = None
                    sk = sibling_key.get(k)
                    if sk is not None and fake_of_key.get(sk):
                        c_idx = fake_of_key[sk][0]
                        if c_idx != idx:
                            clean_idx = c_idx
                            # 膨胀 1 patch (3x3 max-pool on 37x37) -> 排除 sibling 编辑区邻域
                            m_sib_t = torch.from_numpy(
                                (masks[c_idx] > 0.5).astype(np.float32)).view(1, 1, GRID, GRID).cuda()
                            m_dil = F.max_pool2d(m_sib_t, kernel_size=3, stride=1, padding=1).view(-1) > 0.5
                            clean_mask = edit_mask & ~m_dil          # edit_mask 为 torch bool (cuda)
                    if clean_idx is None:
                        r_idx = real_by_src.get(src_prefix(name))
                        if r_idx is not None and r_idx != idx:
                            clean_idx = r_idx
                            clean_mask = edit_mask             # real 全 clean
                    if clean_idx is not None and clean_mask is not None and clean_mask.sum() > 0:
                        X_c = _feat_of(clean_idx, dino_mm, hp_mm, use_hp)
                        with torch.no_grad():
                            z_c = model(X_c).squeeze(0)
                        w_dom = dom_weights.get(labels_data[idx].get("cat"), 1.0) if trace_balance else 1.0
                        term = w_dom * F.softplus(trace_margin - (logits[j] - z_c))[clean_mask]
                        t_terms.append(term)
                        dom = labels_data[idx].get("cat")
                        dom_trace_acc[dom] += float(term.mean().item())
                        dom_trace_cnt[dom] += 1
                        trace_n += 1
                    # L_pair: standard<->exchange 双向 stop-gradient 一致性 (编辑区)
                    #   0.5*|p_A - sg(p_B)| + 0.5*|p_B - sg(p_A)|  (两侧都更新)
                    others = [i2 for i2 in fake_of_key.get(k, []) if i2 != idx]
                    if others:
                        X_p = _feat_of(others[0], dino_mm, hp_mm, use_hp)
                        z_p = model(X_p).squeeze(0)            # 带梯度 (双向)
                        p_other = torch.sigmoid(z_p)
                        p_self = torch.sigmoid(logits[j])
                        em = edit_mask
                        p_terms.append((0.5 * (p_self - p_other.detach()).abs()
                                        + 0.5 * (p_other - p_self.detach()).abs())[em])
                        pair_n += 1
                if t_terms:
                    trace_loss = torch.cat(t_terms).mean()
                if p_terms:
                    pair_loss = torch.cat(p_terms).mean()
                loss = loss + alpha * trace_loss + beta * pair_loss
                # 各域 trace loss 日志 (每 epoch 末尾打印)
                dom_trace_log = {d: f"{dom_trace_acc[d]/max(dom_trace_cnt[d],1):.4f}(n={dom_trace_cnt[d]})"
                                 for d in sorted(dom_trace_acc)}

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{np.mean(losses[-20:]):.4f}",
                              "tr": f"{trace_loss.item():.4f}" if use_trace else None,
                              "pr": f"{pair_loss.item():.4f}" if use_trace else None})
        sched.step()
        if use_trace and dom_trace_log:
            print(f"  各域 trace loss (epoch {epoch+1}): {dom_trace_log}")

        scores, masks_v = predict_patches(val_idx)
        img_auc = roc_auc_score(labels_arr[val_idx], scores.max(axis=1))
        best_miou = 0
        for th in np.linspace(0.1, 0.9, 17):
            pred = (scores >= th).astype(np.float32)
            tp = (pred * masks_v).sum(); fp = (pred * (1 - masks_v)).sum()
            fn = ((1 - pred) * masks_v).sum()
            best_miou = max(best_miou, tp / (tp + fp + fn + 1e-8))
        print(f"  ep{epoch+1}: loss={np.mean(losses):.4f} "
              f"val imgAUC={img_auc:.4f} patch mIoU={best_miou:.4f}")
        if img_auc > best_val:
            best_val = img_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\n训练完成! {(time.time()-t0)/60:.1f} 分钟")
    if best_state:
        model.load_state_dict(best_state)
    suffix = f"_{tag}" if tag else ""
    torch.save(model.state_dict(), OUT / f"weak_sup_v4{suffix}_head.pt")
    print(f"模型: {OUT / f'weak_sup_v4{suffix}_head.pt'}")

    # ---- 测试 (CelebAHQ) ----
    print("\n" + "=" * 60)
    scores_t, masks_t = predict_patches(test_idx)
    img_scores_t = scores_t.max(axis=1)
    y_t = labels_arr[test_idx]
    auc = roc_auc_score(y_t, img_scores_t)
    thr = np.median(img_scores_t)
    yp = (img_scores_t >= thr).astype(int)
    print(f"图像级: Acc={accuracy_score(y_t, yp):.4f} AUC={auc:.4f} "
          f"R.Acc={accuracy_score(y_t[y_t==0], yp[y_t==0]):.4f} "
          f"F.Acc={accuracy_score(y_t[y_t==1], yp[y_t==1]):.4f}")

    fake_mask = y_t == 1

    # Select the fixed localization threshold on validation data only.
    scores_v, masks_v = predict_patches(val_idx)
    val_fake = labels_arr[val_idx] == 1
    best_thr_val, best_miou_val = 0.5, 0.0
    for th in np.linspace(0.05, 0.95, 91):
        pred = (scores_v[val_fake] >= th).astype(np.float32)
        m = masks_v[val_fake]
        tp = (pred * m).sum(); fp = (pred * (1 - m)).sum(); fn = ((1 - pred) * m).sum()
        iou = tp / (tp + fp + fn + 1e-8)
        if iou > best_miou_val:
            best_miou_val, best_thr_val = iou, th
    print(f"验证集选阈值: thr={best_thr_val:.3f} (val fake 全局 mIoU={best_miou_val:.4f})")

    pred_fix = (scores_t[fake_mask] >= best_thr_val).astype(np.float32)
    m = masks_t[fake_mask]
    tp = (pred_fix * m).sum(); fp = (pred_fix * (1 - m)).sum(); fn = ((1 - pred_fix) * m).sum()
    prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
    fixed_miou = tp / (tp + fp + fn + 1e-8)
    fixed_f1 = 2 * prec * rec / (prec + rec + 1e-8)
    print(f"定位固定阈值 (thr={best_thr_val:.3f}): mIoU={fixed_miou:.4f} F1={fixed_f1:.4f}")

    # oracle (测试集上选 best-thr, 仅作补充口径)
    best_miou_all, best_f1_all = 0, 0
    for th in np.linspace(0.05, 0.95, 91):
        pred = (scores_t[fake_mask] >= th).astype(np.float32)
        tp = (pred * m).sum(); fp = (pred * (1 - m)).sum(); fn = ((1 - pred) * m).sum()
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        best_miou_all = max(best_miou_all, iou)
        best_f1_all = max(best_f1_all, 2 * prec * rec / (prec + rec + 1e-8))
    print(f"定位 oracle (补充): mIoU={best_miou_all:.4f} F1={best_f1_all:.4f}")

    sizes = np.array([l.get("size_class") or "unknown" for l in labels_data])[test_idx][fake_mask]
    for sc in ["tiny", "small", "medium", "large"]:
        mm = sizes == sc
        if mm.sum() < 5:
            continue
        best_iou = 0
        for th in np.linspace(0.05, 0.95, 91):
            pred = (scores_t[fake_mask][mm] >= th).astype(np.float32)
            m = masks_t[fake_mask][mm]
            tp = (pred * m).sum(); fp = (pred * (1 - m)).sum(); fn = ((1 - pred) * m).sum()
            best_iou = max(best_iou, tp / (tp + fp + fn + 1e-8))
        print(f"  {sc} (n={mm.sum()}): mIoU={best_iou:.4f}")

    with open(OUT / f"weak_sup_v4{suffix}_results.json", "w") as f:
        json.dump({
            "img_auc": float(auc),
            "img_acc": float(accuracy_score(y_t, yp)),
            "r_acc": float(accuracy_score(y_t[y_t == 0], yp[y_t == 0])),
            "f_acc": float(accuracy_score(y_t[y_t == 1], yp[y_t == 1])),
            "loc_miou": float(best_miou_all),
            "loc_f1": float(best_f1_all),
            "fixed_thr": float(best_thr_val),
            "fixed_loc_miou": float(fixed_miou),
            "fixed_loc_f1": float(fixed_f1),
            "val_fixed_miou": float(best_miou_val),
            "lam_img": float(lam_img),
            "exclude_model": exclude_model,
            "seed": int(seed),
            "use_trace": bool(use_trace),
            "alpha": float(alpha),
            "beta": float(beta),
            "trace_margin": float(trace_margin),
        }, f, indent=1)
    print(f"结果: {OUT / f'weak_sup_v4{suffix}_results.json'}")


def train_v4_budget(epochs=20, batch_size=128, lr=1e-3, hidden=64, lam_img=0.5,
                    seed=42, master_split=None, mask_manifest=None, mask_budget=None,
                    swap_masks=False, real_zero=False, linear_probe=False,
                    amp=False, sel_manifest=None, wd=1e-4, use_hp=True, tag="", cache=None,
                    pixel_off=False):
    """新协议训练器(实验计划 v4 §2.2/§2.3):
    - master split: 固定 train/val indices(三 seed 共用), 不做行级切分;
    - 双 loader: 每 step = image batch(128) + mask batch(128) 拼接 256 样本一次前向一次更新;
      图像标签 256 全部可用(全量免费), 像素项仅对 mask batch 128 计算(BCE + 0.5*Dice, 按 128 归一化);
      像素批从 k 个注记均匀放回 → 每 epoch 像素曝光 = U_img*128 恒定(与 k 无关);
    - k=0+: mask_budget='0' → 无像素流, loss = image BCE;
    - swap_masks: 像素批目标 mask 按 swap manifest 置换(mask-image correspondence control);
    - checkpoint: val image-AUC best epoch(与旧训练器一致); 阈值由 eval_seen500 在 val 固定。
    """
    labels = json.load(open(f"{cache or CACHE}/labels.json"))
    meta = json.load(open(f"{cache or CACHE}/metadata.json"))
    n_total, n_patches, dino_dim = meta["n_total"], meta["n_patches"], meta["dino_dim"]

    dino_mm = np.memmap(f"{cache or CACHE}/dino_tokens.npy", dtype=np.float16, mode="r",
                        shape=(n_total, n_patches, dino_dim))
    hp_mm = np.memmap(f"{cache or CACHE}/highpass.npy", dtype=np.float32, mode="r",
                      shape=(n_total, n_patches, 1))

    split = json.load(open(master_split))
    train_idx = np.array(split["train_idx"], dtype=np.int64)
    val_idx = np.array(split["val_idx"], dtype=np.int64)
    test_idx = np.array(split["test_idx"], dtype=np.int64)
    print(f"[master split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} seed={seed}")
    labels_arr = np.array([l["label"] for l in labels])
    # 全量 mask 数组(与旧训练器同构; real 行全 0; G0' real-zero 桥接用)
    masks = np.zeros((n_total, n_patches), dtype=np.float32)
    for l in labels:
        mp = l.get("mask_path")
        if mp and (IMG_ROOT / mp).exists():
            masks[l["idx"]] = load_mask(mp).ravel()

    # ---- 预算注记 ----
    man = json.load(open(mask_manifest))
    all_anns = man["ordered_annotations"]
    if mask_budget in (None, ""):
        k = len(all_anns)
    elif str(mask_budget).upper() == "ALL":
        k = len(all_anns)
    else:
        k = int(mask_budget)
    k = min(k, len(all_anns))
    anns = all_anns[:k]
    if sel_manifest:
        sel = json.load(open(sel_manifest, encoding="utf-8"))
        ids = [int(x) for x in sel["selected_ann_ids"]]
        by_id = {int(a["ann_id"]): a for a in all_anns}
        anns = [by_id[i] for i in ids]
        k = len(anns)
        print(f"[sel_manifest] {sel_manifest}: k={k}", flush=True)
    ann_rep = np.array([a["rep_idx"] for a in anns], dtype=np.int64)
    # 全注记 mask 预载(14,731 x 1369 fp32 ≈ 80MB): 预算内索引与 swap 目标(可能超出 k)直接查表
    ann_masks_all = np.zeros((len(all_anns), n_patches), dtype=np.float32)
    for j, a in enumerate(all_anns):
        idx = a["rep_idx"]
        mp = labels[idx].get("mask_path")
        if mp and (IMG_ROOT / mp).exists():
            ann_masks_all[j] = load_mask(mp).ravel()
    if sel_manifest:
        # 修复: sel 清单注记的顺序 ≠ 全清单位置; mask 表必须与 anns 按注记对齐
        sel_pos = {int(a["ann_id"]): j for j, a in enumerate(all_anns)}
        ann_masks = np.stack([ann_masks_all[sel_pos[int(a["ann_id"])]] for a in anns])
    else:
        ann_masks = ann_masks_all[:k]
    ann_pos = np.array([a["ann_id"] for a in anns], dtype=np.int64)
    print(f"[budget] k={k} 注记 (unique masks={len(anns)}), unique sources="
          f"{len(set(a['canonical'] for a in anns))}")
    swap_map = None
    if swap_masks:
        sw = json.load(open(f"{OUT}/mask_swap_manifest.json"))
        swap_map = {int(a): int(b) for a, b in sw["swap"].items()}
        # anns 的 ann_id = 全清单位置
        n_sw = sum(1 for p in ann_pos if p in swap_map)
        print(f"[swap] 注记内可交换 {n_sw}/{k}")

    # P1-2 修复: 种子必须在模型创建之前设置 (否则命名 seed 不能复现初始化)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = MLPHead(dino_dim + (1 if use_hp else 0), hidden=hidden).cuda()
    if linear_probe:
        model = torch.nn.Linear(dino_dim + (1 if use_hp else 0), 1).cuda()
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def feat_of(idxs, mask_ok=None):
        dino = torch.from_numpy(dino_mm[idxs])
        parts = [dino]
        if use_hp:
            parts.append(torch.from_numpy(hp_mm[idxs]))
        return torch.cat(parts, dim=-1).cuda().float()

    rng = np.random.RandomState(seed)

    def predict(idxs):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idxs), batch_size):
                b = idxs[i:i + batch_size]
                logits = model(feat_of(b))
                if logits.dim() == 3:
                    logits = logits.squeeze(-1)
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, 0))

    best_val = -1
    best_state = None
    t0 = time.time()
    n_train = len(train_idx)
    U_img = int(np.ceil(n_train / batch_size))
    U_pix = U_img if k > 0 else 0
    print(f"[loader] U_img={U_img}, U_pix={U_pix} (per epoch; 实际像素样本曝光 = "
          f"{n_train if k > 0 else 0}; 末批与图像批等长, 正预算间恒定)")
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n_train)
        img_l, pix_l = [], []
        pix_unique_epoch = set()
        for b0 in range(0, n_train, batch_size):
            bi = train_idx[perm[b0:b0 + batch_size]]
            if len(bi) < batch_size and k > 0:
                # The mask batch matches the actual image batch, including the final partial batch.
                pass
            if k > 0:
                j = rng.choice(k, size=len(bi), replace=True)
                pidx = ann_rep[j]
                pm = ann_masks[j].copy()
                if swap_map is not None:
                    # 目标 = swap manifest 的全局注记位置(可能超出 k 窗口, 用全注记表)
                    tg_pos = np.array([swap_map.get(int(p), int(p)) for p in ann_pos[j]], dtype=np.int64)
                    pm = ann_masks_all[tg_pos].copy()
                X = torch.cat([feat_of(bi), feat_of(pidx)], 0)
                y_img = torch.from_numpy(labels_arr[np.concatenate([bi, pidx])]).float().cuda()
                if amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(X)
                    logits = logits.float()
                else:
                    logits = model(X)
                if logits.dim() == 3:
                    logits = logits.squeeze(-1)
                img_score = torch.sigmoid(logits).max(dim=1).values
                bce_img = F.binary_cross_entropy(img_score, y_img)
                pix_logits = logits[len(bi):]
                tgt = torch.from_numpy(pm).cuda()
                bce_pix = F.binary_cross_entropy_with_logits(pix_logits, tgt)
                dice = dice_loss(pix_logits, tgt)
                if pixel_off:
                    # P0-1 对照: 前向批与图像 BCE 完全保持不变 (仍含 mask 批), 仅移除像素损失项
                    loss = bce_img
                else:
                    loss = bce_img + bce_pix + 0.5 * dice
                if real_zero:
                    # G0' 桥接: 图像批全体像素目标(real 行全 0 = 旧 v4 目标)
                    tgt_ib = torch.from_numpy(masks[bi]).cuda()
                    bce_pix_ib = F.binary_cross_entropy_with_logits(logits[:len(bi)], tgt_ib)
                    dice_ib = dice_loss(logits[:len(bi)], tgt_ib)
                    loss = loss + bce_pix_ib + 0.5 * dice_ib
                img_l.append(float(bce_img.item())); pix_l.append(float((bce_pix + 0.5 * dice).item()))
                pix_unique_epoch.update([int(p) for p in ann_pos[j]])
            else:
                X = feat_of(bi)
                y_img = torch.from_numpy(labels_arr[bi]).float().cuda()
                if amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(X)
                    logits = logits.float()
                else:
                    logits = model(X)
                if logits.dim() == 3:
                    logits = logits.squeeze(-1)
                img_score = torch.sigmoid(logits).max(dim=1).values
                bce_img = F.binary_cross_entropy(img_score, y_img)
                loss = bce_img
                img_l.append(float(bce_img.item()))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        # ---- val: image AUC (best epoch by val image-AUC, 与旧训练器一致) ----
        sv = predict(val_idx)
        yv = labels_arr[val_idx]
        img_auc = roc_auc_score(yv, sv.reshape(len(sv), -1).max(1))
        mean_pix = float(np.mean(pix_l)) if pix_l else 0.0
        mean_img = float(np.mean(img_l))
        print(f"ep{epoch+1}: loss_img={mean_img:.4f} loss_pix={mean_pix:.4f} "
              f"pix/img={mean_pix/max(mean_img,1e-8):.3f} "
              f"pix_unique_epoch={len(pix_unique_epoch)} val_imgAUC={img_auc:.4f} "
              f"({(time.time()-t0)/60:.1f}min)", flush=True)
        if img_auc > best_val:
            best_val = img_auc
            best_state = {k2: v2.clone() for k2, v2 in model.state_dict().items()}

    print(f"\n训练完成 {(time.time()-t0)/60:.1f} 分钟; best val imgAUC={best_val:.4f} (epoch by AUC)")
    model.load_state_dict(best_state)
    suffix = f"_{tag}" if tag else ""
    torch.save(model.state_dict(), OUT / f"weak_sup_v4{suffix}_head.pt")
    print(f"模型: {OUT / f'weak_sup_v4{suffix}_head.pt'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam_img", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--no_hp", action="store_true", help="特征只用 dino (384), 不加高通残差")
    p.add_argument("--exclude_model", type=str, default=None,
                   help="leave-one-out: 排除的 inpainter (Kandinsky_2_2/StableDiffusion_v4/OpenJourney)")
    p.add_argument("--tag", type=str, default="",
                   help="保存文件名后缀 (如 lam01 / loo_kandinsky / hidden32 / nohp)")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子 (划分/shuffle/初始化); 主配置 seed=42")
    p.add_argument("--exclude_json", type=str, default=None,
                   help="评测集 json (imdl_inpx_test.json): 其样本全部排除出训练/验证")
    p.add_argument("--exclude_src_level", action="store_true",
                   help="升级为 source-disjoint: 按源图前缀排除同源图所有属性编辑")
    p.add_argument("--use_trace", action="store_true",
                   help="Content-Controlled Edit-Trace Supervision: L_trace(同源real ranking) + L_pair(std<->exc一致性)")
    p.add_argument("--alpha", type=float, default=0.1, help="L_trace 权重 (验证集上固定)")
    p.add_argument("--beta", type=float, default=0.1, help="L_pair 权重 (验证集上固定)")
    p.add_argument("--trace_margin", type=float, default=0.5, help="L_trace softplus margin")
    p.add_argument("--pair_manifest", type=str, default=None,
                   help="加载固定配对 manifest (四变体共用同一配对)")
    p.add_argument("--trace_balance", action="store_true",
                   help="trace 项按域反比加权 (缓解 CityScapes/SUN 主导)")
    p.add_argument("--cache", type=str, default=None,
                   help="特征缓存目录 (默认 features_cache; 跨 backbone 时指向其他缓存)")
    # ---- 新协议 (master split + 双 loader + mask budget; 实验计划 v4 §2.2/§2.3) ----
    p.add_argument("--master_split", type=str, default=None,
                   help="master split JSON (split_budget_master.json); 提供后忽略内部行级切分, 三 seed 共用同一划分")
    p.add_argument("--mask_manifest", type=str, default=None,
                   help="mask budget manifest (mask_budget_manifest.json)")
    p.add_argument("--mask_budget", type=str, default=None,
                   help="像素注记预算: '0' = k=0+(仅图像级), 'ALL' = 全部注记, 或整数 k")
    p.add_argument("--swap_masks", action="store_true",
                   help="mask-image correspondence control: 像素批目标 mask 按 swap manifest 置换")
    p.add_argument("--real_zero", action="store_true",
                   help="桥接消融 G0': 图像批全体像素目标(real 图全0, 复刻旧 v4 目标)")
    p.add_argument("--pixel_off", action="store_true",
                   help="P0-1 对照: 保持前向批与图像 BCE 不变(仍含 mask 批), 仅移除像素损失项")
    p.add_argument("--linear_probe", action="store_true",
                   help="ladder L1: 线性 probe (Linear(in_dim,1), 无 BN)")
    p.add_argument("--amp", action="store_true",
                   help="mixed-precision (fp16 autocast) 前向加速; 大 backbone/跨 backbone 用")
    p.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--sel_manifest", type=str, default=None,
                   help="主动选择清单 JSON(含 selected_ann_ids): 覆盖 budget 前缀, 按清单训")
    args = p.parse_args()
    if args.master_split or args.mask_budget is not None:
        train_v4_budget(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            hidden=args.hidden, lam_img=args.lam_img, seed=args.seed,
            master_split=args.master_split, mask_manifest=args.mask_manifest,
            mask_budget=args.mask_budget, swap_masks=args.swap_masks,
            real_zero=args.real_zero, linear_probe=args.linear_probe,
            amp=args.amp, sel_manifest=args.sel_manifest, wd=args.wd,
            use_hp=not args.no_hp, tag=args.tag, cache=args.cache,
            pixel_off=args.pixel_off)
    else:
        train_v4(epochs=args.epochs, batch_size=args.batch_size,
                 lr=args.lr, lam_img=args.lam_img, hidden=args.hidden,
                 use_hp=not args.no_hp,
                 exclude_model=args.exclude_model, tag=args.tag, seed=args.seed,
                 exclude_json=args.exclude_json,
                 exclude_src_level=args.exclude_src_level,
                 use_trace=args.use_trace, alpha=args.alpha, beta=args.beta,
                 trace_margin=args.trace_margin,
                 pair_manifest=args.pair_manifest, trace_balance=args.trace_balance,
                 cache=args.cache)
