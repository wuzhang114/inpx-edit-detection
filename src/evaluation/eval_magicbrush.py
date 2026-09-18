"""MagicBrush dev 评测: v4 / FLAME-LAD 同协议 (patch AUROC/AP + mIoU + det AUC)
统一协议: INP-X-only 训练 → INP-X validation threshold (--thr) → zero-shot
用法: python src/evaluation/eval_magicbrush.py --method v4|flame [--head weak_sup_v4_head.pt] [--thr 0.45]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, "D:/lunwen/src")
from baselines.weakly_supervised_v2 import MLPHead
from utils.highpass import cross_diff_highpass, pool_to_patch_grid

DATA = Path("D:/lunwen/data/magicbrush_eval")
GRID = 37
SIZE = 518


def load_v4(head_name):
    import torch.hub as hub
    model = hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).cuda().eval()
    head = MLPHead(385).cuda()
    head.load_state_dict(torch.load(f"D:/lunwen/outputs/{head_name}", map_location="cpu"))
    head.eval()
    return model, head


def load_flame():
    FLAME_ROOT = Path("D:/lunwen/src/evaluation/flame_src/FLAME-main/FLAME")
    sys.path.insert(0, str(FLAME_ROOT))
    from model.ferret_backbone import FerretBackbone
    cfg = json.load(open("D:/lunwen/src/evaluation/flame_model_params.json"))["model_config"]
    sd = torch.load("D:/lunwen/src/evaluation/flame_g2_ladmulti_sam2.pth", map_location="cpu")["model"]
    fb_sd = {k.replace("ferret_backbone.", ""): v for k, v in sd.items() if k.startswith("ferret_backbone.")}
    model = FerretBackbone(dim=96, lad_tau=cfg["lad_tau"], lad_multi_taus=cfg["lad_multi_taus"],
        forensic_operator=cfg["forensic_operator"], coarse_prompt_head=cfg["coarse_prompt_head"],
        coarse_prompt_hidden=cfg["coarse_prompt_hidden"], coarse_prompt_dropout=cfg["coarse_prompt_dropout"],
        coarse_prompt_gate_init=cfg["coarse_prompt_gate_init"], coarse_prompt_gate_max=cfg["coarse_prompt_gate_max"],
        coarse_prompt_area_bias=cfg["coarse_prompt_area_bias"],
        coarse_prompt_signed_residual_max_delta=cfg["coarse_prompt_signed_residual_max_delta"],
        coarse_prompt_unet_gate_init=cfg["coarse_prompt_unet_gate_init"],
        coarse_prompt_unet_gate_max=cfg["coarse_prompt_unet_gate_max"],
        coarse_prompt_unet_signed_residual_max_delta=cfg["coarse_prompt_unet_signed_residual_max_delta"],
        mask_compressor_kernel_size=cfg["mask_compressor_kernel_size"],
        mask_compressor_output=cfg["mask_compressor_output"], legacy_logit_head=cfg["legacy_logit_head"],
    ).cuda().eval()
    missing, unexpected = model.load_state_dict(fb_sd, strict=False)
    assert not missing, f"missing {len(missing)}"
    model.half()
    return model


def scores_v4(model, head, path):
    img = np.asarray(Image.open(path).convert("RGB"))
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    t = (t - mean) / std
    with torch.no_grad():
        tok = model.forward_features(t)["x_norm_patchtokens"]
        hp = F.pad(cross_diff_highpass(t), (0, 1, 0, 1))
        hp_p = pool_to_patch_grid(hp, GRID * GRID)
        X = torch.cat([tok, hp_p], dim=-1).cuda()
        logits = head(X).squeeze(0)
    return torch.sigmoid(logits).cpu().numpy().reshape(GRID, GRID)


def scores_flame(model, path):
    img = Image.open(path).convert("RGB").resize((512, 512), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            _, coarse_mask, _ = model(t)
        cm = torch.sigmoid(coarse_mask).squeeze(0).squeeze(0)
        cm = F.interpolate(cm.unsqueeze(0).unsqueeze(0), size=(GRID, GRID),
                           mode="bilinear", align_corners=False).squeeze().cpu().numpy()
    return cm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["v4", "flame"], required=True)
    p.add_argument("--head", default="weak_sup_v4_head.pt", help="--method v4 时使用的 head")
    p.add_argument("--thr", type=float, default=None,
                   help="INP-X 验证集固定阈值 (统一协议); 不传则报 best-thr")
    args = p.parse_args()

    ids = sorted({f.stem for f in (DATA / "images").glob("*.png")})
    if args.method == "v4":
        model, head = load_v4(args.head)
    else:
        model = load_flame()

    fake_img_scores, real_img_scores = [], []
    auroc_sum = ap_sum = miou_sum = n = 0
    smap_cache, m37_cache = [], []
    for i, sid in enumerate(ids):
        smap = scores_v4(model, head, DATA / "images" / f"{sid}.png") if args.method == "v4" else scores_flame(model, DATA / "images" / f"{sid}.png")
        mask = np.asarray(Image.open(DATA / "masks" / f"{sid}.png").convert("L")) > 127
        h, w = mask.shape
        smap_up = np.asarray(Image.fromarray(smap).resize((w, h), Image.BILINEAR))
        flat_m = mask.ravel().astype(int)
        auroc_sum += roc_auc_score(flat_m, smap_up.ravel())
        ap_sum += average_precision_score(flat_m, smap_up.ravel())
        m37 = (np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((GRID, GRID), Image.NEAREST)) > 127).astype(np.float32)
        smap_cache.append(smap); m37_cache.append(m37)
        best = 0.0
        for th in np.linspace(0.05, 0.95, 91):
            pr = (smap >= th).astype(np.float32)
            tp = (pr * m37).sum(); fp = (pr * (1 - m37)).sum(); fn = ((1 - pr) * m37).sum()
            best = max(best, tp / (tp + fp + fn + 1e-8))
        miou_sum += best
        fake_img_scores.append(float(smap.max()))
        n += 1
        if i % 100 == 99:
            print(f"  {i+1}/{len(ids)}", flush=True)

    # 检测: 前 300 张 source vs 对应 edit
    src_ids = sorted({f.stem for f in (DATA / "sources").glob("*.png")})
    for sid in src_ids:
        smap = scores_v4(model, head, DATA / "sources" / f"{sid}.png") if args.method == "v4" else scores_flame(model, DATA / "sources" / f"{sid}.png")
        real_img_scores.append(float(smap.max()))
    y = np.concatenate([np.zeros(len(real_img_scores)), np.ones(len(src_ids))])
    img_scores = np.concatenate([real_img_scores, fake_img_scores[:len(src_ids)]])
    det_auc = roc_auc_score(y, img_scores)

    res = {"method": args.method, "dataset": "MagicBrush dev", "n": n,
           "patch_auroc": float(auroc_sum / n), "patch_ap": float(ap_sum / n),
           "loc_miou_best37": float(miou_sum / n),
           "det_auc": float(det_auc), "det_n": int(len(src_ids)), "head": args.head}
    if args.thr is not None:
        smap_all = np.concatenate([s.ravel() for s in smap_cache])
        m37_all = np.concatenate([m.ravel() for m in m37_cache])
        pr = (smap_all >= args.thr).astype(np.float32)
        tp = (pr * m37_all).sum(); fp = (pr * (1 - m37_all)).sum(); fn = ((1 - pr) * m37_all).sum()
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        res["fixed_thr"] = args.thr
        res["fixed_loc_miou"] = float(tp / (tp + fp + fn + 1e-8))
        res["fixed_loc_f1"] = float(2 * prec * rec / (prec + rec + 1e-8))
        print(f"[统一协议 thr={args.thr}] mIoU={res['fixed_loc_miou']:.4f} F1={res['fixed_loc_f1']:.4f}")
    print(json.dumps(res, indent=1))
    tag = args.head.replace("_head.pt", "") if args.method == "v4" else "flame"
    thr_tag = f"thr{args.thr}" if args.thr is not None else "bestthr"
    with open(f"D:/lunwen/outputs/magicbrush_{tag}_{thr_tag}_result.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
