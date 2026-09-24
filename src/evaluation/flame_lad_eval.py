"""FLAME LAD-only evaluation on 500 edited and 300 authentic images.

协议与主表一致: 检测 AUC (detection_logit, real vs edit) + 定位 mIoU
(coarse_mask 阈值 91 档 best-thr, 与 v4 同协议)。
说明: 只用 FerretBackbone (LAD 主干) 粗定位, 未含 SAM2 边界精修 — 口径在
对比表注明; FLAME 论文完整数字含 SAM 精修, 略高。
用法: python tmp/flame_lad_eval.py  (在 D:/lunwen 下运行)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

FLAME_ROOT = Path("D:/lunwen/tmp/flame_extract/FLAME-main/FLAME")
sys.path.insert(0, str(FLAME_ROOT))

from model.ferret_backbone import FerretBackbone  # noqa: E402

DATA_JSON = "D:/lunwen/data/imdl_inpx_test.json"
CKPT = "D:/lunwen/FLAME_checkpoints"
GRID = 37
IMG = 512


def main():
    cfg = json.load(open("D:/lunwen/tmp/model_params.json"))["model_config"]
    sd = torch.load(CKPT, map_location="cpu")["model"]
    fb_sd = {k.replace("ferret_backbone.", ""): v for k, v in sd.items()
             if k.startswith("ferret_backbone.")}
    print(f"ferret_backbone keys: {len(fb_sd)}", flush=True)

    model = FerretBackbone(
        dim=96,
        lad_tau=cfg["lad_tau"],
        lad_multi_taus=cfg["lad_multi_taus"],
        forensic_operator=cfg["forensic_operator"],
        coarse_prompt_head=cfg["coarse_prompt_head"],
        coarse_prompt_hidden=cfg["coarse_prompt_hidden"],
        coarse_prompt_dropout=cfg["coarse_prompt_dropout"],
        coarse_prompt_gate_init=cfg["coarse_prompt_gate_init"],
        coarse_prompt_gate_max=cfg["coarse_prompt_gate_max"],
        coarse_prompt_area_bias=cfg["coarse_prompt_area_bias"],
        coarse_prompt_signed_residual_max_delta=cfg["coarse_prompt_signed_residual_max_delta"],
        coarse_prompt_unet_gate_init=cfg["coarse_prompt_unet_gate_init"],
        coarse_prompt_unet_gate_max=cfg["coarse_prompt_unet_gate_max"],
        coarse_prompt_unet_signed_residual_max_delta=cfg["coarse_prompt_unet_signed_residual_max_delta"],
        mask_compressor_kernel_size=cfg["mask_compressor_kernel_size"],
        mask_compressor_output=cfg["mask_compressor_output"],
        legacy_logit_head=cfg["legacy_logit_head"],
    ).cuda().eval()
    missing, unexpected = model.load_state_dict(fb_sd, strict=False)
    print(f"load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("missing sample:", list(missing)[:5], flush=True)
    model.half()

    records = json.load(open(DATA_JSON))
    fake_scores, fake_masks = [], []
    real_scores = []
    with torch.no_grad():
        for img_path, mask_path in records:
            img = Image.open(img_path).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
            t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).unsqueeze(0).float().cuda()
            t = t / 255.0
            with torch.autocast("cuda", dtype=torch.float16):
                _, coarse_mask, det_logit = model(t)
            cm = torch.sigmoid(coarse_mask).squeeze(0).squeeze(0)  # [256,256]
            cm = F.interpolate(cm.unsqueeze(0).unsqueeze(0), size=(GRID, GRID),
                               mode="bilinear", align_corners=False).squeeze().cpu().numpy()
            det = float(torch.sigmoid(det_logit).squeeze().cpu())
            if mask_path == "Negative":
                real_scores.append(det)
            else:
                fake_scores.append(cm)
                m = Image.open(mask_path).convert("L").resize((GRID, GRID), Image.NEAREST)
                fake_masks.append((np.asarray(m) > 127).astype(np.float32))
            if len(fake_scores) + len(real_scores) % 100 == 0:
                print(f"{len(fake_scores)}/{len(real_scores)}", flush=True)

    y = np.concatenate([np.zeros(len(real_scores)), np.ones(len(fake_scores))])
    img_scores = np.concatenate([np.array(real_scores),
                                 np.array([s.max() for s in fake_scores])])
    auc = roc_auc_score(y, img_scores)
    print(f"FLAME LAD 检测 AUC = {auc:.4f} (n_real={len(real_scores)}, n_fake={len(fake_scores)})")

    fs = np.stack(fake_scores)
    fm = np.stack(fake_masks)
    best_miou, best_f1 = 0.0, 0.0
    for th in np.linspace(0.05, 0.95, 91):
        pred = (fs >= th).astype(np.float32)
        tp = (pred * fm).sum(); fp = (pred * (1 - fm)).sum(); fn = ((1 - pred) * fm).sum()
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        best_miou = max(best_miou, tp / (tp + fp + fn + 1e-8))
        best_f1 = max(best_f1, 2 * prec * rec / (prec + rec + 1e-8))
    print(f"FLAME LAD 定位 mIoU = {best_miou:.4f} F1 = {best_f1:.4f} (best-thr, 同主协议)")

    res = {"method": "FLAME-LAD-only", "img_auc": float(auc),
           "loc_miou": float(best_miou), "loc_f1": float(best_f1),
           "n_real": len(real_scores), "n_fake": len(fake_scores),
           "note": "LAD 主干粗定位, 无 SAM2 精修; 检测用 detection_logit"}
    with open("D:/lunwen/outputs/flame_lad_result.json", "w") as f:
        json.dump(res, f, indent=1)
    print("saved -> outputs/flame_lad_result.json")


if __name__ == "__main__":
    main()
