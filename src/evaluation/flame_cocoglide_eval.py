"""FLAME-LAD 在 CocoGlide 400 上的评测 (与 v4 CocoGlide 评测同指标同协议)"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

FLAME_ROOT = Path("D:/lunwen/src/evaluation/flame_src/FLAME-main/FLAME")
if not FLAME_ROOT.exists():
    FLAME_ROOT = Path("D:/lunwen/trail")  # fallback 不存在则报错
sys.path.insert(0, str(FLAME_ROOT))
from model.ferret_backbone import FerretBackbone  # noqa: E402

ASSETS = Path("D:/lunwen/data/cocoglide_400/assets")
CKPT = "D:/lunwen/src/evaluation/flame_g2_ladmulti_sam2.pth"
CFG = "D:/lunwen/src/evaluation/flame_model_params.json"
GRID = 37
NATIVE = 448
IMG = 512


def main():
    cfg = json.load(open(CFG))["model_config"]
    sd = torch.load(CKPT, map_location="cpu")["model"]
    fb_sd = {k.replace("ferret_backbone.", ""): v for k, v in sd.items()
             if k.startswith("ferret_backbone.")}
    model = FerretBackbone(
        dim=96, lad_tau=cfg["lad_tau"], lad_multi_taus=cfg["lad_multi_taus"],
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
    model.half()

    ids = sorted({p.name.split("_")[0] + "_" + p.name.split("_")[1] for p in ASSETS.glob("*_fake.png")})
    auroc_sum = miou_sum = n = 0
    with torch.no_grad():
        for i, sid in enumerate(ids):
            img = Image.open(ASSETS / f"{sid}_fake.png").convert("RGB").resize((IMG, IMG), Image.BILINEAR)
            t = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
            with torch.autocast("cuda", dtype=torch.float16):
                _, coarse_mask, _ = model(t)
            cm = torch.sigmoid(coarse_mask).squeeze(0).squeeze(0)
            cm = F.interpolate(cm.unsqueeze(0).unsqueeze(0), size=(GRID, GRID),
                               mode="bilinear", align_corners=False).squeeze().cpu().numpy()
            mask = np.asarray(Image.open(ASSETS / f"{sid}_mask.png").convert("L")) > 127
            smap_up = np.asarray(Image.fromarray(cm).resize((NATIVE, NATIVE), Image.BILINEAR))
            auroc_sum += roc_auc_score(mask.ravel().astype(int), smap_up.ravel())
            m37 = (np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((GRID, GRID), Image.NEAREST)) > 127).astype(np.float32)
            best = 0.0
            for th in np.linspace(0.05, 0.95, 91):
                pr = (cm >= th).astype(np.float32)
                tp = (pr * m37).sum(); fp = (pr * (1 - m37)).sum(); fn = ((1 - pr) * m37).sum()
                best = max(best, tp / (tp + fp + fn + 1e-8))
            miou_sum += best
            n += 1
            if i % 100 == 99:
                print(f"  {i+1}/{len(ids)}", flush=True)

    res = {"method": "FLAME-LAD (official weights, no SAM2)", "dataset": "CocoGlide 400", "n": n,
           "patch_auroc": float(auroc_sum / n), "loc_miou_best37": float(miou_sum / n)}
    print(json.dumps(res, indent=1))
    with open("D:/lunwen/outputs/flame_cocoglide_result.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

