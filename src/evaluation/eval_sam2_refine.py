"""v4 粗定位 + SAM2 精修 (官方 SAM2.1 base+ 基座, zero-shot, 无 FLAME adapter)

协议: 与主表一致 (500 集, best-thr mIoU/F1)。粗 prompt = v4 分数阈值 0.5。
用法: python src/evaluation/eval_sam2_refine.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "D:/lunwen/src")
from baselines.weakly_supervised_v2 import MLPHead
from utils.highpass import cross_diff_highpass, pool_to_patch_grid

import hydra
from hydra import initialize_config_dir
hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize_config_dir(config_dir="D:/lunwen/src/evaluation/sam2configs", version_base=None)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DATA_JSON = "D:/lunwen/data/imdl_inpx_test.json"
GRID = 37
SIZE = 518
CKPT = "D:/lunwen/data/sam2.1_hiera_base_plus.pt"
CFG = "sam2.1_hiera_b+"

import torch.hub as hub
dino = hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True).cuda().eval()
head = MLPHead(385).cuda()
head.load_state_dict(torch.load("D:/lunwen/outputs/weak_sup_v4_head.pt", map_location="cpu"))
head.eval()

predictor = SAM2ImagePredictor(build_sam2(CFG, CKPT, device="cuda"))


def v4_scores(path):
    img = np.asarray(Image.open(path).convert("RGB"))
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    t = (t - mean) / std
    with torch.no_grad():
        tok = dino.forward_features(t)["x_norm_patchtokens"]
        hp = F.pad(cross_diff_highpass(t), (0, 1, 0, 1))
        hp_p = pool_to_patch_grid(hp, GRID * GRID)
        X = torch.cat([tok, hp_p], dim=-1).cuda()
        logits = head(X).squeeze(0)
    return torch.sigmoid(logits).cpu().numpy().reshape(GRID, GRID)


def sam2_refine(img_rgb, coarse37):
    predictor.set_image(img_rgb)
    coarse = (coarse37 >= 0.5).astype(np.float32)
    coarse_up = np.asarray(Image.fromarray(coarse).resize((256, 256), Image.BILINEAR))
    masks, _, _ = predictor.predict(mask_input=coarse_up[None], multimask_output=False)
    m = masks[0, 0]  # 1024x1024 prob
    m37 = np.asarray(Image.fromarray(m).resize((GRID, GRID), Image.BILINEAR))
    return m37


records = json.load(open(DATA_JSON))
miou_sum = f1_sum = n = 0
with torch.no_grad():
    for i, (img_path, mask_path) in enumerate(records):
        if mask_path == "Negative":
            continue
        img_rgb = np.asarray(Image.open(img_path).convert("RGB"))
        smap = v4_scores(img_path)
        refined = sam2_refine(img_rgb, smap)
        m = Image.open(mask_path).convert("L").resize((GRID, GRID), Image.NEAREST)
        gt = (np.asarray(m) > 127).astype(np.float32)
        best_iou = best_f1 = 0.0
        for th in np.linspace(0.05, 0.95, 91):
            pr = (refined >= th).astype(np.float32)
            tp = (pr * gt).sum(); fp = (pr * (1 - gt)).sum(); fn = ((1 - pr) * gt).sum()
            prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
            best_iou = max(best_iou, tp / (tp + fp + fn + 1e-8))
            best_f1 = max(best_f1, 2 * prec * rec / (prec + rec + 1e-8))
        miou_sum += best_iou; f1_sum += best_f1; n += 1
        if n % 100 == 0:
            print(f"  {n}/500", flush=True)

res = {"method": "v4 + SAM2 refine (official SAM2.1 base+, zero-shot)", "n": n,
       "loc_miou_best": float(miou_sum / n), "loc_f1_best": float(f1_sum / n)}
print(json.dumps(res, indent=1))
with open("D:/lunwen/outputs/v4_sam2_refine_result.json", "w") as f:
    json.dump(res, f, indent=1)

