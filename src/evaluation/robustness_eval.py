"""Evaluate sensitivity to image postprocessing.

流程: 取 imdl_inpx_test.json 同批样本 (500 编辑对 + 300 real) → 图像级后处理
(JPEG Q75/Q90、GaussianBlur σ=1、resize×0.5 再放大) → DINOv2 重抽特征 (免 VAE) →
v4 头评测 检测 AUC + 定位 mIoU。

用法: python src/evaluation/robustness_eval.py [--perturb jpeg75|jpeg90|blur|resize|none]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.weakly_supervised_v2 import MLPHead
from utils.highpass import cross_diff_highpass, pool_to_patch_grid

IMG_ROOT = Path("D:/lunwen/data/INP-X/inpainting_exchange")
DATA_JSON = "D:/lunwen/data/imdl_inpx_test.json"
OUT = Path("D:/lunwen/outputs")
GRID = 37
IMAGE_SIZE = 518


def load_dino():
    import torch.hub as hub
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                           trust_repo=True)
    return model.cuda().eval()


@torch.no_grad()
def extract(img, model, n_patches=1369):
    """返回 [N, 385] = dino tokens (384) + highpass (1), 与 features_cache 管线一致"""
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda()
    t = t / 255.0
    t = F.interpolate(t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                      align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    t = (t - mean) / std
    tok = model.forward_features(t)["x_norm_patchtokens"]  # [1, N, 384]
    hp = cross_diff_highpass(t)                             # [1, 1, H-1, W-1]
    hp = F.pad(hp, (0, 1, 0, 1))                            # 恢复原尺寸
    hp_pooled = pool_to_patch_grid(hp, n_patches)           # [N, 1]
    feats = np.concatenate([tok.squeeze(0).float().cpu().numpy(),
                            hp_pooled.squeeze(0).cpu().numpy()], axis=-1)
    return feats.astype(np.float32)


def perturb(img: np.ndarray, kind: str) -> np.ndarray:
    """img: HWC uint8 RGB"""
    pil = Image.fromarray(img)
    if kind == "jpeg75":
        buf = np.frombuffer(_jpeg_bytes(pil, 75), dtype=np.uint8)
        return np.asarray(Image.open(_bytesio(buf)).convert("RGB"))
    if kind == "jpeg90":
        buf = np.frombuffer(_jpeg_bytes(pil, 90), dtype=np.uint8)
        return np.asarray(Image.open(_bytesio(buf)).convert("RGB"))
    if kind == "blur":
        return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=1.0)).convert("RGB"))
    if kind == "resize":
        small = pil.resize((IMAGE_SIZE // 2, IMAGE_SIZE // 2), Image.LANCZOS)
        return np.asarray(small.resize((pil.width, pil.height), Image.LANCZOS).convert("RGB"))
    return img


def _jpeg_bytes(pil_img, q):
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def _bytesio(b):
    import io
    return io.BytesIO(b.tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--perturb", default="none",
                   choices=["none", "jpeg75", "jpeg90", "blur", "resize"])
    p.add_argument("--head", default="weak_sup_v4_head.pt")
    args = p.parse_args()

    records = json.load(open(DATA_JSON))
    # 编辑对 + real
    items = []
    for img_path, mask_path in records:
        items.append((Path(img_path), None if mask_path == "Negative" else Path(IMG_ROOT / mask_path)))

    model = load_dino()
    head = MLPHead(385).cuda()
    head.load_state_dict(torch.load(OUT / args.head))
    head.eval()

    results = []
    t0 = time.time()
    with torch.no_grad():
        for img_path, mask_path in items:
            img = np.asarray(Image.open(img_path).convert("RGB"))
            img_p = perturb(img, args.perturb)
            tok = extract(img_p, model)
            X = torch.from_numpy(tok).unsqueeze(0).cuda()
            logits = head(X).squeeze(0)
            scores = torch.sigmoid(logits).cpu().numpy().reshape(GRID, GRID)
            img_score = float(scores.max())

            is_fake = mask_path is not None
            if is_fake:
                m = Image.open(mask_path).convert("L").resize((GRID, GRID), Image.NEAREST)
                gt = (np.asarray(m) > 127).astype(np.float32)
                best_iou = 0.0
                for th in np.linspace(0.05, 0.95, 91):
                    pr = (scores >= th).astype(np.float32)
                    tp = (pr * gt).sum(); fp = (pr * (1 - gt)).sum(); fn = ((1 - pr) * gt).sum()
                    best_iou = max(best_iou, tp / (tp + fp + fn + 1e-8))
            else:
                gt = np.zeros((GRID, GRID))
                best_iou = 0.0
            results.append({"fake": is_fake, "score": img_score, "mIoU": best_iou})

    fake = np.array([r["fake"] for r in results])
    scores = np.array([r["score"] for r in results])
    auc = roc_auc_score(fake.astype(int), scores)
    miou = np.mean([r["mIoU"] for r in results if r["fake"]])
    print(f"perturb={args.perturb}: 检测AUC={auc:.4f} 定位mIoU(仅fake)={miou:.4f} "
          f"n={len(results)} 耗时{(time.time()-t0)/60:.1f}min")

    with open(OUT / f"robust_{args.perturb}.json", "w") as f:
        json.dump({"perturb": args.perturb, "auc": float(auc),
                   "loc_miou": float(miou),
                   "head": args.head,
                   "protocol": "det: 500 edited + 300 real; loc: 500 edited only, "
                               "best-threshold mean per-image IoU (37x37)",
                   "n": int(len(results))}, f, indent=1)


if __name__ == "__main__":
    main()
