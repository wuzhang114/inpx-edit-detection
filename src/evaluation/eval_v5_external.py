"""v5 双分支头外部评测: SDXL / MagicBrush / CocoGlide (同上协议: INP-X val 阈值)。

- local 分支 → 定位 (fixed thr 来自 eval500_v5_{tag}.json 的 val_thr)
- global 分支 → 检测 (fake vs source)
- 特征: DINOv2 ViT-S/14 一次提取, 复用给所有 head
输出: outputs/v5_external_{tag}.json
"""
import argparse
import json
import sys
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines.weakly_supervised_v5 import DualBranchHead  # noqa: E402
from utils.highpass import cross_diff_highpass, pool_to_patch_grid  # noqa: E402

SIZE = 518
GRID = 37
N_PATCHES = GRID * GRID
OUT = Path("D:/lunwen/outputs")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB").resize(
        (SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0


def read_mask(path):
    return (np.asarray(Image.open(path).convert("L").resize(
        (GRID, GRID), Image.NEAREST)) > 127).astype(np.float32)


def samples_for(kind):
    root = Path("D:/lunwen/data")
    if kind == "sdxl":
        base = root / "sdxl_edits_500"
        rows = []
        for d in sorted(base.glob("sdxl_*")):
            if (d / "edit.jpg").exists() and (d / "src.jpg").exists() and (d / "mask.png").exists():
                rows.append((d / "edit.jpg", d / "src.jpg", d / "mask.png"))
        return rows
    if kind == "cocoglide":
        base = root / "cocoglide_400" / "assets"
        rows = []
        for fake in sorted(base.glob("*_fake.png")):
            sid = fake.name[:-9]
            source, mask = base / f"{sid}_real.png", base / f"{sid}_mask.png"
            if source.exists() and mask.exists():
                rows.append((fake, source, mask))
        return rows
    if kind == "magicbrush":
        base = root / "magicbrush_eval"
        rows = []
        for fake in sorted((base / "images").glob("*.png")):
            source, mask = base / "sources" / fake.name, base / "masks" / fake.name
            if mask.exists():
                rows.append((fake, source if source.exists() else None, mask))
        return rows
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head_tags", required=True, help="逗号分隔 v5 tags")
    ap.add_argument("--datasets", default="sdxl,magicbrush,cocoglide")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    infos = []
    for tag in [x.strip() for x in args.head_tags.split(",") if x.strip()]:
        meta = json.load(open(OUT / f"weak_sup_v5_{tag}_meta.json", encoding="utf-8"))
        cfg = meta["head_config"]
        ev = json.load(open(OUT / f"eval500_v5_{tag}.json", encoding="utf-8"))
        head = DualBranchHead(cfg["in_dim"], local_hidden=cfg["local_hidden"],
                              global_hidden=cfg["global_hidden"]).cuda()
        head.load_state_dict(torch.load(OUT / f"weak_sup_v5_{tag}_head.pt", map_location="cpu"))
        head.eval()
        infos.append({"tag": tag, "model": head, "thr": float(ev["val_thr"]),
                      "fixed_miou_std": ev["standard"]["fixed_miou"]})

    # 本地加载 DINOv2 (hub 缓存已有; 离线环境不走 torch.hub.load 的网络验证)
    import torch.hub as hub
    hub_dir = Path(hub.get_dir())
    repo_dir = hub_dir / "facebookresearch_dinov2_main"
    if not repo_dir.exists():
        repo_dir = hub_dir / "facebookresearch_dinov2_7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    sys.path.insert(0, str(repo_dir))
    from dinov2.hub.backbones import dinov2_vits14
    backbone = dinov2_vits14(pretrained=False).cuda().eval()
    ckpt = hub_dir / "checkpoints" / "dinov2_vits14_pretrain.pth"
    sd = torch.load(ckpt, map_location="cpu")
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    print(f"backbone loaded (missing={len(missing)} unexpected={len(unexpected)})", flush=True)

    def extract(paths):
        chunks = []
        with torch.no_grad():
            for start in range(0, len(paths), args.batch):
                imgs = np.stack([read_rgb(p) for p in paths[start:start + args.batch]])
                x = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().cuda()
                x = (x - MEAN.cuda()) / STD.cuda()
                tok = backbone.forward_features(x)["x_norm_patchtokens"]
                hp = pool_to_patch_grid(cross_diff_highpass(x), N_PATCHES)
                chunks.append(torch.cat([tok, hp.to(tok.dtype)], dim=-1).float().cpu())
        return torch.cat(chunks, dim=0)

    result = {"protocol": "INP-X-trained v5 dual-branch head, INP-X val threshold",
              "backbone": "DINOv2 ViT-S/14 frozen", "datasets": {}}
    for kind in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        rows = samples_for(kind)
        loc_rows = [r for r in rows if r[0] is not None and r[2] is not None]
        det_rows = [r for r in rows if r[0] is not None and r[1] is not None]
        all_paths, seen = [], set()
        for fake, source, _ in rows:
            for path in (fake, source):
                if path is not None and str(path) not in seen:
                    all_paths.append(path); seen.add(str(path))
        print(f"{kind}: loc={len(loc_rows)} det={len(det_rows)}", flush=True)
        feats = extract(all_paths)
        path_index = {str(p): i for i, p in enumerate(all_paths)}
        masks = np.stack([read_mask(r[2]) for r in loc_rows])
        data = {}
        for info in infos:
            model = info["model"]
            det_all, loc_all = [], []
            with torch.no_grad():
                for start in range(0, len(feats), args.batch):
                    out = model(feats[start:start + args.batch].cuda())
                    det_all.append(torch.sigmoid(out["global"]).cpu().numpy())
                    loc_all.append(torch.sigmoid(out["local"]).cpu().numpy())
            det_all = np.concatenate(det_all)
            loc_all = np.concatenate(loc_all)
            fake_det = np.array([det_all[path_index[str(r[0])]] for r in det_rows])
            src_det = np.array([det_all[path_index[str(r[1])]] for r in det_rows])
            fake_loc = loc_all[[path_index[str(r[0])] for r in loc_rows]]
            pred = (fake_loc >= info["thr"]).astype(np.float32)
            m_flat = masks.reshape(len(masks), -1)
            tp = (pred * m_flat).sum(); fp = (pred * (1 - m_flat)).sum()
            fn = ((1 - pred) * m_flat).sum()
            miou = float(tp / (tp + fp + fn + 1e-8))
            det_auc = float(roc_auc_score(np.r_[np.zeros(len(fake_det)), np.ones(len(fake_det))],
                                          np.r_[src_det, fake_det]))
            flat, flat_m = fake_loc.ravel(), masks.ravel()
            row = {
                "n_loc": int(len(loc_rows)), "n_det": int(len(det_rows)),
                "thr": info["thr"], "fixed_loc_miou": miou,
                "det_auc": det_auc,
                "pixel_auroc": float(roc_auc_score(flat_m, flat)),
                "pixel_ap": float(average_precision_score(flat_m, flat)),
                "fixed_miou_inpx_std": info["fixed_miou_std"],
            }
            data[info["tag"]] = row
            print(f"  {info['tag']}: det={det_auc:.4f} mIoU={miou:.4f} "
                  f"AUROC={row['pixel_auroc']:.4f} (INP-X std {info['fixed_miou_std']:.4f})",
                  flush=True)
        result["datasets"][kind] = data
        del feats
        torch.cuda.empty_cache()
    with open(OUT / "v5_external_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("saved:", OUT / "v5_external_result.json", flush=True)


if __name__ == "__main__":
    main()
