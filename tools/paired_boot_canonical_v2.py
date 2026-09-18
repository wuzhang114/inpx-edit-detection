"""Final fixed-checkpoint canonical cluster bootstrap, no training.

Sorted tie groups allow exact AUROC computation from cluster multiplicities.
Intervals condition on the checkpoints and exclude training-seed uncertainty.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.canonical_source import canonical_source_id
from baselines.weakly_supervised_v4 import MLPHead
OUT = ROOT / "outputs"
CACHE = ROOT / "data/features_cache"
IMG_ROOT = ROOT / "data/INP-X/inpainting_exchange"
N_BOOT = 2000
TAGS = ["b0_s42", "g2u_s42", "po_bALL_s42"]


def auc_prepared(y, score):
    y, score = np.asarray(y).ravel(), np.asarray(score).ravel()
    order = np.argsort(score, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(score[order])) + 1]
    ys = y[order]
    def evaluate(weights):
        w = np.asarray(weights, dtype=np.float64).ravel()[order]
        pos = np.add.reduceat(w * ys, starts)
        neg = np.add.reduceat(w * (1 - ys), starts)
        denom = pos.sum() * neg.sum()
        if denom == 0:
            return np.nan
        return float(np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / denom)
    return evaluate


def bootstrap_pair(y, a, b, sources, pixels=False, binary=None):
    _, inverse = np.unique(sources, return_inverse=True)
    nc = int(inverse.max() + 1)
    rng = np.random.RandomState(20260918)
    aa, bb = auc_prepared(y, a), auc_prepared(y, b)
    for weights in [np.ones(len(sources)), rng.randint(1, 4, nc)[inverse]]:
        ww = np.repeat(weights, y.shape[1]) if pixels else weights
        for scorer, score in [(aa, a), (bb, b)]:
            np.testing.assert_allclose(scorer(ww), roc_auc_score(y.ravel(), score.ravel(), sample_weight=ww), atol=1e-12)
    auc_diffs, iou_diffs = [], []
    if binary is not None:
        gt = y.astype(bool)
        stats = [(np.logical_and(pr, gt).sum(1), np.logical_or(pr, gt).sum(1)) for pr in binary]
    for it in range(N_BOOT):
        mult = rng.multinomial(nc, np.full(nc, 1 / nc))[inverse]
        ww = np.repeat(mult, y.shape[1]) if pixels else mult
        auc_diffs.append(aa(ww) - bb(ww))
        if binary is not None:
            vals = [np.dot(mult, inter) / np.dot(mult, union) for inter, union in stats]
            iou_diffs.append(vals[0] - vals[1])
        if (it + 1) % 500 == 0:
            print(f"bootstrap {it+1}/{N_BOOT}, clusters={nc}, pixels={pixels}", flush=True)
    result = {"n_clusters": nc, "n_boot": N_BOOT,
              "delta_auc": float(aa(np.ones(y.size)) - bb(np.ones(y.size))),
              "ci95_auc": np.nanpercentile(auc_diffs, [2.5, 97.5]).tolist()}
    if iou_diffs:
        result["ci95_iou"] = np.percentile(iou_diffs, [2.5, 97.5]).tolist()
    return result


def main():
    labels = json.loads((CACHE / "labels.json").read_text(encoding="utf-8"))
    meta = json.loads((CACHE / "metadata.json").read_text())
    lookup = {r["path"].replace("\\", "/"): i for i, r in enumerate(labels)}
    assert len(lookup) == len(labels), "Duplicate cache paths"
    records = json.loads((ROOT / "data/imdl_inpx_test.json").read_text())
    records = [r for r in records if r[1] != "Negative"] + [r for r in records if r[1] == "Negative"]
    rels = [str(Path(p).relative_to(IMG_ROOT)).replace("\\", "/") for p, _ in records]
    idx = np.array([lookup[p] for p in rels])
    assert len(set(idx)) == 800
    src = np.array([canonical_source_id(labels[i]["path"], labels[i]["cat"]) for i in idx])
    y = np.array([m != "Negative" for _, m in records], dtype=int)
    ne = int(y.sum())
    assert ne == 500 and len(set(src)) == 783 and len(set(src[:ne])) == 487
    print("Exact paths: 800; canonical clusters detection=783, localization=487", flush=True)
    masks = np.stack([np.asarray(Image.open(IMG_ROOT / m).convert("L").resize((37, 37), Image.Resampling.NEAREST)).ravel() > 127 for _, m in records[:ne]])
    n, patches, dim = (meta[k] for k in ["n_total", "n_patches", "dino_dim"])
    dino = np.memmap(CACHE / "dino_tokens.npy", dtype=np.float16, mode="r", shape=(n, patches, dim))
    hp = np.memmap(CACHE / "highpass.npy", dtype=np.float32, mode="r", shape=(n, patches, 1))
    all_scores, points, provenance = {}, {}, {}
    for tag in TAGS:
        path = OUT / f"weak_sup_v4_{tag}_head.pt"
        sd = torch.load(path, map_location="cpu", weights_only=True)
        in_dim = len(sd["net.0.weight"])
        model = MLPHead(in_dim, hidden=sd["net.1.weight"].shape[0]).cuda().eval()
        model.load_state_dict(sd)
        batches = []
        with torch.no_grad():
            for start in range(0, len(idx), 64):
                ix = idx[start:start+64]
                feats = np.asarray(dino[ix], dtype=np.float32)
                if in_dim == dim + 1:
                    feats = np.concatenate([feats, hp[ix]], axis=-1)
                batches.append(torch.sigmoid(model(torch.from_numpy(feats).cuda())).cpu().numpy().reshape(len(ix), -1))
        sc = np.concatenate(batches)
        ev = json.loads((OUT / f"eval500_eval_{tag}.json").read_text())
        thr = ev["val_thr"]
        pred = sc[:ne] >= thr
        point = {"det_auc": float(roc_auc_score(y, sc.max(1))),
                 "pixel_auroc": float(roc_auc_score(masks.ravel(), sc[:ne].ravel())),
                 "micro_iou": float(np.logical_and(pred, masks).sum() / np.logical_or(pred, masks).sum()), "val_thr": thr}
        for key, oldkey in [("det_auc", "det_auc_800"), ("pixel_auroc", "pixel_auroc"), ("micro_iou", "fixed_miou")]:
            np.testing.assert_allclose(point[key], ev[oldkey], atol=1e-6)
        points[tag], all_scores[tag] = point, sc
        provenance[tag] = {"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(tag, point, flush=True)
    np.savez_compressed(OUT / "paired_boot_final_scores.npz", paths=np.array(rels), sources=src, y=y, masks=masks,
                        **all_scores)
    result = {"n_boot": N_BOOT, "rng_seed": 20260918, "n_clusters_det": 783, "n_clusters_loc": 487,
              "scope": "conditional on fixed seed-42 checkpoints; excludes training randomness",
              "points": points, "provenance": provenance, "comparisons": {}}
    for a, b in [("g2u_s42", "b0_s42"), ("g2u_s42", "po_bALL_s42")]:
        print("Pair (first minus second):", a, b, flush=True)
        sa, sb = all_scores[a], all_scores[b]
        dr = bootstrap_pair(y, sa.max(1), sb.max(1), src)
        lr = bootstrap_pair(masks, sa[:ne], sb[:ne], src[:ne], pixels=True,
                            binary=[sa[:ne] >= points[a]["val_thr"], sb[:ne] >= points[b]["val_thr"]])
        lr["delta_iou"] = points[a]["micro_iou"] - points[b]["micro_iou"]
        result["comparisons"][f"{a}_minus_{b}"] = {"detection": dr, "localization": lr}
        (OUT / "paired_boot_final.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved paired_boot_final.json and reusable score arrays", flush=True)


if __name__ == "__main__":
    main()
