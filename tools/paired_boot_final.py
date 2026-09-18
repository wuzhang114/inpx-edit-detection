"""P0-4 最终修正: canonical 源簇 + 正确配对(g2u)
1) 检测 800 行: b0_s42 vs g2u_s42 (同制度), canonical 簇 bootstrap
2) 像素损失对照: g2u_s42 (pixel-on) vs po_bALL_s42 (pixel-off) —— 同制度同 20ep/batch128/seed42
   报 micro IoU / pixel AUROC / det AUC 及配对 canonical 簇 bootstrap CI
3) 打印簇数 (期望 检测 783 / 定位 487)
输出: outputs/paired_boot_final.json
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path("D:/lunwen")
sys.path.insert(0, str(ROOT / "src"))
from data.canonical_source import canonical_source_id  # noqa: E402

CACHE = ROOT / "data/features_cache"
IMG_ROOT = ROOT / "data/INP-X/inpainting_exchange"
DATA_JSON = ROOT / "data/imdl_inpx_test.json"
OUT = ROOT / "outputs"
GRID = 37
N_BOOT = 2000
RNG = np.random.RandomState(0)


class Head(torch.nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_dim), torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

    def forward(self, x):
        b, n, c = x.shape
        return self.net(x.reshape(b * n, c)).reshape(b, n)


labels = json.load(open(CACHE / "labels.json", encoding="utf-8"))
meta = json.load(open(CACHE / "metadata.json", encoding="utf-8"))
n_total, n_patches, d = meta["n_total"], meta["n_patches"], meta["dino_dim"]
dino_mm = np.memmap(CACHE / "dino_tokens.npy", dtype=np.float16, mode="r", shape=(n_total, n_patches, d))
hp_mm = np.memmap(CACHE / "highpass.npy", dtype=np.float32, mode="r", shape=(n_total, n_patches, 1))
path2idx = {l["path"]: i for i, l in enumerate(labels)}
name2idx = {}
for i, l in enumerate(labels):
    name2idx.setdefault(Path(l["path"]).name, i)

rec = json.load(open(DATA_JSON, encoding="utf-8"))
edited = [(p, m) for p, m in rec if m != "Negative"]
real = [p for p, m in rec if m == "Negative"]

# ---------- canonical source ----------
def canon_of(img_path, mask_path):
    rel = str(Path(img_path).relative_to(IMG_ROOT)).replace("\\", "/")
    idx = path2idx.get(rel)
    if idx is None:
        idx = name2idx[Path(rel).name]
    cat = labels[idx].get("cat")
    return canonical_source_id(Path(rel).name, cat)


src_edit = [canon_of(p, m) for p, m in edited]
src_real = [canon_of(p, "Negative") for p in real]
print(f"canonical 簇: 检测(500+300)={len(set(src_edit+src_real))}  定位(500)={len(set(src_edit))}", flush=True)

def _idx_of(p):
    rel = str(Path(p).relative_to(IMG_ROOT)).replace("\\", "/")
    i = path2idx.get(rel)
    return i if i is not None else name2idx[Path(rel).name]


paths_all = [p for p, _ in edited] + list(real)
idxs_all = np.array([_idx_of(p) for p in paths_all])
y_all = np.array([1] * len(edited) + [0] * len(real))
src_all = np.array(src_edit + src_real)

masks = np.stack([(np.asarray(Image.open(IMG_ROOT / m).convert("L").resize((GRID, GRID), Image.NEAREST)) > 127).astype(np.float32).ravel() for _, m in edited])


def scores(head_file, use_idx, want_patch=False):
    sd = torch.load(OUT / head_file, map_location="cpu")
    in_dim = int(sd["net.0.weight"].shape[0])
    head = Head(in_dim).cuda(); head.load_state_dict(sd); head.eval()
    D = np.take(dino_mm, use_idx, axis=0).astype(np.float32)
    X = D if in_dim == d else np.concatenate([D, np.take(hp_mm, use_idx, axis=0).astype(np.float32)], axis=-1)
    mx, patch = [], []
    with torch.no_grad():
        for i in range(0, len(X), 64):
            lo = head(torch.from_numpy(X[i:i + 64]).cuda())
            sg = torch.sigmoid(lo).cpu().numpy().reshape(len(X[i:i + 64]), -1)
            mx.append(sg.max(1)); patch.append(sg)
    return np.concatenate(mx), np.concatenate(patch)


def cluster_boot(vals, s_arr, metric_fn, label):
    cl = {}
    for k, s in enumerate(s_arr):
        cl.setdefault(s, []).append(k)
    lst = list(cl.values())
    ds = []
    for _ in range(N_BOOT):
        pick = RNG.randint(0, len(lst), len(lst))
        idx = np.concatenate([lst[p] for p in pick])
        v = metric_fn(idx)
        if v is not None:
            ds.append(v)
    ds = np.array(ds)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    print(f"  [{label}] n_clusters={len(lst)} CI95=[{lo:+.5f}, {hi:+.5f}]", flush=True)
    return {"n_clusters": len(lst), "ci95": [float(lo), float(hi)], "mean": float(ds.mean())}


res = {"note": "canonical 源簇 + 正确配对 (g2u 同制度)", "n_clusters_det": int(len(set(src_edit + src_real))),
       "n_clusters_loc": int(len(set(src_edit)))}

# ---------- 1) 检测: b0 vs g2u ----------
s_b0, _ = scores("weak_sup_v4_b0_s42_head.pt", idxs_all)
s_g2u, _ = scores("weak_sup_v4_g2u_s42_head.pt", idxs_all)
auc_b0 = roc_auc_score(y_all, s_b0); auc_g2u = roc_auc_score(y_all, s_g2u)
res["det_b0_vs_g2u"] = {"auc_b0": float(auc_b0), "auc_g2u": float(auc_g2u),
                        "delta_b0_minus_g2u": float(auc_b0 - auc_g2u)}
print(f"检测: b0={auc_b0:.5f}  g2u={auc_g2u:.5f}  Δ(b0-g2u)={auc_b0-auc_g2u:+.5f}", flush=True)
res["det_b0_vs_g2u"].update(cluster_boot(None, src_all,
    lambda idx: (roc_auc_score(y_all[idx], s_b0[idx]) - roc_auc_score(y_all[idx], s_g2u[idx])) if len(set(y_all[idx])) > 1 else None,
    "det b0-g2u"))

# ---------- 2) 像素损失对照: g2u_s42 (on) vs po_bALL_s42 (off) ----------
sc_on, patch_on = scores("weak_sup_v4_g2u_s42_head.pt", idxs_all)
sc_off, patch_off = scores("weak_sup_v4_po_bALL_s42_head.pt", idxs_all)
thr_on = json.load(open(OUT / "eval500_eval_g2u_s42.json", encoding="utf-8"))["val_thr"]
thr_off = json.load(open(OUT / "eval500_eval_po_bALL_s42.json", encoding="utf-8"))["val_thr"]
m_edit = masks
p_on = patch_on[:len(edited)]; p_off = patch_off[:len(edited)]


def micro(pr, m):
    tp = (pr * m).sum(); fp = (pr * (1 - m)).sum(); fn = ((1 - pr) * m).sum()
    return float(tp / (tp + fp + fn + 1e-8))


res["pixel_loss_controlled"] = {
    "pair": ["g2u_s42 (pixel-on)", "po_bALL_s42 (pixel-off)"],
    "thr_on": thr_on, "thr_off": thr_off,
    "iou_on": micro((p_on >= thr_on).astype(np.float32), m_edit),
    "iou_off": micro((p_off >= thr_off).astype(np.float32), m_edit),
    "pixel_auroc_on": float(roc_auc_score(m_edit.ravel(), p_on.ravel())),
    "pixel_auroc_off": float(roc_auc_score(m_edit.ravel(), p_off.ravel())),
    "det_auc_on": float(roc_auc_score(y_all, sc_on)),
    "det_auc_off": float(roc_auc_score(y_all, sc_off)),
}
r = res["pixel_loss_controlled"]
print(f"像素损失对照: IoU {r['iou_on']:.4f}→{r['iou_off']:.4f} | pixelAUROC {r['pixel_auroc_on']:.4f}→{r['pixel_auroc_off']:.4f} | det {r['det_auc_on']:.4f}→{r['det_auc_off']:.4f}", flush=True)

res["pixel_loss_controlled"]["boot_pixel_auroc_on_minus_off"] = cluster_boot(None, np.array(src_edit),
    lambda idx: float(roc_auc_score(m_edit[idx].ravel(), p_on[idx].ravel()) - roc_auc_score(m_edit[idx].ravel(), p_off[idx].ravel())), "pixelAUROC on-off")
res["pixel_loss_controlled"]["boot_det_auc_on_minus_off"] = cluster_boot(None, src_all,
    lambda idx: (roc_auc_score(y_all[idx], sc_on[idx]) - roc_auc_score(y_all[idx], sc_off[idx])) if len(set(y_all[idx])) > 1 else None, "det on-off")

json.dump(res, open(OUT / "paired_boot_final.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved outputs/paired_boot_final.json", flush=True)
