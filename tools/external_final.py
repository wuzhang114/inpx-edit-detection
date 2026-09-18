"""Evaluate final g2u on external images, with old-g2 parity and reusable maps.
No suitable external feature caches were found. One backbone pass per image
serves both endpoints and, on CocoGlide, all seven budget-specific readouts.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baselines.weakly_supervised_v4 import MLPHead
from utils.highpass import cross_diff_highpass, pool_to_patch_grid
from external_uniform_eval import pairs_of
OUT = ROOT / "outputs"


def metrics(pm, gt, thr):
    def miou(t):
        pr = pm >= t
        inter = np.logical_and(pr, gt).sum(1)
        union = np.logical_or(pr, gt).sum(1)
        return float(np.mean(inter/(union+1e-8)))
    thresholds = np.linspace(.05,.95,91)
    vals = [miou(t) for t in thresholds]
    j = int(np.argmax(vals))
    return {"val_thr": float(thr), "mean_image_iou_at_val_thr": miou(thr),
            "mean_image_iou_best37": vals[j], "best_thr": float(thresholds[j]),
            "pooled_pixel_auroc": float(roc_auc_score(gt.ravel(), pm.ravel()))}


def main():
    repo = Path(os.environ["USERPROFILE"]) / ".cache/torch/hub/facebookresearch_dinov2_main"
    model = torch.hub.load(str(repo), "dinov2_vits14", source="local").cuda().eval()
    mean = torch.tensor([.485,.456,.406],device="cuda").view(1,3,1,1)
    std = torch.tensor([.229,.224,.225],device="cuda").view(1,3,1,1)
    res = {"aggregation": "mean per-image IoU on edited images; nearest-neighbor 37x37 masks",
           "thresholds": "validation per head; best37 is test oracle over .05:.01:.95", "datasets": {}}
    for ds in ["cocoglide", "magicbrush", "sdxl", "autosplice"]:
        pairs = pairs_of(ds)
        tags = [f"b{k}_s42" for k in [0,10,50,200,1000,5000]] if ds == "cocoglide" else []
        tags += ["g2_s42", "g2u_s42"]
        heads = {}
        for tag in tags:
            head = MLPHead(385, hidden=64).cuda().eval()
            head.load_state_dict(torch.load(OUT/f"weak_sup_v4_{tag}_head.pt",map_location="cpu",weights_only=True))
            heads[tag] = head
        probs = {tag: [] for tag in tags}; masks = []
        print(ds, len(pairs), "images", flush=True)
        with torch.no_grad():
            for start in range(0,len(pairs),8):
                chunk=pairs[start:start+8]; ims=[]
                for f,m in chunk:
                    arr=np.asarray(Image.open(f).convert("RGB")).copy()
                    t=torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float().cuda()/255
                    ims.append(F.interpolate(t,size=(518,518),mode="bilinear",align_corners=False))
                    masks.append(np.asarray(Image.open(m).convert("L").resize((37,37),Image.Resampling.NEAREST)).ravel()>127)
                x=(torch.cat(ims)-mean)/std
                tokens=model.forward_features(x)["x_norm_patchtokens"]
                hp=pool_to_patch_grid(F.pad(cross_diff_highpass(x),(0,1,0,1)),37*37)
                features=torch.cat([tokens,hp],dim=-1)
                for tag,h in heads.items():
                    probs[tag].append(torch.sigmoid(h(features)).cpu().numpy())
                if (start//8+1)%25==0:
                    print(ds,min(start+8,len(pairs)),"/",len(pairs),flush=True)
        gt=np.stack(masks)
        scores={tag: np.concatenate(v) for tag,v in probs.items()}
        rows={tag: metrics(scores[tag],gt,json.loads((OUT/f"eval500_eval_{tag}.json").read_text())["val_thr"]) for tag in tags}
        # Legacy endpoint parity, allowing the old JSON's four-decimal rounding.
        if ds=="cocoglide":
            old=json.loads((OUT/"cocoglide_budget_curve.json").read_text())["14731"]
        else:
            old=json.loads((OUT/"external_uniform_g2.json").read_text())[ds]
        np.testing.assert_allclose(rows["g2_s42"]["mean_image_iou_best37"],old["miou_best37"],atol=.0002)
        res["datasets"][ds]={"n":len(pairs),"legacy_parity_passed":True,"heads":rows}
        np.savez_compressed(OUT/f"external_final_{ds}_scores.npz",paths=np.array([str(p) for p,_ in pairs]),masks=gt,**scores)
        (OUT/"external_final.json").write_text(json.dumps(res,indent=2),encoding="utf-8")
        print(ds, "g2u", rows["g2u_s42"],flush=True)


if __name__=="__main__":
    main()
