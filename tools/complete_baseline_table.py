"""Re-evaluate saved baseline maps; use one micro-IoU oracle per dataset.

Classical maps retain their saved 8-bit precision and 512px evaluation grid.
FLAME repeats the historical LAD-only 37px evaluation, caching raw maps.
No training; no changes to historical outputs.
"""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
RECORDS = [r for r in json.loads((ROOT/'data/imdl_inpx_test.json').read_text()) if r[1] != 'Negative']


def summarize(tp, union, thresholds):
    tp, union = np.asarray(tp, dtype=np.int64), np.asarray(union, dtype=np.int64)
    micro = tp.sum(0) / np.maximum(union.sum(0), 1)
    mean = (tp / np.maximum(union, 1)).mean(0)
    j = int(np.argmax(micro))
    return dict(n=len(tp), micro_iou=float(micro[j]), mean_image_iou=float(mean[j]),
                threshold=float(thresholds[j]), selection='dataset-level test oracle maximizing micro IoU',
                per_image_oracle_mean=float((tp/np.maximum(union,1)).max(1).mean()),
                threshold_grid=np.asarray(thresholds).tolist())


def classical(tag):
    folder = OUT / tag / 'pred'
    paths = list(folder.glob('*.png')) + list(folder.glob('*.jpg'))
    index = {}
    for p in paths:
        index.setdefault(p.stem, []).append(p)
    tps, unions, used = [], [], []
    digest = hashlib.sha256()
    for image, mask in RECORDS:
        stem = Path(image).stem
        candidates = index.get(stem, [])
        if not candidates:
            candidates = [p for p in paths if p.name.startswith(stem)]
        assert len(candidates) == 1, (stem, candidates)
        p = candidates[0]
        digest.update(str(p.relative_to(ROOT)).encode())
        digest.update(p.read_bytes())
        pred = np.asarray(Image.open(p).convert('L').resize((512,512), Image.Resampling.NEAREST))
        gt = np.asarray(Image.open(mask).convert('L').resize((512,512), Image.Resampling.NEAREST)) > 127
        pos = np.bincount(pred[gt], minlength=256).astype(np.int64)
        neg = np.bincount(pred[~gt], minlength=256).astype(np.int64)
        tp = pos[::-1].cumsum()[::-1]
        fp = neg[::-1].cumsum()[::-1]
        if len(used) < 3:
            for th in [0, 1, 51, 128, 255]:
                binary = pred >= th
                assert int(tp[th]) == int((binary & gt).sum())
                assert int(gt.sum(dtype=np.int64)+fp[th]) == int((binary | gt).sum())
        tps.append(tp); unions.append(gt.sum(dtype=np.int64) + fp)
        used.append(str(p.relative_to(ROOT)))
    assert len(set(used)) == len(RECORDS) == 500
    result = summarize(tps, unions, np.arange(256)/255)
    result.update(grid=512, prediction_files=used, precision='saved 8-bit maps',
                  prediction_digest_sha256=digest.hexdigest())
    return result


def flame():
    import torch
    import torch.nn.functional as F
    from eval_baselines_matched_intervention import build_flame
    cache = OUT/'baseline_table_flame_maps_20260920.npz'
    if not cache.exists():
        model = build_flame()
        maps, masks = [], []
        with torch.inference_mode():
            for i, (image, mask) in enumerate(RECORDS):
                arr = np.asarray(Image.open(image).convert('RGB').resize((512,512), Image.Resampling.BILINEAR)).copy()
                t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float().cuda()/255
                with torch.autocast('cuda', dtype=torch.float16):
                    _, coarse, _ = model(t)
                prob = F.interpolate(torch.sigmoid(coarse), size=(37,37), mode='bilinear', align_corners=False)
                maps.append(prob.float().cpu().numpy().reshape(-1))
                masks.append((np.asarray(Image.open(mask).convert('L').resize((37,37),Image.Resampling.NEAREST))>127).reshape(-1))
                if (i+1)%50 == 0: print('FLAME',i+1,'/500',flush=True)
        np.savez_compressed(cache, maps=np.stack(maps), masks=np.stack(masks), paths=np.array([r[0] for r in RECORDS]))
    with np.load(cache) as d:
        maps, gt = d['maps'], d['masks']
        assert d['paths'].tolist() == [r[0] for r in RECORDS]
    thresholds = np.linspace(.05,.95,91)
    tp, union = [], []
    for threshold in thresholds:
        pred = maps >= threshold
        tp.append((pred & gt).sum(1)); union.append((pred | gt).sum(1))
    result = summarize(np.array(tp).T,np.array(union).T,thresholds)
    result.update(grid=37, cache=str(cache), precision='float16 inference, bilinear resize',
                  historical_micro=0.12800438702106476,
                  checkpoint_sha256=hashlib.sha256((ROOT/'src/evaluation/flame_g2_ladmulti_sam2.pth').read_bytes()).hexdigest())
    result['historical_delta'] = result['micro_iou'] - result['historical_micro']
    result['historical_note'] = 'Fresh run with current local checkpoint/source; historical checkpoint binary was not fingerprinted. Report fresh paired metrics, not a mixture with the old scalar.'
    return result


if __name__ == '__main__':
    target = OUT/'baseline_table_completed_20260920.json'
    results = {}
    for name, tag in [('MVSS-Net','imdl_pred_mvss'),('IML-ViT','imdl_pred_imlvit'),('ObjectFormer','imdl_pred_of')]:
        results[name] = classical(tag)
        print(name, {k:v for k,v in results[name].items() if k not in ['prediction_files','threshold_grid']}, flush=True)
        target.write_text(json.dumps(results,indent=2))
    results['FLAME-LAD'] = flame()
    target.write_text(json.dumps(results,indent=2))
    print('FLAME-LAD', {k:v for k,v in results['FLAME-LAD'].items() if k != 'threshold_grid'},flush=True)
