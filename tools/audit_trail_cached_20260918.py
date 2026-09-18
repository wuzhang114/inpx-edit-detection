"""Audit existing TRAIL maps; all fitted thresholds here are TEST ORACLES.

No inference, training or feature extraction. Reproduces legacy preprocessing
to isolate its threshold-grid error. Not an official TRAIL reproduction.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path('D:/lunwen')
RUN = ROOT / 'outputs/trail_inpx_out'
DATA = ROOT / 'data/dinolizer_inpx500'


def metrics(scores, masks, threshold):
    pred = scores >= threshold
    tp = (pred & masks).sum(axis=1)
    union = (pred | masks).sum(axis=1)
    return {'threshold': float(threshold), 'micro_iou': float(tp.sum()/union.sum()),
            'mean_image_iou': float(np.mean(tp/union))}


def exact_micro_oracle(scores, masks):
    order = np.argsort(-scores.ravel(), kind='stable')
    sc = scores.ravel()[order]
    tp = np.cumsum(masks.ravel()[order], dtype=np.int64)
    end = np.r_[sc[:-1] != sc[1:], True]
    positions = np.flatnonzero(end)
    val = tp[positions] / (masks.sum() + positions + 1 - tp[positions])
    return metrics(scores, masks, sc[positions[np.argmax(val)]])


def main():
    manifest = DATA / 'manifest.csv'
    rows = {r['sample_id']: r for r in csv.DictReader(manifest.open(encoding='utf-8'))}
    run = json.loads((RUN/'run.json').read_text())
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == run['manifest_sha256']
    scores, masks, skipped = [], [], []
    for record in run['records']:
        path = RUN / record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
        row = rows[record['sample_id']]
        assert row['source_id'] == record['source_id']
        with np.load(path, allow_pickle=False) as payload:
            sc = payload['scores'].mean(axis=0)
        sc = np.asarray(Image.fromarray(sc).resize((37, 37), Image.Resampling.BILINEAR)).ravel()
        mk = np.asarray(Image.open(DATA/row['mask']).convert('L').resize((37, 37), Image.Resampling.NEAREST)).ravel() > 127
        if not mk.any():
            skipped.append(record['sample_id'])
            continue
        scores.append(sc)
        masks.append(mk)
    sc, mk = np.stack(scores), np.stack(masks)
    assert np.isfinite(sc).all()
    legacy = max((metrics(sc, mk, t) for t in np.linspace(.05, .95, 91)), key=lambda x:x['micro_iou'])
    oracle = exact_micro_oracle(sc, mk)
    per_oracles = [exact_micro_oracle(s[None], m[None])['micro_iou'] for s,m in zip(sc,mk)]
    within = [roc_auc_score(m, s) for s,m in zip(sc,mk) if m.any() and (~m).any()]
    result = {
        'status': 'diagnostic_test_oracle_not_validation_selected',
        'n': len(sc), 'skipped_empty_at_37_grid': skipped,
        'preprocessing': 'legacy 12-layer mean, bilinear 32-to-37, nearest-neighbor masks, no median3',
        'manifest_sha256_verified': run['manifest_sha256'], 'all_map_sha256_verified': True,
        'score_quantiles': dict(zip(['min','q50','q90','q95','q99','max'], np.quantile(sc,[0,.5,.9,.95,.99,1]).tolist())),
        'fraction_scores_at_least_005': float((sc >= .05).mean()),
        'legacy_005_095_grid_test_oracle': legacy,
        'exact_global_threshold_test_oracle': oracle,
        'per_image_threshold_test_oracle_mean_iou': float(np.mean(per_oracles)),
        'pooled_pixel_auroc': float(roc_auc_score(mk.ravel(),sc.ravel())),
        'within_image_pixel_auroc_mean': float(np.mean(within)),
        'within_image_pixel_auroc_valid_n': len(within),
        'recommendation': 'Withdraw 0.0019 as evidence of TRAIL failure; do not present this test-oracle diagnostic as validation-calibrated performance or official single-layer TRAIL.'
    }
    output = ROOT/'outputs/trail_inpx_threshold_audit_20260918.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
