"""构建 IMDLBenCo 数据目录 (Forged/Mask/RealImage) 用于 4 个数据集"""
import json
import shutil
from pathlib import Path


def build(root, pairs, reals):
    root = Path(root)
    for sub in ['Forged', 'Mask', 'RealImage']:
        (root / sub).mkdir(parents=True, exist_ok=True)
    for i, (img, msk) in enumerate(pairs):
        sid = f'{i:04d}'
        ext = Path(img).suffix or '.jpg'
        shutil.copy2(img, root / 'Forged' / f'{sid}{ext}')
        if msk:
            mext = Path(msk).suffix or '.jpg'
            shutil.copy2(msk, root / 'Mask' / f'{sid}{mext}')
    for j, r in enumerate(reals):
        rext = Path(r).suffix or '.jpg'
        shutil.copy2(r, root / 'RealImage' / f'r{j:03d}{rext}')
    return len(pairs)


# MagicBrush
mb = Path('D:/lunwen/data/magicbrush_eval')
mb_ids = sorted(f.stem for f in (mb / 'images').glob('*.png'))
mb_pairs = [(mb / 'images' / f'{s}.png', mb / 'masks' / f'{s}.png') for s in mb_ids]
mb_reals = [str(p) for p in (mb / 'sources').glob('*.png')]
print('magicbrush:', build('D:/lunwen/data/imdl_magicbrush', mb_pairs, mb_reals))

# SDXL-500
sx = Path('D:/lunwen/data/sdxl_edits_500')
sx_ids = sorted(p.name for p in sx.iterdir() if p.is_dir())[:500]
sx_pairs = [(sx / s / 'edit.jpg', sx / s / 'mask.png') for s in sx_ids]
sx_reals = [str(sx / s / 'src.jpg') for s in sx_ids]
print('sdxl:', build('D:/lunwen/data/imdl_sdxl', sx_pairs, sx_reals))

# CelebAHQ subset: 500 standard edits + 300 real
labels = json.load(open('D:/lunwen/data/features_cache/labels.json'))
celeb = [l for l in labels if l.get('cat') == 'CelebAHQ' and l['label'] == 1
         and l.get('kind') == 'standard'][:500]
root = Path('D:/lunwen/data/INP-X/inpainting_exchange')
c_pairs = [(root / l['path'], root / l['mask_path']) for l in celeb
           if (root / l['path']).exists() and (root / l['mask_path']).exists()]
c_reals = [str(root / l['path']) for l in labels
           if l['label'] == 0 and l.get('cat') == 'CelebAHQ'][:300]
print('celeba:', build('D:/lunwen/data/imdl_celeba', c_pairs, c_reals))
