"""Final qualitative examples: cached features, final g2u endpoint, seed 42.

Adapted from make_qualitative_fig_v14.py. Selection remains maximum / median /
minimum per-image ALL IoU. No new feature extraction. Saves full provenance.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baselines.weakly_supervised_v2 import MLPHead


def main():
    cache, out = ROOT/'data/features_cache', ROOT/'outputs'
    image_root = ROOT/'data/INP-X/inpainting_exchange'
    labels = json.loads((cache/'labels.json').read_text(encoding='utf-8'))
    meta = json.loads((cache/'metadata.json').read_text())
    n,p,d = [meta[k] for k in ['n_total','n_patches','dino_dim']]
    tokens = np.memmap(cache/'dino_tokens.npy',dtype=np.float16,mode='r',shape=(n,p,d))
    hp = np.memmap(cache/'highpass.npy',dtype=np.float32,mode='r',shape=(n,p,1))
    idx = {x['path'].replace('\\','/'):i for i,x in enumerate(labels)}
    pairs = []
    for im,mask in json.loads((ROOT/'data/imdl_inpx_test.json').read_text()):
        if mask == 'Negative':
            continue
        rel = Path(im).relative_to(image_root).as_posix()
        assert rel in idx, rel
        pairs.append({'image':im,'mask':mask,'cache_index':idx[rel]})
    gt = np.stack([np.asarray(Image.open(x['mask']).convert('L').resize((37,37),Image.Resampling.NEAREST)).ravel()>127 for x in pairs])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    maps, info = {}, {}
    tags = ['b0_s42','b200_s42','b1000_s42','g2u_s42']
    for tag in tags:
        head_path = out/f'weak_sup_v4_{tag}_head.pt'
        ev = json.loads((out/f'eval500_eval_{tag}.json').read_text())
        head = MLPHead(d+1,hidden=64).to(device)
        head.load_state_dict(torch.load(head_path,map_location='cpu',weights_only=True))
        head.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0,len(pairs),64):
                ids = [x['cache_index'] for x in pairs[start:start+64]]
                features = torch.cat([torch.from_numpy(tokens[ids]),torch.from_numpy(hp[ids])],dim=-1).to(device).float()
                chunks.append(head(features).squeeze(-1).sigmoid().cpu().numpy())
        probs = np.concatenate(chunks)
        pred = probs>=ev['val_thr']
        tp = (pred&gt).sum(axis=1)
        union = (pred|gt).sum(axis=1)
        per = tp/(union+1e-8)
        micro = float(tp.sum()/union.sum())
        assert abs(micro-ev['fixed_miou'])<1e-7,(tag,micro,ev['fixed_miou'])
        maps[tag] = (probs,per)
        info[tag] = {'threshold':ev['val_thr'],'checkpoint':str(head_path),
                     'sha256':hashlib.sha256(head_path.read_bytes()).hexdigest(),'verified_micro_iou':micro}
        print(tag,micro,flush=True)
    order = np.argsort(maps['g2u_s42'][1])
    nonempty_order = [int(i) for i in order if gt[i].any()]
    picks = [int(order[-1]),int(order[len(order)//2]),nonempty_order[0]]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42})
    fig,axes = plt.subplots(3,6,figsize=(7.16,4.05))
    titles = ['Edited image','Ground truth','k = 0','k = 200','k = 1,000','ALL']
    for r,pi in enumerate(picks):
        image = np.asarray(Image.open(pairs[pi]['image']).convert('RGB').resize((224,224)))
        for c,ax in enumerate(axes[r]):
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(image)
            if c==1:
                ax.imshow(gt[pi].reshape(37,37),cmap='Greens',alpha=.45,vmin=0,vmax=1,extent=(0,224,224,0))
            elif c>1:
                tag = tags[c-2]
                probs = maps[tag][0][pi].reshape(37,37)
                ax.imshow(probs,cmap='magma',alpha=.55,vmin=0,vmax=1,extent=(0,224,224,0))
                pred = probs>=info[tag]['threshold']
                if pred.any() and not pred.all():
                    ax.contour(pred.astype(float),levels=[.5],colors='cyan',linewidths=.8,extent=(0,224,224,0),origin='upper')
            if r==0:
                ax.set_title(titles[c],fontsize=8.5,pad=6)
            if c==0:
                ax.set_ylabel(f'{["High","Median","Failure"][r]}\nALL IoU={maps["g2u_s42"][1][pi]:.2f}',fontsize=8.5)
    fig.subplots_adjust(left=.074,right=.998,bottom=.02,top=.925,wspace=.045,hspace=.06)
    stem = ROOT/'paper/figures/fig_qualitative_final_en'
    for ext in ['pdf','png']:
        fig.savefig(stem.with_suffix('.'+ext),dpi=300)
    result = {'selection_rule':'v14 max and median ALL per-image IoU retained; minimum selected among nonempty 37-grid masks, np.argsort tie order',
              'empty_masks_at_37_grid':int((gt.sum(axis=1)==0).sum()),
              'cohort_n':len(pairs),'heads':info,'selected':[]}
    for label,pi in zip(['high','median','failure'],picks):
        result['selected'].append({'label':label,'cohort_index':pi,**pairs[pi],
            'per_image_iou':{tag:float(maps[tag][1][pi]) for tag in tags}})
    (out/'qualitative_final_provenance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['selected'],indent=2),flush=True)


if __name__=='__main__':
    main()
