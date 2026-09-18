"""Final descriptive budget response, sourced only from refit_final.json.

Two panels, zero-origin y axes, symlog budget with exact zero; three-seed
sample SD. Previous held-out observations are retrospective checks, not a
new prospective validation. No cross-capacity attribution is plotted.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT/'outputs/refit_final.json').read_text())
OUT = ROOT/'paper/figures'


def draw(lang):
    zh = lang == 'zh'
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'Arial', 'DejaVu Sans'] if zh else ['Arial', 'DejaVu Sans'],
        'font.size': 8.5, 'axes.labelsize': 9, 'xtick.labelsize': 8,
        'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'pdf.fonttype': 42,
        'savefig.dpi': 300, 'axes.unicode_minus': False,
    })
    k = np.asarray(DATA['budgets'])
    loc, sd = np.asarray(DATA['loc_mean']), np.asarray(DATA['loc_sd'])
    assert np.allclose(np.asarray(DATA['loc_runs']).mean(axis=0), loc)
    assert np.allclose(np.asarray(DATA['loc_runs']).std(axis=0, ddof=1), sd)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.12))
    fig.subplots_adjust(left=.077, right=.982, bottom=.235, top=.91, wspace=.27)
    blue, orange = '#0072B2', '#D55E00'
    for ax in axes:
        ax.set_xscale('symlog', linthresh=10, linscale=.55)
        ax.set_xlim(-.6, 22000)
        ax.set_ylim(0, 1)
        ax.set_xticks([0,10,200,1000,14731], ['0','10','200','1k','ALL'])
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis='y', color='#DFE4E8', lw=.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.65)
        ax.set_xlabel('训练掩码预算 k' if zh else 'Training-mask budget, k')
    a, b = axes
    a.text(0,1.055,'(a) 定位' if zh else '(a) Localization',transform=a.transAxes,fontsize=10,fontweight='bold')
    b.text(0,1.055,'(b) 检测' if zh else '(b) Detection',transform=b.transAxes,fontsize=10,fontweight='bold')
    a.set_ylabel('全局微平均 IoU' if zh else 'Global micro IoU')
    b.set_ylabel('图像级 AUROC' if zh else 'Image-level AUROC')
    a.errorbar(k,loc,yerr=sd,fmt='o',markersize=4,color=blue,capsize=2.5,lw=1.1,
               label='三种子均值 ± 标准差' if zh else 'Three-seed mean ± SD',zorder=4)
    xx = np.r_[0,np.geomspace(.02,14731,500)]
    m0,amp,g,lk = DATA['models']['Log-sigmoid']['params']
    yy = m0+amp*np.where(xx>0,expit(g*(np.log10(np.maximum(xx,1e-300))-lk)),0.)
    a.plot(xx,yy,'--',color=blue,lw=1.25,
           label='七预算点描述性拟合' if zh else 'Descriptive fit (seven budgets)')
    hold = DATA['log_sigmoid_summary']['heldout_reassessment']
    a.scatter([p['k'] for p in hold],[p['measured'] for p in hold],marker='D',s=24,
              color=orange,edgecolors='white',linewidths=.5,zorder=5,
              label='原留出点，种子 42' if zh else 'Previously held out, seed 42')
    a.legend(loc='upper left',frameon=False,handlelength=2,borderaxespad=.35,labelspacing=.6)
    b.errorbar(k,DATA['det_mean'],yerr=DATA['det_sd'],fmt='o-',markersize=4,color=orange,
               lw=1.1,capsize=2.5,zorder=4)
    b.text(.055,.63,'三种子均值 ± 标准差\n连线仅作视觉引导' if zh else 'Three-seed mean ± SD\nLines guide the eye',
           transform=b.transAxes,fontsize=8.5,color='#434B52',linespacing=1.5)
    note = ('30K 读出；385 维输入（含高通特征）。ALL = 14,731；另用 1,614 个验证掩码。'
            if zh else '30K readout; 385-D input (including high-pass). ALL = 14,731; 1,614 validation masks additional.')
    note2 = '原留出点现作回顾性检验；未用于七预算点拟合。' if zh else 'Previously held-out points are retrospective checks, excluded from the seven-budget fit.'
    fig.text(.077,.080,note,fontsize=8,color='#434B52')
    fig.text(.077,.030,note2,fontsize=8,color='#434B52')
    stem = OUT/f'budget_response_final_{lang}'
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.png'))
    plt.close(fig)
    print(stem, 'endpoints',loc[[0,-1]].tolist(), 'heldout',[p['measured'] for p in hold])


if __name__ == '__main__':
    draw('en')
    draw('zh')
