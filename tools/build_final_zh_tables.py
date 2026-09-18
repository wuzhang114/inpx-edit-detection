"""Generate Chinese result tables from final JSON and assemble authored body.
The manuscript body is authored in final_body_zh.tex; this performs only
deterministic numeric table formatting and preamble/body assembly.
"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
SRC=ROOT/'paper/tifs/_source'
fit=json.loads((OUT/'refit_final.json').read_text())
ext=json.loads((OUT/'external_final.json').read_text())['datasets']
assert set(ext)=={'cocoglide','magicbrush','sdxl','autosplice'}
end=r' \\'
lines=[r'\begin{table}[t]',r'\centering\small\setlength{\tabcolsep}{3pt}',
       r'\caption{主30K预算系列，均值$\pm$样本标准差（3次运行）。全量端点为g2u。}',r'\label{tab:budget_final}',
       r'\begin{tabular}{rcc}',r'\toprule',r'训练掩码$k$ & 定位micro IoU & 检测AUC\\',r'\midrule']
for i,k in enumerate(fit['budgets']):
    lines.append(f"{k} & ${fit['loc_mean'][i]:.4f}\\pm{fit['loc_sd'][i]:.4f}$ & ${fit['det_mean'][i]:.4f}\\pm{fit['det_sd'][i]:.4f}$"+end)
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(SRC/'final_budget_zh.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
lines=[r'\begin{table*}[t]',r'\centering\small',
       r'\caption{最终g2u的外部测试，seed42；定位为逐图IoU均值，AUROC为汇总patch指标。验证阈值均为0.27。Oracle在目标测试集上选阈，不代表可部署阈值。}',
       r'\label{tab:external_final}',r'\begin{tabular}{lrrrrr}',r'\toprule',
       r'数据集 & 编辑图数 & IoU（验证阈值） & IoU（测试oracle） & Oracle阈值 & Patch AUROC\\',r'\midrule']
for ds,label in [('cocoglide','CocoGlide'),('magicbrush','MagicBrush'),('sdxl','SDXL'),('autosplice','AutoSplice')]:
    r=ext[ds]['heads']['g2u_s42']
    lines.append(f"{label} & {ext[ds]['n']} & {r['mean_image_iou_at_val_thr']:.3f} & {r['mean_image_iou_best37']:.3f} & {r['best_thr']:.2f} & {r['pooled_pixel_auroc']:.3f}"+end)
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(SRC/'final_external_zh.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
cg=ext['cocoglide']['heads']
tags=[f'b{k}_s42' for k in [0,10,50,200,1000,5000]]+['g2u_s42']
lines=[r'\begin{table}[t]',r'\centering\footnotesize\setlength{\tabcolsep}{2pt}',
       r'\caption{INP-X训练系列迁移至CocoGlide。IoU为逐图均值；ALL使用最终g2u。}',r'\label{tab:second}',
       r'\resizebox{\columnwidth}{!}{%',r'\begin{tabular}{lrrrrrrr}',r'\toprule',
       r'$k$ & 0 & 10 & 50 & 200 & 1000 & 5000 & ALL\\',r'\midrule']
for name,key in [('验证阈值','val_thr'),('IoU（验证）','mean_image_iou_at_val_thr'),('IoU（oracle）','mean_image_iou_best37'),('Patch AUROC','pooled_pixel_auroc')]:
    lines.append(name+' & '+' & '.join(f'{cg[t][key]:.3f}' for t in tags)+end)
lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
(SRC/'final_cocoglide_zh.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
original=ROOT/'paper/tifs/_archive/2026-09-18_final_revision/tifs2027_main.tex'
preamble=original.read_text(encoding='utf-8').split(r'\begin{document}',1)[0]
body=(SRC/'final_body_zh.tex').read_text(encoding='utf-8')
(SRC/'tifs2027_main.tex').write_text(preamble+body,encoding='utf-8')
print('Generated 3 traceable tables; assembled current Chinese manuscript.')
