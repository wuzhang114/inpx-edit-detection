"""Final source/PDF checks and layout contact sheets, without manuscript edits."""
import json
import re
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import fitz

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'paper/tifs/_source'
QA=ROOT/'paper/tifs/_review/2026-09-18_final'
result={}
for lang,stem in [('zh','tifs2027_main'),('en','tifs2027_main_english')]:
    tex=(SRC/f'{stem}.tex').read_text(encoding='utf-8')
    log=(SRC/f'_build/{stem}.log').read_text(encoding='utf-8',errors='replace')
    bad=re.findall(r'Overfull|undefined|LaTeX Error|Missing number|no line here|Missing character',log)
    assert not bad,(stem,bad)
    pdf=fitz.open(SRC/f'_build/{stem}.pdf')
    text='\n'.join(p.get_text() for p in pdf)
    (QA/f'{stem}_text.txt').write_text(text,encoding='utf-8')
    assert '??' not in text
    abstract=tex.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
    item={'pages':len(pdf),'bad_log_count':len(bad),'abstract_whitespace_words':len(abstract.split())}
    if lang=='en':assert 150<=item['abstract_whitespace_words']<=250
    result[lang]=item
    paths=sorted((QA/f'render_{lang}').glob('page-*.png'))
    assert len(paths)==len(pdf),(paths,len(pdf))
    for start in range(0,len(paths),4):
        pages=[Image.open(p).convert('RGB') for p in paths[start:start+4]]
        w,h=pages[0].size
        sheet=Image.new('RGB',(w*2,(h+28)*2),'#d9d9d9')
        draw=ImageDraw.Draw(sheet)
        for j,im in enumerate(pages):
            x=(j%2)*w;y=(j//2)*(h+28)
            draw.text((x+10,y+5),f'{lang} page {start+j+1}',fill='black')
            sheet.paste(im,(x,y+28))
        sheet.save(QA/f'contact_{lang}_{start+1}.png')
(QA/'build_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
