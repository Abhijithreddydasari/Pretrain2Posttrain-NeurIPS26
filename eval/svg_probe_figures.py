"""Probe visual QA and source-paired bootstrap figures; no new inference."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def load(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def qa(data):
    rows=load(data/'pairs.jsonl');out=data/'qa';out.mkdir(exist_ok=True)
    try:font=ImageFont.truetype('arial.ttf',17)
    except OSError:font=ImageFont.load_default()
    for start in range(0,len(rows),8):
        sheet=Image.new('RGB',(1440,8*210),'white');draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(rows[start:start+8]):
            ims=[Image.open(data/row[f'image{i}']).convert('RGB') for i in (0,1)]
            diff=np.max(abs(np.asarray(ims[0]).astype(int)-np.asarray(ims[1]).astype(int)),axis=2)>25
            ys,xs=np.where(diff); box=(max(0,int(xs.min())-30),max(0,int(ys.min())-30),min(960,int(xs.max())+31),min(960,int(ys.max())+31))
            label=f'{start+j+1}: {row["bench"]} | {row["labels"][0]!r} <-> {row["labels"][1]!r}'
            draw.text((10,j*210),label,font=font,fill='black')
            for i,im in enumerate(ims):
                thumb=im.copy();thumb.thumbnail((190,175));sheet.paste(thumb,(i*720+10,j*210+30))
                crop=im.crop(box);crop.thumbnail((500,175));sheet.paste(crop,(i*720+210,j*210+30))
        sheet.save(out/f'contact_{start//8+1:02d}.png')
    print(f'QA sheets: {out}')

def ci(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(20260905)
    boot=a[rng.integers(0,len(a),size=(4000,len(a)))].mean(axis=1)
    return {'mean':float(a.mean()),'lo':float(np.quantile(boot,.025)),'hi':float(np.quantile(boot,.975)),'n':len(a)}

def analyze(data,scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=load(data/'pairs.jsonl');pcts=[0,20,100]
    qa_selection=json.loads((data/'qa_selection.json').read_text())
    rows=[r for r in rows if r['id'] not in qa_selection['excluded']]
    all_scores={p:load(scores/f'scores_{p:03d}.jsonl') for p in pcts}
    expected={r['id'] for r in rows}
    for p,rr in all_scores.items():
        if len(rr)!=len(expected) or {r['id'] for r in rr}!=expected:raise ValueError(f'Incomplete {p}')
    report={};fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for bench,color in [('svg_diagrams','#2563eb'),('vfig_id','#c34e16')]:
        report[bench]={}; base={r['id']:r for r in all_scores[0] if r['bench']==bench}
        for p in pcts:
            rr=[r for r in all_scores[p] if r['bench']==bench]
            stats={key:ci([r[key] for r in rr]) for key in ['both_correct','G']}
            stats['paired_both_vs_base']=ci([float(r['both_correct'])-float(base[r['id']]['both_correct']) for r in rr])
            stats['paired_G_vs_base']=ci([r['G']-base[r['id']]['G'] for r in rr])
            stats['one_image_accuracy']=ci([((r['d0']>1e-6)+(r['d1']>1e-6))/2 for r in rr])
            stats['ties']=sum(abs(r['d0'])<=1e-6 or abs(r['d1'])<=1e-6 for r in rr)
            stats['blank_prefers_original']=sum(r['blank_margin']>1e-6 for r in rr)/len(rr)
            # Identical blank images cannot reverse a fixed preference.
            stats['blank_both_correct']=0.0
            stats['shuffled_both_correct']=ci([r['shuffled_both_correct'] for r in rr])
            stats['correct_minus_shuffled_both']=ci([float(r['both_correct'])-float(r['shuffled_both_correct']) for r in rr])
            stats['correct_minus_shuffled_G']=ci([r['G']-r['shuffled_G'] for r in rr])
            report[bench][str(p)]=stats
        for ax,key,mult in [(axes[0],'both_correct',100),(axes[1],'G',1)]:
            st=[report[bench][str(p)][key] for p in pcts];y=np.array([s['mean'] for s in st])*mult
            err=np.array([[s['mean']-s['lo'] for s in st],[s['hi']-s['mean'] for s in st]])*mult
            ax.errorbar(pcts,y,yerr=err,marker='o',capsize=4,color=color,label=bench.replace('_',' '))
            ax.set_xticks(pcts);ax.set_xlabel('SFT progress (%)');ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Both image–SVG matches correct (%)');axes[0].set_ylim(-3,103)
    axes[1].set_ylabel('Image–candidate interaction G (nats/token)');axes[1].axhline(0,color='gray',lw=1)
    axes[0].legend(frameon=False);fig.suptitle('Text-location discrimination with valid SVG candidates')
    fig.tight_layout();fig.savefig(scores/'probe_trajectories.png',dpi=220);fig.savefig(scores/'probe_trajectories.pdf')
    (scores/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('outputs/svg_probe_v1'))
    ap.add_argument('--scores',type=Path);args=ap.parse_args()
    if args.scores:analyze(args.data,args.scores)
    else:qa(args.data)
