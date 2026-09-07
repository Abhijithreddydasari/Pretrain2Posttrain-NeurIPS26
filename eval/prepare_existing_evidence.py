"""Publication figures from existing caches; no generation or metric changes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import binomtest
from eval.svg_probe_figures import load,ci

def main():
    out=Path('outputs/analysis/svg_probe_v3');out.mkdir(parents=True,exist_ok=True)
    records=load(Path('outputs/analysis/paper_v1/per_example.jsonl'))
    pcts=[0,5,10,20,40,60,80,100]
    benches=[('SVG-Diagrams','svg_diagrams','#1765ad'),('VFIG-ID','vfig_id','#b64d13'),('VFIG-OOD','vfig_ood','#6d44a3')]
    fig,axes=plt.subplots(1,3,figsize=(11,3.8));report={};paired={}
    for name,prefix,color in benches:
        report[name]={};paired[name]={}
        base={r['id']:r for r in records if r['bench']==name and r['pct']==0}
        for pct in pcts:
            rr=[r for r in records if r['bench']==name and r['pct']==pct]
            tag='base_0pct' if pct==0 else f'pct_{pct:03d}'
            cached=json.loads(Path(f'outputs/metrics/scored_{prefix}_128/{prefix}_{tag}_prompt.json').read_text())
            stats={'validity':ci([r['raw_validity'] for r in rr]),'dino':ci([r['dino_raw'] for r in rr]),
                   'ssim':cached['bootstrap']['ssim_all']}
            report[name][str(pct)]=stats
            changes=[(r,base[r['id']]) for r in rr]
            gained=int(sum(bool(r['raw_validity']) and not bool(b['raw_validity']) for r,b in changes))
            lost=int(sum(bool(b['raw_validity']) and not bool(r['raw_validity']) for r,b in changes))
            paired[name][str(pct)]={
              'validity_delta':ci([r['raw_validity']-b['raw_validity'] for r,b in changes]),
              'dino_delta':ci([r['dino_raw']-b['dino_raw'] for r,b in changes]),
              'validity_gained':gained,'validity_lost':lost,
              'mcnemar_exact_two_sided_p':float(binomtest(min(gained,lost),gained+lost,.5).pvalue) if gained+lost else 1.0}
        for ax,key in zip(axes,['validity','dino','ssim']):
            s=[report[name][str(p)][key] for p in pcts]
            y=[v['mean'] for v in s];lo=[v['lo'] for v in s];hi=[v['hi'] for v in s]
            ax.plot(pcts,y,'o-',color=color,label=name,lw=1.5,ms=4);ax.fill_between(pcts,lo,hi,color=color,alpha=.12)
            ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel('SFT progress (%)');ax.set_ylim(bottom=0)
            ax.spines[['top','right']].set_visible(False)
    for ax,label in zip(axes,['Executable SVG fraction','DINO cosine (invalid = 0)','SSIM (invalid = 0)']):ax.set_ylabel(label)
    axes[0].legend(frameon=False);fig.suptitle('Free generation • 128 fixed examples per benchmark • 95% example-bootstrap intervals')
    fig.tight_layout();fig.savefig(out/'generation_with_uncertainty.png',dpi=220);fig.savefig(out/'generation_with_uncertainty.pdf');plt.close(fig)
    # Existing short-target sensitivity analysis; not a new length threshold tuned on outcomes.
    length={}
    for name,_,_ in benches[:2]:
        length[name]={}
        for pct in pcts:
            rr=[r for r in records if r['bench']==name and r['pct']==pct and r['gold_tokens']<=2048]
            length[name][str(pct)]={'n':len(rr),'validity':ci([r['raw_validity'] for r in rr]),
              'length_limit':ci([r['hit_length_limit'] for r in rr])}
    (out/'existing_generation_summary.json').write_text(json.dumps({'curves':report,'paired_vs_base':paired,'short_targets_le_2048':length,
      'note':'SSIM already existed in scored benchmark JSON; DINO comes from later batched cached analysis. Bands use existing SSIM bootstrap; 4000 resamples for DINO/validity. Short-target analysis exploratory; gold length is not a lower bound on valid alternative reconstructions.'},indent=2))
    # Outcome-independent visual examples from the completed QA review.
    data=Path('outputs/svg_probe_v3');pairs=load(data/'pairs.jsonl')
    chosen=[pairs[i-1] for i in [10,43,28,33]]
    fig,axes=plt.subplots(4,2,figsize=(8,8))
    for row,(pair,label) in enumerate(zip(chosen,['Text labels','Box positions','Connector endpoints','Text + boxes'])):
        images=[Image.open(data/pair[f'image{i}']).convert('RGB') for i in (0,1)]
        diff=np.max(abs(np.asarray(images[0]).astype(int)-np.asarray(images[1]).astype(int)),axis=2)>25
        ys,xs=np.where(diff);box=(max(0,int(xs.min())-45),max(0,int(ys.min())-45),min(960,int(xs.max())+46),min(960,int(ys.max())+46))
        for col,im in enumerate(images):
            axes[row,col].imshow(im.crop(box));axes[row,col].axis('off');axes[row,col].set_title(label+(': original' if col==0 else ': edited'),fontsize=10)
    fig.suptitle('Counterfactual examples (matched crops; full images used for scoring)')
    fig.tight_layout();fig.savefig(out/'probe_examples.png',dpi=220);fig.savefig(out/'probe_examples.pdf');plt.close(fig)
    (out/'example_selection.json').write_text(json.dumps(chosen,indent=2))
    print('Saved existing generation uncertainty curves, short-target sensitivity and probe example figure.')

if __name__=='__main__':main()
