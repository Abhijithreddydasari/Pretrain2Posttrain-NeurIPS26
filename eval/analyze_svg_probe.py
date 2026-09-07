"""Paper-ready statistics and figures for the frozen multi-edit probe."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from eval.svg_probe_figures import load,ci

KINDS=['text','box','arrow','mixed']
LABELS={'text':'Text swap','box':'Box positions','arrow':'Connector endpoints','mixed':'Text + boxes'}
PCTS=[0,20,100]

def rate_ci(values):
    values=np.asarray(values,dtype=float);n=len(values);p=float(values.mean());z=1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {'mean':p,'lo':max(0,center-half),'hi':min(1,center+half),'n':n,'interval':'Wilson 95%'}

def summary(rows,base):
    d0=np.array([r['d0'] for r in rows]);d1=np.array([r['d1'] for r in rows])
    correct=(d0>1e-6)&(d1>1e-6);wrong=(d0<-1e-6)&(d1<-1e-6)
    nreverse=int(correct.sum()+wrong.sum())
    result={'n':len(rows),'both_correct':rate_ci(correct),'both_correct_bootstrap':ci(correct),'G':ci([r['G'] for r in rows]),
      'unrelated_G':ci([r.get('shuffled_G',0.0) for r in rows]),
      'correct_minus_unrelated_G':ci([r['G']-r.get('shuffled_G',0.0) for r in rows]),
      'one_image_accuracy':ci(((d0>1e-6).astype(float)+(d1>1e-6))/2),
      'shuffled_both_correct':rate_ci([r['shuffled_both_correct'] for r in rows]),
      'correct_minus_shuffled_both':ci([float(r['both_correct'])-float(r['shuffled_both_correct']) for r in rows]),
      'delta_both_vs_base':ci([float(r['both_correct'])-float(base[r['id']]['both_correct']) for r in rows]),
      'delta_G_vs_base':ci([r['G']-base[r['id']]['G'] for r in rows]),
      'correct_reversals':int(correct.sum()),'wrong_reversals':int(wrong.sum()),
      'fixed_preference_or_tie':len(rows)-nreverse,
      'assignment_randomization_p':float(binomtest(int(correct.sum()),nreverse,.5,alternative='greater').pvalue) if nreverse else 1.0,
      'blank_original_preference_fraction':float(np.mean([r['blank_margin']>1e-6 for r in rows]))}
    result['median_G']=float(np.median([r['G'] for r in rows]))
    result['positive_G_fraction']=float(np.mean([r['G']>1e-6 for r in rows]))
    gs=np.array([r['G'] for r in rows]);nt=int(np.sum(abs(gs)>1e-6));np_=int(np.sum(gs>1e-6))
    result['joint_assignment_accuracy']=rate_ci(gs>1e-6)
    result['joint_assignment_sign_p']=float(binomtest(np_,nt,.5,alternative='greater').pvalue) if nt else 1.0
    result['delta_joint_assignment_vs_base']=ci([float(r['G']>1e-6)-float(base[r['id']]['G']>1e-6) for r in rows])
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('outputs/svg_probe_v3'))
    ap.add_argument('--scores',type=Path,default=Path('outputs/svg_probe_v3_results'))
    ap.add_argument('--out',type=Path,default=Path('outputs/analysis/svg_probe_v3'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    pairs=load(args.data/'pairs.jsonl');expected={r['id'] for r in pairs}
    runs={p:load(args.scores/f'scores_{p:03d}.jsonl') for p in PCTS}
    for p,rows in runs.items():
        assert len(rows)==len(expected) and {r['id'] for r in rows}==expected,f'Incomplete checkpoint {p}'
        for r in rows:
            assert all(np.isfinite(s['mean_logp']) for s in r['scores'].values())
            assert len({s['target_tokens'] for s in r['scores'].values()})==1
    base={r['id']:r for r in runs[0]}
    protocol=json.loads((args.data/'protocol.json').read_text())
    protocol['n']=len(pairs)
    protocol['n_per_bench']={bench:sum(r['bench']==bench for r in pairs) for bench in sorted({r['bench'] for r in pairs})}
    protocol['caution']='65 eligible examples; one training run; secondary numerical margin; no comparison of raw magnitudes across different metrics.'
    protocol['candidate_serialization']='Default native SVG namespace; every final original/edited raster was visually inspected before main inference.'
    report={'protocol':protocol,
            'groups':{},'caution':'Small edit strata; no seed replication. Box movement may occlude text. Randomization tests concern image correspondence, not training causality.'}
    from structsvg_lib.svg_ops import validate_svg
    profile_bad={r['id']:validate_svg((args.data/r['svg0']).read_text(encoding='utf-8'),try_render=False).errors for r in pairs
                 if not validate_svg((args.data/r['svg0']).read_text(encoding='utf-8'),try_render=False).ok}
    report['reference_profile_audit']={'failed':profile_bad,'note':'All candidates parse and were rendered for QA. Two sources fail stricter generation profile (external anchor hyperlinks or only one drawable element). Retained in frozen primary; profile-eligible sensitivity is supplementary.'}
    groups={'all':lambda r:True,'vfig_id':lambda r:r['bench']=='vfig_id','svg_diagrams':lambda r:r['bench']=='svg_diagrams'}
    groups['profile_eligible_sensitivity']=lambda r:r['id'] not in profile_bad
    for kind in KINDS:
        groups['vfig_id_'+kind]=lambda r,k=kind:r['bench']=='vfig_id' and r['edit']==k
        groups['svg_diagrams_'+kind]=lambda r,k=kind:r['bench']=='svg_diagrams' and r['edit']==k
    flat=[]
    for group,predicate in groups.items():
        if not any(predicate(r) for r in runs[0]):continue
        report['groups'][group]={}
        for p in PCTS:
            rr=[r for r in runs[p] if predicate(r)];s=summary(rr,base)
            report['groups'][group][str(p)]=s
            flat.append({'group':group,'pct':p,'n':s['n'],
                         **{f'{k}_{v}':s[k][v] for k in ('both_correct','G','unrelated_G','correct_minus_unrelated_G','delta_both_vs_base','delta_G_vs_base','shuffled_both_correct') for v in ('mean','lo','hi')},
                         'assignment_randomization_p':s['assignment_randomization_p']})
    # Holm adjust the 12 VFIG edit-by-checkpoint correspondence tests.
    tests=sorted([(report['groups']['vfig_id_'+k][str(p)]['assignment_randomization_p'],k,p) for k in KINDS for p in PCTS])
    previous=0.0
    for rank,(pv,k,p) in enumerate(tests):
        adjusted=min(1,max(previous,pv*(len(tests)-rank)));previous=adjusted
        report['groups']['vfig_id_'+k][str(p)]['assignment_p_holm_12']=adjusted
    tests=sorted([(report['groups']['vfig_id_'+k][str(p)]['joint_assignment_sign_p'],k,p) for k in KINDS for p in PCTS])
    previous=0.0
    for rank,(pv,k,p) in enumerate(tests):
        adjusted=min(1,max(previous,pv*(len(tests)-rank)));previous=adjusted
        report['groups']['vfig_id_'+k][str(p)]['joint_assignment_p_holm_12']=adjusted
    report['metric_definitions']={'both_correct':'Both candidate rankings reverse correctly, d0>epsilon and d1>epsilon. Primary strict measure.',
      'rate_intervals':'Wilson 95% displayed to avoid zero-width bootstrap intervals for 0/12 or 12/12; preregistered example bootstrap also retained. Paired changes and G use example bootstrap.',
      'G':'Sum of matching-vs-mismatching margins across both images. Fixed candidate priors cancel.',
      'joint_assignment_accuracy':'Secondary: G>epsilon, selects correct joint pairing of 2 images with 2 SVGs; does not require each individual ranking to reverse.',
      'epsilon':1e-6,'joint_sign_test':'Exact binomial sign test on non-ties at p=.5; Holm correction for 12 VFIG group/checkpoint tests.',
      'both_randomization':'Under within-source image-label randomization, only correctly or incorrectly reversing pairs can succeed; conditional binomial p=.5 among those pairs. Not an unconditional 25% chance baseline.'}
    # Join original benchmark free generations: do not use validity to select probe examples.
    old=load(Path('outputs/analysis/paper_v1/per_example.jsonl'))
    oldmap={(r['bench'],r['pct'],r['id']):r for r in old}
    report['generation_cross_tab']={}
    for p in PCTS:
        counts={'discriminate_valid':0,'discriminate_invalid':0,'fail_valid':0,'fail_invalid':0}
        joined=[]
        for r in runs[p]:
            bench='VFIG-ID' if r['bench']=='vfig_id' else 'SVG-Diagrams'
            g=oldmap[(bench,p,r['source_id'])];valid=bool(g['raw_validity'])
            counts[('discriminate' if r['both_correct'] else 'fail')+('_valid' if valid else '_invalid')]+=1
            joined.append({'id':r['id'],'source_id':r['source_id'],'pct':p,'edit':r['edit'],'bench':r['bench'],
                           'both_correct':r['both_correct'],'original_image_correct':r['d0']>1e-6,
                           'generation_raw_validity':valid,'generation_dino':g['dino_raw']})
        report['generation_cross_tab'][str(p)]=counts
        (args.out/f'joined_{p:03d}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in joined))
    (args.out/'summary.json').write_text(json.dumps(report,indent=2))
    with (args.out/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    figures(report,args.out)
    print(json.dumps({g:report['groups'][g] for g in ['all','vfig_id','svg_diagrams']},indent=2))

def figures(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    colors=['#1765ad','#b64d13','#6d44a3','#21815c']
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for kind,color in zip(KINDS,colors):
        group=report['groups']['vfig_id_'+kind]
        for ax,key,scale in [(axes[0],'both_correct',100),(axes[1],'G',1)]:
            s=[group[str(p)][key] for p in PCTS];mean=np.array([v['mean'] for v in s])*scale
            err=np.array([[v['mean']-v['lo'] for v in s],[v['hi']-v['mean'] for v in s]])*scale
            offset=(KINDS.index(kind)-1.5)*.9
            ax.errorbar(np.array(PCTS)+offset,mean,yerr=np.maximum(err,0),marker='o',capsize=3,color=color,label=LABELS[kind],lw=1.5)
            ax.set_xticks(PCTS);ax.set_xlabel('SFT progress (%)');ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Both image–SVG matches correct (%)');axes[0].set_ylim(-3,103)
    axes[1].set_ylabel('Image interaction G (nats / SVG token)');axes[1].axhline(0,color='#777777',lw=.7)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False)
    fig.suptitle('VFIG counterfactual discrimination • 12 sources per edit type')
    fig.tight_layout(rect=(0,.09,1,.96));save(fig,out,'probe_by_edit');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,bench in zip(axes,['svg_diagrams','vfig_id']):
        group=report['groups'][bench]
        for key,color,label in [('both_correct','#1765ad','Correct image pair'),('shuffled_both_correct','#777777','Unrelated image pair')]:
            s=[group[str(p)][key] for p in PCTS];y=[v['mean']*100 for v in s]
            err=np.array([[v['mean']-v['lo'] for v in s],[v['hi']-v['mean'] for v in s]])*100
            ax.errorbar(PCTS,y,yerr=np.maximum(err,0),color=color,marker='o',capsize=4,label=label)
        ax.set_title(bench.replace('_',' ')+' (n='+str(group['0']['n'])+')');ax.set_xticks(PCTS)
        ax.set_ylim(-3,103);ax.set_xlabel('SFT progress (%)');ax.set_ylabel('Both matches correct (%)')
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False);fig.tight_layout();save(fig,out,'probe_image_controls');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,3.4));names=['discriminate_valid','discriminate_invalid','fail_valid','fail_invalid']
    labels=['Discriminates + valid generation','Discriminates + invalid generation','Fails discrimination + valid','Fails discrimination + invalid']
    bottom=np.zeros(3)
    for name,label,color in zip(names,labels,['#1765ad','#55a7cc','#b64d13','#dedede']):
        v=np.array([report['generation_cross_tab'][str(p)][name] for p in PCTS]);ax.bar([0,1,2],v,bottom=bottom,color=color,label=label);bottom+=v
    ax.set_xticks([0,1,2],['Base','20%','100%']);ax.set_ylabel('Source diagrams (n=65)')
    ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2,fontsize=8.5)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(rect=(0,0,1,.9));save(fig,out,'discrimination_generation');plt.close(fig)

def save(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=220,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight')

if __name__=='__main__':main()
