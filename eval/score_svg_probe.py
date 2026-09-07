"""Teacher-forced SVG likelihood, frozen candidates and images; no generation."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from PIL import Image
from train.infer_engine import InferEngine
from train.data_utils import prompt_for_row

def score(engine, image, svg):
    prompt=prompt_for_row({'svg':svg},engine.prompt,image=image)
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
    prefix=engine.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    # Encode prefix and full text identically; require exact prefix token boundary.
    p=engine.processor(text=[prefix],images=[image],return_tensors='pt')
    full=engine.processor(text=[prefix+svg],images=[image],return_tensors='pt')
    start=p['input_ids'].shape[1]
    ids=full['input_ids']
    if not torch.equal(p['input_ids'],ids[:,:start]):
        raise RuntimeError('SVG starts across token boundary; refuse ambiguous target mask')
    if ids.shape[1]>8192: raise RuntimeError('Full sequence exceeds frozen 8192 context')
    full={k:v.to(engine.model.device) if hasattr(v,'to') else v for k,v in full.items()}
    with torch.inference_mode():
        output=engine.model(**full,use_cache=False)
        # Token at index t is predicted by logits at t-1. Chunk float32 conversion.
        total=0.0
        for pos in range(start,ids.shape[1],128):
            end=min(pos+128,ids.shape[1])
            logits=output.logits[0,pos-1:end-1,:].float()
            targets=full['input_ids'][0,pos:end]
            total+=float(-torch.nn.functional.cross_entropy(logits,targets,reduction='sum').item())
    n=ids.shape[1]-start
    return {'sum_logp':total,'mean_logp':total/n,'target_tokens':n,'prompt_tokens':start,
            'target_head':engine.processor.tokenizer.decode(ids[0,start:start+8]),
            'target_tail':engine.processor.tokenizer.decode(ids[0,-8:])}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--adapter-root',type=Path,default=Path('/vol/out/e4b_broad_v2'))
    ap.add_argument('--pcts',default='0,20,100')
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(s) for s in (args.data/'pairs.jsonl').read_text().splitlines()]
    qa=json.loads((args.data/'qa_selection.json').read_text())
    rows=[r for r in rows if r['id'] not in qa['excluded']]
    lookup={r['id']:r for r in rows}
    if args.limit: rows=rows[:args.limit]
    engine=InferEngine(Path('/root/configs/model_e4b.yaml'),merge_adapters=False)
    run={'manifest_sha256':hashlib.sha256((args.data/'pairs.jsonl').read_bytes()).hexdigest(),
         'adapter_root':str(args.adapter_root),'pcts':args.pcts,'n':len(rows),
         'model':engine.model_id,'loader':engine.loader,'precision':'bf16; unmerged adapters',
         'qa_selection':qa,
         'protocol':json.loads((args.data/'protocol.json').read_text()),
         'model_commit':getattr(engine.base_model.config,'_commit_hash',None),
         'torch':torch.__version__}
    (args.out/'run.json').write_text(json.dumps(run,indent=2))
    for pct in map(int,args.pcts.split(',')):
        engine.set_adapter(None if pct==0 else args.adapter_root/f'checkpoint_pct_{pct:03d}',tag=str(pct))
        engine.model.eval()
        path=args.out/f'scores_{pct:03d}.jsonl'
        done={json.loads(s)['id'] for s in path.read_text().splitlines()} if path.exists() else set()
        for idx,row in enumerate(rows):
            if row['id'] in done: continue
            started=time.monotonic(); result={**row,'pct':pct,'scores':{}}
            svgs=[(args.data/row[f'svg{i}']).read_text(encoding='utf-8') for i in (0,1)]
            images=[Image.open(args.data/row[f'image{i}']).convert('RGB') for i in (0,1)]
            images.append(Image.new('RGB',images[0].size,'white'))
            shuffled=lookup[row['shuffled_source_id']]
            if shuffled['id']==row['id']:
                shuffled=next(r for r in lookup.values() if r['bench']==row['bench'] and r['id']!=row['id'])
            result['actual_shuffled_source_id']=shuffled['id']
            images.extend(Image.open(args.data/shuffled[f'image{i}']).convert('RGB') for i in (0,1))
            for i,img in enumerate(images):
                for j,svg in enumerate(svgs):
                    result['scores'][f'{i}{j}']=score(engine,img,svg)
            ns={x['target_tokens'] for x in result['scores'].values()}
            if len(ns)!=1: raise RuntimeError('Candidates have unequal contextual token counts')
            if idx==0:
                repeat=score(engine,images[0],svgs[0])
                difference=abs(repeat['mean_logp']-result['scores']['00']['mean_logp'])
                result['repeat_score_abs_difference']=difference
                if difference>1e-6:raise RuntimeError(f'Non-deterministic repeated score: {difference}')
            s={k:v['mean_logp'] for k,v in result['scores'].items()}
            d0=s['00']-s['01'];d1=s['11']-s['10']
            result.update(d0=d0,d1=d1,G=d0+d1,both_correct=d0>1e-6 and d1>1e-6,
                          blank_margin=s['20']-s['21'],seconds=time.monotonic()-started)
            sd0=s['30']-s['31'];sd1=s['41']-s['40']
            result.update(shuffled_d0=sd0,shuffled_d1=sd1,shuffled_G=sd0+sd1,
                          shuffled_both_correct=sd0>1e-6 and sd1>1e-6)
            with path.open('a',encoding='utf-8') as f:f.write(json.dumps(result)+'\n')
            print(f'probe pct={pct} pair={idx+1}/{len(rows)} G={result["G"]:.6f} both={result["both_correct"]} sec={result["seconds"]:.1f}',flush=True)
    print('PROBE COMPLETE',flush=True)

if __name__=='__main__': main()
