"""Construct text, box-position, arrow-endpoint and mixed counterfactuals."""
from __future__ import annotations
import copy,hashlib,itertools,json,random,re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from tokenizers import Tokenizer
from eval.build_svg_probe import MANIFESTS
from train.data_utils import resolve_svg
from structsvg_lib.svg_ops import render_pil

NUM=re.compile(r'-?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')
def tag(e):return e.tag.split('}')[-1]
def number(e,k):return float(e.get(k,'0'))
def endpoint(e):
    if tag(e)=='line':return number(e,'x2'),number(e,'y2')
    d=e.get('d','')
    if re.search('[a-z]',d) or 'Z' in d or 'A' in d or 'H' in d or 'V' in d:return None
    nums=list(NUM.finditer(d))
    return (float(nums[-2].group()),float(nums[-1].group())) if len(nums)>=4 else None
def set_endpoint(e,xy):
    if tag(e)=='line':e.set('x2',f'{xy[0]:g}');e.set('y2',f'{xy[1]:g}');return
    d=e.get('d');nums=list(NUM.finditer(d))
    e.set('d',d[:nums[-2].start()]+f'{xy[0]:g}'+d[nums[-2].end():nums[-1].start()]+f'{xy[1]:g}'+d[nums[-1].end():])

def edit(root,kind,rng):
    parents={c:p for p in root.iter() for c in p}
    elems=list(root.iter())
    if kind=='text':
        leaves=[e for e in elems if tag(e) in ('text','tspan') and not list(e) and e.text
                and 3<=len(e.text.strip())<=24 and any(c.isalpha() for c in e.text)]
        counts=Counter(e.text for e in leaves)
        pairs=[(a,b) for a,b in itertools.combinations(leaves,2) if a.text!=b.text
               and counts[a.text]==counts[b.text]==1 and abs(len(a.text)-len(b.text))<=2]
        if not pairs:return None
        a,b=rng.choice(pairs);labels=[a.text,b.text];a.text,b.text=b.text,a.text
        return {'labels':labels,'operation':'swap two text strings'}
    def clean(e):
        cur=e
        while cur is not None:
            if cur.get('transform') or tag(cur) in ('defs','marker','clipPath'):return False
            cur=parents.get(cur)
        return True
    if kind=='box':
        boxes=[]
        for e in elems:
            if tag(e)!='rect' or not clean(e):continue
            try:
                w,h=number(e,'width'),number(e,'height');x,y=number(e,'x'),number(e,'y')
                if w>=20 and h>=15 and w/h<8:boxes.append(e)
            except ValueError:continue
        pairs=[]
        for a,b in itertools.combinations(boxes,2):
            if parents[a] is not parents[b]:continue
            wa,ha,wb,hb=[number(e,k) for e,k in [(a,'width'),(a,'height'),(b,'width'),(b,'height')]]
            if not .5<=wa/wb<=2 or not .5<=ha/hb<=2:continue
            if abs(number(a,'x')-number(b,'x'))+abs(number(a,'y')-number(b,'y'))<max(wa,ha,wb,hb):continue
            if a.attrib==b.attrib:continue
            pairs.append((a,b))
        if not pairs:return None
        a,b=rng.choice(pairs);old=[[a.get('x','0'),a.get('y','0')],[b.get('x','0'),b.get('y','0')]]
        for k in ('x','y'):av,bv=a.get(k,'0'),b.get(k,'0');a.set(k,bv);b.set(k,av)
        return {'labels':[str(x) for x in old],'operation':'swap two box positions; labels and other geometry stay fixed'}
    if kind=='arrow':
        arrows=[e for e in elems if tag(e) in ('line','path') and clean(e)
                and ('marker-end' in e.attrib or 'marker-end' in e.get('style','')) and endpoint(e)]
        pairs=[(a,b) for a,b in itertools.combinations(arrows,2) if parents[a] is parents[b]
               and sum(abs(x-y) for x,y in zip(endpoint(a),endpoint(b)))>=50]
        if not pairs:return None
        a,b=rng.choice(pairs);pa,pb=endpoint(a),endpoint(b)
        set_endpoint(a,pb);set_endpoint(b,pa)
        return {'labels':[str(pa),str(pb)],'operation':'swap endpoints of two marked connectors; preserve sources and arrow markers'}
    if kind=='mixed':
        a=edit(root,'text',rng);b=edit(root,'box',rng)
        return {'labels':a['labels']+b['labels'],'operation':'text swap plus box-position swap'} if a and b else None
    raise ValueError(kind)

def main():
    out=Path('outputs/svg_probe_v3');out.mkdir(exist_ok=True)
    if (out/'pairs.jsonl').exists():raise RuntimeError('Frozen manifest exists')
    ET.register_namespace('','http://www.w3.org/2000/svg');ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    tok=Tokenizer.from_file('outputs/e4b_broad/e4b_broad/checkpoint_pct_100/tokenizer.json')
    records=[];audit={}
    oldqa=json.loads(Path('outputs/svg_probe_v1/qa_selection.json').read_text())
    excluded={r['source_id'] for r in map(json.loads,Path('outputs/svg_probe_v1/pairs.jsonl').read_text().splitlines()) if r['id'] in oldqa['excluded']}
    for bench,manifest in MANIFESTS.items():
        rows=list(map(json.loads,manifest.read_text().splitlines()));rng=random.Random(20260905);rng.shuffle(rows)
        used=set()
        # Scarce connector edits first; each source contributes only once.
        for kind in ('arrow','mixed','box','text'):
            count=0;reject=Counter()
            for row in rows:
                if count>=12:break
                if row['id'] in used or row['id'] in excluded:continue
                try:
                    root=ET.fromstring(resolve_svg(row))
                    if any(tag(e) in ('script','image','foreignObject','animate','set') for e in root.iter()):
                        reject['non_native_or_dynamic']+=1;continue
                    svg0=ET.tostring(root,encoding='unicode');n0=len(tok.encode(svg0,add_special_tokens=False).ids)
                    if n0>6000:reject['length']+=1;continue
                    im0=None;accepted=None
                    for attempt in range(12):
                        other=copy.deepcopy(root);desc=edit(other,kind,rng)
                        if not desc:break
                        svg1=ET.tostring(other,encoding='unicode');n1=len(tok.encode(svg1,add_special_tokens=False).ids)
                        if n0!=n1:continue
                        if im0 is None:im0=render_pil(svg0,size=960).convert('RGB')
                        im1=render_pil(svg1,size=960).convert('RGB')
                        delta=np.max(abs(np.asarray(im0).astype(int)-np.asarray(im1).astype(int)),axis=2)>25
                        fraction=float(delta.mean())
                        if not .0005<=fraction<=.15:continue
                        accepted=(svg1,im1,desc,fraction);break
                    if accepted is None:reject['no_valid_visible_equal_length_edit']+=1;continue
                    svg1,im1,desc,fraction=accepted
                    key=bench+'_'+kind+'_'+hashlib.sha256(row['id'].encode()).hexdigest()[:10]
                    for i,svg,im in [(0,svg0,im0),(1,svg1,im1)]:
                        (out/f'{key}_{i}.svg').write_text(svg,encoding='utf-8');im.save(out/f'{key}_{i}.png')
                    records.append({'id':key,'source_id':row['id'],'bench':bench,'edit':kind,**desc,
                      'svg_tokens':n0,'changed_pixel_fraction':fraction,
                      **{f'{kind2}{i}':f'{key}_{i}.{ext}' for kind2,ext in [('svg','svg'),('image','png')] for i in (0,1)}})
                    count+=1;used.add(row['id'])
                    print(bench,kind,count,flush=True)
                except Exception as exc:reject[type(exc).__name__]+=1
            audit[bench+'_'+kind]={'accepted':count,'rejections':dict(reject)}
    for bench in MANIFESTS:
        for kind in ('arrow','mixed','box','text'):
            rr=[r for r in records if r['bench']==bench and r['edit']==kind]
            for i,r in enumerate(rr):r['shuffled_source_id']=rr[(i+1)%len(rr)]['id']
    content=''.join(json.dumps(r)+'\n' for r in records);(out/'pairs.jsonl').write_text(content)
    protocol=json.loads(Path('outputs/svg_probe_v2/protocol.json').read_text())
    protocol.update(n=len(records),audit=audit,manifest_sha256=hashlib.sha256(content.encode()).hexdigest(),
       selection='seeded existing 128-example subsets; one source per pair, no reuse across edit groups; up to 12 per bench/group; connector groups first',
       scope='text swaps, box-position swaps, marked connector endpoint swaps, text+box mixed edits; not general visual competence',
       edit='see each pair operation; edits may be semantically implausible; render determines match',
       eligibility='native static SVG, <=6000 target tokens, exactly matched token counts, 0.05%-15% changed raster pixels; visual QA before main inference',
       shuffled_assignment='cyclic next source in same bench and edit group',
       n_per_bench=dict(Counter(r['bench'] for r in records)),
       primary='both-correct accuracy and image-pair randomization test; paired checkpoint changes, 4000 source bootstrap draws; separate edit strata',
       difficulty='visibility and bounded change only; no performance-based difficulty filtering')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'qa_selection.json').write_text(json.dumps({'excluded':{},'review':'Pending raster inspection before main run'},indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
