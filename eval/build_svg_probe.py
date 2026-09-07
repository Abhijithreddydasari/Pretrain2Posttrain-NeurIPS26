"""Freeze a held-out text-location binding probe before checkpoint scoring."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from tokenizers import Tokenizer
from structsvg_lib.svg_ops import render_pil
from train.data_utils import resolve_svg

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    'svg_diagrams': ROOT/'outputs/metrics/eval128_svg_diagrams_ctx8192_remote/eval128_svg_diagrams_ctx8192/eval_subset_svg_diagrams_128_seed42.jsonl',
    'vfig_id': ROOT/'outputs/metrics/eval64_ctx8192_remote/eval64_ctx8192/eval_subset_vfig_id_128_seed42.jsonl',
}

def dumps(root):
    return ET.tostring(root, encoding='unicode')

def build_pair(svg, rng, tokenizer):
    root = ET.fromstring(svg)
    # Leaf text/tspan substitutions preserve all geometry and styling.
    leaves = [e for e in root.iter() if e.tag.split('}')[-1] in ('text','tspan')
              and not list(e) and e.text and 2 <= len(e.text.strip()) <= 24]
    counts = Counter(e.text for e in leaves)
    pairs = [(a,b) for a,b in itertools.combinations(leaves,2)
             if a.text != b.text and counts[a.text] == counts[b.text] == 1
             and abs(len(a.text)-len(b.text)) <= 2]
    rng.shuffle(pairs)
    original = dumps(root)
    n0 = len(tokenizer.encode(original, add_special_tokens=False).ids)
    if n0 > 6000:
        return None, 'over_6000_tokens'
    im0 = render_pil(original, size=960).convert('RGB')
    for a,b in pairs[:30]:
        ta,tb = a.text,b.text
        a.text,b.text = tb,ta
        edited = dumps(root)
        a.text,b.text = ta,tb
        n1 = len(tokenizer.encode(edited, add_special_tokens=False).ids)
        if n0 != n1:
            continue
        im1 = render_pil(edited, size=960).convert('RGB')
        arr0,arr1 = np.asarray(im0).astype(int),np.asarray(im1).astype(int)
        if arr0.shape != arr1.shape:
            continue
        changed = np.max(abs(arr0-arr1),axis=2)>25
        fraction = float(changed.mean())
        if not 0.0002 <= fraction <= 0.05:
            continue
        return {'svg0':original,'svg1':edited,'image0':im0,'image1':im1,
                'labels':[ta,tb],'svg_tokens':n0,'changed_pixel_fraction':fraction}, None
    return None, 'no_equal_token_visible_label_swap'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/svg_probe_v1')
    ap.add_argument('--per-bench',type=int,default=32)
    args=ap.parse_args()
    if (args.out/'pairs.jsonl').exists():
        raise SystemExit('Frozen manifest already exists; use a new output directory.')
    args.out.mkdir(parents=True,exist_ok=True)
    tokenizer=Tokenizer.from_file(str(ROOT/'outputs/e4b_broad/e4b_broad/checkpoint_pct_100/tokenizer.json'))
    records=[]; audit={}
    for bench,path in MANIFESTS.items():
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        rng=random.Random(20260905); rng.shuffle(rows)
        rejected=Counter(); accepted=0
        for row in rows:
            if accepted >= args.per_bench: break
            try:
                pair,reason=build_pair(resolve_svg(row),rng,tokenizer)
            except Exception as exc:
                rejected[type(exc).__name__]+=1
                continue
            if pair is None:
                rejected[reason]+=1; continue
            key=bench+'_'+hashlib.sha256(row['id'].encode()).hexdigest()[:12]
            for i in (0,1):
                (args.out/f'{key}_{i}.svg').write_text(pair[f'svg{i}'],encoding='utf-8')
                pair[f'image{i}'].save(args.out/f'{key}_{i}.png')
            records.append({'id':key,'source_id':row['id'],'bench':bench,'edit':'label_swap',
                            'labels':pair['labels'],'svg_tokens':pair['svg_tokens'],
                            'changed_pixel_fraction':pair['changed_pixel_fraction'],
                            **{f'{kind}{i}':f'{key}_{i}.{ext}' for kind,ext in [('svg','svg'),('image','png')] for i in (0,1)}})
            accepted+=1
        audit[bench]={'available':len(rows),'accepted':accepted,'rejected_before_quota':dict(rejected)}
    content=''.join(json.dumps(r)+'\n' for r in records)
    (args.out/'pairs.jsonl').write_text(content,encoding='utf-8')
    protocol={'seed':20260905,'manifest_sha256':hashlib.sha256(content.encode()).hexdigest(),
      'audit':audit,'selection':'fixed existing 128-example subsets; independent of predictions; seeded source order; one pair per source',
      'edit':'swap two unique leaf text labels, similar character lengths; all geometry and styling fixed',
      'eligibility':'equal SVG token counts; <=6000 SVG tokens; 0.02%-5% raster pixels differ by >25 RGB units',
      'score':'mean log likelihood on SVG tokens only, excluding prompt and turn terminator',
      'pcts':[0,20,100],'conditions':'2 images x 2 SVGs plus identical white image x 2 SVGs',
      'primary':'both image-specific candidate margins > 1e-6; ties fail; source bootstrap 4000 draws',
      'secondary':'interaction G; per-image accuracy; blank preference; paired checkpoint changes',
      'scope':'text-location binding on eligible re-rendered SVGs; not topology, general vision, or free reconstruction',
      'no_result_based_exclusion':True}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(json.dumps(protocol,indent=2))

if __name__=='__main__': main()
