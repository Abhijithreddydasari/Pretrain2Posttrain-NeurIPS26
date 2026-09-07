"""Preserve inspected pairs, restore native default SVG namespace, freeze v2."""
import hashlib,json,shutil
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from structsvg_lib.svg_ops import render_pil

def main():
    src=Path('outputs/svg_probe_v1');dst=Path('outputs/svg_probe_v2')
    if dst.exists():raise RuntimeError('Final probe already exists')
    dst.mkdir()
    ET.register_namespace('','http://www.w3.org/2000/svg')
    ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    tokenizer=Tokenizer.from_file('outputs/e4b_broad/e4b_broad/checkpoint_pct_100/tokenizer.json')
    qa=json.loads((src/'qa_selection.json').read_text())
    rows=[json.loads(s) for s in (src/'pairs.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['id'] not in qa['excluded']]
    for row in rows:
        lengths=[]
        for i in (0,1):
            text=ET.tostring(ET.fromstring((src/row[f'svg{i}']).read_text(encoding='utf-8')),encoding='unicode')
            lengths.append(len(tokenizer.encode(text,add_special_tokens=False).ids))
            (dst/row[f'svg{i}']).write_text(text,encoding='utf-8')
            before=np.asarray(Image.open(src/row[f'image{i}']).convert('RGB'))
            after=np.asarray(render_pil(text,size=960).convert('RGB'))
            assert np.array_equal(before,after), 'Namespace normalization changed pixels'
            shutil.copy2(src/row[f'image{i}'],dst/row[f'image{i}'])
        assert lengths[0]==lengths[1] and max(lengths)<=6000
        row['svg_tokens']=lengths[0]
    for bench in ('svg_diagrams','vfig_id'):
        rr=[r for r in rows if r['bench']==bench]
        for i,r in enumerate(rr):r['shuffled_source_id']=rr[(i+1)%len(rr)]['id']
    content=''.join(json.dumps(r)+'\n' for r in rows)
    (dst/'pairs.jsonl').write_text(content,encoding='utf-8')
    shutil.copy2(src/'qa_selection.json',dst/'qa_selection.json')
    protocol=json.loads((src/'protocol.json').read_text())
    protocol.update(manifest_sha256=hashlib.sha256(content.encode()).hexdigest(),
       candidate_serialization='default native SVG namespace; pixel identity against inspected v1 rasters verified',
       n=61,n_per_bench={'svg_diagrams':29,'vfig_id':32},
       conditions='2 matching images x 2 SVGs; white image x 2 SVGs; next source image pair within same bench x 2 SVGs',
       shuffled_assignment='fixed cyclic next source within bench; identical two candidates; 4 shuffled scores',
       primary='both margins >1e-6; paired correct-minus-shuffled both-correct and paired checkpoint change; source bootstrap 4000',
       inference_precision='bf16 base and unmerged LoRA; score logits converted to float32; SVG targets only',
       pilot='2 pipeline-validation pairs scored in v1 before namespace normalization; final source selection unchanged by scores',
       caution='61 eligible examples, one training run; secondary numerical margin; no comparison of raw magnitudes across different metrics')
    (dst/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps(protocol,indent=2))

if __name__=='__main__':main()
