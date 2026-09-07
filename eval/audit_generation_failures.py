"""Distinguish XML syntax failure from the stricter native-SVG profile."""
import json
from collections import Counter
from pathlib import Path
from structsvg_lib.svg_ops import extract_svg_blob,validate_svg

SPECS={
 'svg_diagrams':Path('outputs/generations/eval128_svg_diagrams_ctx8192_files/eval128_svg_diagrams_ctx8192'),
 'vfig_id':Path('outputs/generations/eval64_ctx8192_files/eval64_ctx8192'),
 'vfig_ood':Path('outputs/generations/eval128_vfig_ood_ctx8192_files/eval128_vfig_ood_ctx8192')}

def main():
    out={}
    for bench,path in SPECS.items():
        out[bench]={}
        for pct in [0,5,10,20,40,60,80,100]:
            tag='base_0pct' if pct==0 else f'pct_{pct:03d}'
            rows=[json.loads(s) for s in (path/f'{bench}_{tag}_prompt.jsonl').read_text().splitlines()]
            counts=Counter();errors=Counter()
            for r in rows:
                text=r.get('pred_text','');blob=extract_svg_blob(text) or text
                val=validate_svg(blob,try_render=False)
                if not val.parse_ok:counts['xml_parse_failure']+=1
                elif not val.ok:counts['parsed_but_profile_failure']+=1
                else:counts['parsed_and_profile_pass']+=1
                for e in val.errors:
                    errors[e.split(':')[0]]+=1
            out[bench][str(pct)]={'n':len(rows),'counts':dict(counts),'overlapping_errors':dict(errors)}
    dest=Path('outputs/analysis/svg_probe_v3/generation_failure_audit.json')
    dest.write_text(json.dumps(out,indent=2));print(json.dumps({b:v['100'] for b,v in out.items()},indent=2))

if __name__=='__main__':main()
