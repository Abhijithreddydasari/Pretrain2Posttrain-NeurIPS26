import math
import random
from xml.etree import ElementTree as ET
import numpy as np
from eval.svg_probe_figures import ci

def test_image_independent_preferences_cancel():
    # Any fixed code preference must cancel across the image swap.
    for a,b in [(-.3,-.6),(-5,-1),(-.2,-.2)]:
        assert (a-b)+(b-a)==0
        assert not ((a-b)>1e-6 and (b-a)>1e-6)

def test_positive_interaction_does_not_imply_both_correct():
    d0,d1=.6,-.1
    assert d0+d1>0
    assert not (d0>0 and d1>0)

def test_bootstrap_constant_difference():
    result=ci([1]*29)
    assert result=={'mean':1.0,'lo':1.0,'hi':1.0,'n':29}

def test_frozen_pairs_preserve_xml_structure_and_label_multiset():
    import json
    from pathlib import Path
    root=Path('outputs/svg_probe_v1')
    if not (root/'pairs.jsonl').exists():
        import pytest
        pytest.skip('Local probe artifact not distributed with unit tests')
    for line in (root/'pairs.jsonl').read_text().splitlines():
        row=json.loads(line)
        a=list(ET.fromstring((root/row['svg0']).read_text(encoding='utf-8')).iter())
        b=list(ET.fromstring((root/row['svg1']).read_text(encoding='utf-8')).iter())
        assert len(a)==len(b)
        assert all(x.tag==y.tag and x.attrib==y.attrib and x.tail==y.tail for x,y in zip(a,b))
        assert sorted(x.text or '' for x in a)==sorted(x.text or '' for x in b)
        assert sum(x.text!=y.text for x,y in zip(a,b))==2

def test_arrow_endpoint_edit_preserves_origin_and_marker():
    from eval.build_multiedit_probe import endpoint,set_endpoint
    e=ET.fromstring('<path d="M 10 20 C 30 40 50 60 70 80" marker-end="url(#a)"/>')
    assert endpoint(e)==(70,80)
    set_endpoint(e,(90,100))
    assert e.get('d')=='M 10 20 C 30 40 50 60 90 100'
    assert e.get('marker-end')=='url(#a)'

def test_unsupported_path_endpoint_is_rejected():
    from eval.build_multiedit_probe import endpoint
    assert endpoint(ET.fromstring('<path d="m 0 0 l 10 20"/>')) is None
    assert endpoint(ET.fromstring('<path d="M 0 0 H 10 V 20"/>')) is None

def test_teacher_forced_score_masks_prompt_and_shifts_logits():
    import torch
    from types import SimpleNamespace
    from PIL import Image
    from eval.score_svg_probe import score
    class Processor:
        tokenizer=SimpleNamespace(decode=lambda ids:'decoded')
        def apply_chat_template(self,*args,**kwargs):return 'PREFIX'
        def __call__(self,*,text,images,return_tensors):
            return {'input_ids':torch.tensor([[0,1]] if text[0]=='PREFIX' else [[0,1,2,3]])}
    logits=torch.zeros(1,4,4)
    logits[0,1,2]=2;logits[0,2,3]=3
    class Model:
        device='cpu'
        def __call__(self,**kwargs):return SimpleNamespace(logits=logits)
    engine=SimpleNamespace(processor=Processor(),model=Model(),prompt='Use {viewbox}')
    result=score(engine,Image.new('RGB',(1,1)),'<svg viewBox="0 0 1 1"></svg>')
    expected=torch.log_softmax(logits[0,1],0)[2]+torch.log_softmax(logits[0,2],0)[3]
    assert result['prompt_tokens']==2 and result['target_tokens']==2
    assert abs(result['sum_logp']-float(expected))<1e-6

def test_probe_summary_accuracy_and_assignment_null():
    from eval.analyze_svg_probe import summary
    rows=[{'id':'a','d0':.2,'d1':.3,'G':.5,'both_correct':True,'shuffled_both_correct':False,'blank_margin':.1},
          {'id':'b','d0':.2,'d1':-.3,'G':-.1,'both_correct':False,'shuffled_both_correct':False,'blank_margin':.1}]
    s=summary(rows,{r['id']:r for r in rows})
    assert s['both_correct']['mean']==.5
    assert s['one_image_accuracy']['mean']==.75
    assert s['assignment_randomization_p']==.5
    assert s['delta_both_vs_base']['mean']==0
