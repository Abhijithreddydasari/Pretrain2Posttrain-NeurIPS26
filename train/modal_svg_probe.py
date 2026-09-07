"""Isolated diagnostic entrypoint; preserves existing Modal training edits."""
from pathlib import Path
import modal
from train.modal_app import image, hf_secret, VOLUME_MOUNTS, vol_out

app=modal.App('svg-grounding-probe')
if modal.is_local():
    image=image.add_local_dir(str(Path(__file__).resolve().parents[1]/'outputs/svg_probe_v3'),remote_path='/root/svg_probe')

@app.function(image=image,gpu='A100-80GB',timeout=7200,secrets=[hf_secret],volumes=VOLUME_MOUNTS)
def run_probe(limit:int=0,pcts:str='0,20,100',run_name:str='svg_probe_v3'):
    import subprocess,sys
    command=[sys.executable,'-m','eval.score_svg_probe','--data','/root/svg_probe',
             '--out',f'/vol/out/{run_name}','--pcts',pcts,'--limit',str(limit)]
    try:
        result=subprocess.run(command,check=False)
    finally:
        vol_out.commit()
    if result.returncode: raise RuntimeError(f'Probe failed: {result.returncode}')
    return {'out':f'/vol/out/{run_name}'}

@app.local_entrypoint()
def main(limit:int=0,pcts:str='0,20,100',run_name:str='svg_probe_v3'):
    print(run_probe.remote(limit=limit,pcts=pcts,run_name=run_name))
