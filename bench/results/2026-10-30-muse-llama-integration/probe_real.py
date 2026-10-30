"""Bounded full-checkpoint validation through the integrated Muse executable."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np

sys.path.insert(0, '/src')
from serve.server import StrataEngine, apply_ready_timeout
from strata_control import muse

out = Path('/out/real')
out.mkdir(exist_ok=False)
binary = Path('/out/build-cuda/bin/strata-muse-llama')
model = Path('/model/Muse-Spark-Open-816B-A42B-UD-IQ1_S-00001-of-00005.gguf')
args = ['--model', str(model), '--max-context', '128', *muse.backend_args('llama-cuda', [0,1,2], 10240, 12)]
report = {'status':'running','binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
          'args':args,'trials':[], 'limits':'BOS diagnostic, short generation, not normal-chat quality; cache retained across resident requests. No speculation.'}

def save():
    (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

engine = None
try:
    save()
    started=time.monotonic()
    with (out/'logits.stderr').open('w') as stderr, (out/'logits.stdout').open('w') as stdout:
        result=subprocess.run([str(binary),*args,'--tokens','200000','--steps','2',
                               '--logits-file',str(out/'bos.json')],stdout=stdout,stderr=stderr,timeout=600)
    assert result.returncode == 0, result.returncode
    bos=json.loads((out/'bos.json').read_text())
    assert bos['tokens']==[200000] and bos['generated_tokens']==[15,220]
    actual=np.asarray(bos['logits'],dtype=np.float32).reshape(-1)
    # Select the full vector, not the prompt-token ID blob.
    prior=next(p for p in Path('/prior/cuda-probe/logits').glob('*.bin') if not p.name.endswith('-tokens.bin'))
    expected=np.fromfile(prior,dtype=np.float32)
    assert actual.shape==expected.shape==(202048,) and np.isfinite(actual).all()
    x,y=actual.astype(np.float64),expected.astype(np.float64)
    report['bos']={'wall_seconds_including_load':time.monotonic()-started,'argmax':int(actual.argmax()),
                   'max_abs_error_vs_prior_cuda':float(np.abs(x-y).max()),
                   'rmse_vs_prior_cuda':float(np.sqrt(np.mean((x-y)**2))),
                   'cosine_vs_prior_cuda':float(x@y/np.sqrt((x@x)*(y@y))),
                   'elementwise_atol_rtol':2e-5,'generated_tokens':bos['generated_tokens']}
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    save()
    bundle=muse.inspect_model(model,128,backend='llama-cuda')
    sidecars=muse.prepare_tokenizer(bundle,out/'tokenizer')
    cfg={'exe':str(binary),'args':args,'arch':'muse','backend':'llama-cuda','tokenizer':str(sidecars),'parallel':1,'engine_ready_s':600}
    muse.validate_config(cfg)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    apply_ready_timeout(cfg)
    started=time.monotonic()
    engine=StrataEngine(str(binary),args,log=str(out/'resident.stderr'))
    report['resident_load_seconds']=time.monotonic()-started
    report['info']=engine.info
    assert engine.info['engine']=='muse-llama' and engine.max_context==128
    report['gpu_allocations_after_ready']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.free','--format=csv'],text=True)
    save()
    for trial in range(3):
        start=time.monotonic()
        tokens=[t for t in engine.generate([200000],32,{'temperature':0},threading.Event()) if t is not None]
        last=dict(engine.last)
        assert len(tokens)==last['generated']==32 and tokens[:2]==[15,220] and last['finish']=='length'
        if report['trials']: assert tokens==report['trials'][0]['tokens']
        item={'trial':trial,'tokens':tokens,'native_counters':last,'request_wall_s':time.monotonic()-start,
              'decode_forward_steps':len(tokens)-1,'actual_decode_steps_s':(len(tokens)-1)*1000/last['decode_ms']}
        report['trials'].append(item); save(); print(json.dumps(item),flush=True)
    cancel=threading.Event(); seen=[]
    for token in engine.generate([200000],32,{'temperature':0},cancel):
        if token is not None: seen.append(token); cancel.set()
    report['cancellation']={'tokens':seen,'counters':dict(engine.last)}
    assert seen==[15] and engine.last['finish']=='cancel'
    replay=[t for t in engine.generate([200000],2,{'temperature':0},threading.Event()) if t is not None]
    assert replay==[15,220] and engine.last['finish']=='length'
    report['after_cancel_replay']=replay
    proc=engine.proc
    engine.close(); engine=None
    report['resident_exit_code']=proc.wait(timeout=15)
    assert report['resident_exit_code']==0
    report['status']='completed'
except BaseException as e:
    report.update(status='failed',error_type=type(e).__name__,error=str(e)); raise
finally:
    if engine is not None: engine.close()
    save()
    print(json.dumps(report),flush=True)
