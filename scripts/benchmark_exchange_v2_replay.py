"""Engineering-only exact replay forward/backward timing, without optimization."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import technician_exchange_v2_experiment as x
from ht_pdm_fjsp.maintenance_dispatch import DispatchState


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    out=args.output_root/(x.VERSION+'_replay_benchmark_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'benchmark_source.py');snap=out/'source_snapshot';snap.mkdir()
    for path in Path(x.__file__).parent.glob('*.py'):shutil.copy2(path,snap/path.name)
    shutil.copy2(x.PLAN,out/'protocol.md');shutil.copy2(x.REGISTRY,out/'seed_registry.json')
    seals={str(p.relative_to(out)):x.sha(p) for p in out.rglob('*') if p.is_file()}
    manifest=dict(status='RUNNING',classification='ENGINEERING_ONLY_NO_TRAINING',
        optimizer_steps=0,parameter_updates=0,scientific_test_opened=False,
        threads=1,warmup=5,trials=20,source_seal=seals)
    x.write_json(out/'manifest.json',manifest);results=[];c=x.settings('smoke');started=time.monotonic()
    try:
        for n,b in tqdm([(n,b) for n in (4,32) for b in (128,256)],desc='Replay equivalence/timing'):
            observations=[]
            for i in range(2):
                env=x.ResourceEnv(c['development_seeds'][i],n,'exchange_ppo');env.reset()
                env.env.state=DispatchState(tuple(3+(j+i)%3 for j in range(n)),(False,)*n,
                    (0,)*n,(0,)*(n//2),(-1,)*(n//2))
                observations.append(env.observe())
            batch=x.learner.tensors([observations[i%2] for i in range(b)])
            model=x.make_model('exchange_ppo',n,c,c['train_seeds'][0]);before=x.learner.parameter_digest(model)
            actions=model.sample(batch,deterministic=True)[0]
            coefficient=torch.linspace(.5,1.5,b)[:,None]

            def step(literal):
                model.zero_grad(set_to_none=True)
                _,log,entropy,value=model.trace(batch,recorded=actions,literal=literal)
                loss=-(log*coefficient).sum(1).mean()-.01*entropy.mean()+.5*(value/x.value_unit(n)).square().mean()
                loss.backward()
                return tuple(v.detach() for v in (log,entropy,value))

            fast=step(False);fast_grad={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
            literal=step(True);grad_error=0.
            for a,z in zip(fast,literal):assert torch.allclose(a,z,atol=2e-5,rtol=0)
            for k,p in model.named_parameters():
                if p.grad is None:assert k not in fast_grad;continue
                assert torch.allclose(fast_grad[k],p.grad,atol=2e-5,rtol=2e-5),k
                grad_error=max(grad_error,float((fast_grad[k]-p.grad).abs().max()))
            del fast_grad
            for _ in range(5):step(True);step(False)
            times={True:[],False:[]}
            for trial in range(20):
                for mode in ((True,False) if trial%2==0 else (False,True)):
                    start=time.perf_counter();step(mode);times[mode].append(time.perf_counter()-start)
            assert before==x.learner.parameter_digest(model)
            med={str(mode):float(np.median(t)) for mode,t in times.items()}
            results.append(dict(n=n,minibatch=b,arm='exchange_ppo',parameter_sha256=before,
                parameters=sum(p.numel() for p in model.parameters()),
                max_log_error=float((fast[0]-literal[0]).abs().max()),max_gradient_error=grad_error,
                literal_seconds=times[True],batched_seconds=times[False],
                literal_median_seconds=med['True'],batched_median_seconds=med['False'],
                speedup=med['True']/med['False']))
            x.write_json(out/'results.json',results)
        assert all(x.sha(out/name)==digest for name,digest in seals.items())
        for p in Path(x.__file__).parent.glob('*.py'):assert x.sha(p)==seals['source_snapshot/'+p.name]
        assert x.sha(x.PLAN)==seals['protocol.md'] and x.sha(x.REGISTRY)==seals['seed_registry.json']
        manifest.update(status='COMPLETED',cases=len(results),elapsed_seconds=time.monotonic()-started,
            numerical_equivalence='PASS',frozen_parameters='PASS')
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        manifest['output_bindings']={str(p.relative_to(out)):dict(sha256=x.sha(p),bytes=p.stat().st_size)
            for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        x.write_json(out/'manifest.json',manifest)
    print(out)


if __name__=='__main__':main()
