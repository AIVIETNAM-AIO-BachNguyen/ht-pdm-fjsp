"""Retrospective V14 training-Q versus frozen selected-action probe."""
from __future__ import annotations
import argparse,csv,hashlib,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_framework_v1 import catalogue,configs
from ht_pdm_fjsp.maintenance_framework_v14_campaign import legal_present,load_checkpoint
from ht_pdm_fjsp.maintenance_requests import state_from_dict
ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'research_result/framework_v14_dev_20261020T185303Z'
SOURCE_AUDIT=ROOT/'research_result/framework_v14_audit_20261020T200552Z/independent_audit.json'
PLAN=ROOT/'research_result/v14_training_credit_probe_plan_vi.md'
CASES=('interchangeable_scale1','reciprocal_scale2');TRAIN=tuple(range(72600000,72600005))
ACTIONS=catalogue();WAIT=ACTIONS.index(())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def write(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize(rows):
    out={}
    for case in CASES:
        cell=[r for r in rows if r['case']==case]
        out[case]={}
        for label,subset in (('early', [r for r in cell if r['early']]),('all',cell)):
            if not subset:
                out[case][label]=dict(count=0);continue
            def mean(pred):return float(np.mean([pred(r) for r in subset]))
            margin=np.array([r['q_margin'] for r in subset],dtype=float)
            positive=[r for r in subset if r['q_margin']>=5]
            negative=[r for r in subset if r['q_margin']<=-5]
            out[case][label]=dict(count=len(subset),seeds=len({r['train_seed'] for r in subset}),
                q_wait_better_fraction=mean(lambda r:r['q_margin']>0),
                q_wait_better_5_fraction=mean(lambda r:r['q_margin']>=5),
                q_margin_mean=float(margin.mean()),q_margin_quantiles=np.quantile(margin,[.1,.25,.5,.75,.9]).tolist(),
                selected_wait_fraction=mean(lambda r:r['selected_action']==WAIT),
                selected_service_given_wait_q5=(float(np.mean([r['selected_action']!=WAIT for r in positive])) if positive else None),
                selected_wait_given_service_q5=(float(np.mean([r['selected_action']==WAIT for r in negative])) if negative else None),
                selected_regret_mean=mean(lambda r:r['selected_regret']),
                selected_regret_5_fraction=mean(lambda r:r['selected_regret']>=5),
                by_seed={str(seed):dict(count=sum(r['train_seed']==seed for r in subset),
                      wait_q5=sum(r['train_seed']==seed and r['q_margin']>=5 for r in subset),
                      selected_service_when_wait_q5=sum(r['train_seed']==seed and r['q_margin']>=5 and r['selected_action']!=WAIT for r in subset))
                      for seed in TRAIN})
    return out
def run(profile,out):
    assert profile in ('smoke','dev');out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1);start=time.monotonic()
    def deadline():
        if time.monotonic()-start>600:raise TimeoutError('V14 training-credit probe deadline')
    manifest=dict(status='RUNNING',profile=profile,started_utc=datetime.now(timezone.utc).isoformat())
    save(out/'manifest.json',manifest)
    try:
        if profile=='dev':
            files=['research_result/probe_v14_training_credit.py','research_result/verify_v14_training_credit.py',
                   'research_result/v14_training_credit_probe_plan_vi.md']
            dirty=subprocess.check_output(['git','diff','--name-only','HEAD','--',*files],cwd=ROOT,text=True).splitlines()
            unknown=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*files],cwd=ROOT,text=True).splitlines()
            if dirty or unknown:raise RuntimeError(f'uncommitted source: {dirty+unknown}')
        source=json.loads((SOURCE/'manifest.json').read_text());audit=json.loads(SOURCE_AUDIT.read_text())
        assert source['status']=='COMPLETED' and audit['status']=='PASS' and Path(audit['run']).resolve()==SOURCE
        for name,digest in source['artifact_hashes'].items():assert sha(SOURCE/name)==digest
        selection=json.loads((SOURCE/'selection.json').read_text())
        train=TRAIN if profile=='dev' else TRAIN[:1]
        models={};checkpoint_hashes={}
        for seed in train:
            item=selection['competition'][str(seed)];cp=SOURCE/item['checkpoint']
            assert sha(cp)==item['sha256'];checkpoint_hashes[str(seed)]=item['sha256']
            models[seed]=load_checkpoint(cp)
        raw=read(SOURCE/'counterfactual_states.csv');rows=[]
        for i,r in tqdm(enumerate(raw),total=len(raw),desc='V14 logged Q/selected policy',unit='row'):
            deadline()
            if r['arm']!='competition' or r['case'] not in CASES:continue
            seed=int(r['train_seed']);iteration=int(r['iteration'])
            if seed not in train or (profile=='smoke' and iteration!=1):continue
            cfg=configs()[r['case']];state=state_from_dict(json.loads(r['original_state']))
            _,_,legal,obs=legal_present(cfg,state,int(r['tick']))
            assert list(legal)==json.loads(r['legal'])
            q={int(k):float(v) for k,v in json.loads(r['scores']).items()}
            assert set(q)==set(legal)
            assert WAIT in q and len(q)>1
            service=min((a for a in legal if a!=WAIT),key=lambda a:q[a])
            with torch.no_grad():
                logits=models[seed](torch.as_tensor(obs).reshape(1,56))[0]
                masked=torch.full_like(logits,-1e9);masked[list(legal)]=logits[list(legal)]
                probs=torch.softmax(masked,dim=0)
            chosen=int(torch.argmax(probs));source_action=int(r['source_action']);assert source_action in legal
            rows.append(dict(source_row=i,case=r['case'],train_seed=seed,iteration=iteration,
                physical_seed=int(r['physical_seed']),tick=int(r['tick']),early=int(r['tick'])<16,
                legal=json.dumps(list(legal)),source_action=source_action,source_wait=source_action==WAIT,
                q_wait=q[WAIT],q_best_service=q[service],q_margin=q[service]-q[WAIT],
                q_wait_rank=1+sum(v<q[WAIT] for v in q.values()),
                selected_action=chosen,selected_wait_probability=float(probs[WAIT]),
                selected_regret=q[chosen]-min(q.values())))
        assert rows
        write(out/'rows.csv',rows)
        summary=dict(status='COMPLETED',profile=profile,rows=len(rows),statistics=summarize(rows))
        save(out/'summary.json',summary)
        manifest.update(status='COMPLETED',elapsed_seconds=time.monotonic()-start,
            finished_utc=datetime.now(timezone.utc).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_run=str(SOURCE),source_manifest_sha256=sha(SOURCE/'manifest.json'),source_audit_sha256=sha(SOURCE_AUDIT),
            source_rows_sha256=sha(SOURCE/'counterfactual_states.csv'),source_selection_sha256=sha(SOURCE/'selection.json'),
            plan_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),
            verifier_sha256=sha(ROOT/'research_result/verify_v14_training_credit.py'),
            checkpoint_hashes=checkpoint_hashes,count=len(rows),
            artifact_hashes={name:sha(out/name) for name in ('rows.csv','summary.json')})
        save(out/'manifest.json',manifest)
        return out
    except Exception as exc:
        manifest.update(status='INCOMPLETE' if isinstance(exc,TimeoutError) else 'FAILED',
                        error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start)
        save(out/'manifest.json',manifest);raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=('smoke','dev'),required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    print(json.dumps(dict(status='COMPLETED',output_dir=str(run(a.profile,a.output_dir)))))
if __name__=='__main__':main()
