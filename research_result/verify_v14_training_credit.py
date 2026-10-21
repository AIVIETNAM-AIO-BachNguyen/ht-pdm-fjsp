"""Independent source-row and selected-policy audit for V14 training-credit probe."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp.maintenance_framework_v1 import catalogue,configs,vector_observation
from ht_pdm_fjsp.maintenance_framework_v14_campaign import load_checkpoint
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import legal_matchings,state_from_dict,submit_requests
ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT/'research_result/framework_v14_dev_20261020T185303Z'
SOURCE_AUDIT=ROOT/'research_result/framework_v14_audit_20261020T200552Z/independent_audit.json'
PLAN=ROOT/'research_result/v14_training_credit_probe_plan_vi.md'
CASES=('interchangeable_scale1','reciprocal_scale2');TRAIN=tuple(range(72600000,72600005))
ACTIONS=catalogue();WAIT=ACTIONS.index(())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def close(a,b,label):
    if not np.isclose(float(a),float(b),rtol=0,atol=1e-7):raise AssertionError(f'{label}: {a} != {b}')
def audit(run):
    start=time.monotonic();torch.set_num_threads(1)
    def deadline():
        if time.monotonic()-start>600:raise TimeoutError('independent V14 credit audit')
    m=json.loads((run/'manifest.json').read_text());profile=m['profile']
    assert m['status']=='COMPLETED' and profile in ('smoke','dev')
    assert sha(PLAN)==m['plan_sha256']
    assert sha(ROOT/'research_result/probe_v14_training_credit.py')==m['script_sha256']
    assert sha(Path(__file__))==m['verifier_sha256']
    assert Path(m['source_run']).resolve()==SOURCE
    assert sha(SOURCE/'manifest.json')==m['source_manifest_sha256']
    assert sha(SOURCE_AUDIT)==m['source_audit_sha256']
    assert sha(SOURCE/'counterfactual_states.csv')==m['source_rows_sha256']
    assert sha(SOURCE/'selection.json')==m['source_selection_sha256']
    source=json.loads((SOURCE/'manifest.json').read_text());source_audit=json.loads(SOURCE_AUDIT.read_text())
    assert source['status']=='COMPLETED' and source_audit['status']=='PASS'
    for name,digest in source['artifact_hashes'].items():assert sha(SOURCE/name)==digest
    for name,digest in m['artifact_hashes'].items():assert sha(run/name)==digest
    selection=json.loads((SOURCE/'selection.json').read_text())
    train=TRAIN if profile=='dev' else TRAIN[:1]
    models={}
    for seed in train:
        item=selection['competition'][str(seed)];cp=SOURCE/item['checkpoint']
        assert sha(cp)==item['sha256']==m['checkpoint_hashes'][str(seed)]
        models[seed]=load_checkpoint(cp)
    raw=read(SOURCE/'counterfactual_states.csv');saved=read(run/'rows.csv')
    expected=[(i,r) for i,r in enumerate(raw) if r['arm']=='competition' and r['case'] in CASES and
              int(r['train_seed']) in train and (profile=='dev' or int(r['iteration'])==1)]
    assert len(expected)==len(saved)==m['count']
    for (i,r),row in tqdm(zip(expected,saved),total=len(saved),desc='Independent V14 training-credit rows',unit='row'):
        deadline();case=r['case'];seed=int(r['train_seed']);tick=int(r['tick'])
        assert int(row['source_row'])==i and row['case']==case and int(row['train_seed'])==seed
        assert int(row['iteration'])==int(r['iteration']) and int(row['physical_seed'])==int(r['physical_seed'])
        assert int(row['tick'])==tick and row['early']==str(tick<16)
        cfg=configs()[case];state=state_from_dict(json.loads(r['original_state']))
        submissions=request_action(cfg,state,tick,'threshold')
        admitted=submit_requests(cfg,state,tick,submissions)
        legal=[a for a,matching in enumerate(ACTIONS) if matching in set(legal_matchings(cfg,admitted))]
        assert legal==json.loads(r['legal'])==json.loads(row['legal'])
        q={int(k):float(v) for k,v in json.loads(r['scores']).items()};assert set(q)==set(legal)
        service=min((a for a in legal if a!=WAIT),key=lambda a:q[a]);source_action=int(r['source_action'])
        assert int(row['source_action'])==source_action and row['source_wait']==str(source_action==WAIT)
        close(row['q_wait'],q[WAIT],'q wait');close(row['q_best_service'],q[service],'q service')
        close(row['q_margin'],q[service]-q[WAIT],'q margin')
        assert int(row['q_wait_rank'])==1+sum(v<q[WAIT] for v in q.values())
        obs=vector_observation(cfg,admitted,tick)
        with torch.no_grad():
            logits=models[seed](torch.as_tensor(obs).reshape(1,56))[0]
            masked=torch.full_like(logits,-1e9);masked[legal]=logits[legal]
            probs=torch.softmax(masked,dim=0)
        chosen=int(torch.argmax(probs))
        assert int(row['selected_action'])==chosen
        close(row['selected_wait_probability'],probs[WAIT],'selected wait probability')
        close(row['selected_regret'],q[chosen]-min(q.values()),'selected regret')
    summary=json.loads((run/'summary.json').read_text())
    assert summary['status']=='COMPLETED' and summary['profile']==profile and summary['rows']==len(saved)
    for case in CASES:
        cell=[r for r in saved if r['case']==case]
        for label,subset in (('early',[r for r in cell if r['early']=='True']),('all',cell)):
            item=summary['statistics'][case][label]
            if not subset:assert item['count']==0;continue
            assert item['count']==len(subset) and item['seeds']==len({r['train_seed'] for r in subset})
            margin=np.array([float(r['q_margin']) for r in subset]);selected=np.array([int(r['selected_action']) for r in subset])
            close(item['q_wait_better_fraction'],np.mean(margin>0),'wait better')
            close(item['q_wait_better_5_fraction'],np.mean(margin>=5),'wait better 5')
            close(item['q_margin_mean'],margin.mean(),'margin mean')
            for got,want in zip(item['q_margin_quantiles'],np.quantile(margin,[.1,.25,.5,.75,.9])):close(got,want,'quantile')
            close(item['selected_wait_fraction'],np.mean(selected==WAIT),'selected wait')
            positive=selected[margin>=5];negative=selected[margin<=-5]
            if len(positive):close(item['selected_service_given_wait_q5'],np.mean(positive!=WAIT),'service despite wait')
            else:assert item['selected_service_given_wait_q5'] is None
            if len(negative):close(item['selected_wait_given_service_q5'],np.mean(negative==WAIT),'wait despite service')
            else:assert item['selected_wait_given_service_q5'] is None
            regret=np.array([float(r['selected_regret']) for r in subset])
            close(item['selected_regret_mean'],regret.mean(),'regret mean')
            close(item['selected_regret_5_fraction'],np.mean(regret>=5),'regret 5')
            for seed in TRAIN:
                x=[r for r in subset if int(r['train_seed'])==seed];b=item['by_seed'][str(seed)]
                assert b['count']==len(x)
                assert b['wait_q5']==sum(float(r['q_margin'])>=5 for r in x)
                assert b['selected_service_when_wait_q5']==sum(float(r['q_margin'])>=5 and int(r['selected_action'])!=WAIT for r in x)
    return dict(status='PASS',run=str(run),profile=profile,rows=len(saved),wall_seconds=time.monotonic()-start)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();out=a.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);report=audit(a.run_dir.resolve())
    (out/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
