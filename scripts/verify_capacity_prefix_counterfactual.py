"""Independent arithmetic/input verification, without simulator/model calls."""
import argparse,csv,hashlib,json,math,shutil
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def csvrows(p):
    with p.open() as f:return list(csv.DictReader(f))
def close(a,b):
    if b is None:assert a is None
    else:assert abs(a-b)<1e-10,(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnostic',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    d=Path(a.diagnostic);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    manifest=read(d/'manifest.json');source=Path(manifest['source']);sm=read(source/'manifest.json')
    assert manifest['status']=='COMPLETED' and manifest['audit_status']=='PASS' and not manifest['test_opened']
    assert sha(source/'manifest.json')==manifest['source_manifest_sha256']
    for root,m in ((d,manifest),(source,sm)):
        for name,v in m['output_bindings'].items():assert sha(root/name)==v['sha256'] and (root/name).stat().st_size==v['bytes']
    rows=csvrows(d/'branches.csv');inputs={r['train_seed']:r for r in read(d/'inputs.json')};summary=read(d/'summary.json')
    assert len(rows)==576 and len(inputs)==3
    table={}
    for t,inp in inputs.items():
        with np.load(source/inp['candidate']['latency_observations']['path'],allow_pickle=False) as f:
            table[t]={key:f[key].copy() for key in f.files}
    def stats(sub):
        result={}
        for b in ('original','wait','shortest','flat_first'):
            group=[r for r in sub if r['branch']==b]
            result[b]=dict(contexts=len(group),mean_gain_per_machine=float(np.mean([float(r['gain_per_machine']) for r in group])) if group else None,
                improved=sum(float(r['gain_per_machine'])>1e-8 for r in group),worsened=sum(float(r['gain_per_machine'])< -1e-8 for r in group),
                unchanged=sum(abs(float(r['gain_per_machine']))<=1e-8 for r in group),changed_tokens=sum(r['original_token']!=r['token'] for r in group))
            for key in ('availability_delta','pending_delta_per_machine','failures_delta_per_machine'):
                result[b]['mean_'+key]=float(np.mean([float(r[key]) for r in group])) if group else None
        return result
    seen=set()
    for r in rows:
        t=int(r['train_seed']);i=int(r['state_index']);token=int(r['token']);action=json.loads(r['action']);obs=table[t];base=inputs[t]['source_episode']
        key=(t,i,r['branch']);assert key not in seen;seen.add(key)
        assert action[0]==token and action[-1]==0
        used=[]
        for j,v in enumerate(action[:2]):
            if v:assert obs['pair_mask'][i,v-1,j] and v not in used;used.append(v)
        duration=round(float(obs['edges'][i,token-1,0,1])*12) if token else 0
        assert duration==int(r['alternative_duration'])
        orig=int(r['original_token']);assert int(r['original_duration'])==(round(float(obs['edges'][i,orig-1,0,1])*12) if orig else 0)
        close(float(r['gain_per_machine']),(float(base['extended_cost'])-float(r['extended_cost']))/4)
        for metric,delta,unit in [('availability','availability_delta',1),('terminal_pending','pending_delta_per_machine',4),('policy_failures','failures_delta_per_machine',4)]:
            close(float(r[delta]),(float(r[metric])-float(base[metric]))/unit)
        if r['branch']=='original':
            assert token==orig and float(r['gain_per_machine'])==0
            assert all(float(r[k])==0 for k in ('availability_delta','pending_delta_per_machine','failures_delta_per_machine'))
        if r['branch']=='wait':assert token==0
        if r['branch']=='shortest':
            legal=np.flatnonzero(obs['pair_mask'][i,:,0]);expected=int(min(legal,key=lambda m:(round(float(obs['edges'][i,m,0,1])*12),m)))+1 if len(legal) else 0
            assert token==expected
    assert len(seen)==576
    def strata(sub):return {name:stats([r for r in sub if pred(int(r['original_duration']))]) for name,pred in dict(all=lambda d:True,four_tick=lambda d:d==4,other_positive=lambda d:d in (1,2),original_wait=lambda d:d==0).items()}
    pooled=strata(rows);per={str(t):strata([r for r in rows if int(r['train_seed'])==t]) for t in inputs}
    def compare(actual,expected):
        if isinstance(actual,dict):
            assert actual.keys()==expected.keys()
            for k,v in actual.items():compare(v,expected[k])
        elif actual is None:assert expected is None
        else:close(actual,expected)
    compare(pooled,summary['strata']);compare(per,summary['per_replica'])
    w=pooled['four_tick']['wait'];checks=dict(sufficient_contexts=w['contexts']>=6 and len(per)==3 and all(v['four_tick']['wait']['contexts']>0 for v in per.values()),mean_cost_gain=w['mean_gain_per_machine']>=1,
        every_replica_positive=all(v['four_tick']['wait']['mean_gain_per_machine']>0 for v in per.values()),availability=w['mean_availability_delta']>=-.01,pending=w['mean_pending_delta_per_machine']<=.025,failures=w['mean_failures_delta_per_machine']<=.25)
    assert checks==summary['engineering_checks'] and all(checks.values())==summary['deferral_explanation_supported']
    result=dict(status='PASS',diagnostic=str(d),diagnostic_manifest_sha256=sha(d/'manifest.json'),source_bindings_verified=len(sm['output_bindings']),diagnostic_bindings_verified=len(manifest['output_bindings']),branches=576,noop_controls=144,engineering_checks=checks,deferral_explanation_supported=all(checks.values()),limitation='Independent arithmetic, legal-token and duration verification; simulator control execution verified by the primary diagnostic, not independently rerun here.')
    shutil.copy2(__file__,out/'verifier.py');(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
