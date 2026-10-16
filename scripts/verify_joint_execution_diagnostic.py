"""Independent arithmetic and immutable-input verification, no rollout/model calls."""
import argparse,csv,hashlib,json,shutil
from collections import defaultdict
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def close(a,b):assert abs(a-b)<1e-10,(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);m=read(source/'manifest.json');s=read(source/'summary.json')
    assert m['status']=='COMPLETED' and m['audit_status']=='PASS' and not m['test_opened'] and m['training_steps']==0
    for name,b in m['output_bindings'].items():assert sha(source/name)==b['sha256'] and (source/name).stat().st_size==b['bytes']
    original=Path(m['source']);assert sha(original/'manifest.json')==m['source_manifest_sha256'];om=read(original/'manifest.json')
    for name,b in om['output_bindings'].items():assert sha(original/name)==b['sha256'] and (original/name).stat().st_size==b['bytes']
    with (source/'episodes.csv').open() as f:rows=list(csv.DictReader(f))
    arms=tuple(s['means']['mean']);trains=tuple(range(250000000,250000003));seeds=tuple(range(260100000,260100064));seen=set();pop={};groups=defaultdict(list)
    for r in rows:
        a,t,mode,rep,seed=r['method'],int(r['train_seed']),r['mode'],int(r['repeat']),int(r['seed']);key=a,t,mode,rep,seed
        assert key not in seen;seen.add(key);assert a in arms and t in trains and seed in seeds and rep in ([-1] if mode=='mean' else range(4))
        assert pop.setdefault(seed,r['population_sha256'])==r['population_sha256'] and float(r['invalid_assignments'])==0 and int(r['physical_ticks'])==60
        groups[mode,a].append(r)
    expected={(a,t,mode,rep,seed) for a in arms for t in trains for mode,rep in [('mean',-1),*(('sample',i) for i in range(4))] for seed in seeds}
    assert seen==expected and len(rows)==7680 and len(pop)==64
    fields=dict(normalized_cost=('extended_cost',4),availability=('availability',1),normalized_pending=('terminal_pending',4),normalized_failures=('policy_failures',4));means={mode:{} for mode in ('mean','sample')}
    for (mode,a),rs in groups.items():
        values={k:sum(float(r[name])/unit for r in rs)/len(rs) for k,(name,unit) in fields.items()};means[mode][a]=values
        for k,v in values.items():close(v,s['means'][mode][a][k])
        for t in trains:
            sub=[r for r in rs if int(r['train_seed'])==t]
            for k,(name,unit) in fields.items():close(sum(float(r[name])/unit for r in sub)/len(sub),s['per_replica'][mode][a][str(t)][k])
    core=('joint_clearing_ppo','forward_clearing_ppo','reverse_clearing_ppo','independent_clearing_ppo');candidate=means['sample'][core[0]];gates={}
    for a in arms[1:]:
        control=means['sample'][a] if a in core else means['mean'][a];v=1-candidate['normalized_cost']/control['normalized_cost'];key='gain_vs_'+a
        close(v,s['mixed_reference_points'][key]);gates[key]=v>=.005 if a==core[3] else v>0 if a in ('mappo','full_mappo') else v>=-.01
    for a in ('ppo','normalized_ppo'):
        for metric,limit,lower in [('availability',-.01,True),('normalized_pending',.025,False),('normalized_failures',.25,False)]:
            key=metric+'_delta_vs_'+a;v=candidate[metric]-means['mean'][a][metric];close(v,s['mixed_reference_points'][key]);gates[key]=v>=limit if lower else v<=limit
    gain=1-candidate['normalized_cost']/means['mean'][core[0]]['normalized_cost'];close(gain,s['joint_sample_vs_mean_gain']);assert gates==s['mixed_reference_gates'] and len(gates)==13
    replica_gain={str(t):1-s['per_replica']['sample'][core[0]][str(t)]['normalized_cost']/s['per_replica']['mean'][core[0]][str(t)]['normalized_cost'] for t in trains}
    checks=dict(own_mean_gain_ge_one_percent=gain>=.01,every_replica_positive=all(v>0 for v in replica_gain.values()),all13_quality_health_gates=all(gates.values()),zero_invalid=True)
    assert checks==s['engineering_checks'] and all(checks.values())==s['stochastic_deployment_trial_motivated']
    physical=read(source/'physical_audit.json');assert len(physical)==24 and sum(r['physical_ticks'] for r in physical)==460800 and sum(r['policy_decisions'] for r in physical)==368640 and all(r['status']=='PASS' for r in physical)
    behavior=read(source/'behavior_replay.json');assert len(behavior)==120 and sum(r['behavior_replay'] is not None for r in behavior)==96
    assert all(r['behavior_replay']['behavior_normalized_max_error']<=1 for r in behavior if r['behavior_replay'] is not None)
    result=dict(status='PASS',source=str(source),source_manifest_sha256=sha(source/'manifest.json'),bindings_verified=len(m['output_bindings']),learning_bindings_verified=len(om['output_bindings']),episodes=7680,physical_ticks_audited_by_independent_primitive=460800,policy_decisions=368640,mean_vs_sample_gain=gain,engineering_checks=checks,stochastic_trial_motivated=all(checks.values()),limitation='This verifier checks stored-data arithmetic and primary primitive-audit completion; primitive replay itself was independently implemented and executed by the diagnostic, not rerun here.')
    shutil.copy2(__file__,out/'verifier.py');(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
