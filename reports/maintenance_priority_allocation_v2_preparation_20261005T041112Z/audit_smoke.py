"""Post-smoke audit: hashes, every physical trace, resource/cost accounting."""
import argparse,csv,hashlib,json,math,sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--repository',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--report',required=True)
a=parser.parse_args()
repo=Path(a.repository); output=Path(a.output)
sys.path.insert(0,str(repo/'src'))
from ht_pdm_fjsp import maintenance_priority_allocation as exp
from ht_pdm_fjsp import maintenance_dispatch as envmod
from ht_pdm_fjsp.maintenance_dispatch_policy import fixed_machine, feasible, batch_observations
import torch
torch.set_num_threads(1)
def read(name):
    with (output/name).open() as f: return list(csv.DictReader(f))
manifest=json.loads((output/'manifest.json').read_text())
summary=json.loads((output/'summary.json').read_text())
assert manifest['status']=='COMPLETED' and manifest['protocol']==exp.PROTOCOL
assert not manifest['sealed_test_evaluated'] and all(manifest['audits'].values())
assert summary['primary_passed'] is None and summary['learning_benefit_with_rule_descriptive'] is None
assert len(summary['family_mean_costs'])==4
for name,h in manifest['source_hashes'].items():
    assert hashlib.sha256((output/'source_snapshot'/name).read_bytes()).hexdigest()==h
    current=repo/'src/ht_pdm_fjsp'/name
    if name.endswith('_plan.md'): current=repo/'docs'/name
    if name.endswith('_seed_registry.json'): current=repo/'configs'/name
    assert hashlib.sha256(current.read_bytes()).hexdigest()==h
expected_bundle=hashlib.sha256(json.dumps(manifest['source_hashes'],sort_keys=True).encode()).hexdigest()
assert expected_bundle==manifest['source_bundle_sha256']
models={}
for path in output.glob('*/train_seed_*/model.pt'):
    model,payload=exp.load_checkpoint(path)
    assert payload['source_bundle_sha256']==expected_bundle
    models[payload['algorithm'],payload['train_seed']]=model
assert len(models)==manifest['expected_models']==3
counts={name:len(read(name)) for name in ['training_episodes.csv','training_progress.csv','episodes.csv','development_episodes.csv','reference_episodes.csv','exact_small_metrics.csv','exact_headroom.csv','exact_headroom_contrasts.csv','reference_contrasts.csv','decisions.csv','reference_decisions.csv','machine_metrics.csv','reference_machine_metrics.csv']}
assert counts['training_episodes.csv']==36 and counts['training_progress.csv']==6
assert counts['episodes.csv']==63 and counts['development_episodes.csv']==3
assert counts['reference_episodes.csv']==22
assert counts['exact_small_metrics.csv']==counts['exact_headroom.csv']==4
assert counts['exact_headroom_contrasts.csv']==counts['reference_contrasts.csv']==3
assert counts['decisions.csv']==243 and counts['reference_decisions.csv']==81
assert counts['machine_metrics.csv']==243 and counts['reference_machine_metrics.csv']==78
results={(r['algorithm'],int(r['train_seed']),r['family'],int(r['eval_seed'])):r for r in read('episodes.csv')+read('reference_episodes.csv')}
replayed=steps=priority_choices=0; max_error=0.0
for filename in ['decisions.csv','reference_decisions.csv']:
    rows=read(filename); groups={}
    for row in rows:
        key=(row['algorithm'],int(row['train_seed']),row['family'],int(row['eval_seed']))
        groups.setdefault(key,[]).append(row)
    for key,traces in groups.items():
        mode,seed,family,episode_seed=key
        config=envmod.family_config(family,episode_seed,'smoke')
        env=envmod.DispatchEnv(config); obs=env.reset(episode_seed)
        model=models.get((mode,seed))
        assert len(traces)==config.horizon
        for row in traces:
            assert int(row['time'])==env.time
            for field in ['ages','failed','pending_wait','remaining','assigned']:
                assert tuple(json.loads(row[field]))==getattr(env.state,field)
            pairs=tuple(tuple(p) for p in json.loads(row['pairs']))
            sequence=[tuple(p) for p in json.loads(row['sequence'])]
            policy_pairs=exp.rule_action(obs) if model is None else model.act(obs,mode,deterministic=True)[0]
            assert pairs==policy_pairs
            if model is not None:
                batch=batch_observations([obs]); _,p,entropy,_=model.decode(batch,mode,exp.batch_sequences([sequence]))
                assert torch.isfinite(p).all() and torch.isfinite(entropy).all()
                if mode=='fixed_priority':
                    used_m=torch.zeros(1,config.machines,dtype=torch.bool)
                    used_t=torch.zeros(1,config.technicians,dtype=torch.bool)
                    for m,j in pairs:
                        mask=feasible(batch,used_m,used_t)
                        assert m==int(fixed_machine(batch,mask,urgent_only=False)[0])
                        used_m[0,m]=True;used_t[0,j]=True;priority_choices+=1
            # Independent interval charge from the pre-transition occupancy.
            occupied={m for m in env.state.assigned if m>=0}|{m for m,j in pairs}
            union=sum(env.state.failed[m] or m in occupied for m in range(config.machines))
            assert float(row['unavailability_cost'])==union*config.unavailable_cost
            starts=sum(config.corrective_cost if env.state.failed[m] else config.preventive_cost for m,j in pairs)
            assert float(row['maintenance_cost'])==starts
            obs,_,_,info=env.step(pairs)
            for field,v in info.items():
                error=abs(float(row[field])-v);max_error=max(max_error,error);assert error<1e-8
            steps+=1
        for field,v in env.metrics.items():
            error=abs(float(results[key][field])-v);max_error=max(max_error,error);assert error<1e-8
        replayed+=1
for row in read('exact_headroom.csv'):
    cost=float(row['exact_policy_cost']);opt=float(row['exact_optimum']);gap=float(row['optimality_gap'])
    assert abs(cost-opt-gap)<1e-8
    assert abs(float(row['exact_selection_regret'])+float(row['exact_allocation_regret'])-gap)<1e-8
    assert abs(float(row['maximum_relative_reduction'])-gap/cost)<1e-8
    assert (row['five_percent_possible_unrestricted']=='True')==(gap+1e-8>=.05*cost)
refs=read('reference_episodes.csv');eps=read('episodes.csv')
for arm,costs in summary['family_mean_costs'].items():
    for family,cost in costs.items():
        rr=[float(r['objective']) for r in (refs if arm==exp.RULE_MODE else eps) if r['family']==family and r['algorithm']==arm]
        assert abs(sum(rr)/len(rr)-cost)<1e-8
assert replayed==84 and steps==324
result={'status':'VERIFIED_ENGINEERING_SMOKE','counts':counts,'replayed_test_episodes':replayed,'replayed_physical_steps':steps,'fixed_priority_dispatch_choices_audited':priority_choices,'max_numeric_error':max_error,'models':len(models),'engineering_gates':len(manifest['audits']),'all_gates_passed':True,'sealed_panels_closed':True,'scientific_gates_applicable':False,'source_bundle_sha256':expected_bundle,'scope':'Shared-simulator full trace replay plus independent occupancy/start-cost, family mean and exact headroom arithmetic; not independent simulator reproduction or full scientific evidence'}
Path(a.report).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
