"""Raw CSV audits and checkpoint decision replay; no retraining or raw edits."""
import collections,csv,hashlib,json,math,pathlib,statistics as stats
import torch
from ht_pdm_fjsp import passive_technician_policy_supervision as exp
from ht_pdm_fjsp import passive_technician_oracle_representation as base
from ht_pdm_fjsp import passive_technician_role_context as role
from ht_pdm_fjsp.passive_technician_marl import PassiveConfig

ROOT=pathlib.Path(__file__).resolve().parents[2]
P=ROOT/'lab_results/ra_qmix_policy_supervision_full_cpu_20261003T181732Z'
OUT=pathlib.Path(__file__).parent
torch.set_num_threads(1)
def read(n):
    with (P/n).open() as f: return list(csv.DictReader(f))
models=read('model_metrics.csv'); states=read('state_metrics.csv'); episodes=read('episodes.csv'); logs=read('training_progress.csv')
manifest=json.loads((P/'manifest.json').read_text())
assert manifest['status']=='COMPLETED' and manifest['git_revision']=='8d15799f1b992b61ea337d20139aec81933f1b49'
assert len(models)==len(list(P.rglob('model.pt')))==240
assert len(states)==107200 and len(episodes)==12000 and len(logs)==24000
assert all((P/f).is_file() for f in manifest['outputs'])
keys=['cell','regime','algorithm','train_seed']
modelidx={tuple(r[k] for k in keys):r for r in models}
stateidx=collections.defaultdict(dict)
for r in states:
    key=tuple(r[k] for k in keys); sid=int(r['state_id'])
    assert sid not in stateidx[key]; stateidx[key][sid]=r
assert len(modelidx)==len(stateidx)==240
assert len({tuple(r[k] for k in keys+['eval_seed']) for r in episodes})==12000
assert {int(r['eval_seed']) for r in episodes}==set(range(97200,97250))
assert all(float(r['invalid_requests'])==0 for r in episodes)
assert all(abs(float(r['objective'])-sum(float(r[k]) for k in ['maintenance_cost','queue_waiting_cost','failure_cost','downtime_cost']))<1e-9 for r in episodes)
assert all(math.isfinite(float(r[k])) for r in logs for k in ['loss','normalized_mse','policy_ce','gradient_norm'])
lg=collections.defaultdict(list)
for r in logs: lg[tuple(r[k] for k in keys)].append(int(r['update']))
assert len(lg)==240 and all(sorted(v)==list(range(100,10001,100)) for v in lg.values())
streams=collections.defaultdict(set)
for r in read('minibatch_streams.csv'): streams[(r['cell'],r['regime'],r['train_seed'])].add(r['minibatch_sha256'])
assert len(streams)==60 and all(len(v)==1 for v in streams.values())
checked_decisions=0; verified_witnesses=0; verified_policy_costs=0
for cell in ['nominal','shared_preference','pressure']:
    population=json.loads((P/cell/'oracle_states.json').read_text())
    q=collections.defaultdict(dict)
    for r in read(cell+'/oracle_q.csv'): q[int(r['state_id'])][tuple(json.loads(r['actions']))]=float(r['q95'])
    teacher={r['state_id']:tuple(r['actions']) for r in json.loads((P/cell/'teacher_actions.json').read_text())}
    witness=json.loads((P/cell/'policy_feasibility.json').read_text())
    lookup={tuple(r['inputs']):r['action'] for r in witness['witness_policy']}
    exemplar=next((P/cell).rglob('model.pt'))
    config=PassiveConfig(**torch.load(exemplar,weights_only=True)['config'])
    oracle=base.ExactOracle(config,progress=False)
    assert len(oracle.keys)==len(population)
    for key in oracle.keys:
        node=oracle.nodes[key]; sid=oracle.key_to_id[key]
        assert node['observations']==tuple(tuple(x) for x in population[sid]['observations'])
        assert all(abs(q[sid][a]-v)<1e-10 for a,v in zip(node['actions'],node['q95']))
    seen={}
    for r in population:
        sid=r['state_id']; feature=tuple(tuple(o) for o in r['observations'])
        if feature in seen: assert seen[feature]==r['split']
        seen[feature]=r['split']
        a=tuple(lookup[role.raw_feature_key(r['observations'],i,True,True)] for i in range(2))
        assert a==teacher[sid] and a in q[sid] and max(q[sid].values())-q[sid][a]<=1e-8
        verified_witnesses+=1
    for key,rs in stateidx.items():
        if key[0]!=cell: continue
        assert len(rs)==len(population)
        for sid,r in rs.items():
            action=tuple(json.loads(r['selected_actions'])); regret=max(q[sid].values())-q[sid][action]
            assert abs(regret-float(r['regret']))<1e-10
            assert (regret<=1e-8)==(r['oracle_optimal']=='True')
            assert (action==teacher[sid])==(r['teacher_joint_agreement']=='True')
            assert tuple(json.loads(r['teacher_actions']))==teacher[sid]
            assert (key[1]=='ceiling' or r['split']=='train')==(r['fit_state']=='True')
        for split in ['all','train','heldout']:
            subset=list(rs.values()) if split=='all' else [r for r in rs.values() if r['split']==split]
            assert abs(stats.mean(float(r['regret']) for r in subset)-float(modelidx[key][split+'_regret']))<1e-10
            assert abs(stats.mean(r['oracle_optimal']=='True' for r in subset)-float(modelidx[key][split+'_oracle_optimal']))<1e-10
        selected_indices={k:oracle.nodes[k]['actions'].index(tuple(json.loads(rs[oracle.key_to_id[k]]['selected_actions']))) for k in oracle.keys}
        for gamma,name in [(base.GAMMA,'discounted_policy_cost'),(1.,'undiscounted_policy_cost')]:
            assert abs(oracle.policy_cost(selected_indices,gamma)-float(modelidx[key][name]))<1e-9
        verified_policy_costs+=1
        if key[2]=='joint_q': continue
        checkpoint=P/cell/key[1]/key[2]/('train_seed_'+key[3])/'model.pt'
        payload=torch.load(checkpoint,weights_only=True)
        assert payload['teacher_sha256']==hashlib.sha256((P/cell/'teacher_actions.json').read_bytes()).hexdigest()
        config=PassiveConfig(**payload['config']); model=exp.load_checkpoint(checkpoint)
        x=torch.stack([base._obs_tensor(r['observations'],config) for r in population])
        mask=torch.tensor([[[any(a[i]==j for a in q[r['state_id']]) for j in range(3)] for i in range(2)] for r in population])
        with torch.no_grad():
            actions=model.agent_q(x).masked_fill(~mask,-torch.inf).argmax(-1).tolist()
            for i,(r,a) in enumerate(zip(population,actions)):
                saved=json.loads(rs[r['state_id']]['selected_actions'])
                if a!=saved:
                    a=model.agent_q(x[i:i+1]).masked_fill(~mask[i:i+1],-torch.inf).argmax(-1)[0].tolist()
                assert a==saved, (key,r['state_id'],a,saved)
                checked_decisions+=1
numeric=[dict(r,train_seed=int(r['train_seed']),**{k:float(v) for k,v in r.items() if k not in keys}) for r in models]
paired,summary=exp.summarize(numeric,exp.settings('full'),True)
stored=json.loads((P/'summary.json').read_text())
assert summary['primary']==stored['primary'] and summary['positive_control']==stored['positive_control']
for r,s in zip(read('paired_seed_metrics.csv'),paired):
    assert all(abs(float(r[k])-v)<1e-10 for k,v in s.items())
means=[]
for regime in ['ceiling','split']:
    for cell in ['nominal','shared_preference','pressure']:
        for arm in exp.ARMS:
            rr=[r for r in models if r['regime']==regime and r['cell']==cell and r['algorithm']==arm]
            means.append(dict(regime=regime,cell=cell,algorithm=arm,**{k:stats.mean(float(r[k]) for r in rr) for k in ['all_regret','heldout_regret','all_oracle_optimal','heldout_oracle_optimal','undiscounted_policy_cost','train_normalized_rmse','heldout_normalized_rmse']}))
with (OUT/'cell_means.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(means[0]));w.writeheader();w.writerows(means)
result=dict(status='ANALYZED',source=str(P),csv_state_regrets_verified=len(states),teacher_state_assignments_verified=verified_witnesses,
            checkpoints_replayed=180,checkpoint_decisions_verified=checked_decisions,exact_policy_costs_recomputed=verified_policy_costs,
            checkpoint_scope='All three shared-agent arms, both regimes, ten seeds; central Joint-Q inference not replayed',
            gates_recomputed=summary)
(OUT/'analysis_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
