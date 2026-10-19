"""Read-only arithmetic, budgets, teacher membership and checkpoint replay."""
import collections,csv,hashlib,json,math,pathlib,statistics as s
import torch
from ht_pdm_fjsp import passive_technician_td_assignment as exp
from ht_pdm_fjsp import passive_technician_oracle_representation as base
from ht_pdm_fjsp import passive_technician_role_context as role
from ht_pdm_fjsp.passive_technician_marl import PassiveConfig
ROOT=pathlib.Path(__file__).resolve().parents[2];P=ROOT/'lab_results/ra_qmix_td_assignment_full_cpu_20261004T053816Z';OUT=pathlib.Path(__file__).parent
torch.set_num_threads(1)
def read(n):
    with (P/n).open() as f:return list(csv.DictReader(f))
m=json.loads((P/'manifest.json').read_text());assert m['status']=='COMPLETED' and m['git_revision']=='5799a0527796caea511edf281700256e2e412a6e'
assert all((P/n).is_file() for n in m['outputs'])
models=read('model_metrics.csv');states=read('state_metrics.csv');logs=read('training_progress.csv');episodes=read('episodes.csv')
assert len(models)==len(list(P.rglob('model.pt')))==90 and len(states)==40200 and len(logs)==13410 and len(episodes)==4500
K=['cell','algorithm','train_seed'];mi={tuple(r[k] for k in K):r for r in models};si=collections.defaultdict(dict)
for r in states:
    key=tuple(r[k] for k in K);sid=int(r['state_id']);assert sid not in si[key];si[key][sid]=r
assert len(mi)==len(si)==90
assert len({tuple(r[k] for k in K+['eval_seed']) for r in episodes})==4500
assert {int(r['eval_seed']) for r in episodes}==set(range(98200,98250))
assert all(float(r['invalid_requests'])==0 and abs(float(r['objective'])-sum(float(r[k]) for k in ['maintenance_cost','queue_waiting_cost','failure_cost','downtime_cost']))<1e-9 for r in episodes)
assert all(math.isfinite(float(r[k])) for r in logs for k in ['td_loss','assignment_ce','loss','gradient_norm'])
lg=collections.defaultdict(list)
for r in logs:lg[tuple(r[k] for k in K)].append(int(r['update']))
expected=list(range(100,14874,100))+[14873]
assert len(lg)==90 and all(v==expected for v in lg.values())
training_counts=collections.Counter()
with (P/'training_episodes.csv').open() as f:
    for r in csv.DictReader(f):
        key=tuple(r[k] for k in K);assert r['algorithm']!='monotonic_policy_control'
        training_counts[key]+=1;assert int(r['episode'])==training_counts[key] and int(r['env_steps'])==int(r['episode'])*6
        assert float(r['invalid_requests'])==0
assert len(training_counts)==60 and set(training_counts.values())=={10000}
hashes=collections.defaultdict(set);coverage=[]
for path in P.glob('*/*/train_seed_*/training_coverage.json'):
    r=json.loads(path.read_text());coverage.append(r);hashes[(r['cell'],r['train_seed'])].add(r['teacher_batch_sha256'])
    control=r['algorithm']=='monotonic_policy_control'
    assert r['optimizer_updates']==14873 and r['env_steps']==(0 if control else 60000)
    assert r['target_syncs']==(0 if control else 29) and r['supervision_heldout_rows']==0
assert len(coverage)==90 and len(hashes)==30 and all(len(v)==1 for v in hashes.values())
replayed=0;teacher_states=0;critical=[];means=[]
for cell in ['nominal','shared_preference','pressure']:
    pop=json.loads((P/cell/'oracle_states.json').read_text());trainids={r['state_id'] for r in pop if r['split']=='train'}
    teacher={r['state_id']:tuple(r['actions']) for r in json.loads((P/cell/'train_teacher_actions.json').read_text())};assert set(teacher)==trainids
    witness=json.loads((P/cell/'policy_feasibility.json').read_text());lookup={tuple(r['inputs']):r['action'] for r in witness['witness_policy']}
    config=PassiveConfig(**torch.load(next((P/cell).rglob('model.pt')),weights_only=True)['config']);oracle=base.ExactOracle(config,progress=False)
    assert len(oracle.keys)==len(pop)
    for key in oracle.keys:
        sid=oracle.key_to_id[key];node=oracle.nodes[key]
        assert node['observations']==tuple(tuple(x) for x in pop[sid]['observations'])
        if sid in trainids:
            a=tuple(lookup[role.raw_feature_key(node['observations'],i,True,True)] for i in range(2));assert a==teacher[sid]
            assert max(node['q95'])-node['q95'][node['actions'].index(a)]<=1e-8;teacher_states+=1
    for key,rs in si.items():
        if key[0]!=cell:continue
        path=P/cell/key[1]/('train_seed_'+key[2])/'model.pt';payload=torch.load(path,weights_only=True)
        assert payload['teacher_sha256']==hashlib.sha256((P/cell/'train_teacher_actions.json').read_bytes()).hexdigest()
        model=exp.load_checkpoint(path)
        x=torch.stack([base._obs_tensor(oracle.nodes[k]['observations'],config) for k in oracle.keys])
        masks=torch.tensor([oracle.nodes[k]['masks'] for k in oracle.keys])
        with torch.no_grad():actions=model.agent_q(x).masked_fill(~masks,-torch.inf).argmax(-1).tolist()
        indices={};cc=oo=0
        for i,k in enumerate(oracle.keys):
            sid=oracle.key_to_id[k];node=oracle.nodes[k];row=rs[sid];saved=json.loads(row['selected_actions'])
            if actions[i]!=saved:
                with torch.no_grad():actions[i]=model.agent_q(x[i:i+1]).masked_fill(~masks[i:i+1],-torch.inf).argmax(-1)[0].tolist()
            assert actions[i]==saved;replayed+=1
            idx=node['actions'].index(tuple(saved));indices[k]=idx;regret=max(node['q95'])-node['q95'][idx]
            assert abs(regret-float(row['regret']))<1e-10 and (regret<=1e-8)==(row['oracle_optimal']=='True')
            if sid not in trainids:assert row['teacher_actions']=='' and row['teacher_agreement']==''
            else:assert tuple(json.loads(row['teacher_actions']))==teacher[sid]
            if node['observations'][0]==node['observations'][1]:
                diag=max(q for a,q in zip(node['actions'],node['q95']) if a[0]==a[1])
                if max(node['q95'])-diag>1e-8:cc+=1;oo+=regret<=1e-8
        for gamma,name in [(base.GAMMA,'discounted_policy_cost'),(1.,'undiscounted_policy_cost')]:
            assert abs(oracle.policy_cost(indices,gamma)-float(mi[key][name]))<1e-9
        for split in ['train','heldout','all']:
            rr=list(rs.values()) if split=='all' else [r for r in rs.values() if r['split']==split]
            assert abs(s.mean(float(r['regret']) for r in rr)-float(mi[key][split+'_regret']))<1e-10
        critical.append(dict(cell=cell,algorithm=key[1],train_seed=key[2],critical_states=cc,optimal_states=oo))
    for a in exp.ARMS:
        rr=[r for r in models if r['cell']==cell and r['algorithm']==a]
        means.append(dict(cell=cell,algorithm=a,**{k:s.mean(float(r[k]) for r in rr) for k in ['undiscounted_policy_cost','train_regret','heldout_regret','train_oracle_optimal','heldout_oracle_optimal']}))
numeric=[dict(r,train_seed=int(r['train_seed']),**{k:float(v) for k,v in r.items() if k not in K}) for r in models]
paired,summary=exp.summarize(numeric,exp.settings('full'),True);stored=json.loads((P/'summary.json').read_text())
assert summary['primary']==stored['primary'] and summary['positive_control']==stored['positive_control']
for r,t in zip(read('paired_seed_metrics.csv'),paired):assert all(abs(float(r[k])-v)<1e-10 for k,v in t.items())
for name,rows in [('cell_means.csv',means),('critical_states.csv',critical)]:
    with (OUT/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result=dict(status='ANALYZED',source=str(P),checkpoint_models_replayed=90,checkpoint_decisions_replayed=replayed,
    exact_costs_recomputed=90,train_teacher_assignments_verified=teacher_states,training_episode_rows_verified=sum(training_counts.values()),
    discounted_oracle_scope='same implementation arithmetic consistency, not independent oracle',gates_recomputed=summary)
(OUT/'analysis_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
