"""Read-only, bounded validation; writes derived artifacts beside this script."""
import csv, json, hashlib, sys
from collections import defaultdict, Counter
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch

REPO = Path('/Users/bachng/Coding/Reinforcement Learning/marl')
CODE = Path('/private/tmp/marl-maintenance-assignment-confidence-v1')
RUN = REPO/'lab_results/maintenance_assignment_confidence_v1_full_cpu_20261011T165402Z'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE/'src'))
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, WaitingEnv, transition
from ht_pdm_fjsp.maintenance_assignment_confidence_policy import calibrate, ConfidenceController
from ht_pdm_fjsp.maintenance_assignment_value_policy import RelativeValueHead
from ht_pdm_fjsp.maintenance_assignment_value_training import predictions
from ht_pdm_fjsp.maintenance_assignment_value_source import verify_source
from tqdm import tqdm

torch.set_num_threads(1)
def read(name):
    with (RUN/name).open() as f: return list(csv.DictReader(f))
def js(name): return json.loads((RUN/name).read_text())
def state(raw): return DispatchState(**{k:tuple(v) for k,v in json.loads(raw).items()})
def action(raw): return tuple(tuple(e) for e in json.loads(raw))
def near(a,b,tol=1e-9): assert np.allclose(a,b,atol=tol,rtol=tol), (a,b)
manifest=js('manifest.json'); cfg=manifest['settings']; summary=js('summary.json')
audit={}
assert manifest['status']=='COMPLETED' and manifest['audit']=='PASS'
for name, record in tqdm(manifest['artifacts'].items(),desc='Hashes'):
    p=RUN/name
    assert p.resolve().is_relative_to(RUN.resolve())
    assert p.stat().st_size==record['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],name
audit['hashes']=len(manifest['artifacts'])
episodes=read('episodes.csv'); index={(r['condition'],int(r['train_seed']),int(r['cohort']),int(r['shock']),r['controller']):r for r in episodes}
assert len(index)==len(episodes)==2880
names=['frozen_ppo','natural_ungated','natural_gated','broad_ungated','broad_gated']
rng=np.random.default_rng(role_seed(cfg['analysis_seed'],'maintenance_assignment_confidence_v1:crossed_bootstrap'))
rd=rng.integers(3,size=(20000,3)); pd=rng.integers(24,size=(20000,24))
wr=np.stack([np.bincount(x,minlength=3) for x in rd])/3
wp=np.stack([np.bincount(x,minlength=24) for x in pd])/24
def matrix(cond,name):
    return np.array([[np.mean([float(index[cond,s,c,z,name]['extended_objective']) for z in cfg['test_shocks']]) for c in cfg['test_cohorts']] for s in cfg['model_seeds']])
def contrast(cond,method,baseline,confidence):
    a,b=matrix(cond,baseline),matrix(cond,method)
    sa=np.einsum('bi,ij,bj->b',wr,a,wp); sb=np.einsum('bi,ij,bj->b',wr,b,wp)
    return {'mean_gain':float((a-b).mean()/a.mean()),'ci':np.quantile((sa-sb)/sa,[(1-confidence)/2,(1+confidence)/2]).tolist(),'confidence':confidence,
            'replica_gains':((a-b).mean(1)/a.mean(1)).tolist()}
contrasts={}
for section in ['primary_contrasts','secondary_contrasts']:
    for key,r in summary[section].items():
        cond=key.split(':')[0]; result=contrast(cond,r['method'],r['baseline'],r['confidence'])
        near(result['mean_gain'],r['relative_gain']);near(result['ci'],r['confidence_interval'])
        contrasts[key]=result
for cond in cfg['conditions']:
    for name in ['natural_ungated','broad_ungated']:
        contrasts[f'{cond}:{name}_vs_frozen_ppo_posthoc']=contrast(cond,name,'frozen_ppo',.95)
    for name in names:
        rows=[r for r in episodes if r['condition']==cond and r['controller']==name]
        for metric,mean in summary['means'][cond][name].items():near(np.mean([float(r[metric]) for r in rows]),mean)
    for s in cfg['model_seeds']:
        for c in cfg['test_cohorts']:
            for z in cfg['test_shocks']:
                base=index[cond,s,c,z,'frozen_ppo']
                for name in ['natural_gated','broad_gated']:
                    r=index[cond,s,c,z,name]
                    assert float(r['assignment_override_ticks'])==0
                    for k,v in base.items():
                        if k.startswith(('base_','tail_','extended_')): assert r[k]==v,(k,cond,s,c,z)
audit['episode_grid_and_statistics']='PASS; all 2880 episodes; 10 registered contrasts plus 4 posthoc'
audit['gated_physical_episode_equality']=1152

configs={(r['split'],r['cohort'],r['condition']):WaitingConfig(**r['config']) for r in js('resolved_config.json')['physical_profiles']}
for filename in ['source_trajectories.csv','probe_prefixes.csv']:
    rows=read(filename)
    for r in tqdm(rows,desc=filename):
        c=configs[r['split'],int(r['cohort']),r['condition']]
        after,cost,metrics=transition(c,state(r['state']),action(r['action']),np.array(json.loads(r['shock_bits']),bool))
        assert asdict(after)==asdict(state(r['following']))
        expected=json.loads(r['metrics']);assert metrics.keys()==expected.keys()
        for k,v in metrics.items():near(v,expected[k])
    audit[filename+'_transition_replays']=len(rows)
rows=read('decisions.csv'); tapes={}; gated_actions={}; base_actions={}
for r in tqdm(rows,desc='Logged test transitions'):
    c=configs['test',int(r['cohort']),r['condition']]; seed=int(r['env_seed'])
    k=(r['condition'],int(r['cohort']),seed)
    if k not in tapes:
        env=WaitingEnv(c);env.reset(seed);tapes[k]=env.events
    after,cost,metrics=transition(c,state(r['state']),action(r['action']),tapes[k][int(r['time'])])
    assert asdict(after)==asdict(state(r['following']))
    for field,v in metrics.items():near(v,json.loads(r['physical_metrics'])[field])
    key=tuple(r[x] for x in ['condition','cohort','shock','train_seed','time'])
    if r['controller']=='frozen_ppo':base_actions[key]=(r['state'],r['action'],r['following'])
    if r['controller'] in ['natural_gated','broad_gated']:gated_actions[key,r['controller']]=(r['state'],r['action'],r['following'])
for (key,name),value in gated_actions.items():assert value==base_actions[key]
audit['logged_test_transition_replays']=len(rows);audit['logged_gated_action_equality']=len(gated_actions)

selections=js('checkpoint_selection.json');calibration=js('calibration.json');assert selections['locked_before_test'] and calibration['locked_before_test']
heads={};prediction_metrics=[]
datasets={(split,s):torch.load(RUN/f'datasets/{split}_{s}.pt',weights_only=True) for s in cfg['model_seeds'] for split in ['train','development','calibration']}
for s in cfg['model_seeds']:
    for arm in ['natural','broad']:
        hs=[]
        for member in range(3):
            sel=selections['selections'][str(s)][f'{arm}:{member}']; best=min(sel['candidates'],key=lambda x:(x['common_development_mse'],x['epoch']))
            assert best==sel['selected']; chosen=sel['selected']
            h=RelativeValueHead(32);h.load_state_dict(torch.load(RUN/chosen['checkpoint'],weights_only=True)['state_dict']);h.eval();hs.append(h)
            with torch.no_grad(): devpred=predictions(h,datasets['development',s])
            near(float(((devpred-datasets['development',s]['targets'])**2).mean()),chosen['common_development_mse'],tol=1e-6)
        heads[s,arm]=hs
        rebuilt,records=calibrate(hs,datasets['calibration',s],cfg)
        for cond in cfg['conditions']:
            saved=calibration['parameters'][str(s)][arm][cond]
            assert rebuilt[cond]['enabled']==saved['enabled'] and rebuilt[cond]['samples']==saved['samples']
            if saved['enabled']:near(rebuilt[cond]['margin'],saved['margin'],tol=2e-6)
        for split in ['development','calibration']:
            d=datasets[split,s]
            with torch.no_grad():p=torch.stack([predictions(h,d) for h in hs]).mean(0)
            for ci,cond in enumerate(cfg['conditions']):
                for oi,origin in enumerate(['natural','probe']):
                    ix=(d['conditions']==ci)&(d['origins']==oi)
                    target=d['targets'][ix];pred=p[ix]
                    prediction_metrics.append(dict(source=s,arm=arm,split=split,condition=cond,origin=origin,labels=int(ix.sum()),
                        mse=float(((target-pred)**2).mean()),zero_mse=float((target**2).mean()),sign_accuracy=float(((target>0)==(pred>0)).float().mean())))
audit['selected_heads_dev_MSE_and_calibration_rebuilt']=18
draws=read('training_draws.csv')
for r in tqdm(draws,desc='Sampling budgets'):
    d=datasets['train',int(r['source_seed'])];ix=torch.tensor(json.loads(r['draw_indices']))
    assert len(ix)==1024 and int(ix.min())>=0 and int(ix.max())<len(d['targets'])
    probes=int(d['origins'][ix].sum());assert probes==(0 if r['arm']=='natural' else 512)==int(r['probe_draws'])
    assert int((d['ticks'][ix]>0).sum())==int(r['late_draws'])
audit['training_draw_rows']=len(draws)

pm=read('pm_choices.csv');unique={}
for r in pm:
    key=tuple(r[x] for x in ['condition','cohort','shock','controller','train_seed','time'])
    if r['controller']=='frozen_ppo':continue
    unique[key]=r
gate_diagnostics=[]
for cond in cfg['conditions']:
    for s in cfg['model_seeds']:
        for name in names[1:]:
            records=[]
            for key,r in unique.items():
                if r['condition']==cond and int(r['train_seed'])==s and r['controller']==name:
                    a=json.loads(r['controller_diagnostics'])['assignment_decode']
                    alt=[v for actionid,v in zip(a['candidates'],a['adjusted_gains']) if actionid!=a['original_id']]
                    if alt:records.append((max(alt),a['near_tie'],a['rejection_reason'],a['changed']))
            margin=calibration['parameters'][str(s)][name.split('_')[0]][cond]['margin']
            gate_diagnostics.append(dict(condition=cond,source=s,controller=name,PM_states_with_choice=len(records),
                max_adjusted_gain=max([r[0] for r in records],default=None),margin=margin,
                near_tie_count=sum(r[1] for r in records),changed=sum(r[3] for r in records),reasons=dict(Counter(r[2] for r in records))))

cache={}; pair_rows=read('label_pairs.csv')
for r in tqdm(pair_rows,desc='All paired labels from saved forecasts'):
    path=r['forecast']
    if path not in cache:
        with np.load(RUN/path) as data:
            identity=json.loads(str(data['identity_json']));costs=data['costs'];cache[path]=(costs,identity)
            near(costs,data['base']+data['tail']);near(costs,sum(data[x] for x in data.files if x.startswith('components_')))
    costs,identity=cache[path];roots=identity['roots'];delta=costs[roots.index(json.loads(r['right']))]-costs[roots.index(json.loads(r['left']))]
    near(delta.mean()/20,float(r['target_normalized']));near(delta.std(ddof=1)/np.sqrt(len(delta))/20,float(r['paired_mc_se_normalized']))
audit['paired_label_algebra']=len(pair_rows)
rootrows=read('root_states.csv'); rootpreds=read('root_predictions.csv')
for r in rootrows:
    if not r['evaluation_forecast']:continue
    with np.load(RUN/r['evaluation_forecast']) as data:costs=data['costs']
    roots=json.loads(r['roots'])
    pred=next(v for v in rootpreds if all(v[x]==r[x] for x in ['condition','cohort','shock','model_seed','tick','origin']))
    acts=json.loads(pred['actions'])
    for name in names[1:]:
        delta=costs[0]-costs[roots.index(acts[name])]
        near(delta.mean(),float(r[f'{name}_gain']));near(delta.std(ddof=1)/np.sqrt(len(delta)),float(r[f'{name}_mc_se']))
audit['root_gain_algebra']=len(rootrows)

# Bounded scalar replay: two roots and one future per available phase/source/condition/origin cell.
from ht_pdm_fjsp.maintenance_assignment_value_forks import scalar_replay
models,source,source_audit=verify_source(REPO/'lab_results/maintenance_assignment_scoring_v1_full_cpu_20261011T131859Z',cfg)
sample={}
for p in sorted((RUN/'forecasts').rglob('*.npz')):
    with np.load(p) as d: ident=json.loads(str(d['identity_json']))
    key=tuple(ident[x] for x in ['phase','model_seed','condition','origin'])
    if key not in sample:sample[key]=p
replayed_paths=replayed_transitions=0
for key,p in tqdm(sample.items(),desc='Stratified scalar forecast replay'):
    with np.load(p) as d:
        ident=json.loads(str(d['identity_json'])); c=WaitingConfig(**ident['config']);st=DispatchState(**{k:tuple(v) for k,v in ident['state'].items()})
        n=d['costs'].shape[0]; ix=sorted(set([0,n-1]));
        expected_shocks=np.random.default_rng(ident['seed']).random(d['shocks'].shape)<c.failure_probability
        assert np.array_equal(expected_shocks,d['shocks'])
        values={x:d[x][ix,:1] for x in ['costs','base','tail','first']}
        values.update(roots=tuple(tuple(tuple(e) for e in ident['roots'][i]) for i in ix),shocks=d['shocks'][:,:1],trace=None)
        for scope in ['components','cutoff','terminal']:
            values[scope]={x.removeprefix(scope+'_'):d[x][ix,:1] for x in d.files if x.startswith(scope+'_')}
    scalar_replay(c,st,values,models[ident['model_seed']],ident['tick'])
    replayed_paths+=len(ix);replayed_transitions+=len(ix)*(60-ident['tick'])
audit['bounded_forecast_replay']={'files':len(sample),'paths':replayed_paths,'transitions':replayed_transitions,'sampling':'first lexicographic file per phase/source/condition/origin; first and last roots; scenario zero'}
audit['source_contract_reverified']=source_audit

result=dict(run=str(RUN),commit=manifest['commit'],audit=audit,contrasts=contrasts,means=summary['means'],
    gate_diagnostics=gate_diagnostics,prediction_metrics=prediction_metrics,calibration=calibration,
    root_diagnostics=js('diagnostic_summary.json'),coverage=js('training_coverage.json'),elapsed_seconds=manifest['elapsed_seconds'])
(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for name,records in [('prediction_metrics.csv',prediction_metrics),('gate_diagnostics.csv',gate_diagnostics)]:
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
print(json.dumps({'audit':audit,'posthoc':{k:v for k,v in contrasts.items() if k.endswith('posthoc')}},indent=2))
