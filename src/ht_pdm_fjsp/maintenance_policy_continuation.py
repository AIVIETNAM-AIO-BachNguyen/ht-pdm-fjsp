"""Do now or WAIT, with the frozen RL policy continuing to its training cutoff."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_timing_diagnostic as previous
from ht_pdm_fjsp import maintenance_joint_rl as joint
from ht_pdm_fjsp import maintenance_opportunity_cost as prior
from ht_pdm_fjsp.maintenance_dispatch import DispatchState,role_seed,validate_state
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig,WaitingEnv
from ht_pdm_fjsp.maintenance_joint_policy import JOINT,CATALOGUE
from ht_pdm_fjsp.maintenance_timing_forks import kind
from ht_pdm_fjsp.maintenance_policy_forks import fork_values,paired_delta,scalar_replay,ARMS,LABELS,COMPONENTS
from ht_pdm_fjsp.maintenance_policy_statistics import summarize

VERSION='maintenance_policy_continuation_v1'
ROOT=Path(__file__).resolve().parents[2]
KEYS=('condition','cohort','shock','model_seed','origin','tick')


def settings(profile):
    if profile not in ('full','smoke'):raise ValueError('unknown profile')
    full=profile=='full'
    old=previous.settings('full')
    return dict(profile=profile,conditions=old['conditions'],origins=old['origins'],
        cohorts=old['cohorts'] if full else old['cohorts'][:1],physical_shocks=old['physical_shocks'] if full else old['physical_shocks'][:1],
        model_seeds=old['model_seeds'] if full else old['model_seeds'][:1],landmarks=old['landmarks'] if full else [42],
        forecast_seed=14100000 if full else 14300000,analysis_seed=14200000 if full else 14300010,
        scenarios=32 if full else 4,horizon=48,tail_ticks=12,bootstrap_samples=20000 if full else 200,
        primary_confidence=.9875,continuations=list(ARMS),root_labels=list(LABELS),torch_threads=1)


def counts(cfg):
    per_tick=len(cfg['conditions'])*len(cfg['origins'])*len(cfg['cohorts'])*len(cfg['physical_shocks'])*len(cfg['model_seeds'])
    states=per_tick*len(cfg['landmarks']);window_sum=sum(60-t for t in cfg['landmarks'])
    return dict(models=len(cfg['model_seeds']),states=states,training_steps=0,new_physical_trajectory_ticks=0,
        max_unique_roots=states*4,max_forecast_transitions=per_tick*window_sum*4*cfg['scenarios']*len(ARMS),
        max_rl_continuation_decision_rows=per_tick*sum(47-t for t in cfg['landmarks'])*4*cfg['scenarios'])


def contract():return json.loads((ROOT/f'configs/{VERSION}_sources.json').read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def action(value):return tuple(map(tuple,json.loads(value)))
def state(value):return DispatchState(**{k:tuple(v) for k,v in json.loads(value).items()})


def verify_source(source,locked):
    source=Path(source).resolve()
    if sha(source/'manifest.json')!=locked['state_source_manifest_sha256']:raise ValueError('state-source manifest hash mismatch')
    m=json.loads((source/'manifest.json').read_text());cfg=previous.settings('full')
    if (m['status']!='COMPLETED' or m['audit']!='PASS' or m['dirty'] or m['protocol']!=previous.VERSION
        or m['commit']!=locked['state_source_commit'] or m['settings']!=cfg or m['actual_counts']!=previous.counts(cfg)):
        raise ValueError('state-source run contract mismatch')
    for name,r in m['artifacts'].items():
        path=(source/name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or path.stat().st_size!=r['bytes'] or sha(path)!=r['sha256']:
            raise ValueError(f'state-source artifact mismatch: {name}')
        if name.startswith('source_snapshot/'):
            base=path.name;current=ROOT/'src/ht_pdm_fjsp'/base if base.endswith('.py') else ROOT/base if base in ('pyproject.toml','uv.lock') else ROOT/'configs'/base
            if sha(current)!=r['sha256']:raise ValueError(f'inherited dependency changed: {base}')
    origin=locked['joint_source_contract']
    if json.loads((source/'source_contract.json').read_text())!=origin or m['source_audit']['source_manifest_sha256']!=origin['manifest_sha256']:
        raise ValueError('joint source provenance changed')
    models={}
    for seed,record in tqdm(list(origin['models'].items()),desc='Verify frozen continuation models'):
        path=source/'frozen_models'/f'seed_{seed}.pt'
        if sha(path)!=record['sha256']:raise ValueError('frozen model hash mismatch')
        model,payload=joint.load_checkpoint(path)
        if payload['algorithm']!=JOINT or payload['train_seed']!=int(seed) or payload['settings']!=joint.settings('full') or any(payload[k]!=record[k] for k in ('update','learned_steps')):
            raise ValueError('frozen model payload mismatch')
        if previous.parameters_sha(model)!=m['frozen_parameter_hashes'][seed]:raise ValueError('source parameter hash mismatch')
        models[int(seed)]=model
    if set(models)!=set(cfg['model_seeds']):raise ValueError('ten source models required')
    resolved=json.loads((source/'resolved_config.json').read_text());panels={}
    if resolved['settings']!=cfg:raise ValueError('source resolved settings mismatch')
    for row in resolved['physical_profiles']:
        d=row['config'];d['service_time']=tuple(map(tuple,d['service_time']));d['restoration']=tuple(map(tuple,d['restoration']))
        c=WaitingConfig(**d);c.validate();panels[row['cohort'],row['condition']]=c
    if set(panels)!=set(itertools.product(cfg['cohorts'],cfg['conditions'])):raise ValueError('source physical grid mismatch')
    rows=list(csv.DictReader((source/'states.csv').open()))
    expected=set(itertools.product(cfg['conditions'],cfg['cohorts'],cfg['physical_shocks'],cfg['model_seeds'],cfg['origins'],cfg['landmarks']))
    seen=set()
    for row in tqdm(rows,desc='Validate complete historical state panel'):
        for key in ('cohort','shock','model_seed','tick','physical_seed','forecast_seed'):row[key]=int(row[key])
        identity=tuple(row[k] for k in KEYS)
        if identity in seen:raise ValueError('duplicate source state')
        seen.add(identity);c=panels[row['cohort'],row['condition']];s=state(row['state']);validate_state(c,s)
        pairs,policy=models[row['model_seed']].select(c,s,48-row['tick'],48)
        if pairs!=action(row['rl_action']) or policy['action_index']!=int(row['action_index']):raise ValueError('source policy root changed')
        if not np.allclose([policy['value'],policy['log_probability']],[float(row['value']),float(row['log_probability'])],rtol=1e-5,atol=1e-5):
            raise ValueError('source policy probability/value changed')
        ref=prior.rule_action(c,s,48-row['tick'],origin['selected_rules'][row['condition']])
        if ref!=action(row['rule_action']):raise ValueError('source reference root changed')
        if row['physical_seed']!=previous.physical_seed(row['cohort'],row['shock']):raise ValueError('source physical role changed')
        if row['forecast_seed']!=previous.forecast_seed(cfg,row['cohort'],row['shock'],row['model_seed'],row['origin'],row['tick']):
            raise ValueError('source forecast role changed')
    if seen!=expected or len(rows)!=2560:raise ValueError('incomplete source state grid')
    return models,panels,rows,dict(status='PASS',verified_source_files=len(m['artifacts']),source_states=len(rows),frozen_models=len(models),
        source_manifest_sha256=locked['state_source_manifest_sha256'],historical_outcomes_previously_opened=True)


def forecast_seed(cfg,row):
    return role_seed(cfg['forecast_seed'],f"{VERSION}:forecast:{row['cohort']}:{row['shock']}:{row['model_seed']}:{row['origin']}:{row['tick']}")


def seed_audit(cfg):
    path=ROOT/f'configs/{VERSION}_seed_registry.json';own=json.loads(path.read_text())['panels']
    historical=joint.integers(joint.exclusions()['historical_raw_panels']);roles=[];fresh=set()
    for p in (ROOT/'configs').glob('*seed_registry.json'):
        if p!=path:historical|=joint.integers(json.loads(p.read_text()))
    source_physical={previous.physical_seed(c,s) for c,s in itertools.product(settings('full')['cohorts'],settings('full')['physical_shocks'])}
    for profile in ('full','smoke'):
        panel=settings(profile);raw={k:panel[k] for k in ('forecast_seed','analysis_seed')}
        if own[profile]!=raw or set(raw.values())&(historical|fresh):raise ValueError('fresh raw seed collision/registry mismatch')
        fresh.update(raw.values())
        roles.extend(forecast_seed(panel,dict(cohort=c,shock=s,model_seed=m,origin=o,tick=t)) for c,s,m,o,t in
            itertools.product(panel['cohorts'],panel['physical_shocks'],panel['model_seeds'],panel['origins'],panel['landmarks']))
        roles.append(role_seed(panel['analysis_seed'],f'{VERSION}:crossed_bootstrap'))
    if cfg!=settings(cfg['profile']) or len(set(roles))!=len(roles) or set(roles)&(fresh|historical|source_physical):
        raise ValueError('settings/derived role collision')
    return dict(status='PASS',unique_derived_roles=len(roles),registry_sha256=sha(path),
        intentional_reuse='historical states/models/physical identities; conditions/roots/continuations paired by forecast tape')


def evaluate(row,c,model,cfg,journals):
    identity={k:row[k] for k in KEYS};s=state(row['state']);rl=action(row['rl_action']);ref=action(row['rule_action'])
    seed=forecast_seed(cfg,row);start=time.perf_counter()
    values=fork_values(c,s,rl,ref,model,seed,cfg['scenarios'],row['tick'],record_trace=cfg['profile']=='smoke')
    seconds=time.perf_counter()-start
    if cfg['profile']=='smoke':scalar_replay(c,s,values,model,row['tick'])
    pm=bool(values['label_indices'][0]!=values['label_indices'][2]);now=bool(rl)
    out=dict(**identity,forecast_seed=seed,source_physical_seed=row['physical_seed'],state=row['state'],
        rl_action=row['rl_action'],rule_action=row['rule_action'],rl_kind=kind(s,rl),rule_kind=kind(s,ref),
        pm_exposure=int(pm),now_exposure=int(now),structural_choice=int(len(joint.feasible(c,s))>1),
        remaining_physical_ticks=60-row['tick'],learned_intervals=48-row['tick'],tail_intervals=12,
        unique_roots=len(values['actions']),forecast_transitions=values['transitions'])
    available={m for pairs in joint.feasible(c,s) for m,j in pairs if not s.failed[m]}
    at_risk={m for m in available if s.ages[m]+1>=c.failure_age}
    pm_machines={m for m,j in rl if not s.failed[m]}
    out.update(busy_workers=sum(v>0 for v in s.remaining),healthy_opportunities=len(available),
        at_risk_opportunities=len(at_risk),young_opportunities=len(available-at_risk),
        pm_starts=len(pm_machines),at_risk_pm_starts=len(pm_machines&at_risk),young_pm_starts=len(pm_machines-at_risk),
        rl_wait=int(not rl),reference_wait=int(not ref))
    for contrast,target,exposure in [('pm','no_pm',pm),('now','wait',now)]:
        for arm in ARMS:
            delta,se=paired_delta(values,arm,'rl_root',target)
            prefix=contrast if arm=='rl' else f'{contrast}_deadline'
            out[f'{prefix}_delta']=delta if exposure else 0.;out[f'{prefix}_mc_se']=se if exposure else 0.
        # Paired continuation interaction uses the very same scenarios.
        i,j=values['label_indices'][0],values['label_indices'][LABELS.index(target)]
        interaction=(values['costs'][0,i]-values['costs'][0,j])-(values['costs'][1,i]-values['costs'][1,j])
        out[f'{contrast}_interaction_delta']=float(interaction.mean()) if exposure else 0.
        out[f'{contrast}_interaction_mc_se']=float(interaction.std(ddof=1)/np.sqrt(cfg['scenarios'])) if exposure else 0.
        out[f'{contrast}_normalized_delta']=out[f'{contrast}_delta']/(4*(60-row['tick']))
    for arm_i,arm in enumerate(ARMS):
        delta,se=paired_delta(values,arm,'rl_root','reference')
        out[f'{arm}_reference_delta']=delta if out['structural_choice'] else 0.
        out[f'{arm}_reference_mc_se']=se if out['structural_choice'] else 0.
        for label,idx in zip(LABELS,values['label_indices'],strict=True):out[f'{arm}_{label}_cost']=float(values['costs'][arm_i,idx].mean())
        for idx,pairs in enumerate(values['actions']):
            journals['root_values'].add(dict(**identity,continuation=arm,action_index=CATALOGUE.index(pairs),action=prior.compact(pairs),
                labels=prior.compact([name for name,i in zip(LABELS,values['label_indices']) if i==idx]),
                total_mean=float(values['costs'][arm_i,idx].mean()),mc_se=float(values['costs'][arm_i,idx].std(ddof=1)/np.sqrt(cfg['scenarios'])),
                base_mean=float(values['base'][arm_i,idx].mean()),tail_mean=float(values['tail'][arm_i,idx].mean()),
                first_step_mean=float(values['first'][arm_i,idx].mean()),
                **{k:float(v[idx].mean()) for k,v in values['component_costs'][arm_i].items()},
                cutoff_busy_workers=float((values['cutoff'][arm_i]['remaining'][idx]>0).sum(1).mean()),
                terminal_busy_workers=float((values['terminal'][arm_i]['remaining'][idx]>0).sum(1).mean()),
                terminal_failed_machines=float(values['terminal'][arm_i]['failed'][idx].sum(1).mean()),
                scenario_costs=prior.compact(values['costs'][arm_i,idx].tolist())))
    journals['rollout_audits'].add(dict(**identity,feasibility='PASS',components='PASS',training_clock='H48_L12_PASS',
        scalar_replay='PASS' if cfg['profile']=='smoke' else 'NOT_SCHEDULED',**values['counters'],
        transitions=values['transitions']))
    return out,values,seconds


def run(output,state_source_dir,profile='smoke',device='cpu'):
    if device!='cpu':raise ValueError('CPU only')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('new/empty directory required')
    output.mkdir(parents=True,exist_ok=True);cfg=settings(profile);torch.set_num_threads(1)
    now=lambda:datetime.now(timezone.utc).isoformat();git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    start=time.perf_counter()
    m=dict(protocol=VERSION,status='RUNNING',started=now(),commit=git('rev-parse','HEAD'),dirty=bool(git('status','--porcelain')),
        settings=cfg,expected_counts=counts(cfg),device=device,platform=platform.platform(),python=platform.python_version(),
        numpy=np.__version__,torch=str(torch.__version__),learning='NOT_APPLICABLE: frozen deterministic inference',
        historical_state_outcomes_previously_opened=True,new_full_forecast_outcomes_opened=False,
        thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    prior.write_json(output/'manifest.json',m);journals={};rows=[];cost_arrays={k:[] for k in ('costs','base','tail')};boundaries={};seconds=0.
    counters=dict(batched_decision_rows=0,near_tie_scalar_rows=0,rl_action_histogram=[0]*21,rl_continuation_transition_rows=0,tail_transition_rows=0)
    try:
        if profile=='full' and m['dirty']:raise ValueError('full requires clean committed code')
        if profile=='full' and any(v!='1' for v in m['thread_environment'].values()):raise ValueError('full requires OMP/OPENBLAS/MKL_NUM_THREADS=1')
        locked=contract();models,panels,source_rows,audit=verify_source(state_source_dir,locked)
        m['source_audit']=audit;m['seed_audit']=seed_audit(cfg)
        selected=[r for r in source_rows if all(r[k] in cfg[v] for k,v in
            [('condition','conditions'),('cohort','cohorts'),('shock','physical_shocks'),('model_seed','model_seeds'),('origin','origins'),('tick','landmarks')])]
        if len(selected)!=counts(cfg)['states']:raise RuntimeError('source selection count mismatch')
        prior.write_json(output/'source_contract.json',locked);prior.write_json(output/'resolved_config.json',dict(settings=cfg,
            physical_profiles=[dict(cohort=k[0],condition=k[1],config=asdict(c)) for k,c in sorted(panels.items())]))
        shutil.copyfile(ROOT/f'configs/{VERSION}_seed_registry.json',output/'seed_registry.json')
        shutil.copyfile(ROOT/f'docs/{VERSION}_plan.md',output/'plan.md')
        snapshot=output/'source_snapshot';snapshot.mkdir()
        for path in list((ROOT/'src/ht_pdm_fjsp').glob('maintenance_*.py'))+[ROOT/'pyproject.toml',ROOT/'uv.lock']:
            shutil.copyfile(path,snapshot/path.name)
        for path in (ROOT/'configs').glob('maintenance_*json'):shutil.copyfile(path,snapshot/path.name)
        initial={}
        for seed in cfg['model_seeds']:
            dest=output/'frozen_models'/f'seed_{seed}.pt';dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(Path(state_source_dir)/'frozen_models'/dest.name,dest)
            restored,_=joint.load_checkpoint(dest)
            if previous.parameters_sha(restored)!=previous.parameters_sha(models[seed]):raise RuntimeError('model weight roundtrip mismatch')
            for c in panels.values():
                s=WaitingEnv(c).state
                if restored.select(c,s,48,48)!=models[seed].select(c,s,48,48):raise RuntimeError('policy roundtrip mismatch')
            models[seed]=restored;initial[seed]=previous.parameters_sha(restored)
        for name in ('states.partial','root_values','rollout_audits'):journals[name]=joint.CountingJournal(output/f'{name}.csv')
        m['new_full_forecast_outcomes_opened']=profile=='full';prior.write_json(output/'manifest.json',m)
        print(json.dumps(dict(protocol=VERSION,budget=counts(cfg),training='NONE',objective='complete H48/L12 remaining returns'),indent=2),flush=True)
        for row in tqdm(selected,desc='Paired RL/deadline continuation states'):
            result,values,elapsed=evaluate(row,panels[row['cohort'],row['condition']],models[row['model_seed']],cfg,journals)
            rows.append(result);journals['states.partial'].add(result);seconds+=elapsed
            indices=list(values['label_indices'])
            for k in cost_arrays:cost_arrays[k].append(values[k][:,indices,:])
            for stage in ('cutoff','terminal'):
                for k in values[stage][0]:boundaries.setdefault(f'{stage}_{k}',[]).append(np.stack([b[k][indices] for b in values[stage]]))
            for k,value in values['counters'].items():counters[k]=(np.asarray(counters[k])+value).tolist() if k=='rl_action_histogram' else counters[k]+value
        if any(previous.parameters_sha(models[k])!=v for k,v in initial.items()):raise RuntimeError('frozen parameters changed')
        summary,cells,strata=summarize(rows,cfg)
        prior.write_json(output/'summary.json',summary);prior.write_csv(output/'profile_cells.csv',cells);prior.write_csv(output/'states.csv',rows)
        prior.write_csv(output/'continuation_strata.csv',strata)
        prior.write_csv(output/'behavior_strata.csv',summary['behavior_strata'])
        np.savez_compressed(output/'raw_forecast_costs.npz',**{k:np.stack(v) for k,v in cost_arrays.items()})
        np.savez_compressed(output/'raw_boundary_states.npz',**{k:np.stack(v) for k,v in boundaries.items()})
        actual=dict(states=len(rows),models=len(cfg['model_seeds']),unique_roots=sum(r['unique_roots'] for r in rows),
            forecast_transitions=sum(r['forecast_transitions'] for r in rows),root_value_rows=journals['root_values'].rows,**counters)
        if (actual['states']!=counts(cfg)['states'] or journals['states.partial'].rows!=len(rows)
            or journals['rollout_audits'].rows!=len(rows) or actual['root_value_rows']!=2*actual['unique_roots']
            or actual['forecast_transitions']>counts(cfg)['max_forecast_transitions']
            or actual['tail_transition_rows']!=2*actual['unique_roots']*cfg['scenarios']*12
            or sum(actual['rl_action_histogram'])!=actual['batched_decision_rows']):raise RuntimeError('budget/counter audit')
        m.update(status='COMPLETED',audit='PASS',actual_counts=actual,fork_seconds=seconds,
            frozen_parameter_hashes={str(k):v for k,v in initial.items()})
    except BaseException as error:
        m.update(status='FAILED',error=f'{type(error).__name__}: {error}');raise
    finally:
        for journal in journals.values():journal.close()
        m['finished']=now();m['elapsed_seconds']=time.perf_counter()-start
        m['artifacts']={str(p.relative_to(output)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        prior.write_json(output/'manifest.json',m)
    return m


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('full','smoke'),default='smoke');parser.add_argument('--device',choices=('cpu',),default='cpu')
    parser.add_argument('--state-source-dir',required=True,type=Path);parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();run(args.output_dir,args.state_source_dir,args.profile,args.device)


if __name__=='__main__':main()
