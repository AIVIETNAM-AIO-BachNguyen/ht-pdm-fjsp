"""Frozen, single-token simulator interventions; never creates learning targets."""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, shutil, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_capacity_factorized_experiment as x
from ht_pdm_fjsp.technician_capacity_service_choice_diagnostic import duration

ROOT=Path(__file__).resolve().parents[2]
VERSION='technician_capacity_prefix_counterfactual_v1'
PLAN=ROOT/'docs'/f'{VERSION}_plan.md'
BRANCHES=('original','wait','shortest','flat_first')
METRICS=('extended_cost','availability','terminal_pending','policy_failures')


@torch.no_grad()
def force_first(model,obs,token):
    """Force one legal first token, keeping candidate T2 conditional unchanged."""
    if not isinstance(token,(int,np.integer)) or not 0<=token<=4:
        raise ValueError('first token range')
    original=model.conditionals;had='conditionals' in model.__dict__
    saved=model.__dict__.get('conditionals');positions=[]
    def conditional(*args):
        log,support=original(*args);position=args[5];positions.append(position)
        if position==0:
            if not support[0,token]:raise ValueError('masked first token')
            log=torch.full_like(log,-1e9);log[:,token]=0
        return log,support
    model.conditionals=conditional
    try:action=model.select(obs)
    finally:
        if had:model.conditionals=saved
        else:delattr(model,'conditionals')
    if positions!=[0,1] or action[0]!=token:raise RuntimeError('single-prefix intervention')
    return action


def shortest_first(obs):
    legal=np.flatnonzero(obs['pair_mask'][:,0])
    return int(min(legal,key=lambda m:(duration(obs,int(m),0),int(m))))+1 if len(legal) else 0


def equal_observation(actual,expected):
    if actual.keys()!=expected.keys() or any(not np.array_equal(actual[k],expected[k]) for k in actual):
        raise RuntimeError('source observation replay mismatch')


def equal_metrics(actual,expected):
    for key,value in actual.items():
        if key not in expected:raise RuntimeError('source missing metric '+key)
        same=(str(value)==str(expected[key]) if isinstance(value,str)
              else abs(float(value)-float(expected[key]))<=1e-8)
        if not same:raise RuntimeError('source metric replay mismatch '+key)


def finish(env,model,first,remaining,guard=lambda:None,expected=None):
    """Advance a private copy only; compare every decision in no-op controls."""
    env=copy.deepcopy(env);obs=env.observe();done=False
    for offset in range(remaining):
        guard()
        if expected is not None:equal_observation(obs,expected[offset])
        action=first if offset==0 else model.select(obs)
        obs,_,done,info=env.step(action)
        if done!=(offset==remaining-1):raise RuntimeError('remaining horizon')
    if not done or info['episode_metrics']['physical_ticks']!=60:
        raise RuntimeError('common deadline tail absent')
    return info['episode_metrics']


def summarize(rows):
    keys={(r['train_seed'],r['state_index']) for r in rows}
    table={(r['train_seed'],r['state_index'],r['branch']):r for r in rows}
    if len(table)!=len(rows) or len(table)!=len(keys)*4 or any((*k,b) not in table for k in keys for b in BRANCHES):
        raise ValueError('incomplete branch grid')
    seeds=sorted({k[0] for k in keys})
    def summary(group):
        result={}
        for branch in BRANCHES:
            selected=[table[*k,branch] for k in group]
            result[branch]=dict(contexts=len(selected),
                mean_gain_per_machine=float(np.mean([r['gain_per_machine'] for r in selected])) if selected else None,
                improved=sum(r['gain_per_machine']>1e-8 for r in selected),
                worsened=sum(r['gain_per_machine']< -1e-8 for r in selected),
                unchanged=sum(abs(r['gain_per_machine'])<=1e-8 for r in selected),
                changed_tokens=sum(r['original_token']!=r['token'] for r in selected),
                **{f'mean_{key}':float(np.mean([r[key] for r in selected])) if selected else None
                   for key in ('availability_delta','pending_delta_per_machine','failures_delta_per_machine')})
        return result
    groups=dict(all=keys,four_tick={k for k in keys if table[*k,'original']['original_duration']==4},
        other_positive={k for k in keys if table[*k,'original']['original_duration'] in (1,2)},
        original_wait={k for k in keys if table[*k,'original']['original_duration']==0})
    result={label:summary(group) for label,group in groups.items()}
    per={str(t):{label:summary({k for k in group if k[0]==t}) for label,group in groups.items()} for t in seeds}
    w=result['four_tick']['wait']
    checks=dict(sufficient_contexts=w['contexts']>=6 and len(seeds)==3 and all(per[str(t)]['four_tick']['wait']['contexts']>0 for t in seeds),
        mean_cost_gain=w['mean_gain_per_machine'] is not None and w['mean_gain_per_machine']>=1.,
        every_replica_positive=bool(seeds) and all((per[str(t)]['four_tick']['wait']['mean_gain_per_machine'] or 0)>0 for t in seeds),
        availability=w['mean_availability_delta'] is not None and w['mean_availability_delta']>=-.01,
        pending=w['mean_pending_delta_per_machine'] is not None and w['mean_pending_delta_per_machine']<=.025,
        failures=w['mean_failures_delta_per_machine'] is not None and w['mean_failures_delta_per_machine']<=.25)
    return dict(strata=result,per_replica=per,engineering_checks=checks,deferral_explanation_supported=all(checks.values()),
        classification='POSTHOC_SIMULATOR_INTERVENTION',goal_pass=False,test_opened=False,training_steps=0,
        limitation='Dependent states from three candidate DEV trajectories, one realized event tape per replica; frozen-candidate continuation only. No population CI or optimal-action labels.')


def run(source,output):
    source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);start=time.monotonic()
    def guard():
        if time.monotonic()-start>600:raise TimeoutError('10min hardstop')
    read=lambda name:json.loads((source/name).read_text())
    manifest=dict(protocol=VERSION,status='RUNNING',goal_pass=False,test_opened=False,training_steps=0,
        fresh_physical_seeds=0,counterfactual_branches=432,noop_replay_branches=144,source_prefix_replays=3,
        source=str(source),source_manifest_sha256=x.sha(source/'manifest.json'),
        git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())))
    x.write_json(out/'manifest.json',manifest)
    try:
        original=read('manifest.json');c=read('resolved_config.json');choices=read('checkpoint_choices.json')
        if manifest['git']['dirty'] or original['status']!='COMPLETED' or original['audit_status']!='PASS' or original['test_opened'] or original['git']['commit']!='a10e049e3f8b4e242efbc879b2d3dc3c13925bf4' or c!=x.settings('development') or len(original['output_bindings'])!=548:
            raise RuntimeError('frozen prerequisite')
        def verify():
            for name,meta in tqdm(original['output_bindings'].items(),desc='Verify548source bindings',leave=False):
                guard();p=source/name
                if x.sha(p)!=meta['sha256'] or p.stat().st_size!=meta['bytes']:raise RuntimeError('source changed '+name)
            for name,h in original['source_seal'].items():
                if x.sha(source/name)!=h:raise RuntimeError('source seal')
                if name.startswith('source_snapshot/') and x.sha(Path(__file__).parent/Path(name).name)!=h:raise RuntimeError('learning source changed')
            if x.sha(source/'manifest.json')!=manifest['source_manifest_sha256']:raise RuntimeError('source manifest changed')
        verify();shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(__file__,out/'diagnostic.py')
        with (source/'development_episodes.csv').open() as f:dev=list(csv.DictReader(f))
        rows=[];controls=[];inputs=[]
        for t in tqdm(c['train_seeds'],desc='Three frozen replicas'):
            candidate=choices[f'4:residual_capacity_ppo:{t}'];flat=choices[f'4:flat_capacity_ppo:{t}']
            model=x.load_model(source/candidate['path'],4,c,t,'residual_capacity_ppo',candidate['steps'])
            flat_model=x.load_model(source/flat['path'],4,c,t,'flat_capacity_ppo',flat['steps'])
            digests=[x.learner.parameter_digest(m) for m in (model,flat_model)]
            meta=candidate['latency_observations']
            if meta['states']!=48 or meta['seed']!=230100000 or x.sha(source/meta['path'])!=meta['sha256']:raise RuntimeError('state binding')
            with np.load(source/meta['path'],allow_pickle=False) as archive:
                saved=[{k:archive[k][i].copy() for k in archive.files} for i in range(48)]
            selected=[r for r in dev if r['method']=='residual_capacity_ppo' and int(r['train_seed'])==t and int(r['seed'])==230100000 and int(r['checkpoint_steps'])==candidate['steps']]
            if len(selected)!=1:raise RuntimeError('unique source episode')
            baseline={k:float(selected[0][k]) for k in METRICS}
            env=x.ResourceEnv(230100000,4,'residual_capacity_ppo');obs=env.reset()
            inputs.append(dict(train_seed=t,candidate=candidate,flat=flat,source_episode=selected[0]))
            for i in tqdm(range(48),desc=f'Intervene replica{t}',leave=False):
                guard();equal_observation(obs,saved[i]);original_action=model.select(obs)
                alternatives=dict(original=int(original_action[0]),wait=0,shortest=shortest_first(obs),flat_first=int(flat_model.select(obs)[0]))
                before_state=copy.deepcopy(env.env.state);before_metrics=copy.deepcopy(env.env.metrics)
                for name,token in alternatives.items():
                    action=force_first(model,obs,token)
                    if name=='original' and not np.array_equal(action,original_action):raise RuntimeError('no-op action mismatch')
                    result=finish(env,model,action,48-i,guard,saved[i:] if token==original_action[0] else None)
                    if token==original_action[0]:equal_metrics(result,selected[0])
                    rows.append(dict(train_seed=t,state_index=i,branch=name,original_token=int(original_action[0]),token=token,
                        original_duration=duration(obs,int(original_action[0])-1,0) if original_action[0] else 0,
                        alternative_duration=duration(obs,token-1,0) if token else 0,action=json.dumps(action.tolist()),
                        gain_per_machine=(baseline['extended_cost']-result['extended_cost'])/4,
                        availability_delta=result['availability']-baseline['availability'],
                        pending_delta_per_machine=(result['terminal_pending']-baseline['terminal_pending'])/4,
                        failures_delta_per_machine=(result['policy_failures']-baseline['policy_failures'])/4,
                        **{k:result[k] for k in METRICS}))
                if env.env.state!=before_state or env.env.metrics!=before_metrics:raise RuntimeError('private branch copy mutated source')
                controls.append(dict(train_seed=t,state_index=i,observation_sha256=hashlib.sha256(obs['state'].tobytes()).hexdigest(),action=json.dumps(original_action.tolist()),noop_metrics_replayed=True))
                obs,_,done,info=env.step(original_action)
            if not done:raise RuntimeError('source replay terminal')
            equal_metrics(info['episode_metrics'],selected[0])
            if digests!=[x.learner.parameter_digest(m) for m in (model,flat_model)]:raise RuntimeError('weights changed')
        if len(rows)!=576 or len(controls)!=144:raise RuntimeError('fixed corpus grid')
        summary=summarize(rows);x.write_csv(out/'branches.csv',rows);x.write_csv(out/'replay_controls.csv',controls)
        x.write_json(out/'inputs.json',inputs);x.write_json(out/'summary.json',summary)
        (out/'report.md').write_text('# Frozen first-technician intervention\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
        verify();manifest.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,
            output_bindings={p.name:dict(sha256=x.sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
        x.write_json(out/'manifest.json',manifest);print(json.dumps(dict(output=str(out),**summary)));return manifest
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);x.write_json(out/'manifest.json',manifest);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();run(a.source,a.output_dir)

if __name__=='__main__':main()
