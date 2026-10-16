"""Frozen policy decisions on existing states; never advances physical time."""
from __future__ import annotations
import argparse,csv,json,shutil,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_capacity_factorized_experiment as x
from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue

ROOT=Path(__file__).resolve().parents[2]
VERSION='technician_capacity_service_choice_diagnostic_v1'
PLAN=ROOT/'docs'/f'{VERSION}_plan.md'
METHODS=('residual_capacity_ppo','flat_capacity_ppo','independent_capacity_ppo','ppo')


@torch.no_grad()
def core_trace(model,obs):
    before=x.learner.parameter_digest(model);original=model.conditionals
    had='conditionals' in model.__dict__;old=model.__dict__.get('conditionals');traces=[]
    def instrument(batch,base,calendar,planned,technician,position,reserved):
        log,support=original(batch,base,calendar,planned,technician,position,reserved)
        traces.append(dict(batch=batch,calendar=calendar.clone(),planned=planned.clone(),
            technician=technician.clone(),position=position,reserved=reserved.clone(),
            log=log.clone(),support=support.clone()))
        return log,support
    model.conditionals=instrument
    try:action=model.select(obs)
    finally:
        if had:model.conditionals=old
        else:delattr(model,'conditionals')
    if len(traces)!=2 or not np.array_equal(action,model.select(obs)) or x.learner.parameter_digest(model)!=before:
        raise RuntimeError('instrumentation changes policy')
    return action,traces


@torch.no_grad()
def under_prefix(model,context):
    batch=context['batch'];base=model.encoder(model.raw_pairs(batch))
    log,support=model.conditionals(batch,base,context['calendar'],context['planned'],
        context['technician'],context['position'],context['reserved'])
    if not torch.equal(support,context['support']):raise RuntimeError('matched prefix support mismatch')
    return log,support


def duration(obs,m,j):return int(round(float(obs['edges'][m,j,1])*12))


def executed_pairs(obs,action,method):
    if method=='ppo':
        actions=catalogue(4,2)
        if not 0<=int(action)<len(actions):raise ValueError('PPO catalogue token')
        requests=list(actions[int(action)])
    else:
        a=np.asarray(action)
        if a.shape!=(3,) or a[-1]!=0 or ((a[:2]<0)|(a[:2]>4)).any():raise ValueError('canonical vector')
        requests=[(int(v)-1,j) for j,v in enumerate(a[:2]) if v]
    if any(not obs['pair_mask'][m,j] for m,j in requests):raise ValueError('masked request')
    pairs=[]
    for m in range(4):
        group=[p for p in requests if p[0]==m]
        if group:
            if len(group)>1 and method!='independent_capacity_ppo':raise ValueError('unresolved duplicate')
            pairs.append(max(group,key=lambda p:(float(obs['edges'][p[0],p[1],2])/duration(obs,*p),-p[1])))
    return pairs


def ppo_observation(obs):
    result=dict(obs)
    result['joint_mask']=np.array([all(obs['pair_mask'][m,j] for m,j in a) for a in catalogue(4,2)],bool)
    return result


def slot_description(context,log,support):
    batch=context['batch'];j=int(context['technician'][0]);p=log[0].exp().cpu().numpy();mode=int(p.argmax())
    allowed=np.flatnonzero(support[0,1:].cpu().numpy())
    durations=[int(round(float(batch['edges'][0,m,j,1])*12)) for m in allowed]
    return dict(legal_count=len(allowed),min_legal_duration=min(durations) if durations else 0,
        probability_go=float(1-p[0]),go_mode=int(mode>0),mode_token=mode,
        selected_duration=int(round(float(batch['edges'][0,mode-1,j,1])*12)) if mode else 0,
        probabilities=json.dumps(p.tolist()))


def summarize(own,matched):
    expected={(r['train_seed'],r['state_index']) for r in matched if r['method']=='residual_capacity_ppo'}
    lookup={(r['method'],r['train_seed'],r['state_index']):r for r in matched}
    if len(lookup)!=2*len(expected) or any((a,*key) not in lookup for key in expected for a in METHODS[:2]):raise ValueError('matched grid')
    slow=[key for key in expected if lookup[('residual_capacity_ppo',*key)]['min_legal_duration']>=2]
    fast=[key for key in expected if lookup[('residual_capacity_ppo',*key)]['min_legal_duration']==1]
    def strata(keys):
        c=[lookup['residual_capacity_ppo',*k] for k in keys];f=[lookup['flat_capacity_ppo',*k] for k in keys]
        return dict(contexts=len(keys),candidate_go_rate=float(np.mean([r['go_mode'] for r in c])) if c else None,
            flat_go_rate=float(np.mean([r['go_mode'] for r in f])) if f else None,
            candidate_go_flat_wait=sum(a['go_mode'] and not b['go_mode'] for a,b in zip(c,f)),
            candidate_wait_flat_go=sum(not a['go_mode'] and b['go_mode'] for a,b in zip(c,f)),
            candidate_mean_pgo=float(np.mean([r['probability_go'] for r in c])) if c else None,
            flat_mean_pgo=float(np.mean([r['probability_go'] for r in f])) if f else None)
    full={}
    for a in METHODS:
        rows=[r for r in own if r['method']==a];starts=sum(r['starts'] for r in rows);long=sum(r['long_starts'] for r in rows)
        full[a]=dict(states=len(rows),executed_starts=starts,long_starts=long,long_fraction=long/starts if starts else 0.,
            service_ticks_committed=sum(r['service_ticks_committed'] for r in rows),
            mean_duration_committed=sum(r['service_ticks_committed'] for r in rows)/starts if starts else 0.)
    s=strata(slow);long_delta=full['residual_capacity_ppo']['long_fraction']-full['flat_capacity_ppo']['long_fraction']
    go_delta=s['candidate_go_rate']-s['flat_go_rate'] if slow else None
    checks=dict(sufficient_slow_contexts=len(slow)>=12,slow_go_mode_difference=go_delta is not None and go_delta>=.20,
                long_start_fraction_difference=long_delta>=.05)
    return dict(slow_only=s,fast_available=strata(fast),own_policy=full,
        slow_go_rate_difference=go_delta,long_fraction_difference=long_delta,
        engineering_checks=checks,legal_option_followup_motivated=all(checks.values()))


def run(source,output):
    source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);start=time.monotonic()
    def guard():
        if time.monotonic()-start>600:raise TimeoutError('10min hardstop')
    read=lambda name:json.loads((source/name).read_text())
    m=dict(protocol=VERSION,status='RUNNING',classification='POSTHOC_FROZEN_STATE_DIAGNOSTIC',goal_pass=False,test_opened=False,
        new_physical_episodes=0,training_steps=0,source=str(source),source_manifest_sha256=x.sha(source/'manifest.json'),
        started_utc=datetime.now(timezone.utc).isoformat(),git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())))
    x.write_json(out/'manifest.json',m)
    try:
        original=read('manifest.json');c=read('resolved_config.json');choices=read('checkpoint_choices.json')
        if (m['git']['dirty'] or original['status']!='COMPLETED' or original['audit_status']!='PASS' or original['test_opened']
            or original['git']['commit']!='a10e049e3f8b4e242efbc879b2d3dc3c13925bf4' or c!=x.settings('development')
            or len(original['output_bindings'])!=548):raise RuntimeError('frozen prerequisite')
        for name,meta in tqdm(original['output_bindings'].items(),desc='Verify548frozen inputs'):
            guard();p=source/name
            if x.sha(p)!=meta['sha256'] or p.stat().st_size!=meta['bytes']:raise RuntimeError('changed source '+name)
        for name,h in original['source_seal'].items():
            if x.sha(source/name)!=h:raise RuntimeError('source seal')
            if name.startswith('source_snapshot/') and x.sha(Path(__file__).parent/Path(name).name)!=h:raise RuntimeError('changed learning source')
        shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(__file__,out/'diagnostic.py');own=[];matched=[];inputs=[]
        for t in tqdm(c['train_seeds'],desc='Three frozen replicas'):
            models={a:x.load_model(source/choices[f'4:{a}:{t}']['path'],4,c,t,a,choices[f'4:{a}:{t}']['steps']) for a in METHODS}
            before={a:x.learner.parameter_digest(v) for a,v in models.items()}
            choice=choices[f'4:residual_capacity_ppo:{t}'];meta=choice['latency_observations'];p=source/meta['path']
            if x.sha(p)!=meta['sha256'] or meta['states']!=48 or meta['seed']!=230100000:raise RuntimeError('retained observations binding')
            inputs.append(dict(train_seed=t,observations=meta,checkpoints={a:choices[f'4:{a}:{t}'] for a in METHODS}))
            with np.load(p,allow_pickle=False) as archive:
                for i in tqdm(range(48),desc=f'Frozen states{t}',leave=False):
                    guard();obs={k:archive[k][i].copy() for k in archive.files}
                    if abs(48-float(obs['global_features'][0])*12-i)>1e-4:raise RuntimeError('physical state tick')
                    candidate_context=None
                    for a,model in models.items():
                        if a=='ppo':action=model.select(ppo_observation(obs))
                        else:
                            action,traces=core_trace(model,obs)
                            if a=='residual_capacity_ppo':candidate_context=traces[1]
                        pairs=executed_pairs(obs,action,a);durations=[duration(obs,*p) for p in pairs]
                        own.append(dict(method=a,train_seed=t,state_index=i,checkpoint_steps=choices[f'4:{a}:{t}']['steps'],
                            submitted=json.dumps(int(action) if np.isscalar(action) else action.tolist()),executed_pairs=json.dumps(pairs),
                            starts=len(pairs),long_starts=sum(d>1 for d in durations),service_ticks_committed=sum(durations)))
                    for a in METHODS[:2]:
                        log,support=under_prefix(models[a],candidate_context)
                        matched.append(dict(method=a,train_seed=t,state_index=i,**slot_description(candidate_context,log,support)))
            if any(x.learner.parameter_digest(models[a])!=before[a] for a in METHODS):raise RuntimeError('model mutated')
        if len(own)!=576 or len(matched)!=288:raise RuntimeError('exact state grid')
        summary=summarize(own,matched)
        summary['per_replica']={str(t):summarize([r for r in own if r['train_seed']==t],[r for r in matched if r['train_seed']==t]) for t in c['train_seeds']}
        summary.update(classification=m['classification'],goal_pass=False,test_opened=False,new_physical_episodes=0,
            limitation='144correlated candidate-visited states from first open DEVseed; shadow choices are not physical-return comparisons or independent test samples.')
        x.write_csv(out/'own_policy_decisions.csv',own);x.write_csv(out/'matched_prefix.csv',matched);x.write_json(out/'inputs.json',inputs);x.write_json(out/'summary.json',summary)
        (out/'report.md').write_text('# Frozen service-choice diagnostic\n\nPOSTHOC DEVELOPMENT ONLY; no new physical trajectory or training.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
        for name,meta in original['output_bindings'].items():
            guard()
            if x.sha(source/name)!=meta['sha256']:raise RuntimeError('source changed during diagnosis')
        if x.sha(source/'manifest.json')!=m['source_manifest_sha256']:raise RuntimeError('source manifest changed')
        m.update(status='COMPLETED',audit_status='PASS',elapsed_seconds=time.monotonic()-start,
                 output_bindings={p.name:dict(sha256=x.sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
        x.write_json(out/'manifest.json',m);print(json.dumps(dict(output=str(out),**summary)));return m
    except Exception as exc:
        m.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);x.write_json(out/'manifest.json',m);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();run(a.source,a.output_dir)

if __name__=='__main__':main()
