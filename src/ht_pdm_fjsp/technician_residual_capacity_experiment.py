"""Residual service-capacity PPO: one registered full N4/K2 comparison."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime,timezone
import csv
import hashlib
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import maintenance_balanced_algorithms as learner
from ht_pdm_fjsp.maintenance_balanced_confirmation_mac import registered_integers
from ht_pdm_fjsp.maintenance_opportunity_cost import Journal,write_json,write_csv
from ht_pdm_fjsp.maintenance_dispatch import role_seed as physical_seed
from ht_pdm_fjsp.technician_reservation_experiment import Tags,pool_results
from ht_pdm_fjsp.technician_residual_capacity_env import VERSION,GRAPH_ARMS,CORE_ARMS,COUNTERS,ResourceEnv,scaled_config,value_unit
from ht_pdm_fjsp.technician_residual_capacity_policy import ResidualCapacityPolicy

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'docs'/f'{VERSION}_plan.md'
REGISTRY=ROOT/'configs'/f'{VERSION}_seed_registry.json'
PROFILES=('engineering','development','confirmation')
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()
def arms(n):
    if n != 4:raise ValueError('N4/K2 first; larger experiments are not registered')
    return (*CORE_ARMS,'ppo','normalized_ppo','mappo','full_mappo')
def settings(profile):
    if profile not in PROFILES:raise ValueError('profile')
    raw=json.loads(REGISTRY.read_text())['panels'][profile];engineering=profile=='engineering'
    return dict(profile=profile,train_seeds=raw['train_seeds'],analysis_seed=raw['analysis_seed'],
        development_seeds=list(range(raw['development_start'],raw['development_start']+raw['development_count'])),
        test_seeds=list(range(raw['test_start'],raw['test_start']+raw['test_count'])) if 'test_start' in raw else [],
        scales=[4],env_steps=384 if engineering else 491520,
        vector_envs=2 if engineering else 16,rollout=192 if engineering else 3072,
        epochs=2 if engineering else 4,minibatch=48 if engineering else 256,
        checkpoints=[192,384] if engineering else [122880,245760,368640,491520],
        hidden=27,baseline_hidden=64,mixer_hidden=32,learning_rate=.0003,clip=.2,entropy_weight=.01,
        value_weight=.5,gradient_clip=.5,gamma=1.,epsilon_start=0.,epsilon_end=0.,torch_threads=1,training_processes=8,
        behavior_log_atol=2e-5,behavior_log_rtol=0.,behavior_value_atol=2e-5,behavior_value_rtol=1e-6,
        value_unit_formula='unavailable_cost*N*48/20',behavior_value_atol_units='normalized',
        bootstrap_samples=200 if engineering else 200000,confirmation_round=2,round_alpha=.05/6,
        confidence=1-(.05/6)/15,max_seconds=43200,teacher_assistance=False,pretraining_steps=0)

def seed_audit():
    history=set();hashes={}
    for path in sorted((ROOT/'configs').glob('*seed_registry.json')):
        if path!=REGISTRY:
            history|=registered_integers(json.loads(path.read_text()));hashes[path.name]=sha(path)
    groups=[];derived=[]
    for profile in PROFILES:
        c=settings(profile);groups.extend([set(c['train_seeds']),set(c['development_seeds']),set(c['test_seeds']),{c['analysis_seed']}])
        derived.append(learner.role_seed(c['analysis_seed'],VERSION+':bootstrap'))
        for t in c['train_seeds']:
            for role in ('graph','ppo','mappo','full_mappo'):
                derived.append(learner.role_seed(t,f'{VERSION}:initialization:{role}'))
        for n in c['scales']:
            for s in (*c['development_seeds'],*c['test_seeds']):derived.append(physical_seed(s,f'{VERSION}:N{n}:physical'))
            for t in c['train_seeds']:
                for a in sorted({rng_role(a) for a in arms(n)}):
                    for role in ('actions','minibatches','replay'):
                        derived.append(learner.role_seed(t,f'{VERSION}:N{n}:{a}:{role}'))
                for w in range(c['vector_envs']):
                    for e in range(c['env_steps']//(48*c['vector_envs'])):
                        derived.append(physical_seed(t,f'{VERSION}:N{n}:training_physical:worker:{w}:episode:{e}'))
    own=set.union(*groups)
    if any(a&b for a,b in itertools.combinations(groups,2)) or own&history:raise ValueError('raw seed overlap')
    if len(derived)!=len(set(derived)) or set(derived)&(own|history):raise ValueError('derived seed overlap')
    return dict(status='PASS',raw_unique=len(own),derived_unique=len(derived),historical_registry_hashes=hashes,
        intentional_pairing='same initial tensors/action/minibatch RNG across graph arms; same physical tapes across all methods per size')

class FullMAPPO(learner.Model):
    is_actor=True
    joint_policy=False
    def __init__(self,hidden=64):
        super().__init__('mappo',hidden);self.algorithm='full_mappo';self.actor=learner.mlp(75,hidden,3)
    def policy(self,batch):return learner.masked(self.actor(batch['full_local']),batch['local_mask'])
    @torch.no_grad()
    def sample(self,batch,generator=None,deterministic=False,epsilon=0):
        logits=self.policy(batch);dist=torch.distributions.Categorical(logits=logits)
        action=logits.argmax(-1) if deterministic else torch.multinomial(dist.probs.reshape(-1,3),1,generator=generator).reshape(logits.shape[:-1])
        return action,dist.log_prob(action),self.critic(batch['state']).squeeze(-1)

def rng_role(algorithm):
    return 'graph' if algorithm in CORE_ARMS else 'ppo' if algorithm=='normalized_ppo' else algorithm


class ValueUnits(torch.nn.Module):
    def __init__(self, net, unit):
        super().__init__()
        self.net = net
        self.unit = float(unit)

    def forward(self, inputs):
        return self.net(inputs) * self.unit


class NormalizedPPO(learner.Model):
    is_actor = True
    joint_policy = True

    def __init__(self, hidden=64):
        super().__init__('ppo', hidden)
        self.algorithm = 'normalized_ppo'
        self.critic = ValueUnits(self.critic, value_unit(4))

    def policy(self, batch):
        return learner.masked(self.actor(batch['state']), batch['joint_mask'])

    def evaluate(self, batch, actions):
        logits = self.policy(batch)
        if actions.shape != logits.shape[:-1] or not batch['joint_mask'].gather(-1, actions[..., None]).all():
            raise ValueError('illegal normalized PPO action')
        log = torch.log_softmax(logits, -1)
        return log.gather(-1, actions[..., None]).squeeze(-1), -(log.exp() * log).sum(-1), self.critic(batch['state']).squeeze(-1)

    @torch.no_grad()
    def sample(self, batch, generator=None, deterministic=False, epsilon=0):
        if not 0 <= epsilon <= 1 or (not deterministic and generator is None):
            raise ValueError('normalized PPO RNG/epsilon')
        logits = self.policy(batch)
        action = logits.argmax(-1) if deterministic else torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)
        log, _, value = self.evaluate(batch, action)
        return action, log, value

    @torch.no_grad()
    def select(self, obs):
        return int(self.sample(learner.tensors([obs]), deterministic=True)[0][0])

    def environment_action(self, action):
        return int(action)


def make_model(a,n,c,t):
    if a not in arms(n):raise ValueError('arm/size contract')
    with torch.random.fork_rng():
        torch.manual_seed(learner.role_seed(t,f'{VERSION}:initialization:'+rng_role(a)))
        if a in CORE_ARMS:return ResidualCapacityPolicy(a,c['hidden'])
        if a=='normalized_ppo':return NormalizedPPO(c['baseline_hidden'])
        model=FullMAPPO(c['baseline_hidden']) if a=='full_mappo' else learner.Model(a,c['baseline_hidden'])
        if a!='ppo':model.critic=ValueUnits(model.critic,value_unit(n))
        return model


def critic_probe(model, obs, n):
    graph = model.algorithm in GRAPH_ARMS
    head = model.value_head if graph else model.critic.net if isinstance(model.critic, ValueUnits) else model.critic
    unit = 1. if model.algorithm == 'ppo' else value_unit(n)
    captures = []
    handle = head[-2].register_forward_hook(lambda module, inputs, output: captures.append(output.detach()))
    try:
        with torch.no_grad():
            values = model.sample(learner.tensors([obs]), deterministic=True)[2]
    finally:
        handle.remove()
    return dict(value_unit=unit, physical_prediction_mean=float(values.mean()),
        normalized_prediction_mean=float(values.mean()/unit),
        hidden_saturation=float(torch.cat([v.reshape(-1) for v in captures]).abs().ge(.99).float().mean()),
        physical_output_lower_bound=float(head[-1].bias.detach()-head[-1].weight.detach().abs().sum())*unit,
        physical_output_upper_bound=float(head[-1].bias.detach()+head[-1].weight.detach().abs().sum())*unit)


def save_model(model,path,n,c,t,steps,probe):
    if path.exists():raise FileExistsError('checkpoint overwrite')
    payload=dict(protocol=VERSION,algorithm=model.algorithm,n=n,train_seed=t,steps=steps,
        parameter_sha256=learner.parameter_digest(model),state_dict=model.state_dict())
    torch.save(payload,path);loaded=load_model(path,n,c,t,model.algorithm,steps)
    if not np.array_equal(model.select(probe),loaded.select(probe)):raise RuntimeError('checkpoint prediction roundtrip')
    return dict(path=str(path),sha256=sha(path),parameter_sha256=payload['parameter_sha256'],value_probe=critic_probe(model,probe,n))

def load_model(path,n,c,t,a,steps):
    p=torch.load(path,map_location='cpu',weights_only=True)
    if any(p.get(k)!=v for k,v in dict(protocol=VERSION,algorithm=a,n=n,train_seed=t,steps=steps).items()):raise RuntimeError('checkpoint metadata')
    model=make_model(a,n,c,t);model.load_state_dict(p['state_dict']);model.eval()
    if learner.parameter_digest(model)!=p['parameter_sha256']:raise RuntimeError('checkpoint digest')
    return model

def evaluate(a,n,t,s,model=None,coordination=None,decisions=None,latencies=None,observations=None):
    tags=dict(method=a,n=n,train_seed=t,seed=s);env=ResourceEnv(s,n,a,journal=Tags(decisions,tags) if decisions else None)
    obs=env.reset();reward=0.;conflicts=rejected=0;times=[]
    for tick in range(48):
        if observations is not None:observations.append({k:v.copy() for k,v in obs.items()})
        before=time.perf_counter();action=model.select(obs) if model else None;elapsed=(time.perf_counter()-before)*1000;times.append(elapsed)
        state_hash=hashlib.sha256(obs['state'].tobytes()).hexdigest();obs,r,done,info=env.step(action)
        reward+=r;diag=info['coordination'];conflicts+=diag['conflicts'];rejected+=diag['rejected_proposals']
        if coordination:coordination.add(dict(tags,tick=tick,observation_sha256=state_hash,
            submitted=json.dumps(int(action) if np.isscalar(action) else np.asarray(action).tolist()) if model else 'null',**diag,decision_ms=elapsed))
    m=info['episode_metrics']
    if not done or abs(-20*reward-m['extended_cost'])>1e-5 or m['invalid_assignments']:raise RuntimeError('episode reconciliation')
    if latencies is not None:latencies.extend(times)
    return dict(tags,**m,conflicts=conflicts,rejected_proposals=rejected,decision_mean_ms=float(np.mean(times)),
        normalized_cost=m['extended_cost']/n,normalized_pending=m['terminal_pending']/n,normalized_failures=m['policy_failures']/n)

def fit_job(job):
    c,n,a,t,out_name,start=job;out=Path(out_name);torch.set_num_threads(1)
    def timeout():
        if (out/'abort.json').exists():raise RuntimeError('sibling technical abort')
        if time.monotonic()-start>c['max_seconds']:raise TimeoutError('registered global deadline')
    model=make_model(a,n,c,t);initial=learner.parameter_digest(model);before=time.monotonic()
    directory=out/f'N{n}'/a/str(t);directory.mkdir(parents=True);(directory/'checkpoints').mkdir()
    journals={key:Journal(directory/f'{key}.csv') for key in ('training_progress','training_episodes','development_episodes')}
    probe=ResourceEnv(c['development_seeds'][0],n,a).reset();choices=[];devrows=[]
    def assess(current,steps,audit=None):
        timeout();current.eval();meta=save_model(current,directory/'checkpoints'/f'step_{steps:09d}.pt',n,c,t,steps,probe);meta['path']=str(Path(meta['path']).relative_to(out));rows=[]
        for s in tqdm(c['development_seeds'],desc=f'Dev N{n}/{a}/{t}/{steps}',leave=False):
            timeout();observations=[] if a=='residual_capacity_ppo' and s==c['development_seeds'][0] else None
            r=evaluate(a,n,t,s,current,observations=observations)
            if observations is not None:
                observation_path=directory/'checkpoints'/f'latency_observations_{steps:09d}.npz'
                np.savez_compressed(observation_path,**{k:np.stack([o[k] for o in observations]) for k in observations[0]})
                meta['latency_observations']=dict(path=str(observation_path.relative_to(out)),sha256=sha(observation_path),states=len(observations),seed=s)
            r['checkpoint_steps']=steps;journals['development_episodes'].add(r);rows.append(r);devrows.append(r)
        choices.append(dict(meta,steps=steps,development_cost=float(np.mean([r['extended_cost'] for r in rows]))));current.train()
    try:
        assess(model,0)
        # Baselines have different policies; graph arms intentionally share RNG tapes.
        role=rng_role(a)
        unit = 1. if a == 'ppo' else value_unit(n)
        cfg=dict(c,rng_namespace=f'{VERSION}:N{n}:{role}:',value_loss_normalizer=unit,
            behavior_value_atol=c['behavior_value_atol']*unit,
            group_gradient_clipping=a in GRAPH_ARMS,log_gradient_groups=a in GRAPH_ARMS)
        audit=learner.fit(model,cfg,t,lambda seed,w:ResourceEnv(seed,n,a,w,True),
            {key:Tags(journals[key],dict(n=n)) for key in ('training_progress','training_episodes')},assess,timeout)
        return dict(n=n,algorithm=a,train_seed=t,initial=initial,parameters=sum(p.numel() for p in model.parameters()),
            audit=dict(audit,n=n,reference=False,from_scratch=True,pretraining_steps=0,seconds_including_development=time.monotonic()-before),devrows=devrows,
            choice=min(choices,key=lambda r:(r['development_cost'],r['steps'])))
    finally:
        for journal in journals.values():journal.close()

def mechanism_states(profile):
    """Fixed, non-training states; confirmation is disjoint by physical time."""
    from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_state
    from ht_pdm_fjsp.technician_reservation_env import graph_observation
    if profile not in PROFILES:raise ValueError('mechanism profile')
    confirmation=profile=='confirmation';c=scaled_config(4)
    if c.service_time[:2]!=((1,2),(1,4)):raise RuntimeError('probe service premises changed')
    states=[]
    for age,failed,other,tick in itertools.product(
            (3,5) if confirmation else (3,4),(False,True),
            (1,2) if confirmation else (0,1),(8,32) if confirmation else (0,24)):
        state=DispatchState((age,age,other,other),(failed,failed,False,False),
                            (0,0,0,0),(0,0),(-1,-1))
        validate_state(c,state)
        obs=graph_observation(c,state,tick)
        if obs['pair_mask'].sum()!=4:raise RuntimeError('probe support premise')
        states.append(dict(state=asdict(state),tick=tick,observation=obs))
    return states


def first_probability(model,obs):
    from ht_pdm_fjsp.technician_completion_policy import initial_calendar
    batch=learner.tensors([obs]);base=model.encoder(model.raw_pairs(batch))
    calendar,planned=initial_calendar(batch)
    log,_=model.conditionals(batch,base,calendar,planned,torch.zeros(1,dtype=torch.long),
                              0,torch.zeros((1,4),dtype=torch.bool))
    return float(log[0,2].exp())


@torch.no_grad()
def measure_mechanism(out,c,choices,choice_hash,timeout=None):
    if sha(out/'checkpoint_choices.json')!=choice_hash:raise RuntimeError('probe choices unsealed')
    states=mechanism_states(c['profile']);records=[];probabilities={};knockout={}
    description=[dict(state=r['state'],tick=r['tick']) for r in states]
    for a,t in tqdm(list(itertools.product(CORE_ARMS,c['train_seeds'])),desc='Fixed capacity mechanism probes'):
        if timeout:timeout()
        choice=choices[f'4:{a}:{t}'];path=out/choice['path']
        if sha(path)!=choice['sha256']:raise RuntimeError('probe checkpoint binding')
        model=load_model(path,4,c,t,a,choice['steps']);before=learner.parameter_digest(model)
        p=[first_probability(model,r['observation']) for r in states]
        probabilities[f'{a}:{t}']=float(np.mean(p));zero=None
        if a=='residual_capacity_ppo':
            try:
                model.algorithm='no_capacity_ppo'
                zero=[first_probability(model,r['observation']) for r in states]
            finally:model.algorithm=a
            knockout[str(t)]=float(np.mean(zero))
        if learner.parameter_digest(model)!=before:raise RuntimeError('probe changed weights')
        records.append(dict(method=a,train_seed=t,checkpoint_steps=choice['steps'],
            checkpoint_sha256=choice['sha256'],probability_machine2=p,zero_message_same_weights=zero))
    result=dict(protocol=VERSION,profile=c['profile'],choices_sha256=choice_hash,
        fixed_states=description,states_sha256=hashlib.sha256(json.dumps(description,sort_keys=True).encode()).hexdigest(),
        fixed_states_per_replica=16,probabilities=probabilities,knockout=knockout,records=records,
        sample_unit='paired training replica; fixed states are not resampled or used in training/selection')
    write_json(out/'mechanism.json',result)
    return result


def validate_mechanism_grid(mechanism,c):
    expected={f'{a}:{t}' for a in CORE_ARMS for t in c['train_seeds']}
    p=mechanism.get('probabilities',{})
    if set(p)!=expected or any(not math.isfinite(v) or not 0<=v<=1 for v in p.values()):
        raise ValueError('incomplete/nonfinite mechanism probability grid')


@torch.no_grad()
def audit_mechanism(out,c,choices):
    data=json.loads((out/'mechanism.json').read_text());validate_mechanism_grid(data,c)
    if (data['protocol']!=VERSION or data['profile']!=c['profile'] or
        data['choices_sha256']!=sha(out/'checkpoint_choices.json') or
        data['fixed_states_per_replica']!=16):raise RuntimeError('mechanism metadata')
    states=mechanism_states(c['profile'])
    description=[dict(state=r['state'],tick=r['tick']) for r in states]
    if (hashlib.sha256(json.dumps(description,sort_keys=True).encode()).hexdigest()!=data['states_sha256'] or
        json.dumps(description,sort_keys=True)!=json.dumps(data['fixed_states'],sort_keys=True)):
        raise RuntimeError('mechanism fixed states binding')
    seen=set();knockout={}
    for record in data['records']:
        a,t=record['method'],record['train_seed'];key=f'{a}:{t}'
        if key in seen or a not in CORE_ARMS or t not in c['train_seeds']:raise RuntimeError('mechanism record grid')
        seen.add(key);choice=choices[f'4:{a}:{t}']
        if record['checkpoint_sha256']!=choice['sha256'] or record['checkpoint_steps']!=choice['steps']:
            raise RuntimeError('mechanism selection binding')
        model=load_model(out/choice['path'],4,c,t,a,choice['steps']);before=learner.parameter_digest(model)
        p=[first_probability(model,r['observation']) for r in states]
        if p!=record['probability_machine2'] or float(np.mean(p))!=data['probabilities'][key]:
            raise RuntimeError('mechanism probability replay')
        if a=='residual_capacity_ppo':
            try:
                model.algorithm='no_capacity_ppo'
                zero=[first_probability(model,r['observation']) for r in states]
            finally:model.algorithm=a
            if zero!=record['zero_message_same_weights']:raise RuntimeError('knockout replay')
            knockout[str(t)]=float(np.mean(zero))
        elif record['zero_message_same_weights'] is not None:raise RuntimeError('unexpected knockout')
        if learner.parameter_digest(model)!=before:raise RuntimeError('mechanism audit changed weights')
    if seen!=set(data['probabilities']) or knockout!=data['knockout']:raise RuntimeError('mechanism aggregate grid')
    return dict(status='PASS',replicas=len(seen),first_conditional_probabilities_replayed=16*len(seen),
                same_weight_knockout_probabilities_replayed=16*len(c['train_seeds']))


def attach_probe_axis(arrays,c,mechanism):
    """Bookkeeping only: constant within episode axis; probe n is train replicas."""
    validate_mechanism_grid(mechanism,c)
    return {n:{a:np.concatenate([v,np.broadcast_to(np.array([
        mechanism['probabilities'].get(f'{a}:{t}',0.) for t in c['train_seeds']])[:,None,None],
        (*v.shape[:2],1))],axis=-1) for a,v in panel.items()} for n,panel in arrays.items()}


def estimates(means,scales):
    if list(scales)!=[4]:raise ValueError('N4-only endpoints')
    candidate=means[4]['residual_capacity_ppo']
    result={f'N4_gain_vs_{other}':1-candidate[0]/means[4][other][0]
            for other in arms(4) if other!='residual_capacity_ppo'}
    for reference in ('ppo','normalized_ppo'):
        for label,index in (('availability',1),('pending_per_machine',2),('failures_per_machine',3)):
            result[f'N4_{label}_difference_vs_{reference}']=candidate[index]-means[4][reference][index]
    result['N4_critical_first_probability_difference']=candidate[4]-means[4]['no_capacity_ppo'][4]
    return result


def guards(intervals,scales,latencies):
    if list(scales)!=[4]:raise ValueError('N4-only gates')
    conservative={k:v['upper' if 'pending_per_machine_difference' in k or 'failures_per_machine_difference' in k else 'lower']
                  for k,v in intervals.items()}
    out=point_checks(conservative)
    out['N4_latency']=latencies['4:residual_capacity_ppo']['p95_ms']<=50
    out['optional_ppo_superiority']=all(intervals[f'N4_gain_vs_{a}']['lower']>=.01 for a in ('ppo','normalized_ppo'))

    return out

def summarize(rows,c,latencies,mechanism):
    metrics=('normalized_cost','availability','normalized_pending','normalized_failures');trains=c['train_seeds'];tests=c['test_seeds']
    lookup={(r['n'],r['method'],r['train_seed'],r['seed']):r for r in rows}
    expected={(n,a,t,s) for n in c['scales'] for a in arms(n) for t in trains for s in tests}|{(n,'deadline',-1,s) for n in c['scales'] for s in tests}
    if len(rows)!=len(expected) or set(lookup)!=expected:raise RuntimeError('exact test grid')
    arrays={n:{a:np.array([[[lookup[n,a,t,s][m] for m in metrics] for s in tests] for t in trains]) for a in arms(n)} for n in c['scales']}
    for n,panel in arrays.items():panel['deadline']=np.tile(np.array([[lookup[n,'deadline',-1,s][m] for m in metrics] for s in tests])[None],(len(trains),1,1))
    arrays=attach_probe_axis(arrays,c,mechanism)
    means={n:{a:v.mean((0,1)) for a,v in panel.items()} for n,panel in arrays.items()};point=estimates(means,c['scales']);draws={k:[] for k in point}
    rng=np.random.default_rng(learner.role_seed(c['analysis_seed'],VERSION+':bootstrap'));nt,ns=len(trains),len(tests)
    for start in tqdm(range(0,c['bootstrap_samples'],128),desc='Crossed bootstrap'):
        b=min(128,c['bootstrap_samples']-start);tc=np.eye(nt)[rng.integers(nt,size=(b,nt))].mean(1);sampled={}
        for n,panel in arrays.items():
            sc=np.eye(ns)[rng.integers(ns,size=(b,ns))].mean(1);sampled[n]={a:np.einsum('bt,bs,tsk->bk',tc,sc,v,optimize=True).T for a,v in panel.items()}
        for key,v in estimates(sampled,c['scales']).items():draws[key].extend(v.tolist())
    tail=(1-c['confidence'])/2;intervals={key:dict(estimate=float(v),lower=float(np.quantile(draws[key],tail)),upper=float(np.quantile(draws[key],1-tail))) for key,v in point.items()}
    timing={k:dict(mean_ms=float(np.mean(v)),p95_ms=float(np.quantile(v,.95)),samples=len(v)) for k,v in latencies.items()}
    checks=guards(intervals,c['scales'],timing);loo=[]
    for omit in range(nt) if nt>1 else []:
        keep=[t for t in range(nt) if t!=omit];mean={n:{a:v[keep].mean((0,1)) for a,v in panel.items()} for n,panel in arrays.items()}
        loo.append(dict(omitted_train_seed=trains[omit],estimates={k:float(v) for k,v in estimates(mean,c['scales']).items()}))
    guard_pass=all(v for k,v in checks.items() if not k.startswith('optional'))
    return dict(classification='ENGINEERING_ONLY' if c['profile']=='engineering' else 'CONFIRMATORY',teacher_assistance=False,
        goal_pass=guard_pass and c['profile']=='confirmation',
        engineering_guard_pass=guard_pass if c['profile']=='engineering' else None,guards=checks,intervals=intervals,
        confidence=c['confidence'],family_size=len(point),round_alpha=c['round_alpha'],confirmation_round=c['confirmation_round'],
        means={str(n):{a:dict(zip(metrics,v[:4].tolist())) for a,v in panel.items()} for n,panel in means.items()},latency=timing,
        per_replica_means={str(n):{a:[dict(zip(metrics,row[:4].tolist())) for row in v.mean(1)] for a,v in panel.items()} for n,panel in arrays.items()},leave_one_replica_out=loo)

def point_checks(point):
    checks={}
    for key,v in point.items():
        if 'availability_difference' in key:checks[key]=bool(v>=-.01)
        elif 'pending_per_machine_difference' in key:checks[key]=bool(v<=.025)
        elif 'failures_per_machine_difference' in key:checks[key]=bool(v<=.25)
        elif key == 'N4_critical_first_probability_difference':checks[key]=bool(v>=.05)
        elif key in tuple('N4_gain_vs_'+a for a in CORE_ARMS[1:]):checks[key]=bool(v>=.005)
        elif key in ('N4_gain_vs_mappo','N4_gain_vs_full_mappo'):checks[key]=bool(v>0)
        else:checks[key]=bool(v>=-.01)
    return checks

def development_summary(rows,c,references,latencies,mechanism):
    validate_mechanism_grid(mechanism,c)
    selected={};means={};health={};checks={}
    for n in c['scales']:
        means[n]={}
        for a in arms(n):
            selected_rows=[]
            for t in c['train_seeds']:
                data=[r for r in rows if r['n']==n and r['method']==a and r['train_seed']==t]
                scores={step:np.mean([r['extended_cost'] for r in data if r['checkpoint_steps']==step]) for step in [0,*c['checkpoints']]}
                step=min(scores,key=lambda x:(scores[x],x));selected[f'{n}:{a}:{t}']=step;selected_rows.extend(r for r in data if r['checkpoint_steps']==step)
            means[n][a]=np.array([np.mean([r[m] for r in selected_rows]) for m in ('normalized_cost','availability','normalized_pending','normalized_failures')])
        ref=[r for r in references if r['n']==n];means[n]['deadline']=np.array([np.mean([r[m] for r in ref]) for m in ('normalized_cost','availability','normalized_pending','normalized_failures')])
    means={n:{a:np.r_[v, np.mean([mechanism['probabilities'].get(f'{a}:{t}',0.) for t in c['train_seeds']])] for a,v in panel.items()} for n,panel in means.items()}
    point=estimates(means,c['scales'])
    checks=point_checks(point)
    timing={k:dict(mean_ms=float(np.mean(v)),p95_ms=float(np.quantile(v,.95)),samples=len(v)) for k,v in latencies.items()}
    latency_pass=timing['4:residual_capacity_ppo']['p95_ms']<=50
    return dict(latency=timing,latency_pass=latency_pass,classification='DEVELOPMENT_ONLY',teacher_assistance=False,test_opened=False,development_screen_pass=all(checks.values()) and latency_pass,
        screen=checks,estimates={k:float(v) for k,v in point.items()},selected_steps=selected,
        means={str(n):{a:dict(zip(('normalized_cost','availability','normalized_pending','normalized_failures'),v.tolist())) for a,v in panel.items()} for n,panel in means.items()})

def validate_bound_training(p,m,c):
    expected={(n,a,t) for n in c['scales'] for a in arms(n) for t in c['train_seeds']}
    for name in ('training_audit.json','checkpoint_choices.json'):
        if not (p/name).is_file() or name not in m['output_bindings'] or sha(p/name)!=m['output_bindings'][name]['sha256']:
            raise ValueError('changed or missing prerequisite training artifact')
    rows=json.loads((p/'training_audit.json').read_text());choices=json.loads((p/'checkpoint_choices.json').read_text())
    opts=c['env_steps']//c['rollout']*c['epochs']*(c['rollout']//c['minibatch'])
    if (len(rows)!=len(expected) or {(r['n'],r['algorithm'],r['train_seed']) for r in rows}!=expected
        or set(choices)!={f'{n}:{a}:{t}' for n,a,t in expected}
        or any(r.get('steps') not in [0,*c['checkpoints']] for r in choices.values())
        or any(r['learned_steps']!=c['env_steps'] or r['physical_ticks']!=c['env_steps']//48*60
            or r['training_episodes']!=c['env_steps']//48 or r['optimizer_steps']!=opts
            or not r['from_scratch'] or r['pretraining_steps']!=0 or not r['changed']
            or r['behavior_normalized_max_error']>1 for r in rows)):
        raise ValueError('prerequisite scratch training/budget grid fails')

def validate_prerequisite(directory,c,source_hashes):
    if directory is None:raise ValueError('confirmation needs audited passing development artifact')
    p=Path(directory);m=json.loads((p/'manifest.json').read_text());s=json.loads((p/'summary.json').read_text());d=json.loads((p/'resolved_config.json').read_text())
    if m['protocol']!=VERSION or m['status']!='COMPLETED' or m['audit_status']!='PASS' or m['test_opened'] or d!=settings('development') or not s['development_screen_pass']:raise ValueError('development prerequisite fails')
    validate_summary_grid(s,d)
    validate_bound_training(p,m,d)
    for name in ('summary.json','audit.json','resolved_config.json','latency.json','mechanism.json'):
        if sha(p/name)!=m['output_bindings'][name]['sha256']:raise ValueError('changed development artifact')
    mechanism=json.loads((p/'mechanism.json').read_text())
    if s.get('mechanism')!=mechanism or mechanism['choices_sha256']!=sha(p/'checkpoint_choices.json'):raise ValueError('mechanism provenance changed')
    validate_mechanism_grid(mechanism,d)
    choices=json.loads((p/'checkpoint_choices.json').read_text())
    if any(choices[k]['steps']!=v for k,v in s['selected_steps'].items()):raise ValueError('selected steps differ from bound choices')
    for name,meta in m['output_bindings'].items():
        if sha(p/name)!=meta['sha256']:raise ValueError('changed bound development output')
    timing=json.loads((p/'latency.json').read_text())
    samples=timing['samples_ms']['4:residual_capacity_ppo']
    if len(samples)!=48*3*len(d['train_seeds']) or not all(math.isfinite(v) and v>=0 for v in samples):raise ValueError('isolated latency sample grid')
    if timing['choices_sha256']!=sha(p/'checkpoint_choices.json') or s['latency']['4:residual_capacity_ppo']['p95_ms']!=float(np.quantile(samples,.95)):raise ValueError('isolated latency binding')
    if not any(s['selected_steps'][f'4:residual_capacity_ppo:{t}']>0 for t in d['train_seeds']):raise ValueError('candidate has no selected trained replica')
    for name,digest in source_hashes.items():
        if m['source_seal'].get(name)!=digest:raise ValueError('source/protocol/registry changed after development')
    for key in ('env_steps','hidden','baseline_hidden','learning_rate','rollout','epochs','minibatch','checkpoints','value_unit_formula','behavior_value_atol_units'):
        if c[key]!=d[key]:raise ValueError('development learning design differs')
    return dict(source=str(p.resolve()),manifest_sha256=sha(p/'manifest.json'),summary_sha256=sha(p/'summary.json'),status='PASS')

def validate_summary_grid(summary,c):
    null={n:{a:np.ones(5) for a in (*arms(n),'deadline')} for n in c['scales']}
    keys=set(estimates(null,c['scales']))
    selected={f'{n}:{a}:{t}' for n in c['scales'] for a in arms(n) for t in c['train_seeds']}
    if (set(summary.get('screen',{}))!=keys or set(summary.get('estimates',{}))!=keys
        or set(summary.get('selected_steps',{}))!=selected
        or any(step not in [0,*c['checkpoints']] for step in summary['selected_steps'].values())
        or not all(isinstance(v,bool) for v in summary['screen'].values())
        or not all(math.isfinite(v) for v in summary['estimates'].values())
        or summary.get('classification')!='DEVELOPMENT_ONLY'
        or summary.get('test_opened') is not False or summary.get('teacher_assistance') is not False
        or not isinstance(summary.get('latency_pass'),bool)
        or summary.get('latency_pass')!=(summary['latency']['4:residual_capacity_ppo']['p95_ms']<=50)
        or summary.get('development_screen_pass')!=(all(summary['screen'].values()) and summary['latency_pass'])):
        raise ValueError('incomplete development summary grid/provenance')
    if summary['screen']!=point_checks(summary['estimates']):
        raise ValueError('summary point gates disagree with registered thresholds')



def schedule_jobs(c,out,start):
    return [(c,n,a,t,str(out),start) for n in c['scales'] for a in arms(n) for t in c['train_seeds']]



def isolated_latency(out,c,choices,choice_hash,timeout=None):
    """Time frozen selected candidate states after all training workers have exited."""
    if sha(out/'checkpoint_choices.json')!=choice_hash:raise RuntimeError('choices unsealed before latency')
    samples=[];contexts=[]
    for t in c['train_seeds']:
        if timeout:timeout()
        choice=choices[f'4:residual_capacity_ppo:{t}'];meta=choice['latency_observations'];path=out/meta['path']
        if sha(path)!=meta['sha256'] or meta['states']!=48 or meta['seed']!=c['development_seeds'][0]:raise RuntimeError('latency observations binding')
        model=load_model(out/choice['path'],4,c,t,'residual_capacity_ppo',choice['steps'])
        with np.load(path,allow_pickle=False) as archive:
            if any(len(archive[k])!=48 for k in archive.files):raise RuntimeError('latency states count')
            for i in range(48):
                if timeout:timeout()
                obs={k:archive[k][i].copy() for k in archive.files}
                model.select(obs)
                for _ in range(3):
                    before=time.perf_counter();model.select(obs);samples.append((time.perf_counter()-before)*1000)
        contexts.append(dict(train_seed=t,checkpoint_steps=choice['steps'],checkpoint_sha256=choice['sha256'],**meta))
    result={'4:residual_capacity_ppo':samples}
    write_json(out/'latency.json',dict(protocol=VERSION,choices_sha256=choice_hash,contexts=contexts,
        warmup_per_state=1,timed_repeats_per_state=3,workers_active=0,torch_threads=torch.get_num_threads(),
        samples_ms=result,description='selected DEV observations, actor-only select, no physics transition'))
    return result

def run(profile,output_dir,development_dir=None):
    c=settings(profile);out=Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('prior run overwrite')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);start=time.monotonic()
    git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()))
    if profile!='engineering' and git['dirty']:raise RuntimeError('commit verified code before research run')
    m=dict(protocol=VERSION,status='RUNNING',profile=profile,git=git,test_opened=False,teacher_assistance=False,
        started_utc=datetime.now(timezone.utc).isoformat(),runtime=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device='cpu'))
    write_json(out/'manifest.json',m);journals={}
    def timeout():
        if time.monotonic()-start>c['max_seconds']:raise TimeoutError('registered hard stop')
    try:
        seed_report=seed_audit();write_json(out/'seed_audit.json',seed_report);write_json(out/'resolved_config.json',c)
        write_json(out/'physics.json',{str(n):asdict(scaled_config(n)) for n in c['scales']})
        snap=out/'source_snapshot';snap.mkdir()
        for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
        history=out/'registry_snapshot';history.mkdir()
        for name in seed_report['historical_registry_hashes']:shutil.copy2(ROOT/'configs'/name,history/name)
        shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'seed_registry.json')
        seals={str(p.relative_to(out)):sha(p) for p in [*snap.iterdir(),*history.iterdir(),out/'protocol.md',out/'seed_registry.json',out/'resolved_config.json',out/'physics.json']}
        m['source_seal']=seals;write_json(out/'manifest.json',m)
        learning={key:digest for key,digest in seals.items() if key.startswith('source_snapshot/') or key in ('protocol.md','seed_registry.json')}
        if profile=='confirmation':
            prerequisite=validate_prerequisite(development_dir,c,learning);write_json(out/'development_prerequisite.json',prerequisite)
        journals={key:Journal(out/f'{key}.csv') for key in ('training_progress','training_episodes','development_episodes','episodes.partial','coordination','decisions')}
        jobs=schedule_jobs(c,out,start);results={};choices={}
        m['stage']='training_pool';write_json(out/'manifest.json',m)
        with ProcessPoolExecutor(max_workers=c['training_processes'],mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(fit_job,job) for job in jobs]
            for r in pool_results(futures,out,m,start,'Random-initialized models'):
                key=f'{r["n"]}:{r["algorithm"]}:{r["train_seed"]}';results[key]=r;choices[key]=r['choice'];m['completed_models']=len(results)
                write_json(out/'checkpoint_choices.partial.json',choices);write_json(out/'manifest.json',m)
        audits=[];devrows=[];initials={};parameters={}
        for _,n,a,t,_,_ in jobs:
            key=f'{n}:{a}:{t}';r=results[key];audits.append(r['audit']);devrows.extend(r['devrows']);initials[key]=r['initial'];parameters[key]=r['parameters']
            for name in ('training_progress','training_episodes','development_episodes'):
                with (out/f'N{n}'/a/str(t)/f'{name}.csv').open() as stream:
                    for row in csv.DictReader(stream):journals[name].add(row)
        for n in c['scales']:
            for t in c['train_seeds']:
                if len({initials[f'{n}:{a}:{t}'] for a in CORE_ARMS})!=1 or len({parameters[f'{n}:{a}:{t}'] for a in CORE_ARMS})!=1:raise RuntimeError('matched graph init/capacity')
        write_json(out/'training_audit.json',audits);write_json(out/'initial_parameter_hashes.json',initials);write_json(out/'parameter_counts.json',parameters)
        write_json(out/'checkpoint_choices.json',choices);choice_hash=sha(out/'checkpoint_choices.json')
        m.update(stage='isolated_latency',choices_sha256=choice_hash,choices_sealed_utc=datetime.now(timezone.utc).isoformat());write_json(out/'manifest.json',m)
        latencies=isolated_latency(out,c,choices,choice_hash,timeout)
        mechanism=measure_mechanism(out,c,choices,choice_hash,timeout)
        if profile=='development':
            refs=[]
            for n in c['scales']:
                for s in tqdm(c['development_seeds'],desc=f'Dev deadline N{n}'):timeout();refs.append(evaluate('deadline',n,-1,s))
            write_csv(out/'development_reference.csv',refs);summary=development_summary(devrows,c,refs,latencies,mechanism);rows=[]
        else:
            m.update(stage='test' if profile=='confirmation' else 'engineering_test',test_opened=profile=='confirmation',engineering_test_opened=profile=='engineering');write_json(out/'manifest.json',m)
            rows=[]
            for n in c['scales']:
                for a,t in [(a,t) for a in arms(n) for t in c['train_seeds']]+[('deadline',-1)]:
                    choice=choices.get(f'{n}:{a}:{t}');model=load_model(out/choice['path'],n,c,t,a,choice['steps']) if choice else None
                    for s in tqdm(c['test_seeds'],desc=f'Test N{n}/{a}/{t}'):
                        timeout();row=evaluate(a,n,t,s,model,journals['coordination'],journals['decisions']);rows.append(row);journals['episodes.partial'].add(row)
            write_csv(out/'episodes.csv',rows);summary=summarize(rows,c,latencies,mechanism)
        summary['mechanism']=mechanism
        write_json(out/'summary.json',summary)
        for journal in journals.values():journal.close()
        journals.clear();m['stage']='independent_audit';write_json(out/'manifest.json',m)
        audit=audit_run(out,c,devrows,rows,audits,choices,timeout);write_json(out/'audit.json',audit)
        if sha(out/'checkpoint_choices.json')!=choice_hash or any(sha(out/name)!=digest for name,digest in seals.items()):raise RuntimeError('sealed input changed')
        if any(sha(out/r['path'])!=r['sha256'] for r in choices.values()):raise RuntimeError('selected checkpoint changed')
        if sha(PLAN)!=seals['protocol.md'] or sha(REGISTRY)!=seals['seed_registry.json'] or any(sha(Path(__file__).parent/p.name)!=seals[f'source_snapshot/{p.name}'] for p in snap.iterdir()):raise RuntimeError('working source changed')
        if any(sha(ROOT/'configs'/name)!=digest for name,digest in seed_report['historical_registry_hashes'].items()):raise RuntimeError('historical registry changed')
        m.update(status='COMPLETED',audit_status='PASS',stage='complete',elapsed_seconds=time.monotonic()-start,training_models=len(audits),test_episodes=len(rows),development_episodes=len(devrows),goal_pass=summary.get('goal_pass',False),completed_utc=datetime.now(timezone.utc).isoformat())
        m['output_bindings']={str(p.relative_to(out)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        timeout();write_json(out/'manifest.json',m);print(json.dumps(dict(output_dir=str(out),status='COMPLETED',audit='PASS',goal_pass=m['goal_pass'],development_screen_pass=summary.get('development_screen_pass'))));return m
    except Exception as exc:
        m.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-start);write_json(out/'manifest.json',m);raise
    finally:
        for journal in journals.values():journal.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=('development','confirmation'),required=True);p.add_argument('--output-dir',required=True);p.add_argument('--development-dir');args=p.parse_args()
    run(args.profile,args.output_dir,args.development_dir)

# Independent scalar replay audit is defined below, before the CLI entry point.

def audit_run(out,cfg,devrows,rows,audits,choices,timeout=None):
    expected={(n,a,t) for n in cfg['scales'] for a in arms(n) for t in cfg['train_seeds']}
    if len(audits)!=len(expected) or {(r['n'],r['algorithm'],r['train_seed']) for r in audits}!=expected:raise RuntimeError('training grid')
    for r in audits:
        if r['learned_steps']!=cfg['env_steps'] or r['physical_ticks']!=cfg['env_steps']//48*60 or r['training_episodes']!=cfg['env_steps']//48 or not r['changed'] or r['behavior_normalized_max_error']>1 or not r['from_scratch'] or r['pretraining_steps']!=0:raise RuntimeError('scratch training/budget/replay contract')
        expected_opts=cfg['env_steps']//cfg['rollout']*cfg['epochs']*(cfg['rollout']//cfg['minibatch'])
        if r['optimizer_steps']!=expected_opts:raise RuntimeError('optimizer budget')
        n=r['n'];a=r['algorithm'];t=r['train_seed']
        zero=out/f'N{n}'/a/str(t)/'checkpoints/step_000000000.pt'
        initial=load_model(zero,n,cfg,t,a,0)
        if learner.parameter_digest(initial)!=learner.parameter_digest(make_model(a,n,cfg,t)):raise RuntimeError('random initialization mismatch')
    expected_dev={(n,a,t,step,s) for n,a,t in expected for step in [0,*cfg['checkpoints']] for s in cfg['development_seeds']}
    if len(devrows)!=len(expected_dev) or {(r['n'],r['method'],r['train_seed'],r['checkpoint_steps'],r['seed']) for r in devrows}!=expected_dev:raise RuntimeError('development grid')
    for n,a,t in expected:
        key=f'{n}:{a}:{t}';data=[r for r in devrows if (r['n'],r['method'],r['train_seed'])==(n,a,t)]
        scores={step:np.mean([r['extended_cost'] for r in data if r['checkpoint_steps']==step]) for step in [0,*cfg['checkpoints']]}
        best=min(scores,key=lambda step:(scores[step],step))
        if choices[key]['steps']!=best or choices[key]['development_cost']!=scores[best]:raise RuntimeError('checkpoint selection replay')
        if sha(out/choices[key]['path'])!=choices[key]['sha256']:raise RuntimeError('checkpoint binding')
    if any(r['invalid_assignments'] for r in (*devrows,*rows)):raise RuntimeError('invalid DEV/test assignment')
    paired={};seen=set()
    with (out/'training_episodes.csv').open() as stream:
        for r in csv.DictReader(stream):
            n=int(r['n']);a=r['algorithm'];t=int(r['train_seed']);w=int(r['worker']);e=int(r['episode_index']);key=(n,t,w,e);identity=key+(a,)
            if identity in seen:raise RuntimeError('duplicate training episode')
            seen.add(identity);tape=(r['physical_seed'],r['population_sha256'])
            if paired.setdefault(key,tape)!=tape:raise RuntimeError('unpaired exogenous training tape')
            if float(r['invalid_assignments']):raise RuntimeError('invalid training assignment')
            if abs(float(r['episode_reward'])*20+float(r['extended_cost']))>1e-5:raise RuntimeError('training reward')
            physical=physical_seed(t,f'{VERSION}:N{n}:training_physical:worker:{w}:episode:{e}')
            if physical!=int(r['physical_seed']):raise RuntimeError('training seed namespace')
    wanted={(n,t,w,e,a) for n,a,t in expected for w in range(cfg['vector_envs']) for e in range(cfg['env_steps']//(48*cfg['vector_envs']))}
    if seen!=wanted:raise RuntimeError('training episode grid')
    if cfg['teacher_assistance'] or cfg['pretraining_steps'] or any((out/name).exists() for name in ('teachers','warm','warm_choices.json')):raise RuntimeError('teacher-free contract')
    if cfg['test_seeds']:
        pops={}
        for r in rows:
            key=r['n'],r['seed']
            if pops.setdefault(key,r['population_sha256'])!=r['population_sha256']:raise RuntimeError('test physical pairing')
        # Independently replay all physical transitions, rather than relying on wrappers.
        from ht_pdm_fjsp.maintenance_dispatch import DispatchState,role_seed
        from ht_pdm_fjsp.maintenance_waiting import transition,WaitingEnv
        replay_count=0;replay_episodes={};lookup={(r['n'],r['method'],r['train_seed'],r['seed']):r for r in rows}
        recorded_coord={}
        with (out/'coordination.csv').open() as f:
            for r in csv.DictReader(f):
                key=(int(r['n']),r['method'],int(r['train_seed']),int(r['seed']),int(r['tick']))
                if key in recorded_coord:raise RuntimeError('duplicate coordination record')
                recorded_coord[key]=r
        with (out/'decisions.csv').open() as f:
            for r in csv.DictReader(f):
                if timeout and replay_count%1000==0:timeout()
                n=int(r['n']);c=scaled_config(n)
                state=DispatchState(**{k:tuple(v) for k,v in json.loads(r['state']).items()})
                pairs=tuple(map(tuple,json.loads(r['action'])))
                key=n,r['method'],int(r['train_seed']),int(r['seed']);tick=int(r['tick'])
                if key not in lookup:raise RuntimeError('unexpected replay episode')
                if key not in replay_episodes:
                    physical=role_seed(key[3],f'{VERSION}:N{n}:physical');initial=WaitingEnv(c);initial.reset(physical)
                    replay_episodes[key]=dict(state=initial.state,events=initial.events,tick=0,
                        metrics={k:0. for k in initial.metrics},broker=dict.fromkeys((*COUNTERS,'conflicts','rejected_proposals'),0),probe=ResourceEnv(key[3],n,key[1]))
                ep=replay_episodes[key]
                if tick!=ep['tick'] or state!=ep['state'] or tuple(json.loads(r['events']))!=ep['events'][tick]:raise RuntimeError('physical tape/state sequence mismatch')
                if tick<48:
                    cr=recorded_coord.get(key+(tick,))
                    if cr is None:raise RuntimeError('missing coordination record')
                    probe=ep['probe'];probe.env.state=state;probe.env.time=tick
                    obs=probe.observe()
                    if hashlib.sha256(obs['state'].tobytes()).hexdigest()!=cr['observation_sha256']:raise RuntimeError('observation hash mismatch')
                    submitted=json.loads(cr['submitted'])
                    if isinstance(submitted,list):submitted=np.asarray(submitted,dtype=int)
                    decoded,diag=probe.resolve(submitted)
                    if decoded!=pairs or any(int(cr[k])!=v for k,v in diag.items()):raise RuntimeError('coordination/action replay mismatch')
                    for field in ep['broker']:ep['broker'][field]+=diag[field]
                else:
                    from ht_pdm_fjsp.technician_reservation_env import deadline
                    if deadline(c,state)!=pairs:raise RuntimeError('common tail policy mismatch')
                following,cost,components=transition(c,state,pairs,json.loads(r['events']))
                for k,v in components.items():ep['metrics'][k]+=v
                ep['state']=following;ep['tick']+=1
                if tick==47:ep['policy_cost']=ep['metrics']['objective']
                actual={k:list(v) for k,v in asdict(following).items()}
                if actual!=json.loads(r['following']) or abs(cost-float(r['cost']))>1e-8:raise RuntimeError('scalar transition replay')
                replay_count+=1
        if replay_count!=60*len(rows):raise RuntimeError('physical test replay count')
        if len(recorded_coord)!=48*len(rows) or set(replay_episodes)!=set(lookup):raise RuntimeError('coordination/replay population')
        for key,ep in replay_episodes.items():
            row=lookup[key]
            if ep['tick']!=60 or abs(ep['policy_cost']-row['policy_cost'])>1e-8 or any(abs(v-row[k])>1e-8 for k,v in ep['metrics'].items()):raise RuntimeError('aggregate physical cost/metric replay')
            if any(row['broker_'+field]!=value for field,value in ep['broker'].items()):raise RuntimeError('aggregate broker counters replay')
    else: replay_count=0
    mechanism_audit=audit_mechanism(out,cfg,choices)
    return dict(mechanism=mechanism_audit,status='PASS',training_models=len(audits),training_episodes=len(seen),paired_training_tapes=len(paired),development_episodes=len(devrows),test_physical_ticks_replayed=replay_count,choices=len(choices),invalid_assignments=0,teacher_assistance=False,pretraining_steps=0,from_scratch_models=len(audits))

if __name__=='__main__':main()
