"""Scratch RL bounded continuation-disagreement credit versus own-Q ranking."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
from sb3_contrib import MaskablePPO

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import (
    CASES,HORIZON,FrameworkEnv,catalogue,configs,evaluate_episode,vector_observation,
)
from ht_pdm_fjsp.maintenance_framework_v17_policy import FullMatchingActor
from ht_pdm_fjsp.maintenance_framework_v14_campaign import V14Controller, load_checkpoint as load_v14_checkpoint
from ht_pdm_fjsp.maintenance_framework_v7_campaign import V7LearnedController,V7RuleController
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers
from ht_pdm_fjsp.maintenance_request_policies import request_action,risk_matching
from ht_pdm_fjsp.maintenance_requests import legal_matchings,submit_requests,transition

ROOT=Path(__file__).resolve().parents[2]
VERSION='maintenance_framework_v18_continuation_disagreement_dev_v1'
PROTOCOL=ROOT/'research_result/framework_v18_dev_protocol_vi.md'
PROPOSAL=ROOT/'research_result/v18_continuation_disagreement_proposal_vi.md'
REGISTRY=ROOT/'configs/maintenance_framework_v18_seed_registry.json'
V7=ROOT/'research_result/framework_v7_full_20261020T012531Z'
V14=ROOT/'research_result/framework_v14_dev_20261020T185303Z'
ARMS=('bounded_disagreement','own_rank')
V14_ARMS=('competition','count_calibrated')
V7_ARMS=('flat_free','shared_free','shared_canonical')
RULES=('fcfs','risk_matching','rollout')
ACTIONS=catalogue()
SOURCES=tuple(ROOT/path for path in (
    'src/ht_pdm_fjsp/maintenance_framework_v18_campaign.py',
    'src/ht_pdm_fjsp/maintenance_framework_v1.py',
    'src/ht_pdm_fjsp/maintenance_framework_v6_policy.py',
    'src/ht_pdm_fjsp/maintenance_framework_v7_campaign.py',
    'src/ht_pdm_fjsp/maintenance_framework_v17_policy.py',
    'src/ht_pdm_fjsp/maintenance_framework_v14_campaign.py',
    'research_result/verify_framework_v18.py',
    'src/ht_pdm_fjsp/maintenance_requests.py',
    'src/ht_pdm_fjsp/maintenance_dispatch.py',
    'src/ht_pdm_fjsp/maintenance_request_policies.py',
    'research_result/framework_v18_dev_protocol_vi.md',
    'research_result/v18_continuation_disagreement_proposal_vi.md',
    'configs/maintenance_framework_v18_seed_registry.json',
    'pyproject.toml','uv.lock',
))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n')

def add_row(path,row):
    path=Path(path);first=not path.exists()
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if first:writer.writeheader()
        writer.writerow(row)

def seed_registry():
    reg=json.loads(REGISTRY.read_text())
    if reg['campaign_version']!=VERSION:raise RuntimeError('registry version')
    groups=[range(reg[key]['start'],reg[key]['start']+reg[key]['count'])
            for key in ('training','selection_development','evaluation_development','smoke')]
    numbers=set().union(*(set(group) for group in groups),
                        {reg['forecast_seed'],reg['bootstrap_seed']})
    assert len(numbers)==sum(len(x) for x in groups)+2
    prior=set()
    for file in (ROOT/'configs').glob('*seed_registry.json'):
        if file!=REGISTRY:prior|=registered_integers(json.loads(file.read_text()))
    if numbers & prior:raise RuntimeError('v17 seed overlap with historical registry')
    return reg,tuple(map(tuple,groups))


def actor_new(seed,arm='bounded_disagreement'):
    torch.manual_seed(seed)
    return FullMatchingActor()


def actor_digest(actor):
    digest=hashlib.sha256()
    for name,value in sorted(actor.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def legal_present(config,state,tick):
    submissions=request_action(config,state,tick,'threshold')
    admitted=submit_requests(config,state,tick,submissions)
    allowed=set(legal_matchings(config,admitted))
    legal=tuple(i for i,a in enumerate(ACTIONS) if a in allowed)
    if not legal:raise AssertionError('no WAIT action')
    return submissions,admitted,legal,vector_observation(config,admitted,tick)


def masked_probabilities(actor,observation,legal):
    with torch.no_grad():
        logits=actor(torch.as_tensor(observation).reshape(1,56)).squeeze(0)
        allowed=torch.full_like(logits,-1e9)
        allowed[list(legal)]=logits[list(legal)]
        return torch.softmax(allowed,dim=0).cpu().numpy()


def policy_action(actor,observation,legal,uniform=None):
    probabilities=masked_probabilities(actor,observation,legal)
    if uniform is None:
        # np.argmax returns the first catalogue index on exact ties.
        return int(np.argmax(probabilities))
    cumulative=0.
    for index in legal:
        cumulative+=float(probabilities[index])
        if uniform<cumulative:return index
    return int(legal[-1])


class V18Controller:
    def __init__(self,actor,name):
        self.actor=actor;self.name=name;self.forecast_transitions=0
    def choose(self,config,state,tick,*,case,seed):
        submissions,_,legal,obs=legal_present(config,state,tick)
        return submissions,ACTIONS[policy_action(self.actor,obs,legal)]


class WaitDiagnostics:
    def __init__(self,controller):
        self.controller=controller;self.name=controller.name
        self.legal_service_ticks=0;self.legal_service_waits=0
    @property
    def forecast_transitions(self):return self.controller.forecast_transitions
    def choose(self,config,state,tick,*,case,seed):
        submissions,matching=self.controller.choose(config,state,tick,case=case,seed=seed)
        admitted=submit_requests(config,state,tick,submissions)
        if any(legal_matchings(config,admitted)):
            self.legal_service_ticks+=1
            self.legal_service_waits+=int(len(matching)==0)
        return submissions,matching


def evaluate_with_wait_fraction(config,case,seed,controller):
    counted=WaitDiagnostics(controller)
    row=evaluate_episode(config,case,seed,counted)
    row['legal_service_ticks']=counted.legal_service_ticks
    row['legal_service_waits']=counted.legal_service_waits
    row['legal_service_wait_fraction']=(counted.legal_service_waits/counted.legal_service_ticks
                                       if counted.legal_service_ticks else 0.)
    return row


def episode_seed(train_seed,iteration,case,index):
    return role_seed(train_seed,f'{VERSION}:train:{iteration}:{case}:{index}')%(2**31)


def state_rank(train_seed,iteration,case,physical_seed,tick):
    key=f'{VERSION}:rank:{train_seed}:{iteration}:{case}:{physical_seed}:{tick}'
    return hashlib.sha256(key.encode()).hexdigest()


def clock_bin(tick):return min(tick//16,2)


def collect(actor,train_seed,iteration,episodes_per_case,cap_per_case,deadline):
    configs_by_case=configs();records=[];tape=[]
    grid=[(case,i) for case in CASES for i in range(episodes_per_case)]
    for case,index in tqdm(grid,desc=f'On-policy {train_seed} iteration {iteration}',unit='episode',leave=False):
        deadline()
        physical_seed=episode_seed(train_seed,iteration,case,index)
        env=FrameworkEnv(configs_by_case[case]);env.reset(seed=physical_seed)
        rng=random.Random(role_seed(train_seed,
            f'{VERSION}:source_policy:{iteration}:{case}:{index}'))
        tape.append((case,physical_seed))
        for tick in range(HORIZON):
            state=env.inner.state
            _,_,legal,obs=legal_present(env.config,state,tick)
            action=policy_action(actor,obs,legal,rng.random())
            if len(legal)>=2:
                records.append(dict(case=case,physical_seed=physical_seed,tick=tick,
                    original_state=state,observation=obs,legal=legal,source_action=action,
                    rank=state_rank(train_seed,iteration,case,physical_seed,tick)))
            _,_,done,_,_=env.step(action)
        assert done
    selected=[];coverage={}
    quotas=(22,21,21) if cap_per_case==64 else (1,1,1)
    for case in CASES:
        for b,cap in enumerate(quotas):
            cell=sorted((x for x in records if x['case']==case and clock_bin(x['tick'])==b),
                        key=lambda x:x['rank'])
            chosen=cell[:cap];selected+=chosen
            coverage[f'{case}/{b}']=dict(eligible=len(cell),selected=len(chosen))
    selected.sort(key=lambda x:x['rank'])
    return selected,coverage,tape


def scenario_bank(anchor,train_seed,iteration,record,scenarios):
    key=f'{VERSION}:bank:{train_seed}:{iteration}:{record["case"]}:{record["physical_seed"]}:{record["tick"]}'
    seed=role_seed(anchor,key)
    rng=random.Random(seed)
    cfg=configs()[record['case']]
    # The bank begins at the current tick and extends through fixed tail H64.
    banks=[]
    for _ in range(scenarios):
        events=tuple(tuple(rng.random()<cfg.failure_probability for _ in range(cfg.machines))
                     for _ in range(record['tick'],cfg.horizon))
        uniforms=tuple(rng.random() for _ in range(record['tick'],HORIZON))
        banks.append((events,uniforms))
    return seed,banks


def forecast_cost(actor,config,record,root,events,uniforms,deadline,continuation):
    """Cost-to-go under forced root and selected continuation on common failures."""
    state=record['original_state'];origin=record['tick'];total=0.;steps=0
    for offset,tick in enumerate(range(origin,config.horizon)):
        deadline()
        submissions,admitted,legal,obs=legal_present(config,state,tick)
        if tick==origin:
            action=root
            if action not in legal:raise AssertionError('illegal counterfactual root')
        elif tick<HORIZON and continuation=='own':
            action=policy_action(actor,obs,legal,uniforms[tick-origin])
        else:
            action=ACTIONS.index(risk_matching(config,admitted,tick))
        state,cost,_=transition(config,state,tick,submissions,ACTIONS[action],events[offset])
        total+=cost;steps+=1
    return total,steps


def bounded_values(own,risk):
    legal=tuple(own)
    if set(legal)!=set(risk):raise ValueError('unmatched Q banks')
    own_mean=sum(own.values())/len(legal)
    risk_mean=sum(risk.values())/len(legal)
    raw={a:(risk[a]-risk_mean)-(own[a]-own_mean) for a in legal}
    correction={a:.5*float(np.clip(raw[a],-40.,40.)) for a in legal}
    return {a:own[a]+correction[a] for a in legal},raw,correction


def counterfactual_values(actor,record,anchor,train_seed,iteration,scenarios,deadline):
    cfg=configs()[record['case']]
    bank_seed,bank=scenario_bank(anchor,train_seed,iteration,record,scenarios)
    own={};risk={};transitions=0
    for action in record['legal']:
        own_total=risk_total=0.
        for events,uniforms in bank:
            value,count=forecast_cost(actor,cfg,record,action,events,uniforms,deadline,'own')
            own_total+=value;transitions+=count
            value,count=forecast_cost(actor,cfg,record,action,events,uniforms,deadline,'risk')
            risk_total+=value;transitions+=count
        own[action]=own_total/scenarios;risk[action]=risk_total/scenarios
    bounded,raw,correction=bounded_values(own,risk)
    return own,risk,bounded,raw,correction,transitions,bank_seed


def masked_logits(actor,obs,mask):
    raw=actor(obs)
    return raw.masked_fill(~mask,-1e9)


def assignment_groups(legal):
    """Legal action indices grouped by exact set of serviced machines."""
    groups={}
    for action in legal:
        machines=tuple(m for m,_ in ACTIONS[action])
        groups.setdefault(machines,[]).append(action)
    return tuple(tuple(group) for group in groups.values() if len(group)>=2)


def conditional_assignment_gain(logp,q,groups):
    """Equal weight per eligible machine set, then per selected state."""
    state_terms=[]
    for row,sets in enumerate(groups):
        terms=[]
        for choices in sets:
            conditional=torch.softmax(logp[row,list(choices)],dim=0)
            returns=q[row,list(choices)]
            terms.append((conditional*(returns.mean()-returns)/10.).sum())
        state_terms.append(torch.stack(terms).mean() if terms else logp[row,0]*0.)
    return torch.stack(state_terms).mean()


def update_actor(actor,records,own_rows,bounded_rows,arm,seed,iteration,deadline,epochs=5):
    obs=torch.tensor(np.asarray([r['observation'] for r in records]),dtype=torch.float32)
    mask=torch.zeros((len(records),len(ACTIONS)),dtype=torch.bool)
    qown=torch.zeros((len(records),len(ACTIONS)),dtype=torch.float32)
    qtarget=torch.zeros_like(qown)
    service=torch.tensor([len(a) for a in ACTIONS],dtype=torch.float32)[None,:]
    chosen=torch.tensor([r['source_action'] for r in records],dtype=torch.long)
    for i,(record,own,bounded) in enumerate(zip(records,own_rows,bounded_rows)):
        for action,value in own.items():
            mask[i,action]=True;qown[i,action]=value
            qtarget[i,action]=bounded[action] if arm=='bounded_disagreement' else value
        assert mask[i,chosen[i]]
    groups=[assignment_groups(record['legal']) for record in records]
    with torch.no_grad():
        old_logits=masked_logits(actor,obs,mask)
        old_logp=torch.log_softmax(old_logits,dim=1)
        old_p=torch.softmax(old_logits,dim=1)
        centered=(qtarget*mask).sum(dim=1,keepdim=True)/mask.sum(dim=1,keepdim=True)
        rank_target=torch.softmax(((centered-qtarget)/10.).masked_fill(~mask,-1e9),dim=1)
        old_service=(old_p*service).sum(dim=1)
    optimizer=torch.optim.Adam(actor.parameters(),lr=3e-4)
    rng=np.random.default_rng(role_seed(seed,f'{VERSION}:minibatch:{iteration}'))
    updates=0;stopped=False;last_loss=0.;last_flow=0.
    for epoch in range(epochs):
        for ids in np.array_split(rng.permutation(len(records)),max(1,int(np.ceil(len(records)/64)))):
            deadline();actor.train();optimizer.zero_grad()
            ids=torch.as_tensor(ids,dtype=torch.long)
            logp=torch.log_softmax(masked_logits(actor,obs[ids],mask[ids]),dim=1)
            p=logp.exp();kl=(old_p[ids]*(old_logp[ids]-logp)).sum(dim=1).mean()
            assignment=.5*conditional_assignment_gain(
                logp,qown[ids],[groups[int(i)] for i in ids])
            main_credit=-(rank_target[ids]*logp.masked_fill(~mask[ids],0.)).sum(dim=1).mean()
            new_service=(p*service).sum(dim=1)
            flow=10*torch.relu(old_service[ids].mean()-new_service.mean()-.05).square()
            loss=main_credit-assignment+kl+flow
            before={k:v.detach().clone() for k,v in actor.state_dict().items()}
            loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),.5);optimizer.step()
            with torch.no_grad():
                new_logp=torch.log_softmax(masked_logits(actor,obs,mask),dim=1)
                whole_kl=float((old_p*(old_logp-new_logp)).sum(dim=1).mean())
            if whole_kl>0.02:
                actor.load_state_dict(before);stopped=True;break
            updates+=1;last_loss=float(loss.detach());last_flow=float(flow.detach())
        if stopped:break
    actor.eval()
    with torch.no_grad():
        new_logp=torch.log_softmax(masked_logits(actor,obs,mask),dim=1)
        new_p=new_logp.exp()
        final_kl=float((old_p*(old_logp-new_logp)).sum(dim=1).mean())
        entropy=float((-(new_p*new_logp).masked_fill(~mask,0.)).sum(dim=1).mean())
        service_after=float((new_p*service).sum(dim=1).mean())
    return dict(updates=updates,stopped_on_kl=stopped,kl=final_kl,entropy=entropy,
                eligible_assignment_groups=sum(map(len,groups)),
                old_service_mean=float(old_service.mean()),new_service_mean=service_after,
                service_flow_penalty_last=last_flow,final_batch_loss=last_loss)


def checkpoint(actor,path,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=actor.state_dict(),metadata=metadata),path)
    return sha(path)


def load_checkpoint(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    actor=actor_new(0,payload['metadata']['arm']);actor.load_state_dict(payload['state_dict']);actor.eval()
    return actor


def v7_models():
    manifest=json.loads((V7/'manifest.json').read_text())
    if manifest['status']!='COMPLETED':raise RuntimeError('v7 comparator source incomplete')
    selection=json.loads((V7/'selection.json').read_text())
    models={};hashes={}
    for arm in V7_ARMS:
        for seed in range(68000000,68000005):
            item=selection[arm][str(seed)]
            path=V7/item['checkpoint']
            if sha(path)!=item['checkpoint_sha256']:raise RuntimeError('v7 checkpoint hash')
            models[(arm,seed)]=MaskablePPO.load(path,device='cpu')
            hashes[f'{arm}/{seed}']=item['checkpoint_sha256']
    return models,hashes,sha(V7/'selection.json')


def v14_models():
    manifest=json.loads((V14/'manifest.json').read_text())
    if manifest['status']!='COMPLETED':raise RuntimeError('V14 comparator incomplete')
    selection=json.loads((V14/'selection.json').read_text())
    if sha(V14/'selection.json')!=manifest['selection_sha256']:raise RuntimeError('V14 selection hash')
    models={};hashes={}
    for arm in V14_ARMS:
        for seed in range(72600000,72600005):
            item=selection[arm][str(seed)];path=V14/item['checkpoint']
            if sha(path)!=item['sha256']:raise RuntimeError('V14 checkpoint hash')
            models[(arm,seed)]=load_v14_checkpoint(path)
            hashes[f'{arm}/{seed}']=item['sha256']
    return models,hashes,sha(V14/'selection.json')


def summarize(episodes,train_seeds,physical_seeds,bootstrap_seed):
    result={'by_case':{},'pooled':{}}
    controllers=sorted({r['controller'] for r in episodes})
    for case in CASES:
        result['by_case'][case]={}
        for name in controllers:
            cell=[r for r in episodes if r['case']==case and r['controller']==name]
            measures=('objective','H_availability','H_failures','queued_cm_ticks',
                      'queued_pm_ticks','H_pending','end_pending','service_starts',
                      'maintenance_cost','unavailability_cost','failure_cost',
                      'tail_cost','selection_seconds','episode_wall_seconds',
                      'forecast_transitions','legal_service_ticks','legal_service_waits',
                      'legal_service_wait_fraction')
            result['by_case'][case][name]=dict(count=len(cell),means={
                metric:float(np.mean([float(r[metric]) for r in cell])) for metric in measures})
            if name in ARMS or name in V14_ARMS or name in V7_ARMS:
                result['by_case'][case][name]['training_seed_cost']={str(seed):float(np.mean([
                    float(r['objective']) for r in cell if int(r['train_seed'])==seed]))
                    for seed in (train_seeds if name in ARMS else range(72600000,72600005)
                                 if name in V14_ARMS else range(68000000,68000005))}
    for name in controllers:
        result['pooled'][name]=float(np.mean([result['by_case'][case][name]['means']['objective'] for case in CASES]))
    lookup={(r['case'],r['controller'],str(r['train_seed']),int(r['seed'])):float(r['objective'])
            for r in episodes}
    matrix=np.asarray([[np.mean([lookup[(case,'bounded_disagreement',str(train),seed)]-
        lookup[(case,'own_rank',str(train),seed)] for case in CASES])
        for seed in physical_seeds] for train in train_seeds])
    rng=np.random.default_rng(bootstrap_seed)
    draws=20000
    ti=rng.integers(0,len(train_seeds),size=(draws,len(train_seeds)))
    pi=rng.integers(0,len(physical_seeds),size=(draws,len(train_seeds),len(physical_seeds)))
    boot=matrix[ti[:,:,None],pi].mean(axis=(1,2))
    result['ablation']=dict(mean=float(matrix.mean()),hierarchical_ci95=np.quantile(boot,[.025,.975]).tolist(),
        train_seed_means={str(seed):float(matrix[i].mean()) for i,seed in enumerate(train_seeds)},
        physical_count=len(physical_seeds),training_count=len(train_seeds),
        wins=int((matrix<0).sum()),ties=int((matrix==0).sum()),losses=int((matrix>0).sum()))
    result['pooled_selection_seconds']={name:float(np.mean([
        result['by_case'][case][name]['means']['selection_seconds'] for case in CASES]))
        for name in controllers}
    control_seed_cost={seed:float(np.mean([lookup[(case,'own_rank',str(seed),physical)]
        for case in CASES for physical in physical_seeds])) for seed in train_seeds}
    strong=sorted(train_seeds,key=lambda seed:(control_seed_cost[seed],seed))[:3]
    strong_wins=sum(result['ablation']['train_seed_means'][str(seed)]<0 for seed in strong)
    risk_dominates=(result['pooled']['risk_matching']<=result['pooled']['bounded_disagreement'] and
        result['pooled_selection_seconds']['risk_matching']<=
        result['pooled_selection_seconds']['bounded_disagreement'])
    result['dev_gate_details']=dict(strong_control_seeds=strong,strong_wins=strong_wins,
                                    risk_dominates=risk_dominates)
    result['dev_go_gate']=bool(result['ablation']['mean']<=-10 and
        sum(x<0 for x in result['ablation']['train_seed_means'].values())>=4 and
        strong_wins>=2 and result['ablation']['hierarchical_ci95'][1]<0 and
        result['pooled']['bounded_disagreement']<=1.02*result['pooled']['rollout'] and
        result['pooled']['bounded_disagreement']<=result['pooled']['competition'] and
        result['pooled_selection_seconds']['rollout']>=
        10*result['pooled_selection_seconds']['bounded_disagreement'] and
        not risk_dominates)
    return result


def run(profile,output):
    if profile not in ('smoke','dev'):raise ValueError(profile)
    out=Path(output).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    started=time.monotonic();budget=1200 if profile=='smoke' else 14400
    def deadline():
        if time.monotonic()-started>=budget:raise TimeoutError('v18 campaign deadline')
    registry,groups=seed_registry()
    train=groups[0] if profile=='dev' else groups[3][:1]
    selection_physical=groups[1] if profile=='dev' else groups[3]
    dev=groups[2] if profile=='dev' else groups[3]
    iterations=6 if profile=='dev' else 1
    episodes_per_case=16 if profile=='dev' else 2
    cap_per_case=64 if profile=='dev' else 3
    scenarios=4 if profile=='dev' else 2
    physical_training=[episode_seed(seed,iteration,case,index)
        for seed in train for iteration in range(1,iterations+1)
        for case in CASES for index in range(episodes_per_case)]
    if len(physical_training)!=len(set(physical_training)) or set(physical_training)&(set(selection_physical)|set(dev)):
        raise RuntimeError('training/development physical seed collision')
    training_physical_digest=hashlib.sha256(json.dumps(physical_training).encode()).hexdigest()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    rel=[str(p.relative_to(ROOT)) for p in SOURCES]
    dirty=subprocess.check_output(['git','diff','--name-only','HEAD','--',*rel],cwd=ROOT,text=True).splitlines()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*rel],
                                      cwd=ROOT,text=True).splitlines()
    if profile=='dev' and (dirty or untracked):raise RuntimeError(f'uncommitted v18 source: {dirty+untracked}')
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCES}
    v7,comparison_hashes,v7_selection_hash=v7_models()
    v14,v14_hashes,v14_selection_hash=v14_models()
    manifest=dict(status='RUNNING',phase='preflight',profile=profile,
        started_utc=datetime.now(timezone.utc).isoformat(),git_commit=commit,
        invocation=[sys.executable,*sys.argv],cwd=str(ROOT),
        runtime=dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__),
        source_hashes=source_hashes,prior_comparator_hashes={**comparison_hashes,**v14_hashes},
        v14_selection_sha256=v14_selection_hash,
        v7_selection_sha256=v7_selection_hash,registry=registry,timeout_seconds=budget,
        training_physical_seed_count=len(physical_training),
        training_physical_seed_sha256=training_physical_digest)
    save_json(out/'manifest.json',manifest)
    shutil.copy2(PROPOSAL,out/'method_proposal.md')
    shutil.copy2(PROTOCOL,out/'protocol.md')
    shutil.copy2(REGISTRY,out/'seed_registry.json')
    save_json(out/'resolved_config.json',dict(iterations=iterations,episodes_per_case=episodes_per_case,
        cap_per_case=cap_per_case,scenarios=scenarios,ppo=None,actor='V18 bounded continuation disagreement versus own-Q ranking',
        optimizer=dict(name='Adam',learning_rate=.0003,epochs=5,minibatch=64,kl_penalty=1,
                       hard_kl_limit=.02,return_scale=10,
                       assignment_lambda={'bounded_disagreement':.5,'own_rank':.5},
                       service_flow_weight=10,service_flow_tolerance=.05),
        cases={k:asdict(v) for k,v in configs().items()},
        checkpoint_selection='eight selection-only physical seeds, four-case mean, tie earlier iteration'))
    try:
        candidate={arm:{} for arm in ARMS}
        starts={};tapes={};totals={arm:0 for arm in ARMS};sampled=[]
        manifest['phase']='training';save_json(out/'manifest.json',manifest)
        for train_seed in train:
            seed_starts=[];tapes[str(train_seed)]={}
            for arm in ARMS:
                deadline();actor=actor_new(train_seed,arm);actor.eval()
                seed_starts.append(actor_digest(actor))
                candidates=[(0,out/'checkpoints'/arm/str(train_seed)/'0.pt')]
                checkpoint(actor,candidates[-1][1],dict(arm=arm,training_seed=train_seed,iteration=0))
                tapes[str(train_seed)][arm]=[]
                for iteration in range(1,iterations+1):
                    selected,coverage,tape=collect(actor,train_seed,iteration,
                        episodes_per_case,cap_per_case,deadline)
                    if not selected:raise RuntimeError('no sampled multi-action states')
                    tapes[str(train_seed)][arm].append(tape)
                    own_rows=[];bounded_rows=[];transitions=0;roots=0;root_gap=[];corrections=[];clipped=0
                    start_iter=time.perf_counter()
                    for record in tqdm(selected,desc=f'Counterfactual {arm} {train_seed}/{iteration}',
                                       unit='state',leave=False):
                        deadline()
                        own,risk,bounded,raw,correction,count,bank_seed=counterfactual_values(actor,record,
                            registry['forecast_seed'],train_seed,iteration,scenarios,deadline)
                        own_rows.append(own);bounded_rows.append(bounded);transitions+=count;roots+=len(own)
                        corrections.extend(correction.values());clipped+=sum(abs(x)>40 for x in raw.values())
                        root_gap.append(own[record['source_action']]-min(own.values()))
                        # Complete provenance for selected state and its same-bank root values.
                        add_row(out/'counterfactual_states.partial.csv',dict(
                            arm=arm,train_seed=train_seed,iteration=iteration,case=record['case'],
                            physical_seed=record['physical_seed'],tick=record['tick'],
                            rank=record['rank'],legal=json.dumps(record['legal']),
                            source_action=record['source_action'],bank_seed=bank_seed,
                            scores_own=json.dumps(own,sort_keys=True),scores_risk=json.dumps(risk,sort_keys=True),
                            scores_bounded=json.dumps(bounded,sort_keys=True),
                            forecast_transitions=count,
                            original_state=json.dumps(asdict(record['original_state']),sort_keys=True)))
                    update=update_actor(actor,selected,own_rows,bounded_rows,arm,train_seed,iteration,deadline)
                    path=out/'checkpoints'/arm/str(train_seed)/f'{iteration}.pt'
                    digest=checkpoint(actor,path,dict(arm=arm,training_seed=train_seed,iteration=iteration))
                    candidates.append((iteration,path))
                    totals[arm]+=transitions
                    add_row(out/'training_trace.partial.csv',dict(arm=arm,train_seed=train_seed,
                        iteration=iteration,source_episodes=len(tape),source_steps=len(tape)*HORIZON,
                        selected_states=len(selected),coverage=json.dumps(coverage,sort_keys=True),
                        forecast_roots=roots,forecast_scenarios=scenarios,
                        forecast_transitions=transitions,
                        source_root_regret_mean=float(np.mean(root_gap)),
                        correction_abs_mean=float(np.mean(np.abs(corrections))),
                        correction_nonzero=sum(abs(x)>1e-9 for x in corrections),
                        correction_clipped=clipped,
                        wall_seconds=time.perf_counter()-start_iter,
                        checkpoint=str(path.relative_to(out)),checkpoint_sha256=digest,
                        actor_digest=actor_digest(actor),**update))
                candidate[arm][train_seed]=candidates
            if len(set(seed_starts))!=1:raise RuntimeError('paired initial actors differ')
            starts[str(train_seed)]=seed_starts[0]
            if tapes[str(train_seed)]['bounded_disagreement'][0]!=tapes[str(train_seed)]['own_rank'][0]:
                raise RuntimeError('first-iteration physical tape differs')
            save_json(out/'training_tapes.json',tapes)
        shutil.copy2(out/'training_trace.partial.csv',out/'training_trace.csv')
        shutil.copy2(out/'counterfactual_states.partial.csv',out/'counterfactual_states.csv')
        save_json(out/'initial_actor_hashes.json',starts)
        manifest['phase']='development';save_json(out/'manifest.json',manifest)
        development=[];grid=[(arm,seed,iteration,path,case,physical)
            for arm in ARMS for seed in train for iteration,path in candidate[arm][seed]
            for case in CASES for physical in selection_physical]
        loaded={}
        for arm,seed,iteration,path,case,physical in tqdm(grid,desc='V18 DEV checkpoint selection',unit='episode'):
            deadline();key=(arm,seed,iteration)
            if key not in loaded:loaded[key]=load_checkpoint(path)
            row=evaluate_with_wait_fraction(configs()[case],case,physical,V18Controller(loaded[key],arm))
            row.update(train_seed=seed,checkpoint_iteration=iteration,
                       checkpoint=str(path.relative_to(out)))
            development.append(row);add_row(out/'development.partial.csv',row)
        shutil.copy2(out/'development.partial.csv',out/'development.csv')
        selection={arm:{} for arm in ARMS}
        for arm in ARMS:
            for seed in train:
                candidates=[]
                for iteration,path in candidate[arm][seed]:
                    cell=[r for r in development if r['controller']==arm and r['train_seed']==seed
                          and r['checkpoint_iteration']==iteration]
                    if len(cell)!=len(CASES)*len(selection_physical):raise AssertionError('incomplete selection checkpoint cell')
                    candidates.append((float(np.mean([r['objective'] for r in cell])),iteration,path))
                value,iteration,path=min(candidates)
                selection[arm][str(seed)]=dict(iteration=iteration,
                    checkpoint=str(path.relative_to(out)),sha256=sha(path),development_mean_cost=value,
                    candidates=[dict(mean_cost=v,iteration=i,checkpoint=str(p.relative_to(out)))
                                for v,i,p in candidates])
        save_json(out/'selection.json',selection)
        manifest['selection_sha256']=sha(out/'selection.json')
        manifest['phase']='development_comparators';save_json(out/'manifest.json',manifest)
        episodes=[]
        controllers=[*ARMS,*V14_ARMS,*V7_ARMS,*RULES]
        grid=[(case,name,seed,physical) for case in CASES for name in controllers
              for seed in (train if name in ARMS else range(72600000,72600005)
                           if name in V14_ARMS else range(68000000,68000005)
                           if name in V7_ARMS else (None,)) for physical in dev]
        selected_models={(arm,seed):load_checkpoint(out/selection[arm][str(seed)]['checkpoint'])
                         for arm in ARMS for seed in train}
        for case,name,seed,physical in tqdm(grid,desc='V18 DEV competitive context',unit='episode'):
            deadline()
            if name in ARMS:
                controller=V18Controller(selected_models[(name,seed)],name)
                checkpoint_name=selection[name][str(seed)]['checkpoint']
            elif name in V14_ARMS:
                controller=V14Controller(v14[(name,seed)],name)
                checkpoint_name='immutable V14 selection'
            elif name in V7_ARMS:
                controller=V7LearnedController(v7[(name,seed)],canonical=(name=='shared_canonical'))
                controller.name=name;checkpoint_name='immutable V7 selection'
            else:
                controller=V7RuleController(name,registry['forecast_seed'],deadline)
                checkpoint_name=''
            row=evaluate_with_wait_fraction(configs()[case],case,physical,controller)
            row.update(train_seed=seed if seed is not None else '',checkpoint=checkpoint_name)
            if abs(row['episode_return']+row['objective'])>1e-8:raise RuntimeError('return/cost mismatch')
            if abs(row['objective']-sum(row[k] for k in
                   ('maintenance_cost','unavailability_cost','failure_cost')))>1e-8:
                raise RuntimeError('cost components mismatch')
            if row['requests_created']-row['requests_completed']!=row['end_pending']+row['end_in_service']:
                raise RuntimeError('request conservation mismatch')
            if row['invalid_assignments']!=0:raise RuntimeError('invalid assignment')
            episodes.append(row);add_row(out/'episodes.partial.csv',row)
        shutil.copy2(out/'episodes.partial.csv',out/'episodes.csv')
        if source_hashes!={str(p.relative_to(ROOT)):sha(p) for p in SOURCES}:
            raise RuntimeError('source changed during campaign')
        if any(sha(out/item['checkpoint'])!=item['sha256']
               for arm in ARMS for item in selection[arm].values()):
            raise RuntimeError('selected checkpoint changed')
        summary=summarize(episodes,train,dev,registry['bootstrap_seed'])
        save_json(out/'summary.json',summary)
        save_json(out/'audit.json',dict(status='PASS',paired_initial_weights=True,
            paired_training_physical_tape=True,source_hashes_stable=True,
            checkpoint_hashes_stable=True,zero_invalid_assignments=True,
            reward_cost_request_reconciled=True,development_episodes=len(development),
            competitive_episodes=len(episodes)))
        manifest.update(status='COMPLETED',phase='done',
            finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-started,
            counts=dict(training_steps=len(train)*len(ARMS)*iterations*episodes_per_case*
                        len(CASES)*HORIZON,training_forecast_transitions=totals,
                        development=len(development),evaluation=len(episodes)),
            actor_parameters=sum(p.numel() for p in actor_new(0).parameters()),
            artifact_hashes={p.name:sha(p) for p in out.iterdir()
                             if p.is_file() and p.name!='manifest.json'})
        save_json(out/'manifest.json',manifest)
        return out
    except Exception as exc:
        manifest.update(status='INCOMPLETE' if isinstance(exc,TimeoutError) else 'FAILED',
            error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-started)
        save_json(out/'manifest.json',manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('smoke','dev'),required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    print(json.dumps(dict(status='COMPLETED',output_dir=str(run(args.profile,args.output_dir)))))

if __name__=='__main__':main()
