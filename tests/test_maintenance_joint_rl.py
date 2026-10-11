import csv
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import itertools
import json

import numpy as np
import pytest
import torch

from ht_pdm_fjsp import maintenance_joint_rl as e
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE, INDEX, JOINT, FIXED, feature_batch
from ht_pdm_fjsp.maintenance_joint_training import complete_targets, fit
from ht_pdm_fjsp.maintenance_joint_statistics import bootstrap_samples, summarize
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, transition
from ht_pdm_fjsp.maintenance_heterogeneous_model import feasible, fixed_candidates
from ht_pdm_fjsp.maintenance_solver_diagnostics import Requests


def test_frozen_full_budget_and_historical_seed_audits():
    cfg=e.settings('full')
    assert e.counts(cfg)==dict(models=20,training_learned_steps=4915200,training_physical_steps=6144000,
        training_episodes=102400,optimizer_steps=76800,development_episodes=6848,test_episodes=21504,
        test_base_decisions=1032192,test_tail_decisions=258048,test_decisions=1290240,logged_test_decisions=10080)
    audit,_=e.seed_audit(cfg)
    assert audit['status']=='PASS' and audit['derived_unique_rng_roles']>100000
    for module in (e.prior,e.boundary):
        assert module.seed_audit(module.settings('full'))['status']=='PASS'
    from ht_pdm_fjsp import maintenance_confirmation as old
    assert old.seed_audit(old.settings('full'))['status']=='PASS'
    with pytest.raises(ValueError,match='registry'):e.seed_audit({**cfg,'analysis_seed':41})


def test_physical_split_exclusions_distribution_and_paired_conditions():
    cfg=e.settings('full');panels,rows=e.build_panels(cfg)
    smoke,_=e.build_panels(e.settings('smoke'))
    signatures={e.boundary.signature(c) for c in panels.values()}
    historical={e.boundary.frozen(s) for s in e.exclusions()['physical_signatures']}
    assert len(signatures)==len(panels)==len(rows)==144
    assert len(historical)==800 and not signatures & historical
    assert not signatures & {e.boundary.signature(c) for c in smoke.values()}
    assert e.build_panels(cfg)==(panels,rows)
    audit,forbidden=e.seed_audit(cfg);sampler=e.training_sampler(cfg,panels,forbidden)
    for index in range(64):
        c,condition,cs,es,attempt=sampler(cfg['train_seeds'][0],index)
        assert condition==e.CONDITIONS[index%2] and cs!=es
        assert e.boundary.signature(c) not in signatures|historical
        assert all(x in (0,2,3,4,5,6) for row in c.service_time for x in row)
        assert set(itertools.chain.from_iterable(c.restoration))<={.25,.5,.75,1.}
        assert sum(x==0 for row in c.service_time for x in row)==(2 if condition=='skill_mask' else 0)
    for cohort in cfg['test_cohorts']:
        dense=panels['test',cohort,'dense'];masked=panels['test',cohort,'skill_mask']
        assert dense.restoration==masked.restoration
        assert (dense.failure_age,dense.failure_probability)==(masked.failure_age,masked.failure_probability)
        assert all(d==m or m==0 for dr,mr in zip(dense.service_time,masked.service_time) for d,m in zip(dr,mr))


@pytest.mark.parametrize('condition',e.CONDITIONS)
def test_masks_exact_feasible_catalogue_assignment_choice_wait_and_future_privacy(condition):
    cfg=e.settings('smoke');c=e.make_config(12700020,condition,6)
    env=WaitingEnv(c);env.reset(12700030)
    joint=e.model_for(cfg,12700000,JOINT);fixed=e.model_for(cfg,12700000,FIXED)
    assert len(CATALOGUE)==21 and CATALOGUE[0]==()
    for t in range(4):
        batch=feature_batch([c],[env.state],[4-t],4)
        for name,mask in [('joint_mask',feasible(c,env.state)),('fixed_mask',fixed_candidates(c,env.state))]:
            assert {CATALOGUE[i] for i in torch.where(batch[name][0])[0]}==set(mask)
        assert batch['joint_mask'][0,0] and batch['fixed_mask'][0,0]
        assert {tuple(m for m,j in a) for a in feasible(c,env.state)}=={tuple(m for m,j in a) for a in fixed_candidates(c,env.state)}
        if t==0:assert int(batch['joint_mask'].sum())>int(batch['fixed_mask'].sum())
        assert torch.equal(batch['edges'][batch['edges'][...,0]==0],torch.zeros_like(batch['edges'][batch['edges'][...,0]==0]))
        before=joint.select(c,env.state,4-t,4)
        private=env.events;env.events=tuple((True,)*4 for _ in private)
        assert before==joint.select(c,env.state,4-t,4)
        env.events=private
        for model in [joint,fixed]:
            action,prob,value=model.sample(batch,generator=torch.Generator().manual_seed(t))
            p,entropy,v=model.evaluate(batch,action)
            torch.testing.assert_close(p,prob);torch.testing.assert_close(v,value)
            assert torch.isfinite(entropy).all()
            model.decode(c,env.state,action[0])
        env.step(before[0])
    assert sum(p.numel() for p in joint.parameters())==sum(p.numel() for p in fixed.parameters())
    assert all(torch.equal(v,fixed.state_dict()[k]) for k,v in joint.state_dict().items())


def test_matching_scores_equivariant_under_machine_and_technician_relabeling():
    cfg=e.settings('smoke');c=e.make_config(12700020,'skill_mask',6)
    env=WaitingEnv(c);env.reset(12700030);s=env.state
    machine_order=(2,0,3,1);tech_order=(1,0)
    machine_map={old:new for new,old in enumerate(machine_order)}
    tech_map={old:new for new,old in enumerate(tech_order)}
    cc=replace(c,service_time=tuple(tuple(c.service_time[m][j] for j in tech_order) for m in machine_order),
        restoration=tuple(tuple(c.restoration[m][j] for j in tech_order) for m in machine_order))
    ss=DispatchState(tuple(s.ages[m] for m in machine_order),tuple(s.failed[m] for m in machine_order),
        tuple(s.pending_wait[m] for m in machine_order),tuple(s.remaining[j] for j in tech_order),
        tuple(machine_map[s.assigned[j]] if s.assigned[j]>=0 else -1 for j in tech_order))
    model=e.model_for(cfg,12700000,JOINT)
    dist,v=model.distribution(feature_batch([c],[s],[4],4));other,vv=model.distribution(feature_batch([cc],[ss],[4],4))
    mapped=[INDEX[tuple(sorted((machine_map[m],tech_map[j]) for m,j in a))] for a in CATALOGUE]
    torch.testing.assert_close(dist.probs[0],other.probs[0,mapped],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(v,vv)


def test_returns_include_entire_tail_and_keep_time_major_order():
    rewards=torch.tensor([[-1.,-2.],[-3.,-4.],[-5.,-6.],[-7.,-8.],[-9.,-10.],[-11.,-12.]])
    target=complete_targets(rewards,4)
    expected=torch.stack([rewards[t:].sum(0) for t in range(4)])
    assert torch.equal(target,expected) and target.shape==(4,2)
    assert torch.equal(target.flatten(),torch.tensor([-36.,-42.,-35.,-40.,-32.,-36.,-27.,-30.]))
    assert not torch.equal(target,complete_targets(rewards[:4],4))
    for bad,h in [(rewards,0),(rewards,7),(torch.tensor([1.]),1),(rewards*torch.nan,4)]:
        with pytest.raises(ValueError):complete_targets(bad,h)


def test_checkpoint_selection_cannot_choose_initial_and_reload_contract(tmp_path):
    cfg=e.settings('smoke');c=e.make_config(12700020,'dense',6);env=WaitingEnv(c);env.reset(12700030)
    model=e.model_for(cfg,12700000,JOINT)
    restored,_=e.save_checkpoint(tmp_path/'model.pt',model,cfg,12700000,1,32,(c,env.state))
    assert restored.select(c,env.state,4,4)==model.select(c,env.state,4,4)
    records=[dict(update=0,mean_extended_cost=0),dict(update=2,mean_extended_cost=10),dict(update=1,mean_extended_cost=10)]
    assert e.choose_checkpoint(records)['update']==1
    with pytest.raises(ValueError):e.choose_checkpoint(records[:1])
    payload=torch.load(tmp_path/'model.pt',weights_only=True);payload['feature_contract']={}
    torch.save(payload,tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='contract'):e.load_checkpoint(tmp_path/'bad.pt')


def synthetic(cfg):
    metrics=('extended_cost_per_machine_tick','extended_objective','base_objective','tail_objective',
        'interior_cost_per_machine_tick','interior_pending_mean','interior_failed_mean','interior_overdue_per_tick',
        'cutoff_pending','cutoff_failed','post_tail_pending','post_tail_failed','nonfixed_assignments',
        'wait_with_choice','preventive_starts','corrective_starts','extended_censored_requests')
    rows=[]
    for cond,c,s in itertools.product(cfg['conditions'],cfg['test_cohorts'],cfg['test_shocks']):
        for name,seed in [('best_rule',-1),*itertools.product(cfg['arms'],cfg['train_seeds'])]:
            cost=100. if name=='best_rule' else 95. if name==FIXED else 90.
            rows.append(dict(condition=cond,cohort=c,shock=s,horizon=cfg['horizon'],controller=name,train_seed=seed,
                **{m:cost for m in metrics}))
    return rows


def test_crossed_bootstrap_paired_replicas_and_profiles_and_four_contrasts():
    a=np.array([[100.,2.],[70.,3.]]);b=np.array([[90.,1.],[65.,2.]])
    replicas=np.array([[0,0],[1,1],[0,1]]);profiles=np.array([[0,1],[0,0],[1,1]])
    expected=[]
    for r,p in zip(replicas,profiles):
        ix=np.ix_(r,p);expected.append((a[ix]-b[ix]).mean()/a[ix].mean())
    np.testing.assert_allclose(bootstrap_samples(a,b,replicas,profiles,True),expected)
    cfg=e.settings('smoke');rows=synthetic(cfg);summary,_=summarize(rows,cfg)
    assert len(summary['primary_contrasts'])==4
    assert all(x['confidence']==.9875 and x['status']=='ENGINEERING_ONLY' for x in summary['primary_contrasts'].values())
    result,_=summarize(rows,{**cfg,'profile':'full'})
    assert all(x['status']=='MATERIAL_GAIN' for x in result['primary_contrasts'].values())
    assert result['primary_contrasts']['dense:joint_vs_best_rule']['estimate']==pytest.approx(.1)
    for bad in [rows[:-1],rows+rows[:1],[{**r,'horizon':1} for r in rows]]:
        with pytest.raises(RuntimeError):summarize(bad,cfg)


def read(out,name):
    with (out/f'{name}.csv').open() as f:return list(csv.DictReader(f))


def strip_time(rows):
    return [{k:v for k,v in r.items() if 'ms_mean' not in k} for r in rows]


def test_end_to_end_smoke_replay_requests_carryover_hashes_training_and_determinism(tmp_path):
    out=tmp_path/'first';manifest=e.run(out)
    assert manifest['status']=='COMPLETED' and manifest['audit']=='PASS'
    assert not manifest['full_test_outcomes_opened'] and manifest['actual_counts']==e.counts(e.settings('smoke'))
    for filename,meta in manifest['artifacts'].items():
        assert hashlib.sha256((out/filename).read_bytes()).hexdigest()==meta['sha256']
    cfg=e.settings('smoke');panels,_=e.build_panels(cfg)
    coverage=json.loads((out/'training_coverage.json').read_text())
    assert all(r['parameters_changed'] and r['complete_tail_returns'] and r['probability_audit'] for r in coverage)
    assert all(r['mixture_counts']==dict(dense=8,skill_mask=8) for r in coverage)
    assert coverage[0]['initial_parameters_sha256']==coverage[1]['initial_parameters_sha256']
    groups=defaultdict(list);key=lambda r:tuple(r[k] for k in ('condition','cohort','shock','controller','train_seed'))
    for row in read(out,'decisions'):groups[key(row)].append(row)
    selection=json.loads((out/'checkpoint_selection.json').read_text())
    assert selection['locked_before_test']
    requests=[];carry=[]
    for result in read(out,'episodes'):
        c=replace(panels['test',int(result['cohort']),result['condition']],horizon=6)
        env=WaitingEnv(c);env.reset(int(groups[key(result)][0]['env_seed']))
        h=4;model=None
        if result['controller']!='best_rule':
            chosen=selection['selections'][f"{result['controller']}:{result['train_seed']}"]
            model,payload=e.load_checkpoint(out/chosen['selected_checkpoint'])
            assert payload['update']>0
        base=Requests(env.state);all_requests=Requests(env.state);totals=defaultdict(float)
        for row in groups[key(result)]:
            t=int(row['time']);state=DispatchState(**{k:tuple(v) for k,v in json.loads(row['state']).items()})
            assert state==env.state
            pairs=tuple(map(tuple,json.loads(row['action'])))
            if t<h and model is not None:assert pairs==model.select(c,state,h-t,h)[0]
            else:
                rule='deadline_matching' if t>=h else selection['selected_rules'][result['condition']]
                assert pairs==e.prior.rule_action(c,state,6-t if t>=h else h-t,rule)
            following,cost,info=transition(c,state,pairs,env.events[t]);env.step(pairs)
            assert following==env.state and json.loads(json.dumps(asdict(following)))==json.loads(row['following'])
            assert info==json.loads(row['physical_metrics'])
            totals[row['phase']]+=cost;all_requests.step(t,state,pairs,following)
            if t<h:base.step(t,state,pairs,following)
            if t+1==h:cut=base.finish(h)
        extended=all_requests.finish(6)
        assert float(result['base_objective'])==totals['base']
        assert float(result['tail_objective'])==totals['tail']
        assert float(result['extended_objective'])==sum(totals.values())
        identity={k:result[k] for k in ('condition','cohort','shock','horizon','controller','train_seed','checkpoint_update')}
        for scope,records in [('cutoff',cut),('extended',extended)]:
            for r in records:requests.append({**identity,'scope':scope,**{k:str(v) for k,v in r.items()},'overdue':str(r['wait']>c.waiting_limit)})
        for r in cut:
            if r['censored']:
                later=next(x for x in extended if (x['machine'],x['request'])==(r['machine'],r['request']))
                carry.append({**identity,**{k:str(v) for k,v in dict(machine=r['machine'],request=r['request'],arrival=r['arrival'],
                    wait_at_cutoff=r['wait'],started_in_tail=later['served'],wait_at_start_or_end=later['wait'],censored_after_tail=later['censored']).items()}})
    assert requests==read(out,'requests') and carry==read(out,'carryover')
    e.run(tmp_path/'second')
    assert strip_time(read(out,'episodes'))==strip_time(read(tmp_path/'second','episodes'))
    assert read(out,'decisions')==read(tmp_path/'second','decisions')
    assert read(out,'training_episodes')==read(tmp_path/'second','training_episodes')
    assert json.loads((out/'summary.json').read_text())==json.loads((tmp_path/'second'/'summary.json').read_text())
    with pytest.raises(ValueError,match='new/empty'):e.run(out)


def test_interrupt_preserves_manifest_partial_journals_and_full_preflight(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'fit',lambda *args,**kwargs:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):e.run(tmp_path/'interrupt')
    manifest=json.loads((tmp_path/'interrupt'/'manifest.json').read_text())
    assert manifest['status']=='FAILED' and 'KeyboardInterrupt' in manifest['error']
    assert not manifest['full_test_outcomes_opened'] and manifest['artifacts']
    original=e.subprocess.check_output
    monkeypatch.setattr(e.subprocess,'check_output',lambda args,**kw: " M dummy" if args==['git','status','--porcelain'] else original(args,**kw))
    with pytest.raises(ValueError,match='clean committed'):e.run(tmp_path/'full',profile='full')
    assert not json.loads((tmp_path/'full'/'manifest.json').read_text())['full_test_outcomes_opened']
