import json
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_joint_execution_diagnostic as d


@pytest.mark.parametrize('arm',(d.x.CORE_ARMS[0],'ppo','normalized_ppo','mappo','full_mappo'))
def test_mean_matches_source_select_and_samples_replay_with_independent_physics(tmp_path,arm):
    torch.set_num_threads(1);c=d.x.settings('engineering');t=c['train_seeds'][0];model=d.x.make_model(arm,c,t)
    before=d.x.learner.parameter_digest(model);physical=d.x.Journal(tmp_path/'physical.csv');policy=d.x.Journal(tmp_path/'policy.csv')
    seeds=[250700001,250700002];all_episodes=[]
    try:
        for mode,repeat in [('mean',-1),('sample',0)]:
            generator=torch.Generator().manual_seed(72) if mode=='sample' else None
            episodes,audit=d.collect(model,arm,seeds,mode,generator,t,repeat,physical,policy,lambda:None)
            if mode=='mean':
                assert audit is None
                for row in episodes:
                    expected=d.x.evaluate(arm,t,row['seed'],model)
                    for key in ('extended_cost','availability','terminal_pending','policy_failures'):assert row[key]==expected[key]
            else:assert audit['behavior_normalized_max_error']<=1
            all_episodes.extend(episodes)
    finally:physical.close();policy.close()
    audit=d.audit_physics(tmp_path,all_episodes,d.primitive_helpers(),lambda:None)
    assert audit['physical_ticks']==240 and audit['policy_decisions']==192
    assert before==d.x.learner.parameter_digest(model) and all(p.grad is None for p in model.parameters())


def synthetic():
    rows=[]
    for arm in d.x.ARMS:
        for t in d.x.settings()['train_seeds']:
            for mode,repeat in [('mean',-1),*(('sample',i) for i in range(4))]:
                cost=(104 if arm==d.x.CORE_ARMS[0] else 100 if arm in ('ppo','normalized_ppo') else 103) if mode=='mean' else (100 if arm==d.x.CORE_ARMS[0] else 103 if arm in d.x.CORE_ARMS else 110)
                rows.append(dict(method=arm,train_seed=t,mode=mode,repeat=repeat,extended_cost=4*cost,availability=.67,terminal_pending=0,policy_failures=0,invalid_assignments=0))
    return rows


def test_fixed_engineering_gate_uses_mean_strong_references_and_every_replica():
    helper=d.primitive_helpers();rows=synthetic();s=d.summarize(rows,helper)
    assert s['stochastic_deployment_trial_motivated'] and len(s['mixed_reference_gates'])==13
    assert s['means']['sample']['ppo']['normalized_cost']==110
    for r in rows:
        if r['method']==d.x.CORE_ARMS[0] and r['mode']=='sample':r['extended_cost']=4*102
    s=d.summarize(rows,helper)
    assert not s['stochastic_deployment_trial_motivated'] and not s['mixed_reference_gates']['gain_vs_ppo']
    rows=synthetic()
    for r in rows:
        if r['method']==d.x.CORE_ARMS[0] and r['mode']=='sample' and r['train_seed']==250000000:r['extended_cost']=4*104
    assert not d.summarize(rows,helper)['engineering_checks']['every_replica_positive']


def test_seed_audit_fresh_raw_and_derived_panels():
    audit=d.seed_audit()
    assert audit['status']=='PASS' and audit['raw_unique']==76 and audit['derived_unique']==100
    assert audit['physical_seeds']==list(range(260100000,260100064))
    assert audit['action_seeds']==list(range(260200000,260200012))
