import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_broker_learning_efficiency as d

spec=importlib.util.spec_from_file_location('efficiency_audit',Path(__file__).resolve().parents[1]/'scripts/verify_broker_learning_efficiency.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def nodes(h):return [dict(method=a,train_seed=t,seed=s,checkpoint_steps=step,normalized_cost=10+step/245760) for a in h.ARMS for t in v.TRAINS for s in v.DEV for step in v.STEPS]


def test_independent_weighted_auc_has_correct_units_and_complete_node_grid():
    h=v.helpers();rows=nodes(h);result=v.auc_reference(rows,h);assert len(result)==768 and all(value==pytest.approx(10.5) for value in result.values())
    production=d.auc_rows(rows,dict(train_seeds=list(v.TRAINS),development_seeds=list(v.DEV)))
    for r in production:assert r['auc_cost_per_machine']==pytest.approx(result[r['method'],r['train_seed'],r['seed']])
    with pytest.raises(AssertionError,match='unique DEV'):v.auc_reference(rows+[rows[0]],h)
    with pytest.raises(AssertionError,match='full6144'):v.auc_reference(rows[:-1],h)


def test_fixed_14_endpoint_math_agrees_with_production_and_rejects_health():
    h=v.helpers();candidate=h.CORE[0];means={a:dict(normalized_cost=99.5 if a==candidate else 100.,availability=.67,normalized_pending=0.,normalized_failures=.1) for a in (*h.ARMS,'deadline',*h.FROZEN)}
    auc={(a,t,s):99 if a==candidate else 100. for a in h.ARMS for t in v.TRAINS for s in v.DEV};average,points,checks,per=v.endpoint_reference(means,auc,h);assert len(checks)==14 and all(checks.values())
    rows=[dict(method=a,train_seed=t,invalid_assignments=0,**m) for a,m in means.items() for t in v.TRAINS];ar=[dict(method=a,train_seed=t,seed=s,auc_cost_per_machine=value) for (a,t,s),value in auc.items()];out=d.summarize(rows,ar,dict(p95_ms=.5),dict(train_seeds=list(v.TRAINS)))
    assert out['checks']==checks and out['per_replica_auc']==per and out['auc_means']==average and out['endpoints']==points
    means[candidate]['normalized_failures']=.36;assert not v.endpoint_reference(means,auc,h)[2]['normalized_failures_delta_vs_ppo']


def test_numpy_helper_panel_override_keeps_source_physical_and_initialization_roles():
    torch.set_num_threads(1);h=v.helpers();assert h.TRAINS==v.TRAINS and h.DEV==v.DEV and h.VERSION==d.x.VERSION and v.STEPS==d.STEPS
    c=d.settings('engineering');t=c['train_seeds'][0]
    for a in h.ARMS:
        expected=h.independent_initial(a,t);actual=d.x.make_model(a,c,t).state_dict();assert list(expected)==list(actual) and all(torch.equal(expected[k],actual[k]) for k in expected)
