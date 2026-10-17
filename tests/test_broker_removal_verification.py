import importlib.util,json,shutil
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_broker_removal_experiment as e

spec=importlib.util.spec_from_file_location('removal_verifier',Path(__file__).resolve().parents[1]/'scripts/verify_broker_removal_development.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


@pytest.mark.parametrize('arm',v.ARMS)
def test_independent_fresh_constructor_rng_consumption_and_tensor_order(arm):
    torch.set_num_threads(1);c=e.settings('engineering');t=c['train_seeds'][0];actual=e.make_model(arm,c,t).state_dict();reference=v.independent_initial(arm,t,v.helpers())
    assert list(actual)==list(reference)
    assert all(torch.equal(actual[k],reference[k]) for k in actual)


@pytest.mark.parametrize('arm',(*v.CORE,'capacity_ppo'))
def test_numpy_dense_conditional_baseline_on_nonconstant_critic_and_masked_states(tmp_path,arm):
    torch.set_num_threads(1);c=e.settings('engineering');t=c['train_seeds'][0];model=e.make_model(arm,c,t)
    if arm!='blind_removal_ppo':
        with torch.no_grad():model.removal[0].weight[:,71:79].copy_(torch.arange(512).reshape(64,8)/1000-.25)
    env=e.ResourceEnv(330700001,4,arm);obs=env.reset();data=[]
    for tick in range(20):
        action=model.select(obs);data.append((obs,action));obs,*_=env.step(action)
    path=tmp_path/'checkpoint.pt';choice=e.save_model(model,path,arm,c,t,491520,data[0][0]);choice['path']=path.name
    result=v.check_neural_group(data,arm,t,tmp_path,{f'{arm}:{t}':choice},c,v.helpers())
    assert result['states']==20 and result['conditional_baseline_max_error']<.001
    if arm!='blind_removal_ppo':assert result['trained_context_weight_norm']>0


def test_gate_boundaries_and_no_implicit_health_relaxation():
    means={a:dict(normalized_cost=100.,availability=.7,normalized_pending=0.,normalized_failures=.1) for a in v.ARMS};means[v.CORE[0]]['normalized_cost']=99.
    points,checks=v.endpoint_gates(means,v.helpers());assert len(checks)==19 and all(checks.values())
    means[v.CORE[0]].update(normalized_cost=99.6,availability=.68,normalized_pending=.026,normalized_failures=.351)
    _,checks=v.endpoint_gates(means,v.helpers());assert not checks['gain_vs_factor_state_ppo']
    assert all(not checks[f'{metric}_delta_vs_{a}'] for a in v.REFS for metric in ('availability','normalized_pending','normalized_failures'))


def test_independent_seed_contract_rejects_changed_registry_or_overlap(tmp_path):
    h=v.helpers();registry=Path(__file__).resolve().parents[1]/'configs/technician_broker_removal_ppo_v1_seed_registry.json';shutil.copy2(registry,tmp_path/'seed_registry.json');historical=tmp_path/'historical.json';historical.write_text('{"train_seeds":[101,102]}');audit=dict(status='PASS',raw_unique=365,derived_unique=82397,confirmation_states_constructed=False,historical_registries={h.sha(historical):str(historical)});(tmp_path/'seed_audit.json').write_text(json.dumps(audit));result=v.seed_contract(tmp_path,h);assert result['raw']==365 and result['derived']==82397
    historical.write_text('{"train_seeds":[330000000]}')
    with pytest.raises(AssertionError,match='historical registry binding'):v.seed_contract(tmp_path,h)
    audit['historical_registries']={h.sha(historical):str(historical)};(tmp_path/'seed_audit.json').write_text(json.dumps(audit))
    with pytest.raises(AssertionError,match='independent raw/derived separation'):v.seed_contract(tmp_path,h)
