import importlib.util,itertools,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_capacity_order_experiment as e

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(SCRIPTS))
import capacity_order_reference as ref
spec=importlib.util.spec_from_file_location('capacity_auditor',SCRIPTS/'verify_capacity_order_development.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


@pytest.mark.parametrize('arm',e.ARMS)
def test_every_fresh_initial_tensor_reconstructed_independently(arm):
    torch.set_num_threads(1);c=e.settings('engineering');t=c['train_seeds'][0];actual=e.make_model(arm,c,t).state_dict();expected=v.independent_initial(arm,t,v.helpers());assert list(actual)==list(expected) and all(torch.equal(actual[k],expected[k]) for k in actual)


@pytest.mark.parametrize('arm',e.CORE)
def test_independent_catalogue_count_conditionals_and81latent_density(arm):
    torch.set_num_threads(1);h=v.helpers();o=h.observation(dict(ages=[3]*4,failed=[False]*4,pending_wait=[0]*4,remaining=[0]*2,assigned=[-1]*2),0);c=e.settings('engineering');m=e.make_model(arm,c,c['train_seeds'][0]).double()
    with torch.no_grad():m.actor[-1].weight.copy_(torch.arange(192).reshape(3,64)/2000-.05)
    weights={k:p.detach().numpy().copy() for k,p in m.state_dict().items()};u=ref.catalogue_tokens(h) if arm in e.CORE[:4] else np.array(list(itertools.product(range(3),repeat=4)));obs=[o]*len(u);b={k:t.double() if t.is_floating_point() else t for k,t in e.learner.tensors(obs).items()};actual=m.evaluate(b,torch.tensor(u))[0].detach().numpy();expected=ref.machine_reference(obs,u,arm,weights,h,False)['log_probability'];np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    if arm in e.CORE[:4]:assert np.exp(actual).sum()==pytest.approx(1.,abs=1e-12)


def test_manual_greedy_machine_and_rawtech_fields_with_urgent_and_busy_states():
    torch.set_num_threads(1);c=e.settings('engineering');h=v.helpers()
    for arm in (*e.CORE,'raw_tech_ppo'):
        m=e.make_model(arm,c,c['train_seeds'][0]);env=e.ResourceEnv(360700001,4,arm);o=env.reset();observations=[];actions=[]
        for tick in range(16):
            a=m.select(o);observations.append(o);actions.append(a);o,*_=env.step(a)
        w={k:p.detach().numpy().astype(float) for k,p in m.state_dict().items()};reference=ref.machine_reference(observations,actions,arm,w,h) if arm in e.CORE else ref.technician_reference(observations,actions,w,h);logs=m.evaluate(e.learner.tensors(observations),torch.tensor(np.array(actions)))[0].detach().numpy();np.testing.assert_allclose(logs,reference['log_probability'],atol=5e-6,rtol=1e-6)


@pytest.mark.parametrize('arm',['capacity_order_ppo','independent_machine_q_ppo','raw_tech_ppo'])
def test_stored_checkpoint_manual_actor_and_value_roundtrip(tmp_path,arm):
    torch.set_num_threads(1);c=e.settings('engineering');t=c['train_seeds'][0];model=e.make_model(arm,c,t);env=e.ResourceEnv(360700001,4,arm);o=env.reset();data=[]
    for tick in range(16):
        u=model.select(o);data.append((o,u));o,*_=env.step(u)
    path=tmp_path/'checkpoint.pt';meta=e.save_model(model,path,arm,c,t,491520,data[0][0]);meta['path']=path.name;result=v.check_neural_group(data,arm,t,tmp_path,{f'{arm}:{t}':meta},c,v.helpers());assert result['states']==16 and result['log_max_error']<5e-6


def test_independent_sixteen_gate_boundaries_match_production():
    c=e.settings();rows=[dict(method=a,train_seed=t,normalized_cost=99. if a==e.CORE[0] else 100.,availability=.67,normalized_pending=0.,normalized_failures=.1,invalid_assignments=0) for a in (*e.ARMS,'deadline') for t in c['train_seeds']];s=e.point_summary(rows,c,dict(p95_ms=.1));points,checks=v.endpoint_gates(s['means'],v.helpers());assert len(checks)==16 and checks==s['checks'] and points==s['endpoints']
    s['means'][e.CORE[0]]['normalized_cost']=99.6;_,checks=v.endpoint_gates(s['means'],v.helpers());assert not checks['gain_vs_fixed_order_ppo'] and not checks['gain_vs_independent_machine_q_ppo']


@pytest.mark.parametrize('arm',e.CORE[:4])
def test_every_one_of64_bipartite_masks_uniform_on_all_legal_matchings(arm):
    torch.set_num_threads(1);h=v.helpers();base=h.observation(dict(ages=[3]*4,failed=[False]*4,pending_wait=[0]*4,remaining=[0]*2,assigned=[-1]*2),0);table=ref.catalogue_tokens(h);observations=[];actions=[];expected=[]
    for active_bits,free_bits in itertools.product(itertools.product((False,True),repeat=4),itertools.product((False,True),repeat=2)):
        legal=np.array(active_bits)[:,None]&np.array(free_bits)[None];allowed=[u for u in table if all(legal[m,t-1] for m,t in enumerate(u) if t)];expected.extend([-np.log(len(allowed))]*len(allowed));observations.extend([dict(base,pair_mask=legal)]*len(allowed));actions.extend(allowed)
    c=e.settings('engineering');m=e.make_model(arm,c,c['train_seeds'][0]).double();weights={k:p.detach().numpy().copy() for k,p in m.state_dict().items()};b={k:t.double() if t.is_floating_point() else t for k,t in e.learner.tensors(observations).items()};actual=m.evaluate(b,torch.tensor(np.array(actions)))[0].detach().numpy();reference=ref.machine_reference(observations,actions,arm,weights,h,False)['log_probability'];np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0);np.testing.assert_allclose(reference,expected,atol=1e-12,rtol=0)
