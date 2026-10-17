import itertools,math
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_typed_gibbs_entropy_units_experiment as e
from ht_pdm_fjsp import technician_typed_gibbs_entropy_units_policy as p
from ht_pdm_fjsp.technician_typed_gibbs_policy import TypedGibbsPolicy
from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue


def observation():return e.helpers().observation(dict(ages=[3]*4,failed=[False]*4,pending_wait=[0]*4,remaining=[0]*2,assigned=[-1]*2),0)
def batch(obs):return {k:v.double() if v.is_floating_point() else v for k,v in e.learner.tensors(obs).items()}
def matching_actions(mask):
    out=[]
    for pairs in catalogue(4,2):
        if not all(mask[m,j] for m,j in pairs):continue
        u=np.zeros(4,int)
        for m,j in pairs:u[m]=j+1
        out.append(u)
    return torch.tensor(np.array(out))


@pytest.mark.parametrize('index',range(3))
def test_frozen_unit_pairs_preserve_base_likelihood_value_map_sample_and_exact_gradients(index):
    torch.set_num_threads(1);torch.manual_seed(400700002);joint=p.EntropyUnitsPolicy(p.CORE[index]).double();machine=p.EntropyUnitsPolicy(p.CORE[index+3]).double();base=TypedGibbsPolicy(p.BASE[p.CORE[index]]).double()
    with torch.no_grad():joint.actor[-1].weight.copy_(torch.arange(128).reshape(2,64)/300-.2)
    machine.load_state_dict(joint.state_dict());base.load_state_dict(joint.state_dict());o=observation();actions=matching_actions(o['pair_mask']);b=batch([o]*len(actions));jl,jh,jv=joint.evaluate(b,actions);ml,mh,mv=machine.evaluate(b,actions);bl,bh,bv=base.evaluate(b,actions)
    torch.testing.assert_close(jl,ml,atol=0,rtol=0);torch.testing.assert_close(ml,bl,atol=0,rtol=0);torch.testing.assert_close(jh,4*mh,atol=0,rtol=0);torch.testing.assert_close(mh,bh,atol=0,rtol=0);torch.testing.assert_close(jv,mv,atol=0,rtol=0);torch.testing.assert_close(mv,bv,atol=0,rtol=0)
    for x,y,scale in ((jl,ml,1),(jh,mh,4)):
        gx=torch.autograd.grad(x.mean(),tuple(joint.actor.parameters()),retain_graph=True);gy=torch.autograd.grad(y.mean(),tuple(machine.actor.parameters()),retain_graph=True)
        for a,c in zip(gx,gy):torch.testing.assert_close(a,c*scale,atol=1e-10,rtol=0)
    for deterministic in (False,True):
        j=joint.sample(b,torch.Generator().manual_seed(400700002),deterministic);m=machine.sample(b,torch.Generator().manual_seed(400700002),deterministic);old=base.sample(b,torch.Generator().manual_seed(400700002),deterministic)
        for a,c,d in zip(j,m,old):torch.testing.assert_close(a,c,atol=0,rtol=0);torch.testing.assert_close(c,d,atol=0,rtol=0)
    probe=joint.entropy_probe(batch([o]));other=machine.entropy_probe(batch([o]));assert probe['base_density']==other['base_density']==p.BASE[p.CORE[index]];assert probe['mean_normalized_entropy']==other['mean_normalized_entropy'];assert joint.float().select(o).tolist()==machine.float().select(o).tolist()==base.float().select(o).tolist()


@pytest.mark.parametrize('algorithm',p.CORE)
def test_all256_cold_masks_uniform_and_correct_entropy_units(algorithm):
    torch.set_num_threads(1);model=p.EntropyUnitsPolicy(algorithm).double()
    for flags in itertools.product((False,True),repeat=8):
        o=observation();o['pair_mask']=np.array(flags).reshape(4,2);actions=matching_actions(o['pair_mask']);b=batch([o]*len(actions));lp,h,_=model.evaluate(b,actions);torch.testing.assert_close(lp,torch.full_like(lp,-math.log(len(actions))),atol=1e-10,rtol=0);expected=math.log(len(actions))*(1 if algorithm in p.CORE[:3] else .25);torch.testing.assert_close(h,torch.full_like(h,expected),atol=1e-10,rtol=0)


def test_base_mapping_is_real_nonzero_gibbs_not_accidentally_myopic_and_select_is_actor_only(monkeypatch):
    torch.set_num_threads(1);g=p.EntropyUnitsPolicy(p.CORE[0]).double();m=p.EntropyUnitsPolicy(p.CORE[1]).double();o=observation()
    with torch.no_grad():g.actor[-1].weight.copy_(torch.arange(128).reshape(2,64)/250-.2)
    m.load_state_dict(g.state_dict());actions=matching_actions(o['pair_mask']);b=batch([o]*21);gl=g.evaluate(b,actions)[0];ml=m.evaluate(b,actions)[0];assert float((gl-ml).abs().max().detach())>1e-5
    def fail(*args):raise AssertionError('critic/catalogue in select')
    monkeypatch.setattr(g.critic,'forward',fail)
    import ht_pdm_fjsp.maintenance_heterogeneous_model as original
    monkeypatch.setattr(original,'catalogue',fail);assert g.float().select(o).shape==(4,)
    env=e.ResourceEnv(400700001,4,p.CORE[0]);assert 'joint_mask' not in env.reset()


def test_busy_worker_type_contract_and_probe_mask_scope():
    torch.set_num_threads(1);model=p.EntropyUnitsPolicy(p.CORE[0]);o=observation();o['technicians'][0,0]=0;o['pair_mask'][:,0]=False;u=model.select(o);assert 1 not in u
    probe=model.entropy_probe(e.learner.tensors([o]));assert probe['mean_initial_available_workers']==1
    b=e.learner.tensors([o]);b['pair_mask'][0,0,0]=True
    with pytest.raises(ValueError,match='eligibility'):model.tables(b)
    with pytest.raises(ValueError):e.ResourceEnv(400700001,8,p.CORE[0])
