import itertools,math
from collections import defaultdict
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_capacity_order_experiment as e
from ht_pdm_fjsp import technician_capacity_order_policy as p
from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue


def full_observation():
    return e.helpers().observation(dict(ages=[3]*4,failed=[False]*4,pending_wait=[0]*4,remaining=[0]*2,assigned=[-1]*2),0)


def batch(observations,double=False):
    b=e.learner.tensors(observations)
    return {k:v.double() if double and v.is_floating_point() else v for k,v in b.items()}


def vectors(mask):
    result=[]
    for pairs in catalogue(4,2):
        if not all(mask[m,j] for m,j in pairs):continue
        u=np.zeros(4,int)
        for m,j in pairs:u[m]=j+1
        result.append(u)
    return np.array(result)


@pytest.mark.parametrize('algorithm',p.CORE[:4])
def test_exact_uniform_physical_matching_prior_all_mask_sizes_and_canonical_density(algorithm):
    torch.set_num_threads(1);model=p.CapacityOrderPolicy(algorithm).double()
    for n,k in itertools.product(range(5),range(3)):
        o=full_observation();o['pair_mask']=np.zeros((4,2),bool);o['pair_mask'][:n,:k]=True;u=vectors(o['pair_mask']);b=batch([o]*len(u),True);log=model.evaluate(b,torch.tensor(u))[0];expected=-math.log(len(u));torch.testing.assert_close(log,torch.full_like(log,expected),atol=1e-12,rtol=0);assert float(log.exp().sum().detach())==pytest.approx(1.,abs=1e-12)


def test_initial_capacity_order_known_and_mask_rejections_and_actor_only_select(monkeypatch):
    o=full_observation();b=batch([o]);assert p.structural_inputs(b,p.CORE[0])[2].tolist()==[[1,3,0,2]]
    assert p.structural_inputs(b,'reverse_capacity_ppo')[2].tolist()==[[0,2,1,3]]
    model=p.CapacityOrderPolicy(p.CORE[0]);bad=b.copy();bad['pair_mask']=b['pair_mask'].clone();bad['pair_mask'][0,0,0]=False
    with pytest.raises(ValueError,match='complete bipartite'):model.sample(bad,torch.Generator().manual_seed(360700002))
    with pytest.raises(ValueError):model.evaluate(b,torch.tensor([[1,1,0,0]]))
    def fail(*args):raise AssertionError('critic inselect')
    monkeypatch.setattr(model.critic,'forward',fail);assert model.select(o).shape==(4,)


def test_independent_physical_q_all81_preimages_and_full_log_gradient():
    torch.set_num_threads(1);torch.manual_seed(360700002);q=p.CapacityOrderPolicy('independent_machine_q_ppo').double();raw=p.CapacityOrderPolicy('independent_machine_p_ppo').double();raw.load_state_dict(q.state_dict())
    with torch.no_grad():q.actor[-1].weight.copy_(torch.arange(192).reshape(3,64)/2000-.05);raw.actor[-1].weight.copy_(q.actor[-1].weight)
    o=full_observation();u=np.array(list(itertools.product(range(3),repeat=4)));b=batch([o]*81,True);a=torch.tensor(u);lp=raw.evaluate(b,a)[0];lq=q.evaluate(b,a)[0];groups=defaultdict(list)
    for i,tokens in enumerate(u):groups[p.decode_machine_tokens(tokens,o['pair_mask'],o['edges'],q.algorithm)[0]].append(i)
    assert len(groups)==21
    for ids in groups.values():torch.testing.assert_close(lq[ids],torch.logsumexp(lp[ids],0).expand(len(ids)),atol=1e-12,rtol=0)
    ids=max(groups.values(),key=len);g1=torch.autograd.grad(lq[ids[0]],tuple(q.actor.parameters()));g2=torch.autograd.grad(torch.logsumexp(lp[ids],0),tuple(raw.actor.parameters()))
    for x,y in zip(g1,g2):torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-10)


def test_no_catalogue_in_core_environment_and_synthetic_polynomial_shapes(monkeypatch):
    import ht_pdm_fjsp.maintenance_heterogeneous_model as original
    def fail(*args):raise AssertionError('core enumeration')
    monkeypatch.setattr(original,'catalogue',fail);o=e.ResourceEnv(360700001,4,p.CORE[0]).reset();assert 'joint_mask' not in o
    for n,k in ((8,4),(16,8)):
        model=p.CapacityOrderPolicy(p.CORE[0],n,k);state_dim=6*n+3*k+4*n*k+9;b=dict(state=torch.zeros(2,state_dim),machines=torch.zeros(2,n,6),technicians=torch.zeros(2,k,3),edges=torch.ones(2,n,k,4),pair_mask=torch.ones(2,n,k,dtype=torch.bool));b['edges'][...,1]=1/12;u,l,_=model.sample(b,torch.Generator().manual_seed(360700002));assert u.shape==(2,n) and torch.isfinite(l).all();l2=model.evaluate(b,u)[0];torch.testing.assert_close(l,l2);assert all(len(set(row[row>0].tolist()))==int((row>0).sum()) for row in u)
