import itertools,math
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_typed_gibbs_experiment as e
from ht_pdm_fjsp import technician_typed_gibbs_policy as p
from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue


def observation():return e.helpers().observation(dict(ages=[3]*4,failed=[False]*4,pending_wait=[0]*4,remaining=[0]*2,assigned=[-1]*2),0)
def batch(obs):return {k:v.double() if v.is_floating_point() else v for k,v in e.learner.tensors(obs).items()}
def actions(mask):
    result=[]
    for pairs in catalogue(4,2):
        if not all(mask[m,j] for m,j in pairs):continue
        u=np.zeros(4,int)
        for m,j in pairs:u[m]=j+1
        result.append(u)
    return torch.tensor(np.array(result),dtype=torch.long)


@pytest.mark.parametrize('algorithm',p.CORE)
def test_uniform_matching_density_all256_type_eligibility_graphs(algorithm):
    torch.set_num_threads(1);model=p.TypedGibbsPolicy(algorithm).double()
    for flags in itertools.product((False,True),repeat=8):
        o=observation();o['pair_mask']=np.asarray(flags).reshape(4,2);a=actions(o['pair_mask']);b=batch([o]*len(a));lp,h,_=model.evaluate(b,a)
        torch.testing.assert_close(lp,torch.full_like(lp,-math.log(len(a))),atol=1e-10,rtol=0);torch.testing.assert_close(h,torch.full_like(h,math.log(len(a))/4),atol=1e-10,rtol=0)


@pytest.mark.parametrize('algorithm',p.CORE)
def test_nonzero_sampling_replay_full_gradient_map_and_exact_entropy(algorithm):
    torch.set_num_threads(1);torch.manual_seed(380700002);model=p.TypedGibbsPolicy(algorithm).double();o=observation()
    with torch.no_grad():model.actor[-1].weight.copy_(torch.arange(128).reshape(2,64)/300-.2)
    b=batch([o]*128);x,log,value=model.sample(b,torch.Generator().manual_seed(380700002));y,other,_=model.sample(b,torch.Generator().manual_seed(380700002));assert torch.equal(x,y);lp,h,v=model.evaluate(b,x);torch.testing.assert_close(log,lp,atol=1e-10,rtol=0);torch.testing.assert_close(value,v,atol=1e-10,rtol=0)
    for a in x.numpy():p.decode_tokens(a,o['pair_mask'])
    a=actions(o['pair_mask']);all_lp,all_h,_=model.evaluate(batch([o]*len(a)),a);expected=-(all_lp.exp()*all_lp).sum()/4;torch.testing.assert_close(all_h,expected.expand_as(all_h),atol=1e-10,rtol=0);chosen=model.trace(batch([o]),deterministic=True)[0];chosen_lp=model.evaluate(batch([o]),chosen)[0];torch.testing.assert_close(chosen_lp,all_lp.max().reshape(1),atol=1e-10,rtol=0)
    loss=-((lp-log).exp()*torch.linspace(-1.,1.,128)).mean()-.01*h.mean();loss.backward();assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.actor.parameters());assert sum(float(t.grad.abs().sum()) for t in model.actor.parameters())>0


def test_busy_workers_raw_contract_and_no_critic_catalogue_select(monkeypatch):
    torch.set_num_threads(1);model=p.TypedGibbsPolicy(p.CORE[0]);o=observation();o['technicians'][0,0]=0;o['pair_mask'][:,0]=False
    def fail(*args,**kwargs):raise AssertionError('forbidden critic/catalogue')
    monkeypatch.setattr(model.critic,'forward',fail)
    import ht_pdm_fjsp.maintenance_heterogeneous_model as original
    monkeypatch.setattr(original,'catalogue',fail);a=model.select(o);assert 1 not in a;p.decode_tokens(a,o['pair_mask']);env=e.ResourceEnv(380700001,4,p.CORE[0]);assert 'joint_mask' not in env.reset()
    with pytest.raises(ValueError):p.decode_tokens(np.array([2,2,0,0]),o['pair_mask'])
    bad=e.learner.tensors([o]);bad['pair_mask'][0,0,0]=True
    with pytest.raises(ValueError,match='eligibility'):model.tables(bad)
    with pytest.raises(ValueError):e.ResourceEnv(380700001,8,p.CORE[0])


def test_synthetic_fixed_two_type_dimensions_and_repeated_profile_contract():
    torch.set_num_threads(1)
    for n,k in ((8,4),(16,8)):
        model=p.TypedGibbsPolicy(p.CORE[0],n,k);b=dict(state=torch.zeros(2,model.state_dim),machines=torch.zeros(2,n,6),technicians=torch.ones(2,k,3),edges=torch.ones(2,n,k,4),pair_mask=torch.ones(2,n,k,dtype=torch.bool));x,lp,_=model.sample(b,torch.Generator().manual_seed(380700002));assert x.shape==(2,n);torch.testing.assert_close(model.evaluate(b,x)[0],lp)
        b['edges'][0,0,2,1]=2
        with pytest.raises(ValueError,match='profile'):model.tables(b)
        b['edges'][0,0,2,1]=1;b['pair_mask'][0,0,2]=False
        with pytest.raises(ValueError,match='eligibility'):model.tables(b)
