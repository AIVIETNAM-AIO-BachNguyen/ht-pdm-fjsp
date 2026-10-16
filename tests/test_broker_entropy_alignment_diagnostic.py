import itertools,json
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_broker_entropy_alignment_diagnostic as d


@pytest.mark.parametrize('masked',[False,True,'forced'])
def test_dense_entropy_and_full_logit_gradients_match_closed_physical_masses(masked):
    h=d.helpers();u,matrix=d.preimage_matrix(h);raw=torch.tensor(np.cos(np.arange(10).reshape(1,2,5)),dtype=torch.float64,requires_grad=True);legal=torch.ones((1,4,2),dtype=torch.bool)
    if masked is True:legal[0,1,0]=False;legal[0,:,1]=False
    if masked=='forced':legal[:]=False
    support=torch.cat([torch.ones((1,2,1),dtype=torch.bool),legal.transpose(1,2)],-1);logp=raw.masked_fill(~support,-1e9).log_softmax(-1);hp,hq,mass=d.entropies(logp,u,matrix);priority=torch.tensor([[[.5,.5],[.5,.25],[.5,.5],[.25,.5]]],dtype=torch.float64);terms=[]
    for pairs in h.baseline_catalogue():
        if not all(legal[0,m,j] for m,j in pairs):continue
        physical=torch.zeros((1,2),dtype=torch.long)
        for m,j in pairs:physical[0,j]=m+1
        logq=d.physical_logprob(logp,physical,legal,priority);terms.append(-(logq.exp()*logq)/2)
    reference=sum(terms);np.testing.assert_allclose(mass.detach().sum(1),1,atol=2e-14);assert torch.all(hp>=hq-2e-14)
    torch.testing.assert_close(hq,reference,atol=2e-14,rtol=2e-14);a=torch.autograd.grad(hq.sum(),raw,retain_graph=True)[0];b=torch.autograd.grad(reference.sum(),raw,retain_graph=True)[0];torch.testing.assert_close(a,b,atol=2e-14,rtol=2e-14)
    # Explicit conditional entropy of proposal aliases verifies the chain rule.
    probability=logp.exp();latent=(probability[:,0,u[:,0]]*probability[:,1,u[:,1]])[0];alias=0.
    for i,p in enumerate(latent):
        if p.detach()>0:alias-=p*(p/mass[0,int(matrix[i].argmax())]).log()/2
    torch.testing.assert_close((hp-hq)[0],alias,atol=2e-14,rtol=2e-14)
    if masked=='forced':assert hp.item()==0 and hq.item()==0 and a.abs().max()==0


def test_exact_uniform_preimage_entropy_is_not_uniform_over21matchings():
    u,m=d.preimage_matrix(d.helpers());hp,hq,mass=d.entropies(torch.full((1,2,5),-np.log(5),dtype=torch.float64),u,m)
    assert hp.item()==pytest.approx(np.log(5));assert (hp-hq).item()==pytest.approx(4*np.log(2)/25)
    assert not np.allclose(mass.numpy(),1/21)


def test_frozen_engineering_collect_returns_and_actor_block_gradient_identity(tmp_path):
    torch.set_num_threads(1);h=d.helpers();c=d.efficiency.settings('engineering');t=c['train_seeds'][0];model=d.x.make_model(d.ARM,c,t);before=d.x.learner.parameter_digest(model)
    # Engineering fixture repeats the single registered ENG physical seed; never320panel.
    batch,actions,logs,value,reward,episodes=d.collect(model,[310700001]*4,torch.Generator().manual_seed(310700002),tmp_path,h)
    stats,vectors,arrays=d.gradients(model,batch,actions,reward,value,h)
    np.testing.assert_allclose(vectors['reward_blocks'].mean(0),vectors['reward'],atol=2e-6,rtol=2e-5);assert all(len(v)==11141 for k,v in vectors.items() if k!='reward_blocks')
    done=torch.zeros((48,4),dtype=torch.bool);done[-1]=True;g=d.x.learner.mc_returns(torch.tensor(reward,dtype=torch.float32),done);a=(g-torch.tensor(value)).reshape(-1);a=(a-a.mean())/(a.std(unbiased=False)+1e-8);np.testing.assert_array_equal(arrays['advantage'].ravel(),a.numpy())
    with torch.no_grad():current,entropy,current_value=model.evaluate(batch,actions)
    torch.testing.assert_close(current,logs,atol=2e-5,rtol=0);np.testing.assert_allclose(arrays['hp'],entropy.numpy(),atol=2e-6)
    assert len(episodes)==4 and before==d.x.learner.parameter_digest(model) and all(p.grad is None for p in model.parameters())


def test_local_projection_sign_degeneracy_and_locked_early_screen():
    m=d.alignment(np.array([-.01,0]),np.array([1.,0]),np.array([-1.,0]));assert m['entropy_gradient_cosine']==-1 and m['replacement_norm_ratio']==pytest.approx(2) and m['projected_replacement']==pytest.approx(2)
    records=[dict(train_seed=t,checkpoint_steps=step,relative_alias_gap=.2,**m) for t in d.TRAINS for step in d.STEPS];assert d.screen(records)['entropy_remedy_further_design_motivated'];json.dumps(d.screen(records),allow_nan=False)
    for r in records:
        if r['checkpoint_steps'] in d.EARLY:r['entropy_gradient_cosine']=.9
    assert not d.screen(records)['entropy_remedy_further_design_motivated']
    zero=d.alignment(np.zeros(2),np.zeros(2),np.zeros(2));assert not zero['entropy_gradients_nondegenerate'] and zero['projected_replacement']==0


def test_fresh67roles_without_constructing_physical_diagnostic_states():
    result=d.seed_audit();assert result['raw_unique']==result['derived_unique']==67 and not result['test_states_constructed']
