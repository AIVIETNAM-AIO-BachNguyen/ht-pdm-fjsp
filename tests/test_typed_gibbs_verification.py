import importlib.util,itertools,sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import typed_gibbs_reference as reference
import verify_typed_gibbs_development as verifier
from ht_pdm_fjsp import technician_typed_gibbs as g
from ht_pdm_fjsp import technician_typed_gibbs_experiment as e


@pytest.mark.parametrize('arm',reference.CORE)
def test_independent_all_physical_densities_entropy_map_all256_eligibility_graphs(arm):
    torch.set_num_threads(1);h=verifier.helpers();canonical=reference.tokens(h);legal=np.array(list(itertools.product((False,True),repeat=8))).reshape(256,4,2);u=np.broadcast_to(np.array([[1.2,-.8],[-.7,1.8],[.4,.3],[.2,-.2]]),(256,4,2)).copy();result=reference.distribution(u,legal,arm,canonical);valid=result['eligible'];bi,ai=np.nonzero(valid);reverse=arm==reference.CORE[2];order=np.arange(3,-1,-1) if reverse else np.arange(4)
    table=g.policy_tables(torch.tensor(u[bi][:,order]),(1,1),torch.tensor(legal[bi][:,order]),mode='gibbs' if arm==reference.CORE[0] else 'myopic');lp=g.matching_log_probability(table,torch.tensor([0,1]),torch.ones((len(bi),2),dtype=torch.bool),torch.tensor(canonical[ai][:,order]));np.testing.assert_allclose(lp.numpy(),result['logp'][bi,ai],atol=1e-10,rtol=0);root=g.root_values(table,torch.tensor([0,1]),torch.ones((len(bi),2),dtype=torch.bool));np.testing.assert_allclose(root['entropy'].numpy(),result['entropy'][bi],atol=1e-10,rtol=0);np.testing.assert_allclose(root['max_log_probability'].numpy(),result['logp'].max(-1)[bi],atol=1e-10,rtol=0)


@pytest.mark.parametrize('arm',e.ARMS)
def test_independent_fresh_constructor_tensors_for_all7(arm):
    torch.set_num_threads(1);c=e.settings('engineering');t=c['train_seeds'][0];actual=e.make_model(arm,c,t).state_dict();expected=verifier.independent_initial(arm,t,verifier.helpers());assert list(actual)==list(expected);assert all(torch.equal(actual[k],expected[k]) for k in actual)


def test_numpy_features_and_nonzero_neural_outputs_and_logged_density():
    torch.set_num_threads(1);h=verifier.helpers();o=h.observation(dict(ages=[3]*4,failed=[False]*4,pending_wait=[0]*4,remaining=[0]*2,assigned=[-1]*2),0);model=e.make_model(e.CORE[0],e.settings('engineering'),380700000).double();b={k:v.double() if v.is_floating_point() else v for k,v in e.learner.tensors([o]).items()};actual=model.inputs(b)[0].detach().numpy();np.testing.assert_array_equal(actual,reference.features([o]))
    with torch.no_grad():model.actor[-1].weight.copy_(torch.arange(128).reshape(2,64)/300-.2)
    weights={k:v.detach().numpy() for k,v in model.state_dict().items()};u=h.mlp(reference.features([o]).reshape(-1,89),weights,'actor').reshape(1,4,2);np.testing.assert_allclose(u,model.actor(model.inputs(b)[0]).detach().numpy(),atol=1e-10,rtol=0)
    result=reference.distribution(u,o['pair_mask'][None],e.CORE[0],reference.tokens(h));action=model.trace(b,deterministic=True)[0];log,entropy,_=model.evaluate(b,action);index=(reference.tokens(h)==action[0].numpy()).all(-1).argmax();np.testing.assert_allclose(log.detach().numpy(),result['logp'][:,index],atol=1e-10,rtol=0);np.testing.assert_allclose(entropy.detach().numpy()*4,result['entropy'],atol=1e-10,rtol=0)


def test_cold_parity_and_nonzero_knockout_no_tie_inflation():
    h=verifier.helpers();canonical=reference.tokens(h);legal=np.ones((1,4,2),bool);zero=np.zeros((1,4,2));uniform=[reference.distribution(zero,legal,a,canonical) for a in e.CORE]
    for r in uniform:np.testing.assert_allclose(r['logp'],-np.log(21),atol=1e-10,rtol=0)
    action=canonical[[0]];m=reference.mechanism(zero,legal,action,canonical);assert m['candidate_action_outside_myopic_map']==0 and m['candidate_action_outside_reverse_map']==0 and m['mean_conditional_tv']<1e-10 and m['mean_future_message_span']<1e-10
    u=np.array([[[1.2,-.8],[-.7,1.8],[.4,.3],[.2,-.2]]]);r=reference.distribution(u,legal,e.CORE[0],canonical);ix=r['logp'].argmax(-1);m=reference.mechanism(u,legal,canonical[ix],canonical);assert m['mean_conditional_tv']>0 and m['max_future_message_span']>0


def test_fixed_cardinalities_and_independent_endpoint_math():
    assert len(e.ARMS)*len(verifier.TRAINS)==21;assert 21*491520==10321920;assert 21*10240==215040;assert 21*7680==161280;assert 21*5*32==3360;assert (21*32+32)*60==42240
    means={a:dict(normalized_cost=99. if a==e.CORE[0] else 100.,availability=.67,normalized_pending=.01,normalized_failures=.1) for a in (*e.ARMS,'deadline')};points,checks=verifier.endpoint_gates(means,verifier.helpers());assert len(checks)==12 and all(checks.values())


def test_independent_raw_ranges_and_prior_capacity_derived_reconstruction(tmp_path):
    # No physical environment, checkpoints, DEV outcomes or RNG sampling.
    import json
    c=e.seed_audit();(tmp_path/'seed_registry.json').write_text(e.REGISTRY.read_text());(tmp_path/'seed_audit.json').write_text(json.dumps(c));r=verifier.seed_contract(tmp_path,verifier.helpers());assert r['raw']==365 and r['derived']==82397 and r['independent_capacity_order_derived']==82424
