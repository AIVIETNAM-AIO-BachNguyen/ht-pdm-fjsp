import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_broker_removal_gradient_diagnostic as d

spec=importlib.util.spec_from_file_location('gradient_auditor',Path(__file__).resolve().parents[1]/'scripts/verify_broker_removal_gradient_diagnostic.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


@pytest.mark.parametrize('context',[False,True])
def test_independent_dense_other_detachment_gradients_vs_closed_vjp(context):
    torch.set_num_threads(1);c=d.e.settings('engineering');m=d.e.make_model(d.e.CORE[0],c,c['train_seeds'][0]).double();env=d.DiagnosticEnv(340700000,4,d.e.CORE[0]);obs=env.reset();observations=[];actions=[];rng=torch.Generator().manual_seed(340700001)
    if context:
        with torch.no_grad():m.removal[0].weight[:,71:79].fill_(.2)
    for _ in range(8):
        a,_,_=m.sample(d.double_batch([obs]),rng);observations.append(obs);actions.append(a[0].numpy());obs,*_=env.step(a[0].numpy())
    actions=np.array(actions);h=d.reference()[1];r=d.reference()[0];w={k:p.detach().numpy().copy() for k,p in m.state_dict().items()};dense=r.dense_baselines(observations,actions,w,h);value=h.mlp(np.stack([o['state'] for o in observations]).astype(float),w,'critic').ravel()*115.2;returns=np.arange(8,dtype=float)-12;manual=v.module('independent_manual','verify_broker_entropy_alignment.py').manual_logp;actual=v.dense_gradients(observations,actions,returns,value,dense['actual'],dense['mixed'],w,h,manual)
    for kind in d.KINDS:
        baseline=value[:,None] if kind=='state' else dense['mixed'] if kind=='conditional' else dense['actual'];adv=np.broadcast_to(returns[:,None]-baseline,(8,2));expected=d.numpy_vjp(observations,actions,adv,w,h,kind=='proposal');np.testing.assert_allclose(actual[kind],expected,atol=1e-11,rtol=1e-9)
    if not context:np.testing.assert_allclose(actual['state'],actual['conditional'],atol=1e-12,rtol=1e-12)
    else:assert np.linalg.norm(actual['state']-actual['conditional'])>1e-4


def test_reference_tolerances_reject_nonfinite_and_wrong_gradient():
    with pytest.raises(AssertionError):v.close([1.],[1.0001],'bad',True)
    with pytest.raises(AssertionError):v.close([np.nan],[0.],'nonfinite')
