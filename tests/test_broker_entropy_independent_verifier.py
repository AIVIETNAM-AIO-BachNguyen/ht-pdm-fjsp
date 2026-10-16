import importlib.util
from pathlib import Path
import numpy as np
import torch
from ht_pdm_fjsp import technician_broker_entropy_alignment_diagnostic as d

spec=importlib.util.spec_from_file_location('entropy_reference',Path(__file__).resolve().parents[1]/'scripts/verify_broker_entropy_alignment.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def model_fixture():
    torch.set_num_threads(1);c=d.efficiency.settings('engineering');model=d.x.make_model(d.ARM,c,c['train_seeds'][0]);w={k:p.detach().numpy().astype(float) for k,p in model.state_dict().items()};return model,w


def test_manual_float32_actor_and_sampler_match_actual_source_on_engineering_state():
    model,w=model_fixture();o=d.x.ResourceEnv(310700001,4,d.ARM).reset();batch=d.x.learner.tensors([o]*64);obs={k:value.numpy() for k,value in batch.items()};logp,_=v.manual_logp(obs,w,torch.float32);actual=model.proposal_log_probs(batch)[0];torch.testing.assert_close(logp,actual,atol=0,rtol=0)
    a=model.sample(batch,torch.Generator().manual_seed(310700002))[0];rng=torch.Generator().manual_seed(310700002);expected=torch.stack([torch.multinomial(logp[:,j].exp(),1,generator=rng).squeeze(-1) for j in range(2)],-1);torch.testing.assert_close(a,expected,atol=0,rtol=0)


def test_closed_mass_manual_float64_full_actor_gradients_match_production(tmp_path):
    model,w=model_fixture();h=d.helpers();batch,actions,logs,value,reward,episodes=d.collect(model,[310700001]*4,torch.Generator().manual_seed(310700002),tmp_path,h);stats,vectors,arrays=d.gradients(model,batch,actions,reward,value,h);obs={k:value.numpy() for k,value in batch.items()};ref,expected=v.reference_gradients(obs,actions.numpy(),arrays['advantage'],w,h)
    for name,gradient in vectors.items():np.testing.assert_allclose(gradient,ref[name],atol=5e-6,rtol=2e-4)
    for name,value in expected.items():np.testing.assert_allclose(value,arrays[name],atol=5e-6,rtol=0)
    for name,value in v.metrics(vectors).items():assert value==stats[name]


def test_independent_primary_screen_uses_only_fixed_early_window():
    metrics=v.metrics(dict(reward=np.array([-.01,0]),proposal=np.array([1.,0]),physical=np.array([-1.,0])));records=[dict(train_seed=t,checkpoint_steps=step,relative_alias_gap=.2,**metrics) for t in d.TRAINS for step in d.STEPS];means,checks=v.primary(records);expected=d.screen(records);assert checks==expected['checks']
    for t,data in means.items():
        for name,value in data.items():v.near(value,expected['primary_replica_metrics'][t][name],name)
    for r in records:
        if r['checkpoint_steps'] in d.EARLY:r['entropy_gradient_cosine']=.99
    assert not all(all(row.values()) for row in v.primary(records)[1].values())
