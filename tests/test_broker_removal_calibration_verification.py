import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_broker_removal_coefficient_calibration as c

spec=importlib.util.spec_from_file_location('calibration_auditor',Path(__file__).resolve().parents[1]/'scripts/verify_broker_removal_calibration.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def test_independent_fit_only_formula_retains_harmful_heldout_coefficient():
    x=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]);h=x*np.array([.5,0]);r=v.fit_and_evaluate(x,x-h,x,x-3*x);assert r['coefficient']==pytest.approx(2.) and r['gain']==pytest.approx(-24.)
    r=v.fit_and_evaluate(x,x,x,x);assert r['coefficient']==0 and r['gain']==0


def test_complete_new_namespace_dense_audit_with_variable_episode_count(tmp_path):
    torch.set_num_threads(1);cfg=c.e.settings('engineering');model=c.e.make_model(c.e.CORE[0],cfg,cfg['train_seeds'][0]).double();record=c.d.collect(model,[350700000]*2,350700001,tmp_path/'case',lambda:None,c.VERSION);h=c.d.reference()[1];r=c.d.reference()[0];g=v.module('varsize_gradient','verify_broker_removal_gradient_diagnostic.py');manual=v.module('actor_manual','verify_broker_entropy_alignment.py').manual_logp;weights={k:p.detach().numpy().copy() for k,p in model.state_dict().items()};gradients,audit=v.verify_case(tmp_path/'case',[350700000]*2,350700001,weights,record,h,r,g,manual,lambda:None);assert audit['physical_transitions']==120 and audit['gradient_vectors']==6 and gradients['state'].shape==(2,11141)
    assert max(audit['gradient_max_errors'].values())<1e-9
