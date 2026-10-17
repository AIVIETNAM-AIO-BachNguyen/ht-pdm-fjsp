import json,time
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_broker_removal_coefficient_calibration as c


def test_fit_coefficient_frozen_on_evaluation_instead_of_oracle_refit():
    x=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]);h=x*np.array([.5,0]);fit=c.coefficient(x,x-h);assert fit['coefficient']==pytest.approx(2.)
    heldout=c.evaluate_coefficient(x,x-3*x,fit['coefficient']);assert heldout['gain']==pytest.approx(-24.)
    assert c.coefficient(x,x-3*x)['coefficient']==pytest.approx(1/3)


def test_initial_invariance_and_all15_nine_gates():
    x=np.array([[1.,2.],[3.,4.],[5.,6.]]);fit=c.coefficient(x,x);assert fit['coefficient']==0
    assert c.evaluate_coefficient(x,x,0)['gain']==0
    records=[dict(train_seed=t,steps=s,evaluation=dict(gain=.08 if s else 0,fixed_one_gain=.04 if s else 0)) for t in c.d.TRAINS for s in c.d.STEPS];_,checks=c.primary(records);assert sum(len(v) for v in checks.values())==9 and all(ok for row in checks.values() for ok in row.values())
    records[1]['evaluation']['gain']=-.01;_,checks=c.primary(records);assert not checks[str(c.d.TRAINS[0])]['material_mean_gain']
    with pytest.raises(AssertionError):c.primary(records[:-1])


def test_new_namespace_and_complete_coefficient_seal_fit_evaluation_roundtrip(tmp_path):
    torch.set_num_threads(1);cfg=c.e.settings('engineering');t=cfg['train_seeds'][0];m=c.e.make_model(c.e.CORE[0],cfg,t);source=tmp_path/'source';out=tmp_path/'out';out.mkdir();p=source/f'N4/{c.e.CORE[0]}/{t}/checkpoints/step_000000000.pt';p.parent.mkdir(parents=True);probe=c.d.DiagnosticEnv(350700000,4,c.e.CORE[0],physical_version=c.VERSION).reset();c.e.save_model(m,p,c.e.CORE[0],cfg,t,0,probe);registry=dict(action_bases=[350700001],fit_physical_seeds=[350700000]*2,evaluation_physical_seeds=[350700000]*2);r=c.node((str(source),str(out),t,0,0,registry,time.monotonic()));assert r['coefficient']['coefficient']==0 and r['evaluation']['gain']==0 and r['optimizer_updates']==0
    assert r['fit']['physical_transitions']==120 and r['evaluation_audit']['physical_transitions']==120;coefficient=out/f'{t}_0/coefficient.json';assert r['coefficient_sha256']==c.e.sha(coefficient) and json.loads(coefficient.read_text())['phase']=='sealed_before_evaluation_collection'


def test_disjoint_fit_evaluation_registered_seed_streams():
    registry=c.e.read(c.REGISTRY);assert not set(registry['fit_physical_seeds'])&set(registry['evaluation_physical_seeds']);audit=c.seed_audit(registry,c.d.reference()[1]);assert audit['raw']==402 and audit['derived']==417 and not audit['confirmation_states_constructed']
