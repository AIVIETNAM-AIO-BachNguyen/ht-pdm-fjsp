"""Safety and intervention checks for the B3 DEV-only diagnosis."""
import json

import pytest
import torch

from scripts import diagnose_prefix_capacity_v3 as d
from ht_pdm_fjsp import technician_prefix_capacity_v3_experiment as x
from ht_pdm_fjsp import technician_prefix_capacity_v3_env as e
from ht_pdm_fjsp.maintenance_balanced_algorithms import parameter_digest,tensors
from ht_pdm_fjsp.maintenance_dispatch import DispatchState


def fixture(tmp_path,*,test_opened=False,audit_hash=None):
    source=tmp_path/'source';source.mkdir()
    (source/'manifest.json').write_text(json.dumps(dict(protocol='technician_prefix_capacity_v3',
        status='COMPLETED',audit_status='PASS',profile='development',test_opened=test_opened)))
    audit=tmp_path/'audit';audit.mkdir()
    (audit/'audit_prefix_capacity_development.py').write_text('verifier')
    (audit/'audit.json').write_text(json.dumps(dict(status='PASS',verdict='DEV_GATE_FAIL',test_opened=False,
        source_artifact=str(source.resolve()),source_manifest_sha256=audit_hash or d.sha(source/'manifest.json'),
        verifier_sha256=d.sha(audit/'audit_prefix_capacity_development.py'))))
    return source,audit,tmp_path/'output'


def test_rejects_test_opened_source(tmp_path):
    source,audit,out=fixture(tmp_path,test_opened=True)
    with pytest.raises(AssertionError):d.run(source,audit,out,'smoke')
    assert json.loads((out/'manifest.json').read_text())['status']=='FAILED'


def test_rejects_independent_audit_hash_mismatch(tmp_path):
    source,audit,out=fixture(tmp_path,audit_hash='wrong')
    with pytest.raises(AssertionError):d.run(source,audit,out,'smoke')
    assert json.loads((out/'manifest.json').read_text())['status']=='FAILED'


def test_capacity_only_suppression_preserves_weights_and_relational_message():
    torch.set_num_threads(1)
    cfg=x.settings('smoke')
    full=x.make_model('reservation_ppo',4,cfg,67800000)
    cap=d.capacity_off(full)
    all_off=x.make_model('mask_only_ppo',4,cfg,67800000)
    all_off.load_state_dict(full.state_dict())
    assert parameter_digest(full)==parameter_digest(cap)==parameter_digest(all_off)
    env=e.ResourceEnv(67900000,4,'reservation_ppo');env.reset()
    env.env.state=DispatchState((3,)*4,(False,)*4,(0,)*4,(0,)*2,(-1,)*2)
    batch=tensors([env.observe()]);chosen=torch.tensor([[1,2]])
    traces=[[],[],[]]
    for model,trace in zip((full,cap,all_off),traces,strict=True):
        model.trace(batch,recorded=chosen,audit=trace)
    assert cap.capacity_suppressed_inputs>0
    assert torch.count_nonzero(traces[1][1]['prefix'])
    assert torch.count_nonzero(traces[1][1]['candidate_capacity'])
    assert not torch.count_nonzero(traces[2][1]['prefix'])
    assert not torch.count_nonzero(traces[2][1]['candidate_capacity'])
    for steps in zip(*traces,strict=True):
        assert torch.equal(steps[0]['legal'],steps[1]['legal'])
        assert torch.equal(steps[0]['legal'],steps[2]['legal'])
