"""Contracts that separate learned assignment scoring, decoding and PM timing."""
import csv
from dataclasses import replace
import hashlib
import itertools
import json
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import maintenance_assignment_scoring as e
from ht_pdm_fjsp import maintenance_assignment_benchmark as b
from ht_pdm_fjsp import maintenance_assignment_diagnostic as d
from ht_pdm_fjsp import maintenance_assignment_forks as f
from ht_pdm_fjsp.maintenance_assignment_policy import ARMS, DECODERS, STOP, tie_argmax, controller
from ht_pdm_fjsp.maintenance_autoregressive_policy import AutoregressivePolicy, feature_batch
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_matching
from ht_pdm_fjsp.maintenance_heterogeneous_model import arrays, membership
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_assignment_statistics import summarize


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def probe(arm=ARMS[0],condition="dense",h=4,tail=2):
    cfg={**b.settings("smoke"),"horizon":h,"tail_ticks":tail,"hidden":32 if h==48 else 16}
    c=b.make_config(18700020,condition,h+tail)
    s=b.initial_state(c,18700020)
    return cfg,c,s,e.model_for(cfg,18700000,arm)


def test_tie_rule_uses_lowest_near_max_and_rejects_empty():
    x=torch.tensor([[-torch.inf,.3,.300009],[-torch.inf,.3,.30002]])
    assert tie_argmax(x).tolist()==[1,2]
    with pytest.raises(ValueError):tie_argmax(torch.full((1,3),-torch.inf))


def test_scorer_is_parameter_matched_and_baseline_preserves_original_logits():
    cfg,c,s,a=probe();other=e.model_for(cfg,18700000,ARMS[1])
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in other.parameters())
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in a.state_dict().items())
    original=AutoregressivePolicy(hidden=cfg["hidden"]);original.load_state_dict(a.state_dict())
    batch=feature_batch([c],[s],[4],4);mask=batch["pair_mask"];count=torch.zeros(1,dtype=torch.long)
    x,_=a.token_distribution(batch,a.encode(batch),mask,count)
    y,_=original.token_distribution(batch,original.encode(batch),mask,count)
    z,_=other.token_distribution(batch,other.encode(batch),mask,count)
    torch.testing.assert_close(x.logits,y.logits)
    assert not torch.equal(x.logits,z.logits)
    # Raw STOP score remains unchanged at identical encoder state/context.
    torch.testing.assert_close(x.logits[:,STOP]-x.logits[:,0],y.logits[:,STOP]-y.logits[:,0])


@pytest.mark.parametrize("arm",ARMS)
@pytest.mark.parametrize("condition",b.CONDITIONS)
def test_probability_sum_gradient_and_sample_reevaluation(arm,condition):
    _,c,s,m=probe(arm,condition)
    original,info,masses,sequences,_=f.policy_scores(c,s,4,m)
    assert sum(masses.values())==pytest.approx(1,abs=2e-6)
    batch=feature_batch([c]*32,[s]*32,[4]*32,4)
    action,oldprob,oldvalue=m.sample(batch,generator=torch.Generator().manual_seed(31))
    prob,entropy,value=m.evaluate(batch,action)
    torch.testing.assert_close(prob,oldprob);torch.testing.assert_close(value,oldvalue)
    (-prob.mean()-.01*entropy.mean()+value.square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in m.actor.parameters())
    for a in action:validate_matching(c,s,m.decode(c,s,a))


@pytest.mark.parametrize("arm",ARMS)
@pytest.mark.parametrize("failed",[(False,)*4,(True,False,False,False)])
def test_conditional_decoder_keeps_machines_and_cm_and_maximizes_mass(arm,failed):
    _,c,s,m=probe(arm)
    s=replace(s,failed=failed)
    batch=feature_batch([c],[s],[4],4)
    greedy=torch.tensor([[0,3,STOP]])
    selected,audits=m.conditional_decode(batch,greedy)
    action=m.decode(c,s,selected[0]);original=m.decode(c,s,greedy[0]);audit=audits[0]
    assert {i for i,j in action}=={i for i,j in original}
    assert {(i,j) for i,j in action if s.failed[i]}=={(i,j) for i,j in original if s.failed[i]}
    assert audit["masses"][audit["candidates"].index(action)]>=max(audit["masses"])-1e-6
    _,_,masses,_,_=f.policy_scores(c,s,4,m)
    for candidate,prob in zip(audit["candidates"],audit["masses"],strict=True):
        assert prob==pytest.approx(masses[candidate],abs=2e-6)


def test_decoder_sums_both_orders_and_original_wins_ties(monkeypatch):
    _,c,s,m=probe();batch=feature_batch([c],[s],[4],4)
    greedy=torch.tensor([[0,3,STOP]])
    def scores(batch,action):
        weights={(0,3):.3,(3,0):.05,(1,2):.2,(2,1):.2}
        return torch.tensor([np.log(weights[tuple(a[:2].tolist())]) for a in action]),torch.zeros(len(action)),torch.zeros(len(action))
    monkeypatch.setattr(m,"evaluate",scores)
    selected,audit=m.conditional_decode(batch,greedy)
    assert m.decode(c,s,selected[0])==((0,1),(1,0))
    monkeypatch.setattr(m,"evaluate",lambda batch,action:(torch.full((len(action),),np.log(.2)),torch.zeros(len(action)),torch.zeros(len(action))))
    selected,_=m.conditional_decode(batch,greedy)
    assert torch.equal(selected,greedy)


@pytest.mark.parametrize("arm",ARMS)
def test_empty_pm_decoder_and_stochastic_training_independent_of_decoder(arm):
    _,c,s,m=probe(arm);batch=feature_batch([c],[s],[4],4)
    empty=torch.tensor([[STOP,-1,-1]])
    result,_=m.conditional_decode(batch,empty);assert torch.equal(result,empty)
    m.decoder="greedy";x=m.sample(batch,generator=torch.Generator().manual_seed(4))
    m.decoder="assignment_mass";y=m.sample(batch,generator=torch.Generator().manual_seed(4))
    assert all(torch.equal(a,b) for a,b in zip(x,y,strict=True))


@pytest.mark.parametrize("arm",ARMS)
@pytest.mark.parametrize("tick",[0,42,47])
def test_h48_clock_fork_scalar_vector_replay(arm,tick):
    _,c,s,m=probe(arm,h=48,tail=12)
    action=m.select(c,s,48-tick,48)[0]
    roots,groups,_=f.candidates(c,s,((0,0),(1,1)))
    assert len(roots)==7 and len(groups["assignment"])==2 and len(groups["timing"])==4
    result=f.fork_roots(c,s,roots,m,117,2,tick,record_trace=True)
    assert result["transitions"]==len(roots)*2*(60-tick)
    assert result["counts"]["decision_rows"]==len(roots)*2*(47-tick)
    assert result["counts"]["tail_rows"]==len(roots)*2*12
    assert f.scalar_replay(c,s,result,m,tick)
    batch=f.ar_batch_features(c,arrays(s,3),48-tick,48)
    scalar=feature_batch([c]*3,[s]*3,[48-tick]*3,48)
    for key in batch:torch.testing.assert_close(batch[key],scalar[key],rtol=0,atol=0)
    selected,_=f.rl_batch(c,arrays(s,3),48-tick,m)
    assert np.array_equal(selected,np.repeat(membership(4,2)[CATALOGUE.index(action)][None].astype(bool),3,axis=0))


def test_seed_registry_physical_exclusion_and_forecast_pairing():
    cfg=b.settings("smoke");audit,_=b.seed_audit(cfg)
    assert audit["status"]=="PASS" and audit["recent_historical_derived_roles_checked"]>10000
    panels,_,training=b.metadata();blocked=b.historical_pool();seen=set(blocked)
    for values in panels.values():
        for c in values.values():
            sig=b.old.boundary.signature(c);assert sig not in seen;seen.add(sig)
    for values in training.values():
        for c,_,_,_,_ in values.values():
            sig=b.old.boundary.signature(c);assert sig not in seen;seen.add(sig)
    identity=dict(condition="dense",cohort=18700040,shock=18700050,model_seed=18700000,tick=0)
    assert d.forecast_seed(cfg,identity,"selection")!=d.forecast_seed(cfg,identity,"evaluation")
    assert f.choose(np.array([[1.,1.],[1.,1.]]),(0,1))==0


def test_checkpoint_roundtrip_and_reject_contract_tampering(tmp_path):
    cfg,c,s,m=probe(ARMS[1]);path=tmp_path/"model.pt"
    restored,_=e.save_checkpoint(path,m,cfg,18700000,1,32,(c,s))
    assert restored.experiment_arm==ARMS[1] and restored.decoder=="greedy"
    payload=torch.load(path,weights_only=True);payload["protocol"]="wrong";torch.save(payload,path)
    with pytest.raises(ValueError):e.load_checkpoint(path)


def test_smoke_artifact_grid_likelihood_and_independent_diagnostics(tmp_path):
    out=tmp_path/"smoke";manifest=e.run(out)
    assert manifest["status"]=="COMPLETED" and manifest["audit"]=="PASS"
    assert manifest["actual_counts"]==e.counts(b.settings("smoke"))
    assert not manifest["full_test_outcomes_opened"]
    audit=manifest["diagnostic_audit"]
    assert audit["common_states"]==8 and audit["model_states"]==16 and audit["parameters_unchanged"]
    coverage=json.loads((out/"training_coverage.json").read_text())
    assert len(coverage)==2 and len({r["initial_parameters_sha256"] for r in coverage})==1
    assert all(r["parameters_changed"] and r["teacher_labels"]==0 and r["probability_audit"] for r in coverage)
    summary=json.loads((out/"summary.json").read_text());assert len(summary["primary_contrasts"])==4
    assert all(r["status"]=="ENGINEERING_ONLY" for r in summary["primary_contrasts"].values())
    with (out/"diagnostic_states.csv").open() as stream:states=list(csv.DictReader(stream))
    tapes={}
    for row in states:
        key=tuple(row[k] for k in ("condition","cohort","shock","model_seed","tick"))
        values=(row["selection_seed"],row["evaluation_seed"])
        assert values[0]!=values[1]
        assert key not in tapes or tapes[key]==values
        tapes[key]=values
    for relative,record in manifest["artifacts"].items():
        path=out/relative
        assert path.stat().st_size==record["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest()==record["sha256"]
    with (out/"episodes.csv").open() as stream:rows=list(csv.DictReader(stream))
    with pytest.raises(RuntimeError):summarize(rows[:-1],b.settings("smoke"))
    with pytest.raises(ValueError):e.run(out)
