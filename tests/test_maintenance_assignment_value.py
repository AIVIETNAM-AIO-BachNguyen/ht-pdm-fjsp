"""Value-target sign, isolation, entity symmetry, clock, source, and cluster contracts."""
from dataclasses import replace
import copy
import json
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import maintenance_assignment_value_benchmark as b
from ht_pdm_fjsp import maintenance_assignment_value_forks as f
from ht_pdm_fjsp import maintenance_assignment_value_training as t
from ht_pdm_fjsp import maintenance_assignment_value_statistics as stats
from ht_pdm_fjsp.maintenance_assignment_value_data import all_assignment_classes
from ht_pdm_fjsp.maintenance_assignment_value_policy import RelativeValueHead,ValueController,inputs,allowed_mask
from ht_pdm_fjsp.maintenance_assignment_value import counts,run
from ht_pdm_fjsp.maintenance_assignment_policy import AssignmentPolicy
from ht_pdm_fjsp.maintenance_autoregressive_policy import feature_batch,STOP
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_dispatch import DispatchState,validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def probe():
    c=WaitingConfig(4,2,60,4,.4,((2,5),(3,2),(5,4),(2,3)),((.5,1.),(.75,.5),(.25,1.),(1.,.25)),waiting_price=12.)
    s=DispatchState((1,2,3,0),(False,)*4,(0,)*4,(0,)*2,(-1,)*2)
    torch.manual_seed(5);m=AssignmentPolicy();m.requires_grad_(False)
    return c,s,m


@pytest.mark.parametrize("failed",[(False,)*4,(True,False,False,False),(True,True,False,False)])
def test_complete_classes_hold_exact_cm_and_do_not_add_corrective_edges(failed):
    c,s,_=probe();s=replace(s,failed=failed);original=((0,0),)
    roots,groups=all_assignment_classes(c,s,original)
    expected_cm=tuple(e for e in original if s.failed[e[0]])
    for machines,group in groups.items():
        assert 1<=len(group)<=2
        for i in group:
            validate_matching(c,s,roots[i]);assert tuple(e for e in roots[i] if s.failed[e[0]])==expected_cm
            assert tuple(m for m,j in roots[i] if not s.failed[m])==machines
    assert roots[0]==original
    if not any(failed):assert len(roots)==21 and sum(len(g)==2 for g in groups.values())==10


def test_value_controller_minimizes_cost_with_fixed_machines_and_exact_cm(monkeypatch):
    c,s,base=probe();batch=feature_batch([c],[s],[48],48)
    monkeypatch.setattr(base,"greedy_tokens",lambda data:torch.tensor([[0,3,STOP]]).repeat(len(data["machines"]),1))
    class Scores:
        def eval(self):return self
        def __call__(self,data):
            values=torch.zeros(len(data["context"]),21);values[:,CATALOGUE.index(((0,1),(1,0)))]=-1
            values[:,0]=-100 # Globally cheap WAIT cannot affect PM timing.
            return values
    ctrl=ValueController(base,Scores());action=ctrl.select(c,s,48,48)[0]
    assert action==((0,1),(1,0));assert ctrl.last_decode_audit[0]["changed"]
    s=replace(s,failed=(True,False,False,False))
    assert ctrl.select(c,s,48,48)[0]==((0,0),(1,1))
    with pytest.raises(ValueError):ctrl.sample(batch,deterministic=False)


def test_tied_value_prefers_original_and_wait_has_no_assignment_alternative(monkeypatch):
    c,s,base=probe();head=RelativeValueHead();head.requires_grad_(False)
    for p in head.parameters():p.zero_()
    ctrl=ValueController(base,head)
    monkeypatch.setattr(base,"greedy_tokens",lambda data:torch.tensor([[1,2,STOP]]).repeat(len(data["machines"]),1))
    assert ctrl.select(c,s,48,48)[0]==((0,1),(1,0))
    mask=allowed_mask(feature_batch([c],[s],[48],48),torch.tensor([0]))
    assert mask.sum()==1 and mask[0,0]


def test_head_entity_permutation_equivariance_and_detached_source():
    c,s,base=probe();head=RelativeValueHead();raw=feature_batch([c],[s],[48],48)
    data=inputs(raw,base);a=head(data)[0]
    mp=[2,0,3,1];tp=[1,0]
    perm={**raw,"machines":raw["machines"][:,mp],"technicians":raw["technicians"][:,tp],
        "edges":raw["edges"][:,mp][:,:,tp],"pair_mask":raw["pair_mask"][:,mp][:,:,tp]}
    other=head(inputs(perm,base))[0]
    for i,action in enumerate(CATALOGUE):
        reindexed=tuple(sorted((mp.index(m),tp.index(j)) for m,j in action))
        torch.testing.assert_close(a[i],other[CATALOGUE.index(reindexed)],rtol=1e-5,atol=1e-6)
    (a[2]-a[1]).backward();assert all(p.grad is None for p in base.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in head.parameters())


def test_pair_regression_learns_continuation_cost_sign_not_probability():
    c,s,base=probe();head=RelativeValueHead();data=inputs(feature_batch([c],[s],[48],48),base)
    left=CATALOGUE.index(((0,0),));right=CATALOGUE.index(((0,1),))
    data.update(pairs=torch.tensor([[0,left,right]]),targets=torch.tensor([1.]),conditions=torch.tensor([0]))
    optimizer=torch.optim.Adam(head.parameters(),lr=.01)
    before=t.mse(head,data)
    for _ in range(50):
        loss=torch.nn.functional.smooth_l1_loss(t.predictions(head,data),data["targets"])
        optimizer.zero_grad();loss.backward();optimizer.step()
    assert t.mse(head,data)<before*.1 and t.predictions(head,data)[0]>.8


def test_shuffle_is_condition_preserving_and_bijective():
    conditions=torch.tensor([0]*17+[1]*13);perm=t.shuffled_permutation(conditions,20100000)
    assert sorted(perm.tolist())==list(range(30));assert torch.equal(conditions,conditions[perm])
    assert not torch.equal(perm,torch.arange(30));assert torch.equal(perm,t.shuffled_permutation(conditions,20100000))


@pytest.mark.parametrize("tick",[0,42,47])
def test_all_twenty_one_roots_scalar_replay_at_original_clock(tick):
    c,s,base=probe();roots,groups=all_assignment_classes(c,s,())
    assert len(roots)==21
    result=f.fork_roots(c,s,roots,base,37,2,tick,record_trace=True)
    assert result["transitions"]==21*2*(60-tick)
    assert result["counts"]["decision_rows"]==21*2*(47-tick)
    assert result["counts"]["tail_rows"]==21*2*12
    assert f.scalar_replay(c,s,result,base,tick)


def test_value_head_is_used_in_continuation_not_just_at_root(monkeypatch):
    c,s,base=probe();head=RelativeValueHead();ctrl=ValueController(base,head)
    calls=[];original=ctrl.sample
    def record(batch,**kwargs):calls.append(len(batch["machines"]));return original(batch,**kwargs)
    monkeypatch.setattr(ctrl,"sample",record)
    result=f.fork_roots(c,s,((),),ctrl,42,2,42,record_trace=True)
    assert len(calls)==5 and result["counts"]["decision_rows"]==10
    assert f.scalar_replay(c,s,result,ctrl,42)


def test_protocol_budgets_and_horizon():
    cfg=b.settings("full");budget=counts(cfg)
    assert budget["train_states"]==1536 and budget["development_states"]==192
    assert budget["test_episodes"]==1728 and budget["ranking_states"]==288 and budget["timing_common_states"]==120
    assert cfg["epochs"]==80 and cfg["checkpoints"]==[20,40,60,80]
    smoke=b.settings("smoke");assert smoke["horizon"]==48 and smoke["tail_ticks"]==12 and smoke["hidden"]==32
    with pytest.raises(ValueError):b.settings("unknown")


def test_metadata_disjoint_from_historical_and_future_phase_is_independent():
    panels,registry,audit=b.metadata();assert audit["status"]=="PASS" and audit["outcomes_used"] is False
    signatures=[r["physical_signature"] for rows in registry.values() for r in rows]
    assert len(signatures)==len(set(signatures))==198
    assert b.seed_audit(b.settings("full"))["status"]=="PASS"
    identity=dict(condition="dense",cohort=20600000,shock=20700000,model_seed=18100000,tick=0)
    cfg=b.settings("full");phases=("train","development","ranking_selection","ranking_evaluation","timing_selection","timing_evaluation")
    assert len({b.future_seed(cfg,identity,p) for p in phases})==6


def test_primary_contrasts_sign_and_complete_grid():
    cfg=b.settings("smoke");metrics=("extended_objective","base_objective","tail_objective","extended_failure_cost","extended_waiting_cost","extended_maintenance_cost","extended_unavailability_cost","preventive_starts","corrective_starts","wait_with_pm_feasible","stop_with_pm_extension_feasible","pm_assignment_alternative_ticks")
    rows=[]
    for cond in cfg["conditions"]:
        for name,cost in (("frozen_ppo",100.),("shuffled_value",110.),("true_value",90.)):
            rows.append(dict(condition=cond,cohort=cfg["test_cohorts"][0],shock=cfg["test_shocks"][0],controller=name,train_seed=cfg["model_seeds"][0],horizon=48,**{k:cost for k in metrics}))
    result,cells=stats.summarize(rows,cfg);assert len(result["primary_contrasts"])==4
    assert all(v["relative_gain"]>0 and v["status"]=="ENGINEERING_ONLY" for v in result["primary_contrasts"].values())
    with pytest.raises(RuntimeError):stats.summarize(rows[:-1],cfg)
    with pytest.raises(RuntimeError):stats.summarize(rows+[rows[0]],cfg)


def test_runner_refuses_overwrite_and_full_dirty_preflight(tmp_path):
    nonempty=tmp_path/"old";nonempty.mkdir();(nonempty/"keep").write_text("unchanged")
    with pytest.raises(ValueError):run(nonempty,tmp_path/"missing")
    assert (nonempty/"keep").read_text()=="unchanged"


def test_source_preflight_rejects_tampered_artifact_before_models(tmp_path):
    import hashlib
    from ht_pdm_fjsp.maintenance_assignment_value_source import verify_source
    from ht_pdm_fjsp import maintenance_assignment_benchmark as previous
    artifact=tmp_path/"cost.csv";artifact.write_text("tampered")
    manifest=dict(status="COMPLETED",audit="PASS",dirty=False,commit="pinned",settings=previous.settings("full"),
        artifacts={"cost.csv":dict(sha256=hashlib.sha256(b"original").hexdigest(),bytes=8)})
    p=tmp_path/"manifest.json";p.write_text(json.dumps(manifest))
    contract=dict(protocol=b.VERSION,source_protocol=previous.VERSION,source_commit="pinned",files={"manifest.json":hashlib.sha256(p.read_bytes()).hexdigest()})
    with pytest.raises(ValueError,match="source artifact mismatch"):
        verify_source(tmp_path,b.settings("smoke"),contract)


def test_empty_cluster_diagnostic_has_no_invented_confidence_interval():
    cfg=b.settings("smoke")
    result,_=stats.conditional([],cfg,"dense","true_value_gain",cfg["test_cohorts"])
    assert result["estimate"] is None and result["confidence_interval"] is None and result["empty_bootstrap_fraction"]==1
