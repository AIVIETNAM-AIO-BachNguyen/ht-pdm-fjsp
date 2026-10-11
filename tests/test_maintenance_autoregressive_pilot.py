"""Scientific-contract tests for sequential joint timing/assignment PPO."""
import csv
from dataclasses import replace
import hashlib
import itertools
import json
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import maintenance_autoregressive_pilot as e
from ht_pdm_fjsp import maintenance_autoregressive_benchmark as b
from ht_pdm_fjsp.maintenance_autoregressive_policy import (ARMS,CATALOGUE_ARM,CONTEXT,FROZEN,STOP,feature_batch)
from ht_pdm_fjsp.maintenance_autoregressive_training import fit,complete_targets
from ht_pdm_fjsp.maintenance_dispatch import DispatchState,eligible_edges,validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv,transition
from ht_pdm_fjsp.maintenance_heterogeneous_model import feasible


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def probe(condition="skill_mask"):
    cfg=e.settings("smoke");c=b.make_config(16700020,condition,6)
    return cfg,c,b.initial_state(c,16700020)


def traces(c,s):
    for matching in feasible(c,s):
        for ordering in itertools.permutations(matching):
            yield tuple([2*m+j for m,j in ordering]+[STOP]+[-1]*(2-len(matching)))


@pytest.mark.parametrize("condition",b.CONDITIONS)
@pytest.mark.parametrize("arm",(CONTEXT,FROZEN))
def test_full_sequence_distribution_normalizes_and_relabels_equivariantly(condition,arm):
    cfg,c,s=probe(condition);model=e.model_for(cfg,16700000,arm)
    tokens=list(traces(c,s));action=torch.tensor(tokens)
    batch=feature_batch([c]*len(tokens),[s]*len(tokens),[4]*len(tokens),4,arm)
    prob,ent,value=model.evaluate(batch,action)
    torch.testing.assert_close(prob.exp().sum(),torch.tensor(1.),atol=1e-6,rtol=1e-6)
    assert torch.isfinite(ent).all() and torch.isfinite(value).all()
    for a in action:model.decode(c,s,a)
    mo=(2,0,3,1);to=(1,0);mm={old:new for new,old in enumerate(mo)};tm={old:new for new,old in enumerate(to)}
    cc=replace(c,service_time=tuple(tuple(c.service_time[m][j] for j in to) for m in mo),
        restoration=tuple(tuple(c.restoration[m][j] for j in to) for m in mo))
    ss=DispatchState(tuple(s.ages[m] for m in mo),tuple(s.failed[m] for m in mo),tuple(s.pending_wait[m] for m in mo),
        tuple(s.remaining[j] for j in to),tuple(mm[s.assigned[j]] if s.assigned[j]>=0 else -1 for j in to))
    mapped=torch.tensor([[2*mm[x//2]+tm[x%2] if 0<=x<STOP else x for x in a] for a in tokens])
    pp,ee,vv=model.evaluate(feature_batch([cc]*len(tokens),[ss]*len(tokens),[4]*len(tokens),4,arm),mapped)
    torch.testing.assert_close(prob,pp,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(ent,ee,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(value,vv)
    # An unordered two-pair action has two distinct likelihood-bearing traces.
    two=[a for a in feasible(c,s) if len(a)==2][0]
    seqs=list(itertools.permutations(two));idx=[tokens.index(tuple([2*m+j for m,j in order]+[STOP])) for order in seqs]
    assert prob[idx].exp().sum()>prob[idx[0]].exp()


@pytest.mark.parametrize("arm",ARMS)
def test_sample_reevaluation_private_future_events_and_gradients(arm):
    cfg,c,s=probe();model=e.model_for(cfg,16700000,arm)
    batch=feature_batch([c]*32,[s]*32,[4]*32,4,arm)
    action,oldprob,oldvalue=model.sample(batch,generator=torch.Generator().manual_seed(23))
    prob,entropy,value=model.evaluate(batch,action)
    torch.testing.assert_close(prob,oldprob);torch.testing.assert_close(value,oldvalue)
    for a in action:validate_matching(c,s,model.decode(c,s,a))
    loss=-(prob.mean()+.01*entropy.mean())+value.square().mean();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.actor.parameters())
    env=WaitingEnv(c);env.reset(123,s);before=model.select(c,s,4,4)
    env.events=tuple((True,)*4 for _ in env.events)
    assert before==model.select(c,env.state,4,4)


def test_paired_parameters_and_real_remaining_graph_context_ablation():
    cfg,c,s=probe();models=[e.model_for(cfg,16700000,a) for a in ARMS]
    assert len({sum(p.numel() for p in model.parameters()) for model in models})==1
    for other in models[1:]:
        assert all(torch.equal(v,other.state_dict()[k]) for k,v in models[0].state_dict().items())
    context,frozen=models[1:];batch=feature_batch([c],[s],[4],4)
    graph=batch["pair_mask"];live=graph.clone();m,j=next((m,j) for m in range(4) for j in range(2) if graph[0,m,j])
    live[:,m,:]=False;live[:,:,j]=False
    encoded=context.encode(batch)
    before=context.resource_context(encoded,graph,graph);after=context.resource_context(encoded,live,graph)
    assert any(not torch.equal(x,y) for x,y in zip(before,after,strict=True))
    initial=frozen.resource_context(encoded,graph,graph);updated=frozen.resource_context(encoded,live,graph)
    assert all(torch.equal(x,y) for x,y in zip(initial,updated,strict=True))
    dc,mc=context.token_distribution(batch,encoded,live,torch.ones(1,dtype=torch.long))
    df,mf=frozen.token_distribution(batch,encoded,live,torch.ones(1,dtype=torch.long))
    assert torch.equal(mc,mf) and mc[0,STOP] and not mc[0,m*2+j]
    assert not torch.equal(dc.logits,df.logits)


@pytest.mark.parametrize("arm",(CONTEXT,FROZEN))
def test_busy_resources_force_stop_without_phantom_probability_or_entropy(arm):
    cfg,c,_=probe("dense");s=DispatchState((1,2,3,1),(False,True,False,False),(0,3,0,0),(2,1),(0,1))
    model=e.model_for(cfg,16700000,arm);batch=feature_batch([c],[s],[4],4,arm)
    a,p,v=model.sample(batch,generator=torch.Generator().manual_seed(0))
    assert a.tolist()==[[STOP,-1,-1]] and p.item()==0 and model.decode(c,s,a[0])==()
    pp,ent,vv=model.evaluate(batch,a);assert pp.item()==ent.item()==0
    assert not batch["pair_mask"].any()
    # Consuming one free worker must never dispatch that worker again.
    s=replace(s,remaining=(2,0),assigned=(0,-1))
    for seed in range(16):
        a,_,_=model.sample(feature_batch([c],[s],[4],4,arm),generator=torch.Generator().manual_seed(seed))
        assert all(j==1 and m!=0 for m,j in model.decode(c,s,a[0]))


@pytest.mark.parametrize("bad",[[8,0,-1],[0,0,8],[0,1,8],[0,2,-1],[-1,-1,-1],[9,-1,-1]])
def test_illegal_prefixes_and_padding_rejected(bad):
    cfg,c,s=probe("dense");model=e.model_for(cfg,16700000,CONTEXT)
    with pytest.raises(ValueError):model.evaluate(feature_batch([c],[s],[4],4),torch.tensor([bad]))


def test_stop_is_optional_before_capacity_and_forced_after_two_pairs():
    cfg,c,s=probe("dense");model=e.model_for(cfg,16700000,CONTEXT)
    batch=feature_batch([c],[s],[4],4);encoded=model.encode(batch)
    dist,mask=model.token_distribution(batch,encoded,batch["pair_mask"],torch.zeros(1,dtype=torch.long))
    assert mask.all() and dist.probs[0,STOP]>0
    for p in model.actor.parameters():p.data.zero_()
    a,_,_=model.sample(batch,deterministic=True)
    assert a[0,-1]==STOP and len(model.decode(c,s,a[0]))==2
    # Explicit first STOP advances physical time once with no service start.
    env=WaitingEnv(c);env.reset(123,s);env.step(model.decode(c,s,torch.tensor([8,-1,-1])))
    assert env.time==1 and env.metrics["jobs"]==0


def test_structural_pm_choices_risk_scenarios_and_fresh_split_metadata():
    cfg=e.settings("full");audit,forbidden=e.seed_audit(cfg)
    assert audit["status"]=="PASS"
    assert e.counts(cfg)==dict(models=9,training_learned_steps=1105920,training_physical_steps=1382400,
        training_episodes=23040,optimizer_steps=17280,development_episodes=1248,test_episodes=1920,
        test_base_decisions=92160,test_tail_decisions=23040,test_decisions=115200,logged_test_decisions=4800)
    with pytest.raises(ValueError):e.seed_audit({**cfg,"analysis_seed":42})
    panels,rows,training=b.metadata();historical=b.historical_pool();all_sigs=set()
    for mode in ("full","smoke"):
        for c in panels[mode].values():
            sig=e.boundary.signature(c);assert sig not in historical|all_sigs;all_sigs.add(sig)
        for c,cond,cs,es,attempt in training[mode].values():
            sig=e.boundary.signature(c);assert sig not in historical|all_sigs;all_sigs.add(sig)
            assert cs!=es and cs not in forbidden and es not in forbidden
        for row in rows[mode]:
            c=panels[mode][row["split"],row["cohort"],row["condition"]];s=b.initial_state(c,row["cohort"])
            legal=eligible_edges(c,s);assert not any(s.failed) and sum(legal[row["focal_machine"]])==2
            if row["condition"]=="skill_mask":assert sum(legal[row["competing_machine"]])==1
    cases={(b.scenario(c,cs)["focal_at_risk"],b.scenario(c,cs)["competitor_at_risk"])
        for c,cond,cs,es,attempt in training["full"].values()}
    assert len(cases)==4
    for split,cohort,cond in panels["full"]:
        dense=panels["full"][split,cohort,"dense"];masked=panels["full"][split,cohort,"skill_mask"]
        assert dense.restoration==masked.restoration
        assert b.initial_state(dense,cohort)==b.initial_state(masked,cohort)
    spec=e.training_sampler(cfg,panels["full"],forbidden)
    assert spec(cfg["train_seeds"][0],0)==spec(cfg["train_seeds"][0],0)


@pytest.mark.parametrize("arm",ARMS)
def test_tail_training_sequence_ppo_and_checkpoint_roundtrip(arm,tmp_path):
    cfg,c,s=probe();model=e.model_for(cfg,16700000,arm)
    class Journal:
        def __init__(self):self.rows=[]
        def add(self,row):self.rows.append(row)
    journals={name:Journal() for name in ("training_progress","training_episodes")};saved=[]
    def spec(seed,index):
        cs=18000000+index;cond=b.CONDITIONS[index%2]
        return b.make_config(cs,cond,6),cond,cs,19000000+index,0
    result=fit(model,cfg,16700000,spec,journals,lambda model,update,steps:saved.append((update,steps)))
    assert result["learned_steps"]==64 and result["physical_steps"]==96 and result["parameters_changed"]
    assert result["probability_audit"] and result["complete_tail_returns"] and result["optimizer_steps"]==8
    assert saved==[(1,32),(2,64)]
    restored,payload=e.save_checkpoint(tmp_path/"model.pt",model,cfg,16700000,2,64,(c,s))
    assert restored.select(c,s,4,4)==model.select(c,s,4,4)
    rewards=torch.tensor([[-1.],[-2.],[-3.],[-4.],[-5.],[-6.]])
    assert complete_targets(rewards,4).flatten().tolist()==[-21.,-20.,-18.,-15.]
    payload["feature_contract"]={};torch.save(payload,tmp_path/"bad.pt")
    with pytest.raises(ValueError,match="contract"):e.load_checkpoint(tmp_path/"bad.pt")


def test_development_selection_and_paired_pilot_statistics():
    records=[dict(update=0,mean_extended_cost=0),dict(update=2,mean_extended_cost=5),dict(update=1,mean_extended_cost=5)]
    assert e.choose_checkpoint(records)["update"]==1
    with pytest.raises(ValueError):e.choose_checkpoint(records[:1])
    cfg=e.settings("smoke");rows=[]
    metrics=("base_objective","tail_objective","extended_cost_per_machine_tick","extended_failure_cost",
        "extended_waiting_cost","extended_maintenance_cost","extended_unavailability_cost","preventive_starts","corrective_starts",
        "wait_with_choice","pm_feasible_ticks","wait_with_pm_feasible","pm_two_worker_opportunity_ticks",
        "stop_with_pm_extension_feasible","pm_assignment_alternative_ticks","pm_starts_with_two_workers","cm_only_assignment_alternative_ticks")
    for cond,cohort,shock in itertools.product(cfg["conditions"],cfg["test_cohorts"],cfg["test_shocks"]):
        for arm,seed in [("best_rule",-1),*itertools.product(ARMS,cfg["train_seeds"])]:
            cost=80 if arm==CONTEXT else 100
            rows.append(dict(condition=cond,cohort=cohort,shock=shock,controller=arm,train_seed=seed,horizon=4,
                extended_objective=cost,**{metric:1. for metric in metrics}))
    summary,differences=e.summarize(rows,cfg)
    assert len(summary["primary_contrasts"])==4 and len(summary["exploratory_rule_contrasts"])==2
    for contrast in summary["primary_contrasts"].values():
        assert contrast["relative_gain"]==pytest.approx(.2) and contrast["confidence"]==.9875
        assert contrast["confidence_interval"]==pytest.approx([.2,.2]) and contrast["status"]=="ENGINEERING_ONLY"
    with pytest.raises(RuntimeError):e.summarize(rows[:-1],cfg)
    with pytest.raises(RuntimeError):e.summarize(rows+[rows[0]],cfg)


def test_runner_smoke_counts_artifact_hashes_and_full_panel_stays_sealed(tmp_path,monkeypatch):
    original=WaitingEnv.step
    def smoke_only(self,pairs):
        assert self.config.horizon==6
        return original(self,pairs)
    monkeypatch.setattr(WaitingEnv,"step",smoke_only)
    output=tmp_path/"smoke";manifest=e.run(output,"smoke")
    assert manifest["status"]=="COMPLETED" and manifest["audit"]=="PASS"
    assert manifest["actual_counts"]==e.counts(e.settings("smoke"))
    assert not manifest["full_test_outcomes_opened"]
    for filename,record in manifest["artifacts"].items():
        path=output/filename
        assert path.stat().st_size==record["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest()==record["sha256"]
    selection=json.loads((output/"checkpoint_selection.json").read_text());assert selection["locked_before_test"]
    assert all(r["update"]>0 for r in selection["selections"].values())
    coverage=json.loads((output/"training_coverage.json").read_text())
    assert len({r["initial_parameters_sha256"] for r in coverage})==1
    assert len({r["parameter_count"] for r in coverage})==1
    with (output/"episodes.csv").open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==16 and all(float(r["pm_two_worker_opportunity_ticks"])>=1 for r in rows)
    with pytest.raises(ValueError,match="new/empty"):e.run(output,"smoke")


def test_full_failure_keeps_manifest_without_opening_outcomes(tmp_path,monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS","2")
    with pytest.raises(ValueError):e.run(tmp_path/"failed","full")
    manifest=json.loads((tmp_path/"failed/manifest.json").read_text())
    assert manifest["status"]=="FAILED" and not manifest["full_test_outcomes_opened"]
