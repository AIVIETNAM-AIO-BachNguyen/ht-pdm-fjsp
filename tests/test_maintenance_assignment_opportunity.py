"""Prospective natural opportunities, nested futures, calibration, and isolation."""
from dataclasses import replace
import itertools
import json
import numpy as np
import pytest
import torch

from ht_pdm_fjsp import maintenance_assignment_opportunity_benchmark as b
from ht_pdm_fjsp.maintenance_assignment_opportunity import counts, run, development_guard
from ht_pdm_fjsp.maintenance_assignment_opportunity_data import source_class, paired_target, subset, collect
from ht_pdm_fjsp.maintenance_assignment_opportunity_training import draw_indices
from ht_pdm_fjsp.maintenance_assignment_opportunity_policy import choose_assignment, calibrate, OpportunityController
from ht_pdm_fjsp.maintenance_assignment_opportunity_statistics import summarize
from ht_pdm_fjsp.maintenance_assignment_policy import AssignmentPolicy
from ht_pdm_fjsp.maintenance_autoregressive_policy import feature_batch, STOP
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig
from ht_pdm_fjsp.maintenance_heterogeneous_model import feasible


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def physical():
    c = WaitingConfig(4, 2, 60, 4, .4, ((2, 5), (3, 2), (5, 4), (2, 3)),
        ((.5, 1.), (.75, .5), (.25, 1.), (1., .25)), waiting_price=12.)
    s = DispatchState((1, 2, 3, 0), (False,)*4, (0,)*4, (0,)*2, (-1,)*2)
    return c, s


def test_source_class_preserves_PM_machine_set_and_exact_CM():
    c, s = physical()
    assert set(source_class(c, s, ((0, 0), (1, 1)))) == {((0, 0), (1, 1)), ((0, 1), (1, 0))}
    failed = replace(s, failed=(True, False, False, False))
    assert source_class(c, failed, ((0, 0), (1, 1))) == (((0, 0), (1, 1)),)
    assert source_class(c, s, ()) == ((),)
    busy = replace(s, remaining=(2, 0), assigned=(3, -1))
    assert source_class(c, busy, ((0, 1),)) == (((0, 1),),)


def test_nested_pair_targets_use_same_first_scenarios_at_every_tick_and_correct_sign():
    roots = (((0, 1),), ((0, 0),))
    costs = np.array([[11., 15., 10., 20.], [1., 3., 4., 6.]])
    l, r, low, low_se = paired_target(costs, roots, 2, 20.)
    _, _, high, high_se = paired_target(costs, roots, 4, 20.)
    assert CATALOGUE[l] == roots[1] and CATALOGUE[r] == roots[0]
    assert low == pytest.approx(.55) and high == pytest.approx(.525)
    assert low_se == pytest.approx(.05)
    assert high_se == pytest.approx(np.std([.5, .6, .3, .7], ddof=1)/2)
    with pytest.raises(ValueError): paired_target(costs, roots, 5, 20.)


def test_landmark_subset_keeps_state_index_mapping():
    data = dict(raw_pairs=torch.zeros(4, 4, 2, 13), context=torch.arange(4)[:, None],
        original_ids=torch.arange(4), pairs=torch.tensor([[1, 1, 2], [3, 2, 3]]),
        targets=torch.tensor([.4, -.2]), cohorts=torch.tensor([10, 20]), ticks=torch.tensor([0, 18]))
    sub = subset(data, torch.tensor([True, False]))
    assert sub['pairs'].tolist() == [[1, 1, 2]] and sub['context'][sub['pairs'][0, 0]].item() == 1
    assert torch.equal(sub['original_ids'], data['original_ids'])


def test_opportunity_draws_pair_low_high_and_balance_clock_bins():
    # Oversample initial labels in the population to expose imbalance bugs.
    cond = torch.tensor([0]*80+[0]*3+[1]*80+[1]*3)
    ticks = torch.tensor([0]*80+[12, 24, 36]+[0]*80+[12, 24, 36])
    data = dict(conditions=cond, ticks=ticks)
    a = draw_indices(data, 'opportunity_low', 10000, np.random.default_rng(13))
    c = draw_indices(data, 'opportunity_high', 10000, np.random.default_rng(13))
    assert torch.equal(a, c)
    frequencies = torch.bincount(ticks[a]//12, minlength=4).float()/len(a)
    assert (frequencies > .22).all() and (frequencies < .28).all()
    landmark = draw_indices(data, 'landmark_low', 10000, np.random.default_rng(13))
    assert float((ticks[landmark]==0).float().mean()) > .9
    with pytest.raises(RuntimeError): draw_indices({k: v[:83] for k, v in data.items()}, 'opportunity_low', 10, np.random.default_rng(1))


def test_original_zero_gain_does_not_force_scalar_fallback():
    margins = dict(enabled=True, margin=.2)
    scores = np.array([[0., -1.]]*3)
    choice, audit = choose_assignment(scores, 0, [0, 1], False, margins)
    assert choice == 1 and not audit['near_tie']
    _, audit = choose_assignment(scores, 0, [0, 1], True, dict(enabled=False, margin=None))
    assert not audit['near_tie'] and audit['rejection_reason'] == 'insufficient_calibration'
    _, audit = choose_assignment(np.array([[0., -1.1e-5]]*3), 0, [0, 1], False, margins)
    assert audit['near_tie']
    _, audit = choose_assignment(scores, 0, [0], False, margins)
    assert not audit['near_tie']


class Scores:
    def eval(self): return self
    def __call__(self, data):
        return torch.tensor([[0., -1.]]).repeat(len(data['context']), 1)


def calibration_data():
    return dict(context=torch.zeros(4, 96), raw_pairs=torch.zeros(4, 4, 2, 13),
        pairs=torch.tensor([[i, 0, 1] for i in range(4)]), original_ids=torch.zeros(4,dtype=torch.long),
        targets=torch.tensor([-.2]*4), mc_se=torch.tensor([.1]*4), conditions=torch.tensor([0,0,1,1]),
        origins=torch.zeros(4,dtype=torch.long), ticks=torch.tensor([0,12,0,12]), cohorts=torch.tensor([1,2,1,2]))


def test_calibration_requires_profiles_and_late_opportunities_independently_of_label_count():
    data = calibration_data(); cfg = {**b.settings('smoke'), 'calibration_minimum':2, 'calibration_profiles':2, 'calibration_late':1}
    result, records = calibrate([Scores()]*3, data, cfg)
    assert len(records)==4 and all(r['enabled'] for r in result.values())
    assert result['dense']['margin']==pytest.approx(1-(.2-1.96*.1))
    clustered = {**data, 'cohorts':torch.ones(4,dtype=torch.long)}
    result, _ = calibrate([Scores()]*3, clustered, cfg)
    assert not result['dense']['enabled'] and result['dense']['physical_profiles']==1
    early = {**data, 'ticks':torch.zeros(4,dtype=torch.long)}
    result, _ = calibrate([Scores()]*3, early, cfg)
    assert not result['dense']['enabled'] and result['dense']['late_samples']==0


def test_development_guard_is_descriptive_and_gain_sign_consistent():
    cfg=b.settings('smoke');data=calibration_data()
    margins={c:dict(enabled=True,margin=.1) for c in b.CONDITIONS}
    result=development_guard([Scores()]*3,data,margins,cfg)
    assert result['dense']['gated_overrides']==2 and result['dense']['paired_cost_gain_per_opportunity']==pytest.approx(4.)
    margins['skill_mask']['enabled']=False
    assert development_guard([Scores()]*3,data,margins,cfg)['skill_mask']['gated_overrides']==0


def test_all_ticks_are_collected_from_real_source_transitions():
    c, _ = physical();cfg={**b.settings('smoke'),'train_cohorts':[25200000],'train_shocks':[25300000]}
    class Source:
        def select(self,c,s,remaining,horizon):
            return max(feasible(c,s),key=len),dict(remaining=remaining)
    class Journal:
        def __init__(self):self.rows=[]
        def add(self,r):self.rows.append(r)
    j,coverage=Journal(),Journal()
    panels={('train',25200000,cond):c for cond in b.CONDITIONS}
    records=collect(panels,cfg,'train',{18100000:Source()},j,coverage)
    assert len(j.rows)==96 and len(coverage.rows)==2 and all(r['identity']['origin']=='natural' for r in records)
    assert sum(r['opportunities'] for r in coverage.rows)==len(records)
    chain=[r for r in j.rows if r['condition']=='dense']
    assert all(chain[i]['following']==chain[i+1]['state'] for i in range(47))
    assert set(r['tick'] for r in chain)==set(range(48))


def test_controller_changes_only_workers_and_scalar_batch_agree(monkeypatch):
    c,s=physical();base=AssignmentPolicy();base.requires_grad_(False)
    monkeypatch.setattr(base,'greedy_tokens',lambda data:torch.tensor([[0,3,STOP]]).repeat(len(data['machines']),1))
    class Head:
        def eval(self):return self
        def __call__(self,data):
            v=torch.zeros(len(data['context']),21)
            v[:,CATALOGUE.index(((0,1),(1,0)))]=-1;v[:,0]=-100
            return v
    margins={c:dict(enabled=True,margin=.1) for c in b.CONDITIONS}
    ctrl=OpportunityController(base,[Head()]*3,'opportunity_high_gated',margins,b.settings('smoke'))
    assert ctrl.select(c,s,48,48)[0]==((0,1),(1,0))
    failed=replace(s,failed=(True,False,False,False))
    assert ctrl.select(c,failed,48,48)[0]==((0,0),(1,1))
    batch=feature_batch([c,c],[s,s],[48,48],48);a,_,_=ctrl.sample(batch,deterministic=True)
    assert ctrl.decode(c,s,a[0])==ctrl.select(c,s,48,48)[0]


def test_fresh_seed_and_physical_splits_and_fixed_budgets():
    _,records,audit=b.metadata(); signatures=[r['physical_signature'] for rs in records.values() for r in rs]
    assert len(set(signatures))==len(signatures)==538 and not audit['outcomes_used']
    assert b.seed_audit(b.settings('smoke'))['status']=='PASS'
    budget=counts(b.settings('full'))
    assert budget['test_episodes']==4032 and budget['trained_heads']==27 and budget['optimizer_steps']==17280
    assert budget['natural_source_transitions']==168192 and budget['logged_test_transitions']==10080
    identity=dict(condition='skill_mask',cohort=24200000,shock=24300000,model_seed=18100000,tick=47,origin='natural')
    assert len({b.future_seed(b.settings('full'),identity,p) for p in ('train','development','calibration','ranking_selection','ranking_evaluation')})==5
    assert b.settings('full')['low_scenarios']==16 and b.settings('full')['train_scenarios']==128


def test_primary_grid_sign_and_missing_rows():
    cfg=b.settings('smoke');rows=[]
    metrics=('extended_objective','base_objective','tail_objective','extended_failure_cost','extended_waiting_cost',
        'extended_maintenance_cost','extended_unavailability_cost','preventive_starts','corrective_starts',
        'wait_with_pm_feasible','pm_assignment_alternative_ticks','assignment_override_ticks','guard_rejected_ticks','batch_amortized_policy_ms_mean')
    for cond,name in itertools.product(b.CONDITIONS,b.NAMES):
        cost=90. if name=='opportunity_high_gated' else 100.
        rows.append(dict(condition=cond,cohort=cfg['test_cohorts'][0],shock=cfg['test_shocks'][0],controller=name,
            train_seed=18100000,horizon=48,**{k:cost for k in metrics}))
    summary,_=summarize(rows,cfg)
    assert len(summary['primary_contrasts'])==4 and len(summary['secondary_contrasts'])==14
    assert all(r['relative_gain']==pytest.approx(.1) for r in summary['primary_contrasts'].values())
    with pytest.raises(RuntimeError):summarize(rows[:-1],cfg)


def test_existing_artifacts_and_failed_preflight_keep_test_sealed(tmp_path):
    prior=tmp_path/'prior';prior.mkdir();(prior/'keep').write_text('intact')
    with pytest.raises(ValueError):run(prior,tmp_path/'missing')
    assert (prior/'keep').read_text()=='intact'
    output=tmp_path/'new'
    with pytest.raises(FileNotFoundError):run(output,tmp_path/'missing')
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['status']=='FAILED' and not manifest['full_test_outcomes_opened']
