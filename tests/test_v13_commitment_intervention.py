"""Focused definitions for the V13 one-edge causal diagnostic."""
import sys
from pathlib import Path

from ht_pdm_fjsp.maintenance_dispatch import DispatchConfig, DispatchState
from ht_pdm_fjsp.maintenance_requests import MaintenanceRequest, RequestState, legal_matchings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research_result'))
from run_v13_commitment_intervention import at_risk, summarize  # noqa: E402


def test_at_risk_distinguishes_occupied_capacity_from_available_worker():
    cfg = DispatchConfig(2, 1, 64, 3, 0.0, ((2,), (2,)), ((1.0,), (1.0,)))
    occupied = RequestState(
        DispatchState((2, 1), (False, False), (0, 0), (1,), (1,)),
        (MaintenanceRequest(0, 1, 0, 'preventive'),
         MaintenanceRequest(1, 1, 0, 'preventive', started_at=0, technician=0)),
        (1, 1))
    legal = legal_matchings(cfg, occupied)
    assert legal == ((),)
    assert at_risk(cfg, occupied, (), legal) == (
        dict(exposed=1, feasible=0, constrained=1), 0)
    available = RequestState(
        DispatchState((2, 1), (False, False), (0, 0), (0,), (-1,)),
        (MaintenanceRequest(0, 1, 0, 'preventive'), None), (1, 0))
    legal = legal_matchings(cfg, available)
    assert ((0, 0),) in legal
    assert at_risk(cfg, available, (), legal) == (
        dict(exposed=1, feasible=1, constrained=0), None)


def test_primary_gate_includes_no_op_episodes():
    seeds = tuple(range(72500000, 72500005))
    physicals = tuple(range(72510000, 72510016))
    rows = []
    for case in ('reciprocal_scale2', 'interchangeable_scale1',
                 'reciprocal_scale3', 'specialist_generalist_scale2'):
        for seed in seeds:
            for index, physical in enumerate(physicals):
                triggered = index < 8
                rows.append(dict(case=case, train_seed=seed, seed=physical,
                                 triggered=int(triggered), cost_delta=-24 if triggered else 0,
                                 exposure_delta=0, failure_delta=0, pm_starts_delta=0))
    result = summarize(rows, seeds, physicals, 'dev')
    assert result['joint_gate']
    assert result['joint_cost_delta'] == -12
    assert result['by_case']['interchangeable_scale1']['trigger_fraction'] == .5
    assert result['by_case']['reciprocal_scale2']['conditional_delta'] == -24
