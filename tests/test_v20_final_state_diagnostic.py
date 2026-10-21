"""Focused invariants for opened-DEV V20 first-divergence diagnostics."""
from __future__ import annotations

import numpy as np

from ht_pdm_fjsp.maintenance_framework_v1 import configs
from research_result import diagnose_v20_final_state as runner
from research_result import verify_v20_final_state as verifier


def test_action_categories_distinguish_gate_and_matching():
    assert runner.classify(0, 0) == 'same'
    assert runner.classify(1, 0) == 'v20_service_other_wait'
    assert runner.classify(0, 1) == 'v20_wait_other_service'
    assert runner.classify(2, 1) == 'v20_higher_cardinality'
    assert runner.classify(1, 2) == 'v20_lower_cardinality'
    assert runner.classify(1, 9) == 'machine_set'
    assert runner.classify(1, 5) == 'assignment'
    for left, right in ((0, 1), (2, 1), (1, 9), (1, 5)):
        assert runner.classify(left, right) == verifier.classify(left, right)


def test_diagnostic_common_tapes_reproduce_independently():
    config = configs()['reciprocal_scale3']
    state = {'physical': {'ages': [1, 2, 3]}, 'requests': []}
    first = runner.scenario_bank(config, 'reciprocal_scale3', 73700003, 73711000,
                                 4, state, 16)
    repeated = verifier.tapes_for(config, 'reciprocal_scale3', 73700003, 73711000,
                                  4, state, 16)
    assert first == repeated
    assert first[0] != runner.scenario_bank(config, 'reciprocal_scale3',
                                            73700003, 73711001, 4, state, 16)[0]
    assert len(first[1]) == 16
    assert len(first[1][0]['events']) == config.horizon - 4


def test_bootstrap_preserves_constant_five_seed_panel():
    cube = np.full((5, 16), 7.25)
    assert runner.bootstrap(cube) == [7.25, 7.25]
    assert verifier.bootstrap(cube) == [7.25, 7.25]


def test_diagnostic_roles_do_not_overlap_historical_seed_roles():
    assert runner.check_seed_roles() > 18000
