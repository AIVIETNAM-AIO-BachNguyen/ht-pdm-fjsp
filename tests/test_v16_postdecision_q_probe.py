import numpy as np
import torch

from ht_pdm_fjsp.maintenance_framework_v1 import configs
from ht_pdm_fjsp.maintenance_requests import initial_state
from research_result.run_v16_postdecision_q_probe import DIM, ValueHead, features, fit


def test_post_release_changes_with_matching_but_pre_does_not():
    cfg = configs()["reciprocal_scale2"]
    state = initial_state(cfg)
    common0, post0, pre0 = features(cfg, state, ())
    m = next(m for m in range(cfg.machines) if cfg.service_time[m][0] > 0)
    common1, post1, pre1 = features(cfg, state, ((m, 0),))
    np.testing.assert_array_equal(pre0, pre1)
    assert post1[0] == cfg.service_time[m][0] / 12
    assert post0[0] == 0
    assert common0.sum() == 0
    assert common1.sum() == 2
    assert all(v >= 1 / 12 for v in post1[2:])


def test_exact_head_capacity_and_output():
    head = ValueHead()
    assert sum(p.numel() for p in head.parameters()) == 9025
    assert head(torch.zeros((2, 21, DIM))).shape == (2, 21)


def test_paired_fit_is_deterministic_and_masked():
    n, a = 8, 3
    x = np.zeros((n, a, DIM), np.float32)
    x[:, 1, 0] = 1
    q = np.zeros((n, a), np.float32)
    q[:, 1] = -5
    mask = np.ones((n, a), bool)
    mask[:, 2] = False
    import time
    one, trace1 = fit(x, q, mask, np.arange(n), 72950000, time.monotonic() + 60)
    two, trace2 = fit(x, q, mask, np.arange(n), 72950000, time.monotonic() + 60)
    assert trace1 == trace2 and len(trace1) == 6
    for p, p2 in zip(one.parameters(), two.parameters()):
        torch.testing.assert_close(p, p2)


def test_specialist_earliest_release_ignores_incompatible_worker():
    cfg = configs()["specialist_generalist_scale2"]
    state = initial_state(cfg)
    _, post, pre = features(cfg, state, ((1, 0),))
    assert cfg.service_time[0][1] == 0
    assert pre[2] == 1 / 12
    assert post[2] == cfg.service_time[1][0] / 12
    assert post[3] == 1 / 12  # generalist can serve machine 1 immediately later
