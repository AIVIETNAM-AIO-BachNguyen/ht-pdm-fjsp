"""V13 matching-specific opportunity cost and fresh campaign contract."""
import torch

from ht_pdm_fjsp.maintenance_framework_v13_policy import ACTIONS, OpportunityCostActor
from ht_pdm_fjsp.maintenance_framework_v13_campaign import ARMS, VERSION, seed_registry


def test_fresh_registry_and_equal_parameter_budget():
    registry, (train, dev, smoke) = seed_registry()
    assert registry['campaign_version'] == VERSION
    assert len(train) == 5 and len(dev) == 16 and len(smoke) == 2
    assert ARMS == ('competition', 'no_competition')
    left, right = OpportunityCostActor(True), OpportunityCostActor(False)
    assert sum(p.numel() for p in left.parameters()) == sum(p.numel() for p in right.parameters())


def test_unserved_demand_creates_matching_interaction():
    torch.manual_seed(4)
    candidate = OpportunityCostActor(True)
    ablation = OpportunityCostActor(False)
    ablation.load_state_dict(candidate.state_dict())
    observation = torch.zeros((1, 56))
    # Both workers are free; every edge is eligible and has positive occupation time.
    observation[:, 34] = 1
    observation[:, 37] = 1
    observation[:, 40:56:2] = 1
    observation[:, 41:56:2] = .5
    a = ACTIONS.index(((0, 0),))
    b = ACTIONS.index(((1, 1),))
    both = ACTIONS.index(((0, 0), (1, 1)))
    logits, penalty, demand = candidate.components(observation)
    local_logits, local_penalty, _ = ablation.components(observation)
    assert torch.isfinite(logits).all() and torch.isfinite(demand).all()
    assert torch.all(demand > 0)
    assert penalty[0, both] < penalty[0, a] + penalty[0, b]
    assert torch.all(local_penalty == 0)
    assert torch.allclose(local_logits[0, both], local_logits[0, a] + local_logits[0, b], atol=1e-6)
    assert logits[0, both] > logits[0, a] + logits[0, b]


def test_competition_term_has_finite_gradient_and_wait_is_legal_score():
    actor = OpportunityCostActor(True)
    x = torch.randn((2,56))
    x[:,41:56:2] = .4
    logits, penalty, _ = actor.components(x)
    assert logits.shape == (2,len(ACTIONS))
    assert penalty.shape == logits.shape
    logits.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())


def test_unrequested_machine_age_changes_technician_opportunity_cost():
    actor = OpportunityCostActor(True)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.machine_encoder[0].weight[0, 0] = 1.
        actor.demand_head[0].weight[0, 0] = 1.
        actor.demand_head[2].weight[0, 0] = 1.
    x = torch.zeros((2, 56))
    x[:, 41:56:2] = .5
    x[1, 2 + 2*8] = 1.  # age of machine 2, which has no current request
    _, penalty, _ = actor.components(x)
    assign_other = ACTIONS.index(((0, 0),))
    serve_aged = ACTIONS.index(((0, 0), (2, 1)))
    assert penalty[1, assign_other] > penalty[0, assign_other]
    assert torch.allclose(penalty[1, serve_aged], penalty[0, serve_aged], atol=1e-6)
