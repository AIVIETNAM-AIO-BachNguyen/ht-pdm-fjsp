import pytest

from ht_pdm_fjsp.maintenance_shared_technician_exploratory_v2 import config
from ht_pdm_fjsp.maintenance_shared_technician_factorial import (
    episode, panel, seed_audit,
)
from ht_pdm_fjsp import maintenance_shared_technician_exploratory_v2 as v2


@pytest.mark.parametrize("condition", v2.CONDITIONS)
def test_v2_service_and_seed_contract(condition):
    c = config(condition)
    assert min(d for row in c.service_time for d in row if d) == 2
    assert seed_audit(v2.REGISTRY)["status"] == "PASS"
    assert len(panel("full", v2.REGISTRY)["sealed_test"]) == 64
    r = episode(condition, "capacity_aware", "fcfs", 93000000,
        config_fn=config, version=v2.VERSION)
    assert r["ticks"] == 60
    assert r["requests"] == r["completions"] + r["pending_end"] + r["active_end"]


def test_v2_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        config("scarce_overlap")
