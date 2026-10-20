import numpy as np
import torch

from ht_pdm_fjsp.maintenance_framework_v1 import catalogue
from ht_pdm_fjsp.maintenance_framework_v6_policy import EdgeSharingActor
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'probe_v8_representability', Path(__file__).resolve().parents[1] /
    'research_result/probe_v8_representability.py')
assert _spec is not None and _spec.loader is not None
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)
ACTIONS, JointScorer = _probe.ACTIONS, _probe.JointScorer
collision_key, local_edge_bytes, split = _probe.collision_key, _probe.local_edge_bytes, _probe.split


def test_exact_collision_key_matches_additive_edge_margin_invariance():
    actions = catalogue()
    wait = actions.index(())
    one = actions.index(((0, 0),))
    x = np.zeros(56, np.float32)
    x[0] = .25
    x[2:10] = np.arange(8, dtype=np.float32)/10
    x[34:36] = [1,0]
    x[40:42] = [1,.25]
    y = x.copy()
    y[10:18] = np.arange(8, dtype=np.float32)+2 # unrelated machine
    assert collision_key(x, wait, one) == collision_key(y, wait, one)
    model = EdgeSharingActor('shared_edge').eval()
    with torch.no_grad():
        logits=model(torch.tensor(np.stack([x,y])))
    assert abs(float((logits[0,one]-logits[0,wait])-
                     (logits[1,one]-logits[1,wait]))) < 1e-6
    y[2] += .1
    assert collision_key(x, wait, one) != collision_key(y, wait, one)
    assert len(local_edge_bytes(x,0,0)) == 14*4


def test_split_is_by_physical_seed_and_joint_scores_every_action():
    rows=[{'physical_seed':seed} for seed in (69100000,69100019,69100020,
                                              69100023,69100024,69100031)]
    got=split(rows)
    assert [len(got[k]) for k in ('train','selection','probe')]==[2,2,2]
    assert ACTIONS == catalogue()
    assert JointScorer()(torch.zeros(3,56)).shape==(3,len(ACTIONS))
