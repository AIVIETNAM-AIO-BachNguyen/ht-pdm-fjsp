"""Focused arithmetic checks for the immutable V11 gradient diagnostic."""
import numpy as np

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('v11_gradient', Path(__file__).resolve().parents[1] / 'research_result/diagnose_v11_gradient.py')
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ACTIONS = _module.ACTIONS
wait_gradient = _module.wait_gradient


def test_wait_gradient_pushes_down_wait_when_service_cheaper():
    service = next(i for i, action in enumerate(ACTIONS) if action)
    p = np.zeros(len(ACTIONS)); p[0] = .75; p[service] = .25
    q = np.zeros(len(ACTIONS)); q[0] = 20; q[service] = 0
    assert np.isclose(wait_gradient(p, q, (0, service)), -.375)
    q[0], q[service] = 0, 20
    assert np.isclose(wait_gradient(p, q, (0, service)), .375)


def test_wait_gradient_requires_service_root():
    p = np.zeros(len(ACTIONS)); p[0] = 1
    q = np.zeros(len(ACTIONS))
    assert wait_gradient(p, q, (0,)) is None
