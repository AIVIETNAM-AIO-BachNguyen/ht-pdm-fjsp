"""Assignment intervention preserves the actor's dispatch scope."""
import importlib.util
from pathlib import Path
import sys

_root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_root/'research_result'))
_spec=importlib.util.spec_from_file_location('v9_assignment_intervention',
    _root/'research_result/intervene_v9_assignment.py')
assert _spec is not None and _spec.loader is not None
_m=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)


def test_assignment_only_preserves_set_and_minimizes_duration():
    cfg=_m.configs()['reciprocal_scale2']
    actions=_m.ACTIONS
    by_set={}
    for i,matching in enumerate(actions):
        by_set.setdefault(frozenset(m for m,_ in matching),[]).append(i)
    for same in by_set.values():
        for chosen in same:
            result=_m.assignment_only(cfg,chosen,same)
            assert frozenset(m for m,_ in actions[result])==frozenset(m for m,_ in actions[chosen])
            assert _m.duration(cfg,result)==min(_m.duration(cfg,a) for a in same)
            if _m.duration(cfg,chosen)==_m.duration(cfg,result):
                assert result==chosen
    assert _m.assignment_only(cfg,actions.index(()),[actions.index(())])==actions.index(())


def test_assignment_only_never_uses_infeasible_action():
    cfg=_m.configs()['specialist_generalist_scale2']
    by_set={}
    for i,matching in enumerate(_m.ACTIONS):
        by_set.setdefault(frozenset(m for m,_ in matching),[]).append(i)
    for same in by_set.values():
        chosen=same[-1]
        assert _m.assignment_only(cfg,chosen,[chosen])==chosen
