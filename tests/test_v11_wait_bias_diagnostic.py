"""V11 stored Q WAIT preference is signed against feasible service roots."""
import importlib.util
from pathlib import Path

_spec=importlib.util.spec_from_file_location('v11_wait_bias',
    Path(__file__).resolve().parents[1]/'research_result/diagnose_v11_wait_bias.py')
assert _spec is not None and _spec.loader is not None
_d=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_d)


def test_wait_advantage_sign_and_legal_service_filter():
    wait=_d.ACTIONS.index(())
    one=_d.ACTIONS.index(((0,0),))
    two=_d.ACTIONS.index(((1,0),))
    assert wait==0
    assert _d.q_wait_delta({wait:5.,one:10.,two:12.},(wait,one,two))==-5.
    assert _d.q_wait_delta({wait:15.,one:10.,two:12.},(wait,one,two))==5.
    assert _d.q_wait_delta({wait:5.},(wait,)) is None
    assert _d.q_wait_delta({one:10.},(one,)) is None


def test_source_checkpoint_grid_loads_and_is_paired_at_initial_weights():
    _,selection,models,hashes,development,selected=_d.load_inputs()
    assert len(models)==len(_d.ARMS)*len(_d.TRAIN)*2
    assert len(development)==4*2*5*7*16
    assert len(selected)==4*2*5*16
    assert selection['reference_credit']['70500001']['iteration']==0
    for train in _d.TRAIN:
        assert hashes[f'reference_credit/{train}/initial']!= ''
        left=models[('reference_credit',train,'initial')]
        right=models[('self_credit',train,'initial')]
        assert all((left.state_dict()[k]==right.state_dict()[k]).all()
                   for k in left.state_dict())
