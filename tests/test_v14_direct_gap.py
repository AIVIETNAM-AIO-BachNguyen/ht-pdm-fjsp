import json
from pathlib import Path

import torch

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import configs
from ht_pdm_fjsp.maintenance_framework_v14_policy import OpportunityCostActor
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers
from ht_pdm_fjsp.maintenance_requests import initial_state
from research_result.diagnose_v14_direct_gap import (
    ACTIONS, ANCHOR, BOOTSTRAP, ROOT, SOURCE, SOURCE_AUDIT, V14, V14_AUDIT,
    bank, classify, root_cost,
)


def test_source_and_forecast_seed_roles_are_frozen_and_disjoint():
    for run,audit in ((SOURCE,SOURCE_AUDIT),(V14,V14_AUDIT)):
        manifest=json.loads((run/'manifest.json').read_text())
        report=json.loads(audit.read_text())
        assert manifest['status']=='COMPLETED' and report['status']=='PASS'
        assert Path(report['run']).resolve()==run
    used=set().union(*(registered_integers(json.loads(p.read_text()))
                       for p in (ROOT/'configs').glob('*seed_registry.json')))
    assert not {ANCHOR,BOOTSTRAP}&used and ANCHOR!=BOOTSTRAP


def test_first_difference_covers_wait_and_assignment():
    wait=next(i for i,a in enumerate(ACTIONS) if not a)
    service=next(i for i,a in enumerate(ACTIONS) if len(a)==1)
    other=next(i for i,a in enumerate(ACTIONS) if len(a)==1 and {m for m,j in a}!={m for m,j in ACTIONS[service]})
    same_machine=next(i for i,a in enumerate(ACTIONS) if len(a)==1 and
                      {m for m,j in a}=={m for m,j in ACTIONS[service]} and a!=ACTIONS[service])
    assert classify(wait,wait)=='same'
    assert classify(service,wait)=='v14_service_other_wait'
    assert classify(wait,service)=='v14_wait_other_service'
    assert classify(service,other)=='machine_set'
    assert classify(service,same_machine)=='assignment'


def test_common_failure_bank_and_continuations():
    torch.set_num_threads(1)
    cfg=configs()['interchangeable_scale1'];state=initial_state(cfg)
    seed,tapes=bank(cfg,'interchangeable_scale1',72600000,73411000,0,2)
    assert seed==role_seed(ANCHOR,'v14_direct_gap:interchangeable_scale1:72600000:73411000:0')
    assert tapes==bank(cfg,'interchangeable_scale1',72600000,73411000,0,2)[1]
    actor=OpportunityCostActor('competition').eval()
    root=next(i for i,a in enumerate(ACTIONS) if not a)
    own,n1=root_cost(cfg,state,0,root,tapes[0],actor,'own',lambda:None)
    risk,n2=root_cost(cfg,state,0,root,tapes[0],actor,'risk',lambda:None)
    assert n1==n2==cfg.horizon
    assert all(isinstance(x,float) for x in (*own,*risk))
