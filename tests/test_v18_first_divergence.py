import json
from pathlib import Path

import torch

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_framework_v1 import catalogue,configs
from ht_pdm_fjsp.maintenance_framework_v17_policy import FullMatchingActor
from ht_pdm_fjsp.maintenance_request_experiment import registered_integers
from ht_pdm_fjsp.maintenance_requests import initial_state
from research_result.diagnose_v18_first_divergence import (
    ACTIONS,ANCHOR,BOOTSTRAP,ROOT,SOURCE,SOURCE_AUDIT,bank,classify,root_cost,
)


def test_prelocked_source_and_seed_roles():
    source=json.loads((SOURCE/'manifest.json').read_text())
    audit=json.loads(SOURCE_AUDIT.read_text())
    assert source['status']=='COMPLETED' and audit['status']=='PASS'
    assert Path(audit['run']).resolve()==SOURCE
    assert source['actor_parameters']==9025
    used=set().union(*(registered_integers(json.loads(p.read_text()))
                       for p in (ROOT/'configs').glob('*seed_registry.json')))
    assert not {ANCHOR,BOOTSTRAP}&used and ANCHOR!=BOOTSTRAP


def test_first_difference_categories_cover_assignment_and_wait():
    wait=next(i for i,a in enumerate(ACTIONS) if not a)
    service=next(i for i,a in enumerate(ACTIONS) if len(a)==1)
    other=next(i for i,a in enumerate(ACTIONS) if len(a)==1 and {m for m,j in a}!={m for m,j in ACTIONS[service]})
    assert classify(wait,wait)=='same'
    assert classify(service,wait)=='bounded_service_other_wait'
    assert classify(wait,service)=='bounded_wait_other_service'
    assert classify(service,other)=='machine_set'
    same_machine=next((i for i,a in enumerate(ACTIONS) if len(a)==1 and
                       {m for m,j in a}=={m for m,j in ACTIONS[service]} and a!=ACTIONS[service]),None)
    assert same_machine is not None and classify(service,same_machine)=='assignment'


def test_common_bank_and_two_continuations_use_same_tape():
    torch.set_num_threads(1)
    cfg=configs()['interchangeable_scale1']
    state=initial_state(cfg)
    seed,tapes=bank(cfg,'interchangeable_scale1','rollout',73300000,73311000,0,2)
    assert seed==role_seed(ANCHOR,'v18_first_divergence:interchangeable_scale1:rollout:73300000:73311000:0')
    assert tapes==bank(cfg,'interchangeable_scale1','rollout',73300000,73311000,0,2)[1]
    actor=FullMatchingActor();actor.eval()
    root=next(i for i,a in enumerate(ACTIONS) if not a)
    own,n1=root_cost(cfg,state,0,root,tapes[0],actor,'own',lambda:None)
    risk,n2=root_cost(cfg,state,0,root,tapes[0],actor,'risk',lambda:None)
    assert n1==n2==cfg.horizon
    assert all(map(lambda x: isinstance(x,float),(own[0],own[1],risk[0],risk[1])))
