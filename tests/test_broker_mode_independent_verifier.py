import importlib.util,itertools,json
from pathlib import Path
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_broker_mode_diagnostic as d

spec=importlib.util.spec_from_file_location('mode_independent',Path(__file__).resolve().parents[1]/'scripts/verify_broker_mode_diagnostic.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


@pytest.mark.parametrize('arm',d.x.ARMS)
def test_independent_numpy_laws_and_brute_mode_match_actual_engineering_actor(arm):
    torch.set_num_threads(1);h=d.helpers();c=d.x.settings('engineering');model=d.x.make_model(arm,c,c['train_seeds'][0]);w={k:p.detach().numpy().astype(float) for k,p in model.state_dict().items()}
    initial,events,_=h.initial_events(d.x.role_seed(290700001,f'{d.x.VERSION}:N4:physical'));states=[];obs=[];actual=[];state=initial
    for tick in range(12):
        o=d.complete_obs(h,state,tick);q=d.query(model,arm,state,o,h);states.append(state);obs.append(o);actual.append(q);state,_=h.primitive_step(state,q['greedy_pairs'],events[tick])
    law,greedy=v.reference_laws(obs,states,arm,w)
    np.testing.assert_allclose(law.sum(1),1,atol=2e-12)
    for i,q in enumerate(actual):
        assert v.CAT[int(law[i].argmax())]==q['mode_pairs']
        assert v.decode(greedy[i],arm,states[i],obs[i])==q['greedy_pairs']
        for pairs,key in ((q['greedy_pairs'],'greedy_logmass'),(q['mode_pairs'],'mode_logmass')):assert np.log(law[i,v.INDEX[pairs]])==pytest.approx(q[key],abs=5e-6,rel=1e-6)


def test_all_24_machine_priority_orders_and_every_latent_proposal():
    h=d.helpers();state=dict(ages=[3]*4,failed=[True]*4,pending_wait=[0]*4,remaining=[0]*2,assigned=[-1]*2);o=d.complete_obs(h,state,0)
    for order in itertools.permutations(range(4)):
        rank=tuple(order.index(m) for m in range(4));current=dict(state,pending_wait=[10-r for r in rank]);proposals,matrix=v.translator(False,rank)
        for i,u in enumerate(proposals):assert v.CAT[int(matrix[i].argmax())]==v.decode(u,'mappo',current,o)
    for arm in d.x.CORE_ARMS:
        priority=h.priority_matrix(o,arm);rank=tuple(int(a!=max(range(2),key=lambda j:(priority[m,j],-j))) for m in range(4) for a in range(2));proposals,matrix=v.translator(True,rank)
        for i,u in enumerate(proposals):assert v.CAT[int(matrix[i].argmax())]==v.decode(u,arm,state,o)


def test_independent_screen_rejection_boundary_and_json():
    rows=[]
    for a in v.h.ARMS:
        for t in v.h.TRAINS:
            for mode in v.MODES:
                rows.append(dict(method=a,train_seed=t,mode=mode,normalized_cost=99 if a==v.h.CORE[0] and mode=='matching_mode' else 100,availability=.67,normalized_pending=0,normalized_failures=.1,invalid_assignments=0,same_state_mismatches=1))
    assert v.summary(rows)['future_polynomial_decoder_trial_motivated'];json.dumps(v.summary(rows))
    for r in rows:
        if r['method']==v.h.CORE[0] and r['mode']=='matching_mode':r['normalized_cost']=99.6
    assert not v.summary(rows)['future_polynomial_decoder_trial_motivated']
