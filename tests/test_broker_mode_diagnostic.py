import itertools,json
from dataclasses import asdict
import numpy as np
import pytest
import torch
from ht_pdm_fjsp import technician_broker_mode_diagnostic as d


def preimages(logp,legal,priority):
    groups={};agents,width=logp.shape
    for proposal in itertools.product(*[[0]+[r+1 for r in range(width-1) if legal[r,a]] for a in range(agents)]):
        mass=float(np.exp(sum(logp[a,t] for a,t in enumerate(proposal))));out=[0]*agents
        for token in set(proposal)-{0}:
            a=max((a for a,t in enumerate(proposal) if t==token),key=lambda a:(priority[token-1,a],-a));out[a]=token
        groups[tuple(out)]=groups.get(tuple(out),0)+mass
    return groups


@pytest.mark.parametrize('agents,resources',[(2,4),(4,2)])
@pytest.mark.parametrize('masked',[False,True])
@pytest.mark.parametrize('tied',[False,True])
def test_generic_core_and_transposed_machine_mass_matches_all_preimages(agents,resources,masked,tied):
    raw=np.cos(np.arange(agents*(resources+1)).reshape(agents,resources+1)*.6);legal=np.ones((resources,agents),bool)
    if masked:legal[:,0]=False;legal[0,-1]=False
    support=np.c_[np.ones(agents,bool),legal.T];raw=np.where(support,raw,-np.inf);logp=raw-np.logaddexp.reduce(raw,axis=1)[:,None]
    priority=np.zeros((resources,agents)) if tied else np.array([[(a+r)%agents for a in range(agents)] for r in range(resources)]);groups=preimages(logp,legal,priority)
    assert sum(groups.values())==pytest.approx(1,abs=2e-12)
    for matching,mass in groups.items():assert np.exp(d.generic_logmass(logp,matching,priority,legal))==pytest.approx(mass,abs=2e-12,rel=2e-12)


def test_physical_mode_differs_from_greedy_in_locked_counterexample():
    h=d.helpers();state=dict(ages=[3]*4,failed=[False]*4,pending_wait=[0]*4,remaining=[0]*2,assigned=[-1]*2);o=d.complete_obs(h,state,0)
    p=np.array([[.01,.60,.30,.04,.05],[.30,.29,.31,.05,.05]])
    class Actor:
        def select(self,obs):return p.argmax(1)
        def proposal_log_probs(self,batch):return torch.tensor(np.log(p),dtype=torch.float32)[None],None
    result=d.query(Actor(),d.x.CORE_ARMS[0],state,o,h)
    assert result['greedy_pairs']==((0,0),(1,1)) and result['mode_pairs']==((0,0),) and result['mismatch']==1
    assert result['mode_logmass']>result['greedy_logmass']


@pytest.mark.parametrize('arm',d.x.ARMS)
def test_actual_frozen_actor_inputs_scalar_physics_and_both_short_rollouts(arm):
    torch.set_num_threads(1);h=d.helpers();c=d.x.settings('engineering');t=c['train_seeds'][0];model=d.x.make_model(arm,c,t);env=d.x.ResourceEnv(290700001,4,arm);actual=env.reset();state={k:list(v) for k,v in asdict(env.env.state).items()};o=d.complete_obs(h,state,0)
    # The reference supplies the union of method-specific actor inputs.
    for key in actual:np.testing.assert_array_equal(actual[key],o[key],err_msg=key)
    np.testing.assert_array_equal(model.select(actual),model.select(o));before=d.x.learner.parameter_digest(model)
    greedy=d.collect(model,arm,t,[290700001],'greedy',h);mode=d.collect(model,arm,t,[290700001],'matching_mode',h)
    assert all(r['physical_ticks']==60 and r['invalid_assignments']==0 for r in greedy+mode)
    assert before==d.x.learner.parameter_digest(model)
    if arm in ('ppo','normalized_ppo'):
        for key in ('normalized_cost','availability','normalized_pending','normalized_failures'):assert greedy[0][key]==mode[0][key]


def test_seeds_closed_panels_and_screen_boundaries():
    audit=d.seed_audit();assert audit['raw_unique']==64 and audit['derived_unique']==64
    h=d.helpers();rows=[]
    for a in d.x.ARMS:
        for t in h.TRAINS:
            for mode in ('greedy','matching_mode'):
                rows.append(dict(method=a,train_seed=t,mode=mode,normalized_cost=99 if a==d.x.CORE_ARMS[0] and mode=='matching_mode' else 100,
                    availability=.67,normalized_pending=0,normalized_failures=.1,invalid_assignments=0,same_state_mismatches=1))
    result=d.summarize(rows,h);assert result['future_polynomial_decoder_trial_motivated'] and len(result['mode_checks'])==13
    json.dumps(result)
    for row in rows:
        if row['method']==d.x.CORE_ARMS[0] and row['mode']=='matching_mode':row['normalized_cost']=99.6
    assert not d.summarize(rows,h)['screen_checks']['own_cost_gain_at_least_half_percent']
