import pytest

from ht_pdm_fjsp import technician_diverse_maintenance_confirmation as c
from ht_pdm_fjsp import technician_diverse_maintenance_experiment as x


def test_scientific_confirmation_requires_passed_development(tmp_path):
    assert len(c.settings('confirmation')['train_seeds'])==5
    assert len(c.settings('confirmation')['development_seeds'])==64
    assert len(c.settings('confirmation')['test_seeds'])==256
    with pytest.raises((FileNotFoundError,AssertionError,RuntimeError)):
        c.prior_gate('confirmation',
            c.ROOT/'artifacts'/'nonexistent_development_artifact')
    (tmp_path/'manifest.json').write_text('{"protocol":"technician_diverse_maintenance_v1","status":"COMPLETED","profile":"smoke","test_opened":true}')
    (tmp_path/'summary.json').write_text('{"development_screen_pass":true}')
    (tmp_path/'audit.json').write_text('{"status":"PASS"}')
    with pytest.raises(RuntimeError,match='development gate failed'):
        c.prior_gate('confirmation',tmp_path)


def test_crossed_inference_uses_matched_mask_and_frozen_rule():
    cfg=dict(c.settings('smoke'),test_seeds=[66000000,66000001])
    rows=[];coordination=[]
    for family,n in map(tuple,cfg['configs']):
        for arm in (*x.ARMS,'fcfs','risk'):
            for train in (cfg['train_seeds'] if arm in x.ARMS else [-1]):
                for seed in cfg['test_seeds']:
                    cost=98. if arm=='reservation_ppo' else 100.
                    rows.append(dict(family=family,n=n,method=arm,train_seed=train,
                        seed=seed,normalized_cost=cost,availability=.7,
                        normalized_pending=.1,normalized_failures=.2,
                        invalid_assignments=0))
                    if arm=='reservation_ppo':
                        coordination.append(dict(family=family,n=n,method=arm,
                            decision_ms=1.,environment_ms=2.))
    result=c.summarize(rows,cfg,'fcfs',coordination)
    assert result['goal_pass']
    assert result['intervals']['pooled_gain_full_vs_mask']['estimate']==pytest.approx(.02)
    assert result['latency']['model_select_p95_ms']==1.
    for row in rows:
        if row['method']=='reservation_ppo':row['normalized_cost']=100.
    assert not c.summarize(rows,cfg,'fcfs',coordination)['goal_pass']
