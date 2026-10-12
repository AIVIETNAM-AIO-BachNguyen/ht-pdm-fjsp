"""Fresh fixed-profile confirmation cannot reselect or overwrite an outcome."""
import json
import pytest
from ht_pdm_fjsp import maintenance_balanced_confirmation_mac as s


def test_fixed_balanced_profile_baseline_and_fresh_roles():
    spec = s.fixed_spec(); c = s.engine.core.make_config(spec)
    assert spec['candidate'] == 'cross_d3_q050_balanced'
    assert c.service_time == ((1,3),(1,3),(3,1),(3,1))
    assert c.restoration == ((.5,1),(.5,1),(1,.5),(1,.5))
    assert s.frozen_choices()[s.CANDIDATE]['base'] == 'isolated_endpoint'
    assert len(s.settings('full')['confirmation_seeds']) == 256
    assert s.seed_audit(s.settings('full'))['status'] == 'PASS'
    assert {100,101,102}.issubset(s.registered_integers({'seed_start':100,'seed_count':3}))
    with pytest.raises(ValueError, match='locked'):
        s.seed_audit({**s.settings('full'), 'scenarios':64})


def test_real_balanced_smoke_replays_complete_grid_and_preserves_old_screen_status(tmp_path):
    out = tmp_path/'run'; m = s.run('smoke', out)
    assert m['status'] == 'COMPLETED' and m['audit'] == 'PASS'
    assert m['scientific_status'] == 'ENGINEERING_ONLY'
    assert m['actual_counts'] == dict(development=0,screening=0,confirmation=10,landmarks=14)
    summary = json.loads((out/'summary.json').read_text())
    assert not summary['prior_screen_passed'] and summary['base'] == 'isolated_endpoint'
    assert len(summary['gates']) == 18
    assert (out/'episodes.partial.csv').read_bytes() == (out/'episodes.csv').read_bytes()
    for name, data in m['files'].items():
        assert s.engine.f.old.screen.sha(out/name) == data['sha256']
    with pytest.raises(FileExistsError):
        s.run('smoke', out)
