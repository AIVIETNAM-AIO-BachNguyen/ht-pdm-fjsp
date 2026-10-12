"""Independent crossed-axis resampling and complete statistical-family checks."""
import numpy as np
import pytest

from ht_pdm_fjsp import maintenance_balanced_benchmark_statistics as s


def fixture():
    cfg = dict(train_seeds=[819200, 819201, 819202], test_seeds=[819210, 819211, 819212, 819213],
        analysis_seed=819220, bootstrap_samples=127, confidence=1-.05/28)
    rows = []
    for ai, method in enumerate(s.ALGORITHMS):
        for ti, train in enumerate(cfg['train_seeds']):
            for ei, seed in enumerate(cfg['test_seeds']):
                rows.append(dict(method=method, train_seed=train, seed=seed,
                    extended_cost=25 + 19*ti + 11*ei + 4*ti*ei + 2*ai,
                    availability=.64 + .025*ti - .007*ei - .01*ai,
                    terminal_pending=.1*ti + .05*ei + .02*ai))
    for bi, method in enumerate(s.BASELINES):
        for ei, seed in enumerate(cfg['test_seeds']):
            rows.append(dict(method=method, train_seed=-1, seed=seed,
                extended_cost=40 + 22*ei + 3*bi, availability=.7-.01*ei,
                terminal_pending=.1*ei))
    return cfg, rows


def test_ratio_of_population_means_and_twenty_eight_family():
    cfg, rows = fixture();result = s.summarize(rows, cfg)
    assert result['family_size'] == 28 and len(result['intervals']) == 28
    assert result['confidence'] == 1-.05/28
    for method in s.ALGORITHMS:
        learner = np.mean([r['extended_cost'] for r in rows if r['method'] == method])
        for baseline in s.BASELINES:
            costs = [r['extended_cost'] for r in rows if r['method'] == baseline]
            expected = 1-learner/np.mean(costs)
            assert result['intervals'][f'{method}_cost_vs_{baseline}']['estimate'] == pytest.approx(expected)
        h6 = [r for r in rows if r['method'] == 'isolated_h6']
        learned = [r for r in rows if r['method'] == method]
        for key, metric in (('availability', 'availability'), ('pending', 'terminal_pending')):
            expected = np.mean([r[metric] for r in learned])-np.mean([r[metric] for r in h6])
            assert result['intervals'][f'{method}_{key}_difference']['estimate'] == pytest.approx(expected)
    # Ratios per episode answer a different question and must not replace ratio of means.
    lookup = {(r['method'], r['train_seed'], r['seed']): r for r in rows}
    wrong = np.mean([1-lookup['ppo', train, seed]['extended_cost']/lookup['deadline_matching', -1, seed]['extended_cost']
        for train in cfg['train_seeds'] for seed in cfg['test_seeds']])
    assert abs(wrong-result['intervals']['ppo_cost_vs_deadline_matching']['estimate']) > .01


def test_independent_indexed_bootstrap_matches_crossed_counts_and_shared_axes():
    cfg, rows = fixture();actual = s.summarize(rows, cfg)
    rng = np.random.default_rng(s.role_seed(cfg['analysis_seed'], 'crossed_bootstrap'))
    train_draws = rng.integers(3, size=(127, 3)); episode_draws = rng.integers(4, size=(127, 4))
    lookup = {(r['method'], r['train_seed'], r['seed']): r for r in rows}
    gains = {method: [] for method in s.ALGORITHMS}
    for td, ed in zip(train_draws, episode_draws):
        # Explicitly duplicate cells in the crossed product; no einsum/count formulation.
        base = np.mean([lookup['deadline_matching', -1, cfg['test_seeds'][e]]['extended_cost'] for e in ed])
        for method in s.ALGORITHMS:
            cost = np.mean([lookup[method, cfg['train_seeds'][t], cfg['test_seeds'][e]]['extended_cost'] for t in td for e in ed])
            gains[method].append(1-cost/base)
    tail = (1-cfg['confidence'])/2
    for method, samples in gains.items():
        expected = np.quantile(samples, [tail, 1-tail])
        interval = actual['intervals'][f'{method}_cost_vs_deadline_matching']
        assert [interval['lower'], interval['upper']] == pytest.approx(expected, abs=1e-12)
        assert interval['upper'] > interval['lower']
    assert actual == s.summarize(rows, cfg)
    assert actual == s.summarize(list(reversed(rows)), cfg)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unexpected_train', 'unexpected_test', 'unexpected_method'])
def test_exact_grid_rejects_missing_duplicate_and_extra_identities(mutation):
    cfg, rows = fixture()
    if mutation == 'missing': rows.pop()
    elif mutation == 'duplicate': rows[-1] = dict(rows[0])
    else:
        key, value = {'unexpected_train': ('train_seed', 999), 'unexpected_test': ('seed', 999),
            'unexpected_method': ('method', 'unknown')}[mutation]
        rows[0][key] = value
    with pytest.raises(RuntimeError, match='exact test'):
        s.summarize(rows, cfg)


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf])
def test_nonpositive_nonfinite_cost_is_rejected(value):
    cfg, rows = fixture();rows[0]['extended_cost'] = value
    with pytest.raises(RuntimeError, match='positive finite'):
        s.summarize(rows, cfg)


def test_statistical_practical_labels_and_health_are_separate():
    cfg, rows = fixture()
    for row in rows:
        row['extended_cost'] = 100.
        row['availability'] = .7
        row['terminal_pending'] = 0.
        if row['method'] == 'ppo': row['extended_cost'] = 99.5
        if row['method'] == 'mappo': row['extended_cost'] = 98.
        if row['method'] == 'iql': row['extended_cost'] = 98.;row['availability'] = .68
        if row['method'] == 'qmix': row['extended_cost'] = 98.;row['terminal_pending'] = .11
    result = s.summarize(rows, cfg)
    assert result['guards']['ppo']['statistically_better_than'] == list(s.BASELINES)
    assert result['guards']['ppo']['practically_better_than'] == []
    assert result['guards']['mappo']['practically_better_than'] == list(s.BASELINES)
    assert result['guards']['ppo']['health_pass'] and result['guards']['mappo']['health_pass']
    assert not result['guards']['iql']['health_pass'] and not result['guards']['qmix']['health_pass']
