"""Held-out one-step PM deferral and technician correction on fresh profiles."""
import argparse
import itertools
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_joint_rl as joint
from ht_pdm_fjsp import maintenance_opportunity_cost as prior
from ht_pdm_fjsp import maintenance_timing_diagnostic as previous
from ht_pdm_fjsp import maintenance_policy_continuation as source
from ht_pdm_fjsp import maintenance_boundary_sensitivity as boundary
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_joint_policy import JOINT, CATALOGUE
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv
from ht_pdm_fjsp.maintenance_timing_forks import without_preventive, kind
from ht_pdm_fjsp.maintenance_headroom_forks import fork_roots, scalar_replay
from ht_pdm_fjsp.maintenance_policy_forks import COMPONENTS
from ht_pdm_fjsp.maintenance_timing_statistics import crossed_ratio

VERSION = 'maintenance_decision_headroom_v1'
ROOT = Path(__file__).resolve().parents[2]
KEYS = source.KEYS
RAW_KEYS = ('cohorts', 'physical_shocks', 'selection_seed', 'evaluation_seed', 'analysis_seed')
PHASES = ('selection', 'evaluation')
CONTRASTS = ('timing', 'assignment')
sha = source.sha


def settings(profile):
    if profile not in ('full', 'smoke'):
        raise ValueError('unknown profile')
    full = profile == 'full'
    registry = json.loads((ROOT / f'configs/{VERSION}_seed_registry.json').read_text())
    return dict(profile=profile, **registry['panels'][profile],
                conditions=['dense', 'skill_mask'], origins=[JOINT, 'best_rule'],
                model_seeds=previous.settings('full')['model_seeds'] if full else previous.settings('full')['model_seeds'][:1],
                landmarks=[6, 18, 30, 42] if full else [42],
                selection_scenarios=32 if full else 4, evaluation_scenarios=64 if full else 4,
                horizon=48, tail_ticks=12, bootstrap_samples=20000 if full else 200,
                primary_confidence=.9875, torch_threads=1)


def counts(cfg):
    trajectories = len(cfg['conditions']) * len(cfg['origins']) * len(cfg['cohorts']) * len(cfg['physical_shocks']) * len(cfg['model_seeds'])
    return dict(models=len(cfg['model_seeds']), trajectories=trajectories,
                trajectory_ticks=trajectories * 48, states=trajectories * len(cfg['landmarks']), training_steps=0,
                max_forecast_transitions=trajectories * sum(60-t for t in cfg['landmarks']) * 3 *
                (cfg['selection_scenarios'] + cfg['evaluation_scenarios']))


def physical_seed(cohort, shock):
    return role_seed(shock, f'{VERSION}:physical:{cohort}')


def forecast_seed(cfg, row, phase):
    if phase not in PHASES:
        raise ValueError('unknown forecast phase')
    return role_seed(cfg[f'{phase}_seed'], f"{VERSION}:{phase}:{row['cohort']}:{row['shock']}:{row['model_seed']}:{row['origin']}:{row['tick']}")


def seed_audit(cfg):
    own = ROOT / f'configs/{VERSION}_seed_registry.json'
    historical = joint.integers(joint.exclusions()['historical_raw_panels'])
    for path in (ROOT / 'configs').glob('*seed_registry.json'):
        if path != own:
            historical |= joint.integers(json.loads(path.read_text()))
    fresh, roles = set(), []
    for profile in ('full', 'smoke'):
        panel = settings(profile)
        for key in RAW_KEYS:
            group = joint.integers(panel[key])
            if group & (fresh | historical):
                raise ValueError('new raw seed collision')
            fresh |= group
        for cohort, shock in itertools.product(panel['cohorts'], panel['physical_shocks']):
            roles.append(physical_seed(cohort, shock))
            for seed, origin, tick, phase in itertools.product(panel['model_seeds'], panel['origins'], panel['landmarks'], PHASES):
                roles.append(forecast_seed(panel, dict(cohort=cohort, shock=shock, model_seed=seed, origin=origin, tick=tick), phase))
        roles.append(role_seed(panel['analysis_seed'], f'{VERSION}:crossed_bootstrap'))
    old_physical = {previous.physical_seed(c, s) for c, s in itertools.product(previous.settings('full')['cohorts'], previous.settings('full')['physical_shocks'])}
    if cfg != settings(cfg['profile']) or len(set(roles)) != len(roles) or set(roles) & (fresh | historical | old_physical):
        raise ValueError('derived RNG/settings collision')
    return dict(status='PASS', unique_derived_roles=len(roles), registry_sha256=sha(own),
                intentional_pairing='conditions/models/origins share physical tape; candidate roots share phase tape; selection and evaluation disjoint')


@lru_cache(maxsize=2)
def build_panels(profile):
    cfg = settings(profile)
    blocked = set(previous.exclusion_pool())
    for old_profile in ('full', 'smoke'):
        old, _ = previous.build_panels(previous.settings(old_profile))
        blocked.update(boundary.signature(c) for c in old.values())
    if profile == 'smoke':
        full, _, _ = build_panels('full')
        blocked.update(boundary.signature(c) for c in full.values())
    prior_count = len(blocked)
    panels, rows = {}, []
    for cohort in cfg['cohorts']:
        for attempt in range(10000):
            generated = {cond: joint.make_config(cohort, cond, 48, attempt) for cond in cfg['conditions']}
            signatures = [boundary.signature(c) for c in generated.values()]
            if len(set(signatures)) == len(signatures) and not set(signatures) & blocked:
                break
        else:
            raise RuntimeError('fresh physical pool exhausted')
        for condition, c in generated.items():
            blocked.add(boundary.signature(c))
            panels[cohort, condition] = c
            rows.append(dict(cohort=cohort, condition=condition, generator_attempt=attempt, config=asdict(c)))
    return panels, rows, dict(status='PASS', excluded_signatures=prior_count, fresh_signatures=len(rows),
                             outcomes_used=False, full_smoke_disjoint=True)


def candidates(c, state, rl):
    validate_matching(c, state, rl)
    no_pm = without_preventive(state, rl)
    machines = {m for m, _ in rl}
    assignments = [a for a in joint.feasible(c, state) if {m for m, _ in a} == machines]
    assignments = [rl] + [a for a in assignments if a != rl]
    actions = tuple(dict.fromkeys([rl, no_pm, *assignments]))
    if not 1 <= len(actions) <= 3:
        raise RuntimeError('N4/K2 candidate budget')
    for a in actions:
        validate_matching(c, state, a)
    return actions, {'timing': tuple(dict.fromkeys([0, actions.index(no_pm)])),
                     'assignment': tuple(actions.index(a) for a in assignments)}


def choose(selection_costs, eligible):
    """Only selection data enter this choice; original root wins exact ties."""
    costs = np.asarray(selection_costs)
    if costs.ndim != 2 or not np.isfinite(costs).all() or not eligible or eligible[0] != 0:
        raise ValueError('invalid selection candidates')
    means = costs.mean(axis=1)
    return min(eligible, key=lambda i: means[i])


def heldout_gain(evaluation_costs, selected):
    samples = np.asarray(evaluation_costs)[0] - np.asarray(evaluation_costs)[selected]
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(len(samples)))


def summarize(rows, cfg):
    index = {tuple(r[k] for k in KEYS): r for r in rows}
    grid = set(itertools.product(cfg['conditions'], cfg['cohorts'], cfg['physical_shocks'], cfg['model_seeds'], cfg['origins'], cfg['landmarks']))
    if set(index) != grid or len(index) != len(rows):
        raise RuntimeError('incomplete/duplicate state grid')
    n, p = len(cfg['model_seeds']), len(cfg['cohorts'])
    rng = np.random.default_rng(role_seed(cfg['analysis_seed'], f'{VERSION}:crossed_bootstrap'))
    rd = rng.integers(n, size=(cfg['bootstrap_samples'], n))
    pd = rng.integers(p, size=(cfg['bootstrap_samples'], p))
    primary, cells, strata = {}, [], []
    for condition, contrast in itertools.product(cfg['conditions'], CONTRASTS):
        num, den = np.zeros((n, p)), np.zeros((n, p))
        for i, seed in enumerate(cfg['model_seeds']):
            for j, cohort in enumerate(cfg['cohorts']):
                group = [index[condition, cohort, shock, seed, origin, tick] for shock, origin, tick in itertools.product(cfg['physical_shocks'], cfg['origins'], cfg['landmarks'])]
                for row in group:
                    exposed, gain = row[f'{contrast}_exposure'], row[f'{contrast}_gain']
                    if exposed not in (0, 1) or not np.isfinite(gain) or (not exposed and gain != 0):
                        raise ValueError('invalid conditional gain')
                num[i, j] = sum(r[f'{contrast}_gain'] for r in group)
                den[i, j] = sum(r[f'{contrast}_exposure'] for r in group)
                cells.append(dict(condition=condition, contrast=contrast, model_seed=seed, cohort=cohort,
                                  numerator=float(num[i, j]), exposure=int(den[i, j])))
        draws = crossed_ratio(num, den, rd, pd)
        finite = draws[np.isfinite(draws)]
        empty = float(1 - len(finite)/len(draws))
        estimate = float(num.sum()/den.sum()) if den.sum() else None
        ci = np.quantile(finite, [.00625, .99375]).tolist() if estimate is not None and empty <= .05 else None
        status = 'INSUFFICIENT' if ci is None else 'POSITIVE_CORRECTION_GAIN' if ci[0] > 0 else 'NEGATIVE_CORRECTION_GAIN' if ci[1] < 0 else 'UNRESOLVED'
        eligible = [r for r in rows if r['condition'] == condition and r[f'{contrast}_exposure']]
        condition_rows = sum(r['condition'] == condition for r in rows)
        primary[f'{condition}:{contrast}'] = dict(estimate=estimate, confidence=.9875, confidence_interval=ci,
            exposure=len(eligible), total_states=condition_rows, coverage=len(eligible)/condition_rows,
            switched=sum(r[f'{contrast}_selected'] != 0 for r in eligible), empty_bootstrap_fraction=empty,
            status='ENGINEERING_ONLY' if cfg['profile'] == 'smoke' else status,
            sign='positive favors selection-chosen one-step correction; negative held-out gains retained',
            units='remaining total simulator cost through tick59')
        for origin, tick in itertools.product(cfg['origins'], cfg['landmarks']):
            group = [r for r in eligible if r['origin'] == origin and r['tick'] == tick]
            strata.append(dict(condition=condition, contrast=contrast, origin=origin, tick=tick, exposure=len(group),
                               switched=sum(r[f'{contrast}_selected'] != 0 for r in group),
                               mean_gain=float(np.mean([r[f'{contrast}_gain'] for r in group])) if group else None))
    return dict(primary_contrasts=primary, strata=strata,
                uncertainty='crossed model/profile bootstrap, conditional on independent fixed forecast tapes',
                multiplicity='four contrasts, approximate Bonferroni 98.75% percentile intervals',
                scope='fresh physical profiles; simulator-assisted one-step correction, no new training'), cells


def evaluate(c, state, identity, model, cfg, journals):
    rl, policy = model.select(c, state, 48-identity['tick'], 48)
    actions, groups = candidates(c, state, rl)
    seeds = {phase: forecast_seed(cfg, identity, phase) for phase in PHASES}
    selection = fork_roots(c, state, actions, model, seeds['selection'], cfg['selection_scenarios'], identity['tick'], record_trace=cfg['profile']=='smoke')
    selected = {name: choose(selection['costs'][0], group) for name, group in groups.items()}
    # Choices are fixed before any independent evaluation cost is generated.
    evaluation = fork_roots(c, state, actions, model, seeds['evaluation'], cfg['evaluation_scenarios'], identity['tick'], record_trace=cfg['profile']=='smoke')
    values = (selection, evaluation)
    if cfg['profile'] == 'smoke':
        for v in values:
            scalar_replay(c, state, v, model, identity['tick'])
    row = dict(**identity, state=prior.compact(asdict(state)), physical_seed=physical_seed(identity['cohort'], identity['shock']),
               rl_action=prior.compact(rl), rl_kind=kind(state, rl), action_index=policy['action_index'],
               actions=prior.compact(actions), unique_roots=len(actions),
               forecast_transitions=sum(v['transitions'] for v in values),
               selection_seed=seeds['selection'], evaluation_seed=seeds['evaluation'],
               remaining_physical_ticks=60-identity['tick'], structural_choice=int(len(joint.feasible(c, state))>1))
    costs = evaluation['costs'][0]
    for name, group in groups.items():
        i = selected[name]
        gain, se = heldout_gain(costs, i)
        row.update({f'{name}_exposure': int(len(group)>1), f'{name}_selected': i,
                    f'{name}_candidates': prior.compact(group), f'{name}_gain': gain, f'{name}_mc_se': se,
                    f'{name}_normalized_gain': gain/(4*(60-identity['tick'])),
                    f'{name}_selection_gain': float((selection['costs'][0, 0]-selection['costs'][0, i]).mean())})
    row['direct_pm_deferral_gain'], row['direct_pm_deferral_mc_se'] = heldout_gain(costs, groups['timing'][-1])
    for phase, v in zip(PHASES, values, strict=True):
        for i, action in enumerate(actions):
            journals['root_values'].add(dict(**identity, phase=phase, root_index=i, action_index=CATALOGUE.index(action),
                action=prior.compact(action), scenarios=v['costs'].shape[-1], total_mean=float(v['costs'][0, i].mean()),
                base_mean=float(v['base'][0, i].mean()), tail_mean=float(v['tail'][0, i].mean()),
                **{k: float(v['component_costs'][0][k][i].mean()) for k in COMPONENTS}))
        journals['rollout_audits'].add(dict(**identity, phase=phase, feasibility='PASS', components='PASS',
            clock='H48_L12_PASS', scalar_replay='PASS' if cfg['profile']=='smoke' else 'NOT_SCHEDULED',
            transitions=v['transitions'], **v['counters']))
    return row, values


def padded(v, name, scenarios):
    a, s = len(v['actions']), v['costs'].shape[-1]
    out = np.full((3, scenarios), np.nan)
    out[:a, :s] = v[name][0]
    return out


def run(output, model_source_dir, profile='smoke', device='cpu'):
    if device != 'cpu':
        raise ValueError('CPU only')
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('new/empty artifact directory required')
    output.mkdir(parents=True, exist_ok=True)
    cfg = settings(profile)
    torch.set_num_threads(1)
    now = lambda: datetime.now(timezone.utc).isoformat()
    git = lambda *args: subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    start = time.perf_counter()
    manifest = dict(protocol=VERSION, status='RUNNING', started=now(), commit=git('rev-parse', 'HEAD'),
        dirty=bool(git('status', '--porcelain')), settings=cfg, expected_counts=counts(cfg), device=device,
        platform=platform.platform(), python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
        new_full_physical_outcomes_opened=False, historical_source_outcomes_previously_opened=True,
        thread_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    prior.write_json(output/'manifest.json', manifest)
    journals, rows, raw, boundaries = {}, [], {}, {}
    transitions, tail_rows, decisions, ties = 0, 0, 0, 0
    histogram = np.zeros(21, dtype=int)
    try:
        if profile == 'full' and (manifest['dirty'] or any(v != '1' for v in manifest['thread_environment'].values())):
            raise ValueError('full requires clean committed code and OMP/OPENBLAS/MKL_NUM_THREADS=1')
        locked = json.loads((ROOT/f'configs/{VERSION}_sources.json').read_text())
        if locked['protocol'] != VERSION:
            raise ValueError('source protocol mismatch')
        models, _, _, audit = source.verify_source(model_source_dir, locked)
        manifest['source_audit'] = audit
        manifest['seed_audit'] = seed_audit(cfg)
        panels, registry, physical_audit = build_panels(profile)
        manifest['physical_audit'] = physical_audit
        prior.write_json(output/'source_contract.json', locked)
        prior.write_json(output/'resolved_config.json', dict(settings=cfg, physical_profiles=registry))
        shutil.copyfile(ROOT/f'configs/{VERSION}_seed_registry.json', output/'seed_registry.json')
        shutil.copyfile(ROOT/f'docs/{VERSION}_plan.md', output/'plan.md')
        snapshot = output/'source_snapshot'
        snapshot.mkdir()
        for path in list((ROOT/'src/ht_pdm_fjsp').glob('maintenance_*.py')) + list((ROOT/'configs').glob('maintenance_*json')) + [ROOT/'pyproject.toml', ROOT/'uv.lock']:
            shutil.copyfile(path, snapshot/path.name)
        initial = {}
        for seed in cfg['model_seeds']:
            dest = output/'frozen_models'/f'seed_{seed}.pt'
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(Path(model_source_dir)/'frozen_models'/dest.name, dest)
            restored, _ = joint.load_checkpoint(dest)
            if sha(dest) != locked['joint_source_contract']['models'][str(seed)]['sha256'] or previous.parameters_sha(restored) != previous.parameters_sha(models[seed]):
                raise RuntimeError('checkpoint roundtrip mismatch')
            for c in panels.values():
                if restored.select(c, WaitingEnv(c).state, 48, 48) != models[seed].select(c, WaitingEnv(c).state, 48, 48):
                    raise RuntimeError('checkpoint policy roundtrip mismatch')
            models[seed] = restored
            initial[seed] = previous.parameters_sha(restored)
        for name in ('trajectories', 'states.partial', 'root_values', 'rollout_audits'):
            journals[name] = joint.CountingJournal(output/f'{name}.csv')
        manifest['new_full_physical_outcomes_opened'] = profile == 'full'
        prior.write_json(output/'manifest.json', manifest)
        print(json.dumps(dict(protocol=VERSION, budget=counts(cfg), training='NONE'), indent=2), flush=True)
        combos = list(itertools.product(cfg['model_seeds'], cfg['cohorts'], cfg['physical_shocks'], cfg['conditions'], cfg['origins']))
        width = max(cfg['selection_scenarios'], cfg['evaluation_scenarios'])
        with tqdm(total=counts(cfg)['states'], desc='Independent correction evaluation') as progress:
            for seed, cohort, shock, condition, origin in tqdm(combos, desc='Frozen multi-seed trajectories'):
                c = panels[cohort, condition]
                env = WaitingEnv(c)
                env.reset(physical_seed(cohort, shock))
                rule = locked['joint_source_contract']['selected_rules'][condition]
                for tick in range(48):
                    state = env.state
                    identity = dict(condition=condition, cohort=cohort, shock=shock, model_seed=seed, origin=origin, tick=tick)
                    if tick in cfg['landmarks']:
                        row, values = evaluate(c, state, identity, models[seed], cfg, journals)
                        rows.append(row)
                        journals['states.partial'].add(row)
                        for name in ('costs', 'base', 'tail'):
                            raw.setdefault(name, []).append(np.stack([padded(v, name, width) for v in values]))
                        for component in COMPONENTS:
                            raw.setdefault(component, []).append(np.stack([padded(dict(v, component=v['component_costs'][0][component][None]), 'component', width) for v in values]))
                        mask = np.zeros((2, 3, width), dtype=bool)
                        for phase, v in enumerate(values):
                            mask[phase, :len(v['actions']), :v['costs'].shape[-1]] = True
                            transitions += v['transitions']
                            tail_rows += v['counters']['tail_transition_rows']
                            decisions += v['counters']['batched_decision_rows']
                            ties += v['counters']['near_tie_scalar_rows']
                            histogram += v['counters']['rl_action_histogram']
                        raw.setdefault('valid', []).append(mask)
                        for stage in ('cutoff', 'terminal'):
                            for name in values[0][stage][0]:
                                dim = values[0][stage][0][name].shape[-1]
                                result = np.full((2, 3, width, dim), -1, dtype=np.int64)
                                for phase, v in enumerate(values):
                                    result[phase, :len(v['actions']), :v['costs'].shape[-1]] = v[stage][0][name]
                                boundaries.setdefault(f'{stage}_{name}', []).append(result)
                        progress.update()
                    pairs = models[seed].select(c, state, 48-tick, 48)[0] if origin == JOINT else prior.rule_action(c, state, 48-tick, rule)
                    validate_matching(c, state, pairs)
                    _, reward, _, info = env.step(pairs)
                    journals['trajectories'].add(dict(**identity, state=prior.compact(asdict(state)), action=prior.compact(pairs),
                        cost=-reward, following=prior.compact(asdict(env.state)), info=prior.compact(info)))
        if any(previous.parameters_sha(models[s]) != value for s, value in initial.items()):
            raise RuntimeError('frozen parameters changed')
        summary, cells = summarize(rows, cfg)
        prior.write_json(output/'summary.json', summary)
        prior.write_csv(output/'states.csv', rows)
        prior.write_csv(output/'profile_cells.csv', cells)
        prior.write_csv(output/'strata.csv', summary['strata'])
        np.savez_compressed(output/'raw_forecast_costs.npz', **{k: np.stack(v) for k, v in raw.items()})
        np.savez_compressed(output/'raw_boundary_states.npz', **{k: np.stack(v) for k, v in boundaries.items()})
        roots = sum(r['unique_roots'] for r in rows)
        scenarios = cfg['selection_scenarios'] + cfg['evaluation_scenarios']
        expected_transitions = sum(r['unique_roots']*scenarios*(60-r['tick']) for r in rows)
        expected_decisions = sum(r['unique_roots']*scenarios*(47-r['tick']) for r in rows)
        if (len(rows) != counts(cfg)['states'] or journals['trajectories'].rows != counts(cfg)['trajectory_ticks'] or
            journals['states.partial'].rows != len(rows) or journals['root_values'].rows != 2*roots or
            journals['rollout_audits'].rows != 2*len(rows) or transitions > counts(cfg)['max_forecast_transitions'] or
            tail_rows != roots*scenarios*12 or transitions != expected_transitions or
            decisions != expected_decisions or histogram.sum() != decisions):
            raise RuntimeError('budget/counter audit mismatch')
        manifest.update(status='COMPLETED', audit='PASS', frozen_parameter_hashes={str(k): v for k, v in initial.items()},
            actual_counts=dict(states=len(rows), models=len(initial), trajectories=len(combos), trajectory_ticks=journals['trajectories'].rows,
                               unique_roots=roots, forecast_transitions=transitions, tail_rows=tail_rows, batched_decision_rows=decisions,
                               near_tie_scalar_rows=ties, rl_action_histogram=histogram.tolist(), training_steps=0))
    except BaseException as error:
        manifest.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for journal in journals.values():
            journal.close()
        manifest['finished'] = now()
        manifest['elapsed_seconds'] = time.perf_counter()-start
        manifest['artifacts'] = {str(p.relative_to(output)): dict(sha256=sha(p), bytes=p.stat().st_size)
                                for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
        prior.write_json(output/'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    parser.add_argument('--model-source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir, args.model_source_dir, args.profile, args.device)


if __name__ == '__main__':
    main()
