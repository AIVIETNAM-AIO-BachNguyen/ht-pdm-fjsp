"""Fresh standalone confirmation of the fixed balanced 2–2 skill profile."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
from tqdm.auto import tqdm
from ht_pdm_fjsp import maintenance_cross_skill_mac as engine

VERSION = 'maintenance_balanced_confirmation_mac_v1'
ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'docs' / f'{VERSION}_plan.md'
REGISTRY = ROOT / 'configs' / f'{VERSION}_seed_registry.json'
CANDIDATE = 'cross_d3_q050_balanced'
BASE = 'isolated_endpoint'
SOURCES = ('maintenance_balanced_confirmation_mac.py', *engine.SOURCES)


def settings(profile):
    if profile not in ('smoke', 'full'):
        raise ValueError('unknown profile')
    raw = json.loads(REGISTRY.read_text())['panels'][profile]
    return dict(profile=profile, horizon=48, tail_ticks=12,
        confirmation_seeds=list(range(raw['confirmation_start'], raw['confirmation_start'] + raw['confirmation_count'])),
        analysis_seed=raw['analysis_seed'], forecast_seed=raw['forecast_seed'],
        scenarios=128 if profile == 'full' else 8,
        bootstrap_samples=20000 if profile == 'full' else 200,
        confidence=1-.05/18, max_seconds=7200 if profile == 'full' else 600,
        audit_seeds=2, landmarks=list(engine.LANDMARKS), training='NOT_APPLICABLE',
        train_seeds=[], development_seeds=[], screening_seeds=[])


def registered_integers(value):
    """Expand start/count ranges as well as explicitly listed historical seeds."""
    values = engine.f.old.integers(value)
    if isinstance(value, dict):
        for key, child in value.items():
            if key.endswith('_start') and key[:-6] + '_count' in value:
                values.update(range(child, child + value[key[:-6] + '_count']))
            values |= registered_integers(child)
    elif isinstance(value, list):
        for child in value:
            values |= registered_integers(child)
    return values


def seed_audit(cfg):
    if cfg != settings(cfg['profile']):
        raise ValueError('locked settings')
    historical = set(); hashes = {}; groups = []; derived = []
    for path in sorted((ROOT / 'configs').glob('*seed_registry.json')):
        if path != REGISTRY:
            historical |= registered_integers(json.loads(path.read_text()))
            hashes[path.name] = engine.f.old.screen.sha(path)
    for profile in ('smoke', 'full'):
        c = settings(profile)
        groups.extend((set(c['confirmation_seeds']), {c['analysis_seed']}, {c['forecast_seed']}))
        derived.extend(engine.physical_seed(s) for s in c['confirmation_seeds'])
        derived.extend(engine.bank_seed(c, 'confirmation', s, tick, held)
            for s in c['confirmation_seeds'] for tick in range(48) for held in (False, True))
        derived.append(engine.role_seed(c['analysis_seed'], f'{engine.VERSION}:bootstrap'))
    own = set.union(*groups)
    if any(a & b for a, b in itertools.combinations(groups, 2)) or own & historical:
        raise ValueError('raw seed overlap')
    if len(derived) != len(set(derived)) or set(derived) & (own | historical):
        raise ValueError('derived seed overlap')
    return dict(status='PASS', raw_unique=len(own), derived_unique=len(derived),
        historical_registry_hashes=hashes, stream_namespace=engine.VERSION,
        intentional_pairing='same physical raw seed across methods; distinct selection/holdout banks')


def fixed_spec():
    return next(s for s in engine.core.specs('full') if s['candidate'] == CANDIDATE)


def frozen_choices():
    registry = json.loads(REGISTRY.read_text())
    if registry['candidate'] != CANDIDATE or registry['base'] != BASE:
        raise ValueError('locked candidate/baseline')
    choice = registry['prior_baseline_choice']
    if choice['base'] != BASE or min(choice['development_means'],
            key=lambda m: (choice['development_means'][m], m)) != BASE:
        raise ValueError('prior development baseline')
    return {CANDIDATE: choice}


def run(profile, output_dir, device='cpu'):
    if device != 'cpu':
        raise ValueError('CPU only')
    cfg = settings(profile); seed_info = seed_audit(cfg)
    choices = frozen_choices(); spec = fixed_spec()
    git = engine.f.old.screen.git_metadata(); out = Path(output_dir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError('refusing overwrite')
    if profile == 'full' and git['dirty']:
        raise RuntimeError('clean committed source required')
    out.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    def timeout():
        if time.monotonic() - started > cfg['max_seconds']:
            raise TimeoutError('registered hard timeout')
    manifest = dict(protocol=VERSION, status='RUNNING', git=git, profile=profile,
        confirmation_outcomes_opened=False, started_utc=datetime.now(timezone.utc).isoformat(),
        runtime=dict(platform=platform.platform(), python=platform.python_version(), numpy=np.__version__, device=device))
    engine.write_json(out / 'manifest.json', manifest)
    try:
        engine.write_json(out / 'candidates.json', [spec])
        engine.write_json(out / 'resolved_config.json', cfg)
        shutil.copy2(PLAN, out / 'protocol.md'); shutil.copy2(REGISTRY, out / 'registry.json')
        snap = out / 'source_snapshot'; snap.mkdir()
        for name in SOURCES:
            shutil.copy2(Path(__file__).parent / name, snap / name)
        rs = out / 'registry_snapshot'; rs.mkdir()
        for name in seed_info['historical_registry_hashes']:
            shutil.copy2(ROOT / 'configs' / name, rs / name)
        seals = {str(p.relative_to(out)): engine.f.old.screen.sha(p)
            for p in (out / 'protocol.md', out / 'registry.json', *snap.iterdir(), *rs.iterdir())}
        selection = dict(winner=CANDIDATE, status='AUTHOR_REQUESTED_FIXED_CONFIRMATION',
            confirmation_outcomes_opened=False, prior_screen_passed=False,
            reason='balanced 2–2 requested after earlier screening; no new selection')
        engine.write_json(out / 'baseline_choices.json', choices)
        engine.write_json(out / 'selection.json', selection)
        manifest.update(source_seal=seals, winner=CANDIDATE,
            baseline_choices_sha256=engine.f.old.screen.sha(out / 'baseline_choices.json'),
            selection_sha256=engine.f.old.screen.sha(out / 'selection.json'),
            confirmation_outcomes_opened=profile == 'full', stage='confirmation')
        engine.write_json(out / 'manifest.json', manifest)
        engine.f.dp.tables(engine.core.make_config(spec))
        rows = []; local = []
        journal = engine.Journal(out / 'decisions.csv')
        partial = engine.Journal(out / 'episodes.partial.csv')
        roots = engine.Journal(out / 'landmark_roots.csv')
        labels = engine.Journal(out / 'landmarks.csv')
        try:
            for seed, method in tqdm(list(itertools.product(cfg['confirmation_seeds'], engine.METHODS)),
                    desc='Confirm fixed balanced 2–2', unit='episode'):
                row, states = engine.episode(cfg, spec, method, seed, BASE, 'confirmation', journal, timeout)
                rows.append(row); partial.add(row)
                for tick, state in states.items():
                    loc = engine.landmark(cfg, spec, state, seed, tick, BASE, 'confirmation', roots, timeout)
                    local.append(loc); labels.add(loc)
        finally:
            journal.close(); partial.close(); roots.close(); labels.close()
        engine.write_csv(out / 'episodes.csv', rows)
        audit = engine.replay(cfg, [spec], choices, out / 'decisions.csv', rows, local,
            list(engine.records(out / 'landmark_roots.csv')), timeout)
        expected = dict(development=0, screening=0, confirmation=len(cfg['confirmation_seeds'])*5,
            landmarks=len(cfg['confirmation_seeds'])*7)
        actual = dict(development=0, screening=0, confirmation=len(rows), landmarks=len(local))
        cap = len(cfg['confirmation_seeds'])*21*cfg['scenarios']*(
            2*sum(60-t for t in range(48)) + 2*sum(60-t for t in engine.LANDMARKS))
        if actual != expected or audit['forecast_transitions'] > cap or any(r['extended_invalid_assignments'] for r in rows):
            raise RuntimeError('counts/feasibility/budget')
        for name, digest in seals.items():
            if engine.f.old.screen.sha(out / name) != digest:
                raise RuntimeError('source seal')
        for name in SOURCES:
            if engine.f.old.screen.sha(Path(__file__).parent / name) != seals[f'source_snapshot/{name}']:
                raise RuntimeError('live source')
        for name, digest in seed_info['historical_registry_hashes'].items():
            if engine.f.old.screen.sha(ROOT / 'configs' / name) != digest:
                raise RuntimeError('registry drift')
        if engine.f.old.screen.sha(PLAN) != seals['protocol.md'] or engine.f.old.screen.sha(REGISTRY) != seals['registry.json']:
            raise RuntimeError('live protocol')
        if choices != frozen_choices() or engine.f.old.screen.sha(out / 'baseline_choices.json') != manifest['baseline_choices_sha256'] or engine.f.old.screen.sha(out / 'selection.json') != manifest['selection_sha256']:
            raise RuntimeError('frozen choices')
        summary = engine.summarize(cfg, spec, BASE, rows, local)
        summary.update(selection=selection, standalone_author_requested=True, prior_screen_passed=False)
        engine.write_json(out / 'summary.json', summary)
        audit.update(expected_counts=expected, actual_counts=actual, forecast_cap=cap, seed_audit=seed_info,
            selection_frozen=True, source_seal_unchanged=True)
        engine.write_json(out / 'audit.json', audit)
        files = {str(p.relative_to(out)): dict(sha256=engine.f.old.screen.sha(p), bytes=p.stat().st_size)
            for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
        timeout()
        manifest.update(status='COMPLETED', audit='PASS', scientific_status=summary['scientific_status'],
            elapsed_seconds=time.monotonic()-started, actual_counts=actual, files=files,
            artifact_bytes=sum(b['bytes'] for b in files.values()))
        engine.write_json(out / 'manifest.json', manifest)
        return manifest
    except Exception as error:
        manifest.update(status='FAILED', error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        engine.write_json(out / 'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('smoke', 'full'), required=True)
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); manifest = run(args.profile, args.output_dir, args.device)
    print(json.dumps({k: manifest[k] for k in ('status', 'audit', 'scientific_status', 'actual_counts', 'elapsed_seconds')}, indent=2))

if __name__ == '__main__':
    main()
