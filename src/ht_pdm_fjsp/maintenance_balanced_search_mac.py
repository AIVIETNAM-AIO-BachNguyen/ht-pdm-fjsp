"""Prospective balanced 2–2 search with health-first selection and fresh confirmation."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import itertools
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
from tqdm.auto import tqdm
from ht_pdm_fjsp import maintenance_cross_skill_mac as engine
from ht_pdm_fjsp import maintenance_balanced_confirmation_mac as prior
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig

VERSION = 'maintenance_balanced_search_mac_v1'
ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'docs'/f'{VERSION}_plan.md'
REGISTRY = ROOT/'configs'/f'{VERSION}_seed_registry.json'
core = engine.core
f = engine.f
CHEAP = engine.CHEAP
METHODS = engine.METHODS
LANDMARKS = engine.LANDMARKS
SOURCES = ('maintenance_balanced_search_mac.py', 'maintenance_balanced_confirmation_mac.py', *engine.SOURCES)
Journal = engine.Journal
write_json = engine.write_json
write_csv = engine.write_csv
records = engine.records
episode = engine.episode
landmark = engine.landmark
baseline_choices = engine.baseline_choices
replay = engine.replay
summarize = engine.summarize


def settings(profile):
    if profile not in ('smoke', 'full'):
        raise ValueError('unknown profile')
    registry = json.loads(REGISTRY.read_text())
    if registry['round'] != 1 or registry['family_alpha'] != .025 or registry['confidence'] != 1-.025/18 or registry['bootstrap_samples'] != 80000:
        raise ValueError('locked spending')
    raw = registry['panels'][profile]
    return dict(profile=profile, horizon=48, tail_ticks=12, scenarios=128 if profile=='full' else 8,
        **{f'{p}_seeds':list(range(raw[f'{p}_start'],raw[f'{p}_start']+raw[f'{p}_count'])) for p in ('development','screening','confirmation')},
        analysis_seed=raw['analysis_seed'], forecast_seed=raw['forecast_seed'],
        bootstrap_samples=80000 if profile=='full' else 200, confidence=1-.025/18,
        max_seconds=7200 if profile=='full' else 600, audit_seeds=2,
        landmarks=list(LANDMARKS), training='NOT_APPLICABLE', train_seeds=[],
        search_series=registry['series'], round=1, family_alpha=.025, stream_namespace=engine.VERSION)


def specifications(profile):
    if profile not in ('full', 'smoke'):
        raise ValueError('unknown profile')
    anchor = core.make_config(prior.fixed_spec())
    result = []
    for d, q in itertools.product((2,3,4), (.625,.75)):
        c = WaitingConfig(**{**asdict(anchor),
            'service_time':((1,d),(1,d),(d,1),(d,1)),
            'restoration':((q,1),(q,1),(1,q),(1,q))})
        c.validate()
        result.append(dict(candidate=f'balanced_d{d}_q{round(q*1000):03d}', selectable=True, config=asdict(c)))
    if profile == 'smoke':
        result = [s for s in result if s['candidate'] in ('balanced_d2_q625','balanced_d4_q750')]
    result.append(dict(candidate='old_balanced_d3_q050', selectable=False, config=asdict(anchor)))
    exchangeable = WaitingConfig(**{**asdict(anchor), 'service_time':((1,1),)*4, 'restoration':((.75,.75),)*4})
    result.append(dict(candidate='exchangeable', selectable=False, config=asdict(exchangeable)))
    return result


def seed_audit(cfg):
    if cfg != settings(cfg['profile']):
        raise ValueError('locked settings')
    historical = set(); hashes = {}; groups = []; derived = []
    for path in sorted((ROOT/'configs').glob('*seed_registry.json')):
        if path != REGISTRY:
            historical |= prior.registered_integers(json.loads(path.read_text()))
            hashes[path.name] = f.old.screen.sha(path)
    for profile in ('smoke','full'):
        c = settings(profile)
        groups.extend(set(c[f'{p}_seeds']) for p in ('development','screening','confirmation'))
        groups.extend(({c['analysis_seed']}, {c['forecast_seed']}))
        derived.extend(engine.physical_seed(s) for p in ('development','screening','confirmation') for s in c[f'{p}_seeds'])
        derived.extend(engine.bank_seed(c,p,s,t,h) for p in ('screening','confirmation') for s in c[f'{p}_seeds'] for t in range(48) for h in (False,True))
        derived.append(engine.role_seed(c['analysis_seed'],f'{engine.VERSION}:bootstrap'))
    own = set.union(*groups)
    if any(a & b for a,b in itertools.combinations(groups,2)) or own & historical:
        raise ValueError('raw seed overlap')
    if len(derived)!=len(set(derived)) or set(derived)&(own|historical):
        raise ValueError('derived seed overlap')
    return dict(status='PASS',raw_unique=len(own),derived_unique=len(derived),historical_registry_hashes=hashes,
        stream_namespace=engine.VERSION,intentional_pairing='same raw physical and forecast seeds across profiles/methods')


def screen_choice(cfg,specs,choices,rows,local):
    # Preserve all prior eligibility gates; change ONLY prospective winner ranking.
    selection = engine.screen_choice(cfg,specs,choices,rows,local)
    for score in selection['scores']:
        values,_ = engine.matrices(rows,local,cfg['screening_seeds'],score['candidate'])
        base = METHODS.index(choices[score['candidate']]['base'])
        health = []
        for method in engine.PLANNERS:
            paired = values[:,METHODS.index(method),1]-values[:,base,1]
            health.append(float(paired.mean()-2*paired.std(ddof=1)/np.sqrt(len(paired))))
        score['health_stability_score'] = min(health)
    eligible = [s for s in selection['scores'] if s['eligible']]
    if cfg['profile']=='smoke':
        eligible = [s for s in selection['scores'] if s['selectable']]
    selection['winner'] = min(eligible,key=lambda s:(-s['health_stability_score'],-s['score'],s['candidate']))['candidate'] if eligible else None
    selection['status'] = 'SELECTED' if eligible else 'NO_SCREEN_CANDIDATE'
    selection['ranking'] = 'max min planner paired availability mean-minus-two-SE; then min cost gain; then ID'
    return selection


def run(profile,output_dir,device='cpu'):
    if device!='cpu':raise ValueError('CPU only')
    cfg=settings(profile);seed_info=seed_audit(cfg);git=f.old.screen.git_metadata();out=Path(output_dir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):raise FileExistsError('refusing overwrite')
    if profile=='full' and git['dirty']:raise RuntimeError('clean committed source required')
    out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    def timeout():
        if time.monotonic()-started>cfg['max_seconds']:raise TimeoutError('registered hard timeout')
    manifest=dict(protocol=VERSION,status='RUNNING',git=git,profile=profile,confirmation_outcomes_opened=False,started_utc=datetime.now(timezone.utc).isoformat(),runtime=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,device=device))
    write_json(out/'manifest.json',manifest)
    try:
        specs=specifications(profile);write_json(out/'candidates.json',specs);write_json(out/'resolved_config.json',cfg)
        shutil.copy2(PLAN,out/'protocol.md');shutil.copy2(REGISTRY,out/'registry.json');snap=out/'source_snapshot';snap.mkdir()
        for name in SOURCES:shutil.copy2(Path(__file__).parent/name,snap/name)
        rs=out/'registry_snapshot';rs.mkdir()
        for name in seed_info['historical_registry_hashes']:shutil.copy2(ROOT/'configs'/name,rs/name)
        seals={str(p.relative_to(out)):f.old.screen.sha(p) for p in (out/'protocol.md',out/'registry.json',*snap.iterdir(),*rs.iterdir())};manifest['source_seal']=seals;write_json(out/'manifest.json',manifest)
        for spec in specs:f.dp.tables(core.make_config(spec))
        journal=Journal(out/'decisions.csv');dev=[];screen=[];locals_=[];roots=[];rows=[];confirm_local=[]
        try:
            for spec,seed,method in tqdm(list(itertools.product(specs,cfg['development_seeds'],CHEAP)),desc='Development baseline selection',unit='episode'):
                row,_=episode(cfg,spec,method,seed,'deadline_matching','development',journal,timeout);dev.append(row)
            write_csv(out/'development_episodes.csv',dev);choices=baseline_choices(cfg,specs,dev);write_json(out/'baseline_choices.json',choices);choice_seal=f.old.screen.sha(out/'baseline_choices.json')
            root_journal=Journal(out/'screening_roots.csv');local_journal=Journal(out/'screening_landmarks.csv')
            try:
                for spec,seed,method in tqdm(list(itertools.product(specs,cfg['screening_seeds'],METHODS)),desc='Screen cross-skill family',unit='episode'):
                    base=choices[spec['candidate']]['base'];row,states=episode(cfg,spec,method,seed,base,'screening',journal,timeout);screen.append(row)
                    for tick,state in states.items():
                        loc=landmark(cfg,spec,state,seed,tick,base,'screening',root_journal,timeout);locals_.append(loc);local_journal.add(loc)
            finally:root_journal.close();local_journal.close()
            write_csv(out/'screening_episodes.csv',screen);selection=screen_choice(cfg,specs,choices,screen,locals_);write_json(out/'selection.json',selection);seal=f.old.screen.sha(out/'selection.json')
            manifest.update(selection_sha256=seal,baseline_choices_sha256=choice_seal,winner=selection['winner'],stage='screened');write_json(out/'manifest.json',manifest)
            if selection['winner']:
                winner=next(s for s in specs if s['candidate']==selection['winner']);base=choices[winner['candidate']]['base'];manifest.update(confirmation_outcomes_opened=profile=='full',stage='confirmation');write_json(out/'manifest.json',manifest)
                partial=Journal(out/'episodes.partial.csv');root_journal=Journal(out/'landmark_roots.csv');local_journal=Journal(out/'landmarks.csv')
                try:
                    for seed,method in tqdm(list(itertools.product(cfg['confirmation_seeds'],METHODS)),desc='Confirm sealed winner',unit='episode'):
                        row,states=episode(cfg,winner,method,seed,base,'confirmation',journal,timeout);rows.append(row);partial.add(row)
                        for tick,state in states.items():
                            loc=landmark(cfg,winner,state,seed,tick,base,'confirmation',root_journal,timeout);confirm_local.append(loc);local_journal.add(loc)
                finally:partial.close();root_journal.close();local_journal.close()
                write_csv(out/'episodes.csv',rows)
        finally:journal.close()
        all_roots=list(records(out/'screening_roots.csv'))+(list(records(out/'landmark_roots.csv')) if rows else [])
        audit=replay(cfg,specs,choices,out/'decisions.csv',dev+screen+rows,locals_+confirm_local,all_roots,timeout)
        if baseline_choices(cfg,specs,dev)!=choices or screen_choice(cfg,specs,choices,screen,locals_)!=selection:raise RuntimeError('selection reconstruction')
        if f.old.screen.sha(out/'selection.json')!=seal or f.old.screen.sha(out/'baseline_choices.json')!=choice_seal:raise RuntimeError('frozen choices')
        expected=dict(development=len(specs)*len(cfg['development_seeds'])*3,screening=len(specs)*len(cfg['screening_seeds'])*5,
                      confirmation=len(cfg['confirmation_seeds'])*5 if rows else 0,landmarks=(len(specs)*len(cfg['screening_seeds'])+(len(cfg['confirmation_seeds']) if rows else 0))*7)
        actual=dict(development=len(dev),screening=len(screen),confirmation=len(rows),landmarks=len(locals_)+len(confirm_local))
        # Registered loose cap includes every root at every planner/diagnostic tick.
        phase_seeds=len(specs)*len(cfg['screening_seeds'])+(len(cfg['confirmation_seeds']) if rows else 0)
        cap=phase_seeds*21*cfg['scenarios']*(2*sum(60-t for t in range(48))+2*sum(60-t for t in LANDMARKS))
        if actual!=expected or audit['forecast_transitions']>cap or any(r['extended_invalid_assignments'] for r in dev+screen+rows):raise RuntimeError('counts/feasibility/budget')
        for name,digest in seals.items():
            if f.old.screen.sha(out/name)!=digest:raise RuntimeError('source seal')
        for name in SOURCES:
            if f.old.screen.sha(Path(__file__).parent/name)!=seals[f'source_snapshot/{name}']:raise RuntimeError('live source')
        for name,digest in seed_info['historical_registry_hashes'].items():
            if f.old.screen.sha(ROOT/'configs'/name)!=digest:raise RuntimeError('registry drift')
        if f.old.screen.sha(PLAN)!=seals['protocol.md'] or f.old.screen.sha(REGISTRY)!=seals['registry.json']:raise RuntimeError('live protocol')
        summary=summarize(cfg,winner,base,rows,confirm_local) if rows else dict(scientific_status='NO_SCREEN_CANDIDATE' if profile=='full' else 'ENGINEERING_ONLY')
        summary['screening']=selection;summary['search_series']=cfg['search_series'];summary['round']=cfg['round'];summary['family_alpha']=cfg['family_alpha'];write_json(out/'summary.json',summary)
        audit.update(expected_counts=expected,actual_counts=actual,forecast_cap=cap,seed_audit=seed_info,selection_frozen=True,source_seal_unchanged=True);write_json(out/'audit.json',audit)
        files={str(p.relative_to(out)):dict(sha256=f.old.screen.sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        timeout();manifest.update(status='COMPLETED',audit='PASS',scientific_status=summary['scientific_status'],elapsed_seconds=time.monotonic()-started,actual_counts=actual,files=files,artifact_bytes=sum(b['bytes'] for b in files.values()));write_json(out/'manifest.json',manifest);return manifest
    except Exception as error:
        manifest.update(status='FAILED',error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started);write_json(out/'manifest.json',manifest);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',choices=('smoke','full'),required=True);p.add_argument('--device',choices=('cpu',),default='cpu');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();m=run(a.profile,a.output_dir,a.device)
    print(json.dumps({k:m[k] for k in ('status','audit','scientific_status','actual_counts','elapsed_seconds')},indent=2))

if __name__=='__main__':main()
