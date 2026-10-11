"""Frozen-policy, same-state preventive deferral and WAIT diagnostic."""
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_joint_rl as joint
from ht_pdm_fjsp import maintenance_opportunity_cost as prior
from ht_pdm_fjsp import maintenance_boundary_sensitivity as boundary
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE, JOINT
from ht_pdm_fjsp.maintenance_timing_forks import fork_values, paired_delta, without_preventive, kind, scalar_replay
from ht_pdm_fjsp.maintenance_timing_statistics import summarize

VERSION = 'maintenance_timing_diagnostic_v1'
ROOT = Path(__file__).resolve().parents[2]
RAW_KEYS = ('cohorts', 'physical_shocks', 'forecast_seed', 'analysis_seed')


def settings(profile):
    if profile not in ('full', 'smoke'):
        raise ValueError('unknown profile')
    full = profile == 'full'
    return dict(profile=profile, conditions=list(joint.CONDITIONS), origins=[JOINT, 'best_rule'],
        model_seeds=list(range(12100000, 12100010)) if full else [12100000],
        cohorts=list(range(13100000,13100008)) if full else [13500000],
        physical_shocks=[13200000,13200001] if full else [13500010],
        forecast_seed=13300000 if full else 13500020, analysis_seed=13400000 if full else 13500030,
        horizon=48, trajectory_ticks=48 if full else 4, landmarks=[6,18,30,42] if full else [0,2],
        scenarios=32 if full else 4, forecast_ticks=24 if full else 6,
        continuation='deadline_matching', bootstrap_samples=20000 if full else 200,
        primary_confidence=.9875, torch_threads=1)


def counts(cfg):
    trajectories = len(cfg['model_seeds'])*len(cfg['conditions'])*len(cfg['cohorts'])*len(cfg['physical_shocks'])*len(cfg['origins'])
    states = trajectories*len(cfg['landmarks'])
    return dict(models=len(cfg['model_seeds']), trajectories=trajectories,
        trajectory_ticks=trajectories*cfg['trajectory_ticks'], states=states,
        max_forecast_transitions=states*len(CATALOGUE)*cfg['scenarios']*cfg['forecast_ticks'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    return json.loads((ROOT/f'configs/{VERSION}_checkpoint_sources.json').read_text())


def verify_source(source, locked):
    """Verify all bytes before loading frozen models or opening new outcomes."""
    source = Path(source).resolve()
    if sha(source/'manifest.json') != locked['manifest_sha256']:
        raise ValueError('source manifest hash mismatch')
    manifest = json.loads((source/'manifest.json').read_text())
    if (manifest['status'] != 'COMPLETED' or manifest['audit'] != 'PASS' or manifest['dirty']
        or manifest['protocol'] != joint.VERSION or manifest['commit'] != locked['commit']
        or manifest['settings'] != joint.settings('full') or manifest['actual_counts'] != joint.counts(joint.settings('full'))):
        raise ValueError('source run contract mismatch')
    for name, record in manifest['artifacts'].items():
        path = (source/name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError(f'source artifact mismatch: {name}')
        if name.startswith('source_snapshot/'):
            relative = name.removeprefix('source_snapshot/')
            current = ROOT/'src/ht_pdm_fjsp'/relative if relative.endswith('.py') else ROOT/relative if relative in ('pyproject.toml','uv.lock') else ROOT/'configs'/relative
            if sha(current) != record['sha256']:
                raise ValueError(f'source dependency changed: {relative}')
    selections = json.loads((source/'checkpoint_selection.json').read_text())
    if not selections['locked_before_test'] or selections['selected_rules'] != locked['selected_rules']:
        raise ValueError('reference rule changed')
    resolved = json.loads((source/'resolved_config.json').read_text())
    _, profiles = joint.build_panels(joint.settings('full'))
    if resolved['physical_profiles'] != json.loads(json.dumps(profiles)):
        raise ValueError('source physical generator mismatch')
    models = {}
    for seed, record in tqdm(list(locked['models'].items()), desc='Verify frozen source models'):
        chosen = selections['selections'][f'{JOINT}:{seed}']
        if any(chosen[k] != record[k] for k in ('update','learned_steps','selected_checkpoint')):
            raise ValueError('source checkpoint selection mismatch')
        path = source/record['selected_checkpoint']
        if sha(path) != record['sha256']:
            raise ValueError('frozen checkpoint hash mismatch')
        model, payload = joint.load_checkpoint(path)
        if (payload['settings'] != joint.settings('full') or payload['train_seed'] != int(seed)
            or payload['algorithm'] != JOINT or any(payload[k] != record[k] for k in ('update','learned_steps'))):
            raise ValueError('frozen checkpoint payload mismatch')
        models[int(seed)] = model
    if set(models) != set(joint.settings('full')['train_seeds']):
        raise ValueError('all ten source seeds required')
    return models, dict(status='PASS', source_manifest_sha256=locked['manifest_sha256'],
        verified_source_files=len(manifest['artifacts']), frozen_models=len(models))


@lru_cache(maxsize=1)
def exclusion_pool():
    """Metadata only: reconstruct every source training configuration, not outcomes."""
    cfg = joint.settings('full')
    panels, _ = joint.build_panels(cfg)
    smoke, _ = joint.build_panels(joint.settings('smoke'))
    blocked = {boundary.frozen(s) for s in joint.exclusions()['physical_signatures']}
    blocked.update(boundary.signature(c) for c in itertools.chain(panels.values(),smoke.values()))
    _, forbidden = joint.seed_audit(cfg)
    sampler = joint.training_sampler(cfg, panels, forbidden)
    for seed in tqdm(cfg['train_seeds'], desc='Regenerate source physical exclusions'):
        for index in range(cfg['env_steps']//cfg['horizon']):
            blocked.add(boundary.signature(sampler(seed,index)[0]))
    return frozenset(blocked)


def build_panels(cfg):
    blocked = set(exclusion_pool())
    if cfg['profile'] == 'smoke':
        full, _ = build_panels(settings('full'))
        blocked.update(boundary.signature(c) for c in full.values())
    panels, rows = {}, []
    for cohort in cfg['cohorts']:
        for attempt in range(10000):
            generated = {cond:joint.make_config(cohort,cond,48,attempt) for cond in cfg['conditions']}
            signatures = [boundary.signature(c) for c in generated.values()]
            if len(set(signatures)) == len(signatures) and not set(signatures)&blocked:
                break
        else:
            raise RuntimeError('fresh physical pool exhausted')
        for condition, c in generated.items():
            blocked.add(boundary.signature(c));panels[cohort,condition] = c
            rows.append(dict(cohort=cohort,condition=condition,generator_attempt=attempt,config=asdict(c)))
    return panels, rows


def physical_seed(cohort, shock):
    return role_seed(shock,f'{VERSION}:physical:{cohort}')


def forecast_seed(cfg, cohort, shock, seed, origin, tick):
    return role_seed(cfg['forecast_seed'],f'{VERSION}:forecast:{cohort}:{shock}:{seed}:{origin}:{tick}')


def seed_audit(cfg):
    own_path = ROOT/f'configs/{VERSION}_seed_registry.json'
    own = json.loads(own_path.read_text())['panels']
    historical = joint.integers(joint.exclusions()['historical_raw_panels'])
    for path in (ROOT/'configs').glob('*seed_registry.json'):
        if path != own_path:
            historical |= joint.integers(json.loads(path.read_text()))
    roles, fresh = [], set()
    for profile in ('full','smoke'):
        panel = settings(profile)
        if own[profile] != {k:panel[k] for k in RAW_KEYS}:
            raise ValueError('fresh seed registry differs from protocol')
        for key in RAW_KEYS:
            group = joint.integers(panel[key])
            if group & (fresh|historical):
                raise ValueError('raw seed collision')
            fresh |= group
        for cohort, shock in itertools.product(panel['cohorts'],panel['physical_shocks']):
            roles.append(physical_seed(cohort,shock))
            roles.extend(forecast_seed(panel,cohort,shock,seed,origin,tick) for seed,origin,tick in
                itertools.product(panel['model_seeds'],panel['origins'],panel['landmarks']))
        roles.append(role_seed(panel['analysis_seed'],f'{VERSION}:crossed_bootstrap'))
    if len(roles) != len(set(roles)) or set(roles)&(fresh|historical):
        raise ValueError('derived RNG role collision')
    if cfg != settings(cfg['profile']):
        raise ValueError('settings changed')
    return dict(status='PASS',own_registry_sha256=sha(own_path),unique_derived_roles=len(roles),
        reused_model_seeds='intentional frozen dependency, never new initialization',
        intentional_pairing='conditions/models/origins share physical tapes; all roots share forecast tapes')


def parameters_sha(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def evaluate_state(c,state,identity,model,rule,cfg,journals):
    tick = identity['tick']
    rl, policy = model.select(c,state,cfg['horizon']-tick,cfg['horizon'])
    reference = prior.rule_action(c,state,cfg['horizon']-tick,rule)
    fs = forecast_seed(cfg,identity['cohort'],identity['shock'],identity['model_seed'],identity['origin'],tick)
    start = time.perf_counter()
    values = fork_values(c,state,fs,cfg['scenarios'],cfg['forecast_ticks'])
    elapsed = time.perf_counter()-start
    if cfg['profile'] == 'smoke':
        scalar_replay(c,state,values)
    no_pm = without_preventive(state,rl)
    pm = no_pm != rl;choice = len(values['candidates'])>1
    delta, se = paired_delta(values,rl,no_pm)
    rd, rs = paired_delta(values,rl,reference)
    lookup = {action:i for i,action in enumerate(values['candidates'])}
    scale = c.machines*cfg['forecast_ticks']
    root_min = float(values['means'].min())
    row = dict(**identity, forecast_seed=fs,physical_seed=physical_seed(identity['cohort'],identity['shock']),
        state=prior.compact(asdict(state)),rl_action=prior.compact(rl),rule_action=prior.compact(reference),
        no_pm_action=prior.compact(no_pm),rl_kind=kind(state,rl),rule_kind=kind(state,reference),
        actions_agree=rl==reference,pm_exposure=int(pm),choice_exposure=int(choice),
        pm_delta=delta/scale if pm else 0.,pm_mc_se=se/scale if pm else 0.,
        rule_delta=rd/scale if choice else 0.,rule_mc_se=rs/scale if choice else 0.,
        rl_cost=float(values['means'][lookup[rl]]),rule_cost=float(values['means'][lookup[reference]]),
        no_pm_cost=float(values['means'][lookup[no_pm]]),sample_min_cost=root_min,
        sample_min_gap=float(values['means'][lookup[rl]])-root_min,
        candidates=len(lookup),forecast_transitions=values['transitions'],**policy)
    healthy = [state.ages[m] for m in range(c.machines) if not state.failed[m] and m not in state.assigned]
    row.update(healthy_hazard_machines=sum(a+1>=c.failure_age for a in healthy),
        healthy_age_band='AT_RISK' if any(a+1>=c.failure_age for a in healthy) else 'BELOW_THRESHOLD',
        maximum_pending_wait=max(state.pending_wait),failure_age=c.failure_age,failure_probability=c.failure_probability,
        forecast_ticks=cfg['forecast_ticks'],scenarios=cfg['scenarios'])
    padded = np.full((len(CATALOGUE),cfg['scenarios']),np.nan)
    for action,i in lookup.items():
        catalogue_index = CATALOGUE.index(action)
        padded[catalogue_index] = values['costs'][i]
        journals['root_values'].add(dict(**identity,action_index=catalogue_index,action=prior.compact(action),
            kind=kind(state,action),mean_cost=float(values['means'][i]),mc_se=float(values['mc_se'][i]),
            first_step_mean=float(values['first_step_mean'][i]),
            **{k:float(v[i]) for k,v in values['component_means'].items()},
            scenario_costs=prior.compact(values['costs'][i].tolist())))
    for m,j in rl:
        if state.failed[m]:
            continue
        alternative = tuple(pair for pair in rl if pair != (m,j))
        d,s = paired_delta(values,rl,alternative)
        journals['preventive_pairs'].add(dict(**identity,machine=m,technician=j,age=state.ages[m],
            threshold=c.failure_age,hazard=c.failure_probability if state.ages[m]+1>=c.failure_age else 0.,
            duration=c.service_time[m][j],restoration=c.restoration[m][j],
            marginal_delta=d/scale,marginal_mc_se=s/scale,alternative=prior.compact(alternative)))
    journals['rollout_audits'].add(dict(**identity,feasibility='PASS',components='PASS',
        scalar_replay='PASS' if cfg['profile']=='smoke' else 'NOT_SCHEDULED',
        scenarios=cfg['scenarios'],forecast_ticks=cfg['forecast_ticks'],transitions=values['transitions']))
    return row,padded,elapsed


def feature_tables(rows):
    groups = defaultdict(list)
    for row in rows:
        state = json.loads(row['state'])
        groups[row['condition'],row['origin'],row['rl_kind'],row['rule_kind'],row['healthy_age_band'],sum(state['failed']),sum(x>0 for x in state['remaining'])].append(row)
    return [dict(condition=k[0],origin=k[1],rl_kind=k[2],rule_kind=k[3],healthy_age_band=k[4],failed_machines=k[5],busy_technicians=k[6],states=len(group),
        pm_exposures=sum(r['pm_exposure'] for r in group),choice_exposures=sum(r['choice_exposure'] for r in group),
        pm_delta=sum(r['pm_delta'] for r in group)/sum(r['pm_exposure'] for r in group) if any(r['pm_exposure'] for r in group) else None,
        rule_delta=sum(r['rule_delta'] for r in group)/sum(r['choice_exposure'] for r in group) if any(r['choice_exposure'] for r in group) else None)
        for k,group in sorted(groups.items())]


def run(output,source_dir,profile='smoke',device='cpu'):
    if device != 'cpu':
        raise ValueError('CPU only')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('new/empty directory required')
    output.mkdir(parents=True,exist_ok=True)
    cfg = settings(profile);torch.set_num_threads(1)
    now = lambda:datetime.now(timezone.utc).isoformat()
    git = lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    start = time.perf_counter()
    manifest = dict(protocol=VERSION,status='RUNNING',started=now(),commit=git('rev-parse','HEAD'),
        dirty=bool(git('status','--porcelain')),settings=cfg,expected_counts=counts(cfg),device=device,
        platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),
        learning='NOT_APPLICABLE: frozen inference only',source_full_outcomes_previously_opened=True,
        new_full_panel_outcomes_opened=False,thread_environment={k:os.environ.get(k) for k in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    prior.write_json(output/'manifest.json',manifest)
    journals = {};rows = [];raw = [];fork_seconds = 0.
    try:
        if profile=='full' and manifest['dirty']:
            raise ValueError('full requires clean committed code')
        if profile=='full' and any(v!='1' for v in manifest['thread_environment'].values()):
            raise ValueError('full requires OMP/OPENBLAS/MKL_NUM_THREADS=1')
        locked = contract();models,source_audit = verify_source(source_dir,locked)
        manifest['source_audit'] = source_audit;manifest['seed_audit'] = seed_audit(cfg)
        panels,registry = build_panels(cfg)
        prior.write_json(output/'source_contract.json',locked)
        prior.write_json(output/'resolved_config.json',dict(settings=cfg,physical_profiles=registry))
        prior.write_json(output/'seed_registry.json',{k:cfg[k] for k in RAW_KEYS})
        prior.write_json(output/'physical_exclusion_audit.json',dict(status='PASS',
            source_training_configurations_regenerated=51200,unique_prior_signatures=len(exclusion_pool()),
            diagnostic_profiles=len(registry),smoke_excludes_full=True))
        shutil.copyfile(ROOT/f'docs/{VERSION}_plan.md',output/'plan.md')
        snapshot = output/'source_snapshot';snapshot.mkdir()
        for path in list((ROOT/'src/ht_pdm_fjsp').glob('maintenance_*.py'))+[ROOT/'pyproject.toml',ROOT/'uv.lock']:
            shutil.copyfile(path,snapshot/path.name)
        for path in (ROOT/'configs').glob('maintenance_*json'):
            shutil.copyfile(path,snapshot/path.name)
        initial_hashes = {}
        for seed in cfg['model_seeds']:
            destination = output/'frozen_models'/f'seed_{seed}.pt';destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(Path(source_dir)/locked['models'][str(seed)]['selected_checkpoint'],destination)
            restored,_ = joint.load_checkpoint(destination)
            if parameters_sha(restored) != parameters_sha(models[seed]):
                raise RuntimeError('checkpoint parameter round trip')
            for c in panels.values():
                env = WaitingEnv(c)
                if restored.select(c,env.state,48,48) != models[seed].select(c,env.state,48,48):
                    raise RuntimeError('checkpoint action/prob/value round trip')
            models[seed] = restored;initial_hashes[seed] = parameters_sha(restored)
        for name in ('trajectories','states.partial','root_values','preventive_pairs','rollout_audits'):
            journals[name] = joint.CountingJournal(output/f'{name}.csv')
        manifest['new_full_panel_outcomes_opened'] = profile=='full'
        prior.write_json(output/'manifest.json',manifest)
        print(json.dumps(dict(protocol=VERSION,budget=counts(cfg),training='NONE',source=str(source_dir)),indent=2),flush=True)
        combos = list(itertools.product(cfg['model_seeds'],cfg['cohorts'],cfg['physical_shocks'],cfg['conditions'],cfg['origins']))
        with tqdm(total=counts(cfg)['states'],desc='Same-state root forks') as progress:
            for seed,cohort,shock,condition,origin in tqdm(combos,desc='Frozen multi-seed trajectories'):
                c = panels[cohort,condition];env = WaitingEnv(c);env.reset(physical_seed(cohort,shock))
                rule = locked['selected_rules'][condition]
                for tick in range(cfg['trajectory_ticks']):
                    state = env.state
                    identity = dict(condition=condition,cohort=cohort,shock=shock,model_seed=seed,origin=origin,tick=tick)
                    if tick in cfg['landmarks']:
                        row,costs,seconds = evaluate_state(c,state,identity,models[seed],rule,cfg,journals)
                        rows.append(row);raw.append(costs);fork_seconds += seconds
                        journals['states.partial'].add(row);progress.update()
                    pairs = models[seed].select(c,state,48-tick,48)[0] if origin==JOINT else prior.rule_action(c,state,48-tick,rule)
                    validate_matching(c,state,pairs)
                    _,reward,_,info = env.step(pairs)
                    journals['trajectories'].add(dict(**identity,state=prior.compact(asdict(state)),action=prior.compact(pairs),
                        kind=kind(state,pairs),cost=-reward,following=prior.compact(asdict(env.state)),info=prior.compact(info)))
        if any(parameters_sha(models[s]) != initial_hashes[s] for s in cfg['model_seeds']):
            raise RuntimeError('frozen model changed')
        summary,cells = summarize(rows,cfg)
        prior.write_json(output/'summary.json',summary);prior.write_csv(output/'profile_cells.csv',cells)
        prior.write_csv(output/'states.csv',rows);prior.write_csv(output/'state_features.csv',feature_tables(rows))
        prior.write_json(output/'behavior.json',summary['behavior'])
        np.savez_compressed(output/'raw_forecast_costs.npz',costs=np.stack(raw))
        actual = dict(models=len(cfg['model_seeds']),trajectories=len(combos),trajectory_ticks=journals['trajectories'].rows,
            states=len(rows),max_forecast_transitions=counts(cfg)['max_forecast_transitions'])
        if actual != counts(cfg) or journals['states.partial'].rows != len(rows):
            raise RuntimeError('count mismatch')
        manifest.update(status='COMPLETED',audit='PASS',actual_counts=actual,
            actual_forecast_transitions=sum(r['forecast_transitions'] for r in rows),candidate_roots=journals['root_values'].rows,
            frozen_parameter_hashes={str(k):v for k,v in initial_hashes.items()},fork_seconds=fork_seconds)
    except BaseException as error:
        manifest.update(status='FAILED',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for journal in journals.values():
            journal.close()
        manifest['finished'] = now();manifest['elapsed_seconds'] = time.perf_counter()-start
        manifest['artifacts'] = {str(p.relative_to(output)):dict(sha256=sha(p),bytes=p.stat().st_size)
            for p in sorted(output.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        prior.write_json(output/'manifest.json',manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('full','smoke'),default='smoke')
    parser.add_argument('--device',choices=('cpu',),default='cpu')
    parser.add_argument('--source-dir',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args = parser.parse_args()
    run(args.output_dir,args.source_dir,args.profile,args.device)


if __name__=='__main__':
    main()
