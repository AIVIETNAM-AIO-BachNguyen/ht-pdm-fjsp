"""One-edge hindsight intervention on audited V13 opened-DEV trajectories."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, HORIZON, FrameworkEnv, configs
from ht_pdm_fjsp.maintenance_framework_v13_campaign import V13Controller
from ht_pdm_fjsp.maintenance_request_policies import request_action
from ht_pdm_fjsp.maintenance_requests import legal_matchings, submit_requests
from run_v13_failure_events import read_rows, selected_actor, sha, bootstrap

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / 'research_result/v13_commitment_intervention_protocol_vi.md'
SOURCE_EXCEPTIONS = {'research_result/framework_v13_dev_protocol_vi.md',
                     'research_result/method_v13_proposal_vi.md'}
TARGETS = ('interchangeable_scale1', 'reciprocal_scale2')


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matching_json(matching):
    return json.dumps(matching, separators=(',', ':'))


def at_risk(config, admitted, matching, legal):
    served = {m for m, _ in matching}
    exposed = feasible = constrained = 0
    first = None
    for m, req in enumerate(admitted.requests):
        if req is None or req.started_at is not None or req.kind != 'preventive':
            continue
        if admitted.physical.failed[m] or admitted.physical.ages[m] + 1 < config.failure_age:
            continue
        if m in served:
            continue
        exposed += 1
        possible = any(any(mm == m for mm, _ in action) for action in legal)
        if possible:
            feasible += 1
        else:
            constrained += 1
            if first is None:
                first = m
    return dict(exposed=exposed, feasible=feasible, constrained=constrained), first


def trajectory(case, physical, controller, deadline, edit=None):
    config = configs()[case]
    env = FrameworkEnv(config)
    env.reset(seed=physical)
    actions = []
    counts = dict(exposed=0, feasible=0, constrained=0, pm_starts=0)
    trigger = None
    reissued = False
    inference_seconds = 0.0
    for tick in range(HORIZON):
        if time.monotonic() > deadline:
            raise TimeoutError('V13 commitment intervention exceeded 20 minutes')
        before = env.inner.state
        submitted = request_action(config, before, tick, 'threshold')
        admitted = submit_requests(config, before, tick, submitted)
        legal = legal_matchings(config, admitted)
        began = time.perf_counter()
        selected, matching = controller.choose(config, before, tick, case=case, seed=physical)
        inference_seconds += time.perf_counter() - began
        assert submitted == selected and matching in legal
        chosen = matching
        if edit is not None and tick == edit['start_tick']:
            edge = (edit['occupied_machine'], edit['technician'])
            assert edge in chosen
            matching = tuple(pair for pair in chosen if pair != edge)
            assert matching in legal
        if edit is not None and tick > edit['start_tick'] and any(
                m == edit['occupied_machine'] for m, _ in matching):
            reissued = True
        risk, target = at_risk(config, admitted, matching, legal)
        for key in ('exposed', 'feasible', 'constrained'):
            counts[key] += risk[key]
        if edit is None and trigger is None and target is not None:
            busy = [j for j in range(config.technicians)
                    if config.service_time[target][j] > 0 and admitted.physical.remaining[j] > 0]
            if busy:
                worker = min(busy, key=lambda j: (-admitted.physical.remaining[j], j))
                occupied = admitted.physical.assigned[worker]
                request = admitted.requests[occupied]
                assert request is not None and request.started_at is not None
                assert request.technician == worker and request.started_at < tick
                trigger = dict(trigger_tick=tick, target_machine=target,
                               technician=worker, occupied_machine=occupied,
                               start_tick=request.started_at,
                               remaining=admitted.physical.remaining[worker],
                               duration=config.service_time[occupied][worker])
        _, _, _, _, info = env.step(env.actions.index(matching))
        counts['pm_starts'] += sum(e['event'] == 'started' and e['kind'] == 'preventive'
                                   for e in info['lifecycle'])
        actions.append(dict(tick=tick, selected=matching_json(chosen),
                            executed=matching_json(matching),
                            exposed=risk['exposed'], feasible=risk['feasible'],
                            constrained=risk['constrained']))
    metrics = env.episode_metrics()
    result = dict(objective=float(metrics['objective']), failure_cost=float(metrics['failure_cost']),
                  H_failures=float(metrics['H_failures']), H_availability=float(metrics['H_availability']),
                  H_pm_starts=counts['pm_starts'], H_risk_exposed=counts['exposed'],
                  H_risk_feasible_exposed=counts['feasible'],
                  H_risk_capacity_exposed=counts['constrained'],
                  inference_seconds=inference_seconds)
    return result, actions, trigger, reissued


def summarize(rows, seeds, physicals, profile):
    by_key = {(r['case'], int(r['train_seed']), int(r['seed'])): r for r in rows}
    result = dict(status='COMPLETED', profile=profile, episodes=len(rows), by_case={}, joint_gate=False)
    for case in CASES:
        group = [by_key[(case, s, p)] for s in seeds for p in physicals]
        cube = np.array([[float(by_key[(case, s, p)]['cost_delta']) for p in physicals]
                         for s in seeds])
        means = cube.mean(axis=1)
        ci = bootstrap(cube, 20000 if profile == 'dev' else 100, 72540002)
        result['by_case'][case] = dict(cost_delta=float(cube.mean()),
            ci95=ci, by_train_seed={str(s):float(means[i]) for i,s in enumerate(seeds)},
            trigger_fraction=float(np.mean([int(r['triggered']) for r in group])),
            triggered_count=sum(int(r['triggered']) for r in group),
            conditional_delta=float(np.mean([float(r['cost_delta']) for r in group
                                             if int(r['triggered'])])) if any(int(r['triggered']) for r in group) else None,
            mean_exposure_delta=float(np.mean([float(r['exposure_delta']) for r in group])),
            mean_failure_delta=float(np.mean([float(r['failure_delta']) for r in group])),
            mean_pm_starts_delta=float(np.mean([float(r['pm_starts_delta']) for r in group])),
            directional_gate=bool(profile == 'dev' and cube.mean() <= -10 and ci[1] < 0 and
                                  sum(means < 0) >= 4))
    result['joint_gate'] = all(result['by_case'][case]['directional_gate'] for case in TARGETS)
    result['joint_cost_delta'] = float(np.mean([result['by_case'][case]['cost_delta'] for case in TARGETS]))
    return result


def run(source, out, profile):
    start = time.monotonic()
    deadline = start + (1200 if profile == 'dev' else 300)
    manifest = json.loads((source/'manifest.json').read_text())
    assert manifest['status'] == 'COMPLETED' and manifest['profile'] == profile
    changed = {name for name,digest in manifest['source_hashes'].items()
               if sha(ROOT/name) != digest}
    assert changed == (SOURCE_EXCEPTIONS if profile == 'smoke' else set()), changed
    for name,digest in manifest['artifact_hashes'].items():
        assert sha(source/name) == digest
    seeds = tuple(range(72500000,72500005)) if profile == 'dev' else (72520000,)
    physicals = tuple(range(72510000,72510016)) if profile == 'dev' else (72520000,72520001)
    selected = {(r['case'],int(r['train_seed']),int(r['seed'])):r
                for r in read_rows(source/'episodes.csv') if r['controller']=='competition'}
    selection = json.loads((source/'selection.json').read_text())
    checkpoint_hashes={}
    actors={}
    for seed in seeds:
        cp=selection['competition'][str(seed)]['checkpoint']
        checkpoint_hashes[str(seed)]=sha(source/cp)
        actors[seed]=V13Controller(selected_actor(source,cp),'competition')
    torch.set_num_threads(1)
    rows=[]; traces=[]
    grid=[(case,seed,p) for case in CASES for seed in seeds for p in physicals]
    for case,seed,p in tqdm(grid,desc='V13 commitment intervention',unit='pair'):
        baseline,base_trace,trigger,_=trajectory(case,p,actors[seed],deadline)
        expected=selected[(case,seed,p)]
        for key in ('objective','failure_cost','H_failures','H_availability'):
            assert np.isclose(baseline[key],float(expected[key]),rtol=0,atol=1e-7), (case,seed,p,key)
        if trigger is not None:
            edge=(trigger['occupied_machine'],trigger['technician'])
            original=json.loads(base_trace[trigger['start_tick']]['executed'])
            assert list(edge) in original
        edited,edit_trace,_,reissued=trajectory(case,p,actors[seed],deadline,trigger)
        row=dict(case=case,train_seed=seed,seed=p,triggered=int(trigger is not None),
                 trigger_tick=trigger['trigger_tick'] if trigger else '',
                 target_machine=trigger['target_machine'] if trigger else '',
                 technician=trigger['technician'] if trigger else '',
                 occupied_machine=trigger['occupied_machine'] if trigger else '',
                 start_tick=trigger['start_tick'] if trigger else '',
                 remaining=trigger['remaining'] if trigger else '',
                 duration=trigger['duration'] if trigger else '',reissued=int(reissued),
                 original_cost=baseline['objective'],edited_cost=edited['objective'],
                 cost_delta=edited['objective']-baseline['objective'],
                 original_failure_cost=baseline['failure_cost'],edited_failure_cost=edited['failure_cost'],
                 original_H_failures=baseline['H_failures'],edited_H_failures=edited['H_failures'],
                 failure_delta=edited['H_failures']-baseline['H_failures'],
                 original_H_availability=baseline['H_availability'],edited_H_availability=edited['H_availability'],
                 original_H_pm_starts=baseline['H_pm_starts'],edited_H_pm_starts=edited['H_pm_starts'],
                 pm_starts_delta=edited['H_pm_starts']-baseline['H_pm_starts'],
                 original_H_risk_exposed=baseline['H_risk_exposed'],edited_H_risk_exposed=edited['H_risk_exposed'],
                 exposure_delta=edited['H_risk_exposed']-baseline['H_risk_exposed'],
                 original_H_risk_feasible_exposed=baseline['H_risk_feasible_exposed'],
                 edited_H_risk_feasible_exposed=edited['H_risk_feasible_exposed'],
                 original_H_risk_capacity_exposed=baseline['H_risk_capacity_exposed'],
                 edited_H_risk_capacity_exposed=edited['H_risk_capacity_exposed'],
                 original_inference_seconds=baseline['inference_seconds'],
                 edited_inference_seconds=edited['inference_seconds'])
        rows.append(row)
        for arm,steps in (('original',base_trace),('edited',edit_trace)):
            traces.extend(dict(case=case,train_seed=seed,seed=p,arm=arm,**step) for step in steps)
    assert len(rows)==len(grid) and len(traces)==len(grid)*2*HORIZON
    summary=summarize(rows,seeds,physicals,profile)
    write_csv(out/'episodes.csv',rows)
    write_csv(out/'actions.csv',traces)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    provenance=dict(status='COMPLETED',profile=profile,source_run=str(source),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_manifest_sha256=sha(source/'manifest.json'),source_episodes_sha256=sha(source/'episodes.csv'),
        source_selection_sha256=sha(source/'selection.json'),checkpoint_hashes=checkpoint_hashes,
        source_document_hash_exceptions=sorted(changed),script_sha256=sha(Path(__file__)),
        protocol_sha256=sha(PROTOCOL),bootstrap_seed=72540002,
        bootstrap_draws=20000 if profile=='dev' else 100,wall_seconds=time.monotonic()-start,
        artifact_hashes={name:sha(out/name) for name in ('episodes.csv','actions.csv','summary.json')})
    (out/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(summary))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--profile',choices=('smoke','dev'),default='dev')
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    run(args.run_dir.resolve(),out,args.profile)

if __name__=='__main__': main()
