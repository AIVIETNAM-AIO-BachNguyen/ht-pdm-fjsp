"""Independent read-only audit of teacher-transfer DEV selection and budgets."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np
from tqdm.auto import tqdm


ARMS = ('teacher_reservation_ppo', 'scratch_reservation_ppo')
PROTOCOL = 'technician_teacher_transfer_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def audit(source, output):
    started = time.monotonic()
    manifest = read_json(source/'manifest.json')
    assert manifest['protocol'] == PROTOCOL
    assert manifest['status'] == 'COMPLETED' and manifest['audit_status'] == 'PASS'
    assert manifest['profile'] in ('development', 'smoke')
    if manifest['profile'] == 'development':
        assert not manifest['test_opened']
    for name, digest in tqdm(manifest['source_seal'].items(), desc='Source seals'):
        assert sha(source/name) == digest, name
    for name, meta in tqdm(manifest['output_bindings'].items(), desc='Output bindings'):
        p = source/name
        assert p.stat().st_size == meta['bytes'] and sha(p) == meta['sha256'], name
    cfg = read_json(source/'resolved_config.json')
    choices = read_json(source/'checkpoint_choices.json')
    summary = read_json(source/'learning_summary.json')
    progress = read_json(source/'training_audit.json')
    initial = read_json(source/'initial_parameter_hashes.json')
    warm = read_json(source/'warm_choices.json')
    rows = read_csv(source/'development_episodes.csv')
    train = read_csv(source/'training_episodes.csv')
    scales, seeds, dev = cfg['scales'], cfg['train_seeds'], cfg['development_seeds']
    steps = [0, *cfg['checkpoints']]
    expected = {(n,a,t,s,d) for n in scales for a in ARMS for t in seeds
                for s in steps for d in dev}
    key = lambda r: (int(r['n']),r['method'],int(r['train_seed']),
                     int(r['checkpoint_steps']),int(r['seed']))
    indexed = {key(r):r for r in rows}
    assert len(rows) == len(expected) and set(indexed) == expected
    for n in scales:
        for t in seeds:
            assert initial[f'{n}:{ARMS[0]}:{t}'] == initial[f'{n}:{ARMS[1]}:{t}']
    assert len(progress) == len(scales)*len(ARMS)*len(seeds)
    for p in progress:
        assert p['learned_steps'] == cfg['env_steps']
        assert p['training_episodes'] == cfg['env_steps']//48
        assert p['physical_ticks'] == cfg['env_steps']//48*60
        assert p['optimizer_steps'] > 0 and p['changed']
        if p['algorithm'] == ARMS[0]:
            assert p['warm_state_exposures'] == cfg['teacher_bank_steps']
            assert p['warm_optimizer_steps'] > 0 and p['bank_sha256']
        else:
            assert p['warm_state_exposures'] == p['warm_optimizer_steps'] == 0
            assert p['bank_sha256'] is None
    assert len(warm) == len(seeds)
    for t in seeds:
        bank = source/'warm'/str(t)/'bank.npz'
        assert sha(bank) == warm[str(t)]['bank']['bank_sha256']
    tape = {}
    seen = set()
    for r in tqdm(train, desc='Paired training tapes', unit='episode'):
        n,a,t,w,e = (int(r['n']),r['algorithm'],int(r['train_seed']),
                     int(r['worker']),int(r['episode_index']))
        k = (n,t,w,e)
        identity = k+(a,)
        assert identity not in seen
        seen.add(identity)
        physical = (r['physical_seed'],r['population_sha256'])
        if k in tape:
            assert tape[k] == physical
        else:
            tape[k] = physical
        assert abs(float(r['episode_reward'])*20+float(r['extended_cost'])) < 1e-5
        assert int(float(r['invalid_assignments'])) == 0
    expected_train = {(n,t,w,e,a) for n in scales for t in seeds
        for w in range(cfg['vector_envs'])
        for e in range(cfg['env_steps']//(48*cfg['vector_envs'])) for a in ARMS}
    assert seen == expected_train
    chosen = {}; by_scale = {}; health = {}; auc = {}; final = {}
    for n in scales:
        by_scale[str(n)] = {}; health[str(n)] = {}; auc[str(n)] = {}; final[str(n)] = {}
        for a in ARMS:
            selected_costs = []; selected_rows = []; auc_values = []; final_values = []
            for t in seeds:
                costs = {s: statistics.fmean(float(indexed[n,a,t,s,d]['extended_cost'])
                    for d in dev) for s in steps}
                picked = min(steps,key=lambda s:(costs[s],s))
                assert choices[f'{n}:{a}:{t}']['steps'] == picked
                chosen[f'{n}:{a}:{t}'] = picked
                selected_costs.append(costs[picked])
                selected_rows.extend(indexed[n,a,t,picked,d] for d in dev)
                auc_values.append(float(np.trapezoid([costs[s]/n for s in steps],steps)
                    /(steps[-1]-steps[0])))
                final_values.append(costs[cfg['env_steps']]/n)
            by_scale[str(n)][a] = statistics.fmean(selected_costs)
            health[str(n)][a] = {field:statistics.fmean(float(r[field]) for r in selected_rows)
                for field in ('availability','normalized_pending','normalized_failures',
                              'invalid_assignments')}
            auc[str(n)][a] = auc_values
            final[str(n)][a] = final_values
    gains = {str(n):1-by_scale[str(n)][ARMS[0]]/by_scale[str(n)][ARMS[1]] for n in scales}
    pooled = 1-sum(by_scale[str(n)][ARMS[0]]/n for n in scales)/sum(
        by_scale[str(n)][ARMS[1]]/n for n in scales)
    guards = {str(n):dict(
        availability=health[str(n)][ARMS[0]]['availability']>=health[str(n)][ARMS[1]]['availability']-.01,
        pending=health[str(n)][ARMS[0]]['normalized_pending']<=health[str(n)][ARMS[1]]['normalized_pending']+.025,
        failures=health[str(n)][ARMS[0]]['normalized_failures']<=health[str(n)][ARMS[1]]['normalized_failures']+.25,
        invalid=health[str(n)][ARMS[0]]['invalid_assignments']==0 and
                health[str(n)][ARMS[1]]['invalid_assignments']==0) for n in scales}
    passed = pooled>=.005 and all(g>=-.01 for g in gains.values()) and sum(
        g>0 for g in gains.values())>=2 and all(all(x.values()) for x in guards.values())
    assert abs(pooled-summary['pooled_gain_teacher_vs_scratch']) < 1e-12
    assert passed == summary['development_screen_pass']
    for n in scales:
        s = str(n)
        assert abs(gains[s]-float(summary['scale_gains'][s])) < 1e-12
        assert guards[s] == summary['health_guards'][s]
        for a in ARMS:
            assert abs(statistics.fmean(auc[s][a])-summary['learning_auc'][s][a]['mean']) < 1e-12
            assert abs(statistics.fmean(final[s][a])-summary['fixed_final_cost_per_machine'][s][a]['mean']) < 1e-12
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),output/Path(__file__).name)
    result = dict(status='PASS',source_artifact=str(source),
        source_manifest_sha256=sha(source/'manifest.json'),
        verifier_sha256=sha(output/Path(__file__).name),
        completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started,profile=manifest['profile'],
        test_opened=manifest['test_opened'],development_episodes=len(rows),
        training_episodes=len(train),matched_training_tapes=len(tape),
        selected_steps=chosen,scale_gains=gains,pooled_gain=pooled,
        health_guards=guards,learning_auc=auc,fixed_final=final,
        development_screen_pass=passed,
        invocation=dict(executable=sys.executable,argv=sys.argv,cwd=str(Path.cwd())))
    (output/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    report=audit(args.artifact.resolve(),args.output_dir.resolve())
    print(json.dumps(dict(status=report['status'],pooled_gain=report['pooled_gain'],
        development_screen_pass=report['development_screen_pass'])))


if __name__=='__main__':main()
