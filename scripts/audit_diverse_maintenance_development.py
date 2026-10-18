"""Independent read-only C1 DEV gate and provenance audit from raw artifacts."""

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


PROTOCOL='technician_diverse_maintenance_v1'
ARMS=('reservation_ppo','mask_only_ppo','independent_ppo','scratch_reservation_ppo')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def check_episode(row):
    assert int(float(row['invalid_assignments']))==0
    assert int(row['requests'])==int(row['completions'])+int(row['tail_pending'])+int(row['active_end'])
    assert int(row['starts'])==int(row['completions'])+int(row['active_end'])
    total=sum(float(row[k]) for k in ('maintenance_cost','unavailability_cost',
        'failure_cost','waiting_cost','terminal_liability'))
    assert abs(total-float(row['extended_cost']))<1e-7


def audit(source,output):
    started=time.monotonic()
    manifest=load(source/'manifest.json')
    assert manifest['protocol']==PROTOCOL and manifest['status']=='COMPLETED'
    assert manifest['audit_status']=='PASS' and manifest['profile'] in ('smoke','development')
    if manifest['profile']=='development':assert not manifest['test_opened']
    for name,digest in tqdm(manifest['source_seal'].items(),desc='C1 source seals'):
        assert sha(source/name)==digest,name
    for name,meta in tqdm(manifest['output_bindings'].items(),desc='C1 output bindings'):
        path=source/name
        assert path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256'],name
    cfg=load(source/'resolved_config.json')
    configs=[tuple(c) for c in cfg['configs']]
    choices=load(source/'checkpoint_choices.json')
    reported=load(source/'summary.json')
    reference_choice=load(source/'reference_selection.json')
    audits=load(source/'training_audit.json')
    initials=load(source/'initial_parameter_hashes.json')
    warm=load(source/'warm_choices.json')
    dev=rows(source/'development_episodes.csv')
    heuristics=rows(source/'development_reference.csv')
    steps=[0,*cfg['checkpoints']]
    seeds=cfg['train_seeds'];physical=cfg['development_seeds']
    expected={(f,n,a,t,s,p) for f,n in configs for a in ARMS
        for t in seeds for s in steps for p in physical}
    key=lambda r:(r['family'],int(r['n']),r['method'],int(r['train_seed']),
                  int(r['checkpoint_steps']),int(r['seed']))
    indexed={key(r):r for r in dev}
    assert len(dev)==len(expected) and set(indexed)==expected
    reference_grid={(f,n,a,p) for f,n in configs for a in ('fcfs','risk') for p in physical}
    ref={(r['family'],int(r['n']),r['method'],int(r['seed'])):r for r in heuristics}
    assert len(heuristics)==len(reference_grid) and set(ref)==reference_grid
    tapes={}
    for r in tqdm([*dev,*heuristics],desc='C1 paired DEV physics',unit='episode'):
        check_episode(r)
        k=r['family'],int(r['n']),int(r['seed'])
        tape=r['physical_seed'],r['population_sha256']
        if k in tapes:assert tapes[k]==tape
        else:tapes[k]=tape
    assert len(audits)==len(configs)*len(ARMS)*len(seeds)
    for item in audits:
        assert item['learned_steps']==cfg['env_steps']
        assert item['physical_ticks']==cfg['env_steps']//48*60
        assert item['training_episodes']==cfg['env_steps']//48
        assert item['optimizer_steps']>0 and item['changed']
        if item['arm']=='scratch_reservation_ppo':
            assert item['warm_state_exposures']==item['warm_optimizer_steps']==0
            assert item['bank_sha256'] is None
        else:
            assert item['warm_state_exposures']==cfg['teacher_bank_steps']
            assert item['warm_optimizer_steps']>0 and item['bank_sha256']
    for f,n in configs:
        for t in seeds:
            assert len({initials[f'{f}:N{n}:{a}:{t}'] for a in ARMS})==1
    assert len(warm)==len(seeds)
    for t in seeds:
        bank=source/'warm'/str(t)/'bank.npz'
        assert sha(bank)==warm[str(t)]['bank']['bank_sha256']
    training_keys=set();training_tapes={}
    with (source/'training_episodes.csv').open(newline='') as stream:
        iterator=csv.DictReader(stream)
        for r in tqdm(iterator,desc='C1 training tapes',unit='episode'):
            f,n,a,t,w,e=(r['family'],int(r['n']),r['arm'],int(r['train_seed']),
                         int(r['worker']),int(r['episode_index']))
            k=f,n,t,w,e
            identity=k+(a,)
            assert identity not in training_keys
            training_keys.add(identity)
            tape=r['physical_seed'],r['population_sha256']
            if k in training_tapes:assert training_tapes[k]==tape
            else:training_tapes[k]=tape
            assert abs(float(r['episode_reward'])*20+float(r['extended_cost']))<1e-5
            check_episode(r)
    expected_training={(f,n,t,w,e,a) for f,n in configs for t in seeds
        for w in range(cfg['vector_envs'])
        for e in range(cfg['env_steps']//(48*cfg['vector_envs'])) for a in ARMS}
    assert training_keys==expected_training
    means={};health={};selected={};auc={};final={}
    for f,n in configs:
        label=f'{f}:N{n}';means[label]={};health[label]={};auc[label]={};final[label]={}
        for a in ARMS:
            costs=[];picked_rows=[];aucs=[];finals=[]
            for t in seeds:
                bystep={s:statistics.fmean(float(indexed[f,n,a,t,s,p]['extended_cost'])/n
                    for p in physical) for s in steps}
                chosen=min(steps,key=lambda s:(bystep[s],s))
                assert choices[f'{label}:{a}:{t}']['steps']==chosen
                selected[f'{label}:{a}:{t}']=chosen
                costs.append(bystep[chosen]);finals.append(bystep[cfg['env_steps']])
                aucs.append(float(np.trapezoid([bystep[s] for s in steps],steps)
                    /(steps[-1]-steps[0])))
                picked_rows.extend(indexed[f,n,a,t,chosen,p] for p in physical)
            means[label][a]=statistics.fmean(costs)
            health[label][a]={m:statistics.fmean(float(r[m]) for r in picked_rows)
                for m in ('availability','normalized_pending','normalized_failures','invalid_assignments')}
            auc[label][a]=aucs;final[label][a]=finals
    heuristic_means={a:statistics.fmean(float(ref[f,n,a,p]['extended_cost'])/n
        for f,n in configs for p in physical) for a in ('fcfs','risk')}
    selected_rule=min(('fcfs','risk'),key=lambda a:(heuristic_means[a],('fcfs','risk').index(a)))
    assert selected_rule==reported['selected_rule']==reference_choice['selected_rule']
    assert not reference_choice['test_opened']
    full=statistics.fmean(means[f'{f}:N{n}']['reservation_ppo'] for f,n in configs)
    mask=statistics.fmean(means[f'{f}:N{n}']['mask_only_ppo'] for f,n in configs)
    pooled=1-full/mask
    rule_gain=1-full/heuristic_means[selected_rule]
    family_gains={f:1-sum(means[f'{f}:N{n}']['reservation_ppo'] for ff,n in configs if ff==f)/
        sum(means[f'{f}:N{n}']['mask_only_ppo'] for ff,n in configs if ff==f)
        for f in {f for f,_ in configs}}
    health_guards={}
    for f in family_gains:
        labels=[f'{f}:N{n}' for ff,n in configs if ff==f]
        diffs={m:statistics.fmean(health[label]['reservation_ppo'][m]-
            health[label]['mask_only_ppo'][m] for label in labels)
            for m in ('availability','normalized_pending','normalized_failures')}
        health_guards[f]=dict(availability=diffs['availability']>=-.01,
            pending=diffs['normalized_pending']<=.025,
            failures=diffs['normalized_failures']<=.25,
            invalid=all(health[label][a]['invalid_assignments']==0
                for label in labels for a in ARMS))
    passed=bool(pooled>=.005 and all(g>=-.01 for g in family_gains.values())
        and rule_gain>=-.01 and all(all(v.values()) for v in health_guards.values()))
    assert abs(pooled-reported['pooled_gain_full_vs_mask'])<1e-12
    assert abs(rule_gain-reported['pooled_gain_full_vs_heuristic'])<1e-12
    assert passed==reported['development_screen_pass']
    for f in family_gains:
        assert abs(family_gains[f]-reported['family_gains'][f])<1e-12
        assert health_guards[f]==reported['health_guards'][f]
    for label in means:
        for a in ARMS:
            assert abs(statistics.fmean(auc[label][a])-reported['learning_auc'][label][a]['mean'])<1e-12
            assert abs(statistics.fmean(final[label][a])-reported['fixed_final_cost_per_machine'][label][a]['mean'])<1e-12
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),output/Path(__file__).name)
    report=dict(status='PASS',source_artifact=str(source),
        source_manifest_sha256=sha(source/'manifest.json'),
        verifier_sha256=sha(output/Path(__file__).name),
        completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started,
        classification='INDEPENDENT_DEVELOPMENT_AUDIT',
        scientific_test_opened=manifest['profile']=='development' and manifest['test_opened'],
        training_episodes=len(training_keys),paired_training_tapes=len(training_tapes),
        development_episodes=len(dev),paired_development_tapes=len(tapes),
        selected_rule=selected_rule,heuristic_means=heuristic_means,
        pooled_gain_full_vs_mask=pooled,pooled_gain_full_vs_heuristic=rule_gain,
        family_gains=family_gains,health_guards=health_guards,
        development_screen_pass=passed,selected_steps=selected,
        invocation=dict(executable=sys.executable,argv=sys.argv,cwd=str(Path.cwd())))
    (output/'audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    result=audit(args.artifact.resolve(),args.output_dir.resolve())
    print(json.dumps(dict(status=result['status'],
        development_screen_pass=result['development_screen_pass'],
        pooled_gain_full_vs_mask=result['pooled_gain_full_vs_mask'])))


if __name__=='__main__':main()
