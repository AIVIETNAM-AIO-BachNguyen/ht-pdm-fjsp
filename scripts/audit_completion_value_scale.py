"""Offline value-head envelopes on opened Completion-Credit pilot DEV data."""
import argparse
import csv
from dataclasses import fields
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tqdm.auto import tqdm
from ht_pdm_fjsp import technician_completion_experiment as x
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.technician_reservation_env import graph_observation


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('diagnostic', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--profile', choices=('smoke', 'audit'), required=True)
    args = parser.parse_args()
    pilot, diagnostic = args.pilot.resolve(), args.diagnostic.resolve()
    pm = json.loads((pilot/'manifest.json').read_text())
    dm = json.loads((diagnostic/'manifest.json').read_text())
    assert pm['status'] == dm['status'] == 'COMPLETED' and pm['audit_status'] == 'PASS'
    assert pm['protocol'] == x.VERSION and pm['profile'] == 'pilot' and not pm['test_opened']
    assert dm['source_manifest_sha256'] == x.sha(pilot/'manifest.json')
    assert dm['scientific_test_opened'] is False and dm['parameter_updates'] == 0
    for folder, manifest in ((pilot, pm), (diagnostic, dm)):
        for name, binding in manifest['output_bindings'].items():
            p = folder/name
            assert p.stat().st_size == binding['bytes'] and x.sha(p) == binding['sha256'], name
    for p in Path(x.__file__).parent.glob('*.py'):
        assert x.sha(p) == pm['source_seal']['source_snapshot/'+p.name], p
    assert x.sha(x.PLAN) == pm['source_seal']['protocol.md']
    assert x.sha(x.REGISTRY) == pm['source_seal']['seed_registry.json']
    cfg = json.loads((pilot/'resolved_config.json').read_text())
    assert not cfg['test_seeds'] and not cfg['teacher_assistance']
    initial = {}
    for row in read(diagnostic/'physical.csv'):
        if row['phase'] == 'policy' and int(row['tick']) == 0:
            key = int(row['n']), int(row['development_seed'])
            state = json.loads(row['state'])
            assert initial.setdefault(key, state) == state
    dev = {(int(r['n']), r['method'], int(r['train_seed']), int(r['checkpoint_steps']), int(r['seed'])): r
           for r in read(pilot/'development_episodes.csv')}
    choices = json.loads((pilot/'checkpoint_choices.json').read_text())
    trains = cfg['train_seeds'][:1] if args.profile == 'smoke' else cfg['train_seeds']
    seeds = cfg['development_seeds'][:1] if args.profile == 'smoke' else cfg['development_seeds']
    jobs = [(n, arm, t, step) for n in cfg['scales'] for arm in x.arms(n) for t in trains
            for step in ([choices[f'{n}:{arm}:{t}']['steps']] if args.profile == 'smoke' else [0, *cfg['checkpoints']])]
    started = time.monotonic()
    out = args.output_root/(x.VERSION+'_value_scale_'+args.profile+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    plan = Path(__file__).resolve().parents[1]/'docs/technician_completion_v1_value_scale_audit_plan.md'
    shutil.copyfile(__file__, out/'audit_source.py')
    shutil.copyfile(plan, out/'audit_plan.md')
    result = dict(status='RUNNING', classification='OFFLINE_OPENED_DEV_DIAGNOSTIC',
        pilot=str(pilot), diagnostic=str(diagnostic), pilot_manifest_sha256=x.sha(pilot/'manifest.json'),
        diagnostic_manifest_sha256=x.sha(diagnostic/'manifest.json'), profile=args.profile,
        scientific_test_opened=False, optimizer_steps=0, parameter_updates=0,
        script_sha256=x.sha(Path(__file__)), plan_sha256=x.sha(plan))
    x.write_json(out/'manifest.json', result)
    torch.set_num_threads(1)
    episodes, summaries = [], []
    try:
        for n, arm, train, step in tqdm(jobs, desc='Frozen value-scale audit'):
            assert time.monotonic()-started < 120, 'registered hard stop'
            path = pilot/f'N{n}'/arm/str(train)/'checkpoints'/f'step_{step:09d}.pt'
            model = x.load_model(path, n, cfg, train, arm, step)
            before = x.learner.parameter_digest(model)
            graph = arm in x.GRAPH_ARMS
            head = model.value_head if graph else model.critic
            assert isinstance(head[-1], torch.nn.Linear) and isinstance(head[-2], torch.nn.Tanh)
            assert head[-1].out_features == 1
            scale = n/4 if graph else 1.
            weight_l1 = float(head[-1].weight.detach().abs().sum())
            bias = float(head[-1].bias.detach())
            lower, upper = (bias-weight_l1)*scale, (bias+weight_l1)*scale
            observations = []
            for seed in seeds:
                state = DispatchState(**{f.name: tuple(initial[n, seed][f.name]) for f in fields(DispatchState)})
                observations.append(graph_observation(x.scaled_config(n), state, 0))
            batch = x.learner.tensors({key: np.stack([o[key] for o in observations]) for key in observations[0]})
            hidden = []
            hook = head[-2].register_forward_hook(lambda module, inp, output: hidden.append(output.detach()))
            with torch.no_grad():
                if graph:
                    recorded = torch.zeros((len(seeds), n//2), dtype=torch.long)
                    values = model.trace(batch, recorded=recorded)[-1][:, 0]
                else:
                    values = head(batch['state']).squeeze(-1)
            hook.remove()
            saturation = (hidden[0].abs() >= .99).float().mean(1).numpy()
            assert (values >= lower-1e-5).all() and (values <= upper+1e-5).all()
            panel = []
            for index, seed in enumerate(seeds):
                observed = -float(dev[n, arm, train, step, seed]['extended_cost'])/20
                row = dict(n=n, method=arm, train_seed=train, checkpoint_steps=step, development_seed=seed,
                    value_lower_bound=lower, value_upper_bound=upper, prediction=float(values[index]),
                    observed_greedy_return=observed, return_outside_envelope=observed < lower or observed > upper,
                    last_hidden_saturation=float(saturation[index]))
                episodes.append(row); panel.append(row)
            assert x.learner.parameter_digest(model) == before
            summaries.append(dict(n=n, method=arm, train_seed=train, checkpoint_steps=step,
                value_lower_bound=lower, value_upper_bound=upper,
                prediction_mean=float(values.mean()), prediction_std=float(values.std(unbiased=False)),
                observed_greedy_return_mean=float(np.mean([r['observed_greedy_return'] for r in panel])),
                return_outside_envelope_fraction=float(np.mean([r['return_outside_envelope'] for r in panel])),
                last_hidden_saturation_mean=float(np.mean(saturation)), checkpoint_sha256=x.sha(path)))
        assert len(episodes) == len(jobs)*len(seeds)
        x.write_csv(out/'per-checkpoint.csv', summaries)
        x.write_csv(out/'per-episode.csv', episodes)
        lines = ['# Frozen value-scale audit', '',
            'Offline opened DEV only. Greedy trajectory returns are not unbiased calibration targets for stochastic-policy critics.',
            'Envelopes are mathematical bounds for each frozen head, not limitations of a retrained architecture.', '',
            '| N | Method | Seed | Step | Lower | Upper | Mean value | Mean observed return | Outside fraction | Hidden saturation |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in summaries:
            lines.append(f'| {r["n"]} | {r["method"]} | {r["train_seed"]} | {r["checkpoint_steps"]} | '+
                ' | '.join(f'{r[k]:.6g}' for k in ('value_lower_bound','value_upper_bound','prediction_mean','observed_greedy_return_mean','return_outside_envelope_fraction','last_hidden_saturation_mean'))+' |')
        (out/'findings.md').write_text('\n'.join(lines)+'\n')
        result.update(status='COMPLETED', checkpoints=len(jobs), initial_predictions=len(episodes),
            frozen_parameter_check='PASS', envelope_check='PASS', elapsed_seconds=time.monotonic()-started)
    except Exception as exc:
        result.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['output_bindings'] = {p.name: dict(sha256=x.sha(p), bytes=p.stat().st_size)
            for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}
        x.write_json(out/'manifest.json', result)
    print(out)


if __name__ == '__main__':
    main()
