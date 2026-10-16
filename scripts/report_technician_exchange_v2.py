"""Read-only report for completed teacher-free Exchange pilot/development."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify_training_grid(source, c, expected):
    # Reconstruct the physical sample grid independently of training_audit.json.
    # Stream the journal: full development contains nearly a million episodes.
    worker_counts, seen, paired = Counter(), set(), {}
    expected_fits = set(expected)
    episodes_per_worker = c['env_steps'] // (48*c['vector_envs'])
    assert episodes_per_worker*48*c['vector_envs'] == c['env_steps']
    with (source/'training_episodes.csv').open() as stream:
        for row in csv.DictReader(stream):
            n,a,t = int(row['n']),row['algorithm'],int(row['train_seed'])
            w,e = int(row['worker']),int(row['episode_index'])
            assert (n,a,t) in expected_fits
            assert 0 <= w < c['vector_envs'] and 0 <= e < episodes_per_worker
            identity = (n,a,t,w,e)
            assert identity not in seen
            seen.add(identity)
            worker_counts[n,a,t,w] += 1
            assert float(row['learned_episode_steps']) == 48
            assert float(row['physical_ticks']) == 60
            assert float(row['invalid_assignments']) == 0
            assert 0 < int(row['learned_steps']) <= c['env_steps']
            tape = (int(row['physical_seed']),row['population_sha256'])
            assert len(tape[1]) == 64 and all(ch in '0123456789abcdef' for ch in tape[1])
            key = (n,t,w,e)
            if key in paired:
                assert paired[key] == tape, key
            else:
                paired[key] = tape
            if a == 'exchange_ppo':
                assert float(row['broker_repair_cancellations']) == 0
    assert len(seen) == len(expected)*c['env_steps']//48
    assert set(worker_counts) == {(n,a,t,w) for n,a,t in expected for w in range(c['vector_envs'])}
    assert all(count == episodes_per_worker for count in worker_counts.values())
    assert len(paired) == len(c['scales'])*len(c['train_seeds'])*c['env_steps']//48
    training_episode_count, paired_tape_count = len(seen), len(paired)
    del seen, paired
    return training_episode_count, paired_tape_count


def report(source, output_root, worktree):
    manifest_path = source / 'manifest.json'
    original_manifest_hash = sha(manifest_path)
    m = json.loads(manifest_path.read_text())
    assert m['protocol'] == 'technician_exchange_ppo_v2'
    assert m['status'] == 'COMPLETED' and m['audit_status'] == 'PASS'
    assert not m['test_opened'] and not m['teacher_assistance']
    # Verify every bound output, including checkpoints and per-fit journals.
    for name, binding in m['output_bindings'].items():
        p = source / name
        assert p.stat().st_size == binding['bytes'] and sha(p) == binding['sha256'], name
    current = {}
    for name, digest in m['source_seal'].items():
        if name.startswith('source_snapshot/'):
            p = worktree / 'src/ht_pdm_fjsp' / Path(name).name
        elif name.startswith('registry_snapshot/'):
            p = worktree / 'configs' / Path(name).name
        elif name == 'protocol.md':
            p = worktree / 'docs/technician_exchange_ppo_v2_plan.md'
        elif name == 'seed_registry.json':
            p = worktree / 'configs/technician_exchange_ppo_v2_seed_registry.json'
        else:
            continue
        assert sha(p) == digest, str(p)
        current[str(p)] = digest
    s = json.loads((source / 'summary.json').read_text())
    c = json.loads((source / 'resolved_config.json').read_text())
    audits = json.loads((source / 'training_audit.json').read_text())
    choices = json.loads((source / 'checkpoint_choices.json').read_text())
    params = json.loads((source / 'parameter_counts.json').read_text())
    initials = json.loads((source / 'initial_parameter_hashes.json').read_text())
    assert s['classification'] == 'DEVELOPMENT_ONLY' and not s['test_opened']
    assert not s['teacher_assistance'] and c['profile'] in ('pilot', 'development')
    core = ['exchange_ppo','unrestricted_exchange_ppo','inverse_mask_ppo','self_repair_ppo','construct_only_ppo','independent_ppo']
    graph = core + ['completion_reference_ppo']
    expected = [(n, a, t) for n in c['scales']
                for a in graph + (['ppo', 'normalized_ppo', 'mappo', 'full_mappo'] if n == 4 else [])
                for t in c['train_seeds']]
    assert len(audits) == len(expected) == m['training_models']
    assert {(a['n'], a['algorithm'], a['train_seed']) for a in audits} == set(expected)
    assert all(a['from_scratch'] and a['pretraining_steps'] == 0
               and a['learned_steps'] == c['env_steps'] for a in audits)
    expected_updates = c['env_steps']//c['rollout']*c['epochs']*(c['rollout']//c['minibatch'])
    assert all(a['optimizer_steps'] == expected_updates and a['physical_ticks'] == c['env_steps']//48*60 for a in audits)
    assert all(a['training_episodes'] == c['env_steps']//48 and a['changed']
               and a['behavior_normalized_max_error'] <= 1 for a in audits)
    assert c['hidden'] == 27 and c['baseline_hidden'] == 64
    if c['profile'] == 'pilot':
        assert (c['scales'], c['env_steps'], c['vector_envs'], c['rollout'], c['epochs'], c['minibatch'], c['checkpoints']) == ([4,32],61440,8,768,4,128,[30720,61440])
    else:
        assert (c['scales'], c['env_steps'], c['vector_envs'], c['rollout'], c['epochs'], c['minibatch'], c['checkpoints']) == ([4,8,16,32],491520,16,3072,4,256,[122880,245760,368640,491520])
    registry = json.loads((source/'seed_registry.json').read_text())
    panel = registry['panels'][c['profile']]
    for name in ('train_seeds','analysis_seed'):
        assert c[name] == panel[name], name
    assert c['development_seeds'] == list(range(panel['development_start'], panel['development_start']+panel['development_count']))
    assert c['value_unit_formula'] == 'unavailable_cost*N*48/20'
    assert c['confidence'] == 1-(.05/6)/53 and c['confirmation_round'] == 2
    for n in c['scales']:
        for t in c['train_seeds']:
            assert len({initials[f'{n}:{a}:{t}'] for a in core}) == 1
            assert len({params[f'{n}:{a}:{t}'] for a in core}) == 1
            assert params[f'{n}:exchange_ppo:{t}'] == 19255
            assert params[f'{n}:completion_reference_ppo:{t}'] == 18444
    training_episode_count, paired_tape_count = verify_training_grid(source,c,expected)
    rows = read_rows(source / 'development_episodes.csv')
    metrics = ('normalized_cost', 'availability', 'normalized_pending', 'normalized_failures')
    steps = [0, *c['checkpoints']]
    curves, replica_values = {}, []
    for n, a, t in expected:
        data = [r for r in rows if int(r['n']) == n and r['method'] == a and int(r['train_seed']) == t]
        by_step = {}
        for step in steps:
            panel = [r for r in data if int(r['checkpoint_steps']) == step]
            assert len(panel) == len(c['development_seeds'])
            assert sorted(int(r['seed']) for r in panel) == sorted(c['development_seeds'])
            assert all(float(r['invalid_assignments']) == 0 for r in panel)
            by_step[step] = {k: float(np.mean([float(r[k]) for r in panel])) for k in metrics}
        selected = min(steps, key=lambda st: (by_step[st]['normalized_cost'], st))
        key = f'{n}:{a}:{t}'
        assert selected == s['selected_steps'][key] == choices[key]['steps']
        curves[n, a, t] = [by_step[st]['normalized_cost'] for st in steps]
        replica_values.append(dict(n=n, method=a, train_seed=t, selected_steps=selected,
                                   parameters=params[key], **by_step[selected]))
    for n in c['scales']:
        for a in graph + (['ppo', 'normalized_ppo', 'mappo', 'full_mappo'] if n == 4 else []):
            panel = [r for r in replica_values if r['n'] == n and r['method'] == a]
            for metric in metrics:
                assert np.isclose(np.mean([r[metric] for r in panel]), s['means'][str(n)][a][metric], rtol=0, atol=1e-10)
    references = read_rows(source / 'development_reference.csv')
    for n in c['scales']:
        panel = [r for r in references if int(r['n']) == n]
        assert sorted(int(r['seed']) for r in panel) == sorted(c['development_seeds'])
        for metric in metrics:
            assert np.isclose(np.mean([float(r[metric]) for r in panel]), s['means'][str(n)]['deadline'][metric], rtol=0, atol=1e-10)
    # Independently reconstruct all endpoints; do not call the experiment's
    # estimate/threshold helpers whose results this report is checking.
    point = {}
    for n in c['scales']:
        candidate = s['means'][str(n)]['exchange_ppo']
        others = ['unrestricted_exchange_ppo','inverse_mask_ppo','self_repair_ppo','construct_only_ppo','independent_ppo','completion_reference_ppo']
        others += ['ppo', 'normalized_ppo', 'mappo', 'full_mappo'] if n == 4 else ['deadline']
        for other in others:
            point[f'N{n}_gain_vs_{other}'] = 1-candidate['normalized_cost']/s['means'][str(n)][other]['normalized_cost']
        for reference in (['ppo', 'normalized_ppo'] if n == 4 else ['deadline']):
            baseline = s['means'][str(n)][reference]
            for metric, label in zip(metrics[1:], ['availability', 'pending_per_machine', 'failures_per_machine']):
                point[f'N{n}_{label}_difference_vs_{reference}'] = candidate[metric]-baseline[metric]
        if n != 4:
            point[f'N{n}_cost_ratio_vs_N4'] = candidate['normalized_cost']/s['means']['4']['exchange_ppo']['normalized_cost']
    large = [n for n in c['scales'] if n != 4]
    for label, other in [('commitment_vs_unrestricted','unrestricted_exchange_ppo'),('commitment_vs_inverse','inverse_mask_ppo'),('exchange','self_repair_ppo'),('repair','construct_only_ppo')]:
        point[f'pooled_{label}_gain'] = float(np.mean([point[f'N{n}_gain_vs_{other}'] for n in large]))
    assert set(point) == set(s['estimates']) == set(s['screen'])
    for key, value in point.items():
        assert np.isclose(value, s['estimates'][key], rtol=0, atol=1e-10), key
        if key.endswith('cost_ratio_vs_N4'): passed = value <= 1.10
        elif key.startswith('pooled_'): passed = value >= .005
        elif 'availability_difference' in key: passed = value >= -.01
        elif 'pending_per_machine_difference' in key: passed = value <= .025
        elif 'failures_per_machine_difference' in key: passed = value <= .25
        elif key == 'N4_gain_vs_mappo' or ('gain_vs_independent_ppo' in key and not key.startswith('N4')): passed = value > 0
        else: passed = value >= (-.02 if key.endswith('gain_vs_deadline') else -.01)
        assert bool(passed) == s['screen'][key], key
    assert s['development_screen_pass'] == all(s['screen'].values())
    assert len(point) == (31 if c['scales'] == [4, 32] else 53)
    assert set(choices) == {f'{n}:{a}:{t}' for n,a,t in expected}
    for row in rows:
        if row['method'] == 'exchange_ppo':
            assert float(row['broker_repair_cancellations']) == 0
    assert sha(manifest_path) == original_manifest_hash
    out = output_root / (m['protocol'] + '_' + c['profile'] + '_report_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    colors = ['#2166ac','#d6604d','#8c6bb1','#5c8655','#a6761d','#17becf','#888888','#333333','#e377c2','#e68a00','#008080']
    methods = graph + ['ppo', 'normalized_ppo', 'mappo', 'full_mappo']
    fig, axs = plt.subplots(1, len(c['scales']), figsize=(7 * len(c['scales']), 5.5), squeeze=False)
    for ax, n in zip(axs[0], c['scales']):
        for a, color in zip(methods, colors):
            values = [curves[n, a, t] for t in c['train_seeds'] if (n, a, t) in curves]
            if not values:
                continue
            values = np.array(values)
            for v in values:
                ax.plot(np.array(steps) / 1000, v, color=color, alpha=.3, linewidth=.7)
            ax.plot(np.array(steps) / 1000, values.mean(0), '-o', color=color, label=a)
            ax.fill_between(np.array(steps) / 1000, values.min(0), values.max(0), color=color, alpha=.08)
        ax.axhline(s['means'][str(n)]['deadline']['normalized_cost'], color='#555555', linestyle='--', label='deadline')
        ax.set(title=f'N{n}, K{n // 2}', xlabel='Joint decisions / replica (thousands)', ylabel='Development H+L cost / machine')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=.15)
    fig.suptitle(f'Exchange PPO v2: {c["profile"]}, {len(c["train_seeds"])} replicas, no scientific test')
    fig.text(.5, .01, 'Faint lines: all replicas. Shading: min–max, not CI. All fits spend the registered budget.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .93))
    fig.savefig(out / 'learning_curves.png', dpi=180)
    fig.savefig(out / 'learning_curves.pdf')
    plt.close(fig)
    with (out / 'selected_replicas.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(replica_values[0]))
        writer.writeheader()
        writer.writerows(replica_values)
    lines = ['# Exchange teacher-free findings', '',
             'DEVELOPMENT_ONLY. No scientific test opened; no scientific confidence interval or alpha spent.',
             f'Source: `{source}`. Frozen commit: `{m["git"]["commit"]}`.',
             f'Independent report verification: all {len(m["output_bindings"])} output bindings and {len(current)} current source/protocol/registry bindings match.',
             f'All {len(audits)} fits from scratch; {c["env_steps"]} decisions per fit. No teacher/pretraining. Elapsed {m["elapsed_seconds"]:.1f} seconds.',
             f'Independently reconstructed {training_episode_count} training episodes and {paired_tape_count} paired physical tapes; all assignments valid.',
             '', '| N | Method | Cost/machine | Availability | Pending/machine | Failures/machine |',
             '|---|---|---:|---:|---:|---:|']
    for n, panel in s['means'].items():
        for a, v in panel.items():
            lines.append(f'| {n} | {a} | {v["normalized_cost"]:.4f} | {v["availability"]:.6f} | {v["normalized_pending"]:.6f} | {v["normalized_failures"]:.6f} |')
    lines += ['', '## Every registered point screen', '', '| Endpoint | Estimate | Screen |', '|---|---:|---|']
    for key, passed in s['screen'].items():
        label = key
        lines.append(f'| {label} | {s["estimates"][key]:.8f} | {"PASS" if passed else "FAIL"} |')
    lines += ['', 'Candidate vs unrestricted exchange tests ordered commitment with identical before-mask inputs, architecture,2K slots and training budget.',
              'Candidate vs inverse-mask tests prefix commitment beyond blocking only the preceding-slot inverse; candidate vs self adds forward bilateral permissions under prefix immutability.',
              'Construct-only computes all2K slots but masks repairs to KEEP; that comparison includes an actionable second pass.',
              'Fresh completion_reference_ppo has18444 parameters and K slots with prefix credit; it is not an equal-internal-compute ablation.',
              'All selected replicas and parameter counts are in selected_replicas.csv. No replica is dropped.',
              '', '## Descriptive learning diagnostics', '',
              '| N | Method | Seed | Seconds incl. dev | Actor gradient | Critic gradient | Actor update | Critic update | Raw advantage SD | Prefix baseline spread |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    progress = read_rows(source / 'training_progress.csv')
    fields = ['actor_grad_norm', 'critic_grad_norm', 'actor_update_norm', 'critic_update_norm', 'allocation_advantage_std', 'prefix_baseline_spread']
    for audit in audits:
        if audit['algorithm'] not in graph:
            continue
        panel = [r for r in progress if int(r['n']) == audit['n'] and r['algorithm'] == audit['algorithm'] and int(r['train_seed']) == audit['train_seed']]
        vals = [float(np.mean([float(r[k]) for r in panel])) for k in fields]
        lines.append(f'| {audit["n"]} | {audit["algorithm"]} | {audit["train_seed"]} | {audit["seconds_including_development"]:.2f} | ' + ' | '.join(f'{v:.8g}' for v in vals) + ' |')
    lines += ['', '## Value learning diagnostics (descriptive)', '',
              '| N | Method | Seed | Mean normalized return SD | Mean normalized value SD | Mean normalized prefix spread |',
              '|---|---|---:|---:|---:|---:|']
    for audit in audits:
        if audit['algorithm'] not in graph: continue
        panel = [r for r in progress if int(r['n']) == audit['n'] and r['algorithm'] == audit['algorithm'] and int(r['train_seed']) == audit['train_seed']]
        vals = [float(np.mean([float(r[k]) for r in panel])) for k in ['normalized_return_std', 'normalized_value_std', 'normalized_prefix_baseline_spread']]
        lines.append(f'| {audit["n"]} | {audit["algorithm"]} | {audit["train_seed"]} | '+' | '.join(f'{v:.8g}' for v in vals)+' |')
    lines += ['', '## Selected checkpoint probes (first DEV observation only)', '',
              '| N | Method | Seed | Selected step | Value unit | Normalized prediction | Last hidden saturation |',
              '|---|---|---:|---:|---:|---:|---:|']
    for n, a, t in expected:
        choice = choices[f'{n}:{a}:{t}']
        probe = choice['value_probe']
        unit = 1. if a == 'ppo' else 12*n*48/20
        assert probe['value_unit'] == unit
        assert probe['physical_output_lower_bound']-1e-5 <= probe['physical_prediction_mean'] <= probe['physical_output_upper_bound']+1e-5
        assert 0 <= probe['hidden_saturation'] <= 1
        assert np.isclose(probe['normalized_prediction_mean'], probe['physical_prediction_mean']/unit, atol=2e-6, rtol=1e-6)
        lines.append(f'| {n} | {a} | {t} | {choice["steps"]} | {unit:.4f} | {probe["normalized_prediction_mean"]:.8f} | {probe["hidden_saturation"]:.6f} |')
    lines += ['', 'Original PPO retains its original critic units; all normalized baselines and graph arms share the fixed physical unit.',
              'Checkpoint probes and training spreads do not establish critic calibration, causal variance reduction or scientific quality.',
              'Core critic consumes physical state only, once per physical decision; prefix baseline spread is zero by construction.',
              'All four N32-only pilot mechanism proxies are exploratory; the registered full pool is N8/N16/N32.']
    lines += ['', 'These diagnostics do not establish variance reduction or causal performance improvements.',
              'Pilot comparisons use equally short baseline fits and do not establish superiority over full-budget PPO.',
              'Pilot alone never authorizes confirmation. Full development and the registered prerequisites are required.',
              'Failed nonpooled pilot cost/health screens prohibit full development. Preserve negative results.']
    lines += ['', '## Selected DEV broker activity (descriptive)', '',
              '| N | Method | Seed | Exchanges/episode | Revisions/episode | Transfers/episode | Drops/episode | Conflicts/episode | Net plan changes/episode | Repair cancellations/episode |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for replica in replica_values:
        n,a,t,step = (replica[k] for k in ('n','method','train_seed','selected_steps'))
        panel = [r for r in rows if int(r['n']) == n and r['method'] == a and int(r['train_seed']) == t and int(r['checkpoint_steps']) == step]
        vals = [float(np.mean([float(r[k]) for r in panel])) for k in ('broker_exchanges','broker_revisions','broker_transfers','broker_drops','broker_conflicts','broker_net_plan_changes','broker_repair_cancellations')]
        lines.append(f'| {n} | {a} | {t} | ' + ' | '.join(f'{v:.6f}' for v in vals) + ' |')
    lines += ['', 'Exchange events include transfers as a subset. Net changes refer to feasible matchings for coordinated arms and proposals for independent.',
              'Activity or zero cancellation does not prove beneficial exchanges, causal mediation, or improvement over matched controls.',
              'Shared current graph/broker is assumed; this is not a network-local decentralized implementation.']
    (out / 'findings.md').write_text('\n'.join(lines) + '\n')
    shutil.copyfile(__file__, out / 'report_source.py')
    assert sha(manifest_path) == original_manifest_hash
    verification = dict(classification='DEVELOPMENT_ONLY', source=str(source),
        source_manifest_sha256=original_manifest_hash, verified_output_bindings=len(m['output_bindings']),
        verified_training_episodes=training_episode_count, verified_paired_tapes=paired_tape_count,
        verified_current_bindings=current, script_sha256=sha(Path(__file__)),
        numpy=np.__version__, matplotlib=matplotlib.__version__,
        outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    (out / 'report_manifest.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--worktree', type=Path, required=True)
    args = parser.parse_args()
    report(args.source.resolve(), args.output_root.resolve(), args.worktree.resolve())
