"""Natural trajectories and explicitly generated resource-release probe states."""
from dataclasses import asdict
import itertools
import random
import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_assignment_confidence_benchmark import initial_state, physical_seed, probe_seed, future_seed
from ht_pdm_fjsp.maintenance_assignment_value_data import all_assignment_classes, save_forecast
from ht_pdm_fjsp.maintenance_assignment_value_forks import fork_roots, scalar_replay
from ht_pdm_fjsp.maintenance_assignment_value_policy import inputs
from ht_pdm_fjsp.maintenance_autoregressive_policy import feature_batch, CONTEXT
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_state, validate_matching
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_opportunity_cost import compact
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, transition


def probe_anchor(config, identity):
    rng = random.Random(probe_seed(identity))
    ages = tuple(rng.randrange(max(1, config.failure_age - 2)) for _ in range(4))
    remaining, assigned = [0, 0], [-1, -1]
    kind = rng.randrange(4)
    if kind:
        eligible = [(m, j) for m in range(4) for j in range(2) if config.service_time[m][j]]
        m, j = rng.choice(eligible)
        remaining[j], assigned[j] = min(kind, config.service_time[m][j]), m
    state = DispatchState(ages, (False,) * 4, (0,) * 4, tuple(remaining), tuple(assigned))
    validate_state(config, state)
    return state


def collect(panels, cfg, split, models, natural_journal, probe_journal, cohorts=None, shocks=None):
    cohorts = cfg[f"{split}_cohorts"] if cohorts is None else cohorts
    shocks = cfg[f"{split}_shocks"] if shocks is None else shocks
    records = []
    grid = list(itertools.product(cfg["conditions"], cohorts, shocks, cfg["model_seeds"]))
    for cond, cohort, shock, seed in tqdm(grid, desc=f"Natural + release states {split}", unit="trajectory"):
        c = panels[split, cohort, cond]; model = models[seed]
        env = WaitingEnv(c); es = physical_seed(cohort, shock)
        env.reset(es, initial_state(c, cohort))
        for tick in range(48):
            state = env.state; original, info = model.select(c, state, 48 - tick, 48)
            identity = dict(condition=cond, cohort=cohort, shock=shock, model_seed=seed, tick=tick, origin="natural")
            if tick in cfg["landmarks"]:
                records.append(dict(identity=identity, config=c, state=state, original=original))
            _, _, _, metrics = env.step(original)
            natural_journal.add({**identity, "split": split, "env_seed": es, "state": compact(asdict(state)),
                "action": compact(original), "following": compact(asdict(env.state)), "metrics": compact(metrics),
                "shock_bits": compact(list(env.events[tick])), "diagnostics": compact(info)})
        for tick in cfg["probe_landmarks"]:
            identity = dict(condition=cond, cohort=cohort, shock=shock, model_seed=seed, tick=tick, origin="probe")
            state = probe_anchor(c, identity)
            events = np.random.default_rng(probe_seed(identity)).random((2, 4)) < c.failure_probability
            for offset in range(2):
                t = tick - 2 + offset
                proposed, _ = model.select(c, state, 48 - t, 48)
                action = tuple(edge for edge in proposed if state.failed[edge[0]])
                following, _, info = transition(c, state, action, events[offset])
                probe_journal.add({**identity, "split": split, "prefix_tick": t, "probe_seed": probe_seed(identity),
                    "state": compact(asdict(state)), "action": compact(action), "following": compact(asdict(following)),
                    "shock_bits": compact(events[offset].tolist()), "metrics": compact(info),
                    "origin_contract": "generated valid anchor + two real CM-preserving/PM-deferred ticks"})
                state = following
            original, _ = model.select(c, state, 48 - tick, 48)
            validate_state(c, state); validate_matching(c, state, original)
            records.append(dict(identity=identity, config=c, state=state, original=original))
    if len(records) != len(grid) * (len(cfg["landmarks"]) + len(cfg["probe_landmarks"])):
        raise RuntimeError("source/probe state grid mismatch")
    return records


def forecast(output, cfg, record, roots, model, phase, scenarios, stats, filename):
    seed = future_seed(cfg, record["identity"], phase)
    r = fork_roots(record["config"], record["state"], roots, model, seed, scenarios,
        record["identity"]["tick"], record_trace=cfg["profile"] == "smoke")
    if cfg["profile"] == "smoke":
        scalar_replay(record["config"], record["state"], r, model, record["identity"]["tick"])
        stats["scalar_replayed_futures"] += len(roots) * scenarios
    stats[f"{phase}_transitions"] += r["transitions"]
    stats["forecast_transitions"] += r["transitions"]
    stats["forecast_futures"] += len(roots) * scenarios
    stats["forecast_roots"] += len(roots)
    for k, value in r["counts"].items():
        stats[k] += value
    path = output / "forecasts" / phase / f"{filename}.npz"
    save_forecast(path, record, roots, r, seed, phase)
    return r, path.relative_to(output).as_posix()


def make_dataset(output, cfg, split, records, models, journals, stats):
    datasets = {}
    for seed in cfg["model_seeds"]:
        selected = [r for r in records if r["identity"]["model_seed"] == seed]
        configs = [r["config"] for r in selected]; states = [r["state"] for r in selected]
        data = inputs(feature_batch(configs, states, [48 - r["identity"]["tick"] for r in selected], 48, CONTEXT), models[seed])
        pairs, targets, ses, conditions, origins, ticks, original_ids = [], [], [], [], [], [], []
        for si, record in enumerate(tqdm(selected, desc=f"{split} paired labels PPO {seed}", unit="state")):
            roots, groups = all_assignment_classes(record["config"], record["state"], record["original"])
            has_choice = any(len(g) > 1 for g in groups.values())
            r, path = forecast(output, cfg, record, roots, models[seed], split, cfg[f"{split}_scenarios"], stats, f"{seed}_{si}") if has_choice else (None, "")
            original_ids.append(CATALOGUE.index(record["original"]))
            journals["label_states"].add({**record["identity"], "split": split, "state_index": si,
                "state": compact(asdict(record["state"])), "original": compact(record["original"]),
                "roots": compact(roots), "forecast": path, "occupied_technicians": sum(x > 0 for x in record["state"].remaining),
                "classes_with_choice": sum(len(g) > 1 for g in groups.values()),
                "source_assignment_choices": len(next(g for g in groups.values() if 0 in g)),
                "singleton_classes": sum(len(g) == 1 for g in groups.values())})
            for machines, group in groups.items():
                if len(group) == 1:
                    continue
                left, right = sorted(group, key=lambda i: CATALOGUE.index(roots[i]))
                delta = (r["costs"][right] - r["costs"][left]) / cfg["reward_scale"]
                mean = float(delta.mean()); se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
                pairs.append([si, CATALOGUE.index(roots[left]), CATALOGUE.index(roots[right])])
                targets.append(mean); ses.append(se); conditions.append(cfg["conditions"].index(record["identity"]["condition"]))
                origins.append(int(record["identity"]["origin"] == "probe")); ticks.append(record["identity"]["tick"])
                journals["label_pairs"].add({**record["identity"], "split": split, "state_index": si,
                    "pm_machines": compact(machines), "left": compact(roots[left]), "right": compact(roots[right]),
                    "target_normalized": mean, "paired_mc_se_normalized": se, "original_class": 0 in group, "forecast": path})
        if not pairs:
            raise RuntimeError("no labels; no outcome-based resampling allowed")
        dataset = {**data, "pairs": torch.tensor(pairs, dtype=torch.long), "targets": torch.tensor(targets),
            "mc_se": torch.tensor(ses), "conditions": torch.tensor(conditions), "origins": torch.tensor(origins),
            "ticks": torch.tensor(ticks), "original_ids": torch.tensor(original_ids)}
        if not all(torch.isfinite(v).all() for v in dataset.values()):
            raise RuntimeError("nonfinite dataset")
        path = output / "datasets" / f"{split}_{seed}.pt"; path.parent.mkdir(exist_ok=True)
        torch.save(dataset, path)
        restored = torch.load(path, weights_only=True)
        if any(not torch.equal(v, restored[k]) for k, v in dataset.items()):
            raise RuntimeError("dataset roundtrip failed")
        datasets[seed] = dataset
    return datasets
