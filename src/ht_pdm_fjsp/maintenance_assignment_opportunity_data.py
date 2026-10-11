"""Actual source assignment classes; shared full tapes and nested low targets."""
from dataclasses import asdict
import itertools
import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_assignment_opportunity_benchmark import initial_state, physical_seed, future_seed, ARMS
from ht_pdm_fjsp.maintenance_assignment_value_data import save_forecast
from ht_pdm_fjsp.maintenance_assignment_value_forks import candidates, fork_roots, scalar_replay
from ht_pdm_fjsp.maintenance_assignment_value_policy import inputs
from ht_pdm_fjsp.maintenance_autoregressive_policy import feature_batch, CONTEXT
from ht_pdm_fjsp.maintenance_joint_policy import CATALOGUE
from ht_pdm_fjsp.maintenance_opportunity_cost import compact
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv


def source_class(config, state, original):
    roots, groups, _ = candidates(config, state, original)
    result = tuple(roots[i] for i in groups["assignment"])
    if not 1 <= len(result) <= 2 or result[0] != original:
        raise RuntimeError("source assignment class contract")
    return result


def collect(panels, cfg, split, models, journal, coverage, cohorts=None, shocks=None):
    cohorts = cfg[f"{split}_cohorts"] if cohorts is None else cohorts
    shocks = cfg[f"{split}_shocks"] if shocks is None else shocks
    records = []
    grid = list(itertools.product(cfg["conditions"], cohorts, shocks, cfg["model_seeds"]))
    for cond, cohort, shock, seed in tqdm(grid, desc=f"Natural {split} opportunities", unit="trajectory"):
        c = panels[split, cohort, cond]; env = WaitingEnv(c); es = physical_seed(cohort, shock)
        env.reset(es, initial_state(c, cohort)); visits = np.zeros(4, int)
        for tick in range(48):
            state = env.state; original, info = models[seed].select(c, state, 48-tick, 48)
            roots = source_class(c, state, original)
            identity = dict(condition=cond, cohort=cohort, shock=shock, model_seed=seed, tick=tick, origin="natural")
            if len(roots) > 1:
                records.append(dict(identity=identity, config=c, state=state, original=original, roots=roots))
                visits[tick//12] += 1
            _, _, _, metrics = env.step(original)
            journal.add({**identity, "split": split, "env_seed": es, "state": compact(asdict(state)),
                "action": compact(original), "following": compact(asdict(env.state)), "metrics": compact(metrics),
                "shock_bits": compact(list(env.events[tick])), "diagnostics": compact(info), "source_assignment_choices": len(roots)})
        coverage.add(dict(condition=cond, cohort=cohort, shock=shock, model_seed=seed, split=split,
            visited_ticks=48, opportunities=int(visits.sum()), early_opportunities=int(visits[0]),
            late_opportunities=int(visits[1:].sum()), counts_by_clock_bin=compact(visits.tolist())))
    return records


def forecast(output, cfg, record, roots, model, phase, scenarios, stats, filename):
    seed = future_seed(cfg, record["identity"], phase)
    r = fork_roots(record["config"], record["state"], roots, model, seed, scenarios,
        record["identity"]["tick"], record_trace=cfg["profile"] == "smoke")
    if cfg["profile"] == "smoke":
        scalar_replay(record["config"], record["state"], r, model, record["identity"]["tick"])
        stats["scalar_replayed_futures"] += len(roots)*scenarios
    stats[f"{phase}_transitions"] += r["transitions"]
    stats["forecast_transitions"] += r["transitions"]; stats["forecast_futures"] += len(roots)*scenarios
    stats["forecast_roots"] += len(roots)
    for k, v in r["counts"].items(): stats[k] += v
    path = output/"forecasts"/phase/f"{filename}.npz"
    save_forecast(path, record, roots, r, seed, phase)
    return r, path.relative_to(output).as_posix()


def paired_target(costs, roots, scenarios, scale):
    if not 2 <= scenarios <= costs.shape[1] or len(roots) != 2:
        raise ValueError("two roots and a valid nested scenario count required")
    left, right = sorted(range(2), key=lambda i: CATALOGUE.index(roots[i]))
    delta = (costs[right, :scenarios] - costs[left, :scenarios])/scale
    return (CATALOGUE.index(roots[left]), CATALOGUE.index(roots[right]),
        float(delta.mean()), float(delta.std(ddof=1)/np.sqrt(scenarios)))


def subset(data, rows):
    # Feature state arrays stay shared; only the label table is restricted.
    state_keys = {"raw_pairs", "context", "original_ids"}
    return {k: v if k in state_keys else v[rows] for k, v in data.items()}


def make_dataset(output, cfg, split, records, models, journals, stats):
    result = {}
    for seed in cfg["model_seeds"]:
        selected = [r for r in records if r["identity"]["model_seed"] == seed]
        if not selected:
            raise RuntimeError("no natural opportunities; no resampling allowed")
        data = inputs(feature_batch([r["config"] for r in selected], [r["state"] for r in selected],
            [48-r["identity"]["tick"] for r in selected], 48, CONTEXT), models[seed])
        columns = {k: [] for k in ("pairs", "targets", "mc_se", "conditions", "origins", "ticks", "cohorts", "original_ids")}
        low_targets, low_ses = [], []
        scenarios = cfg[f"{split}_scenarios"]
        for si, record in enumerate(tqdm(selected, desc=f"{split} natural paired labels {seed}", unit="state")):
            roots = record["roots"]
            values, path = forecast(output, cfg, record, roots, models[seed], split, scenarios, stats, f"{seed}_{si}")
            left, right, target, se = paired_target(values["costs"], roots, scenarios, cfg["reward_scale"])
            _, _, low, low_se = paired_target(values["costs"], roots, cfg["low_scenarios"] if split == "train" else scenarios, cfg["reward_scale"])
            identity = record["identity"]; tick = identity["tick"]
            for k, value in dict(pairs=[si, left, right], targets=target, mc_se=se,
                conditions=cfg["conditions"].index(identity["condition"]), origins=0, ticks=tick,
                cohorts=identity["cohort"], original_ids=CATALOGUE.index(record["original"])).items():
                columns[k].append(value)
            low_targets.append(low); low_ses.append(low_se)
            journals["label_states"].add({**identity, "split": split, "state_index": si,
                "state": compact(asdict(record["state"])), "original": compact(record["original"]),
                "roots": compact(roots), "forecast": path, "source_assignment_choices": 2})
            journals["label_pairs"].add({**identity, "split": split, "state_index": si,
                "left": compact(CATALOGUE[left]), "right": compact(CATALOGUE[right]), "original_class": True,
                "target_normalized": target, "paired_mc_se_normalized": se,
                "low_target_normalized": low, "low_paired_mc_se_normalized": low_se,
                "scenarios": scenarios, "low_scenarios": cfg["low_scenarios"] if split == "train" else scenarios,
                "low_scenario_contract": "same full shocks and costs, scenario indices 0..low-1 at every tick",
                "landmark_eligible": tick in cfg["landmarks"], "forecast": path})
        full = {**data, **{k: torch.tensor(v, dtype=torch.float32 if k in ("targets", "mc_se") else torch.long) for k, v in columns.items()}}
        if set(full["conditions"].tolist()) != {0, 1} or not all(torch.isfinite(v).all() for v in full.values()):
            raise RuntimeError("missing condition labels or nonfinite data; no resampling allowed")
        if split == "train":
            low = {**full, "targets": torch.tensor(low_targets), "mc_se": torch.tensor(low_ses)}
            mask = torch.tensor([t in cfg["landmarks"] for t in full["ticks"].tolist()])
            landmark = subset(low, mask)
            if set(landmark["conditions"].tolist()) != {0, 1}:
                raise RuntimeError("landmark training condition absent; no resampling allowed")
            result[seed] = dict(zip(ARMS, (landmark, low, full), strict=True))
        else:
            result[seed] = full
        saved = result[seed] if split == "train" else {"common": full}
        for arm, d in saved.items():
            path = output/"datasets"/f"{split}_{seed}_{arm}.pt"; path.parent.mkdir(exist_ok=True)
            torch.save(d, path); restored = torch.load(path, weights_only=True)
            if any(not torch.equal(v, restored[k]) for k, v in d.items()):
                raise RuntimeError("dataset roundtrip")
    return result
