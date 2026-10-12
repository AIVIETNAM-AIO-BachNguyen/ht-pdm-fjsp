"""Fixed skill screen with tie-invariant, capacity-controlled physical preferences."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_skill_contention_screen as base
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_opportunity_cost import write_csv, write_json
from ht_pdm_fjsp.maintenance_waiting import ENV_VERSION, WaitingConfig, WaitingEnv, transition

VERSION = "maintenance_skill_contention_screen_v2"
METRIC_VERSION = "pareto_matching_deficit_and_contested_same_subset_v1"
ROOT = Path(__file__).resolve().parents[2]
POLICIES = base.POLICIES
MASKS = {k: v for k, v in base.MASKS.items() if k != "partitioned_control"}
MASKS["overlap3_reverse"] = ((0, 1), (1, 1), (1, 1), (1, 1))
FAVORITES = dict(common=(0, 0, 0, 0), cluster=(0, 0, 0, 1), paired=(0, 1, 0, 1))
DURATIONS = ((1, 2), (1, 3), (2, 3), (2, 4))
PATTERNS = tuple(f"{name}_{fast}_{slow}" for name in FAVORITES for fast, slow in DURATIONS)
CONTROLS = ("overlap3__equal_control", "overlap3__tradeoff_control", "partitioned_control__common_2_4")
LIMITS = dict(skill_contention_rate=.10, contested_choice_rate=.05, availability=.25, terminal_pending=1.)
SEED_PATH = ROOT / "configs" / f"{VERSION}_seed_registry.json"
PLAN_PATH = ROOT / "docs" / f"{VERSION}_plan.md"
decode_config, choose, action_space = base.decode_config, base.choose, base.action_space
aggregate, sha, git_metadata = base.aggregate, base.sha, base.git_metadata


def settings(profile):
    if profile not in ("smoke", "full"):
        raise ValueError("unknown profile")
    raw = json.loads(SEED_PATH.read_text())["panels"][profile]
    return dict(profile=profile, horizon=48, tail_ticks=12, policies=list(POLICIES),
        screening_seeds=list(range(raw["screening_start"], raw["screening_start"] + raw["screening_count"])),
        confirmation_seeds=list(range(raw["confirmation_start"], raw["confirmation_start"] + raw["confirmation_count"])),
        analysis_seed=raw["analysis_seed"], bootstrap_samples=20000 if profile == "full" else 200,
        confidence=.9875, audit_seeds=2 if profile == "full" else 4, limits=LIMITS.copy(),
        max_seconds=7200 if profile == "full" else 600, training="NOT_APPLICABLE", train_seeds=[],
        tail_policy="deadline_matching", metric_version=METRIC_VERSION)


def physical_config(topology, pattern, cfg):
    if topology == "partitioned_control":
        if pattern != "common_2_4": raise ValueError("unknown partitioned control")
        mask = base.MASKS[topology]
    else:
        mask = MASKS[topology]
    if pattern in ("equal_control", "tradeoff_control"):
        if topology != "overlap3": raise ValueError("controls are locked on overlap3")
        favorites, fast, slow = FAVORITES["paired"], 2, 4
    else:
        if pattern not in PATTERNS: raise ValueError("unknown skill pattern")
        name, fast, slow = pattern.split("_")
        favorites, fast, slow = FAVORITES[name], int(fast), int(slow)
    duration, effect = [], []
    for m, allowed in enumerate(mask):
        ds, es = [], []
        for j, valid in enumerate(allowed):
            preferred = j == favorites[m]
            d = 3 if pattern == "equal_control" else fast if preferred else slow
            e = (.5 if preferred else 1.) if pattern == "tradeoff_control" else .75
            ds.append(d if valid else 0); es.append(e)
        duration.append(tuple(ds)); effect.append(tuple(es))
    c = WaitingConfig(4, 2, cfg["horizon"] + cfg["tail_ticks"], 4, .45,
        tuple(duration), tuple(effect), waiting_price=12.)
    c.validate()
    return c


def candidates(cfg):
    grid = list(itertools.product(MASKS, PATTERNS)) + [tuple(name.split("__")) for name in CONTROLS]
    specs = []
    for topology, pattern in grid:
        c = physical_config(topology, pattern, cfg)
        specs.append(dict(candidate=f"{topology}__{pattern}", topology=topology, pattern=pattern,
            selectable=f"{topology}__{pattern}" not in CONTROLS,
            eligible_edges=sum(d > 0 for row in c.service_time for d in row),
            shared_machines=sum(sum(d > 0 for d in row) > 1 for row in c.service_time), config=asdict(c)))
    return specs


def seed_audit(cfg):
    raw_groups, roles = [], []
    for profile in ("full", "smoke"):
        c = settings(profile)
        raw_groups += [set(c["screening_seeds"]), set(c["confirmation_seeds"]), {c["analysis_seed"]}]
        roles += [role_seed(seed, f"{VERSION}:physical") for phase in ("screening", "confirmation")
                  for seed in c[f"{phase}_seeds"]]
        roles.append(role_seed(c["analysis_seed"], f"{VERSION}:bootstrap"))
    if any(a & b for a, b in itertools.combinations(raw_groups, 2)):
        raise ValueError("seed panel overlap")
    def integers(v):
        if isinstance(v, dict): return set().union(set(), *(integers(x) for x in v.values()))
        if isinstance(v, list): return set().union(set(), *(integers(x) for x in v))
        return {v} if isinstance(v, int) and not isinstance(v, bool) else set()
    historical, hashes = set(), {}
    for p in sorted((ROOT / "configs").glob("*seed_registry.json")):
        if p == SEED_PATH: continue
        historical |= integers(json.loads(p.read_text()))
        hashes[p.name] = sha(p)
    old_derived = set()
    for profile in ("full", "smoke"):
        old = base.settings(profile)
        old_derived.update(role_seed(seed, f"{base.VERSION}:physical") for phase in ("screening", "confirmation")
                           for seed in old[f"{phase}_seeds"])
        old_derived.add(role_seed(old["analysis_seed"], f"{base.VERSION}:bootstrap"))
    own = set.union(*raw_groups)
    if own & (historical | old_derived) or len(set(roles)) != len(roles) or set(roles) & (own | historical | old_derived):
        raise ValueError("raw or derived RNG overlap")
    if cfg != settings(cfg["profile"]): raise ValueError("settings must match the locked profile")
    return dict(status="PASS", raw_unique=len(own), derived_unique=len(roles), v1_derived_checked=len(old_derived),
        historical_registry_hashes=hashes, intentional_reuse="same episode seed across candidates/controllers")


def nondominated_workers(c, edges):
    """Keep equal edges and incomparable speed/restoration tradeoffs in full."""
    preferred = {}
    for m in range(c.machines):
        valid = [j for j in range(c.technicians) if edges[m][j]]
        def dominates(j, k):
            dj, dk = c.service_time[m][j], c.service_time[m][k]
            ej, ek = c.restoration[m][j], c.restoration[m][k]
            return dj <= dk and ej >= ek and (dj < dk or ej > ek)
        preferred[m] = {j for j in valid if not any(dominates(k, j) for k in valid)}
    return preferred


def state_metrics(c, state, selected, space=None):
    space = space or action_space(c, state)
    demand, edges, actions = space
    preferred = nondominated_workers(c, edges)
    best_actions = tuple(a for a in actions if all(j in preferred[m] for m, j in a))
    feasible = max(map(len, actions)); best = max(map(len, best_actions))
    choices = base.physical_choices(c, actions)
    best_subsets = {tuple(m for m, _ in a) for a in best_actions if len(a) == feasible}
    contested = any(len(subset) == feasible and subset not in best_subsets for subset in choices)
    legacy = base.state_metrics(c, state, selected, space)
    legacy_conflict = legacy.pop("conflict_rate")
    return dict(**legacy, legacy_proposal_conflict_rate=legacy_conflict,
        skill_contention_rate=int(best < feasible), contested_choice_rate=int(best < feasible and contested),
        feasible_matching_cardinality=feasible, preferred_matching_cardinality=best,
        preferred_deficit=feasible - best,
        legacy_tie_rate=int(any(len(js := [j for j in range(c.technicians) if edges[m][j]]) > 1
            and len({base.quality(c, m, j) for j in js}) < len(js) for m in demand)))


def one_episode(cfg, spec, policy, seed, phase, journal=None):
    c = decode_config(spec); env = WaitingEnv(c)
    physical_seed = role_seed(seed, f"{VERSION}:physical"); env.reset(physical_seed)
    counts, interior, cutoff = defaultdict(float), defaultdict(float), None
    for tick in range(c.horizon):
        state = env.state; space = action_space(c, state)
        action = choose(c, state, policy if tick < cfg["horizon"] else "deadline_matching", space)
        validate_matching(c, state, action)
        metrics = state_metrics(c, state, action, space) if tick < cfg["horizon"] else None
        if metrics is not None:
            for key, value in metrics.items(): counts[key] += value
        _, _, _, info = env.step(action)
        if tick < cfg["horizon"]:
            for key, value in info.items(): interior[key] += value
        if tick + 1 == cfg["horizon"]:
            cutoff = dict(terminal_pending=base.pending(c, env.state), terminal_failed=sum(env.state.failed))
        if journal is not None:
            journal.write(json.dumps(dict(phase=phase, candidate=spec["candidate"], policy=policy,
                seed=seed, physical_seed=physical_seed, tick=tick, state=asdict(state), action=action, metrics=metrics,
                events=env.events[tick], following=asdict(env.state), info=info), allow_nan=False) + "\n")
    if cutoff is None: raise RuntimeError("missing policy cutoff")
    row = dict(phase=phase, candidate=spec["candidate"], policy=policy, seed=seed, physical_seed=physical_seed,
        **{k: v / cfg["horizon"] for k, v in counts.items()},
        availability=1 - interior["unavailable_ticks"] / (c.machines * cfg["horizon"]),
        technician_utilization=interior["service_ticks"] / (c.technicians * cfg["horizon"]),
        **cutoff, tail_pending=base.pending(c, env.state), tail_failed=sum(env.state.failed),
        policy_cost=interior["objective"], tail_cost=env.metrics["objective"] - interior["objective"],
        extended_cost=env.metrics["objective"], policy_failures=interior["failures"],
        policy_overdue_ticks=interior["overdue_waiting_ticks"], **{f"extended_{key}": env.metrics[key] for key in
        ("maintenance_cost", "unavailability_cost", "failure_cost", "waiting_cost", "invalid_assignments")})
    if not np.isclose(sum(row[f"extended_{key}"] for key in
        ("maintenance_cost", "unavailability_cost", "failure_cost", "waiting_cost")), row["extended_cost"], atol=1e-8, rtol=0):
        raise RuntimeError("episode cost reconciliation")
    return row


def rank_screen(cfg, specs, rows):
    means = aggregate(rows); ranking = []
    for spec in specs:
        group = [means[spec["candidate"], p] for p in POLICIES]
        metrics = {k: (max if k == "terminal_pending" else min)(g[k] for g in group) for k in cfg["limits"]}
        failures = [k for k, limit in cfg["limits"].items()
            if (metrics[k] > limit if k == "terminal_pending" else metrics[k] < limit)]
        ranking.append(dict(candidate=spec["candidate"], topology=spec["topology"], pattern=spec["pattern"],
            selectable=spec["selectable"], eligible=spec["selectable"] and not failures,
            rejection=";".join(failures) if spec["selectable"] else "control", **metrics,
            mean_cost=float(np.mean([g["extended_cost"] for g in group]))))
    ranking.sort(key=lambda r: (-int(r["eligible"]), -r["skill_contention_rate"], -r["contested_choice_rate"], r["candidate"]))
    qualified = [r for r in ranking if r["eligible"]]
    winner = qualified[0]["candidate"] if qualified else None
    runner_up = qualified[1]["candidate"] if len(qualified) > 1 else None
    selected = list(dict.fromkeys([c for c in (winner, runner_up, *CONTROLS) if c]))
    return ranking, dict(status="SELECTED" if winner else "NO_ELIGIBLE_SETTING", winner=winner,
        runner_up=runner_up, confirmation_candidates=selected, controls=list(CONTROLS), limits=cfg["limits"],
        rank_rule="min-policy skill contention, then min-policy contested choice, then ID; no cost",
        selection_phase="screening_only", confirmation_outcomes_opened=False, metric_version=METRIC_VERSION)


def confirm(cfg, selection, rows):
    if not selection["winner"]:
        return dict(status="NOT_APPLICABLE", reason="NO_ELIGIBLE_SETTING", winner=None)
    seeds = cfg["confirmation_seeds"]; winner = selection["winner"]; metrics = tuple(cfg["limits"])
    lookup = {(r["candidate"], r["policy"], r["seed"]): r for r in rows}
    values = np.asarray([[[lookup[winner, p, seed][k] for k in metrics] for p in POLICIES] for seed in seeds])
    rng = np.random.default_rng(role_seed(cfg["analysis_seed"], f"{VERSION}:bootstrap"))
    draws = rng.integers(0, len(seeds), size=(cfg["bootstrap_samples"], len(seeds)))
    boot = values[draws].mean(1); point = values.mean(0); alpha = (1 - cfg["confidence"]) / 2
    result = {}
    for j, key in enumerate(metrics):
        upper = key == "terminal_pending"
        distribution = boot[:, :, j].max(1) if upper else boot[:, :, j].min(1)
        estimate = point[:, j].max() if upper else point[:, j].min()
        lo, hi = np.quantile(distribution, [alpha, 1 - alpha])
        passed = hi <= cfg["limits"][key] if upper else lo >= cfg["limits"][key]
        result[key] = dict(estimate=float(estimate), ci=[float(lo), float(hi)], pass_gate=bool(passed))
    return dict(status="CONFIRMED" if all(r["pass_gate"] for r in result.values()) else "NOT_CONFIRMED",
        winner=winner, metrics=result, confidence=cfg["confidence"], metric_version=METRIC_VERSION,
        inference_scope="new seed trajectories in selected fixed setting under two observation policies")


def control_audit(rows):
    for row in rows:
        if row["candidate"] in CONTROLS:
            if row["skill_contention_rate"] != 0 or row["contested_choice_rate"] != 0:
                raise RuntimeError("negative control registered strict competition")
            if row["candidate"] != "overlap3__tradeoff_control" and row["sensitive_choice_rate"] != 0:
                raise RuntimeError("label-only or partitioned control registered physical choices")
        if not 0 <= row["contested_choice_rate"] <= row["skill_contention_rate"] <= 1:
            raise RuntimeError("joint metric is not a subset of primary")
    return dict(status="PASS", negative_control_rows=sum(r["candidate"] in CONTROLS for r in rows))


def replay_audit(path, specs, cfg):
    configs = {r["candidate"]: decode_config(r) for r in specs}; envs, count = {}, 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line); c = configs[row["candidate"]]
            key = row["phase"], row["candidate"], row["policy"], row["seed"]
            if row["physical_seed"] != role_seed(row["seed"], f"{VERSION}:physical"):
                raise RuntimeError("audit seed mismatch")
            if row["tick"] == 0:
                if key in envs: raise RuntimeError("duplicate audited trajectory")
                env = WaitingEnv(c); env.reset(row["physical_seed"]); envs[key] = env
            env = envs[key]; state = base.unpack_state(row["state"])
            if row["tick"] != env.time or state != env.state or tuple(row["events"]) != env.events[env.time]:
                raise RuntimeError("audit state/event continuity")
            action = tuple(map(tuple, row["action"]))
            policy = row["policy"] if row["tick"] < cfg["horizon"] else "deadline_matching"
            if choose(c, state, policy) != action: raise RuntimeError("audit policy mismatch")
            expected_metrics = state_metrics(c, state, action) if row["tick"] < cfg["horizon"] else None
            if row["metrics"] != expected_metrics: raise RuntimeError("audit metric mismatch")
            following, cost, info = transition(c, state, action, row["events"])
            if following != base.unpack_state(row["following"]) or info != row["info"] or cost != info["objective"]:
                raise RuntimeError("independent scalar transition mismatch")
            env.step(action); count += 1
    if any(env.time != env.config.horizon for env in envs.values()): raise RuntimeError("incomplete audited trajectory")
    return dict(status="PASS", replayed_transitions=count, audited_episodes=len(envs), metrics_replayed=True)


def run(profile, output_dir, device="cpu"):
    if device != "cpu": raise ValueError("this non-learning screen is CPU only")
    cfg = settings(profile); specs = candidates(cfg)
    seeds = seed_audit(cfg)
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("refusing to overwrite a previous run")
    output.mkdir(parents=True, exist_ok=True)
    git = git_metadata()
    if profile == "full" and git["dirty"]:
        raise RuntimeError("full run requires a clean committed checkout")
    started = time.monotonic()
    manifest = dict(protocol=VERSION, metric_version=METRIC_VERSION, environment_version=ENV_VERSION, status="RUNNING", profile=profile,
        git=git, started_utc=datetime.now(timezone.utc).isoformat(), runtime=dict(python=platform.python_version(),
        platform=platform.platform(), numpy=np.__version__, device=device),
        confirmation_outcomes_opened=False, full_confirmation_outcomes_opened=False)
    write_json(output / "manifest.json", manifest)
    try:
        write_json(output / "resolved_config.json", dict(settings=cfg, candidates=specs))
        shutil.copy2(SEED_PATH, output / "seed_registry.json")
        shutil.copy2(PLAN_PATH, output / "protocol.md")
        snapshot = output / "source_snapshot"; snapshot.mkdir()
        for name in ("maintenance_skill_contention_screen_v2.py", "maintenance_skill_contention_screen.py", "maintenance_dispatch.py", "maintenance_waiting.py",
                "maintenance_heterogeneous_model.py", "maintenance_opportunity_cost.py"):
            shutil.copy2(Path(__file__).parent / name, snapshot / name)
        def check_timeout():
            if time.monotonic() - started > cfg["max_seconds"]:
                raise TimeoutError("locked experiment timeout")
        def evaluate(phase, selected, journal):
            rows = []
            panel = cfg[f"{phase}_seeds"]
            jobs = list(itertools.product(selected, POLICIES, panel))
            for spec, policy, seed in tqdm(jobs, desc=f"{phase} episodes", unit="episode"):
                check_timeout()
                log = journal if seed in panel[:cfg["audit_seeds"]] else None
                rows.append(one_episode(cfg, spec, policy, seed, phase, log))
            write_csv(output / f"{phase}_episodes.csv", rows)
            return rows
        with (output / "decision_audit.jsonl").open("w") as journal:
            screening = evaluate("screening", specs, journal)
            ranking, selection = rank_screen(cfg, specs, screening)
            write_csv(output / "screening_ranking.csv", ranking)
            write_json(output / "selection.json", selection)
            selection_hash = sha(output / "selection.json")
            selected = [r for r in specs if r["candidate"] in selection["confirmation_candidates"]]
            manifest.update(selection_sha256=selection_hash, phase="CONFIRMATION",
                confirmation_outcomes_opened=bool(selected),
                full_confirmation_outcomes_opened=profile == "full" and bool(selected))
            write_json(output / "manifest.json", manifest)
            confirmation = evaluate("confirmation", selected, journal)
        if sha(output / "selection.json") != selection_hash: raise RuntimeError("selection changed after confirmation")
        expected_screen = len(specs) * len(POLICIES) * len(cfg["screening_seeds"])
        expected_confirm = len(selected) * len(POLICIES) * len(cfg["confirmation_seeds"])
        for rows, phase, chosen in ((screening, "screening", specs), (confirmation, "confirmation", selected)):
            actual = [(r["candidate"], r["policy"], r["seed"]) for r in rows]
            expected = set(itertools.product([r["candidate"] for r in chosen], POLICIES, cfg[f"{phase}_seeds"]))
            if len(actual) != len(expected) or set(actual) != expected: raise RuntimeError("incomplete episode grid")
        audit = replay_audit(output / "decision_audit.jsonl", specs, cfg)
        invalid = sum(r["extended_invalid_assignments"] for r in screening + confirmation)
        if invalid: raise RuntimeError("invalid executed assignments")
        expected_logged = sum(len(chosen) * len(POLICIES) * min(cfg["audit_seeds"], len(cfg[f"{phase}_seeds"]))
            * (cfg["horizon"] + cfg["tail_ticks"]) for phase, chosen in (("screening", specs), ("confirmation", selected)))
        if audit["replayed_transitions"] != expected_logged: raise RuntimeError("audit grid mismatch")
        audit.update(control_audit=control_audit(screening + confirmation), seed_audit=seeds, invalid_assignments=invalid, selection_locked=True,
            screening_episodes=len(screening), confirmation_episodes=len(confirmation),
            expected_screening_episodes=expected_screen, expected_confirmation_episodes=expected_confirm,
            physical_transitions=(len(screening) + len(confirmation)) * (cfg["horizon"] + cfg["tail_ticks"]))
        write_json(output / "audit.json", audit)
        confirmation_result = confirm(cfg, selection, confirmation)
        means = aggregate(confirmation)
        write_json(output / "summary.json", dict(screening=ranking, selection=selection,
            confirmation=confirmation_result, confirmation_means=[dict(candidate=k[0], policy=k[1], **v) for k, v in means.items()],
            interpretation="strict physical dominance contention is distinct from capacity; grid screening is exploratory; no global optimum or RL gain claim"))
        check_timeout()
        files = {str(p.relative_to(output)): dict(sha256=sha(p), bytes=p.stat().st_size)
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        manifest.update(status="COMPLETED", phase="COMPLETED", audit="PASS", elapsed_seconds=time.monotonic() - started,
            expected_counts=dict(screening=expected_screen, confirmation=expected_confirm),
            actual_counts=dict(screening=len(screening), confirmation=len(confirmation)),
            confirmation_status=confirmation_result["status"], files=files,
            artifact_bytes=sum(v["bytes"] for v in files.values()))
        write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="FAILED", elapsed_seconds=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.profile, args.output_dir, args.device)
    print(json.dumps({k: result[k] for k in ("status", "audit", "elapsed_seconds", "actual_counts", "confirmation_status", "artifact_bytes")}))


if __name__ == "__main__": main()
