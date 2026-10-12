"""Screen strict skill competition with unequal committed assignment costs."""

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
from ht_pdm_fjsp import maintenance_skill_contention_screen_v2 as previous
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_opportunity_cost import write_csv, write_json
from ht_pdm_fjsp.maintenance_waiting import ENV_VERSION, WaitingConfig, WaitingEnv, transition

VERSION = "maintenance_skill_contention_screen_v3"
METRIC_VERSION = "same_subset_committed_downtime_cost_contention_v1"
ROOT = Path(__file__).resolve().parents[2]
POLICIES = base.POLICIES
MASKS = {k: previous.MASKS[k] for k in ("overlap3", "overlap3_reverse", "balanced2", "bottleneck2")}
PATTERNS = {f"fast{fast}_slow{''.join(map(str, slow))}": (fast, (fast + 2, *slow))
    for fast, durations in ((1, (2, 3, 5)), (2, (3, 4, 6)))
    for slow in itertools.permutations(durations)}
CONTROLS = ("overlap3__uniform_control", "overlap3__equal_control", "overlap3__tradeoff_control",
    "overlap3__cancellation_control", "partitioned_control__heterogeneous_control")
LIMITS = dict(skill_contention_rate=.10, cost_sensitive_contention_rate=.05,
    selected_cost_sensitive_contention_rate=.03, availability=.25, terminal_pending=1.)
SEED_PATH = ROOT / "configs" / f"{VERSION}_seed_registry.json"
PLAN_PATH = ROOT / "docs" / f"{VERSION}_plan.md"
decode_config, choose, action_space = base.decode_config, base.choose, base.action_space
aggregate, sha, git_metadata = base.aggregate, base.sha, base.git_metadata


def settings(profile):
    if profile not in ("smoke", "full"): raise ValueError("unknown profile")
    raw = json.loads(SEED_PATH.read_text())["panels"][profile]
    return dict(profile=profile, horizon=48, tail_ticks=12, policies=list(POLICIES),
        screening_seeds=list(range(raw["screening_start"], raw["screening_start"] + raw["screening_count"])),
        confirmation_seeds=list(range(raw["confirmation_start"], raw["confirmation_start"] + raw["confirmation_count"])),
        analysis_seed=raw["analysis_seed"], bootstrap_samples=20000 if profile == "full" else 200,
        confidence=.99, audit_seeds=2 if profile == "full" else 4, limits=LIMITS.copy(),
        max_seconds=7200 if profile == "full" else 600, training="NOT_APPLICABLE", train_seeds=[],
        tail_policy="deadline_matching", metric_version=METRIC_VERSION)


def physical_config(topology, pattern, cfg):
    if pattern in ("equal_control", "tradeoff_control"):
        return previous.physical_config(topology, pattern, cfg)
    if pattern == "uniform_control":
        if topology != "overlap3": raise ValueError("uniform control is locked on overlap3")
        return previous.physical_config(topology, "common_1_3", cfg)
    if pattern == "cancellation_control":
        if topology != "overlap3": raise ValueError("cancellation control is locked on overlap3")
        ds = ((1, 0), (2, 4), (1, 3), (2, 4))
    else:
        if pattern == "heterogeneous_control":
            if topology != "partitioned_control": raise ValueError("partitioned control is locked")
            fast, slow, mask = 1, (3, 3, 5, 2), base.MASKS[topology]
        else:
            if pattern not in PATTERNS: raise ValueError("unknown skill pattern")
            fast, slow = PATTERNS[pattern]; mask = MASKS[topology]
        ds = tuple(tuple((fast if j == 0 else slow[m]) if valid else 0 for j, valid in enumerate(row))
                   for m, row in enumerate(mask))
    c = WaitingConfig(4, 2, cfg["horizon"] + cfg["tail_ticks"], 4, .45,
        ds, ((.75, .75),) * 4, waiting_price=12.)
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
    if any(a & b for a, b in itertools.combinations(raw_groups, 2)): raise ValueError("seed panel overlap")
    def integers(v):
        if isinstance(v, dict): return set().union(set(), *(integers(x) for x in v.values()))
        if isinstance(v, list): return set().union(set(), *(integers(x) for x in v))
        return {v} if isinstance(v, int) and not isinstance(v, bool) else set()
    historical, hashes = set(), {}
    for p in sorted((ROOT / "configs").glob("*seed_registry.json")):
        if p == SEED_PATH: continue
        historical |= integers(json.loads(p.read_text())); hashes[p.name] = sha(p)
    old_derived = set()
    for old_module in (base, previous):
        for profile in ("full", "smoke"):
            old = old_module.settings(profile)
            old_derived.update(role_seed(seed, f"{old_module.VERSION}:physical")
                for phase in ("screening", "confirmation") for seed in old[f"{phase}_seeds"])
            old_derived.add(role_seed(old["analysis_seed"], f"{old_module.VERSION}:bootstrap"))
    own = set.union(*raw_groups)
    if own & (historical | old_derived) or len(set(roles)) != len(roles) or set(roles) & (own | historical | old_derived):
        raise ValueError("raw or derived RNG overlap")
    if cfg != settings(cfg["profile"]): raise ValueError("settings must match the locked profile")
    return dict(status="PASS", raw_unique=len(own), derived_unique=len(roles), prior_derived_checked=len(old_derived),
        historical_registry_hashes=hashes, intentional_reuse="same episode seed across candidates/controllers")


def committed_cost_spans(c, actions):
    """Compare obligations of new jobs only, holding the same machine subset."""
    bounds = {}
    for action in actions:
        subset = tuple(m for m, _ in action)
        value = c.unavailable_cost * sum(c.service_time[m][j] for m, j in action)
        low, high = bounds.get(subset, (value, value))
        bounds[subset] = min(low, value), max(high, value)
    return {subset: high - low for subset, (low, high) in bounds.items()}


def state_metrics(c, state, selected, space=None):
    space = space or action_space(c, state)
    metrics = previous.state_metrics(c, state, selected, space)
    maximum = metrics["feasible_matching_cardinality"]
    spans = committed_cost_spans(c, space[2])
    possible = max(v for subset, v in spans.items() if len(subset) == maximum)
    selected_subset = tuple(m for m, _ in selected)
    selected_span = spans[selected_subset]
    threshold = max(c.unavailable_cost, 1e-12)
    strict = metrics["skill_contention_rate"]
    return dict(**metrics, cost_sensitive_contention_rate=int(strict and possible >= threshold),
        selected_cost_sensitive_contention_rate=int(strict and len(selected_subset) == maximum and selected_span >= threshold),
        committed_cost_span_max=possible, selected_committed_cost_span=selected_span)


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
    ranking.sort(key=lambda r: (-int(r["eligible"]), -r["skill_contention_rate"], -r["cost_sensitive_contention_rate"],
        -r["selected_cost_sensitive_contention_rate"], r["candidate"]))
    qualified = [r for r in ranking if r["eligible"]]
    winner = qualified[0]["candidate"] if qualified else None
    runner_up = qualified[1]["candidate"] if len(qualified) > 1 else None
    selected = list(dict.fromkeys([c for c in (winner, runner_up, *CONTROLS) if c]))
    return ranking, dict(status="SELECTED" if winner else "NO_ELIGIBLE_SETTING", winner=winner,
        runner_up=runner_up, confirmation_candidates=selected, controls=list(CONTROLS), limits=cfg["limits"],
        rank_rule="min-rule strict contention, cost-sensitive contention, selected consequence, then ID; no realized cost",
        selection_phase="screening_only", confirmation_outcomes_opened=False, metric_version=METRIC_VERSION)


def control_audit(rows):
    for row in rows:
        if row["candidate"] in CONTROLS:
            if row["cost_sensitive_contention_rate"] != 0 or row["selected_cost_sensitive_contention_rate"] != 0:
                raise RuntimeError("negative control registered cost-sensitive contention")
            if row["candidate"] not in ("overlap3__uniform_control", "overlap3__cancellation_control"):
                if row["skill_contention_rate"] != 0: raise RuntimeError("negative control registered strict preference competition")
        if not 0 <= row["selected_cost_sensitive_contention_rate"] <= row["cost_sensitive_contention_rate"] <= row["contested_choice_rate"] <= row["skill_contention_rate"] <= 1:
            raise RuntimeError("consequence rates are not nested within strict competition")
    return dict(status="PASS", negative_control_rows=sum(r["candidate"] in CONTROLS for r in rows))


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
        for name in ("maintenance_skill_contention_screen_v3.py", "maintenance_skill_contention_screen_v2.py", "maintenance_skill_contention_screen.py", "maintenance_dispatch.py", "maintenance_waiting.py",
                "maintenance_heterogeneous_model.py", "maintenance_opportunity_cost.py"):
            shutil.copy2(Path(__file__).parent / name, snapshot / name)
        for module in (base, previous):
            shutil.copy2(module.SEED_PATH, snapshot / module.SEED_PATH.name)
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
            interpretation="registered same-subset committed downtime consequences; not proof of total cost savings or RL benefit"))
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
