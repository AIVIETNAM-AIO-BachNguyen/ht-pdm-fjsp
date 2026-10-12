"""Prospective fixed-setting skill screen; no learning or future-rollout labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import DispatchState, eligible_edges, role_seed, validate_matching
from ht_pdm_fjsp.maintenance_heterogeneous_model import catalogue
from ht_pdm_fjsp.maintenance_opportunity_cost import write_csv, write_json
from ht_pdm_fjsp.maintenance_waiting import ENV_VERSION, WaitingConfig, WaitingEnv, transition

VERSION = "maintenance_skill_contention_screen_v1"
ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("deadline_matching", "priority_greedy")
MASKS = {
    "overlap3": ((1, 0), (1, 1), (1, 1), (1, 1)),
    "balanced2": ((1, 0), (0, 1), (1, 1), (1, 1)),
    "bottleneck2": ((1, 0), (1, 0), (1, 1), (1, 1)),
    "balanced1": ((1, 0), (1, 0), (0, 1), (1, 1)),
    "bottleneck1": ((1, 0), (1, 0), (1, 0), (1, 1)),
    "partitioned_control": ((1, 0), (1, 0), (0, 1), (0, 1)),
}
PATTERNS = ("paired_2_4", "common_2_4", "paired_2_8", "tradeoff_2_4")
LIMITS = dict(conflict_rate=.10, sensitive_choice_rate=.05, availability=.25, terminal_pending=1.)
SEED_PATH = ROOT / "configs" / f"{VERSION}_seed_registry.json"
PLAN_PATH = ROOT / "docs" / f"{VERSION}_plan.md"


def settings(profile):
    if profile not in ("smoke", "full"):
        raise ValueError("unknown profile")
    raw = json.loads(SEED_PATH.read_text())["panels"][profile]
    return dict(profile=profile, horizon=48, tail_ticks=12, policies=list(POLICIES),
        screening_seeds=list(range(raw["screening_start"], raw["screening_start"] + raw["screening_count"])),
        confirmation_seeds=list(range(raw["confirmation_start"], raw["confirmation_start"] + raw["confirmation_count"])),
        analysis_seed=raw["analysis_seed"], bootstrap_samples=20000 if profile == "full" else 200,
        confidence=.9875, audit_seeds=2 if profile == "full" else 4,
        limits=LIMITS.copy(), max_seconds=7200 if profile == "full" else 600,
        training="NOT_APPLICABLE", train_seeds=[], tail_policy="deadline_matching")


def physical_config(topology, pattern, cfg):
    if pattern not in (*PATTERNS, "equal_control"):
        raise ValueError("unknown skill pattern")
    if pattern == "equal_control" and topology != "overlap3":
        raise ValueError("equal control is defined only on overlap3")
    duration, effect = [], []
    for m, allowed in enumerate(MASKS[topology]):
        favorite = 0 if pattern == "common_2_4" else m % 2
        ds, es = [], []
        for j, valid in enumerate(allowed):
            fast = j == favorite
            d = 3 if pattern == "equal_control" else (2 if fast else (8 if pattern == "paired_2_8" else 4))
            e = (.5 if fast else 1.) if pattern == "tradeoff_2_4" else .75
            ds.append(d if valid else 0)
            es.append(e)
        duration.append(tuple(ds)); effect.append(tuple(es))
    c = WaitingConfig(4, 2, cfg["horizon"] + cfg["tail_ticks"], 4, .45,
        tuple(duration), tuple(effect), waiting_price=12.)
    c.validate()
    return c


def candidates(cfg):
    rows = []
    for topology, pattern in (*itertools.product(MASKS, PATTERNS), ("overlap3", "equal_control")):
        c = physical_config(topology, pattern, cfg)
        rows.append(dict(candidate=f"{topology}__{pattern}", topology=topology, pattern=pattern,
            selectable=topology != "partitioned_control" and pattern != "equal_control",
            eligible_edges=sum(d > 0 for row in c.service_time for d in row),
            shared_machines=sum(sum(d > 0 for d in row) > 1 for row in c.service_time), config=asdict(c)))
    return rows


def decode_config(row):
    data = dict(row["config"])
    for key in ("service_time", "restoration"):
        data[key] = tuple(map(tuple, data[key]))
    return WaitingConfig(**data)


def seed_audit(cfg):
    raw_groups, roles = [], []
    for profile in ("full", "smoke"):
        c = settings(profile)
        raw_groups += [set(c["screening_seeds"]), set(c["confirmation_seeds"]), {c["analysis_seed"]}]
        roles += [role_seed(s, f"{VERSION}:physical") for split in ("screening", "confirmation") for s in c[f"{split}_seeds"]]
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
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    own = set.union(*raw_groups)
    if own & historical or len(set(roles)) != len(roles) or set(roles) & (own | historical):
        raise ValueError("raw or derived RNG overlap")
    if cfg != settings(cfg["profile"]):
        raise ValueError("settings must match the locked profile")
    return dict(status="PASS", raw_unique=len(own), derived_unique=len(roles),
        historical_registry_hashes=hashes, intentional_reuse="same episode seed across candidates/controllers")


def action_space(c, state):
    edges = eligible_edges(c, state)
    demand = tuple(m for m in range(c.machines)
        if m not in state.assigned and (state.failed[m] or state.ages[m] >= c.failure_age - 1))
    requested = set(demand)
    actions = tuple(a for a in catalogue(c.machines, c.technicians)
        if all(m in requested and edges[m][j] for m, j in a))
    return demand, edges, actions


def urgency(c, state, m):
    hazard = (state.ages[m] + 1 >= c.failure_age and not state.failed[m]) * c.failure_probability
    return 100 * state.failed[m] + 10 * hazard + state.ages[m] / 8 + 20 * max(0, state.pending_wait[m] - 3)


def quality(c, m, j):
    return c.restoration[m][j] / c.service_time[m][j]


def choose(c, state, policy, space=None):
    demand, edges, actions = space or action_space(c, state)
    if policy == "deadline_matching":
        return max(actions, key=lambda a: sum(urgency(c, state, m) + quality(c, m, j) for m, j in a))
    if policy != "priority_greedy":
        raise ValueError(policy)
    free, pairs = set(j for j in range(c.technicians) if state.remaining[j] == 0), []
    for m in sorted(demand, key=lambda m: (-urgency(c, state, m), m)):
        available = [j for j in sorted(free) if edges[m][j]]
        if available:
            j = max(available, key=lambda j: quality(c, m, j))
            pairs.append((m, j)); free.remove(j)
    return tuple(sorted(pairs))


def physical_choices(c, actions):
    groups = defaultdict(set)
    for action in actions:
        subset = tuple(m for m, _ in action)
        groups[subset].add(tuple((m, c.service_time[m][j], c.restoration[m][j]) for m, j in action))
    return {subset for subset, physical in groups.items() if len(physical) > 1}


def state_metrics(c, state, selected, space=None):
    demand, edges, actions = space or action_space(c, state)
    free = sum(r == 0 for r in state.remaining)
    largest = max(map(len, actions))
    proposals = []
    for m in demand:
        js = [j for j in range(c.technicians) if edges[m][j]]
        if js:
            proposals.append(max(js, key=lambda j: quality(c, m, j)))
    excess = sum(max(0, n - 1) for n in Counter(proposals).values())
    choices = physical_choices(c, actions)
    return dict(conflict_rate=int(excess > 0), collision_excess=excess,
        sensitive_choice_rate=int(any(len(subset) == largest for subset in choices)),
        selected_sensitive_rate=int(tuple(m for m, _ in selected) in choices),
        capacity_pressure_rate=int(len(demand) > free),
        skill_deficit_rate=int(largest < min(len(demand), free)), demand_count=len(demand))


def pending(c, state):
    return sum(state.failed[m] and m not in state.assigned for m in range(c.machines))


def one_episode(cfg, spec, policy, seed, phase, journal=None):
    c = decode_config(spec)
    env = WaitingEnv(c)
    physical_seed = role_seed(seed, f"{VERSION}:physical")
    env.reset(physical_seed)
    counts, interior, cutoff = defaultdict(float), defaultdict(float), None
    for tick in range(c.horizon):
        state = env.state
        space = action_space(c, state)
        action = choose(c, state, policy if tick < cfg["horizon"] else "deadline_matching", space)
        validate_matching(c, state, action)
        if tick < cfg["horizon"]:
            for key, value in state_metrics(c, state, action, space).items(): counts[key] += value
        _, _, _, info = env.step(action)
        if tick < cfg["horizon"]:
            for key, value in info.items(): interior[key] += value
        if tick + 1 == cfg["horizon"]:
            cutoff = dict(terminal_pending=pending(c, env.state), terminal_failed=sum(env.state.failed))
        if journal is not None:
            journal.write(json.dumps(dict(phase=phase, candidate=spec["candidate"], policy=policy,
                seed=seed, physical_seed=physical_seed, tick=tick, state=asdict(state), action=action,
                events=env.events[tick], following=asdict(env.state), info=info), allow_nan=False) + "\n")
    if cutoff is None: raise RuntimeError("missing policy cutoff")
    row = dict(phase=phase, candidate=spec["candidate"], policy=policy, seed=seed,
        physical_seed=physical_seed, **{k: v / cfg["horizon"] for k, v in counts.items()},
        availability=1 - interior["unavailable_ticks"] / (c.machines * cfg["horizon"]),
        technician_utilization=interior["service_ticks"] / (c.technicians * cfg["horizon"]),
        **cutoff, tail_pending=pending(c, env.state), tail_failed=sum(env.state.failed),
        policy_cost=interior["objective"], tail_cost=env.metrics["objective"] - interior["objective"],
        extended_cost=env.metrics["objective"], policy_failures=interior["failures"],
        policy_overdue_ticks=interior["overdue_waiting_ticks"],
        **{f"extended_{key}": env.metrics[key] for key in
            ("maintenance_cost", "unavailability_cost", "failure_cost", "waiting_cost", "invalid_assignments")})
    components = sum(row[f"extended_{key}"] for key in
        ("maintenance_cost", "unavailability_cost", "failure_cost", "waiting_cost"))
    if not np.isclose(components, row["extended_cost"], atol=1e-8, rtol=0):
        raise RuntimeError("episode cost reconciliation")
    return row


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows: groups[row["candidate"], row["policy"]].append(row)
    excluded = {"phase", "candidate", "policy", "seed", "physical_seed"}
    return {key: {field: float(np.mean([r[field] for r in group])) for field in group[0] if field not in excluded}
        for key, group in groups.items()}


def rank_screen(cfg, specs, rows):
    means = aggregate(rows)
    ranking = []
    for spec in specs:
        group = [means[spec["candidate"], p] for p in POLICIES]
        metrics = {key: min(g[key] for g in group) for key in ("conflict_rate", "sensitive_choice_rate", "availability")}
        metrics["terminal_pending"] = max(g["terminal_pending"] for g in group)
        failures = [k for k, limit in cfg["limits"].items()
            if (metrics[k] > limit if k == "terminal_pending" else metrics[k] < limit)]
        eligible = spec["selectable"] and not failures
        ranking.append(dict(candidate=spec["candidate"], topology=spec["topology"], pattern=spec["pattern"],
            selectable=spec["selectable"], eligible=eligible, rejection=";".join(failures) if spec["selectable"] else "control",
            score=metrics["conflict_rate"] * metrics["sensitive_choice_rate"], **metrics,
            mean_cost=float(np.mean([g["extended_cost"] for g in group]))))
    ranking.sort(key=lambda r: (-int(r["eligible"]), -r["score"], r["candidate"]))
    strongest = sorted(ranking, key=lambda r: (-r["conflict_rate"], r["candidate"]))[0]["candidate"]
    qualified = [r for r in ranking if r["eligible"]]
    winner = qualified[0]["candidate"] if qualified else None
    runner_up = qualified[1]["candidate"] if len(qualified) > 1 else None
    selected = []
    if winner:
        pattern = qualified[0]["pattern"]
        selected = list(dict.fromkeys([winner, *([runner_up] if runner_up else []),
            "overlap3__equal_control", f"partitioned_control__{pattern}"]))
    return ranking, dict(status="SELECTED" if winner else "NO_ELIGIBLE_SETTING", winner=winner,
        runner_up=runner_up, strongest_contention=strongest, confirmation_candidates=selected,
        rule="min-policy conflict * min-policy physical choice; cost not used", limits=cfg["limits"],
        selection_phase="screening_only", confirmation_outcomes_opened=False)


def confirm(cfg, selection, rows):
    if not selection["winner"]:
        return dict(status="NOT_APPLICABLE", reason="NO_ELIGIBLE_SETTING", winner=None)
    seeds = cfg["confirmation_seeds"]
    winner = selection["winner"]
    lookup = {(r["candidate"], r["policy"], r["seed"]): r for r in rows}
    metrics = tuple(cfg["limits"])
    values = np.asarray([[[lookup[winner, p, s][k] for k in metrics] for p in POLICIES] for s in seeds])
    rng = np.random.default_rng(role_seed(cfg["analysis_seed"], f"{VERSION}:bootstrap"))
    draws = rng.integers(0, len(seeds), size=(cfg["bootstrap_samples"], len(seeds)))
    boot = values[draws].mean(1)
    point = values.mean(0)
    alpha = (1 - cfg["confidence"]) / 2
    summary = {}
    for j, key in enumerate(metrics):
        upper_gate = key == "terminal_pending"
        distribution = boot[:, :, j].max(1) if upper_gate else boot[:, :, j].min(1)
        estimate = point[:, j].max() if upper_gate else point[:, j].min()
        low, high = np.quantile(distribution, [alpha, 1 - alpha])
        passed = high <= cfg["limits"][key] if upper_gate else low >= cfg["limits"][key]
        summary[key] = dict(estimate=float(estimate), ci=[float(low), float(high)], pass_gate=bool(passed))
    return dict(status="CONFIRMED" if all(s["pass_gate"] for s in summary.values()) else "NOT_CONFIRMED",
        winner=winner, metrics=summary, confidence=cfg["confidence"],
        inference_scope="new seed trajectories in fixed selected setting under two observation policies")


def unpack_state(data):
    return DispatchState(**{k: tuple(v) for k, v in data.items()})


def replay_audit(path, specs, cfg):
    configs = {r["candidate"]: decode_config(r) for r in specs}
    envs, count = {}, 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line); c = configs[row["candidate"]]
            key = row["phase"], row["candidate"], row["policy"], row["seed"]
            if row["tick"] == 0:
                if key in envs: raise RuntimeError("duplicate audited trajectory")
                env = WaitingEnv(c); env.reset(row["physical_seed"]); envs[key] = env
            env = envs[key]
            state = unpack_state(row["state"])
            if row["tick"] != env.time or state != env.state or tuple(row["events"]) != env.events[env.time]:
                raise RuntimeError("audit state/event continuity")
            action = tuple(map(tuple, row["action"]))
            policy = row["policy"] if row["tick"] < cfg["horizon"] else "deadline_matching"
            if choose(c, state, policy) != action: raise RuntimeError("audit policy mismatch")
            following, cost, info = transition(c, state, action, row["events"])
            if following != unpack_state(row["following"]) or info != row["info"] or cost != info["objective"]:
                raise RuntimeError("independent scalar transition mismatch")
            env.step(action); count += 1
    if any(env.time != env.config.horizon for env in envs.values()):
        raise RuntimeError("incomplete audited trajectory")
    return dict(status="PASS", replayed_transitions=count, audited_episodes=len(envs))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_metadata():
    def command(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return dict(commit=command("rev-parse", "HEAD"), branch=command("branch", "--show-current"),
        dirty=bool(command("status", "--porcelain")))


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
    manifest = dict(protocol=VERSION, environment_version=ENV_VERSION, status="RUNNING", profile=profile,
        git=git, started_utc=datetime.now(timezone.utc).isoformat(), runtime=dict(python=platform.python_version(),
        platform=platform.platform(), numpy=np.__version__, device=device), full_confirmation_outcomes_opened=False)
    write_json(output / "manifest.json", manifest)
    try:
        write_json(output / "resolved_config.json", dict(settings=cfg, candidates=specs))
        shutil.copy2(SEED_PATH, output / "seed_registry.json")
        shutil.copy2(PLAN_PATH, output / "protocol.md")
        snapshot = output / "source_snapshot"; snapshot.mkdir()
        for name in ("maintenance_skill_contention_screen.py", "maintenance_dispatch.py", "maintenance_waiting.py",
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
            manifest.update(selection_sha256=selection_hash, phase="CONFIRMATION", full_confirmation_outcomes_opened=bool(selected))
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
        audit.update(seed_audit=seeds, invalid_assignments=invalid, selection_locked=True,
            screening_episodes=len(screening), confirmation_episodes=len(confirmation),
            expected_screening_episodes=expected_screen, expected_confirmation_episodes=expected_confirm,
            physical_transitions=(len(screening) + len(confirmation)) * (cfg["horizon"] + cfg["tail_ticks"]))
        write_json(output / "audit.json", audit)
        confirmation_result = confirm(cfg, selection, confirmation)
        means = aggregate(confirmation)
        write_json(output / "summary.json", dict(screening=ranking, selection=selection,
            confirmation=confirmation_result, confirmation_means=[dict(candidate=k[0], policy=k[1], **v) for k, v in means.items()],
            interpretation="screening rank is exploratory; no proof of optimal conflict or future RL benefit"))
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
