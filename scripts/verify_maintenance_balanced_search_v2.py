"""Independent shared-context artifact, physical and indexed-bootstrap verifier.

Usage: python verify_maintenance_balanced_search_v2.py RUN_DIRECTORY CHECKOUT [--output FILE]
Reads experiment outcomes only after the manifest says COMPLETED. Does not
import the experiment runner, its selection/statistics functions, or its physics.
"""

import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import subprocess

import numpy as np


VERSION = "maintenance_cross_skill_mac_v1"
PROTOCOL = "maintenance_balanced_search_mac_v2"
CHEAP = ("deadline_matching", "isolated_h6", "isolated_endpoint")
PLANNERS = ("assignment_rollout", "joint_rollout")
METHODS = (*CHEAP, *PLANNERS)
REFERENCES = ("baseline", "quality", "endpoint_dp")
TICKS = (0, 8, 16, 24, 32, 40, 47)
FIELDS = ("extended_cost", "availability", "terminal_pending", "policy_cost")


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def registered_integers(value):
    """Collect explicitly stored integers and expand every start/count panel."""
    if isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, list):
        return set().union(*(registered_integers(child) for child in value))
    if isinstance(value, dict):
        found = set().union(*(registered_integers(child) for child in value.values()))
        for key, start in value.items():
            count_key = key.removesuffix("_start") + "_count"
            if key.endswith("_start") and count_key in value:
                found.update(range(start, start + value[count_key]))
        return found
    return set()


def independent_seed_audit(run, registry):
    historical = set(); hashes = {}
    for path in sorted((run / "registry_snapshot").glob("*seed_registry.json")):
        require(path.name != PROTOCOL + "_seed_registry.json", "own registry excluded from history")
        historical.update(registered_integers(read_json(path)))
        hashes[path.name] = digest(path)
    groups = []; derived = []
    for profile in ("smoke", "full"):
        raw = registry["panels"][profile]
        seeds = {}
        for phase in ("development", "screening", "confirmation"):
            seeds[phase] = list(range(raw[phase + "_start"], raw[phase + "_start"] + raw[phase + "_count"]))
            groups.append(set(seeds[phase]))
            derived.extend(physical(seed) for seed in seeds[phase])
        groups.extend(({raw["analysis_seed"]}, {raw["forecast_seed"]}))
        for phase in ("screening", "confirmation"):
            for seed, tick, held in itertools.product(seeds[phase], range(48), (False, True)):
                derived.append(forecast_seed(raw, phase, seed, tick, held))
        derived.append(role_seed(raw["analysis_seed"], VERSION + ":bootstrap"))
    own = set.union(*groups)
    require(not any(a & b for a, b in itertools.combinations(groups, 2))
        and not own & historical, "fresh raw seed panels/history")
    require(len(derived) == len(set(derived)) and not set(derived) & (own | historical),
        "unique fresh derived stream roles")
    return dict(raw_unique=len(own), derived_unique=len(derived), historical_registry_hashes=hashes)


def role_seed(seed, role):
    return int.from_bytes(hashlib.sha256(f"{seed}:{role}".encode()).digest()[:8], "big")


def physical(seed):
    return role_seed(seed, f"{VERSION}:physical")


def forecast_seed(cfg, phase, seed, tick, held=False):
    return role_seed(cfg["forecast_seed"],
        f'{VERSION}:{phase}:{"holdout" if held else "selection"}:{seed}:{tick}')


def close(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ": fields")
        for key in expected:
            close(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), label + ": length")
        for a, b in zip(actual, expected):
            close(a, b, label)
    elif isinstance(expected, (int, float, np.number)) and not isinstance(expected, bool):
        require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected),
            rel_tol=1e-10, abs_tol=1e-9), label + ": numeric disagreement")
    else:
        require(actual == expected, label + ": disagreement")


def csv_records(path):
    with Path(path).open() as stream:
        yield from csv.DictReader(stream)


def typed_csv(path):
    strings = {"phase", "candidate", "method", "subset", "action"}
    result = []
    for row in csv_records(path):
        result.append({key: value if key in strings else
            float(value) if "." in value or "e" in value.lower() else int(value)
            for key, value in row.items()})
    return result


@lru_cache(None)
def catalogue(n, k):
    actions = [()]
    for size in range(1, min(n, k) + 1):
        for machines in itertools.combinations(range(n), size):
            for workers in itertools.permutations(range(k), size):
                actions.append(tuple(zip(machines, workers)))
    return tuple(sorted(actions))


def valid(c, state, action):
    machines = [m for m, _ in action]
    workers = [j for _, j in action]
    require(len(set(machines)) == len(machines) and len(set(workers)) == len(workers),
        "matching uniqueness")
    for m, j in action:
        require(0 <= m < c["machines"] and 0 <= j < c["technicians"], "matching bounds")
        require(m not in state["assigned"] and state["remaining"][j] == 0
            and c["service_time"][m][j] > 0, "matching resource feasibility")
        require(state["failed"][m] or state["ages"][m] >= c["failure_age"] - 1,
            "matching urgency")


def actions(c, state):
    result = []
    for action in catalogue(c["machines"], c["technicians"]):
        if all(m not in state["assigned"] and state["remaining"][j] == 0
            and c["service_time"][m][j] > 0
            and (state["failed"][m] or state["ages"][m] >= c["failure_age"] - 1)
            for m, j in action):
            result.append(action)
    return tuple(result)


def independent_dp(c):
    """Expected isolated-machine WAIT/start costs, from the scalar event law."""
    horizon, n, ages, limit, k = (c["horizon"], c["machines"], c["max_age"],
        c["waiting_limit"], c["technicians"])
    values = np.zeros((horizon + 1, n, ages + 1, 2, limit + 1))
    gains = np.full((*values.shape, k), -np.inf)
    for h in range(1, horizon + 1):
        for m, age, failed, waiting in itertools.product(range(n), range(ages + 1),
            range(2), range(limit + 1)):
            if failed:
                idle = c["unavailable_cost"] + c["waiting_price"] * (waiting >= limit)
                idle += values[h - 1, m, age, 1, min(limit, waiting + 1)]
            else:
                older = min(ages, age + 1)
                hazard = c["failure_probability"] if older >= c["failure_age"] else 0.
                idle = ((1 - hazard) * values[h - 1, m, older, 0, 0]
                    + hazard * (c["failure_cost"] + values[h - 1, m, older, 1, 0]))
            candidates = [idle]
            if failed or age >= c["failure_age"] - 1:
                for j, duration in enumerate(c["service_time"][m]):
                    if not duration:
                        continue
                    service = c["corrective_cost"] if failed else c["preventive_cost"]
                    service += min(duration, h) * c["unavailable_cost"]
                    if duration < h:
                        restored = math.floor(age * (1 - c["restoration"][m][j]))
                        service += values[h - duration, m, restored, 0, 0]
                    gains[h, m, age, failed, waiting, j] = idle - service
                    candidates.append(service)
            values[h, m, age, failed, waiting] = min(candidates)
    return gains


def edge_score(c, state, tick, method, gains, m, j):
    if method == "deadline_matching":
        hazard = c["failure_probability"] * (not state["failed"][m]
            and state["ages"][m] + 1 >= c["failure_age"])
        return (100 * state["failed"][m] + 10 * hazard + state["ages"][m] / 8
            + 20 * max(0, state["pending_wait"][m] - 3)
            + c["restoration"][m][j] / c["service_time"][m][j])
    h = min(6, 60 - tick) if method == "isolated_h6" else 60 - tick
    return gains[h, m, state["ages"][m], int(state["failed"][m]),
        min(c["waiting_limit"], state["pending_wait"][m]), j]


def cheap(c, state, tick, method, gains):
    pool = actions(c, state)
    scores = [sum(edge_score(c, state, tick, method, gains, m, j) for m, j in a)
        for a in pool]
    return pool[max(range(len(pool)), key=lambda i: (scores[i], -i))]


def subset_actions(c, state, baseline):
    machines = tuple(m for m, _ in baseline)
    return (baseline, *(a for a in actions(c, state) if a != baseline
        and tuple(m for m, _ in a) == machines))


def pending(state):
    return sum(failed and m not in state["assigned"] for m, failed in enumerate(state["failed"]))


def scalar_tick(c, state, action, shocks):
    """Independently reconstruct one simultaneous-start, end-completion interval."""
    valid(c, state, action)
    after = {key: list(value) for key, value in state.items()}
    pm = cm = 0
    for m, j in action:
        after["assigned"][j] = m
        after["remaining"][j] = c["service_time"][m][j]
        if state["failed"][m]:
            cm += 1
        else:
            pm += 1
    busy = {m for m in after["assigned"] if m >= 0}
    unavailable = sum(failed or m in busy for m, failed in enumerate(state["failed"]))
    waiting = sum(failed and m not in busy for m, failed in enumerate(state["failed"]))
    overdue = sum(failed and m not in busy and state["pending_wait"][m] + 1 > c["waiting_limit"]
        for m, failed in enumerate(state["failed"]))
    failures = completed = 0
    for m in range(c["machines"]):
        if m in busy:
            continue
        if state["failed"][m]:
            after["pending_wait"][m] += 1
        else:
            after["ages"][m] = min(c["max_age"], state["ages"][m] + 1)
            if after["ages"][m] >= c["failure_age"] and shocks[m]:
                after["failed"][m] = True
                after["pending_wait"][m] = 0
                failures += 1
    for j, m in enumerate(after["assigned"]):
        if m < 0:
            continue
        after["remaining"][j] -= 1
        if after["remaining"][j] == 0:
            after["ages"][m] = math.floor(after["ages"][m] * (1 - c["restoration"][m][j]))
            after["failed"][m] = False
            after["pending_wait"][m] = 0
            after["assigned"][j] = -1
            completed += 1
    maintenance = pm * c["preventive_cost"] + cm * c["corrective_cost"]
    down = unavailable * c["unavailable_cost"]
    failure = failures * c["failure_cost"]
    base = maintenance + down + failure
    price = overdue * c["waiting_price"]
    info = dict(objective=base + price, maintenance_cost=maintenance, unavailability_cost=down,
        failure_cost=failure, preventive_jobs=pm, corrective_jobs=cm, jobs=pm + cm,
        completed_jobs=completed, failures=failures, unavailable_ticks=unavailable,
        failed_waiting_ticks=waiting, service_ticks=len(busy), invalid_assignments=0,
        base_cost=base, waiting_cost=price, overdue_waiting_ticks=overdue,
        service_adjusted_cost=base + 12 * overdue)
    return after, info


def panel(rows, local, seeds, candidate):
    selected = [r for r in rows if r["candidate"] == candidate]
    lookup = {(r["seed"], r["method"]): r for r in selected}
    require(len(selected) == len(lookup) == 5 * len(seeds)
        and set(lookup) == set(itertools.product(seeds, METHODS)), "method/seed grid")
    labels = [r for r in local if r["candidate"] == candidate]
    landmarks = {(r["seed"], r["tick"]): r for r in labels}
    require(len(labels) == len(landmarks) == 7 * len(seeds)
        and set(landmarks) == set(itertools.product(seeds, TICKS)), "clock landmark grid")
    values = np.asarray([[[lookup[s, m][field] for field in FIELDS] for m in METHODS]
        for s in seeds], dtype=float)
    local_values = np.asarray([[sum(landmarks[s, t][field] for t in TICKS) / 7
        for field in (*["gain_" + r for r in REFERENCES], "choice")] for s in seeds])
    require(np.isfinite(values).all() and np.isfinite(local_values).all()
        and (values[..., 0] > 0).all() and (values[..., 3] > 0).all(), "finite positive costs")
    return values, local_values


def metric_values(v, local, base):
    b = METHODS.index(base); a = METHODS.index("assignment_rollout"); j = METHODS.index("joint_rollout")
    out = {"local_" + reference: local[..., index] for index, reference in enumerate(REFERENCES)}
    out["assignment_cost_gain"] = (v[..., b, 0] - v[..., a, 0]) / v[..., b, 0]
    for m in CHEAP:
        i = METHODS.index(m)
        out["joint_cost_vs_" + m] = (v[..., i, 0] - v[..., j, 0]) / v[..., i, 0]
    for name, i in (("assignment", a), ("joint", j)):
        out[name + "_availability_difference"] = v[..., i, 1] - v[..., b, 1]
        out[name + "_pending_difference"] = v[..., i, 2] - v[..., b, 2]
        out[name + "_absolute_availability"] = v[..., i, 1]
        out[name + "_absolute_pending"] = v[..., i, 2]
        out[name + "_interior_gain"] = (v[..., b, 3] - v[..., i, 3]) / v[..., b, 3]
    out["assignment_choice_fraction"] = local[..., 3]
    require(len(out) == 18, "18-metric family")
    return out


def gate(key, low, high):
    if key.startswith("local_"):
        return low >= .5
    if key.endswith("pending_difference"):
        return high <= .10
    if key.endswith("absolute_pending"):
        return high <= 1
    if key.endswith("availability_difference"):
        return low >= -.01
    if key.endswith("absolute_availability"):
        return low >= .40
    if key == "assignment_choice_fraction":
        return low >= .10
    if key.endswith("interior_gain"):
        return low > 0
    return low >= .01


def screen_eligible(point):
    failures = []
    if point["assignment_cost_gain"] < .015:
        failures.append("assignment_cost_gain")
    failures.extend("joint_cost_vs_" + m for m in CHEAP if point["joint_cost_vs_" + m] < .015)
    failures.extend("local_" + ref for ref in REFERENCES if point["local_" + ref] < .75)
    if point["assignment_choice_fraction"] < .15:
        failures.append("assignment_choice_fraction")
    for p in ("assignment", "joint"):
        for suffix, passes in (("availability_difference", point[p + "_availability_difference"] >= -.005),
            ("pending_difference", point[p + "_pending_difference"] <= .05),
            ("absolute_availability", point[p + "_absolute_availability"] >= .4),
            ("absolute_pending", point[p + "_absolute_pending"] <= .5),
            ("interior_gain", point[p + "_interior_gain"] > 0)):
            if not passes:
                failures.append(p + "_" + suffix)
    return failures


def verify(run, checkout):
    manifest = read_json(run / "manifest.json")
    require(manifest.get("status") == "COMPLETED", "outcomes remain closed until COMPLETED")
    require(manifest.get("audit") == "PASS" and manifest["protocol"] == PROTOCOL, "completed audit/protocol")
    file_set = {str(p.relative_to(run)) for p in run.rglob("*") if p.is_file() and p.name != "manifest.json"}
    require(file_set == set(manifest["files"]), "complete manifested file population")
    for name, metadata in manifest["files"].items():
        path = run / name
        require(path.is_relative_to(run) and path.stat().st_size == metadata["bytes"]
            and digest(path) == metadata["sha256"], "artifact hash/size: " + name)
    require(sum(m["bytes"] for m in manifest["files"].values()) == manifest["artifact_bytes"], "artifact byte total")
    commit = manifest["git"]["commit"]
    source_members = {Path(name).name for name in manifest["source_seal"]
        if name.startswith("source_snapshot/")}
    require({"maintenance_balanced_search_mac_v2.py", "maintenance_balanced_confirmation_mac.py",
        "maintenance_cross_skill_mac.py", "maintenance_cross_skill_core.py",
        "maintenance_joint_frontier_mac.py", "maintenance_restricted_dp.py",
        "maintenance_dispatch.py", "maintenance_waiting.py", "maintenance_heterogeneous_model.py"}
        <= source_members and "maintenance_balanced_search_mac.py" not in source_members,
        "required v2/prior/engine source bindings; v1 driver excluded")
    source_count = 0
    for name, sealed in manifest["source_seal"].items():
        require(digest(run / name) == sealed, "source seal: " + name)
        if name.startswith("source_snapshot/"):
            git_path = "src/ht_pdm_fjsp/" + Path(name).name
        elif name.startswith("registry_snapshot/"):
            git_path = "configs/" + Path(name).name
        elif name == "protocol.md":
            git_path = "docs/" + PROTOCOL + "_plan.md"
        elif name == "registry.json":
            git_path = "configs/" + PROTOCOL + "_seed_registry.json"
        else:
            raise AssertionError("unexpected source seal member")
        blob = subprocess.check_output(["git", "show", f"{commit}:{git_path}"], cwd=checkout)
        require(hashlib.sha256(blob).hexdigest() == sealed, "Git blob binding: " + git_path)
        source_count += 1
    cfg = read_json(run / "resolved_config.json")
    registered = read_json(run / "registry.json")
    require(registered["protocol"] == PROTOCOL and registered["round"] == 2
        and registered["family_alpha"] == .0125 and registered["confidence"] == 1 - .0125 / 18
        and registered["bootstrap_samples"] == 160000
        and registered["series"] == "balanced_2x2_fresh_search_20261012", "locked new-series spending")
    expected_panels = {}
    for profile, starts, counts_ in (("full", (45000000, 45100000, 45200000, 45300000, 45400000), (24, 32, 256)),
        ("smoke", (45500000, 45600000, 45700000, 45800000, 45900000), (2, 2, 2))):
        expected_panels[profile] = {"analysis_seed": starts[3], "forecast_seed": starts[4]}
        for phase, start, count in zip(("development", "screening", "confirmation"), starts[:3], counts_):
            expected_panels[profile].update({phase + "_start": start, phase + "_count": count})
    close(registered["panels"], expected_panels, "locked exact seed allocations")
    seed_integrity = independent_seed_audit(run, registered)
    require(cfg["round"] == 2 and cfg["family_alpha"] == .0125
        and cfg["stream_namespace"] == VERSION and cfg["search_series"] == registered["series"], "registered stream namespace/spending")
    registry = registered["panels"][manifest["profile"]]
    for phase in ("development", "screening", "confirmation"):
        require(cfg[phase + "_seeds"] == list(range(registry[phase + "_start"],
            registry[phase + "_start"] + registry[phase + "_count"])), "registered seed panel")
    require(cfg["analysis_seed"] == registry["analysis_seed"] and cfg["forecast_seed"] == registry["forecast_seed"]
        and cfg["confidence"] == 1 - .0125 / 18 and cfg["landmarks"] == list(TICKS), "registered inference fields")
    require(cfg["scenarios"] == (128 if manifest["profile"] == "full" else 8)
        and cfg["bootstrap_samples"] == (160000 if manifest["profile"] == "full" else 200), "registered effort")
    require(cfg["horizon"] == 48 and cfg["tail_ticks"] == 12
        and cfg["profile"] == manifest["profile"], "registered physical clock")
    specs = read_json(run / "candidates.json")
    configs = {spec["candidate"]: spec["config"] for spec in specs}
    require(len(configs) == len(specs) == (10 if manifest["profile"] == "full" else 4), "candidate population")
    require(sum(bool(s["selectable"]) for s in specs) == (8 if manifest["profile"] == "full" else 2)
        and not any(s["selectable"] for s in specs if s["candidate"] in ("old_balanced_d3_q050", "exchangeable")), "nonselectable controls")
    family = {f"balanced_a{a}_b{b}_u{u:02d}" for (a, b), u in
        itertools.product(((2, 2), (2, 4), (3, 3), (3, 5)), (9, 12))}
    if manifest["profile"] == "smoke":
        family = {"balanced_a2_b4_u09", "balanced_a3_b5_u12"}
    require(set(configs) == family | {"old_balanced_d3_q050", "exchangeable"}, "locked family identities")
    fixed = dict(machines=4, technicians=2, horizon=60, failure_age=4,
        failure_probability=.45, max_age=8, preventive_cost=1., corrective_cost=2.,
        failure_cost=15., waiting_limit=4, waiting_price=12.)
    for spec in specs:
        name = spec["candidate"]
        if name in family:
            _, left, right, price = name.split("_")
            a, b, u = int(left[1:]), int(right[1:]), int(price[1:])
            service = [[1, a], [1, b], [a, 1], [b, 1]]
            restoration = [[.5, 1.], [.5, 1.], [1., .5], [1., .5]]
            require(spec["selectable"] and u in (9, 12), "selectable balanced profile/price")
        elif name == "old_balanced_d3_q050":
            service = [[1, 3], [1, 3], [3, 1], [3, 1]]
            restoration = [[.5, 1.], [.5, 1.], [1., .5], [1., .5]]
            u = 6
        else:
            service = [[1, 1] for _ in range(4)]
            restoration = [[.75, .75] for _ in range(4)]
            u = 6
        close(spec["config"], {**fixed, "unavailable_cost": u,
            "service_time": service, "restoration": restoration}, "locked physical family " + name)
    dev = typed_csv(run / "development_episodes.csv")
    screen = typed_csv(run / "screening_episodes.csv")
    screen_local = typed_csv(run / "screening_landmarks.csv")
    choices = read_json(run / "baseline_choices.json")
    frozen = read_json(run / "selection.json")
    require(digest(run / "baseline_choices.json") == manifest["baseline_choices_sha256"]
        and digest(run / "selection.json") == manifest["selection_sha256"], "sealed choices")
    require(not frozen["confirmation_outcomes_opened"], "selection made before confirmation")
    expected_dev = set(itertools.product(configs, CHEAP, cfg["development_seeds"]))
    actual_dev = {(r["candidate"], r["method"], r["seed"]) for r in dev}
    require(actual_dev == expected_dev and len(dev) == len(expected_dev), "development grid")
    reconstructed = {}
    for name in configs:
        means = {m: float(np.mean([r["extended_cost"] for r in dev if r["candidate"] == name and r["method"] == m]))
            for m in CHEAP}
        reconstructed[name] = dict(base=min(CHEAP, key=lambda m: (means[m], m)), development_means=means)
    close(choices, reconstructed, "frozen baseline reconstruction")
    screens = []
    for spec in specs:
        name = spec["candidate"]
        values, local = panel(screen, screen_local, cfg["screening_seeds"], name)
        point = {k: float(v) for k, v in metric_values(values.mean(0), local.mean(0), choices[name]["base"]).items()}
        failures = screen_eligible(point)
        score = min(point["assignment_cost_gain"], *[point["joint_cost_vs_" + m] for m in CHEAP])
        baseline_index = METHODS.index(choices[name]["base"])
        stability = []
        for method in PLANNERS:
            paired = values[:, METHODS.index(method), 1] - values[:, baseline_index, 1]
            require(len(paired) > 1, "health score sample size")
            stability.append(float(paired.mean() - 2 * paired.std(ddof=1) / math.sqrt(len(paired))))
        screens.append(dict(candidate=name, selectable=spec["selectable"], eligible=bool(spec["selectable"] and not failures),
            score=float(score), point=point, health_stability_score=min(stability), failed_screen_metrics=failures))
    contenders = [s for s in screens if s["eligible"] or (manifest["profile"] == "smoke" and s["selectable"])]
    winner = min(contenders, key=lambda s: (-s["health_stability_score"], -s["score"], s["candidate"]))["candidate"] if contenders else None
    require(winner == frozen["winner"] == manifest["winner"], "sealed winner without fallback")
    close([{k: v for k, v in s.items() if k != "failed_screen_metrics"} for s in screens], frozen["scores"], "screen reconstruction")
    require(frozen["status"] == ("SELECTED" if winner else "NO_SCREEN_CANDIDATE"), "screen selection status")
    require(frozen["ranking"] == "max min planner paired availability mean-minus-two-SE; then min cost gain; then ID", "locked health-first ranking")
    if winner:
        confirm = typed_csv(run / "episodes.csv")
        confirm_local = typed_csv(run / "landmarks.csv")
        require((run / "episodes.partial.csv").read_bytes() == (run / "episodes.csv").read_bytes(), "partial/final episode identity")
        require({r["candidate"] for r in confirm} == {winner}, "single confirmed candidate")
        panel(confirm, confirm_local, cfg["confirmation_seeds"], winner)
        require(manifest["confirmation_outcomes_opened"] == (manifest["profile"] == "full"), "confirmation opening flag")
    else:
        confirm = []; confirm_local = []
        require(not manifest["confirmation_outcomes_opened"] and not (run / "episodes.csv").exists()
            and not (run / "landmarks.csv").exists(), "failed screening cannot open confirmation")
    all_rows = dev + screen + confirm
    lookup = {(r["phase"], r["candidate"], r["method"], r["seed"]): r for r in all_rows}
    require(len(lookup) == len(all_rows), "unique physical episode grid")
    for phase, rows in (("development", dev), ("screening", screen), ("confirmation", confirm)):
        require(all(r["phase"] == phase and r["physical_seed"] == physical(r["seed"])
            and r["extended_invalid_assignments"] == 0 for r in rows), "episode provenance/feasibility")
    dps = {name: independent_dp(c) for name, c in configs.items()}
    active = {}; seen = set(); states = {}; ticks = 0; forecast_total = 0
    for record in csv_records(run / "decisions.csv"):
        phase, name, method, seed = record["phase"], record["candidate"], record["method"], int(record["seed"])
        tick = int(record["tick"]); key = phase, name, method, seed
        require(key in lookup and int(record["physical_seed"]) == physical(seed), "physical decision provenance")
        c = configs[name]; gains = dps[name]
        if tick == 0:
            require(key not in seen, "unique decision trajectory"); seen.add(key)
            rng = random.Random(role_seed(physical(seed), "initial"))
            initial = dict(ages=[rng.randrange(c["failure_age"]) for _ in range(4)],
                failed=[False] * 4, pending_wait=[0] * 4, remaining=[0] * 2, assigned=[-1] * 2)
            active[key] = dict(state=initial, tick=0, rng=random.Random(role_seed(physical(seed), "failures")),
                inside=defaultdict(float), totals=defaultdict(float), changes=defaultdict(float), forecasts=0)
        require(key in active, "decision episode exists")
        trajectory = active[key]; state = trajectory["state"]
        require(trajectory["tick"] == tick and tick in range(60), "consecutive physical ticks")
        close(json.loads(record["state"]), state, "independent physical state")
        events = [trajectory["rng"].random() < c["failure_probability"] for _ in range(4)]
        close(json.loads(record["events"]), events, "role-derived physical events")
        action = tuple(map(tuple, json.loads(record["action"])))
        valid(c, state, action)
        base = "deadline_matching" if phase == "development" else choices[name]["base"]
        baseline = cheap(c, state, tick, base, gains) if tick < 48 else None
        if method in CHEAP or tick >= 48:
            expected = cheap(c, state, tick, method if tick < 48 else "deadline_matching", gains)
            require(action == expected, "independent cheap action")
        diag = json.loads(record["diagnostic"])
        if method in PLANNERS and tick < 48:
            pool = subset_actions(c, state, baseline) if method == "assignment_rollout" else (baseline,
                *(a for a in actions(c, state) if a != baseline))
            require(action in pool, "planner root membership")
            if len(pool) > 1:
                require(diag["planned"] and type(diag["selected_root"]) is int
                    and 0 <= diag["selected_root"] < len(pool) and pool[diag["selected_root"]] == action
                    and diag["candidate_count"] == len(pool)
                    and diag["selection_seed"] == forecast_seed(cfg, phase, seed, tick)
                    and diag["forecast_transitions"] == len(pool) * cfg["scenarios"] * (60 - tick)
                    and math.isfinite(diag["selected_mean"]) and math.isfinite(diag["baseline_mean"])
                    and diag["selected_mean"] <= diag["baseline_mean"] + 1e-10, "planner selection contract")
            else:
                close(diag, dict(planned=False, forecast_transitions=0), "single-root forecast")
        else:
            close(diag, dict(planned=False, forecast_transitions=0), "cheap/tail forecast")
        if tick < 48:
            subset_same = tuple(m for m, _ in baseline) == tuple(m for m, _ in action)
            metric = dict(disagreement_rate=int(baseline != action), subset_change_rate=int(not subset_same),
                assignment_only_change_rate=int(baseline != action and subset_same),
                cardinality_change_rate=int(len(baseline) != len(action)), wait_change_rate=int(bool(baseline) != bool(action)))
            if method == "assignment_rollout":
                require(subset_same, "assignment-only current subset")
            if phase != "development" and method == base and tick in TICKS:
                states[phase, name, seed, tick] = (state, action)
            for field, amount in metric.items():
                trajectory["changes"][field] += amount
        else:
            metric = None
        close(json.loads(record["metrics"]), metric, "decision change metrics")
        after, info = scalar_tick(c, state, action, events)
        close(json.loads(record["following"]), after, "independent physical following")
        close(json.loads(record["info"]), info, "independent scalar costs")
        for field, amount in info.items():
            trajectory["totals"][field] += amount
            if tick < 48:
                trajectory["inside"][field] += amount
        if tick == 47:
            close(lookup[key]["terminal_pending"], pending(after), "terminal pending")
            close(lookup[key]["terminal_failed"], sum(after["failed"]), "terminal failed")
        trajectory["state"] = after; trajectory["tick"] += 1
        trajectory["forecasts"] += diag["forecast_transitions"]
        forecast_total += diag["forecast_transitions"]; ticks += 1
        if tick == 59:
            inside = trajectory["inside"]; totals = trajectory["totals"]
            reconstructed_row = dict(availability=1 - inside["unavailable_ticks"] / 192,
                tail_pending=pending(after), tail_failed=sum(after["failed"]), policy_cost=inside["objective"],
                extended_cost=totals["objective"], tail_cost=totals["objective"] - inside["objective"],
                policy_failures=inside["failures"], policy_overdue_ticks=inside["overdue_waiting_ticks"],
                forecast_transitions=trajectory["forecasts"],
                **{field: value / 48 for field, value in trajectory["changes"].items()},
                **{"extended_" + field: totals[field] for field in ("maintenance_cost", "unavailability_cost", "failure_cost", "waiting_cost", "invalid_assignments")})
            for field, value in reconstructed_row.items():
                close(lookup[key][field], value, "physical aggregate " + field)
            require(math.isfinite(lookup[key]["planning_seconds"]) and lookup[key]["planning_seconds"] >= 0, "planning latency")
            del active[key]
    require(not active and seen == set(lookup) and ticks == len(all_rows) * 60, "complete physical population")
    local_rows = screen_local + confirm_local
    roots = typed_csv(run / "screening_roots.csv") + (typed_csv(run / "landmark_roots.csv") if winner else [])
    root_groups = defaultdict(list)
    for root in roots:
        root_groups[root["phase"], root["candidate"], root["seed"], root["tick"]].append(root)
    local_keys = {(r["phase"], r["candidate"], r["seed"], r["tick"]) for r in local_rows}
    require(len(local_keys) == len(local_rows) and set(root_groups) == local_keys == set(states), "complete root population")
    for local in local_rows:
        key = local["phase"], local["candidate"], local["seed"], local["tick"]
        phase, name, seed, tick = key; c = configs[name]; state, baseline = states[key]
        pool = subset_actions(c, state, baseline)
        rr = {r["root"]: r for r in root_groups[key]}
        require(len(root_groups[key]) == len(rr) == len(pool) and set(rr) == set(range(len(pool))), "unique complete root grid")
        require(json.loads(local["subset"]) == [m for m, _ in baseline], "fixed root subset")
        require(local["selection_seed"] == forecast_seed(cfg, phase, seed, tick)
            and local["holdout_seed"] == forecast_seed(cfg, phase, seed, tick, True)
            and local["selection_seed"] != local["holdout_seed"], "independent root bank roles")
        selected_scores = [rr[i]["selection_mean"] for i in range(len(pool))]
        chosen = next(i for i, value in enumerate(selected_scores) if value <= min(selected_scores) + 1e-10)
        quality = [sum(c["service_time"][m][j] / c["restoration"][m][j] for m, j in a) for a in pool]
        economic = [sum(dps[name][60 - tick, m, state["ages"][m], int(state["failed"][m]),
            min(c["waiting_limit"], state["pending_wait"][m]), j] for m, j in a) for a in pool]
        references = dict(baseline=0, quality=min(range(len(pool)), key=lambda i: (quality[i], i)),
            endpoint_dp=max(range(len(pool)), key=lambda i: (economic[i], -i)))
        for i, action in enumerate(pool):
            require(tuple(map(tuple, json.loads(rr[i]["action"]))) == action and rr[i]["selected"] == int(i == chosen), "root identity/selection")
            require(math.isfinite(rr[i]["selection_mean"]) and math.isfinite(rr[i]["holdout_mean"]), "finite root scores")
            for reference, index in references.items():
                require(rr[i]["reference_" + reference] == int(i == index), "root reference identity")
        duration = [sum(c["service_time"][m][j] for m, j in a) for a in pool]
        restoration = [sum(c["restoration"][m][j] for m, j in a) for a in pool]
        restored = [sum(math.floor(state["ages"][m] * (1 - c["restoration"][m][j])) for m, j in a) for a in pool]
        check = dict(chosen_root=chosen, choice=int(len(pool) > 1), changed=int(chosen != 0),
            duration_span=max(duration) - min(duration), restoration_span=max(restoration) - min(restoration),
            completion_age_span=max(restored) - min(restored),
            forecast_transitions=2 * len(pool) * cfg["scenarios"] * (60 - tick),
            **{"gain_" + ref: rr[index]["holdout_mean"] - rr[chosen]["holdout_mean"] for ref, index in references.items()})
        for field, expected in check.items():
            close(local[field], expected, "root assay " + field)
        if len(pool) == 1 or name == "exchangeable":
            require(all(abs(local["gain_" + r]) < 1e-9 for r in REFERENCES), "zero-choice/exchangeability null")
        forecast_total += check["forecast_transitions"]
    counts = dict(development=len(dev), screening=len(screen), confirmation=len(confirm), landmarks=len(local_rows))
    close(manifest["actual_counts"], counts, "manifest phase counts")
    audit = read_json(run / "audit.json")
    require(audit["seed_audit"]["status"] == "PASS", "runner seed audit passed")
    for field, value in seed_integrity.items():
        close(audit["seed_audit"][field], value, "independent seed audit " + field)
    require(audit["seed_audit"]["stream_namespace"] == VERSION, "seed audit namespace")
    close(audit["actual_counts"], counts, "audit actual counts")
    close(audit["expected_counts"], counts, "audit expected counts")
    require(audit["physical_transitions"] == ticks and audit["episodes"] == len(all_rows)
        and audit["landmarks"] == len(local_rows) and audit["forecast_transitions"] == forecast_total
        and forecast_total <= audit["forecast_cap"], "audited physical/forecast accounting")
    summary = read_json(run / "summary.json")
    require(summary["search_series"] == cfg["search_series"] and summary["round"] == 2
        and summary["family_alpha"] == .0125, "published series spending")
    close(summary["screening"], frozen, "published screen selection")
    comparisons = {}; gates = {}
    if winner:
        v, labels = panel(confirm, confirm_local, cfg["confirmation_seeds"], winner)
        n = len(v); base = choices[winner]["base"]
        random_indices = np.random.default_rng(role_seed(cfg["analysis_seed"], f"{VERSION}:bootstrap")).integers(
            n, size=(cfg["bootstrap_samples"], n))
        # Independent implementation: directly index complete seed blocks, never
        # use the runner's per-seed frequency weights or einsum aggregation.
        boot_v = []; boot_l = []
        for indices in np.array_split(random_indices, max(1, len(random_indices) // 100)):
            boot_v.append(v[indices].mean(axis=1))
            boot_l.append(labels[indices].mean(axis=1))
        boot_v = np.concatenate(boot_v); boot_l = np.concatenate(boot_l)
        point = metric_values(v.mean(axis=0), labels.mean(axis=0), base)
        boot = metric_values(boot_v, boot_l, base)
        alpha = (1 - cfg["confidence"]) / 2
        for key in point:
            bounds = np.quantile(boot[key], [alpha, 1 - alpha]).tolist()
            comparisons[key] = dict(estimate=float(point[key]), ci=bounds, confidence=cfg["confidence"])
            gates[key] = bool(gate(key, *bounds))
        close(summary["comparisons"], comparisons, "independent 18 indexed-bootstrap intervals")
        require(summary["gates"] == gates and summary["candidate"] == winner and summary["base"] == base, "published gates/winner/base")
        scientific = ("CROSS_SKILL_SETTING_CONFIRMED" if all(gates.values()) else "NOT_CONFIRMED") if manifest["profile"] == "full" else "ENGINEERING_ONLY"
    else:
        scientific = "NO_SCREEN_CANDIDATE" if manifest["profile"] == "full" else "ENGINEERING_ONLY"
    require(scientific == manifest["scientific_status"] == summary["scientific_status"], "scientific status")
    physical_components = []
    for phase, phase_rows in (("development", dev), ("screening", screen), ("confirmation", confirm)):
        for name, method in sorted({(r["candidate"], r["method"]) for r in phase_rows}):
            selected_rows = [r for r in phase_rows if r["candidate"] == name and r["method"] == method]
            c = configs[name]
            physical_components.append(dict(phase=phase, candidate=name, method=method,
                unavailable_cost=c["unavailable_cost"], episodes=len(selected_rows),
                mean_policy_unavailable_ticks=float(np.mean([(1 - r["availability"]) * 192 for r in selected_rows])),
                mean_extended_unavailable_ticks=float(np.mean([r["extended_unavailability_cost"] / c["unavailable_cost"] for r in selected_rows])),
                mean_policy_failures=float(np.mean([r["policy_failures"] for r in selected_rows])),
                mean_policy_overdue_ticks=float(np.mean([r["policy_overdue_ticks"] for r in selected_rows])),
                mean_extended_failures=float(np.mean([r["extended_failure_cost"] / c["failure_cost"] for r in selected_rows])),
                mean_extended_overdue_ticks=float(np.mean([r["extended_waiting_cost"] / c["waiting_price"] for r in selected_rows])),
                mean_terminal_pending=float(np.mean([r["terminal_pending"] for r in selected_rows])),
                mean_extended_cost=float(np.mean([r["extended_cost"] for r in selected_rows])),
                mean_maintenance_cost=float(np.mean([r["extended_maintenance_cost"] for r in selected_rows])),
                mean_unavailability_cost=float(np.mean([r["extended_unavailability_cost"] for r in selected_rows])),
                mean_failure_cost=float(np.mean([r["extended_failure_cost"] for r in selected_rows])),
                mean_waiting_cost=float(np.mean([r["extended_waiting_cost"] for r in selected_rows]))))
    result = dict(protocol=PROTOCOL, run=str(run), commit=commit, scientific_status=scientific, winner=winner,
        integrity=dict(status="PASS", hashed_files=len(manifest["files"]), git_bound_source_blobs=source_count,
            phase_counts=counts, physical_ticks=ticks, root_rows=len(roots), forecasts=forecast_total,
            baseline_and_winner_reconstructed=True, all_physical_states_events_and_costs_reconstructed=True,
            seed_roles=dict(raw_unique=seed_integrity["raw_unique"], derived_unique=seed_integrity["derived_unique"],
                historical_scope="historical registered integers and expanded ranges; own derived roles"),
            independent_indexed_bootstrap=bool(winner), independence="independent scalar/indexed-bootstrap implementation reused and adapted in shared context; not blind or cross-model"),
        search_series=cfg["search_series"], round=2, family_alpha=.0125,
        screening=screens, comparisons=comparisons, gates=gates, physical_components=physical_components,
        failed_confirmation_gates=[k for k, passed in gates.items() if not passed],
        limitations="Local assay fixes initial state/tick/subset, not future schedules. Geometric spending applies only to the new declared series; bootstrap coverage is approximate and excludes earlier adaptive searches. Health ranking is a heuristic, not a CI. Raw monetary local gains across downtime prices9/12/6 are not evidence of stronger physical benefit; compare physical ticks/failures/backlog and components. Mean-matched(2,4) versus(3,3) can probe heterogeneity; (3,5) also changes mean burden.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--output", type=Path, help="new analysis file; existing files are refused")
    args = parser.parse_args()
    destination = (args.output or Path(__file__).with_name("analysis.json")).resolve()
    require(not destination.exists(), "refusing to overwrite an earlier analysis; choose a fresh --output")
    result = verify(args.run_directory.resolve(), args.checkout.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["integrity"]["status"], scientific_status=result["scientific_status"],
        winner=result["winner"], output=str(destination)), indent=2))


if __name__ == "__main__":
    main()
