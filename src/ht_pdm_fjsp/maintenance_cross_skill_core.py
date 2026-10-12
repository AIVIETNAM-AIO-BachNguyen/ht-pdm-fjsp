"""Locked reciprocal-skill profiles and same-subset allocation references.

Only the initial allocation differs in a subset pool. Later continuation may
react to different restoration and technician occupancy caused by that choice.
"""

from dataclasses import asdict
import math

from ht_pdm_fjsp import maintenance_joint_frontier_mac as frontier
from ht_pdm_fjsp import maintenance_restricted_dp as dp
from ht_pdm_fjsp.maintenance_dispatch import validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig


def specs(profile="full"):
    """Return fresh serializable specifications; controls cannot win selection."""
    if profile not in ("full", "smoke"):
        raise ValueError("unknown profile")
    anchor = frontier.make_config(frontier.settings("smoke"))
    result = []
    for duration in (2, 3):
        for quality in (.25, .50):
            for group, boundary in (("balanced", 2), ("skewed", 3)):
                rows = tuple(m < boundary for m in range(anchor.machines))
                config = WaitingConfig(**{
                    **asdict(anchor),
                    "service_time": tuple((1, duration) if first else (duration, 1)
                        for first in rows),
                    "restoration": tuple((quality, 1.) if first else (1., quality)
                        for first in rows),
                })
                config.validate()
                name = f"cross_d{duration}_q{round(100 * quality):03d}_{group}"
                result.append(dict(candidate=name, selectable=True, config=asdict(config)))
    if profile == "smoke":
        anchors = {"cross_d2_q025_balanced", "cross_d3_q050_skewed"}
        result = [s for s in result if s["candidate"] in anchors]
    result.append(dict(candidate="old_fast1_slow325", selectable=False, config=asdict(anchor)))
    exchangeable = WaitingConfig(**{**asdict(anchor),
        "service_time": ((1, 1),) * 4, "restoration": ((.75, .75),) * 4})
    result.append(dict(candidate="exchangeable", selectable=False, config=asdict(exchangeable)))
    return result


def make_config(spec):
    raw = dict(spec["config"])
    for name in ("service_time", "restoration"):
        raw[name] = tuple(tuple(row) for row in raw[name])
    config = WaitingConfig(**raw)
    config.validate()
    return config


def subset_pool(c, state, tick, base):
    baseline = frontier.cheap(c, state, tick, base)
    subset = tuple(sorted(m for m, _ in baseline))
    allowed = frontier.old.screen.action_space(c, state)[2]
    if baseline not in allowed:
        raise RuntimeError("baseline is outside urgent matching roots")
    return (baseline, *(a for a in allowed if a != baseline
        and tuple(sorted(m for m, _ in a)) == subset))


def _validate_pool(c, state, pool):
    if not pool or len(set(pool)) != len(pool):
        raise ValueError("nonempty unique allocation pool required")
    subset = tuple(sorted(m for m, _ in pool[0]))
    for action in pool:
        validate_matching(c, state, action)
        if tuple(sorted(m for m, _ in action)) != subset:
            raise ValueError("allocation pool changes machine subset")
        if any(not (state.failed[m] or state.ages[m] >= c.failure_age - 1)
            for m, _ in action):
            raise ValueError("allocation pool starts an unurgent machine")


def reference_indices(c, state, tick, pool):
    """References share the exact subset; baseline wins every exact score tie."""
    _validate_pool(c, state, pool)
    h = c.horizon - tick
    if not 1 <= h <= c.horizon:
        raise ValueError("tick outside episode")
    quality = [sum(c.service_time[m][j] / c.restoration[m][j] for m, j in action)
        for action in pool]
    gains = dp.tables(c).gains
    economic = [sum(gains[h, m, state.ages[m], int(state.failed[m]),
        min(state.pending_wait[m], c.waiting_limit), j] for m, j in action)
        for action in pool]
    if not all(math.isfinite(x) for x in (*quality, *economic)):
        raise RuntimeError("nonfinite allocation reference score")
    return dict(baseline=0, quality=min(range(len(pool)), key=lambda i: (quality[i], i)),
        endpoint_dp=min(range(len(pool)), key=lambda i: (-economic[i], i)))


def completion_span(c, state, pool):
    """Spread of total actual post-service ages, including restoration rounding."""
    _validate_pool(c, state, pool)
    totals = [sum(math.floor(state.ages[m] * (1 - c.restoration[m][j]))
        for m, j in action) for action in pool]
    return float(max(totals) - min(totals))
