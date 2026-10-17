"""Exploratory service-duration stress test using the frozen v1 mechanics."""

import argparse
from pathlib import Path

from ht_pdm_fjsp import maintenance_shared_technician_factorial as base
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig

VERSION = "maintenance_shared_technician_exploratory_v2"
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs" / f"{VERSION}_protocol.md"
REGISTRY = ROOT / "configs" / f"{VERSION}_seed_registry.json"
CONDITIONS = ("overlap_service2", "balanced_service2")


def config(condition):
    if condition == "overlap_service2":
        durations = ((2, 0), (2, 3), (2, 4), (2, 5))
    elif condition == "balanced_service2":
        durations = ((2, 0), (0, 2), (2, 4), (4, 2))
    else:
        raise ValueError(condition)
    c = WaitingConfig(4, 2, base.H + base.TAIL, 4, .45, durations,
        tuple(tuple(.75 for _ in row) for row in durations),
        waiting_limit=4, waiting_price=12.)
    c.validate()
    return c


def run(profile, output_dir):
    return base.run(profile, output_dir, version=VERSION, protocol=PROTOCOL,
        registry=REGISTRY, conditions=CONDITIONS, config_fn=config,
        analysis_seed=93000400, confidence=.95, extra_sources=(Path(__file__),))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(run(args.profile, args.output_dir))


if __name__ == "__main__":
    main()
