"""Supplemental read-only analysis of completed v1/v2 factorial artifacts."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from ht_pdm_fjsp.maintenance_dispatch import role_seed


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    integrity = json.loads((directory / "integrity.json").read_text())
    if manifest["status"] != "COMPLETED" or integrity["status"] != "PASS":
        raise RuntimeError(f"incomplete input: {directory}")
    if manifest["completed_episodes"] != manifest["expected_episodes"] != integrity["episodes"]:
        raise RuntimeError("episode count mismatch")
    episodes = list(csv.DictReader((directory / "episodes.csv").open(newline="")))
    if len(episodes) != manifest["expected_episodes"]:
        raise RuntimeError("episode CSV count mismatch")
    return directory, manifest, integrity, episodes


def exposure(directory, conditions, test_seeds):
    indexed = {}
    trace = directory / "decisions.csv"
    with trace.open(newline="") as stream:
        for row in csv.DictReader(stream):
            tick = int(row["tick"])
            if tick < 48:
                indexed[row["condition"], int(row["seed"]), row["dispatcher"], row["timing"], tick] = row
    result = {}
    for condition in conditions:
        result[condition] = {}
        for dispatcher in ("fcfs", "risk", "myopic", "lookahead"):
            proposal_ticks = pending_ticks = assignment_ticks = both_busy_ticks = 0
            proposal_episodes = assignment_episodes = 0
            for seed in test_seeds:
                changed_request = changed_assignment = False
                for tick in range(48):
                    a = indexed[condition, seed, dispatcher, "threshold", tick]
                    b = indexed[condition, seed, dispatcher, "capacity_aware", tick]
                    proposal = a["proposals"] != b["proposals"]
                    assignment = a["assignment"] != b["assignment"]
                    proposal_ticks += proposal
                    pending_ticks += a["pending_before"] != b["pending_before"]
                    assignment_ticks += assignment
                    both_busy_ticks += all(x > 0 for x in json.loads(a["state"])["remaining"])
                    changed_request |= proposal
                    changed_assignment |= assignment
                proposal_episodes += changed_request
                assignment_episodes += changed_assignment
            result[condition][dispatcher] = dict(policy_ticks=48 * len(test_seeds),
                proposal_diff_ticks=proposal_ticks, pending_diff_ticks=pending_ticks,
                assignment_diff_ticks=assignment_ticks, both_busy_ticks=both_busy_ticks,
                proposal_diff_episodes=proposal_episodes,
                assignment_diff_episodes=assignment_episodes)
    return result


def paired(episodes, conditions, test_seeds, version):
    rows = {(r["condition"], int(r["seed"]), r["timing"], r["dispatcher"]): r
            for r in episodes if r["split"] == "sealed_test"}
    if len(rows) != len(test_seeds) * len(conditions) * 12:
        raise RuntimeError("test panel key count mismatch")
    rng = np.random.default_rng(role_seed(91000400 if version == "v1" else 93000400,
                                              "supplemental_descriptive"))
    draws = rng.integers(0, len(test_seeds), (10000, len(test_seeds)))
    contrasts = {}
    definitions = {
        "threshold_fcfs_minus_reactive_fcfs": (("threshold", "fcfs"), ("reactive", "fcfs")),
        "capacity_aware_fcfs_minus_threshold_fcfs": (("capacity_aware", "fcfs"), ("threshold", "fcfs")),
        "capacity_aware_lookahead_minus_capacity_aware_fcfs": (("capacity_aware", "lookahead"), ("capacity_aware", "fcfs")),
    }
    for condition in conditions:
        contrasts[condition] = {}
        for name, (arm, reference) in definitions.items():
            result = {}
            for metric in ("cost", "policy_cost", "tail_cost", "terminal_liability", "availability",
                           "failures", "failed_waiting_ticks", "pending_h", "pending_end", "utilization"):
                vector = np.asarray([float(rows[condition, seed, *arm][metric])
                                     - float(rows[condition, seed, *reference][metric])
                                     for seed in test_seeds])
                result[metric] = {"mean_difference": float(vector.mean()),
                                  "ci_95_descriptive": np.quantile(vector[draws].mean(1), [.025, .975]).tolist()}
            contrasts[condition][name] = result
    return contrasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", required=True)
    parser.add_argument("--v2", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "COMPLETED",
               "classification": {"v1": "confirmatory", "v2": "exploratory"}, "runs": {},
               "analysis_source_sha256": digest(Path(__file__))}
    (output / "source_snapshot").mkdir()
    shutil.copy2(Path(__file__), output / "source_snapshot" / Path(__file__).name)
    for version, source, conditions in (
        ("v1", args.v1, ("scarce_overlap", "partitioned", "ample")),
        ("v2", args.v2, ("overlap_service2", "balanced_service2"))):
        directory, manifest, integrity, episodes = read_run(source)
        test_seeds = manifest["seeds"]["sealed_test"]
        if len(test_seeds) != 64 or set(test_seeds) & set(manifest["seeds"]["development"]):
            raise RuntimeError("seed split")
        summary["runs"][version] = {"input": str(directory), "commit": manifest["git_commit"],
            "manifest_sha256": digest(directory / "manifest.json"),
            "episodes_sha256": digest(directory / "episodes.csv"),
            "decisions_sha256": digest(directory / "decisions.csv"),
            "integrity": integrity, "exposure": exposure(directory, conditions, test_seeds),
            "supplemental_paired": paired(episodes, conditions, test_seeds, version)}
    if set(json.loads((Path(args.v1) / "manifest.json").read_text())["seeds"]["sealed_test"]) & set(
            json.loads((Path(args.v2) / "manifest.json").read_text())["seeds"]["sealed_test"]):
        raise RuntimeError("cross-study seed overlap")
    (output / "supplementary_analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
