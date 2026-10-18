"""Read-only phase and mechanism audit for the frozen coupled follow-up run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def load_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run, out = args.run_dir, args.output_dir
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    manifest = json.loads((run / "manifest.json").read_text())
    integrity = json.loads((run / "integrity.json").read_text())
    if (manifest["status"], manifest["integrity"], integrity["status"]) != ("COMPLETED", "PASS", "PASS"):
        raise SystemExit("Full completion/integrity gate failed")
    if (manifest["completed_episodes"] != manifest["expected_episodes"] or
            manifest["expected_episodes"] != 1440):
        raise SystemExit("Unexpected episode count")
    if not manifest["selection_utc"] < manifest["test_opened_utc"]:
        raise SystemExit("Test opened before baseline selection")
    for filename, expected in manifest["source_sha256"].items():
        actual = hashlib.sha256((run / "source_snapshot" / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Source hash mismatch: {filename}")
    summary = json.loads((run / "summary.json").read_text())
    rows = load_csv(run / "episodes.csv")
    tests = [r for r in rows if r["split"] == "sealed_test"]
    configs = summary["test_configs"]
    seeds = [str(s) for s in summary["test_shock_seeds"]]
    baseline = summary["selected_simple"]
    expected = {(c, s, t, d) for c in configs for s in seeds
                for t in ("threshold", "capacity_aware")
                for d in ("fcfs", "risk", "myopic", "isolated", "coupled_rollout")}
    indexed = {(r["condition"], r["seed"], r["timing"], r["dispatcher"]): r for r in tests}
    if len(tests) != len(expected) or set(indexed) != expected:
        raise SystemExit("Test panel incomplete")

    def mean_delta(left, right, metric, config=None):
        cs = [config] if config else configs
        return statistics.fmean(float(indexed[c, s, *left][metric]) -
                                float(indexed[c, s, *right][metric])
                                for c in cs for s in seeds)

    contrasts = {
        "A_allocation": (("capacity_aware", "coupled_rollout"), ("capacity_aware", baseline)),
        "B_timing": (("capacity_aware", baseline), ("threshold", baseline)),
        "threshold_allocation": (("threshold", "coupled_rollout"), ("threshold", baseline)),
        "fcfs_vs_selected_capacity": (("capacity_aware", "fcfs"), ("capacity_aware", baseline)),
    }
    phase_metrics = ("cost", "policy_cost", "tail_cost", "terminal_liability",
                     "maintenance_cost", "unavailability_cost", "failure_cost",
                     "waiting_cost", "availability", "failures", "failed_waiting_ticks",
                     "overdue_ticks", "pending_h", "pending_end", "active_end",
                     "requests", "starts", "completions", "request_wait_mean", "utilization")
    effects = {name: {
        "overall": {metric: mean_delta(left, right, metric) for metric in phase_metrics},
        "by_config": {c: {metric: mean_delta(left, right, metric, c)
                          for metric in ("cost", "policy_cost", "tail_cost", "terminal_liability")}
                      for c in configs},
    } for name, (left, right) in contrasts.items()}
    trace = {(r["condition"], r["seed"], r["tick"], r["timing"]): r
             for r in load_csv(run / "decisions.csv") if r["dispatcher"] == baseline}
    trace_differences = {}
    for phase, ticks in (("H", range(48)), ("cleanup", range(48, 60))):
        trace_differences[phase] = {field: sum(
            trace[c, s, str(t), "threshold"][field] !=
            trace[c, s, str(t), "capacity_aware"][field]
            for c in configs for s in seeds for t in ticks)
            for field in ("proposals", "pending_before", "assignment", "state", "cost")}
    diagnostics = {}
    for timing in ("threshold", "capacity_aware"):
        selected = [r for r in load_csv(run / "forecast_diagnostics.csv") if r["timing"] == timing]
        changed = [r for r in selected if r["chosen"] != r["baseline"]]
        diagnostics[timing] = {
            "states": len(selected), "changed": len(changed),
            "decision_mean_delta_on_changed": statistics.fmean(
                float(r["decision_estimate_delta"]) for r in changed),
            "independent_mean_delta_on_changed": statistics.fmean(
                float(r["independent_mean_delta"]) for r in changed),
            "independent_favorable": sum(float(r["independent_mean_delta"]) < 0 for r in changed),
            "independent_worse": sum(float(r["independent_mean_delta"]) > 0 for r in changed),
        }
    result = {
        "status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "full_run": str(run), "full_git_commit": manifest["git_commit"],
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes_verified": len(manifest["source_sha256"]),
        "test_episodes": len(tests), "test_configurations": configs,
        "test_shock_seeds_per_configuration": len(seeds),
        "selected_baseline": baseline, "effects": effects,
        "timing_trace_differences": trace_differences, "diagnostics": diagnostics,
    }
    out.mkdir(parents=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
