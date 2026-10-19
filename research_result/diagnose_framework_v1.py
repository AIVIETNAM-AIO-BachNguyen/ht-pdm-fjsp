"""Exploratory selected-policy action trace on already opened v1 test seeds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_framework_v1 import CASES, FrameworkEnv, configs
from ht_pdm_fjsp.maintenance_request_policies import risk_matching
from ht_pdm_fjsp.maintenance_requests import legal_matchings


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(campaign: Path, output: Path, checkpoint_override: Path | None = None):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((campaign / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED" and manifest["profile"] == "full"
    selection = json.loads((campaign / "selection.json").read_text())
    registry = json.loads((campaign / "seed_registry.json").read_text())
    checkpoint = checkpoint_override.resolve() if checkpoint_override is not None else campaign / selection["checkpoint"]
    model = MaskablePPO.load(checkpoint, device="cpu")
    cases = configs()
    seeds = range(registry["sealed_test"]["start"], registry["sealed_test"]["start"] + registry["sealed_test"]["count"])
    grid = [(case, seed) for case in CASES for seed in seeds]
    accum = defaultdict(lambda: defaultdict(list))
    fields = ("case", "seed", "tick", "legal_count", "queued_pm", "queued_cm", "selected_jobs",
              "wait_with_opportunity", "extendable_action", "pm_starts", "cm_starts", "mean_duration",
              "slower_edge_count", "risk_agree")
    trace = output / "trace.csv"
    with trace.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case, seed in tqdm(grid, desc="Opened-panel diagnostic", unit="episode"):
            config = cases[case]
            env = FrameworkEnv(config)
            env.reset(seed=seed)
            episode = defaultdict(list)
            for tick in range(48):
                _, admitted = env._admitted()
                legal = legal_matchings(config, admitted)
                obs = env._observation()
                mask = env.action_masks()
                index = int(model.predict(obs, action_masks=mask, deterministic=True)[0])
                action = env.actions[index]
                assert action in legal and mask[index]
                queued = [r for r in admitted.requests if r is not None and r.started_at is None]
                durations = [config.service_time[m][j] for m, j in action]
                slower = sum(config.service_time[m][j] > min(config.service_time[m][other]
                    for other in range(config.technicians) if admitted.physical.remaining[other] == 0
                    and config.service_time[m][other] > 0) for m, j in action)
                row = {"case": case, "seed": seed, "tick": tick, "legal_count": len(legal),
                       "queued_pm": sum(r.kind == "preventive" for r in queued),
                       "queued_cm": sum(r.kind == "corrective" for r in queued),
                       "selected_jobs": len(action),
                       "wait_with_opportunity": int(not action and len(legal) > 1),
                       "extendable_action": int(any(set(action) < set(candidate) for candidate in legal)),
                       "pm_starts": sum(not admitted.physical.failed[m] for m, _ in action),
                       "cm_starts": sum(admitted.physical.failed[m] for m, _ in action),
                       "mean_duration": mean(durations) if durations else 0,
                       "slower_edge_count": slower,
                       "risk_agree": int(action == risk_matching(config, admitted, tick))}
                writer.writerow(row)
                for key in fields[3:]:
                    episode[key].append(float(row[key]))
                env.step(index)
            metrics = env.episode_metrics()
            for key, values in episode.items():
                accum[case][key].extend(values)
            for key in ("objective", "H_availability", "H_failures", "queued_cm_ticks", "H_pending"):
                accum[case][key].append(float(metrics[key]))
    summary = {case: {key: {"mean_per_decision_or_episode": mean(values), "sum": sum(values),
                             "count": len(values)} for key, values in metrics.items()}
               for case, metrics in accum.items()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "manifest.json").write_text(json.dumps({"status": "COMPLETED", "label": "exploratory opened-panel diagnostic",
        "source_run": str(campaign), "source_commit": manifest["git"]["commit"],
        "diagnosed_checkpoint": str(checkpoint), "diagnosed_checkpoint_sha256": digest(checkpoint),
        "episodes": len(grid), "decisions": len(grid) * 48,
        "artifact_sha256": {"trace.csv": digest(trace), "summary.json": digest(output / "summary.json")}}, indent=2) + "\n")
    print(json.dumps({"status": "COMPLETED", "output_dir": str(output), "episodes": len(grid)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.output_dir.resolve(), args.checkpoint)


if __name__ == "__main__":
    main()
