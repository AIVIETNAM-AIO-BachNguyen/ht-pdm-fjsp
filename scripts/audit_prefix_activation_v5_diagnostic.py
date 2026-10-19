"""Independent, read-only artifact audit for the exploratory B5 DEV diagnosis.

This checks saved evidence and arithmetic; it does not rerun the policies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import torch
from pathlib import Path
from statistics import mean


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def same_number(actual: object, expected: object) -> None:
    a, b = float(actual), float(expected)
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10), (a, b)


def aggregate(panel: list[dict[str, str]], n: int) -> dict:
    return {
        "episodes": len(panel),
        "cost_per_machine": {
            name: mean(float(r[f"{name}_cost"]) for r in panel) / n
            for name in ("full", "residual_off", "mask")
        },
        **{
            field: sum(int(r[field]) for r in panel)
            for field in (
                "residual_off_changed_states",
                "residual_off_slot_changes",
                "nonzero_residual_stages",
                "mask_checks",
            )
        },
    }


def compare(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected), (set(actual), set(expected))
    for field, value in expected.items():
        if field == "cost_per_machine":
            assert set(actual[field]) == set(value)
            for name, cost in value.items():
                same_number(actual[field][name], cost)
        else:
            assert actual[field] == value, (field, actual[field], value)


def audit(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["classification"] == "EXPLORATORY_DEV_DIAGNOSTIC"
    assert manifest["profile"] == "full"
    assert manifest["test_opened"] is False
    for name, binding in manifest["output_bindings"].items():
        file = path / name
        assert file.stat().st_size == binding["bytes"], name
        assert digest(file) == binding["sha256"], name
    assert digest(path / "diagnose_prefix_activation_v5.py") == manifest["script_sha256"]
    assert digest(path / "diagnostic_plan.md") == manifest["plan_sha256"]

    source = Path(manifest["source_artifact"])
    independent_path = Path(manifest["independent_audit_artifact"])
    assert digest(source / "manifest.json") == manifest["source_manifest_sha256"]
    assert digest(independent_path / "audit.json") == manifest["independent_audit_sha256"]
    source_manifest = json.loads((source / "manifest.json").read_text())
    independent = json.loads((independent_path / "audit.json").read_text())
    assert source_manifest["status"] == "COMPLETED"
    assert source_manifest["protocol"] == "technician_prefix_activation_v5"
    assert source_manifest["test_opened"] is False
    assert independent["status"] == "PASS"
    assert independent["verdict"] == "DEV_GATE_FAIL"
    assert independent["test_opened"] is False
    assert independent["source_manifest_sha256"] == manifest["source_manifest_sha256"]
    assert independent["source_artifact"] == str(source)
    assert independent["verifier_sha256"] == digest(independent_path / "audit_prefix_activation_development.py")
    config = json.loads((source / "resolved_config.json").read_text())
    assert not config["test_seeds"]
    scales = config["scales"]
    trains = config["train_seeds"]
    seeds = config["development_seeds"]
    assert scales == [4, 8] and len(trains) == 3 and len(seeds) == 32

    prior = {
        (int(r["n"]), r["method"], int(r["train_seed"]), int(r["seed"]), int(r["checkpoint_steps"])): r
        for r in rows(source / "development_episodes.csv")
        if r["method"] in ("reservation_ppo", "mask_only_ppo")
    }
    training = json.loads((source / "training_audit.json").read_text())
    training_index = {(r["n"], r["algorithm"], r["train_seed"]): r for r in training}
    branch = config["env_steps"] // 2
    assert branch == 245760
    branch_digests = {}
    for n in scales:
        for train in trains:
            digests = []
            for arm in ("reservation_ppo", "mask_only_ppo"):
                checkpoint = source / f"N{n}" / arm / str(train) / "checkpoints" / f"step_{branch:09d}.pt"
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                assert payload["parameter_sha256"] == training_index[n, arm, train]["precursor_parameter_sha256"]
                digests.append(payload["parameter_sha256"])
            assert digests[0] == digests[1]
            branch_digests[n, train] = digests[0]
    episodes = rows(path / "episodes.csv")
    expected_keys = {(n, train, seed) for n in scales for train in trains for seed in seeds}
    actual_keys = {(int(r["n"]), int(r["train_seed"]), int(r["seed"])) for r in episodes}
    assert len(episodes) == len(expected_keys) == manifest["episodes"] == 192
    assert actual_keys == expected_keys

    for row in episodes:
        n, train, seed = (int(row[k]) for k in ("n", "train_seed", "seed"))
        assert int(row["checkpoint_steps"]) == branch
        assert row["branch_parameter_sha256"] == branch_digests[n, train]
        for kind,arm in (("full", "reservation_ppo"),("mask", "mask_only_ppo")):
            old = prior[n, arm, train, seed, branch]
            assert row["population_sha256"] == old["population_sha256"]
            same_number(row[f"{kind}_cost"], old["extended_cost"])
            same_number(row[f"{kind}_availability"], old["availability"])
            same_number(row[f"{kind}_pending"], float(old["terminal_pending"]) / n)
            same_number(row[f"{kind}_failures"], float(old["policy_failures"]) / n)
            assert float(old["invalid_assignments"]) == 0
        assert int(row["mask_checks"]) == 48 * n // 2
        for kind in ("residual_off",):
            states = int(row[f"{kind}_changed_states"])
            slots = int(row[f"{kind}_slot_changes"])
            assert 0 <= states <= 48 and states <= slots <= 48 * n
        for field in ("nonzero_residual_stages",):
            assert 0 <= int(row[field]) <= int(row["mask_checks"])
        for kind in ("full", "residual_off", "mask"):
            for metric in ("cost", "availability", "pending", "failures"):
                assert math.isfinite(float(row[f"{kind}_{metric}"]))

    summary = json.loads((path / "summary.json").read_text())
    assert summary["classification"] == "EXPLORATORY_DEV_DIAGNOSTIC"
    assert summary["same_weight_interventions_only"] is True
    assert summary["scientific_matched_control_verdict"] == "DEV_GATE_FAIL"
    assert set(summary["by_scale"]) == set(map(str, scales))
    for n in scales:
        panel = [r for r in episodes if int(r["n"]) == n]
        recorded = summary["by_scale"][str(n)]
        compare({k: v for k, v in recorded.items() if k != "per_replica"}, aggregate(panel, n))
        assert set(recorded["per_replica"]) == set(map(str, trains))
        for train in trains:
            part = [r for r in panel if int(r["train_seed"]) == train]
            compare(recorded["per_replica"][str(train)], aggregate(part, n))

    return {
        "status": "PASS",
        "scope": "independent saved-artifact/hash/panel/arithmetic audit; no policy rerollout",
        "classification": "EXPLORATORY_DEV_DIAGNOSTIC",
        "scientific_matched_control_verdict": "DEV_GATE_FAIL",
        "test_opened": False,
        "artifact": str(path),
        "manifest_sha256": digest(path / "manifest.json"),
        "episodes": len(episodes),
        "source_dev_audit_sha256": manifest["independent_audit_sha256"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.artifact.resolve()), indent=2, sort_keys=True))
