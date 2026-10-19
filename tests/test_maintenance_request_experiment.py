import csv
import json

import pytest

from ht_pdm_fjsp import maintenance_request_experiment as experiment
from ht_pdm_fjsp.maintenance_request_policies import family_configs


def test_locked_settings_family_and_raw_seed_audit():
    assert len(family_configs("full")) == 9
    assert len(family_configs("smoke")) == 3
    assert experiment.seed_audit()["status"] == "PASS"
    assert experiment.registered_integers({"eval_start": 20, "eval_count": 3}) >= {20, 21, 22}
    assert not set(experiment.settings("full")["seeds"]) & set(experiment.settings("smoke")["seeds"])


def test_cli_smoke_artifact_replay_and_overwrite_refusal(tmp_path):
    out = tmp_path / "smoke"
    experiment.run("smoke", out)
    manifest = json.loads((out / "manifest.json").read_text())
    audit = json.loads((out / "audit.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert audit["episodes"] == 36 and audit["physical_transitions"] == 720
    assert audit["forecast_reconstruction"] == "ALL"
    assert (out / "episodes.csv").read_bytes() == (out / "episodes.partial.csv").read_bytes()
    assert len(list(csv.DictReader((out / "episodes.csv").open()))) == 36
    for name, expected in manifest["artifact_hashes"].items():
        assert experiment.digest(out / name) == expected
    with pytest.raises(FileExistsError, match="overwrite"):
        experiment.run("smoke", out)
    # Reconstruct typed episode rows and reject a corrupted realized shock.
    rows = []
    for row in csv.DictReader((out / "episodes.csv").open()):
        rows.append({k: (v if k in ("cell", "requester", "dispatcher") else
                        v == "True" if k == "completed_wait_defined" else
                        int(v) if k in ("seed", "physical_seed") else float(v))
                     for k, v in row.items()})
    journal = out / "decisions.jsonl"
    lines = journal.read_text().splitlines()
    first = json.loads(lines[0])
    first["events"][0] = not first["events"][0]
    lines[0] = json.dumps(first)
    journal.write_text("\n".join(lines) + "\n")
    with pytest.raises(AssertionError, match="shock tape"):
        experiment.audit_run(out, experiment.settings("smoke"), family_configs("smoke"), rows, lambda: None)


def test_timeout_writes_failed_manifest(tmp_path, monkeypatch):
    original = experiment.settings
    monkeypatch.setattr(experiment, "settings", lambda profile: {**original(profile), "max_seconds": -1})
    with pytest.raises(TimeoutError):
        experiment.run("smoke", tmp_path / "timeout")
    manifest = json.loads((tmp_path / "timeout" / "manifest.json").read_text())
    assert manifest["status"] == "FAILED" and "TimeoutError" in manifest["error"]


def test_full_run_is_rejected_with_dirty_source(tmp_path, monkeypatch):
    original = experiment.subprocess.check_output
    def fake(command, **kwargs):
        return " M changed.py\n" if command[:2] == ["git", "status"] else original(command, **kwargs)
    monkeypatch.setattr(experiment.subprocess, "check_output", fake)
    with pytest.raises(RuntimeError, match="clean"):
        experiment.run("full", tmp_path / "full")
