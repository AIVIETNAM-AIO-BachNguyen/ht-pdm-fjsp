"""Verify pinned completed PPO run without using its evaluation outcomes as labels."""
import hashlib
import json
from pathlib import Path
from ht_pdm_fjsp import maintenance_assignment_scoring as pilot
from ht_pdm_fjsp import maintenance_assignment_benchmark as previous
from ht_pdm_fjsp.maintenance_assignment_value_benchmark import ROOT,VERSION


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameters_sha(model):
    return hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def verify_source(source_dir,cfg,contract=None):
    source_dir=Path(source_dir)
    if contract is None:contract=json.loads((ROOT/f"configs/{VERSION}_sources.json").read_text())
    if contract["protocol"]!=VERSION or contract["source_protocol"]!=pilot.VERSION:raise ValueError("source contract")
    for name,expected in contract["files"].items():
        if sha(source_dir/name)!=expected:raise ValueError(f"source hash mismatch: {name}")
    m=json.loads((source_dir/"manifest.json").read_text())
    if m["status"]!="COMPLETED" or m["audit"]!="PASS" or m["dirty"] or m["commit"]!=contract["source_commit"] or m["settings"]!=previous.settings("full"):
        raise ValueError("source completion/commit/settings mismatch")
    for name,r in m["artifacts"].items():
        p=source_dir/name
        if not p.resolve().is_relative_to(source_dir.resolve()) or sha(p)!=r["sha256"] or p.stat().st_size!=r["bytes"]:
            raise ValueError(f"source artifact mismatch: {name}")
    for p in (source_dir/"source_snapshot").glob("*.py"):
        current=ROOT/"src/ht_pdm_fjsp"/p.name
        if current.exists() and sha(current)!=sha(p):raise ValueError(f"source code changed: {p.name}")
    selection=json.loads((source_dir/"checkpoint_selection.json").read_text())
    if not selection["locked_before_test"]:raise ValueError("source selection not locked")
    # All three pinned models are checked even in smoke; only requested replicas run.
    models={}
    for raw,pinned in contract["models"].items():
        seed=int(raw);s=selection["selections"][f"additive_ppo:{seed}"]
        if s["selected_checkpoint"]!=pinned["path"] or s["update"]!=pinned["update"] or s["learned_steps"]!=pinned["learned_steps"] or pilot.choose_checkpoint(s["candidates"])["update"]!=pinned["update"]:
            raise ValueError("source selection mismatch")
        if sha(source_dir/pinned["path"])!=pinned["sha256"]:raise ValueError("source checkpoint hash")
        model,p=pilot.load_checkpoint(source_dir/pinned["path"])
        if p["algorithm"]!="additive_ppo" or p["train_seed"]!=seed or p["update"]!=pinned["update"] or p["learned_steps"]!=pinned["learned_steps"] or p["settings"]!=previous.settings("full"):
            raise ValueError("source checkpoint identity")
        model.requires_grad_(False);models[seed]=model
    return {s:models[s] for s in cfg["model_seeds"]},contract,dict(status="PASS",verified_source_artifacts=len(m["artifacts"]),all_pinned_models_verified=len(models),used_models=len(cfg["model_seeds"]))
