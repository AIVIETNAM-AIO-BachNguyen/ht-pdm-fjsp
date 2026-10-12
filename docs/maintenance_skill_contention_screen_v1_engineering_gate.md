## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: engineering validation.
- Origin Date: 2026-10-12.
- Verification Status: VERIFIED for the bounded Mac smoke and tests;
  full screening/confirmation NOT_RUN.
- Version Label: maintenance_skill_contention_screen_v1_engineering_gate.

# Engineering handoff

Branch: `codex/maintenance-skill-contention-screen-v1`.
Base revision: `490f8fb11b7175d40b76f32f506930a12e34c20b`.
Verified changes: the new skill-screen runner, its seed registry, protocol,
11 tests and CLI entry point. No existing dynamics, PPO policy or RL training
implementation changed. Existing user changes in the primary checkout untouched.

Checks:

- `.venv/bin/python -m pytest -q tests/test_maintenance_skill_contention_screen.py`:
  11 passed,2.04s.
- `.venv/bin/python -m pytest -q`:767 passed,225.21s;
  one existing `TqdmExperimentalWarning` from the rich callback.
- `UV_CACHE_DIR=/private/tmp/marl-skill-screen-uv-cache uv lock --check --offline`:
  PASS,72 packages resolved; no lockfile change.
- CLI help and `git diff --check`:PASS.

Explicit local smoke:

```bash
UV_CACHE_DIR=/private/tmp/marl-skill-screen-uv-cache \
uv run --frozen --no-sync python -m ht_pdm_fjsp.maintenance_skill_contention_screen \
  --profile smoke --device cpu \
  --output-dir artifacts/maintenance_skill_contention_screen_v1_smoke_20261012T030356Z
```

COMPLETED/audit PASS in1.708s.25 fixed settings,200 screening +32 confirmation
episodes,13,920 physical transitions. All13,920 logged transitions replayed from
fresh seed tapes and states; policy actions, costs and physical continuity matched.
Executed invalid assignments=0. Selection SHA256 unchanged across confirmation.
15 hashed files,12,762,368 bytes; hashes and sizes independently re-read after exit.
tqdm was visible in both phases. Raw smoke artifacts preserved.

Smoke's scientific gate is **NOT_CONFIRMED** with only4 confirmation seeds.
This is not an engineering failure and is not a full-run result. No setting is
promoted from smoke; cutoffs/seeds/candidates remain unchanged after seeing it.

Full budget:3,200 screening episodes and at most1,024 confirmation episodes;
at most253,440 physical transitions,6,960 audited/logged transitions,
20,000 bootstrap draws. Expected artifacts roughly8–15MB excluding console log;
runtime is a short CPU screen, with a locked2h timeout rather than a promised ETA.
No model training, no future-scenario label generation, no checkpoint prerequisite.

Full artifacts will bind the clean final commit. The smoke was intentionally
run before committing and records dirty=true/base revision; its source snapshots
and protocol bytes bind the tested code. Git commit is created only after these
gates pass. Full execution and rsync remain user-run on lab/Mac respectively.
