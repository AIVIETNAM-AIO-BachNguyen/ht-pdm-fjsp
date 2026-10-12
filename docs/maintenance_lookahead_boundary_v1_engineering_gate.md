## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: authorized engineering implementation and local verification.
- Origin Date: 2026-10-12.
- Verification Status: tests/Mac smoke VERIFIED; full science UNVERIFIED.
- Version Label: maintenance_lookahead_boundary_v1.

# Engineering gate: PASS

The protocol and seed registry were written before any new physical run.
The fixed candidate, seven methods, scenario counts, horizons, primary scope,
interval family, margins and stopping rule were not tuned against smoke results.
No new full outcomes have been opened. This follows the user's authorization
to test a longer window and baseline continuation after the old window.

Branch codex/maintenance-lookahead-boundary-v1 in isolated checkout
/private/tmp/marl-maintenance-assignment-confidence-v1. Five new experiment
files only; previous runners, dependencies and unrelated primary workspace
changes are preserved. No new dependency or learned network.

## Tests and frozen dependencies

uv sync --frozen --extra dev --offline: PASS, 33 packages checked in 7ms.
Use UV_CACHE_DIR=/private/tmp/marl-myopic-lookahead-uv-cache because the default
cache is outside the sandbox. The inherited Python environment is unchanged.

- Targeted: 18 passed, 12.23s.
- Complete suite: 835 passed, 294.31s; one existing SB3 rich/tqdm experimental
  warning in the previous parallel-maintenance smoke test.
- Logs: /private/tmp/maintenance_lookahead_boundary_v1_targeted_tests.log and
  /private/tmp/maintenance_lookahead_boundary_v1_all_tests.log.

Checks cover independent scalar versus vector forecasts at probabilities
0/.45/1 through 60 ticks including busy workers, failure, waiting and completion;
identical random prefixes and exact old h12 scoring; terminal decomposition;
the prior periodic state as a regression fixture; same-subset restrictions and
all six current-state-only planners; common tail/single-root shortcut, ties,
heldout bank excluded from selection and retained negative gains; seed role
uniqueness; paired ratio of means and twelve-interval family; primary guards
and secondary separation; timeout recording and dirty-full refusal; complete
smoke artifacts, replay, tamper detection, hashing and overwrite refusal.
Full-profile synthetic arithmetic never steps a full physical episode.

## Real macOS CLI smoke

```bash
UV_CACHE_DIR=/private/tmp/marl-myopic-lookahead-uv-cache \
uv run --no-sync python -m ht_pdm_fjsp.maintenance_lookahead_boundary \
  --profile smoke --device cpu \
  --output-dir artifacts/maintenance_lookahead_boundary_v1_smoke_20261012T051736Z
```

The real CLI ran in the isolated checkout above, with timestamped adjacent
console log. Tqdm appeared for multi-seed physical evaluation, common-state
diagnostics and replayed forecast audit.

- COMPLETED / audit PASS, 6.958s; scientific status ENGINEERING_ONLY.
- 28 episodes, 1,680 physical transitions, 28 landmarks and 85 root rows.
- All physical decisions/metrics/costs replayed; forecasts regenerated for all
  four smoke seeds and every landmark. Paired state/shock tapes pass.
- 456,024 evaluation forecast transitions plus 456,024 replay transitions;
  evaluation cap4,928,448. Invalid assignments0, full grids/cost components pass.
- 136 unique raw IDs and 7,394 derived roles; 8,000 prior derived roles checked,
  including previous myopic-lookahead selection/holdout/physical/bootstrap roles.
- Independently checked all 23 SHA256/size bindings and source seals;
  payload2,846,036 bytes. Full-open flag false. Source/protocol/registries
  unchanged through execution. Smoke snapshots bind the precommit code bytes.
- No smoke-performance selection/tuning; model/checkpoint save-load N/A.

## Full handoff scope and limits

Exactly 896 paired physical episodes /53,760 physical transitions with 128
new evaluation seeds and 896 common landmarks. Six planning rules use 128
scenarios/root; h12, clipped h24 and remaining-to-tick60 scoring in two scopes.
Maximum17 legal roots. Conservative evaluation forecast cap2,523,365,376;
separate replay compute logged. Timeout7,200s, no outcome stopping or resume.
Expected roughly90–120MB artifacts plus console log; estimate only, no full run.

Primary is assignment_endpoint versus myopic with a 1% practical CI threshold
and availability/pending guards. Joint headroom and repair versus old h12 are
secondary; improvement against old h12 alone is insufficient for the primary.
Endpoint terminal value is explicit Monte Carlo myopic continuation, equivalent
to a remaining-horizon rollout, not a learned head or predicted future replanning.
The observed-state regression is engineering evidence, not a full scientific
result. Full forecast regeneration covers first two seeds only; every physical
transition is replayed, and all remaining root identities/constraints are checked.

The complete test gate passed. Commit/push these five files and give one
exact compound lab command and the canonical Mac rsync -avhP --partial command.
Do not SSH, launch full or transfer. Every run uses a fresh timestamp and rejects
a nonempty output directory.
