## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: authorized engineering execution and verification.
- Origin Date: 2026-10-12.
- Verification Status: VERIFIED for complete tests and Mac smoke; full science UNVERIFIED.
- Version Label: maintenance_myopic_lookahead_v1.

# Engineering gate: PASS

Protocol was saved before the new experiment ran. Candidate, method definitions,
scenario counts, forecast horizon, seed panels, CIs, margins and stopping budget
were not tuned against smoke outcomes. This follows the user's instruction to
keep overlap3__fast1_slow325 exploratory and compare matching with lookahead on
fresh data before any additional setting sweep.

Branch: `codex/maintenance-myopic-lookahead-v1`, isolated checkout
`/private/tmp/marl-maintenance-assignment-confidence-v1`. No changes to existing
v1-v3 code, pyproject.toml, uv.lock or the primary workspace's user changes.

## Tests and dependencies

No new dependencies; `uv sync --frozen --extra dev --offline` passed (33
packages checked, 7ms) using the writable temporary cache, with no environment
changes. The inherited frozen environment was used. New module is run
with the Python module CLI. `uv run --no-sync` needed a writable temporary cache
because the default user cache is outside the sandbox; the failed cache
initialization did not start an episode. No experiment crash or automatic full
retry occurred.

- New targeted suite: **17 passed**, 2.28s.
- Complete suite: **817 passed**, 256.90s; one existing SB3 rich/tqdm
  experimental warning in the old parallel-maintenance smoke test.
- Targeted log: `/private/tmp/maintenance_myopic_lookahead_v1_targeted_tests.log`.
- Complete log: `/private/tmp/maintenance_myopic_lookahead_v1_all_tests.log`.

Checks cover the fixed mask/environment, exactly 17 maximum roots, scalar vs
vector forecasts for probabilities 0/.45/1 including waiting and completion,
initial busy technician, unchanged baseline on reachable states, current-state
only/no real future access, same-subset restriction, baseline wins ties,
single-root/common-tail shortcut, independent selection/holdout bank (including
negative heldout gain retained), fresh seeds/roles, paired bootstrap ratio of
means, cost gains rejected when health guards fail, synthetic full classification,
timeout recording, dirty full refusal, complete artifacts, forecast regeneration,
replay tampering and overwrite refusal. Synthetic arithmetic under a full
configuration does not step any full physical episode or open its outcomes.

## Actual macOS smoke

```bash
UV_CACHE_DIR=/private/tmp/marl-myopic-lookahead-uv-cache \
uv run --no-sync python -m ht_pdm_fjsp.maintenance_myopic_lookahead \
  --profile smoke --device cpu \
  --output-dir artifacts/maintenance_myopic_lookahead_v1_smoke_20261012T044908Z
```

Working directory is the isolated checkout above. Tqdm appears for paired
multi-seed evaluation, independent-bank common-state diagnostics and forecast
audit; the adjacent timestamped console log is preserved.

- COMPLETED / audit PASS, elapsed **1.565s**.
- **12 episodes / 720 physical transitions / 28 common-state landmarks**.
- Replay every physical decision/metric/cost and regenerate forecasts for all
  four smoke seeds and every landmark; 89 root rows checked.
- Actual forecast workload **87,168 transitions**, plus 87,168 separately
  recorded replay transitions, below the locked evaluation cap 887,040.
- Invalid assignments 0; paired initial states/shock tapes, full episode/root
  grids, costs and all source/protocol seals pass.
- 136 unique raw IDs, 7,394 derived role IDs; inherited raw registries and 606
  v1/v2/v3 screen derived IDs checked.
- Independently verified all **21 artifact SHA256/size bindings**; payload
  **1,222,288 bytes**. Source is uncommitted during smoke and bound by snapshot;
  full execution requires a clean committed checkout.
- Scientific status **ENGINEERING_ONLY**, full evaluation outcomes unopened.
  No smoke performance is used to select methods/setting or change protocol.
- Training/model checkpoint save-load: NOT_APPLICABLE.

## Full handoff scope

Exactly **384 paired physical episodes /23,040 transitions**, 896 common-state
landmarks, with 128 scenarios per 12-tick forecast. Full raw evaluation panel
36000000..36000127 is fresh and unopened. Conservative evaluation forecast cap
454,164,480 transitions; actual workload plus audit workload recorded separately.
Fixed timeout 7,200s, no outcome stopping, adaptive scenarios, fallback or resume.
Expect roughly 35–50MB of artifacts plus console log, depending on root count
and JSON field widths; this is an engineering estimate, not an observed full run.

Commit/push these five verified files and give the
user one exact compound lab command and the canonical Mac rsync command. Do not
SSH to bachng@100.111.83.52, launch full or transfer. Each run uses a new UTC
output directory and rejects an existing nonempty directory.
