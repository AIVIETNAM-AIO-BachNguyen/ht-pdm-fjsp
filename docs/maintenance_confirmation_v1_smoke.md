# Maintenance confirmation v1 — macOS engineering evidence

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run; inline implementation and verification
- Origin Date: 2026-10-11
- Verification Status: VERIFIED for engineering smoke reproducibility only
- Version Label: maintenance_confirmation_v1
- Full scientific test: SEALED / not executed on Mac

## Locked scope

Only skill_mask:H48, frozen deadline_matching versus fixed_tail. Fresh 256
physical profiles x 16 shocks x two controllers: 8,192 episodes and 491,520
physical decisions. Historical 181 distinct physical signatures excluded;
all old/current raw seed panels and own smoke/full panels are disjoint.
Full profile construction was checked as METADATA only (256 distinct profiles,
maximum deterministic rejection attempt4); no full test outcome was opened.
Four simultaneous approximate Bonferroni 98.75% bootstrap intervals determine
GO / NO_GO / INCONCLUSIVE; advance to training only on GO. See the locked plan.

## Final checks

- Whole repository: **605 passed, 3 warnings, 199.21 seconds**.
- Focused predecessor/new experiment suite: 32 passed in 4.76 seconds before
  the final descriptive service win/loss correction; the final full suite
  includes the correction and its regression assertion.
- CLI smoke: **COMPLETED, audit PASS**, exit0; tqdm evaluation visible.
- Smoke grid: 8 episodes, 96 base + 24 tail = 120 decisions.
- Smoke settings: H12/L3/S8, two profiles x two shocks x two controllers;
  both controllers use common physical tapes. Independent physical replay,
  request/carryover reconstruction, scalar/batch reconciliation and fixed
  allocation checks pass in the tests.
- Exact reproducibility: decisions/requests/carryover/coordination/time profiles
  and profile differences match across repeated smoke runs; episode values
  match excluding observed latency. Complete trajectories/actions/costs match
  with the singleton shortcut disabled and the inherited boundary episode.
- Full-size H48/L12/S128 forecast checked only on a smoke engineering state.
- Artifact hashes: 42 hash-listed files,
  2,217,094 bytes; all verified.
- Source/config/locked plan snapshots match final working files byte for byte.
- Singleton shortcut: 20 forecasts skipped; 14,000 executed scenario
  ticks versus 15,488 configured without shortcut in THIS smoke.
  No uncomputed score is invented. These work counts are not a full-run speedup.
- Summary scientific decision: ENGINEERING_ONLY;
  full_test_outcomes_opened=false. Smoke outcome values do not establish GO.
- Nonempty-output rejection, interrupted FAILED journaling, full clean-code
  and CPU thread preflight, malformed-grid rejection and statistics tests pass.

Run: maintenance_confirmation_v1_smoke_20261011T041556Z

Local copy: artifacts/maintenance_confirmation_v1_smoke_20261011T041556Z

This smoke precedes the experiment commit: its manifest honestly records the
parent commit 54d17f9ce9c195c59ec6745868ae9b3450c0b1c5 with dirty=True. The immutable snapshots
are the reviewed new sources. The final commit must contain those same bytes;
the full runner records its own clean commit and rejects dirty code.

## Handoff

Full evaluation is handed to the user on bachng@100.111.83.52, repository
~/ht-pdm-fjsp, branch codex/maintenance-confirmation-v1. CPU threads1,
new UTC run directory, tqdm, streaming diagnostics and adjacent console log.
Planning estimate ~2–3h, ~0.4–0.6GB artifacts; neither is guaranteed. No GPU,
training or checkpoint needed. Keep the lab terminal open. No automatic retry
or resume; incomplete runs remain FAILED and do not count as a scientific result.
The assistant did not SSH, launch the full test or start rsync. User handoff
commands follow docs/experiment_workflow.md. Full results can remain inconclusive
at the fixed budget; do not relax margins or add seeds until they pass.
