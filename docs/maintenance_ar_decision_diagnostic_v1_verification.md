## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent.
- Origin Mode: run, inline.
- Origin Date: 2026-10-11.
- Verification Status: ENGINEERING VERIFIED; new full diagnostic outcomes NOT OPENED.
- Version Label: maintenance_ar_decision_diagnostic_v1_verification_v1.
- Overall Confidence: implementation gates passed; no scientific result claimed.

## Verified implementation

The branch codex/maintenance-ar-decision-diagnostic-v1 adds two independent
modules, a diagnostic test suite, a prospective plan and two pinned contracts.
Existing training, policies, transitions and older experiments are unchanged.
No new training or checkpoint selection is performed. Every original CM edge
is preserved. The root menu includes partial/all PM deferral, PM technician
swaps holding chosen machines fixed, and their feasible combinations.

Root choices are made from selection32 only, before evaluation64 is generated.
Negative evaluation gains are retained. All alternatives use common phase
tapes, frozen greedy RL to tick47, and the shared deadline rule at48..59.
The source matching probability sums all possible ordered pair/STOP sequences,
rather than treating one sequence as a whole matching probability.

## Tests

Command: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m pytest -q

694 passed, one existing TqdmExperimentalWarning, in237.10 seconds. The14 new
tests separately passed in6.98 seconds. They exercise candidate completeness
under skills and failed machines, CM preservation, single-tick deferral,
scalar/vector features and decisions including tied logits, original clocks
at0/42/47, every scalar forecast transition and boundary, probability mass over
ordered sequences, selection/evaluation independence, negative gains, empty
exposures and incomplete bootstrap grids, fresh physical exclusions, source
checkpoint tampering, copied checkpoints and replayable NPZ artifacts.
Syntax compilation and git diff --check passed. No unrelated files were edited.

## Local macOS CLI smoke

Run: artifacts/maintenance_ar_decision_diagnostic_v1_smoke_cpu_20261011T121106Z.
Status COMPLETED, audit PASS, elapsed6.153529 seconds. The smoke used one pinned
trained source replica, fresh smoke profile/shock, both conditions, landmarks
0 and42, two independent futures per phase, and the actual H48+12 clock.

Actual counts:2 trajectories,96 trajectory ticks,4 states,14 unique roots,
2856 forecast transitions,672 tail transitions,2128 RL continuation decisions,
62 sequence score rows and10 prefix rows. Every candidate/future was replayed
scalarly: costs, decisions, economic components, cutoff48 and terminal60 match.
All eight phase rollout audits record scalar_replay=PASS. No near-tie fallback
was needed in this CLI smoke; the dedicated tied-score test verifies fallback.
All188 artifact hashes/sizes were rechecked independently;5,205,661 bytes.
Copied source parameters and checkpoint bytes are unchanged.

Pinned source run: maintenance_autoregressive_pilot_v1_full_cpu_20261011T112826Z,
commit01f721b990ba090b10330197248a13795f0205bc. All238 source artifacts were verified
against the pinned manifest, as were unchanged environment/policy source bytes.
Source outcomes were previously analyzed; this is explicitly exploratory.
Physical exclusions include59,941 historical/pilot signatures. New full/smoke
metadata resolves26 signatures with no outcome filtering or overlap. Metadata
resolution does not execute new full states or forecasts.

## Full-size engineering timing, smoke metadata only

artifacts/maintenance_ar_decision_diagnostic_v1_engineering_20261011T121251Z
contains runtime.json. With the same frozen source checkpoint and smoke profile,
seven candidate roots x(32 selection +64 evaluation futures) took0.660500 seconds
at t=0 (40,320 transitions), and0.066936 seconds at t=42 (12,096 transitions).
This tests the real batch dimensions; no new full-profile outcomes were opened.
Runtime is hardware/state dependent and these two samples are not a lab ETA.
No performance effect sizes from engineering/smoke data are used scientifically.

## Lab handoff

Fixed full scope:3 source models,12 fresh profiles,2 physical shocks,2 skill
conditions,5 landmarks;144 trajectories,6912 trajectory ticks,720 states.
Maximum19,740,672 forecast transitions; actual roots depend on feasible PM.
Expected disk use: source snapshot plus copied checkpoints and720 compressed
forecast files, likely tens of MB; this is an estimate, not measured full size.
Each run has a fresh UTC directory, source/seed/clock/frozen-parameter audits,
progress bars and manifest hashes. Full code must be clean and CPU threads1.
The hard timeout is120 minutes; failures retain a FAILED manifest and partial
journals. No resume support; use a new timestamp after fixing an interruption.

The user runs the full command directly in the Ubuntu lab terminal, then runs
the canonical rsync -avhP --partial command on Mac. No SSH, remote terminal
control, lab execution or transfer is performed by the assistant. Keep the lab
terminal open until completion. Scientific interpretation awaits full results.
