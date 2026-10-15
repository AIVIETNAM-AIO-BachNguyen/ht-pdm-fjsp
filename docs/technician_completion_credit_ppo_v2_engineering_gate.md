# Completion-Credit v2 verified engineering gate

Prospective registration commit: 2a7c894. No pilot learning has started.
V1 negative pilot still prohibits V1 full development. V2 requires its own gates.

51 V2 tests PASS (14.54s), including units/loss, unchanged actor tensors under
unit conversion, original/normalized PPO paired initialization and scalar action
routing, no-calendar actor/critic feature removal with feasible reservations,
41-endpoint null/missing-grid checks, calendar mechanism gate, strict checkpoint
identity, serial/spawn determinism, and unchanged tensors with optional diagnostic
logging. The first combined V1/V2 suite passed 87 tests before the final diagnostic
inertness test was added. That new test initially used the old default replay
precision; correcting its fixture to the registered unit-aware precision gives
51/51 PASS. Scientific thresholds, seeds, budgets and objective were not changed.

Seed audit PASS: 387 actual raw seeds, 334525 derived seeds, all mutually/history
disjoint apart from explicit pairing. Source registration audit records hashes
of every historical seed registry, including unopened V1 confirmation panels.

Linux smoke `artifacts/technician_completion_credit_ppo_v2_smoke_20261015T070232Z`:
COMPLETE/audit PASS in32.67s, 14 random fits, 112 training episodes,16 paired tapes,
84 DEV checkpoint episodes,32 engineering test episodes. Independently replayed
all1920 physical/coordination test ticks. Zero invalid assignments, teacher or
pretraining. Independent rehash of all353 output bindings and252 current source/
PLAN/registry bindings PASS. All five graph arms18,444 parameters at both sizes;
original/normalized PPO18,966, local MAPPO15,364, full MAPPO18,052. Every selected
checkpoint contains a value-unit/envelope/saturation probe. Engineering test
panel802... is opened by smoke; scientific confirmation panel812... is NOT opened.
Smoke classification ENGINEERING_ONLY; it does not establish cost/health success.

Full regression COMPLETE: 1158 PASS, one existing tqdm/rich experimental warning,
in758.93s. Log: `/tmp/technician_completion_v2_full_tests_20261015.log`.
PID263783 and session46038 are closed. All353 smoke output bindings and252 current
learning-source/PLAN/registry bindings were reverified after regression completed.
No learning source changed after the passing tests/smoke. The smoke manifest
honestly records the uncommitted pre-verification code state; the implementation
is committed/pushed only after these checks. Once that push succeeds, the
registered pilot can run on this lab. This engineering gate does not establish
scientific quality, a mechanism effect, or goal completion.
