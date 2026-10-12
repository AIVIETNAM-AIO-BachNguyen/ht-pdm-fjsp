## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: authorized engineering execution and verification.
- Origin Date: 2026-10-12.
- Verification Status: VERIFIED for engineering smoke; full science UNVERIFIED.
- Version Label: maintenance_skill_contention_screen_v3.

# Engineering gate: PASS

Protocol was saved before v3 episodes ran. Grid, environment, metrics, five
thresholds, ranking, seed panels and stopping rules were not retuned after smoke.
V1/v2 source and raw artifacts remain unchanged. Branch:
`codex/maintenance-skill-contention-screen-v3`, isolated checkout at
`/private/tmp/marl-maintenance-assignment-confidence-v1`. Primary workspace user
changes were not included.

## Tests and dependencies

`uv sync --frozen --extra dev --offline` passed; no pyproject.toml or uv.lock
change needed. Module CLI is used.

- New v3 tests: **17 passed**,5.27s.
- Complete repository suite: **800 passed**,269.80s; one existing SB3
  rich/tqdm experimental warning in the old parallel-maintenance smoke test.
- Full test log: `/private/tmp/maintenance_skill_contention_screen_v3_tests.log`.

Tests verify same-subset costs, unequal rows with cancelling duration losses,
uniform v2 symmetry, one-job choices excluded from competition, underfilled and
forced-unique actual subsets excluded from the selected gate, busy workers,
capacity versus skill competition, equal/Pareto ties, label invariance, price
scaling, cost-independent ranking, no-winner path, five99% gates under both rules,
fresh v1/v2-separated panels, paired event tapes, consequence-metric tampering,
snapshot hashes, sealed full outcomes and overwrite refusal.

The direct component check runs fixed new jobs through actual service completion
with no subsequent assignments or failure events. Two assignments of M1/M2 have
in-service unavailability costs36and24; their12-unit difference exactly equals
the observed-state committed-cost span. This checks a real economic component,
not a future-rollout target for RL or an assertion about total episode savings.

## macOS smoke

```bash
.venv/bin/python -m ht_pdm_fjsp.maintenance_skill_contention_screen_v3 \
  --profile smoke --device cpu \
  --output-dir artifacts/maintenance_skill_contention_screen_v3_smoke_20261012T040734Z
```

Working directory above; adjacent timestamped `.log` preserves both tqdm phases.

- Manifest COMPLETED, audit PASS, elapsed5.244s.
- **424screening+40control-confirmation=464episodes /27,840transitions**.
- All464episodes/27,840transitions and measured-state metrics replayed.
- Invalid executed assignments0; cost components reconcile; exact episode grids.
- Five negative controls pass:80control rows across the two phases/rules.
-202unique raw and202unique derived IDs;404v1/v2derived IDs checked; historical
  raw registry IDs checked; full episode outcomes unopened.
- Selection frozen and hashed before confirmation; all19artifact hashes/sizes
  independently checked; payload37,557,023bytes.
- Model/checkpoint save/load is NOT_APPLICABLE: no learning in this screen.

Scientific smoke status: **NO_ELIGIBLE_SETTING**, confirmation NOT_APPLICABLE.
Only the five frozen controls are evaluated on the smoke confirmation panel.
For example, bottleneck2/fast2_slow634 has strict competition15.10% and potential
cost-sensitive competition11.98%, but selected cost-sensitive competition0%.
overlap3/fast1_slow253 has10.42%,4.69%,4.17% respectively; it misses the5%
potential-consequence gate on this four-seed panel. No gates or settings changed.
These values are engineering-panel diagnostics, not full scientific evidence.

Maximum full budget:6,784screening+1,792confirmation=8,576episodes,
514,560transitions. If no candidate qualifies, confirmation evaluates only five
controls and the result remains NO_ELIGIBLE_SETTING. Full journal logs first2
seeds per candidate/rule/phase; expected artifact payload roughly25–40MB plus
console log. No model files.

## Handoff

Commit and push the five verified files, then give one exact compound command
for the user to run in the lab terminal at `bachng@100.111.83.52`, plus canonical
`rsync -avhP --partial` for the user to run later on Mac. Do not SSH, start full
run, or transfer. All runs require fresh UTC timestamped output dirs; full run
requires a clean committed checkout. See
`docs/maintenance_skill_contention_screen_v3_plan.md` and
`docs/experiment_workflow.md`.
