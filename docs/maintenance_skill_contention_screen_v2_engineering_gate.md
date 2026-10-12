## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: authorized engineering execution and verification.
- Origin Date: 2026-10-12.
- Verification Status: VERIFIED for engineering smoke only; full science UNVERIFIED.
- Version Label: maintenance_skill_contention_screen_v2.

# Engineering gate: PASS

Protocol was saved before any v2 episode ran. No performance-based changes to
candidate grid, physical environment, metric, selection, thresholds, seeds,
budget or stopping rule were made after smoke outcomes. V1 artifacts and code
remain unchanged. Work is isolated on branch
`codex/maintenance-skill-contention-screen-v2`; primary workspace user changes
were not included.

## Dependency and test evidence

`uv sync --frozen --extra dev --offline` passed using the local cache. No changes
to pyproject.toml or uv.lock were needed; use the module CLI.

- New metric/protocol tests: **16 passed** (6.45s).
- Complete repository suite: **783 passed**, one existing SB3 rich/tqdm warning
  (243.23s). Log: `/private/tmp/maintenance_skill_contention_screen_v2_tests.log`.
- A metadata correction then separated generic confirmation openness from full
  panel openness: smoke opens only its own confirmation panel. The protocol
  wording clarifies that full seed definitions are read for overlap audits,
  while full episodes/outcomes are never opened by smoke. This changed no
  algorithm, episode or statistical rule.
- Final affected v1/v2 tests after that correction: **27 passed** (8.31s).
  Log: `/private/tmp/maintenance_skill_contention_screen_v2_final_tests.log`.

Tests exercise equal-score collisions, incomparable speed/restoration tradeoffs,
strict speed and restoration dominance, technician-label invariance, pure
capacity pressure, compatibility deficits, busy-worker exclusion, forced unique
assignments versus joint consequential choices, negative controls, ranking that
ignores cost, no-winner path, both-rule confirmation, fresh seeds, paired shock
tapes, cost reconciliation, metric tamper detection, full artifact hashes and
overwrite refusal.

## Final macOS smoke

Command:

```bash
.venv/bin/python -m ht_pdm_fjsp.maintenance_skill_contention_screen_v2 \
  --profile smoke --device cpu \
  --output-dir artifacts/maintenance_skill_contention_screen_v2_smoke_20261012T034330Z
```

Working directory: `/private/tmp/marl-maintenance-assignment-confidence-v1`.
Console log: corresponding `artifacts/<run>.log` (tqdm in both phases).

- Manifest COMPLETED, engineering audit PASS, 5.992s.
- 600 screening +40 confirmation = **640 episodes /38,400 physical transitions**.
- All 640 episodes journaled; all 38,400 transitions and their observed-state
  metrics replayed. Costs reconcile; invalid executed assignments=0.
- Negative controls PASS: equal-score, incomparable tradeoff and partitioned
  settings each have zero registered strict skill contention/joint choice.
- 202 raw and202 derived seed IDs unique; checked historical registry raw IDs
  and202 v1 derived IDs. Full episode outcomes unopened.
- Selection frozen and hashed before confirmation; every artifact hash and size
  independently checked. Artifact payload46,865,406bytes.
- A first timestamped smoke remains preserved. Final episode CSVs, ranking,
  selection and scientific summary are byte-identical to that run; only source,
  protocol wording and manifest metadata changed.

Smoke-selected candidate `overlap3__common_2_3` has four-seed confirmation
**NOT_CONFIRMED**: strict competition estimate11.98% [10.94%,12.50%], joint
contested choice5.21% [3.125%,6.25%]; joint lower bound does not reach5%.
Availability and pending gates pass. This is not a final setting decision;
four-seed smoke outcomes are engineering evidence. Gates/grid were not tuned.

No checkpoints are required or produced because this runner does not train RL.
The maximum full payload is9,600screening+1,280confirmation episodes,
652,800transitions. Full logging uses first2seeds per candidate/rule/phase,
so expected raw payload is roughly30–50MB plus console log.

## Authorized handoff

Commit/push these verified files, then show the single lab compound command and
the canonical `rsync -avhP --partial` command. Run full only on user's lab
terminal at `bachng@100.111.83.52`, CPU, fresh UTC timestamp. No lab SSH, full
execution or transfer is performed by assistant. Protocol:
`docs/maintenance_skill_contention_screen_v2_plan.md`.
