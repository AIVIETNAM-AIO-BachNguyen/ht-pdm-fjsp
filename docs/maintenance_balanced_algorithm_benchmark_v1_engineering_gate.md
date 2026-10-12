# Balanced 2–2 algorithm benchmark: engineering gate and lab handoff

## Material Passport
Synthetic fixed simulator, no participant data. User authorized five baselines
plus PPO/MAPPO/IQL/QMIX and canonical lab workflow. This is an engineering
report, not a scientific comparison. Full test outcomes have not been opened.
Setting is frozen balanced_a2_b4_u12, with isolated_h6 reference; no new setting
search or tuning using smoke outcomes. Code prepared in an isolated checkout
because the main working directory contains unrelated edits.

## Engineering result
Gate PASS. Targeted command ran the four new test files plus existing joint
learning environment and balanced search v2 regression files: **62 passed in
15.54s**. The tests cover actual simulator integration, all 21 physical
matchings, legal local proposals/urgency arbitration, no peer/future leakage,
terminal 12-tick continuation, four learners' optimization/masked targets,
QMIX monotonicity, checkpoint corruption, and independent explicit-index
crossed-bootstrap computation. A deliberate cost-component corruption is
rejected by scalar replay.

Successful artifact: `maintenance_balanced_algorithm_benchmark_v1_smoke_mac_20261012T102846Z`.
Status COMPLETED, elapsed 5.001s, CPU/macOS. Four training models, 1,536 learned
transitions, 1,920 physical training ticks / 32 complete training episodes.
Development 24 episodes; test 36 episodes / 2,160 physical ticks and 1,728
policy coordination records. Twelve checkpoint files include four untrained
development diagnostics and eight trained candidates. All four models changed;
PPO/MAPPO 16 optimizer steps each, IQL/QMIX 145 each. Checkpoint choices sealed
before smoke test opening. Every physical test tick replayed; zero invalid
assignments. Sealed full test flag false. External post-run read-only check
verified SHA256 and size for all 250 bound outputs, exact counts, partial/final
CSV identity and selected checkpoint parameter hashes. Total bound size 8.15MB.

Parameter counts: PPO 18,966; MAPPO 15,364; IQL 26,124; QMIX 23,748. Training
wall time and optimizer counts are recorded, since sample parity is not compute
or architecture parity. No smoke performance ranking is used to tune anything.

## Preserved failed smoke artifacts
These remain intact and excluded from scientific evidence. The failures were
integration/audit issues; corrective tests now cover both paths.
- `maintenance_balanced_algorithm_benchmark_v1_smoke_mac_20261012T102318Z`: AttributeError: module 'ht_pdm_fjsp.maintenance_skill_contention_screen_v3' has no attribute 'pending'
- `maintenance_balanced_algorithm_benchmark_v1_smoke_mac_20261012T102540Z`: KeyError: 'population_sha256'

## Full protocol and resource scope
Four algorithms × five training seeds = 20 models, 491,520 learned steps each:
9,830,400 learned transitions / 12,288,000 physical training ticks. Four trained
checkpoint candidates per model, plus initial development diagnostics. A single
trained checkpoint per replica is selected by 64 separate development seeds;
all five replicas are retained. Total 6,400 development episodes. Selected
replicas and five baselines produce 6,400 paired test episodes over 256 new
seeds, with 384,000 physical test ticks. Planners use 128 CRN scenarios.

Primary family: 20 cost comparisons and eight uptime/backlog comparisons,
40,000 crossed-bootstrap samples, confidence 1−.05/28. No mandatory winner.
Full CPU hard limit 24h; timeout is incomplete execution, not poor algorithm
quality. No tested resume contract: interrupted runs remain intact; use a fresh
UTC timestamp. Smoke row-size extrapolation suggests roughly 0.52GB before the
console log/variations; reserve about 1GB for this run. The canonical rsync
copies all remote artifacts, whose total may be larger. Tiny smoke timing is
not a reliable lab runtime estimate.

## Review and statistical fallacy scan
Two bounded native-agent reviews shared conversation context and model family;
they were not blind or independent-model scientific replication. The bootstrap
test uses separate explicit indexing to check the production weighted-count
calculation. Physics replay reconstructs states, shock tapes and scalar costs.

| Check | Applied interpretation |
|---|---|
| Simpson | Paired pooled estimates do not assert improvement in every state. |
| Ecological | Episode/replica estimates do not identify individual machine effects. |
| Berkson | Full seeds fresh; checkpoint selection uses only separate development data. |
| Collider | No filtering test episodes by success, failure, or conflicts. |
| Base rate | Actual failure counts logged; p=.45 is a conditional hazard. |
| Regression to mean | New common test panel; earlier setting-search gains not reused as algorithm evidence. |
| Survivorship | All five training replicas retained; failed runs preserved. |
| Look elsewhere | Fixed 28-comparison family; all registered comparisons reported. |
| Forking paths | Fixed budgets, seeds, checkpoint rule; engineering repairs change no scientific budget. |
| Correlation/causality | Fixed synthetic simulator evidence does not establish plant effects. |
| Reverse causality | Decisions use pre-step state; outcomes follow the physical transition. |

Five replicas give approximate uncertainty. PPO has global execution;
MAPPO/IQL/QMIX local proposals use a fixed central arbiter. IQL and QMIX also
differ in sharing. Model-based rollouts know simulator physics. Consequently
cross-algorithm differences cannot isolate MARL, architecture or learning-rule
benefits. Feedforward implementations do not claim paper-exact replication.

## Handoff
Branch `codex/maintenance-balanced-algorithm-benchmark-v1`. After commit/push,
provide the user exactly one lab compound command using
`uv run python -m ht_pdm_fjsp.maintenance_balanced_algorithm_benchmark --profile full --device cpu --output-dir artifacts/$RUN_ID`
and the canonical `rsync -avhP --partial` command. User runs both commands.
Assistant does not SSH, launch full or transfer. Keep the lab terminal open.
