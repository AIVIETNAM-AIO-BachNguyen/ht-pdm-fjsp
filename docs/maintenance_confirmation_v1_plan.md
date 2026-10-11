# Maintenance confirmation v1 — locked fresh-panel protocol

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan/run, inline execution
- Origin Date: 2026-10-11
- Verification Status: UNVERIFIED until engineering checks; full outcomes sealed
- Version Label: maintenance_confirmation_v1
- Authority: user approved the proposed final confirmation experiment on 2026-10-11.

## Question and hypotheses

Does fixed allocation with a modeled terminal continuation provide more than
1% interior cost headroom over the frozen deadline rule while remaining within
the three inherited service margins? This is a prospectively locked follow-up
chosen after viewing boundary-sensitivity v1. It is not a retroactive change to
that experiment's hypotheses, CIs, or result. Use only fresh held-out profiles
and shock seeds; do not pool this test with the pilot for the decision.

Four directional criteria, all required for GO:
1. Relative interior cost improvement (rule minus planner)/rule > .01.
2. Planner minus rule interior pending machines < .1.
3. Planner minus rule interior total failed machines < .1.
4. Planner minus rule overdue waiting ticks per physical tick < .25/12.
Margins are research investment screens, not calibrated factory requirements.
Service starts and completed repairs are distinct; total failed includes CM
in progress. All state service averages use the state BEFORE each interior
physical tick, identical to boundary v1.

## Physics and frozen controllers

Only skill_mask, N4/K2, H48 and an additional L12 actual tail. Keep every
WaitingEnv coefficient: PM1, CM2, unavailability6, failure15, wait limit4,
overdue price12, original nonpreemptive physics and partial matchings.
Interior window [12,36) covers 24 physical ticks. Cost is normalized by
4 machines x 24 ticks. An interior window is neither stationary performance
nor an infinite-horizon claim; earlier decisions influence it.

Compare best_rule (the frozen deadline_matching) and fixed_tail only.
The baseline was selected on the PREVIOUS development panel for skill_mask:H48;
record the source selection/config/manifest hashes and commit in a committed
frozen_baseline.json. There is no new baseline tuning, development evaluation,
training, or checkpoint. Planner is inherited boundary.forecast_plan with
fixed=True, 128 forecast scenarios per decision, full remaining H lookahead,
adaptive_isolated_matching before H, then L12 deadline_matching. No forecast
horizon cap, new cost coefficient, changed continuation, or learned policy.
All actual trajectories continue L12 under the shared deadline rule without
reset, including ongoing jobs and new failures. Report base, tail, extended
and interior costs separately. No full-run inference on local Mac.

## Panels and independence

Full: 256 new physical configurations x 16 shocks x 2 controllers = 8,192
test episodes, 393,216 base + 98,304 tail = 491,520 physical decisions.
Raw profile seeds 11,100,000..11,100,255; shock seeds 11,200,000..11,200,015;
analysis seed 11,300,000. No training/development seeds. Smoke uses profiles
11,400,000..11,400,001 and shocks 11,500,000..11,500,001, analysis 11,300,001,
H12/L3/S8: 8 episodes and 120 decisions. Smoke is engineering evidence only.

Use the inherited skill_mask generator (base durations 2..5, failure ages3..5,
failure probabilities .25/.45/.6). Exclude physically identical configurations
from BOTH prior full development/test panels AND prior smoke panels; ignore
H and arbitrary machine/worker labels, normalize restoration on absent edges.
Each new profile must also be distinct from all earlier profiles in this panel.
The committed profile_exclusions.json contains signatures, raw historical seed
panels and hash provenance; the lab needs no old artifacts. Construction uses
deterministic rejection up to 10,000 attempts/seed, raises on pool exhaustion.
The sample is from the remaining finite skill_mask profile pool, not a claim
of unconstrained IID uniqueness or generalization to other sizes/topologies.
Only skill_mask is sampled; the old joint acceptance across three conditions
is unnecessary here and is not claimed to induce an identical profile mix.

Smoke excludes the full panel via metadata-only construction. Tests may inspect
full configuration metadata and counts, but never evaluate full test outcomes.
Audit all committed historical seed registries, all prior raw panel seeds, own
smoke/full disjointness, and internal profile/shock/analysis separation.
Physical and independent forecast tapes use domain-separated role_seed names
with the NEW protocol namespace, cohort, shock and physical time; controller
identity is excluded. Matched controllers share actual initial states/events.
Forecast tapes never access actual future events. No future leakage or policy
selection from new test outcomes.

## Statistical decision and fixed stopping

Average the 16 paired shock differences within EACH configuration. Resample
256 configurations with replacement 20,000 times using the frozen analysis
seed; apply the SAME profile draws to all four contrasts. For relative cost,
divide each bootstrap draw's mean paired difference by its resampled mean
baseline cost (not an average of per-profile percentages). Report profile wins
and ties, raw means and absolute deltas. Positive baseline denominator required.

Four two-sided percentile intervals at 98.75% (1-.05/4) provide Bonferroni
family correction with approximate bootstrap coverage, not exact finite-sample
coverage. Uncertainty is profile-bootstrap conditional on the fixed shock panel;
do not treat 4,096 shock episodes as 4,096 independent configurations. Additional
95% intervals and all other metrics are exploratory and cannot change GO.

GO: cost lower CI > .01 AND all three service upper CIs < their margins.
NO_GO: cost upper CI < .01 OR any service lower CI > its margin.
Otherwise INCONCLUSIVE. Equal-to-margin boundaries are INCONCLUSIVE.
INCONCLUSIVE does not mean no effect/equivalence; practical resource rule is to
advance to policy training only on GO. No threshold relaxation or favorable
subgroup selection after viewing data. Failure of one criterion blocks joint GO.

Fixed complete 256x16 panel: no performance-dependent stopping, no extra seeds,
no automatic retry/resume. Failed/interrupted runs retain partial journals and
FAILED manifests; a new attempt needs a fresh UTC directory. No silent overwrite.
On hardware failure, report incomplete; do not infer success from partial data.

Sample size is a finite budget choice, NOT a guarantee of conclusive power.
Pilot estimate ~.99% sits at the 1% threshold. Pilot variance heuristics suggest
~.42 percentage-point 95% cost half-width at 256x16 before the four-way correction;
that approximation uses fixed denominator/normal quantiles, is not a promised
CI, and the new remaining-pool profile mix may differ. Extra data can still leave
the decision inconclusive. No power guarantee is asserted for service gates.

## Computational optimization and runtime

Skip forecasts ONLY when fixed_candidates has exactly one feasible action.
Selecting its sole action is algebraically identical to full argmin. Do not
fabricate the uncomputed score: predicted_action_cost is null, actual forecast
scenarios/ticks/work samples are zero, singleton_skipped=true; configured work
is recorded separately. This changes neither action nor independent per-time
random streams. For >1 candidate call the unchanged inherited forecaster.
Test both actions and complete physical trajectories against the unoptimized
implementation using smoke states; retain the scalar/batch physical audit.

CPU only, OMP/OPENBLAS/MKL thread limits=1. Full runner rejects dirty code and
missing thread limits; smoke may run dirty before commit. Sequential evaluation
with tqdm. Previous fixed_tail H48 cell: 512 episodes ~1,017s action selection;
8x episodes imply ~2.26h before fast-path savings/overheads, illustrative only.
Budget planning ~2–3h on the same lab, ~0.4–0.6GB artifacts; actual profile mix,
hardware/cache and singleton frequency can change it. No GPU is required.
Progress bar/partial journal expose liveness; no outcome-based timeout.

## Artifacts and engineering gate

New timestamped directory: manifest.json, plan.md, resolved_config.json,
seed_registry.json, frozen_baseline.json, source_snapshot/ (all dependent physics,
planner/rule sources, new runner, pyproject, uv.lock, current/historical seed
registries, exclusions, frozen baseline); episodes.partial.csv, episodes.csv,
decisions.csv, requests.csv (cutoff and extended), carryover.csv, latency.csv,
coordination.csv, time_profiles.csv, profile_differences.csv, summary.json.
Console log beside run. Manifest hashes every artifact except itself; records
commit/dirty/runtime/CPU/thread/device/status/counts/seed audit/test-open flag.
No model checkpoint because no training. Streaming per-decision output avoids
retaining the full ~500K decision grid in RAM.

Smoke PASS: exit0, complete grid/counts, coordination/feasibility/fixed allocation
and scalar/batch audits, physical replay/cost reconciliation, paired tape
equality, request/carryover integrity, hash/schema validation, reproducibility
excluding measured wall latency, singleton shortcut equivalence, summary
ENGINEERING_ONLY and full_test_outcomes_opened=false. Test malformed grids,
statistical boundary decisions/correction/bootstrap unit and ratio, seed/panel
exclusions, invalid parameters, clean/thread full preflight, nonempty directory
rejection and interrupted FAILED journaling. Full success means COMPLETED+audit
PASS; it does not require a scientific GO.

Run repository suite and CLI macOS smoke before commit/push. Follow
docs/experiment_workflow.md: user alone runs the full lab command and Mac rsync;
assistant does not SSH, execute the full run, or start transfer. Preserve prior
raw data and unrelated working-tree changes.
