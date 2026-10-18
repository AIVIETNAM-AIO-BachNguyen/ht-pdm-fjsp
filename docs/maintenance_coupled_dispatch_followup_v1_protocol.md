# Coupled dispatch follow-up v1 — prospective protocol

Locked 2026-10-18 after the v1/v2 outcomes were known, **before** any new
development or test outcomes. This is one bounded follow-up; it cannot be
retroactive confirmation of v2's 3.57% timing result. No RL or communication
algorithm. Historical audit source:
`artifacts/maintenance_shared_technician_history_audit_20261018T055849Z/`.

## Question, environment and arms

Can a dispatcher that forecasts the joint technician pool beat a strong simple
rule for the same request process? Does capacity-aware request timing transfer
to new machine–technician matrices? Is there a beneficial timing × allocation
interaction? The primary endpoint is undiscounted total system cost over
H48+common FCFS cleanup tail12+the same terminal liability as v1/v2.

Use the exact v1/v2 `WaitingEnv` physical dynamics, failure age4, p=.45,
max age8, restoration .75, PM1, CM2, unavailable6/tick, failure15,
failed-wait limit4/price12, persistent pending demand and nonpreemptive
committed service. Healthy pending machines keep aging; failed pending machines
wait unavailable; no request vanishes. New matrices vary only skill edges,
capacity K2/K3 and legal service durations2–4; the cost and
failure law do not change. No production scheduling. Configurations and splits
are explicit in the frozen registry; six development and six different test
matrices span hub, balanced, bottleneck, dense3, sparse3 and cross3 topologies.
No test configuration is filtered after observing outcomes.

Request rules: `threshold` and `capacity_aware` exactly as v1/v2. Five
dispatchers for **each** request rule: skill-feasible FCFS, risk priority,
myopic matching, prior isolated-machine lookahead, and new `coupled_rollout`.
Select one simple baseline globally from FCFS/risk/myopic by minimum equal
weight mean development cost over all six development configurations, eight
development shocks and both request rules. Fixed tie order FCFS, risk, myopic.
Save `selection.json` before constructing any test environment. FCFS remains
an explicit reference even if another rule is selected.

Coupled rollout uses the current state, pending pool and request timestamps,
known probability/cost/configuration only. At a decision with pending demand
and at least one eligible idle edge, compare at most ten legal candidate
partial matchings: WAIT, the three simple actions, prior isolated action,
then highest public risk/restoration/duration scoring matchings until ten
unique actions. If there is no pending demand or only one candidate, execute
the selected simple baseline. For each candidate, use four common-random-number
forecast scenarios from an independent role-separated decision bank. Simulate
the candidate now, then the development-selected simple continuation through
the remaining H ticks; cleanup is FCFS for all 12 tail ticks, with the actual
terminal liability formula at tick60. Follow the same request generation,
pending conversion, skill, busy durations, stochastic degradation/failure and
completion mechanics. Choose lowest mean scenario total, with ties favoring
the baseline action then lexical order. Candidate shortlist, four scenarios
and fixed continuation make this a **coupled rollout approximation**, not a
joint oracle or optimum. Physical evaluation uses a separate shock bank; a
third diagnostic bank with 16 scenarios evaluates first relevant test state
per coupled episode without influencing actions. No future physical shocks
are passed to the dispatcher.

No exact joint oracle is planned: the persistent request timestamps, ongoing
service, stochastic failures, H60 and K3 create a state space beyond the
repository's restricted DP contract and this follow-up's CPU budget. Same-
state diagnostics are secondary and never replace closed-loop test cost.

## Falsifiable hypotheses and analysis

Primary paired contrasts, each equal-weight over **six test configurations**
after averaging 16 paired shock seeds within configuration:

* A: `capacity_aware/coupled_rollout` minus
  `capacity_aware/selected_simple` is at most −3% of selected_simple cost.
* B: `capacity_aware/selected_simple` minus `threshold/selected_simple` is at
  most −3% of threshold/selected_simple cost.
* C: [capacity_aware/coupled − capacity_aware/selected_simple] minus
  [threshold/coupled − threshold/selected_simple] is at most −2% of
  threshold/selected_simple cost.

For each, require mean practical margin, a 98.33% two-sided hierarchical
bootstrap interval upper bound <0 (Bonferroni three contrasts), and a negative
effect in >=4/6 test configurations to call SUPPORTED. Bootstrap 10,000
times: resample six configurations with replacement, then within each chosen
configuration resample its 16 aligned shock vectors with replacement. Report
paired absolute/relative effect, interval, per-configuration differences,
config wins, and 95% shock-only intervals separately. If the complete valid
panel fails the support gate, report NOT SUPPORTED; classify INCONCLUSIVE
when the interval still includes a benefit at or above the practical margin.
No post-test change to metrics, margins, alpha, selected baseline or budget.

Secondary: threshold/coupled vs selected simple; coupled vs FCFS and prior
isolated lookahead; all ten registered arms per configuration. Availability,
failures, failed waiting/overdue, pending H/end, request wait, utilization,
starts/completions, valid assignments, H/tail/terminal costs and component
costs. Time dispatch CPU separately by arm and total wall time. Record
proposal/assignment and contention exposure. Interpret H/tail shifts; a gain
only from cleanup must be disclosed. Do not treat shocks inside one matrix as
independent configuration replications. Interpretation conditional on six
purpose-built test matrices and synthetic cost law.

## Panels, budget, stopping and gates

Training seeds/checkpoints: none. Smoke: two development configurations
(`hub_dev`, `balanced_dev`) × raw seed 94000000 × 2 request ×5 dispatchers
=20 episodes; decision scenarios2 for smoke engineering. Full development:
six development configurations × seeds 94000100–94000107 ×10 arms=480
episodes. Full sealed test: six distinct test configurations × seeds
94000200–94000215 ×10 arms=960 episodes. Analysis seed 94000400. Raw panel
ranges are disjoint from v1/v2 and readable seed registries. Physical,
decision-forecast and independent diagnostic shock streams use distinct
SHA256 role namespaces. Same config/seed physical tape is used for every arm.
Four decision scenarios for full; 16 diagnostic scenarios on only the first
eligible test decision per coupled episode. No checkpoint selection, tuning or
early stopping. Stop after exactly 1,440 full episodes or 7,200 wall seconds,
whichever first; interruption/time limit = FAILED, no incomplete performance
claim. Smoke extrapolation is used only to confirm that this fixed design fits
the limit; if infeasible, write a prospective engineering amendment before
opening test. Do not search matrices or methods until a positive result.

Development checks are recorded before test: proposal divergence between
timing rules, pending/resource contention, coupled-vs-selected action
divergence, feasibility and cost reconciliation. Do not remove configurations
or change the algorithm if a mechanism is weak; mark the corresponding claim
limited/inconclusive. Baseline selection is the only development-driven choice.

## Artifacts and integrity

Fresh UTC timestamped directory, refuse overwrite; manifest RUNNING,
COMPLETED or FAILED with Git revision, source/protocol/registry hashes, config
and seed panels, exact counts, selected baseline, runtime and output list.
`source_snapshot/`, `resolved_config.json`, `selection.json`,
`development_episodes.csv`, streaming `episodes.partial.csv`, final
`episodes.csv`, `decisions.csv` for test, `forecast_diagnostics.csv`,
`development_activation.json`, `integrity.json`, `summary.json`, and progress
log. Save selection before test starts; log test-open timestamp. Run tests
covering queue/committed work, legal matching, scenario separation, forecast
physical replay including terminal liability, candidate tie policy, full
source/seed/artifact gates and baseline equivalence. Smoke must pass count,
trace replay, cost/feasibility/conservation and progress bar. Use CPU/tqdm.
Commit/push tested code before direct Ubuntu full run; no SSH/transfer.
