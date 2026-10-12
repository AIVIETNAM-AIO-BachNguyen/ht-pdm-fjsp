## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: prospective code experiment, authorized implementation and Mac smoke.
- Origin Date: 2026-10-12.
- Verification Status: UNVERIFIED until engineering gate; scientific full panel unopened.
- Version Label: maintenance_myopic_lookahead_v1.

# Fixed exploratory candidate: myopic matching versus non-clairvoyant lookahead

## Question and hypothesis

Keep `overlap3__fast1_slow325` as an exploratory candidate selected after v3,
not a v3 winner. No setting sweep, RL training, imitation or future-label data.
Does looking ahead reduce realized cost beyond the existing myopic matching
rule on completely new episode seeds? H1: joint lookahead lowers mean H48+L12
cost by at least 1%, with no practically material availability/backlog harm.
The fresh evaluation panel can assess this fixed comparison; it cannot turn
the candidate into a setting that passed v3's five contention gates.

## Fixed environment and action contract

Use precisely the v3 environment: N4/K2, measured horizon H=48, common L=12
continuation without reset, failure age 4, conditional failure probability .45,
max age 8, PM start cost 1, CM start cost 2, unavailable price 6 per machine tick,
failure event price 15, waiting limit 4 and overdue price 12. Completion
restoration .75 on every valid edge. Duration rows M0..M3, T0/T1:
`[(1,0),(1,3),(1,2),(1,5)]`; mask `10,11,11,11`, fixed across every seed.
Zero means ineligible. This is not a dense mask.

All roots use the v3 action space: free technicians, unoccupied machines that
are failed or age >=3, every feasible partial matching, and empty WAIT.
Do not give lookahead access to preventive jobs younger than the baseline's
action space. All methods use the exact same scalar environment transition,
initial ages and eligibility-independent shock tape for each paired seed.

## Three fixed methods

1. `myopic_matching`: the unchanged v3 `deadline_matching`, maximizing the sum
   of urgency plus restoration/duration. Here "myopic" denotes this current
   state score, not an optimizer of expected one-tick economic cost.
2. `joint_lookahead`: enumerate all legal roots, evaluate cost over 12 ticks
   using 128 fresh forecast scenarios (smoke 8), choose the lowest sampled mean.
   Apply only the root, then replan next real tick. Forecast continuation after
   the root is always `myopic_matching`; this is rollout policy improvement,
   not full stochastic DP or recursive optimal search.
3. `assignment_lookahead`: evaluate the same forecast bank but restrict roots
   to assignments on the exact machine subset chosen by myopic in the current
   observed state. Timing/subset/WAIT at that root are fixed; if there is only
   one such assignment, use it without forecasting. This is an ablation; after
   trajectories diverge it does not hold whole-episode machine choices fixed.

Root order puts myopic first, then the other legal actions in deterministic
catalogue order. Myopic wins exact/numerically equal ties (absolute 1e-10).
An unrestricted root can change machine subset, assignment, cardinality or
WAIT. No access to the actual environment's future events. Planner takes only
config, current state, tick and domain-separated forecast seed. All candidate
roots share a forecast tape (CRN). The two planning methods intentionally share
the selection seed at each episode/tick; real outcome seeds and forecast seeds
are separate roles. Final 12 real ticks use common myopic continuation for all
methods; no re-planning there. All service times <=5, so jobs started in H finish
before H+L, although later failures/backlog may remain.

## Seed panels and no tuning

Registry: `configs/maintenance_myopic_lookahead_v1_seed_registry.json`.

- Full evaluation: 36000000..36000127, 128 paired seeds.
- Full bootstrap raw seed: 36100000; forecast raw seed: 36200000.
- Smoke evaluation: 36300000..36300003, four paired seeds.
- Smoke bootstrap raw seed: 36400000; forecast raw seed: 36500000.
- No train/development seeds. No tuning of method, budget, horizon, margin or
  setting using either smoke performance or full outcomes. Smoke is engineering.
- Physical role `maintenance_myopic_lookahead_v1:physical` derives from episode
  seed. Selection role `...:selection:<episode>:<tick>` and holdout diagnostic
  role `...:holdout:<episode>:<tick>` derive from the profile's forecast raw seed.
  Bootstrap role `...:bootstrap` derives from analysis raw seed.
- Audit raw/derived uniqueness across both profiles and inherited raw registry
  values, including derived physical/bootstrap roles of v1/v2/v3 skill screens.
  Intentional sharing across methods/candidate roots is not an overlap failure.
- Full outcomes open only during a clean committed full run on the lab. Mac
  smoke never runs full seeds, even for an engineering diagnostic.

## Primary, secondary and mechanism metrics

Primary: `1 - mean(joint extended_cost)/mean(myopic extended_cost)` over the
paired fresh episode panel. Extended cost covers all H+L economic and waiting
components. Bootstrap 20,000 paired seed-cluster resamples (smoke 200), keeping
all methods together; ratio of means, not mean of episode percentages.

Five fixed intervals use two-sided **99%** percentile CIs (Bonferroni five,
nominal family coverage >=95%): joint/myopic cost reduction, assignment/myopic
cost reduction, joint/assignment cost reduction, H availability difference
joint-minus-myopic, H terminal pending difference joint-minus-myopic.
`JOINT_HEADROOM_CONFIRMED` only if primary lower bound >=.01, availability
difference lower bound >=-.01 and pending difference upper bound <=.10.
Otherwise `NOT_CONFIRMED`; failure does not establish no headroom. The two
ablation intervals do not determine the primary outcome. These are bootstrap
approximations with iid episode seeds, not finite-sample coverage guarantees.

Secondary descriptive metrics: H policy cost, common-tail cost, cost components,
failures, overdue ticks, availability, utilization, H pending/failed, tail
pending/failed; v3 strict, possible and selected cost-sensitive contention rates
on each method's own trajectory; disagreement versus myopic on the same current
state, subset changes, WAIT changes, assignment-only changes; planner latency,
number of roots and actual forecast-transition budget. Rates denominator H=48.
These rates are policy dependent, not intrinsic setting labels or scientific
selection gates. No new setting selection.

Mechanism diagnostics on **myopic trajectory states** at fixed ticks
`[0,8,16,24,32,40,47]`, with no outcome-based sampling:
evaluate every legal root using the same selection bank as a planner would use,
choose joint and same-subset roots, then evaluate roots using an **independent
holdout forecast bank**, also 128 scenarios/12 ticks (smoke 8). Log each root's
selection/evaluation mean and SE. Log paired holdout root gains versus myopic,
including zero for unchanged choices, and whether assignment/subset/WAIT changed.
Use exploratory 95% seed-cluster CIs over mean gain per seed, with all seven
landmarks retained; no gate or method selection depends on these diagnostics.
These are model-based 12-tick continuation values, not real causal full-episode
savings or omniscient future outcomes, and never training labels.

## Fixed budget and stopping

Full: exactly 384 physical episodes and 23,040 physical transitions, plus 896
common-state landmarks. Smoke: exactly 12 episodes and 720 physical transitions,
plus 28 landmarks. Every episode performs all 60 physical ticks.
This mask has at most 17 roots: one WAIT, seven single edges and nine
two-edge roots; assert the count during preflight.
Conservative forecast cap uses the full N4/K2 catalogue bound 21:
`(2*seeds*48 + 2*seeds*7)*21*scenarios*12`; actual spent transitions logged.
No adaptive scenario count, early outcome stop, fallback method, replacement
setting or resume. Timeout 600s smoke, 7200s full, checked during episode and
forecast loops. Timeout writes FAILED and requires a fresh timestamped run.
Do not relaunch a failed run automatically without understanding/fixing an
engineering cause. Timeout is an engineering bound, not scientific stopping.

## Engineering and scientific gates

Tests: scalar/vectorized forecast equivalence including completion/waiting,
baseline exact equivalence, root eligibility and same-subset restrictions,
baseline-included ties and single-root shortcut, no forecast use of real events,
separate heldout bank, paired tapes, seed disjointness, cost reconciliation,
bootstrap arithmetic, replay tamper detection, complete smoke artifact schema,
overwrite refusal, clean full-run requirement and timeout recording.
Tests and one real macOS smoke must pass before commit/push. No performance
requirement on smoke. Scientific result is only the preregistered primary and
guards on full seeds, with secondary/mechanism results separated.

## Artifacts and audit

Fresh timestamped directories, refuse any nonempty output. Files:
`manifest.json`, `protocol.md`, `seed_registry.json`, `resolved_config.json`,
`episodes.csv`, streaming `episodes.partial.csv`, `decisions.csv` (all physical
decisions/state/event/following/info plus forecast identity and chosen mean),
`landmark_roots.csv`, `landmarks.csv`, `summary.json`, `audit.json`,
`source_snapshot/` (runner, v1-v3 screens, dispatch, waiting, heterogeneous model,
opportunity helpers and the three inherited skill-screen registries).
All files except manifest bound by SHA256/size; manifest records commit/dirty,
runtime, profile, full-panel-open flag, expected/actual counts, timeout budget,
forecast budget and scientific status. No model/checkpoints: no training.

Replay all physical decisions with the independent scalar transition and check
paired initial state/event tapes, feasibility, metrics and costs. Regenerate
forecasts for the first two full seeds/all smoke seeds, and all logged landmark
roots for those seeds, to audit choice restrictions and selection/holdout values.
Root and landmark grids must be exact. Component costs reconcile. No invalid
assignments, NaN/Inf or hidden partial episodes. Source is committed/clean for
full; source/protocol/registry seal must remain unchanged through execution.

## Interpretation and handoff

This is known-model planning headroom in one fixed exploratory environment,
not evidence that RL can learn it or that MARL is necessary. A 12-tick planner
can fail because of Monte Carlo noise, continuation bias or horizon effects;
NOT_CONFIRMED does not prove myopic optimality. Same-subset ablation and common
state holdout diagnostics address different scopes; do not add local gains into
claimed episode savings. Keep all methods/results including losses.

After verified Mac tests/smoke, commit and push the dedicated codex branch.
Show the user the exact compound command for the Ubuntu lab and the canonical
Mac `rsync -avhP --partial` command. Do not SSH, launch full or transfer results.
