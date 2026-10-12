## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: authorized metric correction and experiment engineering.
- Origin Date: 2026-10-12.
- Verification Status: UNVERIFIED for full screening; smoke is engineering evidence.
- Version Label: maintenance_skill_contention_screen_v2.

# Find substantive competition for technician skills

## Amendment and hypothesis

This is a new exploratory follow-up to v1, not a reclassification of its sealed
confirmation. V1's independent proposals picked the lowest worker index in a
quality-score tie. Its selected speed/restoration tradeoff tied .5/2 with 1/4;
52–65% of its reported confirmation conflicts were resolvable within tied best
sets. V1 raw data, source, ranking and registered verdict stay unchanged.

Hypothesis: a broader fixed grid of overlapping skills and faster services has
at least one setting in which machines compete for workers with **strictly better
physical service**, while consequential alternative assignments remain possible.
The operational definition below is falsifiable; no eligible setting is a valid
result. It does not claim to measure all forms of RL resource conflict.

No RL training, future rollout search, teacher labels or learned benchmark
selection. Training budget and train seeds: NOT_APPLICABLE. Ordinary environment
episodes under two fixed observation rules provide state occupancy measurements.

## Locked physical environment and grid

Keep v1 N=4/K=2, H=48 measured ticks plus L=12 deadline_matching continuation,
failure age4, conditional failure probability.45, max age8, PM1/CM2,
unavailable6/machine-tick, failure15/event, soft overdue12 after waiting limit4.
Restoration happens on completion; pending demand is non-FIFO; partial matchings
and WAIT are legal. Seed varies initial ages and the machine/time shock tape,
not the physical matrix. No dense mask.

Six selectable masks, rows M0..M3 / columns T0,T1:

| Topology | Rows |
|---|---|
| overlap3 | 10,11,11,11 |
| overlap3_reverse | 01,11,11,11 |
| balanced2 | 10,01,11,11 |
| bottleneck2 | 10,10,11,11 |
| balanced1 | 10,10,01,11 |
| bottleneck1 | 10,10,10,11 |

Cross each with 12 patterns: three favourite-worker vectors common=(0,0,0,0),
cluster=(0,0,0,1), paired=(0,1,0,1), each crossed with fast/slow durations
(1,2), (1,3), (2,3), (2,4). Compatible edges restore .75. This gives 72 selectable
settings. Add three nonselectable controls:

- overlap3/equal_control: all compatible edges duration3/restoration.75.
- overlap3/tradeoff_control: alternating favourites duration2/restoration.5,
  others duration4/restoration1 (the v1 tied-score winner).
- partitioned_control/common_2_4: mask10,10,01,01, duration2 for T0 and4 for T1,
  restoration.75; no alternative worker for any machine.

Total **75 fixed configurations**. This grid and these thresholds are our design
choices; not asserted to be literature-standard, optimal, or exhaustive.

## Metric definition: capacity held fixed, ties retained

At each measured tick, urgent machines are failed or age>=3, excluding machines
already in service. Valid edges use the existing mask and currently free workers.
For each machine m, technician j strictly dominates k when duration(j)<=duration(k)
and restoration(j)>=restoration(k), with at least one strict inequality. Keep
**every nondominated edge**, including exact ties and speed/restoration tradeoffs.
No arbitrary score, worker-index tie-break or future events enter this metric.

Let A be all feasible partial urgent-machine matchings and B its subset using
only nondominated edges. F=max cardinality(A), P=max cardinality(B).

- **Primary skill_contention_rate:** fraction of all H ticks with P<F. The same
  free-worker capacity can serve F machines, but serving that many requires
  someone to accept a strictly worse physical service. Pure excess demand alone
  does not produce this metric when P=F; compatibility bottlenecks alone may
  cause skill_deficit without producing preference competition.
- **Joint contested_choice_rate:** fraction of H ticks with P<F AND there exists
  a machine subset of size F with at least two physically distinct feasible
  assignments and no all-nondominated assignment of that same subset. This checks
  that strict competition and a consequential assignment choice coexist on the
  same tick, rather than occurring in unrelated states.
- Auxiliary: sensitive_choice_rate and selected_sensitive_rate (same-machine
  physical alternatives); capacity_pressure_rate (urgent machines>free workers);
  skill_deficit_rate (F<min(demand,free)); mean F/P/deficit; legacy proposal conflict
  diagnostic; tie-sensitive legacy flag; availability, utilization, pending/failed
  at H and H+L, complete extended cost decomposition.

All rate denominators are **all 48 measured ticks**, not only decision ticks.
Label-only worker swaps do not count as physical alternatives. If F=0, both
primary and joint indicators are zero. Equal and tradeoff controls must have
P=F; partitioned control must have joint choice zero. Skill contention describes
state-local strict physical dominance, not optimal economic Q-values. Tradeoffs
can matter for RL even though they are not strict dominance.

Observation policies stay v1 deadline_matching and priority_greedy, including
their urgency and restoration/duration scoring. The metric is independent of
that scoring. Both execute feasible actions; common deadline_matching tail is
not reset. The two rules generate different occupancy distributions, not two
independent samples. Same episode seed means same initial ages and shock tape
across every candidate/rule within this protocol.

## Selection, sealed confirmation, seeds

Screen 64 seeds per setting/rule. Eligible candidates must satisfy all four:
min-rule skill_contention_rate>=.10; min-rule contested_choice_rate>=.05;
min-rule availability>=.25; max-rule terminal_pending<=1.
These mean at least 4.8 strict competition ticks and 2.4 contested assignment
ticks per episode on average, with operational-health guards.

Among eligible candidates, rank by min-rule primary descending, then min-rule
joint descending, then candidate ID ascending. No cost or RL reward in ranking.
Freeze winner, runner-up (if any), and all three controls; hash selection.json
**before any confirmation episode runs**. With no winner, confirm the controls
only and report NO_ELIGIBLE_SETTING; never force a fallback or rerank.

Full raw panels: screening32000000..32000063; confirmation32100000..32100127;
analysis32200000. Smoke: screening32300000..32300003;
confirmation32400000..32400003; analysis32500000. Physical seeds use
role_seed(raw, 'maintenance_skill_contention_screen_v2:physical'); bootstrap
uses role_seed(analysis, 'maintenance_skill_contention_screen_v2:bootstrap').
Check raw and derived uniqueness across profiles, historical registry raw seeds,
and v1 derived panels. Smoke never executes the full panel or opens full episode
outcomes. The raw full seed definitions are read only for overlap audits.

Confirm only the frozen winner on **128 new seed pairs**. Use 20,000 bootstrap
resamples of seed clusters, preserving both rule observations together, and
min/max across rules on each draw. Four individual two-sided **98.75%** CIs
(Bonferroni four gates, nominal family-wise coverage>=95%) must satisfy all gates:
lower bounds for the first three, upper bound for pending. No replacements after
confirmation failure. Controls and runner-up are mechanism diagnostics, not
alternate confirmation candidates. No claim of physical-profile generalization
or global maximum beyond this grid and these two rules.

## Budget, stopping and artifacts

Full: 9,600 screening episodes plus at most 1,280 confirmation episodes =
10,880 episodes / 652,800 transitions maximum. Smoke: 600 plus at most40 =640
episodes /38,400 transitions. Run the complete panel without performance-based
early stopping; timeout600s smoke/7,200s full. Failure records manifest FAILED;
no resume; rerun into a new timestamped directory. Progress via tqdm per phase.

Artifacts: manifest.json (commit, clean full-run gate, runtime, hashes, counts,
seal status, protocol/metric version), resolved_config.json, protocol.md,
seed_registry.json, source_snapshot/, screening_episodes.csv,
screening_ranking.csv, selection.json, confirmation_episodes.csv,
decision_audit.jsonl, audit.json, summary.json. Log next to run directory.
Journal stores observed-state metric reconstruction plus physical state/events,
feasible action, next state and costs for first2 full seeds /all4 smoke seeds of
each candidate/rule/phase. Replay scalar transitions and metrics, check episode
grid, costs, feasibility, negative controls and frozen selection. Reject nonempty
output dirs. Model/checkpoint save/load is NOT_APPLICABLE (no training).

Smoke engineering PASS requires counts, feasibility, paired shock tapes, scalar
replay, metric invariance tests, controls, seals, hashes and progress. Scientific
CONFIRMED is not needed for engineering PASS and is not inferred from four seeds.
Run tests plus macOS smoke, commit and push, then hand off exact lab and rsync
commands per docs/experiment_workflow.md; never launch full run or transfer for user.

## Interpretation safeguards

Keep both rule results separately; seed, not tick, is the resampling unit. Retain
all failures/unserved demand. Screening on 72 candidates is exploratory; winner
confirmation on fresh seeds tests only registered gates. Do not claim cost gaps
between rules isolate assignment or forecast RL gains. No tuning of grid, gates,
seed panels or stopping after smoke/confirmation outcomes. A failure to find a
setting is reported directly and requires a separately documented follow-up.
