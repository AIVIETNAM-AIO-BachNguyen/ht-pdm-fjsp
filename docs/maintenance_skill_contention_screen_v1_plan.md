## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: plan and engineering preparation.
- Origin Date: 2026-10-12.
- Verification Status: UNVERIFIED for full screening; smoke is engineering evidence.
- Version Label: maintenance_skill_contention_screen_v1.

# Screen technician skills before training another RL policy

## Question and hypothesis

Which fixed technician-skill configuration creates frequent competition **and**
physically consequential alternatives for assigning the same machines, without
leaving most machines unavailable? Hypothesis: overlapping skills with unequal
service characteristics can create both contention and assignment opportunity;
the narrowest skill graph need not be the most useful RL benchmark.

This is a non-learning environment-screening experiment. It does not train RL,
create counterfactual rollout labels, search future events, or select a setting
because a particular RL method wins. Simulation here means executing ordinary
episodes in the maintenance environment. Training seeds/budget are NOT_APPLICABLE.

## Literature basis and limits

The existing project review recorded heterogeneous repair-time patterns (2,8),
(8,2), and (8,8) in Ruiz Rodríguez et al. (2022), Methods/evaluation/Table 5:
[author manuscript](https://orbilu.uni.lu/bitstream/10993/54345/1/1-s2.0-S0736584522000928-main.pdf).
The notes are in `docs/research/maintenance_nonfifo_lit_review_20261010/phase2_maintenance_sources.md`
in the primary workspace. Direct PDF and publisher retrieval failed again on
2026-10-12; we do not claim a fresh full-text verification. The 2-versus-8 contrast
is inspiration for an adapted machine-worker duration matrix, not a reproduction
of the paper's component model, collision penalties, or failure distribution.

Demirci, Arts and van Houtum (online 2024 / issue 2025),
[primary full text](https://link.springer.com/article/10.1007/s10696-024-09544-y),
Sections 1 and 3, was accessible on 2026-10-12. It studies resource dependence
under a limited maintenance workforce. Cross-trained workers and machine-specific
maintenance rates do not establish worker-specific skill matching. This motivates
measuring workload separately from skill bottlenecks, not copying its dynamics.

Our compatibility masks, restoration tradeoff, screen score and cutoffs below
are **new design decisions**, not literature-recommended optimal settings.

## Fixed environment and candidate matrix

All candidates: N=4, K=2, H=48 policy ticks plus L=12 deadline-matching continuation;
failure age=4, per-tick conditional failure probability=.45, max age=8;
PM start cost=1, CM start cost=2, unavailable cost=6/machine-tick,
failure event cost=15, waiting limit=4, overdue cost=12/request-tick.
Completion repair, non-FIFO pending demand and partial feasible matching keep the
existing environment semantics. WAIT is legal. No dense setting is included.

Six masks, rows M0..M3 / columns T0,T1:

| Topology | Rows | Purpose |
|---|---|---|
| overlap3 | 10,11,11,11 | Broad overlap, one exclusive machine |
| balanced2 | 10,01,11,11 | Two exclusive machines, balanced workers |
| bottleneck2 | 10,10,11,11 | Two exclusive machines compete for T0 |
| balanced1 | 10,10,01,11 | One shared machine |
| bottleneck1 | 10,10,10,11 | Severe T0 bottleneck, one shared machine |
| partitioned_control | 10,10,01,01 | No worker choice; excluded from selection |

Cross each mask with four skill patterns (24 configurations):

- paired_2_4: favourite workers alternate T0,T1,T0,T1; favourite duration2,
  other duration4, all compatible edges restore .75.
- common_2_4: T0 duration2 and T1 duration4 for every machine, restoration .75.
- paired_2_8: alternating favourites, durations2 versus8, restoration .75.
- tradeoff_2_4: alternating favourites are fast (2 ticks, restoration .5),
  others slow (4 ticks, restoration1). Neither speed nor restoration dominates.

Add overlap3/equal_control (duration3/restoration.75 on every compatible edge).
Total **25 fixed physical configurations**, 20 selectable and five controls.
Every machine and worker has at least one valid edge. Parameters do not vary
with episode seed. A winner will become one fixed setting in subsequent RL work.

## Observation policies and physical pairing

Two feasible non-learning controllers, evaluated on every candidate:

1. deadline_matching: existing urgency and restoration/duration scores, globally
   maximize total edge score over eligible urgent partial matchings.
2. priority_greedy: same urgency and urgent-machine definition, visit machines in
   descending urgency then machine index, choose the best free compatible worker
   by restoration/duration, then worker index. Resolve reservations feasibly.

Urgent = failed or age>=failure_age-1, excluding machines already in service.
Both observe the same current state/physical parameters and no future shocks.
They are observation policies, not an exhaustive solver comparison. Their cost
gap bundles machine selection and allocation and is not pure assignment regret.
Tail uses deadline_matching for both, without resetting state at tick48.

Every candidate/controller shares each episode seed's initial-age vector and
machine/time shock tape. Different trajectories remain different trajectories.
Metrics used to select settings are calculated on the first48 ticks only;
extended cost and tail backlog are reported separately.

## Metrics (denominator is all48 policy ticks unless specified)

- conflict_rate: independently requested best free workers collide, before any
  reservations; zero if fewer than two proposals. A structural pressure measure,
  not invalid actions actually sent to the environment.
- sensitive_choice_rate: among maximum-cardinality matchings of urgent machines,
  at least one identical machine subset admits two assignments with different
  per-machine (duration,restoration). Mere worker-label swaps do not count.
- selected_sensitive_rate: that property for the observation controller's actual
  selected machine subset, including CM. Reported separately, not selection score.
- capacity_pressure_rate: demand count exceeds the number of free technicians.
- skill_deficit_rate: maximum feasible matching serves fewer machines than
  min(demand count, free technicians), isolating immediate compatibility deficit.
- availability: 1 - unavailable_machine_ticks/(N*H); busy maintenance also counts
  unavailable. This is an occupancy fraction, not system reliability probability.
- terminal_pending: failed, unserved machines at H; terminal_failed includes CM
  in service. Report both at H and H+L, failures, cost components and overdue ticks.
- collision_excess, technician utilization and urgent demand counts as diagnostics.

## Screen selection, confirmation and uncertainty

Full:64 screening episode seeds,128 fresh confirmation seeds. Smoke:4 and4,
same physical configurations and H48+L12, distinct raw and derived RNG namespaces.
Seed registry is `configs/maintenance_skill_contention_screen_v1_seed_registry.json`.
No training/test generalization over physical profiles is claimed: the setting
is fixed, and confirmation covers new initial conditions/stochastic trajectories.

Aggregate each metric across seeds separately for the two policies. A selectable
candidate must satisfy **under both policies**:
conflict_rate>=.10, sensitive_choice_rate>=.05, availability>=.25,
terminal_pending<=1 machine. These are research screening cutoffs, not factory SLA
or an established economic threshold. Exclude the five controls even if they pass.

Rank eligible candidates by
`min_policy(conflict_rate) * min_policy(sensitive_choice_rate)` descending, then
candidate id ascending. Also publish an unrestricted contention ranking so that
the strongest bottleneck is not confused with the chosen learnable benchmark.
Freeze winner, runner-up, equal control and the winner-pattern partitioned control
before opening confirmation. If none qualifies: NO_ELIGIBLE_SETTING, no forced
winner and no cutoff relaxation. Confirmation does not rerank or substitute one.

For the winner,20,000 seed-cluster paired bootstrap draws; policy trajectories of
one seed stay together. Four approximate Bonferroni98.75% intervals for min-policy
conflict, min-policy sensitive choice, min-policy availability, max-policy pending.
CONFIRMED only if lower bounds pass the first three cutoffs and upper pending bound
passes the fourth. Otherwise NOT_CONFIRMED. Report all selected configs and costs;
CI width is not evidence of equivalence. No confirmatory claim that the winner is
globally strongest: it won a finite exploratory grid under two reference policies.

## Budget, stopping, artifacts and engineering gates

Fixed25*64*2=3,200 screening episodes. Confirmation<=4*128*2=1,024 episodes.
No sequential early stopping or outcome-driven sample expansion. Full CPU,
no checkpoints or GPU requirement. Hard timeout2h, monotonic clock checked per
episode and before completion; failures preserved, never automatically resumed.
Smoke timeout10min. tqdm over both multi-seed phases; no training phase exists.

Artifacts in a new timestamped directory: manifest.json, resolved_config.json,
seed_registry.json, protocol.md, source_snapshot/, screening_episodes.csv,
screening_ranking.csv, selection.json, confirmation_episodes.csv, summary.json,
decision_audit.jsonl, audit.json. Manifest binds every file SHA256/byte count,
Git commit/dirty status, runtime, expected/actual counts and phase status.
Decision audit logs first2 seeds/config/controller/phase (full) or all smoke
seeds, including current/next state, action, event bits and component costs.
Independent scalar replay checks every logged record and selected policy action.
Every episode audits feasibility and cost reconciliation; no outcome filtering.

Tests must cover known Hall deficits, feasible reservations versus proposed
collisions, physical choice versus mere labels, deadline reference agreement,
seed/selection separation, confirmation gates, exact grids and refusal to overwrite.
Mac smoke must finish, all audits pass, all25 configs be evaluated, selection file
hash stay unchanged through confirmation, component sums match and hashes verify.
Smoke performance cannot be promoted into the full setting choice.

## Handoff

Follow docs/experiment_workflow.md: tests + Mac smoke, verified commit/push, then
give the user one exact Ubuntu compound command and one exact rsync command.
No SSH, no assistant-launched lab run, no assistant-started transfer.
