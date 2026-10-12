## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: authorized follow-up experiment engineering.
- Origin Date: 2026-10-12.
- Verification Status: UNVERIFIED for full science; smoke is engineering evidence.
- Version Label: maintenance_skill_contention_screen_v3.

# Screen assignment consequences after removing machine symmetry

## Amendment, question and hypothesis

V2 selected overlap3/common_1_3 and confirmed its registered gates. Subsequent
analysis found all598max-cardinality choice subsets on504contested confirmation
ticks were exchangeable: identical job states and service/restoration rows. The
two rules also had identical trajectories on128seeds. V2 artifacts, source and
verdict are immutable. This is a new exploratory follow-up with fresh panels.

Question: can strict competition for skilled technicians coexist frequently
with alternative assignments of the **same machines** that have different
committed service downtime cost? Hypothesis: unequal slow-worker durations by
machine create at least one eligible setting with this consequence, including
on subsets actually served by both fixed observation rules. No eligible setting
is a valid result; never force a winner.

No RL training, future-rollout search or teacher labels. We execute normal
environment episodes and enumerate feasible assignments from current observed
states. Train seeds and training budget: NOT_APPLICABLE.

## Fixed physical environment and48candidate grid

Keep N4/K2, H48 measured ticks plus L12 common deadline_matching continuation,
failure age4/conditional probability.45/max age8; PM1/CM2, unavailability6 per
machine-tick, failure15/event, overdue12 after waiting limit4. Repair completes
before restoration is applied; non-FIFO pending demand, partial matching and
WAIT retain their semantics. Only the machine-worker duration matrix varies.
All compatible edges restore.75 in selectable candidates; no dense mask.

Four masks, rows M0..M3/columns T0,T1:

| Topology | Rows |
|---|---|
| overlap3 | 10,11,11,11 |
| overlap3_reverse | 01,11,11,11 |
| balanced2 | 10,01,11,11 |
| bottleneck2 | 10,10,11,11 |

T0 has uniform fast duration f on all its compatible edges. T1 has duration
f+2 on M0, and a permutation of the three slow durations on M1,M2,M3. Cross
f=1 with all six permutations of(2,3,5), and f=2 with all six permutations of
(3,4,6). Cross these12patterns with the four masks: **48distinct selectable
configurations**, all with strictly slower T1 on shared machines and unequal
losses from using T1 across those machines.

The previously suggested slow durations(3,5,2) are included. Example with f1:
M0=(1,0), M1=(1,3), M2=(1,5), M3=(1,2). For M1/M2, alternatives have total
durations1+5=6 versus3+1=4; their committed downtime costs differ by12.

Five nonselectable controls (total **53settings**):

- overlap3/uniform_control: v2 winner, rows(1,0),(1,3),(1,3),(1,3), restore.75.
- overlap3/equal_control: all compatible edges duration3/restore.75.
- overlap3/tradeoff_control: v2 incomparable fast2/restore.5 versus slow4/restore1.
- overlap3/cancellation_control: rows(1,0),(2,4),(1,3),(2,4), restore.75. Machine
  rows differ, but the slow-minus-fast penalty is uniformly2. Swapping workers
  between the same shared-machine pair has identical total service time. This
  rejects the weaker notion that merely unequal machine rows imply economic
  allocation consequences.
- partitioned_control/heterogeneous_control: mask10,10,01,01 with T0 durations1
  and T1 slow vector(3,3,5,2), restore.75; no worker alternatives.

These matrices, masks and cutoffs are new design decisions, not claimed to be
literature-standard or optimal. Fast, slow and restoration meanings match the
existing model; durations are integers and costs are per physical tick.

## Metrics: registered consequence rather than label changes

Keep v2 urgent machines (failed or age>=3, excluding in-service) and all feasible
partial matchings A using current free workers. Keep all nondominated edges per
machine: lower/equal duration and greater/equal restoration, one strict to
dominate. Let F=max cardinality(A), P=max cardinality using only nondominated
edges. **skill_contention_rate** remains fraction of measured ticks with P<F.

For each machine subset S of size F, enumerate its feasible assignments A(S),
holding the same machines and current worker capacity fixed. Define
committed_cost(a)=unavailable_cost*sum(duration(m,j) for(m,j) in a), and
span(S)=max committed_cost-min committed_cost over A(S).

- **cost_sensitive_contention_rate**: P<F AND at least one size-F subset has
  span(S)>=one tick of unavailable_cost (6). Uniform cancellation, swapped labels,
  and empty/forced-unique actions do not count.
- **selected_cost_sensitive_contention_rate**: P<F AND the rule's selected
  subset has size F and span>=6. This prevents counting hypothetical subsets
  the observation rules never actually serve, or exploiting an underfilled action.
- Auxiliary: v2 joint/sensitive/selected physical-choice rates; mean maximum
  committed cost span across size-F subsets; mean selected span; nondominated
  deficit; capacity and compatibility pressure; legacy proposals; availability,
  utilization, pending/failed at H and H+L; full cost decomposition.

All rates use **all48measured ticks**. Local spans are deterministic observed-state
mechanism quantities, with no future shocks read. They are not sampled rollouts,
Q-values, labels for RL or proof of whole-episode savings. Because service is
nonpreemptive and unavailable price is constant, sum(duration)*6 is the exact
obligation from the new jobs' own in-service ticks through completion; PM/CM
start charges cancel for the same subset. L12 exceeds max duration6, avoiding
truncating these jobs' service obligations at the measured horizon. Later failures,
restoration effects and crowding of other jobs may offset local savings.

Both observation policies remain v1/v2 deadline_matching and priority_greedy.
Their code/scoring is unchanged. The metric does not depend on score tie-breaks.
Same raw seed creates the same initial ages and machine/time event tape across
candidates and rules, not necessarily the same actions or state occupancy.

## Selection and five registered confirmation gates

Screen64seeds per setting/rule. Eligibility requires all:

1. min-rule skill_contention_rate>=.10;
2. min-rule cost_sensitive_contention_rate>=.05;
3. min-rule selected_cost_sensitive_contention_rate>=.03;
4. min-rule availability>=.25;
5. max-rule terminal_pending<=1.

The consequence gates correspond to2.4possible and1.44actually-served contested
allocation opportunities per48tick episode on average. Rank eligible candidates
by min-rule primary descending, then min-rule cost-sensitive rate descending,
then min-rule selected cost-sensitive rate descending, then ID ascending. Never
rank by realized cost, span magnitude, RL reward or rule performance difference.

Freeze winner, runner-up if present, and all five controls; hash selection.json
before confirmation begins. With no eligible candidate, confirm controls only
and report NO_ELIGIBLE_SETTING. No replacements after a failed confirmation.

Full seeds: screening33000000..33000063, confirmation33100000..33100127,
analysis33200000. Smoke: screening33300000..33300003,
confirmation33400000..33400003, analysis33500000. Physical role
'maintenance_skill_contention_screen_v3:physical'; bootstrap role
'maintenance_skill_contention_screen_v3:bootstrap'. Check unique raw/derived
panels across profiles, all historical registry raw IDs, and derived panels of
v1/v2. Full panel definitions may be read for overlap audits; smoke never
executes full episodes or opens their outcomes.

Confirm only frozen winner on128newseed pairs. Cluster-bootstrap20,000resamples
of seed pairs, preserving both rule observations; use min/max across rules on
each draw. **Five two-sided99% CIs** (Bonferroni five gates, nominal family-wise
coverage>=95%) must each satisfy gate bounds: lower bounds for four rates/health,
upper bound for pending. Runner-up/controls remain diagnostics. Rule cost gaps
are exploratory and bundle machine selection plus assignment, not isolated RL
benefit. Confirmation applies only to this fixed setting under these rules.

## Budget, stopping and artifacts

Full:53*2*64=6,784screening episodes; at most7*2*128=1,792confirmation episodes;
maximum **8,576episodes /514,560transitions**. Smoke:424screening and at most56
confirmation, maximum480episodes /28,800transitions. Complete locked panels with
no outcome-based stopping, timeout600s smoke/7,200s full. On failure preserve
FAILED manifest and partial logs; no resume; create a fresh timestamped directory.

Artifacts follow v2: manifest.json (metric/protocol/commit/runtime/hashes/counts,
generic and full confirmation openness), resolved_config.json, protocol.md,
seed_registry.json, source_snapshot/, screening_episodes.csv, ranking.csv under
screening_ranking.csv, selection.json, confirmation_episodes.csv,
decision_audit.jsonl, audit.json, summary.json and adjacent console log. Journal
includes all measured-state consequence metrics, action/state/events/next state
and costs for first2full/all4smoke seeds of every candidate/rule/phase. Snapshot
includes v1/v2 helper modules and frozen seed registries used in overlap audit.
Reject nonempty output dirs. Show tqdm during both multi-seed phases.

## Engineering and scientific gates

Tests must cover committed cost differences on the same subset, uniform and
nonuniform cancellation, no future information, no underfilled selected gate,
capacity versus skill competition, exact tie retention, worker-label invariance,
restoration tradeoff exclusion, negative controls, both-rule gate checks,
ranking that ignores cost, no-winner path, fresh seeds, paired shocks, metric
tamper replay, artifacts and overwrite refusal. Engineering smoke requires
counts, feasibility, transition/metric replay, controls, hashes, seal and progress
PASS; scientific CONFIRMED is not required on four seeds. Checkpoints N/A.

No retuning of grid/gates/seeds after smoke. Run full suite plus macOS smoke,
commit/push verified files, then give exact lab and rsync handoff per
docs/experiment_workflow.md. Never SSH to lab, launch full run or transfer for user.

## Interpretation safeguards

Keep rule results separate; two labels may still yield the same behavior.
Seed is the resampling unit, not tick. Retain all failures/unserved demand.
48candidate screening is exploratory; fresh winner confirmation tests only the
registered gates. Same-subset committed cost difference demonstrates one
economic component, not lower optimal total cost or a guaranteed RL improvement.
No reranking on test, silent protocol replacement or physical-profile
generalization claim. A failed gate is reported and requires a new follow-up.
