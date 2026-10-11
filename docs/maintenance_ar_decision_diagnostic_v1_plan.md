## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent.
- Origin Mode: run, inline; user approved the preceding proposed diagnostic.
- Origin Date: 2026-10-11.
- Verification Status: PROTOCOL LOCKED BEFORE NEW OUTCOMES; implementation pending.
- Version Label: maintenance_ar_decision_diagnostic_v1.
- Overall Confidence: exploratory mechanism study, not architecture confirmation.

## Question and falsifiable hypotheses

For the three development-selected ar_context_ppo models from the latest pilot,
is remaining H48+12 cost reduced by changing current PM timing, its technician,
or both, and does the learned policy assign low probability to beneficial roots?
H1: selection-chosen PM deferral improves independent continuation cost.
H2: selection-chosen PM technician correction improves independent cost with
exactly the same selected machines and corrective assignments.
H3: jointly choosing PM subset and its technicians improves independent cost.
Positive gain = original RL root cost minus correction root cost. All held-out
negative gains are retained. Zero/negative or uncertain estimates challenge the
corresponding hypothesis. No retraining and no claim of a deployable new policy.

## Source and splits

Source is maintenance_autoregressive_pilot_v1_full_cpu_20261011T112826Z,
commit 01f721b990ba090b10330197248a13795f0205bc. All three context checkpoints
16100000/1/2, development-selected updates 10/40/10, are frozen and hash pinned
in configs/maintenance_ar_decision_diagnostic_v1_sources.json. No best-seed choice.
Source test outcomes were previously opened and inspired this follow-up.

There are no new train or development outcomes and no hyperparameter search.
New diagnostic evaluation: 12 metadata-generated profiles x 2 physical shocks
x 2 skill conditions x 3 frozen replicas = 144 trajectories. Landmarks are
0,6,18,30,42, fixed before outcomes: 720 states, including no-choice states.
Raw roles are recorded in the companion seed registry: profiles17000000..11,
physical17100000/1, selection17200000, evaluation17300000, analysis17400000.
Smoke uses disjoint17500000-series roles, one source replica, one profile/shock,
both conditions and landmarks0,42. It retains H48+12 to test the real clock.

Profiles use the existing pilot generator and initial-state contract, but
canonical physical signatures exclude every historical pool, both pilot
train/dev/test pools, and other new full/smoke profiles. Exclusion is metadata
only; no filtering by outcomes, policy action, cost, or forecast benefit.
Physical trajectory tapes are shared across conditions/replicas per profile.
Future forecast tapes are fresh, independent of trajectory tapes and disjoint
between selection and evaluation; roots share each phase tape. The raw seeds
and derived RNG identities are audited for overlap. Smoke never executes new
full diagnostic trajectories/forecasts; resolving metadata is permitted.

## Root interventions

At every landmark use the frozen model's actual greedy root. Hold every
selected corrective-maintenance (CM, repair of a failed machine) edge fixed.
Only machines already selected for PM may be retained or omitted; no newly
chosen machine or CM delay is introduced. Enumerate feasible PM subsets and
all feasible technician assignments to retained PM machines. N4/K2 implies
at most seven unique roots, including the original root at index0.

- timing: all subsets with their original PM technicians; includes defer all.
- assignment: all original PM machines, alternate technicians, same CM edges.
- joint: union of all feasible subset/assignment combinations.

A timing intervention defers the current start for one tick. The original RL
may start it later, replace it with another PM, or never start it; no forced
rescheduling. Root0 wins exact selection ties. A no-choice state remains in
the grid, with zero exposure/gain for that contrast.

## Continuation and fixed budget

Original physical time is retained: root at t, RL at t+1..47 with feature
remaining=48-current_tick, common deadline_matching at48..59. Do not restart
48 decisions at the intervention or truncate the tail. Total remaining cost,
base and tail cost, four economic cost components, cutoff48 and terminal60
states are recorded separately. Greedy token inference matches the source
policy, including scalar fallback for logits within1e-5 of a tie.

For EACH state and candidate:32 selection futures,64 independent evaluation
futures. Choose each group's root using selection means only, BEFORE generating
any evaluation cost. Evaluate all roots on the independent tape for diagnostics.
Also select best_single from timing union assignment using selection only;
joint minus best_single gain measures incremental benefit of the joint menu.
The direct defer-all-vs-RL comparison is prespecified secondary, no selection.
Smoke uses2 futures per phase and scalar replay for every candidate/future.
Full upper bound:720 x7 x96 x40.8 average remaining ticks =19,740,672
physical forecast transitions (exact clock upper bound computed by runner).
Fixed completion, no outcome-driven stopping; CLI hard timeout120 minutes,
failure writes FAILED manifest, no automatic resume. tqdm shows trajectories
and state diagnostics. The Mac smoke timeout is120 seconds.

## Scores and mechanism outputs

Enumerate every feasible ordered pair/STOP sequence at the original state.
Sum sequence probabilities over pair orderings to get the unordered matching
probability; do not confuse this with one ordered-trace probability or greedy
argmax token selection. The complete matching distribution must sum to1.
Save sequence traces/log probabilities, matching masses and policy value.
Save token probabilities and STOP-vs-best-pair log odds along the actual greedy
prefix, distinguishing forced STOP from optional STOP. Save updated-vs-frozen
context total variation with fixed weights/mask at these prefixes. These are
functional sensitivity descriptions, not interventions on learning gradients.
Relate independent root cost differences and uncertainty to matching mass,
first STOP probability, prefix STOP scores, PM ages and choice availability.
A per-state positive paired95% lower bound is an exploratory label, not a
family-wise significant discovery; no hard state label is used to train.

## Analysis and interpretation

Six exploratory primary contrasts: timing, assignment, joint in each skill
condition. Report exposure coverage, switches, cost gain and99.1667% paired
crossed replica/profile bootstrap intervals (20000 draws; Bonferroni .05/6).
Fixed shocks and landmarks are averaged within cells, never counted as
independent profiles. Conditional exposure ratios resampled with numerators;
if no exposure or >5% empty bootstrap draws, mark INSUFFICIENT, not zero effect.
Three replicas limit interval calibration; this is a small follow-up, not power
confirmation. Report per-replica estimates and all signs, no favorable-seed filter.
Report joint-vs-best-single and defer-all as secondary; no new confirmatory tests.
Root Monte Carlo SE is paired across futures. Crossed CIs condition on the fixed
forecast tapes, so report MC uncertainty separately. Preserve denominators for
PM-free/forced-stop cases. Action-conditioned results describe this policy's
visited states; they do not establish population effect across other policies.
Below-risk-threshold PM is not automatically wrong, and non-wait does not imply
no implicit machine delay. A correction at one step is not an episode-level
policy improvement and simulated candidates are not a real-world oracle.
Do not infer why gradients failed from score/probability correlation alone.

## Artifact contract and gates

Fresh UTC run directory; no overwrite. manifest.json includes commit, dirty
status, versions, seed/source/physical audits, exact expected/actual counts,
clock counters, immutable parameter hashes, elapsed time, completion status
and every artifact hash. Save plan, source contract/seed registry, resolved
physical configurations, source snapshot and copied frozen models. Journals:
trajectories.csv, states.partial.csv/states.csv, root_values.csv, sequence_scores.csv,
prefix_scores.csv, rollout_audits.csv. Binary per-state NPZ stores paired phase
costs, base/tail, components, forecast tapes and cutoff/terminal states. summary.json,
profile_cells.csv and strata.csv retain coverage and uncertainty.

Tests cover CM-preserving candidate completeness, feasible partial deferral,
clock at0/42/47, paired tapes, scalar/vector observation and action equivalence,
full sequence probability normalization, independent selection/tie rules,
conditional bootstrap coverage, source tampering, fresh metadata and artifact
counts/replay. Run full existing suite and Mac smoke before committing/pushing.
Smoke must finish, every scalar replay match, copied checkpoint roundtrip match,
all finite valid costs reconcile, parameter hashes unchanged, and no new full
outcomes opened. Full must use clean committed source and single CPU threads.
Lab execution and transfer are user-run only, per docs/experiment_workflow.md.
