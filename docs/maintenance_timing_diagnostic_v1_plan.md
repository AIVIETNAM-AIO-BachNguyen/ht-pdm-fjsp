# Maintenance timing diagnostic v1 — locked prospective follow-up

## Material Passport

Origin Skill: academic-research-suite / experiment-agent; Origin Mode: inline
plan/run; Date: 2026-10-11; Version: maintenance_timing_diagnostic_v1.
Verification Status: UNVERIFIED pending the engineering gate. User authorizes a
small diagnostic of preventive timing and WAIT on identical states. The source
joint-RL result motivated this follow-up; hypotheses are exploratory. No model
training, checkpoint reselection, reward change or deployment claim.

## Question, hypotheses and controlled comparisons

H1: when frozen joint PPO starts preventive maintenance (PM), deferring those
PM starts for the current interval can have lower subsequent expected cost.
Intervention removes ONLY healthy-machine starts from the PPO root matching;
corrective starts and their technician assignments remain exactly unchanged.
Continuation decides freely from the next interval; this does not promise PM
will start one interval later. Also record single-PM deletion marginals while
holding every other root pair fixed.

H2: on states with any feasible non-WAIT action, PPO's root decision can cost
more than the development-selected strong rule's decision. This comparison
changes the whole matching, including allocation, and is NOT a timing-only
causal effect. Descriptive WAIT/all-PM/CM/mixed agreement and state-feature
tables distinguish early PM, delayed PM and resource occupation.

For each sampled state evaluate EVERY feasible partial matching, including WAIT,
using 32 identical Monte Carlo future shock tapes for every root. From interval
1 onward ALL roots follow deadline_matching for a fixed total24 physical
intervals; no resets, critic bootstrap or zero-cost terminal repair. Existing
repair remaining/owners carry into every fork. State, physical law and random
tape are identical within each fork comparison. Exact root feasibility checked.
This estimates one root action's value UNDER THIS COMMON CONTINUATION and finite
window, not Q under the original learned policy or infinite-horizon optimality.
The sample minimum across roots is an optimistic Monte Carlo diagnostic,
never an oracle label or calibrated optimality bound. No future trajectory
tape is exposed to PPO or used as its forecast.

## Frozen source and fresh physical panels

Source run maintenance_joint_rl_v1_full_cpu_20261011T045434Z, commit
ed032858ebf174738e9c00bfe2dce72b7fff4277. Freeze all ten development-selected
joint_ppo/model.pt checkpoints for seeds12100000..12100009 by SHA256 and source
manifest SHA256. Never choose a best source seed. Strong references inherited:
dense adaptive_isolated_matching, skill_mask deadline_matching.
Check source completion, all source hashes, snapshot/current dependency bytes,
checkpoint contracts/settings/updates and selected-model bytes before any new
environment runs. Input source artifacts are read-only and need no transfer by
the assistant. Missing/corrupted source fails preflight before outcome opening.

Use the SAME independent duration2..6/restoration{.25,.5,.75,1} generator and
WaitingEnv economics as source, N4/K2/H48, dense and skill_mask. New cohorts are
label/horizon-invariant physically disjoint from all prior metadata, source
train/dev/test, new smoke and new full. Regenerate the source's 51,200 training
configurations from frozen seeds and exclusion metadata to reject their physical
signatures too; train reuse within source is allowed. Verify resolved source
dev/test metadata. No selection of profiles/states by action gap or cost outcome.
No new training/dev split: source models and reference rules are frozen;
all fresh full configurations are diagnostic evaluation, not model selection.

Raw full seeds: cohorts13100000..13100007, physical shocks13200000..13200001,
forecast root13300000, bootstrap13400000. Smoke cohorts13500000,
physical13500010, forecast13500020, bootstrap13500030. New raw registry contains
ONLY these fresh roles; reused source model seeds live in source contract.
Audit inherited raw registries, new full/smoke separation and derived roles.
Physical environment tape indexed by cohort/shock, shared across models and
conditions. Forecast seed indexed by cohort/shock/model/origin/tick, independent
of environment tape. CRN paired across actions within each state.

## Sampling, budget and stopping

Full: all10 frozen models x2 conditions x8 cohorts x2 shocks x2 state origins
(on-policy PPO and inherited rule) =640 trajectories. Advance each 48 intervals.
On each trajectory take BEFORE-action states at fixed ticks{6,18,30,42}, even
when both technicians are busy or all actions agree. No cherry-picking disagreement
states. 2,560 state records; forced WAIT states retained with zero choice exposure.
Each model sees both origins; rule-origin trajectories may repeat across models,
declared intentional paired use, not independent observations. No state
deduplication that would change occupancy weighting.
At most21 candidates/state x32 scenarios x24 intervals =41,287,680 forecast
transition samples; record actual candidate/scenario/tick counts. Trajectory
ticks30,720. Singleton WAIT can skip valuation reference choice work, but still
simulate the same32x24 physical transitions and retain its absolute cost.

Smoke: first NUMERIC source model seed only (engineering, not best-seed
selection), both conditions, one fresh cohort/shock, both origins, trajectory
prefix4 with sampled before-states{0,2}; model and rule still receive ORIGINAL
remaining learned horizon48-t. 8 states, 16 trajectory ticks, 4 scenarios and
forecast6 intervals; maximum4,032 forecast transition samples. No new full
panel outcomes run on Mac; metadata construction is allowed. Save/reload
frozen copied models in smoke and compare root action/log-prob/value exactly.

Fixed complete budget; no early success stopping, retry/resume or extra profiles.
All roots, forced states, censoring/busy resources and errors retained. New
timestamp output directories only. Failure/interrupt writes FAILED manifest,
partial states/actions/trajectories and hashes. tqdm for models, trajectories,
fork evaluation and physical split regeneration. Full CPU/thread1, clean code.

## Metrics and uncertainty

Four primary descriptive contrasts, one H1 and H2 per condition. H1 estimates
mean(C(PPO)-C(PPO without PM)) conditional on PPO PM; H2 mean(C(PPO)-C(rule))
conditional on structurally having a non-WAIT option. Positive favors deferral
or rule. Normalize by4x24 machine-intervals. For ineligible cells store zero
numerator and zero exposure; estimate as sum(delta)/sum(exposure), not a mean
of subgroup ratios. Report exposures and negative/zero/positive signs separately.
H1 is effect of root PM removal on the sampled states under common continuation,
not a guaranteed improvement from permanently changing the learned policy.

Paired crossed bootstrap independently resamples ten training replicas and eight
physical profiles, 20,000 shared draws. Keep shocks, landmark ticks and two
origins paired within each cell. Conditional ratios recomputed within each
draw; report empty draws; if >5% empty, mark uncertainty insufficient instead
of fabricating CI. Four two-sided98.75% percentile intervals (Bonferroni family),
approximate coverage; full remains a small exploratory diagnostic. Monte Carlo
SE of each paired state delta retained; intervals conditional on fixed forecast
tapes and physical shock panel. Monte Carlo sign uncertainty is not silently
treated as known true regret. No per-state p-values or outcome-adaptive resampling.
Criterion: CI>0 HIGHER_ROOT_COST, CI<0 LOWER_ROOT_COST, otherwise UNRESOLVED;
zero-exposure INSUFFICIENT. Smoke always ENGINEERING_ONLY, never GO/NO-GO for RL.

Secondary: root immediate and forecast cost, all component costs, WAIT versus
PM/CM/mixed labels, agreement, WAIT with a choice, ages relative to failure
threshold, current hazard, skill graph, remaining busy times, PM duration and
restoration, single-PM deletion effects, empirical root-minimum gaps, PM versus
rule-WAIT subgroup means. Binned descriptive tables never establish causality
or cross-profile generalization. Report both conditions and both state origins.

## Artifacts and workflow

manifest/source_contract/source hashes/protocol/config/fresh seed+physical
registries/exclusion audit; frozen model copies; trajectories.csv,
states.partial.csv/states.csv, root_values.csv with scenario cost arrays,
preventive_pairs.csv, rollout_audits.csv, raw_forecast_costs.npz (fixed catalogue
padding with NaN for infeasible entries), summary.json, profile_cells.csv,
behavior/state-feature tables, timings and counters. All outputs hashed.
Source snapshots include full physics/policy/reference/diagnostic code and lockfile.
Model parameters verified unchanged throughout; learning NOT_APPLICABLE.

Tests: pinned input rejection, raw/physical split audits, same-state/CRN pairing,
scalar replay of all candidate branches on engineering states, keep-CM/drop-PM
and marginal interventions, preservation of repairs across forks, no physical
future access, exact counts/grid/conditional bootstrap, checkpoint round trip,
forced WAIT, nonempty-output rejection, interruption before sealed outcome work.
Whole repository tests and Mac smoke first, then commit/push, exact user lab
command with --source-dir and checkpoint preflight, then Mac rsync. No assistant
SSH, full lab execution or transfers. Estimate compute/storage from engineering
states only; source training is never repeated.
