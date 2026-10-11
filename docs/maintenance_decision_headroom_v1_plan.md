# Maintenance decision headroom v1 — prospective protocol

## Question and material passport
Does a frozen centralized matching PPO leave recoverable one-step opportunities
in PM deferral or technician assignment, on fresh physical profiles, when the
same RL policy continues? This diagnostic motivates a mechanism, not a claim
that a new algorithm is already necessary or superior.

Source: the ten development-selected JOINT checkpoints from
maintenance_joint_rl_v1_full_cpu_20261011T045434Z, archived verbatim in
maintenance_timing_diagnostic_v1_full_cpu_20261011T072307Z. The latter is pinned
by manifest SHA256 2d9cf8caca189635e62239521a340a08860b7c8387e75c85394985a8581f0a27.
Historical results have already been opened. No historical state outcomes enter
new estimators. No new training or development selection is performed.

## Interventions and clock
At each sampled state use deterministic frozen RL to obtain the root action.
Timing candidates: original root or remove every preventive (healthy-machine)
start, keeping every corrective pair and existing service unchanged. Assignment
candidates: every feasible matching with exactly the original selected machine
set; only technician identities can change. Empty roots have no assignment choice.
Candidates are deduplicated (at most three roots for N4/K2).
After this single root intervention the SAME frozen RL model selects all later
actions until global tick 47, with remaining-time observation 48-tick. Every root
continues through the same deadline_matching tail at ticks 48..59. Do not restart
a 48-tick horizon. Do not truncate in-progress service at tick 48.

## Independent selection and evaluation
Use 32 private CRN scenarios to select the minimum-cost timing candidate and,
separately, assignment candidate. Order the original RL root first; exact ties
retain it. Lock choices before opening 64 independent evaluation scenarios.
Evaluation tapes are paired across all candidate roots but disjoint from
selection and physical tapes. Retain negative held-out gains; no clipping or
reselecting by evaluation costs. Forecast RNG is never visible in policy features.
Selectors are simulator-assisted root corrections, not implementable online
policies or a true optimal oracle. Selection noise can understate available gains.

## Primary hypotheses and metrics
Four primary contrasts: dense/skill_mask x timing/assignment. Estimate mean
C(original RL root) - C(selection-chosen root), conditional on a structural
choice for that intervention (PM root for timing; multiple feasible assignments
for assignment). Positive favors the one-step correction. H0: mean gain <= 0.
Use the complete remaining simulator cost through 59; also report cost per
remaining machine-tick, paired MC SE, intervention coverage, selector switch
rate, original/selected cost components, base/tail costs and signed per-state
gains. Direct original-minus-defer PM contrast is descriptive.

Use 20,000 crossed bootstrap draws of ten model replicas and eight physical
profile identities. Keep repeated shocks, origins, ticks and conditions paired.
Four percentile intervals use 98.75% coverage (approximate Bonferroni familywise
95%). No Monte Carlo noise resampling; inference is conditional on fixed tapes.
Empty bootstrap fraction >5% or zero eligibility gives INSUFFICIENT. All
non-switched eligible states remain in the denominator with exactly zero gain.
Report RL-origin-only means separately; reference-origin states diagnose a
broader state distribution and do not establish deployable policy improvement.

## Fixed budget, seeds and stopping
Full: both dense and skill_mask; model seeds 12100000..12100009; fresh cohorts
15100000..15100007; physical shocks15200000/15200001; origins joint_ppo and
best_rule; landmarks6/18/30/42. Exactly 2560 states from640 trajectories of 48
intervals. selection15300000, evaluation15400000, bootstrap15500000, all
role-derived. Regenerate exclusions for all original training/development/test
physical signatures plus the prior diagnostic full/smoke profiles. New full and
smoke physical signatures must be disjoint. No outcome-dependent filtering.

Smoke: separate cohort15600000, shock15600010, selection15600020,
evaluation15600030, bootstrap15600040; first frozen model, both conditions and
origins, landmark42. Four states, four selection/evaluation scenarios, 200
bootstrap draws. Validate scalar replay of ALL roots/scenarios in smoke.
Smoke results are engineering evidence only; full profile never executes on Mac.
No early stopping, seed additions, teacher-label tuning or full resumption.
On failure retain partial journals and FAILED manifest; restart in a new UTC dir.

## Gates and artifact schema
Tests: legal same-machine reassignments including blocked workers/skill masks;
PM removal preserves CM; selection/evaluation separation and signed estimates;
exact batched/scalar clock, components, boundary states and feasible actions;
bootstrap sparse/no-switch eligibility; source/seed guards and output refusal.
Smoke: source hashes and model roundtrip pass, full frozen physical panel still
unopened, all four states and 192 live trajectory transitions complete, progress
visible, immutable parameters, source/effective config saved, scalar replay pass.
Full: clean committed code and CPU thread1, exact 2560 state grid and 30720 live
transitions, source hashes, feasibility, clock/cost/counter audits all pass.
A passed engineering gate does not require a statistically positive result.

Each new UTC directory contains manifest.json (commit, runtime, hashes, counts,
source/seed/physical audits), source_contract.json, resolved_config.json,
seed_registry.json, plan.md, source_snapshot/, frozen_models/seed_*.pt,
trajectories.csv, states.partial.csv, states.csv, root_values.csv,
rollout_audits.csv, raw_forecast_costs.npz (state x phase x padded-root x scenario;
selection padded with NaN after 32 for full; root-valid/scenario-valid masks),
raw_boundary_states.npz, profile_cells.csv, summary.json, and strata.csv.
Root ordering, masks, selection/evaluation role seeds and selected indices are
recorded so held-out estimates can be reconstructed. Log is a sibling .log file.

## Interpretation limits
Timing only tests deferring PM that RL already chose. It does not diagnose
missed PM starts or identify an optimal waiting duration. Assignment only tests
machine sets selected by RL; report opportunity coverage before judging effect.
No gain may reflect limited benchmark opportunity or noisy root selection;
it does not prove optimality. Positive gain supports a targeted algorithmic
hypothesis; a new trained method still requires independent end-to-end testing.
