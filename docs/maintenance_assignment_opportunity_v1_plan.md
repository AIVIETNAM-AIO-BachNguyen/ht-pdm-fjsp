## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-12
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

# Natural assignment opportunities and paired-label precision

Protocol: `maintenance_assignment_opportunity_v1`. Prospective follow-up to
the opened confidence-v1 test. That run diagnosed zero gated overrides,
insufficient skill calibration, and weak transfer from synthetic probes.
Its outcomes motivate this design but never select this run's labels,
checkpoints, margins, seed panels, or stopping rule.

## Hypothesis and comparisons

H1: collecting the actual source-PPO PM assignment class at every naturally
visited opportunity, with clock-stratified sampling, improves closed-loop cost
relative to landmark-only sampling. H2: 128 common-random-number futures per
target, rather than 16, improves assignment predictions and decisions. H3:
calibration with more naturally visited opportunities makes the empirical
guard usable while retaining cost gains. Each can fail; zero gated overrides
is a reported mechanism failure, not an engineering failure or a reason to
lower thresholds after test.

Freeze the same three additive PPO sources (18100000/1/2) pinned by the existing
assignment-value source contract. Environment: N4/K2, dense and skill mask,
original clock H48 + common deadline-matching tail12, CPU. At each intervention
preserve the source PM machine set and every exact CM edge; only technician
assignment changes. Subsequent closed-loop timing can diverge.

Three ensembles, each with three 7,745-parameter relative-value heads:

1. `landmark_low`: naturally visited source assignment opportunities only at
   ticks 0/12/24/36; 16 futures; condition-balanced uniform label sampling.
2. `opportunity_low`: all natural opportunity ticks 0..47; 16 futures;
   condition-balanced and uniform over available 12-tick clock bins.
3. `opportunity_high`: the identical states and draw indices as opportunity_low;
   128 futures. Low targets use the first16 scenarios of each saved high tape,
   preserving pairing across every future tick. Full forecasts are computed
   once and shared; this is a label precision ablation, not an equal simulation
   compute claim. Landmark rows are a subset of those same low targets.

No synthetic probes and no labels from other PM subsets. The landmark arm is
a matched new ablation, not an exact rerun of confidence-v1's all-subset arm.
Evaluate frozen PPO plus ungated/gated versions of all three arms (7 controllers).
Initial weights are paired across arms; low/high opportunity sampling is exact
paired. Adam .001, SmoothL1, clip1, 80 epochs/head, 1,024 draws/epoch, minibatch128.
Checkpoints20/40/60/80 selected separately by common opportunity development MSE
with128 futures; earliest tie. Total27 learned heads,17,280 optimizer steps.

## Splits, calibration, and stopping

Registry `configs/maintenance_assignment_opportunity_v1_seed_registry.json`
contains exact seeds. Full: 128 train profiles ×2 shocks,32 development ×2,
64 calibration ×4,24 sealed test ×4. All three sources and both conditions.
All physical signatures exclude earlier historical/full/smoke panels and are
disjoint across current splits. Raw and derived RNG-role collision audits
include the earlier confidence and assignment-value experiments. Source PPO
seeds and intentional paired streams are the only planned reuse.

Run every preallocated trajectory to48, retain every source assignment class
with two feasible assignments, and forecast only those roots. Do not resample
profiles, extend collection, or stop early based on cost, head outputs, sample
counts, or test outcomes. If a training condition has no labels, fail before
training/test. Log missing clock bins; never invent opportunities.

Common development and calibration targets use128 futures, independently seeded
from train. Guard recipe stays fixed: adjusted gain = ensemble mean gain −1 SD;
margin = max(0,90th percentile [adjusted predicted gain −(observed gain −1.96 MCSE)])
using higher quantile. Enable per arm/source/condition only with at least40
comparisons spanning16 physical profiles and10 late (tick>=12) opportunities;
otherwise retain PPO and report the missing criterion. Repeated shock/state
rows are correlated, not40 independent samples. No statistical safety guarantee.
Record development guard coverage and paired gain after calibration, without
using them to retune the fixed gate. Lock checkpoints/calibration before opening
test. Prior opened tests are not used for optimization.

Test every controller on identical physical tapes: 4,032 episodes,
193,536 base transitions and48,384 tail transitions. Independent root diagnostics
on first8 test profiles/first shock; natural opportunities only,32 selection and
64 evaluation futures. Record learned actions before forecasts; independent
reference selection/evaluation is not an oracle. No timing-improvement claim.

## Metrics and decisions

Primary: relative reduction in extended cost. Four contrasts:
opportunity_high_gated vs frozen PPO and vs opportunity_high_ungated,
separately dense/skill. 20,000 crossed source/profile bootstraps after averaging
the4 shocks;98.75% intervals (approximate Bonferroni), practical margin1%.
Pilot gain signal only if lower bound >1%; below margin if upper bound <1%;
otherwise unresolved. Three source replicas limit inference.

Secondary95% exploratory: opportunity_low vs landmark_low, opportunity_high vs
opportunity_low, each gated/ungated; gated vs ungated by arm. Report all comparisons,
per-source and leave-one-source-out gains. Mechanisms: opportunity counts by
source/condition/clock/profile, target MCSE low/high, sign disagreements, dev
MSE, calibration criteria/margins, eligible/rejected/override counts, single-root
harm and benefit, all cost components, and amortized CPU inference latency.

## Artifacts and engineering gate

New empty UTC-timestamped directory per run; no resume/overwrite. Manifest records
commit, dirty state, platform/runtime, frozen-source hashes, config/seed audits,
expected/actual counts, compute, status/errors, and all artifact SHA256+bytes.
Save source snapshot, resolved physical config, protocol, registry, source contract;
natural trajectory and opportunity coverage CSVs; label_states/label_pairs with
state/root/tick/origin and nested-scenario provenance; full forecast NPZ with shocks,
components and boundary states; all datasets/checkpoints; training_draws/progress;
checkpoint_selection, calibration/predictions, development_guard JSON;
episodes.partial/final, decisions, PM choices, requests/carryover/coordination,
root_predictions/root_states, summaries and profile differences.

Smoke uses independent panels,1 PPO source,3 ensemble members,2 epochs,64 draws,
minibatch32,2/4 low/high futures,4 development/calibration futures,2 selection/
evaluation futures. It runs scalar replay for every forecast path and verifies
all controller scalar/batch actions. PASS requires completed grid, finite trained
heads, save/load exact prediction roundtrip, legal PM/CM preservation, frozen PPO
unchanged, artifact/count/compute audits, nested-target and paired-draw contracts,
progress bars, and no opened full-test outcomes. Scientific gain is not a smoke
requirement. The near-tie scalar fallback checks alternatives only in this new
controller; prior experiment code stays unchanged.

Run required tests and Mac smoke, commit/push, then hand off the exact compound
CPU command for the user at bachng@100.111.83.52 and exact rsync command. Never
SSH/run full lab or transfer on the user's behalf. Expected full artifact size
is several hundred MB depending on opportunity counts; fixed trajectory budget,
at most two forecast roots/state, no data-dependent budget expansion.
