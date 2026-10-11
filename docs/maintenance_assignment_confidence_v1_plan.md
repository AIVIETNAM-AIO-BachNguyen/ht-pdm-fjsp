## Material Passport

Origin Skill: ARS-Codex experiment-agent, inline authorized code experiment.
Date: 2026-10-12. Version: maintenance_assignment_confidence_v1.
Verification Status: protocol locked before new outcomes.

## Question and hypotheses

Can broader assignment-label coverage and a conservative learned override improve
fresh closed-loop H48+tail12 cost over the frozen additive PPO? Previous value-head
results (+0.479% dense, -0.084% skill-mask, uncertain; 94.4% labels at tick0) motivate
this prospective follow-up; no opened test states/labels enter this experiment.
PPO chooses the PM machine set and all CM edges. Only PM technicians can change.
Later trajectories/timing can diverge as a consequence; timing at each intervention
is fixed, not the whole episode schedule. No retraining of the PPO source.

Five controllers: frozen_ppo, natural_ungated, natural_gated,
broad_ungated, broad_gated. Both learned arms use identical three-member value
ensembles, paired initial weights and fixed optimization draws/updates. Natural
data reproduces source-visited 0/12/24/36 landmarks. Broad data gives half the
optimization draws to natural labels and half to transparent release probes at
6/18/30/42. This tests a data-coverage package, not only clock balancing.

Four primary contrasts: broad_gated vs frozen_ppo and vs broad_ungated, each
separately dense/skill_mask. Positive cost reduction must have 98.75% crossed
paired replica/profile bootstrap lower bound above1% for the pilot gain signal.
CI containing zero means unresolved; upper bound below1% challenges material gain.
Natural-vs-broad ungated and gated, and natural gate effects are secondary95%
intervals, exploratory. Three source replicas limit uncertainty calibration.

## Data and probe contract

Fresh configurations use the prior generator family, excluding every historical
pool and every assignment-value-v1 full/smoke configuration, plus mutually
disjoint new train/development/calibration/test/full/smoke configurations.
Source checkpoints are the same three dev-selected additive PPO models from
maintenance_assignment_scoring_v1_full_cpu_20261011T131859Z (de47c393), pinned
by the unchanged assignment-value source contract and verified before compute.

Full train64 profiles, development8, calibration8, test24; train/dev/calibration
one shock/profile, test4. Same physical tapes across source replicas/controllers.
Natural data records frozen PPO trajectories at0/12/24/36. Release probes are
generated valid states, NOT naturally visited states: at t-2, sample healthy
ages below failure threshold and zero/one in-progress eligible service with
remaining0/1/2/3 (bounded by service duration), then execute two REAL simulator
ticks with CM preserved and PM withheld. Record anchor, both transitions and
final state at t in6/18/30/42; use original remaining horizon, never reset H48.
Do not hide no-choice probes or resample until favorable outcomes. Report label
coverage by clock, condition, origin, occupied resources and source replica.
Natural closed-loop test initialization/distribution remains unchanged.

Enumerate all feasible PM machine subsets and assignments while preserving
source CM edges. Only classes with two assignments contribute labels; singleton
states/classes remain in coverage. Targets are paired normalized continuation
cost differences (scale20), with16 train,32 dev/calibration futures per root.
Continuation is frozen PPO until47 and deadline_matching at48..59. Save paired
MCSE; shared futures within phase, independent phases. No analytic oracle, private
future input, old labels or favorable-outcome filtering.

## Fixed training and gate

Three heads/ensemble, hidden32, existing relative-value architecture. Adam1e-3,
SmoothL1 pair regression, gradient clip1,80 epochs,1024 draws/member/epoch,
minibatch128; checkpoints20/40/60/80. Both arms match initial heads and update
count; natural draws are condition-balanced, broad splits draws evenly across
natural and probes then condition/clock-balanced within each origin. Sampling
with replacement and effective unique labels are recorded. Checkpoint selected
per member by common BROAD true-development pair MSE (earliest tie); no calibration
or test used for checkpoint choice. Changing MSE metric is a later experiment.

Ungated chooses lowest ensemble-mean cost among the SAME machine-set/CM class,
original wins ties. Gate predicts original-minus-alternative cost; subtract one
ensemble SD, then a nonnegative condition-specific empirical error margin.
Margin is the90th percentile (higher order statistic) of overoptimism relative
to paired MC mean gain minus1.96MCSE, on independent calibration source-vs-
alternative classes, pooled natural/probe calibration states. Minimum10 examples
per condition; otherwise overrides disabled, clearly reported. Margin fixed
before opening test. Only override if adjusted gain exceeds margin plus1e-5.
This is an empirical guard, NOT a proven confidence bound/safety guarantee.
No forecasts or MC labels at inference. Save all member deltas/disagreement,
margin, original and chosen matching, and rejection reason.

## Evaluation, stopping and artifacts

Full2880 episodes:24 profiles x4 shocks x2 conditions x3 sources x5 controllers.
Primary H48+tail12 total cost; secondary base/tail/components, per-replica gains,
leave-one-replica-out, latency, override rate and negative override magnitudes.
Held-out root diagnostic: first8 test profiles, first shock, four natural and
four probe landmarks,3 replicas,both conditions. Predict BEFORE32 independent
selection and64 evaluation futures for the source's SAME-machine/CM alternatives.
Report gains, harmful overrides, sign/MSE/calibration by natural/probe/clock,
all-state and opportunity denominators. A simulation reference selects on
selection only; not an oracle and not whole-episode performance. No timing arm.

Fixed budgets, no outcome-driven stopping, retries/resume or seed selection.
Full run stops on audit/schema/finite/seed/source error and retains FAILED
manifest, partial CSVs and checkpoints. Empty timestamped output directory only.
CPU/thread1; tqdm for label generation, training and multi-seed evaluation.

Smoke: same H48+12/hidden32, one source,2 train profiles, one per remaining split,
two futures/phase, two epochs/checkpoints1/2,64 draws/member/epoch,minibatch32;
calibration minimum1 engineering-only. Run full repository tests and Mac smoke;
every forecast scalar-replayed, same-machine/exact-CM isolation, source unchanged,
datasets/checkpoints roundtrip, counts/finite/gate determinism audited. Smoke
uses disjoint reserved configurations, does not open full test outcomes.

Artifacts: manifest/source contract/source/config snapshot, resolved physical
and RNG registries, plan, natural source trajectories and release probe traces,
label states/pairs, immutable forecast NPZs, tensor datasets and training draw
journals, progress/checkpoints/selection, calibration predictions/margins,
episodes.partial/episodes/decisions/PM choices/requests/carryover/coordination,
root predictions/forecasts/diagnostics, primary/secondary profile cells,
coverage and effective compute counters, hashes/bytes. Hash all artifacts except
manifest itself. Account simulator-label generation separately from optimization.

Commit/push only after tests+smoke. User runs full on Ubuntu and rsync on Mac per
docs/experiment_workflow.md; assistant never SSH/full-launch/transfers. No claim
of novelty, optimality, improved timing, indefinite-horizon or deployment transfer.
