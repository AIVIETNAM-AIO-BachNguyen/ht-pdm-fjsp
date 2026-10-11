# Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent
- Origin Mode: run, user-authorized implementation of the accepted pilot
- Origin Date: 2026-10-11
- Verification Status: PROTOCOL LOCKED BEFORE IMPLEMENTATION AND OUTCOMES
- Version Label: maintenance_autoregressive_pilot_v1

## Question and hypothesis

Does recomputing the remaining resource graph after each within-tick pair improve
closed-loop maintenance timing and technician allocation? H1: updated-context
autoregressive PPO lowers total episode cost relative to the identical decoder
whose resource context stays fixed. H2: it lowers cost relative to catalogue PPO.
These are falsifiable pilot hypotheses, not assertions of novelty or superiority.

## Environment and benchmark

N4/K2, original WaitingEnv physics and costs, learned H48 plus 12 physical tail
ticks controlled by the same deadline_matching rule. Full returns include tail,
gamma=1, reward=-objective/20, no reset or bootstrap at H48.
Two conditions: dense and structural skill_mask. Configuration-only generation
chooses random machine/technician roles; two flexible machines have both edges;
one specialist machine uses each technician in skill_mask. Durations 2..6 and
restoration {0.25,0.5,0.75,1}; threshold {3,4,5}, failure probability {0.25,0.45,0.6}.
At tick zero all machines are healthy and both workers idle. The focal flexible
machine and a specialist vary independently between young and at-risk ages,
four balanced-in-expectation scenarios; remaining ages are random. No scenario
is accepted/rejected based on planner, rule, RL decisions, costs, or optimal labels.
Every start therefore permits PM with two technicians and a competing machine;
this does not assert that any particular action is optimal. Random relabeling
prevents a fixed worker or machine identity defining scarcity.
Physical signatures canonicalize machine and worker permutations. Exclude all
registered historical joint training/development/test and diagnostic configurations.
New development/test/training and smoke/full configurations are disjoint.

## Arms and policy contract

Three newly trained arms on the same new distribution and tapes:
catalogue_ppo (existing MatchingPolicy joint action head), ar_context_ppo,
ar_frozen_context_ppo. A fourth reference is the best of the seven existing rules,
selected on development separately per condition, locked before test.
All neural arms share encoders, critic, actor dimensions, parameter counts and
exact initial weights per seed; actor inputs have different action semantics.
The decoder selects one of eight pairs or STOP, reserves that row and column,
then selects another pair or STOP. At most two pairs then mandatory STOP.
Physical time advances only once, after the complete matching. No timing rule
forces PM, CM, or waiting. STOP is always feasible. The context arm reaggregates
edge embeddings over the remaining feasible graph and pools remaining machines
and workers after each reservation; the ablation freezes these summaries at the
start of the tick. Both retain identical changing feasibility masks and prefix
length. Mask/prefix information is still available in the ablation; this tests
explicit resource summaries, not removal of all remaining-resource information.
The critic uses the original state only. PPO stores the ordered token sequence
including STOP and sums its conditional log probabilities; it never treats the
unordered matching probability as the sequence probability. Padding is -1 and
forced STOP contributes zero log probability. Entropy bonus sums conditional
entropies along recorded prefixes (a sampled-prefix PPO surrogate during updates).
Deterministic evaluation greedily selects tokens, rather than maximizing the
marginal probability of a whole matching. Catalogue evaluation maximizes its
whole-action probability. These factorization differences are part of H2;
only H1 isolates explicit context updates.

## Fixed budgets, seed split and stopping

Full is a pilot: 3 paired training seeds, 3 arms, 9 models. Per model 122880
learned decisions (2560 complete episodes), 32 vector environments, rollout3072,
40 updates, minibatch256, 4 epochs. Adam3e-4, clip0.2, value0.5, entropy0.01,
gradient clip0.5, hidden32. Nine models total1105920 learned decisions.
Evaluate initial and updates10/20/30/40 on development. Select minimum mean
extended cost among trained checkpoints only; tie selects earliest update.
Six development profiles x2 shocks x2 conditions; 24 sealed test profiles x4
shocks x2 conditions. Raw train16100000..2, dev16200000..5, dev shock16300000..1,
test16400000..23, test shock16500000..3, analysis16600000. Separate167xxxxx smoke
panels are recorded in configs/maintenance_autoregressive_pilot_v1_seed_registry.json.
Separate hashed roles for initialization, action RNG, minibatches, configurations,
initial scenarios, and physical shocks; paired arms share role tapes intentionally.
CPU only; Torch and OMP/OpenBLAS/MKL threads1. No adaptive extension, early stop,
hyperparameter sweep, checkpoint selection using test, or test-driven regeneration.
Failure retains partial journals and FAILED manifest. No automatic resume.
Smoke: H4+tail2, hidden16, vector4, 64 learned decisions/model, updates1/2,
one paired seed, separate development/test profiles. It is engineering evidence.

## Metrics and predeclared analysis

Primary: total closed-loop objective across H48+12 (also normalized cost/tick).
For each condition compare ar_context against catalogue and frozen_context:
four primary contrasts. Gain=(baseline-context)/baseline; minimum material
gain1%. Paired crossed bootstrap over 3 training replicas and24 profiles,
averaging the fixed four shocks within profile; 20000 replicates and two-sided
98.75% percentile intervals (approximate Bonferroni across four). Very few
training replicas limits uncertainty calibration; report all replica gains,
profile differences and leave-one-replica-out estimates. Confidence intervals
are pilot estimates conditional on the shock panel, not confirmatory guarantees.
Reference-rule contrasts are exploratory with95% intervals. Primary success
requires all four lower bounds >1%, and is only a signal to expand replication.
Mixed or null outcomes remain valid results; no RL-impossibility conclusion.
Do not infer good technician choice just because a PM action is feasible.

Secondary: base/tail costs, failure/waiting/PM/CM components, initial scenario,
wait when PM is feasible, STOP while a PM extension is feasible, PM starts with
two initially eligible workers, and PM-specific assignment alternatives holding
the selected machine set and every CM assignment fixed. Count CM-only choices
separately. Log PM choice sets, chosen technicians and relevant state features.
Behavior is descriptive; improvement in both choices ultimately needs paired
same-state timing/assignment interventions under the learned continuation.
PM mechanism contrasts do not replace total-cost endpoints. No scalability,
optimal-policy, publication-novelty or deployment claim from this N4/K2 pilot.

## Artifact schema and gates

Fresh UTC run folder; manifest includes settings, counts, commit, runtime, source
hashes, status and gates. resolved_config.json includes physical profiles and
initial states; seed_registry.json records raw panels and metadata exclusions.
Source snapshot includes all project Python modules and configuration registries.
training_progress.csv/episodes include full returns, likelihood checks and budgets.
development_episodes.csv, checkpoint_selection.json, training_coverage.json;
per-arm/seed checkpoints and model.pt; episodes.partial.csv, episodes.csv,
decisions.csv (selected audit trajectories with token traces and state replay),
pm_choices.csv (all learned-window selected PM edges with initial worker choices),
requests.csv, carryover.csv, coordination.csv, time_profiles.csv;
summary.json and profile_differences.csv. All JSON finite and all artifacts hashed.
Tests cover sequence likelihood and legality, STOP and padding, busy resources,
context ablation, permutation probabilities, future-event privacy, checkpoint
roundtrip, returns through tail, seed/split audits and CLI integration smoke.
Engineering PASS requires complete counts, finite updates, changed parameters,
all sequence likelihood audits, no conflicts, exact scalar/batch physical replay,
paired initialization and checkpoint selection locked before test.
Smoke keeps full_test_outcomes_opened=false and never executes a full test state.
Budget/progress bars plus partial journals monitor process and advancement.
No metric-triggered kill. Local smoke timeout10min; full laboratory run is user
controlled and fixed-budget, with no wall-clock-dependent scientific stopping.

## Handoff

Follow docs/experiment_workflow.md: tests and local Mac smoke, commit/push,
then supply the exact Ubuntu compound command and local rsync command. Assistant
does not launch the lab run or transfer results. No historical checkpoint needed.
