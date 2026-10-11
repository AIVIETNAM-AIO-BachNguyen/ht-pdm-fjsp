## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent, inline.
- Origin Mode: authorized code experiment, local tests/smoke then user-run lab.
- Origin Date: 2026-10-11.
- Verification Status: PROTOCOL LOCKED BEFORE NEW OUTCOMES.
- Version Label: maintenance_assignment_value_v1.
- Overall Confidence: small policy-improvement pilot; not end-to-end/new-algorithm confirmation.

## Question and hypothesis

Does learning relative continuation cost between feasible PM assignments help
select technicians better than frozen PPO probability/greedy scores? Freeze the
three development-selected ADDITIVE PPO models from assignment_scoring_v1
(commit de47c39379f1ae1e8c4d0e64745b90cb6322c550, updates40/30/10). All model
bytes/source artifacts are pinned in the companion source contract. Their old
test results are already opened; do not reuse those states or outcomes as labels.

PPO still chooses the PM machine set and every CM edge. A separate learned
matching-value head selects PM technicians within that same set. This isolates
an assignment-learning step, not joint end-to-end learning of PM timing and
assignment. Timing at the same state is held fixed by construction; subsequent
states/timing can diverge in closed-loop episodes. Primary H1: true-label value
head reduces fresh H48+tail12 episode cost vs the frozen greedy PPO. H2: it
outperforms a parameter/init/update-matched state-shuffled-label value head.
Zero/negative/uncertain gains challenge those hypotheses.

## Data and target

Fresh N4/K2 dense/skill_mask configurations, same generator family and role
permutations. Freeze canonical configuration signatures before running outcomes;
exclude every historical physical pool, both previous assignment benchmark
train/dev/test/smoke panels, and all new full/smoke splits. Seed registry records
all raw roles and derived RNG role collisions are checked.

Training:64 profiles x1 shock x2 conditions x3 frozen source replicas, landmarks
0/12/24/36 =1536 source-visited states. Enumerate every feasible PM machine subset
and its assignments while holding the source's selected CM edges EXACTLY fixed.
All roots (at most21) are deduplicated; assignment comparison classes have one
or two members. Training uses16 common-random-number futures/root. Costs include
root at original t, frozen PPO at t+1..47, common deadline_matching at48..59.
Each pair target=(right cost - left cost)/20, paired MC SE saved. All negative,
zero and positive labels retained; no favorable-gap/state filtering. Singleton
states/classes retained in coverage, contribute no pair regression target.
No analytic oracle or previous test labels; these are new simulator returns
under the frozen RL continuation policy. The value head estimates Delta Q^pi,
up to a state/group offset; it is not a calibrated absolute Q-value.

Development:8 new profiles x1 shock x2 conditions x3 sources, same4 landmarks,
32 fresh futures/root, never pooled into training. Fixed80 epochs, Adam1e-3,
minibatch128, SmoothL1 on paired Delta targets, gradient clip1, hidden32.
Checkpoints20/40/60/80; select minimum true development pair-MSE, earliest tie.
Truth and shuffled heads share architecture, initial weights, minibatch order,
number of labels and optimization updates. Shuffled targets permuted only within
training condition for each replica; save the permutation. Both use true dev
labels for checkpoint selection. No dev/test tuning or best-source-seed choice.

Head uses observable machine/technician/edge features and detached frozen PPO
state embeddings. Trainable pair encoder and whole-matching pooling permit
assignment interactions; no entity IDs or private future events as input.
Inference enumerates only the same-machine/same-CM matching class and minimizes
predicted relative cost. Ties within1e-5 normalized cost prefer original PPO root,
then canonical matching order. No forecasts or target labels at inference.
PPO source bytes remain unchanged; this is one policy improvement step, and
Q^pi training can be off-distribution under repeated value-head deployment.

## Sealed evaluation and timing kept separate

Closed-loop test:24 fresh profiles x4 shocks x2 conditions x3 replicas x3
controllers (frozen PPO, true value, shuffled value) =1728 episodes, each48+12.
No rule superiority claim in this experiment. Four PRIMARY contrasts: true vs
PPO and true vs shuffled, separately dense/skill_mask. Relative cost reduction,
20000 paired crossed replica/profile percentile bootstrap draws, two-sided
98.75% CIs (approximate Bonferroni .05/4), material margin1%. Report base/tail,
four cost components, all replica signs and leave-one-replica-out. Three frozen
replicas limit calibration; no outcome-driven stopping or filtered seeds.

Separate held-out ranking panel:all24 test profiles xFIRST physical shock xboth
conditions x3 sources, landmarks0/24 =288 states. Enumerate the same full
PM-subset classes as training. Independent32 selection and64 evaluation futures
under FROZEN PPO continuation. Save pair-MSE/RMSE, predicted vs measured Delta,
MC uncertainty and cost gain of each controller's assignment on fixed PPO
machine sets. Simulation-chosen comparison root selected using selection tape
only; its independent cost is a noisy reference, not a true optimum. Do not
count all classes/futures as independent profiles. Baseline/shuffled/trained
predictions recorded before generating held-out forecasts.

Timing diagnostic: FIRST4 profiles xFIRST shock xboth conditions x3 sources,
landmarks0/6/18/30/42 =120 common baseline-visited states. For PPO and TRUE value
controller separately, candidates retain/omit its selected PM machines with
ORIGINAL workers, holding CM fixed. Choose timing correction using32 futures,
then measure on64 independent futures with the SAME controller running on at
t+1..47 and common tail12. Own-policy continuation distinguishes actual behavior
from the Q^pi training target. Report timing exposure/gain with exploratory95%
crossed conditional-ratio CIs and per-state MC SE; not a primary algorithm claim.
No new PM machines or CM delay in timing interventions. Blanket PM deferral is
secondary. Assignment and timing menus/continuations are clearly separated.

## Smoke, budget, artifacts and limits

Smoke keeps the REAL source H48+12 and hidden32. One source replica, one fresh
profile/shock per split, both conditions, landmarks0/42,2 futures/phase, two
head epochs/checkpoints1/2. Smoke full/split signatures are disjoint; resolving
full metadata is allowed, executing full new outcomes on Mac is forbidden.
Scalar replay covers smoke training/dev/test/timing candidates, all phases;
checkpoint roundtrip, frozen PPO bytes, finite likelihood/features/loss,
feasible same-machine/fixed-CM decoding, progress and artifact counts must pass.
Run the full existing test suite before commit/push.

Fixed training/dev source trajectory counts1536/192 state records; training
physical source trajectories384x48, dev48x48. Full forecast upper bounds are
computed by runner from exact clock, roots21 and phase budgets; actual roots,
classes, futures/decisions, optimizer steps and elapsed time are audited.
Training cost labels are generated once and shared by truth/shuffle arms.
Fresh UTC directories, refuse overwrite/resume. Single CPU thread full run;
source/preflight failures stop before new compute. Never SSH, full-launch or
rsync on user's behalf; exact compound lab and partial-progress rsync commands
handed off after push per docs/experiment_workflow.md.

Artifacts: manifest, source contract/source snapshots, physical/seed registry,
state trajectories, state/pair tables, immutable per-state NPZ tapes/costs/
components/boundaries, tensor datasets, shuffled permutations, training_progress,
head checkpoints/selection, episodes/decisions/coordination/PM choices, ranking
and timing tables/summary, profile contrasts and artifact hashes. Record source
weights/extra head parameters and effective budgets; not equate forecast compute
with PPO environment steps. No claims of novelty, global optimality, indefinite
horizon superiority, or deployment transfer from this small simulator pilot.
