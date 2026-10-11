# Joint maintenance RL v1 — locked learning experiment

## Material Passport

Origin Skill: academic-research-suite / experiment-agent; Origin Mode: plan/run
inline; Date: 2026-10-11; Verification Status: UNVERIFIED until engineering gate;
Version: maintenance_joint_rl_v1. User explicitly requests RL learning both
WHEN/WHICH MACHINE to maintain and WHICH TECHNICIAN to assign. This supersedes
the fixed-planner confirmation as the research direction. The stopped lab run
remains incomplete historical data; its outcomes are not reused or overwritten.

## Research question and design

Can centralized PPO learn a joint partial matching that reduces physical cost
against a strong rule, and does learning technician assignment add value over
PPO learning only machine selection? Main solver is joint_ppo, NOT a fixed
allocation policy. fixed_ppo is the action-restriction ablation, not a required
production architecture. No planner-headroom gate decides whether to train RL.
Do not claim inability to learn from a negative finite-budget result.

Two trained arms, same architecture, parameter count, paired initialization,
environment/shock schedule, optimizer/budget/checkpoint schedule:
- joint_ppo: all 21 N4/K2 partial matchings, including empty WAIT.
- fixed_ppo: one deterministic duration/restoration-minimizing assignment per
  chosen machine subset (inherited fixed_candidates). Learns WHEN/WHICH MACHINE.
Reference: best_rule, selected separately on development for each condition
from the seven inherited risk/deadline/adaptive/scarcity{0,1,3,6} rules. No new
rule tuning on test. Initialization is evaluated only on development to expose
learning progress; it is ineligible for selected model.pt. No imitation labels,
model-based forecast or actual future information enters PPO observations.

## Environment and NEW profile distribution

N4/K2, learned cutoff H48, actual frozen-rule continuation L12. Preserve original
WaitingEnv nonpreemptive physics, PM1/CM2/unavailability6/failure15, waiting
limit4/price12, max age8. WAIT is always legal; feasibility masks exclude busy
resources and missing skill edges only. PM/CM type follows machine state.

Use richer heterogeneity than the prior favorite-worker generator: EACH of
eight pair durations independently uniform in {2,3,4,5,6}, restoration uniform
in {.25,.5,.75,1}, independently of duration. Failure age uniform{3,4,5},
probability uniform{.25,.45,.6}. This creates speed/restoration tradeoffs and
alternative assignments; it is a new prospective benchmark distribution, NOT
a replication or controlled before/after comparison to old run costs.
Conditions: dense (all skills) and skill_mask (two distinct random machine rows
each lose a random technician edge; every row/column retains eligibility).
Absent-edge restoration is unobservable and normalized in physical signatures.
Both conditions remain possible throughout training; do not claim their
difference is a pure topology effect because effective capacity also changes.
No profiles are selected based on RL/planner outcome, optimum gap or success.

Committed signatures exclude prior opportunity/boundary full+smoke panels and
all pending confirmation metadata (even though cancelled), ignoring H and
machine/worker relabeling. New dev/test profiles are physically disjoint.
Training rejects profiles matching ANY reserved dev/test or historical profile;
reuse within training is allowed and declared. Full smoke is separate and
physically excludes new full dev/test metadata. The lab requires no prior raw
artifacts. No N/K transfer or real-factory claim in this first experiment.

## Policy, returns and equal budgets

Shared machine, technician, pair and global MLP embeddings, hidden32; score
whole feasible matchings with selected-pair, selected/unselected-machine and
remaining-technician context. Shared actor/critic parameterization for both
arms. No entity ID or rotating arbitration feature. Actor emits one categorical
matching action per physical tick; masks are explicit constraints, not learned
constraint evidence. Sample masked likelihood correctly; record/re-evaluate
old probabilities. CPU deterministic inference with first-max tie break.

Inputs use original public observation (machine6, technician3, edge4, global9)
with remaining LEARNED horizon H-t, not H+L-t. Duration/restoration/costs/failure
law are known. No access to env.events from feature construction. Tail L is
fixed throughout the protocol. Encoders pool entities but sizes fixed N4/K2;
permutation-consistent probability tests do not prove scale generalization.

Reward = -objective/20, gamma1. Collect complete physical episodes H+L; after
H, apply shared deadline_matching without reset or policy gradient samples.
Targets at every learned tick include ALL subsequent H+L costs, including
ongoing repairs, new failures and service in the tail. No zero-value cutoff,
heuristic fabricated terminal penalty, critic bootstrap or truncated GAE.
The tail is finite policy evaluation, not infinite-horizon optimality.

Full: 10 independent paired training seeds x 2 arms =20 models. EACH gets
245,760 learned decisions +61,440 controlled tail ticks =307,200 physical
ticks (5,120 episodes). Total6,144,000 training physical ticks. Vector32,
rollout3,072 learned decisions (64 complete episodes), minibatch256, epochs4,
80 updates, lr3e-4, clip.2, entropy.01, value weight.5, gradient clip.5.
Each model3,840 optimizer updates; total76,800. Save initial and updates
{20,40,60,80}; choose minimum equally weighted development extended cost among
trained checkpoints, earliest update on ties. Each seed gets its OWN selected
checkpoint; never choose a best training seed. CPU/thread1 throughout.

Train condition alternates dense/skill_mask per episode (exact50/50 mix).
Counter-derived domain-separated config/environment/action/minibatch seeds
share physical tapes between paired arms; differing stochastic action choices
are expected. Training rejects reserved physical profiles, not just raw seeds.
Fixed complete budget; no extra steps/seeds if test is disappointing, no auto
retry/resume. Interruptions retain partial data and FAILED manifest.

## Splits and evaluation

Raw seeds: training12,100,000..12,100,009; dev config12,200,000..12,200,007;
dev shock12,300,000..12,300,003; test config12,400,000..12,400,063;
test shock12,500,000..12,500,007; analysis12,600,000. Smoke: training12,700,000;
dev config12,700,020; dev shock12,700,030; test config12,700,040;
test shocks12,700,050..12,700,051; analysis12,600,001. Audit every inherited
seed registry plus prior pending/raw panels; smoke/test/train domains disjoint.

Dev: 8 profiles x4 shocks x2 conditions =64 episode cells. Rules448 episodes;
20 models x(1 initial+4 checkpoints)x64 =6,400 model dev episodes. Total6,848.
Test: 64x8x2=1,024 cells x(20 trained policies+best_rule)=21,504 episodes,
1,032,192 learned/base +258,048 tail=1,290,240 physical ticks. Matched physical
initial/failure tapes indexed independently of controller/train replica.
Vector inference batches32 for throughput; batch-amortized timing is labeled
and not presented as individual decision latency. No test checkpoint selection.

Smoke H4/L2, hidden16, vector4, rollout32, budget64 learned decisions/arm,
minibatch16/epochs2, checkpoints{1,2}, one train seed, two conditions, one dev
cell/condition and two test shocks/condition. Full test CONFIGURATION metadata
may be inspected for disjointness; full test outcomes are never run on Mac.
Full-size engineering update benchmark uses separate smoke seeds only and
cannot change hyperparameters based on held-out performance.

## Statistics and interpretation

Four primary contrasts: joint_ppo versus best_rule and versus fixed_ppo,
SEPARATELY for dense and skill_mask. Primary metric: extended cost/machine-tick
over H+L, with common denominator across arms. Report absolute and relative
improvement, all components, base/interior costs and individual training seeds.
Investment margin1% is a research screen, not a factory requirement.

Average paired shocks within each (training replica, physical profile).
Bootstrap paired training replicas AND paired physical profiles independently
20,000 times with shared draws across contrasts; fixed shock panel conditional.
Use the bootstrap mean paired delta / bootstrap mean left baseline for the
relative interval; no mean of per-episode percentages. Primary intervals
98.75% (Bonferroni four); bootstrap coverage approximate, 10 replicas finite.
Do not pretend21,504 episodes are independent learned replicas. Rules have no
training variance, repeated along replica axis only for paired comparison.
Status per contrast: MATERIAL_GAIN if lower>1%; BELOW_MARGIN if upper<1%;
INCONCLUSIVE otherwise. No pooled success hides a failed condition.

Service: interior pending and total failed, overdue/tick, H and H+L backlog,
CM in service, request waits/censoring/carryover. Report paired exploratory95%
CIs for joint-minus-rule service differences and the old margins .1/.1/.25/12;
not simultaneous confirmatory service guarantees or deployment approval.
Cost gain accompanied by service degradation is an explicit tradeoff.
Learning curves/dev initial-to-trained improvement are developmental evidence;
assignment action variety/non-fixed assignment rates are behavior diagnostics,
not proof of calibrated opportunity-cost learning. Full versus fixed separately
trained comparison includes action-space/optimization differences, not a pure
causal effect holding all selected machine decisions constant.

## Artifacts, audits, workflow and compute

Timestamped run: plan/settings/seed/physical registry/exclusions/source snapshots,
manifest with commit/dirty/runtime/threads/counts/test-open flags/file hashes;
training_progress.csv, training_episodes.csv, training_coverage.json, checkpoints
and selected model.pt per arm/seed, dev episodes/selection, episodes.partial.csv,
episodes.csv, requests/carryover, time profiles, coordination, contrasts/profile
differences, summary. Full decision-state journals use a LOCKED audit subpanel:
first2 test profiles x first2 shocks per condition, every model and best rule.
ALL physical eval ticks still get online feasibility/scalar-batch audits and
all episodes/requests retained; no outcome-selected logging.
Persist decisions/actions and PPO likelihood/return audits on smoke. Verify
checkpoint state/log-prob/value/action reload and nonzero weight updates.
Retain both conditions, every training replica, errors and censored requests.

Engineering gates: whole repository tests, independent physical replay, exact
tail return/reward reconciliation, proper paired hierarchy/grid/correction,
all21 matching masks/WAIT/fixed subsets, no future-event features, technician
permutation probabilities, equal initial parameters, seed/physical exclusions,
checkpoint selection never initial, hash/count schema, interruption journaling,
fresh directory rejection and macOS CLI smoke with full_test_outcomes_opened=false.
Runtime estimate is based on engineering update/evaluation timings only; full
20-model training is heavier than a smoke and can take hours. Print exact
fixed budgets before execution, tqdm during training/dev/test, stream journals.
No heavy full forecast planner loop. Document estimated wall time and artifact
range after smoke. User stops old lab run; assistant does not connect to lab.
Commit/push only after gates, then exact lab command and Mac rsync per workflow.
