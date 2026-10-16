# Residual service-capacity PPO v1: one prospective N4/K2 test

Registered before implementation, training or outcomes for this version. This
is one falsifiable operational hypothesis, not a search over encoder sizes,
attention, learning rates, objectives or capacity-message variants. Retain all
negative scarcity/completion/exchange/marginal-matching results and unopened
scientific panels. Random initialization and standard scalar clipped PPO only;
no teacher, distillation, old learned tensor, warm start, reward shaping,
simulator lookahead, future event or auxiliary supervised training.

## Hypothesis and rejection

An explicit summary of service opportunities of technicians later in a fixed
allocation order helps the current technician avoid taking an easy shared
machine while leaving a machine with fewer timely alternatives uncovered.
The representation is deterministic information processing of inputs already
available to every matched arm, not an extra measurement or a learned price.

Earlier scarcity versions used learned bid-derived exclude-own opportunity
prices in a global edge/STOP broker. Completion variants exposed reservation
calendars/credit. Neither established the desired small-setting result. Here
there is no learned price, global STOP, order mixture, repair or prefix critic:
only a fixed capacity summary in a canonical one-slot-per-technician policy.
This is an application-specific representation hypothesis, not a new generic
PPO objective, communication theorem or claim of prior-art novelty. Shared
full current graph and a broker are assumed; no network-local deployment claim.

There is exactly ONE full development comparison for this operationalization.
Failure of a mechanism, quality or health screen rejects this design under the
registered benchmark/budget. Do not change horizons, feature normalization,
encoder, LR, entropy, order or add components to rescue it. Report the failed
comparison; any later distinct hypothesis needs independent motivation and a
new prospective protocol. Failure does not disprove every possible capacity
representation. Passing DEV only authorizes the fixed confirmation, not a claim.

## Deterministic message

Visit technicians once, ascending physical index j=0..K-1. Let U be the urgent,
currently unserved and not provisionally reserved machines. For fixed horizons
h in {1,2,4} physical ticks and later technicians l>j, compute

    F[m,l,h] = 1[m in U, skill(m,l), remaining_l + duration(m,l) <= h]
    D[l,h]   = sum_m F[m,l,h]
    R[m,h]   = sum_{l>j} F[m,l,h] / max(1,D[l,h]).

A later technician contributes one fractional opportunity unit across the
currently urgent machines it could finish one job on within h. This is a fluid
one-job-per-window summary, not a maximum matching, queueing forecast or exact
future capacity. Busy technicians can contribute when their known remaining
service plus a new service fits. No future arrivals, failures or optimal labels.
Use raw skill/duration, machine urgent/unserved flags, current technician
remaining times and own prefix. Floating normalized times are rounded back to
known integer ticks before comparison. Conservation: sum_m R[m,h] equals the
number of later technicians with nonempty support in that window.

At N4/K2 only the first technician receives a nonzero later-capacity message;
the last has zero later capacity. No claim of rich multi-stage capacity updates
at K2. Higher R indicates a later alternative, but no manually signed action
penalty is imposed; PPO must learn its use.

## Five matched arms and policy

- residual_capacity_ppo: R as defined, reservations and prefix calendars.
- no_capacity_ppo: all three message channels zero; same allocated network.
- initial_capacity_ppo: summary uses initial U and ALL technicians at every
  slot; preserves a static service-capacity summary but removes remaining-order
  interpretation. It is not an independent policy.
- permuted_capacity_ppo: cyclically roll true R by +1 machine row; same message
  values/computation but wrong machine association, a fixed negative control.
  This control intentionally breaks machine-permutation equivariance.
- independent_capacity_ppo: true later-capacity based on initial U, original
  calendar throughout, and no reservations. Independent technician proposals
  collide; keep highest restoration/duration, then lower technician index,
  with no retry. Tests compare proposal likelihood, not physical marginal.

All arms compute the same true/static/permuted summaries before choosing their
registered input. Raw M6/T3/E4/G9 pair22 and all original measurements unchanged.
Retain the earlier canonical actor H27: pair encoder22->27, calendar4->27,
bid(4H)->1, WAIT(2H+1)->1. Add a shared MLP(3,H,H) to encode each machine's R,
add it to the edge embedding before unchanged machine/technician/global pools.
The extra MLP has two tanh hidden layers and a linear output, 1620 parameters;
its last layer is zero-initialized in ALL arms. Thus all core arms share exact
initial tensors and initial conditional probabilities for equal legal prefixes.
Total20010 parameters: actor11423, separate physical-state critic8587.
No message enters the scalar critic. No attention or parameter-width sweep.
External PPO18966/localMAPPO15364/fullMAPPO18052 parameter differences disclosed.

One machine-or-WAIT per technician, WAIT does not stop others, mask reservations,
then execute all starts simultaneously. K conditionals per sample, greedy select
or replay; forced WAIT still evaluates the network. Sum conditional log scores
for the joint PPO ratio, average conditional entropy/K. Standard PPO clip.2,
entropy.01, Adam3e-4, four epochs, minibatch256, gamma1, MC returns from the
unchanged 48 decisions plus common 12-tick deadline tail, reward=-cost/20.
Advantage standardized over rollout, state critic unit115.2, value weight.5,
separate actor/critic clipping.5. No allocation-learning/prefix-value mode.

Core likelihood is over canonical proposals (bijective to matching except the
explicit independent-arbitration arm). Auxiliary final token is fixed0 solely
for action-format compatibility, never sampled/learned. There is no latent order
or physical mixture. Candidate never enumerates matchings; conservative
O(N*K^2) construction, parameter count independent of N/K. N4 alone cannot
establish scaling. Larger/generalization runs require confirmed N4 success.

## Locked budgets, splits, tests and stopping

FULL DEV:9 methods (five core plus original PPO, normalized PPO, localMAPPO,
full-informationMAPPO),3 train replicas each,491520 joint decisions/fit,
vector16/rollout3072,7680 optimizer steps/fit. Checkpoints0,122880,245760,
368640,491520. Spend all budgets; earliest minimum mean DEV cost chooses each
replica; retain every replica including step0 selections.27 fits,13,271,040
train decisions,276,480 train episodes;4320 DEV plus32 deadline episodes.
Confirmation:5 FRESH train replicas/method, same fits/budgets, DEV64, TEST256;
45 fits,22,118,400 decisions,460800 train episodes,14400 DEV plus64 deadline,
11520 learned-method TEST plus256 deadline episodes. Seal all choices before
opening tests. Hard12h per grid,8 workers,torch1,tqdm; no extension/early stopping.
Technical failure aborts siblings and preserves FAILED; do not quietly rerun.

Registry fixes DEV train220000000..2/dev220100000..31/analysis220200000;
confirmation train220300000..4/dev220400000..63/test220500000..255/
analysis220600000; engineering train220700000/dev220700001..2/
analysis220700003. Engineering is only finite correctness fixtures, no separate
smoke/profile run; latest user explicitly requests FULL directly on this lab.
No scientific test seeds in engineering. Audit raw/historically registered seed
and derived role collisions, pair physical tapes across all methods, core
initial/action/minibatch RNG roles, preserve original-vs-normalized PPO pairing.
Tests/prospective registration/commit/push precede FULL. No SSH or transfer.

## Mechanism probe and fixed decision rule

After all budgets and checkpoint selection are sealed, measure the first
technician's probability of choosing machine2 in fixed valid states: M1/M2
urgent, same age and failed flag, all technicians free, no services/pending,
M3/M4 nonurgent. In this benchmark T1 takes1 tick on either; T2 takes2 ticks
on M1 and4 on M2. Thus within2 ticks only M1 has a later alternative. Verify
these benchmark constants before learning; do not manufacture a new task.

DEV16 states are Cartesian product: shared urgent age{3,4}, failed{false,true},
other ages{0,1}, tick{0,24}. Confirmation16 disjoint states use urgent age{3,5},
failed{false,true}, other ages{1,2}, tick{8,32}. These are fixed constructed
mechanism states, not sampled test episodes, optimal-action labels or training
examples. No fitting/reselection uses them. Average critical-first probability
per replica. Require candidate minus no_capacity >=.05 (5 percentage points)
on DEV point estimate and confirmation lower confidence bound. A probe alone
cannot establish better physical decisions; all cost/health gates also required.
Record all core probabilities and same-weight candidate zero-message knockout
on these states descriptively; do not train/choose weights based on knockout.

Fifteen mandatory statistical endpoints:8 cost gains,6 health differences vs
both PPO baselines,1 critical-first difference. Gain=1-candidate/reference mean
cost. Required lower gains>=.005 vs ALL FOUR matched controls, >=-.01 vs BOTH
PPO, >0 vs BOTH MAPPO. Health candidate-reference: availability lower>=-.01,
pending/machine upper<=+.025, failures/machine upper<=+.25 vs BOTH PPO.
Zero invalids; isolated selected candidate actor p95<=50ms on retained first-DEV
48 states x3 repeats/replica after training pool closes. Optional practical PPO
superiority requires lower gain>=+.01 vs BOTH PPO using same intervals.

Use unchanged unused round2 alpha=.05/6, Bonferroni15, individual two-sided
confidence1-(.05/6)/15.200000 paired crossed bootstrap draws resample train
replicas and episode seeds, preserving method pairing. Fixed-probe endpoint
resamples paired training replicas ONLY; identical replica weights as cost draw,
no resampling the16 fixed states. Approximate coverage with5 replicas must be
stated; every-replica and leave-one-out reports required. DEV uses exactly the
same point thresholds, no CIs/success claim; no confirmation if any screen fails.
No old test reuse or alpha reset. Cost improvements cannot override health.

## Artifacts and verification

New timestamped directory per run; never overwrite. Store source/protocol/
registry/physics snapshots and hashes, clean pushed commit, config/runtime,
seed audit, initial parameter hashes/counts, every checkpoint and selection,
complete training/DEV/test journals, reward/physical tape hashes, mechanism
state/probability/knockout records, isolated latency, scalar replay audit,
summary15 gates and output bindings. Probe results are separate from physical
metrics: any internal broadcast across episode axes is arithmetic bookkeeping
and does not increase its sample size. Confirmation validates exact source and
all bound passing DEV outputs; modifications invalidate promotion.

Before launch test hand-computed capacity/conservation, busy/urgent/reservation
boundaries, fixed horizon rounding, no future inputs, zero/static/permuted and
independent behavior, random tensor/initial policy parity, finite gradients,
all21 matching likelihood normalization and replay, forced WAIT, collisions and
invalids, original baseline behavior, checkpoint roundtrip, serial/spawn/RNG,
15-endpoint/null/bootstrap/probe selection and prerequisite corruption. Independent
read-only verification after outcomes; preserve negative/technical failures.
Report uncertainty honestly and provide manual rsync -avhP --partial once ready.
