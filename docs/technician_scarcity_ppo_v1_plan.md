# Scarcity-Message PPO v1: prospective teacher-free protocol

Locked before any outcome from this version. User explicitly requests a new
algorithmic contribution, random initialization, no teacher/distillation or
baseline warm start, and direct execution on this Ubuntu lab. Previous V4
scientific tests are historical evidence only and never a new selection panel.

## Hypothesis and prior art boundary

Learned resource bids coupled by bounded differentiable demand/price messages
can protect scarce technicians' alternative uses, improving a matching policy
beyond feasibility reservation alone, while retaining near-central PPO quality.
All learning uses actual on-policy environmental rewards. No teacher labels,
expert demonstrations, counterfactual simulator rollouts or future events.

PPO is the optimizer, not the new contribution. Autoregressive MARL is already
studied by MAT (https://arxiv.org/abs/2205.14953); sequential multi-agent trust
region updates by HAPPO (https://arxiv.org/abs/2109.11251); learned bipartite
matching by Alomrani et al. (https://arxiv.org/abs/2109.10380), and differentiable
matching/auctions by Curry et al. (https://arxiv.org/abs/2106.07877).
This repo also contains deterministic scarcity/opportunity-cost diagnostics.
We claim no invention of PPO, auctions, shadow prices or autoregressive sampling.
Novelty remains provisional: the proposed contribution is a teacher-free
scarcity-message assignment policy and measured intervention beyond reservation
on heterogeneous maintenance resources. Prior-art comparison and limitations
must accompany any positive result.

## Architecture and intervention

Each machine produces learned bids for eligible technicians and a WAIT bid.
Raw inputs are exactly current machine6/technician3/edge4/global9, no identity
features or extra external information. Shared hidden32 pair MLP and pooled
machine/technician/global representations, separate value encoder. Base bids
are computed once per physical decision, not once per technician stage.

At each of K allocation stages, remove already assigned machines/technicians.
Compute three rounds of row-softmax demands (including machine WAIT); update
technician prices p=max(0,p+0.5*(sum_machine demand-1)). Center prices over
available technicians and subtract them from edge bids. These are differentiable
messages computed from learned bids, not expert targets. The broker samples one
eligible edge or STOP; chosen resources are reserved, STOP makes all later tokens
STOP. All sampled assignments execute simultaneously in one physical tick.
The joint probability is the product of these conditional categorical decisions;
PPO uses its exact joint ratio. Entropy is normalized by K. Critic estimates whole
plant return with N/4 output scaling and matching N/4 value-loss normalization.
No large matching catalogue, no pair-attention tensor over all other machines.
Cost per allocation is polynomial; actual N32 training time must be measured.

Three graph arms share parameter counts, random initial tensors, raw information,
training physical tapes, rollout/minibatch budgets and checkpoint-selection rule:

- `scarcity_ppo`: prices plus hard reservation.
- `reservation_ppo`: identical broker/reservations with price exchange disabled.
- `independent_ppo`: no price or reservation exchange; machines independently
  propose technician/WAIT, duplicate technicians resolved by current urgency
  then machine ID; losing proposals are not retried.

The first contrast isolates price messages; the second tests the whole
coordination service. Full graph and a shared broker are assumed. No local-only
agent or network deployment claim. Fixed non-learning capacity masks apply to
all methods; they provide feasibility, not training labels. Prices have no
additional learnable parameters. There is no pretraining or model-input path.

N4 comparisons freshly train catalogue PPO, original local MAPPO, and
full-information MAPPO (global state plus agent ID), with global critic.
Original MAPPO comparison alone cannot establish a coordination contribution.
The stronger full-information control must also be noninferior within 1%.

## Physics, metrics and fixed budgets

Same balanced maintenance scalar physics as V4, N4/N8/N16/N32, K=N/2, four
machine/two skill profiles repeated across all edges, H48 plus common deadline
tail12. Reward=-cost/20, gamma1. Large deadline tail uses polynomial min-cost
flow. Catalogue construction is permitted only for the N4 central reference.
Primary cost is H+L including tail. Availability, pending/machine and failures/
machine are measured at policy cutoff. Record conflicts, rejected proposals,
utilization, invalid assignments, physical events/actions and timing too.

Each full fit spends 491520 physical joint decisions, vector16, rollout3072,
four PPO epochs, minibatch256, lr3e-4, clip0.2, entropy0.01, value0.5,
gradient clip0.5. All baseline fits spend the same step budget. Graph hidden32;
central/local/full MAPPO hidden64. Graph capacity is matched exactly between
intervention arms; differences to baseline capacity are disclosed.
Every fit is random initialization with a recorded parameter digest. Eligible
checkpoints: 0,122880,245760,368640,491520; mean development cost selects earliest
tie. Entire budget spent even if an early checkpoint is selected; all replicas
retained. Checkpoint hashes sealed before scientific tests.

Smoke: N4/N32, one seed, 384 decisions, vector2, rollout192, two epochs,
minibatch48, all N4 baselines and three N32 graph arms. Engineering tests only.
Pilot: N4/N32, two seeds, 61440 decisions, vector8, rollout768, four epochs,
minibatch128, checkpoints30720/61440 plus random0, dev16; no test seeds opened.
Development: all four scales, three seeds, full budgets and dev32, no tests.
Neither pilot nor development can establish scientific improvement. Any change
after development outcomes requires a new version and prospective seed registry.
Per-run hard stop24h; timeout is FAILED, never success. No silent extension.

## Seed splits and confirmation prerequisites

Registry panels use 60000000–61300255 roles; exact splits in the registry.
All raw and namespaced physical/action/minibatch seeds are checked against all
historical registries and across profiles. Same exogenous train/test tapes across
arms at each size; physically different namespaces across sizes. No previous
test episode is reused. Confirmation needs an audited, test-closed full-budget
development run with the same source/configuration and a passing screen:
N4 candidate<=1.01 PPO/full MAPPO/graph controls and <local MAPPO; each large
candidate<=1.02 deadline, <=1.01 reservation, <independent; pooled price gain
>=0.5%; health within the confirmation margins below. Pilot guides whether to
run development; no pilot score can override a failed development prerequisite.

## Locked scientific gates

Previous teacher-assisted round1 spent alpha0.025. This continuation uses round2
alpha=0.05/(2*3)=0.008333333333333333. Thirty endpoints, crossed bootstrap200000,
five training replicas and 256 paired test episodes per scale. Share replica
weights across scales; resample episode weights independently by scale and
pair methods within scale. Individual two-sided confidence=1-alpha/30.
Coverage with five replicas is approximate; report all replicas and leave-one-out.

N4 endpoints: gains vs PPO, local MAPPO, full MAPPO, reservation, independent
(five), plus availability/pending/failure differences vs PPO (three).
Each large: gains vs reservation, independent, deadline (three), three health
differences vs deadline. Three large/N4 cost-per-machine ratios plus one pooled
price-message gain over reservation =30. Pooled gain is the arithmetic mean of
the three per-scale cost gains, computed inside every bootstrap draw.

Required N4: PPO/full MAPPO/reservation/independent gain lower>=-1%; local MAPPO
gain lower>0. Required each large: reservation gain lower>=-1%, independent
gain lower>0, deadline gain lower>=-2%, cost-ratio upper<=1.10. Required pooled
price-message gain lower>=0.5%: passing only independent ablation is insufficient
for the main algorithmic contribution. Optional PPO practical superiority
requires N4 gain lower>=1%; optional positive N4 price effect is reported only.

All sizes: availability difference lower>=-0.01, pending/machine difference
upper<=0.025, failures/machine difference upper<=0.25. Neural p95<=50ms, zero
invalid assignments. Report health deterioration inside margins explicitly.
Timing measures model.select after training stops; excludes graph construction,
environment arbitration/stepping and network transport. Every required gate
must pass; no best-seed reporting, test-driven stopping or reusing opened panels.

## Validation and artifact contract

Tests: exact probability normalization at N4, STOP canonicalization and replay,
finite price gradients, matched capacity/init, scarcity response including busy
resources, permutation equivariance, no future input, no teacher/imported weights,
no catalogue at large sizes, scalar60-tick replay/reward reconciliation, checkpoint
hash/metadata roundtrip, exact train/dev/test grids, fresh seeds, deterministic
serial/spawn execution, corruption rejection and prerequisite enforcement.
Run full suite and local Linux smoke, commit and push before pilot/development.
Use tqdm for training/evaluation and fresh UTC artifact directories.

Artifacts: manifest/resolved config/physics/seed audit, sealed source/protocol/
registry snapshots, initialization and parameter-count records, raw train and dev
journals, checkpoints/choices, coordination/physical decision journals, episodes,
bootstrap intervals and independent scalar physical audit, output hashes, console
log adjacent to the directory. No teachers/banks/warm artifacts. Reports are new
directories; preserve failed runs. Do not SSH or start transfers. Show canonical
rsync -avhP --partial for manual Mac retrieval when completed results exist.

## Engineering amendments before research-quality outcomes

After engineering smoke audit, before any pilot/development run or scientific
test: increase planned bootstrap draws from 40000 to 200000, so round2's extreme
tails have about28 rather than6 Monte Carlo draws. No margin/alpha/endpoint changes.
Construct the redundant tiled full_local observation only for full-information
MAPPO; graph policies never consume it, preserving their policy inputs/weights.
Record descriptive price magnitude/activity and same-weight unpriced matching
changes during evaluation, outside the counterfactual inference timing. These
probes use current state only, cause no physical step and never train the actor.
No extra inferential endpoint is introduced. Retain all original smoke artifacts;
repeat targeted tests and a fresh smoke after these amendments.
