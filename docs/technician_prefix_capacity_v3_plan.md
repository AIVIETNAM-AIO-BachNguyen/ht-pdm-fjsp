# B3 capacity-aware selected-edge prefix: prospective protocol

Locked after independently audited B2 DEV failure and its DEV-only mechanism
diagnostic, before any B3 scientific outcome. B1 and B2 TEST stay sealed and
their negative DEV verdicts remain unchanged. C1 may still be running; no
C1 quality result is used to choose this architecture or panel.

## Falsifiable mechanism and fixed architecture

B2's relational message changed decisions and helped its own fitted weights
when zeroed, but a separately trained mask-only control was better. The V4
balanced simulator already has pair-dependent one-, two- and four-tick
service and restoration. B2 embeds these raw edges but does not explicitly
aggregate the **selected** pairs' future occupancy. B3 adds exactly two
selected-pair context scalars to the existing B2 relational message:

- `busy_load`: sum of earlier selected service durations divided by `4*K`;
- `weak_repair_load`: sum of earlier selected service duration times
  `(1-restoration)` divided by `4*K`.

For a candidate edge, the two actor inputs are its service duration divided
by four multiplied by those loads. WAIT receives the two loads directly.
All scalars are zero when no prior edge was selected. They use current public
edge features and previous actions only, with no future shock or oracle.
This is one targeted representational change; no learning-rate, entropy,
hidden width or checkpoint sweep. The primary hypothesis is that making
action-contingent future capacity explicit yields a **matched trained**
full/mask-only cost gain, not merely same-weight dependence on the message.

Train full reservation + relational/occupancy prefix, mask-only reservation
with exactly the same legal set, order, raw graph, residual-resource pool,
actor/critic dimensions, teacher bank, initialization, optimizer and PPO
budget but both selected-edge signals zeroed, and no-reservation/no-prefix
with the same graph model. Mask-only may exploit all ordinary current-state
information. The primary contrast is full versus a separately trained
mask-only actor; full versus independent measures the wider coordination
service only. Stage-level teacher-forced legal-mask equality, zero invalid
assignments, and parameter-count equality are hard correctness checks.

## Physics, panels, metrics and stopping

Keep B2/V4 balanced physics and costs exactly, N/K=4/2 and 8/4 for DEV and
N/K=4/2, 8/4, 16/8, 32/16 for conditional confirmation; 48 policy plus
12 common tail ticks. The five N4 foundation teachers remain frozen. Three
new development student bundles, 32 new DEV physical seeds per scale; five
new confirmation bundles, 64 new DEV seeds for checkpoint selection and
256 sealed TEST seeds per scale. Raw and role-derived seeds are in the
companion registry and disjoint from B1/B2/A1/C1/V4. Engineering smoke uses
one bundle, N4/N8, 192 bank states, two distillation epochs, 384 PPO
decisions, two DEV and four engineering TEST episodes. Scientific smoke is
not evidence for the claim.

Each graph fit receives 122,880 shared N4 teacher-bank states, 32
distillation epochs and 491,520 new target-scale PPO joint decisions; all
arms use the same optimizer schedule and physical tape. Eligible checkpoints
are 0, 122,880, 245,760, 368,640 and 491,520. Select minimum mean DEV
extended cost per arm/scale/bundle, earliest tie, after spending the full
budget. Report per-replica selected and fixed-final cost, health, learning
AUC/variance, zero-shot warm cost, teacher foundation/bank/distillation/PPO
compute, optimizer steps, wall time, `model.select` and end-to-end latency.
No post-TEST reselection.

DEV gate: pooled full/mask-only selected-cost gain >=0.5%, neither N4 nor
N8 point gain below -1%, zero invalid assignments and per-scale full-minus-
mask health point differences no worse than availability -0.01, terminal
pending/machine +0.025 and policy failures/machine +0.25. Failure leaves
B3 TEST sealed. A passed DEV gate only authorizes confirmation. The B3
scientific hard stop is 48 hours; timeout is INCOMPLETE.

Confirmation primary estimand: `1 - pooled full cost/machine / pooled
mask-only cost/machine`, equal scale and bundle weighting. Require point
gain >=1% and a one-sided multiplicity-adjusted lower CI >0; no scale point
gain <-1%. At each scale require full-minus-mask availability lower CI
>=-0.01, pending/machine upper CI <=+0.025 and failures/machine upper CI
<=+0.25. Require zero invalid assignments and p95 `model.select` <=50 ms
at N32; report end-to-end separately. The inferential family contains 17
endpoints: pooled cost, four scale costs and three health endpoints at each
scale. Use 40,000 crossed bootstrap draws over paired bundles and physical
episodes, independently by scale, one-sided Bonferroni alpha .025/17 per
endpoint. Report every endpoint and adverse replica.

## Execution and evidence contract

Run correctness tests and a small local lab smoke before committing/pushing
verified code; run DEV FULL only after the push. Open confirmation TEST only
if the B3 DEV screen and independent DEV auditor both PASS. Each run has a
fresh timestamped artifact directory with exact invocation, git/source/
protocol/registry/teacher hashes, source snapshot, bank/checkpoint logs,
training/DEV/TEST episode rows, frozen choices, decision/physical audit,
source and output bindings, cost/health/latency, inference and compute. Use
`tqdm` in bank, distillation, PPO and multi-seed evaluation. Independently
recompute cost/health/CI and replay both full and mask-only trajectories
before any B3 claim. Do not SSH or transfer artifacts.
