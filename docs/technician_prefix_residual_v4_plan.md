# B4 direct selected-edge residual: prospective protocol

Locked after the independently audited B3 DEV failure and its DEV-only
same-weight diagnosis, before any B4 code or outcome. B1/B2/B3 negative DEV
verdicts and their sealed scientific TEST panels remain untouched. This is
one bounded follow-up, not a sweep.

## Mechanistic hypothesis

In B3, removing only the new capacity inputs changed 0/4,608 N4 and
2/4,608 N8 choices, while removing all selected-edge channels changed
31/4,608 and 470/4,608 choices. The selected-edge relational message is
used, but B3's separately trained mask-only actor still won at N8 and
the pooled +0.5% DEV screen failed. The falsifiable B4 hypothesis is that
a direct candidate-specific logit path will let the relational prefix guide
assignment enough to beat a separately trained matched mask-only control,
rather than merely improve its own frozen weights.

Keep B3's graph encoder, raw edge/state features, reservation mask,
candidate attention prefix, capacity inputs, actor/critic, WAIT, physics,
costs, N4 foundation teachers, initialization, optimizer, student budgets
and checkpoint rules. Add exactly one residual to each actor candidate logit:
the dot product of the already existing graph `query(candidate_edge)` and
`key(candidate_prefix)`, divided by sqrt(hidden width). Add the analogous
`query(current_technician)` / `key(wait_prefix)` dot product to WAIT.
There are no new parameters or tuned scale coefficients. All selected-prefix
terms are zero at the first decision stage. The mask-only arm uses exactly
the same model and legal masks but zeroes selected-edge and capacity inputs,
so its residual is exactly zero; it still sees the whole ordinary current
graph and residual resource pool. The independent/no-reservation arm is a
service comparator only. Check parameter-count equality, identical
initial tensors, teacher-forced legal mask and resource-pool equality, and
zero invalid assignments. Record prefix-residual values and action
disagreements in engineering checks to establish that the path is live.
Do not reinterpret same-weight ablations as the primary contrast.

## Fixed panels, metrics, stopping

Use the accompanying fresh, disjoint registry. DEV: N/K=4/2 and 8/4,
three training bundles, 32 physical DEV seeds per scale. Conditional
confirmation: N/K=4/2, 8/4, 16/8 and 32/16, five new training bundles,
64 new DEV seeds for checkpoint selection and 256 sealed TEST seeds per
scale. All episodes have 48 policy and 12 common tail ticks. Smoke uses
one separate training bundle, two DEV and four engineering TEST seeds,
a 192-state teacher bank, two distillation epochs and 384 PPO decisions;
smoke is implementation evidence only.

Each scientific arm receives the same 122,880-state shared teacher bank,
32 distillation epochs and 491,520 target-scale PPO joint decisions.
Eligible checkpoints: 0, 122,880, 245,760, 368,640, 491,520; select
minimum mean DEV extended cost per arm/scale/bundle, earliest tie, after
the full budget. Report selected and fixed-final cost, availability,
pending/machine, failures/machine, learning AUC/variance, zero-shot warm
cost, invalid assignments, teacher foundation/bank/distillation/PPO
compute, optimizer steps, wall time, `model.select` and end-to-end latency.
Report all replicas and adverse cases.

The DEV screen is unchanged from B3: pooled full/mask-only selected-cost
gain >=0.5%, neither N4 nor N8 point gain below -1%, zero invalid
assignments, and per-scale full-minus-mask availability >=-0.01,
pending/machine <=+0.025, and failures/machine <=+0.25. Failure seals
B4 TEST. The run hard stop is 48 hours; timeout is INCOMPLETE.

The confirmation primary estimand is
`1 - pooled full cost/machine / pooled mask-only cost/machine`, equal
scale/bundle weighting. Require point gain >=1% and one-sided
multiplicity-adjusted lower CI >0; no scale point gain <-1%.
At each scale require full-minus-mask availability lower CI >=-0.01,
pending/machine upper CI <=+0.025 and failures/machine upper CI <=+0.25.
Require zero invalid assignments and p95 `model.select` <=50 ms at N32;
report end-to-end separately. The inferential family has 17 endpoints:
pooled cost, four scale costs, three health endpoints at each scale.
Use 40,000 crossed bootstrap draws over paired bundles and physical
episodes independently by scale, one-sided Bonferroni alpha .025/17
per endpoint. Report all endpoints.

## Artifact and execution contract

Run focused correctness tests and a small local lab smoke before
committing/pushing verified code. Commit/push code before scientific DEV
FULL. Run confirmation only after DEV screen and independent DEV audit
both PASS, with source/protocol/registry and verifier hashes bound to the
exact DEV manifest. Independently recompute scientific cost/health/CI
and replay full/mask-only trajectories before a claim. Each run gets
a fresh UTC timestamped artifact directory with exact invocation,
commit/source/protocol/registry/teacher hashes, source snapshots,
bank/checkpoint/training/episode rows, choices, physical and decision
audits, output bindings, latency and compute. Use tqdm for teacher-bank
generation, distillation, PPO and multi-seed evaluation. Do not SSH or
transfer artifacts.
