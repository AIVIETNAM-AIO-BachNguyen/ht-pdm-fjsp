# B2 relational selected-edge prefix: prospective protocol

Locked before any B2 outcome. B1 development failed its registered gate;
its TEST remains sealed. The B1 same-weight DEV diagnosis found that removing
the learned prefix changed 621/4,608 N8 decisions and raised cost 2.624%,
while a separately trained mask-only arm almost closed the gap. This is
motivation, not B2 evidence.

## Falsifiable mechanism

The B1 prefix averages selected-edge embeddings with a `1/K` factor and sends
the same vector to every candidate at a stage. This may dilute the signal and
hide which earlier assignment competes with each current candidate. B2 changes
only that message: for each current candidate edge, use scaled dot-product
attention over earlier **selected** edge embeddings, with the candidate as
query. A WAIT contributes no selected edge; if the prefix is empty the message
is zero. The full policy receives this candidate-specific message in the same
input block as B1. The WAIT logit receives the mean of prior selected-edge
embeddings. There are no new trainable parameters; actor/critic and initial
graph observation are otherwise identical to B1. The decoder remains
sequential and polynomial; no large matching catalogue.

Three arms train from matched initial tensors: full relational prefix plus
reservation, mask-only with the identical computation and parameter count but
the message zeroed, and no-reservation/no-prefix. Full and mask-only retain
the same technician order, stage-specific legal sets, residual-resource pool,
reserved fraction, teacher bank, optimization and physical tape. Per-stage
teacher-forced mask equality and zero invalid assignments are hard audits.
The primary contrast is full versus separately trained mask-only. Full versus
no-reservation measures the combined conflict mechanism only. A same-weight
prefix intervention is diagnostic and cannot pass the B gate.

## Budget, panels and gate

Use frozen V4 balanced physics and cost at N/K=4/2, 8/4, 16/8, 32/16,
48 policy plus 12 common tail ticks. The five source N4 PPO teachers are
frozen. For every student bundle count foundation teacher PPO (491,520
decisions), an N4 bank of 122,880 states, 32 distillation epochs, 491,520
target-scale PPO decisions, optimizer updates, phase wall time and inference.
Extra teacher compute is disclosed, not interpreted as compute efficiency.
All new student seeds are in the companion registry and disjoint from B1,
A1, V4 and each other. Do not inspect a sealed TEST unless the matching DEV
gate passes.

Engineering smoke: one bundle, N4/N8, 192 bank states, two pretrain epochs,
384 PPO decisions, 2 DEV and 4 engineering test episodes. Development FULL:
three bundles, N4/N8, 32 fresh DEV episodes per candidate checkpoint.
Checkpoints 0, 122,880, 245,760, 368,640, 491,520; choose minimum mean DEV
extended cost, earliest tie. Train the entire budget regardless of choice.
The DEV gate requires pooled full/mask-only cost gain >=0.5%, neither scale
point gain below -1%, zero invalid assignments, and full versus matched
mask-only availability >=-0.01, terminal pending/machine <=+0.025 and policy
failures/machine <=+0.25. Failure leaves confirmation TEST sealed. A passed
DEV gate only authorizes the prespecified confirmation, not a claim.

Confirmation: five **new** bundles, 64 new DEV episodes for checkpoint
selection and 256 sealed TEST episodes, all four scales. Freeze source,
protocol, registry, architecture, checkpoint rule and analysis before TEST.
No post-TEST reselection. Global hard stop 48 h; timeout is INCOMPLETE.

Primary gain: `1 - mean(full extended cost/machine) /
mean(mask-only extended cost/machine)` with equal scale and bundle weights.
Require pooled point gain >=1% and a one-sided multiplicity-adjusted lower
CI >0. Report each scale/replica, and no scale point gain may be <-1%.
Health reference is matched mask-only on the same physical seeds: for each
scale availability lower CI >=-0.01, pending/machine upper CI <=+0.025,
failures/machine upper CI <=+0.25. The family has 17 endpoints: pooled cost,
four scale costs and three health endpoints at each scale. Crossed bootstrap
over paired training bundles and paired physical episodes independently by
scale, 40,000 draws; one-sided Bonferroni alpha `.025/17` per endpoint.
Report latency `model.select` and end-to-end separately, p95 select <=50 ms
at N32, but latency alone cannot rescue a failed cost/health contrast.

Fresh timestamped artifact directory per run: exact invocation, source and
git hashes, protocol/registry/teacher hashes, bank and pretraining logs,
checkpoints and frozen choices, training/DEV/TEST episode rows, per-replica
cost/health, CI, wall/decisions/updates accounting, decision and legal-mask
audit, scalar physical replay, source/output bindings. Preserve failed runs.
Use `tqdm` in bank, distillation, PPO and multi-seed evaluation. Correctness
tests and small lab smoke, commit and push precede FULL. Run directly here.
