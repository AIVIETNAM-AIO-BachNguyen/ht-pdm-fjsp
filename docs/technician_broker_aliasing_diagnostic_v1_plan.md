# Conditional fixed-broker aliasing diagnostic v1

Register while matched double-budget FULL is live, before reading its terminal
outcomes. This is a conditional read-only diagnostic, NOT another learning trial.
Run ONLY if the frozen source study completes with independent verification PASS
and overall eligibility FALSE. If eligibility passes, refuse and follow the
registered confirmation path instead. Do not change or rescue its verdict.

## Hypothesis and measurement

Hypothesis: the independent-proposal learner has substantial actor-score variance
from proposal distinctions discarded by its fixed winner-only broker, at the
prelocked early checkpoint122880. Mathematical marginalization identities are
already validated; they do not establish material noise in trained models.

Use ONLY `independent_raw_ppo` from source protocol
`technician_raw_reservation_budget_v1`, learning commit
4ade42381681e073efc8a2c7dc9efb43640a3809. Check all three replicas at all nine
checkpoints0/122880/245760/368640/491520/614400/737280/860160/983040. Do not choose
another primary checkpoint after results. Condition on the SAME fixed state
panel within each replica: all32 sourceDEV seeds280100000..31 and ticks0,3,...45
of that replica's recorded selected independent policy trajectories. This is512
contexts/replica,1536 distinct contexts,13824 checkpoint-context evaluations.

These are selected greedy-DEV states, not an on-policy state sample for every
checkpoint. Every reported noise statistic is conditional on this fixed state
panel; it is not total return-gradient variance, training variance, a CI, or
cost evidence. No scientific/test state, new physical rollout or random seed.

For each checkpoint/state reconstruct the actor's two103-dimensional inputs
(raw state71, own technician3, own edges16, zero plan8, zero reservation4,
technician IDj/2). All tensor values come from frozen actor weights or independently
reconstructed current physical state. Compute float64 actor logits/Jacobian at
the stored float32 weights and input values. Form the10x10 Gram matrix JJ^T of
the full11141-parameter SHARED actor's logit Jacobian, not only free-logit scores.
No critic parameters or data-sharing. No optimizer, action sampling or update.

For each masked independent proposal vector u, its categorical score in logit
space is d_u=onehot(u)-p. Enumerate at most25 proposals ONLY in this N4 diagnostic,
resolve with the fixed restoration/duration then ID winner rule, and sum masses
by physical matching S. Independently evaluate the closed-form physical log mass
and analytic logit score d_S. Check exact mass/score/normalization identities.
Compute actor-score Fisher traces:

    F_proposal = sum_u p(u) d_u^T JJ^T d_u
    F_physical = sum_S q(S) d_S^T JJ^T d_S.

The removed component is F_proposal-F_physical>=0. Aggregate ratio is the ratio
of SUMS over all512 contexts, not average individual near-zero ratios. Also log
mean categorical entropy per technician and exact same-machine proposal conflict
probability sum_m p_0(m)*p_1(m). Score variance is for unit action weight, not an
estimated advantage or return. The active-parameter metric depends on the shared
actor representation; do not relabel it as all PPO optimization noise.

## Locked engineering eligibility / rejection

At checkpoint122880, EVERY one of the three replicas must have:
- removed actor-score fraction >=.10;
- mean proposal entropy >=.10 nats per technician;
- mean same-machine proposal conflict probability >=.05.

All numerical mass/score/normalization/variance audits must pass with float64
absolute+relative tolerance2e-10. Check full source and independent-verification
hash bindings before/after; fail on mismatch. This gate is a heuristic practical
screen for a future registered experiment, NOT statistical significance, cost
superiority or proof of a clipping benefit. All nine checkpoint summaries remain
reported even when the primary screen fails; do not substitute a favorable time.
If the screen fails, reject this specific material-early-alias-noise explanation
on the registered panel and do not launch a marginal-ratio learning run from it.
If it passes, a separate bounded matched learning protocol, fresh seed registry,
entropy treatment, fair budget, cost/health/mechanism gates and tests/commit/push
are still required. No automatic full learning trial based on algebra/noise alone.

## Execution, validation and artifacts

No simultaneous training workers for diagnostic execution. One torch/BLAS thread,
tqdm across contexts,15minute hard stop, fail fast and preserve failed artifacts.
Use new timestamped directory; manifest with source/verification manifest hashes,
protocol/script/dependency bindings and outputs; per-context metrics CSV; all27
checkpoint hashes; all1536 panel state/observation bindings; full summaries and
primary eligibility checks. Never modify source or open its sealed CF panels.

Implement as independent mathematical helper plus frozen actor arithmetic,
without simulator or learner imports. Reuse previously verified pure observation
reconstruction only. Necessary deterministic unit fixtures compare analytical
physical scores against exhaustive posterior scores/autodiff, actor Gram against
direct parameter scores, exact masks/forcedWAIT and aggregation boundary math.
Commit/push verified diagnostic before conditional execution. No separate smoke.
This observational state-panel diagnostic cannot establish a causal improvement
from a future learning algorithm, even if its screen passes.
