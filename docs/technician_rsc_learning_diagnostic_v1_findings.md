# RSC learning diagnostic: evidence for one controlled implementation follow-up

Post-hoc DEV analysis, not scientific confirmation. Source RSC remains a negative
7/15 full round with tests closed. Latest user permits necessary tuning; do not
reinterpret that earlier verdict as success or claim that all capacity mechanisms
have been disproved by one configuration.

Diagnostic artifact:
`technician_rsc_learning_diagnostic_v1_20261016T063242688272079Z`.
Protocol685f81c, diagnostic codeac32b1c pushed before execution;6unit checks pass.
All519source bindings verified before and after;75core checkpoints and135curve
points inspected in4.60s.0training updates,0new physical episodes, no scientific
states/seeds opened. Checkpoints and model parameter digests unchanged.

Candidate last-quarter pooled cost falls273.322917→269.132813 (1.5330%); all3
replicas select final491520. More training was a plausible explanation, but the
activation diagnostic exposes a specific representation/optimization concern:

| Final RSC replica | Capacity embedding M1/M2 contrast | Bid hidden M1/M2 contrast | Bid-layer2 saturation | Same-weight zero-message TV |
|---|---:|---:|---:|---:|
|220000000|2.337673|0.000811|1.000000|0.000378|
|220000001|1.089584|0.000059|1.000000|0.000028|
|220000002|3.590555|0.060321|0.888310|0.025829|

These are first-slot measurements on the16registered constructed DEV states,
including nonurgent machines in the saturation fraction. They are not a claim
about all visited states or proof of a causal failure mechanism. Input-gradient
and initial fixed-corpus details are retained in actor_sensitivity.csv; every
replica/checkpoint is included. Training journals confirm genuine actor and
critic Adam updates; no inference that raw gradient norms imply proportional
parameter suppression.

The message embeddings differ while the downstream bid representation largely
loses their contrast in two replicas. In the original flat token softmax,
raising all machine bids also raises GO mass. A new prospective test separates
GO mass from conditional ranking to remove this coupling at the bid logits.
Keep existing encoder, LR, budget, inputs and PPO objective. Do not launch the
previously contemplated2x training-budget test at the same time: changing two
things would obscure the interpretation.

[Factorization protocol](technician_residual_capacity_factorization_v1_plan.md)
registers a matched flat control, common uniform initial token probabilities,
capacity ablations and fresh random fits. Complete-token greedy is identical
in definition across arms. Standard factorized policies/hierarchies are prior
art; no generic novelty or guarantee is claimed. Only its controlled future
cost/health/mechanism comparison can support the proposed application-specific
benefit. Earlier global edge/STOP scarcity failures remain part of the record.
