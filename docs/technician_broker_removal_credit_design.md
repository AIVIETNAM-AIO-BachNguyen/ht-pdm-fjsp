# Broker-aware removal credit — provisional algorithm design

This is design/correctness work, NOT a trained method, full experiment or goal
completion. Preserve all prior negative runs, including entropy-alignment3/12.
No teacher/labels/pretraining/source-model warm-start. Before any learning,
register a separate full N4 protocol with fresh train/DEV/sealedTEST panels,
matched actor/critic info/capacity/budgets and original health/quality gates.

## Specific hypothesis to investigate

The current state-value baseline cannot condition on co-workers' independently
sampled proposals. A technician may receive team-return credit for work a
co-worker would have done anyway. A baseline conditioned on the physical matching
of ONLY the other technicians could remove this source of credit variation.
Marginalizing it consistently through the broker may retain that information
without adding latent-proposal alias noise. Better finite-sample learning/cost
is a hypothesis, not a consequence of the identities below.

Prior prefix-calendar/value-credit trials (completion-creditv2) already failed.
Those critics condition on an ordered prefix; this design concerns simultaneous
independent proposals and removal BEFORE arbitration, including replacement by
losing co-workers. A failed prior trial is not rescued by this design.

## Baseline and conditional expectation

Fix state s, independent U_j~p_j and the original state-only fixed-priority,
winner-only/no-retry broker T. Let U^(-i) replace technician i's token by WAIT.
M^(-i)=T(U^(-i)) depends on other proposals, never U_i. Any DETACHED baseline
b_i(s,M^(-i)) is therefore independent of U_i. Before clipping/normalization,
E[grad log p_i(U_i) b_i(s,M^(-i))]=0. Shared actor parameters do not change this
score identity. A critic may learn this baseline from ordinary own-policy MC
returns; no counterfactual simulator target or teacher is required.
Its target would be E[G|s,M^(-i)], averaging the removed technician's ORIGINAL
policy action, not the return of a world where that technician is absent.
Removal is a counterfactual INPUT construction, not a measured causal cost label.

For physical matching S, the preimage conditional distribution factorizes:
- a matched i proposed its assigned machine with probability1;
- idle j proposes t in A_j(S)={WAIT}+{occupied m whose winner outranks j}, with
  conditional probability p_j(t)/sum_(v in A_j(S))p_j(v).

If i is idle, removing it leaves S unchanged. If i wins machine m, removing it
leaves S without edge(m,i), and MAY promote the highest-priority idle technician
whose latent proposal was m. Write alpha_j=P(U_j=m | S). Then

    w_none = product_(idle j)(1-alpha_j)
    w_j = alpha_j * product_(idle r outranks j at m)(1-alpha_r).

Only legal losing requesters have nonzero alpha. Exact lower-ID tiebreak gives
a total priority order. These weights sum1; at mostKpossible fallback winners
plus none, not all matchings or all proposal vectors. Evaluate b_i on these
possible removal matchings to obtain B_i(s,S)=E[b_i(s,M^(-i))|S].

Let f_i(S)=p_i(m_i) for winners and sum_(v in A_i(S))p_i(v) for idle i, so
q(S)=product_i f_i(S). For a reward/return G depending on latent U only through
physical S and subsequent Markov randomness,

    E[sum_i grad log p_i(U_i)*(G-b_i(s,M^(-i))) | S,G]
        = sum_i grad log f_i(S) * (G-B_i(s,S)).

Reason: a winner's proposal score is fixed given S; an idle technician's removal
matching is fixed given S. Thus the score/baseline covariance term vanishes in
each of these two cases. B_i (including its actor-dependent conditional weights)
must be STOP-GRADIENT in the actor surrogate. Do not differentiate the baseline
as part of the policy objective. Do not confuse S\i with T(U^(-i)); the former
misses replacement requests and in general depends on the removed proposal.

## Scope / contribution boundary / open implementation decisions

Counterfactual baselines and Rao-Blackwellization are prior art:
[COMA](https://arxiv.org/abs/1705.08926),
[MPG](https://arxiv.org/html/1806.05134). No new generic baseline/variance theorem
or first-use claim. The proposed domain mechanism is a tractable removal-baseline
expectation for the fixed winner-only broker, to be checked against closest work.
It does not apply unchanged to autoregressive proposals, learned action-dependent
priorities, retry brokers, hidden history or return dependence on discarded U.

The construction needs at mostK(K+1)critic queries and polynomial bookkeeping,
no matching catalogue in a learned actor. Materializing every counterfactual
matching as a Kvector costs O(K^3)storage; streaming queries could reduce this.
Do not claim a measured scalable implementation or runtime bound excluding the
cost of the critic's actual inputs/NN. Current actor/global broadcast limitations
and no-zero-shot claims remain.

The helper exposes per-technician log f_i. A per-factor clipped surrogate would
be PPO-DERIVED, not the original unchanged scalar-joint PPO objective; its factors
are not standalone normalized action policies. The identity at behavior weights
does not imply a hard KL bound, monotonic improvement or finite-update guarantees.
Choose and lock surrogate/normalization and matched clipping controls before
learning. Shared normalization based on the SAME state-value advantage is needed
if claiming exact Rao-Blackwell comparison; independently normalized advantages
need not obey the same identity.

Fair candidate/control design must train the same state-value and removal-value
critics on the same MC targets/observed other-proposal contexts in EVERY matched
core arm, with private identical initial tensors, equivalent update/compute
budgets. A state-baseline control must retain/train the removal critic as a
shadow, so extra critic information/capacity cannot explain the mechanism gain.
External PPO/MAPPO capacity/info differences need controls/disclosure. No such
learner or performance claim is implemented by this algebra helper.

## Fixed correctness audit before any empirical study

No RNG or scientific seeds. Deterministic finite categorical fixtures N2/K2,
N3/K3 and N2/K4 with tied priorities, heterogeneous priorities, masked and forced
WAIT cases. Explicit enumeration appears ONLY in independent unit references.
Check every physical preimage's conditional removal matching law against the
polynomial weights, factor product vs exact mass, exact conditional gradient
identity and unconditional zero baseline gradient (atol1e-10,float64). Use an
arbitrary nonlinear b_i table and G(S), not learned/oracle targets. Tests have a
60second engineering limit; preserve failures, no changed tolerance to rescue
an incorrect identity. No training outcome, source checkpoint or environment
state needed. Commit/push verified helpers/tests before a future full experiment.
All current scientific CF/test/largerN panels remain sealed.
