# Broker marginalization: conditional derivation, not a validated learning method

Written while the bounded double-budget FULL is running. Do not interpret this
note as its result, launch another learning run, or change that frozen study.
This is a mathematical design investigation for a possible later mechanism
hypothesis. No new trained policy, physical rollout or scientific seed is opened.

## Motivation / prior-art boundary

Independent technician proposals are a legitimate latent-action policy: their
ordinary policy-gradient estimator is unbiased when the broker is part of the
environment. Calling the existing likelihood wrong would be incorrect. However,
a fixed broker can discard proposal information without changing the physical
action; latent-score noise and PPO clipping can then depend on that discarded
information. This needs diagnosis before a learning experiment is justified.

Marginalizing action transformations and its variance argument are established
prior art: [Fujita & Maeda, CAPG (2018)](https://proceedings.mlr.press/v80/fujita18a.html)
and [Eisenach et al., Marginal Policy Gradients (2019), section4](https://arxiv.org/html/1806.05134#S4).
Their generic gradient result is not ours. This note derives a finite matching
likelihood for the particular fixed winner-only broker; the focused review does
not establish novelty over every allocation/action-abstraction method. It keeps
the whole physical-action PPO ratio, unlike the per-subaction alternative in
[Song et al. (2023)](https://arxiv.org/abs/2301.10919).

## Assumptions

At state s, technician j independently samples U_j in {0,...,N}; 0 means WAIT.
The categorical p_j includes the same physical eligibility mask. All actors may
share parameters and see the full state: independence is conditional on s and
independent categorical draws, not parameter independence.

For each machine m, an eligibility-preserving broker keeps exactly one requester:
the highest fixed priority technician (restoration/duration then lower ID in the
current baseline). Its strict total order may depend on s, but not theta or other
requests. Every losing requester becomes WAIT, with no fallback/retry. No cost or
transition depends on rejected requests beyond their effect on the winners.
The policy has no recurrent latent proposal history. These conditions do NOT
hold for serial reservation conditioning, bid-dependent/global matching solvers,
or brokers that reassign losers; the formula must not be used for those.

Let T_s(U)=S be the physical matching, written as winning pairs (m,w_m), and
I(S) the technicians absent from S. Let w_m outrank j at m be written w_m >_m j.

## Exact likelihood without enumerating proposals or matchings

For a particular feasible physical S, define

    L_j(S) = {0} union {m : (m,w_m) in S and w_m >_m j}.

Then

    q_theta(S|s)
      = product_{(m,j) in S} p_j(m|s)
        * product_{j in I(S)} [p_j(0|s)
             + sum_{(m,w_m) in S: w_m >_m j} p_j(m|s)].

Proof: a winning technician must request its assigned machine. An idle technician
can WAIT or request an already served machine where it loses to that machine's
specified winner. Requesting an unserved machine would create an extra winner;
requesting a served machine where it outranks the winner would replace that
winner. Those are exactly the disallowed requests. Under independent proposals,
these allowed events form a Cartesian product, so summing their product masses
factorizes into the displayed product of sums. Every allowed request vector
maps to S and every vector mapping to S is allowed. This also covers empty S
(all technicians must WAIT) and forced-WAIT technicians.

After computing KN proposal logits, one observed-S likelihood takes O(K|S|),
with |S|<=min(N,K), using logsumexp for idle factors. This is a likelihood for
one matching, NOT an algorithm for its global mode or full entropy. It needs no
matching catalogue. The standalone engineering helper additionally validates the
full mask/log-mass matrix and priority permutations; those checks cost O(NK log K)
and are not an optimized production implementation. A fixed-width shared MLP still has its own size/compute
costs; no larger physical scalability result follows from this formula alone.

## What follows, and what does not

For positive distributions on finite masked support and a fixed broker,

    grad log q(S) = E[grad log p(U) | T_s(U)=S].

With an action-value A(s,S), the marginalized score times A is the conditional
expectation of the proposal-score estimator. They have the same mean, and their
covariance difference is positive semidefinite by total covariance. The result
also applies to a Monte Carlo return if, given (s,S), its future distribution is
independent of the discarded current proposal. A state-only baseline is allowed.
This finite proof is an application of the cited established MPG principle.

Likewise the importance ratio satisfies

    q_new(S)/q_old(S) = E_old[p_new(U)/p_old(U) | S].

For fixed A, the PPO clipped scalar f_A(r)=min(r*A,clip(r)*A) is concave in r.
Jensen therefore gives f_A(r_physical)>=E_old[f_A(r_proposal)|S]. This compares
surrogate values on the same old state/action distribution. It does NOT prove
that multi-epoch clipped gradients have lower variance, that the surrogate is a
return bound, or that any update improves maintenance cost. Physical KL also
cannot exceed latent KL under the same fixed map, by data processing.

Example: N1/K2, technician0 always outranks1. For S={(machine0,technician0)},
q(S)=p_0(machine0), regardless of technician1's WAIT/request split. Changing that
split from (.8,.2) to (.2,.8) gives latent ratios .25 and4 while physical ratio1.
For positive advantage1 and epsilon.2, conditional latent clipped surrogate is
.8*.25+.2*1.2=.44 versus physical surrogate1. This is a mathematical example,
not evidence that trained checkpoints have a large harmful aliasing effect.

The exact entropy H(q) is NOT the sum of categorical entropies or obtained from
one log q alone. Replacing the PPO ratio but keeping proposal entropy would be
an explicitly retained auxiliary proposal regularizer, not physical entropy.
An empirical estimate of -log q has a sampling-gradient issue: simply
autodifferentiating it at a detached current sample is not an unbiased entropy
gradient. Any future learning protocol must resolve and match that choice.
Greedy individual proposals also need not yield the global q-mode; learning and
execution changes must not be conflated.

## Validation plan / limits (before implementing checks)

Only deterministic algebra fixtures: no train/DEV/TEST seeds, simulator, learned
checkpoint, random initialization study or physical outcome. Compare the closed
form and its autodiff gradients against exhaustive *unit-test-only* proposal
sums for N/K=(1,2),(2,3),(3,3),(2,4), with fixed logits, priority ties resolved by
ID, masks and forced WAIT. Check normalization, gradient normalization, posterior
scores, mean gradient and variance identities, importance ratios and the clipped
Jensen inequality. Check malformed matchings and an extreme-logit log-domain case.
Stop on any discrepancy; float64 tolerances<=2e-10. Necessary fixture enumeration
is never an actor or likelihood implementation. Run tests and commit/push; do not
run a new learning trial on the strength of algebra alone. Existing budget study
continues independently to its original terminal verdict and independent audit.

A later experiment would first need to measure whether discarded-proposal noise
is substantial in frozen teacher-free policies, before registering a bounded
matched physical-ratio-versus-proposal-ratio PPO comparison. General variance
reduction is known; any contribution would need a domain-specific tractable
coordination likelihood plus actual quality/health/mechanism/scaling evidence.

## Engineering result

The standalone helper and13 deterministic float64 tests PASS (3.37seconds).
Tests compare masses and autodiff gradients with complete proposal sums for all
four registered tiny N/K pairs, both with and without masked/forced-WAIT support;
they verify posterior scores, gradient means, variance decomposition, ratio
conditioning, the clipped Jensen inequality for both advantage signs, the .44
alias-split example, KL contraction, extreme logits and malformed inputs. No
simulator, trained checkpoint, learning outcome, scientific seed or physical
experiment was used. This is algebraic correctness evidence only; a diagnostic
of actual learner noise and a new prospective controlled experiment would still
be necessary. The current double-budget FULL has not been modified or replaced.
