# Exchange PPO v2: ordered commitment and the learned policy

This explanatory appendix describes implementation commit `d42072c` and the
unchanged [registered protocol](technician_exchange_ppo_v2_plan.md). It adds no
training setting, seed, threshold, stopping rule or performance claim. The v2
pilot is still running when this appendix is written. Its purpose is to separate
mechanical properties from hypotheses requiring experiments.

## One physical action, two bounded internal passes

For observed physical state s, let E(s) be the legal machine-technician edges.
A provisional matching x assigns each technician either one machine or zero.
The actor samples a trace z of exactly 2K conditional tokens. The first K slots
construct x in a fixed technician order. The second K slots repair it in that
same order. Only the final matching starts physical services; internal slots
consume no simulated time and receive no intermediate reward.

At repair slot j, the candidate can KEEP, DROP a nonempty assignment, replace
it with a legal unclaimed machine, or request a machine held by technician
h > j. The last operation swaps their provisional assignments if the reciprocal
edge is legal. If j was unassigned, it transfers the requested machine to j
and leaves h unassigned. An already active physical service is never exchanged.

This differs from a centralized PPO actor with a categorical output over a
catalogue of complete matchings. The learned actor scores local edges and
conditions later choices on provisional assignments. PPO remains the learning
objective. The proposed change is the action construction and its coordination
rule, learned from random weights, without a teacher or baseline checkpoint.

## Feasibility, commitment and the limits of those properties

Feasibility follows by induction over internal slots. Start with the empty
matching. Construction adds a legal edge to an unused machine. DROP removes
an edge. Replacement releases the old edge and selects an unclaimed legal
one. A swap checks both recipients and retains distinct machines; a transfer
changes the recipient of one legal edge. KEEP changes nothing. Therefore every
coordinated provisional matching and the final matching are feasible. This
argument does not describe the independent control's conflicting proposals;
its final arbitration removes collisions.

After repair slot j, technician j's assignment cannot change again: later
technicians can change their own assignments or exchange only with a holder
of greater index. Thus the repaired prefix is immutable.

Let xC be the construction matching. If there is a non-KEEP repair, take the
earliest such slot j. All previous repairs were KEEP, so j still holds xC[j]
immediately before its repair. Each permitted non-KEEP token changes that
assignment: duplicate KEEP and DROP from an empty assignment are forbidden.
Commitment makes this changed assignment persist. Consequently:

    any non-KEEP repair => final matching differs from xC.

This proves zero complete repair cancellation for the candidate. It does not
prove that the change reduces cost, improves availability, shortens service,
prevents failure or improves gradient variance. Nor does it make traces unique:
constructing a matching and keeping it can reach the same final matching as
constructing an empty plan and adding a legal assignment during repair.

Every feasible matching still has at least one trace: construct its edges and
KEEP throughout repair. Restricting repair permissions therefore preserves
physical action support. It changes trace support and parameterized action
probabilities; support preservation does not prove equivalent finite-network
expressivity, trainability, or an absence of technician-order bias.

## What probability PPO optimizes

For a fixed arm, the broker decoder D_s and each prefix-dependent legal mask
are deterministic and do not depend on learned parameters. The actor uses

$$p_\theta(z\mid s)=\prod_{l=1}^{2K}
p_\theta(z_l\mid s,z_{<l}).$$

Forced tokens have log probability and entropy zero. Replay reconstructs the
recorded prefixes and masks; it does not replace them with newly greedy choices.
The joint PPO ratio is

$$r(z,s)=\exp\left(\sum_l\log p_\theta(z_l\mid s,z_{<l})
-\sum_l\log p_{old}(z_l\mid s,z_{<l})\right).$$

One physical-state Monte Carlo advantage is used for the whole trace in the
ordinary clipped PPO surrogate. The separate critic consumes physical state
once, without the sampled construction or repair choices. The original physical
cost supplies the return, including the common deadline tail. There is no
imitation loss, teacher label, repair reward or duration penalty.

The stochastic probability of physical matching a is instead

$$q_\theta(a\mid s)=\sum_{z:D_s(z)=a}p_\theta(z\mid s).$$

The implementation does not enumerate this sum. For fixed Q(s,a), the finite
sum identity

$$\nabla_\theta\sum_a q_\theta(a\mid s)Q(s,a)
=\sum_z p_\theta(z\mid s)Q(s,D_s(z))
\nabla_\theta\log p_\theta(z\mid s)$$

explains assigning the physical return to the sampled trace. It does not
establish an unbiased finite-batch clipped/standardized PPO update or a new
policy-gradient theorem. PPO clipping does not impose a hard KL bound. The
entropy regularizer averages conditional entropies at recorded old prefixes;
it is not the exact gradient of full-trace entropy. These qualifications are
unchanged from the [v1 derivation](technician_exchange_ppo_v1_derivation.md).

## What the controls can identify

All six core arms have the same feature maps, 19,255 parameters, initial tensors,
2K slots, optimizer settings and physical training budget. The coordinated
arms have the same observable physical information and provisional-message
format. Their realized prefixes can differ because their legal moves differ.

| Comparison | Mechanism difference being tested |
|---|---|
| Candidate vs unrestricted exchange | Ordered commitment through forward-holder permissions |
| Candidate vs inverse-mask exchange | Prefix commitment beyond forbidding only the immediately preceding bilateral inverse |
| Candidate vs self repair | Bilateral requests to unrepaired holders |
| Candidate vs construct only | An actionable second pass; all 2K neural slots are still computed in the control |
| Candidate vs independent | Shared provisional intentions and coordination, versus own intentions plus collision arbitration |

Generic local rewriting, sequential action decomposition and anti-reversal
masking are established ideas; see the [prior-art matrix](technician_exchange_ppo_v1_prior_art_matrix.md)
and registered protocol. Ordered commitment's domain-specific implementation
and empirical benefits remain a proposed contribution. Zero cancellation alone
cannot establish novelty or a useful learned coordination mechanism.

## Scaling and the evidence still required

The candidate scores N+2 tokens per technician slot, without a catalogue of
complete matchings. With fixed hidden width, the registered conservative bound
is O(NK²) work. Algebraic edge gathering and batched recorded-prefix replay
reduce implementation cost without changing this claim or the learned policy.
Inference remains sequential and skips the critic. A shared current graph and
broker are assumed; this is not a network-local execution protocol. Sizes are
trained separately, so this experiment cannot establish zero-shot transfer.

Measured cost, health, actor latency and matched-control results must meet the
registered gates. Full development requires a passing completed pilot. Any
scientific claim additionally requires the untouched confirmation panel and
registered confidence intervals. No mechanical property above substitutes for
those empirical requirements.
