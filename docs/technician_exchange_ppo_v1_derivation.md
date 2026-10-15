# Exchange PPO v1: policy and learning derivation

This appendix describes implementation commit `3a96ecb`. It does not amend the
registered protocol, thresholds, seeds or running pilot. Performance and novelty
remain empirical questions. The identities below explain the representation;
they are not claimed as new policy-gradient or trust-region theorems.

## A physical decision contains a bounded internal trace

Fix the current observable physical state s, its legal urgent-service pair mask
E(s), N machines and K technicians. Let x_j be technician j's provisional machine,
with zero meaning unassigned. Initially every x_j is zero. A latent action is
z=(z_1,...,z_2K): one construction and one repair slot for each technician, in
fixed order. The deterministic broker produces a final assignment D_s(z).

During construction, a free technician may WAIT or select an unclaimed legal
machine. During repair it may KEEP, DROP its own assignment, replace it with an
unclaimed legal machine, or request another technician's machine. In the last
case the two assignments are swapped, provided the reciprocal pair is legal.
An unassigned requester instead transfers that machine to itself and leaves its
former holder unassigned. Busy physical services are excluded by E(s).

For the coordinated arms, feasibility follows by induction. The empty matching
is feasible. Construction adds at most one legal edge to an unused machine.
DROP removes an edge; an unclaimed replacement releases the old edge first.
A swap replaces two distinct legal edges with two distinct legal edges after
checking both recipients. A transfer moves one edge to an eligible recipient.
KEEP changes nothing. Thus every provisional matching and the final matching
are feasible, and existing physical services are untouched. This argument does
not apply to the independent arm's provisional duplicate proposals; that arm's
final broker resolves conflicts by dropping losing proposals.

Every matching permitted by E(s) has a valid trace: construct precisely its
edges, then KEEP in every repair slot. This establishes support, not an ability
of the finite network or training budget to learn every desired distribution.
Greedy deployment may select a different matching from sampling.

## Trace probability and physical probability

Each legal conditional is a masked categorical distribution. Its support is
determined by s and the recorded prefix, without using network parameters:

$$p_\theta(z\mid s)=\prod_{l=1}^{2K}
 p_\theta(z_l\mid s,z_{<l}),\qquad
 \log p_\theta(z\mid s)=\sum_l\log p_\theta(z_l\mid s,z_{<l}).$$

Forced slots have probability one, log probability zero and entropy zero.
Replaying an old trace reconstructs every prefix and mask before evaluating its
new conditional probability. The behavior replay checks validate this routing.

Several traces can decode to the same physical matching a. Its probability is

$$q_\theta(a\mid s)=\sum_{z:D_s(z)=a}p_\theta(z\mid s).$$

One trace probability is not this marginal. The implementation never computes
that sum or treats a trace ratio as the ratio of the two physical marginals.
Only the final matching enters the physical transition; the internal slots do
not advance time or receive intermediate rewards.

For a fixed action-value function Q(s,a), the following finite-sum identity
explains why the trace representation can receive the physical return:

$$\nabla_\theta\sum_a q_\theta(a\mid s)Q(s,a)
 =\sum_z p_\theta(z\mid s)Q(s,D_s(z))
       \nabla_\theta\log p_\theta(z\mid s).$$

A physical-state baseline b(s) cancels because
sum_z p_theta(z|s) grad log p_theta(z|s) = grad 1 = 0. The implemented critic
therefore needs no sampled construction or repair action. This identity alone
does not establish an unbiased finite-batch, normalized, clipped PPO estimator,
variance reduction, convergence, or improvement of the greedy deployed policy.

## Implemented update

The optimization follows the clipped surrogate of
[Schulman et al., PPO, equation 7](https://arxiv.org/pdf/1707.06347), applied to
the complete latent trace. With old behavior parameters theta_old,

$$r_i=\exp\left[\sum_l\log p_\theta(z_{il}\mid s_i,z_{i,<l})
 -\sum_l\log p_{\theta_{old}}(z_{il}\mid s_i,z_{i,<l})\right],$$

$$L_{clip}=\operatorname{mean}_i\min\{r_i\hat A_i,
 \operatorname{clip}(r_i,1-\epsilon,1+\epsilon)\hat A_i\}.$$

The implementation forms raw advantages G_i - V_old(s_i) and standardizes them
over the rollout batch. G_i is the physical Monte Carlo reward return, with
gamma=1 and reward equal to negative original cost divided by20. There is one
state-only critic call per physical decision. Repeating its scalar into2K slots
for the learner interface leaves the mean value loss equal to
mean_i[(V(s_i)/U-G_i/U)^2], where U=12*N*48/20. Actor and critic parameters and
gradient clipping groups are separate.

The minimized loss is -L_clip -0.01*H_bar +0.5*L_value, with epsilon=0.2.
H_bar averages the current conditional entropies over2K slots and old sampled
prefixes. It is an entropy regularizer evaluated at recorded prefixes. Its
gradient does not include how theta changes the distribution of future
prefixes, so it must not be described as an exact full-trace entropy gradient.
The main candidate and all three core controls use this same update.

## What trace clipping can and cannot imply

For a fixed state and the same decoder D_s, grouping traces by final matching
gives the elementary decomposition

$$D_{KL}(p_{old}\Vert p_\theta)=D_{KL}(q_{old}\Vert q_\theta)
 +\sum_a q_{old}(a)D_{KL}(p_{old}(z\mid a)\Vert p_\theta(z\mid a)).$$

Consequently the stochastic physical KL is at most the trace KL. This follows
by substituting p(z)=q(a)*p(z|a) within each group; the conditional KL terms are
nonnegative. It assumes a common fixed decoder and suitable support. It does
not compare different ablation decoders. PPO clipping does not impose a hard
bound on the trace KL, so the decomposition supplies no new trust-region or
monotonic-performance guarantee. Trace changes that only redistribute mass
within a matching's group can consume optimization effort without changing the
physical policy. The extra trace length may also make joint ratios difficult
to optimize as K increases; this must be judged from the experiments.

## Computational and experimental implications

With fixed hidden width, static pair encoding costs O(NK). Each of2K slots
rebuilds/pools current pair features in O(NK) and scores O(N) choices. Neural
work is O(NK^2); inference working memory is O(NK) with fixed width. Training
retains slot activations and has a larger polynomial memory cost. These are
representation bounds, not measured latency or guarantees of good scaling.
There is no complete-matching catalogue in the candidate. Sizes are trained
separately and a shared current graph/broker is assumed.

Exchange versus self-repair is the primary bilateral-permission ablation:
same physical information, provisional messages, architecture, initialization,
2K calls, objective, optimizer count and physical decision budget. The
construct-only contrast also removes the actionable second pass. The separate
completion reference has a different architecture and K slots; it cannot serve
as an equal-internal-compute control. Generic learned local rewriting already
appears in [Chen and Tian, NeuRewriter](https://proceedings.neurips.cc/paper/2019/file/131f383b434fdf48079bff1e44e2d9a5-Paper.pdf).
The contribution under test is the bounded maintenance allocation mechanism and
its measured cost, health and scaling behavior, subject to the prior-art
boundary in the registered plan.

Correctness evidence is in `tests/test_technician_exchange.py`: independent
scalar/tensor agreement over reachable small prefixes, full physical support
and duplicate-trace mass, immutable active services, machine permutations,
physical-only critic calls, and exact joint-trace learning/budgets. These tests
cover specified finite cases; they do not replace the performance experiments.
