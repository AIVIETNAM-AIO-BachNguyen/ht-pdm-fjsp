# Typed-capacity Gibbs matching policy — provisional design and algebra gate

Design/correctness work only. Capacity-order06c1638 FULL failed12/16 and ALL3
replicas lost to fixed order. Preserve that verdict and all old negatives; do not
promote its fixed-order control or retune the capacity heuristic. No scientific
test/CF/N>4 opens from this document. No learned actor/critic is implemented here.

## Specific next hypothesis

A local token policy normalizes a current choice without weighting the learned
utility of future feasible assignments. Arbitrary processing order can change
that policy even when every trace is legal. A joint physical matching policy
whose continuation messages aggregate the learned utilities of remaining
machines may coordinate competition for worker capacity more directly. Whether
this improves scratch PPO cost is UNPROVEN and needs a separate full protocol.
It is not a claim that the last failure was caused solely by greedy decoding,
encoder, insufficient training or a critic defect.

Proposed current-state neural utilities u_theta(s,m,t) for assigning machine m
to a technician of type t. They are learned policy logits, NOT simulator future
costs, Q-oracle labels or a teacher. A future learner would keep standard PPO,
random initialization, own-policy returns and state critic. At N4 a possible
shared raw actor has state71+ownmachine6+type-edgeprofiles6+machineID4+free-type
counts2=89 inputs,hidden64/output2; actor10050+critic8833=18883parameters. Exact
architecture/loss/budgets/controls/seeds must be prospectively fixed BEFORE any
learning. Do not use this proposal as permission to launch an unspecified sweep.

## Structural assumption and scope

The current benchmark repeats TWO technician service profiles (columns by parity)
at every configured size. Free technicians within each profile have identical
duration/restoration/compatibility across machines. Treat them as labeled members
of a type, retain individual matching identities in probability, and use counts
(a,b) only for availability bookkeeping. Busy workers are excluded; existing
services and their remaining times remain in the original physical state.
Typewise eligibility may differ across machines, but not within one type.

This is polynomial for FIXED TWO TYPES, not arbitrary K distinct technician
profiles: Ctypes need a count-state product prod_t(K_t+1). No unrestricted
permanent/partition-function polynomial algorithm or zero-shot/fully decentralized
claim. Increasing N/K within two profiles is a possible later scaling study;
N4 must pass first. Raw actor parameters may grow with state dimension.

## Exact joint policy, no matching enumeration

For a labeled feasible partial matching M, define

    E_theta(s,M) = sum_(m,j in M) u_theta(s,m,type(j))
    pi_theta(M|s) = exp(E_theta(s,M)) / Z_theta(s).

WAIT contributes0; empty matching legal. Matching support and physical objective
unchanged. Compute weighted suffix partition F_i(a,b) for remaining machines
i..N-1 and a/b free labeled workers of each type:

    F_N(a,b)=1
    F_i(a,b)=F_(i+1)(a,b)
              + a exp(u_i0) F_(i+1)(a-1,b)
              + b exp(u_i1) F_(i+1)(a,b-1).

Omit unavailable/ineligible type terms. The a/b multiplicities count WHICH
labeled worker is selected. Work O(N(K0+1)(K1+1)), log-space messages; no list of
matchings or latent proposal preimages in production. These are elementary
weighted-matching recurrences, not a new generic combinatorial theorem.

Sample WAIT or a type with probabilities proportional to the three terms, then
uniformly select one currently free labeled worker within that type. Concrete
worker probability divides the type probability by its current count. Exact
physical log probability telescopes to E_theta(M)-logZ. Different machine
processing orders yield the SAME physical distribution for the SAME utilities.
Do not stop-gradient logZ or the learned continuation messages in actor density.
They are part of the policy, unlike the detached baselines in the retired study.

Exact labeled entropy can be computed by suffix recursion on conditional type
probabilities, adding p_type*log(available_type_count) for label entropy. The
global physical MAP matching uses max-product/Viterbi with concrete-worker log
probabilities, NOT the most probable aggregate type. No log(a)/log(b) term may
artificially reward a physical MAP merely because a type has more labels.

## Future matched mechanism control, not a learned result

Use SAME neural utilities/state critic/capacity/info/budget. A myopic normalized
policy replaces learned future F_(i+1) messages by unweighted feasible-count
messages C_(i+1) (same eligibility, utilities0); CURRENT choice still uses learned
u_i. Normalize each conditional exactly, retain reservations. Its full physical
likelihood is the product of those conditionals; it generally DOES NOT equal
exp(E)/Z. This is a valid distinct policy, not a biased detached-gradient variant
of Gibbs. Both policies are uniform over feasible labeled matchings at u=0.

Both can have exact physical entropy and exact physical MAP via count-state DP,
so avoid giving only the candidate a stronger decoder/entropy calculation. Reverse
myopic order should be reported; global Gibbs distribution order invariance is a
correctness property, NOT performance evidence. Future training must show benefit
over same-info/capacity myopic control plus PPO/MAPPO and original health bounds.
No actor samples/future costs from an existing PPO are used as labels or weights.

## Prior art / contribution boundary

Autoregressive constrained allocation and sampling-bias corrections are prior art
([NeurIPS2024](https://arxiv.org/abs/2409.18735)); multi-agent decoding/conflict
coordination is prior art ([PARCO](https://arxiv.org/abs/2409.03811)). Gibbs
distributions over matchings are also established ([SODA2026]
(https://epubs.siam.org/doi/epdf/10.1137/1.9781611978971.6)). RL matching policies
are established ([Alomrani et al.](https://arxiv.org/abs/2109.10380)). Do not claim
first Gibbs matching, first partition DP, first RL matching or a novel PPO
objective. This focused review is not an exhaustive novelty clearance.

Provisional contribution would be a teacher-free maintenance policy using exact
learned typed-capacity continuation messages, with evidence that this particular
coupling improves small-setting quality/health and scales under stated profiles.
The recurrence and algebra gate alone cannot establish that contribution.

## Fixed engineering algebra gate before a learning protocol

Deterministic synthetic utilities/eligibility only; no physical environment,
checkpoint, scientific RNG panel, dataset or optimizer. Enumerated references
ONLY in tests for N2/K2, N3/K3(type counts2/1), N4/K4(2/2), N5/K4; unavailable
workers and type masks; zero/positive/negative and extreme utilities. Seed split
not applicable; all fixtures explicit constants, no scientific state constructed.

Compare logZ, every feasible labeled log probability, totalmass1, entropy and its
FULL gradients, logZ gradients/marginals, PPO logratio gradient, physical MAP and
group-label multiplicities against independently enumerated reference. Float64
mass/value/entropy tolerance1e-10, gradients1e-9, MAP1e-10. Verify all24N4machine
orders at fixed utilities give Gibbs equality; show a fixed nonzero utility
fixture differs for myopic orders (TV>1e-3), and both have identical cold
uniform distributions. Fixture u=[[1.2,-.8],[-.7,1.8],[.4,.3],[.2,-.2]].

Tests hard limit60seconds, anyFAILpreserve/reject algebra implementation; no
relaxed tolerances. Tests/commit/push before a fresh timestamped correctness
audit artifact (code/protocol/Git seal, fixture schema, test records/results,
explicit training_steps=0/test_states=0/goal_pass=false). No separate smoke.
Only exact PASS permits drafting a separately registered scratch learning study,
not opening tests/N>4 or claiming better cost/scientific goal completion.
