# Joint latent-utility clearing PPO v1: prospective N4/K2 protocol

Lock before implementation/learning outcomes. New mechanism, not a rescue of
failed RSC/factorization, marginal-order, completion, scarcity or exchange rounds.
Run FULL on this Ubuntu lab, no separate smoke; tests/commit/push first. Random
weights, no teacher, bank, distillation, warm start, future events or cost oracle.
N4/K2 first. Do not open scientific tests or larger-N after a failed DEV gate.

## Evidence and falsifiable hypothesis

Prior RSC capacity probes largely saturated; separating GO/ranking increased
capacity sensitivity but FULL cost/health failed2/16. Frozen service-choice and
576single-T1 interventions did not support a slow-GO or avoid-long-job fix.
Frozen stochastic-credit diagnostic21models/1,344episodes showed fairly calibrated
selected critics; neither family passed prelocked GAE eligibility. Keep the
standard MC PPO estimator and normalized critic, not a new optimizer sweep.

Test simultaneous coupled clearing: each agent's bid for a machine is evaluated
together with the alternative bids for both resources before any reservation is
made. Hypothesis: a learned shared utility matrix with joint one-to-one clearing
learns near-PPO maintenance allocation and beats uncoordinated proposals while
retaining quality relative to irreversible serial clearing, without enumerating
full matchings. A greedy exact-MAP decoder on the older marginal categorical
policy previously gave zero gain; this new latent policy/learning parameterization
is different, and success is not inferred from that failed decoder experiment.

## Policy and likelihood, fully specified

Current information unchanged: machine6, technician3, edge4, global9 ->raw22
per edge. H32 tanh MLP encodes each edge; scorer receives edge, machine mean,
technician mean and global mean (4H), shared scalar utility mean mu_mj. No new
attention, identities, auxiliary prediction, horizon message or hand-set duration
penalty. One NN actor pass for all bids. Separate physical-state critic exactly
the existing marginal policy's H32 raw/calendar/context/value construction;
unit U(N)=28.8N, loss MSE(V/U,G/U), no actor/value parameter sharing.

For every currently legal urgent edge sample u_mj=mu_mj+sigma*epsilon_mj, epsilon
iid Normal(0,1); one explicit RNG draws a full N*K array each decision. Masked
coordinates are deterministic zero and contribute zero log density/entropy.
Shared learnable log-sigma=2*tanh(rho/2), rho initial0 ->sigma1. Train the exact
SUM Gaussian log density over legal coordinates, not probability of the decoded
matching. Every feasible matching has positive-probability latent regions. A
fixed state-dependent decoder maps u to starts; its rules do not depend on theta.
PPO is ordinary joint latent-action clipped PPO, with state-only MC advantage.
Many latent arrays decode to the same matching; latent ratios and entropy are
NOT marginalized matching ratios/entropy. No straight-through, Sinkhorn gradient,
oracle matching labels or critic gradient through the solver.

Decode quantized q=round(1e6*u) to integer utilities. WAIT has utility0. Ignore
nonpositive q, including zero ties; select a maximum-weight feasible partial
bipartite matching by polynomial min-cost flow. One unit per technician, direct
WAIT arc to sink, technician->machine arcs cost -q, machine->sink capacity1.
Stable fixed arc order; equal positive-weight optima follow deterministic solver
tie handling, no claim of permutation-equivariant tie selection. Reject nonfinite
u or |u|>1e6, avoiding int64 overflow. Solver optimizes learned utility, never
true future cost or a simulator search. No listing feasible matchings. Greedy
deployment decodes means, not a claim of MAP matching under the latent policy.

Four core arms share parameters/random tensors, latent support/distribution,
Gaussian RNG, PPO settings, optimizer/physical budget; only decoder differs:
- joint_clearing_ppo: simultaneous maximum-utility matching.
- forward_clearing_ppo: technician0..K-1 takes its best positive remaining bid,
  reserves that machine; equal q chooses lower machine index.
- reverse_clearing_ppo: same but reverse technician order.
- independent_clearing_ppo: each technician chooses best positive bid on original
  support; duplicate machines retain highest bid, then lower technician index;
  losers WAIT with no reassignment.

Common latent initialization does not imply common physical initial policies
because their decoders differ. Record that difference rather than claiming exact
matching-distribution equality. Decoder removal ablation tests the whole joint
clearing service, including preventing/dropping conflicts. Forward/reverse remain
coordinated alternatives; their noninferiority gates are not superiority evidence.
At selected candidate weights also report frozen forward/reverse/independent
decoder rollouts on the same open DEV panel, without labels or retraining;
descriptive mechanism checks, not extra training replicas or scientific tests.

New external original PPO, normalized PPO, local MAPPO and full-info MAPPO are
trained from scratch at the same budget. Local MAPPO has less actor information;
it alone cannot establish contribution. Core parameter count is expected about
19,843 (verify before run); PPO/norm18,966, local15,364, full18,052. Disclose
capacity differences; every mechanism ablation has exactly equal capacity.
Normalized/full/local critic treatment follows the preceding controlled runs;
literal original PPO retains its original unit1/global clipping.

## Prior-art boundary

Structured matching/latent-action RL is established, not invented here:
- [Sinkhorn Policy Gradient](https://arxiv.org/abs/1805.07010) learns permutation
  policies using a continuous Sinkhorn relaxation; ours uses exact integer
  clearing, partial matchings and Gaussian latent-density PPO, no relaxation.
- [Wolpertinger](https://arxiv.org/abs/1512.07679) embeds actions and uses nearest
  neighbors; ours constructs legal assignment by flow without action catalogue.
- [Action Mapping](https://arxiv.org/abs/2412.04327) decouples feasibility from RL;
  ours has a fixed known combinatorial decoder, no pretrained feasibility policy.
- [Projection agents](https://arxiv.org/abs/2605.19721) already decode continuous
  graph-action embeddings. Do not claim first latent projection or general novelty.
- [PPO](https://arxiv.org/abs/1707.06347) supplies the unchanged clipped objective.

Provisional contribution is a teacher-free maintenance coordination design and
its cost/health/ablation/scaling evidence. This focused review is not proof of
broad novelty or comparisons against unimplemented published algorithms.
Agents access the shared current graph and a shared clearing service. This is
a hybrid learned-bid/coordinator design, not communication-free decentralized
execution; network transport/robustness and a deployment distribution are not
validated by simulator timing. The coordinator's objective is learned utility,
not a separately supplied maintenance objective or privileged information.

## Budget, seeds, stopping and artifacts

Frozen balanced_a2_b4_u12 N4/K2,48policy+12common deadline tail, reward=-cost/20,
gamma1. 491,520 decisions/model,16vector envs, rollout3,072,4epochs,minibatch256,
Adam3e-4,clip.2,entropy.01,value.5,gradclip.5. Entropy is mean Gaussian entropy
per active latent coordinate (0 if none), not physical matching entropy. This
differs from categorical baselines, common to all4core arms. Graph actor/critic
clip separately. Checkpoints0/122880/245760/368640/491520; earliest minimum mean
DEVcost per replica, spend whole budget regardless of which checkpoint wins.
8CPU workers, torch/BLAS1, tqdm training and multi-seed evaluation;12h hardstop.

DEV24fits (8methods×3replicas),11,796,480decisions/245,760trainEP/184,320Adamsteps;
3,840DEVepisodes+32deadline reference+288frozen-decoder episodes. Same physical
training tapes per replica/worker and common DEV tapes for all methods. Core
initial/action/minibatch streams shared; PPO/norm paired initialization/streams.
Registry DEVtrain250000000..2/DEV250100000..31/analysis250200000; CFtrain250300000..4,
DEV250400000..63/TEST250500000..255/analysis250600000; engineering250700000..3.
Raw ranges and derived streams audited against history before launch. Scientific
panels remain unopened; deriving seed hashes does not instantiate their states.
No extra training, adaptive seed growth, early success stopping or checkpoint reuse.

Fresh UTC artifact directory, never overwrite: RUNNING/COMPLETED/FAILED manifest,
Git/runtime/protocol/registry/source snapshots+hashes, physics, parameter/initial
tensor checks, full streamed training progress/episodes, DEV grids/choices,
checkpoints, selected actor-only observations/timing, exact decoded assignments,
frozen-decoder metrics, quality/health summary, audit and all output bindings.
Track proposal conflicts/rejections, failures, availability, backlog, service
occupancy and cost decomposition. Tests must cover density math/gradients/replay,
float latent preservation through PPO training, masks/forced-WAIT, sample RNG,
flow optimality against enumeration ONLY in unit tests, greedy counterexamples,
non-enumerating core path, copy/save roundtrip, pairing, budgets and gate semantics.
Core actor has O(NK) NN work plus polynomial flow; unlike K neural reservation
stages. Quantify actual timing including decoder, no unmeasured scaling claim.

## Locked N4 gates and future confirmation contract

13point DEV endpoints, all mandatory:
1. Cost gain vs independent>=.005 (coordination-removal mechanism).
2. Cost gain vs forward and reverse>=-.01 each (coordinated references).
3. Gain vs original and normalized PPO>=-.01 each.
4. Gain vs local and full-info MAPPO>0 each.
5. Versus BOTH PPO references: availability delta>=-.01, pending/machine<=+.025,
   failures/machine per48ticks<=+.25 (six endpoints).
Also zero invalid, isolated candidate NN+decoder p95<=50ms. No superiority claim
over serial controls from noninferiority; no benefit claim from zero conflicts
alone. Report every failed endpoint and all replicas; do not relax health limits.

Only a fully audited passing DEV can authorize fresh5-replica confirmation at
the same budget with64DEV/256sealedTEST, checkpoint choices sealed before tests,
crossed paired bootstrap200,000 resampling train and physical seed axes. Family
13, round2alpha=.05/6 (unused by these DEV failures), confidence1-alpha/13.
Require simultaneous CI bounds satisfying the SAME endpoint thresholds. Claim
superior PPO cost only if its gain CI lower>=.01, locked practical threshold;
near-PPO suffices for quality contribution paired with scale/coordination evidence.
Implement/verify the confirmation harness before opening any scientific test;
this DEV runner must not instantiate that panel. N/K scaling/generalization
requires a separate prospective protocol after confirmed N4, not before.
No SSH or automatic transfer; provide rsync -avhP --partial for result copies.
