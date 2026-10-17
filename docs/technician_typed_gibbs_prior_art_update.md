# Typed-Gibbs prior-art boundary — update during frozen FULL

Read-only literature follow-up after source51ebbec was frozen. This note does
not amend its hypothesis, panels, budgets, gates or final-checkpoint rule. No
learning outcomes were used to select these references. Focused primary-source
review, not exhaustive novelty clearance.

- Zhang, Aberdeen and Vishwanathan, ICML2007, [Conditional Random Fields for
  Multi-agent Reinforcement Learning](https://www.cs.uic.edu/~zhangx/papers/ZhaAbeVis07.pdf),
  sections2.1–2.3/3: joint exponential-family action policies, global partition
  normalization and policy-gradient learning were already combined. They discuss
  structured inference/sampling for multi-agent policies. Thus joint normalization,
  learned messages and differentiating a partition function are NOT generic new
  ideas here. Our inference: the proposed weighted-versus-unweighted continuation
  contrast is related to established globally/locally normalized policy models;
  algebraic order invariance alone cannot establish novelty.

- Hoppe et al., [Structured Reinforcement Learning for Combinatorial
  Decision-Making](https://arxiv.org/html/2505.19053v2), version2 (2025),
  sections2.1/2.2/appendixB.1: neural scores plus a combinatorial actor and a
  regularized distribution on structured actions are established. Their implemented
  learner uses perturbed optimization, action-value targets and Fenchel–Young
  actor updates; our frozen study uses exact physical matching density and ordinary
  on-policy clipped PPO. This distinguishes implementations, not a claim that
  score-and-optimize or teacher-free structured RL is new. Its entropy-regularized
  theoretical distribution includes the familiar exponential-score form.

- Kong et al., [Latent Spherical Flow Policy for Reinforcement Learning with
  Combinatorial Actions](https://arxiv.org/abs/2601.22211), abstract/version2
  (2026): a stochastic latent policy is mapped through a combinatorial solver;
  latent-space value learning is used for efficiency. We did not reproduce their
  code or experiments, and the abstract alone is insufficient for implementation
  equivalence or an empirical ranking. It further rules out claiming first
  stochastic teacher-free solver-backed combinatorial RL.

## Consequence for this research

Keep the provisional contribution narrow: exact labeled matching PPO under TWO
repeated physical technician profiles, concrete count-state coordination with
matched retraining, health constraints and later measured scaling. The recurrence
is elementary weighted counting. We have NOT established a new generic RL
algorithm, a new objective, or a performance contribution. Even a DEV pass needs
confirmation and a defensible domain-specific contribution; do not equate a new
module name with scientific novelty. Existing autoregressive allocation/PARCO/
matching references from the original design still apply. No comparisons above
substitute for a reproduced baseline on this maintenance environment.
