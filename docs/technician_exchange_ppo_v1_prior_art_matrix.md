# Exchange PPO v1: focused prior-art comparison

Primary sources checked on2026-10-15 while the registered pilot was running.
This document clarifies positioning; it does not change that pilot's source,
protocol, seed panels, budgets or gates. The review is focused rather than
exhaustive, and does not establish that the proposed combination is novel.
Statements about Exchange PPO refer to frozen learning commit `3a96ecb`.

| Primary work | Established ingredient | Comparison with the current implementation | Claim boundary |
|---|---|---|---|
| [Schulman et al., Proximal Policy Optimization Algorithms,2017](https://arxiv.org/pdf/1707.06347), section3, equation7 | Clipped probability-ratio surrogate with repeated minibatch updates. | Exchange PPO applies ordinary clipping to a joint latent trace ratio, with a physical-state Monte Carlo advantage. | PPO clipping and the policy-gradient identity are inherited; there is no new trust-region theorem. |
| [Chen and Tian, Learning to Perform Local Rewriting for Combinatorial Optimization, NeurIPS2019](https://proceedings.neurips.cc/paper/2019/file/131f383b434fdf48079bff1e44e2d9a5-Paper.pdf), abstract/section1 | Actor-critic learns region and rewrite choices, including for online job scheduling. | Our maintenance broker constructs and revises provisional assignments in a fixed2K slots before starting service. Internal moves receive no improvement reward or extra physical rollout. | Learned local rewriting, including a scheduling application, is established. A bounded maintenance-specific exchange mechanism still needs empirical and novelty assessment. |
| [Wu et al., Learning Improvement Heuristics for Solving Routing Problems,2019 preprint/revised2020](https://arxiv.org/abs/1912.05784), abstract | Reinforcement learning selects solution improvements for routing using an attention-based policy. The paper reports generalization across problem sizes. | Our action process revises a pending service allocation under reciprocal eligibility and current service calendars. | Neither learned improvement nor size generalization is a new generic contribution. Our registered sizes are trained separately. |
| [Wen et al., Multi-Agent Reinforcement Learning is a Sequence Modeling Problem,2022](https://arxiv.org/abs/2205.14953), abstract | MAT models coordinated agent actions as a sequence using an encoder-decoder and multi-agent advantage decomposition. | Our shared edge encoders and move heads produce construction/repair tokens. The core learner uses one physical-state advantage and a joint trace ratio. | Sequential cooperative action generation is established. We do not implement MAT's update or inherit its performance theorem. Complexity claims require comparing the same computational scope. |
| [Wen et al., Reinforcing Language Agents via Policy Optimization with Action Decomposition,2024](https://arxiv.org/abs/2405.15821), abstract and introduction | POAD decomposes language actions into intra-action tokens and introduces Bellman backup with Action Decomposition for token credit. | Our internal slots are maintenance allocation moves, with a state-only critic and shared Monte Carlo advantage. We do not use BAD or a token-specific critic in the core arms. | Action decomposition and finer intra-action credit are established. The current algorithm must not be described as a new decomposition theorem or POAD-equivalent credit assignment. |
| [Alomrani et al., Deep Policies for Online Bipartite Matching: A Reinforcement Learning Approach, TMLR2022; first preprint2021](https://arxiv.org/abs/2109.10380), abstract | End-to-end RL learns online matching decisions with uncertainty about future arrivals. | Our environment repeatedly allocates heterogeneous technicians to maintenance jobs, retaining active services and permitting only provisional exchanges. | RL-based matching under uncertainty is established. A different application alone does not prove algorithmic novelty. |
| [Begnardi et al., Deep Reinforcement Learning for Two-sided Online Bipartite Matching in Collaborative Order Picking, PMLR222,2024](https://proceedings.mlr.press/v222/begnardi24a.html), abstract | Two-sided human/robot coordination uses GNN-based DRL, with experiments on larger test instances. | Our representation uses shared pair encoders, pooling, and an explicit bounded broker over a shared current graph. Sizes are independently trained. | Graph-based matching coordination and transfer across problem sizes are established. We claim neither the first scalable matching policy nor zero-shot size transfer. |

## What is actually being tested

The candidate design permits a technician to revise a provisional allocation by
changing its own job or bilaterally swapping/transferring another provisional
job, subject to both recipients' eligibility. It rebuilds known completion
descriptors after every internal move and preserves already active services.
The complete bounded trace is sampled and replayed exactly, and only its final
allocation affects physical time and original cost. These are implementation
facts, not evidence that exchanges improve decisions.

The central empirical question is whether bilateral permission helps compared
with self-repair under the same physical information, provisional messages,
architecture, initialization,2K neural decision slots, objective, optimizer count and
physical training budget. Construct-only also removes an actionable repair
pass. Independent proposals remove coordination messages and cross moves.
The separate completion reference uses a different architecture and K slots,
so its contrast cannot isolate bilateral permission at equal internal compute.

The exact trace likelihood, chain rule, baseline cancellation, fixed return
units, scratch initialization and elementary KL decomposition are correctness
and design ingredients. They are not generic mathematical novelty. See the
[implementation derivation](technician_exchange_ppo_v1_derivation.md).

## Evidence needed before a positive research claim

The registered pilot and development gates must pass before confirmation.
Confirmation must establish the cost, health, mechanism and latency endpoints
with its locked seed panels and uncertainty procedure. Algorithmic complexity
alone cannot establish measured latency or retained quality. Runtime exchange
counts alone cannot establish beneficial exchanges or causal mediation.

The current protocol compares PPO, normalized PPO, MAPPO variants, the fresh
completion reference and matched core controls. It does not implement a
head-to-head maintenance adaptation of MAT, POAD, NeuRewriter or the cited
matching policies. Consequently those publications supply prior-art context,
not measured baselines that the current results can be said to outperform.
A publication claiming broader superiority would require additional
prospectively designed comparisons and a wider novelty review.

No outcome or threshold is amended by this comparison. Preserve a negative
pilot or failed mechanism gate as negative evidence rather than claiming
novelty or success from the existence of the implementation.
