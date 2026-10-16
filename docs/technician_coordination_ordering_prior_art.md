# Ordering and conflict-resolution directions after Exchange v2

Focused primary-source review on 2026-10-16, while the frozen diagnostic ran.
This is a boundary for future design, not a new registered algorithm, an
exhaustive novelty search or evidence that one of these papers outperforms our
maintenance policies. No external method has been implemented in this review.

| Primary source | Established mechanism | Consequence for a proposed contribution |
|---|---|---|
| [SeqComm, Multi-Agent Coordination via Multi-Level Communication](https://arxiv.org/abs/2209.12713) | Negotiates decision priority using communicated intentions evaluated through an environment model, then conditions later actions on earlier agents' decisions. | Learned execution priority and sequential communication alone are not new. A world-model or intention-value baseline must be disclosed if used. |
| [PARCO, Parallel AutoRegressive Models for Multi-Agent Combinatorial Optimization](https://arxiv.org/pdf/2409.03811), sections 2 and 4 | Combines agent communication, parallel pointer decoding and priority-based conflict handling for routing and scheduling. Its related-work discussion also identifies learned agent/action ordering. | Parallel bids, learned conflict priority and a scheduling application alone do not establish novelty. Its computational and feasibility assumptions need explicit comparison. |
| [ADAPT, Auction-Based Dynamic Prioritization for Multi-Agent Coordination](https://journals.sagepub.com/doi/10.3233/FAIA251227) | Uses compressed messages, dependency scores, a distributed priority auction and autoregressive action execution. | An auction that selects execution order is already an established direction. Do not infer network-local execution from our shared-graph broker. |
| [A2PO, Order Matters: Agent-by-agent Policy Optimization](https://arxiv.org/abs/2302.06205) | Studies sequential policy updates and their ordering, with an agent-by-agent optimization algorithm. | Policy-update order must be distinguished from the order in which assignments are decoded at execution. Its guarantees do not transfer to ordinary clipped PPO with a changed action representation. |

The earlier [Exchange prior-art matrix](technician_exchange_ppo_v1_prior_art_matrix.md)
covers local rewriting, matching policies and action decomposition. These sources
rule out presenting generic ordering, local repair, auction or parallel
arbitration as a new idea. A credible next contribution needs an exact mechanism
for this resource-allocation problem, a specific difference from these methods,
and evidence from the frozen cost/health/scale gates and matched controls. The
completed v2 pilot and diagnostic supply descriptive motivation only.
