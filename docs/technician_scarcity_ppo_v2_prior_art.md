# Prior-art boundary for opportunity-message PPO v2

Read-only literature review during the frozen pilot, 2026-10-14. This note does
not amend any source, protocol, registry, endpoint or threshold. Novelty remains
provisional: a literature search cannot establish absence of earlier work.

| Primary source | Established component | Boundary for this project |
|---|---|---|
| [Wen et al., MAT](https://arxiv.org/abs/2205.14953) | Online on-policy sequential multi-agent decisions via encoder-decoder | Autoregressive joint PPO is not new here; no MAT guarantee transfers automatically. |
| [Kuba et al., HAPPO/HATRPO](https://arxiv.org/abs/2109.11251) | Advantage decomposition and sequential multi-agent policy updates | Our joint-trace update is not HAPPO, and has no new monotonic-improvement theorem. |
| [Alomrani et al., learned online matching](https://arxiv.org/abs/2109.10380) | End-to-end RL policies for bipartite matching | Learning allocation without enumerating complete matchings is prior art. |
| [Curry et al., differentiable auctions](https://arxiv.org/abs/2106.07877) | Differentiable Sinkhorn allocation for learned auctions | No incentive compatibility or revenue-maximizing auction claim for our heuristic messages. |
| [Tan et al., auction MARL for V2X](https://arxiv.org/abs/2208.04237) | Auction interaction and long-term RL resource allocation | Communication and resource prices in MARL are established. Our full-graph broker cannot claim their partial-information deployment properties. |
| [Link et al., scheduling coordination gap](https://arxiv.org/abs/2604.24117), v3 2026-09-21 | Joint vs modular scheduling depends on scarcity and temporal bottlenecks | Motivates reporting mechanism and physical failure/backlog/availability, rather than generalizing any one-setting coordination advantage. |

Candidate-specific hypothesis, not yet established: a bounded duration-weighted
exclude-own opportunity message improves maintenance allocation over an otherwise
identical reservation broker with matched information, initialization, capacity
and optimization budget. Separate actor/value clipping is a conventional numerical
intervention, measured separately against a globally clipped priced control.

Evidence required by the locked plan: positive pooled large-setting price effect
above 0.5% with simultaneous approximate bootstrap bounds on untouched seeds,
small-setting PPO noninferiority and MAPPO improvement, scaling and all physical
health limits. Pilot/development screens are exploratory and cannot prove these.

Limitations of this review: primary abstract-level inspection, not a completed
systematic full-text review. DARO (Scitepress 2026 paper 149200) appeared in search
but primary PDF retrieval returned HTTP403; its detailed mechanism has not been
verified and is not used as evidence of a specific methodological difference.
