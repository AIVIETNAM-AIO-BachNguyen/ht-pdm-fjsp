# Boundary of the v3 algorithm contribution

Literature-only note during engineering validation. Does not change the locked
learning source, protocol, splits, endpoints, margins or stopping rule.

Conditional action factorization and masking are established techniques.
[Bamford and Ovalle, Conditional Action Trees](https://arxiv.org/pdf/2104.07294),
sectionsII-B/III-C/III-D/IV, describe action components, masks depending on earlier
choices, and their joint conditional probability. Their actor-critic instantiation
uses IMPALA. Our GO/STOP then edge hierarchy is a domain-specific conditional
policy; its product likelihood and conditional masks are not a new general method.
The reservation arm supplies a matched conditional-policy control, not a claim
to reproduce the paper's code or benchmark results.

[Hu and Kaneko, Hierarchical Advantage/PAPPO](https://ieee-cog.org/2021/assets/papers/paper_211.pdf)
study PPO with discrete action choices and continuous parameters, including
specialized advantage baselines for the two parts. V3 has discrete matching
components and a single shared state-value baseline with joint PPO ratios.
It does not implement their hierarchical advantage or prove a new decomposition.

The intended research contribution remains provisional: a maintenance allocation
mechanism that couples duration-weighted other-machine opportunity messages with
a GO/STOP probability/execution rule that avoids pair-count effects. Required
evidence separates messages from factorization, uses identical information,
initialization/capacity/budget controls, and enforces cost plus physical health
limits on untouched seeds. Generic decomposition, PPO training, resource
reservation and separate gradient clipping must not be presented as new.

See the preserved v2 review for MAT, HAPPO, learned matching, auctions and
the2026 scheduling coordination-gap analysis. These checks are a focused review,
not a systematic novelty guarantee or permission to declare v3 successful.
