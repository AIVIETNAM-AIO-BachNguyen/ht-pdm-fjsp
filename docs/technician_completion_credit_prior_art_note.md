# Prior-art boundary for the next completion-credit design

Literature/design note only, before any new algorithm experiment. Does not
amend the failed v3 protocol, thresholds or artifacts. Novelty is provisional.

[COMA](https://arxiv.org/abs/1705.08926) uses a centralized action-value critic
and marginalizes one agent's action while fixing the others. A learned value of
an autoregressive reservation prefix is a different conditioning scheme;
do not call it COMA or claim that generic credit assignment is new.

[MAT](https://arxiv.org/html/2205.14953v2), especially sections2.2–4, already
combines sequential multi-agent decisions, prior actions and on-policy learning.
An autoregressive technician policy alone is not a novel general algorithm.
MAT/HAPPO theoretical guarantees do not transfer automatically to shared
parameters, masks or a different clipped surrogate.

[POAD](https://arxiv.org/html/2405.15821v1), sections4–5.2, is particularly close:
it models partial-action values and optimizes individual action components with
PPO. Its critic uses Bellman targets within/across complete actions and its actor
uses component advantages. Therefore prefix critics and component clipping
must be treated as established prior art, not renamed as our invention.
Our candidate direction would use scratch-trained resource policies, reservation
and completion descriptors, and Monte Carlo continuation targets on the same
physical tapes. Whether that domain-specific mechanism is a useful contribution
must be established by matched controls; this is not a new BAD theorem.

[Branching Policy Optimization](https://arxiv.org/abs/2607.14171) explores
alternate actions by snapshotting and forking executable environments. The
intended technician design will not branch the simulator or gain extra physical
samples; no corresponding variance theorem should be asserted.

[CANON](https://arxiv.org/abs/2509.23962) groups sampled responses using a target
metric. Similar wording “conditional advantage” does not mean that method is
implemented by a prefix-value baseline. Avoid ambiguous algorithm naming.

Research contribution must remain concrete: an allocation/learning mechanism
that handles service occupation and restoration using current information,
avoids matching enumeration, and survives cost/health/coordination ablations.
Include a generic decomposed-policy control or explicitly identify the matched
state-baseline/component-ratio control as an adaptation, not a reproduction of
POAD's published implementation. Do not claim broader novelty from this focused
review. No teacher/fine-tuning escape from the user's requested contribution.
