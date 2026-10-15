# Frozen value-scale audit of the negative Completion-Credit pilot

Hypothesis: the finite checkpoint value heads have insufficient output magnitude
relative to the returns being regressed. This is a diagnostic hypothesis, not a
causal explanation of policy quality or a new algorithm contribution.

Offline audit only: all 66 saved zero/mid/final checkpoints from the 22 completed
pilot fits, every replica and all 16 already opened DEV seeds. Smoke uses selected
checkpoints, first existing replica/DEV seed, all 11 learned method/size pairs.
No simulator, new seeds, optimizer, training, scientific tests or checkpoint
changes. Reconstruct initial observations from the completed diagnostic's saved
physical states. These states are identical across methods within size/seed.

For a final linear value head with tanh-bounded input, report its exact envelope
`[bias - sum(abs(weights)), bias + sum(abs(weights))]`, including the documented
N/4 graph output scale. Verify the module structure and actual outputs within
this envelope. Measure initial empty-prefix prediction spread, saturation of
the last hidden tanh (abs >= .99), and first-decision observed MC return from
each checkpoint's original DEV costs (`-extended_cost/20`). Report mean returns
and fractions outside the envelope, without confidence intervals or gate changes.
The DEV trajectories are greedy while critics train against stochastic-policy
returns: disagreement is a diagnostic, not a direct unbiased calibration test
of the trained value function. Envelope statements are mathematical bounds for
the frozen heads, not bounds on what the architecture could eventually learn.

Require all source output hashes/sizes, unchanged learning-source/protocol/registry
identity, all diagnostic hashes, complete grid, matched initial populations,
checkpoint digests unchanged and all observed predictions inside the envelope.
Hard stop 120 seconds, no extension. Fresh UTC directory: copied script/plan,
per-checkpoint.csv, per-episode.csv, findings.md, manifest with bindings/versions.
Syntax check and smoke first, commit/push analysis code, show then run full audit
directly on this lab. This audit does not authorize full development/confirmation.
