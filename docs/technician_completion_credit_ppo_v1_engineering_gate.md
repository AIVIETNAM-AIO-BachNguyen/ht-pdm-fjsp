# Completion-credit PPO v1 engineering gate

Final amended architecture: hidden27,18444 parameters per graph arm at every
size. Nonlinear state/calendar fusion before per-technician critic pooling,
nonlinear resource contexts, known decoder position/remaining-role inputs.
All four controls have identical parameter tensors at initialization and identical
critic supervision. Only registered actor-credit/surrogate/coordination changes.
No teacher, distillation, warm start or old learned checkpoint.

## Verified final code

- Final targeted completion tests:37 PASS in11.83s. Include unique21-match
  probability mass, exact replay, forced WAIT, WAIT then assignment, finite
  gradients, calendar formula/cache, machine permutation, no future event/large
  catalogue, pre-choice baseline, resource-contention capacity witness,
  reservation-identity fusion, remaining decoder roles, statistical35-family,
  prerequisites, checkpoint metadata and serial/spawn scratch determinism.
- Original PPO384-step training is bitwise equal to the frozen v3 learner at
  `ed0c784`, including final parameter hash and audit. Baseline path remains
  unchanged; allocation-specific learning is gated to new graph models.
- Final complete suite:1107 PASS,1 existing tqdm/rich warning,748.04s.
  Log `/tmp/technician_completion_v1_amended_full_tests_20261014T192500Z.log`.
- Final fresh Linux smoke `technician_completion_credit_ppo_v1_smoke_20261014T193438Z`:
  COMPLETED/auditPASS in29.52s.11 scratch fits,88 training episodes,16 paired
  training tapes,66 DEV episodes,26 engineering-test episodes,1560 independent
  physical/coordination replay ticks.0 invalid assignments/teacher/pretraining.
- Independently rehashed all331 output files with byte-size verification;
 248 current source/protocol/registry bindings match the frozen smoke snapshot.
  Smoke honestly records an uncommitted working tree before the code commit.
  The commit containing this gate records the verified source; research-run
  manifests will record the exact pushed commit.

## Preserved engineering amendments and interpretation

The earlier1104-test PASS and smoke runs184841Z/190045Z apply to hidden30/global
critic pooling, not the final architecture. Those33 checkpoint files and all
nontiming episode fields were identical across the two runs; protocol wording
clarification did not change behavior. Both runs remain preserved. A deterministic
N4 structural counterexample then showed a global-edge-mean critic exactly
aliasing two states with maximum one-tick matching capacities1 and2, before and
after the same reservation. This justified the prospective representation fix
before pilot. An initial random-output assertion was replaced with a deterministic
capacity witness: random weights are not expected to know resource pressure.
Fixture weights are never trained, saved or transferred into experiment models.

Credit vs state-credit isolates actor baseline conditioning with identical actor,
critic and all-prefix critic training. Candidate vs joint-credit changes both
baseline and surrogate; its internal `pooled_clipping_gain`/`clipping_mechanism`
keys describe the full learning recipe and do not establish clipping alone.
No threshold, seed, budget or scientific alpha changed. No pilot or scientific
test outcomes were opened during these amendments.

These tests/smokes establish engineering readiness only. No scientific efficacy,
PPO superiority or novelty claim follows. Pilot22fits on fresh70400000..1/
70500000..15 precedes full development; scientific71200000..255 stays closed.
Generic prefix critics/component PPO already have prior art, notably POAD.
