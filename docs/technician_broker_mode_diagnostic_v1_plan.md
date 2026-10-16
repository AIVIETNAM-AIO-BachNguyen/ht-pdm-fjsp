# Frozen broker matching-mode execution diagnostic v1

Register before any new diagnostic outcomes. Source broker-marginal FULL failed
12/13 (matched cost gain.085%<.5%) and independent verification PASS. Preserve
that verdict. No confirmation, largerN, new learning, checkpoint reselection,
teacher/distillation/warm-start or reward/optimizer/entropy change.

## Single falsifiable diagnostic hypothesis

The trained candidate optimizes the physical matching law q(T(U)) but deployment
uses individual greedy proposals followed by arbitration. Those operations need
not produce the mode of q. Test whether that specific execution mismatch has a
material cost effect in the frozen trained policies. Maximizing probability is
NOT itself maintenance optimality. A negative result rejects this proposed
execution remedy, not the mathematical marginal-gradient identity.

Use all24 ORIGINAL selected checkpoints from frozen learning commit
dcef1f554bf54172aed7f61cfdefe8d8e16b70d9, original choices unchanged. Compare
both greedy and exact physical matching-mode execution for ALL8 methods x3replicas
on64FRESH paired physical diagnostic seeds300100000..63, using source physical
namespace technician_broker_marginal_ppo_v1:N4:physical. No action RNG or new
analysis seed. Total3072episodes/184320physical ticks/147456policy decisions.

All methods receive the same execution comparison: no candidate-only inference
privilege. Original PPO/norm already have21joint tokens, so physical mode MUST
match their original greedy execution exactly. For all four core arms integrate
independent technician proposals through their own fixed quality/ID broker.
For local/full MAPPO integrate independent MACHINE proposals through the original
per-technician highest-urgency/lower-machine-ID arbitration. The latter is the
transposed winner-only map, not the technician quality broker. Never compare
candidate mode against an artificially weaker baseline decoder.

The N4 exact-mode REFERENCE may enumerate21feasible matchings. This is a frozen
DIAGNOSTIC, not a scalable policy or a new matching-catalogue algorithm. Original
actor weights are frozen; no q-mode checkpoint selection, best-of-seeds or
stochastic action search. If useful, a future decoder trial still needs a
polynomial implementation, matched controls/budget, new protocol/seeds and full
training from random weights; no transfer of these frozen weights as warm-start.

## Probability and tie conventions

Read actual float32 neural scores (core conditional log weights, baseline masked
logits), apply original legal supports, then normalize in float64 for stable
reference categorical probabilities. This preserves greedy argmax and defines
the reference physical law up to ordinary floating-point roundoff. Core actors
remain independent across technicians; MAPPO actors across machines. For a
matching, a winner must request its resource and an idle agent may WAIT or
request an occupied resource where it loses to the prescribed winner. Sum this
preimage by the factorized product-of-sums formula. Validate normalization over
all feasible physical matchings at every query. Tests additionally compare it
with exhaustive proposal sums (<=25core/81MAPPO) on tiny deterministic fixtures.

Exact mode is maximum physical matching mass, tied by the existing canonical
PPO matching-catalogue order. No reward-based tiebreak, temperature or threshold.
Greedy submits original model.select proposals; mode submits the unambiguous
representative of the chosen physical matching (othersWAIT). Record original
greedy proposal, greedy physical pairs, mode pairs and model matching log masses.
These scores are NOT behavior log likelihoods of deterministic mode execution.
Representative mode proposals have no conflicts by construction; this is not
learning evidence. Also record expected latent proposal-rejection count for the
underlying independent law to avoid conflating that with zero executed conflicts.

At each policy state compare greedy and mode physical pairs, always at the SAME
current state. Primary mismatch rate uses CANDIDATE GREEDY trajectories only
(3rep x64seeds x48decisions). Mode-trajectory mismatch is descriptive. Actor/critic
weights unchanged before/after; baselinePPO/norm greedy-vs-mode full trajectories
must match. Source/input bindings unchanged before/after.

## Locked screen / rejection

A future polynomial-decoder learning trial is motivated ONLY if ALL hold:
- candidate mode improves own greedy mean H60cost by >=.005;
- own improvement is strictly positive in EACH of3replicas;
- primary same-state physical mismatch rate >=.01;
- original13point gates pass when ALL methods use their matching-mode execution
  (main gain vs quality-proposal>=.005, ID/PPO controlsNI>=-.01, bothMAPPO gains>0,
  six original availability/backlog/failure bounds), with zeroinvalid.

This is a practical engineering screen on fresh diagnosticDEV, not a CI,
scientific superiority test or a replacement source-FULL verdict. Even if all
pass, source remains failed and CF/largerN stay closed pending a new registered
full learning comparison. If any fail, reject THIS exact-mode remedy and do not
auto-launch beam/search/temperature, extra training or decoder tuning from it.
Do not promote a control after seeing it wins. Dense N4 timing is diagnostic
compute only; it does not replace original isolated p95 or establish scalability.

## Execution / artifacts / validation

Commit/push verified code first; directlab FULL diagnostic, no separate smoke.
No active training workers, one torch/BLAS thread, tqdm across all episodes,
15minute fail-fast hardstop, fresh timestamp directory, preserve failed runs.
Audit new64raw/64derived seeds against all historical raw registries and known
learning/diagnostic derived streams, including source family. CF/test states
remain unconstructed. Store protocol/registry/source/verification/choice/code
hashes,24checkpoint hashes; full physical/policy records, paired episode metrics,
3rep means and both mode summaries, original13point math and locked screen.

Use pure dictionary dynamics and seeded event tapes; cross-check EVERY transition
against the repository scalar WaitingEnv transition, cost components and state.
H48health and H60cost remain distinct; common12tail unchanged. Unit fixtures:
all25/81preimages versus reference physical masses, priority/ties/masks/forcedWAIT,
mode-greedy counterexample, original PPO mode equality and canonical token order,
source actor/local/full input reconstruction, short deterministic paired rollouts,
scalar/dictionary physics, source-selection/seed/gate boundaries. No real source
learning outcomes or fresh300panel states in disposable tests. No extra scientific
endpoint, physical oracle future events, test-seed opening or artifact transfer.
