# Frozen stochastic-policy credit diagnostic: prospective procedure

Register before new rollout/gradient outcomes. POSTHOC DEVELOPMENT DIAGNOSTIC,
not confirmation. Original RSC and factored capacity remain failed; no training,
teacher/distillation targets, weight updates or scientific-test opening here.
Previous goal turn made progress: completed 576 registered T1 interventions and
independent artifact verification; the deferral explanation failed. Do not add
duration penalties, a legal-option GO head or always-WAIT from those failures.

## Question and sources

The PPO implementation uses complete-episode MC advantages at gamma=1. Current
critic units already normalize the graph value loss. Inspect value predictions
under the actual stochastic policy, not greedy-policy returns. Is shortening
the policy credit trace a plausible bounded optimization follow-up, given its
value error and finite-batch policy-gradient variability? This is an engineering
question; a positive signal would not establish benefit of GAE or capacity.

Frozen sources:
- RSC development technician_residual_service_capacity_ppo_v1_development_20261016T032534946293781Z,
  b44f114a8bc44ca2fefdfcd4eadf18d1e0f5322d,519 bindings.
- Factored development technician_residual_capacity_factorization_v1_development_20261016T064203318666531Z,
  a10e049e3f8b4e242efbc879b2d3dc3c13925bf4,548 bindings.

Use only protocol-selected checkpoints, no checkpoint reselection. For all three
replicas per source: RSC candidate and no-capacity; factored candidate,
no-capacity and matched flat; factored original PPO and normalized PPO. Exactly
21 models. Cross-source head initialization/training seeds differ: source-family
comparisons are descriptive, not causal normalization comparisons.

## Fresh diagnostic data and fixed budget

New registry: physical raw seeds240100000..63, action raw240200000..2.
Verify disjointness from historical registries and derived-stream collisions.
Every model runs exactly64 stochastic episodes, vectorized64 at48 decisions
plus12common deadline-tail ticks. Same64 physical tapes for every model, derived
using the existing factored-environment physical namespace. Core arms share a
replica-index-specific action RNG stream across sources; PPO/normalized PPO share
their own role. Different supports/distributions can consume draws differently;
no assertion of pathwise-equal actions. No epsilon, greedy rollout, new training
or per-state counterfactual branch. 1,344 episodes /64,512 policy decisions /
80,640 physical ticks in total. 15-minute hard stop, torch/BLAS1, tqdm, fresh
timestamp artifact, no overwrite, tests/commit/push before launch, no smoke.

These are new DEVELOPMENT diagnostic seeds; not scientific confirmation. Fixed
policies are evaluated on held-out tapes but replication unit remains training
seed. No population CI or p value. A frozen actor generates its own data and
critic targets; no policy supplies labels/initialization to another model.

## Measurements and mathematics

Store every sampled action, behavior log probability, value, reward and terminal
episode metrics. Reconcile sum rewards=-extended_cost/20,48decisions,60ticks,
zero invalid assignments and original physical populations across methods.
Replay current log probabilities/values on the recorded actions within original
source tolerances. Retain hashes of checkpoints and unchanged parameter digests.

For every episode/time, independently form float64 MC return G and compare with
the existing learner's float32 mc_returns (atol5e-4). For fixed gamma=1 and lambda
in {1,.95}, compute delta_t=r_t+V_(t+1)-V_t with terminal V=0 and
A_t=delta_t+lambda*A_(t+1). Lambda1 must equal G-V within1e-10 in float64; this is
an identity check, not evidence of estimator accuracy. No new value training.

Per model report return/value means, RMSE, explained variance, MSE skill relative
to constant return mean, and cross-fitted time-only reference (even/odd episode
folds predict each other's mean G_t); report state residual skill descriptively.
Report raw MC/GAE variance ratio, mean-advantage shift divided by MC sd and
centered advantage cosine. The mean shift is NOT a verified bias estimate.
GAE can bias credit when the value function is imperfect.

Compute actor-only score gradients on four predefined episode blocks0..15,
16..31,32..47,48..63 using full physical-decision log probabilities. Advantages
are centered/scaled ONCE across all3,072states separately for MC and GAE. Loss
is -mean(log pi * stopgrad advantage), i.e. the unclipped PPO surrogate gradient
at its behavior parameters. Exclude entropy/value terms to isolate credit; use
no optimizer. Save each block gradient vector. Define finite-block dispersion
D=mean||g_b-mean(g)||²/max(||mean(g)||²,1e-20), and cosine between mean MC/GAE
gradients. This is a noisy four-block diagnostic, not a population variance
estimator or proof of an unbiased gradient. Record zero/tiny-gradient cases.

## Fixed engineering eligibility and limits

A new GAE.95 learning trial is motivated for a source family only if EACH of
its three candidate replicas has: critic explained variance>=.5; absolute mean
prediction error<=.1*return_sd; GAE block D<=.75*MC block D; mean gradient cosine
>=.8; and MC mean gradient squared norm>1e-20. All conditions required, no
threshold tuning. If neither family qualifies, do not automatically train GAE.
If both qualify, prefer original RSC, whose existing DEV cost is lower; do not
choose from the new diagnostic costs. A positive gate only motivates a separate
prospective random-init comparison of MC vsGAE across same-information core
ablations AND baselines at equal budget, preserving old failed verdicts and all
health/mechanism gates. No warm start, extra training budget or new encoder.

GAE is established prior art, not our contribution:
[Schulman et al., GAE](https://arxiv.org/abs/1506.02438) describes the bias/variance
tradeoff; [PPO](https://arxiv.org/abs/1707.06347) already uses advantage estimation.
Any contribution remains a validated capacity-coordination mechanism, not changing
a standard PPO estimator. These sources do not predict success in this environment.

Artifacts: manifest, protocol/code/registry snapshots and hashes, source bindings,
seed audit, checkpoint inputs, all sampled decisions/rewards/values/actions,
episode grid, float64 credit arrays, block gradient arrays, model statistics,
per-family fixed eligibility verdicts, report and untouched-source after-checks.
Technical error aborts as FAILED. Manual rsync -avhP --partial only; no SSH or
automatic artifact transfer.
