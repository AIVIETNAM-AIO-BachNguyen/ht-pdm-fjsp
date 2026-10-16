# Residual-capacity factorization v1: bounded, evidence-led follow-up

Register before implementation or new learning outcomes. Latest user explicitly
permits necessary tuning: an unsuccessful configuration need not refute the broad
capacity hypothesis. Preserve the original RSC v1 negative verdict, all data and
sealed scientific tests; no retrospective rescue. N4/K2 only. Random weights,
standard scalar PPO, no teacher, distillation, old model, warm start or labels.

## Evidence, hypothesis and one intervention

Frozen post-hoc diagnostic source ac32b1c; artifact
`technician_rsc_learning_diagnostic_v1_20261016T063242688272079Z` verified519source
bindings,135curve points,75core checkpoint probes, no new trajectory/update.
Candidate last bid-layer2 |tanh|>.99 fraction averages.962770 on16fixed DEV states;
its same-weight zero-message TV is.008745. First two final replicas have100%
bid-layer2 saturation and M1/M2 hidden contrasts.000811/.0000587 although capacity
embeddings differ2.338/1.090. The third retains.0603contrast. These are associations.
All3 selected the final checkpoint and pooled final-quarter cost improved1.53%,
so insufficient training was plausible; strong lost-contrast evidence instead
motivates ONE normalization intervention first, keeping the original budget.
No2x run or LR/encoder/attention search is launched here.

Hypothesis: the flat per-technician softmax couples raising service probability
to a common upward shift in all machine scores, encouraging saturation that
removes differences conveying later-technician capacity. Separating the service
mass from conditional machine ranking will allow capacity-dependent choices to
learn, improving cost and health at the same physical training budget.

For legal set L, machine scores b_m and WAIT-head score w:

    c = |L|
    z = w - log(c)
    P(WAIT) = sigmoid(z)
    P(m) = sigmoid(-z) * softmax_L(b)_m.

No legal machine forces WAIT probability1; invalid tokens have zero support.
The fixed log(c) offset matches the flat uniform initialization, NOT a claim
that GO/WAIT is cardinality-invariant. Adding a constant to every legal b leaves
the factored full policy unchanged; it changes GO mass in the flat control.
Machine-ranking score gradients sum to zero and GO-mass loss has no gradient
through bid logits in the factored policy. Test these elementary identities;
they do not establish better returns, lower variance or monotonic improvement.

Greedy deployment ALWAYS argmaxes the complete N+1 token probabilities. Do not
use separate binary-threshold greedy: it can select a different physical token
and confound decoding with parameterization, as earlier scarcity diagnostics found.
Sampling uses the same one categorical draw per technician; canonical likelihood
is the product over K physical technician tokens. No detached normalizer,
straight-through gradient, component-specific clipping or entropy bonus.

## Six matched core arms and unchanged learning

- residual_capacity_ppo: factored probabilities, true remaining-capacity message,
  reservations and completion prefixes exactly as originalRSC.
- no_capacity_ppo: same factorization, zero3-channel message.
- initial_capacity_ppo: factorized, initial static capacity of ALL technicians.
- permuted_capacity_ppo: factorized, true message cyclically shifted+1machine row.
- independent_capacity_ppo: factorized, initial state only/no reservations;
  duplicates arbitrated by restoration/duration then lower technician index.
- flat_capacity_ppo: true RSC message/reservations with ORIGINAL flat token
  softmax. Isolates the normalization, same new common initialization.

Same original raw22/prefix features, capacity horizons1/2/4, H27, actor11423 /
critic8587 =20010 parameters, no new network/measurement. Reset ONLY the final
bid and WAIT output linear weights/biases to0 in EVERYcore arm; capacity last
layer remains0 as inRSC. All other tensors retain shared random initialization.
Thus every coordinated core arm has exactly the same initial physical policy,
including flat control, at every common legal prefix. Independent matches local
conditionals at common prefix/mask, but its later mask intentionally differs.
This common head initialization differs from original RSC v1; cross-round
performance cannot isolate its effect. The new matched flat arm is essential.

Same per-technician forward order, scalar state critic, MC rewards from48steps
plus common12tick deadline tail; reward=-cost/20,gamma1, normalized advantages,
critic unit115.2, MSE in normalized units, value weight.5, actor/critic clipping.5.
PPOclip.2, Adam3e-4, entropy.01 on the COMPLETE token categorical entropy averaged
across K;4epochs/minibatch256. No new learning objective or extra experience.
K neural conditional evaluations for sample/replay/select, same for all core.
Conservative O(N*K^2), no full matching enumeration or simulator future lookahead.
Different normalization tensor operations disclosed; no equal wall-time claim.

OriginalPPO18966, normalizedPPO18966, localMAPPO15364 and fullMAPPO18052 parameters
are freshly trained external controls. Information/architecture differences
remain disclosed; localMAPPO alone cannot prove the proposed contribution.

## Prior art and limited contribution

Generic policy factorization is established, e.g.
[Sharma et al.2017 FAR](https://arxiv.org/abs/1705.07269).
Autoregressive coordination is established, e.g.
[MAT](https://arxiv.org/abs/2205.14953); clipped objective is ordinary
[PPO](https://arxiv.org/abs/1707.06347). These papers are context, not implemented
comparisons or a theorem transferable to this model. Earlier local scarcityv3
already tried global GO/STOP then edge choice and failed; do not rename it as
new generic hierarchy. This follow-up is per-PHYSICAL-technician WAIT/service
versus machine ranking with no global STOP, original canonical support, matched
initial probabilities and complete-token greedy decoding. Contribution, if
confirmed, is capacity-aware maintenance coordination with evidence that this
specific parameterization permits its use, not inventing hierarchical RL.

## Fresh panels, full budgets and stopping

Registry fixes DEVtrain230000000..2/DEV230100000..31/analysis230200000;
CFtrain230300000..4/DEV230400000..63/TEST230500000..255/analysis230600000;
engineeringtrain230700000/DEV230700001..2/analysis230700003.366raw seeds, no prior
raw/derived overlap; separate VERSION namespace. Pair all methods' exogenous
physical tapes and core initial/action/minibatch RNG; original/normalizedPPO
pair raw tensors. No old trained weight is loaded.

DEV30full fits (10methods*3replicas), each491520 joint decisions, vector16,
rollout3072,7680optimizer steps. Total14,745,600decisions/307200train episodes;
4800DEVepisodes at5CPs(0/122880/245760/368640/491520)*32plus32deadline.
CF50fresh full fits (5replicas),24,576,000decisions/512000train episodes,
16000DEVplus64deadline,12800learned TESTplus256deadline episodes.
Spend every full budget and retain all replicas, including selectedstep0.
Earliest minimum mean DEVcost chooses each checkpoint; choices sealed before
probes/timing/tests.8workers,torch1,tqdm,hard12h/grid, first technical failure
aborts siblings and preservesFAILED; no quiet overwrite/restart/extension.
Tests and verified commit/push before FULL directly on Ubuntu lab. No separate
smoke run per active user override. N4 confirmation must pass before largerN.

## Sixteen unchanged-or-stronger mandatory endpoints

Keep all original15quality/health/behavior thresholds; add required candidate
cost gain>=.005 vs matched flat normalization control.9cost+6health+1behavior=16.
Cost lower gain>=.005 vs all5matched controls,>=-.01 vs bothPPO,>0 vs bothMAPPO.
Health differences vs BOTH PPO: availability lower>=-.01, pending/machine
upper<=+.025, failures/machine upper<=+.25.0invalids; isolated selected candidate
actor p95<=50ms on retained DEV48states*3repeats/replica after pool closes.
Original16fixedDEVmechanism states, and disjoint16fixedCFstates, unchanged;
critical-first P(M2) candidate-minus-zero difference lower>=.05. It is only a
constructed-state decision proxy, never a label, training input or replacement
for physical outcomes. Track P(WAIT), P(M2|nonWAIT), bid saturation and
capacity-embedding/bid-hidden contrasts descriptively; no additional fitting.

Keep unused round2alpha.05/6, Bonferroni16 individual two-sided confidence
1-(.05/6)/16; paired crossed bootstrap200000 of replicas and physicalepisode
seeds. Probe resamples ONLY paired replicas, fixed16states not resampled.
Approximate coverage with5replicas; report all replicas/leave-one-out. Optional
PPO superiority requires lower gain>=+.01 vs BOTH PPO using same intervals.
DEV screens same POINT thresholds; no success/CI claim. Exact audited passing
DEV/source/protocol/registry/choices required before NEW scientific panel opens.
If this bounded follow-up fails, retain it; do not automatically change LR,
activation, budget, head init, thresholds or open old tests. User allows necessary
tuning; a further change needs evidence and its own prospective protocol.

## Tests and artifacts

Test normalized probabilities/forcedWAIT, full-token greedy, bid-shift invariance,
zero gradient of GO-only loss wrt ranking and zero-sum rank score gradients,
flat sensitivity and exact initial probability/RNG/action parity, all21 canonical
matchings (25independent proposals), replay/finite gradients/critic state-only,
unchanged capacity calculations/no future inputs, independent conflicts,
parameter counts, checkpoint roundtrip, serial/spawn equivalence,16endpoint
threshold/bootstrap/null/missing grids, bound prerequisites and physical60tick
replay. Reuse original experiment runner safeguards with fresh VERSION/tapes.

New timestamp artifacts: source/protocol/registry/physics/clean pushed commit,
seed audit, every checkpoint, initial hashes/counts, complete training/DEV/test
journals, sealed choices, fixed mechanism vectors/knockouts, retained observations,
isolated latency, independent scalar/budget audit,16gate report and hashes.
Keep post-hoc diagnostic evidence separate and unchanged. No SSH/transfer;
provide exact manual rsync -avhP --partial when outputs are available.
