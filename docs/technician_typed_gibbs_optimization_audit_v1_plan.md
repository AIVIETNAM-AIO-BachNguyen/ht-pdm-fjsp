# Frozen typed-Gibbs optimization audit — TRAIN/CFDEV only, no fitting

Confirmationed1002c FAILED4/12 CI gates and one replica primary sign. Preserve
this verdict and all old tests; no N>4 and no new training starts from this note.
Independent audit PASS102.59s proves execution, not robustness. Do not weaken
alpha, health thresholds or practical gains to reinterpret the failed test.

Exploratory evidence ALREADY read: on CFDEV (not TEST), Gibbs cost improved from
368640 to491520 in4/5replicas; one replica still fell markedly in the last segment.
Three myopic and one reverse control had identical CFDEV costs from122880 onward.
These are associations, not proof that longer budget fixes the mechanism or that
more encoder/LR/attention components help. Inspect frozen distribution movement
before deciding whether ONE bounded optimization-horizon follow-up is justified.

## Fixed input panel / no scientific-test reuse

Use immutable CF source technician_typed_gibbs_confirmation_v1_20261017T122835996955Z
and its independent audit technician_typed_gibbs_confirmation_verification_v1_
20261017T125736209944Z. Read manifest status/original verdict and necessary input
SHA bindings; do NOT decode selected_TEST_episodes, TEST states, bootstrap effects
or any TEST performance for this diagnostic. TRAIN progress and fullCFDEVepisode
CSV permitted. Load ALL3core methods/ALL5trainingreplicas/ALL5checkpoints
0/122880/245760/368640/491520. No selection of the failed test replica or an early
checkpoint as replacement winner.

Pool the five existing candidate FINAL CFDEV observation archives,48states each,
physical CFDEVseed380400000, total240recorded states. Keep duplicates and origin
replica/state labels. These states come from five candidate trajectories on ONE
exogenous physical CFDEVseed, not a representative performance evaluation panel.
Hold this same pool fixed for every method/replica/checkpoint. No environment,
new physical seed, rollout, action sampling, optimizer or teacher. Scientific
seed split not applicable to offline fixtures; original CFDEV roles bound and
TEST excluded.18000policy-state evaluations=3×5×5×240.

Manual NumPy neural features/logits from frozen own checkpoints and exact
explicit21matching probabilities via independent Gibbs/myopic reference. Enumeration
ONLY in this offline audit, never proposed production. Verify normalized mass,
entropy range and all file/parameter digests; head0 matches uniform feasibility.
Compute physical entropy / log(number of feasible labeled matchings) only for
supportsize>1; mode probability, fraction mode>=.99, utility range, and late
TV/disagreement between368640 and491520 on SAME fixed states. Exclude near ties
(top-two probability gap<=1e-10) from unique-mode disagreement. Record state
eligibility/opportunity counts. Scope all3matched policies; no latent MAPPO
entropy proxy or extra decoder advantage.

CFDEV descriptors use ALL64seeds per node, cost/availability/pending/failure;
TRAIN descriptors use final40rollout update records, actor/critic displacement
and preclip norm. Missing historical KL, returns/minibatches/gradients cannot be
reconstructed from these summaries; do not claim gradient-cause identification.

## Locked readout / bounded next-action rule

A model is flagged possible probability collapse if >=90% informative states
have mode probability>=.99, late meanTV<=.01 and its paired CFDEV latecostgain
<=.001. This describes a frozen representation, NOT proof of a bad optimizer.

Permit prospectively designing ONLY ONE2x-training-budget study (no LR/encoder/
head/objective change) if Gibbs pooled lateCFDEVcostgain>=.005, at least4/5own
replica late gains>0, and mean latephysicalTV across Gibbs replicas>=.01. The
proposed future study must give EVERYbaseline/control the SAME983040decisions,
fresh train/DEV/TEST panels, inherited quality/health/mechanism gates, no teacher,
no scientific-test reuse and fixed final models. This heuristic merely warrants
a controlled test; it is NOT evidence that extra training works or a contribution.
If ruleFAIL, do not launch a budget rescue from this audit. Preserve outcomes
and require a different evidence-led prospective hypothesis. No4x/budget sweep.

Tests: independent densities already validated by full CF audit. Add synthetic
uniform/degenerate masks, normalized entropy, fixed-state TV/tie handling and
threshold-boundary/missing-record rejection. No scientific physical state in tests.
Test/commit/push before new UTC offlineartifact, no smoke. Hardstop5minutes,tqdm,
refuseoverwrite. Artifact manifest/source/protocol/inputSHA/frozenprobe arrays/
all75node statistics/perstate probabilities/perrep late comparisons/readout.
Record training_steps=0,new_physical_states=0,test_outcomes_used=false,goal_pass=false.
No CI/causal/performance-success claim from this exploratory diagnostic. Verify
input/protocol/source seals again at completion. Goal stays ACTIVE.
