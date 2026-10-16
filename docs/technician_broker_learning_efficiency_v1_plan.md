# Broker physical-marginal PPO: bounded learning-efficiency study v1

Register before ANY new learning outcome. This is a DIFFERENT hypothesis from
the failed983040step final-quality study, not a revised verdict on that study.
Its asymptotic/mechanism gain.085%<.5% remains failed. Frozen exact-mode remedy
and conditional-Q/H4opportunity audit remain rejected. No source weights, oracle
Q labels, teacher, distillation, warm-start, mask change or extra information.

## Evidence and falsifiable question

Opened old curves at122880steps: quality physical-q270.211 vs matched proposal-p
277.146, gain2.502%, positive in3replicas (3.697%,3.582%,.050%). At245760steps
candidate is.173%WORSE; late performance converges. These are exploratory OLD
data, not evidence on this study's fresh panel or an excuse to select that early
checkpoint. Test whether removal of irrelevant latent-proposal score noise
improves EARLY learning over a fixed interval, while maintaining endpoint quality
and health. Noise reduction does not guarantee a clipped-PPO cost improvement.

The policy/algorithm is EXACTLY the frozen broker-marginal design dcef1f5: raw
shared103->64->64->5actor,71->64->64->1critic, independent technician proposals,
quality/ID winner-only broker without retry, no matching catalogue in core
sample/replay/select. Candidate physical q(T(U)) joint PPO ratio, control latent
p(U) joint ratio. All4core arms share initial19974parameter tensors; all8arms
have identical freshly random private critic tensors. Same original PPO-MC,
normalization/clipping/entropy(auxiliary proposal entropy), Adam3e-4, clip.2,
value.5,entropy.01,gamma1,16envs,rollout3072,4epochs,minibatch256. Standard greedy
proposal+broker deployment for ALLcore methods, no mode reference privilege.

Unchanged arms: broker_marginal_ppo(candidate),broker_proposal_ppo(primary
mechanism control),id_marginal_ppo,id_proposal_ppo,PPO,PPO normalized,MAPPOlocal,
MAPPOfull. Do not promote a control after results. Same information/budgets and
declared baseline capacities as prior source: PPO/norm18966,local15364,full18052;
mechanism ablation capacity EXACTLY matched. Same balanced_a2_b4_u12,H48+12tail.

## New fixed budget and measurements

All24DEV fits train from scratch for EXACTLY245760decisions each; no early
stopping or selected-checkpoint primary score. Fixed checkpoint/eval steps:
0,15360,30720,61440,92160,122880,184320,245760. Every checkpoint evaluated on the
same32freshDEV physical seeds. Total5898240decisions,122880training episodes,
92160Adam updates,6144checkpoint DEV episodes.8workers/1torch+BLAS thread each.

Primary learning-efficiency cost AUC = trapezoidal integral of mean H60cost/machine
over these exact eight points, divided by245760. Integrate separately for each
train-replica and physical seed; weight by true step gaps, NOT average points or
best-checkpoint cost. Include step0; identical candidate/control initialization
there is intentional. AUC has units cost/machine, no extra oracle reference or
post-hoc window. Positive gain=1-meanAUC(candidate)/meanAUC(control).

Final quality/health and isolated432sample raw NN+broker latency use ONLY the
fixed245760checkpoint for every method. Save original source driver's earliest
min-DEV choices as ANCILLARY for its existing budget audit, never primary
selection or permission to substitute a different endpoint. Preserve source
learner checkpoint metadata protocol technician_broker_marginal_ppo_v1; outer
experiment manifest has THIS version. All actual init/action/minibatch/replay,
physical-role streams deliberately use the SOURCE namespace with NEW raw seeds.
No monkeypatches, frozen-source edits or transfer learning. Audit derived roles
with the exact source stream formulas for all three prospective panels.

## Fourteen DEV gates and independent audit

Require ALL14points:
1. AUC gain vs quality-proposal >=.005 (new primary mechanism endpoint);
2. FINAL cost gain vs quality-proposal >=-.01;
3-6. FINAL NI vs IDq,IDp,PPO,PPO normalized >=-.01 each;
7-8. FINAL gain vs local/fullMAPPO strictly>0 each;
9-14. FINAL health vs PPO/norm: availability delta>=-.01,pending/machine<=+.025,
failure/machine<=+.25.
Additionally every3replica AUC gain>0,zero invalid over all learning/evaluation,
isolated p95<=50ms,source/input/weight/budget/reward/physical audits PASS.

This does NOT relax or rerun the old final-cost-superiority claim. AUC evidence
would support learning efficiency, not eventual cost superiority. Standard MPG
variance reduction is PRIOR ART, not a new theorem; a task-specific tractable
broker-induced policy and empirically verified efficiency are only proposed
contributions, still need novelty/scaling/confirmation.
[MPG](https://arxiv.org/html/1806.05134) and
[CAPG](https://proceedings.mlr.press/v80/fujita18a.html) motivate marginalizing
action transformations. No claim they guarantee this finite-batch clipped-PPO
AUC, lower optimization variance after every update, or optimal matching.

After any FAILED gate: preserve/reject this bounded efficiency hypothesis;
NO automatic budget/window/cadence/entropy/LR/encoder sweep, no early checkpoint
promotion or CF/test opening. No negative-verdict rescue from a secondary plot.
No largerN before N4scientific confirmation. Twelve-hourfailstop inherited, no
resume; fresh timestamp on any failed restart, preserve all partial runs.

## Sealed confirmation contract, not implemented/opened by DEV runner

Registry fixes DEV310000000..2/DEVphys310100000..31/analysis310200000;
CFtrain310300000..4/DEVphys310400000..63/TEST310500000..255/analysis310600000;
ENG310700000..2. Raw/derived history audit before training; CF/test numeric seeds
may be hashed for exclusion but no states/trajectories constructed.

ONLY independently verified ALL-PASS DEV permits a separately implemented CF
runner:5NEW random-init replicas, same245760budget/cadence/all8arms,64DEV and256
sealedTEST. Fixed endpoint, all8checkpoints/choices sealed by hashes before test;
no DEVselection is needed for primary. Test evaluation covers all8checkpoints to
estimate AUC plus endpoint. Crossed paired bootstrap200000 draws of train replica
and physical seed, same draws across14endpoints. Sequential family r4 has
alpha_r=.05/[4*5]=.0025, each one-sided bound alpha_r/14; lock corrections before
test. Require lower AUC gain bound>=.005, final NI/superiority and sixhealth
bounds as above. Cost superiority to PPO may be claimed ONLY if SAME final PPO
gain lower bound>=.01. ALL required plus feasibility/latency gates. No scientific
claim from DEV, variance algebra, AUC alone or fixed-weight degradation.

## Implementation / artifacts

Reuse source make_model,fit_job and physical/audit routines unchanged, with new
explicit config/raw seeds/budget/cadence. Tests for fresh/disjoint roles,
initial actor/critics/capacities, fixed endpoint binding vs ancillary selection,
trapezoid spacing/units/pairing/complete grid, gates/health/replicas, source
metadata, physics and checkpoint contracts. Necessary correctness checks only;
no separate smoke; verified commit/push before FULL directlab. tqdm in fits,
multi-seed eval and aggregate24fit progress.

New UTCdir, refuse overwrite. Store protocol/registry/config/physics/source seals,
initial tensor hashes/capacities, all training/checkpoint journals and models,
ancillary choices and distinct primary fixed choices, all6144DEV records,
per-replica/per-seed AUCs, fixed-final physical/coordination journals/frozen
ablations/deadline, independent scalar replay, raw isolated timing,14point
summary and before/after source hashes. No SSH/artifact transfer. Manual rsync
provided when results exist. Goal stays ACTIVE until full original objective,
including fair mechanism confirmation/scaling/health, is actually achieved.
