# Typed-Gibbs entropy units v1 — one prospective N4 factorial FULL

Register before new learning or physical panels. Parent typed-Gibbs CF source
ed1002c FAILED4/12CI plus one mainreplicasign; independent audit PASS102.59s.
Frozen TRAIN/CFDEVoptimization audit60feac5 rejected its2x-budget entry rule
(lateTV.001773<.01). Preserve both decisions. This is NOT a budget rescue,
entropy-coefficient sweep, oldTESTreuse or a new objective/novelty claim. User
allows necessary bounded tuning; FULL here, no separate smoke, tests/push first.

## Falsifiable question and prior-art boundary

Does the benefit of learned future matching utilities survive equally informed
myopic retraining after matching entropy is aggregated as a joint action, while
reliably meeting original quality/health bounds? Original core uses exact joint
physical PPO ratio but regularizer H(M|s)/N; catalogue PPO uses H(M|s), same
nominal beta.01. AtN4 this is a defined4x regularization-magnitude contrast.
Original /N choice is valid; it is NOT an algebra/likelihood bug or evidence that
entropy caused failure. Current frozen probes are nearly deterministic, which
motivates this ONE controlled test, not certainty about its result.

Main algorithmic contribution still must be weighted future coordination versus
SAME-entropy myopic control, near-PPO quality/health and later measured scaling.
Entropy normalization is tuning, not a new RL objective. If calibrated myopic
catches Gibbs, reject claimed necessity/benefit of learned future coupling under
this setting. No generic global normalization/partition/Gibbs/structured-RL novelty;
see design/prior-art notes. Prior broker entropy Hp-vs-Hq opposition diagnostic
was negative and stays negative: it asked another question at common /K units.

## Six matched factorial arms + four unchanged external baselines

1 gibbs_joint_entropy_ppo PRIMARYcandidate: original Gibbs density, entropyH.
2 myopic_joint_entropy_ppo PRIMARYcontrol: original myopic density, entropyH.
3 reverse_joint_entropy_ppo: reverse-myopic density, entropyH.
4 gibbs_machine_entropy_ppo: original Gibbs density, entropyH/N.
5 myopic_machine_entropy_ppo: original myopic density, entropyH/N.
6 reverse_machine_entropy_ppo: original reverse-myopic density, entropyH/N.

All6same raw89->64->64->2actor/state71->64->64->1critic,18883parameters/private
identicalfresh tensors, identical raw information/support/DP/likelihood/globalMAP/
physics/sampling roles/learning budget. Zero final utilityhead, uniform feasible
labeled physical matching density at initialization (21fullsupport). New labels
must map to separate BASEdensity identifier; no accidental switch to myopic
from comparing an unfamiliar self.algorithm name. Fixed weights: H-vs-H/N changes
ONLY evaluate's entropy aggregation, never sampler/action/logp/MAP/V or its
likelihood gradients. Actual training paths may diverge after updates.

Original fresh ppo18966/normalized_ppo18966/localmappo15364/fullmappo18052 retain
ORIGINAL entropy conventions/architecture/hyperparams; no baseline weakened by
an entropy change. Same private initial raw statecritic tensors acrossALL10;
PPO/normactor tensors paired. LiteralPPOvalueunit1,others115.2. Information/capacity
comparisons disclosed; causal learnedfuturecontrast is exactly matched withincore.
No teacher/distill/PPOweights/warmstart/labels/additionalcritic/attention/encoder/
LR/optimizer/decoder changes. Contribution unproven until controlled evidence.

## Frozen learning and new panels

STANDARD scalarjoint clipped PPO/common normalizedMCstateadvantage,gamma1,
H48+12commondeadline tail,reward-cost/20,Adam3e-4,clip.2,beta.01,value.5,global
clip.5,16env,rollout3072,4epochs/minibatch256. ONLY core entropyaggregationfactor
1 versus1/N. No epsilon or alternate exploration floor. No longer training.

ALL30fits10arms*3newreplicas receive491520decisions,160rollouts/7680Adam/10240EP.
Total14745600decisions/307200trainEP/230400Adam. Checkpoints0/122880/245760/
368640/491520,32DEV/node=4800DEVepisodes. PRIMARY fixedfinal491520, ancillary
minima descriptive only; no discard of replica/checkpoint/control promotion.
16CPUworkers/torchBLAS1/tqdm/12hhardstop/technical siblingabort/freshUTC/refuse
silentoverwrite/resume. No plateau/win early stopping or secondunitcoefficient.

Registry ownversion, DEVtrain400000000..2/physical400100000..31/analysis400200000;
CFtrain400300000..4/DEV400400000..63/TEST400500000..255/analysis400600000;
ENGtrain400700000/physical400700001/analysis400700002. Pre-registration rawrange
check365candidates versus69unique expanded historicalregistries found0overlap.
Actual own/historical hashed init/action/minibatch/physical/bootstrap roles must
be audited before FULL. Use NEWversionnamespace for original-equivalent physical
reset and privateinitialization/action/minibatch roles. Core rolegraph;PPO/norm
roleppo; owncritic_state/bootstrap namespaces. Pair physics across ALL10.
No new CF/TESTstate constructed in DEV. Old typedv1CFTESTnever reused.

## Fixed common-state entropy readout — diagnostic only

Before learning, generate ONEdeterministic deadline episode on the firstregistered
DEVphysicalseed (400100000); save48policy observations, sourcephysicspopulation/
clock andSHA. This is a shared observation probe ONLY, never actor labels,
training targets, a teacher or a selection score; no model consumes its actions/
rewards during learning. Physicalticks60 recorded separately from train budgets.
Every core checkpoint evaluated on these SAME48states. On supportsize>1 report
physical H/log(feasiblematchingcount), maxphysicalmatching probability, fraction
maxprob>=.99 and meanconditional available counts; no latent entropy proxy.
Computations use typedcountDP, no production catalogue. Source sampler/MAP unchanged.

Secondary earlywindow FIXED122880. Mean entropy gap Gjoint−Gmachine>=.02 AND
mean near-deterministic-fraction reduction>=.10 AND at least2/3replica entropy
gaps>0 descriptively supports delayed saturation on this panel. IfFAIL, do not
claim entropy caused original instability even if final quality improves. Readout
is ONEphysicalseed, not a scientific coverage/cost estimate and not a selection
criterion. Report all6arms/5CP,pairedunit costs/failures and factorial interaction;
no test-time subset fishing or new performance claim from a diagnostic.

## Fifteen original-style quality/health endpoints + stability

Candidate Gjoint costgain1−meanC/meanB vs Mjoint>=.005. Vsreversejoint/ALL3original
H/Ncores/PPO/norm>=−.01 each (6NI). Vslocal/fullMAPPO STRICTLY>0 (2).9cost.
6originalhealth bounds versus BOTHliteral/normPPO: availability delta>=−.01,
pending/machine<=+.025,failures/machine<=+.25. ALL15point gates for DEV.
Every3replica maincostgain>0;zero invalid train/eval;allsource/scratch/budget/
replay/entropyunits/physical auditsPASS;isolated432actor+DP+decoderp95<=50ms
postworkershutdown. No positive costpoint or entropy-readout replaces thesegates.

Only independently verified ALLPASS DEV allows separately implemented/registered
confirmation5freshreplicas/64DEV/256sealedTEST,all10/same491520/fixedfinal,freeze
ALL50models before anyTESTstate.200000paired crossed train/physical bootstrap,
familyround8alpha=.05/(8*9),one-sided/15,linearquantile,samepractical/healthbounds;
strictzero MAPPO tieswithin1e-12FAIL. No reuse of oldtypedTEST/oldfailed alpha.
PPOsuperiorityONLYoverallpass ANDlowergain>=.01vsBOTH P P O/norm. EvenCFPASS does
not finishgoal: largerN/two-profile scope/generalization/contribution stillunproven.
AnyFAILpreserve THISverdict, no beta4xagain/schedule/LR/encoder/budget sweep.

## Tests, artifacts and independent checks

Verify same-weight entropyunit pairs identical fullphysical density/MAP/V/action
RNG stream; entropy and gradients differ EXACTfactorN, not latent entropy;
Gibbs/myopic/reverse basemapping correct; sample/replay/ratio/finite gradients;
uniform support all256tiny masks,typecontracts,no critic/catalogue inselect;
all10fullEPoptimizer/roundtrip/privateinit/coldparity/equalbudget/15gateboundaries;
common probe binding/no labelsconsumed/secondaryreadout bounds/freshraw+actualroles.
Tests before commit/push, then FULL here. No separatesmoke/SSH/transfers.

Fresh artifact manifest/runtime/source/protocol/registry/config/physics/seedseal,
allinitialhashes/counts/checkpoints/TRAIN/DEVgrids/entropyprobe/fixedfinalchoices,
selectedepisodes/step-levelcanonicalutilities/logp/exactH/physicaljournal,raw432
latencysamples/15points/factorialdescriptors/health/conflicts/feasibility/sourceaudit.
Independent manualMLP/explicit21matching entropy/density/MAP/entropyfactorcheck,
primitivefullreplay,initializers/budgets/probe/roles/originalverdict before claiming
quality or opening confirmation. Historical minibatch gradients not saved; do not
claim fulloptimizer replay. Preserve every negative artifact and alltestseals.
