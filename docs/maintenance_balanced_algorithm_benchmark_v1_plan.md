# Fixed balanced 2–2 algorithm benchmark: lab protocol

## Material Passport
Synthetic fixed maintenance simulator, no participant data. Confirmed setting
balanced_a2_b4_u12 is frozen; this protocol measures algorithm performance on
new seeds, not another setting search. Human explicitly chose five planning/
heuristic baselines plus PPO, MAPPO, IQL, QMIX. The current human instruction
restores canonical lab workflow: tests and small smoke on Mac, commit/push,
then provide full Ubuntu command and rsync; assistant does not run full, SSH,
or transfer results. Source primary algorithm references: PPO
https://arxiv.org/abs/1707.06347 ; MAPPO https://arxiv.org/abs/2103.01955 ;
QMIX https://arxiv.org/abs/1803.11485 . These are task-specific feedforward
implementations, not claims to reproduce all original paper hyperparameters.

## Question, hypotheses and frozen physics
Compare algorithms' expected H+L total cost over training replicas and common
heldout physical episodes; determine whether cost gains respect uptime/backlog.
No assumed winner or setting modification after outcomes. Matrices:
service [[1,2],[1,4],[2,1],[4,1]], restoration
[[.5,1],[.5,1],[1,.5],[1,.5]], all edges eligible. N4/K2,H48+common deadline
tail12 without reset; failureage4,p.45,maxage8; PM1,CM2,down12/machine-tick,
failure15,waitlimit4/overdue12. Urgent idle machines only (failed orage>=3),
free technicians only; legal partial matchings and WAIT. Total tail cost is
added to last learned reward, gamma1; episodic team reward=−H+Lcost/20.
No realized future failure shocks enter observations, masks, critics or planners.
Planners sample separate hypothetical shock banks from the known simulator.

## Controllers and information/action contracts
Five baselines: deadline_matching, isolated_h6, isolated_endpoint,
assignment_rollout, joint_rollout. Rollout128CRNscenarios/root using known
simulator physics, baseline isolated_h6 continuation through48, common deadline
tail; baseline fixed from setting selection, not reselected using benchmark
outcomes. Assignment rollout fixes current subset/tick, future decisions may
adapt. Planners have a known transition model; disclose this advantage.

PPO: centralized actor over21legal-matching catalogue entries and global critic.
MAPPO: four parameter-sharing machine actors, each WAIT/technicianA/B proposals,
masked using own urgency/occupancy and public resource status; centralized team
critic. Per-agent PPO ratios with common Monte Carlo returns, mean over agents.
IQL: four independent local Q networks with shared team reward, DoubleQ replay.
QMIX: shared local Q network plus state-conditioned nonnegative hypernetwork
mixing, DoubleQ replay. These are feedforward time-aware baselines; no RNN claim.

Machine actor/Q local features33: ownmachine6, public technicians6 (occupied
worker's machine failure status zeroed), ownedges8, public clock/config9,
one-hot machine identity4. Critic/global PPO/mixer state71 is complete flattened
current observation. Local actors cannot read peer machine ages/failures/waits;
resource broadcast includes availability/remaining time and own assignment.
Four simultaneous machine proposals use one FIXED public urgency arbitration:
for each technician, accept highest urgency100*failed+20*max(wait−3,0)+
10*hazard+age/8, lower machineID ties; losing proposals are not reassigned.
All21matching actions remain representable by collision-free proposals.
Report conflict/rejection counts; arbiter supplies coordination and is not a
learned contribution. PPO has full-state execution while decentralized agents
have local execution; architectures/action factorization differ. Equal sample
budgets do not make an information-matched architecture ablation.

Hazard in arbitration is p when age+1 reaches the failure threshold and the
machine has not already failed, zero otherwise. All actor slots (including
forced WAIT) enter the per-agent MAPPO mean; replay stores original proposals,
including rejected contenders, and terminal transitions have zero bootstrap.

## Locked seeds, budgets, checkpoint selection
Full train rawseeds46000000..46000004 (5 for EACH4algorithms), development
46100000..46100063, sealed test46200000..46200255, analysis46300000,
forecast46400000. Smoke train46500000,dev46600000..1,test46700000..3,
analysis46800000,forecast46900000. Raw roles/splits/profile panels audited
against expanded historical registries. DerivedSHA256 namespace is thisprotocol
for physical evaluation and train worker/episode streams; forecast uses the
existing engine's namespace with fresh forecast rawseed. Intentionally paired
train physical tapes across algorithms for same rawseed/worker/episode, and
common evaluation physics across all controllers. Initialization/actions/replay/
minibatch roles split. No training-stream collision with evaluation roles.

Full EACHmodel491520 learned decisions,16vectorenvs, episodes48policy+12tail;
20models total9,830,400learneddecisions (12,288,000physical ticks),204800training
episodes. Four checkpoint steps122880/245760/368640/491520. Smoke EACHmodel384
steps,2envs,2checkpoints192/384,4models; sameH48/tail12andphysics.
Shared hidden64/mixerhidden32; Adam3e−4; PPO/MAPPO rollout3072,4epochs,
minibatch256,clip.2,entropy.01,valueweight.5,gradclip.5; smoke rollout192,
2epochs,minibatch48. Whole episodes, Monte Carlo undiscounted returns; no
teacher labels, demonstrations or privileged training targets.
Q replay65536transitions,warmup3072,batch128,oneupdate/16newtransitions,
target hardupdate3072transitions,epsilon1→.05 over firsthalf samplebudget;
smoke replay512,warmup96,batch32,frequency2,target96. No epsilon in evaluation.
Report optimizer counts rather than pretending they match on/off-policy.

Select one trained checkpoint PERalgorithm/trainseed by lowest development
meanH+Lcost, earlieststep ties. No choosing best training seed. Untrainedpolicy
on development is diagnostic only; no test checkpoint sweeps. Development
learning curves from all four budgets; test only selected checkpoint permodel.
Train all20models and seal choices before opening test. Full selectedcheckpoint
+5baselines:25controllerreplicas×256=6400testepisodes/384000physical ticks.
Expected development:20*(4trained+1initial)*64=6400episodes; smoke4*3*2=24.
No earlystopping or testseed extension. CPU1torchthread;full24h hardlimit,
smoke1200s; no automatic resume/retry. Interrupted run preserved; newtimestamp
only unless a later resume contract is implemented/tested. No GPU support claim.

## Primary comparisons and statistical interpretation
Baseline h6 stays frozen for health comparisons. Primary20cost comparisons:
each4learnedalgorithms vs EACH5baseline, gain1−meanlearnercost/meanbaselinecost.
Health8comparisons: each4algorithm availabilitydifference and terminalpending
difference vs h6. Locked28CI family, confidence1−.05/28,40000 crossedbootstrap
full (200smoke): independently resample trainingreplica and rawtestepisode
axes, reuse pairedtrainingindices across algorithms and commontestindices
across all methods. Every trainingseed is retained; no treating1280learned
rows as1280independenttrainingreplicas. CIs are approximate, especially5replicas.
A lowercostCI>0 is evidence of improvement; lower>=.01 is practical1%gain.
Health guard: availability lower>=−.01 andpendingupper<=.1; report health loss
if present, even when withinmargin. No mandatorylearnedwinner or claim of
MARL superiority if resultsfail. Pairwiselearnedranking and costcomponents,
failurecounts,tailpending,actionlatency,trainwalltime,parametercounts are
exploratory/descriptive, noadditional familycoverage claim. H-onlycost separately
reported; H+L governscheckpointselectionand primarycomparison to preventtail
horizon deferral. Synthetic fixed-setting evidence is not scale/fieldvalidation.

## Tests, audits and artifacts
Mac unit/integration tests: identical fixedphysics/pairedseed reset, legal masks
and arbitrarysimultaneousproposalresolution, complete tailreward reconciliation,
no peer/futures leakage, checkpointsave/load predictions/parameters, PPO/MAPPO
likelihood ratios, exactoptimizersteps, QMIX monotonic gradients/maskedDoubleQ,
terminaltargets, crossedbootstrapseedunit, smokefullgrid/sourcehashes/nooverwrite.
Mac smoke is engineering only; sealedfulltestpanel remainsunopened. tqdmtraining
and every multiseed evaluation; streamtrainingprogress/episode/partialevaluation.
Commit/push only after tests+smoke pass; canonical labhandoff then stopexecution.

Each timestamped artifact includes manifest git/runtime/status/opening flags,
fixed benchmark/resolvedconfig, prospectiveplan/seedregistry/source/history
SHA256snapshots; training_progress.csv/training_episodes.csv/training_audit.json;
development_episodes/checkpoint_choices; peralgorithm/trainseed checkpoints
with contracts/metadata hashes; modelparametercounts; episodes.partial.csv and
finalepisodes.csv; testdecisions.csv withstate/action/events/nextstate/cost;
coordination.csv arbitration/collisions; summary.json/audit.json.
Checkpointsloaded and predictions compared beforetest. Alltestphysicalticks
replayed; alllegalroots/masks/arbitration rechecked; rawseed role/costaccounting
and noinvalidassignment audits precedeperformance. Testcounts and everytrained
replica/developmentcheckpoint countexact. Boundeddeterministiccheckpoint
reevaluation firsttwo testseeds validatesloggedlearneddecisions; rebuildfirsttwo
rolloutseeds confirms selectionbanks. Source/sealedchoices stablethroughrun.
Largeartifacts ignoredbyGit; reportfailed/incomplete runs honestly.

## Statistical fallacy coverage
11/11 checks: pooledpairedresult not allsubgroupclaim (Simpson);episode/seed
units not individualmachineeffect(ecological);fresh testafterselection(Berkson);
no outcomeconditioning/collider;actual failuresnotp=.45asprevalence(base rate);
pairedbaselines notcross-roundregressiontomean;allreplicas/statesretained
(survivorship);28CIfamily andallcomparisonsreported(lookelsewhere);versioned
fixedprotocol/checkpointselection(nohiddenforks);simulatorinterventionnot
realplantcausality;actionprecedesstateupdate(noreversecausalclaim).
