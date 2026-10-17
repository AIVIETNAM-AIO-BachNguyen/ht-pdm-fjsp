# Typed-Gibbs v1 confirmation — prospective N4/K2, no tuning

Original source51ebbeca434e83077ff4533131af49fd56fcd5cb DEV passed12/12 and all3
replicas after independently verified full21fits/10.32192Mdecisions. Parent run
technician_typed_gibbs_ppo_v1_development_20261017T115852309901Z; independent audit
technician_typed_gibbs_verification_v1_20261017T121421207679Z PASS. Original DEV
candidate cost263.575521/m, main gain6.524%, PPO gain−.153%, fullMAPPO+.006%.
These are DEV points, not PPO superiority or statistical fullMAPPO advantage.
No earlier negative result is superseded. Separate confirmation, before any CF
physical state/learning/test outcome. Goal remains unachieved.

## Frozen question / models / information

Test the SAME learned-future versus unweighted-feasible-future mechanism. Reuse
unchanged source policy/env/learner/model factory/checkpoint/evaluation modules
and all their physics/current information, inputs,18883coreparameters, initialization,
exact labeled density/entropy/MAP, raw statecritic and ordinary scalarjoint PPO.
No teacher/distillation/warmstart/newobjective/component/LR/encoder/order/budget
choice. Core typed_gibbs_ppo vs primarytyped_myopic_ppo and reverse_typed_myopic_ppo;
external ppo/normalized_ppo/mappo/full_mappo. Do not promote a control. See original
FULL protocol and additional prior-art boundary; generic global normalization and
structured RL are established. Two repeated physical technician profiles only;
N>4 remains CLOSED until independently verified confirmation. No scalability claim.

## Fresh sealed panels and identical budget

Use ONLY original family registry's prospectively reserved confirmation panel:
configs/technician_typed_gibbs_ppo_v1_seed_registry.json, no duplicate registry.
Train380300000..4;CFDEV380400000..63;TEST380500000..255;analysis380600000.
CF/test were never constructed in the DEV run or algebra/engineering tests.
Keep family physical/init/action/minibatch namespaces technician_typed_gibbs_ppo_v1;
raw and derived roles are disjoint across all three registered panels. Audit
historical families and within-family splits before collecting CF states.

All35fits=7methods×5freshreplicas;491520decisions/fit,10240trainEP/7680Adam,
160rollouts3072,16vectorenv,4epochs/mb256,Adam3e-4/clip.2/entropy.01/value.5/
globalclip.5/gamma1/H48+12tail/reward−cost/20/valueunit115.2exceptliteralPPO1.
Total17203200decisions/358400trainEP/268800Adam/51200pairedtrainingtapes.
Checkpoints0/122880/245760/368640/491520,64CFDEV each=11200DEVnodes. Fixedfinal
491520 for EVERYmethod/replica, NEVER select from CFDEV or discard a replica.
Ancillary checkpoint minima descriptive only. Full35fit budgets even if DEV
plateaus or looks worse.16CPUworkers/torchBLAS1/tqdm/12hhardstop/siblingtechnical
abort/newUTC/refuseoverwrite/no silent resume or restart. No separate smoke;
necessary tests, commit/push first. Engineering integration uses ONLY existing
engineering physical seed380700001, including repeated test-phase fixture, never
scientific CF/test state; synthetic bootstrap fixtures have no simulator/seeds.

## Phase boundary: freeze first, TEST exactly once

Before constructing ANY TEST state, finish all35fits and fullCFDEV grid; audit
scratch budgets/private initialization/pairedtapes/behavior replay/finite learning.
Freeze ALL35fixedfinal paths/fileSHA/parameterdigests in checkpoint_choices.json
and models_frozen.json. Record choices/seal/protocol/Git hashes and timestamp in
test_gate.json. Only after verifying this complete immutable grid may manifest
mark test_opened=true and workers construct registered TEST states. Test has no
optimizer, model selection or feedback into training. Checkpoint/choice hashes
must remain unchanged at completion. No peek/retry/newseedpanel on a near miss.

Evaluate ALL7×5×256=8960policyepisodes plus256commondeadline=9216TESTepisodes,
552960primitive transitions and442368policy decisions. Parallel16CPU jobs write
private perfit journals, then deterministically merge; include every step, worker
label, full original state/events/cost, canonical utilities/physical logp/exactH
for every core policy decision, feasibility/conflicts/rejections/H48health/H60cost.
Isolated actor+DP+decoder latency before testworkers start, after trainpool closes:
5×3×48=720rawsamples with8warmupstates perreplica; p95<=50ms. Same benchmark/lab
scope; no extrapolated scalability. No critic inselect.

## Confirmatory statistics: exactly inherited twelve endpoints

Primary cost gain1−mean(candidatecost)/mean(controlcost) vsmyopic>=.005.
Other cost gains vsreverse/PPO/norm>=−.01; vslocal/fullMAPPO STRICTLY>0. Sixcost.
Availability candidate-minus-BOTH PPO/norm>=−.01; pending/m<=+.025;
failures/m<=+.25. Sixhealth. ALL12 directional confidence bounds must pass,
plus each5replica's pairedTEST main gain>0,zeroinvalid train/eval,source and
independent execution/seed/density/physical auditsPASS,isolatedp95<=50ms.

Paired crossed train-replica/physical-seed bootstrap200000. Each resample draws
5replica indices with replacement and256physical-seed indices with replacement,
SAME draws for EVERYmethod/metric, and averages the resulting Cartesian product.
Ratio uses resampled mean costs, never average perepisode ratios. Fixed PCG64
np.random.default_rng seeded by registered family learner.role_seed(380600000,
'technician_typed_gibbs_ppo_v1:bootstrap'); this function adds original
maintenance_balanced_algorithm_benchmark_v1 namespace. Audit actual derivedseed.
Generate chunks256, within each draw B×5replicaint64 FIRST then B×256seedint64,
store indicesuint8 (sampling wasint64) and all200000×12effects. No redraw to
change bounds. Quantile methodlinear, oriented one-sided alpha=.05/(7*8)/12
=0.0000744047619047619. This is sequential familyround7 plus Bonferroni12.
Loweralphaquantile for costgains/availability; upper1−alpha for pending/failure.
Strict-zero MAPPO bounds within1e-12 of0 count as numerical ties and FAIL; this
conservative numerical margin is recorded, never used to rescue a result.
Report point/bound/orientation/practicalthreshold/pass for all12, allreplicas and
means, including any failure. Bootstrap is empirical uncertainty with5training
replicas, not an exact distribution-free guarantee; sparse tails are disclosed.

PPO SUPERIORITY only if SAME lowergainbound>=.01 against BOTH literal/normalized
PPO. Not required for near-PPO representation/scalability contribution. No test
win from localMAPPO alone and no superiority from a point estimate. ALLPASS plus
independent audit only permits a separately registered N-scaling study; does not
finish the research goal. AnyFAIL preserves verdict/test/data and requires a
specific prospective follow-up; do not change the original12gates or combine a
favorable testsubset/checkpoint/replica/knockout as rescue.

## Validation / artifacts / independent review

Unit tests for inherited policy/learner remain. Add independent explicit Cartesian
resampling versus count-weighted crossed means, ratio versus mean-ratios,12bounds/
strictMAPPO/1%superiority thresholds, same draw pairing, deterministic bootstrap
seals/indexranges, immutable freeze/no-test-before-gate, complete35grid math,
checkpoint binding/load, engineering fit/freeze/testjournals/merge/replay for all7.
No scientific CF/test environment in tests. Preserve any technical failure.

Fresh manifest/runtime/parent bindings/frozenalgorithm source seal, original family
registry and ownprotocol, seed audit, configs/physics/initial hashes/counts,
every training/CFDEV/checkpoint/progress/episode record, model freeze/testgate,
selected_test_episodes.csv/allpolicy+physicsjournals/latency raws/bootstrap raw
indices/effects/confidence_endpoints.json/summary/audit/seals. Progress tqdm during
fits, every seed-evaluation and bootstrap. Commit/push before FULL here.

Independent verifier implemented and committed separately before scientific
interpretation: manual initializers/NumPyutilities/explicit21matching likelihood/
entropy/MAP, all9216physicalepisodes and budget/seed/freeze gates, all bootstrap
indices regenerated from registeredseed, independently recomputed crossed means/
12orientedbounds and verdict. No source-defined pass alone opens larger N. Frozen
optimizer minibatch gradients were not retained; do not claim fullupdate replay.
Generate report/plots only from source bound to independentPASS; inspect actual
PNGs, record limitations. Manualrsync -avhP --partial only, never initiate transfer.
