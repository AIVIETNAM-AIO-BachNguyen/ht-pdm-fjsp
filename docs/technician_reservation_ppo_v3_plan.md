# Teacher-guided resource reservation PPO v3 — prospective protocol

This protocol is locked before any v3 development outcome. Run on this lab
under the user's explicit override of Mac/manual-lab-handoff instructions.
V1 development completed with N4 reservation cost1210.15 vs independent1121.81
and scale N16 good; no scientific tests. V2 N4 selected reservation1110.76
versus approximately1049.56 PPO (partial development; full fixed run continues).
No confirmation alpha has been spent. Preserve both negative development rounds.

Hypothesis: graph attention and transferring the available centralized PPO's
assignment decisions into a communicating technician policy close the small
quality gap; reservation exchange maintains quality as resources grow. This is
PPO-derived deployment/coordination research, not a claim of inventing general
PPO, autoregressive MARL or distillation. Primary references:
https://arxiv.org/abs/1707.06347 , https://arxiv.org/abs/2205.14953 ,
https://arxiv.org/abs/1511.06295 .

Actor raw inputs: complete current graph (machines6, technicians3, edge4,
global9) plus public normalized row/column IDs2. Separate value encoder;
actor edge embeddings attend to all current machine embeddings with scaled
query/key/value attention, giving each edge a view of competing machines.
Shared weights/hidden64; polynomial O(N^2*K*64) attention and K reservation
stages, no matching catalogue outside N4. No future event input. Communicating
technician agents broadcast selected edge embeddings and reserve machine IDs
in fixed worker order. This requires a shared reservation service; inference
latency excludes network transport, which is not validated here. Capacity is
independent of N/K. Independent ablation has IDENTICAL information, attention,
IDs, parameters, initial weights, teacher data/labels, pretraining updates and
PPO budgets. It receives no reservation or prefix exchange, and fixed initial
summaries. Duplicate proposals use fixed edge quality then workerID arbitration,
losers not reassigned. This tests the entire coordination service, including
collision prevention, not just attention. Less-informed MAPPO alone cannot
establish coordination benefit.

Foundation teachers: ALL five PPO checkpoints chosen on development in the
frozen balanced benchmark artifact maintenance_balanced_algorithm_benchmark_v1
full_cpu_20261012T115140Z (source commit2e19043). All five original MAPPO
checkpoints are references. No source test selection or teacher seed selection.
Bundle i maps deterministically to source46000000+i. Development uses first3;
confirmation all5. Hash every checkpoint and source choices/config/manifest/audit,
verify original metadata/config and parameter digests; copy immutable inputs into
each new run. Current baseline PPO/MAPPO references are evaluated on NEW episodes,
not refit or selected using new tests. They retain their old development-selected
checkpoints and491520 original training decisions, even if selected earlier.
Teacher/source stochastic training uncertainty is represented by bundled replica
resampling, but this is conditional on an existing teacher collection, not five
new independent teacher fits.

For EACH fresh student seed, generate exactly122880 N4 training state exposures
with its mapped PPO teacher:16 environments, whole48 decisions plus12 common
deadline tail; stochastic behavior temperature1.5, deterministic teacherargmax
matching labels. Separate teacher-bank physical workerIDs1000..1015 so no overlap
with PPO fine-tuning workers0..15. No development/test episode enters this bank.
Record raw current observations, tokens WAIT0/machine+1 for K2, return-to-go from
behavior trajectory, physical tape hashes, actual episode counts, dataSHA and
teacherSHA. Both graph arms use the EXACT same bank. Pretrain once per seed/arm
at N4,32epochs, minibatch512, Adam3e-4, hard joint-label NLL/K plus0.1 smoothL1
value loss; clip.5, all fixed epochs, no early stopping/teacher-accuracy selection.
The separate encoders prevent critic gradients changing actor features. Reuse
this same warm checkpoint across all target scales, then fine-tune PPO491520
fresh decisions per arm/scale: vector16, rollout3072,4epochs,minibatch256, gamma1,
lr3e-4, clip.2, entropy.01, value.5, gradclip.5. Checkpoints warm0 and RL122880,
245760,368640,491520 eligible for development selection, earliest tie. Entire
fine-tuning budget spent even if warm checkpoint is selected. Count exposures
and optimizer steps by phase, actual physical decisions, presentations, wall time
and amortization across scales. Graph student receives122880 extra teacher-bank
exposures plus491520 RL decisions, and an already-trained foundation teacher;
DO NOT claim better sample efficiency or fair optimizer-only superiority over
PPO/MAPPO. Extra teacher training is disclosed, never silently charged as zero.
Matched mechanism ablation has exactly the same added training information and
budgets. Claimable main contribution is deployment scalability and coordination.

Physics: exactly frozen balanced_a2_b4_u12 at N4/K2,H48+tail12,reward=-cost/20.
Scale N8/K4,N16/K8,N32/K16 repeats four machine/two technician skill patterns
across all edges, with cross-block assignments. Deadline comparison and common
tail use polynomial min-cost flow at large sizes, original exact deadline atN4.
No catalogue in any large actor,critic,environment,tail or training; teacher bank
exists at N4 only. Report cost per machine,availability,failures/backlog per
machine,conflicts,rejected proposals,parameter counts,graph sizes and theoretical
catalogue counts. Record availability/overdue/utilization in every episode.

Seed registry: development55100000..2,development55200000..31,analysis55300000;
confirmation55400000..4,dev55500000..63,test55600000..255,analysis55700000;
smoke55800000,dev55900000..1,test56000000..3,analysis56100000. Raw panels are
historically disjoint. SHA role namespaces separate warm initialization,
teacher actions,minibatches,fine-tuning per scale/actions/minibatches and
physical training streams. Teacher bank and fine-tuning disjoint workerIDs.
Intentional pairing: same initial weights and bank for both graph arms, same
physical training tape across graph arms per scale, same test tape across all
methods per scale. Development never instantiates confirmation test states.

Development: three student bundles N4/N16, both graph arms; original PPO/MAPPO
atN4 only,deadline bothN.32 development episodes per checkpoint. Screen requires
N4 candidate<=1.01PPO, <MAPPO, <=1.01independent; N16 candidate<=1.02deadline,
<=independent, availability>=deadline-.01,pending/machine<=deadline+.025.
No scientific claims from development. Confirmation only after frozen design
passes development: five fresh bundles, all four scales,64 new development and
256 sealed physical test episodes. Choose each graph replica checkpoint on mean
development extended cost, earliest tie; keep all replicas. Source references
not reselected. Seal every choice/hash BEFORE opening any test. No peeking,
retesting same panel, metric-triggered stopping or silent budget extension.
Fixed global hard stop24h yields INCOMPLETE/FAILED, never claimed success.

Final endpoints: crossed bootstrap training BUNDLE axis and paired episode axis,
40000 draws. Within each draw share bundle weights across scales, but independently
resample episode axes by scale because their physics namespaces are independent;
within scale all methods share episode weights. Bonferroni family24, round1
alpha.025, confidence1-.025/24. Future scientific round r uses fresh panel and
alpha=.05/(r*(r+1)); no alpha spent by development or engineering smoke. CIs are
approximate with five bundles; report per-replica and leave-one-out sensitivity.
Six endpointsN4 (cost gains vsPPO,MAPPO,independent; availability,pending,failures
vsPPO); each large scale five (gains vsindependent,deadline; health vsdeadline),
plus three large/small per-machine cost ratios. Required small: gain vsPPO
lower>=-.01,vsMAPPO lower>0,vsindependent lower>=-.01. Positive small mechanism
reported but not required (as prospectively specified in v2). Required EACH
large: gain vsindependent lower>0,gain vsdeadline lower>=-.02, normalized cost
large/small upper<=1.10. Health at all sizes: availability difference lower>=-.01,
pending per machine upper<=.025, failures per machine upper<=.25 (candidate minus
PPO atN4,deadline large). Single-state neural p95<=50ms, zero invalid assignments.
All required gates must pass. A cost-superiority claim overPPO additionally needs
gain lower>=.01 practical threshold; label it teacher-assisted with extra compute.
Report failures and health losses within margins too. Ablation is simulator
intervention evidence, not field/network deployment validation.

Artifacts: new UTC directory per run; manifest,GIt/runtime,sealed protocol,
registry/source snapshots,source teacher hashes/copies,banks/pretraining journals,
warm and RL checkpoints,per-phase sample/optimizer audits,all training episode
journals and physical tape hashes,development/reference episodes,chosencheckpoint
seal,full final decisions/coordination episodes/latency,summaryCI,gates,audit.
Audit exact grids/counts/bank label legality/initial equality/shared-bank hashes,
checkpoint hashes,budgets and source immutability; independently replay ALL test
physical transitions and submitted proposals/arbitration against scalar physics.
Preserve failures. Four independent CPU processes,torchthreads1,tqdm for bank,
pretraining,PPO and evaluations; parent deterministic journal merging and no
scientific test before all jobs finish. Smoke:bank192,pretrain2epochs/minibatch48,
RL384/checkpoints192,384,vector2,N4/N8,2dev/4engineeringtest episodes, samephase
pipeline. Full existing suite plus new phase/labels/gradient/serialization/process/
seed/statistical/physics tests and local lab smoke BEFORE commit/push and research.
Never edit frozen source while a suite with spawned children or a run is active.
