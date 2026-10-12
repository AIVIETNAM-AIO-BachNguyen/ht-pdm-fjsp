# Teacher-guided resource reservation PPO v4 — prospective protocol

This protocol is locked before any v4 development outcome. Run on this lab
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

Seed registry: development57100000..2,development57200000..31,analysis57300000;
confirmation57400000..4,dev57500000..63,test57600000..255,analysis57700000;
smoke57800000,dev57900000..1,test58000000..3,analysis58100000. Raw panels are
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
Preserve failures. Four independent CPU processes for development, eight for
confirmation and engineering smoke (prospective execution amendment below),
torchthreads1,tqdm for bank,
pretraining,PPO and evaluations; parent deterministic journal merging and no
scientific test before all jobs finish. Smoke:bank192,pretrain2epochs/minibatch48,
RL384/checkpoints192,384,vector2,N4/N8,2dev/4engineeringtest episodes, samephase
pipeline. Full existing suite plus new phase/labels/gradient/serialization/process/
seed/statistical/physics tests and local lab smoke BEFORE commit/push and research.
Never edit frozen source while a suite with spawned children or a run is active.


## Prospective v4 amendments — partial DEVELOPMENT feedback only

V3 continues its frozen full development budget. N4 completed reservation mean
selected cost1061.135 vs sourcePPO1057.552 (0.339% worse), meeting the small
point-estimate screen so far. At N16 warm availability was .66728,.61076,.61951;
first122880-step checkpoint for second replica .62443. Those two poor health
replicas have costs296.27/295.79 per machine. No scientific confirmation tests
have opened. V4 addresses that scale-transfer/health deficit prospectively;
these figures are developmental feedback, not confirmed performance claims.

Both graph arms retain hidden64 and identical capacities and budgets. Add the
pooled embedding of a machine's incident technician edges to its own edge
representation, so bids directly reflect that machine's full skill profile.
Replace single-head competition attention with four heads16 channels, same
query/key/value parameter counts. Teacher distillation uses random independent
machine and technician renamings per state presentation. Permute machines,
technicians,edge tensor,masks and teacher matching tokens together; reconstruct
flat state and catalogue mask consistently. Physics and labels are the same
sample under a bijection, not new oracle queries or additional physical samples.
A dedicated augmentation RNG gives BOTH graph arms EXACTLY the same permutations
and presentations, audited by an identical transformation digest. Public row/column
IDs describe the permuted current layout. Original bank/labels remain unchanged
and independently replayable against the frozen teacher. Do not claim exact
permutation invariance; identifiers remain available for symmetry breaking.

At technician stage j, both arms' planning resource summaries exclude technician
indices<j: their one proposal slot in this tick has already been consumed,
including WAIT. This exclusion depends only on public fixed order. Reservation
also excludes machines chosen by earlier agents and broadcasts their selected
edge prefix; independent receives neither chosen machines nor prefix. The full
initial graph, attention, public order, parameters, teacher transformations,
physical training tapes and budgets match between arms. This contrast isolates
the reservation/message service without giving it a misleading remaining-worker
summary. Live assignment masks remain current physical eligibility, additionally
reserved-machine exclusion in reservation only. All physical assignments execute
simultaneously once the K proposals finish. No prior worker gets a second turn.

Value output is multiplied by N/4 in BOTH graph arms: reward is total plant cost
while warm-start critic is fitted on N4. This uses public graph size, no extra
physical information or samples; improves the initial value calibration across
scales. N4 is unchanged; value weights still transfer without changing shapes.
This is a bundled design refinement, not an isolated causal attribution for
attention, augmentation, worker summaries or value scaling. The primary causal
comparison remains the matched coordination ablation.

Numerical replay audit: compare actor log likelihood and value separately.
New v4 only uses actor absolute tolerance2e-5 (implied ratio drift <~2.1e-5,
far below PPO clip .2); value absolute2e-5 plus relative1e-6, acknowledging that
float32 error scales with total-cost value magnitude. Record separate raw errors
and maximum pointwise normalized error; require normalized<=1 in every job.
Legacy experiments keep their exact original2e-6 settings by default. This is
an engineering precision guard, not changed policy/optimizer/sample budgets.
Tests must accept harmless float32 batched value differences and reject meaningful
actor drift and out-of-tolerance value drift. Checkpoint/parameter/physics hashes
still prevent silently changing the behavior model. A sibling job exception is
written to FAILED manifest immediately; pending jobs are cancelled and active
jobs detect a shared abort marker, preserving all partial artifacts. Technical
aborts are distinct from metric stopping. Do not wait invisibly for other jobs
before exposing the failed run status.

Fresh v4 raw panels57100000..58100000 as listed above, historically disjoint from
v3. Add a hashed teacher_augmentation RNG role. All small/large quality,mechanism,
health,latency and statistical gates remain unchanged. No v4 development outcome
exists at protocol lock. V4 is eligible for first scientific round ONLY if no
v3 scientific test was opened; otherwise increment round/alpha and lock a new
confirmation protocol/panel before any v4 scientific run. Original source PPO/
MAPPO collection remains the same five frozen, old-development-selected models.
Full tests, lab smoke, commit/push precede any v4 development; never edit its
frozen source during tests with spawned children or research.

## Prospective execution amendment before scientific confirmation

Locked on 2026-10-12 UTC while the original four-worker development run finishes.
No scientific confirmation test has opened. That original run and its source
stay frozen in their separate worktree. The candidate actor, critic, physics,
training data, initialization/RNG roles, per-model decisions, optimizer updates,
checkpoint-selection rule, seeds, statistical endpoints, margins, alpha and
24-hour stopping rule are unchanged. Only confirmation and engineering smoke
use eight independent single-thread CPU workers instead of four. Development
keeps four. The lab has sixteen logical CPUs and 64 GiB RAM; this changes
concurrent execution and wall time, not the work or information given to any arm.
Each job owns its RNG and outputs; parent merging follows deterministic job
order. Final latency evaluation occurs after the training pool has shut down.
Do not attribute a training speedup to the algorithm based on this topology.

Engineering hypothesis: changing concurrency preserves every learned tensor
and chosen checkpoint under the same registered smoke panel. Run the complete
test suite, then an eight-worker smoke through the unmodified full pipeline.
Compare ALL checkpoint parameter digests, both warm-start digests and teacher
bank digest against the completed four-worker smoke
technician_reservation_ppo_v4_smoke_20261012T184359Z. Require exact equality,
COMPLETE/audit PASS, unchanged per-model budgets and exact episode grids.
Stop after that fixed smoke budget; no metric-triggered extension. Store the
new smoke in a fresh UTC artifact directory, and save an equivalence report
binding both manifests/checkpoint inventories and this source Git revision.
Commit/push verified execution code before starting scientific confirmation.

The additional latency preflight used engineering raw seeds58300000..58300003
(48 episodes, 2304 neural decisions, no quality endpoint), disjoint from every
registered scientific panel. Preserve its prospective protocol and probe script
in technician_reservation_ppo_v4_engineering_preflight_20261012T200351Z; register
those seeds as engineering history before confirmation. Its results cannot
select a quality checkpoint or substitute for the scientific test grid.
