# Technician reservation PPO v2 — prospective lab protocol

User override 2026-10-12: run directly on this Ubuntu lab, including smoke,
full experiments and analysis. Never transfer artifacts or SSH automatically.

Question: can shared resource agents coordinate technician assignments near a
centralized PPO while avoiding enumeration of joint matchings? Candidate R-PPO
uses per-edge bids from machine state, technician state, edge physics and clock,
pooled current graph messages, and sequential technician reservation. Each
technician selects one machine or WAIT; reservations remove that machine from
later choices. Physical time advances once. This is a communicating agent
implementation with a shared reservation service, not communication-free
execution or a claim of a new general PPO principle. Related sequential MARL:
https://arxiv.org/abs/2205.14953 ; PPO https://arxiv.org/abs/1707.06347 .

Arms: reservation_ppo; independent_ppo, identical parameters/initial weights,
initial information, optimizer and sample budget but no reservation exchange
(independent technician proposals; duplicate machines resolved by fixed edge
quality, then worker ID; losers not reassigned); original local-observation
MAPPO; original catalogue PPO on N4/K2 only. Independent ablation uses a joint
log-likelihood ratio too, isolating the reservation mechanism, including its
collision prevention. PPO and reservation access the same raw current state;
structured pooling differs from flattening. MAPPO has less execution information:
beating it alone never supports a coordination or information-matched claim.
No future realized shock is an input. No oracle labels, warm start or planning
model in any learner. Single shared feedforward architecture, hidden64, all
parameter counts/optimizer steps reported. Fixed worker order is disclosed.

Frozen physics: balanced_a2_b4_u12, same H48+tail12, reward=-cost/20 and gamma1.
N4/K2 is exactly the current setting. Scale N8/K4, N16/K8, N32/K16 replicates
machine four-role and technician two-role service/restoration patterns across
all edges (cross-block assignments allowed). Each extra worker of the same
role has the same physical skill, not a disconnected copy of the small task.
Use polynomial maximum-weight deadline assignment in common tail; N4 uses the
original exact deadline tail for exact historical compatibility. No catalogue,
matching membership matrix or exhaustive mask in large environments, training,
critic, actor or tail. Report per-machine cost/backlog/failures, availability,
conflicts/rejections, mean/p95/p99 single-state decision latency, wall time,
parameter counts, graph size and theoretical catalogue count. Latency includes
neural decoding; environment arbitration and observation costs also measured.

Stage development (not final scientific evidence): 3 train seeds, 491520 steps
per arm/scale, vector16, rollout3072, 4epochs, minibatch256, Adam3e-4, clip.2,
entropy.01/value.5/gradclip.5. Checkpoints122880,245760,368640,491520; 32 dev episodes.
Start N4 plus N16; N8/N32 may be added as diagnostic development before final
confirmation. Original PPO and MAPPO retrained, never selected using old tests.
Choose each replica's checkpoint on mean development H+Lcost, earliest tie.
Development target: candidate mean cost <=1.01*PPO and <MAPPO and <independent
at N4; at scale <=1.02*deadline and <=independent, availability >=deadline-.01,
normalized pending <=deadline+.025. If fail, preserve outcome and amend design
prospectively; never tune on confirmation tests. No mandatory empirical winner.
Within a round fixed budget, no metric-triggered stopping. Hard deadline24h
marks incomplete, not performance failure. User's continuing goal does not
allow claiming success without all gates or silently extending a sealed test.

Final confirmation only after design is frozen on development: 5 fresh paired
training seeds per arm/scale, 491520 steps, same optimizer settings, checkpoints
122880,245760,368640,491520. N4 all4arms; N8,N16,N32 reservation,independent,MAPPO.
64 fresh development and256 sealed test episodes; paired training tapes and
paired test physics per scale. Retain every training replica, never best seed.
Freeze all checkpoint choices before opening any test. A failed final gate
requires fresh test panel and explicit exploratory follow-up protocol; no
optional peeking, repeated testing against the same panel or silent seed growth.

Final statistical gate: crossed bootstrap independently resamples training
replica and paired episode axes, 40000 draws; Bonferroni family24 endpoints:
N4 3cost contrasts plus reservation-vs-PPO availability, pending, failures (6);
three large scales each reservation-vs-independent cost, reservation-vs-deadline
cost and availability/pending/failures (5each=15); three N4-vs-large normalized
cost ratios (3), total24. Approximate percentile CIs confidence1-.05/24. Per
replica estimates and leave-one-out also reported. N4 gain vsMAPPO lower>0,
gain vsindependent lower>0, gain vsPPO lower>=-.01 (1% noninferiority). A claim
of beating PPO additionally requires gain lower>=.01, not merely positive mean.
Health for candidate vsPPO at N4 and deadline at large scales: availability
lower>=-.01; pending difference per machine upper<=.025; failure difference
per machine upper<=.25 per48 ticks. Report losses even within these margins.
Scale cost versus deadline gain lower>=-.02, versus independent lower>0;
normalized cost large/small upper<=1.10, p95 single-state inference <=50ms and
no invalid assignment. Main success requires all N4/scale/mechanism/health gates.
Intervention ablation is simulator-level evidence of the implemented reservation
service, not real-world causality. Zero collisions alone do not prove better cost.

Seeds use new raw ranges recorded in configs/technician_reservation_ppo_v2_seed_registry.json.
Development train53100000..2, dev53200000..31, analysis53300000. Confirmation
train53400000..4, dev53500000..63, test53600000..255, analysis53700000. Smoke
train53800000, dev53900000..1, test54000000..3, analysis54100000. SHA256 roles
separate scale, initialization, action RNG, minibatches and physical tapes;
paired reservation/independent initialization is intentional. Audit expanded
historical registries and all panels. Development never constructs test states.

Artifacts: fresh UTC directory, manifest RUNNING/COMPLETED/FAILED, Git/runtime,
source/protocol/config/registry hashes, resolved physics per scale, seed audit,
training_progress.csv/training_episodes.csv/audits, development_episodes.csv,
checkpoint_choices.json sealed before tests, per-arm/scale/seed checkpoints,
episodes.partial.csv/finalepisodes.csv, coordination.csv (all final decisions),
latency.csv, summary.json, audit.json. Flush journals; preserve failures. Smoke
separate panel,384steps and2checkpoints, 2dev/4engineering-test episodes, N4/N8.
Tests: exact scalar physics and tail, masks/legality/no-catalogue at scale,
sequence likelihood replay and gradients, ablation initial equality, no future
input, checkpoint roundtrip, training/sample counts, pairing, seed audit and
statistical units/gates. Run full suite and smoke, commit/push verified code,
then run development here. Each amended code round repeats tests/smoke/push.

## Prospective amendment A — before any development or final test outcomes

To respect the user's continuing research goal without optional-stopping claims,
confirmation round r spends alpha=.05/(r*(r+1)), so all future rounds together
spend at most .05. This first round reserves alpha=.025; its24-endpoint family
uses confidence1-.025/24. Each later round requires a new prospective protocol,
fresh raw test seeds and an incremented round index; no reopening past tests.
Bootstrap intervals remain approximate with few training replicas. Smoke uses
its original engineering-only confidence and never contributes scientific claims.

## Prospective v2 design amendment — development feedback only

Source evidence: v1 partial development, all three N4 reservation replicas and
all three independent replicas complete. Mean selected dev cost1210.146 versus
1121.812 respectively; no scientific test panel opened. PPO N4 development was
1257.625 at this early budget. V1 continues its entire locked budget in its
original workspace. V2 is an exploratory new design, not a retroactive repair
of v1 or an outcome-confirmed improvement.

Hypothesis: value-gradient interference in the shared edge encoder and missing
explicit selected-assignment messages impede within-tick planning. Separate
actor and value edge encoders; value gradients cannot reach actor encoder.
After each reservation broadcast a cumulative selected-edge embedding, summed
and divided by K; WAIT contributes zero. Both actor edge scoring and WAIT head
receive this prefix message. The independent arm has the SAME new parameters,
initial weights and raw initial observations but zero prefix messages, fixed
initial resource summaries and no reservation masking. The causal contrast
therefore tests the entire reservation/message service, not an isolated claim
about one message type. PPO/MAPPO remain the original baselines with identical
training budget and hyperparameters. Parameter counts remain independent of
scale for the graph actor/critic. No teacher labels or hand-set decision bias.

Development budget491520 per model now matches the final confirmation budget:
do not select a prototype using only its undertrained early regime. Three
replicas at N4/N16, 21models and10321920 learned steps. All other success gates,
health margins, artifacts, bootstrap units and sample/optimizer settings remain
as specified above. All new seed panels are disjoint from v1. Only one design
v2 is locked; no hyperparameter sweep. Confirmation alpha has NOT been spent
by v1 development and this design's first scientific confirmation is round1.

Optional compute implementation: four independent CPU worker processes, each
Torch threads1, with deterministic per-model RNG roles. This is parallel compute,
not additional training samples, replicas or agent delegation. Per-model
journals stream to its own directory, merged deterministically by the parent;
all jobs and checkpoint choices complete before any test. Each training job
has tqdm and an outer model completion bar. Tests must show serial and process
training give identical parameter hashes/selected decisions on smoke. Parent
failure preserves worker journals and a FAILED manifest. Full runs still use
one frozen per-model budget; no timeout-triggered performance selection.

## Prospective amendment B — before ANY v2 development or scientific tests

The user's mechanism requirement is a benefit of coordination under matched
information/budget, not a benefit at every size. V1 already showed an N4
mechanism failure on development; disclose this source of the following design
choice. For v2 the PRIMARY mechanism test is at the three scaling settings:
reservation gain versus independent lower>0 at N8,N16,N32, all predeclared and
all retained. At N4 require noninferiority versus independent (gain lower>=-.01),
in addition to PPO noninferiority and beating MAPPO. Report the N4 positive-gain
flag/CI regardless, but it is not required for the scaling contribution's
success and must not be claimed if it fails. The same24-endpoint confidence
family includes that contrast; no statistical family reduction. The development
screen mirrors the N4 1% noninferiority margin against independent (instead of
requiring strict improvement). This supersedes those two earlier copied v1
requirements only. No v2 development states or scientific test outcomes have
been constructed; only separate engineering smoke has run. This is a disclosed
hypothesis refinement using v1 development, not reinterpretation of a v2 result.

## Prospective amendment C — strengthen the information-matched control

Before any v2 development/scientific outcome, give BOTH graph arms normalized
public machine and technician indices as two edge encoder input features (24
instead of22). Ordered array positions already reveal these identities to
catalogue PPO, MAPPO and the arbiter. No additional physical/health/future data
is supplied. This permits the independent ablation to learn symmetry breaking
when many machines share identical physical features; otherwise forced identical
bids and lowest-index tie resolution could exaggerate the mechanism's scale
benefit. Initial weights, capacities, training budgets and raw information still
match between the two graph arms. No permutation-invariance claim. The graph
parameter count remains independent of N/K. Report these identity features and
tie rules when interpreting any mechanism gain. Earlier v2 smokes are preserved
as engineering versions identified by their source hashes; a final smoke of
this frozen architecture is required before its research launch.
