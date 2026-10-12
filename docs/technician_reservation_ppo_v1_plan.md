# Technician reservation PPO v1 — prospective lab protocol

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

Stage development (not final scientific evidence): 3 train seeds, 122880 steps
per arm/scale, vector16, rollout3072, 4epochs, minibatch256, Adam3e-4, clip.2,
entropy.01/value.5/gradclip.5. Checkpoints30720,61440,92160,122880; 32 dev episodes.
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

Seeds use new raw ranges recorded in configs/technician_reservation_ppo_v1_seed_registry.json.
Development train51000000..2, dev51100000..31, analysis51200000. Confirmation
train51300000..4, dev51400000..63, test51500000..255, analysis51600000. Smoke
train51700000, dev51800000..1, test51900000..3, analysis52000000. SHA256 roles
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
