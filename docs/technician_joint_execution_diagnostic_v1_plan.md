# Frozen Gaussian execution-mode diagnostic

Register before new outcomes. Prior goal turn made progress: FULL24fits completed,
451bindings and65,280primitive transitions independently verified; current7/13DEV
verdict remains negative. No confirmation/larger-N, teacher, learning update,
checkpoint reselection or retrospective rescue here.

Question: does deploying a single stochastic action from the TRAINED policy,
instead of projecting its mean utility, recover enough quality to motivate a
separate controlled learning trial? For nonlinear many-to-one matching decoders,
mean-projection need not maximize physical matching probability; this fact alone
does not imply a performance gap. Do not change sigma, temperature or pick the
best of multiple action draws. Retain every outcome and failure.

Source technician_joint_clearing_ppo_v1_development_20261016T081402007731716Z,
learning690be2d8241b9e5a2625c537e0c684f95674b4d0, COMPLETED/auditPASS,451bindings.
Verify source snapshot/config/checkpoints/output SHA/size before and after.
Use all8methods ×3protocol-selected replicas. Every method runs both original
mean/greedy execution and stochastic sampling on64FRESH diagnostic physical raw
seeds260100000..63 using the source physical namespace. 12fresh action raw seeds
260200000..11, indexed replica*4+repeat. Gaussian core share the core RNG role;
PPO/norm share their categorical role; local/full MAPPO share a separate role.
Audit raw registries across worktrees and known source-derived streams.

Mean/greedy executes model.select separately per state, exactly as source (no
batched rounding/tie semantic change). Stochastic uses model.sample with explicit
RNG, epsilon0, temperature1; one sample/physical decision. Four action-RNG repeats
per model/physical seed are AVERAGED, never winner-selected. Mean64+sample4*64
=320episodes/model;24models=7,680episodes /368,640policy decisions /460,800physical
ticks including the SAME12tick deadline tail. Stochastic batched64 for compute;
all conditional probabilities/values replay within source F32 tolerances.
Single-thread torch/BLAS, sequential models, tqdm,15min hardstop, fresh timestamp
directory/no overwrite; tests/commit/push first; no separate smoke.

Store every physical state/action/event/following/cost, sampled latent/categorical
action, state hash, reward and behavior log/value; episode metrics and fixed-mode
tags. Reconcile reward-cost,48+12ticks, zero invalids, physical populations across
all methods/modes/repeats, unchanged checkpoint parameters. Independently replay
all460,800transitions/cutoff health using the already unit-tested primitive
verifier helpers (no primary simulator transition imports for that check).
These repeats are not additional training replicas or independent physical seeds.
No CI/pvalue or scientific claim from this diagnostic panel.

Report per method/replica all mean versus sampled cost/health, action-repeat
spread, retained critic predictions and original checkpoint steps. Strong
references for quality/health remain original and normalized PPO's MEAN/greedy
policies; do not weaken the comparison by switching only references to sampling.
Also report MAPPO mean references and all stochastic controls, so an apparent
candidate improvement is not isolated from the same execution choice elsewhere.

## Fixed engineering decision

A NEW stochastic-deployment learning trial is motivated ONLY if all hold:
- Joint sampled cost gains>=1% versus its own mean on pooled panel, and positive
  gain in EACH of its three train replicas.
- The SAME13source gate definitions pass with joint SAMPLED as candidate,
  sampled forward/reverse/independent as matched core controls, but original /
  normalized PPO and local/full MAPPO MEAN policies as strong references.
  Thus independent gain>=.005; serial and PPO gains>=-.01; MAPPO gains>0;
  availability>=reference-.01, pending/m<=reference+.025, failures/m<=reference+.25
  against BOTH PPO mean references. All13mandatory, zero invalid.

Failure means no automatic stochastic-deployment training trial or temperature /
variance search. A positive outcome only motivates separately preregistered
random-init FULL with appropriate checkpoint selection and fresh development /
scientific panels; it never reclassifies the source7/13failure as passing. This
diagnostic does not test all possible execution methods or explain architecture
failures if sampling also fails.

Artifacts: manifest with code/source/config/registry hashes, snapshots, seed audit,
all physical and policy journals, episode grid/reward accounting/behavior replay,
independent primitive replay audit, per-method/mode/replica summary and fixed
eligibility. Manual rsync -avhP --partial; no SSH or automatic transfer.
