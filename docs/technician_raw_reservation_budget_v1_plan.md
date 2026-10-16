# Bounded raw-state reservation optimization-budget follow-up

Registered before any new learning outcomes. Preserve raw-state v1 FULL's negative
12/13 verdict (mechanism fails) and all prior results. Direct Ubuntu lab FULL,
no separate smoke; correctness tests and commit/push before launch. N4/K2 only.
No teacher, warm start, demonstrations, shaping, new encoder, tuning sweep or new
RL objective. Policy/physics/optimizer are frozen as in learning commit 16ea7cb.

## One falsifiable optimization hypothesis

Two of three reservation replicas selected the last 491520 checkpoint, whereas
independent controls reached the same better aggregate outcomes by 245760 in two
replicas. Frozen candidate interventions hurt, but retrained independent controls
win; this is NOT evidence the mechanism is already helpful. Test exactly once:
does doubling a matched training budget let prefix/reservation learning catch up
and create the registered mechanism benefit? This is bounded optimization work;
extra training alone is not an algorithmic contribution. Weak late improvement
is a motivation to test, not proof of insufficient training.

From random initialization on fresh seeds, train every one of the same eight
methods for 983040 decisions, twice the old budget. Compare within this new study
short-budget choices (checkpoints <=491520) with long-budget choices (<=983040).
All methods have identical checkpoint opportunities at each budget. All core
arms keep exactly equal parameters, physical sensing, RNG roles and private
on-policy training budgets. All eight get equal freshly seeded critic tensors,
global norm clipping .5; PPO retains literal unit1 and normalized PPO/core/MAPPO
use U4. Architecture, conditional semantics, arbitration, rewards, service physics,
current-state broadcast and prior-art limits remain those of raw-state v1.

## Locked design / seed splits

8 methods x 3 fresh train seeds, N4/K2, 983040 decisions/model =23592960 total;
491520 training episodes, 368640 Adam updates. 16envs, rollout3072, epochs4,
minibatch256, Adam3e-4, PPOclip.2, entropy.01, value.5, globalclip.5, gamma1.
48 policy +12 common-tail ticks, balanced_a2_b4_u12. No changed intervention.
Checkpoints 0,122880,245760,368640,491520,614400,737280,860160,983040;
32 DEV seeds each =6912 checkpoint evaluation conditions. Earliest minimum mean
H+L cost selects separately within each budget; spend the entire long budget.
No pre-checkpoint selection from costs unavailable by the respective cutoff.

DEV train280000000..2; physical280100000..31; analysis280200000.
Sealed potential CF train280300000..4, DEV280400000..63, TEST280500000..255,
analysis280600000; disposable engineering280700000..3. Registry all historical
raw seeds across worktrees and derived streams including raw-state v1. Hashing
CF namespaces does not construct CF physical states. DEV-only CLI. Eight CPU
workers, torch/BLAS1, tqdm; 12h hardstop/technical abort with preserved artifacts.
No metric early stop, resume, seed replacement or extra budgets after outcomes.

## Gates and rejection

At long budget retain exactly the original13 DEV gates: gain>=.005 vs independent;
gain>=-.01 vs mask-only/reverse/PPO/normalized PPO; gain>0 vs local/full MAPPO;
vs both PPOs availability delta>=-.01, pending/m delta<=.025, failures/m H48
delta<=.25. Also zero invalid and isolated actor+decoder p95<=50ms from432 raw
samples after workers close. These primary evaluations use full trajectories and
primitive audits; frozen candidate interventions retained descriptively.

For the optimization explanation additionally require the reservation-versus-
independent cost gain to improve by >=.005 absolute (0.5 percentage points) from
short to long budget, AND candidate selected mean cost to drop >=.005 relative.
The short selection and interaction are arithmetic DEV diagnostics from all
recorded checkpoint rows, not independent scientific tests. Do not conflate a
larger checkpoint menu with causal effects of only the final parameters.
All13 +engineering +both budget conditions must pass before this follow-up is
eligible. No claim prefix alone helps from mask-only noninferiority. No claim
zero-conflict alone proves cost benefit. If any fail, reject this bounded
additional-budget remedy; do NOT automatically double again, change LR, add an
encoder/attention/capacity component or reuse its seeds to rescue this verdict.

Only completed independently verified eligibility can authorize registering a
CF harness at this SAME long budget,5 fresh replicas/64DEV/256 sealed TEST. Freeze
choices before test. Paired crossed bootstrap200000, original13 simultaneous
bounds, unused raw-state family alpha .05/6 (no previous scientific test opened),
confidence1-(.05/6)/13. PPO-superiority only if gain CI lower>=.01; otherwise at
most near-PPO. Optimization diagnostics remain DEV eligibility, not additional
scientific claims. Larger-size work still closed until N4 confirmation passes.

## Artifact contract / validation

New timestamped directory, never overwrite earlier full results. Seal all source,
Git/protocol/config/registry/physics. Retain all24 initial/final/checkpoint tensors,
common initial critic hashes, full paired training tapes/journals/behavior replay,
all6912 DEV rows, short and long checkpoint choices, short_budget_summary.json
and budget_hypothesis.json. Primary selected/frozen/deadline trajectories as v1:
1088 episodes,65280 physical transitions,52224 decisions. Retain432 raw isolated
timings and choice/observation bindings. Seal outputs and keep test_opened=false.

Correctness tests cover fresh namespaced physics, exact same policy/optimizer,
budget and checkpoint accounting, cutoff-limited earliest selection, budget-gate
boundary math, all eight actual PPO updates and scalar physical audits, checkpoint
roundtrip, frozen conditioning restoration and process-vs-serial determinism.
Necessary disposable fixtures are not a separate research smoke. Commit/push
before FULL. Independent verification/report after completion; no success based
on its built-in audit alone. Manual rsync -avhP --partial only; no SSH/transfer.
