# Protocol khóa trước v6 edge-head sharing confirmation

Proposal `research_result/method_v6_proposal_vi.md`. Giữ nguyên simulator,
request threshold, H48+tail16 economic objective, four fixed N4/K2 cases,
56 current features, 21 matching catalogue và legal mask. Three scratch-RL
arms: flat, unshared_edge, shared_edge. PPO hyperparameters v5 giữ nguyên
(n_steps1024, batch256, epochs5, gamma .99, lambda .95, lr3e-4, clip .2,
entropy .01, vf .5, grad clip .5), CPU one Torch thread. Uniform case
mixture. 5 training seeds×3 arms×100.352 steps =1.505.280 steps. Common
case/physical training tape per seed across arms. Checkpoints 50.176 and
100.352. No IL, pretraining, capacity penalty, learning-rate/architecture
sweep or extra FULL.

Fresh registry `configs/maintenance_framework_v6_seed_registry.json`:
train 67000000–04, DEV 67100000–15, sealed TEST 67200000–63, forecast
67300000, bootstrap 67400000, smoke 67500000–01; check no overlap with
historical. DEV 3 arms×5 train seeds×2 checkpoints×4 cases×16 seeds =1.920
episodes. Within each train seed and arm, select one checkpoint by four-case
DEV mean cost (tie smaller step); do not pick a winning train seed. Hash
selections before TEST. TEST learned: 3 arms×5 selected train seeds×4
cases×64 physical seeds =3.840 episodes; plus FCFS/risk/rollout once per
case/physical seed =768, total 4.608. Each learned row records train seed
and checkpoint. All 5 model pairs share same physical seeds; forecast
rollout uses 16 scenarios and horizon6, distinct forecast stream.

Primary per (train seed, physical seed): mean over four cases of shared −
unshared H+tail cost. Point estimate mean over 5×64 cells. Hierarchical
paired bootstrap 20.000 draws resamples 5 training seeds and 64 physical
seeds independently, retaining all four case paired observations. Confirm
only if mean<0 and upper CI95%<0. Secondary pair contrasts shared−flat,
shared−risk/rollout/FCFS, unshared−flat, case-level means/CI descriptive;
no multiplicity-adjusted claim. Record H availability/failure/pending,
end queue, cost components, selection wall/forecast transitions, training
wall, model parameters and per-train-seed primary estimates.

Tests: exact action incidence/WAIT, actor permutation for shared vs
unshared, param count near-match, PPO gradient/save/reload, simulator
regressions. Ubuntu smoke 1024 steps/arm, 2 smoke DEV/eval seeds, independent
audit PASS; commit/push verified inputs before FULL. Campaign hard stop 4h,
tqdm training and multi-seed evaluation, timestamped directories under
`research_result/`, no overwrite. Manifest stores source commit/hashes,
config, splits, tapes, runtime, checkpoint/selection hashes, per-episode
CSV and summary. Independent verifier regenerates tapes, validates hashes,
selection and grids, economics/conservation/feasibility, all bootstrap CIs.
Post hoc true-state permutation logits for each selected shared/unshared
model on original TEST trajectories can be run after audit; no checkpoint
selection or cost claim from that diagnostic.
