# Protocol khóa trước thí nghiệm v5 post-decision capacity

Method proposal: `research_result/method_v5_proposal_vi.md`. V5 chỉ có một
FULL theo protocol này. Cùng bốn config N4/K2 và H48+tail16 của v4,
không đổi physics, request threshold, cost hoặc terminal cleanup. RL là
controller chính; scratch MaskablePPO với mask feasible matching. Four arms:
flat, edge_only, additive_queue, postdecision. Structured arms cùng modules,
initial parameter values per replica; 3 training seeds ×199.680 steps/arm,
checkpoints ở 50.176/100.352/149.504/199.680. Uniform four-case training
mixture; paired case/physical episode tapes per replica. PPO hyperparameters
đúng như v4: n_steps1024, batch256, epochs5, gamma .99, lambda .95,
lr3e-4, clip .2, entropy .01, vf .5, grad clip .5. CPU, one Torch thread.
Không tuning hay teacher. 2.396.160 total PPO steps.

Seed registry duy nhất `configs/maintenance_framework_v5_seed_registry.json`:
3 training 66000000–02, 16 DEV 66100000–15, 64 sealed TEST 66200000–63,
forecast 66300000, bootstrap 66400000, smoke 66500000–01. Mọi seed không
trùng historical registries. DEV: 4 arms ×12 checkpoints ×4 cases ×16 seeds
=3072 episodes. Chọn một checkpoint/arm bằng case-balanced DEV mean cost,
tie-break deterministic score/seed/step. Chỉ sau khi ghi selection hash mới
mở TEST: 7 controllers (4 learned+FCFS/risk/rollout) ×4 cases×64 seeds
=1792 episodes. Forecast rollout dùng 16 scenarios, 6-tick horizon và
independent forecast seed; không đọc future realization. TEST physical seeds
paired. Bootstrap 20.000 draw theo 64 physical seed clusters, mỗi cluster
trung bình đều 4 case. Primary `postdecision − additive_queue`, H+tail16
cost; confirm nếu mean<0 và upper two-sided CI95%<0. Secondary contrasts
postdecision với edge_only, flat, FCFS, risk, rollout; case-level CI,
H availability/failures, queue, tail, cost components, latency và compute.
Không tự động xác nhận secondary sau primary fail.

Tests: action incidence/WAIT, exact residual-wait arithmetic trên synthetic
state, action dependence/complementarity, permutation relabeling, PPO
learning+save/reload, simulator/economics regression. Chạy tests và Ubuntu
smoke 1024 steps/arm với 2 smoke seeds cho DEV/eval, audit độc lập PASS.
Commit+push verified sources/proposal/protocol/registry trước FULL. FULL
hard stop 4 giờ; không resume hoặc overwrite run cũ. Mỗi run/audit có thư
mục timestamp mới dưới `research_result/`, tqdm training/evaluation.
Manifest lưu commit/source hashes, config, seeds, runtime, budgets,
checkpoint hashes, selected checkpoint, training tapes, per-episode raw,
summary, diagnostics. Independent verifier tái tạo tapes, kiểm hashes,
DEV selection, grid, economics/conservation/feasibility và bootstrap.
Post hoc same-weight read-only penalty ablation + actor permutation check
sau independent audit trên 4×64×48 selected-policy TEST states; không dùng
để chọn checkpoint hoặc claim cost. Một FULL thất bại/kết quả âm được giữ
và báo; không chạy adaptive sweep.
