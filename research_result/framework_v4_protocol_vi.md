# Protocol khóa trước v4: capacity opportunity policy học RL từ đầu

Ngày 2026-10-20. Viết trước code v4, DEV/TEST v4, sau khi đã đọc v1–v3.
[Method proposal](method_v4_proposal_vi.md) là rationale. Source nền
5ee36e2. Không dùng imitation/pretraining; PPO là optimizer chung, đối tượng
nghiên cứu là state/action scoring. Không thay kernel, objective hoặc bốn case.

## Câu hỏi, endpoint, rule quyết định

Primary: learned temporal opportunity policy trừ current-queue ablation có
case-balanced H48+tail16 cost thấp hơn 0 trên 64 sealed TEST physical seeds?
Cùng seed dùng cho bốn case và mọi controller. Trên từng seed lấy mean của
bốn case differences, bootstrap 20.000 lần theo nguyên cụm seed từ
65400000; CI hai phía 95%. H1 chỉ đạt nếu mean<0 và upper CI<0. Nếu
temporal giảm cost nhưng H availability, failures hay CM queue xấu đi, báo
riêng và không gọi là cải thiện đồng đều.

Secondary descriptive: current-queue trừ edge-only, temporal trừ edge-only,
temporal trừ flat PPO/risk/rollout/FCFS; cost components, H availability,
failures, queued CM/PM, cutoff/end requests, inference CPU wall, parameter
count, train wall và case-specific contrasts. CI95% descriptive cho ba
case-transfer/component contrasts, không familywise confirmation. Không claim
scalability N/K mới. Permutation equivariance of actor là test/property;
permuted case TEST là robustness diagnostic chứ không dùng để chọn.

## Controller/setting

Tất cả học trên cùng bốn case mixture, mỗi episode chọn một case đều.
N4/K2, threshold PM age≥3, persistent PM→CM, nonpreemptive service,
completion-time restoration, 56 current-state features, 21 partial matching
catalogue và exact legal mask. Objective H48+common risk tail16 giữ nguyên
PM1/CM2/unavailability6/failure15. Physical shocks không vào observation.
Rule FCFS, risk matching và rollout 6 tick ×16 independent forecast dùng
same current information, physics, objective, seeds. Rollout known model
có extra compute, không gọi là oracle exact.

Bốn scratch-RL arms: flat MLP [64,64] cũ; edge-only actor chấm từng
request–worker edge qua shared encoders và cộng thành matching logit;
current-queue thêm learned nonnegative potential của queued PM/CM của các
máy *khác*, chia số worker eligible, nhân thời gian service để trừ khỏi
edge score; temporal thêm potential của máy đang hoạt động age2 ngay trước
threshold PM. Cùng actor base/critic/optimizer ở ba structured arms.
Current-queue và temporal khác đúng gate near-future trong cùng branch.
Mask chung; WAIT logit 0. Critic chỉ phục vụ training. Actor không dùng
machine/worker ID; phép hoán vị action logit phải được unit test.

PPO hyperparameters cố định từ v1: n_steps1024, batch256, epochs5,
gamma.99, GAE.95, clip.2, LR3e-4, entropy.01, value.5, max norm.5.
Mỗi arm × ba training seeds ×199.680 decisions = 2.396.160 PPO decisions.
Checkpoint 50.176/100.352/149.504/199.680 mỗi replica. Physical training
episode-seed sequence và config sequence phải giống nhau giữa arms theo
seed; trajectories có thể khác do actions. Initial weights cùng nhau trong
ba structured arms *trừ* gate-specific context branch? Lưu hashes, không
đòi flat đồng weights vì architecture khác. Không đổi hyperparameters
theo DEV.

## Splits, selection, full grid

Registry configs/maintenance_framework_v4_seed_registry.json:
training 65000000..02; DEV 65100000..15; sealed TEST 65200000..63;
forecast 65300000; bootstrap 65400000; smoke 65500000..01.
Preflight kiểm không trùng registry cũ. DEV chọn *riêng từng arm* một
trong 12 positive-step checkpoints bằng mean case-balanced cost trên bốn
case ×16 seeds; tie nhỏ training seed rồi step sớm. Ghi selection/hashes
trước TEST. DEV grid 4×12×4×16=3.072 episodes. TEST grid
(4 learned + FCFS/risk/rollout)×4×64=1.792 episodes. Không dùng TEST để
chọn hoặc điều chỉnh method. Báo cả checkpoints và mọi case, kể cả âm tính.

## Stopping, engineering gates, artifacts

Ubuntu smoke trước full: một training seed, 1.024 decisions/arm, hai DEV
và hai eval seeds/case, nhỏ chỉ kiểm engineering. Tests phải kiểm policy
logits finite/masked, action remapping exact under machine/worker
permutations, gate controls, no realized future, checkpoint save/reload,
episode accounting; simulator regression tests tiếp tục. Commit và push
verified code/protocol trước FULL. Full hard deadline4h, không extension,
adaptive tuning hoặc restart sau khi mở TEST. Nếu lỗi kỹ thuật, giữ FAILED
artifact và amendment công khai trước rerun với source mới, fresh directory.
tqdm trong train/DEV/TEST. Mỗi run thư mục mới timestamped trong
research_result/, không overwrite.

Manifest: status/phase/commit/branch/source hashes/seed registry/config/
exact invocation/deadline/counts; per-arm training progress, training case
and physical seed tapes, 48 checkpoints, development.csv, selection.json,
episodes.csv, summary.json, audit.json. Raw/checkpoints ngoài Git.
Independent verifier phải recompute source/checkpoint hashes, seed disjoint,
training paired tapes, DEV selections, TEST grid, economics/reward and
request conservation, zero invalid assignments, and registered pooled
bootstrap before scientific interpretation. Small findings report and
reproduction command commit/push after audit.
