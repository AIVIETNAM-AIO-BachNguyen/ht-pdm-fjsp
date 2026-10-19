# Protocol khóa trước v3: rollout teacher, imitation, PPO fine-tuning

Ngày 2026-10-20. Khóa sau khi xem v1/v2 TEST và trước implementation hay
outcome v3. V1/v2 TEST đã mở chỉ dùng để đề xuất giả thuyết; v3 dùng registry
mới hoàn toàn. Source nền `b5cbb59`. V3 vẫn dùng RL policy trực tiếp chọn
matching tại inference; rollout planner chỉ tạo labels trên train seeds và là
baseline ở evaluation. Không đổi simulator, cost hoặc lựa chọn bốn case.

## Câu hỏi, hai giả thuyết đồng chính

V1 PPO chỉ học anchor, kém bounded rollout khoảng 122 cost trên TEST, trong
khi v2 tăng training coverage làm hại reciprocal timing. V3 hỏi (H1) liệu
policy PPO khởi tạo bằng rollout demonstration có cost anchor thấp hơn PPO học
từ đầu với cùng architecture, training tapes và **cùng 199.680 PPO steps/seed**;
và (H2) PPO fine-tuning có tốt hơn chính policy imitation-only trước PPO.
Nếu chỉ H1 đạt, kết luận teacher assistance chứ không kết luận RL fine-tuning
tăng chất lượng; nếu chỉ H2 đạt, không kết luận teacher giúp hơn scratch.

Hai contrast đồng chính trên sealed anchor TEST: teacher-PPO trừ scratch-PPO,
và teacher-PPO trừ warm BC. Mỗi contrast dùng 64 paired physical seeds và
20.000 seed-bootstrap draws. Để kiểm soát hai phép kiểm, dùng two-sided 97,5%
percentile CI cho từng contrast; H1/H2 tương ứng chỉ xác nhận khi paired mean
<0 và cận trên CI <0. Có thể báo paired 95% descriptive trên các case phụ,
không gọi familywise confirmation ở đó. Báo cả gains/losses và mọi checkpoint.

## Setting và thông tin

Anchor `reciprocal_scale2`: N4/K2, requester threshold age≥3, persistent PM
upgradable CM, service nonpreemptive, completion-time restoration; cost
PM1/CM2/unavailable6/failure15. Mỗi controller chọn legal partial matching
hoặc WAIT ở H48; common risk tail16 không reset. 56-feature current-state
observation, 21-action catalogue, exact legal mask, MaskablePPO MLP [64,64]
như v1. Rollout teacher: enumerate legal root matchings, 6-tick forecast,
16 independent scenarios, known model, future risk continuation. Teacher chỉ
nhận current state/config/time và independent forecast, không đọc realized
physical tape. Same physics, objective, current information và physical seeds
cho FCFS/risk/rollout, scratch, warm BC và teacher PPO ở evaluation.

Teacher bank cho mỗi training replica: 32 fresh physical episodes theo rollout
teacher, mỗi episode 48 labeled decisions, tổng 1.536 state/action/mask labels.
Ba bank tổng 96 episodes/4.608 decisions. Seed bank 64600000..64600095
chia 32 liên tiếp mỗi training seed. Dữ liệu có cả mask và observation; nhãn
phải legal. Lưu bank, seed/source hashes, label count, scenario transition count,
wall time. Không lấy labels từ DEV/TEST hay future event tape. Pretrain actor
10 epochs shuffled, batch256, Adam LR3e-4 với masked cross entropy; reset
optimizer state trước PPO. Lưu warm checkpoint và loss mỗi epoch. Teacher bank,
BC compute và PPO compute đều báo cáo; không tuyên bố sample/compute efficiency
thuần so scratch nếu không tính phần thêm này.

Scratch và teacher-PPO mỗi replica khởi tạo **cùng weights trước BC**, cùng
PPO hyperparams v1: rollout1024, batch256, epochs5, gamma.99, GAE.95,
clip.2, LR3e-4, entropy.01, value.5, max norm.5; train mỗi arm đúng
199.680 environment decisions/seed. Với cùng raw training seed, hai arm nhận
cùng chuỗi physical episode seeds và pre-generated shock tapes; audit initial
weight digest và seed sequence phải khớp. Teacher actions khiến trajectories
khác, đây là tác dụng policy, không phải khác shock distribution. PPO checkpoints
50.176, 100.352, 149.504, 199.680; warm-only checkpoint step0.

## Seed split, selection, evaluation

`configs/maintenance_framework_v3_seed_registry.json`: PPO train
64000000..64000002; DEV 64100000..64100015; sealed TEST
64200000..64200063; teacher bank 64600000..64600095; forecast 64300000;
bootstrap 64400000; smoke 64500000..64500001. Tất cả đã kiểm tra không trùng
historical registry. Physical role giữ `maintenance_framework_v1:physical` để
cùng raw seed cho paired controller; teacher forecast được domain-separated
khác physical. Teacher-bank forecast dùng base mới, rollout evaluation cũng
dùng base này nhưng seed role khác nhau.

Chọn riêng một scratch PPO trong 12 positive-step checkpoints, một teacher
PPO trong 12 positive-step checkpoints, và một warm BC trong ba step0
checkpoints bằng **mean H+tail cost trên 16 DEV anchor seeds**, tie training
seed nhỏ rồi step sớm. Không dùng test/case phụ để chọn. Ghi selection/hash
trước khi mở TEST. Evaluation deterministic trên cùng 64 TEST seed/case cho
bốn case v1/v2 và sáu controller FCFS, risk, rollout, scratch PPO, warm BC,
teacher PPO. Anchor là endpoint đồng chính; ba case khác transfer diagnostics.

Primary H+tail economic cost và two paired differences. Secondary H
availability, failures, queued PM/CM, H/end pending/in-service, H/tail costs,
cost components, selection time, forecast transitions, training/bank/BC wall
and model transitions. Report teacher bank cost alongside inference cost.
No health penalty or threshold added; if lower cost harms availability/failure,
report explicitly and do not call it uniformly better.

## Stopping, smoke, artifacts và audit

Full hard deadline6h from runner start; no budget extension, adaptive tuning,
seed extension, or resume. Smoke on Ubuntu lab: one training seed, two teacher
bank episodes, two BC epochs, 1.024 PPO decisions/arm, two DEV and two eval
seeds/case; engineering evidence only. Run relevant tests and smoke, commit/push
verified code first, then FULL directly on authorized Ubuntu lab. tqdm for
teacher bank, BC, both PPO arms, DEV and multi-seed TEST. Each invocation
requires a fresh UTC timestamped `research_result/` subdirectory; no overwrite.
Preserve FAILED/INCOMPLETE artifacts.

Artifacts: `manifest.json` (phase/status, git/source/config hashes, exact
invocation, runtime, splits, deadline, counts), `protocol.md`,
`seed_registry.json`, `resolved_config.json`, per-replica teacher banks and
metadata, BC losses, initial-weight/physical-sequence audit (including both raw
training episode-seed sequences for independent comparison), arm training
progress, warm/checkpoint files, `development.partial.csv`/`.csv`,
`selection.json`, `episodes.partial.csv`/`.csv`, `summary.json`, `audit.json`.
Gates: full grid of 432 DEV and 1.536 TEST episodes, bank label legality,
initial model weights equal per replica, physical seed sequences equal across
training arms, checkpoint save/reload action equality, no future tape access,
zero invalid assignment, objective=components=−episode return, request
conservation, stable source/teacher hashes. After FULL, independent verifier
must recompute grid, selection, hashes, costs, pairing and both registered CIs
before interpreting performance. Raw banks/checkpoints remain outside Git.
