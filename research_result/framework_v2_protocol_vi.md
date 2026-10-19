# Protocol khóa trước: PPO phân phối bốn cấu hình, framework v2

Khóa ngày 2026-10-19 sau khi audit v1, trước code/test/training v2.
V1 TEST đã mở chỉ dùng để tạo giả thuyết, tuyệt đối không dùng lại làm xác
nhận. Source nền v1 commit `2fc6840a63fed765ec1503006765ec55e8dfc9d9`.
Checkpoint đối chứng v1 cố định tại
`research_result/framework_v1_full_20261019T153014Z/checkpoints/62000002/199680.zip`,
SHA256 `6f1400b2e11990d86dbd14d137b85c2f2ac73ccf1c1c380bee8d03a779727cb9`.

## Câu hỏi và giả thuyết

V1 học chỉ trên reciprocal scale 2, mất chất lượng khi topology/scale đổi;
trace mô tả nhiều WAIT hợp lệ ở specialist/generalist. Hỏi liệu quy trình
**train trên hỗn hợp bốn case và chọn checkpoint trên DEV bốn case** giảm
case-balanced H48+tail16 cost so với v1 checkpoint cố định trên TEST mới hay
không. Đây là hiệu ứng gộp của training support và selection rule; không
khẳng định tách riêng nguyên nhân. H1 xác nhận nếu paired pooled mean
`mixture PPO cost − v1 PPO cost` <0 và cận trên paired cluster-bootstrap CI95%
<0. CI chứa 0 là chưa rõ, mean ≥0 bác bỏ hướng cải thiện. Không dùng case
thắng riêng hay baseline yếu để thay H1.

## Setting, arms và công bằng

Giữ nguyên cả bốn case v1, dynamics, requester threshold, H48+tail16 common
risk tail, cost PM1/CM2/unavailable6/failure15, 56 feature observation,
21 matching catalogue, legal mask, PPO architecture/hyperparameters và
199.680 train steps/seed. Mỗi episode train chọn uniform một trong bốn case
trước reset; model không nhận nhãn case riêng nhưng nhận toàn bộ matrix
service-time/eligibility trong observation. Cùng tổng 3 seed ×199.680 steps
=599.040 như v1. V2 và v1 evaluation deterministic, cùng current information,
physical seed, reward/physics/mask. Không thêm waiting penalty. V1 checkpoint
được khóa như historical comparator, không train lại hay chọn lại. Vì DEV rule
khác, đối chiếu không cô lập riêng tác dụng của phân phối train.

Thêm FCFS, risk matching và rollout 6-tick/16-scenario làm references hiện
hành; rollout dùng independent forecast base 63300000, không đọc physical
future tape. Báo compute: selection seconds, episode wall seconds, simulated
forecast transitions. Không coi rollout là oracle tối ưu.

## Split, selection, estimand

Registry `configs/maintenance_framework_v2_seed_registry.json`: training
63000000..63000002, DEV 63100000..63100015, TEST 63200000..63200063,
forecast 63300000, bootstrap 63400000, smoke 63500000..63500001. Đã kiểm
tra không trùng registered historical seeds. Raw physical seed domain giữ
`maintenance_framework_v1:physical` vì cùng kernel; role-separated forecast
mới độc lập. Cùng raw TEST seed dùng trong cả bốn case và năm controller để
paired comparison.

PPO hyperparameters y hệt v1: MLP [64,64], rollout1024, batch256,
epochs5, gamma.99, GAE.95, clip.2, LR3e-4, entropy.01, value.5, max
norm.5. Checkpoints 50.176, 100.352, 149.504, 199.680 mỗi seed. DEV đánh
giá 12 checkpoint ×4 case ×16 seed; chọn checkpoint có arithmetic mean
H+tail cost thấp nhất trên toàn bộ 64 case×seed cell, tie seed nhỏ rồi step
sớm. Chỉ mở TEST sau khi lưu `selection.json`. Không tune thêm từ DEV.

Primary pooled difference: với mỗi physical seed, tính trung bình của bốn
`mixture − v1` cost difference rồi lấy mean 64 seed. Paired cluster bootstrap
20.000 draws lấy lại 64 raw seed có thay thế, mỗi seed giữ nguyên bốn case;
CI percentile hai phía95%. Secondary: per-case cost/paired differences, H
availability, H failures, queued CM/PM exposure, H/end pending và in-service,
H/tail costs, cost components, action time và forecast compute. Per-case CI
95% là mô tả, không multiplicity-adjusted. Báo cả case thua và health xấu;
không gọi giảm cost là cải thiện toàn diện nếu availability/failures xấu.

## Ngân sách, gates, artifact

Full hard deadline6h từ runner start. Không extend, resume hay đổi seed khi
nhìn kết quả. Smoke: 1 seed×1024 train steps, 2 DEV seeds, 2 eval seeds/case;
engineering only. Tests và smoke phải pass, commit/push code verified trước
full. Full run trên Ubuntu lab theo quyền user riêng goal này, không SSH hay
rsync. Dùng tqdm cho train, DEV và TEST. Mỗi run dùng thư mục mới timestamp
trong `research_result/`; từ chối ghi đè.

`manifest.json` ghi phase/status, commit, exact invocation, runtime, source và
v1 checkpoint hashes, splits, counts, timing. `protocol.md`, `seed_registry.json`,
`resolved_config.json`, `training_progress.csv`, `training_case_counts.json`,
`development.partial.csv`/`development.csv`, `selection.json`,
`episodes.partial.csv`/`episodes.csv`, `summary.json`, `audit.json`, 12
`checkpoints/<seed>/<step>.zip`. FAILED/INCOMPLETE giữ partial files. Gates:
đủ 768 DEV và 1.280 TEST episode, checkpoints load/reproduce action, hash
không đổi, exact seed grid, objective=components=−episode return,
request conservation, zero invalid assignment. Sau FULL cần verifier độc lập
trước kết luận.
