# Protocol khóa trước: maintenance coordination framework v1

Ngày khóa: 2026-10-19, trước khi huấn luyện framework v1. Đây là synthetic
development benchmark, không phải kết quả hiệu chuẩn nhà máy. Code gốc là
handoff `82cb1e48c0589cc2aa735d380c8fadfdc2355606`. Phiên này được người
dùng cho phép chạy smoke và full campaign trực tiếp trên Ubuntu lab; các phiên
khác vẫn theo `docs/experiment_workflow.md`.

## Câu hỏi và giả thuyết có thể bác bỏ

Framework có biểu diễn và thi hành nhất quán quyết định chọn subset request,
WAIT và phân công technician khi request PM tồn tại, có thể chuyển CM và sửa
chữa hoàn tất mới phục hồi không? Gate kỹ thuật bác bỏ khi mask cho phép action
không hợp lệ, reward episode khác âm H48+tail16 objective, hoặc checkpoint
không tái lập policy. Giả thuyết thực nghiệm H1: PPO tập trung có mean paired
H+tail cost thấp hơn risk_matching trên fresh sealed anchor panel. H1 bị bác bỏ
nếu paired mean không âm; CI chứa 0 được xem là chưa rõ. Không yêu cầu PPO thắng
để framework hợp lệ. So sánh với FCFS và rollout cũng được báo cáo đủ bốn case.

## Setting và thông tin

N4/K2, threshold requester age>=3 cố định, failure age4/p=.45, cap8,
restoration .75 tại completion, cost PM1/CM2/unavailable6/new failure15.
H48 điều khiển rồi common tail16 không reset: threshold requester + risk_matching
dispatcher. Bốn case cố định: `reciprocal_scale2` anchor với matrix
[[2,6],[2,6],[6,2],[6,2]], `interchangeable_scale1`,
`reciprocal_scale3`, `specialist_generalist_scale2` theo source hiện tại.
Mọi controller nhận cùng config, current request/physical state và time.
Rollout dùng model/forecast riêng, không đọc physical future tape; so sánh
compute phải báo riêng. Không học gửi request, không thêm waiting penalty.

## Seeds và selection

Raw training seeds 62000000..62000002; development 62100000..62100015;
sealed test 62200000..62200063; forecast 62300000; bootstrap 62400000;
smoke 62500000..62500001. `configs/maintenance_framework_v1_seed_registry.json`
là nguồn chính. Đã kiểm tra không trùng mọi `configs/*seed_registry.json` lịch sử
trước khi thêm registry này; seed vật lý là hash domain-separated, cùng raw seed
được dùng cho mọi case/controller trong cùng panel để paired comparison.
Development chỉ dùng anchor để chọn checkpoint. Sealed test không được chạy
hoặc xem trước khi chọn xong checkpoint duy nhất.

Huấn luyện MaskablePPO từ đầu trên anchor, ba independent training seeds,
199.680 environment learning steps mỗi seed (195 rollout ×1024, tổng 599.040,
không vượt trần 200.000/seed và 600.000 toàn campaign). Một bộ hyperparams:
MLP [64,64], n_steps=1024, batch_size=256, n_epochs=5, gamma=.99,
GAE lambda=.95, clip=.2, learning_rate=3e-4, entropy_coef=.01,
value_coef=.5, max_grad_norm=.5. Checkpoints tại 50.176, 100.352,
149.504 và 199.680 steps (rollout boundaries); chọn checkpoint
có mean development H+tail cost thấp nhất trên 16 seed anchor, tie theo
training seed nhỏ rồi step sớm. Không điều chỉnh hyperparams sau dev results.
Policy deterministic khi evaluation. Không chọn theo ba case phụ.

## Metrics, ngân sách, stopping

Primary là mean H+tail economic cost và paired learner-minus-baseline cost.
Secondary: H availability, H failures, CM waiting, PM/CM queued exposure,
request backlog và in-service tại H và end, policy/tail cost, maintenance,
unavailability và failure cost, thời gian chọn action, thời gian cả episode và
forecast transitions.
Paired 95% bootstrap CI trên physical seeds, báo tất cả case và cả losses;
không tuyên bố familywise superiority. Rollout dùng 6-tick/16-scenario forecast
và nhiều model transitions; rule/learner inference có compute khác.

Toàn campaign training + development + sealed evaluation có hard deadline 6h
tính từ runner start. Khi hết hạn, manifest `INCOMPLETE`, giữ partial CSV,
checkpoints và progress; không tự tăng budget hoặc đổi panel. Tests và CPU
smoke phải pass trước commit/push; full chỉ chạy sau push. Mỗi run thư mục
timestamp mới, từ chối ghi đè. Smoke là engineering evidence, không vào kết
quả khoa học. Không dùng tmux, SSH hay rsync tự động.

## Artifact schema và gates

`manifest.json`: protocol, git commit, source/config hashes, runtime, seed
panels, phase/status, counts, deadline. `resolved_config.json`,
`seed_registry.json`, `protocol.md`, `training_progress.csv`,
`development.partial.csv`, `development.csv`, `selection.json`,
`episodes.partial.csv`, `episodes.csv`, `summary.json`, `audit.json`,
`checkpoints/<seed>/<step>.zip`. Progress hiển thị tqdm cho training và
multi-seed evaluation. Mọi episode lưu cost decomposition, H/tail, queue và
runtime. Gate: tất cả episode đủ grid, source hashes không đổi, costs khớp
components và reward, zero invalid assignments, checkpoint load giống action.
Chỉ manifest `COMPLETED` mới được gọi là completed benchmark.
