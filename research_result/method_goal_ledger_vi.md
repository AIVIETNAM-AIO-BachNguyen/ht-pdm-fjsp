# Ledger mục tiêu method điều phối RL

**Trạng thái: COMPLETE với kết luận method có điều kiện; canonical shared vẫn thua risk/rollout pooled.** Mục tiêu sửa đổi yêu cầu bằng chứng về giá trị của
một thiết kế điều phối, không chỉ pipeline huấn luyện hay tăng điểm PPO.
Không đánh dấu complete trước khi contrast method đã khóa được kiểm toán
và giới hạn/điều kiện hiệu quả được báo rõ.

| Mốc | Bằng chứng hiện có | Khoảng trống quyết định |
|---|---|---|
| Bài toán và simulator | Request-lifecycle H48+tail16, N4/K2, PM→CM, service không ngắt, assignment và WAIT; v1–v3 audits PASS. | Synthetic, không có production jobs hoặc N/K biến đổi. |
| Chính sách cũ | Flat 56-feature MLP/21 matching; v1–v3 cho thấy scratch/teacher kém rollout ở anchor, specialist transfer yếu. | Không cô lập được lỗi representation, exploration hay credit. |
| Literature/prior art | Method v4 proposal đối chiếu Learning to Dispatch, DANIEL, online bipartite matching, maintenance RL và planner; historical note đối chiếu Reservation PPO/prefix trong repo. | Chưa có novelty claim; cần đọc thêm full papers và benchmark gần nhất trước khi viết bài. |
| Method v4 | Shared edge score và potential cạnh tranh duration-weighted; ablation edge-only/current-queue/temporal; 4 arms scratch RL. Proposal/protocol ghi trước implementation, commit 7a5075b push trước full. | Tác dụng near-request age2 và giá trị trước rule/planner chưa biết. |
| Correctness v4 | 56 targeted/regression tests PASS; smoke 4.096 train steps/32 DEV/56 eval PASS, independent audit PASS. | Full run và audit độc lập PASS; diagnostic đã replay và kiểm tra actor permutation trên 12.288 states. |
| Bằng chứng method | Primary temporal–current-queue pooled cost; secondary edge-only/flat/risk/rollout; health/compute. | Temporal–queue pooled +8,52 [0,26;17,09], primary FAIL; queue–edge secondary −10,83 [−19,15;−2,66] nhưng không hiệu chỉnh multiple comparisons. Queue/temporal same-weight interventions đổi 83/13 trong 12.288 states. Xem framework_v4_findings_vi.md. |
| Generalization/scalability | Bốn skill/duration cases N4/K2 đã khóa làm panel transfer. | Bốn case N4/K2 cho thấy dị biệt lớn; chưa kiểm chứng N/K mới hoặc outside-simulator transfer. Actor state-level permutation PASS, không phải scalability. |

Các hướng cũ Teacher PPO v3 và Reservation PPO lịch sử là dữ liệu để chọn
câu hỏi và đối chứng, không phải bằng chứng rằng v4 method có novelty.
V4 đã chốt: không đổi gate hay mở lại TEST để cứu H1. Vòng tiếp theo nếu triển khai cần proposal về tương tác của matching với residual capacity, protocol mới và seed mới; chỉ một confirmation round có stopping rule rõ.

## Mốc v5 sau FULL (2026-10-20)

V5 proposal/protocol commit `83dfc8c`, FULL/audit PASS; xem
`research_result/framework_v5_findings_vi.md`. Postdecision−additive queue
primary +1,55 [−1,05;+4,11], FAIL. Same-weight residual penalty chỉ đổi
101/12.288 action dù 1.724 state có legal positive penalty. Additive−edge
v5 −2,21 [−4,95;+0,52], không tái hiện rõ descriptive benefit v4.
Structured postdecision thắng flat/risk v5 (secondary) nhưng không thắng
rollout và không hơn additive; shared pair representation chưa cô lập được
from parameter sharing/action factorization/training variance. Closest
literature: AirME HetGPO ICAPS 2022, reusable online matching và opportunity
loss; không claim novelty. Goal giữ ACTIVE: cần direct structural contrast
với fresh seeds hoặc kết luận rõ rằng chưa có component đủ mạnh.

## Mốc v6 sau FULL (2026-10-20)

V6 proposal/protocol và verified code commit/push `dbf4014` trước FULL;
1.505.280 PPO steps, 1.920 DEV, 4.608 TEST, independent audit PASS; xem
`research_result/framework_v6_findings_vi.md`. Matched-budget shared edge
head so 8 unshared heads có primary −85,59 cost, hierarchical CI95
[−171,24;−0,064], đạt rule khóa nhưng chỉ 3/5 train seeds ủng hộ.
Shared pooled 955,03 còn kém flat 899,08, risk 890,04 và rollout 849,72.
True-state equivariance của shared actor PASS; unshared không equivariant.
Post hoc WAIT replay: unshared một số seeds bỏ service hợp lệ 42–48%,
shared 0%. Weak shared seeds 02/03 gán technician có cùng tập máy nhưng
tổng duration dài hơn min trong 536/516 ticks, nên mất throughput ở
reciprocal cases; chưa có can thiệp closed-loop để chứng minh nguyên nhân.

V4/v5 capacity claims thất bại; v6 chứng minh một inductive bias đã biết
có lợi ích tương đối có giới hạn, chưa thành một phương pháp đủ mạnh để
vượt baselines pooled hoặc chứng minh scalability/generalization. Literature
Learning to Dispatch và AirME/HetGPO đã có graph/shared relational policy;
không claim novelty. Goal giữ ACTIVE. Hướng tiếp phải xét RL chọn request
set phối hợp allocator technician duration/opportunity-aware với protocol
fresh seeds và ablation đúng thành phần; không dùng tuning PPO làm trục.

## Mốc v7 sau FULL (2026-10-20)

Proposal/protocol v7 prelocked, code/tests/smoke audit PASS, commit/push
`8ede081` trước FULL. Raw `framework_v7_full_20261020T012531Z`, independent
audit `framework_v7_audit_20261020T020506Z` PASS; xem
`research_result/framework_v7_findings_vi.md`. RL chọn machine subset,
shortest-duration allocator chọn technician trong subset; cùng physics/cost.
Primary shared canonical − shared free = −47,69, hierarchical CI95
[−85,44;−13,73], PASS. Flat canonical − flat free +0,26
[−8,17;+9,00], không có generic gain rõ. Hiệu quả shared đến từ ba
weak train seeds tự chọn 478/510/558 noncanonical assignments; hai strong
seeds chọn 0 và cost hầu như không đổi. H throughput weak seeds tăng,
nhưng reciprocal vẫn kém. Shared canonical pooled 961,27 còn kém risk
906,53 và rollout 861,02; không claim policy tốt nhất hoặc novelty.

Goal ban đầu và mục tiêu sửa đổi được kết luận tại
`research_result/method_goal_conclusion_v7_vi.md`: framework/simulator/baseline/audit
đã có, hai component method có evidence có điều kiện theo claim khóa.
V7 sửa một failure mode nhưng không cạnh tranh trên pooled panel; chưa
kiểm chứng N/K hoặc benchmark ngoài simulator. Không tái sử dụng TEST v7
như confirmation mới. Một hướng nghiên cứu tiếp sẽ cần goal/protocol mới.
