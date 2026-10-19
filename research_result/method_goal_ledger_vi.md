# Ledger mục tiêu method điều phối RL

**Trạng thái: ACTIVE (v4 primary FAIL, independent audit PASS).** Mục tiêu sửa đổi yêu cầu bằng chứng về giá trị của
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
