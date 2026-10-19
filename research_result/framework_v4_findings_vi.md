# V4 capacity-opportunity: kết quả FULL và giới hạn của method

Ngày phân tích: 2026-10-20 (Asia/Taipei). Đây là kết quả của protocol đã khóa tại
`research_result/framework_v4_protocol_vi.md`, code commit `7a5075ba4b6c73e83541d84c0ef1d5b36432f139`
đã push trước FULL. Raw run: `research_result/framework_v4_full_20261019T172752Z/`.
Independent audit: `research_result/framework_v4_audit_20261019T183015Z/` (PASS).
Hai phân tích cơ chế **post hoc**, không phải endpoint xác nhận, ở
`research_result/framework_v4_gate_diagnostic_20261019T183156Z/` và
`research_result/framework_v4_queue_diagnostic_20261019T183347Z/`.
Một diagnostic thử ban đầu đã dừng tại numerical tie ~6e-8; thư mục lỗi
`research_result/framework_v4_gate_diagnostic_20261019T183039Z/` được giữ,
script đã sửa để dùng chính deterministic selection path của policy.

## Claim đã khóa và phán quyết

Primary H1: thêm healthy age2 near-request demand vào duration-weighted
capacity opportunity sẽ giảm cost H48+tail16 so cùng actor chỉ xem queued
requests. Bốn case N4/K2, mỗi arm 3 replica scratch RL ×199.680 PPO steps,
4 checkpoint/replica, chọn duy nhất bằng DEV 16 seeds/case. TEST 64 physical
seeds mới/case; cùng case/seed/physics/objective cho mọi controller. Bootstrap
20.000 lần theo cluster physical seed, case-balanced. Tiêu chuẩn: mean<0 và
upper CI95%<0.

**H1 FAIL.** Temporal − current-queue = **+8,520 cost**, CI95%
**[+0,258; +17,086]**. Kết quả thậm chí có hướng bất lợi trên panel đã khóa.
Không đổi gate, seed, budget hoặc chọn lại checkpoint sau khi mở TEST.

| Case | Edge-only | Current-queue | Temporal | Risk rule | Rollout |
|---|---:|---:|---:|---:|---:|
| Interchangeable scale1 | 752,94 | 763,83 | 753,94 | 763,83 | 689,14 |
| Reciprocal scale2 | 834,11 | 820,34 | 829,11 | 829,78 | 727,81 |
| Reciprocal scale3 | 1063,02 | 1020,83 | 1061,70 | 1020,45 | 1057,09 |
| Specialist/generalist scale2 | 992,55 | 994,28 | 988,61 | 997,56 | 987,33 |
| Case-balanced | 910,65 | 899,82 | 908,34 | 902,91 | 865,34 |

Các ô là mean cost trên 64 TEST seeds/case, thấp hơn là tốt hơn. Flat PPO
case-balanced = 918,93; FCFS = 954,50. Case-level CI là descriptive:
temporal − current-queue = −9,89 [−17,30;−3,61] ở interchangeable,
+8,77 [−11,16;+29,50] ở reciprocal2, +40,88 [+19,77;+64,25] ở
reciprocal3, và −5,67 [−19,28;+8,11] ở specialist.

Secondary đã ghi trước: current-queue − edge-only = **−10,832**,
cluster CI95% **[−19,148;−2,660]**. Đây là bằng chứng *descriptive* cho
lợi ích của inductive bias queue trong panel này, không được nâng thành
phán quyết primary sau khi H1 fail; không hiệu chỉnh multiple comparisons.
Nó giúp rõ nhất reciprocal3 (−42,19, case CI [−65,86;−21,33]), nhưng làm
xấu interchangeable (+10,89, case CI [+4,22;+18,83]). Temporal − edge-only
= −2,31, CI [−6,09;+1,14]; temporal − flat = −10,59, CI
[−26,11;+5,41]. Temporal − risk = +5,43, CI [−2,93;+14,11];
temporal − rollout = +43,00, CI [+30,66;+55,82]. Rollout sử dụng
~0,69 s/episode selection trên trung bình bốn case, so ~0,040 s/episode
cho temporal; đây là tradeoff compute, không đủ để gọi RL superior. Rollout
không thắng từng case: ở reciprocal3 nó có mean cost 1057,09, xấu hơn
current-queue 1020,83.

## Mechanism check sau TEST

Trên 12.288 trạng thái đúng theo trajectory của checkpoint temporal đã chọn,
healthy age2 competitor xuất hiện trong 6.088 trạng thái. Tắt chỉ phần
near-request cùng weights/mask làm đổi deterministic action **13 lần**
(3 interchangeable, 1 reciprocal2, 0 reciprocal3, 9 specialist). Logit
có thay đổi nhưng thường không đủ đổi argmax. Với checkpoint current-queue,
tắt toàn bộ queue-opportunity term cùng weights/mask đổi action **83 lần**
(1, 4, 3, 75 tương ứng). Đây chỉ là common-state intervention: không đo
cost của closed-loop ablation và không thể quy toàn bộ paired retraining
effect cho những action flip này. Nó gợi ý nhánh capacity price có thể ảnh
hưởng qua dynamics huấn luyện/representation, còn tác dụng trực tiếp lên
chọn matching ở checkpoint chọn ra là hiếm. Scale learned softplus ≈0,127
ở cả hai checkpoint; không dùng nó như shadow price đã calibration.

State-level actor permutation check: đảo nhãn 4 máy và 2 worker cùng remap
assigned-machine, edge rows/columns, catalogue/mask trên 12.288 trạng thái
cho mỗi checkpoint. Max absolute logit error ≤4,77e-7 (temporal) và
≤1,20e-7 (current-queue), zero non-tie deterministic choice mismatch.
955/943 trạng thái tương ứng có near tie nên không đòi argmax trùng chính
xác ở các trạng thái đó. Điều này xác nhận actor equivariance tại N4/K2;
critic, training và transfer N/K khác chưa được xác nhận.

## Correctness và giới hạn suy luận

Independent verifier xác nhận commit/source/checkpoint hashes, fresh split,
full DEV/TEST grids, tái tạo độc lập paired case/physical training tapes,
DEV-only selection, episode economics, request conservation và mọi bootstrap
interval. Campaign self-audit PASS: zero invalid assignment, checkpoint reload,
identical structured initial weights theo replica. Tổng 2.396.160 PPO steps,
3.072 DEV và 1.792 TEST episodes; 48 checkpoint. Training wall cộng theo arm:
flat 504 s, edge-only 584 s, current-queue 665 s, temporal 696 s; policy
structured 9.632 parameters, flat 17.046. Chênh thời gian có thể do runtime
và cấu trúc forward, không diễn giải là compute scaling.

Điều đã biết: scratch RL policy với shared edge scores và queue opportunity
có descriptive improvement trên edge-only, nhất là khi duration reciprocal
khắc nghiệt. Điều chưa đạt: primary temporal anticipation không cải thiện,
queue-price online effect yếu, phương pháp chưa thắng bounded rollout theo
case-balanced cost, chưa có N/K scalability hoặc validation ngoài synthetic
request-lifecycle. Feasibility do legal mask, không phải năng lực học của
actor. Không tuyên bố novelty: shared pair scoring, competition, online
matching và idle/maintenance timing có prior art trong method proposal và
`research_result/method_v4_historical_context_vi.md`.

## Tái lập

Dùng cùng commit và môi trường `uv`:

```bash
uv run pytest -q tests/test_maintenance_framework_v4_policy.py
uv run python -m ht_pdm_fjsp.maintenance_framework_v4_campaign --profile full --device cpu --output-dir "research_result/framework_v4_full_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python research_result/verify_framework_v4.py --run-dir research_result/framework_v4_full_20261019T172752Z --output-dir "research_result/framework_v4_audit_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python research_result/diagnose_framework_v4_gate.py --run-dir research_result/framework_v4_full_20261019T172752Z --output-dir "research_result/framework_v4_gate_diagnostic_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python research_result/diagnose_framework_v4_gate.py --run-dir research_result/framework_v4_full_20261019T172752Z --output-dir "research_result/framework_v4_queue_diagnostic_$(date -u +%Y%m%dT%H%M%SZ)" --selected-arm current_queue --ablated-arm edge_only
```

Các thư mục raw/checkpoint và log lớn được giữ tại lab, nằm ngoài Git. Vì
seed TEST đã mở, rerun cùng seed là replication kỹ thuật chứ không phải
confirmation mới. Một vòng method tiếp theo cần proposal/protocol/seeds
mới trước huấn luyện, và phải giải quyết trực tiếp việc additive price ít
đổi action mà không biến PPO tuning thành trục nghiên cứu.
