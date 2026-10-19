# V5 post-decision capacity: FULL verdict và cơ chế

Ngày phân tích 2026-10-20 (Asia/Taipei). Protocol và proposal đã khóa tại
commit `83dfc8c23f78e735099c5264c8d0e3ba5cc2e5bc` rồi push trước FULL.
Raw `research_result/framework_v5_full_20261019T184108Z/`; independent audit
`research_result/framework_v5_audit_20261019T192957Z/` **PASS**. Diagnostic
post hoc `research_result/framework_v5_diagnostic_20261019T193005Z/` hoàn
thành. Smoke đã audit PASS ở `framework_v5_smoke_20261019T183950Z/` và
`framework_v5_smoke_audit_20261019T184012Z/`; 50 simulator/framework tests
PASS trước FULL. Không sửa source/physics/cost/seed/budget/selection sau TEST.

## Câu hỏi đã khóa

V4 cho thấy additive opportunity term ít đổi action và age2 gate không cải
thiện cost. V5 hỏi liệu score của *toàn matching* dựa trên earliest-start
delay của queued requests còn lại có giảm H48+tail16 cost so additive queue
price hay không. Mỗi arm `flat`, `edge_only`, `additive_queue`,
`postdecision` train RL từ đầu 3 replicas ×199.680 PPO steps; 48 checkpoint.
Structured arms có cùng initial parameters và paired 4-case/physical training
tape theo replica; `additive_queue` khớp v4 queue formula đến roundoff. DEV
4 cases×16 fresh seeds chọn 1 checkpoint/arm trong 12 candidates. Sau khi
ghi selection SHA mới mở TEST 4 cases×64 fresh paired physical seeds ×7
controllers (4 learned+FCFS/risk/rollout). Primary cluster bootstrap 20.000
draws theo 64 physical seeds, cân đều bốn case; confirm mean<0 và upper CI95%<0.

**Primary FAIL:** postdecision − additive_queue = **+1,547 cost**,
cluster CI95% **[−1,047; +4,109]**, 22 seed-cluster wins, 11 ties, 31
losses. CI chứa zero; không chứng minh postdecision tốt hoặc xấu hơn
additive_queue ở pooled panel. Không tăng penalty scale, thêm budget hay
chọn case thuận lợi sau kết quả.

| Case | Flat | Edge-only | Additive queue | Postdecision | Risk | Rollout |
|---|---:|---:|---:|---:|---:|---:|
| Interchangeable scale1 | 804,39 | 754,61 | 750,70 | 753,70 | 759,69 | 688,25 |
| Reciprocal scale2 | 830,38 | 793,80 | 794,42 | 793,83 | 810,48 | 714,86 |
| Reciprocal scale3 | 1035,80 | 1026,03 | 1024,66 | 1024,56 | 1036,52 | 1055,64 |
| Specialist/generalist scale2 | 1022,38 | 994,19 | 990,02 | 993,89 | 993,95 | 978,14 |
| Case-balanced | 923,23 | 892,16 | 889,95 | 891,50 | 900,16 | 859,22 |

Mean H+tail cost; thấp hơn tốt hơn. FCFS case-balanced = 941,04.
Postdecision − additive_queue case-level descriptive CI: interchangeable
+3,00 [+0,75;+5,50], reciprocal2 −0,59 [−2,00;+0,56], reciprocal3 −0,09
[−2,28;+1,56], specialist +3,88 [−5,67;+13,28]. Secondary đã khóa nhưng
không multiple-comparison adjusted: postdecision − edge-only −0,66
[−2,20;+0,85]; postdecision − flat −31,74 [−43,30;−19,79];
postdecision − risk −8,66 [−13,20;−4,16]; postdecision − rollout +32,27
[+22,07;+42,75]. Structured family của v5 rõ ràng tốt hơn flat trên panel
này, nhưng incremental residual-capacity feature **không** tạo lợi ích so
comparator method. Bounded rollout còn tốt hơn case-balanced, dùng trung
bình 0,685 s selection/episode so 0,046 s của postdecision; rollout thua
postdecision ở reciprocal3 (1055,64 vs 1024,56). Tất cả là cùng physical
seeds, physics, objective và current-information contract, nhưng compute
khác nhau.

`additive_queue − edge_only` secondary = −2,21, CI [−4,95;+0,52]. V4 từng
có descriptive −10,83 [−19,15;−2,66] trên panel khác. Lợi ích additive
queue **không lặp lại rõ** ở v5; không gộp hai panel để chọn interpretation
thuận lợi. Exploratory edge-only − flat ở v5 = −31,08, CI [−42,59;−19,39],
nhưng ở v4 = −8,27, CI [−23,93;+8,04]. Vì hai panel chọn checkpoints độc
lập, đây là dấu hiệu hiệu ứng shared edge representation có thể nhạy với
train/selection; không là confirmation đã khóa về component đó.

## Mechanism post hoc và giới hạn

Trên 12.288 state của trajectory selected postdecision checkpoint, 9.368
state có queued request, nhưng chỉ 1.724 state có ít nhất một legal action
với residual penalty dương. Tắt đúng penalty term cùng weights/mask đổi
**101 action** (4 reciprocal2, 20 interchangeable, 2 reciprocal3,
75 specialist). Trace đã replay chính xác cost từng episode trong raw TEST;
intervention chỉ là common-state, **không** đo closed-loop ablated cost.
Penalty dương cho action policy chọn ở 1.578/12.288 state; tuy vậy action
margin thường lớn hơn penalty. Tác dụng trực tiếp lên quyết định hiếm,
phù hợp với primary inconclusive; không chứng minh nguyên nhân duy nhất.
State-level permutation: remap bốn machine, hai worker, busy assigned index,
edge features, action catalogue và mask trên 12.288 state. Max logit error
2,38e−7, zero unambiguous deterministic mismatch; 1.023 near-tie states
được ghi riêng. Chỉ xác nhận actor N4/K2 equivariance, không chứng minh
critic/training invariant hoặc N/K scalability.

Campaign và independent verifier xác nhận 2.396.160 PPO steps, 3.072 DEV,
1.792 TEST, hash source/artifact/checkpoint, unique seed splits, training
tapes tái sinh độc lập, identical structured starts, DEV-only selection,
full grids, reward=−cost, cost components, request conservation, zero invalid
assignment và các bootstrap intervals. Feasibility từ mask. Training wall
cộng theo arm: flat 505 s, edge-only 596 s, additive 677 s, postdecision
764 s; policy 17.046 parameters flat vs 9.632 structured. Latency/compute
đo single-process CPU, không ngoại suy deployment.

## Kết luận method

V5 không xác nhận giả thuyết rằng exact current-queue residual earliest-start
feature cải thiện additive capacity price. V4 cũng đã bác bỏ healthy age2
anticipation. Kết quả tích cực có giới hạn là shared pair/matching structured
family so flat/risk trong v5, nhưng v4 cho thấy độ nhạy panel; chưa cô lập
được lợi ích của weight sharing, action factorization, permutation bias và
checkpoint selection. Không gọi capacity-opportunity hay postdecision là
novel: `research_result/method_v5_literature_context_vi.md` ghi prior art
AirME/HetGPO, online reusable matching và opportunity-cost pricing sau khi
v5 đã khóa. Setting vẫn synthetic N4/K2, exogenous PM threshold, chưa có
production jobs, service-time uncertainty hay validation ngoài simulator.

Một follow-up chỉ hợp lý nếu nó kiểm tra trực tiếp giá trị của *shared pair
representation* với baseline factorized nhưng không chia sẻ weights, flat
MLP và permutation control trên seeds mới; cần đăng ký claim, seed,
train-selection variance và stopping rule trước implementation. Không theo
đuổi việc tăng PPO budget/penalty scale để cứu v5.

## Tái lập

```bash
uv run pytest -q tests/test_maintenance_framework_v5_policy.py
uv run python -m ht_pdm_fjsp.maintenance_framework_v5_campaign --profile full --device cpu --output-dir "research_result/framework_v5_full_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python research_result/verify_framework_v5.py --run-dir research_result/framework_v5_full_20261019T184108Z --output-dir "research_result/framework_v5_audit_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python research_result/diagnose_framework_v5.py --run-dir research_result/framework_v5_full_20261019T184108Z --output-dir "research_result/framework_v5_diagnostic_$(date -u +%Y%m%dT%H%M%SZ)"
```

Raw/checkpoint/log lớn giữ trên lab ngoài Git. Reuse TEST 66200000–63 chỉ là
technical replication, không phải confirmation mới.
