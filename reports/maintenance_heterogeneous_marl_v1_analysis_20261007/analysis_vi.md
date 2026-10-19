## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent.
- Origin Mode: validate; ARS Schema 9.
- Origin Date: 2026-10-07 (Asia/Taipei).
- Verification Status: ANALYZED. Kiểm tra hash, số liệu, physics và checkpoint; không tái huấn luyện.
- Version Label: maintenance_heterogeneous_marl_v1_analysis_v1.
- Overall Confidence: CAUTION đối với suy luận tổng quát về MARL; kết quả primary số liệu được kiểm tra độc lập.
- Source: [maintenance_heterogeneous_marl_v1_full_cpu_20261006T150059Z](/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_heterogeneous_marl_v1_full_cpu_20261006T150059Z/manifest.json), commit `693c348fdfee0d327f0ec8e7680af1f2a76645e7`.

## Kết quả chính

Heterogeneity có tạo tranh chấp cho specialist, nhưng trong cấu hình/ngân sách này,
**học allocation đầy đủ không cải thiện so với allocator rule**. Nhánh chọn máy
và giữ allocation theo rule đạt cost175.8868, nhánh học matching176.2994;
full matching cao hơn0.2346%, chỉ thắng3/10 training replicas. CI97.5% của
chênh lệch chi phí chứa0, và không hỗ trợ mức giảm3% đã đặt trước.

Nhánh MARL hiện tại có cost188.6912, cao hơn centralized matching7.03%, thắng0/10
replicas và làm nominal tăng18.80%. Hai confirmatory hypothesis đều FAIL.
Tuy nhiên, **có hạn chế cấu trúc trong policy MARL đã triển khai**: technician
đồng năng lực có đặc trưng/logit bằng nhau nhưng mọi machine agent dùng cùng
quy tắc argmax phá hòa. Nhiều agent đồng loạt chọn cùng technician; coordinator
chỉ nhận người thắng và để các máy khác chờ dù technician khác rảnh. Đây là
vấn đề của phương pháp được thử; không chứng minh MARL nói chung không phù hợp.

Hai PPO tập trung giảm cost3.00–3.23% so với rule, thắng10/10 replicas với CI
chênh lệch thấp hơn0. Hiệu quả này ổn định trong panel đã giữ cố định, nhưng
**không đạt ngưỡng economic5%**, và base cost cũng chưa đạt3%. Không thay gate
sau khi nhìn kết quả. So sánh với rule không cô lập thuần giá trị học ưu tiên
máy, vì allocator của rule tham chiếu khác allocator cố định của nhánh PPO.

![Primary cost và nominal guard](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/primary_cost_and_nominal.png)

## Integrity và phạm vi kiểm tra

Run COMPLETED, Git clean, đúng commit; thời gian13 giờ06 phút32 giây,218 artifacts,
3.136GB. Hash/size của toàn bộ artifacts khớp;20 source snapshot files khớp Git
commit. Đúng30 models,14,745,600 training steps,1,228,800 training episodes,
307,200 optimizer steps,63,200 dev episodes,144,000 test episodes,
2,592,000 physical test intervals và576,000 machine rows.

Seed panels disjoint với registry; kiểm tra toàn bộ counter-derived training
configuration/shock seeds, budgets và hashes chuỗi cấu hình ghép cặp giữa ba
nhánh.1,470 configuration hashes được tái tính từ generator. Ba nhánh có
72,258 tham số và cùng initial weights theo từng replica. Mọi checkpoint được
load; reference/checkpoint selection được tái tính từ dev. Tất cả10 MARL
replicas DEVELOPMENT_INFEASIBLE do nominal guard, vẫn giữ final checkpoints
theo protocol; không loại các seed xấu.

Đã phát lại toàn bộ2,592,000 physical transitions với shock stream gốc:
state/action/feasibility/following/physical cost và episode totals đều khớp;
machine cost/availability được reconciliation trên576,000 rows.2,160,000 learned
actions được kiểm tra với selected checkpoint. Primary specialized/H12 có
**0 action mismatch**; skill-mask cũng không có mismatch.

Ngoài primary có532/2,160,000 batch-action mismatches (0.0246%) ở homogeneous/
nominal, trên517 episodes. Scalar replay trên logged state khớp242/532; sai
khác log probability tối đa khoảng1.91e-6, batch probability khoảng3.81e-6.
Đây là độ nhạy của lựa chọn gần hòa đối với batch/platform CPU. Phát lại riêng
517 episodes trên Mac làm115 episode costs đổi, tối đa76 đơn vị/episode;
chỉ MARL nominal có đổi tổng cost: mean H12+0.026,H24+0.120 trên5000 episodes
mỗi panel. Không thay số liệu Linux gốc bằng số liệu này. Primary không bị ảnh
hưởng; nominal vẫn vượt guard rất xa. Không ghi toàn bộ action replay là PASS
hoặc VERIFIED: audit lưu `NUMERICAL_ACTION_MISMATCH`; training chưa tái lập.

## Primary N4/K2/H12

Mỗi reference500 cohort/shock episodes; mỗi learned arm10 replicas x500 episodes.
Priced cost = base cost + overdue waiting cost.

| Controller | Priced cost | Base cost | Failures/episode | Waiting violation |
|---|---:|---:|---:|---:|
| Rule chọn từ dev | 181.7520 | 181.1520 | 2.4680 | 5.00% |
| PPO học toàn matching | 176.2994 | 176.2178 | 2.8852 | 0.68% |
| PPO chọn máy + allocation rule | 175.8868 | 175.8124 | 2.8050 | 0.62% |
| MARL machine agents | 188.6912 | 187.8944 | 3.6442 | 4.14% |
| Joint rollout | 174.5420 | 174.5420 | 2.2500 | 0.00% |
| Fixed-allocation rollout | 175.9020 | 175.9020 | 2.3740 | 0.00% |

| Paired contrast | Mean cost difference | CI | Level | Wins | Two-sided t p | Protocol verdict |
|---|---:|---|---:|---:|---:|---|
| central_matching_ppo − central_fixed_allocation_ppo | +0.4126 | [-0.1548, +0.9800] | 97.5% | 3/10 | 0.0826252 | FAIL |
| machine_cooperative_ppo − central_matching_ppo | +12.3918 | [+11.6155, +13.1681] | 97.5% | 0/10 | 1.02232e-11 | FAIL |
| central_matching_ppo − adaptive_isolated_matching | -5.4526 | [-5.8583, -5.0469] | 95% | 10/10 | 2.20674e-10 | FAIL |
| central_fixed_allocation_ppo − adaptive_isolated_matching | -5.8652 | [-6.3610, -5.3694] | 95% | 10/10 | 6.87901e-10 | FAIL |
| machine_cooperative_ppo − adaptive_isolated_matching | +6.9392 | [+6.5344, +7.3440] | 95% | 0/10 | 2.50449e-11 | FAIL |

Đơn vị inference là10 training replicas ghép cặp, conditional trên cùng finite
cohort/shock panel; không coi5000 episodes là5000 mẫu huấn luyện độc lập. Hai
primary contrasts dùng CI97.5% (Bonferroni family2); ba economic contrasts là
secondary/descriptive CI95%. n10 cần giả định paired differences gần chuẩn.
Sensitivity exact sign-flip1024 patterns cho assignment p0.087890625, MARL-vs-
central p0.001953125; kết luận về chiều chênh lệch không thay đổi. Sign-flip cũng
cần symmetry/exchangeability; không phải đảm bảo distribution-free cho mọi mô hình.

Fixed PPO tốt hơn selected rule ở cả10 cohort means, nhưng mức giảm chỉ0.39–
5.44%; không tuyên bố từng shock/máy đều được lợi. Full vs fixed tốt hơn ở1/10
cohort means; MARL tệ hơn full ở10/10 cohort means.

## Conflict đã tăng, nhưng allocation linh hoạt thêm ít giá trị

So sánh cùng adaptive-isolated reference, cùng cohort/shock seed: preferred-worker
excess tăng0.57183 →0.73467 (~28.5%) từ homogeneousK2 sang specializedK2.
Nhưng cost của chính reference này giảm210.606 →181.752 (~13.7%) và utilization
0.856 →0.69467. Row mean service time/restoration giữ cố định không giữ được
effective throughput; specialization vừa tạo tranh chấp vừa cho phép phục vụ
nhanh/tốt hơn khi chọn technician thích hợp. Không suy ra bài toán tổng thể
khó hơn chỉ từ conflict count. Hai rollout comparators specialized cùng có cost
thấp hơn homogeneous, consistent với sự khác biệt này.

Mỗi máy có technician ưu tiên vừa nhanh hơn vừa phục hồi tốt hơn; do đó nhiều
lựa chọn assignment có chất lượng khá rõ. Khi biết tập máy cần phục vụ, rule
min(sum duration/restoration) đã mạnh trong cấu hình này. Đó là cách giải thích
phù hợp với dữ liệu, không phải proof về tối ưu hoặc mọi chế độ heterogeneity.

Full-vs-fixed allocation rollout: H12 specialized174.542 vs175.902, giảm0.773%;
H24 full360.358 vs fixed360.284, full hơi cao hơn0.021%. Skill-mask H12 thêm
allocation chỉ giảm0.125%; H24 full hơi cao hơn0.084%. Homogeneous/nominal hai
planner costs bằng nhau, đúng invariant. Đây là32-scenario rollout với cùng
continuation rule, không phải oracle/upper bound của giá trị allocation tối ưu.

Diagnostic tại **cùng logged full-planner state và forecast scenarios**, primary:
3521/6000 intervals có nhiều allocation khả thi;310/6000 (5.17%) full planner
chọn assignment ngoài rule cố định cho cùng tập máy; chỉ222/6000 (3.70%) có
forecast-cost advantage dương khi mở rộng root action set.94.83% root choices
đã trùng rule allocator. Ngoài root, continuation vẫn cùng rule; không xem số
này là chứng minh dynamic allocation hoàn toàn không có giá trị.

## Hạn chế cấu trúc của nhánh MARL

Trong [policy head](/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_heterogeneous_marl_v1_full_cpu_20261006T150059Z/source_snapshot/maintenance_heterogeneous_policy.py:142),
worker option gồm embedding của technician/edge; positive action indicator giống
nhau cho mọi technician. Với các technician đồng năng lực và rảnh, worker input
bằng nhau. Public machine rank giúp phân biệt máy, nhưng không phân biệt worker
options của một máy. [Greedy evaluation](/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_heterogeneous_marl_v1_full_cpu_20261006T150059Z/source_snapshot/maintenance_heterogeneous_policy.py:224)
chọn index đầu tiên khi hòa. [Coordinator](/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_heterogeneous_marl_v1_full_cpu_20261006T150059Z/source_snapshot/maintenance_heterogeneous_model.py:302)
loại proposals va chạm, không tự phân các máy bị loại sang worker khác.

Diagnostic state:4 machines age2, tất cả hoạt động,4 technicians rảnh, nominal.
Cả10 loaded MARL checkpoints có per-machine free-worker logit spread đúng0,
raw proposals `[1,1,1,1]`, matching chỉ `[(0,0)]`; cả10 central checkpoints nhận
được4 jobs. Đây là posthoc structural diagnostic trên một trạng thái xác định.
Action-space có matching không xung đột không đồng nghĩa deterministic policy
head biểu diễn được cách ghép ấy; số tham số/information match không bảo đảm
khả năng biểu diễn cùng mức.

Trên180,000 actual nominal MARL intervals H12/H24,179,914 intervals có >=2 worker
rảnh. Chỉ150/180000 intervals (0.0833%) bắt đầu nhiều hơn1 service; những ngoại
lệ hiếm phù hợp với numerical tie sensitivity, không phủ nhận thiếu cơ chế phá
hòa ổn định.212,960 nominations,63,887 bị loại (~30.00% pooled proposals) và
mọi rejection này xảy ra khi vẫn còn idle capacity. Mean episode rejection
rate nominal/H12 là31.19% (mẫu số khác pooled rate).

Nominal/H12: rule và centralized full cost168.048,0 failures; MARL199.6468,
2.5256 failures/episode, +18.80% base cost, vượt110% guard. Primary MARL proposal
rejection rate trung bình27.70%; failures3.6442 vs central2.8852. Đây là bằng
chứng implementation/decomposition hiện tại xử lý coordination kém. Chưa cô
lập bao nhiêu trong primary gap7.03% được gây bởi nominal symmetry issue; không
được gán toàn bộ gap đó cho một cơ chế từ diagnostic này.

Training dùng stochastic multinomial, các agent có thể rút worker khác nhau dù
logit hòa; greedy evaluation làm các đề xuất hội tụ về cùng worker. Vì vậy cần
kiểm tra cả policy representation, tie handling và cơ chế coordinated assignment
khi so sánh lại; không diễn giải đơn thuần là budget RL chưa đủ hoặc thiếu conflict.

## Breakdown và sensitivity

Fixed PPO vs selected rule primary: failure cost tăng5.055, unavailability cost
 giảm11.652, maintenance tăng1.2574, overdue giảm0.5256 → total giảm5.8652.
Failures tăng2.468 →2.805 (+13.65%) dù cost giảm; đây là cost trade-off được
objective cho phép, không phải cải thiện mọi chỉ tiêu. Central full cũng tăng
failures so với rule. Không báo là RL làm ít hỏng hơn trong panel này.

| Condition | H | Central full | Central fixed | MARL | MARL wait violation |
|---|---:|---:|---:|---:|---:|
| homogeneous | 12 | 211.273 | 221.378 | 217.760 | 2.08% |
| homogeneous | 24 | 413.877 | 424.067 | 432.493 | 6.08% |
| specialized | 12 | 176.299 | 175.887 | 188.691 | 4.14% |
| specialized | 24 | 355.912 | 353.912 | 380.338 | 16.18% |
| nominal | 12 | 168.048 | 168.051 | 199.647 | 0.00% |
| nominal | 24 | 345.916 | 345.921 | 391.175 | 0.00% |
| skill_mask | 12 | 177.690 | 177.199 | 192.424 | 14.00% |
| skill_mask | 24 | 358.787 | 357.244 | 407.234 | 33.38% |

HomogeneousK2, H24 và skill-mask là sensitivities chưa có trong training mixture.
Chẳng hạn MARL skill-mask/H24 waiting violation33.38%, specialized/H24 16.18%.
Full hơn fixed trên homogeneousK2 không chứng minh học allocation có ích: ở đây
worker interchangeability làm worker-label allocation không có giá trị vật lý;
khác biệt có thể do training/action mask/inductive bias và transfer. Không đổi
primary sang panel thuận lợi. Skill mask làm thay đổi feasible capacity nên giữ
riêng khỏi effect all-compatible heterogeneity. Không so trực tiếp với raw cost
của experiment imitation trước: generator, models và seed panels đã thay đổi.

## Fallacy scan — 11/11 checked

| Pattern | Assessment | Detail |
|---|---|---|
| Simpson's paradox | Checked | Phân tích riêng condition/H/cohort; không pool sensitivity vào primary. |
| Ecological fallacy | CAUTION | Means/replica CI không bảo đảm mọi máy hoặc shock có lợi. |
| Berkson/selection | CAUTION | Synthetic finite128 specialized profiles; không suy rộng sang thực tế/unseen profile. |
| Collider bias | Checked | Dev-only checkpoint guard; giữ cả10 MARL infeasible replicas; không condition trên test success. |
| Base-rate neglect | Checked | Báo waiting baseline5% vs fixed0.62%, không chỉ báo relative reduction. |
| Regression to mean | Checked | Same-panel paired controls, không lấy khác biệt với experiment trước làm causal gain. |
| Survivorship | Checked | Đủ30models/all test grid; giữ failed economic/development gates. |
| Look-elsewhere | CAUTION | Chỉ2 primary Bonferroni; mọi sensitivity/diagnostic là descriptive posthoc. |
| Forking paths | Checked | Protocol/commit hashes trước full run; gates không thay; diagnostic bổ sung được gắn nhãn. |
| Correlation/causation | CAUTION | Conflict count/collision không chứng minh MARL necessity; structural diagnosis không cô lập toàn primary gap. |
| Reverse causality | Checked | Synthetic treatment trước trajectories; policy cũng thay đổi states/conflict đo được, không coi conflict là predictor ngoại sinh. |

## Kết luận nghiên cứu trong phạm vi run

Có lợi ích cost nhỏ nhưng ổn định của tested centralized policies so với selected
rule. Chưa đạt economic impact threshold5%; học toàn allocation không tốt hơn
fixed allocator. Heterogeneity này chưa mở ra allocation advantage lớn quan sát
được. Không dùng thất bại của nhánh MARL hiện tại để chốt MARL không cần thiết:
đã phát hiện hạn chế biểu diễn/phá hòa ở chính phương pháp.

Ưu tiên kỹ thuật tiếp theo là sửa worker symmetry/tie handling và phối hợp
proposals sao cho có thể tạo simultaneous matching khi nhiều technicians rảnh,
rồi mới dùng một so sánh theo protocol mới để đánh giá phương pháp MARL đó.
Đây là đề xuất từ phân tích; turn này không sửa experiment code hoặc launch run.

## Files và reproducibility

- [Integrity](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/integrity_audit.json), [training audit](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/training_audit.json), [gate audit](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/gate_audit.json).
- [Independent contrasts/CIs/p-values](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/independent_statistics.json), [cohort costs](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/cohort_costs.json).
- [Full physics/policy audit](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/physical_policy_audit.json), [numerical sensitivity](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/numerical_replay_diagnostic.json).
- [Worker symmetry diagnosis](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/worker_symmetry_diagnostic.json), [same-state planner root diagnostic](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/planner_root_allocation_diagnostic.json).
- [PNG](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/primary_cost_and_nominal.png), [PDF](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/primary_cost_and_nominal.pdf).
- [Audit script](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/audit_and_analyze.py), [planner diagnostic script](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/planner_allocation_diagnostic.py), [numerical replay script](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_heterogeneous_marl_v1_analysis_20261007/numerical_replay_diagnostic.py).

Raw artifacts được giữ nguyên. Report status ANALYZED; không tái huấn luyện nên
không xác nhận training reproducibility. Cross-platform greedy action sensitivity
được giữ nguyên trong audit, không che bằng một nhãn PASS chung.
