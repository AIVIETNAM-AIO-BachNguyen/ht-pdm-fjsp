# Phân tích maintenance contention imitation v1

## Material Passport

Schema: ARS 9. Material: validation report. Mode: validate. Origin: dữ liệu lab đã chuyển về Mac.
Verification Status: **ANALYZED** — audit artifact và replay; không chạy lại toàn bộ training.
Run: `maintenance_contention_imitation_v1_full_cpu_20261006T063713Z`. Commit: `de85e82e5aa2b5e9a1def742b325b12316101cda`.
Thời gian lab: 2026-10-06T06:37:15.634342+00:00 → 2026-10-06T08:24:55.260318+00:00 (khoảng 107 phút 40 giây).

## Kết luận chính

**Cả ba policy đạt gate kinh tế ở K2/H12; imitation → PPO tốt nhất.** Hybrid giảm cost 7,25% so với baseline và thêm 1,62% so với PPO trực tiếp, thắng 10/10 replica. Tuy nhiên, primary hypothesis yêu cầu ít nhất 2% cải thiện thêm, nên verdict confirmatory vẫn là **FAIL**. Đây là benefit có hướng ổn định nhưng chưa đạt effect size đã khóa; không phải bằng chứng “imitation không có tác dụng”.

Policy tập trung đã học được quyết định điều phối có ích. Kết quả chưa chứng minh MARL cần thiết, và chưa đảm bảo tốt trên mọi workload/capacity.

## Primary và gate kinh tế

| Phương pháp | Priced cost | Base cost | Giảm priced vs ref | Giảm base vs ref | Capture savings của planner | Gate kinh tế |
|---|---:|---:|---:|---:|---:|---|
| Direct PPO | 185.0012 | 185.0012 | 5.72% | 5.51% | 65.65% | PASS |
| Planner imitation | 184.3486 | 184.2646 | 6.05% | 5.89% | 69.47% | PASS |
| Imitation + PPO | 181.9962 | 181.9314 | 7.25% | 7.08% | 83.23% | PASS |
| Reference independent DP | 196.2220 | 195.7900 | — | — | — | — |
| Joint rollout | 179.1300 | 179.0580 | 8.71% | 8.55% | 100% | diagnostic |

Planner là model-based controller khả thi, không phải optimum N4. Capture savings là tỷ lệ so với lợi ích quan sát của controller này, không phải tỷ lệ gần tối ưu.

![Primary cost](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_contention_imitation_v1_analysis_20261006/primary_cost.png>)

Primary hybrid − direct: Δcost **-3.005**, t95% CI **[-3.866, -2.144]**, **10/10 wins**. Reduction **1.62%**, dưới ngưỡng 2%. Nominal và waiting guards đạt. Tất cả seed được giữ lại; không có development-infeasible model.

Mỗi CI dùng 10 paired training-replica means trên cùng 500 episode primary. CI có điều kiện trên panel này; không coi 5.000 episode là 5.000 replica độc lập. Với n=10, t-CI giả định phân phối replica differences đủ phù hợp; sign-only sensitivity hai phía cho 10/10 cùng hướng có p=0,001953, chỉ hỗ trợ hướng benefit, không chứng minh đạt ngưỡng 2%.

Các contrast sau là exploratory, không hiệu chỉnh nhiều so sánh:

| Contrast | Δpriced cost | t95% CI | Wins | Giảm tương đối |
|---|---:|---:|---:|---:|
| planner_imitation_vs_direct_ppo | -0.6526 | [-2.0796, 0.7744] | 6/10 | 0.35% |
| imitation_ppo_vs_planner_imitation | -2.3524 | [-3.2799, -1.4249] | 10/10 | 1.28% |

Imitation đơn thuần chưa hơn PPO trực tiếp rõ ràng: CI chứa 0 và chỉ thắng 6/10. Fine-tuning từ imitation thắng imitation trên 10/10 seed, nhưng có thêm thông tin/compute; không diễn giải thành so sánh hiệu quả compute công bằng.

## Policy đã học gì?

Direct development K2: mean cost khởi tạo 363,89 → checkpoint cuối 180,71. Imitation: cùng khởi tạo → cuối 180,12. Hybrid khởi tạo từ selected BC với cost 179,93 → cuối 178,28. Không lấy các giá trị development để thay thế test result. Learning curve vẫn cải thiện muộn; chưa chứng minh hội tụ hay ngân sách đủ.

BC validation trên các trạng thái có hơn một subset khả thi: agreement **22,11% → 77,85%**, teacher root-score regret **9,684 → 1,325**. Train agreement cuối 78,34%, gần validation. Mức học này cho thấy encoder/action space có thể biểu diễn các quyết định hữu ích; chưa chứng minh biểu diễn hoàn hảo hay loại trừ lỗi optimization/coverage. Overall agreement cuối 80,08% gồm cả forced STOP; luôn báo riêng choice states.

Teacher noise: independent 128-forecast probe đổi quyết định ở **10.94%** trong 320 probe; trong 280 choice states là **12.50%**. Root regret của nhãn 32-forecast dưới probe mean 0.149, p95 0.829, max 7.750. 128 forecast vẫn là ước lượng Monte Carlo, không phải ground truth; 320 probe cố định chỉ mô tả phần validation được kiểm tra.

![Learning diagnostics](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_contention_imitation_v1_analysis_20261006/learning_curve.png>)

## Cơ chế cost ở K2/H12

| Controller | Maintenance | Failure | Downtime | Overdue | Failures/episode | Waiting violation | Utilization |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference | 13.7560 | 42.2100 | 139.8240 | 0.4320 | 2.8140 | 3.40% | 79.85% |
| Direct PPO | 15.8558 | 28.7070 | 140.4384 | 0.0000 | 1.9138 | 0.00% | 94.36% |
| Planner imitation | 14.9278 | 28.0380 | 141.2988 | 0.0840 | 1.8692 | 0.70% | 91.98% |
| Imitation + PPO | 15.2382 | 24.9900 | 141.7032 | 0.0648 | 1.6660 | 0.54% | 93.25% |
| Planner | 14.8560 | 23.5500 | 140.6520 | 0.0720 | 1.5700 | 0.60% | 90.77% |

Hybrid giảm failures từ 2,814 xuống 1,666 (khoảng 40,8%). Tiết kiệm chủ yếu đến từ failure cost: 42,21 → 24,99. Maintenance và downtime tăng nhẹ; vì vậy kết luận đúng là giảm **tổng cost bằng phòng tránh failure**, không phải giảm mọi thành phần hoặc downtime. Hybrid so với direct cũng đánh đổi waiting violation 0% → 0,54%, vẫn dưới guard reference+2pp. Utilization cao không tự chứng minh lịch tối ưu.

![Cost components](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_contention_imitation_v1_analysis_20261006/cost_mechanism.png>)

## Heterogeneity và ngoài phân phối training

Mean hybrid tốt hơn direct trên cả 10 cohort, nhưng cả ba learner đều thua reference ở cohort 147002. Profile này có durations [2,1,3,3], restoration [0,75;1;0,75;1]. PPO đã thấy profile vật lý này trong training của mọi replica; không thể chỉ quy lỗi cho một parameter combination chưa từng thấy. Trong teacher training data, chỉ seed 144007 có đúng 1 trajectory của profile này (các seed còn lại không có); đó là hạn chế coverage của BC, chưa đủ để giải thích duy nhất lỗi của cả ba policy.

| Cohort | Reference | Planner | Direct PPO | Imitation | Hybrid |
|---|---:|---:|---:|---:|---:|
| 147000 | 170.640 | 150.380 | 155.990 | 152.562 | 149.068 |
| 147001 | 218.880 | 206.000 | 211.538 | 209.572 | 208.136 |
| 147002 | 189.280 | 178.300 | 192.648 | 192.078 | 191.560 |
| 147003 | 193.600 | 176.220 | 172.764 | 171.918 | 170.072 |
| 147004 | 195.320 | 176.480 | 174.900 | 175.252 | 174.566 |
| 147005 | 238.260 | 217.000 | 228.138 | 223.222 | 219.126 |
| 147006 | 204.320 | 191.220 | 197.546 | 196.406 | 195.890 |
| 147007 | 171.040 | 155.400 | 161.104 | 161.138 | 159.592 |
| 147008 | 181.020 | 167.120 | 173.626 | 176.948 | 172.696 |
| 147009 | 199.860 | 173.180 | 181.758 | 184.390 | 179.256 |

Cohort 147002: hybrid191,56 > reference189,28, trong khi planner178,30. Vẫn còn lợi ích chưa được học trên profile này. Để xác định vì sao cần kiểm tra state/action errors theo profile; dữ liệu aggregate chưa tách duy nhất approximation, coverage, optimization hoặc teacher visitation shift.

| Test condition | Direct gain vs ref | Imitation gain vs ref | Hybrid gain vs ref |
|---|---:|---:|---:|
| K4 / H12 | -0.26% | +0.00% | -0.04% |
| K2 / H12 | +5.72% | +6.05% | +7.25% |
| K1 / H12 | -0.79% | +1.69% | +0.92% |
| K4 / H24 | -5.70% | -4.41% | -0.79% |
| K2 / H24 | +6.70% | +7.11% | +7.93% |
| K1 / H24 | -4.42% | -3.94% | -2.72% |

Dấu + là cost giảm; dấu − là cost tăng. K4/H12 nominal guard đạt cho cả ba. K1 và H24 không nằm trong training, nên là sensitivity OOD. K1/H12 hybrid chỉ giảm priced0,92% và CI so với reference còn chứa0; waiting violation88,08% so với reference64,2% và planner21,2%. Base cost giảm4,98% nhưng overdue cost tăng24,792 →34,541. K1/H24 cả ba learner tăng tổng cost. Chưa có bằng chứng policy học tốt tradeoff phục vụ/chờ khi capacity này bị bóp thêm.

## So với experiment trước và giới hạn kết luận

Direct PPO giữ pipeline/architecture cũ nhưng đổi training replicas và development/test panels. Gain mới5,72% so với gain cũ1,92% không phải controlled evidence rằng thuật toán đã được cải tiến. Planner gain trên panel mới8,71%, so với5,01% của panel trước; lượng lợi ích có thể khai thác thay đổi theo workload. Cần giữ cả kết quả trước và mới; không chọn riêng panel đạt gate.

Bằng chứng hiện tại hỗ trợ: (1) resource scheduling có economic impact ở operating regime K2 này; (2) một policy tập trung học được phần lớn lợi ích planner; (3) supervision/initialization có benefit vừa phải theo cùng test panel. Nó không chứng minh credit assignment là nguyên nhân duy nhất của gap, không chứng minh transformer/mixer khác sẽ giải quyết gap, và không chứng minh MARL cần thiết.

Nếu tiếp tục, hướng có ích là chẩn đoán state/action regret trên cohort 147002 và K1 trước khi đổi thuật toán, hoặc khóa operating regime K2 rồi xác nhận trên workload panel rộng hơn. Đây là đề xuất tiếp theo; chưa triển khai một full experiment mới.

## Audit và chi phí thực thi

- Manifest COMPLETED, clean commit; **227 artifact hashes** và source/protocol/lock snapshots khớp. Raw artifacts không chỉnh sửa.
- Đủ30 model,180 checkpoint files,9830400 PPO steps/819200 training episodes,256000 optimizer updates;6400 PPO progress rows+800 BC rows.
- Đủ62400 development episodes,105000 test episodes,1890000 intervals và420000 machine rows; test grids/cost/machine totals đúng. Tất cả physical transitions và4 deterministic reference actions được replay.
- Teacher:138240 CSV/tensor rows,11520 trajectories, no reserved seed use; tất cả actual trajectory/behavior/target checks đạt. Full forecast replay720 teacher calls,1080 test planner calls;320 independent noise probes được replay. Không gọi đây là full replay mọi simulated forecast trong dataset.
- BC diagnostic recomputation100 checkpoint/split records khớp trong numerical tolerance; initial direct/BC parameters giống nhau, hybrid initial parameters và parent hash khớp selected BC; BC critic head không đổi.
- Learned batched replay1620000 actions phát hiện **27** tie differences, **3** thuộc hybrid K2/H12. Mọi pair có trạng thái/profile hiện tại đối xứng; scalar replay tại27 logged states khớp27/27 original choices (logit pairs hòa điểm chính xác trên Mac), và27 affected episodes giữ nguyên total cost (mean changes0 ở mọi cell bị ảnh hưởng). Probability/value sai khác tối đa 3.81e-06/2.86e-06. Verdict primary/economic không đổi. Audit status vẫn NUMERICAL_ACTION_MISMATCH, không ghi zero-mismatch PASS.
- No full-training rerun: report status ANALYZED. No claim of exact portability of batched argmax ties.
- Tất cả10 primary physical profiles có trong PPO training của mọi replica; seed disjointness không đồng nghĩa unseen-profile generalization.
- Lab training+development wall time: direct tổng1626s, BC548s, hybrid fine-tuning1624s; teacher generation624s dùng chung BC/hybrid. Hybrid còn thêm teacher/BC compute. Mean K2/H12 inference/episode: learner khoảng8,76ms, rollout56,52ms, reference4,03ms trên runtime lab đã ghi. Không coi đây là benchmark tốc độ portable hoặc compute-matched superiority.

## Fallacy scan — 11/11 checked

| Pattern | Disposition |
|---|---|
| Simpson/group reversal | Report từng cohort/capacity: cohort 147002 reversal vs reference; không tuyên bố mọi workload tốt. |
| Ecological inference | Training-seed means không suy ra mỗi episode/machine tốt hơn. |
| Berkson/selection bias | Dev selection công khai; toàn bộ30 models giữ lại, panel hữu hạn. |
| Collider/control bias | Không điều kiện kết quả primary trên post-treatment success/utilization. |
| Base-rate neglect | Agreement báo riêng choice states, forced STOP được đếm; waiting base rate báo cùng. |
| Regression to mean | Có fresh direct comparator cùng replicas; không dùng dev pre/post làm test effect. |
| Survivorship | Run đầy đủ, không loại seed; no dev-infeasible omission. |
| Look-elsewhere | Một primary contrast; contrasts còn lại exploratory, không corrected-confirmatory claims. |
| Forking paths | Giữ2%/5%/3% và các guards đã khóa, không đổi ngưỡng sau kết quả. |
| Correlation/causal mechanism | Benefit của gói teacher+BC init+compute; không kết luận một nguyên nhân learning gap hoặc MARL necessity. |
| Reverse causality | Training/init trước sealed test; không có test-feedback selection. |

## Files

- analysis.json/statistics.json: metric, CI, dev curves, cohort, numerical replay audit.
- teacher_analysis.json: teacher diagnostics/seed/physics/noise audit.
- numerical_sensitivity.json: 27 cases và full affected-episode cost replay.
- analyze.py,teacher_audit.py,numerical_sensitivity.py,generate_report.py: reproducible read-only analysis.
- primary_cost,learning_curve,cost_mechanism: PNG và PDF xuất độc lập.
