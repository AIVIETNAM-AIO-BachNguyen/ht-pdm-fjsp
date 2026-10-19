# Phân tích maintenance contention learning v1

## Material Passport

- Mode: ARS experiment-agent / validate; language: Vietnamese.
- Raw input: `/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_contention_learning_v1_full_cpu_20261006T023329Z`; commit `82707dbfc81aeeb40c6cd0ad296c4942ab9a1ce6`.
- Verification Status: ANALYZED. Reproduced full physical trace and training
  provenance; replayed all selected policy actions in batches and a planner
  forecast subset. Cross-platform deterministic action reproduction has6
  secondary near-tie mismatches, investigated below. No full training rerun.
- Output: analysis.json, analyze.py, behavior.json, numerical replay/sensitivity
  JSONs, report này, scientific PNG/PDF figures. Raw artifacts remain unchanged.

**PPO học được một phần lợi ích điều phối ở K2, nhưng FAIL protocol về độ lớn
cải thiện.** Effect1,92% priced cost và1,42% base cost là nhỏ hơn ngưỡng5%/3%
đã khóa. 10/10 training seeds thắng và95% CI cost difference dưới0; nominal và
waiting guards đều đạt. Do đó không nên đồng nhất FAIL protocol với không học.
Chưa có MARL trong thí nghiệm này.

## Dữ liệu, ngân sách và audit

COMPLETED, Git clean, đúng commit. Run `2026-10-06T02:33:33.403480+00:00` → `2026-10-06T03:16:49.420178+00:00`,
khoảng43 phút16 giây. Đủ10models,4.915.200 physical steps,409.600 training
episodes,102.400 optimizer steps;22.400 development episodes,45.000 test
episodes,810.000 test intervals,180.000 machine rows. Cả10 seed đủ budget và
80%K2/20%K4 training mix. Teacher labels0, không seed dev/test trực tiếp vào train.

Kiểm tra90 raw file hashes/sizes,60checkpoint files (initial+4fixed budgets mỗi
seed và10selected copies), parameter changes thực sự, source snapshot, seed
splits,3200 progress records/counters/finite losses, tất cả409.600 configuration
và hazard counter roles/hash. Recompute development reference và checkpoint
selection khớp; mọi seed có checkpoint dev-feasible, không drop seed nào.

Replayed all810.000 scalar physical transitions from reconstructed actual
hazard seeds: pre/post state, action feasibility, cost và machine/episode
reconciliation khớp. Reference actions216.000 bước khớp; planner candidate
argmin54.000 bước khớp; forecasts1080 bước thuộc60 episode khớp hoàn toàn.
Replayed540.000 learned action logits/log-probs/values theo batch; sai khác số
log-prob tối đa3,82e-6 và value2,87e-6. Primary K2/H12 có0 action mismatch trong
60.000 bước; phần near-tie secondary xem mục riêng. Không sửa raw sau phân tích.

Các seed cấu hình test mới không đảm bảo physical profile mới: audit cho thấy
cả10 primary profiles đã xuất hiện trong training của từng seed. Generator
hữu hạn, đúng scope protocol. Vì vậy đây là generalization sang seed/draw mới,
không phải đảm bảo chưa thấy các tổ hợp duration/restoration.

## Primary: N4, K2, H12

Reference `independent_dp_matching`, chọn theo priced cost development, cost
199,892. PPO196,0508; joint planner189,870. **PPO giảm1,92%**; planner giảm
**5,01%** trên panel mới. PPO captures **38.33%** observed priced
planner savings và **31.52%** observed base-cost savings. Đây là
fraction của một achievable planner, không phải fraction của optimal headroom.

95% paired training-seed CI của Δcost là
[-4.3529, -3.3295], estimate−3,8412.
Unit10training seeds, từng seed mean trên10cohorts×50paired shocks. Reference
và planner chạy1lần mỗi panel; không nhân bản chúng thành10independent replicas.
CI conditional trên finite common test panel, không phải environment-population CI.

| Gate đã khóa | Quan sát | Đạt? |
|---|---|---|
| Priced-cost reduction≥5% | 1,92% | Không |
| Wins≥8/10 seeds | 10/10 | Có |
| Upper95% CI<0 | −3,3295 | Có |
| Base-cost reduction≥3% | 1,42% | Không |
| K4/H12 base ratio≤1,1 | 1,00132 | Có |
| K2 waiting violation≤reference+2pp | 0,34% so với5,60% | Có |

![Primary cost](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_contention_learning_v1_analysis_20261006/primary_cost.png)

Ngưỡng5% gần mức5,01% mà planner đạt trên panel mới; để đạt gate này PPO cần
xấp xỉ99,7% mức savings quan sát của planner đó. Đây là mô tả độ khó của target,
không đổi protocol sau khi xem kết quả và không coi planner là ceiling.

| Training seed | Selected physical steps | Test cost | Reduction | Priced planner gap capture |
|---|---:|---:|---:|---:|
| 138000 | 491520 | 197.288 | 1.30% | 25.98% |
| 138001 | 491520 | 195.846 | 2.02% | 40.37% |
| 138002 | 368640 | 196.238 | 1.83% | 36.46% |
| 138003 | 491520 | 195.390 | 2.25% | 44.92% |
| 138004 | 491520 | 196.582 | 1.66% | 33.03% |
| 138005 | 491520 | 195.712 | 2.09% | 41.71% |
| 138006 | 491520 | 195.478 | 2.21% | 44.04% |
| 138007 | 368640 | 195.158 | 2.37% | 47.24% |
| 138008 | 491520 | 197.034 | 1.43% | 28.52% |
| 138009 | 491520 | 195.782 | 2.06% | 41.01% |

Capture giữa25,98% và47,24%; consistency qua seed tốt nhưng magnitude chưa đủ.
Không thể suy rằng tăng budget chắc chắn đóng hết gap, hoặc RL nói chung không
thể làm tốt hơn từ một PPO architecture/configuration hiện tại.

## Learning curve và cơ chế

Mean development K2 initial323,195 →191,0525 ở122.880steps →188,3400 ở245.760
→184,3325 ở368.640 →183,1405 ở491.520. K4 initial306,3935 →162,899 ởcuối.
8/10seed chọn checkpoint cuối;138002/138007 chọn368.640. PPO đã học rõ; curve
còn cải thiện ở đoạn cuối nên chưa có bằng chứng chắc về convergence hoặc
mức budget đủ. Development cohort khác test; không so trực tiếp dev/test costs
để suy overfitting chỉ từ khác mean workload.

Mixed stochastic training cost (80%K2/20%K4) mean first20rollouts237,791 →
last20rollouts185,113. Entropy1,403→0,176, value loss10,416→0,321. Đây là
learning diagnostics, không thay thế deterministic held-out policy comparison.

![Learning curve](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_contention_learning_v1_analysis_20261006/learning_curve.png)

| Metric K2/H12 | Reference | PPO | Planner |
|---|---:|---:|---:|
| Maintenance cost | 12.5300 | 14.3708 | 13.2540 |
| Failure cost | 40.3500 | 36.5580 | 29.4000 |
| Downtime cost | 145.9320 | 145.0644 | 147.2160 |
| Overdue cost | 1.0800 | 0.0576 | 0.0000 |
| Failures | 2.6900 | 2.4372 | 1.9600 |
| Waiting violation | 0.0560 | 0.0034 | 0.0000 |
| Utilization | 0.8230 | 0.9529 | 0.8988 |

PPO so reference: Δmaintenance+1,8408, Δfailure−3,7920, Δdowntime−0,8676,
Δoverdue−1,0224, tổng−3,8412. Failure2,690→2,4372 giảm9,40%, so planner1,960
(giảm27,14% so reference). Waiting violation5,60%→0,34%; failed-wait ticks
4,570→1,3074. PPO tốt ở xử lý waiting nhưng còn cách planner về ngăn failure.
Utilization tăng82,30%→95,29%, so planner89,88%: dùng hết technician không tự
đồng nghĩa với lịch tối ưu.

Preventive starts ởage<2 chiếm23,36% PPO,16,66% planner,0% independent DP.
Early PM có thể hợp lý khi dự đoán capacity sẽ bận; rate cao hơn tự nó không
chứng minh lỗi. Dữ liệu cho thấy PPO maintenance cost cao hơn nhưng phòng
failure kém hơn planner; chưa tách causal contribution của early PM, chọn máy,
encoder, critic và policy optimization.

## Heterogeneity qua workload cohorts

| Cohort | PPO mean across seeds | Reference | Δcost | Seed thắng |
|---|---:|---:|---:|---:|
| 141000 | 149.580 | 150.640 | -1.060 | 8/10 |
| 141001 | 251.702 | 246.340 | +5.362 | 0/10 |
| 141002 | 204.854 | 208.120 | -3.266 | 10/10 |
| 141003 | 224.966 | 226.900 | -1.934 | 9/10 |
| 141004 | 164.812 | 181.400 | -16.588 | 10/10 |
| 141005 | 169.430 | 173.940 | -4.510 | 10/10 |
| 141006 | 198.222 | 198.380 | -0.158 | 3/10 |
| 141007 | 193.486 | 208.860 | -15.374 | 10/10 |
| 141008 | 189.514 | 188.780 | +0.734 | 3/10 |
| 141009 | 213.942 | 215.560 | -1.618 | 8/10 |

8/10cohort có mean improvement;141001 và141008 worse. Cohort141001 thua ởcả
10seed. Khoảng83,2% tổng net savings đến từ141004 và141007. Vì thế10/10seed
wins biểu thị độ ổn định training trên cùng panel, không bảo đảm thắng mọi
workload. Đây là lý do cần giữ scope conditional của CI.

## Nominal và sensitivities

| Capacity/H12 | Reference cost | PPO | Planner | PPO priced reduction | PPO base reduction | Seed wins | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| K4 | 166.430 | 166.650 | 166.858 | -0.13% | -0.13% | 0/10 | EXPLORATORY |
| K2 | 199.892 | 196.051 | 189.870 | +1.92% | +1.42% | 10/10 | FAIL |
| K1 | 273.978 | 294.909 | 257.554 | -7.64% | +4.13% | 0/10 | EXPLORATORY |

K4/H12 nominal: cost tăng0,13%, giữ guard≤10%; không có improvement rõ.

K1/H12 là capacity chưa huấn luyện. Priced cost tăng7,64% dù base cost giảm
4,13%. Maintenance+0,043, failure−13,365, downtime+3,084, overdue+31,1688;
waiting cost26,040→57,2088. Violation76,4%→93,22%, max-wait trung bình
5,644→7,4042. Utilization gần100% không giải quyết backlog. Fewer failures
không tự đồng nghĩa reliability tốt hơn khi máy failed chịu downtime/wait
lâu hơn và không thể failure lại trong lúc vẫn failed.

Behavior K1 khi idle và có failed demand: PPO chọn PM thay CM ở60,52% decision
opportunities, so reference33,55%, planner31,48%. Per-machine mean service
starts duration1:1,849→2,690; duration3:1,205→0,977. Duration3 max-wait
3,833→5,272 và violation30,35%→59,93%. Pattern phù hợp bias toward jobs ngắn
và trì hoãn jobs dài. Nó là chẩn đoán OOD, không chứng minh PPO được train
trên K1 cũng sẽ thất bại hoặc việc ưu tiên PM thay CM luôn sai.

H24 exploratory: K2 giảmpriced3,00% vàbase2,49%,10/10seed wins; K4cost tăng
5,28%; K1cost tăng9,62%. HorizonH24 chưa train. K4/H24 có33,56% PM ởage<2,
so reference0%, phù hợp vấn đề over-maintenance trong horizon shift; vẫn cần
intervention để quy nguyên nhân. Không dùng H24 để thay primary FAIL.

![Cost mechanism](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_contention_learning_v1_analysis_20261006/cost_mechanism.png)

## Numerical reproducibility limitation

6/540.000 learned action batch-replay mismatches trên Mac so logged Ubuntu:
5ởK1 và1ởK2/H24, chỉseed138008. Các cặp máy đổi có compact state và duration/
restoration giống nhau. Logits bằng nhau trên Mac hoặc rất gần nhau;1e-6
near-tie rule chưa bảo đảm deterministic tie behavior bất biến hardware/batch.
Scalar replay6states khớp3logged actions,3còn đổi máy. Xem
`numerical_replay_detail.json`. Không có primary K2/H12 mismatch được phát hiện
trong full batched replay; cost/physical transitions logged đều hợp lệ.

Tái chạy6affected episodes với checkpoint nguyên vẹn trên Mac cho costΔ
+7,0,0,−37,+22,0. Nếu thay sáu episode này trong panel, mean PPO cost K1/H12
chỉ+0,0014 vàK1/H24−0,003;K2/H24 không đổi. Các kết luận tăng cost ởK1 và
primary FAIL không đổi. Đây là bounded numerical sensitivity, không phải
chứng nhận mọi future trajectory đều reproducible. Không sửa checkpoint/code/
raw data để ép match; cần numerical tie test trước một experiment tiếp.

## Kết luận và bước kiểm tra phù hợp

Bài toán có economic scheduling headroom; centralized PPO đã khai thác được
một phần trên K2. Hạn chế hiện tại là học đủ giá trị dài hạn của việc chiếm
technician và chọn preventive/corrective schedule, cùng sensitivity khi capacity
và horizon đổi. Dữ liệu không chỉ ra thiếu conflict là nguyên nhân chính;
planner vẫn giảmcost5,01% trong primary panel. Chưa đủ để xác định PPO optimizer,
encoder hoặc budget là nguyên nhân duy nhất của gap còn lại.

Một diagnostic phù hợp tiếp theo là cùng encoder/action space, so direct PPO,
planner imitation và imitation→PPO trên new sealed panels. Nếu imitation tái
hiện được planner mà PPO yếu, đó là evidence hướng optimization/credit
assignment; nếu cả imitation yếu, cần kiểm tra representation/data coverage/
teacher noise trước. Teacher budget phải báo riêng; không gọi đây là fair
compute comparison khi supervision khác. Đây mới là đề xuất, chưa chạy hoặc
sửa thí nghiệm tiếp.

Kết quả không hỗ trợ MARL necessity: policy hiện tại centralized; không có
MARL comparator hoặc ràng buộc distributed execution thực tế. Cần learned
centralized baseline đủ mạnh và information/communication contract trước claim đó.

## All references H12

| Capacity | Controller | Priced cost | Base cost | Waiting violation |
|---|---|---:|---:|---:|
| low | risk_skill_rule | 166.430 | 166.430 | 0.00% |
| low | deadline_matching | 166.430 | 166.430 | 0.00% |
| low | lookahead_matching | 166.430 | 166.430 | 0.00% |
| low | independent_dp_matching | 166.430 | 166.430 | 0.00% |
| low | joint_rollout | 166.858 | 166.858 | 0.00% |
| low | central_subset_ppo | 166.650 | 166.650 | 0.00% |
| middle | risk_skill_rule | 205.466 | 205.466 | 0.00% |
| middle | deadline_matching | 206.370 | 206.370 | 0.00% |
| middle | lookahead_matching | 201.942 | 198.630 | 17.60% |
| middle | independent_dp_matching | 199.892 | 198.812 | 5.60% |
| middle | joint_rollout | 189.870 | 189.870 | 0.00% |
| middle | central_subset_ppo | 196.051 | 195.993 | 0.34% |
| high | risk_skill_rule | 271.110 | 262.806 | 34.80% |
| high | deadline_matching | 271.984 | 260.776 | 42.60% |
| high | lookahead_matching | 279.720 | 245.880 | 74.00% |
| high | independent_dp_matching | 273.978 | 247.938 | 76.40% |
| high | joint_rollout | 257.554 | 248.698 | 39.60% |
| high | central_subset_ppo | 294.909 | 237.700 | 93.22% |

## Statistical fallacy scan:11/11

- Simpson: split capacity/horizon/cohort; no claim of universal benefit from pooled mean.
- Ecological: conditional seed-level mean; not every episode/machine improves.
- Berkson: full fixed panels retained; finite known-model generator scope.
- Collider: no post-treatment filtering by failure, utilization or dev-feasible seed.
- Base rate: absolute costs, failure counts and violation rates accompany percentage changes.
- Regression-to-mean: selection on dev; fresh held-out seed panels; no test-based seed choice.
- Survivorship: all10models, fixed budgets and complete grids;0 dropped seeds.
- Look-elsewhere: one primary; K1/K4/H24 and cohort/behavior diagnostics exploratory.
- Forking paths: frozen thresholds/checkpoints; preserve FAIL; no post-hoc gate replacement.
- Causal overreach: controlled simulator policy effects, no attribution to a single mechanism or MARL.
- Reverse causality: temporal simulator intervention, not cross-sectional regression.

Ten training seeds give an approximate t-CI; no population-of-workloads uncertainty
is claimed. Training seeds independently sample initialization/actions/configs,
while every model shares a fixed evaluation panel. Small CI does not imply
small uncertainty over unseen real plants, laws, skills or workload distributions.
