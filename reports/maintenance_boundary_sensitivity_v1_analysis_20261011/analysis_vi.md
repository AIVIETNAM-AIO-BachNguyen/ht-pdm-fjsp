# Phân tích maintenance boundary sensitivity v1 — 2026-10-11

## Material Passport

Origin: academic-research-suite / experiment-agent / validate; inline analysis. Verification Status: **ANALYZED**.
Integrity, physical replay và thống kê tái tính: **PASS**. Không rerun toàn bộ planner inference hoặc full experiment; không coi audit này là xác nhận tối ưu của planner.

Run: `maintenance_boundary_sensitivity_v1_full_cpu_20261010T101822Z`. Commit: `54d17f9ce9c195c59ec6745868ae9b3450c0b1c5`. Raw data read-only; mọi đầu ra phân tích nằm trong thư mục report này.

## Kết quả chính

Cách xử lý cuối episode ảnh hưởng rõ tới backlog. Khi tính chi phí sau cuối kỳ, số máy hỏng còn chờ bắt đầu sửa giảm; tuy nhiên lợi ích chi phí ở giữa episode dài trong điều kiện primary skill_mask vẫn chưa chắc chắn. Phân công linh hoạt chưa đạt ngưỡng đầu tư 1% đã chốt.

## Dữ liệu và kiểm chứng

Run hoàn tất trong **5.025 giờ** (5 giờ 1 phút 32 giây). Linux x86_64, Python 3.12.3, NumPy 2.5.3; CPU, 16 logical CPUs, OMP/OPENBLAS/MKL=1. Không có training hoặc checkpoint.

- 25 artifacts, toàn bộ checksum và kích thước đúng manifest; source/config/plan trùng commit, dirty=false.
- Đủ 4.032 development episodes và 23.040 test episodes; toàn bộ grid 3 conditions × 3 horizons × 64 profiles × 8 shocks × 5 controllers.
- 921.600 physical decisions được replay; mọi action hợp lệ, state/cost và các chi phí thành phần đúng.
- Tái dựng 304,643 request records ở hai cutoff và 14,389 carryover records. Không reset state hoặc công việc đang sửa tại H.
- 405,504 rule actions được tính lại; 1,800 time-profile rows khớp raw decisions. Planner actions được kiểm tra feasibility và restriction của fixed, không rerun lựa chọn Monte Carlo.
- Best rule được chọn lại từ development và khớp file selection. Selection được code khóa trước test; không chọn lại trên test.
- Seed panels mới tách biệt; 16 dev / 64 test physical profiles mỗi condition không trùng nhau hoặc 96 profiles của v1 trước, kể cả sau hoán vị máy/worker và bỏ horizon.
- Cả 27 contrasts, profile differences và bootstrap CIs đều được tính lại độc lập. 3 primary dùng 98,333333% CI; 24 exploratory dùng 95%. Đơn vị bootstrap là 64 profiles, conditional trên panel 8 shocks, không phải 23.040 episodes hoặc 921.600 ticks.

## Ba hypothesis primary (skill_mask)

| Hypothesis | Effect | Khoảng tin cậy 98,33% | Gate đã chốt | Kết quả |
|---|---:|---:|---|---|
| P1: fixed_zero − fixed_tail, cutoff pending tại H12 | 0.3887 máy | [0.2461, 0.5332] | Lower CI > 0,1 máy | PASS |
| P2: rule − fixed_tail, interior cost tại H48 | 0.986% | [-0.206%, 2.203%] | Lower CI > 1% | INCONCLUSIVE |
| P3: fixed_tail − flexible_tail, interior cost tại H48 | 0.085% | [-0.361%, 0.527%] | Lower CI > 1% | BELOW RESEARCH MARGIN |

Interior H48 là ticks [12,36), tức 12–35. Relative effect là chênh lệch trung bình chia trung bình của controller bên trái; CI bootstrap cùng numerator và denominator. Không báo p-value vì protocol dùng CI.

Profile wins/ties/losses: P1 47/6/11; P2 36/0/28; P3 25/15/24.

P2 chưa đủ bằng chứng planner tốt hơn rule trên 1%, nhưng cũng chưa chứng minh không có lợi ích: CI chứa 0 và vượt 1%. P3 có upper CI 0,527%, dưới margin 1%; đây là đánh giá theo ngưỡng nghiên cứu ở N4/K2, không phải chứng minh hai allocator tương đương hoặc ngưỡng kinh tế nhà máy.

## Vì sao kết quả ngắn hạn dễ gây hiểu nhầm

Tại skill_mask/H12, mọi controller chạy tiếp 12 ticks bằng cùng deadline_matching, giữ nguyên state và công việc đang sửa. Tổng chi phí gồm cả phần sau cutoff:

| Controller | Chi phí trước H12 | 12 ticks sau H12 | Tổng 24 ticks | Pending tại H12 | Tổng máy hỏng tại H12 |
|---|---:|---:|---:|---:|---:|
| Rule tốt nhất trên dev | 192.650 | 199.631 | 392.281 | 0.439 | 0.953 |
| Fixed, terminal=0 | 187.441 | 216.002 | 403.443 | 0.988 | 1.084 |
| Fixed, tail=12 | 198.715 | 191.393 | 390.107 | 0.600 | 0.934 |
| Flexible, tail=12 | 198.582 | 191.023 | 389.605 | 0.580 | 0.926 |

Fixed_zero có vẻ rẻ hơn rule **2,70%** khi chỉ nhìn 12 ticks đầu; khi tính cả phần chạy tiếp, nó lại tốn hơn rule **2,85%**. Fixed_tail tăng chi phí trước cutoff để giảm phần sau; tổng chi phí thấp hơn fixed_zero **3,31%**. Các tỷ lệ tổng chi phí này là secondary descriptive means, không có confirmatory CI bổ sung.

Pending giảm từ **0,988 xuống 0,600 máy**, tức **39,33%**. Nhưng service-start không phải repair completion: CM đang sửa tăng từ 0,096 lên 0,334 máy. Về kế toán trạng thái, khoảng 61,3% chênh lệch pending tương ứng với tăng số máy đang sửa; tổng máy hỏng chỉ giảm từ 1,084 xuống 0,934, tức 13,87%. Không diễn giải mức giảm pending là tất cả đã sửa xong.

Fixed_tail/H12 vẫn còn pending cao hơn rule: 0,600 so với 0,439. Gate service đã khóa kiểm tra interior H48; không chuyển gate sang H12 sau khi xem số liệu, nhưng giữ chênh lệch này trong báo cáo.

## Interior H48 và service guard

| Controller | Cost / machine-tick | Mean pending | Mean total failed | Overdue ticks / physical tick |
|---|---:|---:|---:|---:|
| Rule tốt nhất trên dev | 4.152527 | 0.467529 | 0.983236 | 0.011149 |
| Fixed, terminal=0 | 4.128459 | 0.559652 | 1.030843 | 0.014730 |
| Fixed, tail=12 | 4.111572 | 0.565918 | 1.041260 | 0.013265 |
| Flexible, tail=12 | 4.108093 | 0.567383 | 1.038574 | 0.013672 |

Service guard descriptive đều PASS. Tuy nhiên pending của fixed_tail cao hơn rule **0,098389 máy**, chỉ cách margin +0,1 khoảng **0,001611 máy**. Guard pass sát ngưỡng, không chứng minh dịch vụ tương đương. Tổng failed tăng 0,058024 máy; overdue tăng khoảng 0,002116/tick, dưới margin 0,25/12.

**Timing-learning screen: NOT_ESTABLISHED**, vì P2 không vượt lower-CI margin 1%. P1 pass và service guard pass không thay thế yêu cầu P2.

Ở H48, mean extended cost của fixed_tail = 983,475 và flexible_tail = 985,100. Vì vậy mức lợi rất nhỏ ở interior của flexible không đồng nghĩa nó thắng ở mọi objective. Tổng H48+L12 cũng là finite-horizon outcome, không phải infinite-horizon optimum.

## Heterogeneity và toàn bộ mean costs

Không gộp conditions để che khác biệt. Ở H48, headroom fixed_tail so với rule:

| Condition | Relative interior improvement | CI | Nhãn |
|---|---|---|---|
| homogeneous | 2.849% | [1.845%, 3.884%] | Exploratory, 95% |
| specialized | 1.512% | [0.534%, 2.467%] | Exploratory, 95% |
| skill_mask | 0.986% | [-0.206%, 2.203%] | Primary, 98,33% |

Homogeneous còn lợi ích khoảng 2,85%; specialized khoảng 1,51%, nhưng lower CI dưới 1%. Do đó không kết luận planner hết giá trị trong mọi condition. Conditions thay đổi cả topology lẫn effective capacity; khác biệt giữa conditions không phải pure causal effect của skill mask.

Negative control homogeneous: fixed_tail và flexible_tail có mean bằng nhau ở cả ba horizons, phù hợp worker giống nhau. Primary assignment conclusion chỉ áp dụng setting hiện tại, chưa áp dụng N/K lớn hơn.

| Condition/H | Best rule | Fixed zero: extended cost | Fixed tail: extended cost | Flexible tail: extended cost |
|---|---|---:|---:|---:|
| homogeneous:12 | risk_matching | 397.299 | 386.205 | 386.205 |
| homogeneous:24 | deadline_matching | 591.615 | 578.213 | 578.213 |
| homogeneous:48 | deadline_matching | 985.891 | 973.902 | 973.902 |
| specialized:12 | deadline_matching | 394.902 | 381.979 | 381.902 |
| specialized:24 | deadline_matching | 588.521 | 577.625 | 574.535 |
| specialized:48 | deadline_matching | 979.010 | 967.340 | 966.389 |
| skill_mask:12 | risk_matching | 403.443 | 390.107 | 389.605 |
| skill_mask:24 | deadline_matching | 599.533 | 584.980 | 584.559 |
| skill_mask:48 | deadline_matching | 999.305 | 983.475 | 985.100 |

Không coi chênh lệch với experiment v1 trước là một causal comparison: panel profiles/shocks mới và objective mới khác nhau. Controlled comparison trong run hiện tại là bằng chứng trực tiếp cho boundary treatment.

## Thời gian chạy

Thực tế có **7.087 tỷ scenario-ticks** trong forecast. Thời gian đo bên trong action selection là 4.913 giờ, tương đương **97.77%** wall time. Phần nặng chủ yếu nằm ở planner, không phải 0,813 GB CSV đầu ra.

| skill_mask/H48 | Mean action latency (ms) | P95 (ms) |
|---|---:|---:|
| Rule tốt nhất trên dev | 0.186 | 0.203 |
| Fixed, terminal=0 | 35.782 | 116.595 |
| Flexible, terminal=0 | 38.738 | 147.558 |
| Fixed, tail=12 | 41.400 | 128.979 |
| Flexible, tail=12 | 45.234 | 161.139 |

Latency là đo trong closed-loop run với lịch cache/code hiện tại; không phải benchmark GPU, không equal-compute và không chứng minh khả năng real-time ở nhà máy.
Có 196,707 / 516.096 planner decisions (**38.11%**) chỉ có một feasible candidate nhưng vẫn mô phỏng tương lai. Đây là cơ hội tối ưu runtime cần thiết kế/kiểm tra riêng; không tự sửa code hoặc hứa mức speedup từ số đếm này.

## Censoring và carryover

Observed waits có censoring tại H và H+L. Served-only mean/p95 là conditional metrics; không dùng chúng thay cho toàn bộ request history. Pending requests tại H được nối bằng machine/request ID qua tail, kể cả yêu cầu mới đến đúng boundary có wait=0.

| skill_mask/H12 | Cutoff requests / episode | Censored tại H | Carryover bắt đầu sửa trong tail | Còn censored sau tail |
|---|---:|---:|---:|---:|
| Rule tốt nhất trên dev | 2.713 | 0.439 | 0.439 | 0.000 |
| Fixed, terminal=0 | 2.529 | 0.988 | 0.988 | 0.000 |
| Fixed, tail=12 | 2.541 | 0.600 | 0.600 | 0.000 |
| Flexible, tail=12 | 2.529 | 0.580 | 0.580 | 0.000 |

## Fallacy scan — 11/11 checked

| Loại | Đánh giá trong phạm vi dữ liệu |
|---|---|
| Simpson | Giữ riêng condition/H; headroom khác nhau đáng kể, không suy từ pooled mean. |
| Ecological | Bootstrap ở profile; không suy sang từng máy, nhà máy hoặc mọi N/K. |
| Berkson / sampling selection | Profiles bị lọc để không trùng run cũ; inference giới hạn distribution và selection protocol này. |
| Collider | Không lọc episode theo failure/service outcome để tính effects. Served-only wait chỉ dùng secondary conditional metric. |
| Base-rate neglect | Censoring/served metrics đi kèm request counts và state counts; không bỏ denominators. |
| Regression to mean | Best rule chọn trên dev mới, test không chọn từ extremes; không dùng before/after cross-run để quy causal gain. |
| Survivorship | Đủ toàn bộ grid, không dropout episodes; giữ cả censored requests và carryover. |
| Look-elsewhere | Báo đủ 3 primary với correction; 24 còn lại exploratory, không nâng secondary positive result thành primary. |
| Forking paths | Plan trùng commit trước full run, seeds/budget/windows/margins không thay; explanatory secondary ratios ghi descriptive. |
| Correlation vs causation | Policy treatment kiểm tra trong simulator với paired tapes; không quy ra causal effect nhà máy hoặc topology. |
| Reverse causality | Controller và terminal treatment đặt trước simulation; không suy reverse cause từ post-action backlog. |

## Diễn giải và giới hạn

Theo gate đã khóa, kết quả chưa đủ để mở timing-learning follow-up. Assignment phức tạp cũng chưa vượt investment margin ở primary N4/K2. Điều được hỗ trợ rõ hơn là xử lý đúng boundary và dùng rule baseline mạnh.

Không đồng nhất failure-to-start waiting với thời gian máy hỏng tới khi sửa xong. Tail=12 dưới một rule chung là finite policy-evaluation terminal value; không phải terminal value tối ưu. Interior window vẫn có thể chịu ảnh hưởng của quyết định trước đó, không phải stationary performance. Bootstrap uncertainty conditional trên 8 shocks và approximate planner S128; chưa tích phân mọi nguồn Monte Carlo noise.

Giữ riêng evidence, inference và research screen. Không có deployment claim, không có kết luận mọi RL đều vô ích, và không có thay đổi code/launch/transfer trong lượt phân tích này.

## Artifacts phân tích

- `analyze.py`: checksum/source audit, independent bootstrap, full state/cost/request replay và latency audit.
- `integrity_audit.json`, `physical_replay_audit.json`, `independent_statistics.json`, `latency_analysis.json`.
- `plot.py`, `boundary_summary.png`, `boundary_summary.pdf`: explanatory scientific figure.
- `report.py`: tạo báo cáo này từ số đã audit.

![Boundary summary](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_boundary_sensitivity_v1_analysis_20261011/boundary_summary.png>)
