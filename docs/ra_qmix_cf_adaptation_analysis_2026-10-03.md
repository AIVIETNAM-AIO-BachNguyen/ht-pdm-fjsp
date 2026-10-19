# Phân tích CF × adaptation — 03/10/2026

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: validate.
- Origin Date: 2026-10-03.
- Verification Status: ANALYZED; tính lại từ evaluation CSV và primary interactions; chưa chạy lại training/checkpoint evaluation.
- Version Label: cf_adaptation_analysis_v1.
- Source: `lab_results/ra_qmix_cf_adaptation_full_cpu_20261002T021659Z/`.
- Source commit: `9abc1f64f178938301fa9758e31b94445ac5b951`, clean theo manifest.
- Protocol: `ra_qmix_cf_adaptation_factorial_v1`; đã đọc plan và implementation tại source commit bằng git show.

## Kết luận từ kết quả

Giả thuyết bỏ CF giúp giữ hiệu năng nominal khi adaptation không đạt tiêu chí đã khóa. Primary interaction dương +0,95 cost units, trong khi giả thuyết cần âm; chỉ 5/10 training seeds có interaction âm, yêu cầu ≥8/10. CI95% [−0,961; +2,861] chứa 0: chưa đủ bằng chứng xác lập hướng hiệu ứng. Cũng không thể từ kết quả này kết luận CF có lợi một cách chắc chắn hoặc hai variants tương đương.

Curriculum + replay tiếp tục có tín hiệu stress improvement ở cả Full và no-CF, nhưng cả hai đều không đạt gate giữ nominal. Practical promotion của no-CF thất bại. Kết quả ablation cũ, trong đó no-CF tốt hơn trên stress sau nominal-only training, không tự động áp dụng cho bài toán adaptation.

## Thiết kế và độ đầy đủ

2 algorithms × 2 regimes × 10 paired training seeds, 480384 environment steps/trajectory. Train seeds 85000–85009; evaluation seeds 85100–85149; 3 fixed checkpoints 240192/360288/480384. Tất cả inference chính dùng endpoint cuối; intermediate checkpoints là exploratory. Hai algorithms cùng 38210 parameters, chỉ khác CF loss (lambda 0,05 so với 0). Không có capacity-matched QMIX trong run này.

Run hoàn tất lúc 07:34 ngày 03/10/2026, giờ Đài Bắc. Kiểm tra local thấy đủ mọi output trong manifest, gồm 120 checkpoint files. Evaluation có 24000 rows, mỗi arm 12000 unique keys, zero invalid requests và zero cost reconciliation error. Tại phase boundary, evaluation metrics giữa hai regimes giống hệt trong từng algorithm. No-CF weighted gradients đều bằng 0. Các audits về lịch training, optimizer updates, initialization và traces đều PASS theo summary; chưa độc lập tái tạo toàn bộ audit từ checkpoint tensors/training logs. Manifest báo 1188000 trace rows và sealed panel 201–300 chưa đánh giá; không chứng minh lịch sử chạy ngoài repo.

## Endpoint cuối: cost thấp hơn tốt hơn

| Algorithm | Regime | Nominal | Early failure | Slow service | Combined pressure |
|---|---|---:|---:|---:|---:|
| Full | Nominal-only | 8,450 | 144,451 | 89,126 | 178,862 |
| Full | Curriculum + replay | 8,450 | 88,428 | 80,004 | 152,646 |
| No-CF | Nominal-only | 7,800 | 141,871 | 94,842 | 175,203 |
| No-CF | Curriculum + replay | 8,750 | 90,510 | 82,148 | 155,184 |

Trung bình evaluation trong từng training seed trước, rồi trung bình qua 10 training seeds. Không cộng/gộp raw costs giữa scenarios có horizon khác nhau. Sau adaptation, Full có mean cost thấp hơn no-CF ở cả bốn cells trên panel này; đây là mô tả, không phải bốn kiểm định superiority đã hiệu chỉnh.

| Tiêu chí adaptation so với nominal-only cùng algorithm | Full | No-CF | Ngưỡng khóa |
|---|---:|---:|---|
| Mean paired nominal relative degradation | +1,26% | +15,20% | ≤5% |
| Seeds tăng nominal cost không quá 10% | 6/10 | 6/10 | ≥8/10 |
| Seeds cải thiện stress-max regret | 9/10 | 9/10 | ≥8/10 |
| Mean stress-max regret delta | −0,3180 | −0,2913 | <0 |
| Confirmation gate | FAIL | FAIL | Tất cả tiêu chí đạt |

Stress-max regret delta là thay đổi chỉ số regret so với frozen references, không phải phần trăm giảm raw cost và không phải oracle gap. CI95% supportive cho stress delta: Full [−0,4478; −0,1881], no-CF [−0,4180; −0,1646]. CI cho nominal degradation rộng: Full [−13,71%; +16,23%], no-CF [−9,14%; +39,55%].

Mean of per-seed ratios khác ratio of means. Vì vậy Full có nominal means 8,45 → 8,45 nhưng paired relative degradation +1,26%; no-CF có ratio of means +12,18% nhưng gate dùng mean paired ratios +15,20%. Đây không phải lỗi tính toán.

## Primary interaction và phân tán giữa seeds

Interaction = (curriculum − nominal)_noCF − (curriculum − nominal)_Full. Âm mới hỗ trợ giả thuyết đã khóa.

- Full mean absolute adaptation gap: 0,00.
- No-CF mean absolute adaptation gap: +0,95.
- Interaction: +0,95; SD 2,6714; CI95% [−0,961; +2,861], tái tính khớp summary.
- Interaction âm: 5/10 seeds, cần ≥8/10.
- No-CF curriculum − Full curriculum nominal cost: +0,30.
- No-CF curriculum stress-max regret − Full: +0,04153.

Full nominal regressions >10% ở seeds 85002 (+11,76%), 85004 (+22,22%), 85005 (+33,33%), 85006 (+12,50%). Các improvements lớn như seed 85007 (−40%) bù lại khi lấy mean. No-CF regressions >10% ở 85000 (+35,71%), 85001 (+57,14%), 85005 (+84,62%), 85007 (+15,38%). Mean tốt không chứng minh seed stability.

## Cost decomposition và diễn biến training

Endpoint nominal có zero failure events, zero failure cost và zero downtime cost ở cả bốn arms, trên toàn 2000 final nominal evaluation episodes.

| Algorithm/regime | Maintenance | Queue waiting | Tổng nominal |
|---|---:|---:|---:|
| Full nominal-only | 7,50 | 0,95 | 8,45 |
| Full curriculum | 7,50 | 0,95 | 8,45 |
| No-CF nominal-only | 7,10 | 0,70 | 7,80 |
| No-CF curriculum | 7,40 | 1,35 | 8,75 |

No-CF adaptation tăng +0,30 maintenance và +0,65 waiting: waiting chiếm khoảng 68,4% mean gap +0,95. Đây là phân rã số học của cost; chưa xác lập cơ chế gây tăng waiting. No-CF curriculum so với Full curriculum có maintenance thấp hơn 0,10 nhưng waiting cao hơn 0,40, tổng cao hơn 0,30.

| Steps | Full nominal-only | Full curriculum | No-CF nominal-only | No-CF curriculum |
|---|---:|---:|---:|---:|
| 240192 | 13,772 | 13,772 | 11,890 | 11,890 |
| 360288 | 10,620 | 11,870 | 9,650 | 13,408 |
| 480384 | 8,450 | 8,450 | 7,800 | 8,750 |

Intermediate nominal regression lớn hơn endpoint, và ở 360288 no-CF curriculum còn mean failure events 0,146/episode. Không chọn checkpoint sau khi nhìn bảng này để thay endpoint đã khóa. Nominal-only training tiếp tục làm giảm nominal cost nhưng stress early-failure cost tăng từ phase boundary tới endpoint ở cả algorithms; dữ liệu phù hợp với specialization tradeoff, chưa chứng minh một cơ chế duy nhất.

Gradient diagnostics của Full trong adaptation phase: median weighted-CF/TD gradient norm khoảng 0,097%; khoảng 54,2% sampled cosines âm. CF gradient nhỏ hơn TD ở các mẫu này; cosine âm nhẹ không chứng minh CF gây nominal forgetting. Các mẫu training liên tiếp không phải independent replicates để tăng n; 10 training seeds mới là đơn vị inference.

## Giới hạn diễn giải và fallacy scan

Overall confidence: CAUTION cho attribution CF; stress adaptation có tín hiệu nhất quán nhưng recipe chưa đạt preservation gate. Không báo p-values hoặc multiple-testing superiority claims; secondary/intermediate comparisons chỉ mô tả. Paired Student-t CIs với n=10 phụ thuộc phân bố seed deltas, chưa có xác minh normality; không dùng CI chứa 0 làm bằng chứng equivalence.

Coverage: 11/11 loại đã xét.

| Loại | Nhận xét |
|---|---|
| Simpson's paradox | Không gộp horizons/scenarios; nêu rõ cancellation giữa seeds và khác biệt mean ratios/ratio means. |
| Ecological fallacy | Không suy từ mean cost sang độ ổn định mọi seeds hoặc cơ chế từng máy. |
| Berkson's paradox | Không lọc seeds tốt; scope giới hạn benchmark/panels đã khóa. |
| Collider bias | Primary contrast không điều chỉnh biến hậu can thiệp; không dùng waiting làm causal control. |
| Base rate neglect | Final nominal failures bằng 0 được báo rõ; stress phân tích riêng. |
| Regression to mean | Fresh panels và nominal-only controls giúp kiểm soát; không gọi thay đổi so với run cũ là hiệu ứng CF. |
| Survivorship bias | Manifest completed, toàn bộ declared outputs hiện diện; không bỏ seeds xấu. |
| Look-elsewhere effect | Primary cuối đã khóa; checkpoints/scenario contrasts không được biến thành confirmatory claims. |
| Garden of forking paths | Đọc plan tại source commit, giữ endpoint/gates; chưa chứng minh timestamp preregistration độc lập. |
| Correlation vs causation | Factorial contrast kiểm tra CF intervention trong design này; decomposition/gradient không xác lập mediation. |
| Reverse causality | Intervention trước outcomes; quan hệ gradient/queue/cost vẫn không được coi là causal mechanism. |

Reproducibility verdict: CANNOT_VERIFY training reproducibility; arithmetic consistency của reported primary interval và evaluation aggregates đã kiểm tra. Không chạy training mới, không mở sealed panel.
