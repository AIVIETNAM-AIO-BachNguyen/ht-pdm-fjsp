## Material Passport
- Type: experiment validation and interpretation; Verification Status: **ANALYZED**.
- Protocol: `ra_qmix_heuristic_assignment_v1`; original locked primary verdict: **FAIL**.
- Source run: `lab_results/ra_qmix_heuristic_assignment_full_cpu_20261004T151339Z`, COMPLETED, clean commit `9cff694b3b19f45c4219adf7728ebb5d59900c88`.
- Derived artifacts: `reports/ra_qmix_heuristic_assignment_analysis_20261005T025516Z/`.
- Scope: artifact/budget/seed/provenance audit, replay all saved policies and exact small policy costs; no retraining. Raw artifacts unchanged.
- Earlier run `20261004T124714Z`: FAILED after 22 complete models due missing source path; documented and excluded, not pooled. Both attempts use the same locked seed panel and scientific settings. No performance-driven seed replacement.

## Kết luận
Teacher heuristic dùng khi train giúp giảm cost trên các cấu hình medium/high mới ở 3–5 máy, nhưng policy học được suy giảm rất lớn ở nominal. **Không đạt tiêu chí sử dụng đã chốt**, dù phần diagnostic có cải thiện rõ. Không thay threshold hay loại nominal khỏi đánh giá để chuyển FAIL thành PASS. Đây không phải thất bại do thiếu headroom như experiment TD+oracle trước đó.

## Tính đầy đủ và kiểm chứng
Run chạy 23:13 ngày 04/10 đến 02:41 ngày 05/10/2026, giờ Taipei, khoảng 3 giờ 28 phút. Đủ 90 models, 5.4 triệu real transitions, 1,338,570 optimizer updates, 600,000 training episodes, 13,500 test episodes, 1,800 development episodes, 680 heuristic references. Mọi manifest audit PASS. CPU, Python 3.12.3, torch 2.14.0+cu126.

Đã đối soát độc lập CSV counts/keys, từng model đủ 60,000 steps/14,873 updates/29 target syncs, toàn bộ 600,000 training episode records và 13,410 progress rows, seed registry/split, số parameters bằng nhau giữa arms cùng N. Source snapshot SHA256 khớp manifest, toàn bộ checkpoint và source đang dùng. Hash hai bảng teacher oracle khớp; kiểm tra lại 826 nhãn train đều tối ưu theo exact gamma=.95 DP.

Load/replay 90 checkpoint: khớp mọi metric của **15,300 test/development episodes**; 680 reference episodes cũng khớp. Tính lại **120 small exact policy-cost rows** bằng DP. Primary delta/CI/gate được tính lại từ CSV và khớp summary. Không có invalid requests và các thành phần cost đối soát đúng. Sealed panel 201–300 đóng. Source chỉ tạo oracle ở N=2.

Giới hạn kiểm chứng: đây là replay chính sách, không phải tái train; DP dùng cùng source, không phải oracle độc lập. Không lưu từng transition train, nên không thể replay độc lập kiểm tra reward-cost ở từng bước train; kiểm tra đó được runner thực hiện và ghi audit. Training episode records có invalid_requests=0.

## Primary đã chốt: FAIL vì nominal guardrail
Trung bình đều trên N=3,4,5 × medium/high, sau khi lấy trung bình 50 rollout/model; training seed là đơn vị suy luận:

| Metric | Kết quả | Tiêu chí | Verdict |
|---|---:|---:|---|
| TD-only cost | 147.1728 | baseline | — |
| TD+heuristic cost | 127.4285 | thấp hơn | — |
| Paired cost delta | -19.7443 | giảm >=5% | PASS phần diagnostic |
| Relative reduction | 13.4157% | >=5% | PASS |
| 95% paired t CI, df=9 | [-26.2276, -13.2611] | upper <0 | PASS |
| Improving training seeds | 10/10 | >=8/10 | PASS |
| Nominal TD-only cost | 57.8863 | baseline guardrail | — |
| Nominal TD+heuristic cost | 146.1260 | <=63.6750 | FAIL |
| Nominal relative increase | **+152.4361%** | <=10% | **FAIL** |

CI chỉ lượng hóa biến thiên giữa 10 training replicates trên panel rollout cố định, không bao trùm mọi failure panel/configuration. Normality của paired differences chưa được xác nhận; 10/10 dấu cải thiện là bằng chứng mô tả bổ sung. Không dùng rollout episodes như samples độc lập của training effect.

## Hiệu ứng theo quy mô
Diagnostic là trung bình medium/high. Cost càng thấp càng tốt.

| Machines | Diagnostic TD | Diagnostic TD+H | Giảm | Nominal TD | Nominal TD+H | Nominal tăng |
|---:|---:|---:|---:|---:|---:|---:|
| 2 (secondary) | 12.383 | 11.896 | 3.93% | 11.288 | 11.120 | -1.49% |
| 3 | 70.054 | 47.217 | 32.60% | 20.590 | 24.540 | +19.18% |
| 4 | 154.916 | 135.162 | 12.75% | 44.184 | 183.037 | +314.26% |
| 5 | 216.549 | 199.907 | 7.69% | 108.885 | 230.801 | +111.97% |

Diagnostic cải thiện 10/10, 9/10, 9/10 seeds tương ứng N=3,4,5; nominal xấu hơn ở 4/10, 10/10, 9/10. Do vậy suy giảm nominal ở 4–5 máy không chỉ do một outlier. N=2 diagnostic chỉ thắng 2/10 seeds; không mở rộng kết luận primary sang mọi N/cell.

Nếu lấy trung bình đều cả nominal/medium/high ở N=3–5 (exploratory, không đổi primary), tổng cost **117.4107 → 133.6610**, tăng khoảng **13.84%**. Guardrail bắt đúng đánh đổi thực tế trên panel, không chỉ là một thủ tục pass/fail.

## Phân rã phối hợp và failure
Trên primary diagnostic panel, collisions/episode giảm **2.446 → 1.337**, waiting **14.918 → 9.592**, failures **3.915 → 3.150**, downtime cost **73.392 → 67.717**. Hành vi phối hợp nhìn chung tốt hơn trong panel này, nhưng không đồng đều: N=4 high collisions tăng 1.400 → 2.128, waiting tăng 9.508 → 13.394 trong khi cost vẫn giảm. Nhãn matching không collision không bảo đảm student không collision.

Trên nominal, collisions thấp có thể đi cùng bỏ mặc máy:
- N=4: collisions **4.400 → .110**, jobs **20.352 → 5.034**, failures **.742 → 3.332**, downtime cost **4.352 → 127.968**.
- N=5: collisions **7.564 → 1.744**, jobs **17.298 → 7.982**, failures **2.862 → 4.098**, downtime cost **39.632 → 160.272**.

Replay toàn bộ nominal cho thấy tại N=4/N=5 TD+H, **100% các decision mà một máy đang failed và mask vẫn cho phép request đều chọn defer**. Đây là mô tả trên trajectories thực tế của student, không phải xác suất trên mọi state. Máy index 1 ở N=4 không được bảo trì trong **87%** nominal episodes; máy index 0 ở N=5 không được bảo trì trong **95.2%**. N=4 có **50/500 episodes không bảo trì máy nào** (TD-only: 0/500).

Ví dụ định trước để minh họa (không thay thế aggregate): N=4, train seed 99000, eval seed 99200, student chọn `(0,0,0,0)` cả 12 bước, kể cả khi cả bốn máy đã failed. Teacher trên chính các quan sát này vẫn đề nghị bảo trì ở các bước có lợi. Trace được lưu trong `representative_nominal_traces.csv`.

## Teacher quality và khả năng học là hai vấn đề khác nhau
Direct heuristic reference nominal có cost **14/18/19** ở N=3/4/5, trong khi student là **24.540/183.037/230.801**. Điều này cho thấy proxy heuristic trên nominal không tự nó gây mức sụp đổ quan sát ở student. Không suy ra teacher là tối ưu, hay đã xác định cơ chế neural/gradient.

Để phân biệt lỗi chỉ do cấu hình mới với lỗi ngay ở cấu hình train, thực hiện **posthoc** 9,000 checkpoint rollouts trên train A/B bằng cùng exogenous panel đã có; không fit lại, không chọn hyperparameters hay đổi scientific gate. Thêm 400 direct teacher references trên train A/B. Các số sau là exploratory:

| N | TD train A/B mean | Student TD+H mean | Direct heuristic mean |
|---:|---:|---:|---:|
| 2 | 4.350 | 9.200 | 5.000 |
| 3 | 14.371 | 49.125 | 33.580 |
| 4 | 85.093 | 126.763 | 83.380 |
| 5 | 186.262 | 175.127 | 152.610 |

Tại N=3, teacher proxy đã kém TD-only trên cấu hình train; student còn kém teacher. Tại N=4, direct teacher gần baseline nhưng student kém hơn cả hai. Do đó không thể quy toàn bộ kết quả cho out-of-distribution nominal; có cả vấn đề proxy quality và khoảng cách giữa nhãn teacher với policy học được.

## Đối chứng oracle N=2
Exact test policy costs, trung bình 10 training seeds (không lẫn với sampled rollout cost):

| Cell | TD-only | TD+heuristic | TD+oracle | Exact undiscounted optimum |
|---|---:|---:|---:|---:|
| nominal | 11.4513 | 11.0750 | 9.7500 | 4.0000 |
| medium | 6.0650 | 8.9000 | 16.8043 | 4.0000 |
| high | 18.8581 | 15.7200 | 31.3418 | 11.2500 |
| development | 7.8750 | 6.0000 | 6.5500 | 4.0000 |

Oracle labels train đúng, nhưng student TD+oracle không đạt gần oracle khi deploy. Posthoc trên train A/B, oracle student rollout costs **26.26 / 11.30**, so với TD **4.50 / 4.20**. Đây không chỉ là teacher heuristic chưa đủ chính xác. Khác experiment trước: supervision lần này chỉ trên real replay, mixture cấu hình và cùng TD; trước đó có supervision theo tập oracle trạng thái và đánh giá khác. Không gộp hai experiment để kết luận mâu thuẫn hoặc generalization đã được chứng minh.

Late update logs có CE còn .033–.108 ở N=3–5; TD loss không giảm đồng đều. Loss nhỏ/trung bình không đảm bảo đúng hành động ở state quan trọng. Chưa có gradients tách TD/CE hay phép can thiệp theo coverage, nên không kết luận CE conflict là nguyên nhân đã được chứng minh.

## Điều dữ liệu hỗ trợ và bước tiếp theo
Có bằng chứng supervised coordination có lợi trên panel áp lực mới, cùng monotonic mixer, nhưng recipe hiện tại chưa giữ được chất lượng tổng thể. Chưa có phép so sánh thay mixer trong run này; không đủ bằng chứng đổ lỗi hoặc miễn trách nhiệm cho monotonic architecture. Cũng chưa chứng minh transfer sang số máy chưa train: mỗi N có model riêng, full ordered peer telemetry và ID cố định.

Ưu tiên follow-up nhỏ, có control rõ, kiểm tra học nhãn ở state quan trọng và tương tác TD–CE trước khi đổi mixer. Cần phân biệt chất lượng teacher, supervision coverage, CE weight và bảo toàn nominal; ví dụ teacher-only ceiling trên cùng dữ liệu/feature contract để đo khả năng imitation, sau đó TD+CE với lịch weight đã chốt và nominal anchor có control tương ứng. Đây là đề xuất, chưa triển khai/chạy experiment mới trong lượt phân tích này.

## Statistical fallacy scan — 11/11 checked
| Check | Kết quả/giới hạn |
|---|---|
| 1 Simpson | Report diagnostic vs nominal và từng N; hiệu ứng trái chiều, không che bằng một mean. Không khẳng định nghịch đảo Simpson cổ điển. |
| 2 Ecological | Inference unit là training seed; không suy từ mean sang mọi state/machine/config. |
| 3 Berkson | Panel hand-specified và hạn chế; không suy rộng ngoài các service families đã chốt. |
| 4 Collider | Không lọc theo teacher agreement, collision-free hay thành tích sau train để tính primary. |
| 5 Base rate | Ghi denominators cho 10 seeds, 500 nominal episodes, mask-actionable failed decisions. |
| 6 Regression to mean | Không chọn seeds/cells từ outlier ở kết quả này; có paired TD baseline. |
| 7 Survivorship | Đủ 90/90 models; failed technical attempt 22 models được ghi rõ và không pooled; rerun giữ seed panel. |
| 8 Look elsewhere | Một locked primary gate; breakdown và posthoc training replay là exploratory, không có secondary efficacy p-values. |
| 9 Forking paths | Giữ nguyên threshold/cells/budget; provenance repair chỉ engineering; ghi rõ mọi posthoc check. |
| 10 Correlation/causation | Controlled auxiliary comparison hỗ trợ effect của recipe trong panel; behavioral decomposition không xác định causal mediation hay kiến trúc bottleneck. |
| 11 Reverse causality | Auxiliary assignment đi trước evaluation; không đảo từ policy fail sang proof mixer là nguyên nhân. |

Primary giữ **FAIL**. Kết quả diagnostic tích cực và regression nominal đều được báo cáo; chưa đủ để coi TD+heuristic là candidate triển khai.
