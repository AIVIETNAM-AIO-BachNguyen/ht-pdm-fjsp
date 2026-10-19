## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-12T02:32:33Z
- Verification Status: ANALYZED
- Version Label: validation_v1

# Phân tích maintenance_assignment_opportunity_v1

**Kết luận:** calibration đã đủ mẫu và 128 futures làm nhãn ít nhiễu hơn, nhưng chưa có bằng chứng cải thiện chi phí đáng tin cậy. Opportunity-low ungated có trung bình tốt nhất ở dense; opportunity-high không tốt hơn low. High-gated vẫn giữ PPO trên toàn bộ test.

Run `/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_assignment_opportunity_v1_full_cpu_20261011T174247Z`; commit `490f8fb11b7175d40b76f32f506930a12e34c20b`; code sạch. COMPLETED/PASS, 34.97 phút CPU, 5.190 artifact (300,17 MB). 3 source ×24 profile ×4 shock ×2 điều kiện ×7 controller =4.032 episode. Mỗi condition/controller có288 episode. Chi phí là H48 +tail12, thấp hơn tốt hơn.

## Chi phí closed-loop

| Controller | Dense: chi phí | Giảm so PPO | Skill mask: chi phí | Giảm so PPO | Đổi assignment dense / skill |
|---|---:|---:|---:|---:|---:|
| frozen_ppo | 1187.556 | +0.000% | 1191.663 | +0.000% | 0 / 0 |
| landmark_low_ungated | 1182.396 | +0.434% | 1194.062 | -0.201% | 136 / 46 |
| landmark_low_gated | 1187.556 | +0.000% | 1191.663 | +0.000% | 0 / 0 |
| opportunity_low_ungated | 1179.257 | +0.699% | 1193.931 | -0.190% | 148 / 45 |
| opportunity_low_gated | 1187.295 | +0.022% | 1191.663 | +0.000% | 5 / 0 |
| opportunity_high_ungated | 1180.104 | +0.627% | 1194.059 | -0.201% | 137 / 44 |
| opportunity_high_gated | 1187.556 | +0.000% | 1191.663 | +0.000% | 0 / 0 |

`landmark_low`: cơ hội tự nhiên ở tick0/12/24/36,16 futures. `opportunity_low`: mọi tick có assignment thay thế,16 futures, cân bằng clock bin. `opportunity_high`: cùng trạng thái và cùng draws như low,128 futures. Mọi bản giữ đúng tập máy PM và cạnh CM của PPO tại can thiệp; trajectory tiếp theo có thể đổi lịch. Đây là ablation mới trên panel mới, không phải rerun chính xác natural/broad của confidence-v1.

## Các tiêu chí chính đã khóa

20.000 crossed source/profile bootstrap, trung bình4 shock trong mỗi ô;4 CI98,75% với Bonferroni xấp xỉ; mốc thực dụng1%.

| So sánh | Giảm chi phí | CI98,75% |
|---|---:|---:|
| dense:opportunity_high_gated_vs_frozen_ppo | +0.000% | [+0.000%; +0.000%] |
| dense:opportunity_high_gated_vs_opportunity_high_ungated | -0.631% | [-1.979%; +0.488%] |
| skill_mask:opportunity_high_gated_vs_frozen_ppo | +0.000% | [+0.000%; +0.000%] |
| skill_mask:opportunity_high_gated_vs_opportunity_high_ungated | +0.201% | [-0.143%; +0.757%] |

Cả4 tiêu chí đều BELOW_MARGIN theo quy tắc đã khóa (cận trên dưới1%); không có PILOT_GAIN_SIGNAL. High-gated/PPO có CI[0;0] vì cùng hành vi trong mẫu, không phải bảo đảm tương đương mọi môi trường. High-gated so high-ungated chứa0 ở cả hai điều kiện; chưa chứng minh guard có lợi hay gây hại.

## So sánh phụ và bổ sung

| So sánh | Giảm chi phí | CI95% |
|---|---:|---:|
| dense:opportunity_low_ungated_vs_landmark_low_ungated | +0.265% | [-0.095%; +0.774%] |
| dense:opportunity_low_gated_vs_landmark_low_gated | +0.022% | [+0.000%; +0.128%] |
| dense:opportunity_high_ungated_vs_opportunity_low_ungated | -0.072% | [-0.277%; +0.127%] |
| dense:opportunity_high_gated_vs_opportunity_low_gated | -0.022% | [-0.128%; +0.000%] |
| dense:landmark_low_gated_vs_landmark_low_ungated | -0.436% | [-1.264%; +0.249%] |
| dense:opportunity_low_gated_vs_opportunity_low_ungated | -0.682% | [-1.638%; +0.171%] |
| dense:opportunity_high_gated_vs_opportunity_high_ungated_secondary | -0.631% | [-1.590%; +0.231%] |
| skill_mask:opportunity_low_ungated_vs_landmark_low_ungated | +0.011% | [+0.000%; +0.064%] |
| skill_mask:opportunity_low_gated_vs_landmark_low_gated | +0.000% | [+0.000%; +0.000%] |
| skill_mask:opportunity_high_ungated_vs_opportunity_low_ungated | -0.011% | [-0.062%; +0.000%] |
| skill_mask:opportunity_high_gated_vs_opportunity_low_gated | +0.000% | [+0.000%; +0.000%] |
| skill_mask:landmark_low_gated_vs_landmark_low_ungated | +0.201% | [-0.086%; +0.606%] |
| skill_mask:opportunity_low_gated_vs_opportunity_low_ungated | +0.190% | [-0.090%; +0.589%] |
| skill_mask:opportunity_high_gated_vs_opportunity_high_ungated_secondary | +0.201% | [-0.078%; +0.599%] |
| dense:landmark_low_ungated_vs_frozen_ppo_posthoc | +0.434% | [-0.250%; +1.248%] |
| dense:opportunity_low_ungated_vs_frozen_ppo_posthoc | +0.699% | [-0.159%; +1.650%] |
| dense:opportunity_high_ungated_vs_frozen_ppo_posthoc | +0.627% | [-0.232%; +1.565%] |
| skill_mask:landmark_low_ungated_vs_frozen_ppo_posthoc | -0.201% | [-0.610%; +0.086%] |
| skill_mask:opportunity_low_ungated_vs_frozen_ppo_posthoc | -0.190% | [-0.592%; +0.090%] |
| skill_mask:opportunity_high_ungated_vs_frozen_ppo_posthoc | -0.201% | [-0.602%; +0.078%] |

Các khóa `_posthoc` là so sánh ungated/PPO bổ sung sau khi mở test; không thay tiêu chí chính. Dense opportunity-low/PPO:+0,699%,CI[-0,159%;+1,650%]; high:+0,627%,CI[-0,232%;+1,566%]. Low so landmark:+0,266%,CI[-0,095%;+0,774%]; high so low:−0,072%,CI[-0,277%;+0,127%]. Chưa xác lập H1/H2 về chi phí. Skill ungated có trung bình kém PPO khoảng0,19–0,20%, nhưng CI chứa0 nên chưa chứng minh hại. Các chênh lệch rất nhỏ với CI chạm0 là mô tả thăm dò, không gọi là significant.

Dense ungated giảm trung bình chi phí ở cả3 PPO nguồn; skill có1 nguồn tốt hơn và2 nguồn kém hơn. Không gộp dense/skill thành lợi ích chung. Chỉ3 source nên dấu cùng chiều ở dense vẫn không đủ làm CI hẹp; không so trực tiếp chi phí tuyệt đối với run cũ có panel khác.

## H1: lấy mẫu theo cơ hội thay đổi dữ liệu thế nào?

| Điều kiện / arm | Nhãn train | Tỷ lệ draws tick>=12 |
|---|---:|---:|
| dense / landmark_low | 846 | 10.04% |
| dense / opportunity_low | 1779 | 75.14% |
| dense / opportunity_high | 1779 | 75.14% |
| skill_mask / landmark_low | 260 | 5.79% |
| skill_mask / opportunity_low | 472 | 74.99% |
| skill_mask / opportunity_high | 472 | 74.99% |

2251 cơ hội train (1779 dense,472 skill); landmark giữ1106 trong số này. Dữ liệu đều từ trajectory PPO thật, không có probe tổng hợp. Tỷ lệ draws muộn tăng từ10,04% dense/5,79% skill lên khoảng75%. Thiết kế đồng thời đổi tick coverage và cân bằng clock; không tách nhân quả riêng hai thành phần.

## H2: nhãn chính xác hơn có giúp policy không?

| Điều kiện | MCSE16: mean chuẩn hóa | MCSE128 | Mức giảm MCSE | Số nhãn đổi dấu |
|---|---:|---:|---:|---:|
| dense | 0.6628 | 0.2353 | 64.50% | 241/1779 (13.55%) |
| skill_mask | 0.6317 | 0.2288 | 63.78% | 76/472 (16.10%) |

128 so16 futures làm MCSE giảm khoảng64%, gần tỷ lệ1/sqrt(8). Low dùng đúng16 scenario đầu của cùng full tape ở mọi tick. High không phải ground truth: vẫn có MCSE. Các nhãn tương quan theo trajectory/profile, nên không coi2251 nhãn là2251 mẫu độc lập. Đây là kiểm tra precision của estimator, không phải chứng minh lợi ích policy.

| MSE ensemble trên cùng nhãn128 | Landmark low | Opportunity low | Opportunity high |
|---|---:|---:|---:|
| dense / development | 2.4976 | 2.6521 | 2.6423 |
| dense / calibration | 2.0914 | 2.1242 | 2.1215 |
| skill_mask / development | 6.6489 | 6.0518 | 5.9879 |
| skill_mask / calibration | 2.8812 | 2.6519 | 2.5853 |

High có MSE hơi thấp hơn opportunity-low trên dev/calibration, rõ hơn ở skill. Tuy nhiên giảm sai số value không chuyển thành giảm chi phí test; trên48 root test, dense MSE low2,225/high2,476,skill low2,006/high1,971. Mẫu root nhỏ và chỉ64 futures nên không kết luận thứ hạng tổng quát. Không có bằng chứng từ ablation này để tiếp tục tăng futures chỉ nhằm cải thiện policy.

## H3: gate còn bị thiếu calibration không?

| Source | Dense samples / profiles / late | Skill samples / profiles / late | High margin dense / skill (cost units) |
|---|---:|---:|---:|
| 18100000 | 580 / 64 / 226 | 126 / 31 / 36 | 40.42 / 26.39 |
| 18100001 | 520 / 64 / 179 | 150 / 35 / 43 | 29.28 / 40.85 |
| 18100002 | 480 / 64 / 172 | 151 / 30 / 64 | 37.71 / 40.69 |

Tất cả18 gate (3 source ×3 arm ×2 condition) đều đủ40 samples/16 profiles/10 late và được bật. Skill có126/150/151 mẫu từ30–35 profile, không còn lý do insufficient_calibration của run trước. Nhiều mẫu cùng profile/shock vẫn tương quan.

| Source / condition | High: gain đã trừ SD lớn nhất trên test | Margin |
|---|---:|---:|
| 18100000 / dense | 21.71 | 40.42 |
| 18100001 / dense | 26.83 | 29.28 |
| 18100002 / dense | 19.37 | 37.71 |
| 18100000 / skill_mask | 1.55 | 26.39 |
| 18100001 / skill_mask | 26.87 | 40.85 |
| 18100002 / skill_mask | 21.52 | 40.69 |

High-gated vẫn từ chối toàn bộ487 cơ hội dense và148 skill: adjusted gain không vượt margin. Landmark-gated cũng zero; opportunity-low-gated chỉ5 overrides dense (0 skill),giảm0,022% trung bình. Mọi diagnostic development guard đều zero overrides, nên tình trạng guard hầu như bất động đã hiện trên dev trước khi mở test; protocol vẫn chạy test cố định.

Margin là quantile overoptimism gồm sai số mô hình và1,96 MCSE sau trừ SD ensemble, không phải bằng chứng tin cậy/safety90%. Tại6 hàng calibration quyết định quantile của high, phần predicted-minus-observed là0,987–1,779 chuẩn hóa, lớn hơn phần MC penalty0,308–0,589 (mỗi đơn vị tương đương20 cost). Đây là mô tả tại các hàng quantile, không phải phân rã nhân quả toàn bộ model. Nó cho thấy margin không chỉ do Monte Carlo noise; giảm noise thêm vẫn để lại sai số dự đoán lợi ích. Đủ calibration không đồng nghĩa guard đủ khả năng chọn lọc. High-gated không giữ được lợi ích trung bình của ungated dense vì không thực hiện can thiệp nào. Không retune margin/quantile trên test này.

## Single-root diagnostics

| Điều kiện / ungated arm | Cơ hội | Overrides | Overrides có mean gain âm | Mean gain/cơ hội | CI95% |
|---|---:|---:|---:|---:|---:|
| dense / landmark_low | 33 | 6 | 0 | 4.588 | [0.000;13.976] |
| dense / opportunity_low | 33 | 10 | 2 | 5.932 | [-2.361;16.817] |
| dense / opportunity_high | 33 | 9 | 2 | 4.515 | [-3.005;13.930] |
| skill_mask / landmark_low | 15 | 4 | 1 | 0.690 | [-20.310;9.085] |
| skill_mask / opportunity_low | 15 | 4 | 1 | 0.690 | [-20.310;9.085] |
| skill_mask / opportunity_high | 15 | 4 | 1 | 0.690 | [-20.310;9.085] |

Root panel có33 cơ hội dense,15 skill từ first8 profiles/firstshock. Dense low10 overrides,2 mean harmful; high9,2. Harm lớn nhất28,859 cost. Skill cả3 ungated4 overrides,1 mean harmful (37,234 cost); net gain/cơ hội chỉ0,690 với CI rộng. Đây là một lần đổi assignment rồi PPO continuation, không phải phần trăm episode. Nhãn “harmful” dùng mean của64 futures và vẫn có MC uncertainty. Reference độc lập selection/evaluation không phải oracle.

## Tốc độ và compute

Full run dùng58.800.192 forecast transitions; shared high tape được tính một lần, low chỉ lấy prefix, nên đây là ablation độ chính xác nhãn, không phải ablation ngang chi phí simulator. Mọi learned controller có batch-amortized inference khoảng0,460ms so với PPO0,204–0,205ms (~2,25 lần); batch32, CPU, không phải latency realtime từng quyết định. Near-tie fallback của candidate gốc đã được loại trong code mới; không lấy khác biệt runtime giữa hai run để suy causal effect.

## Đề xuất từ kết quả này

Giữ opportunity-low ungated làm đối chứng nghiên cứu ở dense: nó có trung bình tốt nhất nhưng CI vẫn chứa0, chưa đủ để thay PPO. Chưa nên tăng futures nữa chỉ để kỳ vọng tăng reward. Hướng có ích hơn là kiểm tra lựa chọn quyết định và guard: dùng loss/selection phản ánh gain và downside khi định override, cùng hiệu chỉnh trên nhóm quyết định ứng viên. Đây là giả thuyết mới, không phải cải thiện đã chứng minh; lựa chọn trên dev/calibration mới, khóa trước một test panel mới. Skill cần đánh giá riêng vì hiện không có gain closed-loop dù MSE có cải thiện. Không triển khai experiment mới trong yêu cầu phân tích này.

## Verification và phạm vi

Checksum5190 artifacts khớp; đủ4032 episode/27 heads/17280 updates, config/seed split dựng lại khớp,8 module liên quan khớp commit. Khôi phục27 head, kiểm tra dev MSE/checkpoint selection và dựng lại18 margin/coverage criteria. Kiểm tra2160 draw rows;720 low/high member-epoch pairs có cùng draws và state table. Tính lại4737 high/low paired labels và48 root gains từ stored forecast. Tính độc lập18 registered intervals (bao gồm2 contrast được báo ở cả98,75% và95%) cùng6 posthoc intervals,12 root crossed bootstraps.

Replay vật lý168192 source transitions và10080 logged test transitions. Hai zero-override gated controller có1152 episode với mọi base/tail/extended physical metric trùng PPO;2880 logged gated state/action/following trùng PPO. Bounded scalar forecast replay30 files/60 paths/3194 transitions khớp costs, components và boundary states; không tái chạy58,8 triệu forecast transitions hoặc training. Passport giữ ANALYZED, không tuyên bố full training VERIFIED. Raw artifacts không sửa.

Derived files: `analysis.json`, `prediction_metrics.csv`, `gate_diagnostics.csv`, `validation.log`, `analyze.py`, `report.py`. Scripts dùng worktree venv đã pin và chỉ ghi report directory.

## Fallacy scan — 11/11

| Mẫu suy luận | Kiểm tra / giới hạn |
|---|---|
| Simpson’s paradox | Dense/skill trái dấu ungated/PPO; tách điều kiện và3 nguồn. |
| Ecological fallacy | Không suy mean episode thành mọi assignment; root/episode metrics tách riêng. |
| Berkson’s paradox | Root/labels có điều kiện trên source opportunity; báo denominator và không dùng làm gain toàn episode. |
| Collider bias | Primary cost dùng full grid; không điều kiện hóa theo override/harm để tính hiệu quả. |
| Base rate neglect | Báo487/148 cơ hội test, sample/profile/late calibration,5 và0 overrides. |
| Regression to mean | Không chọn test theo điểm cực đoan; checkpoint chọn dev; mọi state budget cố định. |
| Survivorship bias | Grid4032 đủ, không loại kết quả xấu;27 heads có training/checkpoint. |
| Look-elsewhere effect | Đủ4 primary,14 secondary,6 posthoc; không nâng mean tốt nhất thành significant. |
| Garden of forking paths | Protocol commit trước run; không hạ gate, đổi seeds hoặc kéo dài collection sau test. |
| Correlation != causation | Simulator intervention ghép tapes; H1 thay cả coverage và sampling, không quy riêng một thành phần/ngoại suy production. |
| Reverse causality | Train/dev/calibration trước sealed test; test không chọn checkpoint/gate. |
