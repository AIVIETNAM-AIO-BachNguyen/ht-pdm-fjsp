## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-11T17:17:38Z
- Verification Status: ANALYZED
- Version Label: validation_v1

# Phân tích maintenance_assignment_confidence_v1

**Kết luận:** chưa có bằng chứng thuật toán mới giảm chi phí so với PPO. Hai bản gated trở thành PPO vì không thực hiện lần đổi phân công nào. Dữ liệu probe tăng độ phủ nhưng chưa chuyển thành cải thiện trên trajectory thực.

Run: `/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_assignment_confidence_v1_full_cpu_20261011T165402Z`. Commit `5c4deb15a5242f237549e4ce178167ddc4ed971f`. Hoàn thành trong 16.91 phút; CPU. 3 PPO nguồn × 24 profile test × 4 shock × 2 điều kiện × 5 controller = 2.880 episode. Mỗi điều kiện/controller có 288 episode. Objective là tổng chi phí 48 bước + 12 bước tail; thấp hơn tốt hơn.

## Kết quả closed-loop

| Controller | Dense: chi phí | Dense: giảm so PPO | Skill mask: chi phí | Skill mask: giảm so PPO | Đổi assignment dense / skill |
|---|---:|---:|---:|---:|---:|
| frozen_ppo | 1097.736 | +0.000% | 1139.944 | +0.000% | 0 / 0 |
| natural_ungated | 1096.542 | +0.109% | 1136.733 | +0.282% | 131 / 48 |
| natural_gated | 1097.736 | +0.000% | 1139.944 | +0.000% | 0 / 0 |
| broad_ungated | 1099.826 | -0.190% | 1137.062 | +0.253% | 123 / 41 |
| broad_gated | 1097.736 | +0.000% | 1139.944 | +0.000% | 0 / 0 |

`natural`: chỉ dữ liệu trạng thái PPO gặp tự nhiên. `broad`: một nửa lượt lấy mẫu từ natural, một nửa từ probe tổng hợp ở thời điểm muộn. `gated`: chỉ đổi assignment khi lợi ích dự đoán trừ một SD ensemble vượt margin hiệu chỉnh. PPO chọn tập máy PM và mọi cạnh CM tại từng can thiệp; head chỉ đổi technician. Trajectory sau đó có thể đổi cả lịch PM/CM.

## Bốn so sánh chính

20.000 crossed bootstrap, lấy mẫu lại source và profile; 4 shock được lấy trung bình trong mỗi ô source/profile. CI 98,75% cho bốn so sánh chính, Bonferroni xấp xỉ; ngưỡng lợi ích thực dụng 1%. Chỉ 3 source nên bootstrap không khắc phục được hạn chế số replica.

| So sánh | Giảm chi phí | CI 98,75% |
|---|---:|---:|
| dense:broad_gated_vs_frozen_ppo | +0.000% | [+0.000%; +0.000%] |
| dense:broad_gated_vs_broad_ungated | +0.190% | [-0.431%; +1.099%] |
| skill_mask:broad_gated_vs_frozen_ppo | +0.000% | [+0.000%; +0.000%] |
| skill_mask:broad_gated_vs_broad_ungated | -0.253% | [-1.110%; +0.080%] |

Không so sánh nào chứng minh lợi ích vượt 1%. CI [0;0] của gated/PPO xuất phát từ cùng hành vi trong mẫu này, không phải bằng chứng mọi môi trường đều tương đương hoặc mô hình đã học cách tránh riêng quyết định xấu. CI broad_gated/broad_ungated vẫn chứa 0 ở cả hai điều kiện; dấu âm ở skill không chứng minh bị hại.

## So sánh bổ sung

| So sánh | Giảm chi phí | CI 95% |
|---|---:|---:|
| dense:broad_ungated_vs_natural_ungated | -0.300% | [-0.731%; +0.033%] |
| skill_mask:broad_ungated_vs_natural_ungated | -0.029% | [-0.156%; +0.048%] |
| dense:natural_ungated_vs_frozen_ppo_posthoc | +0.109% | [-0.474%; +0.645%] |
| dense:broad_ungated_vs_frozen_ppo_posthoc | -0.190% | [-0.832%; +0.293%] |
| skill_mask:natural_ungated_vs_frozen_ppo_posthoc | +0.282% | [-0.019%; +0.822%] |
| skill_mask:broad_ungated_vs_frozen_ppo_posthoc | +0.253% | [-0.033%; +0.837%] |

Các so sánh với PPO được tính bổ sung sau khi mở test, không phải tiêu chí chính đã khóa. Mọi CI trên đều chứa 0. Broad/natural cũng chưa chứng minh lợi ích; dense broad kém natural ở trung bình của cả ba source. Không so trực tiếp chi phí tuyệt đối với experiment cũ vì panel test mới.

## Vì sao gate không hoạt động?

Dense: mỗi source/arm có 34 so sánh hiệu chỉnh. Margin tương đương khoảng 52,49–76,53 đơn vị chi phí (giá trị lưu được chia 20). Lợi ích dự đoán sau trừ SD lớn nhất trên các PM-choice test chỉ khoảng 8,74–16,15 với broad, hoặc 11,17–13,79 với natural. Không có lựa chọn nào vượt ngưỡng. Mỗi gated controller từ chối toàn bộ 566 cơ hội dense.

Skill mask: chỉ có 8, 7, 3 so sánh hiệu chỉnh theo ba source; thấp hơn mức tối thiểu 10. Gate bị vô hiệu hóa theo thiết kế, nên từ chối toàn bộ 149 cơ hội. Đây là thiếu dữ liệu hiệu chỉnh cho assignment mà PPO thực sự chọn, dù dữ liệu có nhiều nhãn cho các tập máy khác.

Margin cộng độ bi quan từ cả SD ensemble và 1,96 MCSE của paired forecast. Đây là guard thực nghiệm, không phải bảo đảm an toàn hay xác suất tin cậy 90%. Zero override đồng thời loại bỏ lựa chọn có lợi và có hại; không đánh giá được chất lượng chọn lọc của guard.

## Dữ liệu đa giai đoạn đã làm được gì?

Natural train có 2.886 nhãn pair, 2.496 (86,49%) ở tick 0. Probe thêm 8.091 nhãn ở tick 6/18/30/42. Tỷ lệ lượt huấn luyện ở tick >0 tăng từ 12,07% (natural) lên 87,49% (broad), với cùng 80 epoch × 1.024 draws/head. Tuy nhiên probe là trạng thái tổng hợp sau prefix thật, không phải phân phối PPO gặp tự nhiên.

| MSE ensemble trên nhãn pair | Natural head | Broad head |
|---|---:|---:|
| dense, development, natural | 1.3721 | 1.4847 |
| dense, development, probe | 1.0055 | 1.0014 |
| dense, calibration, natural | 2.3757 | 2.3809 |
| dense, calibration, probe | 2.0682 | 2.1158 |
| skill_mask, development, natural | 0.9255 | 1.2215 |
| skill_mask, development, probe | 0.9001 | 0.8762 |
| skill_mask, calibration, natural | 3.9358 | 4.0379 |
| skill_mask, calibration, probe | 1.6405 | 1.7605 |

MSE theo đơn vị target chuẩn hóa, tính từ ensemble trung bình; mô tả theo số nhãn, không coi các nhãn cùng state là mẫu độc lập. Broad chỉ cải thiện nhẹ dev probe; kém hơn trên dev natural và mọi ô calibration. Độ phủ tăng chưa đủ giúp dự đoán tốt trên trạng thái tự nhiên. Đây là dấu hiệu cần xem lại phân phối dữ liệu/độ nhiễu target, chưa xác định nguyên nhân duy nhất.

## Chẩn đoán một can thiệp

| Điều kiện / origin | Cơ hội | Natural: override / gây hại | Broad: override / gây hại | Natural / Broad: gain trung bình mỗi cơ hội |
|---|---:|---:|---:|---:|
| dense / natural | 24 | 3 / 1 | 2 / 1 | -0.369 / -0.936 |
| dense / probe | 72 | 13 / 5 | 10 / 5 | +0.839 / -0.017 |
| skill_mask / natural | 8 | 1 / 1 | 1 / 1 | -0.336 / -0.336 |
| skill_mask / probe | 13 | 4 / 0 | 4 / 0 | +12.389 / +12.389 |

384 root states; mỗi condition/origin 96 states, nhưng chỉ một phần có assignment thay thế. Tất cả CI 95% gain root đều chứa 0. Dense probe: broad gây hại lớn nhất 52,25 đơn vị chi phí; dense natural: cả hai có một override gây hại 39,64. Skill natural chỉ 8 cơ hội; không đủ để kết luận ổn định. Root gain là một lần đổi assignment rồi tiếp tục PPO với 64 paired futures, không phải phần trăm cải thiện cả episode; reference độc lập select/evaluate không phải oracle.

## Lưu ý đo tốc độ

Phát hiện inefficiency tại `source_snapshot/maintenance_assignment_confidence_policy.py:68`: near-tie kiểm tra cả candidate gốc (gain luôn 0). Với ungated, threshold=1e-5 và tolerance near-tie=2e-5 nên luôn kích hoạt scalar recompute. Gated skill với margin null cũng vậy. Latency hiện tại khoảng 1,61–1,63 ms so với PPO 0,217–0,219 ms; dense gated khoảng 0,482–0,484 ms. Đây là latency amortized batch 32 và bị ảnh hưởng bởi nhánh fallback thừa; không dùng để kết luận chi phí inference bản tối ưu. Không thấy lỗi này giải thích zero override: gate logic vẫn đúng. Chưa sửa code trong yêu cầu phân tích này.

## Hướng thử tiếp theo (đề xuất, chưa triển khai)

Ưu tiên thu thập trạng thái tự nhiên thực sự có ít nhất hai assignment cho đúng tập PM mà PPO chọn, ở từng condition/source và nhiều profile. Bổ sung calibration đủ mẫu ở skill; đánh giá độ phủ, MCSE và tỷ lệ override trên development trước khi mở test mới. Với cùng budget, dành thêm paired futures cho nhãn gần ranh giới quyết định để kiểm tra xem MC noise có làm margin quá lớn không. Nếu đổi gate/quantile, chỉ lựa chọn trên development và calibration mới, khóa trước một panel test mới; không hạ ngưỡng để khớp test vừa mở. Chưa có cơ sở tăng độ phức tạp kiến trúc từ kết quả này.

## Kiểm tra và giới hạn tái lập

Raw run giữ nguyên. Kiểm tra checksum toàn bộ 2.640 artifact; source contract 957 file và 3 PPO; code experiment liên quan khớp commit Git. Tính lại đủ thống kê episode, 16 bootstrap chẩn đoán root, 13.685 target/MCSE từ forecast, lựa chọn 18 head theo dev MSE và margin calibration. Kiểm tra 1.440 dòng sampling theo budget; không chạy lại optimizer.

Replay vật lý: 25.344 transition nguồn + 4.224 prefix probe + 7.200 transition test đã log. Xác nhận 1.152 gated episode có mọi base/tail/extended physical metric trùng PPO, và 2.880 quyết định gated đã log trùng action/state/following của PPO. Replay scalar forecast phân tầng 60 file, 120 future paths, 6.408 transition khớp cost, components, cutoff/terminal state; không tái chạy toàn bộ 30.378.720 forecast transition hay training. Vì vậy passport vẫn ANALYZED, không tuyên bố full run VERIFIED.

Script: `analyze.py`, `report.py`; dữ liệu dẫn xuất: `analysis.json`, `prediction_metrics.csv`, `gate_diagnostics.csv`; log: `validation.log`. Python cần numpy/torch/tqdm trong worktree venv.

## Fallacy scan — 11/11 đã kiểm tra

| Mẫu sai suy luận | Kiểm tra / giới hạn |
|---|---|
| Simpson’s paradox | Tách condition/source; dense và skill broad/PPO trái dấu nên không gộp thành claim chung. |
| Ecological fallacy | Không suy từ mean episode sang mọi state/assignment. |
| Berkson’s paradox | Root diagnostic có điều kiện trên cơ hội; báo denominator và tách probe/natural. |
| Collider bias | Kết quả chính dùng toàn bộ grid; không điều kiện hóa vào override để ước lượng gain episode. |
| Base rate neglect | Báo 566/149 cơ hội, zero overrides và 34/8/7/3 calibration samples. |
| Regression to the mean | Không chọn test profile theo kết quả cực đoan; checkpoint chọn dev. |
| Survivorship bias | Grid đủ 2.880 episode, không loại thất bại/xấu; head đủ 18. |
| Look-elsewhere effect | Báo đủ 4 primary; so sánh bổ sung là exploratory/posthoc, không claim significant. |
| Garden of forking paths | Code/protocol đã khóa trước run; không retune ngưỡng sau test; đề xuất cần test mới. |
| Correlation / causation | Can thiệp ghép physical tapes trong simulator; không suy rộng ra hệ thống thật hoặc quy nguyên nhân duy nhất cho dữ liệu probe. |
| Reverse causality | Train/dev/calibration trước test; không dùng test gain chọn checkpoint/gate; định hướng dữ liệu không đảo thời gian. |
