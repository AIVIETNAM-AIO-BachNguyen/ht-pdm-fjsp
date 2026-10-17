# Removal-credit PPO v1: phương pháp và kết quả DEV

**KHÔNG ĐẠT vòng sàng lọc DEV; giữ nguyên kết quả âm và chưa mở confirmation/test.** 17/19 tiêu chí điểm đạt. Đây là kết quả mô tả trên DEV, chưa có khoảng tin cậy và chưa đạt mục tiêu nghiên cứu tổng thể.

## Giả thuyết và thuật toán

Giả thuyết: thông tin về đề xuất của các technician còn lại giúp baseline phân bổ tín hiệu học tốt hơn, qua đó giảm cost so với cùng policy chỉ dùng V(s). Tất cả actor và critic học từ khởi tạo ngẫu nhiên; không teacher, distillation hay trọng số PPO đã huấn luyện.

Mỗi technician đề xuất WAIT hoặc một máy từ cùng trạng thái thô đầy đủ. Broker cố định chọn technician có restoration/duration tốt hơn khi tranh chấp, hòa thì chọn ID nhỏ hơn; người thua chờ và không thử lại. Policy lõi không liệt kê toàn bộ matching.

Critic ψ dự đoán return thực của policy từ trạng thái và matching khi thay đề xuất technician i bằng WAIT. Đây là baseline dùng return thực để học; không mô phỏng một thế giới thiếu technician và không dùng nhãn cost tương lai. Lấy kỳ vọng baseline trên những đề xuất ẩn cùng tạo ra matching đã quan sát để được B_i. Tính và detach B_i trước optimizer của batch hiện tại, giữ nguyên qua các epoch PPO.

Actor dùng ratio theo từng thừa số của xác suất matching vật lý và advantage riêng (G−B_i), với clip PPO và chuẩn hóa chung dựa trên G−V(s). Đây là surrogate suy ra từ PPO; không tuyên bố giữ nguyên scalar joint PPO hay đảm bảo cải thiện đơn điệu. Critic phụ chỉ dùng trong huấn luyện, suy luận dùng actor và broker.

Sáu nhánh lõi có cùng actor, hai critic, thông tin, tham số 29383, khởi tạo và ngân sách. Đối chứng chính factor-state vẫn huấn luyện critic phụ nhưng actor chỉ dùng V(s); blind-removal xóa kênh matching khi học và truy vấn. Các nhánh proposal và joint kiểm tra cách gán log-probability. Capacity PPO giữ actor 21 matching của PPO normalized và thêm critic phụ; vẫn có khác biệt số tham số 28375 so với 29383. MAPPO local khác thông tin; MAPPO full nhận đầy đủ trạng thái.

## Thiết kế đã khóa

N4/K2, H48 quyết định và H12 tail chung; ba seed huấn luyện mới ×11 nhánh, mỗi mô hình 491520 quyết định. Tổng 16.220.160 quyết định, 337.920 episode và 253.440 Adam update. Đánh giá 32 DEV seeds ở năm checkpoint gồm bước0. Kết luận chính dùng cố định checkpoint 491520, không dùng checkpoint tốt nhất. 19 tiêu chí gồm cải thiện≥0.5% so với factor-state, cải thiện dương so với blind/local/full MAPPO, không kém hơn1% so với các đối chứng cost còn lại, cùng9 giới hạn sức khỏe. Thêm điều kiện cả ba seed thắng đối chứng chính, không assignment sai và p95≤50ms.

## Kết quả cố định cuối kỳ

| Nhánh | Cost/máy H60 | Availability H48 | Pending/máy H48 | Failure/máy H48 |
|---|---:|---:|---:|---:|
| Removal RB | 262.075521 | 0.668891 | 0.000000 | 0.250000 |
| Factor + state V | 261.653646 | 0.667969 | 0.000000 | 0.205729 |
| Blind removal | 261.500000 | 0.668728 | 0.000000 | 0.218750 |
| Proposal + removal | 261.375000 | 0.667969 | 0.000000 | 0.187500 |
| Proposal + state V | 261.791667 | 0.667860 | 0.000000 | 0.210938 |
| Joint q + state V | 261.479167 | 0.668240 | 0.000000 | 0.203125 |
| PPO | 261.898438 | 0.668837 | 0.000000 | 0.138021 |
| PPO normalized | 261.023438 | 0.670410 | 0.000000 | 0.078125 |
| PPO + shadow critic | 261.023438 | 0.670410 | 0.000000 | 0.078125 |
| MAPPO local | 271.023438 | 0.670085 | 0.002604 | 0.700521 |
| MAPPO full | 263.263021 | 0.668945 | 0.000000 | 0.294271 |

| Tiêu chí | Giá trị điểm | Đạt |
|---|---:|---|
| gain_vs_factor_state_ppo | -0.00161234 | Không |
| gain_vs_blind_removal_ppo | -0.00220084 | Không |
| gain_vs_proposal_removal_ppo | -0.00268014 | Có |
| gain_vs_proposal_state_ppo | -0.00108428 | Có |
| gain_vs_joint_state_ppo | -0.00228069 | Có |
| gain_vs_ppo | -0.00067615 | Có |
| gain_vs_normalized_ppo | -0.00403061 | Có |
| gain_vs_capacity_ppo | -0.00403061 | Có |
| gain_vs_mappo | 0.03301529 | Có |
| gain_vs_full_mappo | 0.00451070 | Có |
| availability_delta_vs_ppo | 0.00005425 | Có |
| normalized_pending_delta_vs_ppo | 0.00000000 | Có |
| normalized_failures_delta_vs_ppo | 0.11197917 | Có |
| availability_delta_vs_normalized_ppo | -0.00151910 | Có |
| normalized_pending_delta_vs_normalized_ppo | 0.00000000 | Có |
| normalized_failures_delta_vs_normalized_ppo | 0.17187500 | Có |
| availability_delta_vs_capacity_ppo | -0.00151910 | Có |
| normalized_pending_delta_vs_capacity_ppo | 0.00000000 | Có |
| normalized_failures_delta_vs_capacity_ppo | 0.17187500 | Có |

Gain chính từng seed: 330000000: -0.594811%, 330000001: 0.035868%, 330000002: 0.074487%.
p95 actor+broker: 0.365221ms, đo 432 lượt sau khi worker huấn luyện đóng. Đây là độ trễ trên máy lab N4, chưa chứng minh scaling.

Tiêu chí điểm không đạt: gain_vs_factor_state_ppo, gain_vs_blind_removal_ppo. Điều kiện bổ sung không đạt: every_replica_main_gain_positive.

## Bằng chứng và giới hạn

Bộ kiểm chứng độc lập tái dựng 33 bộ tensor khởi tạo, ngân sách và seed; replay 1.088 episode/65.280 chuyển trạng thái vật lý và 52.224 quyết định policy; kiểm tra actor bằng NumPy, baseline conditional bằng tổng đủ 25 cặp đề xuất, cùng 19 phép tính tiêu chí và 432 mẫu timing. Audit PASS xác nhận tính đúng của báo cáo, không biến một thí nghiệm không đạt thành thành công. Các minibatch và tensor cache lịch sử không được lưu đầy đủ nên chưa replay độc lập từng optimizer update.

Đồ thị residual variance, context sensitivity và fallback mass là chẩn đoán mô tả. Chúng không tự chứng minh cơ chế có ích; bằng chứng chính phải đến từ ablation cùng thông tin/ngân sách và kết quả cost. Trọng số matching khác 0 cũng chưa đủ kết luận phối hợp thành công.

Không có CI trên test mới, không tuyên bố thắng PPO về hiệu năng, không đánh giá generalization hay mở rộng N/K ở vòng này. Nếu DEV không đạt, không tự động thử lại learning rate/entropy/seed/budget; vòng tiếp theo cần nguyên nhân cụ thể và protocol mới trước khi chạy. Các kết quả âm trước đây vẫn giữ nguyên.

Artifact nguồn: `/home/bachng/ht-pdm-fjsp/artifacts/technician_broker_removal_ppo_v1_development_20261017T042628768991Z`.
Audit độc lập: `/home/bachng/ht-pdm-fjsp/artifacts/technician_broker_removal_verification_v1_20261017T044756329259Z`.

![Chất lượng và sức khỏe](/home/bachng/ht-pdm-fjsp/artifacts/technician_broker_removal_report_v1_20261017T044819750990Z/fixed_final_quality_health.png)
![Learning và cơ chế](/home/bachng/ht-pdm-fjsp/artifacts/technician_broker_removal_report_v1_20261017T044819750990Z/learning_and_mechanism.png)

## Đánh giá sau kiểm chứng (2026-10-17)

Kết quả âm được giữ nguyên: 17/19 tiêu chí điểm đạt, nhưng hai tiêu chí chính về credit/co-action đều không đạt, và seed đầu tiên kém đối chứng chính 0,595%. Hai seed còn lại chỉ tốt hơn 0,036% và 0,074%. Không mở confirmation/test hay mở rộng N/K.

So với PPO normalized, candidate cao hơn cost 0,403%; so với PPO gốc cao hơn 0,068%. Candidate tốt hơn MAPPO local 3,302% và full-information MAPPO 0,451%, nhưng các chênh lệch này chưa chứng minh cơ chế mới. Chín giới hạn sức khỏe đạt; failure/máy H48 vẫn cao hơn PPO normalized 0,171875, dù còn trong allowance +0,25 đã khóa. P95 actor+broker 0,365ms ở N4, chưa phải bằng chứng scaling.

Critic mới thực sự đã học: norm trọng số context ở ba seed khoảng 0,357–0,427, độ nhạy matching cuối kỳ khoảng 0,301 SD; bản blind bằng 0. Residual variance của actor advantage giảm trung bình khoảng 2,717% trong 1/4 cuối ngân sách. Tuy nhiên cost không giảm. Vì vậy “critic có học” và “residual variance nhỏ hơn” chưa đủ để ủng hộ giả thuyết giúp quyết định tốt hơn. Không thấy sai số đại số/cache/khởi tạo qua các kiểm tra hiện có; chưa có bằng chứng để tự động cứu hướng này bằng learning rate hay tăng budget.

Câu hỏi chẩn đoán tiếp theo: giảm scalar-return residual có đi kèm giảm phương sai gradient actor thật hay không? Cần dùng cùng policy đóng băng, cùng quỹ đạo mới và cùng thông tin khi so V(s) với removal baseline; không học thêm, không đổi surrogate và không dùng test scientific. Đây là bước kiểm tra nguyên nhân trước khi cân nhắc một protocol học mới.
