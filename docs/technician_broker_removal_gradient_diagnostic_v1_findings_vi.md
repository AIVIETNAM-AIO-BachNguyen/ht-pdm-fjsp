# Removal-credit: chẩn đoán phương sai gradient actor

**KHÔNG ĐẠT ngưỡng chẩn đoán đã khóa.** Kết quả cost âm của FULL trước đó được giữ nguyên. Đây là phân tích mô tả, chưa có CI và không chứng minh một cập nhật PPO hay cost tốt hơn.

## Câu hỏi và thiết kế

Trên cùng policy đóng băng, baseline removal có giảm phương sai gradient của toàn bộ actor dùng chung, thay vì chỉ giảm phần dư return? Dùng tất cả ba seed và năm checkpoint của candidate, mỗi nhóm128 episode mới. Tổng1.920 episode/92.160 quyết định/115.200 chuyển trạng thái; 0 cập nhật trọng số. Tensor checkpoint float32 được giữ nguyên giá trị khi chuyển float64 để lấy mẫu và đo gradient. Không teacher hay fine-tuning actor.

Trên từng episode tính ba vector gradient11.141 chiều: physical factors với V(s), physical factors với conditional removal B_i, và proposal scores với actual removal b_i. Mọi vector dùng cùng quỹ đạo và return thật, chia chung48×2; không clipping, entropy hay chuẩn hóa advantage. Đơn vị thống kê là episode, không coi48 tick tương quan là mẫu độc lập.

Tiêu chí chính: mỗi seed giảm trung bình ít nhất5% trace covariance trên đủ năm checkpoint và cải thiện ở ít nhất3/5 checkpoint. Step0 có V=B theo thiết kế, được kiểm tra và đặt gain chính bằng0 để sai số làm tròn không được tính là học thành công.

## Kết quả

| Seed | Giảm phương sai trung bình | Checkpoint cải thiện | Đạt cả hai |
|---|---:|---:|---|
| 330000000 | 1.5864% | 2/5 | Không |
| 330000001 | 6.7877% | 4/5 | Có |
| 330000002 | 3.4666% | 3/5 | Không |

| Seed | Checkpoint | Gain gradient B/V | Gain B/proposal-b | Ratio phần dư return |
|---|---:|---:|---:|---:|
| 330000000 | 0 | 0.0000% | 17.4004% | 1.000000 |
| 330000000 | 122880 | 5.7705% | 44.4281% | 0.975885 |
| 330000000 | 245760 | 5.2439% | 49.2270% | 0.968249 |
| 330000000 | 368640 | -1.3019% | 29.8998% | 0.973940 |
| 330000000 | 491520 | -1.7803% | 21.5845% | 0.977404 |
| 330000001 | 0 | 0.0000% | 14.2429% | 1.000000 |
| 330000001 | 122880 | 6.1021% | 20.8445% | 0.971956 |
| 330000001 | 245760 | 5.7166% | 16.5913% | 0.968209 |
| 330000001 | 368640 | 18.5541% | 39.3430% | 0.952384 |
| 330000001 | 491520 | 3.5654% | 79.0112% | 0.974077 |
| 330000002 | 0 | 0.0000% | 12.8344% | 1.000000 |
| 330000002 | 122880 | 4.5330% | 20.9384% | 0.973295 |
| 330000002 | 245760 | -0.5024% | 51.0351% | 0.974963 |
| 330000002 | 368640 | 7.4775% | 74.7667% | 0.970382 |
| 330000002 | 491520 | 5.8250% | 64.6759% | 0.979136 |

## Diễn giải có giới hạn

Chỉ một trong ba seed đạt mức giảm phương sai trung bình5%. Ở seed330000000, hai checkpoint cuối có phần dư return nhỏ hơn nhưng phương sai gradient lớn hơn khoảng1,30% và1,78%. Vì vậy việc giảm MSE/variance của return không tự động đem lại tín hiệu actor tốt hơn. Đây là bằng chứng mô tả trên panel này, không phải chứng minh tổng quát hay kết luận mọi baseline removal đều vô ích.

So sánh B với proposal-b có mức giảm lớn hơn, khoảng12,8–79,0% tùy checkpoint. Đó là hiệu ứng của marginalization/Rao-Blackwell so với proposal estimator; đối chứng chính của nghiên cứu đã dùng physical factors. Không được lấy lợi ích này thay cho đóng góp thêm của critic mới so với factor-state.

Không thấy lỗi toán học trong phép đo: kiểm chứng độc lập tái dựng toàn bộ115.200 chuyển trạng thái và return, mọi action draw/baseline, cùng5.760 vector gradient (64.172.160 tọa độ). Reference dùng MLP thủ công và tổng25 latent proposals, chỉ cho gradient đi qua xác suất của technician đang xét; khác với VJP NumPy đóng của lần chạy nguồn. Audit PASS xác nhận phép đo, không thay đổi verdict âm.

Việc học baseline theo gradient variance đã có prior art: [Greensmith et al., JMLR 2004](https://www.jmlr.org/papers/v5/greensmith04a.html) phân tích baseline/value function tối ưu cho variance; [Kuba et al., NeurIPS 2021](https://papers.nips.cc/paper_files/paper/2021/hash/6fe6a8a6e6cb710584efc4af0c34ce50-Abstract.html) nghiên cứu optimal baseline trong MARL. Do đó không coi việc thêm score weight hay đổi critic loss đơn thuần là phát minh mới.

Hướng cần kiểm tra trước một vòng học mới: trong cấu trúc broker và actor dùng chung này, baseline nào giảm được trace covariance trên quỹ đạo chưa dùng để fit? Phải tách fit/đánh giá và dùng đối chứng cùng thông tin/capacity; không chỉ giảm variance trên dữ liệu fit. Chưa có thí nghiệm mới cho hướng đó ở báo cáo này. Không tự động mở CF/test/N lớn, tăng budget, chọn seed hay checkpoint khác.

Nguồn: `/home/bachng/ht-pdm-fjsp/artifacts/technician_broker_removal_gradient_diagnostic_v1_20261017T045418012781Z`.
Kiểm chứng: `/home/bachng/ht-pdm-fjsp/artifacts/technician_broker_removal_gradient_verification_v1_20261017T045701510650Z`.

![Phương sai gradient](/home/bachng/ht-pdm-fjsp/artifacts/technician_broker_removal_gradient_report_v1_20261017T045933678512Z/gradient_variance.png)
