# Hiệu chỉnh hệ số baseline: fit và heldout độc lập

**KHÔNG ĐẠT vòng chẩn đoán: 4/9 điều kiện.** Kết quả FULL cost âm trước đó được giữ nguyên; goal nghiên cứu chưa đạt.

## Giả thuyết và thiết kế

Giả thuyết: hệ số1 chưa phù hợp với gradient actor dùng chung; một hệ số control-variate ước lượng từ dữ liệu riêng có thể giảm nhiễu trên quỹ đạo mới. Công thức hệ số control-variate là prior art, không phải phát minh mới.

Dùng candidate đóng băng ở tất cả ba seed và năm checkpoint. Mỗi nhóm lấy128 episode fit, tính lambda=traceCov(gV,h)/traceVar(h), với h=gV−gB; đóng băng và hash lambda trước khi thu256 episode heldout khác. Hệ số không bị clip, regularize, quét hay chọn lại trên heldout. Nếu control variance≤1e−20 thì lambda0. Tất cả trọng số actor/critic giữ nguyên.

Tổng5.760 episode,276.480 quyết định,345.600 chuyển trạng thái và17.280 vector gradient đầy đủ; 0 optimizer update. Physical/action seeds mới và hai tập tách rời. Vẫn N4/K2, H48+tail12 và reward/health ban đầu. F64 giữ nguyên giá trị tensor checkpoint.

Mỗi seed phải có mean gain trên đủ năm checkpoint≥5%, ít nhất3/5 checkpoint cải thiện, và mean gain tốt hơn baseline hệ số1 trên cùng heldout trajectories. Step0 được kiểm tra V=B, hệ số0 và gain chính0. Đây là chẩn đoán điểm, chưa có CI hay bằng chứng về cập nhật PPO/cost.

## Kết quả heldout

| Seed | Gain hiệu chỉnh | Gain hệ số1 | Checkpoint cải thiện | Đạt cả3 |
|---|---:|---:|---:|---|
| 330000000 | 0.9622% | 2.1225% | 3/5 | Không |
| 330000001 | 20.8711% | 5.3934% | 4/5 | Có |
| 330000002 | -0.3797% | 3.5028% | 2/5 | Không |

| Seed | Checkpoint | Lambda từ fit | Gain trên fit | Gain heldout | Gain fixed1 heldout |
|---|---:|---:|---:|---:|---:|
| 330000000 | 0 | 0.000000 | 0.0000% | 0.0000% | 0.0000% |
| 330000000 | 122880 | 0.344362 | 0.1858% | 1.6537% | 4.0395% |
| 330000000 | 245760 | 4.732306 | 35.8655% | 7.2607% | 2.7225% |
| 330000000 | 368640 | 1.176976 | 0.8558% | 1.7676% | 1.5410% |
| 330000000 | 491520 | 6.948279 | 12.1416% | -5.8709% | 2.3097% |
| 330000001 | 0 | 0.000000 | 0.0000% | 0.0000% | -0.0000% |
| 330000001 | 122880 | 1.610559 | 5.4383% | 9.0615% | 6.6263% |
| 330000001 | 245760 | 31.996067 | 67.6420% | 53.6929% | 5.0106% |
| 330000001 | 368640 | 7.559330 | 52.4245% | 35.7976% | 13.5198% |
| 330000001 | 491520 | 3.818063 | 24.8183% | 5.8035% | 1.8103% |
| 330000002 | 0 | 0.000000 | 0.0000% | 0.0000% | 0.0000% |
| 330000002 | 122880 | 2.635271 | 5.6445% | 3.5514% | 3.0786% |
| 330000002 | 245760 | 2.823876 | 12.2039% | -9.1382% | -1.9821% |
| 330000002 | 368640 | -0.112891 | 0.0276% | -1.1650% | 8.9553% |
| 330000002 | 491520 | 0.631207 | 0.5883% | 4.8533% | 7.4620% |

## Kết luận có giới hạn

Hướng scalar calibration không vượt đủ điều kiện trên dữ liệu mới. Theo protocol, loại cấu hình này; không thử lại bằng clip/regularization, thêm episode, đổi seed hay chọn checkpoint tốt. Một mức giảm trên tập fit không đủ để cứu kết quả heldout.

Bộ kiểm chứng độc lập replay toàn bộ345.600 chuyển trạng thái, action draws, returns/baselines và17.280 vector gradient (192.516.480 tọa độ); tính lại hệ số chỉ từ fit và áp dụng đúng hệ số ấy trên heldout. Audit PASS xác nhận các số liệu, không thay cho thành công khoa học. Scientific confirmation/test và N lớn vẫn đóng.

Nguồn: `/home/bachng/ht-pdm-fjsp/artifacts/technician_broker_removal_coefficient_calibration_v1_20261017T050521333680Z`.
Kiểm chứng: `/home/bachng/ht-pdm-fjsp/artifacts/technician_broker_removal_calibration_verification_v1_20261017T050724270518Z`.

![Fit và heldout](/home/bachng/ht-pdm-fjsp/artifacts/technician_broker_removal_calibration_report_v1_20261017T050913188687Z/fit_heldout_calibration.png)
