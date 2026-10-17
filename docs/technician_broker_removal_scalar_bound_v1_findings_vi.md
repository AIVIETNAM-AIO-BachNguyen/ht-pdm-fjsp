# Giới hạn trộn scalar baseline: phân tích hậu kiểm

**Chỉ là giới hạn tối ưu trên chính mẫu đã có; không phải kết quả held-out hay cải thiện policy.** Công thức giải tích chọn một hệ số cho mỗi checkpoint, không quét hyperparameter. Các kết quả âm trước đây giữ nguyên.

| Seed | Mức giảm phương sai tối ưu trung bình đủ5 checkpoint |
|---|---:|
| 330000000 | 5.7139% |
| 330000001 | 31.5966% |
| 330000002 | 6.2562% |

Đủ tiềm năng trên mẫu để cân nhắc protocol fit/holdout mới; chưa được học hay triển khai hệ số này.

| Seed | Checkpoint | Hệ số tối ưu | Giảm phương sai trên mẫu |
|---|---:|---:|---:|
| 330000000 | 0 | 0.000000 | 0.0000% |
| 330000000 | 122880 | 2.335944 | 8.5753% |
| 330000000 | 245760 | 2.334662 | 7.7896% |
| 330000000 | 368640 | -14.025571 | 8.8158% |
| 330000000 | 491520 | -4.254638 | 3.3890% |
| 330000001 | 0 | 0.000000 | 0.0000% |
| 330000001 | 122880 | 2.768758 | 10.3094% |
| 330000001 | 245760 | 23.063261 | 67.3828% |
| 330000001 | 368640 | 7.044068 | 70.3411% |
| 330000001 | 491520 | 5.025908 | 9.9495% |
| 330000002 | 0 | 0.000000 | 0.0000% |
| 330000002 | 122880 | 2.560605 | 7.2118% |
| 330000002 | 245760 | -0.179425 | 0.0119% |
| 330000002 | 368640 | 1.813811 | 9.3622% |
| 330000002 | 491520 | 4.482759 | 14.6951% |

Đây là control-variate cổ điển, không phải đóng góp thuật toán mới. Fit và đánh giá dùng cùng dữ liệu nên hệ số có thể overfit; không dùng làm bằng chứng xác nhận cơ chế. Không mở scientific test, confirmation hay N lớn.

Nguồn: `/home/bachng/ht-pdm-fjsp/artifacts/technician_broker_removal_gradient_diagnostic_v1_20261017T045418012781Z`.
