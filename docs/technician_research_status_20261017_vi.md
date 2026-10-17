# Trạng thái nghiên cứu technician N4/K2 — 2026-10-17

**Mục tiêu tổng thể chưa đạt.** Đã hoàn tất FULL học từ đầu và hai chẩn đoán
độc lập; loại nhánh removal-credit v1 và cách cứu bằng scalar calibration theo
các điều kiện đã khóa. Scientific confirmation/test và N lớn vẫn đóng. Không
có teacher, distillation hay PPO warm-start trong lần học này.

## Kết quả chính

| Nhánh học từ đầu | Cost/máy H60, fixed final |
|---|---:|
| Removal RB candidate | 262.075521 |
| Factor + state V, đối chứng chính cùng thông tin/capacity | 261.653646 |
| Blind removal | 261.500000 |
| PPO gốc | 261.898438 |
| PPO normalized / capacity PPO | 261.023438 |
| MAPPO local | 271.023438 |
| MAPPO full information | 263.263021 |

33 mô hình ×491.520 quyết định, tổng16.220.160 quyết định và253.440 Adam
update. Ba training seeds mới,32 DEV seeds, fixed final491.520; không đổi sang
checkpoint đẹp hơn. Candidate kém đối chứng chính0,161% và blind0,220%; chỉ
17/19 tiêu chí điểm đạt, điều kiện thắng mỗi training seed cũng không đạt.
Chín giới hạn sức khỏe đạt, không assignment sai; p95 actor+broker0,365ms ở N4.
Các số DEV chưa có CI nên không tuyên bố thắng hay tương đương PPO về mặt
thống kê, dù chênh lệch điểm cost nhỏ. Audit độc lập PASS16,30s.

[Phương pháp, kiểm chứng và đầy đủ19tiêu chí](technician_broker_removal_ppo_v1_findings_vi.md).

## Đã kiểm tra nguyên nhân, thay vì thêm encoder/LR/attention

Critic context có học, baseline thay đổi; return-residual variance cuối kỳ giảm
khoảng2,7%. Nhưng gradient actor có thật sự bớt nhiễu hay không là câu hỏi khác.
Chẩn đoán1.920 episode mới với policy đóng băng đo đầy đủ11.141 tọa độ actor:
mean variance gain trên tất cả năm checkpoint là1,59%/6,79%/3,47%; chỉ một seed
đạt mức5% đã khóa. Một số checkpoint giảm scalar residual mà tăng gradient
variance. Bộ kiểm chứng riêng replay5.760 gradient vectors và toàn bộ115.200
chuyển trạng thái, PASS30,06s.

[Chẩn đoán phương sai gradient và ranh giới prior art](technician_broker_removal_gradient_diagnostic_v1_findings_vi.md).

Đã tính thêm giới hạn giải tích cho scalar mixing trên chính dữ liệu chẩn đoán.
Mức tối ưu trên mẫu còn tiềm năng, nhưng đó là fit trên dữ liệu đánh giá. Vì vậy
đã khóa một phép kiểm tra mới:128fit rồi256heldout độc lập cho mỗi checkpoint;
hash hệ số trước khi thu heldout, không clip/regularize/quét hệ số hay refit.

FULL kiểm tra này gồm5.760 episode mới,276.480 quyết định,345.600 chuyển trạng
thái và0optimizer update. Heldout mean gains0,96%/20,87%/−0,38%; chỉ4/9điều kiện
đạt. Nhánh scalar calibration bị loại. Kiểm chứng độc lập17.280 gradient vectors
(192.516.480tọa độ), physics, fit-only coefficient và heldout verdict PASS88,88s.

[Kết quả fit/heldout và toàn bộ15checkpoint](technician_broker_removal_coefficient_calibration_v1_findings_vi.md).

## Quyết định nghiên cứu

Không chạy thêm biến thể hệ số, clip, regularization, seed hay budget để cứu các
nhánh vừa bị loại. Không dùng seed thứ hai tốt làm kết quả đại diện. Tất cả
protocol, mã đã commit/push, artifact mới và kết quả âm được giữ nguyên.

Kết quả hiện hỗ trợ tính đúng của broker-marginal/removal identities và phép đo
variance; chưa hỗ trợ câu chuyện “thêm removal critic thì quyết định tốt hơn”.
Lợi ích Rao-Blackwell so với proposal-score estimator không thay cho lợi ích
critic mới so với đối chứng đã dùng physical factors. Hệ số control-variate hay
score-weighted baseline cũng đã có prior art, không thể tự nhận là đóng góp mới.

Hướng nghiên cứu tiếp phải tác động được lên phân công và có giả thuyết riêng
được đo trước khi chạy học mới. Cần đối chiếu các thử nghiệm ordered-prefix,
remaining-capacity và broker trước đây để tránh lặp lại một cơ chế đã bị bác bỏ.
Chưa có protocol FULL học tiếp nào được mở từ các kết quả chẩn đoán này. Vẫn
giữ yêu cầu N4/K2, random initialization, cùng thông tin/ngân sách và giới hạn
sức khỏe trước khi xét mở rộng. Goal nghiên cứu tiếp tục ở trạng thái active.
