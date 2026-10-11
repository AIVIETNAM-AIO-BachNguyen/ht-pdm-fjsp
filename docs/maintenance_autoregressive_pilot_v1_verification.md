# Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-11
- Verification Status: VERIFIED ENGINEERING SMOKE; FULL PILOT NOT EXECUTED
- Version Label: maintenance_autoregressive_pilot_v1_mac_gate

## Kết quả kiểm tra trước khi push

680 tests pass (221.77 giây), gồm 24 tests mới cho prototype. Hai warning:
TqdmExperimentalWarning có sẵn trong test SB3 và PytestCacheWarning vì sandbox
không cho ghi cache ngoài writable root; cả hai không phải lỗi test.

CLI smoke: artifacts/maintenance_autoregressive_pilot_v1_smoke_cpu_20261011T112024Z,
COMPLETED/PASS, 4.86 giây, 3 models, 192 learned decisions, 288 physical steps,
48 training episodes, 24 optimizer steps, 32 development episodes và16 test
episodes; 96 physical test decisions replay scalar/batch chính xác.
Mỗi arm có cùng5218 tham số và cùng hash khởi tạo. Cả ba có cập nhật trọng số,
kiểm tra likelihood qua PPO và return bao gồm tail đạt; checkpoint roundtrip đạt.
48 dòng PM choice riêng, 52181 physical signatures lịch sử bị loại theo metadata.
196 artifact files (không tính manifest), tổng5509957 bytes được kiểm tra hash.
Full test outcomes chưa mở. Code snapshot và plan trong smoke trùng byte với
source được chuẩn bị commit; dirty=true là có chủ đích vì smoke chạy trước commit.

Thêm engineering smoke để kiểm tra đúng kích thước pilot và horizon dài:
artifacts/maintenance_autoregressive_pilot_v1_h48_smoke_20261011T112251Z.
H48+tail12, hidden32, vector32, rollout3072, minibatch256, epochs4;
một update/arm trên64 cấu hình PM lựa chọn độc lập từ các engineering role seeds
hash từ smoke seed16700000. Cấu hình này không thuộc train/dev/test của pilot
và đã kiểm tra chữ ký không trùng. Không chạy development/test hoặc chọn model.
COMPLETED/PASS: 9216 learned decisions, 11520 physical steps, 144 optimizer steps.
Mỗi model19650 tham số; likelihood/return/budget/parameter-change checks đạt.
Thời gian/update trên Mac: catalogue0.706s, context0.746s, frozen0.750s.
Đây là kiểm tra số học và timing một update, không phải kết quả nghiên cứu.
Script tái lập engineering smoke được lưu cùng artifacts dưới engineering_smoke.py.

## Pilot bàn giao cho lab

Ba arms: catalogue_ppo, ar_context_ppo, ar_frozen_context_ppo, mỗi arm3 seeds.
9 models x122880 learned steps =1105920; 1382400 physical training steps;
23040 training episodes, 17280 optimizer steps, 1248 development episodes,
1920 sealed test episodes (115200 physical decisions), 4800 audit decision rows.
54 checkpoint/model .pt files; source/config snapshots và CSV/JSON, thường
vài chục MB (PM/request journals phụ thuộc các policy). Không cần checkpoint cũ.
Chi phí huấn luyện ngoại suy trên Mac từ một update khoảng4.4 phút, cộng đánh
giá và I/O; dự trù10–30 phút trên CPU lab là ước lượng thô, không phải đo lab.
Xem tqdm/training_progress.csv để cập nhật ước lượng theo tốc độ thực tế.

## Giới hạn diễn giải

Đây là prototype đầu tiên và pilot3 seeds. Smoke chỉ xác nhận code chạy đúng.
Chưa có bằng chứng thuật toán mới tốt hơn, chọn thời điểm/technician tối ưu,
hay mở rộng được sang N/K lớn. Chỉ đối chiếu hai decoder cô lập cập nhật context;
catalogue so với decoder còn khác cách phân rã action và greedy evaluation.
Trường PM-specific assignment giữ nguyên tập máy và mọi CM assignment;
không dùng CM-only choices để khẳng định kỹ năng chọn technician cho PM.
STOP-with-extension của catalogue diễn tả một matching còn có thể thêm PM;
catalogue không có token STOP tường minh.

## Workflow

Protocol đã khóa trước implementation: docs/maintenance_autoregressive_pilot_v1_plan.md.
Chỉ thay đổi8 files mới của experiment và tài liệu verification này; code cũ
không sửa. Không SSH/lab UI, không chạy full, không thực hiện transfer remote.
Sau commit/push bàn giao compound command và rsync theo docs/experiment_workflow.md.
