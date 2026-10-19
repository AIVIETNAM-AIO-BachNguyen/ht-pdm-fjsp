# Sửa lỗi kiểm tra source trước full run

Traceback người dùng cung cấp dừng trước mô phỏng vì guard cũ dùng mọi dòng
`git status --porcelain` làm điều kiện chặn. Chưa có `git status` của lab nên
chưa xác định file cụ thể đã kích hoạt lỗi; không suy đoán rằng chắc chắn chỉ
là report. Không yêu cầu reset/clean/stash hoặc xóa dữ liệu.

Guard mới chỉ chặn thay đổi input thực thi: toàn bộ src/, configs/ (kể cả
registry lịch sử mà seed audit đọc), pyproject.toml, uv.lock, AGENTS.md và
protocol của experiment. Kiểm tra cả staged/unstaged/deleted/renamed và file
untracked trong các đường dẫn đó. Workspace dirty vẫn được ghi trong manifest;
file ngoài phạm vi input không chặn. Lỗi nay liệt kê chính xác blocking paths.

Kiểm tra macOS ngày19/10/2026: 42 test liên quan PASS trong1,75s, gồm7trường
hợp Git thực tế với report và các kiểu thay đổi source; smoke mới36episode/
720physicaltransition COMPLETED, auditPASS, toàn bộ forecast tái dựng khớp.
Đây là kiểm tra bổ sung cho bản vá; không tuyên bố chạy lại toàn bộ988test.

Protocol có engineering amendment trước smoke; không thay seed, dynamics,
budget, metrics hay phương pháp. Không mở full panel. Lệnh bàn giao dùng
detached worktree mới tại commit đã push, output nằm trong artifacts/ của
repository lab ban đầu để vẫn dùng lệnh rsync cũ. Thư mục source đang dùng
trên lab và run thất bại được giữ nguyên.
