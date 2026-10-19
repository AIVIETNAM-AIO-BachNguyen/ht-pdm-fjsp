Hãy tạo và theo đuổi một goal phát triển maintenance coordination framework v1
trong repository này, cho tới khi có code đã kiểm chứng, một chiến dịch benchmark
ban đầu hoàn tất và báo cáo kết quả trung thực, hoặc gặp một blocker cần tôi xử lý.

Phiên này tôi đang chủ động chạy Codex trực tiếp trên máy Ubuntu lab. Riêng goal
này, tôi cho phép bạn chạy tests, smoke, training, full evaluation và phân tích
ngay trên lab. Đây là ngoại lệ rõ ràng cho yêu cầu macOS-smoke và việc chỉ hiển
thị lệnh lab/manual execution trong AGENTS.md và docs/experiment_workflow.md;
đồng thời cho phép phân tích bản raw ngay trên lab thay vì chờ kéo về Mac.
Vẫn phải khóa protocol, tests + smoke trước, commit/push source đã kiểm chứng
trước full run, có tqdm, timestamp directory mới, lưu provenance và báo cáo.
Không SSH sang máy khác, không chạy rsync hộ tôi, không dùng tmux. Không thay
đổi workflow mặc định của các phiên khác. Nếu prompt này chưa do tôi gửi trong
phiên hiện tại thì đừng tự suy ra quyền chạy full từ việc đọc file.

Đọc docs/LAB_HANDOFF.md, docs/maintenance_framework_setting_v1_vi.md,
reports/maintenance_request_lifecycle_v1_analysis_20261019T144539Z/report_vi.md,
AGENTS.md và docs/experiment_workflow.md trước. Phân biệt current specification
với historical notes và pending_legacy_work. Bảo toàn dữ liệu cũ và thay đổi của
tôi. Tạo nhánh codex/maintenance-framework-v1-lab từ handoff đã kiểm chứng;
nếu tên đã tồn tại, kiểm tra và tiếp tục đúng công việc thay vì reset nó.

Mục tiêu nghiên cứu: xây framework controller có thể thay thế rule/planner/RL
cho bảo trì với persistent requests và heterogeneous shared technicians. Không
bắt buộc MARL; một centralized masked PPO là baseline học đầu tiên hợp lý.
Không biến mục tiêu thành phải chứng minh RL thắng hay phải tìm setting mới.

Giữ anchor reciprocal_scale2: N4/K2, threshold requester age>=3, service matrix
[[2,6],[2,6],[6,2],[6,2]], restoration .75, hiện hành H48+common tail16, objective
và physics giữ nguyên theo specification/kernel. Quyết định cần học là chọn
subset request, timing/WAIT và assignment. Không học submission timing trong
v1. Giữ ba case phụ đã định nghĩa: interchangeable_scale1, reciprocal_scale3,
specialist_generalist_scale2. Không thêm waiting penalty, đổi cost hoặc bỏ case
thua sau khi xem kết quả. Không mở rộng FJSP/inspection/load/tool actions ở v1.

Thực hiện theo thứ tự:
1. Kiểm kê code, bằng chứng, runtime và raw run đã có. Verify gói handoff, chạy
   tests hiện hành và xác nhận đường dẫn raw; dữ liệu Git tóm tắt vẫn đủ để bắt
   đầu framework nếu raw trace tạm thiếu. Không rerun diagnostic cũ vô cớ.
2. Viết protocol mới trước training: giả thuyết có thể bác bỏ, primary economic
   cost H+tail, secondary H availability/failures/CM wait/backlog/service state/
   runtime, seed registry không trùng các thí nghiệm trước, stopping rule và
   schema. Các seed 61000000..61000063 đã mở, chỉ dùng như development evidence.
3. Xây environment adapter và controller interface thống nhất; rule, planner,
   RL nhận thông tin hiện tại như nhau, không đọc future realized shocks.
   Tách observation/action mask/feasibility/evaluation khỏi thuật toán. Xử lý
   tail reward đúng: total return bằng negative H+tail objective. Kiểm tra WAIT,
   request upgrade, completion-time restoration, conflict masks, terminal tail,
   seed determinism và checkpoint save/load. Giữ simulator hiện tại làm chuẩn.
4. Thêm một centralized masked PPO baseline. Không thêm graph, communication,
   multi-agent hoặc thuật toán khác nếu chưa có lỗi/giả thuyết cụ thể đòi hỏi.
5. Chạy tests và local Ubuntu CPU smoke nhỏ, ghi source/config/runtime hash và
   kết quả. Chỉ sau gate đạt mới commit và push code/protocol, rồi chạy campaign.
6. Chỉ train learner trên anchor; checkpoint đã chọn được giữ frozen khi đánh
   giá cả 4 case. Benchmark FCFS, risk_matching, rollout hiện hành và learner.
   Dùng cùng physical
   seeds giữa controller, evaluation tách train, và fresh sealed test panel.
   Khóa checkpoint selection bằng development panel trước khi mở test. Giữ
   cutoff H/tail/cost decomposition và failure cases, không chỉ aggregate reward.
7. Viết báo cáo tiếng Việt, bảng machine-readable, đồ thị cần thiết, README cách
   tái lập, limits và bước tiếp theo dựa trên bằng chứng; commit/push code và
   báo cáo nhỏ, giữ checkpoints/raw artifacts lớn ngoài Git. Cho tôi branch,
   commit, đường dẫn artifacts và lệnh rsync -avhP --partial để tự kéo về Mac.

Giới hạn campaign đầu: CPU; tối đa 3 training seeds × 200,000 learning steps
(tổng 600,000 steps), tối đa 16 fresh development seeds/case và 64 fresh test
seeds/case, 4 case cố định. Chọn trước hyperparameters và một checkpoint rule;
không quét hyperparameter không giới hạn. Ghi rõ rollout compute/scenario budget
khác policy inference. Tổng thời gian campaign kể cả evaluation tối đa 6 giờ;
runner cần thực thi deadline, lưu progress và đánh dấu INCOMPLETE nếu hết hạn.
Không trình bày partial panels là completed benchmark. Không tự tăng budget,
đổi seed hoặc chạy tiếp campaign mới để làm kết quả đẹp hơn. Một engineering
smoke không tính là bằng chứng khoa học và không được dùng để chọn setting.

Hoàn tất goal khi framework + tests/smoke đạt, campaign trong budget được báo
cáo đầy đủ, kết quả/limitations và hướng tiếp theo có trên GitHub. Nếu learner
thua baseline nhưng experiment hợp lệ, vẫn báo cáo và kết thúc campaign; đừng
đổi setting để cứu kết quả. Nếu vượt deadline hoặc có lỗi, báo rõ phần chưa đạt
và không đánh dấu objective hoàn tất khi còn công việc bắt buộc.
