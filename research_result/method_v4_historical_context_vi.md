# Đối chiếu bổ sung sau khi khóa protocol capacity-opportunity v4

Đọc ngày 2026-10-20, **sau khi v4 đã bắt đầu FULL**. Tài liệu này chỉ cập
nhật bối cảnh và giới hạn claim, không sửa proposal/protocol, source đang chạy,
seed, metric, hay điều kiện kết luận. Tên “v4” trong checkout hiện tại là vòng
thứ tư của *request-lifecycle framework*, khác hoàn toàn Reservation PPO V4
trên nhánh lịch sử.

## Bằng chứng lịch sử trong chính repository

- Nhánh origin/codex/technician-reservation-ppo-v4-confirm chứa
  docs/technician_reservation_ppo_v4_confirmation_findings.md. Trên
  benchmark cân bằng N4–N32, teacher-assisted graph reservation gần central
  PPO ở N4; thắng matched independent graph khi N8/N16/N32. Contrast đó
  thay đồng thời mask reservation và selected-prefix, chưa tách được prefix.
  N4 không chứng minh phối hợp thắng central PPO. Benchmark dùng H48+tail12,
  không phải H48+tail16 và objective hiện tại; cost tuyệt đối không so trực
  tiếp. Teacher/pretraining là phần lớn phương pháp, nên không là đối chứng
  scratch-RL cho v4 hiện tại.
- Nhánh origin/codex/reservation-prefix-residual-free-v6-confirmation chứa
  docs/reservation_framework_research_ledger_vi.md. Prefix so với
  mask-only ở B1 **FAIL DEV**: pooled +0,2205% dưới gate +0,5%;
  confirmation TEST vẫn sealed. B1 diagnostic tắt prefix cùng weights làm
  đổi nhiều action N8, nhưng không thay đối chứng retrained. B2–B6 đang là
  các thử nghiệm cơ chế riêng; không được dùng kết quả DEV của chúng làm
  chứng cứ xác nhận claim hiện tại. B6 proposal giữ teacher và prefix
  relational/capacity, khác actor duration-weighted near-request trong
  checkout này.
- Nhánh origin/codex/shared-technician-coupled-followup-v1 chứa
  docs/maintenance_coupled_dispatch_followup_v1_report_vi.md. Trong
  request pool với technician nhiều tick, coupled rollout thua risk về cost
  trung bình trên panel đã khóa; thay thời điểm proposal không đổi assignment
  trong H, tác dụng chủ yếu ở cleanup. Vì vậy “dự báo demand gần tới” trong
  v4 chỉ có giá trị nếu thực sự đổi assignment và giảm cost; có age2 signal
  không đủ. Physics/terminal liability của nhánh đó khác setting v4 hiện tại.
- docs/research/centralized_maintenance_opportunity_cost_idea_20261010.md
  đã nêu ý tưởng opportunity cost của technician và encoder cạnh từ trước,
  chưa được kiểm chứng ở request-lifecycle v4. Tên “capacity opportunity”
  không thể xem là ý tưởng lần đầu hoặc claim novelty.

## Ranh giới diễn giải v4

Mốc khóa trước kết quả là temporal vs current-queue, cùng actor và optimizer,
chỉ thêm eligibility của máy healthy age2 vào potential. Nếu primary âm tính,
ghi là thiết kế temporal gate này không đem lại giá trị trong bốn config/
ngân sách đã thử; không diễn giải là mọi representation theo opportunity cost
đều thất bại. Nếu dương tính, chỉ claim incremental benefit trong panel này,
chưa claim literature novelty, generalization N/K, hay chất lượng trước
heuristic khan hiếm chưa triển khai. Edge-only/current-queue/flat/risk/rollout
và health phải được báo đủ để nhận diện điều kiện hỗ trợ và giới hạn.
