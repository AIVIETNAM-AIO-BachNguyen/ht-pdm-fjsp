# Setting nền đề xuất cho maintenance coordination framework v1

## Mục tiêu và trạng thái lựa chọn

Người dùng xác định ưu tiên là một setting đủ tốt để bắt đầu phát triển
framework; không bắt buộc MARL, centralized RL cũng phù hợp. Đề xuất của
assistant: dùng `reciprocal_scale2` trong kernel request-lifecycle tại commit
`8855c7e3f79424b1d828b2e66ef6fcf941c3fcd9` làm development anchor, dừng quét
setting ở giai đoạn này. Đây là đề xuất kỹ thuật từ kết quả khám phá đã xem,
không phải xác nhận của tác giả về contribution hoặc benchmark công bố.

Setting tốt cho bước phát triển khi có dynamics rõ, feasibility kiểm tra được,
nhiều quyết định hợp lệ với hậu quả khác nhau, áp lực nguồn lực có ý nghĩa,
baseline đủ để đo tiến bộ và quy mô đủ nhỏ để debug. Không yêu cầu RL phải
thắng trước khi bắt đầu, không yêu cầu đã có optimality proof hoặc tính mới.

## Bài toán cụ thể

Bốn máy sản xuất song song bị suy giảm, dùng chung hai technician. Máy cần
quyết định tiếp tục vận hành hay bắt đầu bảo trì; hệ thống quyết định tập máy
được phục vụ và technician phụ trách khi capacity không đủ phục vụ tất cả.
Một technician chỉ làm một việc, mỗi máy tối đa một technician. Công việc đã
bắt đầu không bị ngắt, phục hồi sức khỏe tại lúc hoàn tất.

| Máy | Technician A: thời gian sửa | Technician B: thời gian sửa |
|---|---:|---:|
| M1 | 2 | 6 |
| M2 | 2 | 6 |
| M3 | 6 | 2 |
| M4 | 6 | 2 |

Mọi cặp đều đủ kỹ năng trong anchor; tính không đồng nhất là thời gian sửa.
Không mô tả anchor này là có hard skill exclusion. Case hạn chế kỹ năng được
giữ làm kiểm tra phụ. Cả PM/CM dùng cùng matrix duration, cùng pool worker,
và phục hồi 75% effective age (floor(age_before * .25) tại completion).

Clock rời rạc; máy đang chạy tăng age một đơn vị mỗi tick, cap8. Sau tăng age,
nếu age>=4 thì failure xảy ra với xác suất.45. Máy đang sửa không tăng age,
máy hỏng chờ không sản xuất. Initial age lấy uniform{0,1,2,3}, cùng physical
tape giữa các controller khi so sánh. Đây là synthetic setting để phát triển,
không phải mô hình đã hiệu chỉnh cho một nhà máy cụ thể.

## Yêu cầu chờ và quyết định được học

- Cố định requester threshold: máy đang chạy, chưa có request, age>=3 thì
  tạo PM request. Không học thời điểm gửi request trong framework v1.
- Request tồn tại trong pool chung; không gắn sẵn vào một technician, không
  hủy. Máy chờ PM tiếp tục chạy và suy giảm; hỏng thì cùng request chuyển CM,
  giữ created_at. Failure chưa có request tự tạo CM ở boundary tiếp theo.
- Controller chọn một legal partial matching từ các request đang chờ sang
  các technician rảnh, hoặc WAIT. Chọn subset tạo ưu tiên; WAIT/deferral tạo
  quyết định thời điểm thực hiện. Matching tạo quyết định phân công worker.
- Có21matching trong catalogue N4/K2 trước khi mask. Mask theo request,
  occupancy và eligibility. Một controller tập trung hoàn toàn đủ cho bước đầu.

Không gọi các quyết định trên là toàn bộ tối ưu request timing. Gửi request
sớm miễn phí, vẫn được chạy và có thể bị dispatcher hoãn, nên chưa có đánh đổi
nội tại rõ cho request submission. Tránh thêm penalty để tạo độ khó nhân tạo.

## Objective và thời gian đánh giá

Cost mỗi tick = 1*PM starts + 2*CM starts + 6*unavailable machine-ticks
+15*new failures. Không có artificial waiting penalty; availability loss đã
được phản ánh bằng unavailable cost. Reward RL xuất phát từ negative economic
cost; không đổi coefficients để ép learner thắng.

Giữ H48+common tail16 không reset, cùng tail requester threshold và dispatcher
risk matching như diagnostic. Khi thêm RL wrapper, terminal reward phải phản
ánh đầy đủ tail cost và primary là H+tail; báo riêng H-only, tail, pending và
in-service ở hai cutoff. Tail hữu hạn không chứng minh hết endpoint bias.

Primary: economic cost. Secondary: H availability, failures, CM waiting,
request-wait exposure có tách PM/CM, pending và in-service, latency/compute.
Không dùng tổng PM backlog một mình để suy chất lượng giữa requester khác nhau.
Không tự áp một service floor chưa được đặt trước; nếu framework cần SLA hoặc
availability constraint thì đó là một phiên bản protocol mới có lý do vận hành.

## Bằng chứng để chọn làm development anchor

Run diagnostic `maintenance_request_lifecycle_v1_full_cpu_20261019T141931Z`,
64paired physical seeds/cell, mọi raw artifact hash khớp, zero invalid assignment.
Trên requester threshold tại anchor:

| Chỉ tiêu | FCFS | Risk matching | Rollout6tick/16scenario |
|---|---:|---:|---:|
| H+tail mean cost |885.20|806.16|717.69|
| H availability |54.12%|58.13%|59.20%|
| CM waiting H+tail (machine-tick) |7.95|2.45|1.94|
| Utilization technician H+tail |84.70%|81.20%|80.27%|

Rollout/risk mean gain10.97%, paired difference−88.47 với CI95% khám phá
[−110.64,−65.26];51seed cost thấp hơn,9bằng,4cao hơn. Đây là bằng chứng có
headroom so heuristic đã thử, không phải chứng minh rollout tối ưu hoặc RL
sẽ thắng. Root diagnostic cho thấy phần lớn khác biệt là subset/timing,
chưa cô lập được giá trị assignment. CI trong45contrast không hiệu chỉnh.

## Cấu trúc framework để phát triển

1. Environment adapter: state máy, request, worker và dynamics chung.
2. Controller interface: rule, planner hoặc centralized RL nhận cùng current
   observations và trả matching/WAIT; không đọc realized future shocks.
3. Feasibility layer: kiểm tra/mask hard resource constraints trước execution.
4. Evaluation/logging: cost decomposition, waiting/backlog, event trace và audit.

Ưu tiên adapter Gymnasium và centralized masked PPO như một lựa chọn đầu để
kiểm tra pipeline learning; RL vẫn là module thay thế được. Không cần graph,
communication learning hoặc nhiều agent trong v1. Giữ FCFS, risk matching,
rollout làm comparator; reference exact trên bài nhỏ có thể bổ sung mà không
chặn việc bắt đầu phát triển framework ngay bây giờ.

## Các case phụ đã có, không tiếp tục chọn theo learner thắng

- `interchangeable_scale1`: worker tương đương, control về heterogeneity.
- `reciprocal_scale3`: cùng pattern chuyên môn hóa, sửa lâu hơn, stress case.
- `specialist_generalist_scale2`: có hard eligibility exclusion, kiểm tra tính
  linh hoạt của feasibility interface và tradeoff nguồn lực.

Anchor không thay thế ba case phụ hoặc xóa các kết quả bất lợi đã có. Khi làm
đánh giá xác nhận về framework, dùng fresh seed panels và protocol mới; các
seed hiện tại là development evidence đã mở. Không chạy thêm experiment trong
lượt chốt specification này.

## Ranh giới với production scheduling

Bản nền này là maintenance/resource coordination: không có jobs, operations,
precedence hay rerouting. Bản FJSP cũ là hướng tích hợp sau. Không mô tả v1
đã giải joint production rescheduling; phần inspect/reduce-load/tool-replacement
cũng chưa phải action riêng. Giữ phạm vi này giúp phát triển được framework
cụ thể mà không phải xây lại đồng thời production và health-prediction modules.
