# Góc nhìn để chốt setting: từ tranh chấp sang giá trị quyết định

## Material Passport

- Origin: ARS-Codex experiment-agent, diễn giải kết quả và chẩn đoán đọc-only.
- Date: 2026-10-12, Asia/Taipei.
- Verification Status: **ANALYZED**. Lượt này kiểm lại 19 hash artifact v3, selection, số episode, các mean trọng tâm từ CSV và diagnostic giới hạn trên journal đã lưu; không replay toàn bộ simulator hoặc training.
- Phạm vi: screening non-FIFO N4/K2 và bằng chứng từ các báo cáo liên quan. Các đề xuất bên dưới chưa phải protocol experiment đã khóa.

## 1. Kết quả mới thay đổi cách nhìn thế nào?

Nguồn mới nhất: [v3 report](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_skill_contention_screen_v3_analysis_20261012T041913Z/report_vi.md>) và run `maintenance_skill_contention_screen_v3_full_cpu_20261012T041544Z`, commit `19fb84d5d63977b8486afb6dde742b8864dd72ab`.

**0/48 ứng viên đạt đủ năm gate.** Có 6.784 episode screening; 1.280 episode confirmation chỉ gồm năm đối chứng, không có ứng viên nào được xác nhận. Đây là kết quả hợp lệ của protocol. Không đổi gate hoặc gọi một ứng viên khám phá là winner.

Hai nhóm cho hai bài toán khác nhau:

| Ứng viên / rule | Strict contention | Cost-sensitive contention | Nhóm thực sự chọn có đổi cost | Extended cost |
|---|---:|---:|---:|---:|
| overlap3/fast1_slow325, matching | 8,33% | 3,68% | 3,42% | 466,42 |
| overlap3/fast1_slow325, greedy | 10,84% | 7,65% | 7,39% | 507,75 |
| bottleneck2/fast2_slow634, matching | 14,16% | 10,16% | 0,10% | 942,52 |
| bottleneck2/fast2_slow634, greedy | 14,75% | 9,86% | 0,10% | 950,97 |

Mẫu số rate là toàn bộ 48 tick; mean từ 64 screening seeds/rule. Cost gap overlap3 khoảng 8,14% là kết quả khám phá, gộp timing/selection/assignment và dynamics; chưa đo riêng assignment hay RL gain.

Ở overlap3, fast worker sửa 1 tick, slow worker sửa ba máy chung lần lượt 3/2/5 tick. Với cặp máy có slow durations 2 và 5, hai assignment có tổng duration 3 hoặc 6 tick, chênh 18 đơn vị unavailable cost. Đây là hệ quả dịch vụ thật. Nhưng matching đã chọn assignment có committed cost thấp nhất ở cả 105 lượt selected contested được phân tích trong report. Chưa có bằng chứng cần RL để giải riêng phần đó.

Ở bottleneck2, hai máy độc quyền thường chiếm fast worker. Trong các tick có cơ hội assignment, matching bỏ qua nhóm máy đổi người được chủ yếu vì nó chọn nhóm có máy độc quyền. Bài toán nổi lên là chọn tập máy và phân bổ capacity, không thuần đổi người trên một tập máy cố định.

## 2. Góc nhìn thứ nhất: metric đang đo một loại cơ hội khá hẹp

V3 chỉ đếm cost-sensitive contention khi đồng thời có strict contention, xét subset cardinality tối đa, và chênh committed cost vượt ngưỡng. Những quyết định khác vẫn có thể quan trọng:

- Một máy có hai người phù hợp nhưng việc giữ người đa năng ảnh hưởng tương lai.
- Có thể làm PM ngay hoặc giữ máy hoạt động thêm.
- Chọn phục vụ một máy thay vì hai máy để không chiếm nguồn lực quá sớm.
- Assignment có hậu quả khác dù không xuất hiện strict preference conflict.

**Diagnostic mới, giới hạn hai screening seed có journal:** ở overlap3/fast1_slow325 dưới matching, trong 96 tick có 50 tick mà subset thực sự chọn có committed-cost span ≥6, nhưng chỉ 5 tick vào rate selected cost-sensitive contention. Có 45 tick có span nhưng không có strict contention. Greedy tương ứng 46/96 tick có span, 7 được đếm, 39 nằm ngoài strict contention.

Không ngoại suy các tỷ lệ này thành toàn 64 seed. Đây là minh họa trực tiếp rằng gate contention không bao phủ mọi cơ hội lựa chọn. Span cũng là max–min giữa các assignment, không phải regret của action đang dùng và chưa phải lợi ích dài hạn.

**Đề xuất:** trong protocol mới, báo riêng cơ hội assignment, cơ hội priority/timing và phần giao với contention. Gate cần gắn với câu hỏi nghiên cứu. Không sửa verdict v3.

## 3. Góc nhìn thứ hai: policy tốt có thể làm giảm tần suất tình huống khó

Rate không phải một thuộc tính cố định của duration matrix: policy thay đổi trạng thái máy, thời điểm service và availability của worker. Cùng overlap3, matching có cost thấp hơn greedy và các contention rates cũng thấp hơn.

Điều này chưa chứng minh việc giảm contention là nguyên nhân giảm cost. Tuy nhiên, nó cho thấy lấy min-rate giữa hai rule có thể loại một môi trường nơi một rule xử lý vấn đề khá hiệu quả. Yêu cầu "mọi rule đều thường xuyên gặp conflict" khác với yêu cầu "môi trường có quyết định quan trọng".

**Đề xuất:** dùng một panel state từ nhiều policy đã khóa để so lựa chọn trên cùng state; đồng thời giữ closed-loop evaluation trên toàn episode. Panel state giúp so công bằng, nhưng vẫn phụ thuộc nguồn policy; state thủ công dùng cho cơ chế và báo riêng, không làm mẫu số cho tần suất tự nhiên.

## 4. Góc nhìn thứ ba: phân biệt hậu quả hiện tại với giá trị nhìn trước

V3 xác nhận tổng duration khác nhau. Khi duration đã biết và restoration bằng nhau, matching có thể tối ưu phần downtime cam kết này bằng scoring đơn giản. Đó là baseline cần có trước khi phát triển RL.

Câu hỏi quyết định cho idea opportunity cost là: **sau khi rule/matching đã xử lý tốt chi phí hiện tại, còn những lúc quyết định tốt hơn phải xét tương lai không?**

Kiểm tra nên so myopic matching và scarcity heuristic với lookahead trên cùng state. Tìm trường hợp thứ hạng action đổi theo bối cảnh ngoài máy được phục vụ: tình trạng các máy khác, worker đang bận, khả năng thay thế hoặc time-to-go. Dùng expected cost-to-go, với forecast selection/evaluation tách nhau khi dùng Monte Carlo.

Cần mở cả quyền chọn tập máy khi đánh giá root optimum/reference. [Opportunity-cost diagnostic trước](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_opportunity_cost_v1_analysis_20261010/analysis_vi.md>) đã có assignment reversal khi giữ tập máy, nhưng cả 48 state vẫn có fixed-root gap bằng zero do baseline được chọn thêm PM cho máy khác. Đây là lý do không dừng ở minh họa hai assignment.

Positive gap của reference là bằng chứng có điều kiện theo solver/budget/continuation. Không tìm thấy gap dưới một approximate planner chưa chứng minh optimum đã đạt.

## 5. Góc nhìn thứ tư: tần suất cần đi cùng độ lớn và phần còn cải thiện được

Một cơ hội hiếm nhưng tác động lớn có thể đáng nghiên cứu hơn conflict thường xuyên nhưng chênh cost nhỏ. Ngược lại, max–min span lớn có thể chỉ do có một action rất tệ mà mọi baseline đều tránh.

Cần báo ba đại lượng tách biệt:

1. Tần suất có lựa chọn, với mẫu số all ticks và conditional opportunity đều ghi rõ.
2. Độ lớn expected cost gap của action baseline với reference, cùng uncertainty; không chỉ max–min span.
3. Lợi ích khi dùng policy cải thiện trong closed loop, kể cả thay đổi future state distribution, waiting và backlog.

Mean regret trên panel đầy đủ có thể dùng để ưu tiên diagnostic. Không cộng local regrets rồi gọi là episode gain: can thiệp đầu tiên đã đổi các state tiếp theo.

Các ngưỡng 10%/5%/3% của v3 vẫn được giữ cho run đó. Khi thiết kế mới, chọn ngưỡng dựa vào khả năng đo và research margin, rồi khóa trước confirmation; không hạ ngưỡng chỉ để một cấu hình đang thích đạt.

## 6. Góc nhìn thứ năm: chọn family để kiểm tra nguyên nhân, thay vì một matrix thắng screening

Một benchmark để phát triển thuật toán cần có cả condition đối chứng và miền hoạt động, không chỉ một ma trận duration. Family gọn có thể dùng ba chiều:

| Chiều | Các mức có lý do |
|---|---|
| Khả năng thay thế | Worker tương đương; nhiều máy dùng chung worker; worker đa năng đi kèm specialist |
| Thiệt hại khi dùng người thay thế | Bằng nhau; khác nhau theo máy |
| Áp lực theo thời gian | Ít nhu cầu đồng thời; nhu cầu vừa đủ cạnh tranh; tải cao |

Giữ costs/dynamics/horizon cố định khi kiểm tra duration/skill structure; nếu thay restoration/hazard thì ghi đó là một chiều khác. Các mức tải phải kiểm tra capacity hiệu dụng thay vì chỉ N/K. Thêm reciprocal specialization chỉ khi câu hỏi yêu cầu, không đồng thời mở tất cả yếu tố để tạo độ khó.

Điều kiện worker giống nhau kiểm tra liệu method có tạo lợi ích assignment giả; partitioned topology kiểm tra bài toán ưu tiên dưới skill bottleneck; overlap3 kiểm tra người thay thế với thiệt hại khác nhau. Hoán vị nhãn giữ cùng cấu trúc vật lý, nên là kiểm tra consistency chứ không phải OOD profile mới.

## 7. Cách thu hẹp tiếp theo

Ưu tiên **family overlap3 với substitution penalties không đồng đều**, lấy fast1_slow325 làm điểm neo khám phá. Giữ bottleneck2 làm condition chẩn đoán selection/capacity và uniform/cancellation làm đối chứng. Đây là lựa chọn follow-up từ screening, chưa là setting đã được xác nhận.

Trước vòng quét duration tiếp theo, cần giải quyết câu hỏi value:

1. Đăng ký panel mới và các contrast, budget, stopping và service guards.
2. Kiểm tra myopic matching/heuristic scarcity so lookahead ở setting nhỏ, tách can thiệp selection và assignment, rồi đánh giá joint action đầy đủ.
3. Đo effect trên whole episode/interior với treatment biên chung. [Boundary sensitivity](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_boundary_sensitivity_v1_analysis_20261011/analysis_vi.md>) đã cho thấy lợi ích ngắn hạn có thể đổi dấu sau continuation.
4. Nếu assignment còn headroom theo context, phát triển assignment decoder. Nếu chủ yếu timing/selection, giữ matching đơn giản và tập trung học priority. Nếu rule mạnh đã cạnh tranh ở quality/compute, chưa cần RL mới.

Một seed mới chỉ kiểm tra random trajectories mới trên cùng matrix. Muốn claim family/generalization phải có nhiều physical profile mới, không chỉ nhiều shocks.

Không đặt điều kiện RL phải thắng hoặc planner phải tệ để chọn environment. Setting có thể đạt tính hợp lệ và giá trị quyết định, trong khi không phù hợp với claim assignment-learning hoặc không còn headroom cho algorithm đã chọn.

## 8. Kiểm tra 11 sai lệch suy luận

| Kiểm tra | Cách xử lý |
|---|---|
| Simpson | Giữ riêng overlap3/bottleneck2 và từng rule |
| Ecological | Không suy rate trung bình thành tính chất mọi state/policy |
| Berkson | fast1_slow325 là ứng viên được chú ý sau screening |
| Collider | Selected-subset diagnostic là điều kiện theo policy, không causal episode effect |
| Base rate | Tách all ticks, contested ticks và journal sample 96 tick |
| Regression to mean | Chưa có confirmation ứng viên; cần panel mới |
| Survivorship | Giữ đủ 48 ứng viên không đạt, backlog và tail |
| Look elsewhere | Cost gap 8,14% là exploratory, không thêm confirmatory test |
| Forking paths | Không sửa gate/ranking/verdict v3; đề xuất mới dùng protocol mới |
| Correlation/causation | Rate/cost cùng đổi không xác định cơ chế; cần controlled interventions |
| Reverse causality | Policy tạo state distribution, nên rate thấp có thể đi cùng quyết định tốt |

**11/11 checked.** Chưa có thuật toán mới, full run mới hoặc kết quả xác nhận mới được thực hiện trong lượt này.
