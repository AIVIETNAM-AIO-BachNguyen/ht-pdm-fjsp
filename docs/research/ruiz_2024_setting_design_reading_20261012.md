# Đọc Ruiz-Rodríguez et al. (2024) để thiết kế setting điều phối bảo trì

Ngày: 2026-10-12. Trạng thái: phân tích toàn văn và đề xuất thiết kế; chưa triển khai, chưa chạy experiment mới, chưa khóa protocol.

## Phạm vi và nguồn

Đã đọc toàn bộ 13 trang PDF do người dùng cung cấp, trích xuất văn bản, và kiểm tra ảnh render trang 5–11, gồm Eqs. (1)–(10), Tables 3–8, Figs. 2–8. Không có code của tác giả hoặc dữ liệu gốc trong lần đọc này; không suy từ mô tả sang hành vi implementation chưa kiểm chứng.

Nguồn chính: Marcelo Luis Ruiz-Rodríguez, Sylvain Kubler, Jérémy Robert, Yves Le Traon. *Dynamic maintenance scheduling approach under uncertainty: Comparison between reinforcement learning, genetic algorithm simheuristic, dispatching rules*. Expert Systems with Applications 248, 123404. DOI: [10.1016/j.eswa.2024.123404](https://doi.org/10.1016/j.eswa.2024.123404).

- [PDF người dùng cung cấp](</Users/bachng/Zotero/storage/WX4BQM6B/Ruiz-Rodríguez et al. - 2024 - Dynamic maintenance scheduling approach under uncertainty Comparison between reinforcement learning.pdf>)
- [PDF trong kho University of Luxembourg](https://orbilu.uni.lu/bitstream/10993/62366/1/1-s2.0-S0957417424002690-main.pdf)
- Đối chiếu với [plan contention v3](</Users/bachng/Coding/Reinforcement Learning/marl/docs/maintenance_skill_contention_screen_v3_plan.md>) và source snapshot của run `maintenance_skill_contention_screen_v3_full_cpu_20261012T041544Z`: `maintenance_dispatch.py`, `maintenance_waiting.py`, `maintenance_heterogeneous_model.py`, `maintenance_skill_contention_screen_v3.py`.
- Kết quả biên được tham khảo từ [phân tích boundary sensitivity](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_boundary_sensitivity_v1_analysis_20261011/analysis_vi.md>); không tái audit run trong lần đọc này.

Những điều đánh dấu **Bài báo** là nội dung quan sát trực tiếp. **Suy luận** là diễn giải của người đọc, chưa kiểm chứng bằng code tác giả. **Đề xuất** là lựa chọn cho nghiên cứu của mình, không phải thông số đã được bài báo xác nhận.

## 1. Kết luận phục vụ thiết kế

Bài này cung cấp một mô hình tham chiếu rất gần: nhu cầu bảo trì động, nguồn lực hữu hạn, thời gian sửa phụ thuộc technician, và hai nguồn bất định được tổ chức thành các tổ hợp có lý do. Giá trị lớn nhất để học là cấu trúc bài toán và phép so sánh; bài chưa cung cấp đủ thông tin để coi nó là benchmark có thể tái lập nguyên trạng chỉ từ PDF.

Khuyến nghị: dùng mô hình này làm nền cho **một benchmark không FIFO có mục tiêu vật lý rõ**, rồi ghi tường minh các phần mở rộng của mình. Không dùng “tỷ lệ contention đủ lớn” hoặc “RL thắng” làm định nghĩa setting hợp lệ. Giữ riêng ba câu hỏi:

1. Setting có mô tả một bài toán điều phối nhất quán, có nguồn gốc và tái lập được không?
2. Quyết định về thời điểm, ưu tiên và assignment có ảnh hưởng tới kết quả toàn hệ thống không?
3. Thuật toán học có cải thiện chất lượng/compute so với các solver mạnh không?

Một setting có thể đạt câu hỏi 1 và 2 nhưng RL chưa tạo giá trị ở câu hỏi 3.

## 2. Bài báo thực sự thiết kế gì?

| Thành phần | Nội dung đọc được | Vị trí |
|---|---|---|
| Máy | Ba trạng thái working, breakdown, maintenance; hỏng theo Weibull hai tham số | §3.1, p.5 |
| Technician | Rảnh/bận; duration phụ thuộc cặp máy–technician | §3.1, p.5 |
| Duration | Normal có mean và standard deviation riêng theo cặp; công thức có dấu làm tròn lên | §3.1, p.5 |
| Hiệu quả sửa | Perfect maintenance, as-good-as-new | §3.1, p.5 |
| Mục tiêu vật lý | Tổng machine-time ở trạng thái working | Eq.(1), p.6 |
| Observation được viết | Remaining intervention time, ticket age, remaining technician busy time | Eq.(2), p.6 |
| Action được viết | Một cặp machine–technician hoặc delay | Eq.(3), p.6 |
| RL | Một policy PPO có action masking; actor/critic [256,256], 3M environment steps | §3.2.1, §4.2, pp.6–7,9 |
| Rule | FIFO: việc tới trước, người rảnh đầu tiên. FIBT: vẫn ưu tiên việc tới trước, nhưng chọn người rảnh có thời gian sửa ngắn nhất | §3.2.3, p.7 |
| Bất định failure | β = 1.2 hoặc 3.0 | §4.1.1, p.7 |
| Bất định duration | σ = 0.5 hoặc 1.5 | §4.1.1, p.7 |
| Episode | 168 bước; 1 bước tương ứng 1 giờ | §4.1.1–2, pp.7–8 |
| Planner | GA-simheuristic giải lại mỗi G bước, G ∈ {1,2,4,6,8,12,16,20,24} | §3.2.2, §4.1.2 |
| GA tuning | 324 tổ hợp, mô tả lặp 20 lần, stopping criterion 5 phút; chọn cấu hình theo G | Tables 4–5, p.8 |
| Dữ liệu case study | Lịch sử bảo trì 10 năm; 8 technician; suy ra các phân phối từ ticket logs | §4.1.3, §4.3 |
| Data access | Data available on request | p.12 |

Hai lưu ý về scope:

- Đây là centralized PPO theo cách mô tả action/observation, không phải QMIX. Việc nhiều technician xuất hiện trong môi trường không tự biến bài toán thành MARL.
- “100 Weibull distributions” ở §4.3 không đủ để khẳng định case có 100 máy vật lý. Tác giả chọn 10 giá trị scale và 10 giá trị shape rồi tạo 100 tổ hợp. Quan hệ giữa các tổ hợp này và từng asset vật lý cần dữ liệu/code để làm rõ.

## 3. Những bài học nên kế thừa

### 3.1. Giữ riêng thời điểm, ưu tiên và người thực hiện

**Bài báo:** policy có thể chọn một cặp hoặc delay; p.6 giải thích rằng chờ người sửa nhanh có thể tốt hơn giao người chậm đang rảnh.

**Đề xuất:** bài toán không FIFO của mình phải cho phép partial matching, kể cả matching rỗng. Không bắt buộc luôn phục vụ số việc tối đa. Đây là quyền quyết định cần có trong mô hình, không phải mẹo tăng độ khó.

Ví dụ giải thích, không phải kết quả thực nghiệm: người nhanh còn bận 1 giờ và cần 2 giờ sửa; người chậm đang rảnh cần 6 giờ. Nếu không có chi phí/rủi ro khác, chờ rồi sửa mất 3 giờ tới completion, giao ngay mất 6 giờ. Một rule “giao ngay khi có người rảnh” sẽ bỏ lỡ lựa chọn này; một rule có lookahead cũng có thể xử lý được, chưa cần RL.

### 3.2. Skills phải mô tả quan hệ giữa loại công việc và worker

**Bài báo:** có μ_ij và σ_ij; phần dữ liệu thực dùng duration theo failure type–technician.

**Đề xuất:** tạo duration profiles theo loại công việc và chuyên môn, thay vì chỉ gắn tốc độ chung cho từng người. Đối chứng gồm worker đồng nhất, một người nhanh hơn trên mọi loại việc, và chuyên môn bổ sung nhau. Sparse skill mask chỉ là một condition mở rộng; không cần đưa vào mọi setting.

Ma trận minh họa do mình đề xuất, đơn vị giờ, chưa là cấu hình khóa:

| Loại việc | T0 | T1 |
|---|---:|---:|
| A | 2 | 6 |
| B | 6 | 2 |

Ma trận này tạo khác biệt assignment rõ, nhưng phần lớn lợi ích trước mắt có thể được giải bởi matching theo mean duration. Chỉ dùng nó chưa chứng minh cần học opportunity cost. Phải kiểm tra lợi ích còn lại khi có máy khác đang suy giảm, worker đang bận và nhu cầu tiếp theo.

Giữ duration và restoration tách biệt khi thử nguyên nhân. Không làm người nhanh đồng thời sửa tốt hơn trong experiment chỉ muốn đo ảnh hưởng tốc độ.

### 3.3. Thiết kế theo các yếu tố vật lý có thể diễn giải

**Bài báo:** bốn tổ hợp low/high failure uncertainty × low/high duration uncertainty.

**Đề xuất:** kế thừa cấu trúc factorial, thêm tải và cấu trúc kỹ năng như các chiều có tên và lý do. Tất cả condition được báo cáo; không chọn một matrix vì nó tạo gap đẹp. Dùng pilot để kiểm tra simulator, phạm vi tải và độ chính xác đo, sau đó khóa family trước đánh giá solver trên test.

### 3.4. Đánh giá nhiều mặt và chấp nhận solver khác nhau thắng

**Bài báo, Table 6:** với low failure uncertainty/low duration uncertainty, RL có reward 44.77 và MTTR 4.54; G1 có reward 46.89 và MTTR 5.80. Hai metric xếp hạng khác nhau.

**Bài báo, Table 8:** trên case suy ra từ dữ liệu thực, RL có reward 11.38, MTTR 4.68; FIBT/D2 có reward 5.52, MTTR 2.62. Mean repair duration thấp không đồng nghĩa reward cao.

**Suy luận:** kết quả này phù hợp với việc chọn người sửa nhanh có thể thiếu ưu tiên toàn hệ thống. Nó chưa chứng minh causal mechanism là giữ technician cho tương lai, cũng chưa xác nhận reward cao đồng nghĩa uptime vật lý cao: cần audit Eq.(4)/implementation.

**Đề xuất:** primary là downtime hoặc cost vật lý; báo riêng waiting, service duration, failure-to-completion, availability, backlog và long waits. Không dùng served-only MTTR để đại diện toàn bộ demand.

## 4. Những điểm cần làm rõ hoặc làm chặt hơn

### 4.1. Thay β có thể đồng thời thay mean lifetime

**Bài báo:** §4.1.1 dùng β ∈ {1.2,3.0} để thay độ bất định; Table 3 lại ghi α là mean, β là standard deviation. Cách ghi này không khớp cách định nghĩa scale/shape trong Eq.(10).

**Suy luận toán học:** với Weibull continuous:

E[T] = α Γ(1 + 1/β).

Nếu giữ α, E[T]/α ≈ 0.940656 khi β=1.2 và ≈ 0.892980 khi β=3.0. Do đó đổi shape cũng làm đổi mean khoảng 5.1% giữa hai mức. CV tương ứng khoảng 0.8369 và 0.3634. Các số này được tính từ công thức bằng Python `math.gamma`, không phải kết quả simulator.

**Đề xuất:** khi hỏi tác động của bất định, giữ target mean lifetime L_m, chọn α_m = L_m / Γ(1+1/β_m). Nếu discretize/làm tròn lifetime thì kiểm tra và hiệu chỉnh mean của phân phối sau discretization. β còn đổi age-dependent hazard, nên contrast được gọi là thay hình dạng failure distribution ở cùng mean, không khẳng định một ảnh hưởng uncertainty thuần túy độc lập với hazard.

Triển khai renewal failure đúng: theo operating age và conditional survival; không vẽ lại một lifetime độc lập ở mỗi tick. Quy định rõ tuổi tăng khi nào, có dừng trong repair không, và chỉ reset ở completion.

### 4.2. Normal repair duration cần quy tắc dương và đơn vị

**Bài báo:** §3.1 có dấu ceiling cho Normal duration, nhưng pp.5–7 không nêu xử lý mẫu ≤0. Normal cũng không có maximum hữu hạn, trong khi Eq.(4) dùng max d_ij mà chưa nêu quy tắc xác định bound.

**Đề xuất:** hoặc giữ positive-truncated Normal để gần bài tham chiếu và công bố quy tắc sampling, hoặc dùng positive Gamma/lognormal và gọi rõ là adaptation. Dùng CV để so mức biến động giữa các mean khác nhau. Mọi target mean/CV phải được kiểm tra trên distribution thực sự dùng sau truncation/rounding.

Các mức σ=0.5/1.5 của bài có thể tham khảo nhưng không tự chuyển được sang mọi service matrix: σ=1.5 trên mean 2 khác hẳn σ=1.5 trên mean 8 về relative uncertainty.

### 4.3. Reward và objective chưa được nối đủ rõ

**Bài báo:** Eq.(1) là tổng indicator working; Eq.(4) dựa vào 1 − remaining maintenance time/max duration, không thể hiện trực tiếp indicator working hoặc trạng thái failed-waiting. Tables 6/8 báo Reward và MTTR.

**Suy luận có điều kiện:** nếu failed-but-unserved có remaining repair time bằng 0, công thức in ra có thể cho nó đóng góp lớn như máy đang working. Chưa có code nên không kết luận implementation bị lỗi. Có thể còn quy ước trạng thái/penalty không được viết đầy đủ.

**Đề xuất:** primary reward = âm chi phí vật lý được audit; nếu dùng downtime thì failed-waiting và under-repair đều bị tính. Nếu giữ PM, tính downtime do PM và intervention cost. Reward shaping chỉ được thêm khi quan hệ với objective được chứng minh/kiểm tra; report actual availability độc lập với training reward.

### 4.4. Một cặp mỗi bước có thể tạo bottleneck dispatch

**Bài báo:** Eq.(3) chọn một cặp hoặc delay; §4.1.2 đặt mỗi physical step là 1 giờ. Không thấy pp.5–9 nói rõ có thể dispatch nhiều cặp trong cùng timestamp bằng các substep không làm thời gian trôi.

**Suy luận có điều kiện:** nếu mỗi action làm trôi 1 giờ, hệ thống chỉ có thể bắt đầu một maintenance job mỗi giờ, dù nhiều người cùng rảnh. Khi tăng technician, giới hạn này có thể chi phối kết quả thay vì capacity thực.

**Đề xuất:** ở mỗi physical decision epoch, chọn một partial matching cho toàn bộ idle technicians. Nếu decoder lần lượt chọn cặp, các lựa chọn nội bộ không làm tăng giờ; STOP kết thúc batch. Mọi solver dùng cùng time semantics. Đây là extension so với action được in trong bài, không tự gọi là replication.

### 4.5. Observation còn thiếu quy ước về thông tin tương lai

**Bài báo:** Eq.(2) viết các thời gian còn lại/tuổi ticket, chưa đưa rõ μ_ij, σ_ij, β_i hoặc compatibility vào vector. §3.1/4.2 không đủ để xác định realized duration được biết lúc nào hoặc planner sử dụng random future ra sao.

**Đề xuất:** tất cả methods nhận cùng observable information: machine condition/age, failure/pending status, elapsed waiting, worker status/progress, eligibility và duration forecast. Họ biết model/forecast parameters theo contract; không thấy realized next failure hay duration của công việc chưa bắt đầu.

Hai contract về busy duration đều có thể nghiên cứu, nhưng phải chọn một và dùng chung: (a) duration được quan sát sau khi bắt đầu, nên remaining time biết chính xác; (b) duration vẫn latent, chỉ thấy elapsed time và conditional residual forecast. Contract (b) đòi hỏi belief/progress đủ rõ; không tự gọi raw observation là fully observed MDP.

Bất định ngẫu nhiên vốn có của duration khác forecast miscalibration. Nên nghiên cứu với forecast được hiệu chuẩn trước, rồi tách misspecification thành robustness condition.

### 4.6. Dữ liệu thực được biến đổi, không phải replay nguyên trạng

**Bài báo:** §4.3 chọn distribution có β>1 và scale nhỏ để failures diễn ra trong episode ngắn; tạo Cartesian product 10 scale × 10 shape. §4.1.3 mô tả lọc theo failure type/technician, tính TBF và duration.

**Suy luận:** tái tổ hợp có thể phá quan hệ scale–shape gốc; chỉ giữ scale nhỏ giới hạn phạm vi case về máy cần sửa thường xuyên. TBF theo nhóm technician có thể phản ánh lịch phân công/coverage, chưa chắc là intrinsic failure lifetime của từng asset. Duration trung bình quan sát cũng có thể bị ảnh hưởng bởi loại việc khó mà từng người được giao.

**Đề xuất:** nếu có logs, giữ nguyên fitted parameter tuples và asset/type mapping; kiểm tra machine identity khi tính TBF; tách waiting, active repair và calendar interval. Khi chưa có logs, dùng synthetic family có provenance, không gọi là industrial calibration. Không nhắn tác giả hoặc tải dữ liệu không công khai trong lượt này.

### 4.7. Compute và độ bất định thống kê cần rõ hơn

**Bài báo:** Table 7 cho wall-clock values; §4.2 nói G1 chạy 168 schedules và kết luận real-time dựa vào so với 1 giờ. Table 4 nói stop 5 phút. Phạm vi aggregation của thời gian chưa đủ rõ để tái lập phép so deadline. §4.1.2 nói lặp 20 lần khi tuning GA, nhưng Tables 6–8 không nêu rõ đủ training replicas, test episodes và confidence intervals cho final comparisons.

**Đề xuất:** report per-decision p50/p95/p99, CPU time, tổng episode, hardware, scenario/rollout count, GA stop rule và training/tuning riêng. Không giảm tần suất replan để làm planner thua rồi gọi đó là equal-compute; cố định decision opportunities và sweep search budget, còn replan frequency là một axis riêng.

Data/code không được cung cấp đầy đủ trong PDF để có exact reproduction. Không suy 20 tuning repetitions thành 20 independent PPO training seeds hoặc uncertainty của Tables 6–8.

## 5. Đối chiếu với setting không FIFO đang có

| Thành phần | Paper 2024 | Contention v3 của mình | Hướng thiết kế |
|---|---|---|---|
| Failure law | Weibull | Threshold age 4 rồi Bernoulli p=.45; age cap 8 | Dùng renewal wear-out law làm condition tham chiếu; công bố law hiện tại nếu giữ |
| Service duration | Stochastic theo cặp | Deterministic integer theo cặp | Có mean profiles rõ; thêm uncertainty độc lập |
| Restoration | Perfect | .75 trên các cạnh selectable | Perfect ở reference; imperfect là extension riêng |
| Skill structure | Pair-dependent duration | T0 nhanh chung; T1 chậm khác nhau; compatibility masks | Thêm reciprocal specialization có lý do; không phải thay toàn bộ bằng mask phức tạp |
| Action | Một cặp hoặc delay theo Eq.(3) | Legal partial matching | Giữ matching và WAIT; chuẩn hóa dispatch/time semantics |
| Objective | Uptime; reward theo remaining time | PM/CM cost, unavailable6, failure15, overdue12 sau4 ticks | Objective vật lý và cost ratios có lý do; deadlines là condition riêng |
| Horizon | 168 giờ | 48 tick + common continuation12 | Quy định physical unit và service/lifetime scale; kiểm tra boundary |
| Benchmark | FIFO/FIBT, PPO, GA-S | Matching/greedy và planner ở các study trước | Giữ baseline mạnh của mình, thêm reference methods để nối literature |

V3 giúp kiểm tra một cơ chế trong mô hình synthetic đang có. Các ngưỡng strict contention 10%, consequence5%, selected3% không được paper này hỗ trợ như tiêu chuẩn chung. Không sửa verdict hoặc threshold của run v3 sau khi đọc paper.

## 6. Đề xuất một setting có thể chốt tiếp

### 6.1. Scope nghiên cứu

**Centralized scheduling of maintenance under skill-dependent service times and stochastic failures.** Scheduler chọn khi nào bắt đầu, ưu tiên máy/ticket nào và technician nào; non-FIFO, resource-constrained, nonpreemptive. Mục tiêu đo bằng downtime/cost toàn hệ thống.

Phải ghi rõ hai lớp khi tiếp nối code PM+CM của mình:

- **Reference dispatch condition:** chỉ điều phối các repair tickets phát sinh bởi failure; gần mô tả ticket scheduling của paper hơn.
- **Research extension PM+CM:** có preventive intervention trước failure. Quy định chính xác eligibility/ticket release, việc PM pending có tiếp tục hoạt động không, chuyển PM sang CM khi hỏng, và restoration/cost. Không suy từ bài này rằng cơ chế PM trigger hiện tại của mình đã được xác nhận.

Khuyến nghị giữ PM+CM làm câu hỏi của mình nếu muốn nghiên cứu ưu tiên bảo trì; CM-only là condition tham chiếu để tách giá trị timing phòng ngừa. Cùng một simulator và switch có nghĩa rõ, không ghép kết quả thành một claim chung.

### 6.2. Hợp đồng vật lý tối thiểu

1. Máy độc lập về degradation, liên kết qua technician hữu hạn. Chưa thêm travel, fatigue, mentoring hoặc production routing.
2. Theo dõi operating age, failure state, pending request và trạng thái service. Một machine chỉ có một active request/service theo semantics công bố.
3. Renewal failure với wear-out parameters theo machine/type. Maintenance hiệu lực tại completion; reference reset age0.
4. Duration positive, phụ thuộc task type–technician, forecast parameters được biết/quan sát theo cùng contract cho mọi solver.
5. Feasible partial matching; máy/worker tối đa một nhiệm vụ cùng lúc; WAIT hợp lệ. Không impose FIFO hoặc max-cardinality.
6. Decision epoch và event order thống nhất. Nếu giữ periodic1h để gần paper, cho phép batch dispatch trong timestamp và tính hoàn thành/downtime nhất quán. Nếu dùng event-driven, ghi rõ đó là adaptation.
7. Không cung cấp true future failure time/duration trước khi xảy ra. Planner chỉ sample forecast trajectories độc lập với realized evaluation randomness.
8. Horizon theo physical time; dừng logging không xóa backlog, jobs in service hoặc nghĩa vụ chi phí. Báo cả cutoff và continuation/interior sensitivity.

### 6.3. Family có lý do

Một bản pilot gọn, các mức sau là **đề xuất của mình**:

- Anchor N4/K2 để có exact/reference checks. Các mức N8/K3 và N16/K6 là candidate scale-transfer tests, chưa phải quy mô chuẩn từ paper.
- Ba skill structures: homogeneous; universally faster worker; reciprocal specialists. Giữ restoration đồng nhất ở primary. Sparse masks chỉ ở robustness/skill-eligibility condition.
- Hai failure shapes β=1.2/3.0 kế thừa paper, điều chỉnh scale để giữ mean theo từng load condition.
- Duration uncertainty dùng CV hoặc một tương đương được hiệu chỉnh từ σ=.5/1.5 trên reference mean; kiểm tra realized moments.
- Ba áp lực nhu cầu: thấp/vừa/cao qua target operating lifetime. Kiểm tra bằng frozen reference policy trong calibration panel riêng, gồm utilization từng worker, pending, waiting, availability và throughput. N/K hoặc một công thức pooled load đơn giản không đủ khi skills và blocking ảnh hưởng capacity.

Không nhất thiết chạy toàn bộ Cartesian product ngay. Khóa primary skill/load conditions trước, bốn uncertainty cells là một module; scale/forecast error là module transfer. Báo đầy đủ những condition đã đăng ký, kể cả RL không thắng.

Thay matrix kỹ năng có thể đổi effective capacity dù row means bằng nhau. Khi diễn giải pure skill effect cần matched-load comparison hoặc kiểm soát/report capacity riêng; không gọi raw cost difference là chỉ do chất lượng assignment.

### 6.4. Objective và baselines

Primary nên chọn một trong hai và khóa trước:

- **Downtime:** tổng thời gian failed-waiting hoặc maintenance, chuẩn hóa theo machine-time.
- **Economic cost:** downtime loss + intervention/failure costs được nêu rõ theo physical units/ratios; reward = âm cost.

Nếu giữ cost đang có, ghi nó là synthetic research calibration và khảo sát sensitivity, không gán cho nhà máy. Overdue penalty4/12 không được bài này xác nhận. Deadline có thể là soft service constraint được giải thích, không cần trở thành yếu tố chính để tạo ưu thế RL.

Secondary: actual availability, failure count, PM/CM count, throughput, waiting/failure-to-completion distribution, unserved/censored requests, terminal failed/pending/in-service, worker utilization.

Baselines cần có:

1. FIFO và FIBT để nối với literature; cùng feasibility/time contract.
2. Priority + min-expected-duration matching; rule có cân nhắc chờ người phù hợp và scarcity. Chọn/tune trên development.
3. Mean-model và stochastic rolling-horizon planner ở nhiều search budgets; cùng decision opportunities và observable information.
4. Plain masked centralized PPO dùng cùng information/action semantics.
5. Exact dynamic reference cho discretized small instances; hoặc đủ-budget reference được ghi là approximate. Perfect-information oracle nếu có chỉ là bound được gắn nhãn, không baseline online bình thường.

Matching tối ưu cost tức thời là baseline cần thiết: nó tách việc giải bài toán ghép tĩnh khỏi learning of future opportunity cost. Học assignment chỉ có căn cứ nếu còn giá trị sau baseline này, hoặc đạt quality/compute trade-off tốt hơn.

### 6.5. Split và evidence gates cho một experiment tương lai

Chưa khóa seed IDs/counts ở đây vì chưa triển khai. Protocol experiment tiếp theo phải đăng ký hypothesis, estimand, train/dev/test splits, physical profile generator, independent RL training replicas, paired evaluation randomness, statistical unit, stopping rule, compute budget và artifact schema trước run.

Tách calibration (physical load), development (solver tuning), và test (evaluation). Frozen transfer test phải có physical profiles/type mixtures mới; new shock seeds trên cùng matrix chỉ là trajectory generalization. Bootstrap/CI theo profile và training replica phù hợp, không coi từng tick là independent sample.

Hai loại gate độc lập:

- **Setting validity:** dynamics/time/cost audit; admissible information; meaningful skill/load profiles; boundary/censoring handled; reproducible config; health metrics interpretable.
- **Algorithm investment:** joint-action lookahead vs strong myopic baseline có effect đo được hoặc learned policy đạt compute advantage; effect giữ qua fresh profiles. Không bắt mọi condition phải có gap, không lấy worst-action span hoặc strict conflict frequency làm proxy cho episode benefit.

## 7. Những gì chưa thể kết luận từ bài

- Bài chưa chứng minh kiến trúc opportunity-cost/assignment decoder của mình là mới hoặc sẽ tốt hơn.
- Chưa có causal ablation tách timing, ticket selection, assignment và WAIT đủ để quy lợi ích cho từng cơ chế.
- Reward differences trong Tables6/8 không được tự đổi thành uptime percentages.
- Chưa có bằng chứng zero-shot tăng cả số máy và technician chỉ từ những phần đã đọc.
- Chưa tái lập numerical results; thiếu source/data/config đầy đủ. Các absence statements trên giới hạn ở §3–5 và Tables3–8 đã kiểm tra.

Hướng tiếp theo có căn cứ là hoàn thiện **specification của benchmark tham chiếu và extension PM+CM**, rồi làm kiểm tra vật lý và baseline. Chưa cần thay mixer/đầu assignment hoặc tìm thêm một conflict threshold để setting “pass”.
