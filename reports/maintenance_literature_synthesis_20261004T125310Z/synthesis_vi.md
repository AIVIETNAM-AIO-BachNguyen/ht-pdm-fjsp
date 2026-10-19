# Từ các paper đến một bài toán nghiên cứu chung

Ngày rà soát: 2026-10-04. Đây là tổng hợp có mục tiêu để định hình problem, không phải systematic review hay xác nhận toàn bộ tính mới.

## 1. Phạm vi và mức bằng chứng

Folder `/Users/bachng/Research/1/` có 19 PDF, tương ứng 16 công trình sau khi loại ba cặp trùng nội dung công trình; 287 trang ở tập không trùng. Đã trích xuất văn bản toàn bộ, sàng lọc abstract và đọc chọn lọc phần mô hình, thực nghiệm, kết luận. Trích xuất toàn bộ không có nghĩa đã đọc sâu mọi trang.

Đọc kỹ mô hình của Rodríguez et al., Ruiz-Hernández et al., Demirci et al., An et al., Yu et al., các phần mô hình liên quan của Dahite et al. và Sanhueza et al. Đọc abstract/kết luận và các đoạn liên quan của nhóm còn lại. Kiểm tra thêm ảnh trang 7 của Rodríguez và trang 8 của Demirci để xác nhận các giả định quan trọng. Cột phạm vi đọc ở bảng dưới giới hạn những điều có thể suy luận.

Đã tải thêm hai PDF công khai vào `additional_papers/`: Tseremoglou & Santos (2024), và Lee et al. (2026, preprint). Đọc mô hình/giới hạn có liên quan của hai bài này. Đối chiếu thêm MIRA (2026) qua nội dung nhà xuất bản được công cụ tìm kiếm lập chỉ mục; chưa tải được PDF vì HTTP 403. Không coi MIRA là đã đọc toàn văn. Hồ sơ inventory, văn bản trích xuất và log tải được giữ trong cùng thư mục.

## 2. Problem chung ở cấp vận hành

**Làm thế nào quyết định máy nào cần được bảo trì, vào lúc nào, và bởi nguồn lực nào, khi nhiều máy đang suy giảm cùng chia sẻ năng lực bảo trì hữu hạn, để giảm tổn thất dài hạn của cả hệ thống?**

Đây là giao điểm của các dòng nghiên cứu, không phải mọi paper đều có đủ tất cả các yếu tố sau:

- Tình trạng máy và dự báo tuổi thọ còn lại (RUL) cho biết rủi ro; chúng chưa tự quyết định lịch khả thi.
- Nguồn lực hữu hạn khiến quyết định cho một máy ảnh hưởng thời điểm các máy khác được phục vụ.
- Kỹ năng kỹ thuật viên ảnh hưởng khả năng nhận việc, thời gian sửa hoặc kết quả sửa, tùy mô hình.
- Bảo trì sớm gây mất sản lượng và chi phí; bảo trì muộn gây hỏng và chờ sửa. Tối ưu cho từng máy riêng lẻ có thể không tối ưu cho cả hệ thống.

Từ đó, câu hỏi trung tâm không nên bắt đầu bằng QMIX, graph, counterfactual loss hay priority tag. Cần bắt đầu bằng mối liên hệ giữa **rủi ro chờ đợi, thời gian chiếm dụng nguồn lực và giá trị sản xuất**.

Ví dụ minh họa do người đọc xây dựng: máy A gần hỏng hơn B, nhưng A có thể được một thợ phổ thông xử lý; B chỉ được phục vụ nhanh bởi một chuyên gia. Phân chuyên gia cho A có thể làm B phải chờ quá lâu. Quyết định tốt phụ thuộc tổn thất khi chờ, khả năng thay thế thợ và thời gian hoàn thành, không thể suy ra chỉ từ thứ tự RUL. Ví dụ không chứng minh một quy tắc luôn tối ưu.

## 3. Các dòng nghiên cứu và bằng chứng phản chứng

| Công trình | Quyết định/chỗ mạnh được đọc | Phạm vi đọc và giới hạn diễn giải |
|---|---|---|
| [Ruiz-Hernández et al., 2020](https://doi.org/10.1016/j.cor.2020.104927) | Phân nguồn lực bảo trì cho nhiều máy suy giảm; imperfect maintenance; chỉ số Whittle | Mô hình §3 và kết luận §7. PM hoàn tất trước epoch tiếp theo; action chọn máy dưới giới hạn số repairmen, không phân danh tính/kỹ năng từng người. Giới hạn này thuộc mô hình đang đọc. |
| [Hossini et al., 2026](https://doi.org/10.1016/j.ress.2025.111613) | Reliability/QoS và chi phí bảo trì hệ thống chiếu sáng; RL so với SA | Abstract, kết luận §6. Tác giả nêu technician availability, spare parts và advanced scheduling là hướng mở. Không kết luận mọi RL cùng có giới hạn này. |
| [An et al., 2021](https://doi.org/10.1016/j.eswa.2021.115693) | Kết hợp flexible job-shop, PM và repairmen khác kỹ năng/hiệu suất/chi phí, có timetable | Mô hình §2 và kết luận §6. PM có time window cho trước; thời gian xử lý đã biết. Là phản chứng cho tuyên bố chưa ai joint schedule với heterogeneous repairmen. |
| [Rodríguez et al., 2022](https://doi.org/10.1016/j.rcim.2022.102406) | Machine agents chọn hoãn hoặc cặp technician–component; thời gian sửa phụ thuộc chuyên môn | Mô hình §§3.1–3.2 và thực nghiệm §§4.1–4.4. Đã có MARL cùng technicians khác nhau. Chọn trùng thợ cùng lúc bị phạt và restart môi trường (§4.4, trang 7), khác một hệ thống tiếp tục vận hành với backlog. |
| [Ding et al., 2025](https://doi.org/10.1016/j.ress.2025.111394) | RUL prediction và MARL phối hợp production/maintenance | Abstract, kết luận §5 và các đoạn liên quan. Tác giả nêu RUL deterministic, chưa tính uncertainty của prediction (trang 15). Có joint production–maintenance; chưa dùng phạm vi đọc này để khẳng định thiếu mọi loại ràng buộc nhân lực. |
| [Pérez Hernández et al., 2024](https://doi.org/10.1016/j.engappai.2023.107362) | Localised subnetworks khai thác quan hệ mạng tài sản; trade-off chất lượng và độ phức tạp | Abstract, kết luận §9. System coupling qua mạng không đồng nhất với coupling qua technician queues. |
| [Zhang et al., 2025](https://doi.org/10.1016/j.eswa.2025.127034) | Component policies và system-centred re-optimization dưới system constraints | Abstract, kết luận §5. Không quy kết thiếu kỹ năng/thời gian chiếm dụng thợ khi chưa kiểm tra đầy đủ method theo các trục đó. |
| [Chen & Liu, 2025](https://doi.org/10.1016/j.aei.2025.103553) | Hai agent chuyên trách replacement và inspection; dự báo RUL xác suất, nhiều mục tiêu | Abstract, kết luận §4. Không phải mỗi component là một agent cạnh tranh thợ. |
| [Garapati & Kanna, 2025](https://doi.org/10.5220/0014152200004932) | Digital twin + MARL cho dữ liệu IIoT | Abstract/kết luận; DOI được xác nhận ở SciTePress. References trong PDF có chuỗi DOI placeholder `XXXXX`; cần kiểm tra lại references trước khi dựa vào lập luận nền của bài. Không dùng làm nguồn chính cho một kết luận định lượng. |
| [Rokhforoz & Fink, 2021, preprint](https://arxiv.org/abs/2112.10459) | Joint bidding–maintenance với safety filter cho generation units | Abstract, kết luận §6. Bài toán cạnh tranh thị trường khác bài toán cooperative dùng chung crew. |
| [Mbuli et al., 2020](https://doi.org/10.1049/iet-cps.2019.0063) | Multi-agent system lập lịch/reactive maintenance cho mobile fleet | Abstract, kết luận §5. MAS không tự động có nghĩa MARL. Health/RUL chưa phải cơ chế trung tâm được chứng minh trong phạm vi này. |
| [Demirci et al., 2025; online 2024](https://doi.org/10.1007/s10696-024-09544-y) | Condition-based maintenance với capacity; Whittle index và lower bound; đánh giá tới 160 machines/16 repairmen | Mô hình §§3.1–3.2, kết luận §5. Cross-trained repairmen, chuyển máy nhanh, repair rate theo máy; không phân ma trận thời gian machine–technician. Không nói bài này bỏ qua scarcity/queues. |
| [Sanhueza et al., 2026](https://doi.org/10.1016/j.cie.2026.111801) | Skills, shift patterns, relocation giữa bases, task assignment và outsourcing | Mô hình §§3.1–3.4 và abstract. Tập công việc có sẵn, processing time cho trước, aircraft ground windows; không tự suy ra thời điểm PM từ stochastic health trong các phần đã đọc. |
| [Tusar & Sarker, 2024](https://doi.org/10.1016/j.ref.2024.100616) | Phân thợ theo kỹ năng và shift, giảm vị trí thiếu người và cân bằng workload | Abstract, phần notation/model liên quan, kết luận trang 15–16. Tác giả nêu dữ liệu input tĩnh và kỹ năng cố định là giới hạn. |
| [Dahite et al., 2022](https://doi.org/10.3390/math10111807) | Joint maintenance scheduling và technician routing; random failures, expected failure/maintenance cost | Abstract, các phần mô hình §§3.3–3.4 và kết luận §6. Time-based PM với maintenance windows; không gọi mô hình này hoàn toàn deterministic. |
| [Yu et al., 2023](https://doi.org/10.3390/math11214562) | Skill/proficiency, technician-dependent service times, routing/selection task dưới capacity | Abstract, mô hình §3 liên quan và kết luận §6. Tập task/profit/time window cho trước; experience-dependent ở đây không đủ để khẳng định kỹ năng học và thay đổi nội sinh theo thời gian. |
| [Tseremoglou & Santos, 2024](https://doi.org/10.1016/j.ress.2023.109582), tải thêm | RUL distributions/POMDP cho prognostic tasks, sau đó DRL fleet scheduling, 4M resources gồm manpower theo skill | Đọc §§3.1–3.4 và method/kết luận liên quan. Hai stage có trao đổi thông tin slot/resources; không được mô tả là stage 1 hoàn toàn mù resources. Là phản chứng cho gap rộng “chưa ai đưa uncertainty cùng resource constraints vào maintenance planning”. |
| [Lee et al., 2026, preprint](https://arxiv.org/abs/2609.13566), tải thêm | So sánh planning, DQN/PPO và heuristic trên shared-crew multi-asset maintenance | Đọc §§3–4 và giới hạn/kết luận. Planning áp zero-failure hard constraint, vanilla RL soft penalty; không phải so sánh cùng toàn bộ constraints. Một crew, maintenance một step; giới hạn chính xác RUL/discrete state dựa failure time và ba seeds được tác giả nêu. |
| [Babu et al., MIRA, 2026](https://doi.org/10.3390/fi18080430), đối chiếu thêm | Graph/MARL kết hợp maintenance, production rescheduling, communication và action repair | Đọc các đoạn primary publisher được search lập chỉ mục. Controller dùng health deterministic vì uncertainty-validation gate không đạt; đánh giá trong simulation. Là đối thủ gần cho hướng graph + feasibility/action repair; chưa dùng để kết luận về đầy đủ technician semantics. |

## 4. Điều cần sửa trong bản review cũ

Nhận định trong file `Literature Review — MARL for Maintenance with Shared Heterogeneous Technicians.md` rằng chưa có MARL kết hợp maintenance và heterogeneous technicians là quá mạnh: Rodríguez (2022) đã làm trực tiếp điều này và PDF nằm ngay trong folder.

Tương tự, các từ khóa “joint scheduling”, “uncertainty + limited resources” hay “system-level optimization” không đủ tạo tính mới: An (2021), Demirci (2025) và Tseremoglou & Santos (2024) là những bằng chứng cần đối chiếu. Có optimization model xử lý contention rõ ràng; không thể gán chung nhược điểm “không hiểu tranh chấp nguồn lực” cho các phương pháp centralized.

Sửa năm: Ruiz-Hernández là 2020, không phải 2024. Demirci có online date 2024 và journal issue 2025. Hossini có DOI/online year 2025 nhưng journal issue 2026. Chen & Liu dùng hai agents theo chức năng, không phải một agent/component.

## 5. Câu hỏi nghiên cứu hẹp nên kiểm chứng

**Khi thời gian chờ và thời gian sửa phụ thuộc vào kỹ thuật viên, quyết định đồng thời thời điểm bảo trì và phân công thợ có giảm tổn thất hệ thống hơn cách xếp ưu tiên từ tình trạng máy rồi mới phân thợ hay không, và lợi ích xuất hiện trong điều kiện nào?**

Đây là câu hỏi ứng viên và giả thuyết của người đọc, chưa là xác nhận novelty. Các thuật toán đã có có thể giải quyết một phần hoặc toàn bộ trong những biến thể nhất định. Cần đối chiếu method sát hơn trước khi dùng trong proposal/paper.

Một đóng góp rõ ràng có thể là xác định cơ chế và ranh giới: mức tranh chấp nguồn lực, độ lệch kỹ năng, thời gian sửa và độ bất định nào khiến policy ưu tiên theo health thất bại; khi nào baseline đơn giản vẫn đủ tốt. Việc chứng minh này có giá trị hơn chỉ thêm một module rồi báo reward tăng.

Thứ tự logic để thiết kế giải pháp:

1. Cố định physics và quyết định: request vào queue hay dispatch ngay; FIFO có bắt buộc không; máy còn chạy khi chờ không; repair có ngắt được không; health reset ở start hay completion; thợ chỉ khác thời gian hay cả eligibility/kết quả. Không đổi đồng thời các semantics để tạo lợi thế cho thuật toán.
2. Tách bài toán cốt lõi health–maintenance–workforce khỏi job-shop, routing, spare parts và communication nếu chúng chưa cần cho câu hỏi.
3. Lập chính sách dễ hiểu dựa tổn thất kỳ vọng nếu chờ, rồi phối hợp assignment và lookahead theo lịch nguồn lực. Một matching tại thời điểm hiện tại chỉ bảo đảm một số ràng buộc tức thời, không tự tối ưu việc chiếm dụng qua nhiều bước.
4. So sánh health/RUL-priority + skill-aware assignment với joint lookahead/MPC và baseline index phù hợp giả định. Nếu biến thể không indexable, gọi rõ là heuristic lấy cảm hứng từ index, không gọi là Whittle policy được chứng minh.
5. Chỉ chọn RL/MARL khi có lợi ích cần thiết về dynamics, tính toán hoặc thông tin phân tán. “Machine là agent” là quyết định thiết kế; không tự chứng minh cần decentralized execution.

## 6. Điều kiện để kết quả sau này có sức thuyết phục

Chưa triển khai hoặc chạy experiment trong lượt rà soát này. Nếu nghiên cứu tiếp, cần protocol riêng theo `docs/experiment_workflow.md`.

Các baseline phải có cùng observations, action feasibility, failure constraints và objective; không so oracle với partial observation rồi gán toàn bộ chênh lệch cho architecture. Cần báo chi phí hệ thống, availability/production loss, failures, waiting time, utilization và thời gian quyết định; reward đơn lẻ không đủ.

Trước tiên tìm counterexample nhỏ và nghiệm tối ưu cho instance nhỏ. Sau đó mới kiểm tra khác capacity/skill mix/service duration/load; thêm RUL uncertainty khi đã có mô hình observation và calibration phù hợp. Prediction chính xác hơn không được mặc nhiên coi là policy tốt hơn. Nếu baseline đơn giản đã giải tốt vùng đang quan tâm, nên thu hẹp hoặc đổi giả thuyết thay vì thêm module.

Kết luận ở mức hiện tại: đã xác định một problem vận hành thống nhất và câu hỏi ứng viên có thể kiểm chứng. Chưa chứng minh “chưa ai làm”, chưa chứng minh phương pháp nào sẽ thắng, và không tổng hợp các phần trăm improvement từ các paper thành một hiệu quả chung vì nhiệm vụ/baseline/constraints khác nhau.
