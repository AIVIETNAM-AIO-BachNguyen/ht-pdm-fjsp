# Method proposal v4: học giá trị năng lực technician khi quyết định matching

Viết ngày 2026-10-20, trước implementation v4 và trước các TEST seed v4.
Đây là đề xuất phương pháp cần bác bỏ được, chưa phải novelty claim.

## Vấn đề cụ thể

Flat PPO v1/v2/v3 nhận 56 feature nhưng chấm 21 matching bằng các output
không chia sẻ tham số theo cặp máy–technician. Nó phải tự học lại cùng một
quan hệ duration/urgency ở nhiều vị trí nhãn. Trên TEST v3, scratch và
teacher-assisted policy còn thua rule/planner ở nhiều case; specialist có
corrective queue lớn và fine-tuning làm xấu kết quả so imitation-only.
Điều này không chứng minh nguyên nhân là representation, nhưng tạo câu hỏi
kiểm chứng. Trong kernel, gán technician j cho máy m khóa j trong d(m,j)
tick; một máy chưa có request ở age2 có thể tạo PM ở tick kế tiếp. Giá trị
của cạnh (m,j) vì vậy phụ thuộc cả demand hiện tại và gần tới của các máy
khác chỉ j hoặc rất ít người phục vụ được.

## Ý tưởng và cơ chế

Policy centralized RL học từ đầu, dùng cùng observation hiện tại và reward
H48+tail16. Mỗi cặp hợp lệ có base score từ shared machine encoder, worker
encoder, service-duration edge và clock. Matching logit là tổng edge score;
WAIT có logit 0. Feasibility vẫn do mask. Thêm một nhánh *duration-weighted
capacity opportunity*: mạng học potential không âm cho từng máy từ state
hiện tại. Với mỗi technician j, tổng hợp potential của các máy khác có
eligible edge đến j, chia theo số technician thay thế của từng máy. Trừ khỏi
score của (m,j) một giá không âm tỷ lệ d(m,j) và tổng potential đối thủ
(loại m). Khi j bị chiếm lâu và là năng lực hiếm cho demand khác, edge nhận
điểm thấp hơn. Đây là inductive bias cho hậu quả thời gian, không phải một
shadow price kinh tế đã được hiệu chuẩn hay một reward shaping term.

Ba phiên bản cùng architecture/optimizer: edge-only (không nhánh giá),
current-queue (chỉ queued PM/CM tham gia potential), temporal (queued cộng
máy đang hoạt động age2, ngay trước requester threshold age3). Gate age2
chỉ dùng state hiện tại và luật requester công khai; không đọc future tape.
Mọi phiên bản được học RL từ random initialization, không có imitation.
Flat PPO cùng training budget là đối chứng architecture tổng quát. Actor
chia sẻ encoder và đối xứng theo hoán vị nhãn máy/technician ở N4/K2; critic
có thể không đối xứng, nên chỉ claim actor permutation equivariance nếu
unit test xác nhận. Chưa claim transfer N/K mới vì observation/action adapter
v4 vẫn fixed-size.

## Giả thuyết và đối chứng cần thiết

H1 (primary): temporal giảm case-balanced H+tail cost so current-queue trên
fresh four-case TEST sau DEV-only selection. Paired cluster bootstrap theo
64 physical seeds, 20.000 draws, CI95% upper<0. Nếu H1 không đạt, nhánh
near-future chưa có bằng chứng đem lại giá trị ở setting/budget này.

H2 (secondary): current-queue vs edge-only đo lợi ích competition hiện có;
temporal vs edge-only đo tổng giá trị của nhánh; temporal vs flat PPO đo
giá trị toàn cấu trúc. Các CI95% theo cùng panel chỉ descriptive vì chưa
hiệu chỉnh multiple comparisons. So thêm risk matching, FCFS, bounded
rollout cùng physics/information/objective/seeds. Đo H availability,
failures, CM queue, invalid assignment, cost components, selection latency,
training time và model parameters. Kiểm tra hoán vị nhãn bằng exact action
logit remapping và TEST trajectories trên config hoán vị; không dùng panel
permutation để chọn checkpoint. Một rule khan hiếm có thể được thêm trong
vòng kế tiếp, nhưng không tự coi method thắng heuristics nếu chưa so.

Nếu temporal chỉ hơn flat mà không hơn current-queue, không gán thắng lợi
cho anticipation. Nếu current-queue và temporal cùng hơn edge-only, kết
luận chỉ competition hiện tại có giá trị. Nếu rule/rollout vẫn tốt hơn,
nêu rõ gap. Không đổi cost/physics hoặc chọn case có lợi. Không thêm
training steps/adaptive sweep sau khi xem TEST.

## Đối chiếu literature tại thời điểm đề xuất

- [Zhang et al., Learning to Dispatch, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html)
  học dispatch rule bằng graph RL từ đầu, size-agnostic cho JSSP. Tinh thần
  thiết kế theo graph rồi kiểm chứng transfer là tiền lệ; v4 không nhận
  GNN/dispatch RL là mới. JSSP của họ không phải persistent stochastic
  PM→CM với shared nonpreemptive repair technicians.
- [Wang et al., DANIEL, 2023](https://arxiv.org/abs/2305.05119)
  đã mô hình hóa cạnh tranh operation–machine, học priority cho cặp.
  Shared pair scores/competition ở v4 vì vậy không tự tạo novelty.
- [Alomrani et al., Deep Policies for Online Bipartite Matching, 2021](https://arxiv.org/abs/2109.10380)
  cho thấy skip, bất định demand tương lai và permutation-invariant policy
  đã được nghiên cứu trong online matching. V4 cần kiểm chứng cơ chế
  duration/near-request riêng, không claim phát minh skip hay invariant actor.
- [Lamprecht et al., condition-oriented maintenance RL, 2021](https://arxiv.org/abs/2108.12298)
  nêu vai trò idle/timing trong flow line; trì hoãn PM bằng RL không mới.
- [Ruiz-Rodríguez et al., dynamic maintenance scheduling, 2024](https://doi.org/10.1016/j.eswa.2024.123404)
  so RL, GA simheuristic, dispatch rules dưới uncertainty; mạnh hơn RL ở
  một số điều kiện tần suất replanning. Đây là lý do giữ planner/rule mạnh.
- [Dynamic repair and maintenance of heterogeneous machines, COR 2026](https://doi.org/10.1016/j.cor.2026.107517)
  nghiên cứu repairer, degradation, index rule và rollout RL; rất gần mục
  tiêu tài nguyên hiếm, nhưng publisher abstract chỉ xác nhận single repairer
  và network routing. Không suy từ abstract rằng họ thiếu cơ chế v4.
- [GQN-EREA, Applied Soft Computing 2026](https://doi.org/10.1016/j.asoc.2026.116088)
  đã dùng heterogeneous graph và rule/action composition cho scheduling
  cùng limited maintenance resources. Do đó v4 không claim joint maintenance
  resource graph là mới; cần đọc full paper trước novelty statement.

Khoảng trống ở đây là **câu hỏi thực nghiệm trong simulator cụ thể**:
nhánh potential của demand age2, cân với thời gian chiếm dụng và khả năng
thay thế technician, có cải thiện policy RL học từ đầu so chính architecture
không có nhánh ấy không? Đó là claim v4 được khóa; chưa phải claim đầu tiên
trong literature.
