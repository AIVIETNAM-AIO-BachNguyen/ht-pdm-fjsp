# Các experiment cần làm để đánh giá hướng MARL

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-08
- Verification Status: UNVERIFIED
- Version Label: marl_followup_proposals_v1
- Trạng thái: đề xuất thiết kế, chưa phải protocol đã khóa; chưa implement hoặc chạy experiment mới.
- Bằng chứng đầu vào: `analysis_vi.md`, commit `4b1cffb0cc1e2413e27f0e2c3a7730f8d3e3d7a8`, source snapshot của full run 20261007T151955Z.

## Mục tiêu

Tìm điều kiện mà MARL tạo giá trị và xác định nút thắt của prototype. Không thiết kế một môi trường để buộc MARL thắng. Không cần chứng minh MARL thắng mọi centralized controller có thông tin hoàn hảo; cần chứng minh lợi ích dưới cùng constraint triển khai và objective.

Đã có MARL prototype: cooperative local actor + team training. Điều chưa có là bằng chứng đủ mạnh để đầu tư thêm thuật toán/kiến trúc. Ba experiment dưới đây trả lời ba câu hỏi khác nhau, cần chạy theo thứ tự và có thể dừng nếu bằng chứng không ủng hộ.

## A. Ưu tiên trước: audit decoding và giới hạn thông tin

**RQ:** khoảng cách chất lượng đến từ cách chọn action, thiếu thông tin hay learning?

**Dữ kiện code:** central matching chọn argmax trên matching catalogue. Serial policy chọn argmax conditional tại mỗi bước reservation. Greedy theo prefix không bảo đảm matching có joint probability lớn nhất. Đây là khác biệt implementation đã xác nhận, chưa phải nguyên nhân đã chứng minh của cost gap.

**Hypothesis có thể bác bỏ:** giữ nguyên checkpoint, đổi serial greedy sang exact joint-MAP làm thay đổi đáng kể chất lượng. Nếu không cải thiện, decoding chưa giải thích được khoảng cách; nếu cải thiện, cần tách decoding khỏi đề xuất learning algorithm mới.

**Thiết kế vòng đầu không cần train mới:**

1. Giữ toàn bộ checkpoint đã chọn, không chọn lại bằng test. So serial greedy với joint-MAP cho cooperative local và cooperative full. MAP liệt kê matching hợp lệ, tính tổng conditional log-probability theo đúng prefix, rồi chọn matching có xác suất lớn nhất. Kiểm tra decoder trên toy distribution và kiểm tra mask/likelihood normalization.
2. Có thể thêm stochastic execution với policy RNG tách biệt như diagnostic; không dùng nhiều rollout RNG để giả tạo thêm replica training độc lập.
3. Central matching, central fixed, rule và planner giữ nguyên làm reference. So actor full và local để quan sát information gap, nhưng không gọi đây là can thiệp chỉ đổi thông tin vì chúng có checkpoint training khác nhau.
4. Primary specialized H12; homogeneous, skill-mask và H24 là sensitivity. Common-state panel từ development dùng cho log-probability và timing; closed-loop paired episodes dùng cho cost. Action thay đổi sẽ làm state trajectory thay đổi, nên không chỉ so action trên log cũ.

**Giới hạn quan trọng:** joint-MAP là decoder có điều phối tập trung, không phải một solver đã triển khai phân tán. Nó tối đa probability, không tối thiểu expected cost; cải thiện cost là giả thuyết cần đo. Không gọi MAP là oracle hay upper bound chất lượng.

**Metrics:** paired priced/base cost; waiting violation và overdue ticks; terminal pending theo request; action-disagreement frequency; log-probability difference; mean/P95 decoding time; memory. Báo riêng cost theo failure, downtime và maintenance.

**Seeds/budget:** dùng đủ 10 checkpoint seed/arm hiện có, zero optimizer update. Tạo development shock panel và sealed test shock panel mới bằng namespace riêng trong seed registry; common environment shocks giữa decoder. Existing labeled profiles không còn là unseen profiles; test mới chỉ kiểm tra noise mới conditional trên các profile này. Số profile/shock phải được khóa trước full run, không tăng sau khi thấy CI.

**Analysis/stopping:** paired differences lấy trung bình trong mỗi training replica; CI theo replica, không theo tick. Hai contrast decoder cho local/full có multiplicity correction nếu cùng là primary. Báo raw cost difference cùng CI; mức giảm 2% có thể là candidate practical gate để thảo luận, chưa phải yêu cầu triển khai. Smoke kiểm tra engineering và runtime, không chọn decoder theo sealed test. Fixed evaluation grid, không optional stopping.

**Sau A, nếu cần retraining:** cùng global observation/team objective/feasible action support, so catalogue head với serial head dưới decoding đã định nghĩa rõ. Ablate critic hoặc advantage một yếu tố mỗi lần; giữ PPO ratio, reward scale, optimizer và entropy contract tương ứng. Current comparison đổi reward, critic và ratio cùng lúc nên không xác định được nguyên nhân. Exact entropy và sampled-prefix entropy phải được phân biệt. Có thể dùng toy finite-horizon DP để đo regret; optimum này chỉ đúng cho toy horizon/objective đã khóa, không phải full benchmark.

**Quyết định sau A:** sửa decoder/protocol nếu đó là nút thắt; nghiên cứu optimization/credit assignment nếu full-information serial policy vẫn kém với decoder tương ứng; nghiên cứu memory/communication nếu gap xuất hiện chủ yếu khi giới hạn thông tin. Không chọn attention hay counterfactual credit assignment trước khi thấy bằng chứng tương ứng.

## B. Experiment chính cho lý do dùng MARL: information và communication constraints

**RQ:** MARL giữ được chất lượng tốt hơn các solver khác khi không thể lấy global state mới ở mỗi quyết định hay không?

**Nhu cầu triển khai còn thiếu:** chưa có measurement về sensor/network, decision deadline, update frequency hoặc tổ chức vận hành. Nếu chưa có, ghi rõ đây là synthetic sensitivity; không gọi đó là chứng minh nhà máy cần MARL.

**Thiết kế:** giữ N=4/K=2 và dynamics/cost/service semantics hiện tại. Đổi observation-delivery contract, không đổi failure process để giúp một solver. Local actors có own state/history, own skill edges và public resource acknowledgements. Peer state chỉ được nhận qua message có timestamp; critic global chỉ dùng trong training.

Grid đề xuất để khảo sát (chưa khóa): full fresh telemetry; peer update mỗi 2 hoặc 4 tick; delivery delay 0/1/2 tick. Loss chỉ nên thêm thành experiment riêng sau khi causal delivery, acknowledgement và timeout fallback đã được test. Tick là đơn vị simulator, chưa có quy đổi thời gian vật lý. Không chạy một Cartesian grid lớn trước khi smoke xác định chi phí và câu hỏi primary.

**Controllers bắt buộc:**

- Local rule/auction mạnh: dùng risk/wait/skill score và shared reservation/assignment protocol, không chỉ “mỗi máy chọn technician ưa thích”.
- Independent local RL.
- Cooperative local MARL hiện tại.
- Centralized RL hoặc planner có timestamped stale state/history và cùng message budget; cho phép compression/event-triggered telemetry và model-based prediction hợp lệ. Không chỉ cấp zero state cho trung tâm rồi gọi baseline thất bại.
- Full-information central reference để định lượng cost của restriction; đây là reference có nhiều thông tin hơn, không phải đối chứng ngang điều kiện hay oracle.

Reservation/dispatch/acknowledgement của MARL cũng chịu delay/byte budget. Feasibility checker chỉ bảo vệ tài nguyên; không được âm thầm dùng hidden peer urgency để chọn thay actor. Tất cả solver có cùng physical start boundary, timeout/fallback và xác nhận lock hợp lệ. Communication cost đo từ payload/message thực trong simulator có schema rõ; wall-network measurements là validation triển khai riêng.

**Primary metrics:** priced cost dưới constraint đã chọn trước, cùng guard về waiting và terminal pending; end-to-end decision latency gồm frontend, messaging và arbitration; bytes/messages/rounds, stale-state age và deadline miss. Đừng chỉ đo forward pass của actor. Local policy không được miễn chi phí thu thập state hoặc reservation.

**Training/selection:** train riêng các learned arm dưới observation constraint tương ứng; giữ development tuning budget tương đương và công khai wall compute. Dùng namespace seed mới cho training, development, test, shock và communication. Fixed training budget sau preflight; test mới sealed, không dùng test của run hiện tại để chọn hyperparameter. Báo zero-shot exposure và matched-training adaptation riêng nếu cùng làm cả hai.

**Success gate cần chốt theo use case:** một operating point phải đạt cost/waiting/terminal limits và tạo lợi ích về quality hoặc latency/message cost so với baseline mạnh khả thi cùng constraint. Nếu dùng đề xuất non-inferiority cost margin 2%, phải kiểm tra CI theo margin, không coi “CI chứa zero” là equivalence. Nếu claim giảm communication ít nhất 50%, phải tính cả reservation/acknowledgement. Những con số 2%/50% là candidate margin nghiên cứu, chưa phải requirement từ nhà máy.

**Quyết định sau B:** nếu MARL tạo operating point tốt và tái lập qua seed, có cơ sở nghiên cứu local memory, message selection hoặc communication-aware policy. Nếu rule/central prediction đáp ứng tốt hơn, chọn chúng; bản thân restriction chưa làm MARL cần thiết.

## C. Scalability chỉ làm khi A/B hoặc nhu cầu vận hành chỉ ra vấn đề

**RQ:** khi N tăng, MARL có lợi thế về chất lượng đạt được trong deadline hoặc total compute/message cost không?

Grid đề xuất N=4/8/16/32, giữ N/K và skill-graph workload có định nghĩa tương ứng. Báo normalized cost/machine/tick, workload per technician và độ tập trung preference để phân biệt scaling với tăng contention. Không coi H24 là scalability về số máy.

Baseline tập trung phải mở rộng hợp lý: pair scoring + assignment solver, factorized/autoregressive central policy hoặc planner bị giới hạn compute có anytime output. Matching catalogue hiện tại cố định N=4 không phải baseline duy nhất ở N lớn. Để catalogue combinatorial hết RAM rồi tuyên bố MARL scale tốt là kết luận không đủ căn cứ. Arbitrator có thể trở thành bottleneck của MARL và phải tính vào latency/rounds.

Đo trained-at-size và transfer-to-larger-size riêng; dùng training budgets và model capacity tương ứng đã khóa. Report quality–latency–communication frontier thay vì một mean cost. Failure/timeout là outcome, không âm thầm bỏ trial. Timeout policy và fallback phải thống nhất trong protocol.

**Quyết định sau C:** nếu central scalable baselines không đạt deadline nhưng MARL đạt cả deadline và quality guard, có cơ sở đầu tư thuật toán scalability. Nếu central vẫn đáp ứng, size lớn tự nó không biện minh MARL.

## Những điểm còn yếu hiện tại

| Điểm còn thiếu/yếu | Bằng chứng | Experiment xử lý |
|---|---|---|
| Chưa có lợi ích so với strong central solver | Full MARL +5,34% cost so với central; local MARL +2,45% so với rule | A, sau đó B |
| Decoding chưa được tách khỏi learning | Catalogue joint argmax vs serial conditional greedy | A trước tiên |
| Chưa định vị bottleneck credit assignment | Cooperative vs independent đổi reward, critic, ratio cùng lúc | Ablation sau A |
| Waiting robustness kém ở skill mask | H24 violation local/full 29,45%/33,98%, central 2,53–3,17% | A sensitivity, B guard |
| Chưa kiểm chứng distributed execution | Chung process/frontend; byte model; serial reservation | B |
| Chưa đo scaling | N cố định 4; K/H sensitivity không thay N | C |
| Chưa chứng minh starvation-free | Request censored cuối episode; zero overdue có thể chỉ do horizon | Report request arrivals, terminal pending, censoring; cùng scoring window và nếu có follow-up/drain thì định nghĩa riêng trước run |
| Generalization còn hẹp | Split theo labeled profile; equivalence classes có thể overlap | Panel mới có claim đúng phạm vi; thiết kế unseen structure riêng nếu muốn claim đó |

## Artifact và workflow khi chuyển sang implementation

Mỗi experiment có protocol/seed registry riêng, manifest ghi checkpoint hashes, commit, device và observation/communication contract. Lưu `episodes.csv`, `request_metrics.csv`, `decoding_audit.csv`, `messages.csv` (B/C), `latency.csv`, `summary.json`, model/checkpoint nếu retrain. Schema ghi rõ schema_version, physical timestamp, message timestamp, sender/receiver, information age, action/fallback và resource-lock outcome. Logging sampling được khóa để tránh lặp lại 5,5 GB decision CSV không cần thiết; feasibility counters và aggregate counts vẫn đầy đủ.

Implement trên branch riêng, tests + smoke Mac trước, rồi commit/push. Sau push mới có exact compound lab command và exact `rsync -avhP --partial` command cho người dùng tự chạy. Training/evaluation có tqdm, artifacts dùng timestamp mới. Không SSH hay chạy full trên lab thay người dùng. Đây mới là đề xuất roadmap; chưa có runner, nên chưa có lệnh full hợp lệ để handoff.

## Nguồn phương pháp

CTDE cho phép local execution sau centralized training, nhưng không bảo đảm communication contract của bài toán cụ thể: [QMIX](https://arxiv.org/abs/1803.11485). Counterfactual baseline là một hướng xử lý multi-agent credit assignment, không phải phương án tự động cần thêm vào benchmark này: [COMA](https://arxiv.org/abs/1705.08926).
