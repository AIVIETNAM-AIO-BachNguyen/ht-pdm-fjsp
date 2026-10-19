# Kết luận nghiên cứu: RL điều phối bảo trì nguồn lực dùng chung

**Kết luận:** đã xây dựng và kiểm chứng một framework centralized RL chạy được:
simulator request/physical, observation, legal-action mask, PPO controller,
teacher/heuristic/planner baselines, split DEV/TEST, provenance và verifier độc
lập. Trên TEST mới v3, rollout-assisted PPO tốt hơn scratch PPO ở anchor, nhưng
chưa chứng minh PPO fine-tuning hơn imitation-only. RL chưa vượt planner mạnh
hoặc heuristic một cách nhất quán trên bốn case. Đây là nền nghiên cứu có thể
phát triển tiếp; chưa có bằng chứng novelty, optimality hay tính sẵn sàng cho
nhà máy. Kết luận dựa trên cả ba vòng dương và âm, không thay physics/objective
hay chọn case sau khi thấy TEST.

## Hợp đồng ra quyết định và tradeoff

Một centralized controller quyết định mỗi tick. Input: state hiện tại của bốn
máy (tuổi, hỏng, PM/CM request), hai technician (rảnh/bận, thời gian còn lại),
service eligibility/duration và clock. Không có realized future events. Request
PM phát sinh theo threshold age≥3, failure sinh CM, PM đang chờ có thể nâng
thành CM. Action là partial matching request–technician hoặc WAIT: controller
chọn subset, ưu tiên, gán worker và thời điểm start. Feasibility do mask và
simulator giữ, không phải policy tự học. Service không ngắt, một technician
không thể phục vụ hai máy, phục hồi tại completion. Chọn một job chiếm worker
nhiều tick, làm máy khác chờ, suy giảm và có thể hỏng; vì vậy một matching tốt
trước mắt không luôn tốt theo thời gian. Reward âm của PM/CM start cost,
unavailable machine-ticks và new failures, tính H48 + common risk tail16.
Không thêm waiting penalty để tạo thắng lợi giả.

## Bằng chứng tính đúng và kết quả

Kernel có unit tests cho conservation, request lifecycle, completion,
feasibility, randomness và terminal effects; independent scalar replay trước
goal đã kiểm 221.184 transitions. Framework tách kernel, controller, action
feasibility và evaluation. V1/v2/v3 targeted tests lần lượt 48/50/52 PASS;
Ubuntu smoke từng vòng PASS. Ba FULL COMPLETED và ba verifier độc lập PASS.
V3 audit còn kiểm teacher-bank physical replay/label legality, initial policy
hashes, paired training seed sequences, DEV selection, cost/reward accounting,
checkpoint hashes và toàn bộ intervals. Zero invalid assignment không chứng
minh PPO tự học ràng buộc.

| Vòng | Câu hỏi đã khóa | Kết quả TEST |
|---|---|---|
| V1, source 2fc6840 | PPO anchor so risk/rollout | Anchor PPO 817,45 vs risk 839,91, paired CI95% PPO−risk [−60,94;+12,92]; chưa xác nhận vượt risk, thua rollout 695,62. Transfer yếu. |
| V2, source 526272a | Train/chọn bốn case có tốt hơn V1? | Pooled v2−v1 +8,64, CI95% [−0,68;+18,91], H1 không đạt; specialist tốt hơn v1 41,30 nhưng anchor/scale3 tệ hơn. |
| V3, source cfb870e | Rollout teacher giúp PPO, PPO có hơn imitation-only? | Anchor teacher−scratch −44,92, CI97,5% [−62,06;−28,73], H1 đạt. Teacher−warm −6,58 [−25,27;+4,89], H2 chưa đạt. Joint rule âm tính. |

V3 giữ cùng simulator/objective và bốn case, thêm train-only teacher labels
(96 episodes, 4.608 decisions, 1.082.304 forecast transitions), BC và paired
PPO runs (1.198.080 decisions), 432 DEV và 1.536 sealed TEST episodes. Anchor
cost: teacher 769,38, warm BC 775,95, scratch 814,30, risk 802,72, rollout
703,72. Teacher thấp hơn risk nhưng vẫn cao hơn rollout 65,66. Trên
interchangeable teacher 763,05 > risk 753,84; trên specialist teacher 1.028,30
> risk 1.009,64, rollout 976,83 và warm 1.011,63. Chỉ reciprocal scale3 có
teacher 945,38 thấp hơn risk 993,41 và bounded rollout 1.039,77, là transfer
diagnostic, không phải endpoint chọn model. Health không đồng đều: anchor
failure giảm so scratch nhưng H availability teacher 60,16% hơi thấp hơn
scratch 60,24%; specialist teacher CM queue và availability kém warm.

Post hoc trace cho thấy teacher và warm có cùng cost ở 49/64 anchor seeds;
fine-tuning chỉ đổi outcome ở ít trajectories. Specialist teacher WAIT khi có
việc legal 12,47% ticks so warm 7,29%, CM queue tăng. Vì action làm trạng thái
tương lai khác nhau, các trace này gợi cơ chế chứ không xác nhận nhân quả.
Teacher giúp khởi tạo policy so scratch, có extra planner/data cost; chưa
chứng minh PPO cải thiện so chỉ imitation. V3 inference learned policy nhanh
hơn rollout trên CPU single process nhưng training/teacher compute khác lớn.

Raw source/evidence trong research_result/: framework_v1_findings_vi.md,
framework_v2_findings_vi.md, framework_v3_findings_vi.md và các thư mục
framework_v{1,2,3}_full_* cùng audit timestamped. Raw/checkpoints/logs được
giữ ngoài Git; compact reports, protocols, code và verifiers được commit/push.

## Giới hạn và hướng phát triển

N4/K2 hazard synthetic, bốn configs và finite tail chưa đại diện production
scheduling, data nhà máy hay quy mô N/K biến đổi. Controller không chọn lúc
phát request; requester threshold cố định. Flat 56-feature MLP và 21 fixed
matching actions có thể khó biểu diễn opportunity cost theo request–worker edge
và không scale tự nhiên. PPO delayed credit, exploration, teacher horizon
6 tick, imitation state coverage và DEV selection đều có thể góp phần vào
khoảng cách với rollout; hiện chưa tách được nguyên nhân. Rollout là bounded
forecast, không phải optimum và thua risk trên scale3. Các CI bootstrap theo
physical seeds không bao phủ hết uncertainty do ba training seeds và model
selection. Không kết luận RL thiếu tiềm năng, cũng không gọi framework hiện tại
là phương pháp vượt baselines tổng quát.

Vòng tiếp theo có căn cứ là structured request–worker policy và tập dữ liệu
on-policy của trạng thái mà BC/PPO tự ghé, với prelocked test mới, cùng
teacher-only/scratch/planner controls và compute. Chỉ mở production routing
hoặc RL request timing sau khi xác định thêm khả năng kiểm chứng dynamics.
Kết quả này là một mốc nghiên cứu có kết luận rõ, không phải kết thúc câu hỏi
khoa học.

## Tái lập trên Ubuntu lab

Chạy từ source commit tương ứng; mỗi run sinh thư mục mới và không ghi đè.
V3 exact procedure:

~~~bash
cd /home/bachng/ht-pdm-fjsp
uv sync --frozen
uv run pytest -q tests/test_maintenance_framework_v1.py tests/test_maintenance_framework_v2.py tests/test_maintenance_framework_v3.py tests/test_maintenance_dispatch.py tests/test_maintenance_requests.py tests/test_maintenance_request_experiment.py
RUN_ID="framework_v3_replica_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python -m ht_pdm_fjsp.maintenance_framework_v3_campaign --profile full --device cpu --output-dir "research_result/$RUN_ID"
AUDIT_ID="framework_v3_audit_$(date -u +%Y%m%dT%H%M%SZ)"
uv run python research_result/verify_framework_v3.py --run-dir "research_result/$RUN_ID" --output-dir "research_result/$AUDIT_ID"
~~~

V1/v2 reproduction và comparator checkpoint requirements còn trong từng
findings/protocol; v2 pins original v1 selected checkpoint SHA256 để tránh
âm thầm thay comparator. Mọi manifest giữ exact seed/config/source hashes.
