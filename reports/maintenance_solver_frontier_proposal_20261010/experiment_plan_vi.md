# Đề xuất experiment: đường biên chất lượng–tính toán của các solver tập trung

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-10 Asia/Taipei
- Verification Status: UNVERIFIED; chưa implement hoặc chạy experiment này
- Version Label: maintenance_solver_frontier_proposal_v1
- Status: PROPOSED; seed registry, ngân sách tuning và inference protocol phải được khóa trước full run

## Mục tiêu và giới hạn

Người dùng xác nhận nghiên cứu là giả lập, chưa có yêu cầu vận hành mục tiêu.
Vì vậy không gán nhãn “thực tế”, không tự đặt deadline rồi tuyên bố planner không
đáp ứng triển khai. Câu hỏi là: trên benchmark đang nghiên cứu, rule, rollout
planner và centralized RL có những đánh đổi chất lượng–tính toán nào, và kết quả
có đủ lý do để ưu tiên một hướng phát triển tiếp theo không?

Giả thuyết kiểm tra: centralized RL có thể tạo một điểm đánh đổi có ích so với
rule và planner; cũng cho phép kết luận rule/planner đáp ứng tốt benchmark, hoặc
chưa phân biệt được các phương pháp. Không yêu cầu RL thắng.

Experiment horizon đang chạy có thể được người dùng ngắt bằng Ctrl+C trong
terminal lab. Giữ nguyên checkpoint, log và partial artifacts. Không sử dụng
phần run bị ngắt như một kết quả confirmatory hoàn chỉnh; không đổi raw manifest
thành COMPLETED. Theo workflow, assistant không SSH hoặc điều khiển lab.

## Thiết kế một experiment, chưa cần training mới

- Giữ nguyên physics, prices, waiting semantics, observations, action feasibility
  và finite-horizon objective của benchmark hiện có.
- Bốn condition hiện có: nominal, specialized, skill_mask, homogeneous;
  N=4, K theo condition hiện có, evaluation H12/H24.
- Không thêm condition để tạo lợi thế cho RL. Không suy rộng sang quy mô lớn;
  network/checkpoint hiện tại không được mặc nhiên coi là hỗ trợ N/K khác.
- Centralized RL: cả 10 final checkpoints central_matching_ppo, covered training,
  từ maintenance_training_coverage_v1_full_cpu_20261009T044236Z, seeds
  180000..180009. Không chọn seed tốt nhất, không dùng checkpoint horizon partial,
  không retrain/tune/reselect weights. Ghi hash và nguồn checkpoint đầy đủ.
- Rule: bốn reference policies hiện có, chọn một theo development objective cho
  mỗi condition; tie-break cố định. Selection H12 giống protocol trước, đánh giá
  cả H12/H24; báo rõ H24 chưa được dùng chọn rule.
- Planner: rollout planner tập trung với ba ngân sách 8/32/128 forecast scenarios.
  Mỗi ngân sách được chọn continuation trong cùng bốn rules trên development,
  không ép dùng continuation tốt nhất của rule đơn lẻ. Các root matching đều
  hợp lệ; forecast không được thấy realized future test shocks.
- Giữ lookahead bằng toàn bộ remaining horizon cho cả ba ngân sách. Không đồng
  thời đổi horizon và scenario budget trong experiment này.
- Giữ implementation planner hiện có làm đối chứng engineering. Thực hiện
  profiling và tối ưu có cơ sở nếu có bottleneck; nếu thay đổi, kiểm tra cùng
  forecasts cho root scores và matching tương đương, trong tolerance khóa trước.
  Không cố tình dùng phiên bản chậm. Chỉ gọi đây là rollout planner, không phải
  planner tối ưu toàn cục hay đại diện cho mọi phương pháp không học.

## Development/test, ngân sách và stopping

- Dùng 16 development profile labels và 32 test profile labels hiện có. Các
  profiles đã mở ở những nghiên cứu trước; không tuyên bố unseen profiles.
- Development: đề xuất 20 fresh shock seeds; test: đề xuất 20 fresh sealed shock
  seeds, chung cho tất cả phương pháp và H12/H24. Numeric seeds và derived planner
  forecast seeds chỉ khóa sau khi kiểm tra toàn bộ historical registry.
- Smoke dùng physical profiles của full TRAIN pool và seed panel riêng; không
  mở fresh full test shocks. Train panel mới: không có, vì checkpoint frozen.
- Planner forecasts dùng seed stream riêng với environment shocks, có common
  random numbers giữa root candidates và nested forecast samples giữa budgets.
  Budget khác nhau có thể sinh trajectories khác nhau; không tuyên bố trajectories
  identical chỉ vì cùng exogenous seeds.
- Sau development selection, chạy đúng toàn bộ test grid, không early stopping
  khi nhìn thấy winner, không thêm ngân sách sau khi xem test. Nếu max budget vẫn
  cải thiện, ghi rằng chưa xác định được frontier ngoài phạm vi đã thử.
- Với 1 selected rule + 3 selected planner configurations + 10 RL replicas,
  test có 14×4×2×32×20 = 71,680 episodes và 1,290,240 physical ticks.
- Development, common-state timing và profiling có fixed counts riêng phải khóa
  trước execution; không coi test count trên là toàn bộ runtime.
- Giới hạn RAM/runtime engineering được xác định bằng TRAIN-only smoke/profiling
  trước sealed test. Nếu timeout, giữ failure artifact; không diễn giải solver
  failed như một phép đo objective hoàn chỉnh và không tự thay test panel.

## Metrics và latency contract

- Primary decision evidence: objective và end-to-end decision latency, xem theo
  từng condition/horizon; không chọn condition thắng sau khi đã nhìn test.
- Secondary: cost components, waiting-violation rate, overdue ticks, served và
  horizon-censored waiting, terminal pending, failure/down time; feasibility.
- Latency: p50/p95/p99, max, cold start, warm inference; bao gồm observation
  preparation, action/matching/feasibility handling và planner forecasts.
- Dùng cùng CPU, thread settings, process isolation và runtime. Warm cache contract
  giống nhau; timing order được randomize, repetitions cố định sau smoke.
- Đo trên shared states từ development reference trajectories để tách ảnh hưởng
  solver khỏi state distribution. Đo thêm latency trên on-policy test trajectories
  để thấy workload thực, báo riêng hai phép đo, không trộn populations.
- Resource: wall time, CPU time, peak resident memory, checkpoint size; rule/planner
  tuning cost, RL training và development selection cost được ghi riêng.
- Training đã trả trước không phải chi phí bằng 0. Chỉ tính điểm hòa vốn compute
  khi có đo CPU time phù hợp cùng hardware, và inference saving thật; không cộng
  trực tiếp wall time từ hai hardware khác nhau.
- Có thể trình bày cost dưới các budget latency như sensitivity curve, nhưng các
  budget ấy là biến khảo sát, không phải operational SLA.

## Phân tích và tiêu chí định hướng

- Paired objective/waiting differences trên cùng profile/shock panel; công bố
  cả 10 training replicas của RL. Không coi các replica và hàng nghìn episode
  chia sẻ shocks là những mẫu hoàn toàn độc lập.
- Báo effect sizes và confidence intervals. Khóa trước inference method:
  ưu tiên crossed resampling profile/shock và resampling RL training replicas;
  familywise uncertainty cho các so sánh định hướng phải được định nghĩa trước
  test. CI chỉ theo training replicas phải ghi conditional on fixed test panel.
- Vẽ cost–latency frontier kèm waiting và memory. Các điểm gần nhau hoặc intervals
  rộng phải mang nhãn chưa phân biệt, không gọi dominance chắc chắn.
- Không dùng một pooled mean để che condition có waiting/cost regression. Không
  dùng “không significant” để tuyên bố equivalence. Nếu cần non-inferiority,
  practical margins phải được thống nhất trước full test, không suy ra từ data.
- Nếu rule gần planner/RL trong phạm vi uncertainty, ưu tiên làm rõ bài toán và
  cải thiện benchmark/baseline trước kiến trúc mới; đây chưa phải chứng nhận rule
  đáp ứng vận hành vì chưa có yêu cầu triển khai.
- Nếu planner tạo kết quả chất lượng mạnh trong nhiều budget và RL không có điểm
  đánh đổi hữu ích, ưu tiên nghiên cứu/cải thiện planner.
- Nếu RL có lợi thế chất lượng hoặc giữ chất lượng với compute thấp hơn và không
  có waiting regression đáng kể theo margins khóa trước, ưu tiên centralized RL;
  phải tính chi phí học và độ ổn định qua replicas.
- Nếu mỗi phương pháp có vùng lợi thế khác nhau, hướng tiếp theo là lựa chọn
  solver/hybrid; không chọn một winner duy nhất bằng cách đổi trọng số hậu nghiệm.
- Kết quả này không xác định MARL cần thiết, optimality gap của mọi planner, hay
  deployment deadline. Muốn đánh giá scale phải có experiment riêng và RL phù
  hợp dimension; muốn đánh giá robustness phải khóa perturbations có động cơ rõ.

## Artifact schema và workflow

New UTC timestamp directory, immutable raw artifacts, no silent overwrite.
Manifest: protocol version, commit, checkpoint/source hashes, actual runtime,
hardware/thread settings, seed/profile audits, expected/actual counts, run status.
Files: resolved_config.json, seed_registry.json, checkpoint_provenance.json,
development_episodes.csv, reference_selection.json, planner_selection.json,
episodes.partial.csv/episodes.csv, request_metrics.csv, feasibility_audit.json,
latency_states.json, latency.csv, resource_usage.csv, paired_metrics.csv,
summary.json; analysis và plots nằm trong report directory mới.

Implementation sau khi khóa protocol: branch codex/ riêng, tests và small Mac
smoke, tqdm development/test loops, commit/push verified code, rồi đưa người dùng
compound lab command và rsync -avhP --partial. Assistant không launch lab full run
hoặc transfer. Đây là đề xuất protocol, chưa phải code đã kiểm tra và bàn giao.
