## Material Passport

Dữ liệu mô phỏng tổng hợp, không có dữ liệu người tham gia. Phân tích run
`maintenance_balanced_algorithm_benchmark_v1_full_cpu_20261012T115140Z`
trên setting 2–2 cố định `balanced_a2_b4_u12`, code commit `2e19043`.
Nguồn: [/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_balanced_algorithm_benchmark_v1_full_cpu_20261012T115140Z](/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_balanced_algorithm_benchmark_v1_full_cpu_20261012T115140Z/manifest.json).
Bản raw chỉ được đọc. Verification Status: **ANALYZED**; không chạy lại training.
Phân tích số học độc lập bởi agent chính trong cùng cuộc trò chuyện, không phải
review mù hoặc xác minh bằng model khác.

## Kết quả chính

**PPO có cost trung bình thấp nhất trong bốn thuật toán học. Joint rollout có
cost trung bình thấp nhất toàn bộ benchmark; CI chưa phân biệt chắc chắn PPO
với joint rollout. Run này chưa chứng minh lợi thế của MARL.**

Run hoàn tất trong 101.31 phút trên CPU.
Hoàn tất lúc 21:33 ngày 12/10/2026, múi giờ Asia/Taipei. Đủ 20 model, mỗi model
491.520 learned transitions; tổng 9.830.400 learned / 12.288.000 physical ticks.
Đủ 6.400 episode development và 6.400 episode test trên 256 physical seed mới.
Mỗi thuật toán học giữ đủ năm training replica; mean không chọn seed tốt nhất.

Cost dưới đây tính cả 48 bước quyết định và 12 bước tail chung; thấp hơn là tốt.
Cột giảm cost so h6 là point estimate; các so sánh baseline với baseline chỉ
mang tính mô tả. Availability chỉ đo trong 48 bước H. Pending H đếm máy hỏng
đang chờ, chưa được technician nhận; không bao gồm máy đang bảo trì.

| Phương pháp | Mean cost H+L | Giảm cost so h6 | Availability H | Pending H |
|---|---:|---:|---:|---:|
| Joint rollout | 1046.32 | +3.97% | 66.740% | 0.0000 |
| PPO | 1050.08 | +3.62% | 66.793% | 0.0000 |
| Assignment rollout | 1056.27 | +3.06% | 66.429% | 0.0000 |
| MAPPO | 1086.93 | +0.24% | 66.773% | 0.0156 |
| Isolated h6 | 1089.55 | +0.00% | 66.960% | 0.0000 |
| Isolated endpoint | 1092.80 | -0.30% | 66.956% | 0.0000 |
| QMIX | 1145.80 | -5.16% | 67.272% | 0.0688 |
| Deadline matching | 1147.46 | -5.31% | 66.636% | 0.0938 |
| IQL | 1151.96 | -5.73% | 67.198% | 0.1258 |

## Hiệu ứng và độ bất định đã đăng ký

Primary family gồm 20 so sánh cost và tám so sánh health. CI từng so sánh
99,8214% (1−.05/28), 40.000 crossed bootstrap, resample độc lập training-replica
và physical-episode, giữ paired indices giữa phương pháp. Đây là CI bootstrap
xấp xỉ, đặc biệt chỉ có năm training replica. Dấu dương là cost thấp hơn.
Không có p-value đăng ký; không tạo thêm p-value sau khi xem kết quả.

| So sánh | Giảm cost | CI đăng ký | Diễn giải |
|---|---:|---:|---|
| PPO / Deadline matching | +8.486% | [+6.804, +10.161]% | Giảm cost, lower CI ≥1% |
| PPO / Isolated h6 | +3.623% | [+2.482, +4.820]% | Giảm cost, lower CI ≥1% |
| PPO / Isolated endpoint | +3.909% | [+2.831, +5.019]% | Giảm cost, lower CI ≥1% |
| PPO / Assignment rollout | +0.586% | [-0.039, +1.221]% | Chưa phân biệt chắc chắn |
| PPO / Joint rollout | -0.360% | [-0.899, +0.170]% | Chưa phân biệt chắc chắn |
| MAPPO / Deadline matching | +5.275% | [+1.878, +8.369]% | Giảm cost, lower CI ≥1% |
| MAPPO / Isolated h6 | +0.241% | [-3.220, +3.208]% | Chưa phân biệt chắc chắn |
| MAPPO / Isolated endpoint | +0.537% | [-2.962, +3.532]% | Chưa phân biệt chắc chắn |
| MAPPO / Assignment rollout | -2.903% | [-6.355, -0.171]% | Cost cao hơn |
| MAPPO / Joint rollout | -3.882% | [-7.347, -1.173]% | Cost cao hơn |
| IQL / Deadline matching | -0.392% | [-2.692, +1.519]% | Chưa phân biệt chắc chắn |
| IQL / Isolated h6 | -5.727% | [-7.910, -3.791]% | Cost cao hơn |
| IQL / Isolated endpoint | -5.413% | [-7.620, -3.447]% | Cost cao hơn |
| IQL / Assignment rollout | -9.060% | [-11.126, -7.185]% | Cost cao hơn |
| IQL / Joint rollout | -10.097% | [-12.200, -8.150]% | Cost cao hơn |
| QMIX / Deadline matching | +0.144% | [-1.488, +1.790]% | Chưa phân biệt chắc chắn |
| QMIX / Isolated h6 | -5.162% | [-6.884, -3.523]% | Cost cao hơn |
| QMIX / Isolated endpoint | -4.850% | [-6.583, -3.159]% | Cost cao hơn |
| QMIX / Assignment rollout | -8.477% | [-10.072, -6.913]% | Cost cao hơn |
| QMIX / Joint rollout | -9.508% | [-11.162, -7.911]% | Cost cao hơn |

PPO tốt hơn chắc chắn ba heuristic (deadline, h6, endpoint), với lower CI vượt
ngưỡng cải thiện thực tế 1%. PPO so assignment rollout: +0,586%, CI
[−0,039%, +1,221%]; chưa chắc chắn. PPO so joint rollout: −0,360%, CI
[−0,899%, +0,170%]; chưa chắc chắn. CI chứa zero không chứng minh hai phương
pháp tương đương, vì không đăng ký kiểm định tương đương.

MAPPO chỉ vượt deadline chắc chắn; so h6 gần zero, CI rộng. MAPPO kém cả hai
rollout chắc chắn trong family đã đăng ký. IQL/QMIX kém h6, endpoint và hai
rollout; chưa phân biệt chắc chắn với deadline.

## Health guard

Guard: lower CI availability difference ≥−1 điểm phần trăm; upper CI pending
H difference ≤0,1 máy so h6. PASS không có nghĩa health được cải thiện.

| Thuật toán | Δ availability so h6, CI (điểm %) | Δ pending H, CI (máy) | Guard |
|---|---:|---:|---|
| PPO | -0.1664 pp [-0.2702, -0.0718] | +0.0000 [+0.0000, +0.0000] | PASS |
| MAPPO | -0.1868 pp [-0.3563, -0.0306] | +0.0156 [+0.0000, +0.0526] | PASS |
| IQL | +0.2384 pp [-0.0411, +0.3912] | +0.1258 [+0.0273, +0.3359] | Chưa đạt |
| QMIX | +0.3125 pp [+0.2275, +0.3964] | +0.0688 [+0.0248, +0.1250] | Chưa đạt |

PPO uptime giảm 0,1664 điểm phần trăm, CI hoàn toàn âm, nhưng nằm trong margin
cho phép. Pending H bằng zero ở mọi test episode của PPO. MAPPO cũng mất nhẹ
uptime và vẫn PASS. IQL không đạt backlog guard. QMIX có point backlog 0,06875
<0,1 nhưng upper CI 0,125 >0,1: **chưa xác lập được backlog non-inferiority**;
không diễn giải là mọi episode đều vi phạm margin.

## Vì sao PPO giảm cost? (mô tả, chưa phải causal ablation)

So h6, cost PPO giảm 39.476.
Phân rã H+L: maintenance cost tăng
0.851, downtime cost tăng
8.166, failure cost giảm
48.492. Waiting cost bằng zero.
Lợi ích cost chủ yếu đến từ failure cost thấp hơn; không phải uptime tốt hơn.
Trong H, số failure trung bình giảm từ 2,945 xuống 0,627.

PPO H-only cost 836,738 so h6 866,320, giảm khoảng
3.41%; tail cost cũng thấp hơn
(213,341 so 223,234). Không thấy lợi ích chỉ do đẩy cost ra ngoài H.
Tuy vậy PPO tail pending mean 0,009375, còn joint rollout bằng zero; pending H
bằng zero không đồng nghĩa hết backlog sau tail hoặc mọi máy đều khỏe.

## Technician skill và tranh chấp (exploratory)

Fast technician là A trên M0/M1, B trên M2/M3, duration1/restoration.5.
Slow technician trên các máy tương ứng có duration2 hoặc4/restoration1.
Tỷ lệ dưới đây tính assignment thực thi trong H, không phải chỉ proposal.
Rejection là số proposal bị arbiter loại mỗi episode, không phải assignment sai.

| Phương pháp | Tỷ lệ dùng technician chậm | Rejection/episode | Failure H/episode | CPU ms/quyết định |
|---|---:|---:|---:|---:|
| PPO | 3.267% | 0.000 | 0.627 | 0.307 |
| MAPPO | 0.010% | 4.774 | 2.148 | 0.334 |
| IQL | 0.641% | 13.507 | 6.201 | 0.275 |
| QMIX | 0.025% | 13.137 | 5.888 | 0.159 |
| Isolated h6 | 9.289% | 0.000 | 2.945 | 0.177 |
| Assignment rollout | 3.309% | 0.000 | 0.801 | 10.510 |
| Joint rollout | 2.689% | 0.000 | 0.660 | 17.395 |

PPO dùng technician chậm khoảng 3,27% assignment, gần hai rollout; chủ yếu
trên các máy có slow duration2. PPO không chọn slow duration4 trong test panel
này. MAPPO/QMIX gần như luôn chọn technician nhanh; IQL cũng >99% nhanh.
Đây là dấu hiệu policy địa phương thiên về lựa chọn nhanh, phối hợp chưa tốt,
không chứng minh nguyên nhân duy nhất của cost cao.

Các thuật toán local còn proposal bị tranh chấp: MAPPO ~4,77/episode, IQL
~13,51, QMIX ~13,14. PPO chọn matching chung nên không phát sinh loại tranh chấp
proposal này theo thiết kế. So sánh rejection của PPO với local actors không
phải một ablation công bằng để xác định hiệu quả learning rule.

Assignment rollout thấp hơn h6 3.06% ở panel này:
việc chọn technician vẫn có giá trị trong setting 2–2. Đây là so sánh baseline
mô tả ngoài primary CI family, không gán CI confirmatory mới. Joint rollout
thấp hơn assignment rollout khoảng
0.94%.
Assignment rollout giữ subset của h6 tại current root trên state riêng của nó;
không phải toàn bộ trajectory luôn có cùng subset với trajectory h6.

Waiting/overdue cost bằng zero ở mọi phương pháp, nên benchmark này không cung
cấp bằng chứng về ích lợi xử lý overdue penalty dù knob waiting_price12 tồn tại.
Inference latency PPO ~0,307ms so joint rollout ~17,395ms, chậm hơn PPO khoảng
56.6 lần ở phép đo này. Đây là timing mô tả
batch1 trên lab CPU, loại trừ training cost, không phải đánh giá deployment.

## Training replica và development curves

| Thuật toán | Khoảng mean test cost của năm replica | Replica thấp hơn h6 | Mean training wall/model, bỏ callback |
|---|---:|---:|---:|
| PPO | 1045.48–1051.98 | 5/5 | 154.8 s |
| MAPPO | 1058.12–1120.11 | 3/5 | 224.5 s |
| IQL | 1146.71–1172.93 | 0/5 | 332.1 s |
| QMIX | 1142.15–1146.71 | 0/5 | 326.0 s |

PPO ổn định hơn MAPPO trong năm seed quan sát. MAPPO có ba replica tốt hơn h6,
hai replica kém hơn: không chọn ba seed tốt để làm kết luận. IQL/QMIX không
replica nào thấp hơn h6. Trên toàn 256 seed, số panel hành vi độc nhất: PPO3,
MAPPO5, IQL2, QMIX2. Trùng hành vi quan sát không có nghĩa trùng tham số hay
làm giảm số training replica đã giữ theo protocol.

Development PPO tiến bộ rõ qua các budget; MAPPO cũng tiến bộ nhưng phân tán.
QMIX chọn checkpoint122.880 cho cả năm replica; bốn seed có dev cost bằng nhau
ở nhiều checkpoint nên rule tie chọn checkpoint sớm nhất. IQL chọn 245.760
hoặc368.640. Đây là dấu hiệu plateau trên panel development hiện tại, không
chứng minh mọi configuration của Q learning không thể học.

![CI giảm cost](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_balanced_algorithm_benchmark_analysis_20261012/analysis_20261012T135523Z/cost_vs_h6.png)

![Development learning curves](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_balanced_algorithm_benchmark_analysis_20261012/analysis_20261012T135523Z/development_learning_curves.png)

## Audit và provenance

- 338 output bindings khớp SHA256/size, tổng 519.953.778 bytes; partial/final
  episode CSV identical, checkpoint-choice seal đúng.
- Grid test 25 controller replica ×256 seed, grid development 20 model ×5
  checkpoints (gồm initial) ×64 seed, 100 checkpoint files đều đúng.
- Tính lại checkpoint selection từ development đúng rule minimum/earlier tie.
- 204.800 training episodes, 51.200 paired initial-state/shock populations;
  physical role seed và population hash đều khớp giữa bốn thuật toán.
- Optimizer counts PPO/MAPPO7.680/model, IQL/QMIX30.529/model; final hash và
  selected-checkpoint hash khớp progress log. Equal samples không equal compute.
- Lab scalar replay PASS cho384.000 physical ticks,307.200 coordination rows,
  zero invalid assignments; hai test seed đầu đã được replay checkpoint/planner.
  Mac không chạy lại toàn bộ simulator/training.
- Tính lại đủ40.000 bootstrap draws bằng Cartesian indexing thay weighted
  count/einsum, cả28 CI khớp; sai số lớn nhất 2.64e-15.
- Plot phase ở temporary learner venv thiếu matplotlib sau khi đã lưu xong số
  học. Tạo figure bằng main venv hiện có, không rerun inference/training.
  `verification.json`, `plot_manifest.json` giữ riêng provenance hai phase.

## Statistical fallacy scan — 11/11 checked

| Loại | Kết quả kiểm tra trên dữ liệu này |
|---|---|
| Simpson | PPO5/5 thấp hơn h6, IQL/QMIX0/5; MAPPO3/5 thấp hơn nên mean không đại diện mọi training replica. |
| Ecological | Đơn vị inference là training replica và physical episode; không suy hiệu ứng riêng từng máy từ mean. |
| Berkson | Setting đã được chọn trước; dữ liệu chỉ nói về setting này. Test mới, checkpoint chọn dev riêng. |
| Collider | Không lọc episode theo collision, failure hoặc success; cơ chế chỉ mô tả, không điều kiện hóa thành causal effect. |
| Base rate | p=.45 là conditional hazard; số failure thực tế được báo cáo, không xem45% là prevalence episode. |
| Regression to mean | Gain so common fresh h6 trên panel mới; không dùng mức tăng của round setting search làm bằng chứng thuật toán. |
| Survivorship | Đủ20/20 model, năm seed cho mỗi algorithm, không bỏ replica kém hay checkpoint training thất bại. |
| Look elsewhere | Hiển thị toàn20 cost và8 health CI; mechanism/baseline-vs-baseline ngoài family được đánh dấu exploratory. |
| Forking paths | Budgets, seeds, lựa chọn checkpoint và CI family khớp protocol; không retune bằng test hoặc chọn best seed. |
| Correlation/causation | Failure-cost decomposition là accounting; collision association chưa phải causal ablation, không suy lợi ích nhà máy. |
| Reverse causality | Matching từ state trước transition; không thấy lấy outcome tương lai vào action trong contract/source audit. |

## Giới hạn diễn giải

PPO có global execution, còn MAPPO/IQL/QMIX có local observation và fixed
central arbiter. IQL bốn network độc lập; QMIX shared network. Rollouts biết
simulator physics và dùng128 forecast scenarios. Vì khác information, action
factorization, sharing và compute, bảng này không cô lập lợi ích của learning
rule hoặc tính multi-agent. Các baseline feedforward cũng không là replication
đầy đủ nguyên paper. Năm replica và một setting tổng hợp chưa là bằng chứng
scale/generalization/field validity. Kết quả hiện tại hỗ trợ mô tả rằng học
matching chung có thể đạt cost gần planner với inference nhẹ; lợi thế của
MARL trên setting này vẫn chưa được xác lập.
