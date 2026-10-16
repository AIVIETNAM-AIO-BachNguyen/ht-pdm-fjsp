# Sampling không giải quyết gap của joint-clearing Gaussian

Chẩn đoán đã khóa hoàn tất **COMPLETED/audit PASS trong231,769466giây**:24frozen
models ×64fresh physical seeds ×(1mean+4action repeats)=7.680episodes,
368.640policy decisions,460.800physical ticks. Không update/teacher/temperature
search/checkpoint reselection/scientific test. Source FULL vẫn không đạt7/13.

Protocolcd45cf7, code6f410de,7testsPASS3,10s rồi commit/push trước chạy trên lab.
Artifact `technician_joint_execution_diagnostic_v1_20261016T124544178125062Z`.
Fresh physical260100000..63/action260200000..11,76raw/100derived auditPASS.
Source690be2d,451bindings, selected weights và source manifest giữ nguyên.

## Mean versus stochastic, mọi method đều giữ đủ repeats

| Method | Mean cost/máy | Sampled cost/máy | Mean failure/máy | Sampled failure/máy |
|---|---:|---:|---:|---:|
| Joint |277,406250|282,344076|0,949219|1,158529|
| Forward |279,605469|284,411784|1,126302|1,146484|
| Reverse |279,100260|276,791992|1,110677|0,950195|
| Independent |278,858073|292,431315|1,024740|1,511719|
| PPO |261,417969|262,174479|0,114583|0,224284|
| PPO normalized |261,949219|262,752604|0,097656|0,254232|
| MAPPO local |269,486979|271,867513|0,611979|0,734701|
| MAPPO full |264,671875|265,787435|0,360677|0,432617|

Cost gồm48policy+12common-tail ticks; failure chỉ48policyticks, chia4máy.
Mỗi sampled decision chỉ lấy MỘT action. Bốn repeats là các trajectory độc lập
về action RNG trên cùng64physical tapes; lấy trung bình tất cả, không best-of4.
Chúng không phải thêm training replicas và không tạoCI/pvalue.

Joint sampled cost cao hơn own mean **1,779998%**. Ba replica có gain lần lượt
−1,936892% /−1,285830% /−2,111094%: cả ba đều kém hơn. Joint availability giảm
0,672553→0,667677, pending/máy tăng0→0,014323 và failures tăng0,949219→1,158529.

Gate cố định yêu cầu own-mean gain≥1%, mọi replica gain>0, đồng thời13gate chất
lượng/health cũ đều đạt với stochastic core controls nhưng PPO/norm/MAPPO MEAN
làm strong references. Chỉ6/13gate đó qua; eligibility tổng thểFALSE. Không mở
stochastic-deployment training trial hoặc tìm nhiệt độ/variance khác từ kết quả.
Reverse cải thiện khi sampling vẫn không gần PPO/đạt failure bound; không đổi
candidate sau khi nhìn thấy kết quả.

## Kiểm chứng

451source bindings kiểm trước/sau;0invalid, reward-cost khớp, cùng exogenous
populations giữa methods/modes/repeats; đủ96sample behavior log/value replays
trong tolerance source. Mean select B1 khớp semantics gốc, không đổi do batch
rounding. Tất cả checkpoint digests và grad fields không đổi.

Đã replay độc lập toàn460.800transitions với primitive helpers kiểm chứng/unit
tests từ vòng trước, gồmraw observation hashes, latent/categorical decoder,
state/event chains, transition/aggregate costs,48cutoff health và common tail.
Verifier arithmetic riêng d40adde kiểm83diagnostic+451sourcebindings, đủ7.680grid,
mean/per-replica summaries,13gate, behavior/physical-audit counts: PASS. Nó không
chạy lại primitive replay lần hai; giới hạn đó ghi trong verification.json.

Kết quả bác bỏ cách sửa cụ thể “chỉ đổi deployment từ mean sang một sample là
đủ” trên checkpoint/panel này; không chứng minh mọi mean projection đều tối ưu
hay mọi Gaussian policy vô dụng. Gap train/deploy vẫn là đặc tính phải nêu rõ,
nhưng không phải lời giải được dữ liệu hiện tại ủng hộ. Không rescue sourceFULL,
không mở test xác nhận hoặc largerN, không teacher/fine-tuning weights.

Lượt nghiên cứu kế tiếp nếu chọn action factorization từ raw-state PPO cần
protocol riêng trước học, mọi weights random, cơ chế reservation có ablation
cùng thông tin/capacity/budget, giữ strong references và health limits ban đầu.
Không gán causal failure cho pooling từ so sánh khác kiến trúc/distribution.

Copy thủ công về Mac khi cần (agent không thực hiện transfer):

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" &&
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```
