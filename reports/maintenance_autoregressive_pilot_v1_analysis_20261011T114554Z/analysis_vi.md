## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent.
- Origin Mode: validate, inline.
- Origin Date: 2026-10-11.
- Verification Status: VERIFIED EVALUATION AND STATISTICS; ANALYZED TRAINING.
- Version Label: maintenance_autoregressive_pilot_v1_analysis_v1.
- Source: maintenance_autoregressive_pilot_v1_full_cpu_20261011T112826Z.
- Overall Confidence: CAUTION — pilot3 training replicas,24 independent profile
  identities, fixed4 shocks/profile; diagnostics có phạm vi riêng.

## Kết luận

**Prototype đã huấn luyện và chạy đúng, nhưng chưa có bằng chứng rằng cập nhật
ngữ cảnh tài nguyên làm RL phối hợp thời điểm PM và technician tốt hơn. Policy
greedy hiện tại chủ động dùng tài nguyên khi PM vẫn khả thi, chưa thể hiện được
khả năng chủ động chừa tài nguyên/chờ.**

Run CPU hoàn tất trong618,91 giây (10 phút19 giây), đủ9 models,1105920 learned
training decisions và1920 test episodes. 238 files,34,375 MB,165 source/config
files khớp commit01f721b990ba090b10330197248a13795f0205bc; clean source, threads1.
Đã tái lập trên Mac toàn bộ1920 evaluation episodes,115200 physical transitions:
mọi episode field ngoài thời gian inference đều khớp. Thêm4800 logged transitions
và2304 ordered token likelihoods kiểm tra lại; scalar greedy matching không khác
quyết định batch ghi log. Không retrain hoặc truy cập lab; raw artifacts giữ nguyên.

## Đo điều gì và cách đọc

Mọi arm train mới trên cùng phân phối PM có lựa chọn: catalogue PPO, decoder
pair/STOP với context cập nhật, và cùng decoder giữ context đầu tick. Ba training
seeds/arm, cùng19650 parameters và cùng khởi tạo theo replica, cùng physical tapes.
H48 policy decisions cộng12 tail ticks dùng chung deadline_matching, cost càng
thấp càng tốt. Checkpoint và rule chọn trên development, trước khi mở test.
Đối chiếu hai decoder cô lập phần context. Catalogue-vs-decoder còn khác phân rã
action và quy tắc argmax toàn matching so với greedy token.

## Chi phí episode

| Policy | Dense: mọi KTV làm được mọi máy | Skill mask: có giới hạn kỹ năng |
|---|---:|---:|
| Rule chọn trên development |1159,33|1191,75|
| PPO catalogue |1212,53|1232,51|
| Decoder giữ context đầu tick |1214,07|1242,45|
| Decoder cập nhật context |1219,91|1246,95|

Rule thắng trung bình. Context decoder có chi phí cao hơn rule5,23%/4,63%,
catalogue0,61%/1,17%, frozen decoder0,48%/0,36% tương ứng dense/skill_mask.
Không so trực tiếp số tuyệt đối với thí nghiệm cũ: benchmark và seed panel mới.

## Bốn so sánh chính đã khóa

Gain = (chi phí đối chứng − chi phí context)/chi phí đối chứng; dương là có lợi.

| Cấu hình / đối chứng | Gain TB | CI98,75% | Diễn giải theo margin1% |
|---|---:|---:|---|
| Dense / catalogue |-0,61%|[-5,62%;5,71%]|Chưa rõ|
| Dense / frozen context |-0,48%|[-1,53%;0,39%]|Không đạt tín hiệu gain≥1%|
| Skill mask / catalogue |-1,17%|[-4,29%;1,40%]|Chưa rõ|
| Skill mask / frozen context |-0,36%|[-1,60%;0,73%]|Không đạt tín hiệu gain≥1%|

Cả bốn CI đều chứa0: không kết luận context gây hại hoặc tương đương đối chứng.
Riêng hai ablation có upper bound dưới1% nên hypothesis về material gain không
được ủng hộ trong pilot này. Không so với1% rồi gọi đó là proof of zero effect.
Các CI percentile dùng paired crossed bootstrap20000 draws, resample3 replicas
và24 profile identities, giữ4 shocks mỗi profile. Bốn CI98,75% là Bonferroni
xấp xỉ; chỉ3 replicas khiến coverage/độ chính xác còn hạn chế, không coi1920
episodes là1920 mẫu độc lập. Các CI dựng lại độc lập từ raw CSV và khớp summary.

Context-vs-rule là exploratory: dense gain-5,23%,CI95%[-8,69%;-2,23%]; skill
gain-4,63%,CI95%[-7,61%;-2,10%]. Đây là tín hiệu bất lợi có phạm vi pilot, không
phải additional confirmatory tests. Cả3 replicas của context đều thua rule.
So catalogue, seed16100002 context tốt hơn4,20% dense và0,62% skill; hai seeds
còn lại thua. Không lấy riêng seed tốt làm kết luận kiến trúc thắng.

## Thời điểm PM: chủ động chờ vẫn là điểm yếu

Trên toàn learned window của context policy (3 seeds ×96 episodes/condition):

| Hành vi | Dense | Skill mask |
|---|---:|---:|
| Ticks còn PM khả thi |6409|6476|
| WAIT hoàn toàn khi PM khả thi |0|0|
| Kết thúc matching khi còn có thể thêm một PM |0|0|
| PM đã bắt đầu |3235|3413|
| PM bắt đầu dưới ngưỡng rủi ro tức thời |464 (14,34%)|628 (18,40%)|

Tổng12885 PM-feasible ticks không có lần chủ động WAIT hay chừa chỗ cho một PM
còn khả thi. Không tính forced STOP khi mọi technician bận là quyết định chờ.
Rule có160/2262=7,07% và155/2250=6,89% WAIT trên ticks PM khả thi.
Frozen decoder có40/6428=0,62% và67/6536=1,03%; catalogue cũng không WAIT.

Đây là hành vi của **greedy/argmax evaluation**. Network vẫn cho STOP xác suất
dương: ở các audit states còn action pair, mean first-STOP probability của context
là5,28% dense và7,03% skill, nhưng STOP không thắng argmax. Chưa đo outcome của
stochastic deployment; không khẳng định policy chưa bao giờ sample STOP lúc train.

Không chọn WAIT cũng không có nghĩa RL không trì hoãn máy nào:4 máy/2 technicians
vẫn buộc nhiều máy đợi khi workers bận, và việc ưu tiên máy là quyết định thời điểm
ngầm. Phát hiện ở đây là **thiếu lựa chọn chủ động giữ worker rảnh khi còn việc
PM khả thi**, không phải phủ nhận mọi dạng scheduling. PM dưới ngưỡng cũng không
tự động là sai: duration/restoration/rủi ro tương lai có thể làm PM sớm có lợi.
Cần cùng-state counterfactuals với continuation của policy để gán đúng/sai.

## Technician cho PM: đã có lựa chọn, chưa chứng minh chọn tốt

Context policy có1416 PM starts dense và736 skill với hai technicians ban đầu
khả thi. Nếu giữ đúng tập máy đã chọn và mọi CM assignments, có497 dense và131
skill ticks vẫn thay được technician cho PM (994/262 PM edges có lựa chọn).
Các CM-only alternative ticks là339/126 và không được dùng cho kết luận PM.
Chữ ký benchmark đảm bảo focal healthy machine có hai workers ở tick0, nên
assignment không bị đóng cứng bởi input. Nhưng busy workers và matching đầy
vẫn làm nhiều closed-loop ticks không có lựa chọn assignment riêng.

Số choice không chứng minh chất lượng: tổng cost chưa thắng các đối chứng, và
pilot này không rollout cùng-state alternative PM technician sau lựa chọn.
Không có cơ sở nói thuật toán đã chọn technician tốt hoặc đã học được timing
và assignment tốt đồng thời. Cũng không có cơ sở nói RL không thể học được.

## Decoder có dùng context cập nhật không?

Diagnostic exploratory thêm trên các audit trajectories có2 profiles ×2 shocks,
không phải full panel: giữ **nguyên trọng số đã học** của context policy, chỉ
thay remaining-resource summaries ở bước sau cặp đầu bằng context đầu tick.
Không train lại và không thay mask. Đây là kiểm tra độ nhạy output, không phải
một phương pháp đối chứng đã đánh giá cost hay phép chứng minh nhân quả training.

| Cấu hình | States sau cặp đầu | States bước2 còn pair/STOP choice | Greedy token đổi | Median total variation |
|---|---:|---:|---:|---:|
| Dense |252|52|0|0,000123|
| Skill mask |262|50|0|0,000328|

Ngữ cảnh vector có thay đổi (median L2≈0,223/0,206), nhưng phân phối token đổi
rất nhỏ; tối đa total variation0,00129/0,00210. Không có đổi hành động ở102 states
có lựa chọn bước2. Tín hiệu này gợi ý phần context động hiện ảnh hưởng yếu đến
decision head, phù hợp việc ablation chưa có gain. Không ngoại suy102 states
này thành toàn bộ24 profiles hoặc mọi kiến trúc context khác.

Ở N4/K2, cập nhật bắt đầu **sau cặp đầu**; bước đầu của cùng weights có context
giống nhau, và chỉ một technician còn rảnh ở bước2. Vì vậy cập nhật không trực
tiếp đổi technician của cặp đầu qua forward pass; nó có thể tác động qua learning
credit của cả sequence. Điều này hạn chế sức mạnh của pilot nhỏ để kiểm tra
benefit của cập nhật nhiều lần ở K lớn; cũng không phải lý do bỏ qua kết quả âm.

## Có học hay chỉ chạy code?

Trọng số cả9 models thay đổi; training/development cost giảm rõ từ policy khởi
tạo. Ví dụ context seed16100000 development mean từ2207,13 xuống1121,83;
seed16100001 từ1856,92 xuống1113,29; seed16100002 từ1748,54 xuống1089,75.
Đây là learning về behavior/cost trên development, không phải superiority hoặc
optimal timing. Các selected context checkpoints là10/40/10; frozen10/40/10;
catalogue10/40/40. Không chọn checkpointinitial0 dù có đo, không đổi bằng test.

Context development gần plateau từ update10; train tới40 không cải thiện đều.
Giảm training cost ở các episodes khác nhau không tự động là causal evidence
cho context. Chỉ train dài hơn hoặc thêm seeds mà giữ nguyên learning setup chưa
có bằng chứng sẽ giải quyết thiếu selective waiting.

## Bước tiếp theo đề xuất

Giữ catalogue và frozen decoder làm controls. Trước khi mở rộng K/N hoặc chạy
nhiều seeds hơn, kiểm tra tín hiệu học và score của **STOP/thời điểm** trên những
trạng thái đã xác nhận bằng independent rollout là chờ tốt hơn hoặc làm ngay
tốt hơn. Đồng thời kiểm tra cách resource summaries tương tác với score từng
cặp, vì forward sensitivity hiện rất yếu. Các khả năng như chuẩn hóa value/return,
tách phần representation actor/critic hoặc dùng query–candidate interaction
cần được chốt thành ablation mới, không gọi là nguyên nhân đã xác minh.
Không sửa code experiment hay khởi chạy thí nghiệm mới trong lượt phân tích này.

## Statistical/methodological fallacy scan:11/11 checked

| Loại | Đánh giá trong phân tích này |
|---|---|
| Simpson |Tách dense/skill, replicas và4 initial-risk scenarios. Mean context-vs-catalogue khác dấu ở seed02; không che giấu heterogeneity.|
| Ecological |Không suy cost episode mean thành mọi decision PM/technician đúng.|
| Berkson |Generator chỉ dùng structural constraints/seed; không lọc profile bằng outcome. Choice strata chỉ mô tả.|
| Collider |PM choice opportunities phụ thuộc policy trajectory; không so conditional means giữa policies như causal timing gain.|
| Base rate |WAIT có denominator PM-feasible; STOP forced tách choice; PM choices tách CM.|
| Regression to mean |Initial vs trained dev improvements là mô tả; trained controls và test dùng cho architecture comparison.|
| Survivorship |Đủ9 planned models và toàn1920 episodes; không drop seed/profile xấu.|
| Look-elsewhere |Báo đủ4 primary contrasts;2 rule contrasts và context sensitivity ghi exploratory. Không chọn best seed.|
| Forking paths |Fixed protocol, budget, seed registry và dev selection verified; post-hoc diagnostics không nâng thành confirmatory.|
| Correlation/causation |Paired trained-arm comparison kiểm tra architecture; PM ages/WAIT/resource sensitivity không xác minh nguyên nhân cost gap.|
| Reverse causality |Architecture đặt trước training; behavior do trajectory/cost cùng tác động nên không gán directional mechanism từ mô tả.|

## Reproducibility và artifacts

Evaluation replay: REPRODUCIBLE, exact non-latency outputs cho toàn1920 episodes;
4800 raw logged transitions khớp,2304 log-likelihoods within5e-5 float tolerance;
statistical replay khớp tới numerical tolerance1e-10. Không so timing Mac/lab.
Training replay: NOT RUN, chỉ kiểm tra journals/budgets/checkpoints/initialization;
không gọi toàn bộ stochastic training là đã tái lập. Development có55 rows với
served-wait mean/p95 trống vì chưa có served CM request ở cutoff; đã kiểm tra
served_requests=0, đây là mean không xác định, không phải PPO nonfinite update.
File-level hash check lặp lại sau analysis để đảm bảo raw artifacts bất biến.

- audit.json: integrity, seed/split, replay và likelihood gates.
- primary_and_reference_results.csv: effect sizes/CI dựng lại độc lập.
- profile_gains.csv,replica_means.csv,initial_scenario_means.csv.
- pm_behavior.csv,logged_policy_diagnostics.csv.
- context_functional_sensitivity.csv,context_sensitivity_summary.json.
- training_curves.csv,development_curves.csv,selected_models.csv.
- pilot_results.png/pdf,development_learning.png/pdf.
