## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent.
- Origin Mode: validate, inline.
- Origin Date: 2026-10-11.
- Verification Status: SOURCE AND STATISTICS VERIFIED; cross-platform forecast replay PARTIALLY REPRODUCIBLE (1/158784 futures).
- Version Label: maintenance_ar_decision_diagnostic_v1_analysis_v1.
- Source: maintenance_ar_decision_diagnostic_v1_full_cpu_20261011T123034Z.
- Overall Confidence: CAUTION — exploratory follow-up,3 training replicas,12 physical profile identities; intervals conditional on fixed MC tapes.

## Kết luận

**Điểm yếu được chỉ ra rõ hơn hiện nay là phân công technician cho PM trong
điều kiện dense. Chưa đủ bằng chứng rằng policy nói chung nên chờ nhiều hơn,
và chưa chứng minh phối hợp sửa cả timing và assignment tốt hơn chỉ sửa một
thành phần ở từng trạng thái.**

Giữ nguyên trạng thái, thời điểm, tập máy được chọn và mọi việc sửa máy hỏng,
đổi technician bằng lựa chọn trên32 futures độc lập đã giảm chi phí trung bình
8,70 trên74 trạng thái có lựa chọn ở dense; CI99,17%[0,16;22,99]. Cả3 replicas
có gain assignment dương. Đây là correction do simulator chọn, không phải
RL đã học được phân công tốt. Gain này khoảng0,76% của mean remaining cost
trên đúng các trạng thái eligible; không phải giảm8,70% chi phí episode.

Hoãn có chọn lọc cho gain trung bình1,52 dense/2,27 skill_mask, nhưng cả hai CI
đều chứa0. Hoãn TẤT CẢ PM đã chọn lại làm remaining cost tăng trung bình7,02/8,37.
Vì vậy không thể lấy việc greedy policy không WAIT rồi kết luận mọi PM sớm là
sai, hoặc thêm WAIT thường xuyên sẽ giải quyết được bài toán.

## Thiết kế và kiểm chứng

Run dùng branch codex/maintenance-ar-decision-diagnostic-v1, commit
89eb91e10e99a707ab3886915e14790ecf2ebaae, clean source và CPU threads1. Hoàn tất
trong210,18 giây (3 phút30 giây),906 artifacts,24,565 MB. Có đúng3 models,
144 trajectories,6912 trajectory ticks,720 landmark states,1654 unique roots,
7.810.560 physical forecast transitions; không có training mới.

Ba context PPO checkpoints phát sinh từ pilot trước, được chọn trên development
ở updates10/40/10 và pin hash. Không chọn riêng seed tốt. Diagnostic có12 cấu
hình mới ×2 physical shocks ×2 điều kiện kỹ năng ×3 replicas. Landmarks khóa
0,6,18,30,42. Source outcomes đã được mở trước khi thiết kế follow-up; các
kết luận của run này là exploratory, không đổi thành confirmatory.

Ở mỗi trạng thái giữ nguyên root CM (corrective maintenance: sửa máy đã hỏng).
PM (preventive maintenance: bảo trì phòng ngừa) chỉ được bỏ bớt hoặc đổi worker
cho những máy mà RL đã chọn; không thêm máy PM mới. Timing giữ technician gốc;
assignment giữ nguyên tập máy; joint cho phép cả hai. Tối đa7 roots/state.
Policy chạy tiếp từ t+1..47 với remaining=48-current_tick, rồi chung rule ở48..59.
Không reset thêm48 quyết định từ landmark, không truncate phần tail.

Mỗi root được chọn bằng mean của32 selection futures, trước khi dùng64 evaluation
futures độc lập. Roots chia sẻ tape trong từng phase. Mọi gain âm được giữ.
Không gọi sample minimum trên evaluation là oracle hay dùng nó để chọn phương án.

Đã kiểm tra hash/bytes mọi906 artifacts,169 source/config bytes khớp Git commit,
source/checkpoint contracts, seed-role audits và59.941 physical exclusions.
Tái lập tất cả144 trajectories trên Mac:6912 states/actions/transitions/costs khớp.
Tái lập toàn bộ720 states,5964 sequence probabilities và1279 prefixes; score
sai số trong5e-5. Dựng độc lập6 bootstrap contrasts và216 profile cells, khớp
summary đến1e-10. Raw artifacts và parameter hashes giữ nguyên.

## Sáu so sánh chính đã khóa

Gain = cost của root RL trừ cost của root được chọn trên selection. Số dương
có lợi. Đơn vị là **chi phí còn lại tới hết tick59, mỗi trạng thái có lựa chọn**;
không cộng các gain landmark thành episode gain. Denominator giữa assignment
và timing/joint khác nhau nên không so trực tiếp8,70 với7,42 để kết luận
assignment luôn tốt hơn joint.

| Điều kiện | Can thiệp bước đầu | Gain | CI99,17% | Có lựa chọn / panel | Diễn giải |
|---|---|---:|---|---:|---|
| Mọi technician làm được mọi máy | Chỉ hoãn PM | +1.52 | [-1.75; 7.72] | 130/360 | Chưa rõ |
| Mọi technician làm được mọi máy | Chỉ đổi technician | +8.70 | [0.16; 22.99] | 74/360 | Tín hiệu lợi ích trong diagnostic |
| Mọi technician làm được mọi máy | Kết hợp | +7.42 | [0.55; 20.10] | 130/360 | Tín hiệu lợi ích trong diagnostic |
| Có giới hạn kỹ năng | Chỉ hoãn PM | +2.27 | [-0.69; 6.80] | 146/360 | Chưa rõ |
| Có giới hạn kỹ năng | Chỉ đổi technician | +7.81 | [-3.48; 33.47] | 28/360 | Chưa rõ |
| Có giới hạn kỹ năng | Kết hợp | +5.66 | [-0.24; 15.37] | 146/360 | Chưa rõ |

CI dùng paired crossed bootstrap20000 draws, resample3 replicas và12 profile
identities; physical shocks vàlandmarks nằm trong cell. Sáu interval99,1667%
là Bonferroni xấp xỉ. Không coi720 states hoặc158784 future rollouts là720/158784
mẫu độc lập. Chỉ3 replicas làm coverage của bootstrap hạn chế. MC SE của từng
root được lưu riêng; CI chính điều kiện trên tapes đã dùng, không tích phân
mọi nguồn bất định simulator.

Assignment và joint dense có lower bound dương nhưng khá gần0, nên đây là tín
hiệu có phạm vi diagnostic, chưa phải xác nhận một thuật toán mới. Skill_mask
chỉ có28 assignment opportunities; khoảng tin cậy rất rộng. CI chứa0 không
chứng minh không có lợi ích hay hai phương án tương đương.

| Điều kiện | Can thiệp | Seed16100000 | Seed16100001 | Seed16100002 |
|---|---|---:|---:|---:|
| dense | timing | -0.004 | +1.376 | +3.041 |
| dense | assignment | +5.539 | +13.956 | +6.493 |
| dense | joint | +5.520 | +10.107 | +6.709 |
| skill_mask | timing | +1.691 | +0.949 | +4.650 |
| skill_mask | assignment | +2.609 | +3.002 | +14.491 |
| skill_mask | joint | +2.990 | +4.229 | +10.721 |

Bỏ từng replica hoặc profile: point estimate dense assignment luôn dương
(min6,03 khi bỏ replica;min6,33 khi bỏ profile), dense joint cũng luôn dương
(min6,14/min5,45). Đây là kiểm tra độ nhạy của estimate, không phải thêm các
significance tests. Dense timing có thể đổi dấu khi bỏ một profile (min-0,078).

![Các so sánh chính](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_ar_decision_diagnostic_v1_analysis_20261011T124812Z/counterfactual_results.png)

## Cơ hội phân công chủ yếu nằm ở đầu episode

Dense có74 assignment-eligible states:72 ởtick0,2 ởtick42. Skill_mask có28,
tất cả ởtick0. Vì vậy dữ liệu mới chủ yếu kiểm tra **phân công PM ban đầu khi
cả hai workers còn rảnh**; chưa xác nhận chất lượng tái phân công xuyên episode,
nhiều technician hơn, hoặc tình huống hàng đợi technician phức tạp.

Từ720 landmarks, timing/joint có130 dense và146 skill_mask states có lựa chọn;
những trạng thái còn lại thường không có PM nào trong root RL để bỏ/đổi. Chúng
vẫn ở panel nhưng có exposure0; không gán chúng là các quyết định PM tối ưu.
Diagnostic này không kiểm tra thêm PM vào máy mà RL chưa chọn.

| Điều kiện | Nhóm | Eligible | Chọn correction | Gain dương/âm sau correction | MC95 label dương | Correction có xác suất thấp hơn RL |
|---|---|---:|---:|---|---:|---|
| dense | timing | 130 | 55 | 34/21 | 7 | 55/55 |
| dense | assignment | 74 | 40 | 31/9 | 17 | 30/40 |
| dense | joint | 130 | 78 | 57/21 | 22 | 72/78 |
| skill_mask | timing | 146 | 66 | 47/19 | 15 | 66/66 |
| skill_mask | assignment | 28 | 11 | 9/2 | 5 | 10/11 |
| skill_mask | joint | 146 | 72 | 54/18 | 23 | 71/72 |

MC95 labels ở đây là lower bound normal-approximation gain-1,96*pairedSE>0,
không sửa multiplicity qua hàng trăm states. Chúng chỉ giúp mô tả ví dụ/headroom,
không phải các phát hiện có family-wise confidence hoặc nhãn training chắc chắn.
Không bỏ các correction âm vì selection sampling noise hoặc vì RL root vốn tốt.

## Hoãn và phối hợp: phân biệt từng trạng thái với luật chung

Các secondary intervals dưới đây là95% exploratory, không thuộc sáu tests chính:

| Điều kiện | So sánh | Gain | CI95% exploratory |
|---|---|---:|---|
| Mọi technician làm được mọi máy | Joint so với best-single chọn trên selection | +1.39 | [-0.22; 4.31] |
| Mọi technician làm được mọi máy | Hoãn toàn bộ PM so với RL | -7.02 | [-14.03; -1.06] |
| Có giới hạn kỹ năng | Joint so với best-single chọn trên selection | +2.43 | [-0.39; 6.93] |
| Có giới hạn kỹ năng | Hoãn toàn bộ PM so với RL | -8.37 | [-13.95; -3.46] |

Joint menu đã chọn một root cần đổi cả PM subset và technician ở19 dense và16
skill_mask states. Tuy nhiên joint so với best-single (best của timing union
assignment, cũng chọn bằngselection) có CI chứa0 ởcảhai điều kiện. Không dùng
joint-vs-RL gain dương để khẳng định “hai cơ chế phối hợp đã thắng sửa một cơ chế”.

Selective timing corrections gồm29 defer-all và26 partial-PM ởdense;27 và39
ởskill_mask. Có những trạng thái waiting lợi, nhưng mean timing correction vẫn
chưa rõ. Luật blanket defer-all gây hại trung bình khác với selective choice.
Các kết luận secondary không nâng thành xác nhận chung về mọi phân phối trạng thái.

Trong các trajectory gốc, policy không chủ động WAIT hoặc STOP sớm khi còn PM
extension khả thi:0/1546 dense và0/1528 skill_mask PM-feasible ticks. Đây là
**greedy evaluation**; STOP vẫn có xác suất dương, không nói training chưa từng
sample STOP. Trì hoãn ngầm qua chọn ưu tiên máy vẫn có thể xảy ra khi workers bận.
PM starts dưới ngưỡng rủi ro tức thời là84/873=9,62% dense và147/974=15,09%
skill_mask; không tự động coi các PM đó là sai.

## Policy có phân biệt technician và dùng resource context không?

Giữ đúng tập máy và chỉ xét hai feasible assignments, xác suất có điều kiện của
phân công RL có trung vị50,55% dense/50,18% skill_mask. Xác suất được cộng qua mọi
thứ tự decoder, rồi normalize trong hai assignments; không phải tần suất deploy
và không phải một ordered trace. Phân phối khá phẳng tự nó chưa chứng minh action
sai. Independent rollout bổ sung bằng chứng chất lượng của các correction.

Trong40 dense assignment corrections,30 có matching mass thấp hơn RL root;
15/17 positive MC95 examples cũng thuộc nhóm đó. Còn10 corrections có matching
mass cao hơn RL root nhưng greedy token decoding vẫn không chọn chúng. Vì vậy
điểm yếu có thể liên quan cả scoring lẫn cách decoder greedily chọn từng cặp;
không quy toàn bộ thành một lỗi duy nhất của score head. Skill_mask có10/11
corrections được policy gán matching mass thấp hơn root đã chọn.

![Xác suất hai phân công PM](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_ar_decision_diagnostic_v1_analysis_20261011T124812Z/assignment_probability.png)

Với cùng weights, chỉ freeze remaining-resource context sau cặp đầu và giữ live
mask, có86 dense và98 skill_mask prefixes còn choice. Greedy action đổi0/184.
Median total variation0,000138/0,000204;max0,000968/0,001457. Context vector thực
sự thay đổi (median L2~0,145/0,181), nhưng output đổi rất nhỏ. Đây là functional
sensitivity trên toàn landmark panel mới; không chứng minh nguyên nhân training.

N4/K2 chỉ còn một worker sau cặp đầu. Cập nhật context không trực tiếp quyết định
technician đầu tiên trong forward pass; first-prefix context luôn giống nhau cho
cùng weights. Kết quả không phủ nhận mọi kiến trúc context hoặc scalability ởK>2.
Chưa đo phân rã actor/critic gradients: không suy nguyên nhân từ loss scale.

## Ví dụ minh họa đã kiểm chứng

Lấy median-gain example trong17 dense assignment corrections có positive MC95
label (chọn để minh họa, không đại diện mọi trạng thái): profile17000007,
shock17100001,seed16100001,tick0. Máy0 và1 cùng tuổi4,ngưỡng failure5; cả hai
technicians rảnh, hai máy còn lại tuổi0/1.

| Root | Chi phí remaining TB64 evaluation futures |
|---|---:|
| RL: máy0→technician1,máy1→technician0 |1081,47|
| Vẫn PM ngay hai máy, đổi: máy0→technician0,máy1→technician1 |1054,56|
| Hoãn cả hai PM |1096,48|

Đổi technician giảm26,91 (pairedMCSE5,77), trong khi hoãn làm cost tăng15,02.
Ởmáy0,tech0 làm2 ticks so vớitech1 làm4 ticks, cùngrestoration0,5. Ởmáy1,
tech1 làm4 ticks vàrestoration1,0,tech0 làm3 ticks vàrestoration0,5. Chọn nhanh
nhất riêng lẻ chưa mô tả toàn bộ tradeoff. Matching mass của phân công RL là
0,20060, phương án đổi là0,18319: policy ưu tiên cao hơn root có cost cao hơn
trong ví dụ này. Đây là kết quả simulated continuation, không phải guarantee
cho mỗi future hoặc technician ởthực tế.

## Sai khác tái lập Mac/lab: nhỏ, được giữ nguyên và báo rõ

**Forecast replay là PARTIALLY_REPRODUCIBLE, không gọi exact toàn run.** Một
future trong158784 futures khác cost; tất cả evaluation futures khớp. Lệch ở
state640:dense,profile17000008,shock17100000,seed16100002,tick0,selection root6,
future29. Labcost1294,Mac1264; sai khácmean root selection là30/32=0,9375.
Root này không được chọn ởbất kỳ contrast/best-single; mọi selection index khớp,
primary gains và sáu CI không đổi. Tapes, root first-step cost và mọi source hash
khớp; không thay raw array bằng replay values khi dựng thống kê.

Debug scalar trên Mac cũng ra1264. Ởtick35 có hai CM tokens5/7 bằng nhau trên
Mac; chọn5. Force token7 ởđúng bước đó tái tạo labcost1294 và cảcutoff48/terminal60
states. Điều này hỗ trợ giải thích do argmax gần hòa giữa nền tảng, nhưng full
lab forecast không lưu action trace nên đây là **inference**, không quan sát trực
tiếp lab token. Không sửa policy/code hoặc rerun training trong lượt phân tích.
Các experiment sau nên cố định quy tắc near-tie nếu cần tái lập across platforms.

Lần checker đầu dừng vì yêu cầu exact quá nghiêm cho nhánh này; thư mục derived
20261011T124352Z được đánh dấuINCOMPLETE_EXACT_REPLAY_CHECK. Báo cáo hiện tại
20261011T124812Z đã audit đủ720 states và giữ đầy đủ sai khác, không loại nhánh
hay tạo số liệu tốt hơn. Original lab run vẫn COMPLETED/PASS.

## Hệ quả cho bước phát triển thuật toán

Bằng chứng mới xác nhận có headroom phân công PM mà greedy RL hiện tại bỏ lỡ
trong dense, chủ yếu ởđầu episode. Ưu tiên một ablation RL cải thiện cách chấm
điểm/giải mã phân công cặp máy–technician, giữ timing/STOP và ngân sách đối chứng
để đo riêng hiệu quả. Chưa chốt một cấu trúc mới là lời giải, chưa chứng minh
RL đã học tốt timing và assignment đồng thời, chưa chứng minh novelty.

Không ưu tiên luật “chờ nhiều hơn” chỉ vì WAIT bằng0. Nếu sửa timing, mục tiêu
là phân biệt waiting có lợi trên từng state, chứ không tăng WAIT frequency.
Không thêm đồng thời STOP gate, đổi scorer, normalization và actor/critic
representation rồi quy toàn bộ gain cho một thành phần. Không lấy nhãn simulator
của diagnostic này làm sealed test cho experiment tiếp theo.

## Statistical/methodological fallacy scan:11/11 checked

| Loại | Đánh giá |
|---|---|
| Simpson |Tách dense/skill, replicas vàlandmarks. Dense assignment gain tập trungtick0; không che biến thiênprofile hoặc dense timing seed0~0.|
| Ecological |Mean correction gain không suy thành mọi PM/technician action sai hoặc episode policy sẽ cải thiện bằng tổnggains.|
| Berkson |Fresh profiles chỉ lọc structural signatures, không bằngoutcomes. Eligible state conditioning được báocoverage.|
| Collider |Choice availability là kết quảtrajectorycủamodel; không so policy populations bằngconditionalmean hoặc coi assignment coveragecausal.|
| Base rate |Có denominator74/360,28/360,130/360,146/360; forced STOP vàPM-free không tínhchoice.|
| Regression to mean |Selection32 vàevaluation64 độc lập, không chọn bằngevaluation; mọi gain âm được giữ.|
| Survivorship |Đủ3 models,144 trajectories,720states; không drop replica/profile/MCfuture lệch hoặcnegative correction.|
| Look-elsewhere |Đủ6 primarycontrasts,Bonferroni99,17%; secondary95%,MCstate labels vàví dụ được đánh dấuexploratory.|
| Forking paths |Fixed protocol,source,seeds,landmarks,budget verified; post-hocjackknife vàtie debugdisclosed; không sửa raw/data để quaexactcheck.|
| Correlation/causation |Root interventions có diễn giải within simulator; probabilities/context sensitivity không xác minh nguyên nhân optimization hoặc hiệu quả thuật toán mới.|
| Reverse causality |Root decision đi trước simulated cost, nhưng visited state/availability phụ thuộcpolicyhistory; không gán chiều causal cho scoreflatness.|

## Artifacts và giới hạn

- audit.json,analysis_metadata.json: integrity,replay,version vàscope.
- primary_results.csv,secondary_results.csv,profile_cells_verified.csv,profile_gains.csv,leave_one_out.csv.
- verified_root_values.csv,verified_prefix_scores.csv,trajectory_behavior.csv.
- timing_corrections.csv,assignment_corrections.csv,joint_corrections.csv,mechanism_summary.csv.
- replay_field_differences.csv,replay_future_differences.csv,replay_selection_differences.csv.
- affected_future_scalar_check.json,affected_future_tie_debug.json.
- counterfactual_results.png/pdf,assignment_probability.png/pdf.

Chưa rerun stochastic training, không truy cập lab, không sửa raw artifacts.
Nội suy từ12profiles/3replicas,conditionalfixedMCtapes vàchủ yếuinitialassignment
là giới hạn chính; không dùng diagnostic này thay cho đánh giá thuật toán mới.
