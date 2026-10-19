## Material Passport

- Origin Skill: ars-codex:academic-research-suite / experiment-agent.
- Origin Mode: validate, inline; local transferred data only.
- Origin Date: 2026-10-11.
- Version Label: maintenance_assignment_scoring_v1_analysis.
- Verification Status: SOURCE, STATISTICS AND EVALUATION REPLAY AUDITED; training not rerun.
- Overall Confidence: CAUTION — exploratory pilot with only three training replicas.

## Kết luận

Thí nghiệm chưa ủng hộ việc chốt phiên bản thuật toán mới này. Bổ sung tương tác
cặp máy–tài nguyên chưa tạo cải thiện chi phí chắc chắn; đổi từ chọn token tham
lam sang tối đa xác suất phân công cũng chưa cho lợi ích rõ. Mạng mới có biểu
hiện chờ/STOP nhiều hơn, nhưng chưa chứng minh chọn đúng timing và đúng technician
cùng lúc. Đây là kết quả về hai thay đổi cụ thể trong pilot, không phủ định hướng RL.

## Run và tính hợp lệ

Run `maintenance_assignment_scoring_v1_full_cpu_20261011T131859Z`, commit `de47c39379f1ae1e8c4d0e64745b90cb6322c550`, COMPLETED/PASS, code sạch. Lab chạy 13.34 phút. 957 artifact, 66.68 MB; toàn bộ hash/size và 177 source/config trùng commit. Sáu model, mỗi model122880 learned steps,19650 tham số; cùng trọng số khởi tạo, cấu hình và các stream training trong mỗi cặp seed. Không teacher labels; complete return H48+tail12. Checkpoint được chọn trên development trước test; cả ba seed được giữ. 2496 test episodes,360 trạng thái chung/720 scorer-state diagnostics. Raw artifacts giữ nguyên.

Replay2496 episodes/149760 physical decisions và18720 logged decisions: 0 sai khác field episode, 0 sai khác action/state/cost logs; bỏ qua latency phần cứng. Replay155328 root-futures/7478784 transitions: 0 sai khác field, 0 khác chọn root; trạng thái replay EXACT. Không chạy lại training; kiểm chứng này không phải bằng chứng training lặp lại từng bit.

## Chi phí toàn episode

Chi phí thấp hơn tốt hơn. “Đủ kỹ năng” nghĩa cả hai technician có thể làm mọi máy; “giới hạn kỹ năng” có cạnh máy–technician bị loại.

| Controller | Đủ kỹ năng | Giới hạn kỹ năng |
|---|---:|---:|
| Rule tốt nhất trên dev | 1128.00 | 1187.24 |
| RL đối chứng, greedy | 1146.13 | 1184.25 |
| RL đối chứng, chọn mass | 1144.55 | 1183.08 |
| RL tương tác, greedy | 1150.76 | 1176.81 |
| RL tương tác, chọn mass | 1146.81 | 1177.82 |

## Bốn so sánh chính đã khóa trước

Gain dương là giảm chi phí. CI98.75% crossed bootstrap trên replica/profile,20000draws; Bonferroni xấp xỉ cho bốn so sánh. Ngưỡng cải thiện có ý nghĩa thực tế1%.

| Điều kiện / decoder | Giảm chi phí | CI98.75% | Kết quả |
|---|---:|---|---|
| Đủ kỹ năng / greedy | -0.40% | [-1.42; +0.67]% | Dưới ngưỡng1% |
| Đủ kỹ năng / mass | -0.20% | [-1.29; +0.67]% | Dưới ngưỡng1% |
| Giới hạn kỹ năng / greedy | +0.63% | [-0.21; +1.81]% | Chưa rõ |
| Giới hạn kỹ năng / mass | +0.44% | [-0.29; +1.42]% | Chưa rõ |

Không có CI nào đạt cận dưới>1%. Hai điều kiện đủ kỹ năng có cận trên<1%,
nhưng CI vẫn chứa0: không được diễn giải thành chứng minh thuật toán mới làm
xấu đi. Với skill constraints, cả ba seed có point gain dương dưới cả hai
 decoder, nhưng ba replicas không đủ để gọi đây là cải thiện chắc chắn.

Secondary decoder gains: đối chứng +0.138% dense/+0.099% skill; tương tác
+0.343% dense/−0.086% skill, toàn bộ CI95% chứa0. So với rule, mạng tương tác
+0.879% skill nhưng−2.018% dense (greedy); các CI95% cũng chứa0. Không chốt vượt
rule. So sánh skill với rule có base cost933.81>928.89 và tail242.99<258.35;
point advantage toàn episode nhờ phần tail, không đồng nghĩa thắng ở mọi cửa sổ.
Không so mean lần này với pilot cũ như một paired gain vì các panel đã đổi.

## Chọn technician: vẫn còn yếu

Decoder mass chỉ thay worker của PM ở trạng thái hiện tại, giữ EXACT machine
set và CM edges. Nó cộng xác suất qua tất cả thứ tự token hợp lệ. Toàn episode
sau đó có thể đổi trạng thái/timing; giữ máy tại mỗi intervention không khóa
lịch PM cả episode. Highest policy probability không phải ước lượng trực tiếp
chi phí của phân công.

Trên các trạng thái có alternate assignment, conditional mass của lựa chọn
greedy gần50%: đối chứng dense50.16%/skill50.14%; tương tác dense50.35%/
skill50.57%. Việc thêm tương tác chưa tạo phân biệt assignment mạnh trong panel
này. Những số này mô tả scorer output, không phải độ chính xác chọn technician.

Diagnostic dưới đây chọn một chỉnh sửa bằng32 selection futures rồi đo bằng
64 futures độc lập. Gain là chi phí còn lại của root RL trừ root đã chỉnh sửa,
RL tự chạy tiếp đến48 rồi tail12; không phải % giảm chi phí của một thuật toán.

| Scorer / điều kiện | Gap phân công | CI95% | Exposure /180 |
|---|---:|---|---:|
| additive_ppo / dense | +1.48 | [-3.05; +11.37] | 40 |
| additive_ppo / skill_mask | +17.93 | không đủ exposure | 10 |
| interaction_ppo / dense | +3.89 | [-1.11; +13.52] | 40 |
| interaction_ppo / skill_mask | +10.78 | [+0.00; +20.20] | 17 |

Dense assignment gap chưa tái hiện tín hiệu dương rõ trong panel mới. Skill
đối chứng chỉ10 exposures,6.53% bootstrap draws không có exposure nên không đủ CI đáng tin; tương tác17 với CI chạm0.
Không được kết luận gap giảm chỉ vì17.93 xuống10.78: denominators/machine sets
và continuation khác nhau. Hầu hết exposure assignment ở tick0 (dense36/40;
skill10/10 đối chứng,16/17 tương tác), nên dữ liệu chủ yếu nói về phân công ban đầu.

Trong17 skill exposures của mạng mới, simulator chọn đổi worker11 lần:10 gain
dương,8 có paired MC95% lower bound dương,6 trong8 có mass THẤP hơn phân công RL.
Đây là nhãn exploratory từng state, không hiệu chỉnh nhiều so sánh; gợi ý scorer
vẫn ưu tiên sai ở một số trường hợp, không chứng minh cơ chế gradient gây sai.

## Timing được giữ riêng

| Scorer / điều kiện | Chờ khi PM khả thi | Tỷ lệ | STOP còn mở rộng PM |
|---|---:|---:|---:|
| additive_ppo / dense | 8/6317 | 0.13% | 44 |
| additive_ppo / skill_mask | 7/6200 | 0.11% | 84 |
| interaction_ppo / dense | 52/6389 | 0.81% | 150 |
| interaction_ppo / skill_mask | 109/6300 | 1.73% | 231 |

Mạng mới chờ nhiều hơn; PM ở tuổi dưới ngưỡng nguy cơ giảm từ14.58% xuống12.93%
dense và21.24% xuống17.94% skill. Những PM đó không mặc nhiên sai và non-wait
không loại trừ trì hoãn máy do nguồn lực. Không dùng median STOP trên cả forced
STOP states để suy ra chủ động chờ.

Gap timing của mạng mới: dense+2.08 CI95%[−2.17;7.09] trên63/180 states;
skill+2.13[0.45;3.68] trên71/180, cả ba replicas point gain dương. Skill là tín
hiệu exploratory rằng vẫn có chỉnh sửa timing có lợi; nó không phải một trong
bốn primary contrasts, không chứng minh “timing đã được giải quyết”.
Đối chứng: dense+1.72[−1.88;5.30], skill+2.81[−0.03;6.60]. Blanket bỏ toàn bộ
PM đang chọn có mean gain âm: đối chứng−3.54/−2.86, tương tác−4.95/−3.65
(dense/skill). Không suy ra nên chờ tất cả PM; cần chọn theo trạng thái.

## Hàm ý bước tiếp theo

Chưa nên chốt bản interaction+mass thành thuật toán hoàn chỉnh. Thử nghiệm
hợp lý tiếp theo là kiểm tra cách học giá trị tương đối giữa các phân công
trên cùng trạng thái, thay vì chỉ sửa feature scorer/decoder. Có thể thiết kế
assignment value/ranking head dự đoán chi phí tiếp diễn của các phân công giữ
máy cố định, dùng counterfactual training futures mới và loss so sánh cặp, rồi
kiểm tra khả năng xếp hạng trên fresh held-out states. Đây là giả thuyết kế tiếp,
chưa được implement/đảm bảo hiệu quả. Không dùng panel diagnostic/test đã mở
làm training labels; timing tiếp tục là kiểm tra riêng.

## Kiểm tra11/11 ngụy biện thống kê

1. Simpson: báo riêng dense/skill và từng decoder; hướng point effect khác nhau giữa điều kiện.
2. Ecological: không biến mean episode thành mọi state đều đúng.
3. Berkson: states thuộc baseline-visited panel, không đại diện mọi trạng thái.
4. Collider: eligibility phụ thuộc action; không so gap với denominator khác như cùng quần thể.
5. Base rate: luôn báo exposure và mẫu số PM-feasible; forced STOP tách khỏi lựa chọn chủ động.
6. Regression to mean: checkpoint chọn dev trước test; root chọn selection trước independent evaluation.
7. Survivorship: đủ sáu models/ba replicas và mọi episode; giữ cả gain âm/zero.
8. Look-elsewhere: bốn primary có hiệu chỉnh; secondary/12 diagnostics/per-stateMC là exploratory.
9. Forking paths: protocol khóa; không chọn best seed; leave-one-replica-out được lưu.
10. Correlation/causation: interventions kiểm tra phiên bản cụ thể trong simulator; output mass không chứng minh gradient mechanism.
11. Reverse causality: training trước sealed test; không dùng test/diagnostic labels để train.

Giới hạn: chỉ3replicas,24 test profiles và6 diagnostic profiles; crossed CI
conditional trên shock/future tapes, không tích hợp toàn bộ Monte Carlo error.
Per-state pairedSE lưu riêng. Kết quả không xác nhận novelty, scale transfer,
hiệu quả ngoài simulator hay optimum timing/assignment.


![Bốn hiệu ứng chi phí](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_assignment_scoring_v1_analysis_20261011T133827Z/primary_cost_effects.png)

![Timing và assignment tách riêng](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_assignment_scoring_v1_analysis_20261011T133827Z/separate_diagnostics.png)
