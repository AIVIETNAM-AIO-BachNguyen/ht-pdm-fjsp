# Phân tích diagnostic thời điểm phòng ngừa và WAIT

## Material Passport

Origin Skill: academic-research-suite / experiment-agent; Origin Mode: validate
inline; Date: 2026-10-11; Version: maintenance_timing_diagnostic_v1;
Verification Status: ANALYZED, numerically VERIFIED for the complete frozen
evaluation/tapes and four primary CIs. Không train lại, không lấy seed mới,
không coi việc replay cùng tape là một replication độc lập.

## Kết luận chính

Policy ưu tiên phòng ngừa cho máy có nguy cơ hỏng ở bước kế tiếp, nhưng rất ít
chủ động WAIT khi vẫn có thể khởi động một công việc. Kết quả chưa chứng minh
policy phòng ngừa sai trên toàn bộ trạng thái: bỏ PM hiện tại có chi phí trung
bình cao hơn giữ PM, và cả bốn khoảng tin cậy chính đều chứa 0. Dữ liệu chỉ ra
các trường hợp máy chưa có nguy cơ tức thời là nhóm đáng kiểm tra kỹ hơn.

Run `maintenance_timing_diagnostic_v1_full_cpu_20261011T072307Z`, commit
`e02b504ae6f0755c6f5d150e41f3ada9ca2ee6f4`, COMPLETED / audit PASS. Thời gian
lab **74.90 giây**, khoảng 1 phút 15 giây; artifacts **31.98MB**. Đủ 10 model
đã đóng băng, 640 quỹ đạo, 2.560 trạng thái, 12.007 root actions hợp lệ và
9.221.376 bước forecast thực tế. Không có model bị chọn lại.

## Policy chọn phòng ngừa theo nguy cơ như thế nào?

Đếm từng cơ hội: máy khỏe, chưa được bảo trì, còn ít nhất một kỹ thuật viên
rảnh có kỹ năng tương thích. Khi máy thỏa điều kiện này, xác suất được RL chọn
PM trong các trạng thái mẫu là:

| condition | at_risk | eligible_machine_opportunities | selected_pm_pairs | selection_fraction |
| --- | --- | --- | --- | --- |
| dense | False | 1186 | 62 | 0.052 |
| dense | True | 677 | 322 | 0.476 |
| skill_mask | False | 1152 | 100 | 0.087 |
| skill_mask | True | 520 | 308 | 0.592 |

Tỷ lệ chọn máy có nguy cơ ở bước kế tiếp là **47,6% / 59,2%**, so với
**5,2% / 8,7%** khi chưa có nguy cơ, tương ứng dense / skill_mask. Đây là
mô tả hành vi ưu tiên nguy cơ; các cơ hội có thể còn khác nhau về duration,
restoration, máy cạnh tranh và kỹ thuật viên bận. Không coi các tỷ lệ này là
thí nghiệm can thiệp riêng trên tuổi hoặc bằng chứng nguồn gốc học của mạng.

Trong 792 PM pairs có 630 pair trên máy nguy cơ cao và 162 pair trên máy chưa
có nguy cơ ở bước kế tiếp. Hazard=0 chỉ nói bước kế tiếp; máy vẫn có thể hỏng
trong các bước sau.

## WAIT chủ động khác WAIT bắt buộc

| condition | choice_states | RL_WAIT | RL_WAIT_pct | rule_WAIT | rule_WAIT_pct |
| --- | --- | --- | --- | --- | --- |
| dense | 742 | 8 | 1.078 | 273 | 36.792 |
| skill_mask | 753 | 12 | 1.594 | 94 | 12.483 |

Gộp hai điều kiện, RL chủ động WAIT **20/1.495 = 1,34%**, rule **367/1.495 =
24,55%**. Các mẫu này gồm cả trạng thái do RL và rule tạo ra. Trong 1.085
quyết định WAIT của RL, **1.065 = 98,16%** là trạng thái không còn root
matching nào ngoài WAIT. Vì vậy thấy WAIT trong log chưa đủ để nói policy đã
chọn chờ nhằm giữ tài nguyên. Tỷ lệ WAIT chủ động trên riêng quỹ đạo RL:
dense 6/372=1,61%, skill_mask 9/423=2,13%; kết luận hành vi tương tự.

WAIT ít không tự động nghĩa là WAIT chưa tốt: số lần đúng để WAIT phụ thuộc
nguy cơ, độ dài công việc và trạng thái tài nguyên. Cần đi cùng giá trị fork.

![Tần suất WAIT và chọn PM theo nguy cơ](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_timing_diagnostic_v1_analysis_20261011T073603Z/timing_behavior.png>)

## Khi bỏ PM hiện tại, kết quả có tốt hơn không?

H1 xóa riêng các pair PM khỏi root của RL, giữ nguyên mọi pair CM và kỹ thuật
viên của chúng. Từ bước sau, cả hai nhánh dùng deadline_matching. Đây là
đo hiệu ứng quyết định hiện tại; không phải cấm PM mãi hoặc hẹn chắc PM vào
bước kế tiếp. H2 thay toàn bộ root bằng reference đã chọn trên development:
dense adaptive_isolated_matching; skill_mask deadline_matching.

Trong bảng, delta = C(root RL) − C(root so sánh), **tổng chi phí của 24 bước**.
Số âm lợi cho root RL; số dương lợi cho bỏ PM hiện tại / root của rule.
Đây là đơn vị chi phí simulator, không phải phần trăm hay tiền thực tế.

| contrast | exposures | delta | CI_low | CI_high | verdict |
| --- | --- | --- | --- | --- | --- |
| dense:pm_deferral | 347 | -2.182 | -6.443 | 2.201 | UNRESOLVED |
| dense:rule_root | 742 | 2.549 | -0.530 | 6.489 | UNRESOLVED |
| skill_mask:pm_deferral | 381 | -2.712 | -8.684 | 0.025 | UNRESOLVED |
| skill_mask:rule_root | 753 | 1.091 | -0.064 | 2.770 | UNRESOLVED |

H1: giữ PM lợi trung bình 2,18 / 2,71 đơn vị, nhưng độ bất định còn lớn.
H2: root của rule lợi trung bình 2,55 / 1,09 đơn vị, cũng chưa tách được khỏi
0 theo ngưỡng đã khóa. **Cả bốn UNRESOLVED**. Không đổi sang CI95% để gọi
skill_mask H1 “có ý nghĩa”; CI98,75% của H1 skill_mask vẫn có đầu trên
**+0,0252** đơn vị tổng chi phí. Chưa thể kết luận tương đương, thắng rule,
hoặc policy sai vì PM quá nhiều.

CI là crossed bootstrap 10 model x8 physical cohorts, 20.000 draws, giữ
shock/origin/landmark paired. Conditional ratio là tổng delta/tổng exposure,
không lấy trung bình các cell ratios. Không có bootstrap draw thiếu exposure.
Coverage98,75% là xấp xỉ Bonferroni bốn exploratory contrasts, conditional
trên tape MC32 và shock panel cố định; không bao toàn bộ bất định MC/simulator.

![Bốn contrast và CI](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_timing_diagnostic_v1_analysis_20261011T073603Z/primary_contrasts.png>)

## Cơ chế: PM đúng nguy cơ khác PM trên máy còn trẻ

Xóa một pair PM, giữ nguyên các pair còn lại, cho mô tả sau. Mỗi mean là
chi phí root hiện tại trừ chi phí root bỏ pair đó, trên cùng 32 forecast tapes.
Các pair trên một trạng thái có thể tương tác; không cộng các marginal này
thành hiệu ứng xóa toàn bộ PM. Các nhóm được xem sau kết quả, không thêm
p-value hoặc CI confirmatory cho chúng.

| condition | at_risk | PM_pairs | mean_C_PM_minus_C_drop |
| --- | --- | --- | --- |
| dense | False | 62 | 6.283 |
| dense | True | 322 | -4.683 |
| skill_mask | False | 100 | 2.532 |
| skill_mask | True | 308 | -4.363 |

Trên máy chưa có nguy cơ bước kế tiếp, bỏ riêng pair PM tiết kiệm trung bình
6,28 đơn vị ở dense / 2,53 ở skill_mask. Trên máy đã có nguy cơ, giữ PM tiết
kiệm trung bình 4,68 / 4,36. Đây là dấu hiệu cần phân biệt ngữ cảnh khi quyết
định “làm ngay” và “giữ người rảnh”, chứ không áp dụng cùng cách cho mọi máy.

Nhóm xóa toàn bộ PM khi **mọi máy được chọn PM đều chưa có nguy cơ** khác
nhau giữa hai origins:

| condition | origin | n | mean |
| --- | --- | --- | --- |
| dense | best_rule | 10 | 16.753 |
| dense | joint_ppo | 36 | 9.128 |
| skill_mask | best_rule | 19 | -3.822 |
| skill_mask | joint_ppo | 69 | 2.399 |

Đặc biệt skill_mask / rule-origin có mean âm (giữ PM lợi), trong khi
RL-origin mean dương. Vì vậy không tuyên bố “PM trên máy trẻ luôn sai”.
Hai origins giữ đầy đủ trong các CSV, không lựa một origin thuận kết luận.
Kết quả H1 aggregate theo origin đều âm ở cả hai điều kiện:

| condition | origin | exposures | mean |
| --- | --- | --- | --- |
| dense | best_rule | 162 | -3.354 |
| dense | joint_ppo | 185 | -1.156 |
| skill_mask | best_rule | 164 | -4.506 |
| skill_mask | joint_ppo | 217 | -1.356 |

Chi phí thành phần, tính trung bình đúng trên các exposure của từng contrast:

| condition | contrast | maintenance_cost | unavailability_cost | failure_cost | waiting_cost |
| --- | --- | --- | --- | --- | --- |
| dense | pm | -0.295 | 4.027 | -5.879 | -0.035 |
| dense | rule | 0.099 | 0.824 | 2.011 | -0.385 |
| skill_mask | pm | -0.585 | 5.197 | -6.801 | -0.524 |
| skill_mask | rule | -0.051 | 1.071 | -0.063 | 0.134 |

H1: PM tăng thời gian máy không khả dụng, đồng thời giảm failure cost;
phần giảm failure cost thắng trung bình trên panel này. H2 dense: chênh
root RL/rule đến nhiều từ failure cost; H2 skill_mask: chủ yếu từ
unavailability. H2 còn thay machine/worker allocation nên không quy toàn bộ
chênh này cho riêng timing hoặc riêng technician choice.

Ví dụ minh họa hậu kiểm, chọn pair dense gần mean của nhóm tương ứng:
cohort13100007/shock13200001/seed12100005/RL-origin/tick6. Máy1 tuổi2,
threshold5, một kỹ thuật viên rảnh; người còn lại đang sửa máy0 và còn2 bước.
RL bắt đầu PM máy1 kéo dài4 bước; rule WAIT. Forecast mean PM359,16,
WAIT351,84: chờ tiết kiệm7,31. MCSE của paired delta **5,21** nên một ca
này không chứng minh lợi ích chắc chắn; chỉ minh họa việc đặt người rảnh vào
một công việc PM dài khi máy chưa có nguy cơ tức thời.

## Phạm vi suy luận và bước tiếp theo

Đây là giá trị một root dưới **common rule continuation**, không phải Q khi
chính policy RL chạy tiếp. Horizon24 cố định còn có thể kéo dài vượt cutoff
của episode gốc, nhất là tick42 (policy nhận remaining learned horizon6;
source base48+tail12 kết thúc ở60, forecast này kéo tới66). Ngữ cảnh finite
horizon cần được giữ khi diễn giải, không coi diagnostic là nguyên nhân đã
chứng minh của full-run performance gap trước đó.

Sample minimum qua nhiều roots có selection optimism; không dùng nó làm
oracle regret hoặc target “action đúng” không điều kiện. Các dấu delta theo
state/pair chỉ là dấu mean MC32, không phải xác suất một quyết định chắc đúng.

Hướng RL vẫn có tín hiệu hành vi hữu ích: ưu tiên máy nguy cơ cao. Câu hỏi
cần đào sâu là giá trị của giữ tài nguyên rảnh so với khởi động PM/CM và
chọn máy/người, trên cùng trạng thái. Bước kiểm tra kế tiếp hợp lý là đối
chiếu root WAIT/PM với **RL chạy tiếp**, giữ horizon/cutoff/tail như training,
để xác định thiếu WAIT có thật sự làm policy RL tốn chi phí hơn. Sau đó mới
chọn thay đổi training/critic có kiểm chứng. Đây là đề xuất, chưa chạy một
experiment mới hoặc sửa model theo kết quả vừa mở.

## Kiểm tra tái lập và 11 fallacies

- 85 artifact hashes/byte sizes, toàn bộ snapshot code/config/lockfile so
  commit, source full manifest169 artifacts và10 checkpoints: PASS.
- 51.200 source train configurations được tái dựng để kiểm tra fresh physical
  split;16 profiles mới disjoint source và smoke. Seed roles: PASS.
- Toàn bộ2.560 states /12.007 roots /9.221.376 forecast transition samples
  replay đúng raw NPZ, CSV scenario costs, mean, MCSE, immediate cost, và
  bốn components. Bằng simulator scalar gốc: kiểm tra scenario đầu cho
  **mọi root của mọi state**, tổng288.168 interval transitions. Không nói
  rằng cả32 scenarios đều được kiểm tra scalar.
- Replay640 quỹ đạo /30.720 physical intervals, mọi frozen-policy/rule action,
  trạng thái trước/sau, metric và landmark occupancy: PASS.
- Toàn bộ792 PM deletion marginals, keep-CM intervention và frozen weights:
  PASS. Bốn CIs tái tính bằng implementation bootstrap độc lập: PASS.
- Không sửa file dữ liệu nguồn đã hash; phân tích lưu trong thư mục mới.

Fallacy scan **11/11**:

| Loại | Đánh giá trong phân tích này |
|---|---|
| Simpson | Origin/condition/cohort được phân tầng; young-PM group skill_mask đảo dấu giữa origins, được nêu rõ. |
| Ecological | Không suy mọi state đúng/sai từ mean; model/profile mới là axes của CI. |
| Berkson/selection | Landmark cố định, giữ forced-WAIT; PM eligibility là estimand conditional, không generalize ra mọi state. |
| Collider | PM/risk subgroups là mô tả; không điều chỉnh chúng như biến kiểm soát để chứng minh tác động training. |
| Base rate | Báo denominator1.495 choice states và forced-WAIT; PM selection dùng eligible machine opportunities. |
| Regression to mean | Giữ cả10 dev-selected models, không chọn best seed mới; không coi replay là learning improvement. |
| Survivorship | Grid đầy đủ, không drop busy/agreement states hoặc state đắt. |
| Look-elsewhere | Báo cả4 primary intervals; posthoc bins/examples không được gán ý nghĩa thống kê. |
| Forking paths | Protocol/seed/budget giữ nguyên; descriptive follow-up tách khỏi primary. |
| Correlation/causation | Root removal là controlled simulator intervention dưới rule; không quy cho hiệu quả toàn RL hoặc real-world causality. |
| Reverse causality | Action trước future shocks; mức risk/selection association không được coi là riêng hiệu ứng tuổi. |

Input artifacts: [manifest](</Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_timing_diagnostic_v1_full_cpu_20261011T072307Z/manifest.json>), [summary](</Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_timing_diagnostic_v1_full_cpu_20261011T072307Z/summary.json>),
[locked plan](</Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_timing_diagnostic_v1_full_cpu_20261011T072307Z/plan.md>). Evidence: [audit](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_timing_diagnostic_v1_analysis_20261011T073603Z/audit.json>),
[state diagnostics](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_timing_diagnostic_v1_analysis_20261011T073603Z/state_diagnostics.csv>), [primary recomputed](</Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_timing_diagnostic_v1_analysis_20261011T073603Z/primary_recomputed.json>).
