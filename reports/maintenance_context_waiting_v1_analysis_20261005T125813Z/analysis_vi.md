# Phân tích maintenance context × waiting v1

## Material Passport

- Mode: ARS experiment validate.
- Verification Status: **ANALYZED**. Có audit/replay và tính lại thống kê; chưa chạy lại training độc lập.
- Protocol: `maintenance_context_waiting_v1`, commit `e2ec53e3239881aaf2d7db57aa7b72c64342165f`.
- Raw run: `/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/maintenance_context_waiting_v1_full_cpu_20261005T064927Z`.
- Giả thuyết và ngưỡng lấy từ plan đã khóa trong `source_snapshot/maintenance_context_waiting_v1_plan.md`.
- Bản phân tích không sửa raw artifacts, code experiment, seed panel, checkpoint hay tiêu chí pass/fail.

## Kết quả chính

**H_wait đạt toàn bộ tiêu chí đã khóa; H_context không đạt.** Trong benchmark và
ngân sách này, việc định lượng chi phí chờ sửa quá ngưỡng tạo tác động lớn lên
chất lượng phục vụ, trong khi thêm context riêng vào allocation head chưa tạo
lợi ích cost rõ. Đây là bằng chứng về thiết kế mục tiêu phục vụ của bài toán;
chưa chứng minh cần MARL hay một kiến trúc phức tạp hơn.

Bảng cost là trung bình bằng trọng số của bốn primary families N3/N4 nominal và
pressure. Cột overdue là trung bình bằng trọng số N3/N4 pressure, tính trên
**số tick chờ vượt 4 / máy / episode**, không phải toàn bộ thời gian chờ.

| Nhánh | Base economic cost | Overdue tick/máy/episode | p95 max wait N3/N4 pressure |
|---|---:|---:|---:|
| Local, cost only | 146.2330 | 0.49317 | 8 / 9 |
| Context, cost only | 145.8055 | 0.57583 | 9 / 9 |
| Local, thêm phí chờ | 147.7245 | 0.00375 | 2 / 3 |
| Context, thêm phí chờ | 147.2025 | 0.00258 | 2 / 4 |
| Risk/skill rule | 148.1050 | 0.00000 | 1 / 2 |

Rule chỉ được chạy một lần trên cùng panel; không nhân bản thành mười lần
training. Khoảng tin cậy dùng training seed làm đơn vị độc lập, có điều kiện
trên evaluation panel cố định. Pooled p95 dùng toàn bộ machine-episode records;
không dùng các records này như những lần training độc lập.

## Hai kiểm định xác nhận

**H_context: context_cost so với local_cost.** Giảm 0.4275 cost, tương đương
**0.2923%**, thắng **6/10 seed**. CI paired 97.5%, đã Bonferroni cho hai giả
thuyết, là **[-1.7003, +0.8453]**; p hai phía 0.39066, p điều chỉnh 0.78132.
Nominal mean tốt hơn 0.3018%, nhưng không đạt mức giảm 5%, số seed thắng >=8
hay CI upper <0. Không thể xem context technician là nút thắt cost chính được
xác nhận bởi run này. Không suy rộng thành kết luận mọi representation đều vô ích.

**H_wait: context_wait so với context_cost.** Overdue giảm
**0.575833 → 0.002583 tick/máy/episode**, tức **99.5514%**, thắng **10/10 seed**.
CI 97.5% cho chênh lệch là **[-0.95599, -0.19051]**; p hai phía 0.003011,
p Bonferroni 0.006022. p95 wait bằng 2 ở N3 pressure và 4 ở N4 pressure, đều
đạt B=4. Primary base cost tăng **0.9581%**; nominal base cost tăng **0.0393%**,
đều trong guard <=10%. Đạt toàn bộ sáu tiêu chí khóa trước.

Chi phí base tăng trung bình 1.397; CI 95% mô tả **[-0.3540, +3.1480]**.
Vì vậy +0.96% là point estimate, không phải một mức tăng được đảm bảo trong
mọi tình huống. Hai phía cost và service được báo cáo riêng, không lấy reward
có phí chờ của một nhánh so với reward không có phí chờ của nhánh khác.

Sensitivity mô tả: kiểm định sign-flip trên 10 paired differences cho H_wait
p=0.001953 hai phía, H_context p=0.376953. Không thay thế gate paired-t đã khóa.
Normality của paired differences chưa được xác lập với chỉ mười training seed;
CI không bao gồm uncertainty do lấy mẫu panel cấu hình/evaluation mới.

## Cơ chế và tradeoff

N4 pressure minh họa rõ thay đổi priority giữa bảo trì phòng ngừa và sửa hỏng:

| Metric/episode | Context cost only | Context + phí chờ |
|---|---:|---:|
| Base cost | 235.964 | 240.544 |
| PM starts | 5.574 | 4.970 |
| CM starts | 1.786 | 2.968 |
| Failure events | 3.206 | 3.998 |
| Unavailable machine ticks | 29.788 | 28.278 |
| Failed waiting ticks, tất cả | 8.456 | 5.134 |
| Overdue ticks/máy | 0.7030 | 0.0045 |
| p95 maximum pending wait | 9 | 4 |
| Machine records có max wait >4 | 369/2000 (18.45%) | 9/2000 (0.45%) |
| Terminal failed, chưa từng có service start trong episode | 205/2000 (10.25%) | 0/2000 |

Cost tăng 4.580 ở family này (+1.94%): maintenance +1.760, failure-event cost
+11.880, unavailability cost -9.060. Policy phục vụ CM nhiều hơn, dành ít lượt
PM hơn, và giảm chờ/không khả dụng. Số failure events tăng là một tradeoff
quan sát được; không tự nó chứng minh fleet health xấu hơn, vì việc sửa xong
cũng đưa máy trở lại hoạt động và chịu hazard. Hai nhánh có exposure khác nhau.

Chỉ tối ưu cost đã tính downtime đúng nhưng vẫn cho phép hy sinh một số máy
để phòng ngừa các failure event đắt tiền trên máy khác. Đây không phải phát
hiện một lỗi tính cost. Nếu yêu cầu thực tế không chấp nhận chờ lâu, yêu cầu
đó cần xuất hiện rõ trong mục tiêu/ràng buộc.

Hiệu ứng giảm overdue cũng xuất hiện với head local: **99.2396%**, thắng 10/10
seed. Khi đều có phí chờ, thêm context chỉ giảm 0.3534% base cost, CI 95% mô tả
[-1.8273, +0.7833]. Hai đối chiếu này là secondary, không thêm kiểm định xác nhận.
Chúng phù hợp với việc thay đổi yêu cầu phục vụ là yếu tố có tác động lớn hơn
việc thêm context ở allocation head trong thiết kế hiện tại.

## Đối chứng rule giới hạn kết luận về thuật toán

Context_wait rẻ hơn rule **0.6094%** trên primary panel, thắng 8/10 training
seed; CI 95% mô tả cho delta cost **[-1.9247, +0.1197]** vẫn chứa 0.
Rule không có overdue tick trên hai primary pressure families, và p95 N3/N4
là 1/2, tốt hơn 2/4 của context_wait. Vì vậy **chưa chứng minh policy học tạo
lợi ích cost rõ hơn heuristic khi cùng xét chất lượng phục vụ**.

Context_cost rẻ hơn rule 1.5526%, nhưng vi phạm service threshold thường xuyên.
Không được dùng nhánh cost-only này để tuyên bố lời giải tốt hơn cho bài toán
có yêu cầu phục vụ. Ngược lại, không thể nói H_wait vô giá trị chỉ vì rule đã
đạt: nó chỉ ra policy học phản ứng với yêu cầu được định nghĩa lại; chưa xác
lập giá trị gia tăng của một phương pháp mới so với đối chứng mạnh.

## Horizon dài, scale và exact nhỏ

N4 pressure H=24 là OOD, secondary: context_wait giảm overdue
**1.953 → 0.0115** (99.4112%), thắng 10/10, p95 **18 → 4**;
base cost **479.562 → 490.364** (+2.25%). Hiệu ứng không chỉ xuất hiện ở H=12,
nhưng chưa phải kết luận steady state. Vẫn còn 23/2000 records (1.15%) vượt B.

N5 pressure, secondary: overdue giảm **1.4644 → 0.0548**, base cost tăng khoảng
5.24%, p95 **10 → 4**. Tuy vậy 123/2500 records (4.92%) vượt B, max wait 6.
Kết quả transfer hỗ trợ giảm tail waiting nhưng không bảo đảm hard deadline.

Exact N2/K2 dùng base cost, không phải waiting-constrained optimum:
optimal 30.5; context_cost trung bình 30.6667, gap 0.1667, allocation regret 0,
10/10 seed có gap <0.2. Ba nhánh còn lại trung bình 30.8667, gap 0.3667,
9/10 seed có gap <0.2. Rule 32.4444. Những con số này mô tả exact cell nhỏ;
không suy ra ceiling N3/N4 hay nhu cầu MARL. Có giá trị context ở một số lỗi
allocation nhỏ, nhưng chưa thành lợi ích cost chính trên primary panel.

## Audit và phạm vi tái hiện

Raw manifest COMPLETED, Git sạch, đúng commit/source bundle và seed registry.
Đủ 40 checkpoints, 4.8 triệu transitions, 160000 optimizer updates, 400000
training episodes, 10000 rollout logs, 16000 learned test, 800 development,
420 reference và 41 exact rows. 23/23 engineering audits đạt; sealed panels
đóng. Run CPU trên Linux, torch 2.14.0+cu126; thời gian khoảng 5 giờ 8 phút.

Audit mới trên Mac đã replay **16400 test episodes / 207050 physical ticks**,
đối chiếu state, tick-cost, machine waits/starts và source/checkpoint contracts.
Tính scalar economic cost và boundary overdue riêng khỏi giá trị trả về của
simulator. Simulator replay vẫn dùng cùng implementation, chưa phải simulator
độc lập. Tất cả 400000 training records đã reconcile objective/base/wait cost;
26975 records có waiting penalty dương.

Đã chạy lại inference checkpoint trên **202000 logged learned-policy states**
ở Mac theo batch256: **0 sequence mismatch, 0 physical-matching mismatch**,
đủ 40 model và source snapshot trùng. Near-tie fix ổn trên panel đã kiểm tra;
không được khẳng định bảo đảm bitwise mọi platform. Không chạy lại training.

Learning curves có cải thiện trong cả bốn nhánh, matching size cuối khoảng
0.50 cặp/bước và loss hữu hạn; không có collapse STOP toàn bộ như tiny smoke.
Mean training objectives giữa cost/wait arms khác định nghĩa nên không dùng
chúng để xếp hạng chất lượng cross-arm.

Thống kê paired, primary weights, p95 và cả hai bộ gate đã tính lại độc lập từ
CSV, khớp summary. Raw-file SHA256 lưu riêng. Fallacy scan **11/11** trong
`statistics.json`: không loại seed, không coi machine records là replicates,
không thay primary bằng cell thuận lợi, các phân tích thêm được đánh dấu
secondary/descriptive, không kết luận MARL từ hiệu ứng mục tiêu hoặc context.

## Hướng chốt problem

Bằng chứng phù hợp với problem: **điều phối PM/CM và technician để tối ưu chi
phí kinh tế dưới yêu cầu chất lượng phục vụ máy hỏng**. “Tránh bỏ mặc máy” cần
một tiêu chí đo được. Giá B=4, giá phạt12 chỉ là assumption thử nghiệm; cần đối
chiếu yêu cầu thực tế trước khi xem chúng là đặc tả ứng dụng.

Penalty hiện tại là soft cost, p95 đạt không có nghĩa mọi máy đều được sửa
trong 4 tick. Run đủ để xác nhận tác động của yêu cầu chờ đã chọn trong
benchmark. Chưa đủ để chốt một thuật toán mới vượt rule rõ rệt hoặc cần MARL.
PPO hiện tại vẫn là một scheduler trung tâm, không tự trở thành MARL.

![Kết quả context và waiting](/Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_context_waiting_v1_analysis_20261005T125813Z/context_waiting_results.png)

## Artifacts phân tích

- `statistics.json`, `paired_seed_recomputed.csv`, `primary_metrics.csv`: thống kê/gates độc lập.
- `family_metrics.csv`, `machine_tail_metrics.csv`, `learning_curves.csv`, `exact_aggregates.csv`: chi tiết và cơ chế.
- `replay_audit.json/.log`: state/cost audit; `policy_replay.json/.log`: checkpoint inference đối chiếu Mac/Linux.
- `raw_file_sha256.csv`: inventory/hash raw files; không chỉnh raw.
- `analyze.py`, `policy_replay.py`, `plot.py`, `analysis.log`: source/trace phân tích.
- `context_waiting_results.png/.svg`: hình tĩnh đã xem và kiểm tra bố cục.
